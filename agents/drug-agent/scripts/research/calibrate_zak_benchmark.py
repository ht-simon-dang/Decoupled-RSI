#!/usr/bin/env python
"""Calibrate the independent ZAK benchmark against real development controls.

Step 7 of the per-target evolution program. This is the entry point that turns
:mod:`drug_agent.selfevolve.evaluation.target_benchmark`'s rules into the two
frozen records every later step is judged against:

``configs/research/zak-c9r/qualification.json``
    Appendix A §3.3's qualification: exact required metrics, the derived
    structure-score threshold, geometry/property/novelty rules, the evaluation
    cap, and the evaluator's own content address.
``configs/research/zak-c9r/calibration.json``
    What was measured to get there: the scorer and its version, run-to-run
    determinism, wrapper-versus-upstream parity, the activity-gate operating
    point search, discrimination against *measured* inactives, and the label
    permutation controls.

**Only development controls are used.** Every row this script reads is screened
through Step 6's protected exclusion registry before it is used for anything
(:func:`~drug_agent.selfevolve.target.splits.screen_records`), and the screen's
counts go into the calibration record. That is the prerequisite Step 6 marked
"not yet applicable" and handed forward: nothing fitted, selected or thresholded
here has seen a held-out compound.

It is deliberately not a unit test. It needs the real ``vina`` binary, the real
6JUU entry under ``.external/`` and about ten minutes.
``tests/research/test_target_metrics.py`` holds the parts that run anywhere.

Usage::

    conda run --no-capture-output -n drug-agent \
        python scripts/research/calibrate_zak_benchmark.py

Exit status is 0 when both records were written, 1 when a mandatory calibration
could not be established and the records were therefore not frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / ".external" / "research" / "zak-c9r"
DERIVED = SNAPSHOT / "derived"
PROTECTED = SNAPSHOT / "protected"
OUT_DIR = REPO / "configs" / "research" / "zak-c9r"

#: Fixed before any control was docked, so the sample cannot be reshuffled to
#: change an answer. 42 is the seed both engines were verified with at repair R1.
SEED = 42

#: How many measured-inactive development controls to dock. Bounded because
#: each dock is a real search; chosen before the scores existed, and the sample
#: is taken deterministically by stable hash of the compound key.
N_INACTIVE_CONTROLS = 24

#: Repeats of one ligand at a fixed seed, to *measure* rather than assume
#: backend determinism before freezing a numerical tolerance.
N_DETERMINISM_REPEATS = 3

#: A second seed for the same ligand and box. Repeats at one seed establish
#: reproducibility at that seed and nothing about whether the seed matters, and
#: conflating the two is how "deterministic" gets claimed for a stochastic search.
SEED_PROBE = 7

#: Appendix A §3.3 caps the expensive independent pipeline at 200 unique
#: molecules per arm per seed. That is the independent evaluator's cap; it is
#: not ``settings.dock_budget``, which is the *generation-side* structure
#: agent's per-batch limit and does not apply here.
TRACK_B_EVALUATION_CAP = 200

#: The registered independent structure scorer, chosen before candidate search.
SCORER_ENGINE = "vina"
SCORER_METRIC = "vina_affinity"

#: Step 15B. The generation-quality endpoints whose gates are derived here, and
#: the capability each one comes from. Both are derived by the SAME frozen rule
#: as the structure threshold — the median of the valid known-active development
#: controls — so a widened gate set cannot smuggle in a friendlier rule.
GENERATION_QUALITY_GATES = {
    "safety": {"metric": "herg", "capability": "safety.assess"},
    "synthetic_accessibility": {"metric": "sa_score", "capability": "synth.assess"},
}

#: The content address of the structure-only qualification this record succeeds.
#: Hard-coded rather than read from the file this script overwrites, so a second
#: run does not chain lineage onto its own output.
SUPERSEDED_QUALIFICATION_HASH = "sha256:c5e20f2aa0ff2c4c558830d2f0773837f293b5151925f87b4ec47a0e64026131"

#: Step 15A §8.3's operator decision for the Stage D sweep.
STAGE_D_UNIQUE_PER_ARM_PER_SEED = 50
STAGE_D_SEEDS = 3
STAGE_D_RAW_ATTEMPT_CEILING = 2000


def _load_redock_module() -> Any:
    """Reuse repair R1's receptor/box preparation rather than re-deriving it.

    Two scripts deriving the same search box independently is two chances to
    derive a different one, and the box is part of the frozen evaluator.
    """
    path = Path(__file__).resolve().parent / "dock_zak_reference_ligand.py"
    spec = importlib.util.spec_from_file_location("zak_redock", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise SystemExit(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run the Step 5/6 build scripts first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Control selection — development partition only, screened at the boundary
# ---------------------------------------------------------------------------
def select_controls() -> dict[str, Any]:
    """Known-active and measured-inactive development controls, screened.

    A compound is a *known active* when some measurement of it is a conclusive
    positive under the frozen label policy, and a *measured inactive* when some
    measurement is a conclusive negative and none is a positive. Untested
    molecules are never converted into negatives.
    """
    from drug_agent.selfevolve.target.splits import label_class, screen_records
    from drug_agent.util import stable_hash_int

    rows = _json(DERIVED / "measurements.json")
    membership = _json(PROTECTED / "split_membership.json")["membership"]
    registry = _json(PROTECTED / "protected_exclusion_registry.json")
    family_of = _json(PROTECTED / "split_membership.json")["family_of"]

    by_compound: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_compound.setdefault(str(row["inchikey"]), []).append(row)

    # Everything this calibration may look at, offered to the boundary exactly
    # as a fresh acquisition would be.
    offered = [
        {
            "molecule_id": str(rs[0]["molecule_id"]),
            "connectivity_key": rs[0].get("connectivity_key"),
            "family_id": family_of.get(inchikey),
            "aliases": sorted({str(r["molecule_id"]) for r in rs} | {inchikey}),
            "inchikey": inchikey,
            "partition": membership.get(inchikey),
        }
        for inchikey, rs in sorted(by_compound.items())
        if membership.get(inchikey) in {"train", "development"}
    ]
    screen = screen_records(offered, registry)
    # The same boundary, offered the held-out compounds, as a positive control
    # that it would in fact refuse them.
    held_out = [
        {
            "molecule_id": str(rs[0]["molecule_id"]),
            "connectivity_key": rs[0].get("connectivity_key"),
            "family_id": family_of.get(inchikey),
            "aliases": sorted({str(r["molecule_id"]) for r in rs} | {inchikey}),
        }
        for inchikey, rs in sorted(by_compound.items())
        if membership.get(inchikey) == "test"
    ]
    held_out_screen = screen_records(held_out, registry)

    admitted = {str(r["inchikey"]) for r in screen["admitted"]}
    actives: list[dict[str, Any]] = []
    inactives: list[dict[str, Any]] = []
    for inchikey, rs in sorted(by_compound.items()):
        if membership.get(inchikey) != "development" or inchikey not in admitted:
            continue
        classes = {label_class(r["value_M"], str(r.get("relation") or "=")) for r in rs}
        entry = {
            "compound_key": inchikey,
            "molecule_id": str(rs[0]["molecule_id"]),
            "smiles": str(rs[0]["molecule"]["standard_smiles"]),
            "n_measurements": len(rs),
        }
        if "positive" in classes:
            actives.append(entry)
        elif "negative" in classes:
            inactives.append(entry)

    ordered = sorted(inactives, key=lambda e: stable_hash_int(f"{SEED}:{e['compound_key']}"))
    sampled = sorted(ordered[:N_INACTIVE_CONTROLS], key=lambda e: e["compound_key"])
    return {
        "actives": actives,
        "inactives": sampled,
        "n_inactives_available": len(inactives),
        "boundary": {
            "n_offered": screen["n_screened"],
            "n_admitted": screen["n_admitted"],
            "n_quarantined": screen["n_quarantined"],
            "held_out_offered": held_out_screen["n_screened"],
            "held_out_quarantined": held_out_screen["n_quarantined"],
            "held_out_admitted": held_out_screen["n_admitted"],
            "split_hash": registry.get("split_hash"),
            "note": (
                "every compound this calibration reads passed Step 6's exclusion boundary "
                "first; the held-out row is a positive control that the same boundary "
                "refuses them"
            ),
        },
        "selection_rule": (
            "development partition only; known active = some measurement is a conclusive "
            "positive under the frozen label policy; measured inactive = some measurement "
            "is a conclusive negative and none is a positive; inactives sampled "
            f"deterministically by stable_hash_int('{SEED}:<compound_key>'), first "
            f"{N_INACTIVE_CONTROLS}"
        ),
    }


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------
def dock_one(adapter: Any, inputs: dict[str, Any], entry: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """One control's structure score as an endpoint record, never a bare float."""
    from drug_agent.selfevolve.target.api import endpoint_result

    started = time.time()
    try:
        poses = adapter.dock(
            candidate_id=entry["compound_key"],
            smiles=entry["smiles"],
            receptor_pdb=inputs["receptor_pdb"],
            receptor_id=inputs["receptor_id"],
            center=inputs["center"],
            size=inputs["size"],
            n_poses=9,
            seed=seed,
            exhaustiveness=8,
        )
    except Exception as exc:
        return {
            **entry,
            "endpoint": endpoint_result("FAILED", None),
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - started, 2),
        }
    scores = [float(p.score) for p in poses if p.score is not None]
    if not scores:
        return {
            **entry,
            "endpoint": endpoint_result("FAILED", None),
            "error": "the engine returned no scored pose",
            "seconds": round(time.time() - started, 2),
        }
    from drug_agent.selfevolve.evaluation.target_benchmark import most_favourable

    best = most_favourable(scores, SCORER_METRIC)
    return {
        **entry,
        "endpoint": endpoint_result("SUCCEEDED", best),
        "error": None,
        "n_poses": len(poses),
        "engine_version": poses[0].engine_version,
        "pose_molblock": poses[0].pose_ref,
        "seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Step 15B: the generation-quality endpoints, on the same development controls
# ---------------------------------------------------------------------------
def score_generation_quality(entries: list[dict[str, Any]], *, cache: Path | None) -> dict[str, Any]:
    """``herg`` and ``sa_score`` for a control set, through the real backends.

    No generated molecule is scored here and none exists: these are the same
    development controls the structure threshold was derived from, read through
    ``safety.assess`` (the released ADMET-AI Chemprop ensembles) and
    ``synth.assess`` (RDKit's ``sascorer``). Both are cached on the standardized
    SMILES so re-freezing the record is seconds and does not re-run a neural
    ensemble.

    An unscored molecule is recorded as an ``endpoint_result("FAILED", None)``,
    never as a favourable number and never dropped: a control the panel could
    not reach must reduce ``n_valid`` rather than quietly improve a median.
    """
    from drug_agent.selfevolve.target.api import endpoint_result
    from drug_agent.tools.safety_tool import SafetyAdapter
    from drug_agent.tools.synthesis_tool import SynthesisAdapter

    cached: dict[str, Any] = {}
    if cache is not None and cache.is_file():
        cached = json.loads(cache.read_text()).get("by_smiles", {})

    wanted = [e for e in entries if str(e["smiles"]) not in cached]
    if wanted:
        synth = SynthesisAdapter()
        safety = SafetyAdapter()
        if not synth.available():
            raise SystemExit("synth.assess is unavailable here; the SA gate cannot be derived")
        if not safety.panel.available():
            raise SystemExit(
                "safety.assess has no toxicity panel here "
                f"({safety.panel.unavailable_reason}); the hERG gate cannot be derived"
            )
        safety.prefetch([str(e["smiles"]) for e in wanted])
        for entry in wanted:
            smiles = str(entry["smiles"])
            plan = synth.assess(str(entry["compound_key"]), smiles)
            profile = safety.assess(str(entry["compound_key"]), smiles)
            endpoints = {e.endpoint: e for e in profile.endpoints}
            herg = endpoints.get("hERG")
            cached[smiles] = {
                "sa_score": None if plan.sa_score is None else float(plan.sa_score),
                "sa_label": plan.sa_label,
                "herg": None if herg is None or herg.score is None else float(herg.score),
                "herg_model_id": None if herg is None else herg.model_id,
                "herg_calibrated": None if herg is None else bool(herg.calibrated),
                "n_structural_alerts": len(endpoints["structural_alerts"].alerts or [])
                if "structural_alerts" in endpoints
                else None,
            }
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    {
                        "backend_safety": safety.backend,
                        "backend_synth": synth.backend,
                        "by_smiles": cached,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

    rows = []
    for entry in entries:
        row = cached[str(entry["smiles"])]
        rows.append(
            {
                "compound_key": entry["compound_key"],
                "herg": endpoint_result("SUCCEEDED", row["herg"])
                if row["herg"] is not None
                else endpoint_result("FAILED", None),
                "sa_score": endpoint_result("SUCCEEDED", row["sa_score"])
                if row["sa_score"] is not None
                else endpoint_result("FAILED", None),
                "raw": row,
            }
        )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Track A: development labels, the failure penalty and the permutation control
# ---------------------------------------------------------------------------
def track_a_calibration() -> dict[str, Any]:
    """The predeclared regression failure penalty and an executed shuffle control.

    The penalty is derived from **development** pActivity labels alone, by a
    rule fixed before any candidate exists: the 90th percentile of the absolute
    deviation of the development labels about their own median. A compound an
    arm fails to score is therefore charged more than the trivial
    median-constant predictor's typical error, so dropping a difficult compound
    can never be the cheaper option.

    The permutation control needs a predictor with real signal to destroy, so it
    uses a 1-nearest-neighbour Tanimoto reference **fitted on the train
    partition only**. That probe is a calibration instrument, not a registered
    baseline — Step 8 owns baselines — and it is labelled as such in the record.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    from drug_agent.selfevolve.evaluation.target_benchmark import (
        regression_endpoint,
        regression_shuffle_control,
    )
    from drug_agent.selfevolve.target.api import endpoint_result

    RDLogger.DisableLog("rdApp.*")
    manifest = _json(DERIVED / "split_manifest.json")
    stratum = str(manifest["stratum_id"])
    rows = _json(DERIVED / "measurements.json")
    membership = _json(PROTECTED / "split_membership.json")["membership"]

    exact: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("stratum_id") != stratum or not row.get("exact"):
            continue
        key = str(row["inchikey"])
        exact.setdefault(key, row)

    train = {k: v for k, v in exact.items() if membership.get(k) == "train"}
    dev = {k: v for k, v in exact.items() if membership.get(k) == "development"}
    dev_labels = {k: float(v["p_value"]) for k, v in sorted(dev.items())}

    median = statistics.median(dev_labels.values())
    deviations = sorted(abs(v - median) for v in dev_labels.values())
    index = max(0, math.ceil(0.9 * len(deviations)) - 1)
    penalty = float(deviations[index])
    trivial = statistics.fmean(deviations)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles: str) -> Any:
        mol = Chem.MolFromSmiles(smiles)
        return generator.GetFingerprint(mol) if mol is not None else None

    train_fps = []
    for key, row in sorted(train.items()):
        fp = fingerprint(str(row["molecule"]["standard_smiles"]))
        if fp is not None:
            train_fps.append((key, fp, float(row["p_value"])))

    predictions: dict[str, dict[str, Any]] = {}
    for key, row in sorted(dev.items()):
        fp = fingerprint(str(row["molecule"]["standard_smiles"]))
        if fp is None or not train_fps:
            predictions[key] = endpoint_result("FAILED", None)
            continue
        best = max(train_fps, key=lambda t: DataStructs.TanimotoSimilarity(fp, t[1]))
        predictions[key] = endpoint_result("SUCCEEDED", best[2])

    pool = sorted(dev_labels)
    observed = regression_endpoint(
        pool=pool, labels=dev_labels, predictions=predictions, failure_penalty=penalty
    )
    shuffle = regression_shuffle_control(
        pool=pool,
        labels=dev_labels,
        predictions=predictions,
        failure_penalty=penalty,
        n_permutations=200,
        seed=SEED,
    )
    # A positive control on the control itself, on the REAL development labels:
    # an oracle predictor must have its error destroyed by permuting them. If it
    # did not, the permutation machinery would be broken and a weak result from
    # the probe above would be uninterpretable.
    oracle = regression_shuffle_control(
        pool=pool,
        labels=dev_labels,
        predictions={k: endpoint_result("SUCCEEDED", v) for k, v in dev_labels.items()},
        failure_penalty=penalty,
        n_permutations=200,
        seed=SEED,
    )
    return {
        "primary_mode": "regression",
        "primary_metric": "mae_pactivity",
        "stratum_id": stratum,
        "ranking_branch": "INSUFFICIENT_TEST_DATA",
        "ranking_branch_note": (
            "Step 6 measured 0 positives in the development partition of the primary "
            "stratum. EF10 is implemented, hand-checked and reportable as exploratory; "
            "it is not the primary Track A metric for this target."
        ),
        "failure_penalty": penalty,
        "failure_penalty_rule": (
            "90th percentile of |development pActivity label - development median|, "
            "computed on the development partition of the primary stratum only, before "
            "any candidate exists"
        ),
        "failure_penalty_reference": {
            "development_median_pactivity_absolute_deviation": trivial,
            "note": (
                "the trivial median-constant predictor's mean absolute error on the same "
                "labels; the penalty is deliberately above it so an arm can never gain by "
                "declining to score a difficult compound"
            ),
        },
        "n_development_exact": len(dev_labels),
        "n_train_exact": len(train_fps),
        "calibration_probe": {
            "kind": "1-nearest-neighbour Morgan r2/2048 Tanimoto reference",
            "fitted_on": "train partition only",
            "purpose": (
                "a probe with real signal, so the label permutation control has something "
                "to destroy. NOT a registered baseline: Step 8 owns baselines and nothing "
                "here is a Track A result."
            ),
            "observed": {
                k: observed[k] for k in ("status", "mae", "coverage", "n_pool", "n_scored", "n_failed")
            },
        },
        "shuffle_control": shuffle,
        "shuffle_positive_control": {
            **oracle,
            "predictor": "oracle (the development labels themselves)",
            "purpose": (
                "proves the permutation control fires on the real development labels, so a "
                "weak p-value from the probe above is a statement about the probe and not "
                "about the control"
            ),
        },
    }


def measure_endpoint_determinism(entries: list[dict[str, Any]], *, cache: Path | None) -> dict[str, Any]:
    """How far ``herg`` moves between repeats of the identical batch.

    Measured before the precision in :data:`ENDPOINT_SCALES` is relied on, not
    assumed from it — the same order Step 7 used for the docking tolerance.
    ``safety_tool``'s own card names float32 reduction-order noise as a known
    failure mode; this is the number. ``sa_score`` is a deterministic fragment
    lookup and is not repeated.

    Cached with the endpoint values, so re-freezing the records does not re-run
    the ensembles.
    """
    payload: dict[str, Any] = {}
    if cache is not None and cache.is_file():
        payload = json.loads(cache.read_text())
        if payload.get("determinism") is not None:
            return dict(payload["determinism"])

    from drug_agent.selfevolve.evaluation.target_benchmark import ENDPOINT_SCALES
    from drug_agent.tools.safety_tool import SafetyAdapter

    smiles = [str(e["smiles"]) for e in entries]
    repeats: list[list[float]] = []
    for _ in range(N_DETERMINISM_REPEATS):
        adapter = SafetyAdapter()
        adapter.prefetch(smiles)
        run: list[float] = []
        for entry in entries:
            profile = adapter.assess(str(entry["compound_key"]), str(entry["smiles"]))
            endpoints = {e.endpoint: e for e in profile.endpoints}
            herg = endpoints.get("hERG")
            run.append(float("nan") if herg is None or herg.score is None else float(herg.score))
        repeats.append(run)

    deviation = max(
        abs(repeats[j][k] - repeats[0][k]) for j in range(len(repeats)) for k in range(len(entries))
    )
    precision = int(ENDPOINT_SCALES["herg"]["precision"])
    record = {
        "metric": "herg",
        "n_repeats": len(repeats),
        "n_molecules": len(entries),
        "max_abs_deviation": deviation,
        "frozen_precision_decimals": precision,
        "rounding_absorbs_the_wobble": deviation < 0.5 * (10.0**-precision),
        "note": (
            "three repeats of the identical batch through the identical ensembles on this "
            "host. The deviation is the float32 reduction-order noise safety_tool's own card "
            "names, and the frozen precision is roughly two orders of magnitude coarser, so a "
            "molecule cannot cross the bar by being scored twice. sa_score is a deterministic "
            "fragment-table lookup and is not repeated"
        ),
    }
    if cache is not None and cache.is_file():
        payload["determinism"] = record
        cache.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return record


def novelty_union_block() -> dict[str, Any]:
    """The 30-versus-611 reconciliation, stated rather than inherited.

    Two exclusion sets are in play and they are not alternatives — one is a term
    in the other. ``novelty.exclusion_smiles`` below holds the **30 reference
    controls**, which is the *base*, and it stays exactly 30 because
    ``baseline_arms._common_exclusion_set`` derives the operative set from it at
    run time by unioning the admitted training pool. The set the de-novo primary
    count actually uses is that **locked union of 611**, whose ``union_hash`` the
    frozen protocol records as MEASURED-stable. Plan v1.1 §3.2 requires the union;
    this record uses the union and says so here.
    """
    protocol = _json(OUT_DIR / "protocol.json")
    lock = protocol["novelty_exclusion"]
    return {
        "set": "the locked union of admitted resources across arms",
        "n_base_controls": lock["n_base_controls"],
        "n_training_pool": lock["n_training_pool"],
        "n_union": lock["n_union"],
        "union_hash": lock["union_hash"],
        "frozen_before_generation_evaluation": lock["frozen_before_generation_evaluation"],
        "locked_under": lock["locked_under"],
        "derivation": (
            "sorted(set(novelty.exclusion_smiles) | set(admitted train-partition standardized "
            "SMILES)), computed by baseline_arms._common_exclusion_set and hashed as "
            "sha256 of the newline-joined sorted set"
        ),
        "reconciliation": (
            "the 30 SMILES serialized under novelty.exclusion_smiles are the reference-control "
            "BASE, not the operative de-novo exclusion set, and they are deliberately left at 30 "
            "so the runtime derivation keeps reproducing n_base 30 / n_training 581 / n_union 611 "
            "and the locked union_hash EXACTLY. The operative set for the de-novo primary table "
            "is the union of 611. Plan v1.1 §3.2 requires the union; Step 7's record predates the "
            "training pool and its own notes said Step 8 would add it. Nothing here silently "
            "inherits the 30"
        ),
        "stability": lock["stability"],
        "library_retrieval_is_separate": lock["library_retrieval_is_separate"],
        "applies_to": "DE_NOVO_PRIMARY",
    }


def conditioning_block() -> dict[str, Any]:
    """The conditioning-class stratification, read from Step 15A's own record."""
    survey = _json(OUT_DIR / "generator-availability.json")
    admitted = set(survey["admitted_to_sweep"])
    classes: dict[str, list[str]] = {}
    synthesis_aware: list[str] = []
    for row in survey["summary"]:
        arm = str(row["arm"])
        if arm not in admitted:
            continue
        classes.setdefault(str(row["conditioning"]), []).append(arm)
        if row.get("synthesis_aware"):
            synthesis_aware.append(arm)
    return {
        "rule": (
            "the primary qualifying-scaffold table is reported STRATIFIED BY CONDITIONING "
            "CLASS. A table pooled across classes may be reported only as a secondary, "
            "labelled POOLED_ACROSS_CONDITIONING"
        ),
        "why": (
            "the arms are not conditioned alike. Three of the nine admitted arms never see "
            "the target at all, so a pooled table silently compares an unconditioned prior "
            "against a pocket-conditioned model and calls the difference a method effect. "
            "Step 8 already observed the consequence in the other direction: REINVENT4's "
            "single scaffold comes from a prior that has never heard of ZAK "
            "(docs/research/baselines.md:46)"
        ),
        "classes": {k: sorted(v) for k, v in sorted(classes.items())},
        "n_admitted_arms": len(admitted),
        "synthesis_aware": sorted(synthesis_aware),
        "synthesis_aware_rule": (
            "on the synthetic-accessibility gate and on any retrosynthesis endpoint, an arm "
            "that can report a route holds a structural advantage. It is reported as such and "
            "never absorbed into a headline"
        ),
        "pocket_conditioned_extra_input": {
            "molecule.generate.pocketxmol": "also requires pocket_center",
        },
        "not_admitted": {
            "molecule.generate.pocket2mol": (
                "EXECUTED_NO_YIELD — ran, consumed 135.8 GPU-seconds, returned four valid "
                "distinct structures of 6/6/7/9 heavy atoms, every one below the declared "
                "floor of 10. A measured yield of zero from a working backend, not an "
                "absent arm, and not an affinity of zero"
            ),
            "molecule.generate.molpilot": (
                "UNAVAILABLE — the driver raised before drawing a sample (upstream "
                "np.compat.long, removed in NumPy 2.0, against numpy 2.5.3). 5.5 s, 0.0 GPU-s"
            ),
        },
        "source": "configs/research/zak-c9r/generator-availability.json (step-15A, run-05)",
    }


def caps_block() -> dict[str, Any]:
    """Appendix A §3.3's ceiling and the operator's Stage D choice inside it."""
    total = STAGE_D_UNIQUE_PER_ARM_PER_SEED * STAGE_D_SEEDS
    return {
        "appendix_a_ceiling": {
            "raw_generation_attempts_per_arm_per_seed": STAGE_D_RAW_ATTEMPT_CEILING,
            "unique_molecules_to_the_expensive_pipeline_per_arm_per_seed": TRACK_B_EVALUATION_CAP,
            "seeds": 1,
        },
        "stage_d": {
            "unique_molecules_per_arm_per_seed": STAGE_D_UNIQUE_PER_ARM_PER_SEED,
            "seeds": STAGE_D_SEEDS,
            "unique_molecules_per_arm_total": total,
            "raw_generation_attempts_per_arm_per_seed": STAGE_D_RAW_ATTEMPT_CEILING,
            "source": "operator decision, step-15A §8.3",
            "reason": (
                "the frozen protocol makes the deployment seed the replication unit and states "
                "that the molecule is not, so seeds buy inference and molecules do not"
            ),
        },
        "identical_across_arms": True,
        "consistency": (
            f"{STAGE_D_UNIQUE_PER_ARM_PER_SEED} <= evaluation_cap {TRACK_B_EVALUATION_CAP}, so the "
            "Stage D budget sits strictly inside Appendix A §3.3's ceiling and no gate below was "
            "chosen to fit a budget"
        ),
        "a_timeout_is_not_a_yield": (
            "an arm that cannot reach the common cap has its shortfall logged as a backend "
            "failure with its consumed cost. Appendix A §3.3 forbids reporting that as a "
            "fixed-budget qualifying yield"
        ),
    }


def reporting_block(self_check: dict[str, Any] | None = None) -> dict[str, Any]:
    """§2.3's anti-gaming provisions, written into the rule before any search."""
    expected_yield: dict[str, Any] = {}
    if self_check and self_check.get("status") == "EXECUTED":
        n_eval = int(self_check["n_evaluated"])
        n_pass = int(self_check["n_qualifying_compounds"])
        expected_yield = {
            "known_active_pass_rate": f"{n_pass} of {n_eval}",
            "what_was_measured": (
                "the full widened gate set applied to the REAL known-active development "
                "controls - the compounds a human being measured as active against this "
                "target - with only the property and novelty gates disabled, because those "
                "controls are in the exclusion set by construction"
            ),
            "funnel": dict(self_check["funnel"]),
            "consequence": (
                f"real known ZAK actives clear this bar at {n_pass} in {n_eval}. A qualifying "
                "yield of ZERO from a generator arm is therefore the EXPECTED outcome at this "
                "bar, not a surprise, not a broken arm, not an unavailable capability and not "
                "an affinity of zero. Read a zero as a measured fixed-budget yield against a "
                "bar that most known actives also fail"
            ),
            "why_the_bar_was_not_loosened": (
                "every threshold above was derived by the frozen rule from development "
                "controls before any candidate existed. Loosening one because the pass rate "
                "looks low is the retrofit Appendix A §3.3 forbids, and the low pass rate is "
                "information about the bar rather than a defect in it"
            ),
            "how_to_report_it": (
                "quote this pass rate beside any arm's qualifying count, so a reader can see "
                "what the bar costs a molecule that really is active"
            ),
        }
    return {
        **({"expected_yield": expected_yield} if expected_yield else {}),
        "correlation_sign_convention": (
            "vina_affinity, minimizedAffinity and Boltz-2 pred_value are LOWER_IS_BETTER, so a "
            "PERFECT potency predictor has Spearman -1.0; CNNscore, CNNaffinity, "
            "probability_binary and any pActivity estimate are HIGHER_IS_BETTER, so a perfect "
            "predictor is +1.0. Near zero is the bad case in both directions. Every correlation "
            "reported under this rule must carry its perfect value beside it"
        ),
        "retracted_claim_not_to_be_reintroduced": (
            "an earlier dispatcher framing - 'the imitated endpoint's correlation with real "
            "measured potency is NEGATIVE in 19 of 21 arm-seed cells', offered as evidence that "
            "docking is a weak proxy - was WITHDRAWN and must not be quoted from this record or "
            "any successor. It inverts the sign convention above: negative is the CORRECT "
            "direction, so those cells are evidence the students tracked potency the right way. "
            "It is also uninformative at n = 4-11 per cell. See "
            ".research/dispatch/correction-dispatcher-sign-error.md and "
            "docs/research/track-c-rebaseline.md §3. The conclusion it was used to support - "
            "that a favourable docking score is not evidence of potency here - stands on the "
            "evidence in limits_of_a_passing_molecule instead"
        ),
        "the_trap": (
            "QED is maximised by bland drug-likeness and SA is minimised by making molecules "
            "simple; both are satisfied by uninteresting molecules nobody wants. A gate set "
            "widened without this paragraph is a gate set an optimiser walks around"
        ),
        "qed_is_not_a_gate": (
            "QED is computed and reported in the unconstrained distributions and is NOT a "
            "qualification gate, precisely because it is the endpoint the trap names"
        ),
        "sa_is_a_floor_not_an_objective": (
            "sa_score enters as a pass/fail bar at the known actives' own median, never as a "
            "term to minimise. The primary metric stays qualifying_scaffolds, a COUNT. Three "
            "frozen constraints make SA-minimisation self-defeating rather than merely "
            "discouraged: the heavy-atom window 10-70 excludes fragment-scale simplicity (it "
            "is what measured Pocket2Mol's yield at zero), the structure threshold of -10.526 "
            "kcal/mol is hard for a small simple molecule to reach because docking scores "
            "scale with size, and the novelty union of 611 excludes the simple known molecules "
            "outright"
        ),
        "reported_separately_never_gating": [
            "new-scaffold novelty rate",
            "nearest-neighbour Tanimoto similarity to the exclusion union",
        ],
        "why_separately": (
            "Appendix A §3.3: scaffold novelty is not required for an improved analog-design "
            "method, so it informs the reader and does not gate the count"
        ),
        "unconstrained_distributions_required": [
            "vina_affinity",
            "herg",
            "sa_score",
            "descriptors.mw",
            "descriptors.logp",
            "qed",
            "n_structural_alerts",
            "heavy_atoms",
        ],
        "unconstrained_distributions_rule": (
            "every endpoint's unconstrained distribution is reported alongside the qualifying "
            "count, together with the raw qualifying-compound count, so a reader can see what "
            "the gates removed rather than only what survived"
        ),
        "generated_3d_quality": {
            "capability": "structure.validate.posebusters",
            "rule": (
                "generated 3D quality is evaluated with PoseBusters beside the docking score "
                "and is reported for every submitted pose. The geometry_limits above are a "
                "VALIDITY screen (0.8 A non-bonded, deliberately far looser than PoseBusters' "
                "0.75 x sum of vdW radii, i.e. 1.65 A for an H...H contact); they are not a "
                "pose-quality benchmark and must not be reported as one"
            ),
            "references": [
                "PoseBusters, arXiv:2308.05777",
                "GenBench3D, arXiv:2407.04424",
            ],
        },
        "generation_funnel_required": (
            "raw attempts, invalid proposals, duplicates and heavy-atom-window rejections are "
            "counted and reported per arm per seed; a backend failure is a missing scientific "
            "outcome with its consumed cost, never a zero"
        ),
        "boltz2_excluded": (
            "Boltz-2 is EXCLUDED from this scoring rule. Its contamination audit "
            "(docs/research/boltz2-contamination-audit.md) is INCONCLUSIVE with one route "
            "open, and separately its headline fails its own Holm correction"
        ),
    }


def limits_block() -> list[str]:
    """What a molecule that passes every gate does and does not license."""
    return [
        "A computational qualification is not biological activity. A molecule that passes "
        "every gate here has not been shown to bind ZAK, to inhibit it, or to be safe; it "
        "has been shown to clear a set of computed bars derived from six known actives.",
        "SIGN CONVENTION, stated because it is the thing most easily got backwards: "
        "vina_affinity and minimizedAffinity are LOWER_IS_BETTER while pActivity is "
        "HIGHER_IS_BETTER, so a PERFECT docking-based potency predictor has Spearman -1.0 and "
        "a value near ZERO is the bad case. Any correlation quoted against a docking score in "
        "this programme must carry its perfect value beside it.",
        "On that convention the registered structure scorer carries no usable potency signal "
        "on this target. Vina's development rank signal is +0.078 (p = 0.663, bootstrap 95% CI "
        "[-0.285, +0.435]) where -1.0 would be perfect: near zero AND on the wrong side of it. "
        "On the better-powered TRAIN partition (n = 102) it is -0.09133 (p = 0.361) - correctly "
        "signed, still indistinguishable from zero. GNINA's minimizedAffinity is +0.035 on the "
        "same convention: two independent engines, 272 runs, 0 failures, coverage 1.000, both "
        "near zero and both on the wrong side. The TRAIN-fitted calibration "
        "pActivity = -0.04335*vina + 5.20171 moves the prediction by 0.195 pActivity units "
        "across the full 4.49 kcal/mol development spread - a constant with jitter. A "
        "favourable vina_affinity is therefore not evidence of potency here.",
        "Redocking is a pipeline sanity check, not an affinity benchmark.",
        "Generated molecules have no known native pose, so no native-pose RMSD may be "
        "invented for them. Any RMSD reported for a generated molecule is to a computed "
        "pose and must say so.",
        "The hERG gate is an empirical operating point on an UNCALIBRATED score. "
        "safety_tool sets calibrated=False on every panel endpoint; passing it is not a "
        "statement about cardiac risk, and failing it is not a statement about harm.",
        "sa_score is a heuristic fragment-contribution score, not a synthesis success "
        "probability and not a route. Only prexsyn and synflownet can report a route at all.",
        "Structural-alert matching flags substructures, not toxicity.",
        "The activity gate is UNCALIBRATED and Track B stays registered "
        "STRUCTURE_ONLY_COMPUTATIONAL. Nothing in this record may be described as an "
        "activity prediction.",
        "Every threshold here is a median of six numbers. Six known-active development "
        "controls is not a power calculation, and no gate below is a validated filter.",
        "A qualifying yield of zero is a measured fixed-budget result, not an affinity of "
        "zero and not an unavailable arm.",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-docking", action="store_true", help="diagnostics only; writes nothing")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DERIVED / "calibration_docking_cache.json",
        help=(
            "where the docked control results are cached. Docking 30 real ligands takes tens "
            "of minutes and rebuilding the frozen records from the same scores must not "
            "require repeating it. Delete the file to force a fresh docking run."
        ),
    )
    parser.add_argument(
        "--generation-quality-cache",
        type=Path,
        default=DERIVED / "generation_quality_cache.json",
        help=(
            "where the controls' herg / sa_score endpoint values are cached, keyed on the "
            "standardized SMILES. Delete the file to re-run the ADMET-AI ensembles."
        ),
    )
    args = parser.parse_args(argv)

    from drug_agent.selfevolve.evaluation.target_benchmark import (
        CALIBRATED,
        STRUCTURE_ONLY_COMPUTATIONAL,
        WRAPPER_PARITY_TOLERANCE,
        build_calibration,
        build_qualification,
        calibrate_activity_gate,
        calibration_hash,
        derive_endpoint_threshold,
        derive_structure_threshold,
        discrimination_check,
        endpoint_gate_diagnostics,
        enrichment_record,
        evaluate_portfolio,
        label_shuffle_control,
        numerical_tolerance,
        parity_check,
        qualification_hash,
        register_track_b_name,
        upstream_direct_dock,
    )
    from drug_agent.tools.vina_tool import DockingAdapter
    from drug_agent.util import hash_bytes

    controls = select_controls()
    print(
        f"controls: {len(controls['actives'])} known-active, "
        f"{len(controls['inactives'])} measured-inactive "
        f"(of {controls['n_inactives_available']} available); boundary "
        f"{controls['boundary']['n_admitted']}/{controls['boundary']['n_offered']} admitted, "
        f"held-out {controls['boundary']['held_out_quarantined']}/"
        f"{controls['boundary']['held_out_offered']} refused"
    )
    track_a = track_a_calibration()
    print(
        f"track A: {track_a['n_development_exact']} development exact compounds, "
        f"failure penalty {track_a['failure_penalty']:.4f} pActivity, "
        f"probe MAE {track_a['calibration_probe']['observed']['mae']:.4f}, "
        f"shuffle p={track_a['shuffle_control']['p_value']:.4f}, "
        f"oracle positive control p={track_a['shuffle_positive_control']['p_value']:.4f}"
    )
    if args.skip_docking:
        print(json.dumps({"controls": controls, "track_a": track_a}, indent=2, default=str))
        return 0

    adapter = DockingAdapter()
    if not adapter.available():
        raise SystemExit(f"vina is unavailable here: {adapter.profile.unavailable_reason}")

    redock = _load_redock_module()
    cached = json.loads(args.cache.read_text()) if args.cache and args.cache.is_file() else None
    with tempfile.TemporaryDirectory(prefix="zak_calibrate_") as tmp:
        work = Path(tmp)
        inputs = redock.prepare_inputs(work)
        inputs["receptor_pdb"] = Path(inputs["receptor_pdb"]).resolve()
        receptor_sha = hash_bytes(Path(inputs["receptor_pdb"]).read_bytes())
        reference_smiles = str(inputs["smiles"])

        if cached is not None and cached.get("receptor_sha256") == receptor_sha:
            print(f"reusing the docked controls cached at {args.cache}")
            repeats = [float(v) for v in cached["determinism_repeats"]]
            direct = cached["upstream_direct"]
            active_rows = cached["active_rows"]
            inactive_rows = cached["inactive_rows"]
            seed_probe = cached.get("seed_probe")
            if seed_probe is None:
                seed_probe = dock_one(
                    adapter,
                    inputs,
                    {"compound_key": "reference_C9R", "smiles": reference_smiles},
                    seed=SEED_PROBE,
                )["endpoint"]["value"]
                cached["seed_probe"] = seed_probe
                args.cache.write_text(json.dumps(cached, indent=2, default=str) + "\n")
        else:
            # 1. Determinism at a fixed seed, measured rather than assumed.
            repeats = []
            for _ in range(N_DETERMINISM_REPEATS):
                record = dock_one(
                    adapter,
                    inputs,
                    {"compound_key": "reference_C9R", "smiles": reference_smiles},
                    seed=SEED,
                )
                if record["endpoint"]["value"] is not None:
                    repeats.append(float(record["endpoint"]["value"]))

            # 2. Wrapper parity against the upstream tools, same inputs, same seed.
            direct = upstream_direct_dock(
                smiles=reference_smiles,
                receptor_pdb=inputs["receptor_pdb"],
                center=inputs["center"],
                size=inputs["size"],
                seed=SEED,
                exhaustiveness=8,
                n_poses=9,
            )

            # 2b. The same ligand at a different seed, so "deterministic at a
            #     fixed seed" is not confused with "the seed does not matter".
            seed_probe = dock_one(
                adapter,
                inputs,
                {"compound_key": "reference_C9R", "smiles": reference_smiles},
                seed=SEED_PROBE,
            )["endpoint"]["value"]

            # 3. The controls themselves.
            active_rows = [dock_one(adapter, inputs, e, seed=SEED) for e in controls["actives"]]
            inactive_rows = [dock_one(adapter, inputs, e, seed=SEED) for e in controls["inactives"]]
            if args.cache:
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                args.cache.write_text(
                    json.dumps(
                        {
                            "receptor_sha256": receptor_sha,
                            "seed": SEED,
                            "determinism_repeats": repeats,
                            "seed_probe": seed_probe,
                            "upstream_direct": direct,
                            "active_rows": active_rows,
                            "inactive_rows": inactive_rows,
                        },
                        indent=2,
                        default=str,
                    )
                    + "\n"
                )
                print(f"cached the docked controls at {args.cache}")

    tolerance = numerical_tolerance(repeats)
    tolerance["seed"] = SEED
    tolerance["seed_sensitivity"] = {
        "other_seed": SEED_PROBE,
        "other_seed_score": seed_probe,
        "abs_difference": (
            None if seed_probe is None else abs(float(seed_probe) - float(tolerance["median"]))
        ),
        "note": (
            "measured, not assumed: the repeats above fix the seed, so they establish "
            "reproducibility at that seed and say nothing about seed sensitivity. This row "
            "is the same ligand and the same box at a different seed."
        ),
    }
    print(
        f"determinism: {repeats} -> tolerance {tolerance['tolerance']:.6f}; "
        f"seed {SEED_PROBE} gives {seed_probe}"
    )
    parity = [
        parity_check(
            direct=float(direct["best_score"]),
            wrapped=float(tolerance["median"]),
            tolerance=WRAPPER_PARITY_TOLERANCE,
            label="vina dock of the reference ligand: wrapped adapter vs upstream-direct",
        )
    ]
    print(f"parity: direct {direct['best_score']} vs wrapped {tolerance['median']} -> {parity[0]['status']}")

    engine_versions = sorted(
        {r["engine_version"] for r in active_rows + inactive_rows if r.get("engine_version")}
    )
    structure_threshold = derive_structure_threshold(
        active_scores=[r["endpoint"] for r in active_rows], metric=SCORER_METRIC
    )
    activity_gate = calibrate_activity_gate(
        active_scores=[r["endpoint"] for r in active_rows],
        inactive_scores=[r["endpoint"] for r in inactive_rows],
        metric=SCORER_METRIC,
    )
    discrimination = discrimination_check(
        active_scores=[r["endpoint"] for r in active_rows],
        inactive_scores=[r["endpoint"] for r in inactive_rows],
        metric=SCORER_METRIC,
    )
    # Step 15B: the generation-quality gates, on the same known-active
    # development controls and by the same frozen derivation rule.
    quality_actives = score_generation_quality(controls["actives"], cache=args.generation_quality_cache)
    quality_inactives = score_generation_quality(controls["inactives"], cache=args.generation_quality_cache)
    endpoint_determinism = measure_endpoint_determinism(
        controls["actives"], cache=args.generation_quality_cache
    )
    print(
        f"herg determinism: {endpoint_determinism['n_repeats']} repeats, max deviation "
        f"{endpoint_determinism['max_abs_deviation']:.3e}, frozen precision "
        f"{endpoint_determinism['frozen_precision_decimals']} dp"
    )
    endpoint_thresholds: dict[str, Any] = {}
    endpoint_diagnostics: list[dict[str, Any]] = []
    for gate_name, spec in sorted(GENERATION_QUALITY_GATES.items()):
        metric = spec["metric"]
        derived = derive_endpoint_threshold(
            metric=metric,
            active_values=[r[metric] for r in quality_actives["rows"]],
        )
        derived["capability"] = spec["capability"]
        derived["gate"] = gate_name
        endpoint_thresholds[gate_name] = derived
        endpoint_diagnostics.append(
            endpoint_gate_diagnostics(
                threshold_record=derived,
                values=[r[metric] for r in quality_actives["rows"]],
                label="known-active development controls (the derivation set)",
            )
        )
        endpoint_diagnostics.append(
            endpoint_gate_diagnostics(
                threshold_record=derived,
                values=[r[metric] for r in quality_inactives["rows"]],
                label="measured-inactive development controls (NOT the derivation set)",
            )
        )
        print(
            f"{gate_name} gate ({metric}): {derived['status']} "
            f"{derived['threshold']} (n_valid {derived['n_valid']}, "
            f"n_invalid {derived['n_invalid']}, min_controls {derived['min_controls']})"
            + (f" reason {derived['reason']}" if derived.get("reason") else "")
        )

    track_b_name = register_track_b_name(activity_gate=activity_gate)
    print(
        f"structure threshold: {structure_threshold['status']} "
        f"{structure_threshold['threshold']}; activity gate: {activity_gate['status']} "
        f"({activity_gate['reason']}); discrimination: {discrimination['status']} "
        f"AUC={discrimination['auc']}; track B registered as {track_b_name}"
    )

    # 4. A real label-shuffle control on the real backend's own output: rank the
    #    docked development controls by score and permute their activity labels.
    scored = [
        (r["compound_key"], float(r["endpoint"]["value"]), 1)
        for r in active_rows
        if r["endpoint"]["value"] is not None
    ] + [
        (r["compound_key"], float(r["endpoint"]["value"]), 0)
        for r in inactive_rows
        if r["endpoint"]["value"] is not None
    ]
    shuffle_controls = [track_a["shuffle_control"]]
    ranking_control = None
    if any(label for _k, _s, label in scored) and len(scored) >= 10:
        ids = [k for k, _s, _l in scored]
        # More negative is better, so the ranking score is the negated energy.
        ranking_scores = [-s for _k, s, _l in scored]
        labels = [label for _k, _s, label in scored]
        ranking_control = label_shuffle_control(
            labels=labels,
            scores=ranking_scores,
            molecule_ids=ids,
            fraction=0.1,
            n_permutations=200,
            seed=SEED,
        )
        ranking_control["observed_record"] = enrichment_record(labels, ranking_scores, 0.1, molecule_ids=ids)
        shuffle_controls.append(ranking_control)
        print(
            f"structure-score EF10 on the development controls: "
            f"{ranking_control['observed']:.3f}, permutation p={ranking_control['p_value']:.4f}"
        )

    qualification = build_qualification(
        target_id="CHEMBL3886",
        track_b_name=track_b_name,
        structure_threshold=structure_threshold,
        activity_gate=activity_gate,
        property_limits=[
            {"property": "descriptors.mw", "op": "<=", "value": 500.0},
            {"property": "descriptors.logp", "op": "<=", "value": 5.0},
            {"property": "n_structural_alerts", "op": "==", "value": 0},
        ],
        exclusion_smiles=[e["smiles"] for e in controls["actives"] + controls["inactives"]],
        evaluation_cap=TRACK_B_EVALUATION_CAP,
        dock_budget_note=(
            "settings.dock_budget (default 8) is the GENERATION-side structure agent's "
            "per-batch docking cap and does not apply to this independent evaluator, which "
            f"docks every unique submitted molecule up to evaluation_cap={TRACK_B_EVALUATION_CAP} "
            "(Appendix A §3.3). A candidate-side run that leaves dock_budget at 8 is making a "
            "scientific choice about its own search, and must record it in its ExperimentPlan."
        ),
        split_hash=str(_json(DERIVED / "split_manifest.json")["split_hash"]),
        data_hash=str(_json(DERIVED / "split_manifest.json")["data_hash"]),
        target_manifest_hash=str(_json(DERIVED / "split_manifest.json")["target_manifest_hash"]),
        endpoint_thresholds=endpoint_thresholds,
        novelty_union=novelty_union_block(),
        conditioning=conditioning_block(),
        caps=caps_block(),
        reporting=reporting_block(),
        limits=limits_block(),
        supersedes={
            "qualification_hash": SUPERSEDED_QUALIFICATION_HASH,
            "frozen_by": "step-07, re-frozen through step-14",
            "superseded_by": "step-15B",
            "what_changed": (
                "the gate set is widened by the two generation-quality endpoints (herg, "
                "sa_score), the de-novo novelty set is stated as the locked union of 611 "
                "rather than the 30-control base, the comparison is stratified by "
                "conditioning class, the Stage D caps are recorded, and the anti-gaming "
                "provisions and the limits of a passing molecule are written into the rule"
            ),
            "what_did_not_change": (
                "every Step 7 frozen value: structure_threshold -10.526 vina_affinity "
                "LOWER_IS_BETTER inclusive, its derivation rule and min_controls 3, the "
                "UNCALIBRATED activity gate and its 10/10 floor and its 30 serialized "
                "candidates, the geometry limits, the three property limits including "
                "n_structural_alerts == 0, the counting rule, evaluation_cap 200, the "
                "registered Track B name STRUCTURE_ONLY_COMPUTATIONAL, and the heavy-atom "
                "window of 10-70 that lives in the generator probes"
            ),
            "registered_before_candidate_search": True,
        },
        notes=[
            "Derived from development controls only; every input passed Step 6's exclusion boundary first.",
            "novelty.exclusion_smiles is the 30-molecule reference-control BASE. The operative "
            "de-novo exclusion set is the locked union of 611 recorded under "
            "novelty.primary_de_novo_exclusion, and the base is kept at 30 so the runtime "
            "derivation keeps reproducing that union and its hash exactly.",
            "Qualifying computationally is not evidence of biological activity.",
            "The two generation-quality gates were derived by the SAME frozen rule as the "
            "structure threshold - the median of the valid known-active development controls - "
            "from 6 valid controls against a min_controls of 3. That floor is the median floor, "
            "not Appendix A §3.2's 10/10 classification floor, which applies to a gate that "
            "estimates a true-positive and a false-positive rate and which this one does not.",
            "The hERG gate is an empirical operating point on an UNCALIBRATED score.",
            "evaluator_hash is the content address of every file the judgement depends on. "
            "If it moves, this record is stale: re-run this script (the docked controls are "
            "cached) and record the new hashes. test_the_frozen_records_still_match_the_"
            "evaluator_that_produced_them fails until that is done.",
        ],
    )

    calibration = build_calibration(
        target_id="CHEMBL3886",
        scorer={
            "engine": SCORER_ENGINE,
            "metric": SCORER_METRIC,
            "engine_versions": engine_versions,
            "adapter": "drug_agent.tools.vina_tool.DockingAdapter",
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": SEED,
            "receptor": "ATOM records of .external/research/zak-c9r/rcsb/6JUU.pdb",
            "receptor_sha256": receptor_sha,
            "box_center": list(inputs["center"]),
            "box_size": list(inputs["size"]),
            "secondary_available": "gnina (structure.dock.gnina); not the registered scorer",
            "note": (
                "An empirical scoring function over coordinates. It is a computation, not a "
                "potency prediction, and redocking the deposited ligand is a pipeline sanity "
                "check rather than an affinity benchmark (Appendix A §3.3)."
            ),
        },
        structure_threshold=structure_threshold,
        activity_gate=activity_gate,
        discrimination=discrimination,
        tolerance=tolerance,
        parity=parity,
        controls={
            "partition": "development",
            "n_actives": len(active_rows),
            "n_actives_scored": sum(1 for r in active_rows if r["endpoint"]["value"] is not None),
            "n_inactives": len(inactive_rows),
            "n_inactives_scored": sum(1 for r in inactive_rows if r["endpoint"]["value"] is not None),
            "n_inactives_available": controls["n_inactives_available"],
            "selection_rule": controls["selection_rule"],
            "boundary": controls["boundary"],
            "active_compound_keys": [r["compound_key"] for r in active_rows],
            "inactive_compound_keys": [r["compound_key"] for r in inactive_rows],
            "scores": {r["compound_key"]: r["endpoint"]["value"] for r in active_rows + inactive_rows},
            "failures": [
                {"compound_key": r["compound_key"], "error": r["error"]}
                for r in active_rows + inactive_rows
                if r["endpoint"]["value"] is None
            ],
        },
        track_a=track_a,
        track_b_name=track_b_name,
        shuffle_controls=shuffle_controls,
        notes=[
            "Score orientation is registered, not inferred: vina_affinity is LOWER_IS_BETTER, "
            "and so are herg and sa_score. An unregistered metric still raises INVALID_ENUM.",
            "Discrimination on known controls is sanity evidence for the pipeline, never a "
            "claim of novel affinity accuracy.",
            "Public/pretraining exposure of these compounds is UNKNOWN, and this host has no "
            "OS access boundary (Step 6 measured a candidate reading the protected registry). "
            "No run here may be described as sealed.",
        ],
    )

    # 5. Run the frozen gates on the development controls themselves, so the
    #    qualification is exercised end to end rather than only serialized.
    self_check: dict[str, Any] = {"status": "SKIPPED", "reason": qualification["status"]}
    if qualification["status"] == CALIBRATED:
        quality_by_key = {r["compound_key"]: r for r in quality_actives["rows"]}
        portfolio = evaluate_portfolio(
            [
                {
                    "molecule_id": r["compound_key"],
                    "smiles": r["smiles"],
                    "structure_score": r["endpoint"],
                    "properties": {"descriptors": {"mw": 0.0}},
                    "pose_molblock": r.get("pose_molblock") or "",
                    "endpoints": {
                        "herg": quality_by_key[r["compound_key"]]["herg"],
                        "sa_score": quality_by_key[r["compound_key"]]["sa_score"],
                    },
                }
                for r in active_rows
            ],
            qualification={**qualification, "property_limits": [], "novelty": {"exclusion_smiles": []}},
        )
        self_check = {
            "status": "EXECUTED",
            "what": (
                "the frozen structure-score, geometry and generation-quality gates applied to "
                "the known-active development controls, with the property and novelty gates "
                "disabled because the controls are by construction in the exclusion set"
            ),
            "n_evaluated": portfolio["n_evaluated"],
            "n_qualifying_compounds": portfolio["n_qualifying_compounds"],
            "qualifying_scaffolds": portfolio["qualifying_scaffolds"],
            "funnel": portfolio["funnel"],
            "applied_endpoint_gates": portfolio["applied_endpoint_gates"],
            "unapplied_endpoint_gates": portfolio["unapplied_endpoint_gates"],
            "note": (
                "a control that clears its own median on one endpoint need not clear it on "
                "another, so this count is expected to fall as the gate set widens. It is a "
                "wiring check, not a yield"
            ),
        }
        print(
            f"self-check: {self_check['n_qualifying_compounds']}/{self_check['n_evaluated']} "
            f"known-active controls clear their own median threshold "
            f"({self_check['qualifying_scaffolds']} scaffolds)"
        )
        # The pass rate of REAL known actives through the full widened gate set is
        # load-bearing context for whoever reads a generator's yield, so it goes
        # into the rule itself rather than only into the calibration's funnel. It
        # can only be written after the gates have been applied, so the reporting
        # block is completed here and the record's own address recomputed — the
        # same pattern `baseline_arms._common_exclusion_set` uses for the derived
        # qualification. Nothing a gate reads changes: `evaluate_portfolio` never
        # consults `reporting`, so the self-check above is valid for this record.
        qualification["reporting"] = reporting_block(self_check)
        qualification.pop("qualification_hash", None)
        qualification["qualification_hash"] = qualification_hash(qualification)
        self_check["qualification_hash"] = qualification["qualification_hash"]
        self_check["qualification_hash_note"] = (
            "the gates were applied before reporting.expected_yield could be written from their "
            "own result, so the address above is the final record's. Only prose moved between "
            "the two, and evaluate_portfolio reads no field that changed"
        )
    calibration["qualification_self_check"] = self_check
    calibration["generation_quality"] = {
        "what": (
            "the two gates Step 15B added, and everything measured to derive them. No "
            "generated molecule was scored: these are the same development controls the "
            "structure threshold came from"
        ),
        "capabilities": {
            "safety.assess": (
                "structural alerts (RDKit SMARTS) plus hERG/AMES/DILI from the released "
                "ADMET-AI Chemprop ensembles, authors' TDC training, published hERG AUROC "
                "0.839. Nothing is fitted at call time"
            ),
            "synth.assess": "RDKit Contrib sascorer, Ertl & Schuffenhauer (2009)",
        },
        "uncalibrated_scores": (
            "safety_tool sets calibrated=False on every panel endpoint and its "
            "data_restrictions say so. The hERG gate is therefore an empirical operating "
            "point on an uncalibrated score, not a calibrated probability. This module "
            "previously shipped a fabricated model and was repaired under ADR-0011; the "
            "numbers here come from the repaired path"
        ),
        "thresholds": endpoint_thresholds,
        "diagnostics": endpoint_diagnostics,
        "determinism": endpoint_determinism,
        "control_values": {
            "known_active_development_controls": [
                {
                    "compound_key": r["compound_key"],
                    "herg": r["raw"]["herg"],
                    "sa_score": r["raw"]["sa_score"],
                    "sa_label": r["raw"]["sa_label"],
                    "n_structural_alerts": r["raw"]["n_structural_alerts"],
                }
                for r in quality_actives["rows"]
            ],
            "measured_inactive_development_controls": [
                {
                    "compound_key": r["compound_key"],
                    "herg": r["raw"]["herg"],
                    "sa_score": r["raw"]["sa_score"],
                }
                for r in quality_inactives["rows"]
            ],
            "note": (
                "development-partition controls only. These are computed model outputs, not "
                "measured labels; no held-out compound appears here"
            ),
        },
        "permissiveness_of_the_herg_gate": (
            "reported because it is the honest reading of the number and not because it "
            "moves it: ADMET-AI's own severity ladder calls a panel score >= 0.7 HIGH, and "
            "the bar derived from the six known ZAK actives sits well above that, because "
            "five of those six actives are themselves scored above 0.89. The gate is "
            "therefore a 'no worse than the known actives' bar and NOT a safety filter. "
            "Substituting 0.7 would be choosing a threshold after seeing the data, which is "
            "the one thing Appendix A §3.3 forbids"
        ),
    }
    calibration["calibration_hash"] = calibration_hash(calibration)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"wrote {args.out / 'qualification.json'} and {args.out / 'calibration.json'}")
    print(f"qualification_hash {qualification['qualification_hash']}")
    print(f"calibration_hash   {calibration['calibration_hash']}")

    if qualification["status"] != CALIBRATED:
        print(
            "MANDATORY CALIBRATION INVALID: the Track B structure threshold was not "
            f"established ({structure_threshold['reason']}). The records above record that "
            "fact; nothing may be scored against them.",
            file=sys.stderr,
        )
        return 1
    if track_b_name == STRUCTURE_ONLY_COMPUTATIONAL:
        print(
            "Track B is registered STRUCTURE_ONLY_COMPUTATIONAL: there is no defensible "
            "activity gate on these controls. This name is frozen before candidate search "
            "and must not be removed after seeing candidates."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
