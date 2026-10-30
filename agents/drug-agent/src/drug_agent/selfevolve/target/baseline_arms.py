"""The reference baseline arms, executed from their manifests without an LLM.

Each function here is one arm of Appendix A §4's matrix for the ZAK/C9R target,
reached as ``python -m drug_agent.selfevolve.target.baseline_arms --arm <id>
--manifest <path> --output <dir>``. The manifest is the single source of truth
for preparation settings, seeds, caps and the declared tuning budget, so an arm
is re-run by re-reading a committed file rather than by reconstructing an
argument list.

**Track A is this target's primary track** — regression on ``mae_pactivity`` over
the ``CHEMBL3886|Ki|F|BAO_0000019`` stratum, decided at Step 6 and calibrated at
Step 7. Every Track A arm predicts the *development* pool. The held-out partition
is not read, not scored and not fitted on: the test partition stays sealed until
a protocol lock exists, and each arm offers its candidate training rows to Step
6's exclusion boundary and records what the boundary refused.

**Track B is secondary and registered ``STRUCTURE_ONLY_COMPUTATIONAL``.** Its
arms submit a bounded portfolio to the frozen independent evaluator. The
portfolio cap here is a *declared small probe*, not Appendix A §3.3's
200-molecule confirmatory cap; Step 9 owns the integrated budget service and
until it exists these runs are bounded by local process limits and charged to a
provisional ledger.

Nothing in this module reads a held-out label, and nothing consults an LLM.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .baselines import (
    load_manifest,
    pool_hash,
    portfolio_status,
    require_uncontaminated,
    sha256_file,
    sha256_text,
    validate_baseline_manifest,
    validate_output,
)

Record = dict[str, Any]


def _repo_root() -> Path:
    """The checkout whose campaign snapshots and frozen records apply.

    Normally this module's own checkout. Inside a **candidate** checkout it is
    not: a candidate is a git worktree, the campaign's snapshots under
    ``.external/`` are gitignored, and the frozen calibration belongs to the
    referee rather than to the thing being evaluated. Step 10's dispatcher
    therefore passes the parent's root to the child as
    ``DRUGAGENT_REPO_ROOT``, and it is honoured only when it really holds the
    snapshot — so a stale variable in an operator's shell cannot silently
    redirect an ordinary run, and a wrong one fails loudly on the missing file
    rather than quietly scoring different data.
    """
    default = Path(__file__).resolve().parents[4]
    override = os.environ.get("DRUGAGENT_REPO_ROOT", "").strip()
    if override:
        candidate = Path(override).resolve()
        if candidate != default and (candidate / ".external" / "research" / "zak-c9r").is_dir():
            return candidate
    return default


REPO = _repo_root()
SNAPSHOT = REPO / ".external" / "research" / "zak-c9r"
DERIVED = SNAPSHOT / "derived"
PROTECTED = SNAPSHOT / "protected"
CONFIG_DIR = REPO / "configs" / "research" / "zak-c9r"

#: The partition every arm in this step reports on. The held-out partition is
#: not read here by any arm, for any purpose.
EVALUATED_PARTITION = "development"


# ---------------------------------------------------------------------------
# Shared preparation — identical for every Track A arm, by construction
# ---------------------------------------------------------------------------
def _json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run the Step 5/6/7 build scripts first")
    return json.loads(path.read_text())


def _fingerprint_generator(radius: int, n_bits: int) -> Any:
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def prepare_track_a(preparation: Mapping[str, Any]) -> Record:
    """Stratum rows, the exclusion boundary, the fit pool and the eval pool.

    Every Track A arm calls this and nothing else for its data, so "identical
    data and evaluator preparation across arms" is a property of the code rather
    than a promise in a table.
    """
    from rdkit import RDLogger

    from .splits import screen_records

    RDLogger.DisableLog("rdApp.*")

    manifest = _json(DERIVED / "split_manifest.json")
    stratum = str(manifest["stratum_id"])
    declared = str(preparation.get("stratum_id") or stratum)
    if declared != stratum:
        raise SystemExit(
            f"the manifest declares stratum {declared!r} but the frozen split names "
            f"{stratum!r}; an arm may not silently choose a different endpoint"
        )
    rows = _json(DERIVED / "measurements.json")
    membership = _json(PROTECTED / "split_membership.json")["membership"]
    family_of = _json(PROTECTED / "split_membership.json")["family_of"]
    registry = _json(PROTECTED / "protected_exclusion_registry.json")

    by_compound: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_compound.setdefault(str(row["inchikey"]), []).append(row)

    def _offer(inchikey: str, rs: list[dict[str, Any]]) -> Record:
        return {
            "molecule_id": str(rs[0]["molecule_id"]),
            "connectivity_key": rs[0].get("connectivity_key"),
            "family_id": family_of.get(inchikey),
            "aliases": sorted({str(r["molecule_id"]) for r in rs} | {inchikey}),
            "inchikey": inchikey,
        }

    # Offer EVERY compound, including the held-out ones, exactly as a fresh
    # acquisition would arrive. The counts below are then a measurement of the
    # boundary rather than an assertion that it was respected.
    offered = [_offer(k, rs) for k, rs in sorted(by_compound.items())]
    screen = screen_records(offered, registry)
    admitted = {str(r["inchikey"]) for r in screen["admitted"]}
    quarantined = {str(offered[int(r["record_index"])]["inchikey"]) for r in screen["quarantined"]}
    held_out = {k for k, v in membership.items() if v == "test"}
    wrongly_admitted = sorted(admitted & held_out)

    exact: dict[str, dict[str, Any]] = {}
    for key, rs in sorted(by_compound.items()):
        for row in rs:
            if row.get("stratum_id") == stratum and row.get("exact"):
                exact.setdefault(key, row)
                break

    train = {k: v for k, v in exact.items() if membership.get(k) == "train" and k in admitted}
    # The Track A fit set above is the exactly measured rows of the primary
    # stratum. The permitted *library* is wider: every admitted train-partition
    # compound, whatever endpoint it was measured on. Track B's retrieval control
    # searches this, and the common novelty exclusion set is built from it, so a
    # generator cannot qualify by re-proposing a training compound that happened
    # to be measured on a different endpoint.
    library = {
        k: rs[0] for k, rs in sorted(by_compound.items()) if membership.get(k) == "train" and k in admitted
    }
    evaluation = {
        k: v for k, v in exact.items() if membership.get(k) == EVALUATED_PARTITION and k in admitted
    }
    pool = sorted(evaluation)
    labels = {k: float(evaluation[k]["p_value"]) for k in pool}

    calibration = _json(CONFIG_DIR / "calibration.json")
    penalty = float(calibration["track_a"]["failure_penalty"])

    return {
        "stratum_id": stratum,
        "train": train,
        "library": library,
        "evaluation": evaluation,
        "pool": pool,
        "labels": labels,
        "family_of": family_of,
        "failure_penalty": penalty,
        "split_hash": str(registry.get("split_hash") or manifest.get("split_hash") or ""),
        "data_hash": str(manifest.get("data_hash") or ""),
        "boundary": {
            "n_offered": screen["n_screened"],
            "n_admitted": screen["n_admitted"],
            "n_quarantined": screen["n_quarantined"],
            "n_held_out_offered": len(held_out),
            "n_held_out_admitted": len(wrongly_admitted),
            "n_held_out_quarantined": len(quarantined & held_out),
            "evaluated_partition": EVALUATED_PARTITION,
            "note": (
                "every compound this arm could have fitted on was offered to Step 6's "
                "protected registry first; n_held_out_admitted is the leak count and must "
                "be zero. No arm in this step reads, fits or scores the held-out partition."
            ),
        },
    }


def _regression_result(
    *,
    manifest: Mapping[str, Any],
    prepared: Mapping[str, Any],
    predictions: Mapping[str, Any],
    model: Mapping[str, Any],
    tuning: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> Record:
    from drug_agent.selfevolve.evaluation.target_benchmark import (
        evaluator_hash,
        regression_endpoint,
    )

    endpoint = regression_endpoint(
        pool=list(prepared["pool"]),
        labels=dict(prepared["labels"]),
        predictions=dict(predictions),
        failure_penalty=float(prepared["failure_penalty"]),
    )
    return {
        "contract_version": "1.0.0",
        "status": endpoint["status"],
        "baseline_id": str(manifest["baseline_id"]),
        "arm": str(manifest["arm"]),
        "track": str(manifest["track"]),
        "primary_metric": "mae_pactivity",
        "primary_value": endpoint["mae"],
        "coverage": float(endpoint["coverage"]),
        "n_pool": int(endpoint["n_pool"]),
        "n_scored": int(endpoint["n_scored"]),
        "n_failed": int(endpoint["n_failed"]),
        "evaluated_partition": EVALUATED_PARTITION,
        "pool_hash": pool_hash(prepared["pool"]),
        "train_pool_hash": pool_hash(prepared["train"]),
        "n_train": len(prepared["train"]),
        "stratum_id": str(prepared["stratum_id"]),
        "boundary": dict(prepared["boundary"]),
        "endpoint": endpoint,
        "model": dict(model),
        "tuning": dict(tuning),
        "cost": dict(cost),
        "evaluator_hash": evaluator_hash(),
        "split_hash": str(prepared["split_hash"]),
        "data_hash": str(prepared["data_hash"]),
        "llm": _llm_usage(),
    }


#: The modules that actually reach a language-model provider. Importing
#: ``drug_agent.llm.config`` or ``drug_agent.llm.policy`` is not talking to a
#: model — the evaluation kernel's integrity manifest pulls them in — so a check
#: that treated any ``drug_agent.llm.*`` import as "an LLM was used" would cry
#: wolf on every arm and be ignored by the third one.
LLM_CALL_SITES: tuple[str, ...] = (
    "drug_agent.llm.ollama",
    "drug_agent.llm.bridge",
    "drug_agent.llm.harness",
    "anthropic",
    "openai",
    "ollama",
)


def _llm_usage() -> Record:
    """Observed, not promised: what LLM machinery this process actually loaded."""
    imported = sorted(name for name in sys.modules if name.startswith("drug_agent.llm"))
    call_sites = sorted(name for name in LLM_CALL_SITES if name in sys.modules)
    return {
        "modules_imported": imported,
        "provider_call_sites_imported": call_sites,
        "reached_a_provider": bool(call_sites),
        "calls_observed": 0 if not call_sites else None,
        "evidence_class": "executed" if not call_sites else "not tested",
        "note": (
            "An arm is a manifest-driven computation: it proposes nothing, plans nothing and "
            "explains nothing. `modules_imported` is non-empty because the evaluation "
            "kernel's integrity manifest imports drug_agent.llm.policy to hash it; no module "
            "that can reach a provider is loaded, so no call can have been made. If "
            "`reached_a_provider` were true the call count would be unknown, not zero."
        ),
    }


def _cost(
    started: float, *, retries: int = 0, failures: int = 0, extra: Mapping[str, Any] | None = None
) -> Record:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_seconds": round(time.time() - started, 3),
        "cpu_user_seconds": round(usage.ru_utime + children.ru_utime, 3),
        "cpu_system_seconds": round(usage.ru_stime + children.ru_stime, 3),
        "max_rss_kb": int(max(usage.ru_maxrss, children.ru_maxrss)),
        "gpu_seconds": None,
        "gpu_telemetry": "UNAVAILABLE",
        "retries": int(retries),
        "failures": int(failures),
        **dict(extra or {}),
    }


# ---------------------------------------------------------------------------
# Track A · B0 — the frozen parent package's own activity model
# ---------------------------------------------------------------------------
def arm_a_b0_parent_rf(manifest: Mapping[str, Any], output: Path) -> Record:
    """``model.activity`` exactly as the parent package ships it.

    This is B0: the comparator every later evolution claim is measured against.
    Its hyperparameters are the shipped defaults and are *not* tuned here — that
    is what B2 is for, and conflating them would delete the distinction Appendix
    A §4 draws between a parent and a conventionally adapted parent.
    """
    from drug_agent.selfevolve.target.api import endpoint_result
    from drug_agent.tools.sklearn_tool import BaselineActivityModel

    prep = manifest["preparation"]
    prepared = prepare_track_a(prep)
    started = time.time()

    model = BaselineActivityModel(
        endpoint=str(prep["endpoint_label"]),
        n_estimators=int(prep["n_estimators"]),
        max_depth=None if prep["max_depth"] is None else int(prep["max_depth"]),
        seed=int(prep["seed"]),
    )
    train_smiles = [str(r["molecule"]["standard_smiles"]) for r in prepared["train"].values()]
    train_y = [float(r["p_value"]) for r in prepared["train"].values()]
    model.fit(train_smiles, train_y)

    predictions: dict[str, Any] = {}
    for key in prepared["pool"]:
        smiles = str(prepared["evaluation"][key]["molecule"]["standard_smiles"])
        try:
            out = model.predict_one(smiles)
        except Exception as exc:  # a refusal is a failure with a reason, never a number
            predictions[key] = endpoint_result("FAILED", None)
            print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        predictions[key] = endpoint_result("SUCCEEDED", float(out.value))

    return _regression_result(
        manifest=manifest,
        prepared=prepared,
        predictions=predictions,
        model={
            "kind": "parent package model.activity",
            "backend": "sklearn-rf",
            "model_version": model.model_version,
            "capability_id": BaselineActivityModel.capability_id,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "seed": model.seed,
            "fitted_on": "train partition only, after the exclusion boundary",
        },
        tuning={
            "n_configurations_evaluated": 0,
            "budget": dict(manifest["tuning_budget"]),
            "note": "B0 is the frozen parent: its settings are the shipped defaults, untuned",
            "failures": [],
        },
        cost=_cost(started),
    )


# ---------------------------------------------------------------------------
# Track A · B1 — the simple similarity/fingerprint baseline
# ---------------------------------------------------------------------------
def arm_a_b1_similarity(manifest: Mapping[str, Any], output: Path) -> Record:
    """1-nearest-neighbour Morgan Tanimoto: Appendix A §4's simple target baseline.

    Step 7 already measured that this family of model has essentially no signal
    on this target — a label-shuffle control on the real development labels gave
    p = 0.313 while an oracle positive control on the same labels fired at
    p = 0.005. That is a statement about the predictor, and it is reported as the
    measured baseline it is. It is emphatically *not* a reason to tune B1 until
    it looks respectable: the declared tuning budget for this arm is zero
    configurations and the manifest says so before the number exists.
    """
    from rdkit import Chem, DataStructs

    from drug_agent.selfevolve.target.api import endpoint_result

    prep = manifest["preparation"]
    prepared = prepare_track_a(prep)
    started = time.time()

    generator = _fingerprint_generator(int(prep["morgan_radius"]), int(prep["morgan_bits"]))

    def fingerprint(smiles: str) -> Any:
        mol = Chem.MolFromSmiles(smiles)
        return generator.GetFingerprint(mol) if mol is not None else None

    train_fps: list[tuple[str, Any, float]] = []
    n_unparseable = 0
    for key, row in sorted(prepared["train"].items()):
        fp = fingerprint(str(row["molecule"]["standard_smiles"]))
        if fp is None:
            n_unparseable += 1
            continue
        train_fps.append((key, fp, float(row["p_value"])))

    predictions: dict[str, Any] = {}
    neighbour_similarity: list[float] = []
    for key in prepared["pool"]:
        fp = fingerprint(str(prepared["evaluation"][key]["molecule"]["standard_smiles"]))
        if fp is None or not train_fps:
            predictions[key] = endpoint_result("FAILED", None)
            continue
        best = max(train_fps, key=lambda t: DataStructs.TanimotoSimilarity(fp, t[1]))
        neighbour_similarity.append(DataStructs.TanimotoSimilarity(fp, best[1]))
        predictions[key] = endpoint_result("SUCCEEDED", best[2])

    return _regression_result(
        manifest=manifest,
        prepared=prepared,
        predictions=predictions,
        model={
            "kind": f"1-nearest-neighbour Morgan r{prep['morgan_radius']}/{prep['morgan_bits']} Tanimoto",
            "backend": "rdkit",
            "n_train_fingerprints": len(train_fps),
            "n_train_unparseable": n_unparseable,
            "mean_nearest_neighbour_similarity": (
                round(statistics.fmean(neighbour_similarity), 6) if neighbour_similarity else None
            ),
            "fitted_on": "train partition only, after the exclusion boundary",
        },
        tuning={
            "n_configurations_evaluated": 0,
            "budget": dict(manifest["tuning_budget"]),
            "note": (
                "declared zero before the number existed. Step 7 measured this predictor "
                "family at p = 0.313 under a label-shuffle control on the real development "
                "labels; tuning it until it looked respectable would be exactly the move the "
                "declared budget exists to forbid"
            ),
            "failures": [],
        },
        cost=_cost(started),
    )


# ---------------------------------------------------------------------------
# Track A · B2 — conventional adaptation of the same starting model
# ---------------------------------------------------------------------------
def _grouped_folds(
    keys: Sequence[str], family_of: Mapping[str, str], *, n_folds: int, seed: int
) -> list[list[str]]:
    """Family-level folds, so a fold boundary cannot cut a structural family.

    The same rule Step 6 used for the split itself, applied inside the training
    partition. A random compound-level fold would let an analogue of a validation
    compound sit in the fold's own training half and make every configuration
    look better than it is.
    """
    from drug_agent.util import stable_hash_int

    families: dict[str, list[str]] = {}
    for key in keys:
        families.setdefault(str(family_of.get(key) or f"singleton:{key}"), []).append(key)
    ordered = sorted(families, key=lambda f: (stable_hash_int(f"{seed}:{f}"), f))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for index, family in enumerate(ordered):
        folds[index % n_folds].extend(sorted(families[family]))
    return folds


def arm_a_b2_standard_adaptation(manifest: Mapping[str, Any], output: Path) -> Record:
    """The same starting model and data, adapted the conventional way.

    "Conventional adaptation" here is what a competent practitioner does with a
    RandomForest and a new target: refit it on the target's own permitted
    training data and select hyperparameters by grouped cross-validation *inside
    that training partition*. The development pool this arm is scored on is never
    used to select anything, and the held-out partition is never touched at all.

    The grid, the fold count and the wall-clock ceiling are in the manifest and
    were declared before the first configuration was fitted. Every configuration
    that was evaluated — including any that failed — is reported, so the tuning
    effort spent on the baseline is on the record next to the baseline's number.
    """
    from drug_agent.selfevolve.target.api import endpoint_result
    from drug_agent.tools.sklearn_tool import BaselineActivityModel

    prep = manifest["preparation"]
    budget = manifest["tuning_budget"]
    prepared = prepare_track_a(prep)
    started = time.time()

    keys = sorted(prepared["train"])
    smiles_of = {k: str(prepared["train"][k]["molecule"]["standard_smiles"]) for k in keys}
    y_of = {k: float(prepared["train"][k]["p_value"]) for k in keys}
    folds = _grouped_folds(keys, prepared["family_of"], n_folds=int(prep["cv_folds"]), seed=int(prep["seed"]))

    grid: list[Record] = []
    for n_estimators in prep["grid"]["n_estimators"]:
        for max_depth in prep["grid"]["max_depth"]:
            grid.append({"n_estimators": int(n_estimators), "max_depth": max_depth})
    max_configs = int(budget["max_configurations"])
    max_seconds = float(budget["max_wall_seconds"])
    if len(grid) > max_configs:
        raise SystemExit(
            f"the declared grid has {len(grid)} configurations but the declared tuning "
            f"budget allows {max_configs}; widen the budget in the manifest deliberately "
            f"rather than overspending it silently"
        )

    trials: list[Record] = []
    failures: list[Record] = []
    exhausted = False
    for config in grid:
        if time.time() - started > max_seconds:
            exhausted = True
            failures.append({"config": config, "reason": "TUNING_WALL_BUDGET_EXHAUSTED"})
            continue
        errors: list[float] = []
        trial_started = time.time()
        try:
            for index, fold in enumerate(folds):
                holdout = set(fold)
                fit_keys = [k for k in keys if k not in holdout]
                if not fit_keys or not fold:
                    continue
                model = BaselineActivityModel(
                    endpoint=str(prep["endpoint_label"]),
                    n_estimators=config["n_estimators"],
                    max_depth=None if config["max_depth"] is None else int(config["max_depth"]),
                    seed=int(prep["seed"]) + index,
                )
                model.fit([smiles_of[k] for k in fit_keys], [y_of[k] for k in fit_keys])
                for key in fold:
                    predicted = float(model.predict_one(smiles_of[key]).value)
                    errors.append(abs(predicted - y_of[key]))
        except Exception as exc:
            failures.append({"config": config, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        trials.append(
            {
                "config": config,
                "cv_mae": round(statistics.fmean(errors), 6) if errors else None,
                "n_cv_predictions": len(errors),
                "seconds": round(time.time() - trial_started, 2),
            }
        )

    scored = [t for t in trials if t["cv_mae"] is not None]
    if not scored:
        raise SystemExit("no configuration produced a cross-validated score; nothing to select")
    # Lower MAE wins; ties break on the smaller model, then on the declared order,
    # so the selection rule cannot depend on dictionary iteration.
    best = min(
        scored,
        key=lambda t: (
            t["cv_mae"],
            t["config"]["n_estimators"],
            math.inf if t["config"]["max_depth"] is None else t["config"]["max_depth"],
        ),
    )
    selected = best["config"]

    model = BaselineActivityModel(
        endpoint=str(prep["endpoint_label"]),
        n_estimators=int(selected["n_estimators"]),
        max_depth=None if selected["max_depth"] is None else int(selected["max_depth"]),
        seed=int(prep["seed"]),
    )
    model.fit([smiles_of[k] for k in keys], [y_of[k] for k in keys])

    predictions: dict[str, Any] = {}
    for key in prepared["pool"]:
        smiles = str(prepared["evaluation"][key]["molecule"]["standard_smiles"])
        try:
            predictions[key] = endpoint_result("SUCCEEDED", float(model.predict_one(smiles).value))
        except Exception as exc:
            predictions[key] = endpoint_result("FAILED", None)
            print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr)

    return _regression_result(
        manifest=manifest,
        prepared=prepared,
        predictions=predictions,
        model={
            "kind": "parent model.activity, conventionally adapted",
            "backend": "sklearn-rf",
            "model_version": model.model_version,
            "capability_id": BaselineActivityModel.capability_id,
            "selected_config": selected,
            "seed": model.seed,
            "fitted_on": "train partition only, after the exclusion boundary",
        },
        tuning={
            "n_configurations_evaluated": len(trials),
            "n_configurations_failed": len(failures),
            "budget": dict(budget),
            "budget_exhausted": exhausted,
            "selection_rule": (
                f"lowest mean absolute error under {prep['cv_folds']}-fold cross-validation "
                f"grouped by structural family, computed on the TRAIN partition only; ties "
                f"break to the smaller forest"
            ),
            "trials": trials,
            "failures": failures,
            "note": (
                "the development pool this arm is scored on selected nothing; the held-out "
                "partition was neither read nor scored"
            ),
        },
        cost=_cost(started, failures=len(failures)),
    )


# ---------------------------------------------------------------------------
# Track A · B6 — Chemprop, the external competitor Appendix A §4 names
# ---------------------------------------------------------------------------
def arm_a_b6_chemprop(manifest: Mapping[str, Any], output: Path) -> Record:
    """A directed message-passing network fitted to the same permitted rows.

    Chemprop ships no pretrained ZAK model and no pretrained potency panel, so
    "is Chemprop available" and "is there a trained checkpoint for this endpoint"
    are two different questions; this arm answers the second by training one, on
    the train partition only, under the epoch budget its manifest declares.
    """
    from drug_agent.selfevolve.target.api import endpoint_result
    from drug_agent.tools.properties.chemprop_tool import ChempropAdapter

    prep = manifest["preparation"]
    budget = manifest["tuning_budget"]
    prepared = prepare_track_a(prep)
    started = time.time()

    checkpoint_dir = Path(prep.get("checkpoint_dir") or (output / "chemprop"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter = ChempropAdapter(checkpoint_dir=checkpoint_dir)
    adapter.require_available()

    keys = sorted(prepared["train"])
    checkpoint = adapter.train(
        endpoint=str(prep["endpoint_label"]),
        smiles=[str(prepared["train"][k]["molecule"]["standard_smiles"]) for k in keys],
        y=[float(prepared["train"][k]["p_value"]) for k in keys],
        epochs=int(prep["epochs"]),
        val_fraction=float(prep["val_fraction"]),
        batch_size=int(prep["batch_size"]),
        seed=int(prep["seed"]),
    )

    pool = list(prepared["pool"])
    panels = adapter.predict(
        [str(prepared["evaluation"][k]["molecule"]["standard_smiles"]) for k in pool],
        endpoint=str(prep["endpoint_label"]),
        candidate_ids=pool,
    )
    predictions: dict[str, Any] = {}
    for key, panel in zip(pool, panels, strict=True):
        values = [v for v in panel.endpoints.values() if isinstance(v, (int, float))]
        if not panel.ok or not values:
            predictions[key] = endpoint_result("FAILED", None)
            continue
        predictions[key] = endpoint_result("SUCCEEDED", float(values[0]))

    path = Path(checkpoint.path)
    return _regression_result(
        manifest=manifest,
        prepared=prepared,
        predictions=predictions,
        model={
            "kind": "Chemprop D-MPNN",
            "backend": "chemprop",
            "capability_id": adapter.capability_id,
            "model_version": adapter.version,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path) if path.is_file() else None,
            "n_train": checkpoint.n_train,
            "n_val": checkpoint.n_val,
            "epochs": int(prep["epochs"]),
            "seed": int(prep["seed"]),
            "fitted_on": "train partition only, after the exclusion boundary",
        },
        tuning={
            "n_configurations_evaluated": 1,
            "budget": dict(budget),
            "note": (
                "one declared configuration, no architecture search. Appendix A §4 requires "
                "the baseline's tuning effort to be recorded, and the recorded effort here is "
                "a single fit at the manifest's epoch budget"
            ),
            "failures": [],
        },
        cost=_cost(started, extra={"epochs": int(prep["epochs"])}),
    )


# ---------------------------------------------------------------------------
# Track B — the frozen independent evaluator over a bounded portfolio
# ---------------------------------------------------------------------------
def _common_exclusion_set(qualification: Mapping[str, Any], prepared: Mapping[str, Any]) -> Record:
    """The frozen novelty exclusion set, extended by this step's training pool.

    Step 7's qualification record says in its own notes that Step 8 adds the
    training pool to the common exclusion set. The union is computed once and
    applied identically to every Track B arm, so it can only make qualification
    harder and cannot advantage one arm over another. The parent record is not
    edited: the derived qualification carries both hashes.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import (
        NOVELTY_DIGEST_ALGORITHM,
        novelty_digests,
        qualification_hash,
    )

    parent_novelty = dict(qualification.get("novelty") or {})
    base = list(parent_novelty.get("exclusion_smiles") or [])
    base_digest_rule = parent_novelty.get("exclusion_digests") or {}
    base_digests = list(base_digest_rule.get("digests") or ())
    training = sorted({str(row["molecule"]["standard_smiles"]) for row in prepared["library"].values()})

    # The parent may carry its exclusion set as plaintext, as digests, or as
    # both, and the union has to be taken in whichever space the parent uses:
    # unioning plaintext into a digest set would silently drop the digests, and
    # the novelty gate would then stop enforcing the thing it exists to enforce.
    if base_digests:
        union_digests = sorted(set(base_digests) | set(novelty_digests(training)))
        union = sorted(set(base))  # empty in the digest-carried record; kept for the counts below
        n_enforced = len(union) + len(union_digests)
    else:
        union_digests = []
        union = sorted(set(base) | set(training))
        n_enforced = len(union)

    derived = json.loads(json.dumps(qualification))
    derived["novelty"] = {
        **parent_novelty,
        "n_excluded": n_enforced,
        "exclusion_smiles": union,
        **(
            {
                "exclusion_digests": {
                    "algorithm": NOVELTY_DIGEST_ALGORITHM,
                    "n": len(union_digests),
                    "digests": union_digests,
                }
            }
            if union_digests
            else {}
        ),
        "extended_by": (
            "Step 8 training pool: every admitted train-partition compound, after the "
            "exclusion boundary, on any endpoint"
        ),
        "parent_qualification_hash": qualification.get("qualification_hash"),
    }
    derived.pop("qualification_hash", None)
    derived["qualification_hash"] = qualification_hash(derived)
    return {
        "qualification": derived,
        "n_base": len(base) + len(base_digests),
        "n_training": len(training),
        "n_union": n_enforced,
        # union_hash keeps the promise the frozen protocol makes about it. protocol.json records, as
        # MEASURED, that "the union_hash is a content address over standardized MOLECULE IDENTITIES
        # and does not move when the evaluator's address moves", and that re-running reproduces
        # sha256:5ec21d5a... EXACTLY. Quietly re-pointing the same key at a hash of the DIGESTS would
        # have made that sentence false and left a reader unable to tell a carrier change from a set
        # change. So the key is computed over identities when identities are there, and is absent
        # rather than redefined when they are not.
        "union_hash": (None if union_digests else sha256_text("\n".join(union))),
        "union_digest_hash": (sha256_text("\n".join(union_digests)) if union_digests else None),
        "union_hash_basis": (
            "UNAVAILABLE: the exclusion set is carried as digests, so no address over standardized "
            "molecule identities can be computed here. union_digest_hash is an address over the "
            "DIGESTS and is NOT comparable with a union_hash from the plaintext era"
            if union_digests
            else "standardized molecule identities"
        ),
        "parent_qualification_hash": qualification.get("qualification_hash"),
        "derived_qualification_hash": derived["qualification_hash"],
    }


def _dock_portfolio(
    molecules: Sequence[tuple[str, str]],
    *,
    scorer: Mapping[str, Any],
    workdir: Path,
) -> list[Record]:
    """Score every submitted molecule with the registered independent scorer."""
    import importlib.util

    from drug_agent.selfevolve.evaluation.target_benchmark import most_favourable
    from drug_agent.selfevolve.target.api import endpoint_result
    from drug_agent.tools.vina_tool import DockingAdapter

    spec = importlib.util.spec_from_file_location(
        "zak_redock", REPO / "scripts" / "research" / "dock_zak_reference_ligand.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise SystemExit("could not load the receptor/box preparation module")
    redock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(redock)
    inputs = redock.prepare_inputs(workdir)

    adapter = DockingAdapter()
    adapter.require_available()

    # --- why a thread cap is read from the environment and not from the scorer ----------------
    # vina's default is every core it can see, so two docking passes on one host do not run twice
    # as fast -- they run about nine times slower EACH. Measured on this 12-core host on
    # 2026-09-25: 16.2 s per molecule with the machine idle, 146.5 s per molecule with four other
    # passes running, on the same nine molecules with byte-identical results.
    #
    # The cap is safe because vina's SCORE does not depend on it. Measured the same day, three
    # molecules at --cpu unset, 4 and 2, at the frozen seed 42 and exhaustiveness 8: every pose
    # score identical to the last digit, recorded in 0925-pfm-evolve/vina-cpu-equivalence.json.
    # So this changes scheduling and nothing else.
    #
    # It is read from DRUGAGENT_VINA_CPU rather than added to the scorer record on purpose. The
    # scorer lives in calibration.json, which is content-addressed; putting a scheduling knob in it
    # would move a hash that every prior measurement declares, to express something that provably
    # does not change a measurement. Unset means the previous behaviour exactly.
    cpu_cap: int | None = None
    raw_cap = os.environ.get("DRUGAGENT_VINA_CPU")
    if raw_cap:
        try:
            cpu_cap = max(1, int(raw_cap))
        except ValueError:
            cpu_cap = None

    rows: list[Record] = []
    for molecule_id, smiles in molecules:
        started = time.time()
        try:
            poses = adapter.dock(
                candidate_id=molecule_id,
                smiles=smiles,
                receptor_pdb=inputs["receptor_pdb"],
                receptor_id=inputs["receptor_id"],
                center=inputs["center"],
                size=inputs["size"],
                n_poses=int(scorer["n_poses"]),
                seed=int(scorer["seed"]),
                exhaustiveness=int(scorer["exhaustiveness"]),
                cpu=cpu_cap,
            )
        except Exception as exc:
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "structure_score": endpoint_result("FAILED", None),
                    "pose_molblock": "",
                    "properties": {},
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - started, 2),
                }
            )
            continue
        scores = [float(p.score) for p in poses if p.score is not None]
        if not scores:
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "structure_score": endpoint_result("FAILED", None),
                    "pose_molblock": "",
                    "properties": {},
                    "error": "the engine returned no scored pose",
                    "seconds": round(time.time() - started, 2),
                }
            )
            continue
        best = most_favourable(scores, str(scorer["metric"]))
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "structure_score": endpoint_result("SUCCEEDED", best),
                "pose_molblock": poses[0].pose_ref,
                "properties": _descriptors(smiles),
                "engine_version": poses[0].engine_version,
                "error": None,
                "seconds": round(time.time() - started, 2),
            }
        )
    return rows


def _generation_quality_endpoints(molecules: Sequence[tuple[str, str]]) -> dict[str, Record]:
    """``herg`` and ``sa_score`` for a whole portfolio, as endpoint records.

    Step 15B widened the Track B gate set, and a gate the evaluation path cannot
    feed is a gate that silently fails every molecule. These are batched, because
    ADMET-AI's cost is dominated by per-call Lightning setup rather than by batch
    size, and a molecule the panel could not reach is an
    ``endpoint_result("FAILED", None)`` — never a favourable number and never a
    missing key, because ``evaluate_portfolio`` reads an absent endpoint as
    ``NOT_EVALUATED`` and refuses the molecule either way.
    """
    from drug_agent.selfevolve.target.api import endpoint_result
    from drug_agent.tools.safety_tool import SafetyAdapter
    from drug_agent.tools.synthesis_tool import SynthesisAdapter

    synth = SynthesisAdapter()
    safety = SafetyAdapter()
    if safety.panel.available():
        safety.prefetch([s for _mid, s in molecules])
    out: dict[str, Record] = {}
    for molecule_id, smiles in molecules:
        sa: float | None = None
        if synth.available():
            with contextlib.suppress(Exception):
                sa = synth.assess(molecule_id, smiles).sa_score
        herg: float | None = None
        with contextlib.suppress(Exception):
            profile = safety.assess(molecule_id, smiles)
            endpoint = next((e for e in profile.endpoints if e.endpoint == "hERG"), None)
            herg = None if endpoint is None else endpoint.score
        out[molecule_id] = {
            "herg": endpoint_result("SUCCEEDED", float(herg))
            if herg is not None
            else endpoint_result("FAILED", None),
            "sa_score": endpoint_result("SUCCEEDED", float(sa))
            if sa is not None
            else endpoint_result("FAILED", None),
        }
    return out


def _descriptors(smiles: str) -> Record:
    """The molecular properties the frozen qualification's limits read."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    from drug_agent.tools.rdkit_tool import Cheminformatics

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    alerts = 0
    try:
        chem = Cheminformatics()
        alerts = len(chem.structural_alerts_from_smiles(smiles))
    except Exception:
        alerts = 0
    descriptors: Record = {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
    }
    # qed joins the descriptors because the operator's PFM standards gate on it, and a property
    # limit reads ONLY from this mapping -- `_property_passes` treats an absent key as a failure,
    # so a qed limit against descriptors without qed would refuse every molecule.
    #
    # It is a property limit rather than an endpoint gate because `score_orientation()` raises for
    # any metric absent from SCORE_ORIENTATION, `qed` is absent, and that table is inside the
    # GUARDED evaluator: registering it there would move the evaluator address every campaign
    # record declares. The property-limit path needs no orientation and supports the strict `>` the
    # operator wrote.
    #
    # Additive by construction: the parent rule's limits name mw, logp and n_structural_alerts, so
    # adding a key changes no measurement taken before the PFM standards were applied.
    # NOT suppressed. The comment above says a qed limit against descriptors without qed refuses
    # every molecule -- and a swallowed exception here is the one thing that reproduces exactly that,
    # per molecule, recorded as a genuine properties breach and indistinguishable from a real
    # low-QED result. mw and logp above are computed unguarded; so is this.
    from rdkit.Chem import QED

    descriptors["qed"] = float(QED.qed(mol))
    return {
        "descriptors": descriptors,
        "n_structural_alerts": int(alerts),
    }


def _portfolio_result(
    *,
    manifest: Mapping[str, Any],
    prepared: Mapping[str, Any],
    submitted: Sequence[tuple[str, str]],
    generator: Mapping[str, Any],
    output: Path,
    qualification_rule: str,
) -> Record:
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluate_portfolio, evaluator_hash

    started = time.time()
    prep = manifest["preparation"]
    qualification = _json(CONFIG_DIR / "qualification.json")
    # A bare read of a record this code may not have been built for. CONFIG_DIR comes from
    # DRUGAGENT_REPO_ROOT, which worker.py points at the PARENT checkout, while a candidate workspace
    # pins its own src/ at the commit it was created from -- so the record can be newer than the
    # evaluator reading it. Measured consequence of exactly that pairing: an evaluator predating the
    # 2026-09-30 novelty re-freeze reads the newer record's exclusion set as empty and the novelty
    # gate admits everything, silently, 72 qualifying scaffolds against the matched pairing's 60.
    # Refuse instead.
    from drug_agent.selfevolve.target.generation_sweep import QUALIFICATION_HASH
    from drug_agent.selfevolve.target.novelty_guard import (
        require_enforceable_novelty,
        require_qualification_address,
    )

    require_qualification_address(qualification, QUALIFICATION_HASH, what="Track B portfolio scoring")
    require_enforceable_novelty(qualification, what="Track B portfolio scoring")
    common = _common_exclusion_set(qualification, prepared)
    derived = common["qualification"]
    if qualification_rule == "LIBRARY_RETRIEVAL":
        # Appendix A §3.3: a declared library-retrieval benchmark uses a separate
        # qualification rule and must NOT be mixed with the de novo primary
        # table. Novelty against the training pool is the rule that a retrieval
        # arm cannot satisfy by construction, so it is reported both ways and
        # the de novo number is the one that stays comparable.
        retrieval = json.loads(json.dumps(derived))
        retrieval["novelty"] = {**derived["novelty"], "exclusion_smiles": [], "n_excluded": 0}
        # Both carriers, not only the plaintext one: the derived record may hold
        # the exclusion set as digests, and leaving those in place beside
        # n_excluded 0 is a record that contradicts itself — which the evaluator
        # now refuses outright rather than scoring against.
        retrieval["novelty"].pop("exclusion_digests", None)
        retrieval.pop("qualification_hash", None)
        from drug_agent.selfevolve.evaluation.target_benchmark import qualification_hash

        retrieval["qualification_hash"] = qualification_hash(retrieval)
    else:
        retrieval = None

    # Absolute: the docking backend shells out to obabel and vina, which do not
    # necessarily inherit this process's working directory, and a relative
    # receptor path silently becomes "0 molecules converted".
    workdir = (output / "docking").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rows = _dock_portfolio(list(submitted), scorer=prep["scorer"], workdir=workdir)
    # Step 15B's generation-quality gates need their endpoints on every row. The
    # qualification record decides whether they are applied; this only makes sure
    # a gate that IS applied is applied to a number rather than to an absence.
    endpoints = _generation_quality_endpoints(list(submitted))
    for row in rows:
        row["endpoints"] = endpoints.get(str(row["molecule_id"]), {})
    (output / "portfolio_rows.json").write_text(json.dumps(rows, indent=2))

    evaluated = evaluate_portfolio(rows, qualification=derived)
    secondary = evaluate_portfolio(rows, qualification=retrieval) if retrieval else None

    n_failed = sum(1 for r in rows if r["structure_score"]["status"] != "SUCCEEDED")
    return {
        "contract_version": "1.0.0",
        "status": portfolio_status(n_evaluated=int(evaluated["n_evaluated"]), n_scoring_failed=n_failed),
        "baseline_id": str(manifest["baseline_id"]),
        "arm": str(manifest["arm"]),
        "track": str(manifest["track"]),
        "primary_metric": "qualifying_scaffolds",
        "primary_value": float(evaluated["qualifying_scaffolds"]),
        "qualification_rule": qualification_rule,
        "track_b_name": str(derived.get("track_b_name")),
        "n_submitted": len(submitted),
        "n_evaluated": int(evaluated["n_evaluated"]),
        "n_qualifying_compounds": int(evaluated["n_qualifying_compounds"]),
        "n_scoring_failed": n_failed,
        "funnel": dict(evaluated["funnel"]),
        "evaluated_partition": EVALUATED_PARTITION,
        "portfolio_cap": int(prep["portfolio_cap"]),
        "portfolio_cap_note": (
            "a DECLARED SMALL PROBE, bounded by local process limits because Step 9's "
            "integrated budget service does not exist yet. Appendix A §3.3's confirmatory "
            "cap is 200 unique molecules per arm per seed and this is not that run."
        ),
        "secondary_retrieval_evaluation": (
            {
                "qualifying_scaffolds": secondary["qualifying_scaffolds"],
                "n_qualifying_compounds": secondary["n_qualifying_compounds"],
                "rule": (
                    "novelty gate removed. Appendix A §3.3 keeps a declared library-retrieval "
                    "benchmark on a separate rule and forbids mixing it with the de novo "
                    "primary table; this number is reported apart from primary_value and is "
                    "never comparable to a de novo arm's count."
                ),
            }
            if secondary
            else None
        ),
        "generator": dict(generator),
        "common_exclusion": {k: v for k, v in common.items() if k != "qualification"},
        "boundary": dict(prepared["boundary"]),
        "scorer": dict(prep["scorer"]),
        "cost": _cost(started, failures=n_failed),
        "evaluator_hash": evaluator_hash(),
        "split_hash": str(prepared["split_hash"]),
        "data_hash": str(prepared["data_hash"]),
        "llm": _llm_usage(),
    }


def arm_b_b0_pfm_frozen(manifest: Mapping[str, Any], output: Path) -> Record:
    """B0 for Track B: the frozen parent generator, at the real ZAK pocket."""
    from drug_agent.tools.pfm_tool import PFMGeneratorAdapter

    require_uncontaminated("Track B B0 (frozen PFM generator)")
    prep = manifest["preparation"]
    prepared = prepare_track_a(prep)

    pocket = REPO / str(prep["pocket_relpath"])
    if not pocket.is_file():
        raise SystemExit(f"the ZAK pocket is missing at {pocket}")
    adapter = PFMGeneratorAdapter()
    adapter.require_available()
    cap = int(prep["portfolio_cap"])
    batch = adapter.generate(
        n_molecules=int(prep["n_generation_attempts"]),
        pocket={"pdb_path": str(pocket)},
        seed=int(prep["seed"]),
        batch_size=int(prep["batch_size"]),
        max_runtime_seconds=float(prep["generation_timeout_seconds"]),
    )
    unique = _unique_submission(batch, cap)
    return _portfolio_result(
        manifest=manifest,
        prepared=prepared,
        submitted=unique,
        generator={
            "backend": batch.backend,
            "model_id": batch.model_id,
            "model_version": batch.model_version,
            "checkpoint": batch.config.get("checkpoint"),
            "checkpoint_sha256": (
                sha256_file(batch.config["checkpoint"])
                if batch.config.get("checkpoint") and Path(batch.config["checkpoint"]).is_file()
                else None
            ),
            "pocket": str(pocket),
            "pocket_sha256": sha256_file(pocket),
            "n_generation_attempts": int(prep["n_generation_attempts"]),
            "n_raw": batch.n_raw,
            "n_valid": batch.n_valid,
            "n_unique": batch.n_unique,
            "validity": batch.validity,
            "uniqueness": batch.uniqueness,
            "warnings": list(batch.warnings),
            "errors": list(batch.errors),
        },
        output=output,
        qualification_rule="DE_NOVO_PRIMARY",
    )


def arm_b_b6_reinvent4(manifest: Mapping[str, Any], output: Path) -> Record:
    """B6 for Track B: REINVENT4's unconditional prior, sampled as it ships.

    The prior knows nothing about ZAK. That is the honest description of this
    comparison and it is recorded on the arm: an unconditional de novo sampler is
    the external competitor Appendix A §4 names, and steering it at a target is
    the transfer-learning run mode, which this repository has no supported route
    to (see the ``fit`` capability probe).
    """
    from drug_agent.tools.generation.reinvent4_tool import REINVENT4_PRIOR_RELPATH, Reinvent4Adapter

    prep = manifest["preparation"]
    prepared = prepare_track_a(prep)
    adapter = Reinvent4Adapter()
    adapter.require_available()
    batch = adapter.generate(
        n_molecules=int(prep["n_generation_attempts"]),
        seed=int(prep["seed"]),
        batch_size=int(prep["batch_size"]),
        max_runtime_seconds=float(prep["generation_timeout_seconds"]),
    )
    prior = Path(adapter.ext.root or "") / REINVENT4_PRIOR_RELPATH
    unique = _unique_submission(batch, int(prep["portfolio_cap"]))
    return _portfolio_result(
        manifest=manifest,
        prepared=prepared,
        submitted=unique,
        generator={
            "backend": batch.backend,
            "model_id": batch.model_id,
            "model_version": batch.model_version,
            "prior": str(prior),
            "prior_sha256": sha256_file(prior) if prior.is_file() else None,
            "upstream_ref": batch.config.get("upstream_ref"),
            "conditioning": "none (unconditional prior; it has no knowledge of ZAK)",
            "n_generation_attempts": int(prep["n_generation_attempts"]),
            "n_raw": batch.n_raw,
            "n_valid": batch.n_valid,
            "n_unique": batch.n_unique,
            "validity": batch.validity,
            "uniqueness": batch.uniqueness,
            "warnings": list(batch.warnings),
            "errors": list(batch.errors),
        },
        output=output,
        qualification_rule="DE_NOVO_PRIMARY",
    )


def arm_b_b1_analog_retrieval(manifest: Mapping[str, Any], output: Path) -> Record:
    """B1 for Track B: the retrieval / analog-search control.

    Appendix A §4's simple target baseline for a generation track is retrieval,
    not generation: take the permitted library and pick the compounds most like
    the reference ligand. It is the cheapest thing that could work, and it is the
    bar a generator has to clear to have contributed anything.
    """
    from rdkit import Chem, DataStructs

    prep = manifest["preparation"]
    prepared = prepare_track_a(prep)
    target_manifest = _json(DERIVED / "target_manifest.json")
    reference_smiles = str(
        prep.get("reference_smiles")
        or target_manifest.get("reference_ligand", {}).get("standard_smiles")
        or ""
    )
    if not reference_smiles:
        raise SystemExit("no reference ligand SMILES available for the retrieval control")

    generator = _fingerprint_generator(int(prep["morgan_radius"]), int(prep["morgan_bits"]))
    reference = Chem.MolFromSmiles(reference_smiles)
    if reference is None:
        raise SystemExit(f"the reference ligand SMILES did not parse: {reference_smiles!r}")
    reference_fp = generator.GetFingerprint(reference)

    scored: list[tuple[float, str, str]] = []
    for key, row in sorted(prepared["library"].items()):
        smiles = str(row["molecule"]["standard_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(reference_fp, generator.GetFingerprint(mol))
        scored.append((similarity, key, smiles))
    scored.sort(key=lambda t: (-t[0], t[1]))
    cap = int(prep["portfolio_cap"])
    chosen = scored[:cap]
    return _portfolio_result(
        manifest=manifest,
        prepared=prepared,
        submitted=[(key, smiles) for _sim, key, smiles in chosen],
        generator={
            "backend": "rdkit-retrieval",
            "model_id": "baseline.retrieval.tanimoto",
            "model_version": "1.0.0",
            "library": (
                "every admitted train-partition compound on any endpoint, after the exclusion boundary"
            ),
            "n_library": len(scored),
            "reference_ligand_smiles": reference_smiles,
            "selection_rule": (
                f"top {cap} by Morgan r{prep['morgan_radius']}/{prep['morgan_bits']} Tanimoto "
                f"to the reference ligand, ties broken by compound key"
            ),
            "max_similarity": round(chosen[0][0], 6) if chosen else None,
            "min_selected_similarity": round(chosen[-1][0], 6) if chosen else None,
        },
        output=output,
        qualification_rule="LIBRARY_RETRIEVAL",
    )


def _unique_submission(batch: Any, cap: int) -> list[tuple[str, str]]:
    """Deduplicate a generation batch and cap it, keeping generation order."""
    from rdkit import Chem

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for index, molecule in enumerate(batch.molecules):
        smiles = str(getattr(molecule, "smiles", "") or "")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append((f"{batch.backend}_{index:04d}", canonical))
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
ARMS: dict[str, Any] = {
    "A-B0-parent-rf": arm_a_b0_parent_rf,
    "A-B1-similarity-1nn": arm_a_b1_similarity,
    "A-B2-standard-adaptation-rf": arm_a_b2_standard_adaptation,
    "A-B6-chemprop": arm_a_b6_chemprop,
    "B-B0-pfm-frozen": arm_b_b0_pfm_frozen,
    "B-B1-analog-retrieval": arm_b_b1_analog_retrieval,
    "B-B6-reinvent4-prior": arm_b_b6_reinvent4,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest = validate_baseline_manifest(load_manifest(args.manifest))
    if str(manifest["baseline_id"]) != args.arm:
        raise SystemExit(
            f"--arm {args.arm!r} does not match the manifest's baseline_id {manifest['baseline_id']!r}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    # An arm is a real-backend lane. If the session-wide generator substitution
    # ever reached one of these processes, every molecule it produced would be a
    # fixture's output wearing a backend's name.
    contamination = require_uncontaminated(f"baseline arm {args.arm}")

    result = ARMS[args.arm](manifest, args.output)
    result["contamination_check"] = contamination
    result["manifest_hash"] = sha256_text(json.dumps(manifest, sort_keys=True))
    validate_output(result, manifest["output_schema"])
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "baseline_id": result["baseline_id"],
                "status": result["status"],
                "primary_metric": result["primary_metric"],
                "primary_value": result["primary_value"],
            }
        )
    )
    return 0 if result["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
