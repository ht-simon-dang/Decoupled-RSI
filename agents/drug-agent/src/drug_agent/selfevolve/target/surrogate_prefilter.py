"""A *prediction* of the expensive gates, where the lookup is forbidden.

WHAT THIS ADDS TO ``submission_policy``
---------------------------------------
:mod:`drug_agent.selfevolve.target.submission_policy` spends a fixed budget of 150
submissions by filtering an oversampled pool on the gates computable from a SMILES --
``properties``, ``safety``/hERG, ``synthetic_accessibility``/SA. It deliberately
cannot filter on the two gates that decide most of the outcome, ``structure_score``
(vina_affinity, LOWER IS BETTER) and ``geometry``, because both need a docked pose.
That module refuses outright when either reaches its filter, and the refusal is
right: selecting on a gate that needs a pose means submitting the molecules an
expensive measurement already passed, which is a lookup of the answer and not a
prediction.

This module does the other thing. It fits a surrogate on the docking outputs of
molecules from PRIOR runs and uses it to RANK an oversampled pool of molecules that
have never been docked. The ranked molecules contribute no docking output to the
model, at training time or at inference time; that absence is what makes the number
a prediction.

THE LINE, AND HOW IT IS MADE STRUCTURAL
---------------------------------------
The line is not "do not peek". The line is::

    the surrogate may be trained on docking outputs of molecules from PRIOR runs;
    it may NEVER see, as an input feature or as a training label, any docking
    output of a molecule in the pool it is currently ranking.

Convention will not hold that line, so four independent refusals do. All four raise;
none warns.

1. THE FEATURISER IS SMILES-ONLY. :func:`featurise` accepts records and reads
   exactly one key, ``smiles``. A record carrying ``structure_score``,
   ``pose_molblock``, ``posebusters``, ``geometry`` or any other name in
   :data:`DOCKING_DERIVED_KEYS` is refused before a single bit is computed, and a
   caller-supplied extra feature whose NAME contains a docking token
   (``vina``, ``dock``, ``pose``, ``affinity``, ``busters``, ``rmsd``) is refused
   too. Training rows are stripped to ``{"smiles": ...}`` before they reach it, so
   the featuriser never holds a label in the first place.

2. A POOL CARRYING DOCKING OUTPUT CANNOT BE RANKED. :meth:`SurrogatePrefilter.
   score_pool` re-runs the same key check over every member. A molecule that
   already has a ``structure_score`` is a molecule somebody docked; ranking it is
   the lookup, so it is refused rather than scored.

3. A MOLECULE THE MODEL WAS TRAINED ON CANNOT BE RANKED. The fitted model carries
   the SHA-256 digest of every canonical SMILES in its training set. If any pool
   member's digest is in that set, :meth:`SurrogatePrefilter.score_pool` refuses:
   for that molecule the "prediction" would be a memorised label. Dropping the
   overlap is available, but only through :meth:`SurrogatePrefilter.
   drop_trained_overlap`, which makes the removal explicit and counted.

4. A RUN THE MODEL WAS TRAINED ON CANNOT BE RANKED. The model also carries the run
   ids it was fitted on. Molecules inside one run are near-duplicates of one another
   -- same generator, same seed, same pool -- so a model that saw *any* of a run is
   not making a prediction about the rest of it. Pool members declaring a
   ``run_id`` the model trained on are refused.

WHY THE SPLIT IS RUN-LEVEL AND NOT RANDOM
-----------------------------------------
The same near-duplication makes a random train/test split dishonest: a held-out
molecule's siblings sit in the training set and the reported number measures
interpolation between near-copies, not generalisation to the next pool.
:func:`run_level_folds` holds out whole runs and asserts that no run appears on both
sides. Every metric in :func:`evaluate_grouped` is computed that way, and the two
baselines that matter -- random selection, and the cheap-gate prefilter alone -- are
computed on the identical held-out runs so the comparison is like for like.

Holding out the run is necessary but NOT sufficient, and the measurement says so:
33% of the held-out molecules under leave-one-run-out turned out to be the same
structure as a molecule in some training run. Those are dropped, counted, and
reported. Left in, they would have been scored from memory.

WHAT IS REPORTED, AND WHAT WOULD MAKE IT NOT WORTH DEPLOYING
------------------------------------------------------------
Both heads are fitted and both are reported: a regression on ``vina_affinity`` and a
binary classification on "passes ``structure_score`` AND ``geometry``". Neither is
assumed better. A surrogate that does not beat the cheap prefilter on the held-out
runs is not worth deploying, and :func:`counterfactual_replay` is the cheap, honest
way to find that out -- it replays recorded pools whose ground truth is already on
disk, at a fixed submission budget, with no new docking at all.

MEASURED, ON THIS TARGET
------------------------
Corpus: 9,238 docked molecules over 64 runs and 10 generators, 7,336 of them distinct
structures; 1,973 (21.4%) clear both expensive gates. Every row was generated by this
project; nothing held out is read.

Held-out RUNS (8 folds, training duplicates removed, 6,197 molecules scored)::

                                ROC-AUC   P@150   Spearman(affinity)
    classification head            0.918   0.717                0.732
    regression head                0.887   0.706                0.784
    cheap-gate prefilter alone     0.512   0.438                   --
    random                            --   0.243                   --

Held-out GENERATORS (5 folds, the harder test -- the model has never seen the
generator it is ranking) hold up: classification 0.867 ROC-AUC and 0.465 P@150
against the same 0.460 / 0.305 cheap-prefilter baseline.

The cheap prefilter's ROC-AUC of ~0.5 is the finding underneath the rest: passing the
gates a SMILES decides says almost NOTHING about passing the docking gate. The two
filters are close to independent, which is why composing them helps.

Counterfactual replay, budget 150, generator held out of training entirely, over the
recorded molecules of that generator's runs (qualifying compounds)::

    pool                       generation_order   cheap_prefilter   surrogate_rank
    pocketxmol   (1,043)                     83               107              134
    synflownet     (300)                     83               112              137
    reinvent4      (300)                      5                 5                9

reinvent4's 9 is every qualifying molecule in that pool: the surrogate found all of
them, and the ceiling -- not the ranking -- is that arm's problem. Counted as the
registered endpoint, qualifying SCAFFOLDS, the numbers are identical except for
synflownet, where the three rules give 55 / 73 / 93.

Wired into ``submission_policy`` as ``surrogate_rank``, the rule COMPOSES rather than
replaces: cheap gates first, then the surrogate ranks the survivors, then the top N
fill the budget. With a constant surrogate it reduces exactly to
``cheap_gate_prefilter``, because ties are broken by generation order.

WHAT THIS NUMBER IS NOT
-----------------------
It is a replay over molecules that were docked ONCE, by the campaigns that recorded
them. The oversampled pools those campaigns drew from -- 1,914 candidates for runB,
1,985 for runD, 1,920 for runF -- were never docked beyond the 150 each submitted, so
no replay can say what the surrogate would have found in the other ~1,770. The replay
above therefore re-spends a budget over molecules whose outcome is known, which is a
real like-for-like comparison of selection rules and NOT a measured campaign result.
Only a campaign run under ``surrogate_rank`` can produce one of those.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drug_agent.selfevolve.target.submission_policy import EXPENSIVE_GATES

Record = dict[str, Any]

#: Record keys that carry, or are derived from, a docked pose. None of these may
#: reach the featuriser, and a molecule carrying one may not be ranked.
DOCKING_DERIVED_KEYS = frozenset(
    {
        "structure_score",
        "pose_molblock",
        "pose",
        "poses",
        "pose_quality",
        "posebusters",
        "geometry",
        "vina_affinity",
        "docking_score",
        "docked_pose",
        "binding_affinity",
        "affinity",
        "rmsd",
        "engine_version",
    }
)

#: Substrings that make a FEATURE NAME inadmissible. A feature may be named only
#: after a quantity computable from a SMILES; anything whose name suggests a pose is
#: refused whatever it actually computes, because the name is the only thing a
#: reader of the funnel gets to check.
DOCKING_DERIVED_TOKENS = (
    "vina",
    "dock",
    "pose",
    "affinity",
    "busters",
    "rmsd",
    "structure_score",
    "geometry",
    "binding",
)

#: The two gates this module PREDICTS. This is ``submission_policy.EXPENSIVE_GATES``
#: itself, imported rather than restated, so the two modules cannot drift into
#: disagreeing about which gates need a pose.
PREDICTED_GATES = EXPENSIVE_GATES

#: Fingerprint geometry, matching ``distill.DEFAULT_BASE_CONFIG`` so that this
#: programme has one featurisation convention rather than two.
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

#: Physico-chemical descriptors appended to the fingerprint. Every one is a function
#: of the SMILES alone; none is named after a pose.
DESCRIPTOR_NAMES = (
    "desc_mw",
    "desc_logp",
    "desc_tpsa",
    "desc_hbd",
    "desc_hba",
    "desc_rotatable_bonds",
    "desc_ring_count",
    "desc_aromatic_rings",
    "desc_fraction_csp3",
    "desc_heavy_atoms",
    "desc_qed",
)

HEAD_REGRESSION = "regression"
HEAD_CLASSIFICATION = "classification"
HEADS = (HEAD_REGRESSION, HEAD_CLASSIFICATION)

DEFAULT_CONFIG: Record = {
    "fingerprint_radius": FINGERPRINT_RADIUS,
    "fingerprint_bits": FINGERPRINT_BITS,
    "descriptors": True,
    "learner": "hist_gradient_boosting",
    "max_iter": 200,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "seed": 42,
}


class SurrogateLineError(ValueError):
    """Raised when something would turn the prediction back into a lookup."""


# ---------------------------------------------------------------------------
# 1. the refusals
# ---------------------------------------------------------------------------
def assert_feature_name_is_docking_free(name: str) -> None:
    """Refuse a feature whose NAME claims a pose-derived quantity."""
    lowered = str(name).lower()
    for token in DOCKING_DERIVED_TOKENS:
        if token in lowered:
            raise SurrogateLineError(
                f"feature {name!r} names a docking-derived quantity ({token!r}); the surrogate "
                f"predicts the expensive gates from a SMILES alone, and a feature that reads a "
                f"pose would make its output a lookup of the answer it is meant to predict"
            )


def assert_record_is_docking_free(record: Mapping[str, Any], *, what: str) -> None:
    """Refuse a molecule record that carries any docking output.

    Presence of the key is enough; a ``None`` value is not a defence, because a
    record shaped to carry a docking output is a record somebody intends to fill.
    """
    present = sorted(k for k in record if str(k) in DOCKING_DERIVED_KEYS)
    if present:
        raise SurrogateLineError(
            f"{what} carries docking output {present}; a molecule that already has a pose is a "
            f"molecule an expensive measurement already judged. Ranking it would be a lookup and "
            f"not a prediction, so it is refused rather than scored"
        )


def sanitise_for_featurisation(record: Mapping[str, Any]) -> Record:
    """Strip a record to the one key the featuriser is allowed to read."""
    assert_record_is_docking_free(record, what="a record offered to the featuriser")
    smiles = record.get("smiles")
    if not isinstance(smiles, str) or not smiles:
        raise SurrogateLineError("a record offered to the featuriser has no 'smiles'")
    return {"smiles": smiles}


def smiles_digest(smiles: str) -> str:
    """Content address of a canonical structure. Never the structure itself."""
    canonical = _canonical(smiles)
    return hashlib.sha256((canonical or f"UNPARSEABLE:{smiles}").encode()).hexdigest()


def _canonical(smiles: str) -> str | None:
    from drug_agent.selfevolve.evaluation.target_benchmark import _standardized

    return _standardized(str(smiles or ""))


# ---------------------------------------------------------------------------
# 2. featurisation, from a SMILES and nothing else
# ---------------------------------------------------------------------------
def feature_names(*, bits: int = FINGERPRINT_BITS, descriptors: bool = True) -> list[str]:
    names = [f"fp_{i:04d}" for i in range(bits)]
    if descriptors:
        names += list(DESCRIPTOR_NAMES)
    return names


def _descriptor_vector(mol: Any) -> list[float]:
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Lipinski.NumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(mol.GetNumHeavyAtoms()),
        float(QED.qed(mol)),
    ]


def featurise(
    records: Sequence[Mapping[str, Any]],
    *,
    radius: int = FINGERPRINT_RADIUS,
    bits: int = FINGERPRINT_BITS,
    descriptors: bool = True,
    extra_features: Mapping[str, Any] | None = None,
) -> tuple[Any, list[int], list[str]]:
    """Morgan bits (+ physico-chemical descriptors) for a list of molecule records.

    Returns ``(X, usable_indices, names)``. Follows ``distill._featurise``'s
    convention -- Morgan radius 2 over 2048 bits, unusable SMILES dropped rather
    than imputed -- and adds the refusals this module exists for.

    Raises:
        SurrogateLineError: a record carrying a docking output, or an extra feature
            whose name claims a pose-derived quantity.
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    extra = dict(extra_features or {})
    for name in sorted(extra):
        assert_feature_name_is_docking_free(name)
    clean = [sanitise_for_featurisation(record) for record in records]

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(bits))
    names = feature_names(bits=int(bits), descriptors=bool(descriptors)) + sorted(extra)
    vectors: list[Any] = []
    usable: list[int] = []
    for index, record in enumerate(clean):
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            continue
        row = list(np.array(generator.GetFingerprint(mol), dtype=np.float64))
        if descriptors:
            row += _descriptor_vector(mol)
        for name in sorted(extra):
            row.append(float(extra[name](record["smiles"])))
        vectors.append(np.asarray(row, dtype=np.float64))
        usable.append(index)
    if not vectors:
        return np.zeros((0, len(names)), dtype=np.float64), [], names
    return np.vstack(vectors), usable, names


# ---------------------------------------------------------------------------
# 3. the training corpus: docking outputs of PRIOR runs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingRow:
    """One docked molecule from a prior run: the features' source and both labels."""

    run_id: str
    arm: str
    molecule_id: str
    smiles: str
    vina_affinity: float | None
    structure_score_passes: bool
    geometry_passes: bool
    #: Whether this molecule clears the gates a SMILES already decides. It is the
    #: BASELINE the surrogate has to beat, never a feature: nothing derived from it
    #: reaches the featuriser, and it is carried on the row only so the two can be
    #: compared on identical held-out molecules.
    cheap_gates_pass: bool = True

    @property
    def passes_expensive(self) -> bool:
        return self.structure_score_passes and self.geometry_passes

    def as_feature_record(self) -> Record:
        """The ONLY thing this row is allowed to hand the featuriser."""
        return {"smiles": self.smiles}


def default_corpus_sources(repo_root: Path | str) -> list[Path]:
    """Every on-disk table of docked molecules this project generated.

    Held-out material is never a source: ``protected/`` is excluded by path, and
    nothing under it is opened, listed or stat-ed.
    """
    root = Path(repo_root)
    patterns = (
        ".research/**/molecules.json",
        ".external/research/*/derived/**/molecules.json",
        ".external/research/*/derived/**/scoring.json",
    )
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if "protected" in path.parts:
                continue
            out.append(path)
    return out


def _rows_of(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and "rows" in payload:
        return list(payload["rows"])
    if isinstance(payload, list):
        return list(payload)
    return []


def assemble_training_rows(
    sources: Sequence[Path | str],
    *,
    qualification: Mapping[str, Any],
    repo_root: Path | str | None = None,
    arm_of: Mapping[str, str] | None = None,
) -> list[TrainingRow]:
    """Read docked molecules off disk and label them with the FROZEN evaluator.

    ``structure_score`` and ``geometry`` are decided by the evaluator's own
    predicates -- :func:`at_least_as_favourable` against the frozen threshold, and
    :func:`geometry_checks` over the recorded pose -- rather than by a second
    definition living here. A molecule whose docking did not succeed is a labelled
    NEGATIVE, exactly as the evaluator treats it, and never a missing row.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import (
        _property_passes,
        at_least_as_favourable,
        geometry_checks,
    )

    threshold_record = qualification["structure_threshold"]
    threshold = float(threshold_record["threshold"])
    metric = str(threshold_record["metric"])
    limits = list(qualification.get("property_limits") or [])
    # The cheap-gate BASELINE is judged here too, and it is held to the same rule
    # ``submission_policy`` holds its own filter to: a calibrated gate that module
    # has not classified as docking-free is refused rather than guessed at, so a
    # gate added to the frozen qualification later cannot quietly become part of a
    # baseline that is supposed to need no pose.
    from drug_agent.selfevolve.target.submission_policy import CHEAP_ENDPOINT_GATES

    calibrated = {
        name: rec
        for name, rec in sorted(dict(qualification.get("endpoint_thresholds") or {}).items())
        if rec.get("status") == "CALIBRATED"
    }
    unclassified = sorted(set(calibrated) - set(CHEAP_ENDPOINT_GATES) - set(PREDICTED_GATES))
    if unclassified:
        raise SurrogateLineError(
            f"the frozen qualification applies endpoint gate(s) {unclassified} that "
            f"submission_policy has not classified as docking-free; the cheap-gate baseline "
            f"cannot include a gate that might read a pose"
        )
    endpoint_gates = {n: r for n, r in calibrated.items() if n in set(CHEAP_ENDPOINT_GATES)}
    base = Path(repo_root) if repo_root is not None else None
    names = dict(arm_of or {})

    rows: list[TrainingRow] = []
    for source in sources:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(path.parent.relative_to(base)) if base is not None else str(path.parent)
        arm = names.get(run_id) or _arm_hint(path)
        for entry in _rows_of(payload):
            smiles = entry.get("smiles")
            score_record = entry.get("structure_score")
            if not isinstance(smiles, str) or not smiles or not isinstance(score_record, Mapping):
                continue
            value = score_record.get("value") if score_record.get("status") == "SUCCEEDED" else None
            affinity = float(value) if isinstance(value, (int, float)) else None
            structure_ok = affinity is not None and at_least_as_favourable(affinity, threshold, metric)
            geometry_ok = bool(geometry_checks(str(entry.get("pose_molblock") or ""))["passes"])
            # The cheap-gate verdict is read back off the RECORDED panel with the
            # evaluator's own predicates rather than recomputed, so the baseline is
            # the same judgement the campaign actually applied.
            properties = entry.get("properties") or {}
            cheap_ok = all(_property_passes(properties, limit) for limit in limits)
            panel = entry.get("endpoints") or {}
            for gate_record in endpoint_gates.values():
                observed = panel.get(str(gate_record["metric"]))
                measured = None
                if isinstance(observed, Mapping) and observed.get("status") == "SUCCEEDED":
                    measured = observed.get("value")
                if measured is None:
                    # The evaluator's convention: an endpoint nobody computed FAILS.
                    cheap_ok = False
                    continue
                precision = gate_record.get("precision")
                rounded = round(float(measured), int(precision)) if precision is not None else float(measured)
                if not at_least_as_favourable(
                    rounded, float(gate_record["threshold"]), str(gate_record["metric"])
                ):
                    cheap_ok = False
            rows.append(
                TrainingRow(
                    run_id=run_id,
                    arm=arm,
                    molecule_id=str(entry.get("molecule_id") or ""),
                    smiles=smiles,
                    vina_affinity=affinity,
                    structure_score_passes=structure_ok,
                    geometry_passes=geometry_ok,
                    cheap_gates_pass=cheap_ok,
                )
            )
    return rows


def _arm_hint(path: Path) -> str:
    """Best-effort generator name for a run directory, for leave-one-ARM-out."""
    for candidate in ("funnel.json", "arm.json"):
        probe = path.parent / candidate
        if not probe.exists():
            continue
        try:
            record = json.loads(probe.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("arm", "name"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        generator = record.get("generator")
        if isinstance(generator, Mapping) and isinstance(generator.get("name"), str):
            return str(generator["name"])
    probe = path.parent / "result.json"
    if probe.exists():
        try:
            record = json.loads(probe.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        backend = record.get("backend")
        if isinstance(backend, str) and backend:
            return backend
    return "UNKNOWN"


def corpus_counts(rows: Sequence[TrainingRow]) -> Record:
    """Counts only. This module never emits a structure."""
    by_run: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_run.setdefault(row.run_id, {"n": 0, "n_expensive_pass": 0, "n_docked": 0})
        bucket["n"] += 1
        bucket["n_expensive_pass"] += int(row.passes_expensive)
        bucket["n_docked"] += int(row.vina_affinity is not None)
    return {
        "n_rows": len(rows),
        "n_runs": len(by_run),
        "n_arms": len({r.arm for r in rows}),
        "n_distinct_smiles": len({smiles_digest(r.smiles) for r in rows}),
        "n_docking_succeeded": sum(1 for r in rows if r.vina_affinity is not None),
        "n_structure_score_pass": sum(1 for r in rows if r.structure_score_passes),
        "n_geometry_pass": sum(1 for r in rows if r.geometry_passes),
        "n_expensive_pass": sum(1 for r in rows if r.passes_expensive),
        "expensive_pass_rate": round(sum(1 for r in rows if r.passes_expensive) / len(rows), 4)
        if rows
        else None,
        "per_run": dict(sorted(by_run.items())),
    }


# ---------------------------------------------------------------------------
# 4. the run-level split
# ---------------------------------------------------------------------------
def run_level_folds(
    rows: Sequence[TrainingRow],
    *,
    n_folds: int = 5,
    seed: int = 0,
    group: str = "run_id",
) -> list[tuple[list[str], list[str]]]:
    """Partition the GROUPS, not the molecules, into *n_folds* train/test pairs.

    Molecules inside one run are near-duplicates of one another. A random split puts
    a molecule's siblings in the training set and measures interpolation between
    near-copies; the number it reports is flattering and false. Whole groups are
    held out here, and :func:`assert_grouped_split` re-checks the property on the
    way out rather than trusting the arithmetic above it.

    *group* is ``"run_id"`` for leave-one-run-out or ``"arm"`` for the stricter
    leave-one-generator-out.
    """
    if group not in ("run_id", "arm"):
        raise ValueError(f"unknown grouping {group!r}; expected 'run_id' or 'arm'")
    groups = sorted({getattr(row, group) for row in rows})
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if len(groups) < n_folds:
        raise ValueError(f"{len(groups)} {group} group(s) cannot make {n_folds} folds")
    order = list(groups)
    rng = _rng(seed)
    rng.shuffle(order)
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for index, name in enumerate(order):
        buckets[index % n_folds].append(name)
    folds: list[tuple[list[str], list[str]]] = []
    for held in buckets:
        test = sorted(held)
        train = sorted(g for g in groups if g not in set(test))
        assert_grouped_split(train, test)
        folds.append((train, test))
    return folds


def assert_grouped_split(train_groups: Iterable[str], test_groups: Iterable[str]) -> None:
    """Refuse a split that puts one group on both sides."""
    both = sorted(set(train_groups) & set(test_groups))
    if both:
        raise SurrogateLineError(
            f"group(s) {both} appear in BOTH the training and the held-out side. Molecules inside "
            f"one run are near-duplicates, so this split would report interpolation between "
            f"near-copies as generalisation"
        )


def _rng(seed: int) -> Any:
    import random

    return random.Random(int(seed))


# ---------------------------------------------------------------------------
# 5. the fitted surrogate
# ---------------------------------------------------------------------------
@dataclass
class SurrogatePrefilter:
    """A fitted surrogate, and the record of what it is NOT allowed to rank.

    ``score_pool`` returns HIGHER-IS-BETTER scores whichever head is fitted: the
    regression head predicts ``vina_affinity`` (lower is better) and is negated, so
    a caller ranking descending never has to remember the orientation.
    """

    head: str
    config: Record
    model: Any
    trained_runs: frozenset[str]
    trained_arms: frozenset[str]
    trained_smiles_digests: frozenset[str]
    n_training_rows: int
    feature_names: list[str] = field(repr=False, default_factory=list)

    @property
    def model_hash(self) -> str:
        payload = {
            "head": self.head,
            "config": dict(sorted(self.config.items())),
            "feature_spec": {
                "n_features": len(self.feature_names),
                "fingerprint_radius": self.config.get("fingerprint_radius"),
                "fingerprint_bits": self.config.get("fingerprint_bits"),
                "descriptors": bool(self.config.get("descriptors")),
            },
            "trained_runs": sorted(self.trained_runs),
            "n_training_rows": self.n_training_rows,
            "training_set_digest": hashlib.sha256(
                "".join(sorted(self.trained_smiles_digests)).encode("utf-8")
            ).hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # -- the refusals, again, at the point of use ---------------------------
    def check_pool(self, pool: Sequence[Mapping[str, Any]]) -> None:
        """Refuse a pool this model is not entitled to rank.

        Three separate ways a pool stops being a prediction, each its own error.
        """
        for member in pool:
            assert_record_is_docking_free(member, what=f"pool member {member.get('molecule_id')!r}")
        offending_runs = sorted(
            {
                str(m["run_id"])
                for m in pool
                if m.get("run_id") is not None and str(m["run_id"]) in self.trained_runs
            }
        )
        if offending_runs:
            raise SurrogateLineError(
                f"this surrogate was fitted on run(s) {offending_runs} and is being asked to rank "
                f"molecules from them. Molecules inside one run are near-duplicates, so a model "
                f"that saw any of the run is recalling it rather than predicting it"
            )
        seen = sum(
            1 for m in pool if smiles_digest(str(m.get("smiles") or "")) in self.trained_smiles_digests
        )
        if seen:
            raise SurrogateLineError(
                f"{seen} of {len(pool)} pool molecules are in this surrogate's TRAINING set, whose "
                f"labels are docking outputs of those same molecules. Scoring them would return a "
                f"memorised measurement, which is the lookup this module exists to avoid. Call "
                f"drop_trained_overlap() if the removal is what you want; it is counted, not silent"
            )

    def drop_trained_overlap(self, pool: Sequence[Mapping[str, Any]]) -> tuple[list[Record], Record]:
        """Remove pool members this model was trained on, and say how many."""
        kept = [
            dict(m)
            for m in pool
            if smiles_digest(str(m.get("smiles") or "")) not in self.trained_smiles_digests
        ]
        return kept, {
            "n_pool": len(pool),
            "n_dropped_seen_in_training": len(pool) - len(kept),
            "n_rankable": len(kept),
        }

    def score_pool(self, pool: Sequence[Mapping[str, Any]]) -> list[float]:
        """Higher is better. Reads ``smiles`` and nothing else.

        A molecule the featuriser cannot parse scores ``-inf``: unrankable is not
        the same as bad, but it must never sort ABOVE a molecule the model actually
        judged, and the evaluator's own convention is that an unmeasurable molecule
        fails rather than defaults favourably.
        """
        pool = list(pool)
        self.check_pool(pool)
        if not pool:
            return []
        matrix, usable, _names = featurise(
            [{"smiles": str(m.get("smiles") or "")} for m in pool],
            radius=int(self.config["fingerprint_radius"]),
            bits=int(self.config["fingerprint_bits"]),
            descriptors=bool(self.config["descriptors"]),
        )
        scores = [float("-inf")] * len(pool)
        if not usable:
            return scores
        raw = self._predict(matrix)
        for position, index in enumerate(usable):
            scores[index] = float(raw[position])
        return scores

    def _predict(self, matrix: Any) -> Any:
        if self.head == HEAD_REGRESSION:
            # vina_affinity is LOWER_IS_BETTER; negate so the caller may always sort
            # descending and never has to remember which way the metric runs.
            return -self.model.predict(matrix)
        return self.model.predict_proba(matrix)[:, 1]

    def card(self) -> Record:
        """What the funnel records about this model. Counts and hashes only."""
        return {
            "head": self.head,
            "model_hash": self.model_hash,
            "learner": self.config.get("learner"),
            "n_training_rows": self.n_training_rows,
            "n_training_runs": len(self.trained_runs),
            "n_training_arms": len(self.trained_arms),
            "n_features": len(self.feature_names),
            "features_are_smiles_only": True,
            "gates_predicted": list(PREDICTED_GATES),
            "orientation": "higher_is_better",
        }


class ConstantSurrogate:
    """A surrogate that ranks nothing. Keeps every refusal, changes no order.

    Useful as the null arm: composed into ``surrogate_rank`` it must reproduce
    ``cheap_gate_prefilter`` exactly, because ties break on generation order.
    """

    head = "constant"

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)
        self.trained_runs: frozenset[str] = frozenset()
        self.trained_smiles_digests: frozenset[str] = frozenset()

    @property
    def model_hash(self) -> str:
        return hashlib.sha256(f"constant:{self.value!r}".encode()).hexdigest()

    def check_pool(self, pool: Sequence[Mapping[str, Any]]) -> None:
        for member in pool:
            assert_record_is_docking_free(member, what=f"pool member {member.get('molecule_id')!r}")

    def score_pool(self, pool: Sequence[Mapping[str, Any]]) -> list[float]:
        pool = list(pool)
        self.check_pool(pool)
        return [self.value] * len(pool)

    def card(self) -> Record:
        return {
            "head": self.head,
            "model_hash": self.model_hash,
            "note": "a constant score; surrogate_rank reduces to cheap_gate_prefilter",
        }


def _build_learner(head: str, config: Mapping[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    name = str(config.get("learner", "hist_gradient_boosting"))
    if name == "hist_gradient_boosting":
        kwargs = {
            "max_iter": int(config.get("max_iter", 200)),
            "learning_rate": float(config.get("learning_rate", 0.08)),
            "max_leaf_nodes": int(config.get("max_leaf_nodes", 31)),
            "min_samples_leaf": int(config.get("min_samples_leaf", 20)),
            "l2_regularization": float(config.get("l2_regularization", 1.0)),
            "random_state": int(config.get("seed", 42)),
        }
        if head == HEAD_REGRESSION:
            return HistGradientBoostingRegressor(**kwargs)
        return HistGradientBoostingClassifier(**kwargs)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        kwargs = {
            "n_estimators": int(config.get("n_estimators", 300)),
            "min_samples_leaf": int(config.get("min_samples_leaf", 2)),
            "n_jobs": -1,
            "random_state": int(config.get("seed", 42)),
        }
        if head == HEAD_REGRESSION:
            return RandomForestRegressor(**kwargs)
        return RandomForestClassifier(**kwargs)
    if name == "linear":
        return (
            Ridge(alpha=float(config.get("alpha", 1.0)))
            if head == HEAD_REGRESSION
            else LogisticRegression(max_iter=2000, C=float(config.get("C", 1.0)))
        )
    raise ValueError(f"unknown learner {name!r}; expected hist_gradient_boosting, random_forest or linear")


def train_surrogate(
    rows: Sequence[TrainingRow],
    *,
    head: str = HEAD_CLASSIFICATION,
    config: Mapping[str, Any] | None = None,
) -> SurrogatePrefilter:
    """Fit one head on the docking outputs of the runs in *rows*.

    The fitted model remembers which runs, which generators and which structures it
    saw, because those are exactly the things it must later refuse to rank.
    """
    import numpy as np

    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; expected one of {list(HEADS)}")
    merged = dict(DEFAULT_CONFIG)
    merged.update(dict(config or {}))
    usable_rows = [r for r in rows if head == HEAD_CLASSIFICATION or r.vina_affinity is not None]
    if not usable_rows:
        raise ValueError(f"no training row carries a label for the {head} head")

    matrix, usable, names = featurise(
        [r.as_feature_record() for r in usable_rows],
        radius=int(merged["fingerprint_radius"]),
        bits=int(merged["fingerprint_bits"]),
        descriptors=bool(merged["descriptors"]),
    )
    if matrix.shape[0] == 0:
        raise ValueError("no training row survived featurisation")
    kept = [usable_rows[i] for i in usable]
    if head == HEAD_REGRESSION:
        target = np.asarray([float(r.vina_affinity) for r in kept], dtype=np.float64)
    else:
        target = np.asarray([int(r.passes_expensive) for r in kept], dtype=np.int64)
        if len(set(target.tolist())) < 2:
            raise ValueError("the classification head needs both a passing and a failing example")

    model = _build_learner(head, merged)
    model.fit(matrix, target)
    return SurrogatePrefilter(
        head=head,
        config=merged,
        model=model,
        trained_runs=frozenset(r.run_id for r in kept),
        trained_arms=frozenset(r.arm for r in kept),
        trained_smiles_digests=frozenset(smiles_digest(r.smiles) for r in kept),
        n_training_rows=len(kept),
        feature_names=names,
    )


# ---------------------------------------------------------------------------
# 6. held-out-RUN evaluation, with the baselines beside it
# ---------------------------------------------------------------------------
def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return None
    value = spearmanr(list(a), list(b)).statistic
    return None if value is None or math.isnan(float(value)) else round(float(value), 4)


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return None
    finite = [s if math.isfinite(s) else -1e18 for s in scores]
    return round(float(roc_auc_score(list(labels), finite)), 4)


def precision_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float | None:
    """Fraction of the top-*k* by score that carry a positive label."""
    if not labels or k < 1:
        return None
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))
    top = order[: min(k, len(order))]
    return round(sum(labels[i] for i in top) / len(top), 4)


def evaluate_grouped(
    rows: Sequence[TrainingRow],
    *,
    head: str,
    n_folds: int = 5,
    seed: int = 0,
    group: str = "run_id",
    config: Mapping[str, Any] | None = None,
    at_k: int = 150,
) -> Record:
    """Cross-validate with WHOLE GROUPS held out, and report the baselines beside it.

    The two baselines are computed over the identical held-out rows, so the three
    numbers are comparable by construction:

    * ``random`` -- the expected yield of picking *k* at random, which is the
      held-out positive rate.
    * ``cheap_prefilter_only`` -- the positive rate among the molecules that pass
      the gates a SMILES already decides. This is the bar the surrogate has to
      clear to be worth deploying, not the random bar.
    """
    per_fold: list[Record] = []
    folds = run_level_folds(rows, n_folds=n_folds, seed=seed, group=group)
    for train_groups, test_groups in folds:
        assert_grouped_split(train_groups, test_groups)
        train_set = set(train_groups)
        test_set = set(test_groups)
        train_rows = [r for r in rows if getattr(r, group) in train_set]
        test_rows = [r for r in rows if getattr(r, group) in test_set]
        if not train_rows or not test_rows:
            continue
        model = train_surrogate(train_rows, head=head, config=config)
        # The held-out rows carry docking outputs (they are the ground truth), so
        # they are offered to the model as SMILES-only records -- the same refusal
        # the deployment path enforces, applied to the evaluation path.
        # ``molecule_id`` repeats across runs (every run has a ``gen_0000``), so the
        # pool is keyed by POSITION here and never by id; aligning on a non-unique
        # key would silently mix runs back together.
        pool = [{"molecule_id": f"{i:06d}", "smiles": r.smiles} for i, r in enumerate(test_rows)]
        rankable, overlap = model.drop_trained_overlap(pool)
        kept = [int(m["molecule_id"]) for m in rankable]
        scored_rows = [test_rows[i] for i in kept]
        scores = model.score_pool(rankable)
        labels = [int(r.passes_expensive) for r in scored_rows]
        cheap_scores = [float(r.cheap_gates_pass) for r in scored_rows]
        affinities = [r.vina_affinity for r in scored_rows]
        finite = [
            (s, a) for s, a in zip(scores, affinities, strict=True) if a is not None and math.isfinite(s)
        ]
        # Pooled metrics mix generators, and a model can separate two generators
        # without saying anything useful about ranking WITHIN one pool. The
        # per-run breakdown is the number a submission policy actually spends: one
        # pool, one generator, one ordering decision.
        per_run: list[Record] = []
        for run_id in sorted({r.run_id for r in scored_rows}):
            subset = [(r, s) for r, s in zip(scored_rows, scores, strict=True) if r.run_id == run_id]
            sub_labels = [int(r.passes_expensive) for r, _ in subset]
            sub_scores = [s for _, s in subset]
            sub_finite = [
                (s, r.vina_affinity) for r, s in subset if r.vina_affinity is not None and math.isfinite(s)
            ]
            per_run.append(
                {
                    "run_id": run_id,
                    "n": len(subset),
                    "positive_rate": round(sum(sub_labels) / len(sub_labels), 4) if sub_labels else None,
                    "spearman_vs_affinity": _spearman(
                        [s for s, _ in sub_finite], [-a for _, a in sub_finite]
                    ),
                    "roc_auc": _roc_auc(sub_labels, sub_scores),
                    "precision_at_50": precision_at_k(sub_labels, sub_scores, 50),
                    "baseline_cheap_precision_at_50": precision_at_k(
                        sub_labels, [float(r.cheap_gates_pass) for r, _ in subset], 50
                    ),
                }
            )
        per_fold.append(
            {
                "per_run": per_run,
                "held_out_groups": test_groups,
                "n_train_rows": len(train_rows),
                "n_test_rows": len(scored_rows),
                "overlap": overlap,
                "model_hash": model.model_hash,
                "positive_rate": round(sum(labels) / len(labels), 4) if labels else None,
                # regression is reported against the TRUE affinity, with the sign
                # flipped back so a positive Spearman means "ranks the good ones high"
                "spearman_vs_affinity": _spearman([s for s, _ in finite], [-a for _, a in finite]),
                "roc_auc": _roc_auc(labels, scores),
                f"precision_at_{at_k}": precision_at_k(labels, scores, at_k),
                "precision_at_50": precision_at_k(labels, scores, 50),
                # The two baselines, on the IDENTICAL held-out molecules. Random is
                # the held-out positive rate; the cheap prefilter is the positive
                # rate among the molecules a SMILES already clears. The second is
                # the bar that decides whether this is worth deploying.
                "baseline_random_precision": round(sum(labels) / len(labels), 4) if labels else None,
                "baseline_cheap_roc_auc": _roc_auc(labels, cheap_scores),
                f"baseline_cheap_precision_at_{at_k}": precision_at_k(labels, cheap_scores, at_k),
                "baseline_cheap_precision_at_50": precision_at_k(labels, cheap_scores, 50),
                "n_cheap_survivors": int(sum(cheap_scores)),
            }
        )

    def _mean(key: str) -> float | None:
        values = [f[key] for f in per_fold if f.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    run_rows = [entry for fold in per_fold for entry in fold["per_run"]]

    def _run_mean(key: str) -> float | None:
        values = [e[key] for e in run_rows if e.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "head": head,
        "group": group,
        "n_folds": len(per_fold),
        "split": "GROUPED: whole runs held out. A random split would leak near-duplicates",
        "folds": per_fold,
        "mean_pooled_over_folds": {
            "spearman_vs_affinity": _mean("spearman_vs_affinity"),
            "roc_auc": _mean("roc_auc"),
            f"precision_at_{at_k}": _mean(f"precision_at_{at_k}"),
            "precision_at_50": _mean("precision_at_50"),
            "baseline_random_precision": _mean("baseline_random_precision"),
            "baseline_cheap_roc_auc": _mean("baseline_cheap_roc_auc"),
            f"baseline_cheap_precision_at_{at_k}": _mean(f"baseline_cheap_precision_at_{at_k}"),
            "baseline_cheap_precision_at_50": _mean("baseline_cheap_precision_at_50"),
            "positive_rate": _mean("positive_rate"),
        },
        "mean_within_held_out_run": {
            "n_runs": len(run_rows),
            "spearman_vs_affinity": _run_mean("spearman_vs_affinity"),
            "roc_auc": _run_mean("roc_auc"),
            "precision_at_50": _run_mean("precision_at_50"),
            "baseline_cheap_precision_at_50": _run_mean("baseline_cheap_precision_at_50"),
            "positive_rate": _run_mean("positive_rate"),
            "note": (
                "pooled metrics mix generators and reward separating them; these are the "
                "within-one-pool numbers a submission policy actually spends"
            ),
        },
    }


# ---------------------------------------------------------------------------
# 7. counterfactual replay against recorded pools
# ---------------------------------------------------------------------------
def replay_pool(
    pool: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    ground_truth: Mapping[str, bool],
    cheap_survivors: Mapping[str, bool],
    scaffold_of: Mapping[str, str] | None = None,
    surrogate: Any | None = None,
    random_trials: int = 500,
    seed: int = 0,
) -> Record:
    """Count how many of *budget* would have qualified under each selection rule.

    *pool* is in generation order and carries ``molecule_id`` and ``smiles`` ONLY --
    a pool that carries a docking output is refused, because that is the lookup.
    *ground_truth*, *cheap_survivors* and *scaffold_of* are keyed by ``molecule_id``
    and are held by the CALLER; the surrogate never sees any of them.

    When *scaffold_of* is supplied the registered endpoint is counted too:
    ``qualifying_scaffolds`` is the primary metric and a selection that concentrates
    its qualifying molecules on few scaffolds is worth less than its compound count
    suggests. The scaffold count uses the frozen evaluator's own
    :func:`qualifying_scaffolds`, not a second definition.
    """
    for member in pool:
        assert_record_is_docking_free(member, what=f"replay pool member {member.get('molecule_id')!r}")
    ids = [str(m["molecule_id"]) for m in pool]
    qualifies = [bool(ground_truth.get(i, False)) for i in ids]
    survives = [bool(cheap_survivors.get(i, False)) for i in ids]

    out: Record = {
        "n_pool": len(pool),
        "budget": budget,
        "n_qualifying_in_pool": sum(qualifies),
        "n_cheap_survivors": sum(survives),
        "rules": {},
    }

    scaffolds = dict(scaffold_of or {})

    def count(indices: Sequence[int]) -> Record:
        record: Record = {
            "n_selected": len(indices),
            "n_qualifying": sum(1 for i in indices if qualifies[i]),
            "budget_filled": len(indices) == budget,
        }
        if scaffolds:
            record["n_qualifying_scaffolds"] = _qualifying_scaffolds(
                [(ids[i], scaffolds.get(ids[i]), qualifies[i]) for i in indices]
            )
        return record

    order_gen = list(range(len(pool)))[:budget]
    out["rules"]["generation_order"] = count(order_gen)

    order_cheap = [i for i in range(len(pool)) if survives[i]][:budget]
    out["rules"]["cheap_gate_prefilter"] = count(order_cheap)

    rng = _rng(seed)
    totals = 0
    scaffold_totals = 0
    for _ in range(random_trials):
        draw = list(range(len(pool)))
        rng.shuffle(draw)
        totals += sum(qualifies[i] for i in draw[:budget])
        if scaffolds:
            scaffold_totals += _qualifying_scaffolds(
                [(ids[i], scaffolds.get(ids[i]), qualifies[i]) for i in draw[:budget]]
            )
    out["rules"]["random"] = {
        "n_selected": min(budget, len(pool)),
        "n_qualifying": round(totals / random_trials, 2),
        "note": f"mean over {random_trials} uniform draws",
    }
    if scaffolds:
        out["rules"]["random"]["n_qualifying_scaffolds"] = round(scaffold_totals / random_trials, 2)

    if surrogate is not None:
        eligible = [i for i in range(len(pool)) if survives[i]]
        scores = surrogate.score_pool(
            [{"molecule_id": ids[i], "smiles": pool[i]["smiles"]} for i in eligible]
        )
        ranked = sorted(range(len(eligible)), key=lambda j: (-scores[j], j))
        order_sur = [eligible[j] for j in ranked][:budget]
        out["rules"]["surrogate_rank"] = count(order_sur)
        out["rules"]["surrogate_rank"]["model_hash"] = surrogate.model_hash
    return out


def _qualifying_scaffolds(entries: Sequence[tuple[str, str | None, bool]]) -> int:
    """The registered endpoint, counted by the frozen evaluator's own function."""
    from drug_agent.selfevolve.evaluation.target_benchmark import qualifying_scaffolds

    rows = [
        {"molecule_id": mid, "scaffold_id": scaffold, "qualifies": bool(ok)}
        for mid, scaffold, ok in entries
        if scaffold
    ]
    return qualifying_scaffolds(rows) if rows else 0


def counterfactual_replay(
    *,
    rows: Sequence[TrainingRow],
    qualifying: Mapping[str, Mapping[str, bool]],
    cheap: Mapping[str, Mapping[str, bool]],
    pool_order: Mapping[str, Sequence[str]],
    held_out_groups: Sequence[str],
    scaffolds: Mapping[str, Mapping[str, str]] | None = None,
    head: str = HEAD_CLASSIFICATION,
    budget: int = 150,
    config: Mapping[str, Any] | None = None,
    group: str = "run_id",
) -> Record:
    """Replay one held-out run's recorded pool without docking anything new.

    The surrogate is fitted on every OTHER run, so nothing it saw comes from the
    pool it ranks. The ground truth is the evaluation already on disk, which is why
    this costs no GPU-seconds and produces a like-for-like count at a fixed budget.

    Any pool molecule the model happened to see in another run is removed from the
    pool for EVERY rule, not only from the surrogate's: dropping it for one arm and
    not the others would change the denominator between rules and the counts would
    stop being comparable. The number removed is reported per run.
    """
    held = set(held_out_groups)
    train_rows = [r for r in rows if getattr(r, group) not in held]
    assert_grouped_split(sorted({getattr(r, group) for r in train_rows}), sorted(held))
    model = train_surrogate(train_rows, head=head, config=config)

    by_run: dict[str, dict[str, TrainingRow]] = {}
    for row in rows:
        by_run.setdefault(row.run_id, {})[row.molecule_id] = row

    group_of = {row.run_id: getattr(row, group) for row in rows}
    results: Record = {"head": head, "model_hash": model.model_hash, "budget": budget, "runs": {}}
    for run_id in sorted(pool_order):
        # Replaying a run whose GROUP was in the training set would be the lookup
        # again, one level up: the model saw its siblings.
        if group_of.get(run_id, run_id) not in held:
            raise SurrogateLineError(
                f"run {run_id!r} is offered for replay but its {group} was NOT held out of "
                f"training; the model saw its siblings and would be recalling, not predicting"
            )
        order = list(pool_order.get(run_id) or [])
        known = by_run.get(run_id, {})
        pool = [{"molecule_id": mid, "smiles": known[mid].smiles} for mid in order if mid in known]
        if not pool:
            continue
        rankable, overlap = model.drop_trained_overlap(pool)
        if not rankable:
            continue
        record = replay_pool(
            rankable,
            budget=budget,
            ground_truth=qualifying.get(run_id, {}),
            cheap_survivors=cheap.get(run_id, {}),
            scaffold_of=(scaffolds or {}).get(run_id),
            surrogate=model,
        )
        record["overlap"] = overlap
        results["runs"][run_id] = record
    return results


__all__ = [
    "ConstantSurrogate",
    "DEFAULT_CONFIG",
    "DESCRIPTOR_NAMES",
    "DOCKING_DERIVED_KEYS",
    "DOCKING_DERIVED_TOKENS",
    "FINGERPRINT_BITS",
    "FINGERPRINT_RADIUS",
    "HEADS",
    "HEAD_CLASSIFICATION",
    "HEAD_REGRESSION",
    "PREDICTED_GATES",
    "SurrogateLineError",
    "SurrogatePrefilter",
    "TrainingRow",
    "assemble_training_rows",
    "assert_feature_name_is_docking_free",
    "assert_grouped_split",
    "assert_record_is_docking_free",
    "corpus_counts",
    "counterfactual_replay",
    "default_corpus_sources",
    "evaluate_grouped",
    "featurise",
    "feature_names",
    "precision_at_k",
    "replay_pool",
    "run_level_folds",
    "sanitise_for_featurisation",
    "smiles_digest",
    "train_surrogate",
]
