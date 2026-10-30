"""The independent Track A/B/C benchmark for one target, and its calibration.

Step 7 of the per-target evolution program. Everything downstream measures
improvement with what is in this module, so it is written to be *checkable*
rather than merely correct: each metric's denominator is stated, each threshold
is derived by a frozen rule from development controls only, and each direction
of "better" is registered rather than assumed.

**Where it sits.** Inside :mod:`drug_agent.selfevolve.evaluation`, so
:mod:`.integrity` already hashes it and
:func:`~drug_agent.selfevolve.evaluation.integrity.candidate_touches_kernel`
refuses a candidate that edits it at admission. A benchmark a candidate can
rewrite is not a benchmark. :func:`enrichment_factor` and
:func:`qualifying_scaffolds` are re-exported from
:mod:`drug_agent.selfevolve.target.api`, which Appendix A §7.2 names as the
acceptance surface; the implementations live here.

**Five decisions worth reading before trusting a number this module produces.**

*Orientation is registered, not inferred.* Vina and GNINA report
more-negative-is-better; pActivity reports higher-is-better; MAE reports
lower-is-better. :func:`score_orientation` raises for a metric nobody
registered, because a default direction is how an evaluator silently rewards the
worse molecule.

*An absence is never a number.* Every pool-level metric takes
:func:`~drug_agent.selfevolve.target.api.endpoint_result` records, not floats.
A compound that failed to score is charged a **predeclared** penalty for
regression and the **worst rank** for ranking; it is never dropped, because
dropping the hard compounds and quoting the mean of what is left is the cheapest
way to win. Coverage is reported beside every metric, the scored-subset mean is
computed and labelled *secondary*, and a coverage regression fails the
comparison outright.

*Row order decides nothing.* Ranking ties are broken by molecule id. A tie that
straddles the top-k boundary with no ids to break it is **refused**, not
resolved by the order the rows happened to arrive in. Appendix A §7.2's
convenience signature has no id parameter, so one was added as a keyword-only,
defaulted, backward-compatible extension and recorded in the implementation map.

*Qualification thresholds come from development controls, by a frozen rule.*
Structure score at least as favourable as the **median of the valid known-active
development controls**; declared geometry, identity, property and novelty checks
pass. The activity gate takes the **lowest** development threshold attaining
false-positive rate ≤ 0.10 and true-positive rate ≥ 0.20 — and when there are
too few controls to establish one, it is marked ``UNCALIBRATED`` and the Track B
name is forced to ``STRUCTURE_ONLY_COMPUTATIONAL``. Appendix A §3.3: do not
silently remove it after seeing candidates.

*An invalid mandatory calibration fails closed.* :func:`require_calibrated`
raises :class:`CalibrationError` rather than returning a permissive default, and
:func:`evaluate_portfolio` refuses to produce a qualifying-scaffold count
against a qualification whose structure threshold was never established.

**Empty scaffolds are not one scaffold.** Step 6 had to make this call for
family ids and this module reuses its convention: an acyclic molecule's scaffold
id is ``acyclic:<molecule_id>``, so two acyclic molecules are two scaffolds. A
count that collapsed every acyclic molecule into one shared empty scaffold would
under-report a generator's yield by exactly the molecules it is best at.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...util import content_hash
from ..target.contracts import CONTRACT_VERSION, ContractError, ErrorCategory
from .integrity import kernel_digest

Record = dict[str, Any]

#: This module's own contract with the programme, not the repository's version.
#: 1.1.0 (step 15B): the Track B qualification record gained the generation-quality
#: endpoint gates, the de-novo novelty union, the conditioning-class
#: stratification and the declared limits. No 1.0.0 value moved; the shape grew.
BENCHMARK_VERSION = "1.1.0"

_CONTRACT = "TargetBenchmark"

# ---------------------------------------------------------------------------
# Score orientation
# ---------------------------------------------------------------------------
HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
LOWER_IS_BETTER = "LOWER_IS_BETTER"

#: Which way "better" points for every metric this programme reports. There is
#: deliberately no default: an unregistered metric raises. An inverted
#: comparison is the classic way an evaluator rewards the worse molecule while
#: staying precise to four decimal places, and it is invisible in the output.
SCORE_ORIENTATION: dict[str, str] = {
    # Track A
    "pActivity": HIGHER_IS_BETTER,
    "mae_pactivity": LOWER_IS_BETTER,
    "ef10": HIGHER_IS_BETTER,
    "pr_auc": HIGHER_IS_BETTER,
    "spearman": HIGHER_IS_BETTER,
    "coverage": HIGHER_IS_BETTER,
    # Track B — empirical docking functions, kcal/mol, more negative is better.
    "vina_affinity": LOWER_IS_BETTER,
    "minimizedAffinity": LOWER_IS_BETTER,
    # GNINA's CNN heads are pose/affinity *scores*, not energies: higher is
    # better. Registering them beside the energies is the point of this table.
    "CNNscore": HIGHER_IS_BETTER,
    "CNNaffinity": HIGHER_IS_BETTER,
    "qualifying_scaffolds": HIGHER_IS_BETTER,
    # Track B generation quality (step 15B). Both are LOWER_IS_BETTER and both
    # are stated rather than inferred, because neither name carries its own
    # direction: `herg` is a blockade score, so a *higher* number is a *worse*
    # molecule, and `sa_score` is Ertl & Schuffenhauer's synthetic-accessibility
    # score, where 1 is trivial to make and 10 is intractable.
    "herg": LOWER_IS_BETTER,
    "sa_score": LOWER_IS_BETTER,
    # Track C
    "wall_seconds": LOWER_IS_BETTER,
    "gpu_seconds": LOWER_IS_BETTER,
    "oracle_calls": LOWER_IS_BETTER,
    "expensive_evaluations": LOWER_IS_BETTER,
}


def score_orientation(metric: str) -> str:
    """Which direction is better for *metric*.

    Raises:
        ContractError: ``INVALID_ENUM`` for a metric nobody registered. There is
            no default direction on purpose.
    """
    try:
        return SCORE_ORIENTATION[str(metric)]
    except KeyError:
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"{metric!r} has no registered score orientation; register it in "
            f"SCORE_ORIENTATION before comparing on it. Known: {sorted(SCORE_ORIENTATION)}",
            contract=_CONTRACT,
            field="metric",
        ) from None


def is_better(candidate: float, reference: float, metric: str) -> bool:
    """``True`` when *candidate* is strictly more favourable than *reference*."""
    orientation = score_orientation(metric)
    a, b = float(candidate), float(reference)
    return bool(a > b) if orientation == HIGHER_IS_BETTER else bool(a < b)


def at_least_as_favourable(candidate: float, reference: float, metric: str) -> bool:
    """``True`` when *candidate* is as good as *reference* or better.

    Inclusive by declaration: the qualification rule is "at least as favourable
    as the median", so a molecule sitting exactly on the threshold qualifies.
    """
    orientation = score_orientation(metric)
    a, b = float(candidate), float(reference)
    return bool(a >= b) if orientation == HIGHER_IS_BETTER else bool(a <= b)


def most_favourable(values: Sequence[float], metric: str) -> float:
    """The best of *values* under *metric*'s registered orientation."""
    if not values:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            "no values to compare",
            contract=_CONTRACT,
            field="values",
        )
    orientation = score_orientation(metric)
    return max(values) if orientation == HIGHER_IS_BETTER else min(values)


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------
#: Appendix A §3.2's primary ranking metric is EF at the top 10 %.
EF_FRACTION = 0.1

#: Appendix A §5: essential validity/coverage constraints carry a default
#: non-regression tolerance of two percentage points.
COVERAGE_TOLERANCE = 0.02

#: Appendix A §5's default quality non-inferiority margin for Track C.
DEFAULT_QUALITY_MARGIN = 0.05

#: How a compound that failed to score is charged, per branch. Both are
#: predeclared here rather than decided when a comparison is run.
REGRESSION_FAILURE_POLICY = "PREDECLARED_ABSOLUTE_PENALTY"
RANKING_FAILURE_POLICY = "WORST_RANK"

#: Step 6 decided Track A's primary mode on development counts alone:
#: ``regression``. The ranking branch is ``INSUFFICIENT_TEST_DATA`` for this
#: target (development: 0 positives / 109 negatives). EF10 is implemented,
#: tested and reportable as exploratory; it is not the primary Track A metric
#: here, and nothing in this module may promote it because regression looks
#: hard.
TRACK_A_PRIMARY_MODE = "regression"
TRACK_A_RANKING_STATUS = "INSUFFICIENT_TEST_DATA"

#: An acyclic molecule has an empty Bemis-Murcko scaffold. Step 6 established
#: that empty scaffolds are not one family; this is the same rule for scaffold
#: counting, with an explicit per-molecule id so two acyclic molecules are two
#: scaffolds rather than one.
ACYCLIC_SCAFFOLD_PREFIX = "acyclic:"

#: Verdicts a calibration record can carry.
CALIBRATED = "CALIBRATED"
UNCALIBRATED = "UNCALIBRATED"

#: The Track B registered name when there is no defensible activity gate
#: (Appendix A §3.3). Chosen before candidate search and never swapped after.
STRUCTURE_ONLY_COMPUTATIONAL = "STRUCTURE_ONLY_COMPUTATIONAL"
STRUCTURE_AND_ACTIVITY_GATED = "STRUCTURE_AND_ACTIVITY_GATED"

#: The work order's exact operating point for the optional activity gate.
ACTIVITY_GATE_OPERATING_POINT: dict[str, float] = {
    "max_false_positive_rate": 0.10,
    "min_true_positive_rate": 0.20,
}

#: How many development controls an activity gate needs before a threshold
#: derived from them means anything. This is Appendix A §3.2's classification
#: feasibility guard — the only count floor this programme has declared — and it
#: is applied here rather than a number chosen after seeing the data.
ACTIVITY_GATE_MIN_CONTROLS: dict[str, int] = {"actives": 10, "inactives": 10}

#: A median is not a statistic below three observations. Declared before the
#: controls were docked.
STRUCTURE_THRESHOLD_MIN_CONTROLS = 3

#: The smallest numerical difference this programme will ever call real. A
#: frozen tolerance is never tighter than this even when repeated runs agree
#: exactly, because exact agreement over three repeats does not establish
#: bit-for-bit determinism across hosts.
DEFAULT_TOLERANCE_FLOOR = 1e-6

#: How far the repository's docking adapter may sit from the upstream tools'
#: own answer on the same inputs before the wrapper is called wrong.
#:
#: **Measured before it was declared**, which is the only way a tolerance means
#: anything. Two ligands docked into the ZAK 6JUU receptor at seed 42, once
#: through :class:`~drug_agent.tools.vina_tool.DockingAdapter` and once through
#: :func:`upstream_direct_dock` (RDKit + Meeko + Open Babel + the ``vina``
#: executable, no repository code in the path):
#:
#: ===================================== ========== ========== ==========
#: ligand                                 wrapped    direct     |Δ|
#: ===================================== ========== ========== ==========
#: the deposited C9R reference             −14.385    −14.366    0.019
#: CCOc1ccc(NC(=O)c2ccccc2)cc1              −8.794     −8.817    0.023
#: ===================================== ========== ========== ==========
#:
#: The residual is the ligand-preparation path, not the search: the adapter
#: round-trips the embedded conformer through an SDF on its way to Meeko and
#: loses coordinate precision doing it, while the search itself is bit-for-bit
#: reproducible at a fixed seed (see the determinism measurement in
#: ``configs/research/zak-c9r/calibration.json``). 0.1 kcal/mol is roughly five
#: times the largest observed difference and under 1 % of a −14 kcal/mol score.
WRAPPER_PARITY_TOLERANCE = 0.1

#: Geometry sanity limits, in Angstrom. **Deliberately far looser than
#: PoseBusters** (arXiv:2308.05777), whose non-bonded clash criterion is 0.75 ×
#: the sum of the two van der Waals radii — 1.65 Å for an H···H contact against
#: the 0.8 Å below. This is a validity screen that refuses physically impossible
#: geometry, not a pose-quality benchmark; an evaluation that wants PoseBusters'
#: criterion must declare it and pass it in through ``limits``. Executed on the
#: real development controls it refused one docked pose of six, on a 0.67 Å
#: H···H contact.
GEOMETRY_LIMITS: dict[str, float] = {
    "min_interatomic_distance_A": 0.8,
    "min_bond_length_A": 0.7,
    "max_bond_length_A": 2.5,
}

#: The gates a Track B molecule passes, in evaluation order.
QUALIFICATION_GATES: tuple[str, ...] = (
    "identity",
    "geometry",
    "properties",
    "structure_score",
    "novelty",
)

#: The declared admissible range of each endpoint this programme thresholds, and
#: the decimal precision its scores are frozen at. Both are properties of the
#: **endpoint**, declared from the model or the published scale that produces it,
#: never from the observed controls — which is what lets the vacuity test in
#: :func:`derive_endpoint_threshold` be a statement about the gate rather than a
#: statement about the sample.
#:
#: ``precision`` exists because a threshold is only meaningful to the precision
#: its backend reproduces. Measured on this host, three repeats of the identical
#: six-molecule ADMET-AI batch moved ``hERG`` by at most **5.96e-08** — the
#: float32 reduction-order noise ``safety_tool``'s own card names — so six
#: decimals is roughly two orders of magnitude coarser than the wobble and a
#: molecule cannot cross the bar by being scored twice. ``sascorer`` is a
#: deterministic fragment-table lookup and the adapter already rounds it to three.
ENDPOINT_SCALES: dict[str, dict[str, Any]] = {
    "herg": {
        "admissible_range": [0.0, 1.0],
        "precision": 6,
        "calibrated_score": False,
        "source": "safety.assess -> ADMET-AI Chemprop hERG ensemble (TDC-trained)",
        "note": (
            "an UNCALIBRATED sigmoid output, not a probability of harm: "
            "safety_tool sets calibrated=False on every panel endpoint and says so in "
            "its data_restrictions. A threshold on it is an empirical operating point "
            "on an uncalibrated score"
        ),
    },
    "sa_score": {
        "admissible_range": [1.0, 10.0],
        "precision": 3,
        "calibrated_score": False,
        "source": "synth.assess -> RDKit Contrib sascorer (Ertl & Schuffenhauer 2009)",
        "note": (
            "a heuristic fragment-contribution score on a fixed 1-10 scale; 1 is "
            "trivially accessible and 10 is intractable. Not a synthesis success "
            "probability and not a route"
        ),
    },
}

#: Why an endpoint threshold can fail to calibrate, beyond too few controls.
VACUOUS_THRESHOLD = "VACUOUS_THRESHOLD"
IMPOSSIBLE_THRESHOLD = "IMPOSSIBLE_THRESHOLD"
INSUFFICIENT_CONTROLS = "INSUFFICIENT_CONTROLS"


class CalibrationError(RuntimeError):
    """A mandatory calibration is invalid, so nothing may be scored against it."""


def require_calibrated(record: Mapping[str, Any], *, what: str) -> None:
    """Raise unless *record* carries ``status == CALIBRATED``.

    The fail-closed rule. A permissive default here would mean a candidate
    scored against a threshold that was never established.
    """
    status = record.get("status")
    if status != CALIBRATED:
        raise CalibrationError(
            f"{what} is {status!r} ({record.get('reason')}); refusing to score anything "
            f"against an invalid calibration"
        )


def evaluator_hash() -> str:
    """The content address of the whole evaluation kernel, this module included.

    A candidate that changed any file the judgement depends on changes this
    value, and every record this module freezes carries it.
    """
    return f"sha256:{kernel_digest()}"


# ---------------------------------------------------------------------------
# Small shared validators
# ---------------------------------------------------------------------------
def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ContractError(
            ErrorCategory.INVALID_TYPE,
            f"{value!r} is not a number",
            contract=_CONTRACT,
            field=field,
        ) from None
    if not math.isfinite(number):
        raise ContractError(
            ErrorCategory.NON_FINITE_VALUE,
            f"{value!r} is not finite and is therefore not a measurement",
            contract=_CONTRACT,
            field=field,
        )
    return number


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractError(
            ErrorCategory.EMPTY_IDENTIFIER,
            "a blank identifier cannot address, deduplicate or tie-break anything",
            contract=_CONTRACT,
            field=field,
        )
    return text


def _endpoint_value(record: Any, *, field: str) -> float | None:
    """The value of an endpoint record, or ``None`` when it did not succeed.

    Raises for anything that is not an endpoint record at all, because a bare
    float carries no way to distinguish a measured zero from an absence.
    """
    if not isinstance(record, Mapping) or "status" not in record or "value" not in record:
        raise ContractError(
            ErrorCategory.INVALID_TYPE,
            f"expected an endpoint record with 'status' and 'value' (see "
            f"drug_agent.selfevolve.target.api.endpoint_result), got {record!r}",
            contract=_CONTRACT,
            field=field,
        )
    if record["status"] != "SUCCEEDED":
        return None
    return _finite(record["value"], field=field)


def _scores_and_invalid(entries: Iterable[Any], *, field: str) -> tuple[list[float], int]:
    """Valid numbers from a sequence of endpoint records or plain floats.

    Plain floats are accepted so a hand-written calibration fixture stays
    readable; anything shaped like an endpoint record is read through the
    status/value contract, so a ``FAILED`` control never becomes a datapoint.
    """
    values: list[float] = []
    invalid = 0
    for entry in entries:
        if isinstance(entry, Mapping):
            value = _endpoint_value(entry, field=field)
            if value is None:
                invalid += 1
            else:
                values.append(value)
        else:
            values.append(_finite(entry, field=field))
    return values, invalid


# ---------------------------------------------------------------------------
# Track A — ranking: enrichment factor
# ---------------------------------------------------------------------------
def _ranked_indices(scores: Sequence[float], molecule_ids: Sequence[str] | None, k: int) -> list[int]:
    """Indices ordered best-first, with ties broken by molecule id.

    With no ids, a tie that *straddles* the top-k boundary is refused: which of
    the tied molecules is "in the top 10 %" would otherwise be decided by the
    order the rows arrived in, which is not a property of the science.
    """
    n = len(scores)
    if molecule_ids is not None:
        return sorted(range(n), key=lambda i: (-scores[i], molecule_ids[i]))
    order = sorted(range(n), key=lambda i: -scores[i])
    if 0 < k < n and scores[order[k - 1]] == scores[order[k]]:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"a tie at score {scores[order[k]]!r} straddles the top-{k} boundary and row "
            f"order would decide the winner; pass molecule_ids= to break it deterministically",
            contract=_CONTRACT,
            field="scores",
        )
    return order


def _validate_ranking_inputs(
    labels: Sequence[int],
    scores: Sequence[float],
    fraction: float,
    molecule_ids: Sequence[str] | None,
) -> tuple[list[int], list[float], list[str] | None, int]:
    if len(labels) != len(scores):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"labels and scores must have equal length; got {len(labels)} and {len(scores)}",
            contract=_CONTRACT,
            field="labels",
        )
    if molecule_ids is not None and len(molecule_ids) != len(labels):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"molecule_ids must have equal length with labels; got {len(molecule_ids)} and {len(labels)}",
            contract=_CONTRACT,
            field="molecule_ids",
        )
    n = len(labels)
    if n == 0:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            "an enrichment factor over an empty dataset has no denominator",
            contract=_CONTRACT,
            field="labels",
        )
    frac = _finite(fraction, field="fraction")
    if not (0.0 < frac <= 1.0):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"fraction must be in (0, 1]; got {fraction!r}",
            contract=_CONTRACT,
            field="fraction",
        )
    clean_labels: list[int] = []
    for label in labels:
        if isinstance(label, bool) or label not in (0, 1):
            raise ContractError(
                ErrorCategory.INVALID_VALUE,
                f"labels must be binary 0/1; got {label!r}. An intermediate-potency "
                f"compound is excluded from the binary benchmark, never coerced",
                contract=_CONTRACT,
                field="labels",
            )
        clean_labels.append(int(label))
    clean_scores = [_finite(s, field="scores") for s in scores]
    ids: list[str] | None = None
    if molecule_ids is not None:
        ids = [_identifier(m, field="molecule_ids") for m in molecule_ids]
        duplicated = sorted({m for m in ids if ids.count(m) > 1})
        if duplicated:
            raise ContractError(
                ErrorCategory.INVALID_VALUE,
                f"each molecule contributes at most once; duplicate molecule_ids: {duplicated}",
                contract=_CONTRACT,
                field="molecule_ids",
                missing=duplicated,
            )
    n_actives = sum(clean_labels)
    if n_actives == 0:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            "no active in the dataset: the overall active fraction is the enrichment "
            "factor's denominator, so EF is undefined rather than zero or infinite",
            contract=_CONTRACT,
            field="labels",
        )
    return clean_labels, clean_scores, ids, n_actives


def enrichment_record(
    labels: Sequence[int],
    scores: Sequence[float],
    fraction: float = EF_FRACTION,
    *,
    molecule_ids: Sequence[str] | None = None,
) -> Record:
    """Every number behind :func:`enrichment_factor`, so it can be checked.

    ``EF = (actives in the top ceil(fraction * N) / selected count) / (actives / N)``
    — Appendix A §3.2 exactly. ``ceil`` and not ``floor``: at N = 15 and
    fraction 0.1 the selected count is 2, and an evaluator using ``floor`` would
    select 1 and report a different benchmark.
    """
    clean_labels, clean_scores, ids, n_actives = _validate_ranking_inputs(
        labels, scores, fraction, molecule_ids
    )
    n = len(clean_labels)
    k = math.ceil(_finite(fraction, field="fraction") * n)
    k = max(1, min(k, n))
    order = _ranked_indices(clean_scores, ids, k)
    selected = order[:k]
    n_hits = sum(clean_labels[i] for i in selected)
    prevalence = n_actives / n
    return {
        "contract_version": CONTRACT_VERSION,
        "metric": "ef10" if abs(fraction - 0.1) < 1e-12 else f"ef{fraction:g}",
        "fraction": float(fraction),
        "n": n,
        "k": k,
        "n_actives": n_actives,
        "n_hits": n_hits,
        "prevalence": prevalence,
        "ef": (n_hits / k) / prevalence,
        "tie_break": "molecule_id" if ids is not None else "no tie straddles the cut",
        "selected_molecule_ids": [ids[i] for i in selected] if ids is not None else None,
    }


def enrichment_factor(
    labels: Sequence[int],
    scores: Sequence[float],
    fraction: float = EF_FRACTION,
    *,
    molecule_ids: Sequence[str] | None = None,
) -> float:
    """Enrichment at the top *fraction* of a ranked pool.

    Appendix A §7.2 declares ``enrichment_factor(labels, scores, fraction=0.1)``.
    ``molecule_ids`` is a keyword-only, defaulted extension: without it a tie
    straddling the cut is refused rather than resolved by row order. Existing
    positional callers are unaffected, and the extension is recorded in
    ``docs/research/implementation-map.md``.

    >>> enrichment_factor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
    10.0

    Raises:
        ContractError: ``INVALID_VALUE`` for unequal lengths, an empty dataset,
            a non-binary label, an out-of-range fraction, a dataset with no
            active, a duplicate molecule id, or an unbreakable tie;
            ``NON_FINITE_VALUE`` for a NaN or infinite score;
            ``EMPTY_IDENTIFIER`` for a blank molecule id.
    """
    return float(enrichment_record(labels, scores, fraction, molecule_ids=molecule_ids)["ef"])


def _average_precision(labels_in_order: Sequence[int]) -> float | None:
    """Average precision over an already-ranked pool (PR-AUC, secondary)."""
    total = sum(labels_in_order)
    if total == 0:
        return None
    hits = 0
    acc = 0.0
    for index, label in enumerate(labels_in_order, start=1):
        if label:
            hits += 1
            acc += hits / index
    return acc / total


def ranking_endpoint(
    *,
    pool: Sequence[str],
    labels: Mapping[str, int],
    predictions: Mapping[str, Mapping[str, Any]],
    fraction: float = EF_FRACTION,
) -> Record:
    """EF over the identical held-out pool, with failures ranked worst.

    Every method ranks the same pool. A compound whose scorer failed is not
    dropped — it is placed below every compound that scored, which is the
    prespecified worst-rank policy of Appendix A §3.2, and it stays in the
    denominator. Coverage, raw hit counts and prevalence are reported beside
    the metric.
    """
    ids = _unique_pool(pool)
    scored: dict[str, float] = {}
    failures: list[Record] = []
    for molecule_id in ids:
        record = predictions.get(molecule_id)
        if record is None:
            failures.append({"molecule_id": molecule_id, "reason": "NO_PREDICTION"})
            continue
        value = _endpoint_value(record, field=f"predictions[{molecule_id}]")
        if value is None:
            failures.append({"molecule_id": molecule_id, "reason": str(record["status"])})
        else:
            scored[molecule_id] = value
    missing_labels = [m for m in ids if m not in labels]
    if missing_labels:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            f"no label for pool members {missing_labels}; a pool member without a label "
            f"cannot be scored and must not be silently removed from the denominator",
            contract=_CONTRACT,
            field="labels",
            missing=missing_labels,
        )
    clean_labels = {m: _binary_label(labels[m]) for m in ids}
    order = sorted(scored, key=lambda m: (-scored[m], m)) + sorted(f["molecule_id"] for f in failures)
    n = len(ids)
    k = max(1, min(math.ceil(_finite(fraction, field="fraction") * n), n))
    n_actives = sum(clean_labels.values())
    n_hits = sum(clean_labels[m] for m in order[:k])
    prevalence = n_actives / n if n else 0.0
    status = "SUCCEEDED" if n_actives > 0 else "INSUFFICIENT_POSITIVES"
    return {
        "contract_version": CONTRACT_VERSION,
        "metric": "ef10" if abs(fraction - 0.1) < 1e-12 else f"ef{fraction:g}",
        "status": status,
        "n_pool": n,
        "k": k,
        "n_scored": len(scored),
        "n_failed": len(failures),
        "coverage": len(scored) / n if n else 0.0,
        "n_actives": n_actives,
        "n_hits": n_hits,
        "prevalence": prevalence,
        "ef": ((n_hits / k) / prevalence) if n_actives else None,
        "failure_policy": RANKING_FAILURE_POLICY,
        "failures": sorted(failures, key=lambda f: f["molecule_id"]),
        "ranked_molecule_ids": order,
        "primary_for_target": TRACK_A_PRIMARY_MODE == "ranking",
        "note": (
            "Track A's primary mode for this target is "
            f"{TRACK_A_PRIMARY_MODE!r}; the ranking branch is {TRACK_A_RANKING_STATUS} "
            "(Step 6 measured 0 positives in the development partition) and is "
            "exploratory, never confirmatory."
        ),
        "secondary": {
            "pr_auc": _average_precision([clean_labels[m] for m in order]),
            "raw_hits": n_hits,
            "note": "PR-AUC, raw hit counts and prevalence are secondary (Appendix A §3.2).",
        },
    }


def _binary_label(value: Any) -> int:
    if isinstance(value, bool) or value not in (0, 1):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"labels must be binary 0/1; got {value!r}",
            contract=_CONTRACT,
            field="labels",
        )
    return int(value)


def _unique_pool(pool: Sequence[str]) -> list[str]:
    ids = [_identifier(m, field="pool") for m in pool]
    duplicated = sorted({m for m in ids if ids.count(m) > 1})
    if duplicated:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"the evaluated pool must name each compound once; duplicates: {duplicated}",
            contract=_CONTRACT,
            field="pool",
            missing=duplicated,
        )
    return ids


# ---------------------------------------------------------------------------
# Track A — regression: pActivity MAE with coverage and a predeclared penalty
# ---------------------------------------------------------------------------
def regression_endpoint(
    *,
    pool: Sequence[str],
    labels: Mapping[str, float],
    predictions: Mapping[str, Mapping[str, Any]],
    failure_penalty: float,
) -> Record:
    """Mean absolute pActivity error over the identical held-out pool.

    ``failure_penalty`` is mandatory and has no default: a compound that failed
    to score is charged it, in pActivity units, so that dropping a difficult
    compound can never improve the primary number. The scored-subset mean is
    computed as well, and labelled secondary, because that is exactly the
    quantity favourable missingness inflates.
    """
    penalty = _finite(failure_penalty, field="failure_penalty")
    if penalty <= 0.0:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"the failure penalty must be positive; got {failure_penalty!r}. A penalty of "
            f"zero rewards an arm for not scoring the hard compounds",
            contract=_CONTRACT,
            field="failure_penalty",
        )
    ids = _unique_pool(pool)
    missing_labels = [m for m in ids if m not in labels]
    if missing_labels:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            f"no label for pool members {missing_labels}; the pool is fixed and a member "
            f"without a label must be removed from the split, not from this computation",
            contract=_CONTRACT,
            field="labels",
            missing=missing_labels,
        )

    errors: dict[str, float | None] = {}
    failures: list[Record] = []
    for molecule_id in ids:
        record = predictions.get(molecule_id)
        if record is None:
            failures.append({"molecule_id": molecule_id, "reason": "NO_PREDICTION"})
            errors[molecule_id] = None
            continue
        value = _endpoint_value(record, field=f"predictions[{molecule_id}]")
        if value is None:
            failures.append({"molecule_id": molecule_id, "reason": str(record["status"])})
            errors[molecule_id] = None
        else:
            label = _finite(labels[molecule_id], field=f"labels[{molecule_id}]")
            errors[molecule_id] = abs(value - label)

    scored = [e for e in errors.values() if e is not None]
    n = len(ids)
    n_failed = n - len(scored)
    charged = scored + [penalty] * n_failed
    coverage = len(scored) / n if n else 0.0
    status = "SUCCEEDED" if scored else "INSUFFICIENT_COVERAGE"
    return {
        "contract_version": CONTRACT_VERSION,
        "metric": "mae_pactivity",
        "status": status,
        "mae": statistics.fmean(charged) if scored else None,
        "n_pool": n,
        "n_scored": len(scored),
        "n_failed": n_failed,
        "coverage": coverage,
        "failure_policy": REGRESSION_FAILURE_POLICY,
        "failure_penalty": penalty,
        "failures": sorted(failures, key=lambda f: f["molecule_id"]),
        "pool": ids,
        "absolute_errors": errors,
        "secondary": {
            "mae_scored_subset": statistics.fmean(scored) if scored else None,
            "note": (
                "SECONDARY ONLY. The scored-subset mean ignores every compound the arm "
                "failed to score, so it rewards favourable missingness; the primary "
                "number above charges the predeclared penalty over the full pool."
            ),
        },
    }


def compare_regression(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    coverage_tolerance: float = COVERAGE_TOLERANCE,
) -> Record:
    """Paired point estimate plus the coverage non-regression gate.

    The gate comes first and it is not advisory: an arm that scores fewer of the
    held-out compounds than the baseline cannot be accepted on a better mean,
    however much better the mean is on what it did score.
    """
    if list(candidate.get("pool") or []) != list(baseline.get("pool") or []):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            "the candidate and the baseline were scored on different pools; every method "
            "must rank or predict the identical held-out pool",
            contract=_CONTRACT,
            field="pool",
        )
    tolerance = _finite(coverage_tolerance, field="coverage_tolerance")
    coverage_delta = float(candidate["coverage"]) - float(baseline["coverage"])
    gate = "PASS" if coverage_delta >= -tolerance else "FAIL"

    common = [
        m
        for m in candidate["pool"]
        if candidate["absolute_errors"].get(m) is not None and baseline["absolute_errors"].get(m) is not None
    ]
    subset = {
        "molecule_ids": common,
        "n": len(common),
        "candidate_mae": statistics.fmean([candidate["absolute_errors"][m] for m in common])
        if common
        else None,
        "baseline_mae": statistics.fmean([baseline["absolute_errors"][m] for m in common])
        if common
        else None,
    }

    delta = None
    relative = None
    decision = "COVERAGE_REGRESSION"
    if gate == "PASS":
        if candidate["mae"] is None or baseline["mae"] is None:
            decision = "INSUFFICIENT_COVERAGE"
        else:
            delta = float(baseline["mae"]) - float(candidate["mae"])
            relative = delta / float(baseline["mae"]) if baseline["mae"] else None
            if is_better(candidate["mae"], baseline["mae"], "mae_pactivity"):
                decision = "LOWER_MAE"
            elif float(candidate["mae"]) == float(baseline["mae"]):
                decision = "EQUAL_MAE"
            else:
                decision = "HIGHER_MAE"
    return {
        "contract_version": CONTRACT_VERSION,
        "metric": "mae_pactivity",
        "orientation": score_orientation("mae_pactivity"),
        "coverage_gate": gate,
        "coverage_delta": coverage_delta,
        "coverage_tolerance": tolerance,
        "candidate_coverage": float(candidate["coverage"]),
        "baseline_coverage": float(baseline["coverage"]),
        "delta_mae": delta,
        "relative_reduction": relative,
        "decision": decision,
        "note": (
            "A point estimate only. BENCHMARK_ACCEPTED requires the paired uncertainty "
            "interval declared in Appendix A §5 to exclude no improvement; this module "
            "does not compute that interval and no decision here is an acceptance."
        ),
        "secondary": {
            "common_scored_subset": subset,
            "note": (
                "SECONDARY ONLY. The common scored subset is a descriptive analysis and "
                "never rescues a failed coverage gate."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Label-shuffle controls
# ---------------------------------------------------------------------------
def label_shuffle_control(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
    molecule_ids: Sequence[str] | None = None,
    fraction: float = EF_FRACTION,
    n_permutations: int = 200,
    seed: int = 0,
) -> Record:
    """Does permuting the labels destroy the enrichment factor?

    If it does not, the metric is measuring something other than the
    association between the ranking and the labels. The permutation p-value is
    the share of permutations reaching the observed value or better, with the
    observed value counted (the conservative ``(r + 1) / (n + 1)`` form).
    """
    observed = enrichment_factor(labels, scores, fraction, molecule_ids=molecule_ids)
    rng = random.Random(seed)
    permuted: list[float] = []
    pool = list(labels)
    for _ in range(int(n_permutations)):
        rng.shuffle(pool)
        permuted.append(enrichment_factor(pool, scores, fraction, molecule_ids=molecule_ids))
    at_least = sum(1 for value in permuted if value >= observed)
    p_value = (at_least + 1) / (len(permuted) + 1)
    return {
        "contract_version": CONTRACT_VERSION,
        "control": "label_shuffle",
        "metric": "ef10" if abs(fraction - 0.1) < 1e-12 else f"ef{fraction:g}",
        "observed": observed,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "permuted_mean": statistics.fmean(permuted) if permuted else None,
        "permuted_max": max(permuted) if permuted else None,
        "p_value": p_value,
        "destroys_the_metric": bool(p_value <= 0.01),
        "note": (
            "A permutation that does not move the metric means the metric is not "
            "measuring the ranking/label association it claims to."
        ),
    }


def regression_shuffle_control(
    *,
    pool: Sequence[str],
    labels: Mapping[str, float],
    predictions: Mapping[str, Mapping[str, Any]],
    failure_penalty: float,
    n_permutations: int = 200,
    seed: int = 0,
) -> Record:
    """Does permuting the labels make the regression error worse?

    The same question as :func:`label_shuffle_control`, with the direction
    flipped: MAE is lower-is-better, so a permutation should *raise* it.
    """
    ids = _unique_pool(pool)
    observed_record = regression_endpoint(
        pool=ids, labels=labels, predictions=predictions, failure_penalty=failure_penalty
    )
    observed = observed_record["mae"]
    if observed is None:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            "no compound scored, so there is no error for a permutation to move",
            contract=_CONTRACT,
            field="predictions",
        )
    rng = random.Random(seed)
    values = [float(labels[m]) for m in ids]
    permuted: list[float] = []
    for _ in range(int(n_permutations)):
        rng.shuffle(values)
        shuffled = dict(zip(ids, values, strict=True))
        record = regression_endpoint(
            pool=ids, labels=shuffled, predictions=predictions, failure_penalty=failure_penalty
        )
        if record["mae"] is not None:
            permuted.append(float(record["mae"]))
    at_least = sum(1 for value in permuted if value <= observed)
    p_value = (at_least + 1) / (len(permuted) + 1)
    return {
        "contract_version": CONTRACT_VERSION,
        "control": "label_shuffle",
        "metric": "mae_pactivity",
        "observed": float(observed),
        "n_permutations": len(permuted),
        "seed": int(seed),
        "permuted_mean": statistics.fmean(permuted) if permuted else None,
        "permuted_min": min(permuted) if permuted else None,
        "p_value": p_value,
        "destroys_the_metric": bool(p_value <= 0.01),
    }


# ---------------------------------------------------------------------------
# Scaffolds
# ---------------------------------------------------------------------------
def scaffold_id(smiles: str, molecule_id: str) -> str:
    """The scaffold a molecule counts under, with acyclic families made explicit.

    A cyclic molecule counts under its canonical Bemis-Murcko scaffold. An
    acyclic one has no scaffold at all, and Step 6 already established that
    empty scaffolds are not one family — so it counts under
    ``acyclic:<molecule_id>``, which keeps two acyclic molecules two scaffolds.

    Raises:
        ContractError: ``INVALID_VALUE`` for chemistry RDKit cannot parse.
    """
    key = _identifier(molecule_id, field="molecule_id")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"{smiles!r} is not a parseable structure, so it has no scaffold",
            contract=_CONTRACT,
            field="smiles",
        )
    core = MurckoScaffold.GetScaffoldForMol(mol)
    canonical = Chem.MolToSmiles(core) if core is not None else ""
    return canonical if canonical else f"{ACYCLIC_SCAFFOLD_PREFIX}{key}"


def qualifying_scaffolds(rows: Sequence[Mapping[str, Any]]) -> int:
    """Unique scaffolds among the molecules that passed every required gate.

    Each *molecule* counts once, and each *scaffold* counts once — Appendix A
    §3.3. Compound deduplication happens first, by ``molecule_id``; two rows
    that disagree about one molecule are refused rather than silently resolved.
    A blank scaffold id is refused too, naming
    :func:`scaffold_id`'s acyclic convention, because treating every empty
    scaffold as the same one collapses a generator's acyclic output into a
    single scaffold.

    Raises:
        ContractError: ``EMPTY_IDENTIFIER`` for a blank molecule id,
            ``MISSING_FIELD`` for a missing ``qualifies`` flag or a blank
            scaffold id, ``INVALID_TYPE`` for a non-boolean flag, and
            ``INVALID_VALUE`` when two rows disagree about one molecule.
    """
    seen: dict[str, tuple[str, bool]] = {}
    for row in rows:
        molecule_id = _identifier(row.get("molecule_id"), field="molecule_id")
        if "qualifies" not in row:
            raise ContractError(
                ErrorCategory.MISSING_FIELD,
                f"molecule {molecule_id!r} carries no 'qualifies' flag; an absent gate "
                f"result is not a failed gate and must not be assumed either way",
                contract=_CONTRACT,
                field="qualifies",
                missing=["qualifies"],
            )
        flag = row["qualifies"]
        if not isinstance(flag, bool):
            raise ContractError(
                ErrorCategory.INVALID_TYPE,
                f"'qualifies' must be a bool; molecule {molecule_id!r} carries {flag!r}",
                contract=_CONTRACT,
                field="qualifies",
            )
        scaffold = str(row.get("scaffold_id") or "").strip()
        if not scaffold:
            raise ContractError(
                ErrorCategory.MISSING_FIELD,
                f"molecule {molecule_id!r} has no scaffold_id. An acyclic molecule has an "
                f"empty Bemis-Murcko scaffold and needs an explicit family id — use "
                f"scaffold_id(), which returns {ACYCLIC_SCAFFOLD_PREFIX}<molecule_id>",
                contract=_CONTRACT,
                field="scaffold_id",
                missing=["scaffold_id"],
            )
        current = (scaffold, flag)
        previous = seen.get(molecule_id)
        if previous is not None and previous != current:
            raise ContractError(
                ErrorCategory.INVALID_VALUE,
                f"two rows disagree about molecule {molecule_id!r}: {previous} vs {current}",
                contract=_CONTRACT,
                field="molecule_id",
            )
        seen[molecule_id] = current
    return len({scaffold for scaffold, flag in seen.values() if flag})


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def geometry_checks(molblock: str, *, limits: Mapping[str, float] | None = None) -> Record:
    """Named validity checks on a 3D structure, in the spirit of PoseBusters.

    Reports which checks failed rather than raising, because an invalid pose is
    a fact about that molecule and the rest of the portfolio still has to be
    evaluated. A structure that does not parse fails ``parses`` and nothing
    further is claimed about it.
    """
    lim = dict(GEOMETRY_LIMITS)
    lim.update(limits or {})
    checks: dict[str, bool | None] = dict.fromkeys(
        ("parses", "has_conformer", "is_3d", "finite_coordinates", "min_interatomic_distance", "bond_lengths")
    )
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = None
    text = str(molblock or "")
    # A `drug_agent.contracts.structure.Pose` carries its mol block through a
    # field whose whitespace handling strips the leading (empty) title line, and
    # a V2000 block without that line does not parse at all -- RDKit reads the
    # counts line one line early and reports "Cannot convert ' 0.' to unsigned
    # int". Restoring it is not leniency about invalid chemistry: it is reading
    # this repository's own pose contract. A block that is genuinely malformed
    # still fails, which `test_an_unparseable_block_fails_identity_rather_than_
    # raising` pins.
    for attempt in (text, f"\n{text}") if text and not text.startswith("\n") else (text,):
        try:
            mol = Chem.MolFromMolBlock(attempt, removeHs=False, sanitize=True)
        except Exception:  # RDKit raises a family of parse errors
            mol = None
        if mol is not None:
            break
    checks["parses"] = mol is not None
    n_atoms = 0
    if mol is not None:
        n_atoms = mol.GetNumAtoms()
        checks["has_conformer"] = mol.GetNumConformers() > 0
        if checks["has_conformer"]:
            conf = mol.GetConformer()
            points = [conf.GetAtomPosition(i) for i in range(n_atoms)]
            coords = [(p.x, p.y, p.z) for p in points]
            checks["finite_coordinates"] = all(math.isfinite(v) for point in coords for v in point)
            z_values = [c[2] for c in coords]
            spread = (max(z_values) - min(z_values)) if z_values else 0.0
            checks["is_3d"] = bool(conf.Is3D() and spread > 1e-3)
            if checks["finite_coordinates"]:
                shortest = None
                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        distance = math.dist(coords[i], coords[j])
                        shortest = distance if shortest is None else min(shortest, distance)
                checks["min_interatomic_distance"] = (
                    shortest is None or shortest >= lim["min_interatomic_distance_A"]
                )
                lengths = [
                    math.dist(
                        coords[bond.GetBeginAtomIdx()],
                        coords[bond.GetEndAtomIdx()],
                    )
                    for bond in mol.GetBonds()
                ]
                checks["bond_lengths"] = all(
                    lim["min_bond_length_A"] <= length <= lim["max_bond_length_A"] for length in lengths
                )
    failed = sorted(name for name, ok in checks.items() if ok is False)
    return {
        "contract_version": CONTRACT_VERSION,
        "passes": not failed and checks["parses"] is True,
        "failed_checks": failed,
        "checks": checks,
        "is_3d": checks["is_3d"] is True,
        "n_atoms": n_atoms,
        "limits": lim,
    }


# ---------------------------------------------------------------------------
# Calibration: structure threshold, activity gate, discrimination
# ---------------------------------------------------------------------------
def derive_structure_threshold(
    *,
    active_scores: Sequence[Any],
    metric: str,
    min_controls: int = STRUCTURE_THRESHOLD_MIN_CONTROLS,
) -> Record:
    """The frozen Track B structure-score threshold, from development controls.

    Appendix A §3.3's derivation rule, unchanged: *structure score at least as
    favourable as the median of valid known-active development controls*. A
    control whose scorer failed is invalid and contributes nothing — it does not
    become a very bad score, which would drag the median in the permissive
    direction.
    """
    orientation = score_orientation(metric)
    values, invalid = _scores_and_invalid(active_scores, field="active_scores")
    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "metric": metric,
        "orientation": orientation,
        "rule": (
            "median of the valid known-active development controls' "
            f"{metric}; a molecule qualifies when its score is at least as favourable "
            "as this value (inclusive)"
        ),
        "n_valid": len(values),
        "n_invalid": invalid,
        "min_controls": int(min_controls),
        "control_partition": "development",
    }
    if len(values) < int(min_controls):
        record.update(
            {
                "status": UNCALIBRATED,
                "reason": "INSUFFICIENT_CONTROLS",
                "threshold": None,
                "detail": (
                    f"{len(values)} valid known-active development control(s); "
                    f"{min_controls} are required before a median means anything"
                ),
            }
        )
        return record
    record.update(
        {
            "status": CALIBRATED,
            "reason": None,
            "threshold": float(statistics.median(values)),
            "detail": None,
        }
    )
    return record


def derive_endpoint_threshold(
    *,
    metric: str,
    active_values: Sequence[Any],
    min_controls: int = STRUCTURE_THRESHOLD_MIN_CONTROLS,
    scale: Mapping[str, Any] | None = None,
) -> Record:
    """A generation-quality gate, by the **same frozen rule** as the structure score.

    Appendix A §3.3 fixes one derivation rule for Track B qualification
    thresholds: *at least as favourable as the median of the valid known-active
    development controls*. :func:`derive_structure_threshold` applies it to
    ``vina_affinity``; this applies the identical rule to any other registered
    endpoint, so a widened gate set cannot smuggle in a second, friendlier rule.

    ``min_controls`` defaults to :data:`STRUCTURE_THRESHOLD_MIN_CONTROLS` (3),
    which is the *median* floor — a median is not a statistic below three
    observations. It is deliberately **not**
    :data:`ACTIVITY_GATE_MIN_CONTROLS` (10 actives and 10 inactives): that is
    Appendix A §3.2's *classification* feasibility guard, and it applies to a
    gate whose derivation estimates a true-positive and a false-positive rate
    from two labelled classes. A one-sided median of a single class estimates
    neither rate and needs neither class, so borrowing the classification floor
    here would refuse a gate for failing a test it is not taking. The two floors
    are different because the two jobs are different.

    Two ways a threshold can fail to mean anything, both checked and both
    reported as ``UNCALIBRATED`` rather than quietly loosened:

    ``VACUOUS_THRESHOLD``
        the derived bar sits at or beyond the permissive end of the endpoint's
        **declared** admissible range, so nothing the endpoint can produce could
        ever fail it. The range comes from :data:`ENDPOINT_SCALES`, i.e. from the
        model or the published scale, never from the observed controls.
    ``IMPOSSIBLE_THRESHOLD``
        no valid known-active development control passes its own bar. For an
        inclusive median this cannot happen with real numbers, and the branch
        exists so that a future endpoint whose values are not totally ordered by
        the registered orientation fails loudly instead of silently.

    A gate that calibrates is still only a bar relative to the known actives. It
    is not a safety verdict, and the record says so.
    """
    orientation = score_orientation(metric)
    spec = dict(scale or ENDPOINT_SCALES.get(metric) or {})
    precision = spec.get("precision")
    admissible = spec.get("admissible_range")

    raw, invalid = _scores_and_invalid(active_values, field="active_values")
    values = [round(v, int(precision)) for v in raw] if precision is not None else list(raw)

    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "metric": metric,
        "orientation": orientation,
        "rule": (
            f"median of the valid known-active development controls' {metric}, each rounded "
            f"to {precision} decimal place(s) first; a molecule qualifies when its score is "
            "at least as favourable as this value (inclusive). The median itself is NOT "
            "rounded: the median of two values at precision p can need p+1 digits, and "
            "rounding it would move the bar after it was derived"
        )
        if precision is not None
        else (
            f"median of the valid known-active development controls' {metric}; a molecule "
            "qualifies when its score is at least as favourable as this value (inclusive)"
        ),
        "rule_is_the_same_as": "structure_threshold (Appendix A §3.3's frozen derivation rule)",
        "n_valid": len(values),
        "n_invalid": invalid,
        "min_controls": int(min_controls),
        "min_controls_rationale": (
            "the median floor, not the classification floor: a median is not a statistic "
            "below three observations, and Appendix A §3.2's 10/10 guard applies to a gate "
            "that estimates a true-positive and a false-positive rate from two labelled "
            "classes, which this one does not"
        ),
        "control_partition": "development",
        "control_values": sorted(values),
        "precision": precision,
        "admissible_range": list(admissible) if admissible else None,
        "calibrated_score": spec.get("calibrated_score"),
        "endpoint_source": spec.get("source"),
        "endpoint_note": spec.get("note"),
    }

    if len(values) < int(min_controls):
        record.update(
            {
                "status": UNCALIBRATED,
                "reason": INSUFFICIENT_CONTROLS,
                "threshold": None,
                "n_controls_passing_their_own_bar": None,
                "detail": (
                    f"{len(values)} valid known-active development control(s); "
                    f"{min_controls} are required before a median means anything"
                ),
            }
        )
        return record

    threshold = float(statistics.median(values))
    passing = sum(1 for v in values if at_least_as_favourable(v, threshold, metric))

    if admissible is not None:
        low, high = float(admissible[0]), float(admissible[1])
        permissive = high if orientation == LOWER_IS_BETTER else low
        vacuous = threshold >= permissive if orientation == LOWER_IS_BETTER else threshold <= permissive
        if vacuous:
            record.update(
                {
                    "status": UNCALIBRATED,
                    "reason": VACUOUS_THRESHOLD,
                    "threshold": threshold,
                    "n_controls_passing_their_own_bar": passing,
                    "detail": (
                        f"the derived bar {threshold} sits at or beyond the permissive end "
                        f"{permissive} of {metric}'s declared admissible range "
                        f"[{low}, {high}], so no value the endpoint can produce could fail "
                        "it. A gate that excludes nothing is recorded as uncalibrated, not "
                        "tightened until it produces a pleasing number"
                    ),
                }
            )
            return record

    if passing == 0:
        record.update(
            {
                "status": UNCALIBRATED,
                "reason": IMPOSSIBLE_THRESHOLD,
                "threshold": threshold,
                "n_controls_passing_their_own_bar": 0,
                "detail": (
                    f"no valid known-active development control passes the bar {threshold} "
                    f"derived from those same controls; {metric} cannot be gated this way"
                ),
            }
        )
        return record

    record.update(
        {
            "status": CALIBRATED,
            "reason": None,
            "threshold": threshold,
            "n_controls_passing_their_own_bar": passing,
            "detail": None,
        }
    )
    return record


def endpoint_gate_diagnostics(
    *,
    threshold_record: Mapping[str, Any],
    values: Sequence[Any],
    label: str,
) -> Record:
    """How permissive a derived gate is on a control set it was NOT derived from.

    Reported, never acted on. A gate derived from six known actives can be very
    permissive without being vacuous, and the only honest thing to do with that
    is to publish the number beside the gate. Moving the bar because this row
    looked bad would be choosing a threshold after seeing the data, which is the
    one thing Appendix A §3.3 forbids.
    """
    metric = str(threshold_record["metric"])
    threshold = threshold_record.get("threshold")
    scored, invalid = _scores_and_invalid(values, field=f"{label}_values")
    precision = threshold_record.get("precision")
    if precision is not None:
        scored = [round(v, int(precision)) for v in scored]
    if threshold is None:
        return {
            "set": label,
            "metric": metric,
            "n_valid": len(scored),
            "n_invalid": invalid,
            "n_passing": None,
            "note": "the gate is UNCALIBRATED, so it admits and excludes nothing",
        }
    passing = sum(1 for v in scored if at_least_as_favourable(v, float(threshold), metric))
    return {
        "set": label,
        "metric": metric,
        "threshold": float(threshold),
        "n_valid": len(scored),
        "n_invalid": invalid,
        "n_passing": passing,
        "median": float(statistics.median(scored)) if scored else None,
        "note": (
            "a permissiveness diagnostic on a set the threshold was not derived from. "
            "It is reported beside the gate and is never a reason to move it"
        ),
    }


def calibrate_activity_gate(
    *,
    active_scores: Sequence[Any],
    inactive_scores: Sequence[Any],
    metric: str,
    operating_point: Mapping[str, float] | None = None,
    min_controls: Mapping[str, int] | None = None,
) -> Record:
    """The optional activity gate, or an honest ``UNCALIBRATED``.

    The rule is the work order's, exactly: choose the **lowest** development
    threshold attaining false-positive rate ≤ 0.10 and true-positive rate ≥
    0.20. "Lowest" is read as *strictest*, which for a lower-is-better score
    such as a docking energy is the numerically lowest threshold — the wording
    and the orientation agree for this programme's scorer, and for a
    higher-is-better score the strictest is the numerically highest. If no
    threshold attains the operating point, or there are too few controls to
    establish one, the gate is marked ``UNCALIBRATED``: a nicer number is not
    chosen instead.
    """
    point = dict(ACTIVITY_GATE_OPERATING_POINT)
    point.update(operating_point or {})
    floors = dict(ACTIVITY_GATE_MIN_CONTROLS)
    floors.update(min_controls or {})
    orientation = score_orientation(metric)

    actives, invalid_actives = _scores_and_invalid(active_scores, field="active_scores")
    inactives, invalid_inactives = _scores_and_invalid(inactive_scores, field="inactive_scores")

    def gated(threshold: float, values: Sequence[float]) -> int:
        if orientation == LOWER_IS_BETTER:
            return sum(1 for v in values if v <= threshold)
        return sum(1 for v in values if v >= threshold)

    candidates: list[Record] = []
    if actives and inactives:
        for threshold in sorted(set(actives) | set(inactives)):
            tpr = gated(threshold, actives) / len(actives)
            fpr = gated(threshold, inactives) / len(inactives)
            candidates.append(
                {
                    "threshold": float(threshold),
                    "true_positive_rate": tpr,
                    "false_positive_rate": fpr,
                    "attains_operating_point": bool(
                        fpr <= point["max_false_positive_rate"] and tpr >= point["min_true_positive_rate"]
                    ),
                }
            )

    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "metric": metric,
        "orientation": orientation,
        "operating_point": point,
        "min_controls": floors,
        "n_actives": len(actives),
        "n_inactives": len(inactives),
        "n_invalid": invalid_actives + invalid_inactives,
        "control_partition": "development",
        "candidates": candidates,
        "selection_rule": (
            "the strictest threshold attaining both bounds; for a LOWER_IS_BETTER score "
            "that is the numerically lowest, which is the work order's wording"
        ),
    }
    if len(actives) < floors["actives"] or len(inactives) < floors["inactives"]:
        record.update(
            {
                "status": UNCALIBRATED,
                "reason": "INSUFFICIENT_CONTROLS",
                "threshold": None,
                "true_positive_rate": None,
                "false_positive_rate": None,
                "detail": (
                    f"{len(actives)} valid known-active and {len(inactives)} valid measured-inactive "
                    f"development controls; the declared floor is {floors['actives']} and "
                    f"{floors['inactives']} (Appendix A §3.2's classification guard)"
                ),
            }
        )
        return record

    attaining = [c for c in candidates if c["attains_operating_point"]]
    if not attaining:
        record.update(
            {
                "status": UNCALIBRATED,
                "reason": "NO_OPERATING_POINT",
                "threshold": None,
                "true_positive_rate": None,
                "false_positive_rate": None,
                "detail": (
                    f"no threshold reaches true-positive rate ≥ {point['min_true_positive_rate']} "
                    f"while holding false-positive rate ≤ {point['max_false_positive_rate']}"
                ),
            }
        )
        return record

    chosen = (
        min(attaining, key=lambda c: c["threshold"])
        if orientation == LOWER_IS_BETTER
        else max(attaining, key=lambda c: c["threshold"])
    )
    record.update(
        {
            "status": CALIBRATED,
            "reason": None,
            "threshold": chosen["threshold"],
            "true_positive_rate": chosen["true_positive_rate"],
            "false_positive_rate": chosen["false_positive_rate"],
            "detail": None,
        }
    )
    return record


def register_track_b_name(*, activity_gate: Mapping[str, Any]) -> str:
    """The Track B name, decided by the gate and fixed before candidate search.

    Appendix A §3.3: with no defensible activity gate the registered track name
    must explicitly read ``STRUCTURE_ONLY_COMPUTATIONAL``, and it is not to be
    removed after seeing candidates.
    """
    return (
        STRUCTURE_AND_ACTIVITY_GATED
        if activity_gate.get("status") == CALIBRATED
        else STRUCTURE_ONLY_COMPUTATIONAL
    )


def discrimination_check(
    *, active_scores: Sequence[Any], inactive_scores: Sequence[Any], metric: str
) -> Record:
    """Can the structure scorer tell known actives from *measured* inactives?

    With no measured inactive controls the answer is ``UNTESTED`` — not
    "passed". Decoys are not substituted for measured negatives here; a caller
    that wants that comparison must declare it and pass them in explicitly.
    """
    orientation = score_orientation(metric)
    actives, invalid_actives = _scores_and_invalid(active_scores, field="active_scores")
    inactives, invalid_inactives = _scores_and_invalid(inactive_scores, field="inactive_scores")
    note = (
        "Separating known actives from measured inactives is sanity evidence that the "
        "pipeline is wired correctly. It is NOT a claim of novel affinity accuracy, and "
        "a redocking result is a pipeline check rather than an affinity benchmark "
        "(Appendix A §3.3)."
    )
    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "metric": metric,
        "orientation": orientation,
        "n_actives": len(actives),
        "n_inactives": len(inactives),
        "n_invalid": invalid_actives + invalid_inactives,
        "median_active": statistics.median(actives) if actives else None,
        "median_inactive": statistics.median(inactives) if inactives else None,
        "note": note,
    }
    if not inactives or not actives:
        record.update(
            {
                "status": "UNTESTED",
                "auc": None,
                "reason": (
                    "no measured inactive development controls were scored, so structure-score "
                    "discrimination is untested here"
                    if not inactives
                    else "no valid known-active development control was scored"
                ),
            }
        )
        return record
    wins = 0.0
    for a in actives:
        for i in inactives:
            if a == i:
                wins += 0.5
            elif is_better(a, i, metric):
                wins += 1.0
    record.update(
        {
            "status": "TESTED",
            "auc": wins / (len(actives) * len(inactives)),
            "reason": None,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Backend determinism and wrapper parity
# ---------------------------------------------------------------------------
def numerical_tolerance(repeats: Sequence[float], *, floor: float = DEFAULT_TOLERANCE_FLOOR) -> Record:
    """Freeze an allowed numerical tolerance from *observed* repeated runs.

    Both docking engines take a seed, so determinism is measurable rather than
    assumable; this takes the measurement and turns it into the tolerance every
    later parity comparison is judged against. One observation cannot establish
    a spread and is refused.
    """
    values = [_finite(v, field="repeats") for v in repeats]
    if len(values) < 2:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"a tolerance needs at least two repeated observations; got {len(values)}",
            contract=_CONTRACT,
            field="repeats",
        )
    median = statistics.median(values)
    deviation = max(abs(v - median) for v in values)
    return {
        "contract_version": CONTRACT_VERSION,
        "n_repeats": len(values),
        "values": values,
        "median": median,
        "max_abs_deviation": deviation,
        "floor": float(floor),
        "tolerance": max(deviation, float(floor)),
    }


def parity_check(*, direct: float, wrapped: float, tolerance: float, label: str) -> Record:
    """Does the repository's adapter reproduce the upstream tool's own answer?"""
    a = _finite(direct, field="direct")
    b = _finite(wrapped, field="wrapped")
    limit = _finite(tolerance, field="tolerance")
    difference = abs(a - b)
    return {
        "contract_version": CONTRACT_VERSION,
        "label": str(label),
        "direct": a,
        "wrapped": b,
        "abs_difference": difference,
        "tolerance": limit,
        "status": "AGREE" if difference <= limit else "DISAGREE",
    }


def upstream_direct_dock(
    *,
    smiles: str,
    receptor_pdb: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    seed: int = 42,
    exhaustiveness: int = 8,
    n_poses: int = 9,
    binary: str | None = None,
    obabel: str | None = None,
    timeout_s: float = 600.0,
) -> Record:
    """Dock with the upstream tools only — no repository adapter in the path.

    RDKit embeds the ligand, Meeko types it, Open Babel prepares the receptor,
    and the ``vina`` executable searches. Nothing here calls
    :class:`~drug_agent.tools.vina_tool.DockingAdapter`, which is the point: the
    evaluator must be able to check the platform's wrapper against the tools it
    wraps without trusting the wrapper to report on itself.

    The only thing borrowed from the adapter is *where the binary is*, and only
    when it is not on ``PATH``; a file path is not a scientific claim.
    """
    exe = binary or shutil.which("vina")
    if exe is None:  # the locator, not the wrapper's science
        from ...tools.vina_tool import find_vina

        exe = find_vina()
    if exe is None:
        raise CalibrationError("no 'vina' executable is resolvable; cannot run an upstream check")
    babel = obabel or shutil.which("obabel")
    if babel is None:
        raise CalibrationError("no 'obabel' executable is resolvable; cannot prepare the receptor")

    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise CalibrationError(f"unparseable ligand SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) or 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise CalibrationError(f"could not embed a 3D conformer for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)

    with tempfile.TemporaryDirectory(prefix="upstream_vina_") as tmp:
        work = Path(tmp)
        setups = MoleculePreparation().prepare(mol)
        text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise CalibrationError(f"meeko could not write a ligand PDBQT: {err}")
        ligand = work / "ligand.pdbqt"
        ligand.write_text(str(text))

        receptor = work / "receptor.pdbqt"
        convert = subprocess.run(
            [babel, str(receptor_pdb), "-O", str(receptor), "-xr"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=work,
            check=False,
        )
        if not receptor.is_file() or "ATOM" not in receptor.read_text():
            raise CalibrationError(
                f"obabel produced no receptor PDBQT (exit {convert.returncode}): "
                f"{(convert.stderr or '')[-400:]}"
            )

        out = work / "out.pdbqt"
        argv = [
            exe,
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--out",
            str(out),
            "--center_x",
            f"{float(center[0]):g}",
            "--center_y",
            f"{float(center[1]):g}",
            "--center_z",
            f"{float(center[2]):g}",
            "--size_x",
            f"{float(size[0]):g}",
            "--size_y",
            f"{float(size[1]):g}",
            "--size_z",
            f"{float(size[2]):g}",
            "--seed",
            str(int(seed)),
            "--exhaustiveness",
            str(int(exhaustiveness)),
            "--num_modes",
            str(int(n_poses)),
        ]
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=work,
            check=False,
            env={**os.environ},
        )
        if proc.returncode != 0 or not out.is_file():
            raise CalibrationError(
                f"vina exited {proc.returncode} with no output:\n{(proc.stderr or proc.stdout)[-800:]}"
            )
        scores: list[float] = []
        for line in out.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                scores.append(float(line.split()[3]))
    if not scores:
        raise CalibrationError("vina wrote an output file with no REMARK VINA RESULT line")
    return {
        "engine": "vina",
        "path": "upstream-direct (rdkit + meeko + obabel + the vina executable)",
        "binary": exe,
        "scores": scores,
        "best_score": most_favourable(scores, "vina_affinity"),
        "metric": "vina_affinity",
        "argv": argv[1:],
    }


# ---------------------------------------------------------------------------
# Track B — qualification and the portfolio
# ---------------------------------------------------------------------------
def _standardized(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(mol) if mol is not None else None


#: How a novelty exclusion set is carried when it is carried as digests rather
#: than as plaintext SMILES. Unsalted and therefore verifiable: anyone holding a
#: claimed plaintext set can recompute the digests and confirm the record encodes
#: exactly that set. A salt would raise the cost of a dictionary attack against a
#: public compound library, and would also have to be distributed to every
#: worktree that evaluates anything; it is not used, and the residual exposure is
#: recorded in the record itself rather than implied by its absence.
NOVELTY_DIGEST_ALGORITHM = "sha256(standardized_canonical_smiles)"


def novelty_digest(standardized_smiles: str) -> str:
    """The exclusion-set digest of one already-standardized canonical SMILES.

    The input must be what :func:`_standardized` returns, because the digest is
    an address of that exact string: hashing a non-canonical form of the same
    molecule would silently miss.
    """
    return hashlib.sha256(str(standardized_smiles).encode("utf-8")).hexdigest()


def novelty_digests(smiles: Iterable[str]) -> list[str]:
    """Sorted digests of *smiles*, standardizing each one first.

    Anything that does not parse is dropped, exactly as
    :func:`build_qualification` drops it from a plaintext exclusion set, so the
    two carriers describe the same set.
    """
    out = {novelty_digest(s) for s in (_standardized(x) for x in smiles) if s}
    return sorted(out)


_PROPERTY_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def qualification_hash(record: Mapping[str, Any]) -> str:
    """Content address of a qualification record, excluding its own hash field."""
    payload = {k: v for k, v in record.items() if k != "qualification_hash"}
    return content_hash(payload)


def calibration_hash(record: Mapping[str, Any]) -> str:
    """Content address of a calibration record, excluding its own hash field."""
    payload = {k: v for k, v in record.items() if k != "calibration_hash"}
    return content_hash(payload)


def build_qualification(
    *,
    target_id: str,
    track_b_name: str,
    structure_threshold: Mapping[str, Any],
    activity_gate: Mapping[str, Any],
    property_limits: Sequence[Mapping[str, Any]],
    exclusion_smiles: Sequence[str],
    evaluation_cap: int,
    dock_budget_note: str,
    split_hash: str,
    data_hash: str,
    target_manifest_hash: str,
    notes: Sequence[str] = (),
    endpoint_thresholds: Mapping[str, Mapping[str, Any]] | None = None,
    novelty_union: Mapping[str, Any] | None = None,
    conditioning: Mapping[str, Any] | None = None,
    caps: Mapping[str, Any] | None = None,
    reporting: Mapping[str, Any] | None = None,
    limits: Sequence[str] = (),
    supersedes: Mapping[str, Any] | None = None,
) -> Record:
    """The frozen ``qualification.json`` payload (Appendix A §3.3).

    Exact required metrics, thresholds, evaluator hash and novelty rules,
    written **before** candidate search. Its own content address is the last
    field computed, so a later edit is detectable.

    ``endpoint_thresholds`` are the generation-quality gates (step 15B). Only the
    ones that actually **calibrated** join ``required_gates``: a gate that could
    not be established is still serialized, with its reason, so the decision is
    inspectable — but it is never listed as required, because a required gate
    nothing enforces is worse than an absent one.
    """
    standardized = sorted({s for s in (_standardized(x) for x in exclusion_smiles) if s})
    gates = dict(endpoint_thresholds or {})
    calibrated_gates = sorted(name for name, rec in gates.items() if rec.get("status") == CALIBRATED)
    uncalibrated_gates = sorted(name for name in gates if name not in calibrated_gates)
    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "record_kind": "target_qualification",
        "target_id": target_id,
        "track": "B",
        "track_b_name": track_b_name,
        "status": CALIBRATED if structure_threshold.get("status") == CALIBRATED else UNCALIBRATED,
        "reason": structure_threshold.get("reason"),
        "primary_metric": "qualifying_scaffolds",
        "required_gates": [*QUALIFICATION_GATES, *calibrated_gates],
        "gates_declared_but_not_required": uncalibrated_gates,
        "structure_threshold": dict(structure_threshold),
        "activity_gate": dict(activity_gate),
        "endpoint_thresholds": {name: dict(rec) for name, rec in sorted(gates.items())},
        "geometry_limits": dict(GEOMETRY_LIMITS),
        "property_limits": [dict(limit) for limit in property_limits],
        "novelty": {
            "rule": (
                "a molecule whose standardized canonical SMILES is in the frozen common "
                "exclusion set cannot qualify; the set is frozen across comparator arms"
            ),
            "n_excluded": len(standardized),
            "exclusion_smiles": standardized,
            **({"primary_de_novo_exclusion": dict(novelty_union)} if novelty_union else {}),
        },
        "counting_rule": (
            "each evaluated molecule contributes at most once and each scaffold contributes "
            "at most once; an acyclic molecule counts under "
            f"{ACYCLIC_SCAFFOLD_PREFIX}<molecule_id> rather than a shared empty scaffold"
        ),
        "evaluation_cap": int(evaluation_cap),
        "dock_budget_note": dock_budget_note,
        **({"caps": dict(caps)} if caps else {}),
        **({"conditioning_stratification": dict(conditioning)} if conditioning else {}),
        **({"reporting": dict(reporting)} if reporting else {}),
        "limits_of_a_passing_molecule": list(limits),
        **({"supersedes": dict(supersedes)} if supersedes else {}),
        "score_orientation": dict(SCORE_ORIENTATION),
        "split_hash": split_hash,
        "data_hash": data_hash,
        "target_manifest_hash": target_manifest_hash,
        "evaluator_hash": evaluator_hash(),
        "registered_before_candidate_search": True,
        "contains_measured_labels": False,
        "label_disclosure": (
            "No measured potency value appears in this record. What a control set does "
            "disclose by construction is the activity *class* of the development-partition "
            "compounds it names; the development partition is permitted data and only the "
            "test partition is protected. No held-out compound is named here."
        ),
        "notes": list(notes),
    }
    record["qualification_hash"] = qualification_hash(record)
    return record


def build_calibration(
    *,
    target_id: str,
    scorer: Mapping[str, Any],
    structure_threshold: Mapping[str, Any],
    activity_gate: Mapping[str, Any],
    discrimination: Mapping[str, Any],
    tolerance: Mapping[str, Any],
    parity: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    track_a: Mapping[str, Any],
    track_b_name: str,
    shuffle_controls: Sequence[Mapping[str, Any]],
    notes: Sequence[str] = (),
) -> Record:
    """The frozen ``calibration.json`` payload: operating points and hashes."""
    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "record_kind": "target_evaluator_calibration",
        "target_id": target_id,
        "scorer": dict(scorer),
        "score_orientation": dict(SCORE_ORIENTATION),
        "structure_threshold": dict(structure_threshold),
        "activity_gate": dict(activity_gate),
        "activity_gate_operating_point": dict(ACTIVITY_GATE_OPERATING_POINT),
        "discrimination": dict(discrimination),
        "tolerance": dict(tolerance),
        "parity": [dict(p) for p in parity],
        "controls": dict(controls),
        "track_a": dict(track_a),
        "track_b_name": track_b_name,
        "shuffle_controls": [dict(c) for c in shuffle_controls],
        "failure_policies": {
            "regression": REGRESSION_FAILURE_POLICY,
            "ranking": RANKING_FAILURE_POLICY,
        },
        "coverage_tolerance": COVERAGE_TOLERANCE,
        "evaluator_hash": evaluator_hash(),
        "registered_before_candidate_search": True,
        "contains_measured_labels": False,
        "label_disclosure": (
            "No measured potency value appears in this record. What a control set does "
            "disclose by construction is the activity *class* of the development-partition "
            "compounds it names; the development partition is permitted data and only the "
            "test partition is protected. No held-out compound is named here."
        ),
        "notes": list(notes),
    }
    record["calibration_hash"] = calibration_hash(record)
    return record


def evaluate_portfolio(molecules: Sequence[Mapping[str, Any]], *, qualification: Mapping[str, Any]) -> Record:
    """Apply the frozen Track B gates to a portfolio and count its scaffolds.

    Refuses to run at all when the qualification's mandatory structure
    threshold was never established: a qualifying count produced against an
    invalid calibration would look exactly like a real one.
    """
    require_calibrated(qualification, what="Track B qualification")
    threshold_record = qualification["structure_threshold"]
    require_calibrated(threshold_record, what="Track B structure-score threshold")
    metric = str(threshold_record["metric"])
    threshold = float(threshold_record["threshold"])
    limits = qualification.get("property_limits") or []
    novelty_rule = qualification.get("novelty") or {}
    excluded = set(novelty_rule.get("exclusion_smiles") or [])
    # The exclusion set may be carried as plaintext SMILES, as digests of the
    # same standardized SMILES, or as both, and the membership test is on the
    # standardized canonical SMILES either way — so the two carriers enforce the
    # identical rule. Digests exist because every campaign has to read this
    # record to learn the gates it is judged by, and a plaintext set turned that
    # unavoidable read into a free copy of every molecule this programme has
    # already accepted.
    excluded_digests: set[str] = set()
    digest_rule = novelty_rule.get("exclusion_digests")
    if digest_rule is not None:
        if not isinstance(digest_rule, Mapping):
            raise ValueError(
                "novelty.exclusion_digests must be a record carrying 'algorithm' and 'digests'; "
                f"found {type(digest_rule).__name__}"
            )
        algorithm = str(digest_rule.get("algorithm") or "")
        if algorithm != NOVELTY_DIGEST_ALGORITHM:
            # Fail closed. An exclusion set this evaluator cannot compute over
            # would make every molecule look novel, which is indistinguishable
            # from a correctly empty set — the one failure a novelty gate must
            # never have, and the shape of the exploit that scored a perfect 150
            # of 150 by resubmitting molecules this host had already accepted.
            raise ValueError(
                f"novelty.exclusion_digests declares algorithm {algorithm!r}, which this evaluator "
                f"cannot compute; it computes {NOVELTY_DIGEST_ALGORITHM!r}. Refusing to evaluate, "
                "because an unreadable exclusion set would pass every molecule"
            )
        excluded_digests = {str(d) for d in (digest_rule.get("digests") or ())}
        declared_digests = digest_rule.get("n")
        if declared_digests is not None and int(declared_digests) != len(excluded_digests):
            raise ValueError(
                f"novelty.exclusion_digests declares n={int(declared_digests)} but carries "
                f"{len(excluded_digests)} distinct digests"
            )
    # The declared size and the enforced size have to agree. They did not once:
    # the record named a 611-molecule de-novo union while the field this function
    # reads held 30, so every Track B count on this host was measured against a
    # rule the record did not name. An absent count is not invented here, but a
    # present one is checked.
    declared_total = novelty_rule.get("n_excluded")
    if declared_total is not None and int(declared_total) != len(excluded) + len(excluded_digests):
        raise ValueError(
            f"novelty declares n_excluded={int(declared_total)} but this evaluator can enforce "
            f"{len(excluded)} plaintext SMILES plus {len(excluded_digests)} digests; a declared "
            "exclusion set wider than the enforced one is exactly the defect this check catches"
        )
    declared_gates = dict(qualification.get("endpoint_thresholds") or {})
    endpoint_gates = {n: r for n, r in sorted(declared_gates.items()) if r.get("status") == CALIBRATED}
    unapplied_gates = [
        {"gate": n, "status": r.get("status"), "reason": r.get("reason"), "metric": r.get("metric")}
        for n, r in sorted(declared_gates.items())
        if r.get("status") != CALIBRATED
    ]

    seen: set[str] = set()
    rows: list[Record] = []
    funnel = {
        "n_submitted": len(molecules),
        "n_duplicate_submissions": 0,
        "n_invalid": 0,
        "n_geometry_failed": 0,
        "n_properties_failed": 0,
        "n_structure_score_failed": 0,
        "n_scoring_failed": 0,
        "n_novelty_failed": 0,
    }
    for name in endpoint_gates:
        funnel[f"n_{name}_failed"] = 0
        funnel[f"n_{name}_not_evaluated"] = 0

    for entry in molecules:
        molecule_id = _identifier(entry.get("molecule_id"), field="molecule_id")
        if molecule_id in seen:
            funnel["n_duplicate_submissions"] += 1
            continue
        seen.add(molecule_id)

        gates: dict[str, Record] = {}
        failed: list[str] = []
        canonical = _standardized(str(entry.get("smiles") or ""))
        gates["identity"] = {"passes": canonical is not None, "canonical_smiles": canonical}
        row_scaffold: str | None = None
        if canonical is None:
            failed.append("identity")
            funnel["n_invalid"] += 1
        else:
            row_scaffold = scaffold_id(canonical, molecule_id)

            geometry = geometry_checks(str(entry.get("pose_molblock") or ""))
            gates["geometry"] = geometry
            if not geometry["passes"]:
                failed.append("geometry")
                funnel["n_geometry_failed"] += 1

            properties = entry.get("properties") or {}
            breaches = [limit for limit in limits if not _property_passes(properties, limit)]
            gates["properties"] = {"passes": not breaches, "breaches": breaches}
            if breaches:
                failed.append("properties")
                funnel["n_properties_failed"] += 1

            score = _endpoint_value(entry.get("structure_score"), field=f"structure_score[{molecule_id}]")
            if score is None:
                gates["structure_score"] = {
                    "passes": False,
                    "reason": "SCORING_FAILED",
                    "value": None,
                    "threshold": threshold,
                }
                failed.append("structure_score")
                funnel["n_scoring_failed"] += 1
            else:
                ok = at_least_as_favourable(score, threshold, metric)
                gates["structure_score"] = {
                    "passes": ok,
                    "reason": None if ok else "LESS_FAVOURABLE_THAN_THRESHOLD",
                    "value": score,
                    "threshold": threshold,
                    "metric": metric,
                }
                if not ok:
                    failed.append("structure_score")
                    funnel["n_structure_score_failed"] += 1

            novel = canonical not in excluded and novelty_digest(canonical) not in excluded_digests
            gates["novelty"] = {"passes": novel}
            if not novel:
                failed.append("novelty")
                funnel["n_novelty_failed"] += 1

            # Generation-quality gates (step 15B). A molecule nobody scored is
            # never a clean one: an absent endpoint fails its gate as
            # NOT_EVALUATED and is counted as such, which is the same convention
            # `safety_tool` uses for a molecule its panel did not reach.
            endpoints = entry.get("endpoints") or {}
            for name, gate_record in endpoint_gates.items():
                gate_metric = str(gate_record["metric"])
                bar = float(gate_record["threshold"])
                precision = gate_record.get("precision")
                # An absent key means nobody offered this molecule to that
                # capability at all; a present record that did not succeed means
                # the capability was asked and failed. Both refuse the molecule,
                # and neither is allowed to become a favourable number. Anything
                # else — a bare float, say — still raises, because a bare float
                # cannot distinguish a measured zero from an absence.
                observed = (
                    _endpoint_value(endpoints[gate_metric], field=f"{gate_metric}[{molecule_id}]")
                    if gate_metric in endpoints
                    else None
                )
                if observed is None:
                    gates[name] = {
                        "passes": False,
                        "reason": "NOT_EVALUATED",
                        "value": None,
                        "threshold": bar,
                        "metric": gate_metric,
                    }
                    failed.append(name)
                    funnel[f"n_{name}_failed"] += 1
                    funnel[f"n_{name}_not_evaluated"] += 1
                    continue
                value = round(observed, int(precision)) if precision is not None else observed
                ok = at_least_as_favourable(value, bar, gate_metric)
                gates[name] = {
                    "passes": ok,
                    "reason": None if ok else "LESS_FAVOURABLE_THAN_THRESHOLD",
                    "value": value,
                    "threshold": bar,
                    "metric": gate_metric,
                }
                if not ok:
                    failed.append(name)
                    funnel[f"n_{name}_failed"] += 1

        rows.append(
            {
                "molecule_id": molecule_id,
                "scaffold_id": row_scaffold,
                "qualifies": not failed,
                "failed_gates": failed,
                "gates": gates,
            }
        )

    countable = [
        {"molecule_id": r["molecule_id"], "scaffold_id": r["scaffold_id"], "qualifies": r["qualifies"]}
        for r in rows
        if r["scaffold_id"]
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "SUCCEEDED",
        "track": "B",
        "track_b_name": qualification.get("track_b_name"),
        "primary_metric": "qualifying_scaffolds",
        "qualifying_scaffolds": qualifying_scaffolds(countable),
        "n_evaluated": len(seen),
        "n_qualifying_compounds": sum(1 for r in rows if r["qualifies"]),
        "rows": rows,
        "funnel": funnel,
        "applied_endpoint_gates": sorted(endpoint_gates),
        "unapplied_endpoint_gates": unapplied_gates,
        "qualification_hash": qualification.get("qualification_hash"),
        "evaluator_hash": evaluator_hash(),
        "note": (
            "A zero here is a measured fixed-budget yield, not an affinity of zero, and "
            "qualifying computationally is not evidence of biological activity."
        ),
    }


def _property_passes(properties: Mapping[str, Any], limit: Mapping[str, Any]) -> bool:
    key = str(limit.get("property") or "")
    current: Any = properties
    for part in key.split("."):
        current = current.get(part) if isinstance(current, Mapping) else None
    if current is None:
        return False
    op = _PROPERTY_OPS.get(str(limit.get("op") or "<="))
    if op is None:
        return False
    try:
        return bool(op(current, limit.get("value")))
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Track C — efficiency
# ---------------------------------------------------------------------------
def efficiency_comparison(
    *,
    candidate_cost: Mapping[str, Any],
    baseline_cost: Mapping[str, Any],
    candidate_quality: Mapping[str, Any],
    baseline_quality: Mapping[str, Any],
    quality_metric: str,
    resource_metric: str,
    margin: float = DEFAULT_QUALITY_MARGIN,
) -> Record:
    """Resource reduction, gated on prespecified quality non-inferiority.

    Costs and qualities arrive as endpoint records, so telemetry this
    deployment cannot measure stays ``UNAVAILABLE`` and never becomes a zero
    cost — which would make the unmeasured arm look free.
    """
    cost_orientation = score_orientation(resource_metric)
    quality_orientation = score_orientation(quality_metric)
    c_cost = _endpoint_value(candidate_cost, field="candidate_cost")
    b_cost = _endpoint_value(baseline_cost, field="baseline_cost")
    c_quality = _endpoint_value(candidate_quality, field="candidate_quality")
    b_quality = _endpoint_value(baseline_quality, field="baseline_quality")

    base: Record = {
        "contract_version": CONTRACT_VERSION,
        "track": "C",
        "resource_metric": resource_metric,
        "resource_orientation": cost_orientation,
        "quality_metric": quality_metric,
        "quality_orientation": quality_orientation,
        "margin": float(margin),
        "candidate_cost": c_cost,
        "baseline_cost": b_cost,
        "candidate_quality": c_quality,
        "baseline_quality": b_quality,
    }
    if None in (c_cost, b_cost, c_quality, b_quality) or not b_cost:
        base.update(
            {
                "status": "UNAVAILABLE",
                "resource_reduction": None,
                "quality_degradation": None,
                "decision": "UNMEASURED",
                "note": (
                    "Unavailable telemetry is unknown, never zero (Appendix A §4). No "
                    "efficiency claim can rest on an unmeasured arm."
                ),
            }
        )
        return base

    assert c_cost is not None and b_cost is not None
    assert c_quality is not None and b_quality is not None
    reduction = (b_cost - c_cost) / b_cost if cost_orientation == LOWER_IS_BETTER else None
    raw = (c_quality - b_quality) if quality_orientation == LOWER_IS_BETTER else (b_quality - c_quality)
    degradation = raw / abs(b_quality) if b_quality else None
    if degradation is not None and degradation > float(margin):
        decision = "QUALITY_REGRESSION"
    elif reduction is not None and reduction > 0:
        decision = "CHEAPER_AND_NON_INFERIOR"
    else:
        decision = "NO_REDUCTION"
    base.update(
        {
            "status": "SUCCEEDED",
            "resource_reduction": reduction,
            "quality_degradation": degradation,
            "decision": decision,
            "note": (
                "Appendix A §5's practical-gain target is a 20 % reduction with the quality "
                "non-inferiority margin satisfied; this is a point estimate, not an acceptance."
            ),
        }
    )
    return base


__all__ = [
    "ACTIVITY_GATE_MIN_CONTROLS",
    "ACTIVITY_GATE_OPERATING_POINT",
    "ACYCLIC_SCAFFOLD_PREFIX",
    "BENCHMARK_VERSION",
    "CALIBRATED",
    "COVERAGE_TOLERANCE",
    "DEFAULT_QUALITY_MARGIN",
    "DEFAULT_TOLERANCE_FLOOR",
    "EF_FRACTION",
    "ENDPOINT_SCALES",
    "GEOMETRY_LIMITS",
    "IMPOSSIBLE_THRESHOLD",
    "INSUFFICIENT_CONTROLS",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "QUALIFICATION_GATES",
    "RANKING_FAILURE_POLICY",
    "REGRESSION_FAILURE_POLICY",
    "SCORE_ORIENTATION",
    "WRAPPER_PARITY_TOLERANCE",
    "STRUCTURE_AND_ACTIVITY_GATED",
    "STRUCTURE_ONLY_COMPUTATIONAL",
    "STRUCTURE_THRESHOLD_MIN_CONTROLS",
    "TRACK_A_PRIMARY_MODE",
    "TRACK_A_RANKING_STATUS",
    "UNCALIBRATED",
    "VACUOUS_THRESHOLD",
    "CalibrationError",
    "at_least_as_favourable",
    "build_calibration",
    "build_qualification",
    "calibrate_activity_gate",
    "calibration_hash",
    "compare_regression",
    "derive_endpoint_threshold",
    "derive_structure_threshold",
    "discrimination_check",
    "endpoint_gate_diagnostics",
    "efficiency_comparison",
    "enrichment_factor",
    "enrichment_record",
    "evaluate_portfolio",
    "novelty_digest",
    "novelty_digests",
    "NOVELTY_DIGEST_ALGORITHM",
    "evaluator_hash",
    "geometry_checks",
    "is_better",
    "label_shuffle_control",
    "most_favourable",
    "numerical_tolerance",
    "parity_check",
    "qualification_hash",
    "qualifying_scaffolds",
    "ranking_endpoint",
    "regression_endpoint",
    "regression_shuffle_control",
    "register_track_b_name",
    "require_calibrated",
    "scaffold_id",
    "score_orientation",
    "upstream_direct_dock",
]
