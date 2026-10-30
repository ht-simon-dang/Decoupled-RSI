"""Synthetic distillation with a teacher table that says where it came from.

Step 12 of the per-target evolution program. Two separate things live here and
the separation is the point.

**A teacher pass is not a training step.** Step 11's third campaign proposed
``distill``, named a real teacher endpoint, and failed because the teacher rows
did not exist: this platform has no single action that both produces teacher
scores and learns from them. That is not a bug to route around by letting
``distill`` quietly score its own molecules — a student whose teacher scores
were produced inside the same call has no record of *which* computation made
them. So the teacher pass is a **first-class artifact of its own**:
:func:`build_teacher_table` runs the named computational endpoint, charges it to
the ledger's ``TEACHER_GENERATION`` phase, and writes a content-addressed table
in which every row carries teacher/checkpoint, molecule/pose/receptor,
evaluator, experiment and data-partition identity. :func:`run_ablation` then
*consumes* that file and never scores anything.

**A teacher score is not a measurement.** Appendix A §6 keeps measured-label and
teacher-score fields apart, and :func:`validate_teacher_row` is where that stops
being a convention. A row whose ``source_kind`` is ``teacher_prediction`` may
not carry an experimental ``evidence_type``, may not name a measured endpoint as
its ``metric``, and may not put a measured quantity in ``value`` — the measured
pActivity of the same compound travels beside it in ``measured_p_activity``,
where nothing can train on it by accident. The rejection is
``UNSUPPORTED_CLAIM``, not ``MISSING_FIELD``, so a mislabelled row is
distinguishable from an incomplete one.

What the student learns here is therefore stated precisely: the **calibrated
docking endpoint** under the frozen receptor, box and seed. It is a Track C
efficiency artifact. Better agreement with the teacher is better imitation of a
computation and is **not** evidence of stronger biology; the descriptive
correlation against measured potency is reported separately and labelled as
descriptive.

The generator route the work order names first — REINVENT4 transfer learning on
selected molecules — is not taken, because it is not reachable. Step 8 measured
``molecule.generate.reinvent4`` as ``AVAILABLE_EXECUTED`` for sampling and
``UNAVAILABLE`` for fitting, and :func:`training_capability_report` re-probes
that on this host at run time rather than quoting it. An honest ``UNAVAILABLE``
is recorded; no configuration field is passed to an adapter that would ignore
it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import ContractError, ErrorCategory

Record = dict[str, Any]

#: Bumped when the meaning of a teacher row changes. Stored in every table.
CONTRACT_VERSION = "1.0.0"

_CONTRACT = "TeacherRow"


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
#: What produced the number in ``value``.
SOURCE_KINDS: tuple[str, ...] = ("teacher_prediction", "measured_label")

#: Evidence labels a row may carry. The first group is computational, the second
#: experimental. ``MEASURED`` and ``EXPERIMENTAL`` are *recognised spellings* of
#: an experimental label rather than unknown strings: a row that says
#: ``MEASURED`` has to be refused for saying something false about its evidence,
#: not for using a word the vocabulary happens not to list. That is what makes
#: the negative test prove what it claims to prove.
COMPUTATIONAL_EVIDENCE: tuple[str, ...] = ("PREDICTED", "CALCULATED", "INFERRED")
EXPERIMENTAL_EVIDENCE: tuple[str, ...] = ("OBSERVED", "CURATED", "MEASURED", "EXPERIMENTAL")
EVIDENCE_TYPES: tuple[str, ...] = COMPUTATIONAL_EVIDENCE + EXPERIMENTAL_EVIDENCE

#: Curation is a *separate* axis from evidence. A human-reviewed teacher
#: prediction is still a prediction; ``CURATED`` as a curation status does not
#: promote it, which is exactly why the two cannot share one field.
CURATION_STATUSES: tuple[str, ...] = ("UNREVIEWED", "AUTOMATIC", "HUMAN_REVIEWED", "SUPERSEDED")

#: Endpoint states, mirrored from the frozen Appendix A §7.1 contract.
TEACHER_STATES: tuple[str, ...] = ("SUCCEEDED", "FAILED", "UNAVAILABLE", "INVALID_INPUT")

#: Where the structure itself came from. The distinction the synthetic-weight
#: sweep is *about*: anchor rows are real compounds with real provenance,
#: synthetic rows are structures a generator invented.
STRUCTURE_ORIGINS: tuple[str, ...] = ("REAL_MEASURED_COMPOUND", "GENERATED_STRUCTURE")

#: Metrics that are measured biology. A ``teacher_prediction`` naming one of
#: these has claimed to be an assay, whatever its ``evidence_type`` says.
MEASURED_METRICS: frozenset[str] = frozenset(
    {"pactivity", "p_activity", "pki", "pic50", "pkd", "ki", "ic50", "kd", "ec50", "affinity_nm"}
)

#: Partitions a training row may declare. ``test`` is listed so a row that names
#: it is refused *by name* rather than slipping through as an unknown string.
#:
#: ``data_partition`` says **which side of the frozen split this row may be used
#: on**, not where the molecule came from. A generated structure is not a member
#: of any partition; it declares ``train`` because that is the only side it is
#: admissible on, and ``structure_origin`` is the field that says it was
#: generated. Keeping those two facts in two fields is the same separation as
#: keeping the measured and the predicted quantity apart.
PARTITIONS: tuple[str, ...] = ("train", "development", "test")

#: Required on every row. Absence of a key is never evidence of absence of a
#: measurement, so ``measured_p_activity`` and ``uncertainty`` are required to be
#: *present* even when their value is null.
REQUIRED_TEACHER_FIELDS: tuple[str, ...] = (
    "row_id",
    "source_kind",
    "evidence_type",
    "curation_status",
    "structure_origin",
    "status",
    "metric",
    "value",
    "unit",
    "teacher_id",
    "teacher_checkpoint",
    "molecule_id",
    "pose_id",
    "receptor_hash",
    "evaluator_hash",
    "experiment_id",
    "data_partition",
    "family_id",
    "smiles",
    "measured_p_activity",
    "uncertainty",
    "uncertainty_state",
    "selection_rationale",
)

#: Accepted but not required. Anything outside both sets is ``UNKNOWN_FIELD``:
#: a row carrying a key nobody declared is a row whose meaning is not agreed.
OPTIONAL_TEACHER_FIELDS: tuple[str, ...] = (
    "connectivity_key",
    "aliases",
    "engine_version",
    "n_poses",
    "objective_vector",
    "error_code",
    "seconds",
    "notes",
    "contract_version",
)

#: Identity a row must name for its number to be traceable. Named separately
#: from :data:`REQUIRED_TEACHER_FIELDS` because the work order enumerates
#: exactly these and a test asserts the two agree.
IDENTITY_FIELDS: tuple[str, ...] = (
    "teacher_id",
    "teacher_checkpoint",
    "molecule_id",
    "pose_id",
    "receptor_hash",
    "evaluator_hash",
    "experiment_id",
    "data_partition",
)

#: Identity fields that are legitimately null in one stated case each.
#: ``pose_id`` has no value when the teacher call did not succeed — a failed
#: backend call is a missing outcome, not a pose — and a ``measured_label`` row
#: has no pose or receptor at all because an assay is not a docking run.
_POSE_OPTIONAL_STATES: frozenset[str] = frozenset({"FAILED", "UNAVAILABLE", "INVALID_INPUT"})


def _fail(
    category: ErrorCategory,
    message: str,
    *,
    field: str | None = None,
    missing: Sequence[str] = (),
    contract: str = _CONTRACT,
) -> Any:
    raise ContractError(category, message, contract=contract, field=field, missing=missing)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_hash(payload: Any) -> str:
    """Content address of a JSON-serialisable record, independent of key order."""
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        _fail(
            ErrorCategory.EMPTY_IDENTIFIER,
            f"{field} is blank; a blank identifier cannot address, deduplicate or trace anything",
            field=field,
        )
    return text


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
def validate_teacher_row(row: Mapping[str, Any]) -> Record:
    """Validate one distillation training row and return it as JSON-ready data.

    Step 12 owns this; :func:`drug_agent.selfevolve.target.api.validate_teacher_row`
    is the stable name the program's acceptance checks call.

    Three things it refuses to do, because each one is how a synthetic training
    set stops being traceable:

    * it never lets a teacher prediction wear an experimental label. A row whose
      ``source_kind`` is ``teacher_prediction`` and whose ``evidence_type`` is
      any of :data:`EXPERIMENTAL_EVIDENCE` is ``UNSUPPORTED_CLAIM`` on
      ``evidence_type`` — and so is the mirror image, a ``measured_label`` row
      claiming ``PREDICTED``. The same rule catches the relabelling done through
      the back door: a ``teacher_prediction`` whose ``metric`` is a measured
      endpoint (:data:`MEASURED_METRICS`), or which puts a number in
      ``measured_p_activity`` *and* claims it as its own ``value``;
    * it never accepts an unnamed computation. Teacher, checkpoint, molecule,
      pose, receptor, evaluator, experiment and partition are all required
      (:data:`IDENTITY_FIELDS`), and ``pose_id`` may be null only when the
      teacher call did not succeed, because a failed backend call has no pose
      and is not a low score;
    * it never reads a missing key as a null value. ``measured_p_activity`` and
      ``uncertainty`` must be *present*; an absent key is refused as
      ``MISSING_FIELD`` so "we did not measure it" and "we did not say" stay
      different statements.

    The status/value pairing is not restated here: it is delegated to
    :func:`drug_agent.selfevolve.target.api.endpoint_result`, which Step 2 froze,
    so a genuine ``0.0`` survives and a non-finite value is refused.

    >>> row = example_teacher_row()
    >>> validate_teacher_row(row)["source_kind"]
    'teacher_prediction'

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``MISSING_FIELD`` for an absent key, ``UNKNOWN_FIELD`` for
            an undeclared one, ``INVALID_ENUM`` for an unregistered vocabulary
            value, ``EMPTY_IDENTIFIER`` for a blank identifier,
            ``INVALID_REFERENCE`` for a hash that is not a content address,
            ``INCONSISTENT_STATUS`` for a status/value disagreement,
            ``TEST_LEAKAGE`` for a row from the held-out partition, and
            ``UNSUPPORTED_CLAIM`` for mislabelled evidence.
    """
    if not isinstance(row, Mapping):
        _fail(ErrorCategory.INVALID_TYPE, f"a teacher row must be a mapping, not {type(row).__name__}")

    missing = [f for f in REQUIRED_TEACHER_FIELDS if f not in row]
    if missing:
        _fail(
            ErrorCategory.MISSING_FIELD,
            f"a teacher row is missing {missing}; an absent key is not a null value",
            field=missing[0],
            missing=missing,
        )
    allowed = set(REQUIRED_TEACHER_FIELDS) | set(OPTIONAL_TEACHER_FIELDS)
    unknown = sorted(k for k in row if k not in allowed)
    if unknown:
        _fail(
            ErrorCategory.UNKNOWN_FIELD,
            f"a teacher row declares {unknown}, which no contract defines",
            field=unknown[0],
            missing=unknown,
        )

    out: Record = {k: row[k] for k in REQUIRED_TEACHER_FIELDS}
    for key in OPTIONAL_TEACHER_FIELDS:
        if key in row:
            out[key] = row[key]
    out["contract_version"] = CONTRACT_VERSION

    source_kind = _text(row["source_kind"], field="source_kind")
    if source_kind not in SOURCE_KINDS:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"source_kind {source_kind!r} is not one of {list(SOURCE_KINDS)}",
            field="source_kind",
        )
    evidence_type = _text(row["evidence_type"], field="evidence_type")
    if evidence_type not in EVIDENCE_TYPES:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"evidence_type {evidence_type!r} is not one of {list(EVIDENCE_TYPES)}",
            field="evidence_type",
        )
    curation = _text(row["curation_status"], field="curation_status")
    if curation not in CURATION_STATUSES:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"curation_status {curation!r} is not one of {list(CURATION_STATUSES)}",
            field="curation_status",
        )
    origin = _text(row["structure_origin"], field="structure_origin")
    if origin not in STRUCTURE_ORIGINS:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"structure_origin {origin!r} is not one of {list(STRUCTURE_ORIGINS)}",
            field="structure_origin",
        )
    status = _text(row["status"], field="status")
    if status not in TEACHER_STATES:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"status {status!r} is not one of {list(TEACHER_STATES)}",
            field="status",
        )
    partition = _text(row["data_partition"], field="data_partition")
    if partition not in PARTITIONS:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"data_partition {partition!r} is not one of {list(PARTITIONS)}",
            field="data_partition",
        )
    if partition == "test":
        _fail(
            ErrorCategory.TEST_LEAKAGE,
            "a teacher row from the held-out partition cannot be a training row; the "
            "exclusion boundary quarantines it before training, and naming it here "
            "would be the leak the boundary exists to stop",
            field="data_partition",
        )

    # -- identity ---------------------------------------------------------
    for field in IDENTITY_FIELDS:
        if field == "pose_id":
            continue
        # An assay has no receptor. A null there is the honest record; a
        # borrowed docking receptor hash would be a false provenance.
        if field == "receptor_hash" and source_kind == "measured_label" and row[field] is None:
            continue
        _text(row[field], field=field)
    for field in ("teacher_checkpoint", "receptor_hash", "evaluator_hash"):
        value = row[field]
        if value is None and field == "receptor_hash" and source_kind == "measured_label":
            continue
        text = str(value or "")
        if not _is_content_address(text):
            _fail(
                ErrorCategory.INVALID_REFERENCE,
                f"{field}={text!r} is not a content address of the form '<algo>:<hexdigest>'; "
                f"a provenance field that cannot be resolved is not provenance",
                field=field,
            )
    pose_id = row["pose_id"]
    if (source_kind == "teacher_prediction" and status == "SUCCEEDED") or pose_id is not None:
        _text(pose_id, field="pose_id")
    _text(row["row_id"], field="row_id")
    _text(row["family_id"], field="family_id")
    _text(row["smiles"], field="smiles")
    _text(row["metric"], field="metric")
    _text(row["selection_rationale"], field="selection_rationale")

    # -- status / value ---------------------------------------------------
    from .api import endpoint_result

    try:
        checked = endpoint_result(status, row["value"])
    except ValueError as exc:
        _fail(ErrorCategory.INCONSISTENT_STATUS, str(exc), field="value")
    out["value"] = checked["value"]
    if status == "SUCCEEDED":
        _text(row["unit"], field="unit")
    uncertainty_state = _text(row["uncertainty_state"], field="uncertainty_state")
    if uncertainty_state not in ("MEASURED", "UNAVAILABLE"):
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"uncertainty_state {uncertainty_state!r} is not one of ['MEASURED', 'UNAVAILABLE']",
            field="uncertainty_state",
        )
    uncertainty = row["uncertainty"]
    if uncertainty_state == "MEASURED":
        out["uncertainty"] = _finite(uncertainty, field="uncertainty")
    elif uncertainty is not None:
        _fail(
            ErrorCategory.INVALID_VALUE,
            f"uncertainty_state is UNAVAILABLE and uncertainty is {uncertainty!r}; "
            f"an unmeasured uncertainty is null, never a number",
            field="uncertainty",
        )
    measured = row["measured_p_activity"]
    if measured is not None:
        out["measured_p_activity"] = _finite(measured, field="measured_p_activity")

    # -- label semantics --------------------------------------------------
    if source_kind == "teacher_prediction":
        if evidence_type in EXPERIMENTAL_EVIDENCE:
            _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"source_kind='teacher_prediction' with evidence_type={evidence_type!r}: this row "
                f"was produced by the computational endpoint {row['teacher_id']!r} and labelling "
                f"it as experimental evidence would let a teacher score be trained on, reported "
                f"and cited as a measured affinity. A teacher prediction carries one of "
                f"{list(COMPUTATIONAL_EVIDENCE)}; the measured quantity for the same compound "
                f"travels beside it in 'measured_p_activity'",
                field="evidence_type",
            )
        if str(row["metric"]).strip().lower() in MEASURED_METRICS:
            _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"source_kind='teacher_prediction' with metric={row['metric']!r}: a teacher row "
                f"may not name a measured biological endpoint as the quantity it produced, "
                f"whatever its evidence_type says. Name the computational endpoint the teacher "
                f"actually computed",
                field="metric",
            )
        if measured is not None and out["value"] is not None and float(measured) == float(out["value"]):
            _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                "a teacher row whose 'value' equals its 'measured_p_activity' has copied the "
                "measurement into the teacher field; the two quantities are kept apart so that "
                "nothing trains on a measured label by accident",
                field="value",
            )
    else:
        if evidence_type in COMPUTATIONAL_EVIDENCE:
            _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"source_kind='measured_label' with evidence_type={evidence_type!r}: a real "
                f"measurement labelled as a computation loses the provenance that makes it "
                f"usable as a label. A measured row carries one of "
                f"{list(EXPERIMENTAL_EVIDENCE)}",
                field="evidence_type",
            )
        if str(row["metric"]).strip().lower() not in MEASURED_METRICS:
            _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"source_kind='measured_label' with metric={row['metric']!r}: a measured row must "
                f"name a measured endpoint, one of {sorted(MEASURED_METRICS)}",
                field="metric",
            )
    return out


def _is_content_address(text: str) -> bool:
    if ":" not in text:
        return False
    algo, _, digest = text.partition(":")
    hexadecimal = all(c in "0123456789abcdef" for c in digest)
    return bool(algo) and algo.islower() and len(digest) >= 32 and hexadecimal


def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(ErrorCategory.INVALID_TYPE, f"{field}={value!r} is not a number", field=field)
    if not math.isfinite(number):
        _fail(
            ErrorCategory.NON_FINITE_VALUE,
            f"{field}={value!r} is not finite and is therefore not a measurement",
            field=field,
        )
    return number


def example_teacher_row(**overrides: Any) -> Record:
    """A complete, valid teacher row. **A synthetic engineering fixture.**

    It exists so a negative test can start from something that *validates* and
    change exactly one field, which is the only way a rejection proves what it
    claims to prove. It is not research data and carries no real measurement.
    """
    row: Record = {
        "row_id": "tr_0001",
        "source_kind": "teacher_prediction",
        "evidence_type": "PREDICTED",
        "curation_status": "AUTOMATIC",
        "structure_origin": "REAL_MEASURED_COMPOUND",
        "status": "SUCCEEDED",
        "metric": "vina_affinity",
        "value": -8.0,
        "unit": "kcal/mol",
        "teacher_id": "structure.dock.vina",
        "teacher_checkpoint": "sha256:" + "0" * 64,
        "molecule_id": "CHEMBL0000001",
        "pose_id": "pose_tr_0001_0",
        "receptor_hash": "sha256:" + "1" * 64,
        "evaluator_hash": "sha256:" + "2" * 64,
        "experiment_id": "exp_example",
        "data_partition": "train",
        "family_id": "fam_0001",
        "smiles": "c1ccccc1",
        "measured_p_activity": 7.5,
        "uncertainty": None,
        "uncertainty_state": "UNAVAILABLE",
        "selection_rationale": "synthetic fixture",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Capability reality — measured here, never quoted
# ---------------------------------------------------------------------------
#: The path this step actually takes, and the reason it is not the first one the
#: work order proposes. Kept as a constant so the ablation record, the student
#: card and the test all name the same thing.
CHOSEN_PATH = "scorer_distillation_of_the_calibrated_docking_endpoint"


def training_capability_report(*, execute_sample: bool = False) -> Record:
    """Which student training paths this host can really reach.

    Reuses Step 8's :func:`drug_agent.selfevolve.target.baselines.probe_capability`
    rather than re-probing in a second dialect, and reports the ``fit`` verb for
    every generator the work order proposes before choosing a path. ``sample``
    is *inspected* by default because executing a generator costs real seconds
    and this report is asked for on every ablation run; pass
    ``execute_sample=True`` for the executing probe.
    """
    from .baselines import probe_capability

    def _adapter_status(loader: Any) -> Record:
        try:
            adapter = loader()
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "evidence_class": "unavailable",
                "evidence": f"constructing the adapter raised {type(exc).__name__}: {exc}",
                "missing_verb": None,
                "unblock": "read the reason above",
            }
        ok = bool(adapter.available())
        return {
            "status": "AVAILABLE" if ok else "UNAVAILABLE",
            "evidence_class": "inspected",
            "evidence": f"{type(adapter).__name__}.available() == {ok}",
            "missing_verb": None if ok else "sample",
            "unblock": None if ok else str(adapter.profile.unavailable_reason or "unspecified"),
        }

    def _fit(capability_id: str) -> Record:
        probe = probe_capability(capability_id, "fit")
        status = str(probe["status"])
        return {
            "status": "UNAVAILABLE" if status == "UNAVAILABLE" else status,
            "evidence_class": str(probe["evidence_class"]),
            "evidence": str(probe["evidence"]),
            "missing_verb": "fit" if status == "UNAVAILABLE" else None,
            "unblock": probe.get("unblock"),
            "reason": probe.get("reason"),
        }

    def _sample(capability_id: str, loader: Any) -> Record:
        if not execute_sample:
            return _adapter_status(loader)
        probe = probe_capability(capability_id, "sample")
        return {
            "status": "AVAILABLE" if str(probe["status"]).startswith("AVAILABLE") else "UNAVAILABLE",
            "evidence_class": str(probe["evidence_class"]),
            "evidence": str(probe["evidence"]),
            "missing_verb": None if str(probe["status"]).startswith("AVAILABLE") else "sample",
            "unblock": probe.get("unblock"),
        }

    def _reinvent() -> Any:
        from drug_agent.tools.generation.reinvent4_tool import Reinvent4Adapter

        return Reinvent4Adapter()

    def _pfm() -> Any:
        from drug_agent.tools.pfm_tool import PFMGeneratorAdapter

        return PFMGeneratorAdapter()

    backends = {
        "reinvent4": {
            "capability_id": "molecule.generate.reinvent4",
            "sample": _sample("molecule.generate.reinvent4", _reinvent),
            "fit": _fit("molecule.generate.reinvent4"),
        },
        "pfm": {
            "capability_id": "molecule.generate.pfm",
            "sample": _sample("molecule.generate.pfm", _pfm),
            "fit": _fit("molecule.generate.pfm"),
        },
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "backends": backends,
        "chosen_path": CHOSEN_PATH,
        "why_not_generator_adaptation": (
            "The work order proposes REINVENT4 adaptation on eligible selected molecules first. "
            "Neither generator adapter in this repository exposes a training verb, so target "
            "fine-tuning is not reachable through a supported entry point on this host. Rather "
            "than pass a synthetic-loss-weight field to an adapter that would ignore it, the "
            "student here distils the calibrated docking endpoint, which Appendix A \u00a73.1 "
            "registers as Track C and names as the expensive evaluation this target is bottlenecked "
            "on. The UNAVAILABLE answers above are measured on this host at run time."
        ),
        "student_learns": (
            "the named computational endpoint in the teacher table, never a measured affinity"
        ),
    }


# ---------------------------------------------------------------------------
# Pareto handling
# ---------------------------------------------------------------------------
ORIENTATIONS: tuple[str, ...] = ("LOWER_IS_BETTER", "HIGHER_IS_BETTER")


def pareto_front(
    items: Sequence[Mapping[str, Any]],
    *,
    objectives: Mapping[str, str],
    key: str = "molecule_id",
) -> Record:
    """The non-dominated set, with the incomparabilities left visible.

    Selecting distillation examples is a multi-objective problem — a favourable
    teacher score, an unusual scaffold, a property profile — and collapsing it
    into one number is a preference nobody declared. So this returns the front
    and *says* which members are incomparable, which are exactly tied, and
    whether a total order exists at all. ``total_order`` is ``None`` whenever any
    two items are incomparable; a caller that needs a deterministic sequence
    takes ``front`` (sorted by *key*, which is a tie-break for reproducibility
    and explicitly not a preference).

    Raises:
        ContractError: ``INVALID_ENUM`` for an unregistered orientation,
            ``MISSING_FIELD`` for an item lacking an objective or the key, and
            ``NON_FINITE_VALUE`` for an objective that is not a measurement.
    """
    if not objectives:
        _fail(ErrorCategory.MISSING_FIELD, "a Pareto front needs at least one objective", field="objectives")
    for name, orientation in objectives.items():
        if orientation not in ORIENTATIONS:
            _fail(
                ErrorCategory.INVALID_ENUM,
                f"orientation {orientation!r} for {name!r} is not one of {list(ORIENTATIONS)}",
                field=name,
            )
    names = sorted(objectives)
    ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    for item in items:
        if key not in item:
            _fail(ErrorCategory.MISSING_FIELD, f"an item has no {key!r} to identify it", field=key)
        ids.append(_text(item[key], field=key))
        row: list[float] = []
        for name in names:
            if name not in item:
                _fail(
                    ErrorCategory.MISSING_FIELD,
                    f"item {item[key]!r} has no objective {name!r}",
                    field=name,
                )
            value = _finite(item[name], field=name)
            row.append(value if objectives[name] == "HIGHER_IS_BETTER" else -value)
        vectors.append(tuple(row))
    if len(set(ids)) != len(ids):
        _fail(ErrorCategory.INVALID_VALUE, f"duplicate {key} values cannot address a front", field=key)

    def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        return all(x >= y for x, y in zip(a, b, strict=True)) and any(
            x > y for x, y in zip(a, b, strict=True)
        )

    n = len(ids)
    front = [ids[i] for i in range(n) if not any(dominates(vectors[j], vectors[i]) for j in range(n))]
    dominated = [i for i in ids if i not in set(front)]
    incomparable: list[list[str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vectors[i], vectors[j]
            if a != b and not dominates(a, b) and not dominates(b, a):
                incomparable.append(sorted((ids[i], ids[j])))
    groups: dict[tuple[float, ...], list[str]] = {}
    for identifier, vector in zip(ids, vectors, strict=True):
        groups.setdefault(vector, []).append(identifier)
    tied = [sorted(g) for g in groups.values() if len(g) > 1]
    total: list[str] | None = None
    if not incomparable:
        ranked = sorted(zip(vectors, ids, strict=True), key=lambda p: (tuple(-x for x in p[0]), p[1]))
        total = [i for _v, i in ranked]
    return {
        "contract_version": CONTRACT_VERSION,
        "objectives": {name: objectives[name] for name in names},
        "n_items": n,
        "front": sorted(front),
        "dominated": sorted(dominated),
        "incomparable_pairs": sorted(incomparable),
        "n_incomparable_pairs": len(incomparable),
        "tied_groups": sorted(tied),
        "total_order": total,
        "tie_break": "molecule id, for reproducibility only; it expresses no preference",
        "note": (
            "total_order is null whenever two items are incomparable. Forcing a total ranking "
            "over incomparable objective vectors would invent a preference nobody declared."
        ),
    }


# ---------------------------------------------------------------------------
# Diversity and collapse
# ---------------------------------------------------------------------------
def _scaffold_id(smiles: str, molecule_id: str) -> str:
    """Bemis-Murcko scaffold, with acyclic molecules kept apart.

    Step 7 froze the ``acyclic:<molecule_id>`` convention for the qualifying
    scaffold count; the same convention is used here so a diversity number and a
    qualification number cannot silently disagree about what a scaffold is.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"unparsed:{molecule_id}"
    core = MurckoScaffold.GetScaffoldForMol(mol)
    text = Chem.MolToSmiles(core) if core is not None else ""
    return text or f"acyclic:{molecule_id}"


def student_diversity(rows: Sequence[Mapping[str, Any]]) -> Record:
    """Family and scaffold concentration of a training set or a selection.

    Reported because "the student got better" and "the student was trained on
    forty analogues of one compound" are the same sentence often enough that the
    second has to be visible.
    """
    n = len(rows)
    if n == 0:
        return {
            "n_rows": 0,
            "n_families": 0,
            "max_family_share": None,
            "n_scaffolds": 0,
            "unique_scaffold_fraction": None,
            "n_unique_structures": 0,
            "mean_nn_tanimoto": None,
            "telemetry": "UNAVAILABLE",
        }
    families: dict[str, int] = {}
    scaffolds: dict[str, int] = {}
    smiles: list[str] = []
    for row in rows:
        families[str(row.get("family_id") or "")] = families.get(str(row.get("family_id") or ""), 0) + 1
        text = str(row.get("smiles") or "")
        smiles.append(text)
        sid = _scaffold_id(text, str(row.get("molecule_id") or ""))
        scaffolds[sid] = scaffolds.get(sid, 0) + 1
    nn = _mean_nearest_neighbour_tanimoto(smiles)
    return {
        "n_rows": n,
        "n_families": len(families),
        "max_family_share": max(families.values()) / n,
        "n_scaffolds": len(scaffolds),
        "unique_scaffold_fraction": len(scaffolds) / n,
        "n_unique_structures": len(set(smiles)),
        "mean_nn_tanimoto": nn,
        "telemetry": "MEASURED",
        "note": (
            "max_family_share is the largest single structural family's share of the rows; a "
            "value near 1.0 means the set is one family wearing many molecule ids."
        ),
    }


def _mean_nearest_neighbour_tanimoto(smiles: Sequence[str]) -> float | None:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            fingerprints.append(generator.GetFingerprint(mol))
    if len(fingerprints) < 2:
        return None
    totals = []
    for i, fp in enumerate(fingerprints):
        others = fingerprints[:i] + fingerprints[i + 1 :]
        totals.append(max(DataStructs.BulkTanimotoSimilarity(fp, others)))
    return float(sum(totals) / len(totals))


#: Below this ratio of the reference spread, a student has stopped discriminating.
#: Declared here, before any student was fitted, rather than chosen afterwards.
COLLAPSE_DISPERSION_RATIO = 0.25


def collapse_indicators(predictions: Sequence[float], reference: Sequence[float] | None = None) -> Record:
    """Is the student still telling molecules apart, or predicting the mean?

    A distilled student that has collapsed reports a near-constant value and can
    still post a respectable MAE when the teacher's own spread is small. The
    dispersion ratio against the teacher's spread is the honest check, and the
    threshold is a module constant declared before any fit.
    """
    values = [_finite(v, field="prediction") for v in predictions]
    if not values:
        return {
            "n": 0,
            "prediction_sd": None,
            "reference_sd": None,
            "dispersion_ratio": None,
            "n_distinct_predictions": 0,
            "collapsed": None,
            "telemetry": "UNAVAILABLE",
        }
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    ref_sd: float | None = None
    ratio: float | None = None
    if reference is not None and len(reference) > 0:
        ref_values = [_finite(v, field="reference") for v in reference]
        ref_mean = sum(ref_values) / len(ref_values)
        ref_sd = math.sqrt(sum((v - ref_mean) ** 2 for v in ref_values) / len(ref_values))
        ratio = (sd / ref_sd) if ref_sd > 0 else None
    distinct = len({round(v, 6) for v in values})
    collapsed: bool | None
    if distinct <= 1:
        collapsed = True
    elif ratio is None:
        collapsed = None
    else:
        collapsed = ratio < COLLAPSE_DISPERSION_RATIO
    return {
        "n": len(values),
        "prediction_sd": sd,
        "reference_sd": ref_sd,
        "dispersion_ratio": ratio,
        "n_distinct_predictions": distinct,
        "collapse_dispersion_ratio_threshold": COLLAPSE_DISPERSION_RATIO,
        "collapsed": collapsed,
        "telemetry": "MEASURED",
    }


# ---------------------------------------------------------------------------
# Training-set assembly
# ---------------------------------------------------------------------------
#: At most this many rows from one structural family, unless a caller says
#: otherwise. Two rather than one because a family with a stereochemical pair is
#: not the same thing as forty analogues of one hit.
DEFAULT_MAX_PER_FAMILY = 2


def build_student_training_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    synthetic_weight: float,
    exclusion_registry: Mapping[str, Any] | None = None,
    max_per_family: int = DEFAULT_MAX_PER_FAMILY,
    holdout_families: Sequence[str] = (),
    holdout_row_ids: Sequence[str] = (),
) -> Record:
    """Assemble one arm's training rows, with the synthetic weight applied.

    Order matters and is fixed: **validate, screen, hold out, drop unscored,
    deduplicate, then weight**. Screening before deduplication means a protected
    compound cannot survive by being the first member of its family; weighting
    last means the weight is applied to rows that were already going to be used,
    so the arms differ in exactly one factor.

    ``synthetic_weight`` is a per-row loss weight, handed to the learner as
    scikit-learn's ``sample_weight``. **At 0.0 the synthetic rows are removed
    rather than zero-weighted**, because "this row counted for nothing" and
    "this row was not in the training set" are different claims and only the
    second is the no-distillation control.

    The frozen real-data anchor — the rows whose ``structure_origin`` is
    ``REAL_MEASURED_COMPOUND`` — is required at every weight, so no arm can
    become pure synthetic training by accident.

    Raises:
        ContractError: every category :func:`validate_teacher_row` can raise,
            plus ``INVALID_VALUE`` for a negative or non-finite weight and
            ``MISSING_COMPONENT`` when the anchor would be empty.
    """
    weight = _finite(synthetic_weight, field="synthetic_weight")
    if weight < 0:
        _fail(
            ErrorCategory.INVALID_VALUE,
            f"synthetic_weight={weight!r}; a negative weight would train the student to "
            f"disagree with its teacher, which is not an ablation of distillation",
            field="synthetic_weight",
        )
    if int(max_per_family) < 1:
        _fail(
            ErrorCategory.INVALID_VALUE,
            f"max_per_family={max_per_family!r} would admit nothing",
            field="max_per_family",
        )

    validated = [validate_teacher_row(row) for row in rows]
    n_input = len(validated)

    boundary: Record = {
        "registry_declared": exclusion_registry is not None,
        "n_offered": n_input,
        "n_admitted": n_input,
        "n_quarantined": 0,
        "quarantined_molecule_ids": [],
        "note": (
            "no exclusion registry was supplied, so nothing was screened; a caller assembling a "
            "real training set passes the Step 6 protected registry"
        ),
    }
    if exclusion_registry is not None:
        from .splits import screen_records

        offered = [
            {
                "molecule_id": row["molecule_id"],
                "connectivity_key": row.get("connectivity_key"),
                "family_id": row["family_id"],
                "aliases": list(row.get("aliases") or []),
            }
            for row in validated
        ]
        screen = screen_records(offered, exclusion_registry)
        blocked = {int(q["record_index"]) for q in screen["quarantined"]}
        boundary = {
            "registry_declared": True,
            "n_offered": int(screen["n_screened"]),
            "n_admitted": int(screen["n_admitted"]),
            "n_quarantined": int(screen["n_quarantined"]),
            "quarantined_molecule_ids": sorted(validated[i]["molecule_id"] for i in blocked),
            "quarantine_reasons": [
                {
                    "molecule_id": validated[int(q["record_index"])]["molecule_id"],
                    "matched_on": q.get("matched_on"),
                    "category": q.get("category"),
                }
                for q in screen["quarantined"]
            ],
            "note": (
                "protected examples are removed here, before anything is fitted; the count is a "
                "measurement of the boundary rather than an assertion that it was respected"
            ),
        }
        validated = [row for i, row in enumerate(validated) if i not in blocked]

    held = {str(f) for f in holdout_families}
    held_rows = {str(r) for r in holdout_row_ids}
    before_holdout = len(validated)
    validated = [row for row in validated if row["family_id"] not in held and row["row_id"] not in held_rows]
    n_dropped_holdout = before_holdout - len(validated)

    scored = [row for row in validated if row["status"] == "SUCCEEDED"]
    n_unscored = len(validated) - len(scored)

    # Deterministic deduplication: sort by (family, row id) so the surviving
    # members of a family do not depend on the order the table arrived in.
    ordered = sorted(scored, key=lambda r: (str(r["family_id"]), str(r["row_id"])))
    per_family: dict[str, int] = {}
    seen_structures: set[str] = set()
    kept: list[Record] = []
    n_dropped_family = 0
    n_dropped_structure = 0
    for row in ordered:
        structure = str(row["smiles"])
        if structure in seen_structures:
            n_dropped_structure += 1
            continue
        family = str(row["family_id"])
        if per_family.get(family, 0) >= int(max_per_family):
            n_dropped_family += 1
            continue
        per_family[family] = per_family.get(family, 0) + 1
        seen_structures.add(structure)
        kept.append(row)

    anchor = [r for r in kept if r["structure_origin"] == "REAL_MEASURED_COMPOUND"]
    synthetic = [r for r in kept if r["structure_origin"] == "GENERATED_STRUCTURE"]
    control_arm = weight == 0.0
    if control_arm:
        synthetic = []

    if not anchor:
        _fail(
            ErrorCategory.MISSING_COMPONENT,
            "the training set has no real-data anchor. Appendix A \u00a76 requires a frozen "
            "real-data/prior anchor to be preserved when testing iterative synthetic training, "
            "and a student fitted only on generated structures has no such anchor",
            field="structure_origin",
        )

    training: list[Record] = []
    for row in anchor:
        training.append({**row, "role": "anchor", "sample_weight": 1.0})
    for row in synthetic:
        training.append({**row, "role": "synthetic", "sample_weight": weight})
    training.sort(key=lambda r: str(r["row_id"]))

    anchor_ids = sorted(r["molecule_id"] for r in anchor)
    return {
        "contract_version": CONTRACT_VERSION,
        "synthetic_weight": weight,
        "control_arm": control_arm,
        "mixture": "per-row scikit-learn sample_weight; anchor rows are always 1.0",
        "rows": training,
        "n_input": n_input,
        "anchor_rows": len(anchor),
        "synthetic_rows": len(synthetic),
        "n_unscored": n_unscored,
        "n_dropped_holdout": n_dropped_holdout,
        "boundary": boundary,
        "dedup": {
            "max_per_family": int(max_per_family),
            "n_dropped_duplicate_family": n_dropped_family,
            "n_dropped_duplicate_structure": n_dropped_structure,
            "n_families": len(per_family),
            "rule": "rows sorted by (family_id, row_id); the first max_per_family of each family survive",
        },
        "anchor_pool_hash": sha256_text("\n".join(anchor_ids)),
        "training_set_hash": canonical_hash(
            [[r["row_id"], r["sample_weight"], r["value"]] for r in training]
        ),
        "diversity": student_diversity(training),
        "label_semantics": (
            "the target is the teacher's own metric; measured_p_activity travels with each row "
            "and is never a training target"
        ),
    }


# ---------------------------------------------------------------------------
# The student
# ---------------------------------------------------------------------------
#: Learners this module will fit, and what each one does with a row weight.
#: ``consumes_sample_weight`` is not documentation: :func:`train_student`
#: refuses a learner that does not, because passing a weight to an estimator
#: that ignores it is the ignored-configuration-field failure in person.
LEARNERS: dict[str, Record] = {
    "weighted_gradient_boosting": {
        "import": ("sklearn.ensemble", "GradientBoostingRegressor"),
        "consumes_sample_weight": True,
        "note": (
            "sample_weight scales each row's contribution to the squared-error loss at every "
            "boosting stage, so a weight change moves the fitted function"
        ),
    },
    "weighted_random_forest": {
        "import": ("sklearn.ensemble", "RandomForestRegressor"),
        "consumes_sample_weight": True,
        "note": (
            "sample_weight weights the split criterion and the leaf means. A forest grown to "
            "single-sample leaves can make that inert, so this learner pins min_samples_leaf "
            "to at least 2; tests/research/test_distillation_lineage.py asserts weight "
            "sensitivity for every learner in this table rather than trusting the note"
        ),
        "floor": {"min_samples_leaf": 2},
    },
    "weighted_ridge": {
        "import": ("sklearn.linear_model", "Ridge"),
        "consumes_sample_weight": True,
        "note": "sample_weight enters the weighted least-squares normal equations directly",
    },
}

#: The action-level backend names the dispatcher advertises, and the learner each
#: one means. Declared here so ``actions.py`` and this module cannot disagree
#: about what ``backend='sklearn_gbr'`` fits.
BACKEND_LEARNERS: dict[str, str] = {
    "sklearn_gbr": "weighted_gradient_boosting",
    "sklearn_rf": "weighted_random_forest",
    "sklearn_ridge": "weighted_ridge",
}

DEFAULT_BASE_CONFIG: Record = {
    "learner": "weighted_gradient_boosting",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 2,
    "subsample": 1.0,
    "fingerprint_radius": 2,
    "fingerprint_bits": 2048,
    "seed": 42,
}


class StudentModel:
    """A fitted student and everything needed to call it again.

    Deliberately a small object rather than a bare estimator: the featuriser's
    radius and bit count are part of the artifact, and a student reloaded
    without them would silently score a different representation.
    """

    def __init__(self, estimator: Any, *, featuriser: Mapping[str, Any], card: Mapping[str, Any]) -> None:
        self.estimator = estimator
        self.featuriser = dict(featuriser)
        self.card = dict(card)

    def featurise(self, smiles: Sequence[str]) -> Any:
        return _featurise(smiles, radius=int(self.featuriser["radius"]), bits=int(self.featuriser["bits"]))

    def predict(self, smiles: Sequence[str]) -> list[float | None]:
        """Predict the teacher's metric. ``None`` where the SMILES will not parse.

        An unparseable structure is a missing outcome, never a low score, so it
        comes back as ``None`` and the caller's coverage drops.
        """
        features, usable = self.featurise(smiles)
        out: list[float | None] = [None] * len(smiles)
        if not usable:
            return out
        predicted = self.estimator.predict(features)
        for position, index in enumerate(usable):
            out[index] = float(predicted[position])
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StudentModel(metric={self.card.get('metric')!r}, learner={self.card.get('learner')!r})"


#: Classes that a student pickled by the ``python -m ...target.distill`` entry
#: point records under ``__main__`` rather than under this module.
#:
#: Running a module with ``-m`` imports it *as* ``__main__``, so every class
#: defined in it has ``__module__ == "__main__"`` and pickle writes that name
#: into the file. The 24 students the Step 12 sweep actually executed were saved
#: that way, and could therefore be reloaded only by a process that happened to
#: have ``StudentModel`` in its own ``__main__`` — which is not "the file is the
#: artifact". Step 12's own reload test passed because it trained the student
#: inside pytest, where this module is imported under its real name.
#:
#: Two repairs, and they are different repairs. The ``__main__`` block at the
#: bottom of this file now delegates to the canonically imported module, so
#: nothing saved from here on records ``__main__`` again. This table is the
#: other half: it lets the students already on disk be reloaded, without
#: re-training them and therefore without changing bytes whose digests are
#: recorded in ``ablation.json`` and ``repeated.json``.
_LEGACY_MAIN_CLASSES: dict[str, type] = {"StudentModel": StudentModel}


@contextlib.contextmanager
def _legacy_main_aliases() -> Iterator[None]:
    """Make ``__main__.<name>`` resolve to this module's class, then undo it.

    Pickle resolves a global as ``getattr(sys.modules[module], name)``, and
    ``sys.modules["__main__"]`` always exists, so binding the name there is
    enough and needs no joblib internals. An existing attribute of the same name
    is left alone and restored: a process that really does define its own
    ``StudentModel`` in ``__main__`` keeps it.
    """
    main = sys.modules.get("__main__")
    if main is None:  # pragma: no cover - there is always a __main__
        yield
        return
    restore: list[tuple[str, Any]] = []
    added: list[str] = []
    for name, cls in _LEGACY_MAIN_CLASSES.items():
        if hasattr(main, name):
            restore.append((name, getattr(main, name)))
        else:
            added.append(name)
        setattr(main, name, cls)
    try:
        yield
    finally:
        for name, value in restore:
            setattr(main, name, value)
        for name in added:
            with contextlib.suppress(AttributeError):
                delattr(main, name)


def _featurise(smiles: Sequence[str], *, radius: int, bits: int) -> tuple[Any, list[int]]:
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    vectors: list[Any] = []
    usable: list[int] = []
    for index, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            continue
        vectors.append(np.array(generator.GetFingerprint(mol), dtype=np.float32))
        usable.append(index)
    if not vectors:
        return np.zeros((0, bits), dtype=np.float32), []
    return np.vstack(vectors), usable


def train_student(
    training_set: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
    output_dir: Path | str,
    teacher_endpoint: str | None = None,
) -> Record:
    """Fit one student on one arm's rows and save it as a reloadable artifact.

    The saved file is the artifact: :func:`load_student` reads it without the
    training data, the teacher table or this module's campaign state.
    Reproducing a student by retraining it is a *different* artifact and is not
    what "reloadable" means.

    Every arm of an ablation passes the identical ``base_config``; its content
    address is recorded in the card, so "matched base" is checkable rather than
    asserted. A gradient-boosted forest has no incremental checkpoint to warm
    start from, so the matched base here is the matched configuration and the
    matched anchor pool, and the card says exactly that rather than implying a
    shared set of weights.

    Raises:
        ContractError: ``INVALID_ENUM`` for an unregistered learner,
            ``UNSUPPORTED_CLAIM`` for a learner that would ignore the weights,
            and ``MISSING_COMPONENT`` when no row survives featurisation.
    """
    import joblib
    import numpy as np

    config = dict(DEFAULT_BASE_CONFIG)
    config.update(dict(base_config or {}))
    name = str(config["learner"])
    if name not in LEARNERS:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"learner {name!r} is not one of {sorted(LEARNERS)}",
            field="learner",
        )
    if not LEARNERS[name]["consumes_sample_weight"]:
        _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            f"learner {name!r} does not consume sample_weight; handing it the synthetic weight "
            f"would advertise an ablation that the fit ignores",
            field="learner",
        )

    rows = list(training_set["rows"])
    if not rows:
        _fail(ErrorCategory.MISSING_COMPONENT, "no training row survived assembly", field="rows")
    metrics = {str(r["metric"]) for r in rows}
    if len(metrics) != 1:
        _fail(
            ErrorCategory.INVALID_VALUE,
            f"the training rows name {sorted(metrics)}; a student learns one endpoint, and "
            f"mixing metrics would make its output unnameable",
            field="metric",
        )
    metric = metrics.pop()
    endpoints = {str(r["teacher_id"]) for r in rows}
    endpoint = teacher_endpoint or "+".join(sorted(endpoints))

    features, usable = _featurise(
        [str(r["smiles"]) for r in rows],
        radius=int(config["fingerprint_radius"]),
        bits=int(config["fingerprint_bits"]),
    )
    if not usable:
        _fail(ErrorCategory.MISSING_COMPONENT, "no training SMILES could be featurised", field="smiles")
    targets = np.array([float(rows[i]["value"]) for i in usable], dtype=np.float64)
    weights = np.array([float(rows[i]["sample_weight"]) for i in usable], dtype=np.float64)

    floor = dict(LEARNERS[name].get("floor") or {})
    for field, minimum in floor.items():
        if int(config.get(field) or 0) < int(minimum):
            config[field] = int(minimum)
    module_name, class_name = LEARNERS[name]["import"]
    module = __import__(module_name, fromlist=[class_name])
    estimator_cls = getattr(module, class_name)
    if name == "weighted_gradient_boosting":
        estimator = estimator_cls(
            n_estimators=int(config["n_estimators"]),
            max_depth=None if config["max_depth"] is None else int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            subsample=float(config["subsample"]),
            random_state=int(config["seed"]),
        )
    elif name == "weighted_random_forest":
        estimator = estimator_cls(
            n_estimators=int(config["n_estimators"]),
            max_depth=None if config["max_depth"] is None else int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=int(config["seed"]),
            n_jobs=1,
        )
    else:
        estimator = estimator_cls(alpha=float(config.get("alpha", 1.0)), random_state=int(config["seed"]))

    started = time.time()
    estimator.fit(features, targets, sample_weight=weights)
    fit_seconds = time.time() - started

    base_config_hash = canonical_hash(dict(sorted(config.items())))
    card: Record = {
        "contract_version": CONTRACT_VERSION,
        "kind": "distilled_student",
        "learner": name,
        "learner_class": f"{module_name}.{class_name}",
        "metric": metric,
        "unit": str(rows[0].get("unit") or ""),
        "teacher_endpoint": endpoint,
        "teacher_checkpoints": sorted({str(r["teacher_checkpoint"]) for r in rows}),
        "receptor_hashes": sorted({str(r["receptor_hash"]) for r in rows if r["receptor_hash"]}),
        "evaluator_hashes": sorted({str(r["evaluator_hash"]) for r in rows}),
        "experiment_ids": sorted({str(r["experiment_id"]) for r in rows}),
        "data_partitions": sorted({str(r["data_partition"]) for r in rows}),
        "synthetic_weight": float(training_set["synthetic_weight"]),
        "control_arm": bool(training_set["control_arm"]),
        "n_train_rows": len(usable),
        "n_anchor_rows": int(training_set["anchor_rows"]),
        "n_synthetic_rows": int(training_set["synthetic_rows"]),
        "anchor_pool_hash": str(training_set["anchor_pool_hash"]),
        "training_set_hash": str(training_set["training_set_hash"]),
        "base_config": dict(sorted(config.items())),
        "base_config_hash": base_config_hash,
        "featuriser": {
            "kind": "rdkit_morgan_bits",
            "radius": int(config["fingerprint_radius"]),
            "bits": int(config["fingerprint_bits"]),
        },
        "matched_base": (
            "this learner has no incremental checkpoint to warm start from, so the matched base "
            "across arms is the identical base_config_hash and the identical anchor_pool_hash "
            "recorded above, and no shared set of weights is claimed"
        ),
        "predicts": "the named computational teacher endpoint, not a measured affinity",
        "measured_label_used": False,
        "measured_label_note": (
            "measured_p_activity travels with every teacher row and is never a target, a feature "
            "or a sample weight here. Agreement with this student is agreement with a docking "
            "computation; no claim about binding follows from it."
        ),
        "fit_seconds": round(fit_seconds, 4),
        "entrypoint": [
            "python",
            "-c",
            "from pathlib import Path; import sys; "
            "from drug_agent.selfevolve.target.distill import load_student; "
            "print(load_student(Path(sys.argv[1])).predict(sys.argv[2:]))",
        ],
    }
    student = StudentModel(estimator, featuriser=card["featuriser"], card=card)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student_path = out / "student.joblib"
    joblib.dump(student, student_path)
    card["student_sha256"] = sha256_bytes(student_path.read_bytes())
    card_path = out / "student_card.json"
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True, default=str), encoding="utf-8")
    rows_path = out / "training_rows.json"
    rows_path.write_text(
        json.dumps(
            {k: v for k, v in training_set.items() if k != "rows"} | {"rows": [dict(r) for r in rows]},
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "student_path": str(student_path),
        "card_path": str(card_path),
        "training_rows_path": str(rows_path),
        "student_sha256": card["student_sha256"],
        "card": card,
        "base_config_hash": base_config_hash,
        "n_train_rows": len(usable),
        "synthetic_weight": float(training_set["synthetic_weight"]),
        "fit_seconds": round(fit_seconds, 4),
    }


def load_student(path: Path | str) -> StudentModel:
    """Reload a saved student. The file is the artifact; nothing else is needed.

    That sentence is the contract, and making it true for the students already
    on disk needed :func:`_legacy_main_aliases`: see its note.
    """
    import joblib

    target = Path(path)
    if not target.is_file():
        _fail(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            f"{target} is not a file; a student that cannot be reloaded is not an artifact",
            field="student_path",
        )
    with _legacy_main_aliases():
        obj = joblib.load(target)
    if not isinstance(obj, StudentModel):  # pragma: no cover - corrupted artifact
        _fail(
            ErrorCategory.INVALID_TYPE,
            f"{target} holds a {type(obj).__name__}, not a StudentModel",
            field="student_path",
        )
    return obj


def evaluate_student(
    student: StudentModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    teacher_seconds: Mapping[str, float] | None = None,
) -> Record:
    """Score a student against fresh teacher rows it has never seen.

    The reported agreement is against **teacher scores**. It is a measurement of
    how well a cheap model imitates a named computation, and Appendix A \u00a76 is
    explicit that this is not a measured affinity. The descriptive correlation
    against measured potency is computed separately by
    :func:`descriptive_potency_correlation` and is labelled there.
    """
    scored = [validate_teacher_row(r) for r in rows]
    scored = [r for r in scored if r["status"] == "SUCCEEDED"]
    if not scored:
        return {
            "status": "INSUFFICIENT_DATA",
            "n_pool": 0,
            "reason": "no fresh teacher row carries a score",
        }
    started = time.time()
    predictions = student.predict([str(r["smiles"]) for r in scored])
    inference_seconds = time.time() - started
    pairs = [(float(r["value"]), float(p)) for r, p in zip(scored, predictions, strict=True) if p is not None]
    n_failed = len(scored) - len(pairs)
    if not pairs:
        return {
            "status": "FAILED",
            "n_pool": len(scored),
            "n_scored": 0,
            "coverage": 0.0,
            "reason": "the student could not featurise a single fresh molecule",
        }
    kept = [(r, p) for r, p in zip(scored, predictions, strict=True) if p is not None]
    truth = [t for t, _p in pairs]
    predicted = [p for _t, p in pairs]
    errors = [abs(t - p) for t, p in pairs]
    per_molecule = [
        {
            "molecule_id": str(r["molecule_id"]),
            "structure_origin": str(r["structure_origin"]),
            "family_id": str(r["family_id"]),
            "teacher": float(r["value"]),
            "predicted": float(p),
            "abs_error": abs(float(r["value"]) - float(p)),
        }
        for r, p in kept
    ]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum((t - p) ** 2 for t, p in pairs) / len(pairs))
    rho = _spearman(truth, predicted)
    teacher_total = None
    if teacher_seconds:
        observed = [
            float(teacher_seconds[r["molecule_id"]]) for r in scored if r["molecule_id"] in teacher_seconds
        ]
        teacher_total = sum(observed) if observed else None
    return {
        "status": "SUCCEEDED",
        "metric": f"student_mae_vs_teacher[{student.card.get('metric')}]",
        "n_pool": len(scored),
        "n_scored": len(pairs),
        "n_failed": n_failed,
        "coverage": len(pairs) / len(scored),
        "mae_vs_teacher": mae,
        "rmse_vs_teacher": rmse,
        "spearman_vs_teacher": rho,
        "teacher_sd": collapse_indicators(truth)["prediction_sd"],
        "collapse": collapse_indicators(predicted, truth),
        "per_molecule": per_molecule,
        "student_inference_seconds": round(inference_seconds, 4),
        "teacher_seconds_on_this_pool": teacher_total,
        "speedup_vs_teacher": (teacher_total / inference_seconds)
        if teacher_total and inference_seconds > 0
        else None,
        "claim_scope": (
            "agreement with a named computational endpoint. Better agreement is better imitation "
            "of docking and is not evidence about binding or potency."
        ),
    }


#: Bootstrap resamples for a paired effect interval. Declared here rather than
#: chosen after seeing a result.
BOOTSTRAP_RESAMPLES = 5000


def paired_effect(
    arm: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    alpha: float = 0.05,
) -> Record:
    """Paired difference in absolute error, with a percentile bootstrap interval.

    Paired **per molecule** on the shared holdout, because the two arms scored
    the identical pool and an unpaired comparison would throw that away.

    What this interval does and does not cover is stated in the record it
    returns: it resamples the holdout molecules, so it covers the holdout's
    sampling variability and **not** the variability of the fit itself. One fit
    per arm gives no estimate of the second, which is why
    :func:`repeat_ablation` exists and why a single-seed result is reported as
    ``INCONCLUSIVE`` however its interval falls.
    """
    import random

    a = {str(r["molecule_id"]): float(r["abs_error"]) for r in arm.get("per_molecule") or []}
    b = {str(r["molecule_id"]): float(r["abs_error"]) for r in control.get("per_molecule") or []}
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return {
            "status": "INSUFFICIENT_DATA",
            "n_paired": len(shared),
            "reason": "fewer than three molecules were scored by both arms",
        }
    deltas = [a[m] - b[m] for m in shared]
    mean = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    means: list[float] = []
    n = len(deltas)
    for _ in range(int(n_bootstrap)):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int((alpha / 2) * len(means))]
    high = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return {
        "status": "SUCCEEDED",
        "metric": "delta_abs_error_vs_teacher (arm - control)",
        "orientation": "LOWER_IS_BETTER, so a negative mean favours the distilled arm",
        "n_paired": n,
        "mean_delta": mean,
        "median_delta": sorted(deltas)[n // 2],
        "n_improved": sum(1 for d in deltas if d < 0),
        "n_worsened": sum(1 for d in deltas if d > 0),
        "interval": [low, high],
        "interval_kind": f"{int((1 - alpha) * 100)}% percentile bootstrap, {int(n_bootstrap)} resamples",
        "excludes_no_improvement": high < 0.0,
        "covers": (
            "the sampling variability of the holdout molecules only. It does NOT cover the "
            "variability of the fit: one fit per arm gives no estimate of that, and a result "
            "from a single seed is INCONCLUSIVE whatever this interval does"
        ),
    }


def descriptive_potency_correlation(student: StudentModel, rows: Sequence[Mapping[str, Any]]) -> Record:
    """Rank correlation between the student's output and measured pActivity.

    **Descriptive, and labelled as such.** The student predicts a docking score;
    pActivity is a different quantity in different units measured by a different
    process. Reporting the correlation is informative — it is the number that
    says whether imitating the teacher bought anything biological — and the
    honest answer is allowed to be "no".
    """
    usable = [
        r
        for r in (validate_teacher_row(row) for row in rows)
        if r["measured_p_activity"] is not None and r["smiles"]
    ]
    if len(usable) < 3:
        return {
            "status": "INSUFFICIENT_DATA",
            "n": len(usable),
            "reason": "fewer than three rows carry a measured pActivity",
            "claim_scope": "descriptive only",
        }
    predictions = student.predict([str(r["smiles"]) for r in usable])
    pairs = [
        (float(r["measured_p_activity"]), float(p))
        for r, p in zip(usable, predictions, strict=True)
        if p is not None
    ]
    if len(pairs) < 3:
        return {"status": "INSUFFICIENT_DATA", "n": len(pairs), "claim_scope": "descriptive only"}
    measured = [m for m, _p in pairs]
    predicted = [p for _m, p in pairs]
    return {
        "status": "SUCCEEDED",
        "n": len(pairs),
        "spearman_student_vs_measured_pactivity": _spearman(measured, predicted),
        "orientation_note": (
            "the student predicts vina_affinity, which is LOWER_IS_BETTER, against pActivity, "
            "which is HIGHER_IS_BETTER; a negative rho is the direction that would indicate "
            "agreement in sign"
        ),
        "claim_scope": (
            "DESCRIPTIVE. This is a correlation between a docking surrogate and a measured "
            "potency on the development partition. It is not a Track A benchmark result, it was "
            "not used to choose an arm, and no claim of stronger biology follows from it."
        ),
    }


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(list(a), list(b)).statistic)
    return None if math.isnan(value) else value


# ---------------------------------------------------------------------------
# The teacher pass — a first-class artifact of its own
# ---------------------------------------------------------------------------
# Step 11's campaign C named a real teacher endpoint and failed because the
# teacher rows did not exist. The architectural answer taken here is the first
# of the two the dispatcher named: **a separate scoring pass whose output is a
# first-class artifact**. ``build_teacher_table`` docks; ``run_ablation`` learns;
# neither does the other's job. The alternative — letting ``distill`` score its
# own molecules — was rejected because the resulting student could not say which
# computation produced its labels, which is the whole thing this step is for.
def _repo_root() -> Path:
    default = Path(__file__).resolve().parents[4]
    override = os.environ.get("DRUGAGENT_REPO_ROOT", "").strip()
    if override:
        candidate = Path(override).resolve()
        if candidate != default and (candidate / ".external" / "research" / "zak-c9r").is_dir():
            return candidate
    return default


def _snapshot(repo_root: Path | None = None) -> Path:
    return (repo_root or _repo_root()) / ".external" / "research" / "zak-c9r"


def _json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run the Step 5/6/7 build scripts first")
    return json.loads(path.read_text(encoding="utf-8"))


def teacher_identity(repo_root: Path | None = None) -> Record:
    """Who the teacher is, as a content address rather than a name.

    A physics-based scorer has no learned checkpoint, so its ``checkpoint`` is
    the pinned configuration that makes its numbers reproducible: engine, seed,
    exhaustiveness, pose count, receptor digest and search box. Calling that a
    checkpoint is not a metaphor — change any of it and the teacher is a
    different teacher, which is exactly what a checkpoint id has to detect.
    """
    root = repo_root or _repo_root()
    calibration = _json(root / "configs" / "research" / "zak-c9r" / "calibration.json")
    scorer = dict(calibration["scorer"])
    identity = {
        "engine": scorer["engine"],
        "metric": scorer["metric"],
        "engine_versions": list(scorer["engine_versions"]),
        "adapter": scorer["adapter"],
        "seed": int(scorer["seed"]),
        "exhaustiveness": int(scorer["exhaustiveness"]),
        "n_poses": int(scorer["n_poses"]),
        "receptor_sha256": scorer["receptor_sha256"],
        "box_center": [float(v) for v in scorer["box_center"]],
        "box_size": [float(v) for v in scorer["box_size"]],
    }
    return {
        "teacher_id": f"structure.dock.{scorer['engine']}",
        "teacher_checkpoint": canonical_hash(identity),
        "receptor_hash": str(scorer["receptor_sha256"]),
        "metric": str(scorer["metric"]),
        "unit": "kcal/mol",
        "identity": identity,
        "checkpoint_note": (
            "an empirical scoring function has no learned weights; this content address is over "
            "the pinned engine, seed, exhaustiveness, pose count, receptor digest and box, which "
            "is what has to change for the teacher to be a different teacher"
        ),
    }


def select_teacher_molecules(
    *,
    n_anchor: int,
    n_synthetic: int,
    seed: int = 42,
    repo_root: Path | None = None,
    generator_backend: str = "reinvent4",
    generator_oversample: int = 3,
) -> Record:
    """Choose what the teacher pass will score, before it scores anything.

    Anchor molecules are real train-partition compounds taken through Step 6's
    exclusion boundary, **one per structural family**, so the teacher table
    cannot be dominated by one series before a student ever sees it. Synthetic
    molecules are sampled from the generator and screened against the permitted
    training library, so a "generated" row cannot be a training compound in
    disguise.
    """
    root = repo_root or _repo_root()
    os.environ.setdefault("DRUGAGENT_REPO_ROOT", str(root))
    from .baseline_arms import prepare_track_a

    manifest = _json(_snapshot(root) / "derived" / "split_manifest.json")
    prepared = prepare_track_a({"stratum_id": manifest["stratum_id"]})
    family_of = dict(prepared["family_of"])

    import random

    rng = random.Random(seed)
    by_family: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for key, row in sorted(prepared["library"].items()):
        family = str(family_of.get(key) or f"singleton:{key}")
        by_family.setdefault(family, []).append((key, row))
    families = sorted(by_family)
    rng.shuffle(families)

    measured: dict[str, float] = {}
    for key, row in prepared["train"].items():
        try:
            measured[str(key)] = float(row["p_value"])
        except (KeyError, TypeError, ValueError):
            continue

    anchor: list[Record] = []
    for family in families:
        if len(anchor) >= n_anchor:
            break
        # Within a family, prefer the member that also carries an exact primary
        # -stratum measurement. The teacher score is the training target either
        # way; the measured value rides along in its own field and makes the
        # descriptive potency correlation computable on more than a handful of
        # rows. The family is still the unit of selection, so this changes which
        # member represents a family and never how many families there are.
        key, row = sorted(by_family[family], key=lambda pair: (pair[0] not in measured, pair[0]))[0]
        smiles = str(row["molecule"]["standard_smiles"])
        anchor.append(
            {
                "molecule_id": str(row["molecule_id"]),
                "inchikey": key,
                "connectivity_key": row.get("connectivity_key"),
                "smiles": smiles,
                "family_id": family,
                "structure_origin": "REAL_MEASURED_COMPOUND",
                "data_partition": "train",
                "measured_p_activity": measured.get(key),
                "selection_rationale": (
                    "one compound per admitted train-partition structural family, family order "
                    f"shuffled at seed {seed}, preferring the family member that also carries an "
                    "exact primary-stratum measurement; the family cap keeps a single series from "
                    "dominating the teacher table"
                ),
            }
        )

    generation: Record = {"requested": n_synthetic, "backend": generator_backend}
    synthetic: list[Record] = []
    if n_synthetic > 0:
        known = {str(r["molecule"]["standard_smiles"]) for r in prepared["library"].values()}
        known |= {str(r["molecule"]["standard_smiles"]) for r in prepared["evaluation"].values()}
        if generator_backend == "reinvent4":
            from drug_agent.tools.generation.reinvent4_tool import Reinvent4Adapter

            adapter: Any = Reinvent4Adapter()
        else:
            from drug_agent.tools.pfm_tool import PFMGeneratorAdapter

            adapter = PFMGeneratorAdapter()
        if not adapter.available():
            generation["status"] = "UNAVAILABLE"
            generation["reason"] = str(adapter.profile.unavailable_reason or "unspecified")
        else:
            want = max(n_synthetic * max(1, int(generator_oversample)), n_synthetic)
            started = time.time()
            batch = adapter.generate(n_molecules=want, seed=seed, batch_size=min(want, 256))
            generation.update(
                {
                    "status": "EXECUTED",
                    "n_raw": int(batch.n_raw),
                    "model_id": batch.model_id,
                    "model_version": batch.model_version,
                    "seconds": round(time.time() - started, 3),
                    "errors": list(batch.errors)[:5],
                }
            )
            seen: set[str] = set()
            for molecule in batch.molecules:
                smiles = str(getattr(molecule, "smiles", "") or "")
                if not smiles or smiles in seen or smiles in known:
                    continue
                seen.add(smiles)
                identifier = f"gen_{len(synthetic):04d}"
                synthetic.append(
                    {
                        "molecule_id": identifier,
                        "inchikey": None,
                        "connectivity_key": None,
                        "smiles": smiles,
                        "family_id": "gen:" + _scaffold_id(smiles, identifier),
                        "structure_origin": "GENERATED_STRUCTURE",
                        "data_partition": "train",
                        "measured_p_activity": None,
                        "selection_rationale": (
                            f"sampled from {generator_backend} at seed {seed}; deduplicated and "
                            "screened against the permitted training library and the development "
                            "pool, so a generated row cannot be a known compound in disguise"
                        ),
                    }
                )
                if len(synthetic) >= n_synthetic:
                    break
            generation["n_unique_novel"] = len(synthetic)

    return {
        "contract_version": CONTRACT_VERSION,
        "seed": seed,
        "anchor": anchor,
        "synthetic": synthetic,
        "n_anchor": len(anchor),
        "n_synthetic": len(synthetic),
        "generation": generation,
        "boundary": dict(prepared["boundary"]),
        "split_hash": str(prepared["split_hash"]),
        "data_hash": str(prepared["data_hash"]),
        "stratum_id": str(prepared["stratum_id"]),
    }


def build_teacher_table(
    molecules: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    output_dir: Path | str,
    repo_root: Path | None = None,
    ledger: Any = None,
    dock_call_cap: int | None = None,
    resume: bool = True,
) -> Record:
    """Score *molecules* with the named teacher endpoint and write the table.

    This is the only place in this module that runs the expensive endpoint. It
    is resumable by construction: every scored molecule is appended to
    ``teacher_rows.jsonl`` as soon as it is scored, and a re-run with
    ``resume=True`` reads that file and skips what is already there, so a killed
    pass costs the molecules it had not yet reached and nothing else. The
    reservation ``request_key`` carries the same idempotence into the ledger, so
    a resumed molecule is not charged twice.

    Teacher generation is charged to the ledger's ``TEACHER_GENERATION`` phase,
    which Appendix A §4 requires and which no earlier step had ever spent.
    """
    root = repo_root or _repo_root()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    teacher = teacher_identity(root)
    from ..evaluation.target_benchmark import evaluator_hash
    from .baseline_arms import _dock_portfolio  # the same scorer every Track B arm uses

    evaluator = evaluator_hash()
    rows_path = out / "teacher_rows.jsonl"
    poses_path = out / "teacher_poses.json"

    done: dict[str, Record] = {}
    if resume and rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[str(record["molecule_id"])] = record
    poses: dict[str, str] = {}
    if resume and poses_path.is_file():
        poses = dict(json.loads(poses_path.read_text(encoding="utf-8")))

    pending = [m for m in molecules if str(m["molecule_id"]) not in done]
    if ledger is not None and pending and dock_call_cap is not None:
        ledger.declare("teacher_dock_calls", float(dock_call_cap))

    started_all = time.time()
    workdir = out / "docking"
    workdir.mkdir(parents=True, exist_ok=True)
    seconds_by_molecule: dict[str, float] = {str(k): float(v.get("seconds") or 0.0) for k, v in done.items()}
    n_failed = 0
    for molecule in pending:
        molecule_id = str(molecule["molecule_id"])
        reservation = None
        if ledger is not None:
            reservation = ledger.reserve(
                "teacher_dock_calls",
                1.0,
                owner=experiment_id,
                request_key=f"{experiment_id}:dock:{molecule_id}",
                phase="TEACHER_GENERATION",
                note="one teacher scoring call",
            )
            if reservation.get("already_completed"):
                continue
        started = time.time()
        scored = _dock_portfolio(
            [(molecule_id, str(molecule["smiles"]))],
            scorer=teacher["identity"],
            workdir=workdir,
        )[0]
        seconds = time.time() - started
        endpoint = dict(scored["structure_score"])
        status = str(endpoint["status"])
        pose_molblock = str(scored.get("pose_molblock") or "")
        pose_id = sha256_text(pose_molblock) if (status == "SUCCEEDED" and pose_molblock) else None
        if pose_id:
            poses[pose_id] = pose_molblock
        if status != "SUCCEEDED":
            n_failed += 1
        row: Record = {
            "row_id": f"tr_{experiment_id}_{molecule_id}",
            "source_kind": "teacher_prediction",
            "evidence_type": "CALCULATED",
            "curation_status": "AUTOMATIC",
            "structure_origin": str(molecule["structure_origin"]),
            "status": status,
            "metric": teacher["metric"],
            "value": endpoint["value"],
            "unit": teacher["unit"] if status == "SUCCEEDED" else None,
            "teacher_id": teacher["teacher_id"],
            "teacher_checkpoint": teacher["teacher_checkpoint"],
            "molecule_id": molecule_id,
            "pose_id": pose_id,
            "receptor_hash": teacher["receptor_hash"],
            "evaluator_hash": evaluator,
            "experiment_id": experiment_id,
            "data_partition": str(molecule["data_partition"]),
            "family_id": str(molecule["family_id"]),
            "smiles": str(molecule["smiles"]),
            "measured_p_activity": molecule.get("measured_p_activity"),
            "uncertainty": None,
            "uncertainty_state": "UNAVAILABLE",
            "selection_rationale": str(molecule["selection_rationale"]),
            "connectivity_key": molecule.get("connectivity_key"),
            "engine_version": scored.get("engine_version"),
            "n_poses": teacher["identity"]["n_poses"],
            "error_code": scored.get("error"),
            "seconds": round(seconds, 3),
        }
        validate_teacher_row(row)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        poses_path.write_text(json.dumps(poses, indent=0, sort_keys=True), encoding="utf-8")
        done[molecule_id] = row
        seconds_by_molecule[molecule_id] = round(seconds, 3)
        if ledger is not None and reservation is not None:
            ledger.settle(
                reservation["reservation_id"],
                consumed=1.0,
                status="SETTLED" if status == "SUCCEEDED" else "FAILED",
                note=f"{molecule_id}: {status} in {seconds:.1f}s",
                extra_costs={"teacher_wall_seconds": round(seconds, 3)},
            )

    ordered = [done[str(m["molecule_id"])] for m in molecules if str(m["molecule_id"]) in done]
    for row in ordered:
        validate_teacher_row(row)
    table: Record = {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "teacher_table",
        "experiment_id": experiment_id,
        "teacher": teacher,
        "evaluator_hash": evaluator,
        "n_rows": len(ordered),
        "n_succeeded": sum(1 for r in ordered if r["status"] == "SUCCEEDED"),
        "n_failed": sum(1 for r in ordered if r["status"] != "SUCCEEDED"),
        "n_anchor": sum(1 for r in ordered if r["structure_origin"] == "REAL_MEASURED_COMPOUND"),
        "n_synthetic": sum(1 for r in ordered if r["structure_origin"] == "GENERATED_STRUCTURE"),
        "teacher_wall_seconds": round(sum(seconds_by_molecule.values()), 3),
        "seconds_by_molecule": seconds_by_molecule,
        "rows": ordered,
        "poses_path": str(poses_path),
        "note": (
            "produced by a scoring pass that is a separate, first-class artifact. "
            "run_ablation consumes this file and scores nothing itself, so a student's labels "
            "can always be traced to the computation that made them."
        ),
    }
    table_path = out / "teacher_table.json"
    payload = json.dumps(table, indent=2, sort_keys=True, default=str).encode("utf-8")
    table_path.write_bytes(payload)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "teacher_table_manifest",
        "experiment_id": experiment_id,
        "teacher_id": teacher["teacher_id"],
        "teacher_checkpoint": teacher["teacher_checkpoint"],
        "receptor_hash": teacher["receptor_hash"],
        "evaluator_hash": evaluator,
        "metric": teacher["metric"],
        "table_path": str(table_path),
        "table_sha256": sha256_bytes(payload),
        "rows_jsonl_sha256": sha256_bytes(rows_path.read_bytes()) if rows_path.is_file() else None,
        "poses_sha256": sha256_bytes(poses_path.read_bytes()) if poses_path.is_file() else None,
        "n_rows": table["n_rows"],
        "n_succeeded": table["n_succeeded"],
        "n_failed": table["n_failed"],
        "teacher_wall_seconds": table["teacher_wall_seconds"],
        # Wall clock of *this invocation*, which is near zero on a resumed pass.
        # It lives here and not in the table because the table is a content-
        # addressed artifact: re-running a completed pass must reproduce the same
        # bytes, and a table carrying its own elapsed time never could.
        "pass_wall_seconds": round(time.time() - started_all, 3),
        "n_scored_this_invocation": len(pending),
    }
    (out / "teacher_table_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    table["manifest"] = manifest
    return table


def load_teacher_table(path: Path | str) -> Record:
    """Read a teacher table and validate every row in it before use."""
    record = _json(Path(path))
    rows = [validate_teacher_row(row) for row in record["rows"]]
    return {**record, "rows": rows}


# ---------------------------------------------------------------------------
# The ablation
# ---------------------------------------------------------------------------
#: Appendix A §6 / the Step 12 work order. The 0.0 arm is the control and is run,
#: not assumed.
DEFAULT_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 1.0)


def choose_holdout_families(
    rows: Sequence[Mapping[str, Any]], *, fraction: float = 0.25, seed: int = 42
) -> list[str]:
    """Whole families held out of every arm, chosen before any arm is fitted.

    Whole families, because a fresh evaluation that shares a structural family
    with the training set measures memorisation. Chosen from the row identifiers
    alone with a declared seed, so the holdout cannot drift once a number is
    known.
    """
    import random

    families = sorted({str(r["family_id"]) for r in rows if r["status"] == "SUCCEEDED"})
    if not families:
        return []
    rng = random.Random(seed)
    shuffled = list(families)
    rng.shuffle(shuffled)
    take = max(1, round(len(shuffled) * float(fraction)))
    return sorted(shuffled[:take])


def run_ablation(
    teacher_table: Mapping[str, Any],
    *,
    output_dir: Path | str,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    base_config: Mapping[str, Any] | None = None,
    exclusion_registry: Mapping[str, Any] | None = None,
    max_per_family: int = DEFAULT_MAX_PER_FAMILY,
    holdout_fraction: float = 0.25,
    seed: int = 42,
    experiment_id: str = "",
    ledger: Any = None,
    development_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Record:
    """Train one student per synthetic weight on a matched setup, and compare.

    Everything except the weight is held identical: the same teacher table, the
    same exclusion registry, the same family cap, the same held-out families,
    the same base configuration and the same anchor pool. The comparison this
    supports is therefore "what did the synthetic rows do", and nothing wider.

    The ``0.0`` arm is the no-distillation control. It really trains, on the
    anchor alone, with the identical recipe.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [validate_teacher_row(r) for r in teacher_table["rows"]]
    config = dict(DEFAULT_BASE_CONFIG)
    config.update(dict(base_config or {}))
    holdout = choose_holdout_families(rows, fraction=holdout_fraction, seed=seed)
    holdout_rows = [r for r in rows if str(r["family_id"]) in set(holdout) and r["status"] == "SUCCEEDED"]
    seconds_by_molecule = dict(teacher_table.get("seconds_by_molecule") or {})

    arms: list[Record] = []
    for weight in weights:
        arm_dir = out / f"weight-{weight:g}".replace(".", "p")
        reservation = None
        if ledger is not None:
            reservation = ledger.reserve(
                "student_fits",
                1.0,
                owner=experiment_id or "distill",
                request_key=f"{experiment_id}:fit:seed{seed}:{weight:g}",
                phase="ADAPTATION",
                note=f"student fit at synthetic weight {weight:g}",
            )
            if reservation.get("already_completed"):
                reservation = None
        training_set = build_student_training_set(
            rows,
            synthetic_weight=float(weight),
            exclusion_registry=exclusion_registry,
            max_per_family=max_per_family,
            holdout_families=holdout,
        )
        trained = train_student(
            training_set,
            base_config=config,
            output_dir=arm_dir,
            teacher_endpoint=str(teacher_table["teacher"]["teacher_id"]),
        )
        student = load_student(Path(trained["student_path"]))
        evaluation = evaluate_student(student, holdout_rows, teacher_seconds=seconds_by_molecule)
        # Reported per origin as well as overall, because "the synthetic rows
        # helped" and "the synthetic rows helped on other synthetic molecules"
        # are different findings and only the first is interesting.
        by_origin = {
            origin: evaluate_student(
                student,
                [r for r in holdout_rows if r["structure_origin"] == origin],
                teacher_seconds=seconds_by_molecule,
            )
            for origin in STRUCTURE_ORIGINS
        }
        descriptive = descriptive_potency_correlation(
            student, development_rows if development_rows is not None else holdout_rows
        )
        if ledger is not None and reservation is not None:
            ledger.settle(
                reservation["reservation_id"],
                consumed=1.0,
                note=f"weight {weight:g}: {trained['fit_seconds']:.2f}s",
                extra_costs={"fit_wall_seconds": float(trained["fit_seconds"])},
            )
        arms.append(
            {
                "synthetic_weight": float(weight),
                "control_arm": bool(training_set["control_arm"]),
                "student_path": trained["student_path"],
                "student_sha256": trained["student_sha256"],
                "card_path": trained["card_path"],
                "base_config_hash": trained["base_config_hash"],
                "anchor_pool_hash": training_set["anchor_pool_hash"],
                "training_set_hash": training_set["training_set_hash"],
                "n_train_rows": trained["n_train_rows"],
                "n_anchor_rows": training_set["anchor_rows"],
                "n_synthetic_rows": training_set["synthetic_rows"],
                "boundary": training_set["boundary"],
                "dedup": training_set["dedup"],
                "diversity": training_set["diversity"],
                "fit_seconds": trained["fit_seconds"],
                "evaluation": evaluation,
                "evaluation_by_structure_origin": by_origin,
                "descriptive_potency_correlation": descriptive,
            }
        )

    control = next((a for a in arms if a["control_arm"]), None)
    comparison: list[Record] = []
    for arm in arms:
        if control is None or arm is control:
            continue
        a_mae = arm["evaluation"].get("mae_vs_teacher")
        c_mae = control["evaluation"].get("mae_vs_teacher")
        delta = None if (a_mae is None or c_mae is None) else float(a_mae) - float(c_mae)
        effect = paired_effect(arm["evaluation"], control["evaluation"], seed=seed)
        comparison.append(
            {
                "synthetic_weight": arm["synthetic_weight"],
                "paired_effect": effect,
                "mae_vs_teacher": a_mae,
                "control_mae_vs_teacher": c_mae,
                "delta_mae_vs_control": delta,
                "relative_change": (delta / c_mae) if (delta is not None and c_mae) else None,
                "direction": "LOWER_IS_BETTER",
                "n_holdout": arm["evaluation"].get("n_scored"),
                "by_structure_origin": {
                    origin: {
                        "mae_vs_teacher": arm["evaluation_by_structure_origin"][origin].get("mae_vs_teacher"),
                        "control_mae_vs_teacher": control["evaluation_by_structure_origin"][origin].get(
                            "mae_vs_teacher"
                        ),
                        "n_scored": arm["evaluation_by_structure_origin"][origin].get("n_scored"),
                    }
                    for origin in STRUCTURE_ORIGINS
                },
                "paired_interval": effect.get("interval"),
                "interval_note": (
                    "the interval is a paired percentile bootstrap over the holdout molecules. "
                    "It covers holdout sampling and NOT fit variability, so one seed is "
                    "INCONCLUSIVE by Appendix A §5 whatever its sign; repeat_ablation adds the "
                    "across-seed spread."
                ),
            }
        )

    from ..evaluation.target_benchmark import evaluator_hash as _live_evaluator_hash

    live_evaluator = _live_evaluator_hash()
    table_evaluator = str(teacher_table.get("evaluator_hash") or "")
    live_teacher = teacher_identity()
    evaluator_drift = {
        "evaluator_hash_at_teacher_pass": table_evaluator,
        "evaluator_hash_now": live_evaluator,
        "moved": table_evaluator != live_evaluator,
        "teacher_checkpoint_at_teacher_pass": str(teacher_table["teacher"]["teacher_checkpoint"]),
        "teacher_checkpoint_now": live_teacher["teacher_checkpoint"],
        "teacher_moved": str(teacher_table["teacher"]["teacher_checkpoint"])
        != live_teacher["teacher_checkpoint"],
        "note": (
            "The evaluator address covers every file the judgement depends on, and it moves when "
            "a step adds a function to the guarded acceptance surface. The teacher checkpoint "
            "covers the engine, seed, exhaustiveness, pose count, receptor digest and box -- the "
            "things a docking score actually depends on. If the evaluator moved while the "
            "teacher checkpoint did not, the teacher scores in this table are still the scores "
            "that configuration produces, and nothing here was re-stamped to hide the difference."
        ),
    }
    matched = {
        "base_config_hash": sorted({a["base_config_hash"] for a in arms}),
        "anchor_pool_hash": sorted({a["anchor_pool_hash"] for a in arms}),
        "holdout_families": holdout,
        "n_holdout_rows": len(holdout_rows),
        "matched": len({a["base_config_hash"] for a in arms}) == 1
        and len({a["anchor_pool_hash"] for a in arms}) == 1,
    }
    record: Record = {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "distillation_ablation",
        "experiment_id": experiment_id,
        "chosen_path": CHOSEN_PATH,
        "teacher": dict(teacher_table["teacher"]),
        "evaluator_hash": str(teacher_table.get("evaluator_hash") or ""),
        "evaluator_drift": evaluator_drift,
        "weights": [float(w) for w in weights],
        "seed": seed,
        "base_config": dict(sorted(config.items())),
        "matched_setup": matched,
        "arms": arms,
        "comparison_against_no_distillation": comparison,
        "teacher_cost": {
            "teacher_wall_seconds": teacher_table.get("teacher_wall_seconds"),
            "n_teacher_calls": teacher_table.get("n_rows"),
            "n_teacher_failures": teacher_table.get("n_failed"),
            "charged_to_phase": "TEACHER_GENERATION",
            "note": (
                "teacher generation is charged to the adaptation budget. A failed docking call "
                "is a missing outcome with a recorded cost, never a low score."
            ),
        },
        "claim_scope": (
            "Track C. The students imitate a named computational endpoint. Any difference "
            "between arms is a difference in how well a cheap model reproduces docking, and no "
            "claim about binding, potency or biology follows from it."
        ),
    }
    path = out / "ablation.json"
    payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode("utf-8")
    path.write_bytes(payload)
    record["ablation_path"] = str(path)
    record["ablation_sha256"] = sha256_bytes(payload)
    return record


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 42, n_bootstrap: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05
) -> Record:
    """Percentile bootstrap of the mean, over whatever unit *values* are.

    Used here over **seeds**, which is the replication unit the within-holdout
    bootstrap cannot see. Seven points is few; the record says so rather than
    letting a narrow-looking interval speak for itself.
    """
    import random

    data = [float(v) for v in values]
    rng = random.Random(seed)
    n = len(data)
    means = sorted(sum(data[rng.randrange(n)] for _ in range(n)) / n for _ in range(int(n_bootstrap)))
    return {
        "mean": sum(data) / n,
        "interval": [
            means[int((alpha / 2) * len(means))],
            means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))],
        ],
        "interval_kind": f"{int((1 - alpha) * 100)}% percentile bootstrap over {n} seeds, "
        f"{int(n_bootstrap)} resamples",
        "n": n,
        "instability_note": (
            "the replication unit is the seed: each one re-chooses the held-out families and "
            "re-fits every arm. With this few seeds the interval is an instability estimate, not "
            "a precise one, and Appendix A §5 asks for it to be reported as such."
        ),
    }


def repeat_ablation(
    teacher_table: Mapping[str, Any],
    *,
    output_dir: Path | str,
    seeds: Sequence[int],
    **options: Any,
) -> Record:
    """Run the whole matched ablation once per seed and report the spread.

    One fit on one holdout has no estimate of its own variability, and a paired
    bootstrap over that single holdout cannot supply one. Each seed here
    re-chooses the held-out families **and** re-fits every arm, so the spread
    across seeds is an estimate of the thing the within-holdout interval is
    blind to. Every arm inside one seed still sees the identical holdout, so the
    comparison stays paired.

    With a handful of seeds this is an estimate of instability, not a precise
    interval, and the returned record says so rather than quoting a tight number
    from five points.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[Record] = []
    for seed in seeds:
        record = run_ablation(
            teacher_table,
            output_dir=out / f"seed-{int(seed)}",
            seed=int(seed),
            **options,
        )
        runs.append(record)

    by_weight: dict[float, list[Record]] = {}
    for record in runs:
        for row in record["comparison_against_no_distillation"]:
            by_weight.setdefault(float(row["synthetic_weight"]), []).append(
                {
                    "seed": record["seed"],
                    "delta_mae_vs_control": row["delta_mae_vs_control"],
                    "relative_change": row["relative_change"],
                    "n_holdout": row["n_holdout"],
                    "paired_interval": row["paired_interval"],
                    "paired_mean_delta": (row["paired_effect"] or {}).get("mean_delta"),
                }
            )

    summary: list[Record] = []
    for weight in sorted(by_weight):
        rows = by_weight[weight]
        deltas = [float(r["delta_mae_vs_control"]) for r in rows if r["delta_mae_vs_control"] is not None]
        if not deltas:
            summary.append({"synthetic_weight": weight, "status": "INSUFFICIENT_DATA", "n_seeds": 0})
            continue
        mean = sum(deltas) / len(deltas)
        interval = _bootstrap_mean_interval(deltas, seed=int(min(seeds)))
        summary.append(
            {
                "synthetic_weight": weight,
                "status": "SUCCEEDED",
                "n_seeds": len(deltas),
                "per_seed_delta_mae": deltas,
                "mean_delta_mae": mean,
                "across_seed_interval": interval,
                "excludes_no_improvement": bool(interval["interval"][1] < 0.0),
                "min_delta_mae": min(deltas),
                "max_delta_mae": max(deltas),
                "n_seeds_favouring_distillation": sum(1 for d in deltas if d < 0),
                "sign_consistent": all(d < 0 for d in deltas) or all(d > 0 for d in deltas),
                "direction": "LOWER_IS_BETTER",
                "precision_note": (
                    f"{len(deltas)} seeds. This is a spread, not a precise interval; Appendix A "
                    "§5 asks for instability to be reported rather than bootstrap precision "
                    "overstated at small sample sizes."
                ),
            }
        )
    record = {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "distillation_ablation_repeats",
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(runs),
        "chosen_path": CHOSEN_PATH,
        "teacher": dict(teacher_table["teacher"]),
        "across_seed_summary": summary,
        "runs": [
            {
                "seed": r["seed"],
                "ablation_path": r.get("ablation_path"),
                "matched": r["matched_setup"]["matched"],
                "n_holdout_rows": r["matched_setup"]["n_holdout_rows"],
                "arms": [
                    {
                        "synthetic_weight": a["synthetic_weight"],
                        "n_train_rows": a["n_train_rows"],
                        "mae_vs_teacher": a["evaluation"].get("mae_vs_teacher"),
                        "spearman_vs_teacher": a["evaluation"].get("spearman_vs_teacher"),
                        "collapsed": a["evaluation"].get("collapse", {}).get("collapsed"),
                        "max_family_share": a["diversity"]["max_family_share"],
                        "student_sha256": a["student_sha256"],
                        "descriptive_potency_spearman": a["descriptive_potency_correlation"].get(
                            "spearman_student_vs_measured_pactivity"
                        ),
                        "descriptive_potency_n": a["descriptive_potency_correlation"].get("n"),
                    }
                    for a in r["arms"]
                ],
            }
            for r in runs
        ],
        "claim_scope": (
            "Track C, computational-endpoint imitation. Repeating the ablation over seeds "
            "measures how stable the effect of the synthetic rows is; it says nothing about "
            "binding, potency or any other biological quantity."
        ),
    }
    path = out / "repeated.json"
    payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode("utf-8")
    path.write_bytes(payload)
    record["repeats_path"] = str(path)
    record["repeats_sha256"] = sha256_bytes(payload)
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m drug_agent.selfevolve.target.distill",
        description=(
            "Two stages, deliberately separate: 'teacher' runs the expensive named endpoint and "
            "writes a first-class teacher table; 'ablation' trains one student per synthetic "
            "weight on that table and compares them against the zero-weight control."
        ),
    )
    parser.add_argument("--stage", required=True, choices=("capabilities", "select", "teacher", "ablation"))
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--teacher-table", default="")
    parser.add_argument("--selection", default="")
    parser.add_argument("--n-anchor", type=int, default=120)
    parser.add_argument("--n-synthetic", type=int, default=80)
    parser.add_argument("--dock-call-cap", type=int, default=0)
    parser.add_argument("--student-fit-cap", type=int, default=12)
    parser.add_argument("--weights", default="0.0,0.25,1.0")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--max-per-family", type=int, default=DEFAULT_MAX_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default="",
        help="comma-separated seeds; runs the whole matched ablation once per seed and reports "
        "the across-seed spread, which is the variability one holdout cannot estimate",
    )
    parser.add_argument("--ledger", default="")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.stage == "capabilities":
        print(json.dumps(training_capability_report(), indent=2, sort_keys=True, default=str))
        return 0

    root = _repo_root()
    out = (
        Path(args.output_dir)
        if args.output_dir
        else (_snapshot(root) / "derived" / "distill" / (args.experiment_id or "run"))
    )
    out.mkdir(parents=True, exist_ok=True)

    ledger = None
    if args.ledger:
        from .budget import BudgetLedger

        ledger = BudgetLedger(args.ledger, campaign_id=args.experiment_id)
        if args.stage == "ablation":
            ledger.declare("student_fits", float(args.student_fit_cap))

    if args.stage == "select":
        selection = select_teacher_molecules(
            n_anchor=args.n_anchor, n_synthetic=args.n_synthetic, seed=args.seed, repo_root=root
        )
        (out / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: v for k, v in selection.items() if k not in ("anchor", "synthetic")},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    if args.stage == "teacher":
        selection = _json(Path(args.selection) if args.selection else out / "selection.json")
        molecules = list(selection["anchor"]) + list(selection["synthetic"])
        table = build_teacher_table(
            molecules,
            experiment_id=args.experiment_id or "distill",
            output_dir=out,
            repo_root=root,
            ledger=ledger,
            dock_call_cap=args.dock_call_cap or None,
            resume=not args.no_resume,
        )
        print(json.dumps(table["manifest"], indent=2, sort_keys=True, default=str))
        return 0

    table = load_teacher_table(Path(args.teacher_table) if args.teacher_table else out / "teacher_table.json")
    registry_path = _snapshot(root) / "protected" / "protected_exclusion_registry.json"
    registry = _json(registry_path) if registry_path.is_file() else None
    development = _development_teacher_rows(out, root)
    options: dict[str, Any] = {
        "weights": [float(w) for w in str(args.weights).split(",") if w.strip()],
        "exclusion_registry": registry,
        "max_per_family": args.max_per_family,
        "holdout_fraction": args.holdout_fraction,
        "experiment_id": args.experiment_id or "distill",
        "ledger": ledger,
        "development_rows": development,
    }
    if args.seeds.strip():
        repeats = repeat_ablation(
            table,
            output_dir=out / "ablation-repeats",
            seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
            **options,
        )
        print(
            json.dumps(
                {k: v for k, v in repeats.items() if k != "runs"}, indent=2, sort_keys=True, default=str
            )
        )
        return 0
    record = run_ablation(table, output_dir=out / "ablation", seed=args.seed, **options)
    summary = {k: v for k, v in record.items() if k not in ("arms",)}
    summary["arms"] = [
        {
            "synthetic_weight": a["synthetic_weight"],
            "n_train_rows": a["n_train_rows"],
            "n_synthetic_rows": a["n_synthetic_rows"],
            "mae_vs_teacher": a["evaluation"].get("mae_vs_teacher"),
            "spearman_vs_teacher": a["evaluation"].get("spearman_vs_teacher"),
            "collapsed": a["evaluation"].get("collapse", {}).get("collapsed"),
            "student_sha256": a["student_sha256"],
        }
        for a in record["arms"]
    ]
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def _development_teacher_rows(out: Path, root: Path) -> list[Record] | None:
    """Development-partition teacher rows, when a pass over them exists.

    Optional by design: the descriptive potency correlation is a secondary,
    explicitly descriptive number, and its absence is reported as
    ``INSUFFICIENT_DATA`` rather than filled in from somewhere else.
    """
    path = out / "development" / "teacher_table.json"
    if not path.is_file():
        return None
    return list(load_teacher_table(path)["rows"])


if __name__ == "__main__":  # pragma: no cover - operator entry point
    # Delegate to the module imported under its real name, not to this copy.
    # ``python -m drug_agent.selfevolve.target.distill`` imports this file as
    # ``__main__``, so a student trained by *this* copy pickles its class as
    # ``__main__.StudentModel`` and is reloadable only where that name happens
    # to exist. Running the canonically imported module instead means the
    # artifact records ``drug_agent.selfevolve.target.distill.StudentModel`` and
    # is reloadable anywhere the package is importable.
    from drug_agent.selfevolve.target.distill import _cli as _canonical_cli

    raise SystemExit(_canonical_cli())


__all__ = [
    "BACKEND_LEARNERS",
    "BOOTSTRAP_RESAMPLES",
    "CHOSEN_PATH",
    "COLLAPSE_DISPERSION_RATIO",
    "COMPUTATIONAL_EVIDENCE",
    "CONTRACT_VERSION",
    "CURATION_STATUSES",
    "DEFAULT_BASE_CONFIG",
    "DEFAULT_MAX_PER_FAMILY",
    "DEFAULT_WEIGHTS",
    "EVIDENCE_TYPES",
    "EXPERIMENTAL_EVIDENCE",
    "IDENTITY_FIELDS",
    "LEARNERS",
    "MEASURED_METRICS",
    "OPTIONAL_TEACHER_FIELDS",
    "ORIENTATIONS",
    "PARTITIONS",
    "REQUIRED_TEACHER_FIELDS",
    "SOURCE_KINDS",
    "STRUCTURE_ORIGINS",
    "StudentModel",
    "TEACHER_STATES",
    "build_student_training_set",
    "build_teacher_table",
    "canonical_hash",
    "choose_holdout_families",
    "collapse_indicators",
    "descriptive_potency_correlation",
    "evaluate_student",
    "example_teacher_row",
    "load_student",
    "load_teacher_table",
    "paired_effect",
    "pareto_front",
    "repeat_ablation",
    "run_ablation",
    "select_teacher_molecules",
    "sha256_bytes",
    "sha256_text",
    "student_diversity",
    "teacher_identity",
    "train_student",
    "training_capability_report",
    "validate_teacher_row",
]
