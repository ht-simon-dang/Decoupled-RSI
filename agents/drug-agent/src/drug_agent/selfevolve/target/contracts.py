"""Campaign, artifact, plan, result and status contracts for target evolution.

These seven records are the boundary the per-target evolution program crosses at
every step: a **Campaign** is one evolution run against one target under one
resource regime; an **EvolutionArtifact** is what an experiment produced and how
to call it; an **ExperimentPlan** is what was asked for before it ran; an
**EndpointResult** is one measurement or one honest absence; an
**ExperimentResult** is what the run cost and how it ended; a **Nomination** is
the single development-only choice a campaign submits; and a **Comparison** is
the paired statement made about it afterwards.

Three design decisions are load-bearing and are the reason this module is not a
bag of dictionaries:

**Nothing here couples what an artifact *is* to how it *did*.** ``kind`` lives on
the artifact; the scientific decision lives on the Comparison. A new
architecture is allowed to be ``NO_IMPROVEMENT`` and new weights are allowed to
be ``IMPROVED``, so no step can quietly infer evidence from novelty.

**A required field is required even when it is empty or null.** Every field in
Appendix A §7.1 is declared without a default. ``parent_id: None`` is a root
artifact and is legal; an artifact whose record simply *omits* ``parent_id`` has
lost its lineage and is refused. The same holds for ``target_manifest_hash``:
without it an artifact is not scoped to any target, and a serialization that
dropped it can never be repaired later because nothing knows what it was for.
``extra="forbid"`` is the other half — an unknown key is refused rather than
ignored, because an ignored key is how a requirement gets silently disabled.

**Structural validity is not existence.** ``validate_record`` checks that a hash
is a well-formed content address. It does *not* open the store — a synthetic
digest is perfectly valid input to a schema test. Only
:func:`verify_artifact_references` touches real content-addressed files, and it
is what must run before an artifact is scheduled or resolved.

Validation reuses the platform's own machinery: the pydantic ``_Rec`` base from
:mod:`drug_agent.selfevolve.schemas`, the endpoint status/value contract frozen
in :mod:`.api`, and the single process-exit mapping in
:mod:`drug_agent.cli.main`. Nothing here re-implements any of the three.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import ConfigDict, ValidationError, field_validator, model_validator

from ...contracts.enums import StrEnum, TrustLabel
from ..schemas import _Rec

Record = dict[str, Any]

#: Bumped when a contract's meaning changes. It is serialized with every record
#: so a stored artifact says which rules it was written under.
CONTRACT_VERSION = "1.0.0"

#: Canonical content address, as produced by :func:`drug_agent.util.hash_bytes`
#: and consumed by :meth:`drug_agent.storage.artifacts.ArtifactStore.exists`:
#: ``"<algo>:<hex digest>"``. A bare hex string is refused because it does not
#: say which algorithm produced it.
_CONTENT_ADDRESS = re.compile(r"^[a-z0-9]+:[0-9a-f]{32,128}$")


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
class ErrorCategory(StrEnum):
    """Stable reasons a record is refused.

    Callers and tests branch on these, never on message text. Adding a member is
    a compatible change; renaming one is not.
    """

    MISSING_FIELD = "MISSING_FIELD"  # a required key is absent from the record
    UNKNOWN_FIELD = "UNKNOWN_FIELD"  # a key no contract declares
    INVALID_TYPE = "INVALID_TYPE"  # present, but not the declared type
    INVALID_ENUM = "INVALID_ENUM"  # not one of the registered values
    EMPTY_VALUE = "EMPTY_VALUE"  # present and blank where content is required
    EMPTY_IDENTIFIER = "EMPTY_IDENTIFIER"  # a blank id cannot address anything
    INVALID_REFERENCE = "INVALID_REFERENCE"  # not a well-formed content address
    UNVERIFIABLE_REFERENCE = "UNVERIFIABLE_REFERENCE"  # well-formed, not in the store
    MISSING_COMPONENT = "MISSING_COMPONENT"  # the kind requires a component it lacks
    ENTRYPOINT_NOT_ARGV = "ENTRYPOINT_NOT_ARGV"  # not a callable argument vector
    ACTION_NOT_PERMITTED = "ACTION_NOT_PERMITTED"  # registered, but this campaign did not allow it
    PROTECTED_PATH = "PROTECTED_PATH"  # the plan reaches a file the referee depends on
    HASH_MISMATCH = "HASH_MISMATCH"  # a declared digest disagrees with the produced bytes
    INCONSISTENT_STATUS = "INCONSISTENT_STATUS"  # status disagrees with the record
    NON_FINITE_VALUE = "NON_FINITE_VALUE"  # NaN/inf is not a measurement
    INVALID_VALUE = "INVALID_VALUE"  # out of range or self-contradictory
    BUDGET_UNDECLARED = "BUDGET_UNDECLARED"  # no enforceable cap to schedule against
    COST_UNRECORDED = "COST_UNRECORDED"  # consumption was not recorded at all
    TEST_LEAKAGE = "TEST_LEAKAGE"  # held-out evidence reached a development decision
    TARGET_MISMATCH = "TARGET_MISMATCH"  # the artifact is scoped to a different target manifest
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"  # the decision outruns its own evidence


class ContractError(ValueError):
    """A record was refused. Subclasses ``ValueError`` per Appendix A §7.2.

    Attributes:
        category: the stable :class:`ErrorCategory`.
        contract: which contract refused it.
        field: the offending field, when one field is responsible.
        missing: the missing keys or unverifiable references, when relevant.
    """

    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        *,
        contract: str,
        field: str | None = None,
        missing: Sequence[str] = (),
    ) -> None:
        self.category = category
        self.contract = contract
        self.field = field
        self.missing = list(missing)
        where = f"{contract}.{field}" if field else contract
        super().__init__(f"[{category.value}] {where}: {message}")


# ---------------------------------------------------------------------------
# Registered vocabularies (Appendix A §3.1, §4, §5, §7.1)
# ---------------------------------------------------------------------------
class ArtifactKind(StrEnum):
    """What an evolution artifact *is*. Never what it achieved."""

    ALGORITHM = "ALGORITHM"
    MODEL_ARCHITECTURE = "MODEL_ARCHITECTURE"
    WEIGHTS = "WEIGHTS"
    TRAINED_MODEL = "TRAINED_MODEL"
    SCORER = "SCORER"
    SAMPLING_POLICY = "SAMPLING_POLICY"
    RECEPTOR_STRUCTURE = "RECEPTOR_STRUCTURE"
    RECEPTOR_ENSEMBLE = "RECEPTOR_ENSEMBLE"
    DATA_OR_LIBRARY = "DATA_OR_LIBRARY"
    COMPOSITE_PACKAGE = "COMPOSITE_PACKAGE"


class Track(StrEnum):
    """The registered benchmark track (Appendix A §3.1).

    ``STRUCTURE_ONLY_COMPUTATIONAL`` is the Track B variant the specification
    requires to be registered *by that name* when no defensible activity gate
    exists. It is a distinct member precisely so it cannot be silently dropped
    after candidates have been seen.
    """

    A = "A"
    B = "B"
    C = "C"
    STRUCTURE_ONLY_COMPUTATIONAL = "STRUCTURE_ONLY_COMPUTATIONAL"


class ResourceRegime(StrEnum):
    CONTROLLED = "CONTROLLED"
    ACQUISITION_ENABLED = "ACQUISITION_ENABLED"


class CampaignState(StrEnum):
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    NOMINATED = "NOMINATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ScientificDecision(StrEnum):
    """A scientific outcome. None of these is a software crash."""

    IMPROVED = "IMPROVED"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ClaimScope(StrEnum):
    """Which comparator a decision is about (Appendix A §5).

    Recorded separately for every comparator so that "better than its parent" can
    never be reported as "better than every existing tool".
    """

    PARENT_IMPROVED = "PARENT_IMPROVED"
    STANDARD_ADAPTATION_IMPROVED = "STANDARD_ADAPTATION_IMPROVED"
    EXTERNAL_BASELINE_IMPROVED = "EXTERNAL_BASELINE_IMPROVED"


class EndpointState(StrEnum):
    """Mirrors the frozen tuple in :data:`drug_agent.selfevolve.target.api.ENDPOINT_STATES`.

    The two are cross-checked at validation time, so the mirror cannot drift.
    """

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"


class ExperimentStatus(StrEnum):
    """Execution outcome of one experiment.

    Every member is a key of ``drug_agent.cli.main._RUN_EXIT_CODES``; the
    ``exit_code`` field is validated against ``run_exit_code`` rather than
    against a second copy of that table.
    """

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"
    CANCELLED = "CANCELLED"


class FailureCategory(StrEnum):
    """Why an experiment did not succeed. ``NONE`` exactly when it did.

    ``INFRASTRUCTURE_ERROR`` and ``CANDIDATE_ERROR`` are deliberately distinct:
    a transient infrastructure fault is retried and charged, a deterministic
    candidate fault is a result.
    """

    NONE = "NONE"
    CANDIDATE_ERROR = "CANDIDATE_ERROR"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    TIMEOUT = "TIMEOUT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    INVALID_CANDIDATE = "INVALID_CANDIDATE"
    #: The proposal was authorized and admitted, and the backend it needs does
    #: not exist on this host. Deliberately distinct from
    #: ``INFRASTRUCTURE_ERROR``: a transient fault is retried, a missing verb is
    #: an answer, and retrying it would only spend budget to learn the same
    #: thing. It is not in :data:`drug_agent.selfevolve.target.worker._RETRYABLE`.
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"


#: Evidence status of a measurement, reused from the platform's own taxonomy so
#: an endpoint carries the same provenance vocabulary as every other scientific
#: number in the repository. It is orthogonal to :class:`ArtifactKind`.
EvidenceType = TrustLabel


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------
def _fail(
    category: ErrorCategory,
    message: str,
    *,
    contract: str,
    field: str | None = None,
    missing: Sequence[str] = (),
) -> ContractError:
    return ContractError(category, message, contract=contract, field=field, missing=missing)


def _require_reference(value: Any, *, contract: str, field: str) -> None:
    if not isinstance(value, str) or not _CONTENT_ADDRESS.match(value):
        raise _fail(
            ErrorCategory.INVALID_REFERENCE,
            f"{value!r} is not a canonical content address of the form '<algo>:<hex>'",
            contract=contract,
            field=field,
        )


def _require_reference_list(values: Any, *, contract: str, field: str) -> None:
    for item in values or []:
        _require_reference(item, contract=contract, field=field)


def _require_identifier(value: Any, *, contract: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _fail(
            ErrorCategory.EMPTY_IDENTIFIER,
            "a blank identifier addresses nothing",
            contract=contract,
            field=field,
        )


def _require_text(value: Any, *, contract: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _fail(
            ErrorCategory.EMPTY_VALUE,
            "must carry content; a blank value records nothing",
            contract=contract,
            field=field,
        )


def _require_nonempty_sequence(values: Any, *, contract: str, field: str) -> None:
    if not values:
        raise _fail(
            ErrorCategory.EMPTY_VALUE,
            "must list at least one entry",
            contract=contract,
            field=field,
        )
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise _fail(
                ErrorCategory.EMPTY_VALUE,
                f"entry {item!r} is blank",
                contract=contract,
                field=field,
            )


def _require_finite(value: Any, *, contract: str, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            f"{value!r} is not a number",
            contract=contract,
            field=field,
        ) from exc
    if not math.isfinite(numeric):
        raise _fail(
            ErrorCategory.NON_FINITE_VALUE,
            f"{value!r} is not finite and is not a measurement",
            contract=contract,
            field=field,
        )
    return numeric


def _require_budget(amounts: Mapping[str, Any], *, contract: str, field: str) -> None:
    """An enforceable cap: at least one resource, every amount known and finite.

    ``None`` is how *unknown consumption* is recorded (see :func:`_require_cost`),
    and an unknown cap is not a cap — it would schedule work against nothing.
    """
    if not amounts:
        raise _fail(
            ErrorCategory.BUDGET_UNDECLARED,
            "no resource cap declared; nothing can be scheduled against it",
            contract=contract,
            field=field,
        )
    for resource, amount in amounts.items():
        _require_text(resource, contract=contract, field=field)
        if amount is None:
            raise _fail(
                ErrorCategory.BUDGET_UNDECLARED,
                f"cap for {resource!r} is unknown; an unknown cap is not a cap",
                contract=contract,
                field=field,
            )
        if _require_finite(amount, contract=contract, field=field) < 0:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"cap for {resource!r} is negative",
                contract=contract,
                field=field,
            )


def _require_cost(amounts: Mapping[str, Any], *, contract: str, field: str, allow_empty: bool) -> None:
    """Recorded consumption. ``None`` means *unavailable telemetry*, never zero."""
    if not amounts and not allow_empty:
        raise _fail(
            ErrorCategory.COST_UNRECORDED,
            "consumption was not recorded; unknown telemetry is recorded as null, not omitted",
            contract=contract,
            field=field,
        )
    for resource, amount in amounts.items():
        _require_text(resource, contract=contract, field=field)
        if amount is None:
            continue  # unavailable, and deliberately distinguishable from 0.0
        if _require_finite(amount, contract=contract, field=field) < 0:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"consumption of {resource!r} is negative",
                contract=contract,
                field=field,
            )


def _require_metrics(values: Mapping[str, Any], *, contract: str, field: str) -> None:
    for metric, value in values.items():
        _require_text(metric, contract=contract, field=field)
        if value is None:
            continue  # a missing measurement, not a zero
        _require_finite(value, contract=contract, field=field)


# ---------------------------------------------------------------------------
# Base record
# ---------------------------------------------------------------------------
class _TargetRec(_Rec):
    """Reuses the selfevolve pydantic base, but refuses unknown fields.

    ``_Rec`` ignores extras, which is right for records normalized out of a
    journal that may grow keys. It is wrong here: these records gate scheduling
    and promotion, and an ignored key is how a requirement gets disabled without
    anyone seeing it.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), str_strip_whitespace=True)

    #: Serialized with every record so a stored artifact says which rules it was
    #: written under.
    contract_version: str = CONTRACT_VERSION

    REFERENCE_FIELDS: ClassVar[tuple[str, ...]] = ()
    REFERENCE_LIST_FIELDS: ClassVar[tuple[str, ...]] = ()
    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ()
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def _validate_contract(self) -> _TargetRec:
        name = type(self).__name__
        for field in self.IDENTIFIER_FIELDS:
            _require_identifier(getattr(self, field), contract=name, field=field)
        for field in self.REFERENCE_FIELDS:
            value = getattr(self, field)
            if value is None:
                continue  # nullable references are declared as such per contract
            _require_reference(value, contract=name, field=field)
        for field in self.REFERENCE_LIST_FIELDS:
            _require_reference_list(getattr(self, field), contract=name, field=field)
        for field in self.TEXT_FIELDS:
            _require_text(getattr(self, field), contract=name, field=field)
        self._validate_record()
        return self

    def _validate_record(self) -> None:
        """Per-contract cross-field rules."""

    def to_record(self) -> Record:
        """The JSON-serializable mapping. Nothing is dropped, including nulls."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# The seven contracts (Appendix A §7.1)
# ---------------------------------------------------------------------------
class Campaign(_TargetRec):
    """One evolution run: one target, one track, one regime, one budget."""

    campaign_id: str
    brief_hash: str
    target_id: str
    target_manifest_hash: str
    track: Track
    resource_regime: ResourceRegime
    #: The frozen parent package the campaign evolves from (baseline B0).
    parent_artifact_id: str
    data_hash: str
    split_hash: str
    evaluator_hash: str
    #: ``{resource: cap}``. Declared as nullable so that a ``None`` cap reaches
    #: :func:`_require_budget` and is refused as ``BUDGET_UNDECLARED`` — "an
    #: unknown cap is not a cap" — rather than as an anonymous type error. No
    #: ``None`` survives validation.
    budget: dict[str, float | None]
    seed: int
    protocol_hash: str
    #: Lifecycle state. Not in the §7.1 required set; defaulted, never inferred.
    state: CampaignState = CampaignState.RUNNING

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("campaign_id", "target_id", "parent_artifact_id")
    REFERENCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "brief_hash",
        "target_manifest_hash",
        "data_hash",
        "split_hash",
        "evaluator_hash",
        "protocol_hash",
    )

    def _validate_record(self) -> None:
        _require_budget(self.budget, contract="Campaign", field="budget")
        if self.seed < 0:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                "a seed must be non-negative to be reproducible",
                contract="Campaign",
                field="seed",
            )


class EvolutionArtifact(_TargetRec):
    """A persisted artifact with lineage to an experiment and a callable interface.

    ``kind`` says what it is. It says nothing about whether it helped: that is a
    :class:`Comparison`, and the two are deliberately not joined here.
    """

    artifact_id: str
    #: The artifact this one descends from. ``None`` is a declared root; the key
    #: itself is mandatory, so lineage cannot be lost by omission.
    parent_id: str | None
    kind: ArtifactKind
    #: The target manifest this artifact is scoped to. Compatibility with a
    #: target is this field and nothing else — see :meth:`is_compatible_with`.
    target_manifest_hash: str
    #: An argument vector, e.g. ``["python", "-m", "pkg.score", "--config", "c.json"]``.
    #: Never an interpolated shell string: a string would be word-split by
    #: someone downstream, and whoever controls the artifact would control the
    #: shell.
    entrypoint: list[str]
    code_hash: str
    #: Empty for a non-neural algorithm. Required to be non-empty only for the
    #: kinds that are *defined* by their weights.
    weights_hashes: list[str]
    structure_hashes: list[str]
    environment_hash: str
    training_data_hashes: list[str]
    #: The experiment that produced it. An artifact without one has no lineage
    #: and cannot be attributed.
    experiment_id: str

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("artifact_id", "experiment_id")
    REFERENCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "target_manifest_hash",
        "code_hash",
        "environment_hash",
    )
    REFERENCE_LIST_FIELDS: ClassVar[tuple[str, ...]] = (
        "weights_hashes",
        "structure_hashes",
        "training_data_hashes",
    )

    #: Kinds that are defined by a component: ``kind -> (field, minimum count)``.
    #: Everything absent from this table is representable with code alone, which
    #: is what lets an algorithm with no weights be a first-class artifact.
    COMPONENT_REQUIREMENTS: ClassVar[dict[str, tuple[str, int]]] = {
        ArtifactKind.WEIGHTS.value: ("weights_hashes", 1),
        ArtifactKind.TRAINED_MODEL.value: ("weights_hashes", 1),
        ArtifactKind.RECEPTOR_STRUCTURE.value: ("structure_hashes", 1),
        ArtifactKind.RECEPTOR_ENSEMBLE.value: ("structure_hashes", 2),
        ArtifactKind.DATA_OR_LIBRARY.value: ("training_data_hashes", 1),
    }

    @field_validator("entrypoint", mode="before")
    @classmethod
    def _entrypoint_is_argv(cls, value: Any) -> Any:
        if isinstance(value, str):
            raise _fail(
                ErrorCategory.ENTRYPOINT_NOT_ARGV,
                "entrypoint is an argument vector, not an interpolated shell command; "
                f"pass {value.split()!r} rather than {value!r}",
                contract="EvolutionArtifact",
                field="entrypoint",
            )
        return value

    def _validate_record(self) -> None:
        if not self.entrypoint or any(not part.strip() for part in self.entrypoint):
            raise _fail(
                ErrorCategory.ENTRYPOINT_NOT_ARGV,
                f"{self.entrypoint!r} is not a callable argument vector",
                contract="EvolutionArtifact",
                field="entrypoint",
            )
        if self.parent_id is not None:
            _require_identifier(self.parent_id, contract="EvolutionArtifact", field="parent_id")
        requirement = self.COMPONENT_REQUIREMENTS.get(str(self.kind))
        if requirement is not None:
            field, minimum = requirement
            present = len(getattr(self, field))
            if present < minimum:
                raise _fail(
                    ErrorCategory.MISSING_COMPONENT,
                    f"a {self.kind} artifact needs at least {minimum} entry in {field}, has {present}",
                    contract="EvolutionArtifact",
                    field=field,
                )
        if str(self.kind) == ArtifactKind.COMPOSITE_PACKAGE.value and not (
            self.weights_hashes or self.structure_hashes or self.training_data_hashes
        ):
            raise _fail(
                ErrorCategory.MISSING_COMPONENT,
                "a composite package must reference at least one component artifact",
                contract="EvolutionArtifact",
                field="weights_hashes",
            )

    def is_compatible_with(self, target_manifest_hash: str) -> bool:
        """Compatibility is explicit: the artifact names the target it is for."""
        return self.target_manifest_hash == target_manifest_hash


class ExperimentPlan(_TargetRec):
    """What an experiment intends, registered before it runs."""

    experiment_id: str
    campaign_id: str
    hypothesis: str
    kind: ArtifactKind
    primary_track: Track
    parent_id: str | None
    #: The action to take. The allowed menu is supplied by the caller at
    #: validation time (Step 10's ``validate_plan``); here it must merely be a
    #: named action rather than blank.
    action: str
    inputs: dict[str, Any]
    expected_outputs: list[str]
    #: What this experiment changes relative to its parent. An experiment that
    #: changes nothing is not an experiment.
    changed_factors: list[str]
    #: Nullable for the same reason as :attr:`Campaign.budget`; no ``None``
    #: survives validation.
    requested_budget: dict[str, float | None]
    development_metric: str
    stop_rule: str

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("experiment_id", "campaign_id")
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = (
        "hypothesis",
        "action",
        "development_metric",
        "stop_rule",
    )

    def _validate_record(self) -> None:
        if self.parent_id is not None:
            _require_identifier(self.parent_id, contract="ExperimentPlan", field="parent_id")
        _require_nonempty_sequence(self.changed_factors, contract="ExperimentPlan", field="changed_factors")
        _require_nonempty_sequence(self.expected_outputs, contract="ExperimentPlan", field="expected_outputs")
        _require_budget(self.requested_budget, contract="ExperimentPlan", field="requested_budget")


class EndpointResult(_TargetRec):
    """One measurement, or one honest absence of a measurement.

    The status/value rule is not restated here. It is enforced by calling
    :func:`drug_agent.selfevolve.target.api.endpoint_result`, which Step 2 froze:
    a value exists only for ``SUCCEEDED``, a measured ``0.0`` survives, and a
    non-finite value is refused because NaN compares false against every
    threshold and is indistinguishable from a criterion legitimately missed.
    """

    status: EndpointState
    value: float | None
    unit: str | None
    metric: str
    molecule_id: str | None
    receptor_hash: str | None
    evaluator_hash: str
    #: Provenance of the number — observed, calculated, predicted. Independent of
    #: ``status`` and of any artifact's ``kind``.
    evidence_type: EvidenceType
    error_code: str | None
    cost: dict[str, float | None]

    REFERENCE_FIELDS: ClassVar[tuple[str, ...]] = ("receptor_hash", "evaluator_hash")
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("metric",)

    def _validate_record(self) -> None:
        from .api import ENDPOINT_STATES, endpoint_result  # local: api imports this module

        if tuple(s.value for s in EndpointState) != tuple(ENDPOINT_STATES):
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                f"endpoint states drifted from the frozen api contract {ENDPOINT_STATES}",
                contract="EndpointResult",
                field="status",
            )
        if (str(self.status) == EndpointState.SUCCEEDED.value) != (self.value is not None):
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                f"status={self.status} carries value={self.value!r}; a value is required for "
                "SUCCEEDED and forbidden otherwise",
                contract="EndpointResult",
                field="value",
            )
        if self.value is not None:
            _require_finite(self.value, contract="EndpointResult", field="value")
        try:
            endpoint_result(str(self.status), self.value)
        except ValueError as exc:  # pragma: no cover - the checks above precede it
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                str(exc),
                contract="EndpointResult",
                field="value",
            ) from exc
        for field in ("unit", "molecule_id", "error_code"):
            value = getattr(self, field)
            if value is not None:
                _require_text(value, contract="EndpointResult", field=field)
        _require_cost(self.cost, contract="EndpointResult", field="cost", allow_empty=True)


class ExperimentResult(_TargetRec):
    """How one experiment ended, what it produced and what it consumed.

    A failed experiment is a complete record: it has a failure category, an
    honest ``None`` for every metric it did not measure, and a cost. A failure
    with no recorded cost would make the campaign's ledger lie.
    """

    experiment_id: str
    status: ExperimentStatus
    #: Must equal ``drug_agent.cli.main.run_exit_code(status)``. There is one
    #: exit-status mapping in this repository and this is not a second copy.
    exit_code: int
    candidate_revision: str
    #: Where the candidate's code was actually imported from, so a run that
    #: silently imported the operator's tree instead of the candidate checkout is
    #: visible in the record.
    import_origin: str
    artifacts: list[str]
    #: ``{metric: value or None}``. ``None`` is *not measured*, never zero.
    development_metrics: dict[str, float | None]
    #: ``{resource: amount or None}``. ``None`` is *unknown telemetry*, never zero.
    actual_cost: dict[str, float | None]
    failure_category: FailureCategory
    intervention_log: list[dict[str, Any]]

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("experiment_id",)
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("candidate_revision", "import_origin")

    def _validate_record(self) -> None:
        from ...cli.main import run_exit_code  # local: keeps the CLI out of import order

        expected = run_exit_code(str(self.status))
        if self.exit_code != expected:
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                f"status={self.status} maps to exit code {expected}, record says {self.exit_code}",
                contract="ExperimentResult",
                field="exit_code",
            )
        succeeded = str(self.status) == ExperimentStatus.SUCCEEDED.value
        if succeeded != (str(self.failure_category) == FailureCategory.NONE.value):
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                f"status={self.status} with failure_category={self.failure_category}; "
                "NONE is permitted exactly when the experiment succeeded",
                contract="ExperimentResult",
                field="failure_category",
            )
        for artifact_id in self.artifacts:
            _require_identifier(artifact_id, contract="ExperimentResult", field="artifacts")
        _require_metrics(self.development_metrics, contract="ExperimentResult", field="development_metrics")
        _require_cost(self.actual_cost, contract="ExperimentResult", field="actual_cost", allow_empty=False)


class Nomination(_TargetRec):
    """The single artifact a campaign submits, chosen on development evidence.

    ``test_access_count`` must be zero. Appendix A §5 puts nomination strictly
    before any final-test result is revealed, so a nomination record that has
    already touched the held-out set is not a late-caught mistake — it is the
    evidence that the result is contaminated, and it is refused here.
    """

    campaign_id: str
    artifact_id: str
    protocol_hash: str
    development_selection_rule: str
    nominated_at: str
    test_access_count: int

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("campaign_id", "artifact_id")
    REFERENCE_FIELDS: ClassVar[tuple[str, ...]] = ("protocol_hash",)
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("development_selection_rule", "nominated_at")

    def _validate_record(self) -> None:
        if self.test_access_count < 0:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                "a count cannot be negative",
                contract="Nomination",
                field="test_access_count",
            )
        if self.test_access_count != 0:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"the held-out set was accessed {self.test_access_count} time(s) before "
                "nomination; nomination is a development-only decision",
                contract="Nomination",
                field="test_access_count",
            )


class Comparison(_TargetRec):
    """A paired statement about one candidate against one named comparator.

    Sign convention, and it is load-bearing: ``effect`` and ``interval`` are
    stated as **improvement**, positive meaning better, whatever the metric's
    own direction. An MAE reduction of 0.09 is ``effect = +0.09``. Without a
    fixed convention an interval could not be checked against a decision at all.

    ``IMPROVED`` therefore requires an interval whose lower bound excludes no
    improvement and every declared constraint satisfied (Appendix A §5). A
    favorable point estimate without that support is ``INCONCLUSIVE``, and this
    record refuses to call it anything else.
    """

    track: Track
    primary_metric: str
    candidate_id: str
    baseline_id: str
    paired_results: list[dict[str, Any]]
    effect: float
    #: ``[low, high]`` uncertainty interval on ``effect``, same sign convention.
    interval: list[float]
    #: ``{constraint: satisfied}``. The numbers behind each one live in
    #: ``paired_results``; this is the decision-relevant summary.
    constraints: dict[str, bool]
    decision: ScientificDecision
    claim_scope: ClaimScope

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("candidate_id", "baseline_id")
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("primary_metric",)

    def _validate_record(self) -> None:
        if not self.paired_results:
            raise _fail(
                ErrorCategory.EMPTY_VALUE,
                "a comparison needs at least one paired campaign result",
                contract="Comparison",
                field="paired_results",
            )
        _require_finite(self.effect, contract="Comparison", field="effect")
        if len(self.interval) != 2:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"interval must be [low, high], got {self.interval!r}",
                contract="Comparison",
                field="interval",
            )
        low, high = (
            _require_finite(bound, contract="Comparison", field="interval") for bound in self.interval
        )
        if low > high:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"interval [{low}, {high}] is inverted",
                contract="Comparison",
                field="interval",
            )
        if str(self.decision) != ScientificDecision.IMPROVED.value:
            return
        if low <= 0:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"IMPROVED requires the paired interval to exclude no improvement; [{low}, {high}] does not",
                contract="Comparison",
                field="interval",
            )
        unmet = sorted(name for name, ok in self.constraints.items() if not ok)
        if unmet:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"IMPROVED requires every declared constraint to hold; unmet: {unmet}",
                contract="Comparison",
                field="constraints",
                missing=unmet,
            )


# ---------------------------------------------------------------------------
# Registry, frozen field table and the validation entry points
# ---------------------------------------------------------------------------
CONTRACTS: dict[str, type[_TargetRec]] = {
    "Campaign": Campaign,
    "EvolutionArtifact": EvolutionArtifact,
    "ExperimentPlan": ExperimentPlan,
    "EndpointResult": EndpointResult,
    "ExperimentResult": ExperimentResult,
    "Nomination": Nomination,
    "Comparison": Comparison,
}

#: Appendix A §7.1, transcribed. This is the drift anchor: the models are
#: checked against it, and a field quietly given a default stops being required
#: and fails that check. Changing this table is a deliberate protocol change.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "Campaign": (
        "campaign_id",
        "brief_hash",
        "target_id",
        "target_manifest_hash",
        "track",
        "resource_regime",
        "parent_artifact_id",
        "data_hash",
        "split_hash",
        "evaluator_hash",
        "budget",
        "seed",
        "protocol_hash",
    ),
    "EvolutionArtifact": (
        "artifact_id",
        "parent_id",
        "kind",
        "target_manifest_hash",
        "entrypoint",
        "code_hash",
        "weights_hashes",
        "structure_hashes",
        "environment_hash",
        "training_data_hashes",
        "experiment_id",
    ),
    "ExperimentPlan": (
        "experiment_id",
        "campaign_id",
        "hypothesis",
        "kind",
        "primary_track",
        "parent_id",
        "action",
        "inputs",
        "expected_outputs",
        "changed_factors",
        "requested_budget",
        "development_metric",
        "stop_rule",
    ),
    "EndpointResult": (
        "status",
        "value",
        "unit",
        "metric",
        "molecule_id",
        "receptor_hash",
        "evaluator_hash",
        "evidence_type",
        "error_code",
        "cost",
    ),
    "ExperimentResult": (
        "experiment_id",
        "status",
        "exit_code",
        "candidate_revision",
        "import_origin",
        "artifacts",
        "development_metrics",
        "actual_cost",
        "failure_category",
        "intervention_log",
    ),
    "Nomination": (
        "campaign_id",
        "artifact_id",
        "protocol_hash",
        "development_selection_rule",
        "nominated_at",
        "test_access_count",
    ),
    "Comparison": (
        "track",
        "primary_metric",
        "candidate_id",
        "baseline_id",
        "paired_results",
        "effect",
        "interval",
        "constraints",
        "decision",
        "claim_scope",
    ),
}

_PYDANTIC_ERROR_CATEGORY = {
    "missing": ErrorCategory.MISSING_FIELD,
    "extra_forbidden": ErrorCategory.UNKNOWN_FIELD,
}


def _model_for(contract: type[_TargetRec] | str) -> type[_TargetRec]:
    if isinstance(contract, str):
        try:
            return CONTRACTS[contract]
        except KeyError:
            raise ValueError(f"unknown contract {contract!r}; known: {sorted(CONTRACTS)}") from None
    if contract.__name__ not in CONTRACTS:
        raise ValueError(f"{contract.__name__} is not a registered target contract")
    return contract


def _category_for(error: Mapping[str, Any]) -> ErrorCategory:
    kind = str(error.get("type", ""))
    if kind in _PYDANTIC_ERROR_CATEGORY:
        return _PYDANTIC_ERROR_CATEGORY[kind]
    if "enum" in kind or kind == "literal_error":
        return ErrorCategory.INVALID_ENUM
    return ErrorCategory.INVALID_TYPE


def _reraise(exc: ValidationError, contract: str) -> ContractError:
    """Translate pydantic's report into one stable category.

    A :class:`ContractError` raised inside a validator is handed back unchanged;
    everything else is classified by pydantic's own error type so callers never
    have to read a message to find out what went wrong.
    """
    errors = exc.errors()
    for error in errors:
        original = (error.get("ctx") or {}).get("error")
        if isinstance(original, ContractError):
            return original
    first = errors[0]
    loc = first.get("loc") or ()
    field = str(loc[0]) if loc else None
    category = _category_for(first)
    if category is ErrorCategory.UNKNOWN_FIELD:
        unknown = sorted(str(e["loc"][0]) for e in errors if e.get("type") == "extra_forbidden")
        return ContractError(
            category,
            f"unknown field(s) {unknown}; an ignored field is a silently disabled requirement",
            contract=contract,
            field=unknown[0] if len(unknown) == 1 else None,
            missing=unknown,
        )
    if category is ErrorCategory.MISSING_FIELD:
        absent = sorted(str(e["loc"][0]) for e in errors if e.get("type") == "missing")
        order = REQUIRED_FIELDS.get(contract, ())
        ordered = [f for f in order if f in absent] + [f for f in absent if f not in order]
        return ContractError(
            category,
            f"required field(s) {ordered} are absent; a required field stays required even "
            "when its value is empty or null",
            contract=contract,
            field=ordered[0] if ordered else None,
            missing=ordered,
        )
    return ContractError(
        category,
        f"{first.get('msg', 'invalid value')} (input={first.get('input')!r})",
        contract=contract,
        field=field,
    )


def validate_record(contract: type[_TargetRec] | str, record: Mapping[str, Any]) -> Record:
    """Validate ``record`` against ``contract`` and return it as JSON-ready data.

    Args:
        contract: a registered contract class or its name.
        record: the mapping to validate. Not mutated.

    Returns:
        The validated record as a JSON-serializable ``dict``, including every
        required key — nulls are kept, because a dropped null is a lost field.

    Raises:
        ContractError: a :class:`ValueError` carrying a stable
            :class:`ErrorCategory`, the contract, the offending field and, for
            missing/unknown keys, the full list.
    """
    model = _model_for(contract)
    if not isinstance(record, Mapping):
        raise ContractError(
            ErrorCategory.INVALID_TYPE,
            f"expected a mapping, got {type(record).__name__}",
            contract=model.__name__,
        )
    try:
        return model.model_validate(dict(record)).to_record()
    except ValidationError as exc:
        raise _reraise(exc, model.__name__) from exc


def artifact_reference_hashes(record: Mapping[str, Any]) -> list[str]:
    """Every content address an artifact record depends on, in a stable order."""
    validated = validate_record(EvolutionArtifact, record)
    hashes = [validated["code_hash"], validated["environment_hash"]]
    for field in ("weights_hashes", "structure_hashes", "training_data_hashes"):
        hashes.extend(validated[field])
    seen: dict[str, None] = {}
    for item in hashes:
        seen.setdefault(item, None)
    return list(seen)


def verify_artifact_references(record: Mapping[str, Any], store: Any) -> Record:
    """Structural validation **plus** proof the referenced bytes actually exist.

    This is the runtime check, and it is deliberately separate from
    :func:`validate_record`: a schema test may legitimately use a synthetic
    digest, but an artifact about to be scheduled, resolved or promoted must
    reference content the store really holds.

    Args:
        record: an artifact record.
        store: anything with ``exists(content_hash) -> bool``, i.e. a
            :class:`drug_agent.storage.artifacts.ArtifactStore`.

    Returns:
        The validated record.

    Raises:
        ContractError: ``UNVERIFIABLE_REFERENCE``, with ``.missing`` listing
            every content address the store does not hold.
    """
    validated = validate_record(EvolutionArtifact, record)
    missing = [h for h in artifact_reference_hashes(validated) if not store.exists(h)]
    if missing:
        raise ContractError(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            f"{len(missing)} referenced artifact(s) are not in the store: {missing}",
            contract="EvolutionArtifact",
            field="artifact_id",
            missing=missing,
        )
    return validated


#: The four hash families an accepted experiment has to be able to name. They
#: live on :class:`EvolutionArtifact` — Appendix A §7.1 fixes
#: :class:`ExperimentResult`'s field list and this is deliberately not a second
#: copy of them — so an ``ExperimentResult`` "lists" them by referencing
#: artifacts that carry them, and :func:`require_experiment_provenance` is what
#: makes that reference mandatory rather than optional.
PROVENANCE_HASH_FIELDS: tuple[str, ...] = (
    "code_hash",
    "weights_hashes",
    "structure_hashes",
    "environment_hash",
)


def require_experiment_provenance(
    result: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    store: Any = None,
    run_environment_hash: str | None = None,
) -> Record:
    """Refuse to accept an experiment that cannot say what it produced.

    Step 10's acceptance rule. An ``ExperimentResult`` names artifacts by id;
    this resolves every one of those ids to a validated
    :class:`EvolutionArtifact` and returns the code, weights, structure and
    environment content addresses together with the plan's ``changed_factors``.
    Nothing here reads a metric: a worker's own numbers are not evidence, and
    the provenance record exists so the *independent* evaluator has something
    verifiable to score.

    Three things it refuses to do:

    * it never accepts an artifact id with no record behind it. A result listing
      ``art_x`` while no artifact record exists is a claim, not a lineage;
    * it never accepts a record whose referenced bytes are absent. With a
      ``store``, every content address is checked through
      :func:`verify_artifact_references` — which is how "weights with no code"
      is caught, because ``code_hash`` is required and has to resolve;
    * it never accepts an experiment that changed nothing. ``changed_factors``
      comes from the plan and must be non-empty, so an experiment cannot be
      attributed to a factor it never varied.

    Args:
        result: an :class:`ExperimentResult` record.
        plan: the :class:`ExperimentPlan` the result answers.
        artifacts: the artifact records the run produced, in any order.
        store: optional content-addressed store; when given, every reference is
            verified to exist.
        run_environment_hash: the environment the run itself executed in,
            recorded even when the run produced no artifact — a failed
            experiment still has to say where it ran.

    Returns:
        A JSON-ready provenance record.

    Raises:
        ContractError: ``MISSING_FIELD`` for an unresolvable or duplicated
            artifact id, ``INVALID_VALUE`` for an artifact that names a
            different experiment, ``EMPTY_VALUE`` for empty changed factors, and
            ``UNVERIFIABLE_REFERENCE`` (from
            :func:`verify_artifact_references`) for absent bytes.
    """
    validated_result = validate_record(ExperimentResult, result)
    validated_plan = validate_record(ExperimentPlan, plan)
    experiment_id = validated_result["experiment_id"]
    if validated_plan["experiment_id"] != experiment_id:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"the result is for {experiment_id!r} and the plan is for "
            f"{validated_plan['experiment_id']!r}; provenance joins one plan to one result",
            contract="ExperimentResult",
            field="experiment_id",
        )
    if not validated_plan["changed_factors"]:  # pragma: no cover - the plan contract precedes this
        raise _fail(
            ErrorCategory.EMPTY_VALUE,
            "an experiment that changed nothing has nothing to attribute",
            contract="ExperimentPlan",
            field="changed_factors",
        )

    by_id: dict[str, Record] = {}
    for record in artifacts:
        validated = (
            verify_artifact_references(record, store)
            if store is not None
            else validate_record(EvolutionArtifact, record)
        )
        artifact_id = validated["artifact_id"]
        if artifact_id in by_id:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"two artifact records share the id {artifact_id!r}",
                contract="EvolutionArtifact",
                field="artifact_id",
            )
        if validated["experiment_id"] != experiment_id:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"artifact {artifact_id!r} names experiment {validated['experiment_id']!r}, "
                f"not {experiment_id!r}; lineage is not transferable",
                contract="EvolutionArtifact",
                field="experiment_id",
            )
        by_id[artifact_id] = validated

    unresolved = [a for a in validated_result["artifacts"] if a not in by_id]
    if unresolved:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            f"the result lists {len(unresolved)} artifact id(s) with no artifact record: "
            f"{unresolved}. An id without a record is a claim, not an artifact",
            contract="ExperimentResult",
            field="artifacts",
            missing=unresolved,
        )

    rows: list[Record] = []
    for artifact_id in validated_result["artifacts"]:
        art = by_id[artifact_id]
        rows.append(
            {
                "artifact_id": artifact_id,
                "kind": art["kind"],
                "parent_id": art["parent_id"],
                "entrypoint": list(art["entrypoint"]),
                "code_hash": art["code_hash"],
                "weights_hashes": list(art["weights_hashes"]),
                "structure_hashes": list(art["structure_hashes"]),
                "environment_hash": art["environment_hash"],
                "training_data_hashes": list(art["training_data_hashes"]),
            }
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "experiment_id": experiment_id,
        "campaign_id": validated_plan["campaign_id"],
        "action": validated_plan["action"],
        "status": validated_result["status"],
        "changed_factors": list(validated_plan["changed_factors"]),
        "candidate_revision": validated_result["candidate_revision"],
        "import_origin": validated_result["import_origin"],
        "run_environment_hash": run_environment_hash,
        "artifacts": rows,
        "references_verified": store is not None,
        "n_artifacts": len(rows),
    }


__all__ = [
    "CONTRACTS",
    "CONTRACT_VERSION",
    "REQUIRED_FIELDS",
    "ArtifactKind",
    "Campaign",
    "CampaignState",
    "ClaimScope",
    "Comparison",
    "ContractError",
    "EndpointResult",
    "EndpointState",
    "ErrorCategory",
    "EvidenceType",
    "EvolutionArtifact",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentStatus",
    "FailureCategory",
    "PROVENANCE_HASH_FIELDS",
    "Nomination",
    "Record",
    "ResourceRegime",
    "ScientificDecision",
    "Track",
    "artifact_reference_hashes",
    "require_experiment_provenance",
    "validate_record",
    "verify_artifact_references",
]
