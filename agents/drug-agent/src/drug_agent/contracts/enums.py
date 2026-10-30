"""Canonical enumerations shared across all contracts.

Enums are string-valued so they serialize cleanly to JSON/YAML/Parquet and are
stable identifiers in persisted artifacts.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum whose ``str()`` is the bare value (portable across py311)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class TrustLabel(StrEnum):
    """Epistemic provenance of a value. Every scientific number carries one."""

    OBSERVED = "OBSERVED"  # measured in the physical world (experimental assay)
    CURATED = "CURATED"  # human-curated from a trusted database
    CALCULATED = "CALCULATED"  # deterministic computation from inputs (e.g. RDKit descriptor)
    PREDICTED = "PREDICTED"  # statistical/ML model output
    INFERRED = "INFERRED"  # derived by reasoning over other evidence
    HYPOTHESIZED = "HYPOTHESIZED"  # a scientific hypothesis, not yet tested
    PROPOSED = "PROPOSED"  # a suggested action/experiment


class TaskState(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_FINAL = "FAILED_FINAL"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"


TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED_FINAL,
        TaskState.CANCELLED,
        TaskState.SUPERSEDED,
    }
)


class RunState(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PAUSED = "PAUSED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CostClass(StrEnum):
    """Coarse expected cost of invoking a capability."""

    TRIVIAL = "TRIVIAL"  # microseconds, pure python
    CHEAP = "CHEAP"  # milliseconds, local compute
    MODERATE = "MODERATE"  # seconds, heavier local compute
    EXPENSIVE = "EXPENSIVE"  # minutes+, GPU / simulation / external service


class Determinism(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    STOCHASTIC = "STOCHASTIC"


class Qualifier(StrEnum):
    """Measurement qualifier for assay/prediction values."""

    EXACT = "="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    APPROX = "~"
    RANGE = "range"
    CENSORED = "censored"


class ApplicabilityDomain(StrEnum):
    IN_DOMAIN = "IN_DOMAIN"
    BORDERLINE = "BORDERLINE"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    DEFER = "DEFER"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class Severity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BriefKind(StrEnum):
    TARGET_BASED = "target_based"
    PHENOTYPE_BASED = "phenotype_based"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class GenerationMethod(StrEnum):
    """De-novo candidate-generation backend selected by a brief."""

    NONE = "none"  # no generative model; library + seed analogues only
    PFM = "pfm"  # Perturbed Flow Matching, structure-conditioned 3D generation


class CapabilityStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    MOCKED = "MOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"
