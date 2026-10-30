"""Per-target evolution: contracts, benchmark plumbing and the public API.

Created at Step 2 of the per-target evolution program with the endpoint
status/value contract, and extended at Step 3 with the seven data contracts of
Appendix A §7.1 in :mod:`.contracts`. The rest of the surface declared in §7.2 is
added by the step that owns each function; this package deliberately contains no
placeholder that would import cleanly and compute nothing.
"""

from __future__ import annotations

from .api import ENDPOINT_STATES, endpoint_result, resolve_artifact, validate_artifact
from .contracts import (
    CONTRACT_VERSION,
    CONTRACTS,
    REQUIRED_FIELDS,
    ArtifactKind,
    Campaign,
    CampaignState,
    ClaimScope,
    Comparison,
    ContractError,
    EndpointResult,
    ErrorCategory,
    EvolutionArtifact,
    ExperimentPlan,
    ExperimentResult,
    ExperimentStatus,
    FailureCategory,
    Nomination,
    ResourceRegime,
    ScientificDecision,
    Track,
    artifact_reference_hashes,
    validate_record,
    verify_artifact_references,
)

__all__ = [
    "CONTRACTS",
    "CONTRACT_VERSION",
    "ENDPOINT_STATES",
    "REQUIRED_FIELDS",
    "ArtifactKind",
    "Campaign",
    "CampaignState",
    "ClaimScope",
    "Comparison",
    "ContractError",
    "EndpointResult",
    "ErrorCategory",
    "EvolutionArtifact",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentStatus",
    "FailureCategory",
    "Nomination",
    "ResourceRegime",
    "ScientificDecision",
    "Track",
    "artifact_reference_hashes",
    "endpoint_result",
    "resolve_artifact",
    "validate_artifact",
    "validate_record",
    "verify_artifact_references",
]
