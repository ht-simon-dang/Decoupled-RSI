"""Prediction and safety contracts."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import EvidenceRef, IdentifiedModel, Quantity, SchemaModel
from .enums import ApplicabilityDomain, Severity, TrustLabel


class Prediction(IdentifiedModel):
    """A single endpoint estimate for a candidate.

    Every prediction is self-describing: endpoint, value+unit+uncertainty,
    the model/tool version that produced it, and applicability. This directly
    satisfies acceptance criterion #10.
    """

    id_prefix: str = "pred"
    candidate_id: str = ""
    molecule_inchikey: str | None = None
    endpoint: str = Field(description="e.g. 'pIC50', 'logP', 'herg_block', 'sa_score'.")
    kind: str = Field(default="property", description="'activity'|'property'|'safety'|'admet'.")

    quantity: Quantity
    applicability: ApplicabilityDomain = ApplicabilityDomain.UNKNOWN
    applicability_score: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated: bool = False

    model_id: str = Field(description="Model/tool that produced it.")
    model_version: str = "0.0.0"
    trust_label: TrustLabel = TrustLabel.PREDICTED
    limitation: str | None = Field(
        default=None, description="Explicit statement when uncertainty is unavailable."
    )
    evidence: list[EvidenceRef] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class SafetyEndpoint(SchemaModel):
    """One toxicity/safety endpoint result."""

    endpoint: str = Field(description="e.g. 'hERG', 'AMES', 'hepatotox', 'structural_alerts'.")
    score: float | None = Field(default=None, description="Probability/severity score if numeric.")
    flag: bool | None = Field(default=None, description="Binary alert if applicable.")
    severity: Severity = Severity.NONE
    applicability: ApplicabilityDomain = ApplicabilityDomain.UNKNOWN
    calibrated: bool = False
    model_id: str = ""
    model_version: str = "0.0.0"
    detail: str | None = None
    limitation: str | None = None
    alerts: list[str] = Field(default_factory=list)


class SafetyProfile(IdentifiedModel):
    """Aggregate safety profile that preserves per-endpoint detail.

    Deliberately distinguishes "no detected signal" from "safe": ``no_signal``
    means the panel found nothing, never a safety claim.
    """

    id_prefix: str = "safety"
    candidate_id: str = ""
    endpoints: list[SafetyEndpoint] = Field(default_factory=list)
    max_severity: Severity = Severity.NONE
    n_flags: int = 0
    no_signal: bool = False
    summary: str = ""
    disclaimer: str = (
        "No detected signal is not a claim of safety. External in-vitro/in-vivo "
        "validation is required before any safety conclusion."
    )
