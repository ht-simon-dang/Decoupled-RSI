"""Evidence-plane contracts: artifacts, evidence items, decisions."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import EvidenceRef, IdentifiedModel, Provenance
from .enums import DecisionOutcome, TrustLabel


class ArtifactRef(IdentifiedModel):
    """Immutable, content-addressed reference to a stored artifact."""

    id_prefix: str = "artifact"
    kind: str = Field(description="e.g. 'parquet', 'json', 'sdf', 'png', 'html'.")
    uri: str = Field(description="Backend URI, e.g. 'cas://sha256/<hash>'.")
    content_hash: str = ""
    size_bytes: int = 0
    media_type: str | None = None
    produced_by: str | None = Field(default=None, description="Capability/task that produced it.")
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(IdentifiedModel):
    """A single, source-attributed unit of evidence with a trust label."""

    id_prefix: str = "ev"
    program_id: str | None = None
    run_id: str | None = None
    kind: str = Field(description="e.g. 'literature', 'assay', 'structure', 'target_disease'.")
    statement: str = Field(description="What this evidence asserts (plain text).")
    trust_label: TrustLabel = TrustLabel.CURATED
    provenance: Provenance
    subject: str | None = Field(default=None, description="Entity the evidence is about.")
    value: dict[str, Any] = Field(default_factory=dict, description="Structured payload.")
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    related: list[EvidenceRef] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None


class DecisionRecord(IdentifiedModel):
    """An auditable decision (promote/reject/defer) with reasons and inputs."""

    id_prefix: str = "dec"
    run_id: str = ""
    subject_id: str = Field(description="Candidate/entity the decision is about.")
    outcome: DecisionOutcome
    rationale: str
    reasons: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    decided_by: str = Field(default="system", description="Agent/human that decided.")
    policy_version: str | None = None
