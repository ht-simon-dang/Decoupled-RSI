"""Experiment-proposal contract — the platform's forward-looking output."""

from __future__ import annotations

from pydantic import Field

from .base import EvidenceRef, IdentifiedModel
from .enums import TrustLabel


class ExperimentProposal(IdentifiedModel):
    """A proposed next experiment to reduce uncertainty on a candidate."""

    id_prefix: str = "exp"
    run_id: str = ""
    candidate_id: str | None = None
    title: str = ""
    hypothesis: str = ""
    assay_type: str = Field(description="e.g. 'biochemical_IC50', 'hERG_patch', 'ADME_microsomal'.")
    rationale: str = ""
    expected_information_gain: str | None = None
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    est_cost_class: str | None = None
    reduces_uncertainty_on: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    trust_label: TrustLabel = TrustLabel.PROPOSED
