"""Ranking-plane contracts: scorecards, portfolio, sensitivity."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import IdentifiedModel, SchemaModel
from .enums import ApplicabilityDomain, DecisionOutcome


class EndpointScore(SchemaModel):
    """Per-endpoint entry on a candidate scorecard."""

    endpoint: str
    raw_value: float | None = None
    normalized: float | None = Field(default=None, description="0..1, higher = better.")
    unit: str | None = None
    direction: str = "maximize"
    uncertainty: float | None = None
    applicability: ApplicabilityDomain = ApplicabilityDomain.UNKNOWN
    model_id: str | None = None
    model_version: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    missing: bool = False


class Scorecard(IdentifiedModel):
    """A full, auditable scorecard for one candidate (acceptance criterion #10, §10)."""

    id_prefix: str = "score"
    run_id: str = ""
    candidate_id: str = ""
    inchikey: str | None = None
    canonical_smiles: str | None = None

    endpoints: list[EndpointScore] = Field(default_factory=list)
    hard_constraints_passed: bool = True
    hard_constraint_outcomes: dict[str, bool] = Field(default_factory=dict)

    utility: float | None = Field(default=None, description="Optional display-only scalar utility.")
    evidence_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    conflict_score: float | None = Field(default=None, ge=0.0, le=1.0)
    out_of_domain_penalty: float = 0.0

    pareto_optimal: bool = False
    pareto_rank: int | None = None
    diversity_cluster: int | None = None

    decision: DecisionOutcome = DecisionOutcome.DEFER
    decision_reasons: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)


class PortfolioMember(SchemaModel):
    candidate_id: str
    inchikey: str | None = None
    canonical_smiles: str | None = None
    role: str = Field(default="shortlist", description="'shortlist'|'reserve'.")
    cluster: int | None = None
    utility: float | None = None
    pareto_rank: int | None = None


class Portfolio(IdentifiedModel):
    """The final diverse Pareto portfolio for a run."""

    id_prefix: str = "portfolio"
    run_id: str = ""
    shortlist: list[PortfolioMember] = Field(default_factory=list)
    reserve: list[PortfolioMember] = Field(default_factory=list)
    n_candidates_considered: int = 0
    n_pareto: int = 0
    n_clusters: int = 0
    hypervolume: float | None = None
    diversity: float | None = None
    evidence_coverage: float | None = None
    objectives: list[dict[str, Any]] = Field(default_factory=list)


class SensitivityRow(SchemaModel):
    scenario: str
    description: str
    shortlist_ids: list[str] = Field(default_factory=list)
    jaccard_vs_base: float | None = None
    changed: bool = False


class SensitivityReport(IdentifiedModel):
    id_prefix: str = "sens"
    run_id: str = ""
    base_shortlist: list[str] = Field(default_factory=list)
    rows: list[SensitivityRow] = Field(default_factory=list)
