"""Synthesizability contracts: routes, steps, plans."""

from __future__ import annotations

from pydantic import Field

from .base import IdentifiedModel, SchemaModel
from .enums import Severity, TrustLabel


class BuildingBlock(SchemaModel):
    smiles: str
    available: bool = False
    catalog: str | None = None
    price_class: str | None = Field(default=None, description="'cheap'|'moderate'|'costly'|'unknown'.")


class RouteStep(SchemaModel):
    order: int
    reaction_type: str | None = None
    description: str
    reactants: list[str] = Field(default_factory=list)
    product: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk: Severity = Severity.LOW


class SynthesisRoute(IdentifiedModel):
    id_prefix: str = "route"
    target_smiles: str = ""
    n_steps: int = 0
    steps: list[RouteStep] = Field(default_factory=list)
    building_blocks: list[BuildingBlock] = Field(default_factory=list)
    feasible: bool | None = None
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    engine: str = "mock"
    engine_version: str = "0.0.0"
    trust_label: TrustLabel = TrustLabel.PREDICTED


class SynthesisPlan(IdentifiedModel):
    """Synthesizability assessment for a candidate."""

    id_prefix: str = "synth"
    candidate_id: str = ""
    sa_score: float | None = Field(default=None, description="Synthetic accessibility (1 easy - 10 hard).")
    sa_label: str | None = None
    routes: list[SynthesisRoute] = Field(default_factory=list)
    best_route_id: str | None = None
    risk_factors: list[str] = Field(default_factory=list)
    building_block_availability: float | None = Field(default=None, ge=0.0, le=1.0)
    trust_label: TrustLabel = TrustLabel.PREDICTED
    limitation: str = "Heuristic estimate; no laboratory validation implied."
