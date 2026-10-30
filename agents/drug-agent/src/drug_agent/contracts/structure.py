"""Structure-based contracts: proteins, poses, simulation results."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import IdentifiedModel, Provenance, SchemaModel
from .enums import TrustLabel


class ProteinStructure(IdentifiedModel):
    """Receptor structure + preparation metadata."""

    id_prefix: str = "prot"
    accession: str | None = None
    name: str | None = None
    sequence_checksum: str | None = None
    isoform: str | None = None
    construct_label: str | None = Field(default=None, description="Construct/boundaries used.")
    mutations: list[str] = Field(default_factory=list)
    coverage: str | None = None
    pdb_id: str | None = None
    method: str | None = Field(default=None, description="X-ray/cryo-EM/model.")
    resolution: float | None = None
    quality_notes: str | None = None
    prepared: bool = False
    preparation_assumptions: list[str] = Field(default_factory=list)
    structure_ref: str | None = Field(default=None, description="Artifact ref to coordinates.")
    provenance: Provenance | None = None


class InteractionFingerprint(SchemaModel):
    residues: list[str] = Field(default_factory=list)
    interaction_types: list[str] = Field(default_factory=list)
    bits: list[int] = Field(default_factory=list)


class Pose(IdentifiedModel):
    """A docking pose for a candidate against a receptor."""

    id_prefix: str = "pose"
    candidate_id: str = ""
    receptor_id: str | None = None
    rank: int = 0
    score: float | None = Field(default=None, description="Docking score (lower = better).")
    score_unit: str = "kcal/mol"
    rmsd_lb: float | None = None
    rmsd_ub: float | None = None
    interaction_fp: InteractionFingerprint | None = None
    pose_ref: str | None = Field(default=None, description="Artifact ref to pose coordinates.")
    engine: str = "mock"
    engine_version: str = "0.0.0"
    trust_label: TrustLabel = TrustLabel.CALCULATED
    qc_flags: list[str] = Field(default_factory=list)


class SimulationResult(IdentifiedModel):
    """A molecular-dynamics / free-energy protocol result."""

    id_prefix: str = "sim"
    candidate_id: str = ""
    receptor_id: str | None = None
    protocol: str = Field(default="mock_min", description="Simulation protocol id.")
    engine: str = "mock"
    engine_version: str = "0.0.0"
    n_steps: int = 0
    observables: dict[str, float] = Field(default_factory=dict)
    stability_score: float | None = None
    converged: bool = False
    trust_label: TrustLabel = TrustLabel.CALCULATED
    qc_flags: list[str] = Field(default_factory=list)
    artifact_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
