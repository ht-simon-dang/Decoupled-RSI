"""Chemistry contracts: Molecule, Candidate, AssayRecord."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import IdentifiedModel, Provenance, Quantity
from .enums import Qualifier, TrustLabel


class Molecule(IdentifiedModel):
    """A standardized chemical identity with descriptors and provenance.

    ``input_smiles`` is preserved verbatim; ``canonical_smiles`` /
    ``inchikey`` are the standardized identity used for de-duplication.
    """

    id_prefix: str = "mol"
    input_smiles: str = Field(description="Verbatim input representation.")
    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    formula: str | None = None

    standardized: bool = False
    standardizer_version: str | None = None
    sanitization_ok: bool = True
    has_stereo: bool = False
    stereo_defined: bool = False
    is_salt: bool = False
    is_mixture: bool = False
    parent_inchikey: str | None = Field(default=None, description="Desalted parent identity.")

    descriptors: dict[str, float] = Field(default_factory=dict)
    fingerprint_ref: str | None = Field(default=None, description="Artifact ref to fingerprint.")
    structural_alerts: list[str] = Field(default_factory=list)

    provenance: Provenance | None = None
    lineage: list[str] = Field(
        default_factory=list, description="Seed/parent molecule IDs this derives from."
    )
    source: str | None = None
    errors: list[str] = Field(default_factory=list)


class AssayRecord(IdentifiedModel):
    """A single bioactivity measurement, preserving original value/unit/qualifier."""

    id_prefix: str = "assay"
    molecule_inchikey: str | None = None
    molecule_smiles: str | None = None
    target_id: str | None = None
    target_name: str | None = None

    endpoint: str = Field(description="e.g. 'IC50', 'Ki', 'pIC50'.")
    measurement: Quantity
    assay_type: str | None = None
    assay_description: str | None = None
    relation: Qualifier = Qualifier.EXACT
    year: int | None = None
    series_id: str | None = Field(default=None, description="Med-chem series for group splits.")

    provenance: Provenance
    trust_label: TrustLabel = TrustLabel.OBSERVED


class Candidate(IdentifiedModel):
    """A molecule under evaluation, accumulating predictions and a scorecard.

    The ``scorecard`` and ``decision`` fields are populated by the ranking
    plane; predictions are attached by specialist agents.
    """

    id_prefix: str = "cand"
    run_id: str = ""
    molecule_id: str = ""
    inchikey: str | None = None
    canonical_smiles: str | None = None
    origin: str = Field(default="sourced", description="'seed' | 'analogue' | 'library'.")
    lineage: list[str] = Field(default_factory=list)

    passed_hard_filters: bool | None = None
    filter_failures: list[str] = Field(default_factory=list)

    prediction_ids: list[str] = Field(default_factory=list)
    scaffold: str | None = None
    diversity_cluster: int | None = None

    # Populated by ranking plane; kept generic to avoid a circular import.
    scorecard: dict[str, Any] | None = None
    decision: str | None = None
