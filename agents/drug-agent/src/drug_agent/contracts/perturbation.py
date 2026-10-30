"""Single-cell / organoid perturbation contracts."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import IdentifiedModel, SchemaModel
from .enums import ApplicabilityDomain, TrustLabel


class QCSummary(SchemaModel):
    n_cells: int = 0
    n_genes: int = 0
    n_donors: int | None = None
    median_counts: float | None = None
    pct_mito: float | None = None
    flags: list[str] = Field(default_factory=list)


class DifferentialGene(SchemaModel):
    gene: str
    log2fc: float
    score: float
    pval: float | None = None
    direction: str = Field(default="up", description="'up'|'down'.")


class PerturbationSignature(IdentifiedModel):
    """A perturbation's effect signature and restoration metric."""

    id_prefix: str = "perturb"
    candidate_id: str | None = None
    dataset_id: str = ""
    cell_context: str | None = None
    qc: QCSummary = Field(default_factory=QCSummary)

    top_genes: list[DifferentialGene] = Field(default_factory=list)
    signature_score: float | None = Field(
        default=None, description="Signature-scoring statistic vs disease axis."
    )
    restoration_score: float | None = Field(
        default=None, description="Desired-state restoration (higher = more restored)."
    )
    context_applicability: ApplicabilityDomain = ApplicabilityDomain.UNKNOWN

    method: str = "signature_reversal_baseline"
    method_version: str = "0.0.0"
    split_strategy: str | None = Field(default=None, description="e.g. 'donor', 'study'.")
    trust_label: TrustLabel = TrustLabel.PREDICTED
    limitation: str = "In-silico signature only; not a phenotypic efficacy claim."
    metadata: dict[str, Any] = Field(default_factory=dict)
