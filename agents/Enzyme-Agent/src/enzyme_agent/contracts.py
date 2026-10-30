from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

AA = "ACDEFGHIKLMNPQRSTVWY"
Sequence = Annotated[str, Field(min_length=20, max_length=1500, pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$")]
Status = Literal["measured", "predicted", "inferred", "not_assessed", "not_applicable", "technical_failure"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, validate_assignment=True)


class Evidence(Contract):
    endpoint: str = Field(min_length=1, max_length=200)
    status: Status
    value: float | str | None = None
    unit: str = "not_applicable"
    scale: Literal["linear", "log10", "ln", "categorical"] = "linear"
    conditions: dict[str, str | float | bool] = Field(default_factory=dict)
    source: str = Field(min_length=1, max_length=1000)
    version: str = Field(min_length=1, max_length=200)
    uncertainty: str = "not calibrated"
    applicability: str = "project-specific; not clinical evidence"
    substrate_id: str | None = None
    environment_id: str = "target"

    @model_validator(mode="after")
    def consistent(self):
        if self.status in {"not_assessed", "not_applicable", "technical_failure"} and self.value is not None:
            raise ValueError("Missing/inapplicable/failed evidence cannot contain a scientific value")
        if self.status in {"measured", "predicted", "inferred"} and self.value is None:
            raise ValueError("Present evidence requires a value")
        return self


class Gate(Contract):
    status: Literal["supported", "unknown", "contradicted"] = "unknown"
    rationale: str = "Evidence required"
    sources: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def cited(self):
        if self.status != "unknown" and not self.sources:
            raise ValueError("A supported/contradicted gate requires sources")
        return self


class Substrate(Contract):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=300)
    kind: Literal["small_molecule", "protein", "glycan", "nucleic_acid", "material"]
    role: Literal["target", "off_target", "protect", "competitor", "harmful_product"]
    chemical_id: str | None = None
    smiles: str | None = Field(default=None, max_length=4000)


class Environment(Contract):
    id: str = "target"
    species: str = "human"
    tissue: str = "unknown"
    compartment: str = "unknown"
    ph: float | None = Field(default=None, ge=0, le=14)
    temperature_c: float | None = Field(default=None, ge=-20, le=120)
    substrate_range_molar: list[float] | None = None
    cofactors: list[str] = Field(default_factory=list)
    dependencies: dict[str, Literal["met", "external", "unknown"]] = Field(default_factory=dict)
    effective_lifetime_hours: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def ranges(self):
        r = self.substrate_range_molar
        if r is not None and (len(r) != 2 or not 0 <= r[0] <= r[1]):
            raise ValueError("Substrate range must be [low, high] mol/L")
        return self


class Reaction(Contract):
    name: str = "unresolved"
    ec: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+\.\d+$")
    rhea: str | None = Field(default=None, pattern=r"^RHEA:\d+$")
    products: list[str] = Field(default_factory=list)
    co_reactants: list[str] = Field(default_factory=list)
    direction: str = "unresolved"
    product_fate: str = "unknown"
    mechanism: str = "unknown"
    mechanism_sources: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)


class Product(Contract):
    name: str = "unmodified research protein"
    form: Literal["recombinant", "fusion", "local_delivery", "mrna", "undecided"] = "undecided"
    precursor_sequence: Sequence | None = None
    mature_start: int | None = Field(default=None, ge=1, le=1500)
    mature_end: int | None = Field(default=None, ge=1, le=1500)
    processing: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    oligomer: str = "unknown"
    targeting: str = "unknown"
    expression_host: str = "unknown"
    auxiliary_components: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def boundaries(self):
        if (self.mature_start is None) != (self.mature_end is None):
            raise ValueError("Both mature boundaries are required")
        if self.mature_start is not None:
            if not self.precursor_sequence or not self.mature_start <= self.mature_end <= len(self.precursor_sequence):
                raise ValueError("Mature boundaries must lie within the precursor")
        return self


class Reference(Contract):
    name: str = Field(min_length=1, max_length=200)
    sequence: Sequence
    accession: str | None = None
    sources: list[str] = Field(default_factory=list)
    pdb: str | None = Field(default=None, max_length=4_000_000)
    chain: str = Field(default="A", pattern=r"^[A-Za-z0-9]$")
    protected_positions: list[int] = Field(default_factory=list, max_length=1500)
    catalytic_positions: list[int] = Field(default_factory=list, max_length=100)
    protection_sources: list[str] = Field(default_factory=list)
    product: Product = Field(default_factory=Product)

    @model_validator(mode="after")
    def positions(self):
        for p in self.protected_positions + self.catalytic_positions:
            if not 1 <= p <= len(self.sequence):
                raise ValueError("Positions are 1-based mature sequence indices")
        if self.product.mature_start is not None:
            s = self.product.precursor_sequence[self.product.mature_start - 1:self.product.mature_end]
            if s != self.sequence:
                raise ValueError("Mature sequence differs from the specified precursor segment")
        return self


class Budget(Contract):
    candidates: int = Field(default=4, ge=1, le=32)
    shortlist: int = Field(default=4, ge=1, le=16)
    # Retained for historical briefs. All non-rejected candidates are evaluated;
    # this legacy field can no longer disable or cap independent folding.
    fold_candidates: int = Field(default=33, ge=0, le=33)
    tool_calls: int = Field(default=104, ge=1, le=128)
    wall_seconds: int = Field(default=3600, ge=30, le=14400)
    experiment_slots: int = Field(default=4, ge=0, le=32)


class EvaluationPolicy(Contract):
    """Predeclared computational filters, not calibrated functional thresholds."""
    min_plddt: float = Field(default=0.70, ge=0, le=1)
    min_ptm: float = Field(default=0.50, ge=0, le=1)
    max_backbone_rmsd: float = Field(default=3.0, gt=0, le=50)
    max_catalytic_ca_rmsd: float = Field(default=2.0, gt=0, le=20)
    max_catalytic_sidechain_rmsd: float = Field(default=3.0, gt=0, le=20)
    max_single_mutation_ddg: float = Field(default=2.0, ge=-20, le=20)
    max_nonlocal_clashes: int = Field(default=0, ge=0, le=100)


class Criterion(Contract):
    endpoint: str
    layer: Literal["expression", "solubility", "processing", "artificial_substrate", "disease_substrate", "localization", "disease_system"]
    substrate_id: str
    environment_id: str
    conditions: dict[str, str | float | bool]
    unit: str
    threshold: float
    direction: Literal["ge", "le"]
    source: str = Field(min_length=1)


class Brief(Contract):
    name: str = Field(default="Enzyme research", min_length=1, max_length=200)
    entry: Literal["disease", "enzyme", "reaction"] = "enzyme"
    disease: str = "unspecified"
    subtype: str = "unspecified"
    endpoint: str = "unresolved"
    falsifier: str = "unresolved"
    template: Literal["fabry", "pku", "cf_dnase"] | None = None
    route: Literal["conservative", "family_expansion", "de_novo", "product", "research"] = "conservative"
    generators: list[Literal["ligandmpnn", "proteinmpnn", "esm3", "zymctrl", "simple", "library", "natural", "rfdiffusion3", "rfdiffusion2"]] = Field(default_factory=lambda: ["esm3"], min_length=1, max_length=4)
    reference: Reference | None = None
    reaction: Reaction = Field(default_factory=Reaction)
    panel: list[Substrate] = Field(default_factory=list, max_length=32)
    environments: list[Environment] = Field(default_factory=lambda: [Environment()], min_length=1, max_length=8)
    direction_gate: Gate = Field(default_factory=Gate)
    product_gate: Gate = Field(default_factory=Gate)
    access_gate: Gate = Field(default_factory=Gate)
    evidence: list[Evidence] = Field(default_factory=list, max_length=200)
    qualification_criteria: list[Criterion] = Field(default_factory=list, max_length=32)
    library: list[Sequence] = Field(default_factory=list, max_length=32)
    design_positions: list[int] = Field(default_factory=list, max_length=128)
    bottleneck: Literal["catalysis", "selectivity", "stability", "expression", "processing", "localization", "mechanism"] = "stability"
    design_intent: str = "Constrained engineering; test changes independently"
    budget: Budget = Field(default_factory=Budget)
    evaluation_policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    temperature: float = Field(default=0.7, gt=0, le=2)
    fold_steps: int = Field(default=50, ge=5, le=200)
    fold_loops: int = Field(default=3, ge=1, le=5)
    rfd3_steps: int = Field(default=200, ge=20, le=400)
    rfd3_ligands: list[Annotated[str,Field(pattern=r'^[A-Z0-9]{1,3}$')]] = Field(default_factory=list,max_length=16)
    max_mutations: int = Field(default=16, ge=1, le=128)
    natural_search: bool = False
    parent_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.generators)) != len(self.generators):
            raise ValueError("Duplicate generators")
        if len({s.id for s in self.panel}) != len(self.panel):
            raise ValueError("Substrate ids must be unique")
        if len({e.id for e in self.environments}) != len(self.environments):
            raise ValueError("Environment ids must be unique")
        if self.design_positions:
            if not self.reference:
                raise ValueError("Design positions require a reference")
            protected = set(self.reference.protected_positions + self.reference.catalytic_positions)
            if protected.intersection(self.design_positions):
                raise ValueError("Design positions overlap protected/catalytic residues")
            if any(not 1 <= p <= len(self.reference.sequence) for p in self.design_positions):
                raise ValueError("Design position outside mature sequence")
        if "library" in self.generators and not self.library and not self.reference and self.route != "research":
            raise ValueError("Library mode requires supplied sequences/reference")
        if "zymctrl" in self.generators and self.route not in {"family_expansion", "research"}:
            raise ValueError("ZymCTRL is a family expansion generator")
        return self


class Candidate(Contract):
    id: str
    sequence: Sequence
    parent_id: str | None
    product: Product
    origin: str
    provenance: dict
    design_intent: str
    mutations: list[str] = Field(default_factory=list)
    state: Literal["generated", "exploratory", "computational_priority", "nominated", "biochemical_validated", "disease_system_validated", "rejected"] = "generated"
    evidence: list[Evidence] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    pdb: str | None = None
    evaluation: dict = Field(default_factory=dict)


class Observation(Contract):
    candidate_id: str = Field(min_length=1, max_length=80)
    product_id: str = Field(min_length=1, max_length=80)
    endpoint: str = Field(min_length=1, max_length=200)
    layer: Literal["expression", "solubility", "processing", "artificial_substrate", "disease_substrate", "localization", "disease_system"]
    status: Literal["measured", "technical_failure"] = "measured"
    value: float | None = None
    unit: str = Field(min_length=1, max_length=100)
    conditions: dict[str, str | float | bool]
    substrate_id: str
    environment_id: str
    expression_host: str
    processing_state: str
    batch: str
    replicate: str
    source: str = Field(min_length=1,max_length=1000)
    version: str = Field(min_length=1,max_length=200)
    reference_control: str
    detection: Literal["quantified", "below_limit", "above_limit", "not_applicable"] = "quantified"
    detection_limit: float | None = None
    qualification_threshold: float | None = None
    threshold_direction: Literal["ge", "le"] = "ge"
    threshold_source: str | None = None
    failure_category: Literal["expression", "solubility", "processing", "catalysis", "selectivity", "localization", "causal_hypothesis", "technical"] | None = None

    @model_validator(mode="after")
    def valid_result(self):
        if self.status == "technical_failure" and (self.value is not None or self.qualification_threshold is not None):
            raise ValueError("Technical failure cannot be an activity value or qualification")
        if self.status=='technical_failure' and self.detection in {'below_limit','above_limit'}:
            raise ValueError('Technical failure cannot be a censored biological measurement')
        if self.status=='measured' and self.detection=='not_applicable':
            raise ValueError('Measured results must be quantified or explicitly censored')
        if self.status == "measured" and self.detection == "quantified" and self.value is None:
            raise ValueError("Quantified measurement requires a value")
        if self.detection in {"below_limit", "above_limit"} and (self.detection_limit is None or self.value is not None):
            raise ValueError("Censored results require a detection limit, not an exact value")
        if self.qualification_threshold is not None and not self.threshold_source:
            raise ValueError("Qualification thresholds require a predeclared source")
        if not self.conditions or not self.source or not self.batch or not self.replicate or not self.reference_control:
            raise ValueError("Measurement context, source, batch, replicate and control are required")
        return self


class ToolResult(Contract):
    tool: str
    status: Literal["ok", "unavailable", "error", "not_applicable"]
    data: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class Unavailable(RuntimeError):
    pass
