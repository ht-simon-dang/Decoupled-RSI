"""Strict scientific contracts. Missing evidence never becomes a zero score."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sequence = Annotated[str, Field(min_length=20, max_length=3000, pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$")]
Identifier = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")]
Status = Literal["hypothesis", "predicted", "measured", "unknown", "technical_failure", "censored"]
Generator = Literal[
    "library",
    "rfantibody",
    "boltzgen",
    "iggm",
    "igdesign",
    "antifold",
    "p_iggen",
    "lichen",
    "sapiens",
    "humatch",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, validate_assignment=True)


class Evidence(Contract):
    endpoint: str = Field(min_length=1, max_length=100)
    status: Status = "unknown"
    value: float | str | None = None
    unit: str = "not_applicable"
    source: str = Field(min_length=1, max_length=1000)
    version: str = Field(min_length=1, max_length=200)
    conditions: dict[str, str | float | bool] = Field(default_factory=dict)
    uncertainty: str = "not calibrated"
    limitation: str = "Research evidence; no clinical conclusion"

    @model_validator(mode="after")
    def present(self):
        if self.status in {"unknown", "technical_failure"} and self.value is not None:
            raise ValueError("Unknown/failed evidence must have a null value")
        if self.status not in {"unknown", "technical_failure"} and self.value is None:
            raise ValueError("Present evidence requires its original value or censoring bound")
        return self


class Gate(Contract):
    status: Literal["supported", "hypothesis", "unknown", "contradicted"] = "unknown"
    rationale: str = "Evidence required"
    sources: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def sourced(self):
        if self.status in {"supported", "contradicted"} and not self.sources:
            raise ValueError("Supported/contradicted statements require sources")
        return self


class NumberedResidue(Contract):
    index: int = Field(ge=1, le=3000)
    label: str = Field(pattern=r"^[0-9]+[A-Z]*$")
    region: Literal["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    structure_residue: str | None = None


class Chain(Contract):
    role: Literal["VH", "VL", "VHH"]
    sequence: Sequence
    scheme: Literal["IMGT", "AHo", "Kabat"] = "IMGT"
    numbering: list[NumberedResidue] = Field(default_factory=list, max_length=3000)
    numbering_source: str | None = None

    @model_validator(mode="after")
    def reversible(self):
        if self.numbering:
            if sorted(r.index for r in self.numbering) != list(range(1, len(self.sequence) + 1)):
                raise ValueError("Numbering must cover each sequence position exactly once")
            if len({r.label for r in self.numbering}) != len(self.numbering) or not self.numbering_source:
                raise ValueError("Numbering requires unique labels and provenance")
            structural = [r.structure_residue for r in self.numbering if r.structure_residue]
            if len(structural) != len(set(structural)):
                raise ValueError("Structure residue mapping must be reversible")
        return self


class Product(Contract):
    format: Literal["IgG", "Fab", "scFv", "VHH", "fusion", "bispecific", "ADC", "conditional"] = "IgG"
    valency: int = Field(default=2, ge=1, le=8)
    fc: str = "unresolved"
    fc_requirement: Literal["retain", "reduce", "enhance", "none", "unresolved"] = "unresolved"
    chain_organization: str = "paired VH/VL; constant regions not supplied"
    linker: str | None = None
    fusion_modules: list[str] = Field(default_factory=list, max_length=10)
    conjugation_version: str | None = None
    sequence_scope: Literal["variable_domains"] = "variable_domains"


class Binder(Contract):
    name: str = Field(default="supplied binder", max_length=200)
    chains: list[Chain] = Field(min_length=1, max_length=2)
    pairing_source: Literal["natural_observation", "experimental", "model_proposed", "user_supplied"]
    source: str = Field(min_length=1, max_length=1000)
    product: Product = Field(default_factory=Product)
    structure_pdb: str | None = Field(default=None, max_length=5_000_000)

    @model_validator(mode="after")
    def paired(self):
        roles = sorted(c.role for c in self.chains)
        if roles not in [["VH", "VL"], ["VHH"]]:
            raise ValueError("Supply a paired VH/VL or an explicitly identified VHH; isolated VH is not VHH")
        if self.product.format == "VHH" and roles != ["VHH"]:
            raise ValueError("VHH format requires a VHH chain")
        if roles == ["VHH"] and self.product.format in {"IgG", "Fab", "scFv"}:
            raise ValueError("Single-domain constructs require a VHH or explicitly defined fusion format")
        return self


class AntigenState(Contract):
    id: Identifier
    entity: str = Field(min_length=1, max_length=200)
    species: str = "human"
    isoform: str = "unresolved"
    state: str = "unresolved"
    role: Literal["positive", "negative"] = "positive"
    sequence: Sequence | None = None
    structure_pdb: str | None = Field(default=None, max_length=5_000_000)
    structure_chain: str = Field(default="A", pattern=r"^[A-Za-z0-9]$")
    presentation: str = "unresolved"
    glycan_context: Literal["known", "unknown", "not_applicable"] = "unknown"
    membrane_context: Literal["known", "unknown", "not_applicable"] = "unknown"
    full_context: bool = False
    crop_to_full: dict[str, str] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list, max_length=30)


class Epitope(Contract):
    id: Identifier = "primary"
    state_id: Identifier
    residues: list[int] = Field(default_factory=list, max_length=500)
    forbidden_residues: list[int] = Field(default_factory=list, max_length=500)
    rationale: str = "Functional relationship unresolved"
    accessibility: Gate = Field(default_factory=Gate)
    evidence: list[Evidence] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def positions(self):
        if any(i < 1 for i in self.residues + self.forbidden_residues):
            raise ValueError("Epitope residues use 1-based antigen sequence positions")
        if set(self.residues) & set(self.forbidden_residues):
            raise ValueError("Required and forbidden epitope contacts overlap")
        return self


class Constraints(Contract):
    scheme: Literal["IMGT", "AHo", "Kabat"] = "IMGT"
    editable: dict[str, list[str]] = Field(default_factory=dict)
    protected: dict[str, list[str]] = Field(default_factory=dict)
    max_mutations: int = Field(default=12, ge=0, le=200)
    preserve: list[str] = Field(default_factory=list, max_length=30)
    allow_length_change: bool = False

    @model_validator(mode="after")
    def disjoint(self):
        for role in set(self.editable) | set(self.protected):
            if role not in {"VH", "VL", "VHH"}:
                raise ValueError("Constraint keys must be chain roles")
            if set(self.editable.get(role, [])) & set(self.protected.get(role, [])):
                raise ValueError("Editable and protected positions overlap")
        return self


class Budget(Contract):
    candidates: int = Field(default=4, ge=1, le=32)
    shortlist: int = Field(default=4, ge=1, le=16)
    tool_calls: int = Field(default=64, ge=1, le=512)
    wall_seconds: int = Field(default=3600, ge=1, le=14400)
    experiment_slots: int = Field(default=4, ge=0, le=32)


class EvaluationPolicy(Contract):
    min_iptm: float = Field(default=0.6, ge=0, le=1)
    min_plddt: float = Field(default=0.7, ge=0, le=1)
    min_epitope_coverage: float = Field(default=0.5, ge=0, le=1)
    max_ca_clashes: int = Field(default=0, ge=0, le=100)
    family_identity: float = Field(default=0.9, ge=0.5, le=1)
    calibration: str = "Uncalibrated computational triage; not binding probability"


class Brief(Contract):
    name: str = Field(default="Antibody research", min_length=1, max_length=200)
    entry: Literal["disease", "target", "sequence", "complex", "feedback"] = "disease"
    route: Literal["research", "de_novo", "optimize", "pairing", "humanize", "library"] = "research"
    disease: str = Field(default="unspecified", max_length=2000)
    patient_subgroup: str = "unresolved"
    tissue: str = "unresolved"
    target_cell: str = "unresolved"
    mechanism: str = "unresolved"
    direction: Literal["block", "agonize", "clear", "bridge", "deliver", "unresolved"] = "unresolved"
    direction_gate: Gate = Field(default_factory=Gate)
    access_gate: Gate = Field(default_factory=Gate)
    desired_endpoints: dict[str, str] = Field(default_factory=dict)
    delivery_requirements: str = "unresolved"
    stop_conditions: list[str] = Field(
        default_factory=lambda: ["budget exhausted", "no independently qualified candidates"]
    )
    states: list[AntigenState] = Field(default_factory=list, max_length=16)
    epitopes: list[Epitope] = Field(default_factory=list, max_length=16)
    reference: Binder | None = None
    library: list[Binder] = Field(default_factory=list, max_length=32)
    product: Product = Field(default_factory=Product)
    constraints: Constraints = Field(default_factory=Constraints)
    generators: list[Generator] = Field(default_factory=lambda: ["rfantibody"], min_length=1, max_length=5)
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    budget: Budget = Field(default_factory=Budget)
    evaluation_policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    parent_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def coherent(self):
        states = {s.id: s for s in self.states}
        if len(states) != len(self.states) or len(set(self.generators)) != len(self.generators):
            raise ValueError("State ids and generators must be unique")
        if len({e.id for e in self.epitopes}) != len(self.epitopes):
            raise ValueError("Epitope ids must be unique")
        for e in self.epitopes:
            if e.state_id not in states:
                raise ValueError("Epitope references an unknown antigen state")
            seq = states[e.state_id].sequence
            if seq and any(i > len(seq) for i in e.residues + e.forbidden_residues):
                raise ValueError("Epitope position outside the antigen sequence")
        if self.budget.shortlist > self.budget.candidates:
            raise ValueError("Shortlist cannot exceed candidate budget")
        if len(self.library) > self.budget.candidates:
            raise ValueError("Supplied library exceeds candidate budget")
        if self.reference:
            chains = {c.role: c for c in self.reference.chains}
            for role in set(self.constraints.editable) | set(self.constraints.protected):
                if role not in chains:
                    raise ValueError("Constraints refer to an absent reference chain")
                chain = chains[role]
                if chain.numbering:
                    labels = {r.label for r in chain.numbering}
                    requested = self.constraints.editable.get(role, []) + self.constraints.protected.get(
                        role, []
                    )
                    if chain.scheme != self.constraints.scheme or not set(requested) <= labels:
                        raise ValueError("Constraint numbering does not match the reference mapping")
        return self


class Candidate(Contract):
    id: str = Field(pattern=r"^[a-f0-9]{64}$")
    variable_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    binder: Binder
    origin: str
    provenance: dict
    parent_id: str | None = None
    mutations: list[str] = Field(default_factory=list)
    family: str = "unassigned"


class ToolResult(Contract):
    tool: str
    status: Literal["ok", "unavailable", "error", "timeout"]
    data: dict = Field(default_factory=dict)
    error: str | None = None
    provenance: dict = Field(default_factory=dict)


class Observation(Contract):
    id: Identifier
    candidate_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    endpoint: Literal[
        "expression",
        "binding",
        "epitope",
        "pose",
        "mechanism",
        "developability",
        "off_target",
        "on_target_off_tissue",
    ]
    status: Literal["measured", "censored", "technical_failure"]
    value: float | str | None = None
    unit: str = Field(min_length=1, max_length=100)
    censoring: Literal["none", "below", "above", "interval"] = "none"
    upper_bound: float | None = None
    outcome: Literal["positive", "negative", "undetermined"] = "undetermined"
    conditions: dict[str, str | float | bool] = Field(min_length=1)
    batch: str = Field(min_length=1)
    glycoform: str = "unknown"
    replicate: str = Field(min_length=1)
    control: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def measurement(self):
        if self.status == "technical_failure":
            if self.value is not None or self.outcome != "undetermined" or self.censoring != "none":
                raise ValueError("Technical failure is not a negative measurement")
        elif self.value is None:
            raise ValueError("Measurements/censored records require a raw value or bound")
        if (self.status == "censored") != (self.censoring != "none"):
            raise ValueError("Censored measurements require an explicit censoring direction")
        if self.censoring == "interval":
            if (
                not isinstance(self.value, (float, int))
                or self.upper_bound is None
                or self.value > self.upper_bound
            ):
                raise ValueError("Interval censoring needs ordered numerical bounds")
        return self


class Unavailable(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass
