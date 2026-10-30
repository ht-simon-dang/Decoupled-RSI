"""Typed scientific boundaries. Sequence identity and use context are separate."""
from __future__ import annotations

import hashlib
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from Bio.Seq import Seq


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def identity(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rna(value: str) -> str:
    value = "".join(value.split()).upper().replace("T", "U")
    if set(value) - set("ACGU"):
        raise ValueError("Sequence must contain only A/C/G/U (DNA T is normalized to U)")
    return value


def translate(cds: str) -> str:
    if not cds or len(cds) % 3 or not cds.startswith("AUG") or cds[-3:] not in {"UAA", "UAG", "UGA"}:
        raise ValueError("CDS requires AUG, complete codons and exactly one terminal stop")
    protein = str(Seq(cds).translate())
    if "*" in protein[:-1]:
        raise ValueError("Internal termination codon")
    return protein[:-1]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Context(Contract):
    species: Literal["human", "mouse", "other", "unspecified"] = "human"
    target_cell: str = Field(default="unspecified", max_length=160)
    non_target_cells: list[str] = Field(default_factory=list, max_length=8)
    delivery: str = Field(default="unspecified", max_length=300)
    dose: str = Field(default="unspecified", max_length=200)
    time_window_hours: tuple[float, float] | None = None
    assay: str = Field(default="unspecified", max_length=200)

    @model_validator(mode="after")
    def ordered(self):
        if self.time_window_hours and not 0 <= self.time_window_hours[0] < self.time_window_hours[1]:
            raise ValueError("Time window must satisfy 0 <= start < end")
        return self


class MolecularConfig(Contract):
    form: Literal["linear_nonreplicating"] = "linear_nonreplicating"
    cap: Literal["Cap0", "Cap1", "Cap2", "uncapped", "unspecified"] = "unspecified"
    poly_a_length: int = Field(default=0, ge=0, le=300)
    modification: Literal["unmodified", "m1Psi", "Psi", "other", "unspecified"] = "unspecified"
    retained_elements: list[str] = Field(default_factory=list, max_length=20)


class Regions(Contract):
    utr5: str = Field(default="", max_length=3000)
    cds: str = Field(min_length=6, max_length=18003)
    utr3: str = Field(default="", max_length=5000)

    @field_validator("utr5", "cds", "utr3", mode="before")
    @classmethod
    def normalize(cls, v):
        return rna(v)

    @property
    def transcript(self):
        return self.utr5 + self.cds + self.utr3

    @property
    def boundaries(self):
        n, m = len(self.utr5), len(self.cds)
        return {"coordinate_system": "0-based half-open; poly(A) separate", "utr5": [0, n],
                "cds": [n, n + m], "utr3": [n + m, len(self.transcript)]}


class LockedSpan(Contract):
    region: Literal["utr5", "cds", "utr3"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Objective(Contract):
    metric: Literal["translation_efficiency", "ribosome_load", "rna_stability", "chemical_degradation",
                    "protein_peak", "protein_auc", "functional_duration", "target_expression",
                    "non_target_expression", "mfe", "start_accessibility", "gc_fraction", "length_nt"]
    role: Literal["primary", "secondary", "hard", "explanatory"] = "primary"
    direction: Literal["maximize", "minimize", "target"] = "maximize"
    target: float | None = None
    threshold: float | None = None
    stage: Literal["computational", "experimental"] = "experimental"

    @model_validator(mode="after")
    def target_required(self):
        if self.direction == "target" and self.target is None:
            raise ValueError("A target objective needs a target value")
        if self.role == "hard" and self.threshold is None:
            raise ValueError("A hard endpoint needs a threshold (tolerance for target objectives)")
        return self


class Budget(Contract):
    candidates: int = Field(default=8, ge=1, le=64)
    shortlist: int = Field(default=4, ge=1, le=24)
    tool_calls: int = Field(default=32, ge=1, le=256)
    seconds: int = Field(default=1800, ge=10, le=14400)
    rounds: int = Field(default=1, ge=1, le=4)
    synthesis_slots: int | None = Field(default=None, ge=0, le=96)
    experiment_slots: int | None = Field(default=None, ge=0, le=96)


class JointProposal(Contract):
    regions: Regions
    provider: str = Field(min_length=1,max_length=160)
    version: str = Field(min_length=1,max_length=160)
    request_id: str = Field(min_length=1,max_length=200)
    source: str = Field(min_length=1,max_length=1000)


class Brief(Contract):
    name: str = Field(default="mRNA design", max_length=200)
    mode: Literal["full", "cds", "utr", "diagnose", "iterate", "transfer"] = "full"
    protein: str | None = Field(default=None, max_length=6000)
    protein_version: str = Field(default="user-v1", max_length=160)
    application: str = Field(default="research", max_length=2000)
    reference: Regions | None = None
    reference_config: MolecularConfig | None = None
    parent_run: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    fixed_regions: list[Literal["utr5", "cds", "utr3"]] = Field(default_factory=list)
    locked_spans: list[LockedSpan] = Field(default_factory=list, max_length=100)
    forbidden_motifs: list[str] = Field(default_factory=list, max_length=30)
    required_motifs: list[str] = Field(default_factory=list, max_length=30)
    context: Context = Field(default_factory=Context)
    molecule: MolecularConfig = Field(default_factory=MolecularConfig)
    objectives: list[Objective] = Field(default_factory=lambda: [Objective(metric="protein_auc")], max_length=20)
    budget: Budget = Field(default_factory=Budget)
    generators: list[Literal["gemorna", "lineardesign", "synonymous", "smart5utr", "parade", "joint", "codongpt", "utrgan", "mrnautilus", "promorna", "mrna_gpt"]] = Field(default_factory=lambda: ["gemorna", "lineardesign", "smart5utr", "parade", "codongpt", "utrgan", "mrnautilus", "promorna", "mrna_gpt"])
    joint_proposals: list[JointProposal] = Field(default_factory=list,max_length=64)
    evaluators: list[Literal["vienna", "ribonn", "parade", "optimus5", "saluki", "rnadegformer", "utr_lm", "mrnabert", "degscore"]] = Field(default_factory=lambda: ["vienna", "ribonn", "parade", "optimus5", "saluki", "rnadegformer", "utr_lm", "mrnabert", "degscore"])
    utr_length: Literal["short", "medium", "long"] = "short"
    seed: int = Field(default=42, ge=0, le=2**31-1)
    temperature: Literal[1.0] = 1.0  # Author sampler exposes no configurable temperature.
    linear_lambda: float = Field(default=1.0, ge=0, le=100)
    allow_fallback: bool = True
    workflow_version: Literal["v1", "v2"] = "v1"
    assumptions: list[str] = Field(default_factory=list, max_length=40)

    @field_validator("protein", mode="before")
    @classmethod
    def normalize_protein(cls, v):
        if v is None:
            return v
        v = "".join(v.split()).upper().removesuffix("*")
        if not v or not v.startswith("M") or set(v) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("Provide a canonical protein starting with M; payload changes require a new protein version")
        return v

    @field_validator("forbidden_motifs", "required_motifs")
    @classmethod
    def motifs(cls, values):
        if any(not x or len(x) > 200 for x in values):
            raise ValueError("Motifs must contain 1..200 nt")
        return [rna(x) for x in values]

    @model_validator(mode="after")
    def consistency(self):
        if self.reference:
            p = translate(self.reference.cds)
            if self.protein and self.protein != p:
                raise ValueError("Reference CDS does not encode the locked protein")
            if self.protein is None:
                object.__setattr__(self, "protein", p)
        if self.mode in {"cds", "utr", "diagnose", "iterate", "transfer"} and not self.reference:
            raise ValueError(f"{self.mode} mode requires an explicitly partitioned reference construct")
        if (self.fixed_regions or self.locked_spans) and not self.reference:
            raise ValueError("Locked regions need a reference")
        for span in self.locked_spans:
            if span.start >= span.end or span.end > len(getattr(self.reference, span.region)):
                raise ValueError("Locked span outside reference region")
        if len({o.metric for o in self.objectives}) != len(self.objectives):
            raise ValueError("Duplicate objective metrics")
        if not self.generators:
            raise ValueError("At least one generation route is required")
        return self

    @property
    def locks(self):
        return set(self.fixed_regions) | ({"utr5", "utr3"} if self.mode == "cds" else {"cds"} if self.mode == "utr" else {"utr5", "cds", "utr3"} if self.mode == "diagnose" else set())


class Construct(Contract):
    regions: Regions
    molecule: MolecularConfig
    protein_version: str
    sources: dict = Field(default_factory=dict)
    parent_ids: list[str] = Field(default_factory=list)
    optimized_metrics: list[str] = Field(default_factory=list)
    hypothesis: str = "exploratory combination; experimental benefit unknown"

    @property
    def id(self):
        # Context, name, lineage and protein-version aliases do not change molecular identity.
        return identity({"regions": self.regions.model_dump(), "molecule": self.molecule.model_dump()})

    @property
    def sequence_id(self):
        return hashlib.sha256(self.sequence.encode()).hexdigest()

    @property
    def sequence(self):
        return self.regions.transcript + "A" * self.molecule.poly_a_length

    def card(self):
        return {**self.model_dump(), "id": self.id, "sequence_id": self.sequence_id,
                "sequence": self.sequence, "boundaries": self.regions.boundaries,
                "poly_a_boundary": [len(self.regions.transcript), len(self.sequence)]}


class Measurement(Contract):
    metric: str
    status: Literal["ok", "unavailable", "out_of_scope", "error", "not_evaluated"]
    value: float | None = None
    unit: str
    source: str
    version: str
    evidence: Literal["calculated", "model_prediction", "mechanistic_hypothesis", "measured_this_context", "not_evaluated"]
    scope: str
    reason: str | None = None
    optimized_by_generator: bool = False
    independent: bool = True
    uncertainty: dict | None = None
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def no_fake_zero(self):
        if (self.status == "ok") != (self.value is not None):
            raise ValueError("Only successful measurements may have a numeric value")
        return self
