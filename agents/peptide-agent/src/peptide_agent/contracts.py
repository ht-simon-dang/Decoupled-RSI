from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

AA = "ACDEFGHIKLMNPQRSTVWY"
Sequence = Annotated[str, StringConstraints(pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$", min_length=2, max_length=80)]
Target = Annotated[str, StringConstraints(pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$", min_length=2, max_length=900)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Generate(Contract):
    target_sequence: Target
    length: int = Field(default=15, ge=5, le=50)
    count: int = Field(default=8, ge=1, le=64)
    top_k: int = Field(default=3, ge=1, le=20)
    seed: int = Field(default=1729, ge=0, le=2**32 - 1)


class Analyze(Contract):
    sequences: list[Sequence] = Field(min_length=1, max_length=128)
    ph: float = Field(default=7.4, ge=0, le=14)
    topology: Literal["linear", "head_to_tail"] = "linear"


class BoltzGenerate(Contract):
    target_sequence: Target | None = None
    target_pdb: str | None = Field(default=None, max_length=2_000_000)
    target_chain: str = Field(default="A", pattern=r"^[A-Za-z0-9]$")
    length: int = Field(default=12, ge=5, le=50)
    count: int = Field(default=2, ge=1, le=8)
    seed: int = Field(default=1729, ge=0, le=2**32-1)
    topology: Literal["linear", "head_to_tail"] = "linear"
    sampling_steps: int = Field(default=500, ge=20, le=500)
    refold: bool = False

    @model_validator(mode="after")
    def target(self):
        if not self.target_sequence and not self.target_pdb:
            raise ValueError("BoltzGen requires a target sequence or PDB")
        if self.target_chain == "B":
            raise ValueError("Chain B is reserved for the designed peptide; rename target chain")
        return self


class Structure(Contract):
    pdb: str = Field(min_length=1, max_length=2_000_000)
    receptor_chain: str = Field(default="A", min_length=1, max_length=1)
    peptide_chain: str = Field(default="B", min_length=1, max_length=1)


class Constraints(Contract):
    max_abs_charge: float = Field(default=10.0, ge=0, le=80)
    max_gravy: float = Field(default=2.0, ge=-4.5, le=4.5)
    max_cysteines: int = Field(default=4, ge=0, le=80)


class Brief(Contract):
    name: str = Field(default="peptide-design", min_length=1, max_length=100)
    source: Literal["library", "pepmlm", "boltzgen"] = "pepmlm"
    target_sequence: Target | None = None
    sequences: list[Sequence] = Field(default_factory=list, max_length=128)
    length: int = Field(default=15, ge=5, le=50)
    count: int = Field(default=8, ge=1, le=64)
    top_k: int = Field(default=3, ge=1, le=20)
    seed: int = Field(default=1729, ge=0, le=2**32 - 1)
    shortlist: int = Field(default=5, ge=1, le=32)
    ph: float = Field(default=7.4, ge=0, le=14)
    constraints: Constraints = Field(default_factory=Constraints)
    topology: Literal["linear", "head_to_tail"] = "linear"
    target_pdb: str | None = Field(default=None, max_length=2_000_000)
    target_chain: str = Field(default="A", pattern=r"^[A-Za-z0-9]$")
    sampling_steps: int = Field(default=500, ge=20, le=500)
    refold: bool = False
    assess_toxicity: bool = False
    assess_affinity: bool = False

    @model_validator(mode="after")
    def coherent(self):
        if self.source == "pepmlm" and (not self.target_sequence or self.sequences):
            raise ValueError("pepmlm requires target_sequence and no library sequences")
        if self.source == "library" and not self.sequences:
            raise ValueError("library requires sequences")
        if self.source == "pepmlm" and self.topology != "linear":
            raise ValueError("PepMLM is linear-only; use BoltzGen for cyclic design")
        if self.source == "boltzgen":
            BoltzGenerate(**{k: getattr(self, k) for k in BoltzGenerate.model_fields})
            if self.sequences:
                raise ValueError("BoltzGen generates sequences; remove supplied sequences")
        if self.assess_affinity and self.source != "boltzgen":
            raise ValueError("Workflow affinity requires BoltzGen complexes; use the structure tool for supplied PDB")
        return self


class Plan(Contract):
    skill: Literal["design-peptide", "analyze-peptide"]
    brief: Brief

    @model_validator(mode="after")
    def scope(self):
        if self.skill == "analyze-peptide" and self.brief.source != "library":
            raise ValueError("analysis skill cannot invoke generation")
        if self.skill == "design-peptide" and self.brief.source == "library":
            raise ValueError("design-peptide requires pepmlm or boltzgen")
        return self


class ToolResult(Contract):
    tool: str
    status: Literal["ok", "unavailable", "error"]
    trust: Literal["CALCULATED", "PREDICTED", "INFERRED", "UNAVAILABLE"]
    data: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Unavailable(RuntimeError):
    """A real capability or model artifact is missing; never substitute fake data."""
