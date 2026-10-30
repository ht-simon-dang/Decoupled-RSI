"""Research brief and program contracts — the platform's primary input."""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import IdentifiedModel, SchemaModel
from .enums import BriefKind, GenerationMethod, ObjectiveDirection


class TargetSpec(SchemaModel):
    """A molecular target for target-based discovery."""

    name: str = Field(description="Human name, e.g. 'Soluble epoxide hydrolase'.")
    gene_symbol: str | None = None
    uniprot: str | None = Field(default=None, description="UniProt accession, e.g. P34913.")
    chembl_target_id: str | None = None
    organism: str | None = "Homo sapiens"
    pdb_ids: list[str] = Field(default_factory=list)


class PhenotypeSpec(SchemaModel):
    """A phenotypic / cell-state objective for phenotype-based discovery."""

    description: str = Field(description="Desired phenotypic change.")
    cell_context: str | None = Field(default=None, description="Cell type / tissue / organoid.")
    disease_state_label: str | None = None
    desired_state_label: str | None = None
    dataset_hint: str | None = Field(default=None, description="Fixture/dataset identifier hint.")


class Objective(SchemaModel):
    """An optimization objective used for Pareto ranking."""

    name: str = Field(description="Endpoint name, e.g. 'pIC50', 'logP', 'sa_score'.")
    direction: ObjectiveDirection
    weight: float = Field(default=1.0, ge=0.0, description="Display-utility weight only.")
    target_value: float | None = Field(default=None, description="Optional desired value.")
    unit: str | None = None
    required: bool = Field(default=False, description="Whether the endpoint must be present.")


class Constraint(SchemaModel):
    """A hard filter applied before ranking."""

    name: str
    property: str = Field(description="Property key the constraint applies to.")
    op: str = Field(description="One of: <, <=, >, >=, ==, in, not_in, is_true, is_false.")
    value: float | int | str | bool | list[str] | None = None
    description: str | None = None
    rationale: str | None = None


class PocketSpec(SchemaModel):
    """A binding site, as consumed by structure-conditioned generative models.

    ``pdb_path`` must already be a *pocket* file (a few hundred atoms), not a
    whole structure: PFM samples inside the box implied by these coordinates, so
    a full-protein PDB produces a meaningless pocket size. Use the upstream
    ``clean_pocket`` helper (or any pocket-extraction tool) to cut one.
    """

    receptor_id: str | None = Field(default=None, description="Receptor/ProteinStructure id.")
    pdb_id: str | None = Field(default=None, description="Source PDB accession, if any.")
    pdb_path: str | None = Field(default=None, description="Path to a prepared pocket PDB file.")
    center: list[float] | None = Field(
        default=None,
        description=(
            "Pocket centre [x, y, z] in Angstrom. NOT provenance: tasks that denoise inside a "
            "box require it — PocketXMol sbdd / sbdd_simple / dock / dock_flexible and RxnFlow "
            "refuse without it, and OMTRA falls back to the structure's heavy-atom mean and "
            "warns. TargetDiff, Pocket2Mol and PocketFlow legitimately run without one."
        ),
    )
    radius: float = Field(default=10.0, gt=0.0, le=50.0, description="Pocket radius in Angstrom.")
    notes: str | None = None


class GenerationSpec(SchemaModel):
    """De-novo generation configuration for a brief.

    Parameters only — *whether* generation runs is decided by
    :attr:`ResearchBrief.candidate_source` (which the validator keeps in sync).
    """

    method: GenerationMethod = GenerationMethod.NONE
    #: Where the binding site comes from. ``brief`` uses :attr:`pocket` exactly as
    #: given and is the default, so every recorded brief and fixture produces the
    #: graph it always produced. ``detect`` schedules ``agent.pocket`` on the
    #: structure this run's target resolves to and prefers what it finds — the
    #: only way to obtain a centre a detector actually predicted.
    pocket_source: str = Field(default="brief", description="brief | detect")
    pocket_detector: str = Field(default="auto", description="auto | p2rank | fpocket | vnegnn")
    n_molecules: int = Field(default=50, ge=1, le=5000, description="Molecules to sample.")
    batch_size: int = Field(default=10, ge=1, le=200, description="Molecules per model call.")
    pocket: PocketSpec | None = Field(default=None, description="Required by structure-based methods.")
    seed: int | None = Field(default=None, description="Sampling seed; defaults to the run seed.")
    max_runtime_seconds: float = Field(default=1800.0, gt=0.0, description="Hard wall-clock cap.")
    min_heavy_atoms: int = Field(default=10, ge=1, description="Post-hoc validity filter.")
    max_heavy_atoms: int = Field(default=70, ge=1, description="Post-hoc validity filter.")
    keep_invalid: bool = Field(
        default=False, description="Keep unparseable outputs for QC instead of discarding them."
    )


class ResearchBrief(IdentifiedModel):
    """Versioned research brief. The unit of work submitted to the platform."""

    id_prefix: str = "brief"
    kind: BriefKind
    title: str
    program_id: str | None = Field(default=None, description="Owning research program.")
    author: str | None = None
    description: str = ""

    target: TargetSpec | None = None
    phenotype: PhenotypeSpec | None = None

    objectives: list[Objective] = Field(default_factory=list)
    hard_constraints: list[Constraint] = Field(default_factory=list)

    seed_smiles: list[str] = Field(
        default_factory=list, description="Optional seed molecules to expand/screen."
    )
    candidate_source: str = Field(
        default="fixture_library",
        description=(
            "Connector/strategy for sourcing candidates. '+'-separated tokens from "
            "{fixture_library, library, seed, analogue, pfm, none}; 'fixture_library' "
            "expands to library+seed+analogue."
        ),
    )
    generation: GenerationSpec | None = Field(
        default=None, description="De-novo generation parameters (see candidate_source)."
    )
    #: Run the target-evidence agent before any chemistry exists.
    #:
    #: Opt-in, and defaulted off, for the reason ``GenerationSpec.pocket_source``
    #: is: a brief that does not ask builds exactly the graph it always built.
    #: ``agent.target`` was registered, contracted and complete, and no task in
    #: ``orchestration/workflow.py`` ever scheduled it — so the two target tools
    #: the pool advertised could not be called by any run at all.
    assess_target_evidence: bool = Field(
        default=False,
        description="Assemble association, tractability, safety and essentiality evidence first.",
    )
    max_candidates: int = Field(default=200, ge=1, le=100_000)
    shortlist_size: int = Field(default=10, ge=1, le=1000)

    dataset_snapshot: str | None = Field(default=None, description="Pinned data snapshot name.")
    approval_gates: list[str] = Field(
        default_factory=lambda: ["portfolio_review"],
        description="Named human-approval gates to pause at.",
    )
    budget_tokens: int | None = Field(default=None, description="Optional LLM token budget.")
    budget_seconds: int | None = Field(default=None, description="Optional wall-clock budget.")
    tags: list[str] = Field(default_factory=list)

    def candidate_sources(self) -> frozenset[str]:
        """Normalize ``candidate_source`` into primitive sourcing strategies.

        Returns a subset of ``{"library", "seed", "analogue", "pfm"}``. Unknown
        tokens are ignored here and reported by ``validate_brief`` instead, so a
        typo degrades to the default rather than crashing a run mid-flight.
        """
        return _parse_candidate_source(self.candidate_source)

    def unknown_candidate_sources(self) -> list[str]:
        """Tokens in ``candidate_source`` this build does not understand."""
        return [
            tok
            for tok in _split_candidate_source(self.candidate_source)
            if tok not in _CANDIDATE_SOURCE_ALIASES
        ]

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> ResearchBrief:
        if self.kind == BriefKind.TARGET_BASED and self.target is None:
            raise ValueError("target_based brief requires a 'target' block")
        if self.kind == BriefKind.PHENOTYPE_BASED and self.phenotype is None:
            raise ValueError("phenotype_based brief requires a 'phenotype' block")
        if not self.objectives:
            raise ValueError("brief must declare at least one objective")
        # Objective names must be unique.
        names = [o.name for o in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError("objective names must be unique")
        # Keep candidate_source and generation mutually consistent, so neither
        # half can silently disable the other. candidate_source is the selector;
        # generation only ever carries parameters.
        if self.generation is not None and self.generation.method != GenerationMethod.NONE:
            method = str(self.generation.method)
            if method not in self.candidate_sources():
                self.candidate_source = f"{self.candidate_source}+{method}".lstrip("+")
        elif "pfm" in self.candidate_sources():
            if self.generation is None:
                self.generation = GenerationSpec(method=GenerationMethod.PFM)
            else:
                # A `method: none` block alongside `candidate_source: …+pfm` is
                # contradictory. The selector wins — but only the method flips;
                # the operator's n_molecules/pocket/window are preserved.
                self.generation.method = GenerationMethod.PFM
        if self.generation is not None and self.generation.min_heavy_atoms > self.generation.max_heavy_atoms:
            raise ValueError("generation.min_heavy_atoms must be <= generation.max_heavy_atoms")
        return self


class ResearchProgram(IdentifiedModel):
    """A long-lived container that isolates memory/knowledge across briefs."""

    id_prefix: str = "program"
    name: str
    description: str = ""
    owner: str | None = None
    brief_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# candidate_source parsing
# ---------------------------------------------------------------------------
#: Token -> primitive sourcing strategies it enables.
_CANDIDATE_SOURCE_ALIASES: dict[str, frozenset[str]] = {
    "fixture_library": frozenset({"library", "seed", "analogue"}),
    "library": frozenset({"library"}),
    "compound_library": frozenset({"library"}),
    "seed": frozenset({"seed"}),
    "seeds": frozenset({"seed"}),
    "analogue": frozenset({"analogue"}),
    "analogues": frozenset({"analogue"}),
    "seed_analogues": frozenset({"seed", "analogue"}),
    "pfm": frozenset({"pfm"}),
    "generated": frozenset({"pfm"}),
    "none": frozenset(),
}


def _split_candidate_source(value: str) -> list[str]:
    raw = (value or "").replace(",", "+").replace("|", "+").replace(" ", "+")
    return [tok.strip().lower() for tok in raw.split("+") if tok.strip()]


def _parse_candidate_source(value: str) -> frozenset[str]:
    tokens = _split_candidate_source(value)
    if not tokens:
        return _CANDIDATE_SOURCE_ALIASES["fixture_library"]
    out: set[str] = set()
    matched = False
    for tok in tokens:
        if tok in _CANDIDATE_SOURCE_ALIASES:
            matched = True
            out |= _CANDIDATE_SOURCE_ALIASES[tok]
    if not matched:
        # Every token was unrecognized: fall back to the documented default.
        return _CANDIDATE_SOURCE_ALIASES["fixture_library"]
    return frozenset(out)
