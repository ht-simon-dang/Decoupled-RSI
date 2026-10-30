"""Structured outputs the master agent produces (the LLM's *decisions*, never
scientific numbers). These are deliberately lenient (``extra="ignore"``, coercing
validators) because a 27B quantized model's JSON is not always pristine — the
harness self-repairs, and these schemas absorb the rest.

Leniency here is specific, not blanket. Each coercion below exists because the
model was observed doing that exact thing: writing ``key`` where the contract
says ``property``, naming a success metric the evaluation agent cannot compute,
or spelling a direction as "higher". Absorbing a *synonym* is free; absorbing a
wrong *value* would not be, so nothing here invents or rounds a number.

Mapping to the drug_agent backend:
  * ``ChemicalRequirements`` -> ``ResearchBrief`` (see backend.py)
  * ``objectives``           -> ``brief.objectives`` (Objective)
  * ``property_constraints`` -> ``brief.hard_constraints`` (Constraint, dotted keys)
  * ``seed_smiles``          -> ``brief.seed_smiles`` (RDKit-validated first)
  * ``generator``            -> ``brief.candidate_source`` (library / PFM de-novo)
  * ``n_generated``          -> ``brief.generation.n_molecules``
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=(), str_strip_whitespace=True)


def _rename(data: Any, synonyms: dict[str, str], canonical: str) -> Any:
    """Move a synonym key onto its canonical name, if the canonical is absent.

    Used in ``mode="before"`` validators. A present canonical key always wins, so
    a model that emits both loses nothing.
    """
    if not isinstance(data, dict) or canonical in data:
        return data
    for alias, target in synonyms.items():
        if alias in data:
            data = {**data, target: data[alias]}
            break
    return data


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
#: The metric keys the evaluation agent can actually compute. A criterion using
#: any other key is unscorable — it can never be marked met, so a plan full of
#: invented keys makes ACCEPT unreachable and the run burns every cycle. The
#: authoritative definitions live in ``drug_agent.agents.llm.evaluation``.
METRIC_KEYS: frozenset[str] = frozenset(
    {
        "max_pIC50",
        "median_pIC50",
        "qed_median",
        "sa_median",
        "herg_median",
        "restoration_median",
        "hard_constraint_pass_rate",
        "mean_pairwise_distance",
        "n_scaffold_clusters",
        "shortlist_filled",
        "in_domain_fraction",
    }
)

#: Near-misses seen from real models, mapped onto the key that means the same
#: thing. Only unambiguous renamings belong here.
_METRIC_SYNONYMS: dict[str, str] = {
    "pic50": "max_pIC50",
    "max_pic50": "max_pIC50",
    "best_pic50": "max_pIC50",
    "potency": "max_pIC50",
    "median_pic50": "median_pIC50",
    "qed": "qed_median",
    "qed_score": "qed_median",
    "median_qed": "qed_median",
    "drug_likeness": "qed_median",
    "sa": "sa_median",
    "sa_score": "sa_median",
    "median_sa": "sa_median",
    "synthetic_accessibility": "sa_median",
    "herg": "herg_median",
    "median_herg": "herg_median",
    "restoration_score": "restoration_median",
    "diversity": "mean_pairwise_distance",
    "chemical_diversity": "mean_pairwise_distance",
    "mean_tanimoto_distance": "mean_pairwise_distance",
    "scaffold_clusters": "n_scaffold_clusters",
    "n_clusters": "n_scaffold_clusters",
    "shortlist_size": "shortlist_filled",
    "constraint_pass_rate": "hard_constraint_pass_rate",
    "hard_constraint_pass": "hard_constraint_pass_rate",
    "in_domain": "in_domain_fraction",
}


def normalize_metric(raw: str) -> str:
    """Map a model-written metric name onto a computable key where possible.

    Unknown keys are returned unchanged rather than dropped: the planner is
    allowed to say what it cares about, and the caller decides what to do with a
    criterion nothing can measure. Silently deleting it would hide the gap.
    """
    key = (raw or "").strip()
    if key in METRIC_KEYS:
        return key
    folded = key.lower().replace(" ", "_").replace("-", "_")
    if folded in _METRIC_SYNONYMS:
        return _METRIC_SYNONYMS[folded]
    for known in METRIC_KEYS:  # case-only difference (pIC50 vs pic50)
        if known.lower() == folded:
            return known
    return key


class SuccessCriterion(_LLMModel):
    metric: str = Field(description="a metric key the evaluation agent can compute")
    target: str = Field(description="human-readable target, e.g. '>= 7.0'")
    rationale: str = ""

    @field_validator("metric", "target", "rationale", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("metric", mode="after")
    @classmethod
    def _known_metric(cls, v: str) -> str:
        return normalize_metric(v)

    @property
    def scorable(self) -> bool:
        """Whether the evaluation agent can compute this criterion at all."""
        return self.metric in METRIC_KEYS


class SubTask(_LLMModel):
    id: str = ""
    description: str = ""
    agent: str = "chemical"  # chemical | evaluation | master


class TaskPlan(_LLMModel):
    goal_restatement: str = ""
    interpretation: str = ""
    strategy: str = ""
    subtasks: list[SubTask] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    planned_cycles: int = 2

    @field_validator("planned_cycles", mode="before")
    @classmethod
    def _coerce_cycles(cls, v: Any) -> int:
        try:
            return max(1, min(6, int(v)))
        except (TypeError, ValueError):
            return 2


# --------------------------------------------------------------------------
# Chemical requirements (the transcribed "generation prompt")
# --------------------------------------------------------------------------
class ObjectiveSpec(_LLMModel):
    name: str
    direction: str = "maximize"  # maximize | minimize
    weight: float = 1.0
    target_value: float | None = None
    required: bool = False

    @field_validator("direction", mode="before")
    @classmethod
    def _norm_direction(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in ("max", "maximize", "maximise", "up", "higher"):
            return "maximize"
        if s in ("min", "minimize", "minimise", "down", "lower"):
            return "minimize"
        return "maximize"

    @field_validator("weight", mode="before")
    @classmethod
    def _coerce_weight(cls, v: Any) -> float:
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return 1.0

    @model_validator(mode="before")
    @classmethod
    def _accept_name_synonyms(cls, data: Any) -> Any:
        return _rename(data, {"objective": "name", "metric": "name", "key": "name"}, "name")


class PropertyConstraint(_LLMModel):
    property: str  # dotted backend key, e.g. descriptors.mw, n_structural_alerts
    op: str = "<="
    value: Any = None
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_field_synonyms(cls, data: Any) -> Any:
        # Observed from qwen3.8:27b during refinement: every constraint came
        # back as {"key": "descriptors.mw", "op": "<=", "value": 530}. The
        # content was right; only the field name was invented.
        data = _rename(data, {"key": "property", "name": "property", "field": "property"}, "property")
        data = _rename(data, {"operator": "op", "comparator": "op", "cmp": "op"}, "op")
        return _rename(data, {"threshold": "value", "val": "value", "limit": "value"}, "value")

    @field_validator("op", mode="before")
    @classmethod
    def _norm_op(cls, v: Any) -> str:
        s = str(v).strip()
        return s if s in ("<", "<=", ">", ">=", "==", "!=") else "<="


class ChemicalRequirements(_LLMModel):
    kind: str = "target_based"  # target_based | phenotype_based
    title: str = "LLM-directed discovery program"
    target_name: str | None = None
    target_gene: str | None = None
    target_uniprot: str | None = None
    phenotype_description: str | None = None
    cell_context: str | None = None
    primary_objective: str = ""
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    property_constraints: list[PropertyConstraint] = Field(default_factory=list)
    seed_smiles: list[str] = Field(default_factory=list)
    avoid_scaffolds: list[str] = Field(default_factory=list)
    novelty: str = "medium"  # low | medium | high
    diversity_requirement: str = "medium"
    # Where candidate structures come from. This is the MASTER AGENT's decision:
    # whether a goal needs new chemistry or a search of known chemistry is a
    # scientific judgement about the goal, and the console used to force it on the
    # operator, who has less to go on. An operator can still pin it (`--generator`)
    # for an ablation; when they do not, the transcription step chooses and the
    # master agent holds it across refinement cycles.
    #
    # The wire tokens are unchanged so recorded history still replays: "pfm" names
    # the de-novo ARM, not the model. Which model samples it is `generator_backend`.
    generator: str = "library"  # library | pfm | library+pfm
    # Which de-novo model samples the arm. "auto" is the deployment default (PFM).
    # Selected from the installed generator catalogue, which the master agent is
    # shown; an uninstalled name is refused and downgraded, never silently swapped.
    generator_backend: str = "auto"
    # How many molecules the de-novo arm samples. An LLM lever — when novelty or
    # diversity is the diagnosed failure, sampling more de-novo structures is the
    # targeted fix.
    n_generated: int = 40
    # --- the binding site -------------------------------------------------
    # Pocket-conditioned generators refuse without a prepared pocket, and the
    # de-novo PocketXMol tasks refuse without its numeric CENTRE. Until these
    # fields existed the master agent had no way to ask for either: the only
    # pocket in the system was one an operator had configured, and the prompt told
    # the model the site was none of its business. It could therefore choose
    # pocketxmol and be guaranteed to fail.
    #
    # "detect" schedules the pocket agent (p2rank / fpocket) on the structure the
    # target resolves to; "brief" uses the operator's configured pocket as-is;
    # "auto" lets the bridge decide from whether a structure is nameable.
    pocket_source: str = "auto"  # auto | detect | brief
    pocket_detector: str = "auto"  # auto | p2rank | fpocket | vnegnn
    # PDB accessions pinning WHICH structure to detect in. The pocket agent
    # resolves these against the local RCSB cache; no network call is made. The
    # model may name a structure, never invent coordinates.
    pdb_ids: list[str] = Field(default_factory=list)
    # --- assessment depth -------------------------------------------------
    # Which OPTIONAL assessment arms this run switches on. Every one names a tool
    # that is installed here and declared reachable in `agents/registry`, and
    # every one was, until these fields existed, reachable by nothing: the agent
    # constructors were called with the arms pinned off and no run parameter,
    # profile or decision could pass anything else.
    #
    # They cost real time, which is why they default to off and why the master
    # agent is told what each buys before it asks. Turning one on is a scientific
    # judgement about the goal — an ADMET-heavy brief wants the learned panel; a
    # brief whose success criteria mention makeability wants the route search —
    # so it is the master agent's call, exactly as the candidate source is.
    #: Learned property panels: any of admetai, chemprop, unimol.
    property_backends: list[str] = Field(default_factory=list)
    #: Retrosynthetic search: none | aizynthfinder.
    route_backend: str = "none"
    #: Co-folded complex prediction: none | boltz2 | protenix.
    complex_backend: str = "none"
    #: Run the target-evidence dossier before any chemistry exists.
    target_evidence: bool = False
    max_candidates: int = 200
    shortlist_size: int = 8
    generation_notes: str = ""

    @field_validator("generator", mode="before")
    @classmethod
    def _norm_generator(cls, v: Any) -> str:
        s = str(v or "").strip().lower().replace(",", "+").replace(" ", "")
        has_pfm = "pfm" in s or "denovo" in s or "de-novo" in s or "diffusion" in s
        has_lib = "librar" in s or "fixture" in s or "analog" in s
        if has_pfm and has_lib:
            return "library+pfm"
        if has_pfm:
            return "pfm"
        return "library"

    @field_validator("generator_backend", mode="before")
    @classmethod
    def _norm_generator_backend(cls, v: Any) -> str:
        """Normalize to a bare engine name; availability is checked upstream.

        Only the shape is enforced here. Whether the named engine is installed is
        a deployment question the master agent answers against the live tool pool,
        because a schema that silently rewrote an unavailable name to "auto" would
        be exactly the substitution this system refuses to make.
        """
        s = str(v or "auto").strip().lower()
        s = s.rsplit(".", 1)[-1] if s.startswith("molecule.generate.") else s
        return s or "auto"

    @field_validator("pocket_source", mode="before")
    @classmethod
    def _norm_pocket_source(cls, v: Any) -> str:
        s = str(v or "auto").strip().lower()
        if "detect" in s or "find" in s:
            return "detect"
        if "brief" in s or "config" in s or "operator" in s:
            return "brief"
        return "auto"

    @field_validator("pocket_detector", mode="before")
    @classmethod
    def _norm_pocket_detector(cls, v: Any) -> str:
        """Only the names ``select_detector`` accepts; anything else means auto.

        An unknown detector name is the model guessing, and guessing here would
        raise deep inside the pocket agent. "auto" is a real, always-valid choice,
        so it is the safe reading of an unrecognised one.
        """
        s = str(v or "auto").strip().lower()
        return s if s in ("auto", "p2rank", "fpocket", "vnegnn") else "auto"

    @field_validator("pdb_ids", mode="before")
    @classmethod
    def _norm_pdb_ids(cls, v: Any) -> list[str]:
        """Keep only things shaped like a PDB accession, upper-cased and deduped.

        Never invents one. A model that writes prose here contributes nothing
        rather than sending the structure cache looking for a sentence.
        """
        if v is None:
            return []
        items = v if isinstance(v, (list, tuple)) else [v]
        out: list[str] = []
        for item in items:
            token = str(item or "").strip().upper()
            shaped = len(token) == 4 and token.isalnum() and any(c.isdigit() for c in token)
            if shaped and token not in out:
                out.append(token)
        return out

    @field_validator("property_backends", mode="before")
    @classmethod
    def _norm_property_backends(cls, v: Any) -> list[str]:
        """Keep only panels this agent really drives, deduped and ordered.

        An unknown name is dropped rather than raised: the molecule agent raises
        on one, and a hallucinated backend should cost the run that panel, not
        the run. What is kept is spelled the way `PROPERTY_TOOL_IDS` spells it.
        """
        known = ("admetai", "chemprop", "unimol")
        if isinstance(v, str):
            v = v.replace(",", "+").split("+")
        out: list[str] = []
        for item in v or []:
            name = str(item or "").strip().lower().rsplit(".", 1)[-1]
            if name in known and name not in out:
                out.append(name)
        return out

    @field_validator("route_backend", mode="before")
    @classmethod
    def _norm_route_backend(cls, v: Any) -> str:
        s = str(v or "none").strip().lower().rsplit(".", 1)[-1]
        return "aizynthfinder" if s in ("aizynthfinder", "aizynth", "auto") else "none"

    @field_validator("complex_backend", mode="before")
    @classmethod
    def _norm_complex_backend(cls, v: Any) -> str:
        s = str(v or "none").strip().lower().rsplit(".", 1)[-1]
        if s in ("boltz2", "boltz", "auto"):
            return "boltz2"
        return "protenix" if s == "protenix" else "none"

    def resolved_pocket_source(self) -> str:
        """``pocket_source`` with ``auto`` decided from what this request can name.

        Detection needs a structure to detect in, and the pocket agent resolves
        one from a PDB id or the target's identity. A request that names neither
        has nothing to detect in, so ``auto`` there means "use the brief's pocket"
        rather than scheduling a task that can only report that it found nothing.
        """
        if self.pocket_source in ("detect", "brief"):
            return self.pocket_source
        nameable = bool(self.pdb_ids or self.target_gene or self.target_uniprot or self.target_name)
        return "detect" if nameable else "brief"

    @field_validator("n_generated", mode="before")
    @classmethod
    def _coerce_n_generated(cls, v: Any) -> int:
        try:
            return max(1, min(2000, int(v)))
        except (TypeError, ValueError):
            return 40

    @field_validator("kind", mode="before")
    @classmethod
    def _norm_kind(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if "pheno" in s:
            return "phenotype_based"
        return "target_based"

    @field_validator("max_candidates", mode="before")
    @classmethod
    def _coerce_maxc(cls, v: Any) -> int:
        try:
            return max(20, min(2000, int(v)))
        except (TypeError, ValueError):
            return 200

    @field_validator("shortlist_size", mode="before")
    @classmethod
    def _coerce_short(cls, v: Any) -> int:
        try:
            return max(3, min(50, int(v)))
        except (TypeError, ValueError):
            return 8


# --------------------------------------------------------------------------
# Critique + refinement (the ablation core)
# --------------------------------------------------------------------------
def _coerce_str(v: Any) -> str:
    return "" if v is None else str(v)


class CriterionVerdict(_LLMModel):
    metric: str = ""
    target: str = ""
    observed: str = ""
    met: bool = False
    comment: str = ""

    @field_validator("metric", "target", "observed", "comment", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return _coerce_str(v)


class Critique(_LLMModel):
    per_criterion: list[CriterionVerdict] = Field(default_factory=list)
    overall_verdict: str = "REFINE"  # ACCEPT | REFINE | REPLAN
    diagnosis: str = ""
    confidence: str = "medium"
    summary: str = ""

    @field_validator("overall_verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if s.startswith("ACCEPT"):
            return "ACCEPT"
        if s.startswith("REPLAN"):
            return "REPLAN"
        return "REFINE"


class ParameterChange(_LLMModel):
    parameter: str = ""
    old_value: str = ""
    new_value: str = ""
    reason: str = ""

    @field_validator("parameter", "old_value", "new_value", "reason", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return _coerce_str(v)


class RequirementsAdjustment(_LLMModel):
    changes: list[ParameterChange] = Field(default_factory=list)
    strategy_note: str = ""
    adjusted_requirements: ChemicalRequirements


# --------------------------------------------------------------------------
# Final cross-cycle selection (``dynamic`` ceiling mode)
# --------------------------------------------------------------------------
class BestCandidateSelection(_LLMModel):
    """The one candidate, across every cycle, the master agent would take forward.

    Note what this object deliberately does NOT have: a field for a potency, a
    QED, a utility, or any other score. Under ADR-0011 the master agent decides
    and explains; it ranks over numbers the tool plane already computed and may
    never restate, round, or invent one. Giving it nowhere to write a number is
    cheaper than checking afterwards whether it did — the caller re-reads every
    score for the chosen id from the evaluation report, and the model's own text
    reaches the user only as ``reason``.

    ``candidate_id`` is likewise checked against the ids that were offered; a
    name that was not in the table is a hallucination, not a selection.
    """

    candidate_id: str = ""
    cycle: int = 0
    reason: str = ""
    runner_up_id: str = ""
    tradeoff: str = ""
    confidence: str = "medium"

    @field_validator("candidate_id", "reason", "runner_up_id", "tradeoff", "confidence", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return _coerce_str(v)

    @field_validator("cycle", mode="before")
    @classmethod
    def _coerce_cycle(cls, v: Any) -> int:
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0

    @model_validator(mode="before")
    @classmethod
    def _accept_id_synonyms(cls, data: Any) -> Any:
        data = _rename(data, {"id": "candidate_id", "chosen": "candidate_id"}, "candidate_id")
        return _rename(data, {"justification": "reason", "rationale": "reason"}, "reason")


def candidate_source_label(generator: Any, generator_backend: Any = None) -> str:
    """How a candidate source reads to a human, naming the model that really ran.

    The wire token for the de-novo arm is ``"pfm"`` and has been since there was
    only one generative model. There are now eleven installed, and the token did
    not change — recorded history replays against it — so every human-facing
    surface inherited a label that names one specific model for work any of the
    eleven may have done. An operator who watched PocketXMol sample for four
    minutes then read "candidate source pfm" in the summary concluded, correctly
    from what they were shown and wrongly about what happened, that the platform
    had ignored their instruction and run PFM.

    So the arm is spelled "de-novo" wherever a person reads it, and the engine is
    named beside it. ``pfm`` the model appears only when PFM the model ran.
    """
    gen = str(generator or "library").strip().lower()
    engine = str(generator_backend or "").strip().lower()
    denovo = "pfm" in gen or "denovo" in gen
    if not denovo:
        return "library"
    arm = "de-novo" if engine in ("", "auto") else f"de-novo ({engine})"
    return f"library + {arm}" if "librar" in gen else arm
