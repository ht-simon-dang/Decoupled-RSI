"""Typed records for the evolution loop.

Three families live here:

* **Experience** — :class:`Episode` and :class:`Step`: what the user-serving
  system actually did, normalized from the request journal. Every field is
  either present in the journal or absent; nothing is imputed. A field the
  journal does not carry is ``None``, and the analysis layer is written to
  expect that.
* **Findings** — :class:`Finding`, :class:`FailureCluster`, :class:`Hypothesis`:
  what the analysis concluded, always with the episode ids it concluded it from.
* **Change** — :class:`Candidate`, :class:`EvaluationRecord`,
  :class:`PromotionDecision`: a proposed modification, the evidence produced
  about it, and what was decided.

These are pydantic models so they round-trip to JSON and can be schema-exported
alongside the platform's own contracts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..contracts.enums import StrEnum


class _Rec(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=(), str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------
class Step(_Rec):
    """One model exchange inside an episode, with what it cost and produced.

    This is the unit attribution works over: the Agent-Lightning-style
    ``(prompt, response, outcome)`` triplet, minus the reward — we have no
    gradient to feed, only an analysis to inform.
    """

    index: int = 0
    phase: str = ""  # plan | transcribe | critique | refine
    cycle: int = 0
    attempt: int = 0
    mode: str = ""  # schema | fenced | repair — how the reply was requested
    decode: str = ""  # strict | relaxed | truncated — how it was recovered
    ok: bool = True
    error: str | None = None
    elapsed_s: float = 0.0
    eval_tokens: int | None = None
    prompt_tokens: int | None = None
    done_reason: str | None = None
    n_repairs: int = 0
    thinking_chars: int = 0
    content_chars: int = 0
    #: The decoded object, when the journal recorded one.
    parsed: dict[str, Any] | None = None
    #: sha256 of the system+user prompt actually sent. Lets us tell whether two
    #: episodes ran the same prompt without storing the prompt twice.
    prompt_sha256: str = ""


class ToolUse(_Rec):
    """Aggregated use of one ``(tool_id, operation)`` within an episode."""

    tool_id: str
    operation: str
    backend: str | None = None
    n_calls: int = 0
    n_errors: int = 0
    total_ms: float = 0.0
    mean_ms: float | None = None
    #: Quality metrics the tool reported about its own output (means).
    metrics: dict[str, float | None] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @property
    def is_mocked(self) -> bool:
        return (self.backend or "").lower() == "mock"


class CycleRecord(_Rec):
    """One refinement cycle: what was asked for and what came back."""

    cycle: int = 0
    n_candidates: int | None = None
    shortlist_size: int | None = None
    n_criteria_met: int | None = None
    n_criteria_total: int | None = None
    headline_utility: float | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    verdict: str | None = None  # ACCEPT | REFINE | REPLAN
    diagnosis: str = ""
    #: Notes the evaluation agent attached — dropped seeds, domain warnings,
    #: the most-violated constraint. Free text, but written by deterministic code.
    notes: list[str] = Field(default_factory=list)
    #: Why the pipeline stopped, when it did not reach scoring. Without this the
    #: experience corpus could not tell "the program tried and scored badly" from
    #: "the program crashed and scored nothing" — the two look identical once the
    #: metrics are all null, and only one of them is a prompt problem.
    failure_reason: str | None = None
    #: The pipeline run's terminal state for this cycle (SUCCEEDED / FAILED / ...).
    pipeline_state: str | None = None
    #: Field-level diff the master agent computed for the refinement, if any.
    adjustment_diff: list[dict[str, str]] = Field(default_factory=list)
    #: What the model *said* it changed. Compared against the diff by analysis.
    adjustment_changes: list[dict[str, str]] = Field(default_factory=list)


class Episode(_Rec):
    """One user request, normalized. The unit of experience."""

    episode_id: str
    started_at: float | None = None
    finished_at: float | None = None
    duration_s: float | None = None
    source: str = "unknown"  # cli | web | api
    status: str = "UNKNOWN"  # SUCCEEDED | FAILED
    instruction: str = ""
    #: Coarse class used to condition every comparison. Derived, never invented:
    #: read off the transcribed requirements when present, else the instruction.
    task_class: str = "unknown"
    model: str | None = None
    strategy: str | None = None  # adaptive | naive
    generator: str | None = None
    max_cycles: int | None = None
    prompt_version: str | None = None
    #: Success criteria the planner set, and the ones excluded as unmeasurable.
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    unscorable_criteria: list[dict[str, Any]] = Field(default_factory=list)
    requirements: dict[str, Any] | None = None
    steps: list[Step] = Field(default_factory=list)
    tools: list[ToolUse] = Field(default_factory=list)
    cycles: list[CycleRecord] = Field(default_factory=list)
    outcome: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    #: Where this came from, so any claim can be traced back to bytes on disk.
    journal_dir: str = ""

    # --- derived, all None-safe ---------------------------------------
    @property
    def final_verdict(self) -> str | None:
        for cyc in reversed(self.cycles):
            if cyc.verdict:
                return cyc.verdict
        v = self.outcome.get("verdict")
        return str(v) if v else None

    @property
    def accepted(self) -> bool:
        return self.final_verdict == "ACCEPT"

    @property
    def n_llm_calls(self) -> int:
        return len(self.steps)

    @property
    def eval_tokens(self) -> int:
        return sum(s.eval_tokens or 0 for s in self.steps)

    @property
    def llm_seconds(self) -> float:
        return round(sum(s.elapsed_s for s in self.steps), 3)

    @property
    def n_tool_calls(self) -> int:
        return sum(t.n_calls for t in self.tools)

    @property
    def n_tool_errors(self) -> int:
        return sum(t.n_errors for t in self.tools)

    @property
    def best_cycle(self) -> CycleRecord | None:
        scored = [c for c in self.cycles if c.n_criteria_met is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: (c.n_criteria_met or 0, c.headline_utility or 0.0))

    @property
    def criteria_met_rate(self) -> float | None:
        best = self.best_cycle
        if best is None or not best.n_criteria_total:
            return None
        return round((best.n_criteria_met or 0) / best.n_criteria_total, 4)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
class Severity(StrEnum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Finding(_Rec):
    """One measured observation about the user-serving system.

    ``evidence`` is a list of episode ids (optionally ``id#cycle``). A finding
    with no evidence is a bug in the analyzer, and the store rejects it.
    """

    finding_id: str
    title: str
    detector: str  # which detector produced it, for reproducibility
    severity: Severity = Severity.MEDIUM
    #: How many episodes exhibit it, out of how many it could apply to.
    n_affected: int = 0
    n_applicable: int = 0
    #: Numbers backing the claim. Detectors compute these; nothing is asserted
    #: without one.
    stats: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    detail: str = ""
    #: "association" or "causal". Detectors that cannot separate a tool's effect
    #: from task difficulty must say "association" — see analysis/attribution.py.
    inference: str = "association"

    @property
    def rate(self) -> float | None:
        return round(self.n_affected / self.n_applicable, 4) if self.n_applicable else None


class FailureCluster(_Rec):
    """A group of findings that plausibly share one structural cause."""

    cluster_id: str
    label: str
    findings: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    #: Which part of the system the cluster points at — used as the archive's
    #: behaviour dimension, so a planner fix and a refiner fix coexist.
    surface: str = "unknown"
    summary: str = ""
    #: Cheap descriptor for dedup across cycles.
    signature: str = ""


class Hypothesis(_Rec):
    """A falsifiable proposal for why a cluster happens and what would fix it."""

    hypothesis_id: str
    cluster_id: str
    statement: str
    mechanism: str = ""
    predicted_effect: str = ""
    #: How this could be shown WRONG. A hypothesis without one is not testable
    #: and the verifier is told to reject it.
    falsifier: str = ""
    proposed_change: str = ""
    candidate_class: str = "PromptPatch"
    #: Verifier verdict: SUPPORTED | REFUTED | INSUFFICIENT_EVIDENCE
    verdict: str = ""
    verifier_notes: str = ""
    revision_of: str | None = None
    author: str = ""  # which backend/role wrote it
    evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Change
# ---------------------------------------------------------------------------
class CandidateClass(StrEnum):
    PROMPT_PATCH = "PromptPatch"
    TOOL_DESCRIPTION_PATCH = "ToolDescriptionPatch"
    TOOL_SCHEMA_PATCH = "ToolSchemaPatch"
    TOOL_IMPLEMENTATION_PATCH = "ToolImplementationPatch"
    COMPOSITE_TOOL = "CompositeTool"
    NEW_TOOL = "NewTool"
    NEW_AGENT = "NewAgent"
    AGENT_ROUTING_PATCH = "AgentRoutingPatch"
    VERIFICATION_PATCH = "VerificationPatch"
    MEMORY_PATCH = "MemoryPatch"
    RETRIEVAL_PATCH = "RetrievalPatch"
    CONTEXT_POLICY_PATCH = "ContextPolicyPatch"
    PLANNING_PATCH = "PlanningPatch"
    RECOVERY_POLICY_PATCH = "RecoveryPolicyPatch"


class CandidateStatus(StrEnum):
    DRAFT = "DRAFT"
    EVALUATED = "EVALUATED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"
    PROMOTED = "PROMOTED"
    QUARANTINED = "QUARANTINED"  # tampering suspected — never promotable


class Candidate(_Rec):
    """A proposed modification to the user-serving system.

    Two representations, both evaluable by the same kernel:

    * ``overlay`` — a :class:`drug_agent.llm.policy.RuntimePolicy` payload. The
      user-serving code reads it at startup; the default is inert, so an empty
      overlay reproduces today's behaviour exactly.
    * ``patch`` — a unified diff applied inside an isolated workspace, for
      changes an overlay cannot express (new modules, tool implementations).

    A candidate may carry both. It may never carry a change to the evaluation
    kernel: :mod:`~drug_agent.selfevolve.evaluation.integrity` rejects that
    before the candidate is scored.
    """

    candidate_id: str
    parent_id: str | None = None
    generation: int = 0
    candidate_class: CandidateClass = CandidateClass.PROMPT_PATCH
    hypothesis: str = ""
    hypothesis_id: str | None = None
    failure_modes_addressed: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    expected_benefit: str = ""
    possible_regressions: list[str] = Field(default_factory=list)
    evaluation_plan: list[str] = Field(default_factory=list)
    #: Runtime overlay payload (see drug_agent.llm.policy).
    overlay: dict[str, Any] = Field(default_factory=dict)
    #: Unified diff, applied in an isolated workspace.
    #:
    #: Exempt from the base record's ``str_strip_whitespace``. A diff is bytes,
    #: not prose: stripping the trailing newline off the final hunk makes
    #: ``git apply`` reject the whole patch with "corrupt patch at line N", so
    #: with the model-wide strip in force *every* single-hunk candidate patch
    #: was unapplicable. Leading whitespace matters too — a context line is a
    #: space followed by the line.
    patch: Annotated[str, StringConstraints(strip_whitespace=False)] = ""
    files_changed: list[str] = Field(default_factory=list)
    status: CandidateStatus = CandidateStatus.DRAFT
    created_at: float = 0.0
    author: str = ""
    notes: str = ""

    def surfaces(self) -> list[str]:
        """Overlay keys that actually change behaviour.

        ``policy_id`` and ``notes`` are labels, not surfaces: two candidates that
        differ only in what they call themselves are the same experiment, and a
        cell named after a label would be meaningless.
        """
        return sorted(k for k in self.overlay if k not in _OVERLAY_METADATA and not k.startswith("_"))

    def novelty_descriptor(self) -> str:
        """A stable descriptor of *what this candidate touches*.

        ShinkaEvolve's novelty rejection needs a cheap "have we tried this
        already" key. Two candidates with the same descriptor differ only in
        prose, and with an evaluation budget in the tens that is not worth a
        benchmark run.
        """
        surfaces = self.surfaces()
        payload = json.dumps(
            {
                "class": str(self.candidate_class),
                "surfaces": surfaces,
                "overlay": _canon(self.overlay),
                "patch": hashlib.sha256(self.patch.encode()).hexdigest() if self.patch else "",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def cell(self) -> str:
        """Quality-diversity cell: ``class/primary-surface``.

        Keeps a planner fix and a refiner fix from evicting each other in the
        archive, which is the whole point of a behaviour-indexed archive at this
        scale.
        """
        surfaces = self.surfaces()
        primary = surfaces[0] if surfaces else ("patch" if self.patch else "none")
        return f"{self.candidate_class}/{primary}"


#: Overlay keys that name a candidate rather than change its behaviour.
_OVERLAY_METADATA = frozenset({"policy_id", "notes", "rationale"})


def _canon(obj: Any) -> Any:
    """Canonicalize for hashing: prose fields collapse to a length.

    Two candidates whose only difference is the wording of a ``notes`` field are
    the same experiment.
    """
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in sorted(obj.items()) if k not in _OVERLAY_METADATA}
    if isinstance(obj, list):
        return [_canon(v) for v in obj]
    if isinstance(obj, str) and len(obj) > 160:
        return f"<str:{len(obj)}>"
    return obj


class TaskResult(_Rec):
    """One benchmark task run, scored by the kernel."""

    task_id: str
    arm: str = "baseline"  # baseline | candidate id
    ok: bool = True
    error: str | None = None
    #: Gold criteria (kernel-owned, candidate-blind) met / total.
    gold_met: int = 0
    gold_total: int = 0
    gold_score: float = 0.0
    #: The system's own verdict, and whether it agreed with the gold rubric.
    verdict: str | None = None
    false_accept: bool = False
    false_refine: bool = False
    #: Independently recomputed chemistry (kernel does its own RDKit pass).
    independent: dict[str, Any] = Field(default_factory=dict)
    #: What the agent itself claimed. Recorded, never used for promotion.
    self_reported: dict[str, Any] = Field(default_factory=dict)
    #: Cost.
    llm_calls: int = 0
    eval_tokens: int = 0
    wall_s: float = 0.0
    tool_calls: int = 0
    #: Calibration: |self-set target - achieved| over efficacy criteria.
    infeasible_targets: int = 0
    n_targets: int = 0
    journal_dir: str = ""


class EvaluationRecord(_Rec):
    """Independently generated evidence about one candidate.

    Produced only by :class:`~drug_agent.selfevolve.evaluation.kernel.EvaluationKernel`.
    Nothing else in this package is allowed to construct one, and the promotion
    gate refuses a record whose integrity fields do not verify.
    """

    record_id: str
    candidate_id: str
    arm: str = "candidate"
    mode: str = "program"  # invariants | replay | program
    benchmark_id: str = ""
    benchmark_version: str = ""
    #: sha256 manifest of the kernel + benchmark at evaluation time.
    kernel_digest: str = ""
    kernel_verified: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    #: Every command the kernel ran, with exit status and captured output. The
    #: brief is explicit: "all tests passed" from a model is not evidence.
    commands: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[TaskResult] = Field(default_factory=list)
    aggregates: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None


class PromotionDecision(_Rec):
    candidate_id: str
    decision: str = "REJECT"  # PROMOTE | REJECT | ARCHIVE | NEEDS_HUMAN
    reasons: list[str] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    baseline_record: str | None = None
    candidate_record: str | None = None
    comparison: dict[str, Any] = Field(default_factory=dict)
    decided_at: float = 0.0
    requires_human: bool = True


class CycleLog(_Rec):
    """The evolver's own experience — one line per evolution cycle."""

    cycle: int
    started_at: float = 0.0
    finished_at: float = 0.0
    parent_candidate: str | None = None
    cluster_id: str | None = None
    hypothesis_id: str | None = None
    candidate_id: str | None = None
    research_sources: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    records: list[str] = Field(default_factory=list)
    decision: str | None = None
    reason: str = ""
    backend_calls: dict[str, int] = Field(default_factory=dict)
    backend_seconds: dict[str, float] = Field(default_factory=dict)
    wall_s: float = 0.0
    notes: str = ""
