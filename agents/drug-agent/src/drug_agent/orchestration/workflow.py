"""Workflow builder: constructs the persisted task DAG from a brief.

Target-based and phenotype-based briefs produce *different* graphs: target runs
structure-based assessment; phenotype runs single-cell perturbation as a
required objective. Both share the evidence → source → standardize → filter →
assess → integrate → portfolio → approval → report → bundle spine.
"""

from __future__ import annotations

from ..contracts.brief import ResearchBrief
from ..contracts.enums import BriefKind
from ..contracts.run import RetryPolicy, Task


def build_task_graph(brief: ResearchBrief, run_id: str, *, transient_task_retries: int = 0) -> list[Task]:
    """Build the run's task DAG.

    ``transient_task_retries`` is the extra-attempt budget for tasks that invoke
    an external capability. It was previously unreachable: ``add`` took a
    ``retries`` argument, no call site ever passed it, so every task got
    ``max_attempts=1`` and *nothing in the system ever retried* — a single
    connection reset in one tool call failed the whole run. The engine's retry
    branch existed and was dead code.

    Retries are deliberately not universal. A task with no ``capability_id``
    runs entirely in process and has no transient failure mode, so retrying it
    would spend budget to reach the same answer. And the engine only retries a
    ``RetryableAgentError`` or an unexpected exception — an ``AgentError`` is an
    agent stating that this is a *result*, and results are not retried.
    """
    tasks: list[Task] = []
    by_name: dict[str, Task] = {}

    def add(
        name: str,
        deps: list[str],
        *,
        capability_id: str | None = None,
        approval_gate: str | None = None,
        params: dict | None = None,
        retries: int | None = None,
    ) -> Task:
        if retries is None:
            retries = max(0, transient_task_retries) if capability_id else 0
        t = Task(
            run_id=run_id,
            name=name,
            capability_id=capability_id,
            depends_on=[by_name[d].id for d in deps],
            approval_gate=approval_gate,
            params=params or {},
            retry_policy=RetryPolicy(max_attempts=retries + 1, backoff_seconds=0.0),
        )
        tasks.append(t)
        by_name[name] = t
        return t

    is_target = brief.kind == BriefKind.TARGET_BASED
    objective_names = {o.name for o in brief.objectives}

    add("validate_brief", [])
    add("resolve_entities", ["validate_brief"])
    # Pocket detection, when the brief asked for it. `agent.pocket` was complete,
    # registered and contracted, but no task ever scheduled it — so the blackboard
    # entry `DesignAgent.pocket_for` reads FIRST could never exist, and the only
    # surviving pocket source carried no centre. Every pocket-conditioned model
    # except PFM then failed on a contract field nothing could produce.
    #
    # Opt-in, and defaulted off in `GenerationSpec`, so a brief that does not ask
    # builds exactly the graph it always built.
    generation = brief.generation
    detect_pocket = bool(
        "pfm" in brief.candidate_sources() and generation is not None and generation.pocket_source == "detect"
    )
    if detect_pocket and generation is not None:
        add(
            "detect_pocket",
            ["resolve_entities"],
            capability_id="agent.pocket",
            params={
                "detector": generation.pocket_detector,
                "pocket_radius": (generation.pocket.radius if generation.pocket else 10.0),
                "n_prepared": 1,
            },
        )
    # Target evidence, when the brief asked for it. Same opt-in shape, and for the
    # same reason: `agent.target` is "the first agent in the program and the only
    # one that runs before chemistry exists" (AGENT_NOTES), and until now it never
    # ran, because nothing scheduled it. It goes before the literature sweep so a
    # dossier about the protein is on the blackboard when evidence is assembled.
    if brief.assess_target_evidence:
        add(
            "assemble_target_evidence",
            ["resolve_entities"],
            capability_id="agent.target",
        )
    add("assemble_evidence", ["resolve_entities"], capability_id="agent.literature")
    add("generate_hypotheses", ["assemble_evidence"], capability_id="agent.discovery")
    add("create_data_snapshot", ["resolve_entities"])
    add("run_baselines", ["create_data_snapshot"])
    add(
        "source_candidates",
        ["create_data_snapshot"] + (["detect_pocket"] if detect_pocket else []),
        capability_id="agent.molecule",
    )
    add("standardize_candidates", ["source_candidates"], capability_id="agent.molecule")
    add("apply_hard_filters", ["standardize_candidates"])

    # Parallel assessment fan-out (dependencies permitting).
    assessment_tasks: list[str] = []
    add(
        "predict_activity_and_properties",
        ["apply_hard_filters", "run_baselines"],
        capability_id="agent.molecule",
    )
    assessment_tasks.append("predict_activity_and_properties")
    add("assess_synthesis", ["apply_hard_filters"], capability_id="agent.synthesis")
    assessment_tasks.append("assess_synthesis")
    add("assess_safety", ["apply_hard_filters"], capability_id="agent.safety")
    assessment_tasks.append("assess_safety")

    if is_target:
        add("assess_structure", ["apply_hard_filters"], capability_id="agent.structure")
        assessment_tasks.append("assess_structure")
    if not is_target or "restoration_score" in objective_names:
        add("assess_perturbation", ["apply_hard_filters"], capability_id="agent.perturbation")
        assessment_tasks.append("assess_perturbation")

    add("validate_all_results", assessment_tasks, capability_id="agent.evaluation")
    add("integrate_objectives", ["validate_all_results"])
    add("build_pareto_portfolio", ["integrate_objectives"])
    add("index_knowledge", ["build_pareto_portfolio"], capability_id="agent.memory")

    # Approval gate (if configured), then experiment proposals.
    gate = brief.approval_gates[0] if brief.approval_gates else None
    if gate:
        add("await_human_approval", ["build_pareto_portfolio"], approval_gate=gate)
        add("propose_experiments", ["await_human_approval"], capability_id="agent.discovery")
    else:
        add("propose_experiments", ["build_pareto_portfolio"], capability_id="agent.discovery")

    add("generate_report", ["propose_experiments", "index_knowledge", "generate_hypotheses"])
    add("create_reproducibility_bundle", ["generate_report"])

    return tasks
