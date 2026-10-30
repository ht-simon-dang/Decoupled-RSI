"""The mutation operators — how a hypothesis becomes something evaluable.

Two representations, and the choice between them is not stylistic:

**Overlay** — a :class:`~drug_agent.llm.policy.RuntimePolicy` payload. Reversible,
diffable, and bindable per-arm inside one process, which is what makes a paired
baseline/candidate comparison possible at all on a benchmark this expensive. The
overlay surface is small on purpose: it can change what the planner is told and
how the platform repairs model output, and nothing else. It cannot make the
master agent invent a number or bypass the tool plane.

**Patch** — a unified diff applied in an isolated workspace, for changes an
overlay cannot express: a new module, a tool implementation, a new agent. More
expressive, much more expensive to evaluate, and it is the representation the
:mod:`~drug_agent.selfevolve.evaluation.integrity` admission check exists for.

Each builder below states its hypothesis, the findings it addresses, and how it
could fail — the ``possible_regressions`` field is not decoration, it is what the
promotion gate's guarded metrics are chosen from.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from ..schemas import Candidate, CandidateClass


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def build_capability_envelope(**kw: Any) -> Candidate:
    """Tell the planner what this deployment can actually achieve.

    *Hypothesis.* The planner sets unreachable targets because nothing publishes
    the system's operating range. Given a tool-measured envelope, it will set
    criteria inside it, its ACCEPT/REFINE verdict will start agreeing with an
    independent rubric, and the portfolio it produces will not get worse.

    *How this could be wrong.* The envelope is 25 lines of extra context in an
    already long prompt; the model may spend attention on it and transcribe
    worse. Or it may read "achievable: 6.4582" as permission to stop trying, and
    the portfolio degrades. Both are visible: ``gold_score`` is the guard and
    ``kind_accuracy`` catches transcription damage.

    *Why this is not the forbidden move.* It changes the criteria the agent sets
    for itself, which is exactly the thing a self-improving system must not be
    rewarded for. It is therefore *not* scored on those criteria: promotion is
    decided by a rubric the candidate never sees, over molecules it produced.
    """
    return Candidate(
        candidate_id=_new_id("cand"),
        candidate_class=CandidateClass.PLANNING_PATCH,
        hypothesis=(
            "The planner sets success criteria outside the deployment's achievable range because "
            "no component publishes that range. Given a tool-measured capability envelope in the "
            "planning and critique prompts, it will set feasible criteria and its accept/refine "
            "decision will agree with an independent assessment of the portfolio."
        ),
        failure_modes_addressed=["infeasible_targets", "never_accepted", "mocked_backend_scored"],
        expected_benefit=(
            "decision_accuracy rises (fewer false REFINEs); infeasible_target_rate falls; "
            "gold_score unchanged."
        ),
        possible_regressions=[
            "longer prompt degrades transcription (watch kind_accuracy)",
            "planner aims low and the portfolio degrades (watch gold_score — the promotion guard)",
            "more context per call raises token cost (watch mean_eval_tokens)",
        ],
        evaluation_plan=[
            "gate 0: platform regression suite must pass",
            "gate 2: 6 benchmark tasks, baseline vs candidate, paired by task",
            "guard: gold_score must not regress by more than max_regression",
        ],
        overlay={
            "policy_id": "capability-envelope",
            "capability_envelope": True,
            "notes": "Injects the platform.envelope measurement into the plan and critique prompts.",
        },
        files_changed=[],
        created_at=time.time(),
        author=kw.get("author", "builtin"),
        **{k: v for k, v in kw.items() if k in {"parent_id", "generation", "hypothesis_id", "evidence"}},
    )


def build_refine_guard(**kw: Any) -> Candidate:
    """Stop refinement from deleting requirements it never said it was changing.

    *Hypothesis.* ``objectives`` and ``property_constraints`` default to empty
    lists, so an omission in a schema-constrained decode is a deletion. Carrying
    forward entries the model did not name in its own ``changes`` list restores
    the program the critique actually diagnosed.

    *How this could be wrong.* The model may sometimes *intend* to drop a
    constraint without naming it — the whole point of relaxing a filter that
    eliminates every candidate. If the carry-forward is too eager it will block
    a legitimate relaxation. That is why the match on ``changes`` is generous:
    it fails open, back to today's behaviour.
    """
    return Candidate(
        candidate_id=_new_id("cand"),
        candidate_class=CandidateClass.RECOVERY_POLICY_PATCH,
        hypothesis=(
            "Adaptive refinement silently empties the objective and constraint lists because the "
            "schema defaults them to []. Carrying forward entries the model did not report "
            "changing keeps the next cycle scoring the same program the critique diagnosed."
        ),
        failure_modes_addressed=["silent_requirement_loss", "metric_gaming_observed"],
        expected_benefit=(
            "requirement entries survive refinement unless the model says otherwise; "
            "hard_constraint_pass_rate can no longer reach 1.0 by deletion."
        ),
        possible_regressions=[
            "blocks a deliberate but unreported relaxation (watch gold_score on cycle-2 tasks)",
            "no effect at all in a one-cycle benchmark — needs program_cycles >= 2 to bind",
        ],
        evaluation_plan=[
            "gate 1: replay the recorded refine exchange from mrun_333fbbc07a32",
            "gate 0: platform regression suite",
        ],
        overlay={
            "policy_id": "refine-guard",
            "refine_carry_forward": True,
            "refine_max_drop_fraction": 0.5,
            "notes": "Merge guard over the refinement step.",
        },
        files_changed=[],
        created_at=time.time(),
        author=kw.get("author", "builtin"),
        **{k: v for k, v in kw.items() if k in {"parent_id", "generation", "hypothesis_id", "evidence"}},
    )


def build_goal_aware_criteria(**kw: Any) -> Candidate:
    """Drop efficacy criteria the program's goal kind cannot produce.

    *Hypothesis.* A ``max_pIC50`` criterion on a phenotype-rescue program is
    unscorable for the same reason an invented metric key is: nothing can
    compute it. The platform already excludes and reports the second case; the
    first is only detectable after transcription, which is where this repairs it.

    *How this could be wrong.* This raises the agent's own met/total ratio by
    removing criteria, which is the shape of metric-gaming. It is therefore
    scored the same way as everything else — on the gold rubric — and the drop
    is reported in the event stream rather than applied silently.
    """
    return Candidate(
        candidate_id=_new_id("cand"),
        candidate_class=CandidateClass.VERIFICATION_PATCH,
        hypothesis=(
            "Efficacy criteria that cannot apply to the program's goal kind are unscorable in "
            "exactly the sense the platform already handles. Excluding them after transcription "
            "and reporting the exclusion removes criteria that can never be met."
        ),
        failure_modes_addressed=["goal_metric_mismatch", "never_accepted"],
        expected_benefit="phenotype programs stop being scored on potency metrics they cannot produce",
        possible_regressions=[
            "raises the agent's own met-rate without improving the portfolio — the gold rubric is "
            "the check, and it is not affected by this change",
            "a mis-transcribed kind now silently removes the *right* criteria (watch kind_accuracy)",
        ],
        evaluation_plan=["gate 2 on the two phenotype tasks", "gate 0 regression suite"],
        overlay={
            "policy_id": "goal-aware-criteria",
            "goal_aware_criteria": True,
            "notes": "Post-transcription criteria repair.",
        },
        files_changed=[],
        created_at=time.time(),
        author=kw.get("author", "builtin"),
        **{k: v for k, v in kw.items() if k in {"parent_id", "generation", "hypothesis_id", "evidence"}},
    )


def build_combined(**kw: Any) -> Candidate:
    """Envelope + goal-aware criteria + refinement guard, as one lineage step.

    Kept separate from its parts so the archive can hold all four and the
    comparison can say which component carried the effect.
    """
    return Candidate(
        candidate_id=_new_id("cand"),
        candidate_class=CandidateClass.PLANNING_PATCH,
        hypothesis=(
            "The planner-criteria failures and the refinement failures are independent, so the "
            "three targeted changes should compose without interfering."
        ),
        failure_modes_addressed=[
            "infeasible_targets",
            "never_accepted",
            "goal_metric_mismatch",
            "silent_requirement_loss",
        ],
        expected_benefit="the union of the three, if they do not interact",
        possible_regressions=[
            "combined prompt growth degrades transcription",
            "if the parts interact, the comparison cannot say which one did what",
        ],
        evaluation_plan=["gate 0", "gate 1", "gate 2 on all six tasks"],
        overlay={
            "policy_id": "combined-v1",
            "capability_envelope": True,
            "goal_aware_criteria": True,
            "refine_carry_forward": True,
            "refine_max_drop_fraction": 0.5,
            "notes": "Composition of the three targeted candidates.",
        },
        files_changed=[],
        created_at=time.time(),
        author=kw.get("author", "builtin"),
        **{k: v for k, v in kw.items() if k in {"parent_id", "generation", "hypothesis_id", "evidence"}},
    )


def build_prompt_addendum(
    *,
    phase: str,
    text: str,
    hypothesis: str,
    addressed: list[str] | None = None,
    **kw: Any,
) -> Candidate:
    """A free-text prompt patch — the operator the hypothesis stage writes into.

    This is how a model-authored proposal becomes evaluable without letting a
    model write code into the production path: it proposes *text*, appended to
    one named phase's system prompt, and the benchmark decides whether it helped.
    """
    field = {
        "plan": "plan_prompt_addendum",
        "transcribe": "transcribe_prompt_addendum",
        "critique": "critique_prompt_addendum",
        "refine": "refine_prompt_addendum",
    }.get(phase)
    if field is None:
        raise ValueError(f"unknown prompt phase {phase!r}")
    return Candidate(
        candidate_id=_new_id("cand"),
        candidate_class=CandidateClass.PROMPT_PATCH,
        hypothesis=hypothesis,
        failure_modes_addressed=list(addressed or []),
        expected_benefit=kw.pop("expected_benefit", "see hypothesis"),
        possible_regressions=[
            "prompt growth costs tokens and attention (watch mean_eval_tokens, kind_accuracy)",
            "instructions that contradict the existing rules produce worse structured output "
            "(watch n_decode_failures)",
        ],
        evaluation_plan=["gate 0", "gate 2 on all six tasks"],
        overlay={"policy_id": f"prompt-{phase}", field: text, "notes": hypothesis[:200]},
        files_changed=[],
        created_at=time.time(),
        author=kw.get("author", "hypothesis"),
        **{k: v for k, v in kw.items() if k in {"parent_id", "generation", "hypothesis_id", "evidence"}},
    )


#: The operators the controller may apply, keyed by the failure surface they aim
#: at. Mapping by surface rather than by name is what lets the controller pick an
#: operator for a cluster it opened without a model in the loop.
CANDIDATE_BUILDERS: dict[str, Callable[..., Candidate]] = {
    "capability_envelope": build_capability_envelope,
    "goal_aware_criteria": build_goal_aware_criteria,
    "refine_guard": build_refine_guard,
    "combined": build_combined,
}

BUILDERS_BY_SURFACE: dict[str, list[str]] = {
    "planner.criteria": ["capability_envelope", "goal_aware_criteria", "combined"],
    "refiner.requirements": ["refine_guard"],
    "transcriber.seeds": [],
    "tools.routing": [],
    "harness.decoding": [],
    "controller.cost": ["capability_envelope"],
}


def build_candidate(name: str, **kw: Any) -> Candidate:
    builder = CANDIDATE_BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"unknown candidate builder {name!r}; have {sorted(CANDIDATE_BUILDERS)}")
    return builder(**kw)


def describe_builders() -> list[dict[str, Any]]:
    out = []
    for name, builder in sorted(CANDIDATE_BUILDERS.items()):
        cand = builder()
        out.append(
            {
                "name": name,
                "class": str(cand.candidate_class),
                "overlay": cand.overlay,
                "addresses": cand.failure_modes_addressed,
                "hypothesis": cand.hypothesis,
                "cell": cand.cell(),
            }
        )
    return out


__all__ = [
    "BUILDERS_BY_SURFACE",
    "CANDIDATE_BUILDERS",
    "build_candidate",
    "build_prompt_addendum",
    "describe_builders",
]
