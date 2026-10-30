"""Running one benchmark task under one arm, and scoring it here.

The shape of a task run, and why it is this shape:

1. **The real master agent runs the real program.** No mock of the thing under
   test. The arm's :class:`~drug_agent.llm.policy.RuntimePolicy` is bound with a
   ``ContextVar`` for the duration, so a baseline arm and a candidate arm can run
   back to back in one process without either inheriting the other.

2. **The kernel then re-executes the pipeline itself.** The agent reports
   numbers; the kernel does not read them for scoring. It takes the
   ``ChemicalRequirements`` the agent produced, runs
   :func:`drug_agent.llm.bridge.run_pipeline` again — 2.3 s, deterministic, same
   result — and measures *that*. What the agent claimed is still recorded, under
   ``self_reported``, so a divergence between claim and re-execution is visible.

3. **Chemistry is recomputed from SMILES.** QED, PAINS, pairwise distance and the
   brief's own hard constraints are recomputed with RDKit inside the kernel. That
   tier does not depend on the platform's scorecards being right.

4. **Gold criteria are scored, then the agent's verdict is judged against them.**
   The verdict taken is the *effective* one — the master agent forces ACCEPT when
   every success criterion is met, whatever the critique said — because that is
   what the user would have seen. The raw critique verdict is recorded beside it.
   A run that clears the rubric and was refused is a *false refine*; one that
   fails the rubric and was accepted is a *false accept*. Those two are the
   quality signal that a candidate cannot improve by making its own criteria
   easier, because the rubric is not its criteria.

Cost: one program is three or four local-model calls, minutes each. Everything
else in this file is seconds. That ratio is why the benchmark runs one refinement
cycle by default and why the deterministic re-execution is free enough to do on
every task.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
import uuid
from typing import Any

from ...llm.config import Config
from ...llm.policy import RuntimePolicy, use_policy
from ...llm.schemas import ChemicalRequirements
from ..schemas import TaskResult
from ..target.api import endpoint_result
from .benchmark import GOLD_PASS_THRESHOLD, BenchmarkTask, GoldCriterion
from .metrics import cross_check, kernel_metrics


class TaskRunError(RuntimeError):
    """The arm could not be run at all (model down, timeout, crash)."""


def _quiet() -> contextlib.AbstractContextManager[Any]:
    """Swallow the platform's structlog-to-stdout while a task runs.

    The Engine logs a span per agent step to stdout. That is right for an
    operator watching a run and wrong for a harness whose stdout is a report,
    so it is captured rather than suppressed — nothing is lost, it just does not
    interleave with the evidence.
    """
    return contextlib.redirect_stdout(io.StringIO())


async def _run_program(
    task: BenchmarkTask,
    *,
    cfg: Config,
    cycles: int,
    request_id: str,
) -> dict[str, Any]:
    from ...agents.llm.master_agent import MasterAgent
    from ...llm.events import RunSession

    session = RunSession(request_id)
    agent = MasterAgent(cfg)
    journal = agent.open_journal(
        session,
        task.instruction,
        source="selfevolve-benchmark",
        model=cfg.model,
        strategy=cfg.refine_strategy,
        generator=task.generator,
        n_generated=cfg.n_generated,
        max_cycles=cycles,
        benchmark_task=task.task_id,
    )
    await agent.run(
        session,
        task.instruction,
        strategy=cfg.refine_strategy,
        max_cycles=cycles,
        generator=task.generator,
        source="selfevolve-benchmark",
        journal=journal,
    )
    return {
        "result": session.result,
        "error": session.error,
        "journal_dir": str(journal.dir) if journal is not None else "",
        "n_events": len(session.log),
    }


#: The pipeline states that mean the kernel's own re-execution produced a real
#: observation. Everything else is an absence, however many zeros it left behind.
_SUCCESSFUL_KERNEL_STATES = {"SUCCEEDED"}


def kernel_execution_endpoint(run: Any) -> dict[str, Any]:
    """Whether the kernel's re-execution of the pipeline measured anything.

    ``run`` is a :class:`drug_agent.llm.bridge.BackendRun`. A run that ends
    ``FAILED`` still has a full metric row — ``shortlist_size`` 0,
    ``hard_constraint_pass_rate`` 0.0, medians ``None`` — and those zeros used to
    be scored against the gold rubric and averaged into the arm's score. They are
    not measurements of a portfolio; they are the absence of one.

    The value carried on success is the shortlist size, which is the one number
    that is always defined for a completed run and is 0.0 for a completed run
    that legitimately shortlisted nothing — a measured zero, not an absence.
    """
    state = str(getattr(run, "state", "") or "")
    portfolio = getattr(run, "portfolio", None) or {}
    if state not in _SUCCESSFUL_KERNEL_STATES:
        record = endpoint_result("FAILED" if state else "UNAVAILABLE", None)
        record["error_code"] = f"kernel re-execution ended {state or 'with no state'}" + (
            f": {run.error}" if getattr(run, "error", None) else ""
        )
        record["kernel_state"] = state
        record["kernel_run_id"] = getattr(run, "run_id", "")
        return record
    record = endpoint_result("SUCCEEDED", float(len(portfolio.get("shortlist") or [])))
    record["error_code"] = None
    record["kernel_state"] = state
    record["kernel_run_id"] = getattr(run, "run_id", "")
    return record


def score_gold(
    gold: tuple[GoldCriterion, ...], metrics: dict[str, Any]
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    """Score the gold rubric, recording *why* a criterion was not scored.

    Returns ``(rows, met, scored, unscorable, invalid)``. Three outcomes that
    used to be one:

    ``SUCCEEDED``
        the metric was produced and compared; ``met`` is true or false.
    ``UNAVAILABLE``
        the metric was not produced at all.
    ``INVALID_INPUT``
        the metric was produced as NaN or an infinity. That compared false
        against every threshold and was therefore counted as a criterion the arm
        missed, which is a computational fault reported as a scientific result.

    ``met`` is unchanged for every criterion that really was compared, so the
    rubric's scale and :data:`GOLD_PASS_THRESHOLD` mean exactly what they meant
    before; only the reason a criterion went unscored is now on the record.
    """
    rows: list[dict[str, Any]] = []
    met = scored = unscorable = invalid = 0
    for crit in gold:
        raw = metrics.get(crit.metric)
        ok, observed = crit.score(metrics)
        if raw is None:
            endpoint = endpoint_result("UNAVAILABLE", None)
            ok, observed = None, None
            unscorable += 1
        else:
            try:
                endpoint = endpoint_result("SUCCEEDED", float(raw))
            except (TypeError, ValueError):
                endpoint = endpoint_result("INVALID_INPUT", None)
                ok, observed = None, None
                invalid += 1
        rows.append(
            {
                "metric": crit.metric,
                "op": crit.op,
                "value": crit.value,
                "observed": observed,
                "met": ok,
                "why": crit.why,
                "endpoint": endpoint,
            }
        )
        if ok is None:
            continue
        scored += 1
        met += 1 if ok else 0
    return rows, met, scored, unscorable, invalid


def run_task(
    task: BenchmarkTask,
    *,
    policy: RuntimePolicy,
    arm: str,
    cfg: Config | None = None,
    cycles: int = 1,
    timeout_s: float = 2400.0,
    journal_dir: str | None = None,
) -> TaskResult:
    """Run one task under one arm and score it. Never raises for a task failure.

    Benchmark programs journal to their **own** root, not to the platform's
    request journal. A benchmark run is not a user request: letting it land in
    the same directory would mean the analyzer studies the evolver's own
    experiments as if they were user experience, and would put every holdout
    instruction into the history the candidate generator reads. Both are
    leakage, and the second one would quietly invalidate the holdout split.
    """
    cfg = cfg or Config()
    if journal_dir:
        cfg = cfg.with_overrides(journal_dir=str(journal_dir))
    started = time.time()
    request_id = f"bench_{task.task_id[:18]}_{uuid.uuid4().hex[:8]}"

    with use_policy(policy):
        try:
            with _quiet():
                outcome = asyncio.run(
                    asyncio.wait_for(
                        _run_program(task, cfg=cfg, cycles=cycles, request_id=request_id),
                        timeout=timeout_s,
                    )
                )
        except Exception as exc:
            return TaskResult(
                task_id=task.task_id,
                arm=arm,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                gold_total=len(task.gold),
                wall_s=round(time.time() - started, 3),
                # Every exit from this function carries an endpoint record, so a
                # reader never has to infer a missing measurement from a default.
                independent={"endpoint": endpoint_result("FAILED", None)},
            )

        result = outcome.get("result") or {}
        if not result:
            return TaskResult(
                task_id=task.task_id,
                arm=arm,
                ok=False,
                error=outcome.get("error") or "the program produced no result",
                gold_total=len(task.gold),
                wall_s=round(time.time() - started, 3),
                journal_dir=outcome.get("journal_dir", ""),
                independent={"endpoint": endpoint_result("FAILED", None)},
            )

        best = result.get("best") or {}
        requirements = best.get("requirements") or {}
        # The *effective* decision, not the raw critique. The master agent
        # overrides a REFINE to ACCEPT when every success criterion was met
        # (`master.py`: `if report.all_criteria_met: verdict = "ACCEPT"`), and
        # reading the critique alone would miss exactly the case a candidate
        # that makes criteria reachable is supposed to produce. Both are kept.
        critique_verdict = ((best.get("critique") or {}).get("overall_verdict")) or result.get("verdict")
        all_criteria_met = bool((best.get("evaluation") or {}).get("all_criteria_met"))
        verdict = "ACCEPT" if all_criteria_met else critique_verdict

        # --- the kernel's own execution ---------------------------------
        try:
            with _quiet():
                req = ChemicalRequirements.model_validate(requirements)
                from ...llm.bridge import requirements_to_brief_dict, run_pipeline

                brief, _seeds, _dropped = requirements_to_brief_dict(req, cfg)
                run = run_pipeline(req, cfg)
                measured = kernel_metrics(run, req.shortlist_size, brief)
        except Exception as exc:
            return TaskResult(
                task_id=task.task_id,
                arm=arm,
                ok=False,
                error=f"kernel re-execution failed: {type(exc).__name__}: {exc}",
                gold_total=len(task.gold),
                wall_s=round(time.time() - started, 3),
                journal_dir=outcome.get("journal_dir", ""),
                independent={"endpoint": endpoint_result("FAILED", None)},
            )

        # The re-execution can *return* without succeeding: `run_pipeline` hands
        # back a BackendRun whose state is FAILED and whose portfolio is empty.
        # That used to fall straight through to `ok=True` with a metric row of
        # zeros, and `aggregate` then averaged those zeros into the arm's score.
        execution = kernel_execution_endpoint(run)
        if execution["status"] != "SUCCEEDED":
            return TaskResult(
                task_id=task.task_id,
                arm=arm,
                ok=False,
                error=str(execution["error_code"]),
                gold_total=len(task.gold),
                wall_s=round(time.time() - started, 3),
                journal_dir=outcome.get("journal_dir", ""),
                independent={"endpoint": execution},
                self_reported={"verdict": verdict, "critique_verdict": critique_verdict},
            )

    metrics = measured["metrics"]
    gold_rows, met, scored, unscorable, invalid = score_gold(task.gold, metrics)

    gold_total = len(task.gold)
    gold_score = round(met / gold_total, 4) if gold_total else 0.0
    gold_pass = gold_score >= GOLD_PASS_THRESHOLD
    accepted = verdict == "ACCEPT"

    self_reported = {
        "verdict": verdict,
        "critique_verdict": critique_verdict,
        "all_criteria_met": all_criteria_met,
        "n_criteria_met": (best.get("evaluation") or {}).get("n_criteria_met"),
        "n_criteria_total": (best.get("evaluation") or {}).get("n_criteria_total"),
        "criteria": [
            {"metric": c.get("key"), "target": c.get("target"), "value": c.get("value"), "met": c.get("met")}
            for c in ((best.get("evaluation") or {}).get("criteria") or [])
        ],
        "headline_utility": (best.get("evaluation") or {}).get("headline_utility"),
        "kind": requirements.get("kind"),
        "expected_kind": task.expected_kind,
        "kind_correct": requirements.get("kind") == task.expected_kind,
        "policy": (result.get("policy") or {}).get("behaviour_version"),
        "prompt_version": (result.get("policy") or {}).get("prompt_version"),
        "cycles_run": len(result.get("cycles") or []),
        "unscorable_criteria": [c.get("metric") for c in (result.get("unscorable_criteria") or [])],
    }

    reported_criteria: list[dict[str, Any]] = list(self_reported["criteria"])  # type: ignore[arg-type]
    infeasible, n_targets = _target_feasibility(reported_criteria, metrics)
    cost = _cost_from_journal(outcome.get("journal_dir", ""), result)

    return TaskResult(
        task_id=task.task_id,
        arm=arm,
        ok=True,
        gold_met=met,
        gold_total=gold_total,
        gold_score=gold_score,
        verdict=verdict,
        false_accept=bool(accepted and not gold_pass),
        false_refine=bool(not accepted and gold_pass),
        independent={
            **measured["independent"],
            "gold": gold_rows,
            "gold_scored": scored,
            "gold_unscorable": unscorable,
            "gold_invalid": invalid,
            "gold_pass": gold_pass,
            "cross_check": cross_check(metrics),
            # The kernel run's identity and its state, which used to be one
            # field: `kernel_run_id` held `metrics["state"]`, so the run could
            # not be found and its state was filed under the wrong name.
            "kernel_run_id": execution["kernel_run_id"],
            "kernel_state": execution["kernel_state"],
            "endpoint": endpoint_result("SUCCEEDED", gold_score),
            "metrics": metrics,
        },
        self_reported={
            **self_reported,
            "cost_source": cost["source"],
            "llm_seconds": cost.get("llm_seconds"),
            "n_repairs": cost.get("n_repairs"),
            "n_decode_failures": cost.get("n_decode_failures"),
        },
        llm_calls=cost["llm_calls"],
        eval_tokens=cost["eval_tokens"],
        wall_s=round(time.time() - started, 3),
        tool_calls=cost["tool_calls"],
        infeasible_targets=infeasible,
        n_targets=n_targets,
        journal_dir=outcome.get("journal_dir", ""),
    )


def _target_feasibility(criteria: list[dict[str, Any]], metrics: dict[str, Any]) -> tuple[int, int]:
    """How many of the agent's own targets were unreachable *in this run*.

    Compared against what the run actually achieved rather than against the
    envelope, so it stays a property of the episode. A diagnostic, never a gold
    criterion: an arm cannot improve it by aiming low, because aiming low does
    not move the gold rubric.
    """
    from ...selfevolve.analysis.analyzer import parse_target

    n = 0
    infeasible = 0
    for crit in criteria:
        metric = str(crit.get("metric") or "")
        parsed = parse_target(str(crit.get("target") or ""))
        observed = metrics.get(metric)
        if parsed is None or observed is None:
            continue
        n += 1
        op, threshold = parsed
        gap = float(observed) - threshold
        missed = (
            (op in (">=", ">") and gap < 0)
            or (op in ("<=", "<") and gap > 0)
            or (op == "==" and abs(gap) > 1e-9)
        )
        if missed:
            infeasible += 1
    return infeasible, n


def _cost_from_journal(journal_dir: str, result: dict[str, Any]) -> dict[str, Any]:
    """Model calls, tokens and tool calls, read from the journal the run wrote.

    The journal is the measurement, not the result payload: it is written by the
    platform's own instrumentation as the run happens, and it records the token
    counts the model reported rather than any count reconstructed afterwards. If
    it is missing (journaling disabled), the shape of the program still gives an
    exact call count — plan + transcribe + one critique per cycle + one refine
    per non-final cycle — and that fallback is flagged.
    """
    import json
    from pathlib import Path

    n_cycles = len(result.get("cycles") or [])
    fallback_calls = 2 + n_cycles + max(0, n_cycles - 1)
    manifest = Path(journal_dir) / "manifest.json" if journal_dir else None
    if manifest is None or not manifest.is_file():
        return {
            "llm_calls": fallback_calls,
            "eval_tokens": 0,
            "tool_calls": 0,
            "llm_seconds": 0.0,
            "source": "reconstructed (no journal)",
        }
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "llm_calls": fallback_calls,
            "eval_tokens": 0,
            "tool_calls": 0,
            "llm_seconds": 0.0,
            "source": "reconstructed (unreadable journal)",
        }
    llm = data.get("llm") or {}
    tools = data.get("tools") or {}
    return {
        "llm_calls": int(llm.get("n_exchanges") or fallback_calls),
        "eval_tokens": int(llm.get("eval_tokens") or 0),
        "tool_calls": int(tools.get("n_calls") or 0),
        "llm_seconds": float(llm.get("seconds") or 0.0),
        "n_repairs": int(llm.get("n_repairs") or 0),
        "n_decode_failures": int(llm.get("n_decode_failures") or 0),
        "source": "journal",
    }


__all__ = ["TaskRunError", "run_task"]
