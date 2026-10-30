"""Promotion: comparison, statistics, and the checks that must all pass.

The comparison is **paired by task**. Both arms run the same six benchmark
tasks, so the honest statistic is the per-task difference, not the difference of
two independent means. At n=6 a paired bootstrap interval is wide and the gate
says so rather than dressing six numbers up as significance.

The checks are ordered so the cheap and disqualifying ones come first, and each
one names what it would take to pass:

1. the evaluation kernel verified — the candidate could not have edited its judge;
2. both arms ran the same benchmark version *and the same kernel build*, so the
   numbers are comparable;
3. the candidate did not touch the guarded file set;
4. any tool the candidate changed clears a **held-out** conformance suite the
   candidate did not write — passing the call that motivated a tool certifies
   that call, not the capability (arXiv:2604.00392);
5. gate 0 (the platform's own test suite) passed, with the exit status recorded;
6. the primary metric improved by at least ``min_effect``;
7. **no guarded metric regressed** — this is the anti-gaming check, and it is why
   the primary metric being "the agent's verdict agrees with an independent
   rubric" cannot be won by making the agent's own criteria easier;
8. the improvement is not paid for entirely in compute;
9. the result is reproducible — the record carries the commands and the
   environment needed to re-run it.

Nothing here promotes automatically. ``auto_promote`` defaults to false and the
decision that comes back is ``NEEDS_HUMAN`` even when everything passes, because
a self-modifying system that installs its own changes into production is the one
thing the brief is unambiguous about.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any

from ..config import EvolveConfig
from ..schemas import Candidate, EvaluationRecord, PromotionDecision

#: The metric a candidate must improve. Chosen because the historical record
#: says it is broken (0 of 5 requests accepted, several with good portfolios),
#: and because it cannot be moved by weakening the agent's own criteria — it is
#: scored against a rubric the candidate never sees.
PRIMARY_METRIC = "decision_accuracy"

#: Metrics that must not get worse. ``gold_score`` is the guard that makes the
#: primary metric safe: it measures the molecules that came out, so an arm that
#: buys agreement by aiming low loses here.
GUARDED_METRICS = ("gold_score", "kind_accuracy")

#: Reported, weighed, but never automatically decisive.
COST_METRICS = ("mean_llm_calls", "mean_eval_tokens", "mean_wall_s")


def _task_map(record: EvaluationRecord) -> dict[str, Any]:
    return {t.task_id: t for t in record.tasks if t.ok}


def _decision_correct(task: Any) -> float:
    return 0.0 if (task.false_accept or task.false_refine) else 1.0


_PER_TASK = {
    "decision_accuracy": _decision_correct,
    "gold_score": lambda t: float(t.gold_score),
    "kind_accuracy": lambda t: 1.0 if t.self_reported.get("kind_correct") else 0.0,
    "mean_llm_calls": lambda t: float(t.llm_calls),
    "mean_eval_tokens": lambda t: float(t.eval_tokens),
    "mean_wall_s": lambda t: float(t.wall_s),
    "infeasible_target_rate": lambda t: (t.infeasible_targets / t.n_targets) if t.n_targets else 0.0,
}


def paired_bootstrap(diffs: list[float], *, resamples: int = 2000, seed: int = 20260827) -> dict[str, Any]:
    """Percentile bootstrap over the per-task differences.

    Reported with the sample size attached and an explicit warning below n=10,
    because a 95% interval from six paired observations is a description of six
    numbers, not evidence about a population.
    """
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None, "note": "no paired tasks"}
    mean = statistics.mean(diffs)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(max(1, resamples)):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    out: dict[str, Any] = {
        "n": n,
        "mean": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": sum(1 for d in diffs if d == 0),
        "resamples": resamples,
    }
    if n < 10:
        out["warning"] = (
            f"n={n}: this interval describes {n} paired observations. It is not a significance "
            "test and must not be read as one."
        )
    if n >= 6:
        out["sign_test_p"] = _sign_test_p(n_pos, n_neg)
    return out


def _sign_test_p(pos: int, neg: int) -> float | None:
    """Two-sided exact sign test over non-zero differences."""
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return round(min(1.0, 2.0 * tail / (2**n)), 4)


def compare(
    baseline: EvaluationRecord, candidate: EvaluationRecord, cfg: EvolveConfig | None = None
) -> dict[str, Any]:
    """Paired comparison of two arms over the tasks both completed."""
    cfg = cfg or EvolveConfig()
    b_tasks, c_tasks = _task_map(baseline), _task_map(candidate)
    shared = sorted(set(b_tasks) & set(c_tasks))
    metrics: dict[str, Any] = {}
    for name, extract in _PER_TASK.items():
        diffs = [extract(c_tasks[t]) - extract(b_tasks[t]) for t in shared]
        metrics[name] = {
            "baseline": round(statistics.mean([extract(b_tasks[t]) for t in shared]), 4) if shared else None,
            "candidate": round(statistics.mean([extract(c_tasks[t]) for t in shared]), 4) if shared else None,
            "delta": round(statistics.mean(diffs), 4) if diffs else None,
            "paired": paired_bootstrap(diffs, resamples=cfg.bootstrap_resamples),
            "per_task": {t: round(d, 4) for t, d in zip(shared, diffs, strict=False)},
        }
    by_split: dict[str, dict[str, Any]] = {}
    from ..evaluation.kernel import _split_of

    for split in sorted({_split_of(t) for t in shared}):
        ids = [t for t in shared if _split_of(t) == split]
        by_split[split] = {
            name: {
                "baseline": round(statistics.mean([extract(b_tasks[t]) for t in ids]), 4),
                "candidate": round(statistics.mean([extract(c_tasks[t]) for t in ids]), 4),
                "delta": round(statistics.mean([extract(c_tasks[t]) - extract(b_tasks[t]) for t in ids]), 4),
                "n": len(ids),
            }
            for name, extract in _PER_TASK.items()
            if name in (PRIMARY_METRIC, *GUARDED_METRICS)
        }
    return {
        "n_paired_tasks": len(shared),
        "tasks": shared,
        "baseline_only": sorted(set(b_tasks) - set(c_tasks)),
        "candidate_only": sorted(set(c_tasks) - set(b_tasks)),
        "metrics": metrics,
        "by_split": by_split,
        "baseline_record": baseline.record_id,
        "candidate_record": candidate.record_id,
        "primary_metric": PRIMARY_METRIC,
        "guarded_metrics": list(GUARDED_METRICS),
    }


GATE_CHECKS = (
    "kernel_verified",
    "same_benchmark",
    "kernel_untouched",
    "tool_conformance",
    "invariants_pass",
    "primary_improved",
    "no_guarded_regression",
    "cost_acceptable",
    "reproducible",
    "holdout_survives",
)


def decide(
    candidate: Candidate,
    baseline: EvaluationRecord,
    candidate_record: EvaluationRecord,
    cfg: EvolveConfig | None = None,
) -> PromotionDecision:
    """Run every gate check and return the decision plus the evidence for it."""
    cfg = cfg or EvolveConfig()
    from ..evaluation.integrity import candidate_touches_kernel

    cmp = compare(baseline, candidate_record, cfg)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str, **extra: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail, **extra})

    check(
        "kernel_verified",
        baseline.kernel_verified and candidate_record.kernel_verified,
        "the evaluation kernel was byte-identical before and after both arms",
        baseline_digest=baseline.kernel_digest[:16],
        candidate_digest=candidate_record.kernel_digest[:16],
    )
    same_bench = (baseline.benchmark_id, baseline.benchmark_version) == (
        candidate_record.benchmark_id,
        candidate_record.benchmark_version,
    )
    same_kernel = baseline.kernel_digest == candidate_record.kernel_digest
    check(
        "same_benchmark",
        same_bench and same_kernel,
        f"{baseline.benchmark_id}@{baseline.benchmark_version} vs "
        f"{candidate_record.benchmark_id}@{candidate_record.benchmark_version}"
        + ("" if same_kernel else "; the two arms were measured by DIFFERENT kernel builds"),
        same_kernel=same_kernel,
    )
    touched = candidate_touches_kernel(candidate.files_changed)
    check(
        "kernel_untouched",
        not touched,
        "candidate changes no file inside the integrity manifest"
        if not touched
        else f"candidate touches guarded files: {touched}",
    )
    from ..evaluation.conformance import run_for_candidate

    conformance = run_for_candidate(candidate)
    check(
        "tool_conformance",
        conformance["passed"],
        (
            "this candidate changes no tool implementation"
            if not conformance["required"]
            else f"held-out conformance for {conformance['capabilities']}: "
            + ("passed" if conformance["passed"] else "FAILED")
        ),
        conformance=conformance,
    )

    invariants = [c for c in candidate_record.commands if c.get("label")]
    check(
        "invariants_pass",
        bool(invariants) and all(c.get("ok") for c in invariants),
        "gate 0 commands executed by the kernel, with exit status recorded"
        if invariants
        else "no invariant commands were executed for this candidate",
        commands=[{"label": c["label"], "exit_status": c["exit_status"]} for c in invariants],
    )

    primary = cmp["metrics"].get(PRIMARY_METRIC, {})
    delta = primary.get("delta")
    check(
        "primary_improved",
        delta is not None and delta > cfg.min_effect,
        f"{PRIMARY_METRIC} {primary.get('baseline')} → {primary.get('candidate')} "
        f"(Δ {delta}, min_effect {cfg.min_effect})",
        paired=primary.get("paired"),
    )

    regressions = []
    for name in GUARDED_METRICS:
        d = cmp["metrics"].get(name, {}).get("delta")
        if d is not None and d < -cfg.max_regression:
            regressions.append({"metric": name, "delta": d})
    check(
        "no_guarded_regression",
        not regressions,
        "no guarded metric regressed beyond tolerance"
        if not regressions
        else f"guarded regression: {regressions}",
        guarded={n: cmp["metrics"].get(n, {}).get("delta") for n in GUARDED_METRICS},
    )

    cost = {n: cmp["metrics"].get(n, {}).get("delta") for n in COST_METRICS}
    calls_delta = cost.get("mean_llm_calls") or 0.0
    check(
        "cost_acceptable",
        calls_delta <= 0.5,
        f"model calls per task Δ {calls_delta}; token Δ {cost.get('mean_eval_tokens')}; "
        f"wall Δ {cost.get('mean_wall_s')}s",
        cost=cost,
    )
    check(
        "reproducible",
        bool(candidate_record.environment.get("git_commit") is not None) and bool(candidate_record.tasks),
        "the record carries the environment, the benchmark fingerprint and per-task rows",
        environment={
            k: candidate_record.environment.get(k)
            for k in ("python", "git_commit", "ollama_model", "benchmark")
        },
    )

    holdout = cmp["by_split"].get("holdout", {}).get(PRIMARY_METRIC, {})
    holdout_delta = holdout.get("delta")
    check(
        "holdout_survives",
        holdout_delta is None or holdout_delta >= 0.0,
        f"holdout split ({holdout.get('n', 0)} tasks): {PRIMARY_METRIC} Δ {holdout_delta}"
        if holdout
        else "no holdout tasks in this comparison",
        holdout=holdout,
    )

    failed = [c["check"] for c in checks if not c["passed"]]
    hard_fail = {"kernel_verified", "same_benchmark", "kernel_untouched"} & set(failed)

    if hard_fail:
        decision, reasons = "REJECT", [f"disqualifying check failed: {sorted(hard_fail)}"]
    elif not failed:
        decision = "PROMOTE" if cfg.auto_promote else "NEEDS_HUMAN"
        reasons = [
            "every gate check passed",
            f"{PRIMARY_METRIC} improved by {delta} over {cmp['n_paired_tasks']} paired tasks",
            "guarded metrics held",
        ]
        if cmp["n_paired_tasks"] < 10:
            reasons.append(
                f"sample size is {cmp['n_paired_tasks']} paired tasks — the effect is measured, "
                "not established; a human should read the per-task rows before promoting"
            )
        if not cfg.auto_promote:
            reasons.append("auto_promote is off, which is the default: promotion needs a human")
    elif failed == ["primary_improved"] and not regressions:
        decision, reasons = (
            "ARCHIVE",
            ["no improvement on the primary metric, but nothing regressed — kept as a stepping stone"],
        )
    else:
        decision, reasons = "REJECT", [f"checks failed: {failed}"]

    return PromotionDecision(
        candidate_id=candidate.candidate_id,
        decision=decision,
        reasons=reasons,
        checks=checks,
        baseline_record=baseline.record_id,
        candidate_record=candidate_record.record_id,
        comparison=cmp,
        decided_at=time.time(),
        requires_human=not cfg.auto_promote,
    )


def render_report(candidate: Candidate, decision: PromotionDecision) -> str:
    """The human-readable promotion report."""
    cmp = decision.comparison
    lines = [
        f"# Promotion report — {candidate.candidate_id}",
        "",
        f"**Decision: {decision.decision}**"
        + ("  (human approval required)" if decision.requires_human else ""),
        "",
        f"- class: `{candidate.candidate_class}`  ·  cell: `{candidate.cell()}`",
        f"- parent: `{candidate.parent_id or '—'}`  ·  generation {candidate.generation}",
        f"- overlay: `{candidate.overlay}`",
        "",
        "## Hypothesis",
        "",
        candidate.hypothesis or "(none recorded)",
        "",
        "## Evidence",
        "",
        f"- baseline record `{cmp.get('baseline_record')}`",
        f"- candidate record `{cmp.get('candidate_record')}`",
        f"- {cmp.get('n_paired_tasks', 0)} paired tasks: {', '.join(cmp.get('tasks', []))}",
        "",
        "| metric | baseline | candidate | Δ | paired 95% CI | n+/n− |",
        "| --- | --: | --: | --: | --- | --- |",
    ]
    for name, row in (cmp.get("metrics") or {}).items():
        paired = row.get("paired") or {}
        ci = paired.get("ci95")
        marker = (
            " **(primary)**"
            if name == cmp.get("primary_metric")
            else (" *(guarded)*" if name in (cmp.get("guarded_metrics") or []) else "")
        )
        lines.append(
            f"| {name}{marker} | {row.get('baseline')} | {row.get('candidate')} | {row.get('delta')} | "
            f"{ci if ci else '—'} | {paired.get('n_positive', 0)}/{paired.get('n_negative', 0)} |"
        )
    if cmp.get("by_split"):
        lines += [
            "",
            "### By split",
            "",
            "| split | metric | baseline | candidate | Δ | n |",
            "| --- | --- | --: | --: | --: | --: |",
        ]
        for split, rows in cmp["by_split"].items():
            for name, row in rows.items():
                lines.append(
                    f"| {split} | {name} | {row['baseline']} | {row['candidate']} | {row['delta']} "
                    f"| {row['n']} |"
                )
    lines += ["", "## Gate checks", "", "| check | result | detail |", "| --- | --- | --- |"]
    for c in decision.checks:
        lines.append(f"| `{c['check']}` | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    lines += ["", "## Reasons", ""]
    lines += [f"- {r}" for r in decision.reasons]
    lines += [
        "",
        "## Possible regressions this candidate declared",
        "",
    ]
    lines += [f"- {r}" for r in (candidate.possible_regressions or ["(none declared)"])]
    return "\n".join(lines)


__all__ = [
    "COST_METRICS",
    "GATE_CHECKS",
    "GUARDED_METRICS",
    "PRIMARY_METRIC",
    "compare",
    "decide",
    "paired_bootstrap",
    "render_report",
]
