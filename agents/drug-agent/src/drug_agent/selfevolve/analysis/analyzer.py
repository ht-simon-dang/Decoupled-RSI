"""Failure detectors over normalized episodes.

Each detector is a small function that answers one question with numbers, cites
the episodes it looked at, and returns ``None`` when the question does not apply
to the data available. That shape is the point:

* a detector that cannot compute its statistic returns nothing rather than a
  hedge, so the report never contains a claim with no arithmetic behind it;
* every :class:`~drug_agent.selfevolve.schemas.Finding` carries
  ``n_affected / n_applicable`` and an ``evidence`` list of episode ids, so a
  reader can check it in one command;
* a detector that can only observe co-occurrence sets ``inference="association"``.
  Only :mod:`~drug_agent.selfevolve.analysis.attribution` ever upgrades that, and
  only where a matched comparison exists.

The detectors are deliberately *specific*. A generic "the LLM sometimes gets
things wrong" detector is unfalsifiable and unactionable; "five of five plans set
a potency target above the measured ceiling of 6.4582" names a mechanism, a
number, and a fix.
"""

from __future__ import annotations

import itertools
import re
import statistics
from collections.abc import Callable
from typing import Any

from ..schemas import Episode, Finding, Severity

#: Which tool produces each scored endpoint — used to tell when a success
#: criterion is being scored against a mocked backend.
ENDPOINT_TOOL: dict[str, str] = {
    "max_pIC50": "model.activity",
    "median_pIC50": "model.activity",
    "in_domain_fraction": "model.activity",
    "qed_median": "chem.standardize",
    "sa_median": "synth.assess",
    "herg_median": "safety.assess",
    "restoration_median": "perturb.assess",
    "mean_pairwise_distance": "chem.standardize",
    "n_scaffold_clusters": "chem.standardize",
}

_TARGET = re.compile(r"(<=|>=|==|!=|<|>)?\s*([-+]?\d*\.?\d+)")

#: Metrics where a *lower* value is better, so "target beyond the envelope"
#: means below the achievable value rather than above it.
_MINIMIZE = frozenset({"sa_median", "herg_median"})


def parse_target(target: str) -> tuple[str, float] | None:
    m = _TARGET.search(target or "")
    if not m:
        return None
    try:
        return (m.group(1) or ">="), float(m.group(2))
    except ValueError:
        return None


def _feasible(metric: str, op: str, threshold: float, achievable: float) -> bool:
    """Could a criterion of this shape ever be met on this deployment?

    ``achievable`` is the envelope's best-achievable value for the metric. The
    comparison respects the metric's direction: ``sa_median <= 3.0`` is
    infeasible when the best (lowest) achievable median is 5.1, and
    ``max_pIC50 >= 7.5`` is infeasible when the ceiling is 6.46.
    """
    minimize = metric in _MINIMIZE
    if op in (">=", ">"):
        return achievable >= threshold if not minimize else True
    if op in ("<=", "<"):
        return achievable <= threshold if minimize else True
    if op == "==":
        # An equality target is only reachable if the envelope reaches it.
        if abs(achievable - threshold) < 1e-9:
            return True
        return achievable <= threshold if minimize else achievable >= threshold
    return True


def _crit_pairs(ep: Episode) -> list[tuple[str, str]]:
    return [(str(c.get("metric") or ""), str(c.get("target") or "")) for c in ep.criteria]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def detect_infeasible_targets(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Success criteria set beyond what this deployment can reach.

    The single most consequential pattern in the historical record, and the one
    the platform had no way to notice: a target outside the envelope cannot be
    met by any amount of refinement, so it does not make the program stricter,
    only unsatisfiable — and the run burns its whole cycle budget discovering
    that.
    """
    envelope: dict[str, Any] = ctx.get("envelope") or {}
    metrics = envelope.get("metrics") or {}
    if not metrics:
        return None
    infeasible: list[dict[str, Any]] = []
    n_criteria = 0
    affected: list[str] = []
    for ep in episodes:
        hit = False
        for metric, target in _crit_pairs(ep):
            rng = metrics.get(metric)
            if not rng or not rng.get("available") or rng.get("best") is None:
                continue
            parsed = parse_target(target)
            if parsed is None:
                continue
            n_criteria += 1
            op, threshold = parsed
            if not _feasible(metric, op, threshold, float(rng["best"])):
                infeasible.append(
                    {
                        "episode": ep.episode_id,
                        "metric": metric,
                        "target": target,
                        "achievable": rng["best"],
                        "mocked": bool(rng.get("mocked")),
                    }
                )
                hit = True
        if hit:
            affected.append(ep.episode_id)
    if not infeasible:
        return None
    by_metric: dict[str, int] = {}
    for row in infeasible:
        by_metric[row["metric"]] = by_metric.get(row["metric"], 0) + 1
    return Finding(
        finding_id="infeasible_targets",
        title="Plans set success targets outside the deployment's measured capability envelope",
        detector="detect_infeasible_targets",
        severity=Severity.HIGH,
        n_affected=len(affected),
        n_applicable=len(episodes),
        stats={
            "n_infeasible_criteria": len(infeasible),
            "n_scored_criteria": n_criteria,
            "infeasible_rate": round(len(infeasible) / n_criteria, 4) if n_criteria else None,
            "by_metric": dict(sorted(by_metric.items(), key=lambda kv: -kv[1])),
            "examples": infeasible[:12],
            "envelope_source": envelope.get("fingerprint"),
        },
        evidence=affected,
        detail=(
            "The planner has no information about what this deployment can produce, so it sets "
            "targets from general medicinal-chemistry practice. Every criterion outside the "
            "envelope is guaranteed to read as missed."
        ),
    )


def detect_never_accepted(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """The user-serving loop never terminates successfully."""
    done = [ep for ep in episodes if ep.cycles]
    if not done:
        return None
    accepted = [ep for ep in done if ep.accepted]
    burned = [ep for ep in done if ep.max_cycles and len(ep.cycles) >= ep.max_cycles and not ep.accepted]
    return Finding(
        finding_id="never_accepted",
        title="Programs never reach ACCEPT and burn the whole cycle budget",
        detector="detect_never_accepted",
        severity=Severity.HIGH if not accepted else Severity.MEDIUM,
        n_affected=len(done) - len(accepted),
        n_applicable=len(done),
        stats={
            "accept_rate": round(len(accepted) / len(done), 4),
            "budget_burn_rate": round(len(burned) / len(done), 4),
            "mean_criteria_met_rate": _mean([ep.criteria_met_rate for ep in done]),
            "mean_llm_calls": _mean([float(ep.n_llm_calls) for ep in done]),
            "mean_wall_s": _mean([ep.duration_s for ep in done]),
            "verdicts": _counts([ep.final_verdict for ep in done]),
        },
        evidence=[ep.episode_id for ep in done if not ep.accepted],
        detail=(
            "A loop whose exit condition is never reached is a pipeline with extra steps. "
            "Cost is the whole budget every time; the user waits for cycles that cannot change "
            "the verdict."
        ),
    )


def detect_goal_metric_mismatch(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Efficacy criteria that cannot apply to the program's goal kind."""
    from ...llm.repair import inapplicable_metrics

    rows: list[dict[str, Any]] = []
    affected: list[str] = []
    applicable = 0
    for ep in episodes:
        kind = str((ep.requirements or {}).get("kind") or "")
        if not kind:
            continue
        applicable += 1
        blocked = inapplicable_metrics(kind)
        bad = [m for m, _t in _crit_pairs(ep) if m in blocked]
        if bad:
            affected.append(ep.episode_id)
            rows.append({"episode": ep.episode_id, "kind": kind, "metrics": bad})
    if not rows:
        return None
    return Finding(
        finding_id="goal_metric_mismatch",
        title="Plans score efficacy metrics that the program's goal kind cannot produce",
        detector="detect_goal_metric_mismatch",
        severity=Severity.HIGH,
        n_affected=len(affected),
        n_applicable=applicable,
        stats={"cases": rows, "n_criteria": sum(len(r["metrics"]) for r in rows)},
        evidence=affected,
        detail=(
            "The planning prompt states the rule explicitly and the model did not follow it. "
            "The goal kind is not known until transcription, so the platform cannot repair this "
            "at plan time — but it can at dispatch time, where it already excludes metrics it "
            "cannot compute at all."
        ),
    )


def detect_silent_requirement_loss(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Refinements that delete requirement entries they did not report changing.

    The schema gives ``objectives`` and ``property_constraints`` a default of
    ``[]``, so an omission in a schema-constrained decode *is* a deletion. When
    that happens the next cycle scores a different program from the one the
    critique diagnosed.
    """
    rows: list[dict[str, Any]] = []
    affected: list[str] = []
    n_adjustments = 0
    for ep in episodes:
        for cyc in ep.cycles:
            if not cyc.adjustment_diff:
                continue
            n_adjustments += 1
            removed = [
                d
                for d in cyc.adjustment_diff
                if str(d.get("new")) == "removed" or str(d.get("new")).strip() == "—"
            ]
            if not removed:
                continue
            named = " ".join(
                str(c.get("parameter", "")) + " " + str(c.get("reason", "")) for c in cyc.adjustment_changes
            ).lower()
            unreported = [d for d in removed if _field_key(d.get("field", "")) not in named]
            if unreported:
                affected.append(ep.episode_id)
                rows.append(
                    {
                        "episode": ep.episode_id,
                        "cycle": cyc.cycle,
                        "n_removed": len(removed),
                        "n_unreported": len(unreported),
                        "n_reported_changes": len(cyc.adjustment_changes),
                        "fields": [d.get("field") for d in unreported][:12],
                    }
                )
    if not rows:
        return None
    return Finding(
        finding_id="silent_requirement_loss",
        title="Adaptive refinement deletes objectives and constraints it never reported changing",
        detector="detect_silent_requirement_loss",
        severity=Severity.HIGH,
        n_affected=len(set(affected)),
        n_applicable=n_adjustments,
        stats={"cases": rows, "n_adjustments": n_adjustments},
        evidence=sorted(set(affected)),
        detail=(
            "Omission is deletion under a schema whose list fields default to empty. The visible "
            "consequence is a metric that improves for the wrong reason: with every hard "
            "constraint gone, hard_constraint_pass_rate reads 1.0."
        ),
    )


def _field_key(field: str) -> str:
    """``objective:qed`` / ``constraint:descriptors.mw`` → the identifier part."""
    return str(field).split(":", 1)[-1].strip().lower()


def detect_metric_gaming(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """A metric that improved because the thing it measures was removed.

    Specifically: ``hard_constraint_pass_rate`` rising to 1.0 in the same cycle
    that the constraint list was emptied. This is the user-serving system doing,
    unprompted, exactly what the brief forbids the *evolver* from doing — which
    is why the benchmark in this package is scored against a rubric no arm can
    see.
    """
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        prev_rate: float | None = None
        prev_diff: list[dict[str, str]] = []
        for cyc in ep.cycles:
            rate = cyc.metrics.get("hard_constraint_pass_rate")
            if prev_rate is not None and rate is not None:
                removed_constraints = [
                    d
                    for d in prev_diff
                    if str(d.get("field", "")).startswith("constraint:") and str(d.get("new")) == "removed"
                ]
                if removed_constraints and rate > prev_rate and abs(rate - 1.0) < 1e-9:
                    rows.append(
                        {
                            "episode": ep.episode_id,
                            "cycle": cyc.cycle,
                            "before": prev_rate,
                            "after": rate,
                            "constraints_removed": len(removed_constraints),
                        }
                    )
            prev_rate = rate
            prev_diff = list(cyc.adjustment_diff)
    if not rows:
        return None
    return Finding(
        finding_id="metric_gaming_observed",
        title="A success metric improved because its subject was deleted, not because the portfolio improved",
        detector="detect_metric_gaming",
        severity=Severity.HIGH,
        n_affected=len({r["episode"] for r in rows}),
        n_applicable=len(episodes),
        stats={"cases": rows},
        evidence=sorted({r["episode"] for r in rows}),
        detail=(
            "hard_constraint_pass_rate reached 1.0 in the cycle that emptied the constraint list. "
            "Nothing about the candidates changed for the better."
        ),
    )


def detect_refinement_ineffective(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Cycle-over-cycle change in criteria met, by refinement strategy."""
    deltas: list[dict[str, Any]] = []
    for ep in episodes:
        scored = [c for c in ep.cycles if c.n_criteria_met is not None]
        for a, b in itertools.pairwise(scored):
            deltas.append(
                {
                    "episode": ep.episode_id,
                    "strategy": ep.strategy,
                    "from": a.cycle,
                    "to": b.cycle,
                    "d_met": (b.n_criteria_met or 0) - (a.n_criteria_met or 0),
                    "d_utility": _sub(b.headline_utility, a.headline_utility),
                }
            )
    if not deltas:
        return None
    improved = [d for d in deltas if d["d_met"] > 0]
    return Finding(
        finding_id="refinement_ineffective",
        title="A second cycle does not increase the number of criteria met",
        detector="detect_refinement_ineffective",
        severity=Severity.MEDIUM,
        n_affected=len(deltas) - len(improved),
        n_applicable=len(deltas),
        stats={
            "transitions": deltas,
            "n_improved": len(improved),
            "mean_d_met": _mean([float(d["d_met"]) for d in deltas]),
            "by_strategy": {
                s: _mean([float(d["d_met"]) for d in deltas if d["strategy"] == s])
                for s in sorted({str(d["strategy"]) for d in deltas})
            },
        },
        evidence=sorted({d["episode"] for d in deltas}),
        detail=(
            "With so few transitions this is a description, not a test. It is reported because "
            "the extra cycle is the single largest cost item in a request."
        ),
        inference="association",
    )


def detect_unscorable_criteria(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Criteria naming a metric outside the evaluation agent's vocabulary."""
    rows = [
        {"episode": ep.episode_id, "metrics": [c.get("metric") for c in ep.unscorable_criteria]}
        for ep in episodes
        if ep.unscorable_criteria
    ]
    if not rows:
        return None
    return Finding(
        finding_id="unscorable_criteria",
        title="Plans name success metrics the evaluation agent cannot compute",
        detector="detect_unscorable_criteria",
        severity=Severity.LOW,
        n_affected=len(rows),
        n_applicable=len(episodes),
        stats={"cases": rows},
        evidence=[r["episode"] for r in rows],
        detail="Already handled by the master agent, which excludes and reports them. Tracked to "
        "confirm the handling keeps working.",
    )


def detect_invalid_seeds(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Seed SMILES the model proposed that RDKit rejected."""
    rows: list[dict[str, Any]] = []
    total_seeds = 0
    for ep in episodes:
        seeds = (ep.requirements or {}).get("seed_smiles") or []
        total_seeds += len(seeds)
        dropped = _dropped_seeds(ep)
        if dropped:
            rows.append({"episode": ep.episode_id, "dropped": dropped, "proposed": len(seeds)})
    if not rows:
        return None
    n_dropped = sum(len(r["dropped"]) for r in rows)
    return Finding(
        finding_id="invalid_seed_smiles",
        title="Proposed seed structures that RDKit could not parse",
        detector="detect_invalid_seeds",
        severity=Severity.LOW,
        n_affected=len(rows),
        n_applicable=len(episodes),
        stats={
            "n_dropped": n_dropped,
            "n_proposed": total_seeds,
            "drop_rate": round(n_dropped / total_seeds, 4) if total_seeds else None,
            "cases": rows,
        },
        evidence=[r["episode"] for r in rows],
        detail="Handled safely (invalid seeds never reach the backend) but it silently narrows the "
        "search the model intended.",
    )


_SEED_NOTE = re.compile(r"RDKit rejected (\d+) invalid proposed seed SMILES: \[(.*?)\]")


def _dropped_seeds(ep: Episode) -> list[str]:
    """SMILES the evaluation agent recorded RDKit rejecting, per episode.

    Read from the evaluation report's own notes rather than recomputed: the
    platform already made this determination with the real toolchain, and
    re-deciding it here would be a second opinion nobody asked for.
    """
    out: list[str] = []
    for cyc in ep.cycles:
        for note in cyc.notes:
            m = _SEED_NOTE.search(note)
            if not m:
                continue
            for raw in m.group(2).split(","):
                smi = raw.strip().strip("'\"")
                if smi and smi not in out:
                    out.append(smi)
    return out


def detect_mocked_backend_scored(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Success criteria scored against a tool running its deterministic mock."""
    mocked_tools = {t.tool_id for ep in episodes for t in ep.tools if t.is_mocked}
    if not mocked_tools:
        return None
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        hits = [
            {"metric": m, "tool": ENDPOINT_TOOL[m]}
            for m, _t in _crit_pairs(ep)
            if ENDPOINT_TOOL.get(m) in mocked_tools
        ]
        if hits:
            rows.append({"episode": ep.episode_id, "criteria": hits})
    if not rows:
        return None
    return Finding(
        finding_id="mocked_backend_scored",
        title="Success criteria are scored against tools running a deterministic stand-in",
        detector="detect_mocked_backend_scored",
        severity=Severity.MEDIUM,
        n_affected=len(rows),
        n_applicable=len(episodes),
        stats={"mocked_tools": sorted(mocked_tools), "cases": rows},
        evidence=[r["episode"] for r in rows],
        detail=(
            "The platform is careful to label mocked tools everywhere they surface, but the "
            "planner is never told, so it sets hard numeric targets against a surrogate."
        ),
    )


def detect_unused_tool_output(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Tools called on every candidate whose output reaches no decision.

    Association only: the outputs do reach the run bundle as evidence artefacts,
    so "unused" means "used by neither the ranking plane nor any success
    criterion in these episodes", which is what the detector can actually see.
    """
    tracked = {"structure.dock", "structure.simulate"}
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        objectives = {str(o.get("name")) for o in (ep.requirements or {}).get("objectives", [])}
        scored = {m for m, _t in _crit_pairs(ep)}
        for tool in ep.tools:
            if tool.tool_id not in tracked or not tool.n_calls:
                continue
            reaches = {"structure.dock": "docking_score", "structure.simulate": "stability"}[tool.tool_id]
            if reaches in objectives or reaches in scored:
                continue
            rows.append(
                {
                    "episode": ep.episode_id,
                    "tool": tool.tool_id,
                    "operation": tool.operation,
                    "n_calls": tool.n_calls,
                    "total_ms": tool.total_ms,
                    "backend": tool.backend,
                }
            )
    if not rows:
        return None
    by_tool: dict[str, dict[str, float]] = {}
    for r in rows:
        agg = by_tool.setdefault(r["tool"], {"n_calls": 0.0, "total_ms": 0.0, "episodes": 0.0})
        agg["n_calls"] += r["n_calls"]
        agg["total_ms"] += r["total_ms"]
        agg["episodes"] += 1
    return Finding(
        finding_id="unused_tool_output",
        title="Structure tools run on every candidate but reach neither the ranking nor any criterion",
        detector="detect_unused_tool_output",
        severity=Severity.LOW,
        n_affected=len({r["episode"] for r in rows}),
        n_applicable=len(episodes),
        stats={"by_tool": by_tool, "cases": rows[:20]},
        evidence=sorted({r["episode"] for r in rows}),
        detail=(
            "`build_scorecard` iterates the brief's objectives, so a pose score or an MD "
            "stability estimate can only influence ranking if an objective names it — none did. "
            "They are still written to the run bundle as evidence, which is why this is LOW: the "
            "cost is real (about 370 calls per cycle) but so is the artefact."
        ),
        inference="association",
    )


def detect_decode_health(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Repairs, truncation, and undecodable replies — the harness's own health."""
    steps = [s for ep in episodes for s in ep.steps]
    if not steps:
        return None
    truncated = [s for s in steps if s.done_reason == "length"]
    repaired = [s for s in steps if s.attempt > 0]
    failed = [s for s in steps if not s.ok]
    by_phase: dict[str, dict[str, Any]] = {}
    for s in steps:
        agg = by_phase.setdefault(s.phase, {"n": 0, "tokens": [], "seconds": [], "thinking": []})
        agg["n"] += 1
        if s.eval_tokens is not None:
            agg["tokens"].append(float(s.eval_tokens))
        agg["seconds"].append(s.elapsed_s)
        agg["thinking"].append(float(s.thinking_chars))
    summary = {
        phase: {
            "n": agg["n"],
            "mean_tokens": _mean(agg["tokens"]),
            "mean_seconds": _mean(agg["seconds"]),
            "mean_thinking_chars": _mean(agg["thinking"]),
        }
        for phase, agg in sorted(by_phase.items())
    }
    severity = Severity.INFO if not (truncated or failed) else Severity.MEDIUM
    return Finding(
        finding_id="decode_health",
        title="Structured-decoding health of the reasoning harness",
        detector="detect_decode_health",
        severity=severity,
        n_affected=len(truncated) + len(failed),
        n_applicable=len(steps),
        stats={
            "n_steps": len(steps),
            "n_truncated": len(truncated),
            "n_repaired": len(repaired),
            "n_undecodable": len(failed),
            "by_phase": summary,
        },
        evidence=sorted({s_ep.episode_id for s_ep in episodes}),
        detail="Reported so a candidate that changes prompts can be checked for making decoding worse.",
    )


def detect_cost_profile(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Where a request's wall time actually goes."""
    if not episodes:
        return None
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        if ep.duration_s is None:
            continue
        rows.append(
            {
                "episode": ep.episode_id,
                "wall_s": ep.duration_s,
                "llm_s": ep.llm_seconds,
                "llm_fraction": round(ep.llm_seconds / ep.duration_s, 4) if ep.duration_s else None,
                "tool_calls": ep.n_tool_calls,
                "llm_calls": ep.n_llm_calls,
            }
        )
    if not rows:
        return None
    return Finding(
        finding_id="cost_profile",
        title="Wall time is almost entirely language-model time",
        detector="detect_cost_profile",
        severity=Severity.INFO,
        n_affected=len(rows),
        n_applicable=len(episodes),
        stats={
            "mean_wall_s": _mean([float(r["wall_s"]) for r in rows]),
            "mean_llm_fraction": _mean([r["llm_fraction"] for r in rows]),
            "mean_llm_calls": _mean([float(r["llm_calls"]) for r in rows]),
            "rows": rows,
        },
        evidence=[r["episode"] for r in rows],
        detail=(
            "Sets the price of every experiment in this package: the deterministic pipeline is "
            "seconds, one model call is minutes. Candidates that reduce model calls are cheap "
            "wins; optimising the pipeline is not."
        ),
    )


def detect_pipeline_crash(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Cycles whose pipeline stopped before anything could be scored.

    The failure class that was invisible: a run whose three cycles all died at
    `source_candidates` recorded `SUCCEEDED` with null metrics, so every
    metric-shaped detector read it as a program that tried and scored badly. It
    is not — it is a program that never ran, usually because the generator it
    chose was handed inputs it refuses. That is a prompt problem, and nothing
    could see it.
    """
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        crashed = [c for c in ep.cycles if c.failure_reason]
        if not crashed:
            continue
        reasons = sorted({str(c.failure_reason)[:200] for c in crashed})
        rows.append(
            {
                "episode": ep.episode_id,
                "n_cycles": len(ep.cycles),
                "n_crashed": len(crashed),
                "all_cycles": len(crashed) == len(ep.cycles),
                "distinct_reasons": len(reasons),
                "reasons": reasons,
            }
        )
    if not rows:
        return None
    total_lost = sum(r["n_crashed"] for r in rows)
    repeated = [r for r in rows if r["n_crashed"] > 1 and r["distinct_reasons"] == 1]
    return Finding(
        finding_id="pipeline_crash",
        title="Cycles whose pipeline stopped before scoring",
        detector="detect_pipeline_crash",
        severity=Severity.HIGH if repeated else Severity.MEDIUM,
        n_affected=len(rows),
        n_applicable=len(episodes),
        stats={
            "n_cycles_lost": total_lost,
            "n_episodes_wholly_lost": sum(1 for r in rows if r["all_cycles"]),
            "n_episodes_repeating_one_reason": len(repeated),
            "cases": rows,
        },
        evidence=[r["episode"] for r in rows],
        detail=(
            "A crashed cycle scored nothing, which is not the same as scoring badly. An episode "
            "that repeated ONE reason across several cycles spent its whole budget on a failure "
            "the refinement step could not or did not address."
        ),
    )


def detect_discarded_adjustment(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Refinements the model made that the platform then threw away.

    Distinct from `detect_silent_requirement_loss`, which is about requirements
    the model DELETED. This is the mirror image: changes the model asked for that
    never reached the next cycle, so it re-ran the same requirements and got the
    same result. Measured on a real run whose refiner diagnosed a missing pocket
    centre correctly and had its fix overwritten three times.
    """
    rows = [
        {
            "episode": ep.episode_id,
            "cycle": c.cycle,
            "claimed": [str(ch.get("parameter") or ch.get("field")) for ch in c.adjustment_changes],
        }
        for ep in episodes
        for c in ep.cycles
        if c.adjustment_changes and not c.adjustment_diff
    ]
    if not rows:
        return None
    return Finding(
        finding_id="discarded_adjustment",
        title="Refinements the model proposed that never took effect",
        detector="detect_discarded_adjustment",
        severity=Severity.HIGH,
        n_affected=len({r["episode"] for r in rows}),
        n_applicable=len(episodes),
        stats={"n_cycles": len(rows), "cases": rows},
        evidence=sorted({str(r["episode"]) for r in rows}),
        detail=(
            "The model named changes and the field-level diff was empty, so the next cycle ran the "
            "previous requirements. Either a guard rejected the change without saying so, or the "
            "field the model moved is one refinement is not allowed to move."
        ),
    )


#: The optional assessment arms, mapped to the tools each one switches on. Kept
#: here rather than imported from `agents.llm.master_agent` because the analyzer
#: runs offline and that module opens an HTTP client on import.
_OPTIONAL_ARMS: dict[str, tuple[str, ...]] = {
    "property_backends": (
        "property.admet.admetai",
        "property.predict.chemprop",
        "property.represent.unimol",
    ),
    "route_backend": ("synth.route.aizynthfinder",),
    "complex_backend": ("structure.complex.boltz2", "structure.predict.protenix"),
    "target_evidence": ("target.evidence.opentargets", "target.essentiality.depmap"),
}


def _arm_requested(requirements: dict[str, Any] | None, field: str) -> bool:
    """Whether an episode's requirements switched this arm on."""
    value = (requirements or {}).get(field)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none")
    if isinstance(value, list):
        return bool(value)
    return bool(value)


def detect_unrequested_capability(episodes: list[Episode], ctx: dict[str, Any]) -> Finding | None:
    """Installed capabilities the master agent has never once asked for.

    The measurable form of "the agent does not know what it can use". Every arm
    here is installed, wired, and switched on by a field of the requirements —
    and each spent a long time switched on by nothing at all, because the field
    did not exist and the prompt never mentioned the tool. That is invisible from
    the metrics: a program that never asked for a route search and a program for
    which route search is genuinely irrelevant produce identical scorecards.

    A never-requested arm is not automatically a bug. It is a question worth
    asking of the prompt, which is what a finding is for.
    """
    try:
        from ...tools.registry import tool_pool

        installed = {c.tool_id for c in tool_pool() if str(c.status) == "AVAILABLE"}
    except Exception:
        return None
    scored = [ep for ep in episodes if ep.requirements]
    if not scored:
        return None

    rows: list[dict[str, Any]] = []
    for field, tool_ids in sorted(_OPTIONAL_ARMS.items()):
        present = sorted(t for t in tool_ids if t in installed)
        if not present:
            continue  # not installed here: never asking for it is correct
        n_asked = sum(1 for ep in scored if _arm_requested(ep.requirements, field))
        rows.append(
            {
                "field": field,
                "tools": present,
                "n_requested": n_asked,
                "n_episodes": len(scored),
            }
        )
    never = [r for r in rows if r["n_requested"] == 0]
    if not never:
        return None
    return Finding(
        finding_id="unrequested_capability",
        title="Installed capabilities the master agent never asks for",
        detector="detect_unrequested_capability",
        severity=Severity.MEDIUM,
        n_affected=len(never),
        n_applicable=len(rows),
        stats={"never_requested": never, "all_arms": rows},
        evidence=sorted({ep.episode_id for ep in scored})[:20],
        detail=(
            "These arms are installed and reachable through a requirements field, and no episode "
            "in the corpus switched one on. Either the goals genuinely did not need them — which "
            "the prompt should be able to show — or the model was never told the field exists. "
            "Arms never requested: " + "; ".join(f"{r['field']} ({', '.join(r['tools'])})" for r in never)
        ),
    )


DETECTORS: list[Callable[[list[Episode], dict[str, Any]], Finding | None]] = [
    detect_pipeline_crash,
    detect_discarded_adjustment,
    detect_unrequested_capability,
    detect_infeasible_targets,
    detect_never_accepted,
    detect_goal_metric_mismatch,
    detect_silent_requirement_loss,
    detect_metric_gaming,
    detect_mocked_backend_scored,
    detect_refinement_ineffective,
    detect_unscorable_criteria,
    detect_invalid_seeds,
    detect_unused_tool_output,
    detect_decode_health,
    detect_cost_profile,
]


def analyze(episodes: list[Episode], *, envelope: dict[str, Any] | None = None) -> list[Finding]:
    """Run every detector. A detector that raises is skipped, loudly in the result."""
    ctx: dict[str, Any] = {"envelope": envelope or {}}
    findings: list[Finding] = []
    for detector in DETECTORS:
        try:
            finding = detector(episodes, ctx)
        except Exception as exc:  # a broken detector must not lose the others
            findings.append(
                Finding(
                    finding_id=f"detector_error:{detector.__name__}",
                    title=f"Detector {detector.__name__} failed",
                    detector=detector.__name__,
                    severity=Severity.INFO,
                    stats={"error": f"{type(exc).__name__}: {exc}"},
                    evidence=[],
                    detail="Reported rather than swallowed: a silent detector looks like a clean system.",
                )
            )
            continue
        if finding is not None:
            findings.append(finding)
    order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    return sorted(findings, key=lambda f: (order.get(f.severity, 9), f.finding_id))


# --- small helpers ---------------------------------------------------------
def _mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return round(statistics.mean(xs), 4) if xs else None


def _sub(a: float | None, b: float | None) -> float | None:
    return round(a - b, 4) if a is not None and b is not None else None


def _counts(values: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


__all__ = ["DETECTORS", "ENDPOINT_TOOL", "analyze", "parse_target"]
