"""Gate 1 — deterministic replay of recorded model output.

The expensive part of evaluating this platform is the model: minutes per call.
The *cheap* part is that the journal already contains, verbatim, what the model
said. For any candidate that changes how the platform **post-processes** model
output — a merge guard, a criteria repair, a schema coercion — the recorded
replies are a complete and exact benchmark, with no model in the loop and no
sampling noise.

Replay is safe by construction here because it never re-executes anything with a
side effect. It re-decodes a recorded reply and runs it through the candidate's
deterministic code path; no tool is called, no network is touched, no run is
created. That is the whole reason this gate can sit before the expensive one.

Two replay suites exist, because the journal supports exactly two:

``refine_merge``
    Every recorded refinement, with the requirements that went in and the
    ``RequirementsAdjustment`` that came back. Measures how many requirement
    entries survived, and how many were lost without the model saying so. The
    history contains one episode where 8 of 8 entries were silently deleted and
    one where nothing was — a positive and a negative control.

``criteria_repair``
    Every recorded plan paired with the transcribed goal kind. Measures how many
    success criteria are unscorable for that kind.

A candidate whose overlay touches neither of these gets an empty, honest result
rather than a fabricated pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...llm.policy import RuntimePolicy
from ...llm.repair import merge_adjustment, repair_criteria_for_goal
from ...llm.schemas import ChemicalRequirements, RequirementsAdjustment, SuccessCriterion
from ...observability.journal import iter_request_dirs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def collect_refine_cases(journal_root: Path) -> list[dict[str, Any]]:
    """Every recorded refinement, with its inputs, from the request journals."""
    cases: list[dict[str, Any]] = []
    if not journal_root.is_dir():
        return cases
    for d in iter_request_dirs(journal_root):
        chat = _read_jsonl(d / "chat.jsonl")
        events = _read_jsonl(d / "events.jsonl")
        req_by_cycle: dict[int, dict[str, Any]] = {}
        for e in events:
            if e.get("type") == "cycle_started":
                req_by_cycle[int(e.get("cycle") or 0)] = (e.get("data") or {}).get("requirements") or {}
            elif e.get("type") == "requirements_ready" and 1 not in req_by_cycle:
                req_by_cycle[1] = (e.get("data") or {}).get("requirements") or {}
        for entry in chat:
            if entry.get("phase") != "refine" or not isinstance(entry.get("parsed"), dict):
                continue
            cycle = int(entry.get("cycle") or 0)
            before = req_by_cycle.get(cycle)
            if not before:
                continue
            cases.append(
                {
                    "episode": d.name,
                    "cycle": cycle,
                    "requirements_before": before,
                    "adjustment": entry["parsed"],
                    "journal": str(d),
                }
            )
    return cases


def replay_refine_merge(policy: RuntimePolicy, journal_root: Path) -> dict[str, Any]:
    """Replay every recorded refinement under ``policy``.

    The measured quantity is ``silent_loss`` — requirement entries the model
    removed while not naming them in its own ``changes`` list. Under the default
    policy this is whatever the platform actually did; under a candidate with
    ``refine_carry_forward`` it should be zero, and the number of entries the
    model *did* name as removed should be unchanged.
    """
    rows: list[dict[str, Any]] = []
    for case in collect_refine_cases(journal_root):
        try:
            before = ChemicalRequirements.model_validate(case["requirements_before"])
            adj = RequirementsAdjustment.model_validate(case["adjustment"])
        except Exception as exc:
            rows.append(
                {"episode": case["episode"], "cycle": case["cycle"], "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        after = adj.adjusted_requirements.model_copy(deep=True)
        # The master agent pins program identity before any merge; mirror that
        # so the replay measures the merge and nothing else.
        after.kind = before.kind
        after.generator = before.generator

        notes: list[str] = []
        if policy.refine_carry_forward:
            after, notes = merge_adjustment(
                before, after, adj.changes, max_drop_fraction=policy.refine_max_drop_fraction
            )

        named = " ".join(f"{c.parameter} {c.reason}" for c in adj.changes).lower()
        kept_obj = {o.name for o in after.objectives}
        kept_con = {c.property for c in after.property_constraints}
        lost_obj = [o.name for o in before.objectives if o.name not in kept_obj]
        lost_con = [c.property for c in before.property_constraints if c.property not in kept_con]
        silent = [n for n in lost_obj + lost_con if not _named(n, named)]
        rows.append(
            {
                "episode": case["episode"],
                "cycle": case["cycle"],
                "n_before": len(before.objectives) + len(before.property_constraints),
                "n_after": len(after.objectives) + len(after.property_constraints),
                "n_reported_changes": len(adj.changes),
                "lost": lost_obj + lost_con,
                "silent_loss": silent,
                "n_silent_loss": len(silent),
                "guard_notes": notes,
            }
        )
    total_silent = sum(r.get("n_silent_loss", 0) for r in rows)
    return {
        "suite": "refine_merge",
        "n_cases": len(rows),
        "n_silent_loss": total_silent,
        "n_cases_with_silent_loss": sum(1 for r in rows if r.get("n_silent_loss")),
        "rows": rows,
    }


def _named(field: str, blob: str) -> bool:
    low = field.lower()
    if low in blob:
        return True
    return any(part in blob for part in low.split(".") if len(part) > 2)


def collect_criteria_cases(journal_root: Path) -> list[dict[str, Any]]:
    """Every recorded plan, paired with the goal kind that was transcribed."""
    cases: list[dict[str, Any]] = []
    if not journal_root.is_dir():
        return cases
    for d in iter_request_dirs(journal_root):
        events = _read_jsonl(d / "events.jsonl")
        criteria: list[dict[str, Any]] = []
        kind = ""
        for e in events:
            if e.get("type") == "plan_ready":
                criteria = list((e.get("data") or {}).get("criteria") or [])
            elif e.get("type") == "requirements_ready":
                kind = str(((e.get("data") or {}).get("requirements") or {}).get("kind") or "")
        if criteria and kind:
            cases.append({"episode": d.name, "kind": kind, "criteria": criteria})
    return cases


def replay_criteria_repair(policy: RuntimePolicy, journal_root: Path) -> dict[str, Any]:
    """How many planned criteria are inapplicable to the goal kind, under ``policy``."""
    rows: list[dict[str, Any]] = []
    for case in collect_criteria_cases(journal_root):
        crits = [SuccessCriterion.model_validate(c) for c in case["criteria"]]
        dropped: list[SuccessCriterion] = []
        kept = crits
        if policy.goal_aware_criteria:
            kept, dropped = repair_criteria_for_goal(crits, case["kind"])
        from ...llm.repair import inapplicable_metrics

        blocked = inapplicable_metrics(case["kind"])
        remaining_bad = [c.metric for c in kept if c.metric in blocked]
        rows.append(
            {
                "episode": case["episode"],
                "kind": case["kind"],
                "n_criteria": len(crits),
                "n_kept": len(kept),
                "n_dropped": len(dropped),
                "dropped": [c.metric for c in dropped],
                "still_inapplicable": remaining_bad,
            }
        )
    return {
        "suite": "criteria_repair",
        "n_cases": len(rows),
        "n_still_inapplicable": sum(len(r["still_inapplicable"]) for r in rows),
        "rows": rows,
    }


SUITES = {
    "refine_merge": replay_refine_merge,
    "criteria_repair": replay_criteria_repair,
}


def applicable_suites(policy: RuntimePolicy) -> list[str]:
    """Which replay suites this policy could possibly move.

    A candidate that changes nothing a suite measures gets an empty result and
    the record says "not applicable", rather than a pass it did not earn.
    """
    out: list[str] = []
    if policy.refine_carry_forward or policy.refine_max_drop_fraction < 1.0:
        out.append("refine_merge")
    if policy.goal_aware_criteria:
        out.append("criteria_repair")
    return out


def run_replay(
    policy: RuntimePolicy, journal_root: Path, *, suites: list[str] | None = None
) -> dict[str, Any]:
    """Run the named suites (or every suite) and return the raw results."""
    names = suites if suites is not None else list(SUITES)
    started = time.time()
    return {
        "suites": {name: SUITES[name](policy, journal_root) for name in names if name in SUITES},
        "elapsed_s": round(time.time() - started, 3),
        "journal_root": str(journal_root),
        "policy": policy.behaviour_version(),
    }


__all__ = [
    "SUITES",
    "applicable_suites",
    "collect_criteria_cases",
    "collect_refine_cases",
    "replay_criteria_repair",
    "replay_refine_merge",
    "run_replay",
]
