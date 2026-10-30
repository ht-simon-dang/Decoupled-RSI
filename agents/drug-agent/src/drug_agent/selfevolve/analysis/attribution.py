"""Tool-use attribution — how useful is a tool, really?

The naive inference is a trap:

    successful episodes used tool X  ⟹  tool X causes success

Tool usage here is decided by the brief's objectives and the operator's
generator choice, both of which also determine how hard the task is. PFM runs on
the requests that asked for novel chemotypes, and those are the requests with the
most demanding criteria — so a plain correlation would report that the de-novo
generator causes failure.

What this module does instead, in increasing order of strength:

1. **Conditional description.** Every rate is computed *within* a task class.
   Marked ``association``.
2. **Output utilization.** A structural question, not a statistical one: does the
   tool's output reach any objective, any success criterion, or any ranking
   decision in this episode? Answering "no" needs no counterfactual, so it is
   marked ``structural`` — the strongest claim available at this sample size.
3. **Matched pairs.** Two episodes of the same task class differing in exactly
   one tool's presence. Reported when they exist and marked ``matched`` — still
   not causal with n=1, and the record says so.

Nothing here is ever marked ``causal``. A causal claim would need an ablation
run through the evaluation kernel, which is what
:mod:`~drug_agent.selfevolve.evaluation.kernel` is for; the kernel writes its own
records and does not go through this module.
"""

from __future__ import annotations

import statistics
from typing import Any

from ..schemas import Episode, Finding, Severity

#: Where each tool's output can possibly land. A tool whose landing site appears
#: in neither the brief's objectives nor the plan's criteria did work nobody read.
TOOL_OUTPUT_SINKS: dict[str, list[str]] = {
    "chem.standardize": [
        "qed",
        "descriptors.mw",
        "descriptors.logp",
        "n_structural_alerts",
        "qed_median",
        "mean_pairwise_distance",
        "n_scaffold_clusters",
    ],
    "model.activity": ["pIC50", "max_pIC50", "median_pIC50", "in_domain_fraction"],
    "synth.assess": ["sa_score", "sa_median"],
    "safety.assess": ["hERG", "herg_median"],
    "perturb.assess": ["restoration_score", "restoration_median"],
    "molecule.generate.pfm": ["__candidate_source__"],
    "structure.dock": ["docking_score"],
    "structure.simulate": ["stability"],
    "platform.envelope": ["__planning__"],
}


def _sinks_of(ep: Episode) -> set[str]:
    """Every name in this episode that a tool's output could have reached."""
    names: set[str] = set()
    for obj in (ep.requirements or {}).get("objectives", []) or []:
        if obj.get("name"):
            names.add(str(obj["name"]))
    for con in (ep.requirements or {}).get("property_constraints", []) or []:
        if con.get("property"):
            names.add(str(con["property"]))
    for crit in ep.criteria:
        if crit.get("metric"):
            names.add(str(crit["metric"]))
    if (ep.requirements or {}).get("generator", "library") != "library":
        names.add("__candidate_source__")
    return names


def attribute_tools(episodes: list[Episode]) -> dict[str, Any]:
    """Per-tool statistics, conditioned on task class, plus utilization."""
    classes = sorted({ep.task_class for ep in episodes})
    by_tool: dict[str, dict[str, Any]] = {}

    for ep in episodes:
        sinks = _sinks_of(ep)
        for tool in ep.tools:
            entry = by_tool.setdefault(
                tool.tool_id,
                {
                    "tool_id": tool.tool_id,
                    "backends": set(),
                    "n_episodes": 0,
                    "n_calls": 0,
                    "n_errors": 0,
                    "total_ms": 0.0,
                    "operations": set(),
                    "used_in": [],
                    "unused_in": [],
                    "by_task_class": {},
                    "reported_metrics": {},
                },
            )
            entry["backends"].add(tool.backend or "")
            entry["operations"].add(tool.operation)
            entry["n_calls"] += tool.n_calls
            entry["n_errors"] += tool.n_errors
            entry["total_ms"] += tool.total_ms
            for key, value in tool.metrics.items():
                if value is None:
                    continue
                entry["reported_metrics"].setdefault(key, []).append(float(value))
            cls = entry["by_task_class"].setdefault(
                ep.task_class, {"episodes": set(), "n_calls": 0, "n_errors": 0, "criteria_met": []}
            )
            cls["episodes"].add(ep.episode_id)
            cls["n_calls"] += tool.n_calls
            cls["n_errors"] += tool.n_errors
            if ep.criteria_met_rate is not None:
                cls["criteria_met"].append(ep.criteria_met_rate)

        seen_tools = {t.tool_id for t in ep.tools}
        for tool_id in seen_tools:
            entry = by_tool[tool_id]
            entry["n_episodes"] += 1
            reachable = set(TOOL_OUTPUT_SINKS.get(tool_id, []))
            (entry["used_in"] if (reachable & sinks) else entry["unused_in"]).append(ep.episode_id)

    out: dict[str, Any] = {"n_episodes": len(episodes), "task_classes": classes, "tools": []}
    for tool_id, entry in sorted(by_tool.items()):
        n_ep = entry["n_episodes"]
        used = len(entry["used_in"])
        out["tools"].append(
            {
                "tool_id": tool_id,
                "backends": sorted(b for b in entry["backends"] if b),
                "operations": sorted(entry["operations"]),
                "n_episodes": n_ep,
                "n_calls": entry["n_calls"],
                "n_errors": entry["n_errors"],
                "error_rate": round(entry["n_errors"] / entry["n_calls"], 6) if entry["n_calls"] else None,
                "total_ms": round(entry["total_ms"], 3),
                "mean_ms_per_call": round(entry["total_ms"] / entry["n_calls"], 4)
                if entry["n_calls"]
                else None,
                "output_utilization": round(used / n_ep, 4) if n_ep else None,
                "used_in": sorted(entry["used_in"]),
                "unused_in": sorted(entry["unused_in"]),
                "reported_metrics": {
                    k: {"n": len(v), "mean": round(statistics.mean(v), 6)}
                    for k, v in sorted(entry["reported_metrics"].items())
                },
                "by_task_class": {
                    cls: {
                        "n_episodes": len(vals["episodes"]),
                        "n_calls": vals["n_calls"],
                        "n_errors": vals["n_errors"],
                        "mean_criteria_met_rate": round(statistics.mean(vals["criteria_met"]), 4)
                        if vals["criteria_met"]
                        else None,
                    }
                    for cls, vals in sorted(entry["by_task_class"].items())
                },
                "inference": "association (conditioned on task class); utilization is structural",
            }
        )
    out["matched_pairs"] = _matched_pairs(episodes)
    out["caveat"] = (
        f"{len(episodes)} episodes across {len(classes)} task classes. No per-tool comparison here "
        "has the sample size for a causal claim; utilization is a structural property of the "
        "brief and does not need one."
    )
    return out


def _matched_pairs(episodes: list[Episode]) -> list[dict[str, Any]]:
    """Same task class, differing in exactly one tool's presence."""
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(episodes):
        for b in episodes[i + 1 :]:
            if a.task_class != b.task_class:
                continue
            ta, tb = {t.tool_id for t in a.tools}, {t.tool_id for t in b.tools}
            diff = ta ^ tb
            if len(diff) != 1:
                continue
            tool = next(iter(diff))
            with_ep, without_ep = (a, b) if tool in ta else (b, a)
            pairs.append(
                {
                    "tool_id": tool,
                    "task_class": a.task_class,
                    "with": {"episode": with_ep.episode_id, "criteria_met_rate": with_ep.criteria_met_rate},
                    "without": {
                        "episode": without_ep.episode_id,
                        "criteria_met_rate": without_ep.criteria_met_rate,
                    },
                    "inference": "matched (n=1) — descriptive, not a test",
                }
            )
    return pairs


def utilization_findings(episodes: list[Episode]) -> list[Finding]:
    """Turn zero-utilization tools into findings the loop can act on."""
    report = attribute_tools(episodes)
    findings: list[Finding] = []
    for tool in report["tools"]:
        if tool["output_utilization"] is None or tool["output_utilization"] > 0.0:
            continue
        findings.append(
            Finding(
                finding_id=f"tool_output_unused:{tool['tool_id']}",
                title=(f"{tool['tool_id']} ran on every candidate but influenced no ranking decision"),
                detector="utilization_findings",
                severity=Severity.LOW,
                n_affected=tool["n_episodes"],
                n_applicable=report["n_episodes"],
                stats={
                    "n_calls": tool["n_calls"],
                    "total_ms": tool["total_ms"],
                    "backends": tool["backends"],
                    "sinks": TOOL_OUTPUT_SINKS.get(tool["tool_id"], []),
                    "unused_in": tool["unused_in"],
                },
                evidence=tool["unused_in"],
                detail=(
                    "Structural, and verified against the code rather than inferred: hard filters "
                    "read only RDKit descriptors and structural alerts (`h_apply_hard_filters`), "
                    "and `build_scorecard` iterates the brief's objectives — so this tool's output "
                    "can only reach a ranking decision if an objective or criterion names it, and "
                    "none did. The results are still written to the run bundle (safety register, "
                    "structures/), which is why this is LOW rather than waste."
                ),
                inference="structural",
            )
        )
    return findings


__all__ = ["TOOL_OUTPUT_SINKS", "attribute_tools", "utilization_findings"]
