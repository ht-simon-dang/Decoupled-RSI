"""Grouping findings into a manageable number of structural failure modes.

The brief sketches ``raw episodes → summaries → embedding/clustering →
representative examples → LLM analysis``. With eleven findings over five
episodes, an embedding step would be theatre: it would consume a model call to
rediscover a grouping that is already explicit in the data, and it would make
the grouping unreproducible.

So clustering here is **deterministic and structural**: findings are grouped by
the *surface* they point at — the part of the system a fix would have to change.
That surface is also the archive's behaviour dimension, so a planner fix and a
refiner fix stay separable throughout the loop.

The design note that matters for scale: the surface map below is the only thing
that would need to change if this ran over ten thousand episodes instead of five.
At that size a clustering step over structured summaries earns its keep, and
:func:`cluster_findings` is where it would go. It does not earn its keep now, and
building it anyway would mean shipping an untested path as if it were load-bearing.
"""

from __future__ import annotations

import hashlib

from ..schemas import FailureCluster, Finding, Severity

#: finding id (or prefix) → the surface a fix would have to touch.
SURFACE_OF: dict[str, str] = {
    "infeasible_targets": "planner.criteria",
    "never_accepted": "planner.criteria",
    "unscorable_criteria": "planner.criteria",
    "goal_metric_mismatch": "planner.criteria",
    "mocked_backend_scored": "planner.criteria",
    "silent_requirement_loss": "refiner.requirements",
    "metric_gaming_observed": "refiner.requirements",
    "refinement_ineffective": "refiner.requirements",
    "invalid_seed_smiles": "transcriber.seeds",
    "unused_tool_output": "tools.routing",
    "tool_output_unused": "tools.routing",
    "decode_health": "harness.decoding",
    "cost_profile": "controller.cost",
}

_LABELS: dict[str, str] = {
    "planner.criteria": "The planner sets success criteria this deployment cannot satisfy",
    "refiner.requirements": "Refinement rewrites the requirement set in ways nobody asked for",
    "transcriber.seeds": "Proposed chemical seeds are lost before they reach the pipeline",
    "tools.routing": "Tools run whose output no decision reads",
    "harness.decoding": "Structured decoding health",
    "controller.cost": "Where the cost of a request goes",
}

_SEVERITY_RANK = {Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.INFO: 0}


def surface_of(finding_id: str) -> str:
    if finding_id in SURFACE_OF:
        return SURFACE_OF[finding_id]
    prefix = finding_id.split(":", 1)[0]
    return SURFACE_OF.get(prefix, "unknown")


def cluster_findings(findings: list[Finding]) -> list[FailureCluster]:
    """Group findings by surface, newest severity wins, ordered by severity."""
    buckets: dict[str, list[Finding]] = {}
    for f in findings:
        buckets.setdefault(surface_of(f.finding_id), []).append(f)

    clusters: list[FailureCluster] = []
    for surface, group in buckets.items():
        severity = max(group, key=lambda f: _SEVERITY_RANK.get(f.severity, 0)).severity
        episodes = sorted({e for f in group for e in f.evidence})
        signature = hashlib.sha256(("|".join(sorted(f.finding_id for f in group))).encode()).hexdigest()[:12]
        headline = sorted(group, key=lambda f: -_SEVERITY_RANK.get(f.severity, 0))[0]
        clusters.append(
            FailureCluster(
                cluster_id=f"cl_{surface.replace('.', '_')}",
                label=_LABELS.get(surface, surface),
                findings=[f.finding_id for f in group],
                episodes=episodes,
                severity=severity,
                surface=surface,
                summary=(
                    f"{len(group)} finding(s) over {len(episodes)} episode(s). "
                    f"Most severe: {headline.title} ({headline.n_affected}/{headline.n_applicable})."
                ),
                signature=signature,
            )
        )
    return sorted(clusters, key=lambda c: (-_SEVERITY_RANK.get(c.severity, 0), c.cluster_id))


__all__ = ["SURFACE_OF", "cluster_findings", "surface_of"]
