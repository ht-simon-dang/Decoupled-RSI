"""Frozen-pool workflow comparison. Never promotes from a proxy-only gain."""

from .contracts import Brief, Candidate
from .delivery import public_run
from .evaluation import evaluate, evaluator_hash, select, similarity


def compare(run):
    brief = Brief.model_validate(run["brief"])
    public = public_run(run)
    if public["delivery"]["state"] not in {"awaiting_human_review", "no_qualified_candidates"}:
        return {
            "status": "blocked",
            "reason": "A completed run with current independent evaluation is required",
        }
    candidates = [Candidate.model_validate(c) for c in run["payload"].get("candidates", [])]
    assessments = {
        c.id: evaluate(c, brief, run["payload"].get("evaluation_calls", {}).get(c.id, {})) for c in candidates
    }
    passed = sorted(
        [c for c in candidates if assessments[c.id]["status"] == "pass"],
        key=lambda c: (-(assessments[c.id]["ranking_value"] or 0), c.id),
    )
    limit = min(brief.budget.shortlist, brief.budget.experiment_slots)
    baseline = passed[:limit]
    proposed = select(candidates, assessments, brief)
    families = []
    for c in baseline:
        if not any(similarity(c, p) >= brief.evaluation_policy.family_identity for p in families):
            families.append(c)
    return {
        "status": "replayed",
        "evaluator_hash": evaluator_hash(),
        "same_pool": True,
        "same_budget": limit,
        "baseline": {
            "policy": "confidence_then_id",
            "selected_count": len(baseline),
            "family_count": len(families),
        },
        "proposal": {
            "policy": "independent_gates_then_family_coverage",
            "selected_count": len(proposed),
            "family_count": len(proposed),
        },
        "promotion": "not_promoted",
        "experimental_gain": None,
        "limitations": [
            "Within-pool policy replay is not a frozen independent biological benchmark",
            "Prospective measurements and held-out antigen/family/time splits are required before promotion",
            "Qualification thresholds, failed samples and test sets were not altered",
        ],
        "rollback": "Keep the original run and fixed baseline; no default policy was changed",
    }
