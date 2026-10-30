"""The only public candidate boundary: CLI, HTTP, MCP, feedback and exports."""

from __future__ import annotations

from .config import digest
from .contracts import Brief, Candidate
from .evaluation import POLICY_ID, evaluate, evaluator_hash, identities, select


def public_run(run):
    raw = run.get("payload", {})
    view = {k: run[k] for k in ("id", "state", "created", "updated")}
    view.update(
        name=run.get("brief", {}).get("name", "Qwen planning"),
        research=raw.get("research", {}),
        diagnosis=raw.get("diagnosis"),
        cost=raw.get("cost", {"tool_calls": 0, "tool_seconds": 0}),
        elapsed_seconds=raw.get("elapsed_seconds", 0),
        candidates=[],
        evaluations=[],
        portfolio=[],
        delivery={
            "policy_id": POLICY_ID,
            "state": "withheld",
            "generated_count": len(raw.get("candidates", [])),
            "evaluated_count": len(raw.get("assessments", {})),
            "delivered_count": 0,
        },
        diagnostics=[
            {"candidate_id": k, "status": a.get("status"), "reasons": a.get("reasons", [])}
            for k, a in raw.get("assessments", {}).items()
        ],
        events=[{k: e[k] for k in ("id", "time", "kind") if k in e} for e in run.get("events", [])],
    )
    if run["state"] != "complete":
        return view
    if not raw.get("candidates"):
        view["delivery"]["state"] = "research_only"
        return view
    try:
        brief = Brief.model_validate(run["brief"])
        if raw.get("evaluator_hash") != evaluator_hash() or raw.get("brief_hash") != digest(
            brief.model_dump()
        ):
            raise ValueError("Evaluation policy/brief changed")
        candidates = [Candidate.model_validate(c) for c in raw["candidates"]]
        if any((c.id, c.variable_id) != identities(c.binder) for c in candidates):
            raise ValueError("Candidate identity changed")
        # Recompute from tool records, not from a persisted 'pass' bit or shortlist.
        assessments = {
            c.id: evaluate(c, brief, raw.get("evaluation_calls", {}).get(c.id, {})) for c in candidates
        }
        chosen = select(candidates, assessments, brief)
        allowed = {c["candidate_id"] for c in chosen}
        for candidate in candidates:
            if candidate.id in allowed:
                card = candidate.model_dump()
                card["binder"]["structure_pdb"] = None
                view["candidates"].append(card)
        view["evaluations"] = [assessments[c.id] for c in candidates if c.id in allowed]
        view["portfolio"] = chosen
        view["delivery"].update(
            state="awaiting_human_review" if allowed else "no_qualified_candidates",
            delivered_count=len(allowed),
        )
    except (ValueError, KeyError, TypeError):
        view["delivery"]["state"] = "invalid_evaluation_withheld"
    return view


def allowed_candidate(run, candidate_id):
    if not any(c["id"] == candidate_id for c in public_run(run)["candidates"]):
        raise ValueError("Candidate is not eligible for human delivery under the current evaluation")
    return next(c for c in run["payload"]["candidates"] if c["id"] == candidate_id)
