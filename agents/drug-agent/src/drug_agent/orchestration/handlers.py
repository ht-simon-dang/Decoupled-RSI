"""Task handlers: the per-step workflow logic.

Each handler takes ``(ctx, task)`` and returns a small JSON-able payload dict.
Agent steps delegate to :func:`invoke_agent`; control/ranking steps do the work
directly. Handlers are deterministic given their persisted inputs, which is what
makes the run resumable and cacheable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..agents.base import AgentError
from ..contracts.brief import ResearchBrief
from ..contracts.enums import DecisionOutcome
from ..contracts.evidence import DecisionRecord
from ..contracts.predictions import Prediction
from ..contracts.scorecard import Scorecard
from ..data.connectors import default_connectors
from ..data.normalize import build_assay_record
from ..ranking.filters import evaluate_constraints
from ..ranking.portfolio import build_portfolio, sensitivity_analysis
from ..ranking.scorecards import build_scorecard, compute_bounds
from ..tools.sklearn_tool import BaselineActivityModel
from .invoke import invoke_agent

if TYPE_CHECKING:
    from ..contracts.run import Task
    from .context import RunContext

Handler = Callable[["RunContext", "Task"], dict[str, Any]]


def _brief(ctx: RunContext) -> ResearchBrief:
    if "brief" in ctx.blackboard:
        return ctx.blackboard["brief"]
    data = ctx.store.get_object("brief", ctx.run.brief_id)
    brief = ResearchBrief.model_validate(data)
    ctx.blackboard["brief"] = brief
    return brief


# --- control-plane steps ---------------------------------------------------


def h_validate_brief(ctx: RunContext, task: Task) -> dict[str, Any]:
    brief = _brief(ctx)  # model_validate re-checks the schema
    return {"valid": True, "kind": str(brief.kind), "n_objectives": len(brief.objectives)}


def h_resolve_entities(ctx: RunContext, task: Task) -> dict[str, Any]:
    brief = _brief(ctx)
    conns = default_connectors()
    resolved: dict[str, Any] = {}
    # Keyed on the PHENOTYPE block, not on the presence of a target. A phenotype
    # brief may now also name a protein (so a structure can be resolved and a
    # pocket detected), and reading that as a target run would set
    # target_id="TARGET", which the bioactivity connector filters on — silently
    # dropping every assay row from a phenotype program that happened to name a
    # gene. The phenotype is what decides the run's shape; the target name rides
    # along for structure resolution.
    if brief.phenotype is not None:
        resolved = {
            "target_id": "PHENOTYPE",
            "target_name": brief.phenotype.description,
        }
        if brief.target is not None:
            resolved["structure_target_name"] = brief.target.name
            resolved["uniprot"] = brief.target.uniprot
    elif brief.target:
        resolved = {
            "target_id": brief.target.chembl_target_id or "TARGET",
            "target_name": brief.target.name,
            "uniprot": brief.target.uniprot,
        }
    else:
        resolved = {"target_id": "PHENOTYPE", "target_name": "phenotype"}
    ctx.blackboard["resolved"] = resolved

    # Load evidence-source snapshots for the literature agent.
    for cname, otype in (
        ("literature", "literature"),
        ("opentargets", "target_disease"),
        ("pdb", "structure_meta"),
    ):
        snap = conns[cname].load()
        for i, rec in enumerate(snap.records):
            ctx.store.put_object(otype, {"id": f"{otype}_{i}", **rec}, run_id=ctx.run.id)
        ctx.store.put_object("data_card", {"id": f"card_{snap.name}", **snap.data_card()}, run_id=ctx.run.id)
    return resolved


def h_create_data_snapshot(ctx: RunContext, task: Task) -> dict[str, Any]:
    conns = default_connectors()
    # compounds
    comp = conns["pubchem"].load()
    for i, rec in enumerate(comp.records):
        ctx.store.put_object("compound", {"id": f"cmp_{i}", **rec}, run_id=ctx.run.id)
    ctx.store.put_object("data_card", {"id": f"card_{comp.name}", **comp.data_card()}, run_id=ctx.run.id)
    # bioactivity -> assay records
    target_id = (ctx.blackboard.get("resolved") or {}).get("target_id")
    bio = conns["chembl"].load(target_id=target_id if target_id and target_id.startswith("TARGET") else None)
    for rec in bio.records:
        ar = build_assay_record(rec, source="chembl_fixture", snapshot_id=bio.snapshot_id)
        ctx.store.put_object("assay", ar, run_id=ctx.run.id)
    ctx.store.put_object("data_card", {"id": f"card_{bio.name}", **bio.data_card()}, run_id=ctx.run.id)
    ctx.run.tags = list(dict.fromkeys([*ctx.run.tags, bio.snapshot_id]))
    return {
        "n_compounds": len(comp.records),
        "n_assays": len(bio.records),
        "snapshot": bio.snapshot_id,
        "content_hash": bio.content_hash,
    }


def h_run_baselines(ctx: RunContext, task: Task) -> dict[str, Any]:
    """Baseline-first: compare the RF model to random/constant baselines under a
    scaffold split, so we never present a model as good merely because it trains.
    """
    import numpy as np

    from ..tools.rdkit_tool import Cheminformatics as Chem

    assays = ctx.store.list_objects("assay", run_id=ctx.run.id)
    raw_pairs = [
        (a["molecule_smiles"], (a.get("measurement") or {}).get("value"))
        for a in assays
        if a.get("molecule_smiles")
    ]
    pairs: list[tuple[str, float]] = [(s, float(v)) for s, v in raw_pairs if v is not None]
    if len(pairs) < 8:
        ctx.store.put_object(
            "baseline_metrics", {"id": "baseline", "note": "insufficient data"}, run_id=ctx.run.id
        )
        return {"trained": False, "n": len(pairs)}

    chem = Chem()
    # Scaffold split (generalization-relevant).
    scaffolds = [chem.scaffold(s) or s for s, _ in pairs]
    uniq = sorted(set(scaffolds))
    from ..util import stable_hash_int

    test_scaffolds = {sc for sc in uniq if stable_hash_int(sc, modulo=3) == 0}
    train = [(s, v) for (s, v), sc in zip(pairs, scaffolds, strict=True) if sc not in test_scaffolds]
    test = [(s, v) for (s, v), sc in zip(pairs, scaffolds, strict=True) if sc in test_scaffolds]
    if not train or not test:
        train, test = pairs[: len(pairs) * 2 // 3], pairs[len(pairs) * 2 // 3 :]

    model = BaselineActivityModel(endpoint="pIC50", seed=ctx.run.seed)
    model.fit([s for s, _ in train], [v for _, v in train])
    ctx.blackboard["activity_model"] = model

    y_true = np.array([v for _, v in test])
    y_pred = np.array([model.predict_one(s).value for s, _ in test])
    y_mean = float(np.mean([v for _, v in train]))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mae_const = float(np.mean(np.abs(y_true - y_mean)))
    # Spearman-ish rank correlation.
    from scipy.stats import spearmanr

    rho = float(spearmanr(y_true, y_pred).statistic) if len(test) > 2 else float("nan")
    beats = mae < mae_const
    metrics = {
        "id": "baseline",
        "endpoint": "pIC50",
        "split": "scaffold",
        "n_train": len(train),
        "n_test": len(test),
        "mae": round(mae, 4),
        "mae_constant_baseline": round(mae_const, 4),
        "spearman": round(rho, 4),
        "beats_constant_baseline": beats,
        "model_version": model.model_version,
    }
    ctx.store.put_object("baseline_metrics", metrics, run_id=ctx.run.id)
    return metrics


def h_apply_hard_filters(ctx: RunContext, task: Task) -> dict[str, Any]:
    brief = _brief(ctx)
    constraints = brief.hard_constraints
    candidates = ctx.store.list_objects("candidate", run_id=ctx.run.id)
    n_pass = 0
    for c in candidates:
        mol = ctx.store.get_object("molecule", c["molecule_id"]) or {}
        props = {
            "descriptors": mol.get("descriptors", {}),
            "n_structural_alerts": len(mol.get("structural_alerts", [])),
            "structural_alerts": mol.get("structural_alerts", []),
        }
        passed, outcomes = evaluate_constraints(constraints, props)
        c["passed_hard_filters"] = passed
        c["filter_failures"] = [o.name for o in outcomes if not o.passed]
        ctx.store.put_object("candidate", c, run_id=ctx.run.id)
        n_pass += passed
    return {"n_candidates": len(candidates), "n_passed": n_pass, "n_failed": len(candidates) - n_pass}


def h_integrate_objectives(ctx: RunContext, task: Task) -> dict[str, Any]:
    brief = _brief(ctx)
    objectives = brief.objectives
    candidates = ctx.store.list_objects("candidate", run_id=ctx.run.id)
    preds = ctx.store.list_objects("prediction", run_id=ctx.run.id)
    preds_by_cand: dict[str, list[Prediction]] = {}
    endpoint_values: dict[str, list[float]] = {o.name: [] for o in objectives}
    for p in preds:
        obj = Prediction.model_validate(p)
        preds_by_cand.setdefault(obj.candidate_id, []).append(obj)
        if obj.endpoint in endpoint_values and obj.quantity.value is not None:
            endpoint_values[obj.endpoint].append(obj.quantity.value)

    # Make-it-or-fail reaches the ranking plane too (ADR-0011). A required
    # objective with no predictions at all is NOT "worst for everyone" — the
    # Pareto front treats an all-missing column as neutral, so it silently
    # disappears and utility is renormalised over the objectives that remain.
    # A run would then ship a ranked shortlist whose primary objective nothing
    # computed. `Objective.required` existed but was read nowhere; this is the
    # one place it can mean something.
    missing_required = [o.name for o in objectives if o.required and not endpoint_values.get(o.name)]
    if missing_required:
        raise AgentError(
            "cannot rank: no predictions exist for required objective(s) "
            f"{', '.join(sorted(missing_required))}. The capability that produces them is "
            "UNAVAILABLE — see the capability matrix. Ranking on the remaining objectives "
            "would silently drop the one the brief called required."
        )

    bounds = compute_bounds(endpoint_values)

    n = 0
    for c in candidates:
        sc = build_scorecard(
            run_id=ctx.run.id,
            candidate_id=c["id"],
            inchikey=c.get("inchikey"),
            smiles=c.get("canonical_smiles"),
            objectives=objectives,
            predictions=preds_by_cand.get(c["id"], []),
            bounds=bounds,
            hard_passed=bool(c.get("passed_hard_filters")),
            constraint_outcomes={
                name: (name not in (c.get("filter_failures") or []))
                for name in [ct.name for ct in brief.hard_constraints]
            },
            tool_versions={"chem": "rdkit", "model": "baseline-rf-1.0.0"},
        )
        ctx.store.put_object("scorecard", sc, run_id=ctx.run.id)
        n += 1
    return {"n_scorecards": n, "n_objectives": len(objectives)}


def h_build_pareto_portfolio(ctx: RunContext, task: Task) -> dict[str, Any]:
    brief = _brief(ctx)
    scorecards = [Scorecard.model_validate(s) for s in ctx.store.list_objects("scorecard", run_id=ctx.run.id)]
    result = build_portfolio(
        run_id=ctx.run.id,
        scorecards=scorecards,
        objectives=brief.objectives,
        shortlist_size=brief.shortlist_size,
    )
    ctx.store.put_object("portfolio", result.portfolio, run_id=ctx.run.id)
    for sc in result.scorecards:
        ctx.store.put_object("scorecard", sc, run_id=ctx.run.id)
        ctx.store.put_object(
            "decision",
            DecisionRecord(
                run_id=ctx.run.id,
                subject_id=sc.candidate_id,
                outcome=sc.decision
                if isinstance(sc.decision, DecisionOutcome)
                else DecisionOutcome(sc.decision),
                rationale="; ".join(sc.decision_reasons) or "n/a",
                reasons=sc.decision_reasons,
                inputs={
                    "pareto_rank": sc.pareto_rank,
                    "utility": sc.utility,
                    "hard_constraints_passed": sc.hard_constraints_passed,
                },
                decided_by="agent.ranking",
            ),
            run_id=ctx.run.id,
        )

    base_shortlist = [m.candidate_id for m in result.portfolio.shortlist]
    sens = sensitivity_analysis(
        run_id=ctx.run.id,
        scorecards=scorecards,
        objectives=brief.objectives,
        shortlist_size=brief.shortlist_size,
        base_shortlist=base_shortlist,
    )
    ctx.store.put_object("sensitivity", sens, run_id=ctx.run.id)
    return {
        "n_shortlist": len(result.portfolio.shortlist),
        "n_reserve": len(result.portfolio.reserve),
        "n_pareto": result.portfolio.n_pareto,
        "hypervolume": result.portfolio.hypervolume,
        "diversity": result.portfolio.diversity,
    }


def h_await_human_approval(ctx: RunContext, task: Task) -> dict[str, Any]:
    # Engine handles the gate mechanics; reaching here means it was approved.
    return {"gate": task.approval_gate, "approved": True}


def h_generate_report(ctx: RunContext, task: Task) -> dict[str, Any]:
    from ..reporting.report import generate_report

    return generate_report(ctx)


def h_create_bundle(ctx: RunContext, task: Task) -> dict[str, Any]:
    from ..reporting.bundle import create_bundle

    return create_bundle(ctx)


# --- agent-backed steps ----------------------------------------------------


def _record_warnings(ctx: RunContext, task: Task, agent_id: str, res: Any) -> None:
    """Persist an agent's batch warnings where something can read them back.

    Agents have always written `AgentResult.warnings` — "docking produced NO
    poses", "route search produced NOTHING" — and nothing ever read them. They
    went into the task payload behind an artifact reference and stopped there, so
    the master agent could switch an optional arm on, watch it produce nothing,
    and be told only that some criteria were unmeasured. The cause and the
    symptom were in different places and only the symptom crossed the bridge.

    One small object per warning, in the run's own store, so the LLM bridge, the
    report and the bundle all reach them the same way they reach everything else.
    """
    for i, warning in enumerate(getattr(res, "warnings", None) or []):
        try:
            ctx.store.put_object(
                "agent_warning",
                {
                    "id": f"warn_{task.name}_{i}",
                    "task": task.name,
                    "agent_id": agent_id,
                    "message": str(warning),
                },
                run_id=ctx.run.id,
            )
        except Exception:  # a warning must never be able to fail the task
            return


def _agent_step(
    action: str, agent_id: str, extra_inputs: Callable[[RunContext], dict] | None = None
) -> Handler:
    def handler(ctx: RunContext, task: Task) -> dict[str, Any]:
        resolved = ctx.blackboard.get("resolved", {})
        inputs = {"brief": _brief(ctx).model_dump(mode="json"), **resolved}
        if extra_inputs:
            inputs.update(extra_inputs(ctx))
        res = invoke_agent(ctx, agent_id, action, inputs=inputs, task_id=task.id)
        _record_warnings(ctx, task, agent_id, res)
        return {"agent": agent_id, "action": action, **res.payload}

    return handler


HANDLERS: dict[str, Handler] = {
    "validate_brief": h_validate_brief,
    "resolve_entities": h_resolve_entities,
    "assemble_target_evidence": _agent_step("assemble_target_evidence", "agent.target"),
    "assemble_evidence": _agent_step("assemble_evidence", "agent.literature"),
    "generate_hypotheses": _agent_step("generate_hypotheses", "agent.discovery"),
    "detect_pocket": _agent_step("detect_pocket", "agent.pocket"),
    "create_data_snapshot": h_create_data_snapshot,
    "run_baselines": h_run_baselines,
    "source_candidates": _agent_step("source_candidates", "agent.molecule"),
    "standardize_candidates": _agent_step("standardize_candidates", "agent.molecule"),
    "apply_hard_filters": h_apply_hard_filters,
    "predict_activity_and_properties": _agent_step("predict_activity_and_properties", "agent.molecule"),
    "assess_synthesis": _agent_step("assess_synthesis", "agent.synthesis"),
    "assess_structure": _agent_step("assess_structure", "agent.structure"),
    "assess_perturbation": _agent_step("assess_perturbation", "agent.perturbation"),
    "assess_safety": _agent_step("assess_safety", "agent.safety"),
    "validate_all_results": _agent_step("validate_all_results", "agent.evaluation"),
    "integrate_objectives": h_integrate_objectives,
    "build_pareto_portfolio": h_build_pareto_portfolio,
    "index_knowledge": _agent_step("index_knowledge", "agent.memory"),
    "await_human_approval": h_await_human_approval,
    "propose_experiments": _agent_step("propose_experiments", "agent.discovery"),
    "generate_report": h_generate_report,
    "create_reproducibility_bundle": h_create_bundle,
}
