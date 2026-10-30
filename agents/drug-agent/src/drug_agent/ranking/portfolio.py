"""Portfolio construction: Pareto + diversity + decisions + sensitivity.

Ties scorecards together into a diverse Pareto portfolio with promote/reject/
defer decisions (with reasons), portfolio-level metrics, and a sensitivity
analysis over alternate objective weights/thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.brief import Objective
from ..contracts.enums import DecisionOutcome
from ..contracts.scorecard import (
    Portfolio,
    PortfolioMember,
    Scorecard,
    SensitivityReport,
    SensitivityRow,
)
from .diversity import cluster_by_scaffold, mean_pairwise_distance, select_diverse
from .pareto import hypervolume, pareto_front, pareto_rank


@dataclass
class RankedResult:
    portfolio: Portfolio
    scorecards: list[Scorecard]


def _objective_matrix(scorecards: list[Scorecard], objectives: list[Objective]) -> np.ndarray:
    """Normalized (higher-better) objective matrix; missing -> NaN."""
    rows = []
    for sc in scorecards:
        by_ep = {e.endpoint: e for e in sc.endpoints}
        row = []
        for obj in objectives:
            e = by_ep.get(obj.name)
            row.append(np.nan if (e is None or e.missing or e.normalized is None) else e.normalized)
        rows.append(row)
    return np.array(rows, dtype=float) if rows else np.zeros((0, len(objectives)))


def build_portfolio(
    *,
    run_id: str,
    scorecards: list[Scorecard],
    objectives: list[Objective],
    shortlist_size: int = 10,
    reserve_size: int = 10,
) -> RankedResult:
    if not scorecards:
        return RankedResult(Portfolio(run_id=run_id), [])

    eligible = [sc for sc in scorecards if sc.hard_constraints_passed]
    ineligible = [sc for sc in scorecards if not sc.hard_constraints_passed]

    # Pareto over eligible candidates only (hard constraints gate ranking).
    matrix = _objective_matrix(eligible, objectives)
    if len(eligible) > 0:
        front_mask = pareto_front(matrix)
        ranks = pareto_rank(matrix)
    else:
        front_mask = np.zeros(0, dtype=bool)
        ranks = np.zeros(0, dtype=int)

    for sc, on_front, rank in zip(eligible, front_mask, ranks, strict=True):
        sc.pareto_optimal = bool(on_front)
        sc.pareto_rank = int(rank)

    # Diversity clustering over eligible candidates.
    smiles = [sc.canonical_smiles or "" for sc in eligible]
    clusters = cluster_by_scaffold(smiles) if smiles else []
    for sc, cl in zip(eligible, clusters, strict=True):
        sc.diversity_cluster = int(cl)

    # Recompute display utility from *this call's* objective weights so the
    # sensitivity analysis genuinely reflects weight changes.
    weight_by = {o.name: o.weight for o in objectives}
    for sc in eligible:
        present = [
            (e.normalized, weight_by.get(e.endpoint, 1.0))
            for e in sc.endpoints
            if not e.missing and e.normalized is not None
        ]
        if present:
            wsum = sum(w for _, w in present) or 1.0
            sc.utility = round(sum(n * w for n, w in present) / wsum, 4)

    # Selection score: prefer Pareto (lower rank), then utility, penalize OOD.
    def sel_score(sc: Scorecard) -> float:
        rank_term = -(sc.pareto_rank or 999)
        util = sc.utility or 0.0
        return rank_term * 10 + util - sc.out_of_domain_penalty

    ids = [sc.candidate_id for sc in eligible]
    scores = [sel_score(sc) for sc in eligible]
    shortlist_ids = select_diverse(ids, smiles, scores, shortlist_size, clusters=clusters)
    shortlist_set = set(shortlist_ids)

    # Reserve: next-best diverse candidates not in shortlist.
    remaining_ids = [i for i in ids if i not in shortlist_set]
    rem_smiles = [smiles[ids.index(i)] for i in remaining_ids]
    rem_scores = [scores[ids.index(i)] for i in remaining_ids]
    rem_clusters = [clusters[ids.index(i)] for i in remaining_ids]
    reserve_ids = (
        select_diverse(remaining_ids, rem_smiles, rem_scores, reserve_size, clusters=rem_clusters)
        if remaining_ids
        else []
    )
    # Decisions with reasons.
    by_id = {sc.candidate_id: sc for sc in eligible}
    for cid, sc in by_id.items():
        reasons = []
        if cid in shortlist_set:
            sc.decision = DecisionOutcome.PROMOTE
            reasons.append("on shortlist (diverse Pareto selection)")
            if sc.pareto_optimal:
                reasons.append("Pareto-optimal")
        elif sc.out_of_domain_penalty >= 0.5:
            sc.decision = DecisionOutcome.DEFER
            reasons.append("out-of-domain; needs data before promotion")
        elif (sc.pareto_rank or 99) <= 2:
            sc.decision = DecisionOutcome.DEFER
            reasons.append("near-Pareto reserve candidate")
        else:
            sc.decision = DecisionOutcome.REJECT
            reasons.append(f"dominated (Pareto rank {sc.pareto_rank})")
        sc.decision_reasons = reasons

    for sc in ineligible:
        sc.decision = DecisionOutcome.REJECT
        failed = [k for k, v in sc.hard_constraint_outcomes.items() if not v]
        sc.decision_reasons = [f"failed hard constraint(s): {', '.join(failed) or 'unknown'}"]

    # Portfolio object.
    def member(cid: str, role: str) -> PortfolioMember:
        sc = by_id[cid]
        return PortfolioMember(
            candidate_id=cid,
            inchikey=sc.inchikey,
            canonical_smiles=sc.canonical_smiles,
            role=role,
            cluster=sc.diversity_cluster,
            utility=sc.utility,
            pareto_rank=sc.pareto_rank,
        )

    shortlist = [member(c, "shortlist") for c in shortlist_ids]
    reserve = [member(c, "reserve") for c in reserve_ids]

    front_pts = matrix[front_mask] if len(eligible) else np.zeros((0, len(objectives)))
    hv = round(hypervolume(np.nan_to_num(front_pts, nan=0.0)), 5) if front_pts.size else 0.0
    shortlist_smiles = [by_id[c].canonical_smiles or "" for c in shortlist_ids]
    diversity = round(mean_pairwise_distance(shortlist_smiles), 4) if len(shortlist_ids) > 1 else 0.0
    ev_cov = round(float(np.mean([sc.evidence_quality or 0.0 for sc in eligible])), 4) if eligible else 0.0

    portfolio = Portfolio(
        run_id=run_id,
        shortlist=shortlist,
        reserve=reserve,
        n_candidates_considered=len(scorecards),
        n_pareto=int(front_mask.sum()),
        n_clusters=len(set(clusters)) if clusters else 0,
        hypervolume=hv,
        diversity=diversity,
        evidence_coverage=ev_cov,
        objectives=[{"name": o.name, "direction": str(o.direction), "weight": o.weight} for o in objectives],
    )
    return RankedResult(portfolio, scorecards)


def sensitivity_analysis(
    *,
    run_id: str,
    scorecards: list[Scorecard],
    objectives: list[Objective],
    shortlist_size: int,
    base_shortlist: list[str],
) -> SensitivityReport:
    """Recompute the shortlist under alternate weights/thresholds; report drift."""
    rows: list[SensitivityRow] = []
    base_set = set(base_shortlist)

    scenarios: list[tuple[str, str, list[Objective]]] = []
    # Equal weights.
    scenarios.append(
        (
            "equal_weights",
            "All objective weights set to 1.0",
            [o.model_copy(update={"weight": 1.0}) for o in objectives],
        )
    )
    # Emphasize each objective in turn (2x weight).
    for obj in objectives:
        upd = [
            o.model_copy(update={"weight": (o.weight * 2 if o.name == obj.name else o.weight)})
            for o in objectives
        ]
        scenarios.append((f"emphasize_{obj.name}", f"Double weight on {obj.name}", upd))

    for name, desc, objs in scenarios:
        # Rebuild only selection (reuse existing normalized scorecards).
        result = build_portfolio(
            run_id=run_id,
            scorecards=[sc.model_copy(deep=True) for sc in scorecards],
            objectives=objs,
            shortlist_size=shortlist_size,
        )
        ids = [m.candidate_id for m in result.portfolio.shortlist]
        inter = len(base_set & set(ids))
        union = len(base_set | set(ids)) or 1
        jac = round(inter / union, 3)
        rows.append(
            SensitivityRow(
                scenario=name,
                description=desc,
                shortlist_ids=ids,
                jaccard_vs_base=jac,
                changed=set(ids) != base_set,
            )
        )

    return SensitivityReport(run_id=run_id, base_shortlist=base_shortlist, rows=rows)
