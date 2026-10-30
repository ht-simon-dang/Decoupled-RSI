"""Scorecard construction: endpoint normalization, utility, evidence, conflict."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.brief import Objective
from ..contracts.enums import ApplicabilityDomain, ObjectiveDirection
from ..contracts.predictions import Prediction
from ..contracts.scorecard import EndpointScore, Scorecard


@dataclass
class NormBounds:
    lo: float
    hi: float


def normalize_value(
    value: float, direction: ObjectiveDirection, bounds: NormBounds, target: float | None = None
) -> float:
    """Map a raw value to a [0,1] score where higher is always better."""
    lo, hi = bounds.lo, bounds.hi
    if hi <= lo:
        return 0.5
    if target is not None:
        # Closeness-to-target: 1 at target, decaying to 0 at the far bound.
        span = max(hi - target, target - lo, 1e-9)
        return float(max(0.0, 1.0 - abs(value - target) / span))
    frac = (value - lo) / (hi - lo)
    if direction == ObjectiveDirection.MINIMIZE:
        frac = 1.0 - frac
    return float(min(1.0, max(0.0, frac)))


def compute_bounds(predictions_by_endpoint: dict[str, list[float]]) -> dict[str, NormBounds]:
    bounds: dict[str, NormBounds] = {}
    for ep, vals in predictions_by_endpoint.items():
        finite = [v for v in vals if v is not None]
        if finite:
            bounds[ep] = NormBounds(min(finite), max(finite))
    return bounds


_APPLICABILITY_PENALTY = {
    ApplicabilityDomain.IN_DOMAIN: 0.0,
    ApplicabilityDomain.BORDERLINE: 0.25,
    ApplicabilityDomain.OUT_OF_DOMAIN: 0.6,
    ApplicabilityDomain.UNKNOWN: 0.4,
}


def build_scorecard(
    *,
    run_id: str,
    candidate_id: str,
    inchikey: str | None,
    smiles: str | None,
    objectives: list[Objective],
    predictions: list[Prediction],
    bounds: dict[str, NormBounds],
    hard_passed: bool,
    constraint_outcomes: dict[str, bool],
    tool_versions: dict[str, str] | None = None,
) -> Scorecard:
    pred_by_ep = {p.endpoint: p for p in predictions}
    endpoint_scores: list[EndpointScore] = []
    penalties: list[float] = []

    for obj in objectives:
        pred = pred_by_ep.get(obj.name)
        if pred is None or pred.quantity.value is None:
            endpoint_scores.append(
                EndpointScore(
                    endpoint=obj.name,
                    direction=str(obj.direction),
                    missing=True,
                    applicability=ApplicabilityDomain.UNKNOWN,
                )
            )
            penalties.append(_APPLICABILITY_PENALTY[ApplicabilityDomain.UNKNOWN])
            continue
        b = bounds.get(obj.name, NormBounds(pred.quantity.value, pred.quantity.value))
        norm = normalize_value(pred.quantity.value, obj.direction, b, obj.target_value)
        endpoint_scores.append(
            EndpointScore(
                endpoint=obj.name,
                raw_value=pred.quantity.value,
                normalized=round(norm, 4),
                unit=pred.quantity.unit,
                direction=str(obj.direction),
                uncertainty=pred.quantity.std,
                applicability=pred.applicability,
                model_id=pred.model_id,
                model_version=pred.model_version,
                evidence_ids=[e.evidence_id for e in pred.evidence],
            )
        )
        penalties.append(_APPLICABILITY_PENALTY.get(pred.applicability, 0.4))

    present = [e for e in endpoint_scores if not e.missing]
    norms = [e.normalized for e in present if e.normalized is not None]
    weights = [next((o.weight for o in objectives if o.name == e.endpoint), 1.0) for e in present]

    # Display-only weighted utility.
    if norms:
        wsum = sum(weights) or 1.0
        utility = sum(n * w for n, w in zip(norms, weights, strict=True)) / wsum
    else:
        utility = None

    # Evidence quality: fraction of endpoints present & in/borderline domain.
    n_obj = len(objectives) or 1
    in_dom = sum(
        1
        for e in present
        if e.applicability in {ApplicabilityDomain.IN_DOMAIN, ApplicabilityDomain.BORDERLINE}
    )
    evidence_quality = round((len(present) / n_obj) * (in_dom / max(len(present), 1)), 4)

    # Conflict: spread of normalized endpoint scores (large trade-offs).
    if len(norms) >= 2:
        mean = sum(norms) / len(norms)
        conflict = (sum((n - mean) ** 2 for n in norms) / len(norms)) ** 0.5
    else:
        conflict = 0.0

    ood_penalty = round(sum(penalties) / len(penalties), 4) if penalties else 0.0

    return Scorecard(
        run_id=run_id,
        candidate_id=candidate_id,
        inchikey=inchikey,
        canonical_smiles=smiles,
        endpoints=endpoint_scores,
        hard_constraints_passed=hard_passed,
        hard_constraint_outcomes=constraint_outcomes,
        utility=round(utility, 4) if utility is not None else None,
        evidence_quality=evidence_quality,
        conflict_score=round(conflict, 4),
        out_of_domain_penalty=ood_penalty,
        tool_versions=tool_versions or {},
    )
