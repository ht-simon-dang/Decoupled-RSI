"""Fail-closed delivery gate. Research prioritization, never a claim of efficacy."""
from __future__ import annotations
import math
from .contracts import identity

POLICY = {
    "version": "high-potential-v1",
    "minimum_primary_gain": 0.05,
    "gain_scale": "directional delta / max(1, abs(baseline)); engineering threshold, not a confidence bound",
    "minimum_independent_predictors": 2,
    "requires": ["complete evaluation", "all hard constraints", "full-molecule structure",
                 "independent primary evidence", "matched baseline", "no secondary regression"],
    "interpretation": "Computational high-potential for follow-up; experimental efficacy remains unproven",
}
POLICY_ID = identity(POLICY)


def assess(candidate, evaluation, baseline, base_evaluation, brief):
    from .evaluation import check_construct
    reasons, comparisons = [], []
    if any(c["status"] != "pass" for c in check_construct(candidate,brief)):
        reasons.append("current_construct_constraints_failed")
    if candidate.sources.get("baseline"):
        reasons.append("reference_control_not_a_new_recommendation")
    if not evaluation or evaluation.get("protocol") != POLICY_ID:
        reasons.append("evaluation_missing_or_legacy_protocol")
    if not evaluation or evaluation.get("status") != "computationally_valid":
        reasons.append("evaluation_or_hard_constraints_incomplete")
    if evaluation and evaluation.get("context_id") != identity(brief.context.model_dump()):
        reasons.append("evaluation_context_mismatch")
    if not baseline or not base_evaluation or base_evaluation.get("status") != "computationally_valid" or base_evaluation.get("protocol") != POLICY_ID or base_evaluation.get("context_id") != identity(brief.context.model_dump()):
        reasons.append("valid_baseline_evaluation_required")
    if baseline and candidate.molecule != baseline.molecule:
        reasons.append("baseline_molecular_configuration_mismatch")
    if evaluation and (not evaluation.get("evaluation_id") or any(c["status"] != "pass" for c in evaluation.get("checks", []))):
        reasons.append("durable_successful_evaluation_required")
    metrics = (evaluation or {}).get("metrics", [])
    base_metrics = (base_evaluation or {}).get("metrics", [])
    ok = [m for m in metrics if m["status"] == "ok" and m.get("value") is not None and math.isfinite(m["value"])]
    for name in brief.evaluators:
        rows = [m for m in metrics if m["source"] == name]
        if not rows or any(m["status"] in {"error", "unavailable", "not_evaluated"} for m in rows):
            reasons.append("requested_evaluator_incomplete:" + name)
    if not {"mfe", "ensemble_free_energy", "start_accessibility"} <= {m["metric"] for m in ok if m["source"] == "vienna"}:
        reasons.append("complete_full_construct_structure_required")
    learned = {m["source"] for m in ok if m["evidence"] == "model_prediction" and m["independent"] and not m["optimized_by_generator"]}
    if len(learned) < POLICY["minimum_independent_predictors"]:
        reasons.append("at_least_two_independent_predictors_required")
    for objective in [o for o in brief.objectives if o.role == "hard"]:
        values=[m["value"] for m in ok if m["metric"] == objective.metric and (objective.stage == "computational" or m["evidence"] == "measured_this_context")]
        def satisfied(v):
            return v >= objective.threshold if objective.direction == "maximize" else v <= objective.threshold if objective.direction == "minimize" else abs(v-objective.target) <= objective.threshold
        if not values or not all(satisfied(v) for v in values):
            reasons.append("current_hard_constraint_unmet:"+objective.metric)
    primary = [o for o in brief.objectives if o.role == "primary"]
    if not primary:
        reasons.append("explicit_primary_objective_required")
    for objective in [o for o in brief.objectives if o.role in {"primary", "secondary"}]:
        valid = [m for m in ok if m["metric"] == objective.metric and (objective.role != "primary" or (m["independent"] and not m["optimized_by_generator"]))
                 and (m["evidence"] == "measured_this_context" if objective.stage == "experimental" else m["evidence"] in {"model_prediction", "calculated"})]
        if objective.role == "primary":
            valid = [m for m in valid if m["evidence"] in {"model_prediction", "measured_this_context"}]
        comparable = []
        for value in valid:
            ref = next((m for m in base_metrics if m["status"] == "ok" and all(m.get(k) == value.get(k) for k in ("metric", "source", "version", "unit", "scope", "evidence"))), None)
            if not ref or not math.isfinite(ref["value"]):
                continue
            gain = value["value"] - ref["value"]
            if objective.direction == "minimize": gain = -gain
            if objective.direction == "target": gain = abs(ref["value"]-objective.target)-abs(value["value"]-objective.target)
            normalized = gain / max(1.0, abs(ref["value"]))
            item = {"metric": objective.metric, "source": value["source"], "value": value["value"], "baseline": ref["value"], "unit": value["unit"], "directional_gain": gain, "scaled_gain": normalized}
            comparisons.append(item); comparable.append(normalized)
        if not comparable:
            reasons.append("independent_matched_endpoint_required:" + objective.metric)
        elif min(comparable) < (POLICY["minimum_primary_gain"] if objective.role == "primary" else -1e-9):
            reasons.append("insufficient_gain_or_regression:" + objective.metric)
    return {"id": candidate.id, "status": "filtered_out" if reasons else "high_potential", "reasons": sorted(set(reasons)),
            "comparisons": comparisons, "independent_predictors": sorted(learned), "policy_id": POLICY_ID,
            "evaluation_id": (evaluation or {}).get("evaluation_id")}
