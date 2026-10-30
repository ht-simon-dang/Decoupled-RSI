"""Independent full-construct validation and evidence-preserving multiobjective selection."""
from __future__ import annotations
import math
from collections import Counter
from difflib import SequenceMatcher
import importlib.metadata
import numpy as np

from .contracts import Brief, Construct, Measurement, translate, identity
from .registry import MODELS, manifest
from .filtering import assess, POLICY, POLICY_ID


def check_construct(construct: Construct, brief: Brief):
    checks = []
    def add(name, passed, reason):
        checks.append({"name": name, "stage": "computational", "status": "pass" if passed else "fail", "reason": reason})
    try:
        p = translate(construct.regions.cds)
        add("protein_identity", p == brief.protein, "Complete translation must match the locked protein")
    except ValueError as exc:
        add("protein_identity", False, str(exc))
    for name in sorted(brief.locks):
        add("fixed_" + name, getattr(construct.regions, name) == getattr(brief.reference, name), "Exact reference region identity")
    for span in brief.locked_spans:
        actual, ref = getattr(construct.regions, span.region), getattr(brief.reference, span.region)
        add(f"fixed_{span.region}_{span.start}_{span.end}", actual[span.start:span.end] == ref[span.start:span.end], "0-based half-open locked span")
    for motif in brief.required_motifs:
        add("required_" + motif, motif in construct.regions.transcript, "Required motif in assembled transcript")
    for motif in brief.forbidden_motifs:
        add("forbidden_" + motif, motif not in construct.regions.transcript, "Forbidden motif in assembled transcript")
    add("molecular_configuration", construct.molecule == brief.molecule or construct.sources.get("baseline") is True,
        "Configuration is explicit; original baseline can retain its original configuration")
    return checks


def missing(metric, source, reason, status="not_evaluated", unit="unknown"):
    return Measurement(metric=metric, status=status, unit=unit, source=source, version="not_available",
                       evidence="not_evaluated", scope=MODELS.get(source, {}).get("scope", "Context-specific evidence needed"), reason=reason).model_dump()


def measured(metric, value, unit, source, version, scope, *, prediction=False, optimized=False, independent=True, provenance=None, uncertainty=None):
    return Measurement(metric=metric, status="ok", value=float(value), unit=unit, source=source, version=version,
                       evidence="model_prediction" if prediction else "calculated", scope=scope,
                       optimized_by_generator=optimized, independent=independent and not optimized,
                       provenance=provenance or {}, uncertainty=uncertainty).model_dump()


def basic(construct):
    seq = construct.sequence
    version = "sequence-rules-v1"
    result = [measured("length_nt", len(seq), "nt", "sequence_rules", version, "Entire molecule including explicit poly(A)"),
              measured("gc_fraction", (seq.count("G") + seq.count("C")) / len(seq), "fraction", "sequence_rules", version, "Entire molecule including explicit poly(A)")]
    upstream_aug = [i for i in range(len(construct.regions.utr5) - 2) if construct.regions.utr5[i:i+3] == "AUG"]
    import re
    homopolymer = max((len(m.group()) for m in re.finditer(r"A+|C+|G+|U+", construct.regions.transcript)), default=0)
    return result, {"upstream_aug_positions": upstream_aug, "longest_transcript_homopolymer": homopolymer,
                    "start_context": (construct.regions.utr5[-6:] + construct.regions.cds[:9]),
                    "notes": ["Motifs and GC do not establish immune safety or manufacturing quality", "Upstream AUGs are potential translation events, not experimentally verified uORFs"]}


def vienna(construct):
    import RNA
    seq = construct.sequence
    scope = MODELS["vienna"]["scope"]
    if len(seq) > 2000:
        return [missing(k, "vienna", "Full molecule exceeds 2000 nt application resource limit; no truncation performed", "out_of_scope", u) for k, u in [("mfe", "kcal/mol"), ("ensemble_free_energy", "kcal/mol"), ("start_accessibility", "mean unpaired probability")]], {}
    md = RNA.md()
    md.temperature = 37.0
    fc = RNA.fold_compound(seq, md)
    structure, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    ensemble_structure, ensemble = fc.pf()
    bpp = fc.bpp()
    start = len(construct.regions.utr5)
    a, b = max(0, start-12), min(len(seq), start+15)
    unpaired = []
    for zero in range(a, b):
        i = zero+1
        paired = sum(bpp[j][i] for j in range(1, i)) + sum(bpp[i][j] for j in range(i+1, len(seq)+1))
        unpaired.append(max(0.0, min(1.0, 1-paired)))
    optimized = "mfe" in construct.optimized_metrics
    provenance = {"temperature_c": 37, "sequence_scope": "full molecule including poly(A)", "start_window": [a,b],
                  "chemistry_caveat": "Standard RNA parameters; modification not represented in thermodynamics"}
    result = [measured("mfe", mfe, "kcal/mol", "vienna", RNA.__version__, scope, optimized=optimized, provenance=provenance),
              measured("ensemble_free_energy", ensemble, "kcal/mol", "vienna", RNA.__version__, scope, independent=not optimized, provenance=provenance),
              measured("start_accessibility", np.mean(unpaired), "mean unpaired probability", "vienna", RNA.__version__, scope, independent=not optimized, provenance=provenance)]
    # Full structure and local accessibility share one ensemble; they are not independent votes.
    return result, {"mfe_dot_bracket": structure, "ensemble_annotation": ensemble_structure,
                    "start_window": [a, b], "start_unpaired_probabilities": unpaired,
                    "ensemble_diversity": float(fc.mean_bp_distance())}


def predict_batch(constructs, brief, gateway):
    output = {c.id: [] for c in constructs}
    if "ribonn" in brief.evaluators:
        eligible = []
        for c in constructs:
            if brief.context.species not in {"human", "mouse"} or len(c.regions.utr5) > 1381 or len(c.regions.cds+c.regions.utr3) > 11937:
                output[c.id].append(missing("translation_efficiency", "ribonn", "Unsupported species or region length", "out_of_scope"))
            else:
                eligible.append(c)
        if eligible:
            result = gateway.call("ribonn", {"constructs": [c.card() for c in eligible], "species": brief.context.species})
            if result["status"] != "ok":
                for c in eligible:
                    output[c.id].append(missing("translation_efficiency", "ribonn", result.get("reason", "Tool unavailable"), result["status"]))
            else:
                rows = {r["id"]: r for r in result["data"]["predictions"]}
                if set(rows) != {c.id for c in eligible}:
                    raise ValueError("RiboNN candidate identity mismatch")
                for c in eligible:
                    row = rows[c.id]
                    cell = "TE_" + brief.context.target_cell
                    common = dict(source="ribonn", version=manifest().get("RiboNN", {}).get("revision", "installed"), scope=MODELS["ribonn"]["scope"], prediction=True,
                                  provenance={**result.get("provenance", {}), "ensemble": result["data"]["ensemble"], "model_ids": result["data"]["model_ids"], "cell_scores": row["scores"], "domain_shift": "IVT RNA / delivery / modification not calibrated", "input": "5UTR-CDS-3UTR; excludes poly(A) per author input convention"})
                    if cell in row["scores"]:
                        output[c.id].append(measured("translation_efficiency", row["scores"][cell], "normalized TE (model scale)", **common,
                                                     uncertainty={"ensemble_sd": row["ensemble_sd"][cell], "calibrated": False, "interpretation": "Model dispersion, not confidence interval"}))
                    else:
                        output[c.id].append(missing("translation_efficiency", "ribonn", "Requested cell is not an author model output; mean across cells is separately labeled", "out_of_scope"))
                    output[c.id].append(measured("mean_cell_te", np.mean(list(row["scores"].values())), "mean normalized TE across author cells", **common))
    if "optimus5" in brief.evaluators:
        eligible = [c for c in constructs if len(c.regions.utr5) == 50 and c.molecule.modification == "m1Psi"]
        for c in constructs:
            if c not in eligible:
                output[c.id].append(missing("ribosome_load", "optimus5", "Requires exactly 50 nt 5UTR and m1Psi; no trimming", "out_of_scope", "MRL"))
        if eligible:
            result = gateway.call("smart", {"kind": "optimus5", "sequences": [c.regions.utr5 for c in eligible]})
            for i, c in enumerate(eligible):
                if result["status"] != "ok":
                    output[c.id].append(missing("ribosome_load", "optimus5", result.get("reason", "Tool failure"), result["status"], "MRL"))
                else:
                    output[c.id].append(measured("ribosome_load", result["data"]["scores"][i], "MRL", "optimus5", manifest().get("Smart5UTR", {}).get("revision", "installed"), MODELS["optimus5"]["scope"], prediction=True,
                                                 independent="ribosome_load" not in c.optimized_metrics, provenance={"dataset_lineage": "GSE114002/egfp_m1pseudo2", "correlated_with": ["Smart5UTR"], **result.get("provenance", {})}))
    if "parade" in brief.evaluators:
        cell = brief.context.target_cell
        for region, n, codes in [("utr5", 50, {"c1", "c2", "c4", "c6", "c17"}), ("utr3", 240, {"c1", "c2", "c4", "c6", "c17", "c13"})]:
            eligible = [c for c in constructs if len(getattr(c.regions, region)) == n and cell in codes and set(brief.context.non_target_cells) <= codes and c.molecule.modification in {"unmodified", "unspecified"}]
            for c in constructs:
                if c not in eligible:
                    output[c.id].append(missing("parade_"+region+"_activity", "parade", "Exact region length, chemistry and author cell codes required", "out_of_scope"))
            if not eligible:
                continue
            result = gateway.call("parade", {"region": region, "sequences": [getattr(c.regions, region) for c in eligible], "cells": list(dict.fromkeys([cell]+brief.context.non_target_cells))})
            for i,c in enumerate(eligible):
                if result["status"] != "ok":
                    output[c.id].append(missing("parade_"+region+"_activity", "parade", result.get("reason", "Tool failure"), result["status"]))
                else:
                    scores = result["data"]["predictions"][i]["scores"]
                    for code, value in scores.items():
                        output[c.id].append(measured("parade_"+region+"_"+code, value, "reporter mass-center score", "parade", manifest().get("parade", {}).get("revision", "installed"), MODELS["parade"]["scope"], prediction=True,
                                                     optimized="parade_activity" in c.optimized_metrics, provenance={"region": region, "cell_code": code, "absolute_target_protein": "not_evaluated", **result.get("provenance", {})}))
    from .optional_evaluation import predict
    return predict(constructs,brief,gateway,output)


def evaluate(construct, brief, predictions):
    checks = check_construct(construct, brief)
    values, features = basic(construct)
    structure = {}
    if "vienna" in brief.evaluators:
        try:
            more, structure = vienna(construct)
            values.extend(more)
        except Exception as exc:
            values.extend(missing(k, "vienna", str(exc), "error") for k in ["mfe", "start_accessibility", "ensemble_free_energy"])
    values.extend(predictions)
    for objective in brief.objectives:
        value = next((v for v in values if v["metric"] == objective.metric and v["status"] == "ok"), None)
        if value is None:
            values.append(missing(objective.metric, "experimental_endpoint", "No matched measurement or calibrated predictor for this endpoint"))
        if objective.role == "hard":
            if objective.stage == "experimental" or value is None or value["status"] != "ok":
                status = "pending"
            else:
                actual = value["value"]
                passed = actual >= objective.threshold if objective.direction == "maximize" else actual <= objective.threshold if objective.direction == "minimize" else abs(actual-objective.target) <= objective.threshold
                status = "pass" if passed else "fail"
            checks.append({"name": objective.metric, "stage": objective.stage, "status": status, "reason": "Task endpoint constraint; missing evidence is pending"})
    failed = any(x["status"] == "fail" for x in checks)
    pending = any(x["status"] == "pending" for x in checks)
    return {"protocol": POLICY_ID, "construct_id": construct.id, "context_id": identity(brief.context.model_dump()), "context": brief.context.model_dump(),
            "checks": checks, "metrics": values, "features": features, "structure": structure,
            "status": "rejected" if failed else "pending_constraints" if pending else "computationally_valid",
            "experimental_success": False, "missing_endpoints": [v["metric"] for v in values if v["status"] != "ok"],
            "next_evidence": "Matched protein time course, RNA persistence, identity/function and quality controls; record batch, dose, delivery and modifications"}


def dominates(a, b, objectives):
    # Compare only identical complete evidence support. Unknown cannot lose to a number.
    def vals(row):
        return {m["metric"]: m for m in row["metrics"] if m["status"] == "ok"}
    av, bv = vals(a), vals(b)
    dimensions = [o for o in objectives if o.role in {"primary", "secondary"} and o.stage == "computational"]
    keys = {o.metric for o in dimensions}
    if not keys or not keys <= av.keys() or not keys <= bv.keys():
        return False
    better = False
    for o in dimensions:
        x, y = av[o.metric], bv[o.metric]
        if (x["source"],x["version"],x["unit"],x["scope"]) != (y["source"],y["version"],y["unit"],y["scope"]):
            return False
        u, v = x["value"], y["value"]
        if o.direction == "minimize": u,v = -u,-v
        elif o.direction == "target": u,v = -abs(u-o.target),-abs(v-o.target)
        if u < v: return False
        better |= u > v
    return better


def portfolio(candidates, evaluations, brief):
    cards = {c.id: c for c in candidates}
    by_id = {e["construct_id"]: e for e in evaluations}
    baseline = next((c for c in candidates if c.sources.get("baseline")), None)
    gates = [assess(c, by_id.get(c.id), baseline, by_id.get(baseline.id) if baseline else None, brief) for c in candidates]
    allowed = {g["id"] for g in gates if g["status"] == "high_potential"}
    valid = [e for e in evaluations if e["construct_id"] in allowed]
    frontier = [e["construct_id"] for e in valid if not any(dominates(other,e,brief.objectives) for other in valid if other is not e)]
    chosen, remaining = [], list(frontier)
    slots = min(brief.budget.shortlist,brief.budget.synthesis_slots if brief.budget.synthesis_slots is not None else brief.budget.shortlist)
    while remaining and len(chosen) < slots:
        best = max(remaining,key=lambda x:(min((1-SequenceMatcher(None,cards[x].sequence,cards[y].sequence,autojunk=False).ratio() for y in chosen),default=1),x))
        chosen.append(best);remaining.remove(best)
    return {"frontier":frontier,"selected":[{"id":x,"role":"high_potential","reason":"Passed independent evaluation and strict baseline-improvement filter","policy_id":POLICY_ID} for x in chosen],
            "filter_policy":POLICY,"filter_policy_id":POLICY_ID,"filter_results":gates,
            "rejected":[{"id":e["construct_id"],"reasons":[x for x in e["checks"] if x["status"]=="fail"]} for e in evaluations if e["status"]=="rejected"],
            "ranking_basis":"Strict evaluation gate, then Pareto frontier and sequence diversity; no filling with failed candidates",
            "unmeasured_primary_endpoints":[o.metric for o in brief.objectives if o.role=="primary" and not any(m["metric"]==o.metric and m["status"]=="ok" for e in valid for m in e["metrics"])]}
