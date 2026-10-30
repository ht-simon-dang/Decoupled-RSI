"""Context-bound, append-only experimental feedback and conservative calibration."""
from __future__ import annotations
import json
import time
from collections import defaultdict
from typing import Literal
import numpy as np
from pydantic import Field, model_validator
from .contracts import Contract, Context, Brief, identity, canonical
from .storage import quota


class Experiment(Contract):
    construct_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    context: Context
    metric: Literal["translation_efficiency", "ribosome_load", "rna_abundance", "rna_stability", "chemical_degradation", "protein_concentration", "protein_peak", "protein_auc", "functional_duration", "target_expression", "non_target_expression", "protein_function", "immune_response", "manufacturing_quality"]
    value: float | None = None
    unit: str = Field(min_length=1, max_length=100)
    time_hours: float = Field(ge=0)
    batch: str = Field(min_length=1, max_length=100)
    replicate: str = Field(min_length=1, max_length=100)
    control_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    study: str = Field(min_length=1, max_length=200)
    assay: str = Field(min_length=1, max_length=200)
    quality: Literal["valid", "technical_failure", "manufacturing_failure", "quality_failed", "missing"]
    notes: str = Field(default="", max_length=2000)
    dataset_role: Literal["experimental", "synthetic_fixture"] = "experimental"

    @model_validator(mode="after")
    def values(self):
        if self.quality == "valid" and self.value is None:
            raise ValueError("Valid measurements need a value; missing is not zero")
        if self.quality in {"technical_failure", "missing"} and self.value is not None:
            raise ValueError("Technical failure/missing must not provide an efficacy value")
        return self


def import_experiments(store, records):
    if not 1 <= len(records) <= 20000:
        raise ValueError("Import 1..20000 records")
    parsed = [Experiment.model_validate(r) for r in records]
    keys = []
    for r in parsed:
        store.construct(r.construct_id); store.construct(r.control_id)
    # Validate the complete batch before an atomic insertion; duplicate identity with a new
    # value is a conflict, not an invisible correction to historical measurements.
    with quota(sum(len(canonical(r.model_dump())) for r in parsed) * 3 + 1_000_000), store.db:
        for r in parsed:
            body = r.model_dump()
            key = identity({k: body[k] for k in ("construct_id", "context", "metric", "time_hours", "batch", "replicate", "study", "assay", "dataset_role")})
            old = store.db.execute("SELECT body FROM experiments WHERE id=?", (key,)).fetchone()
            if old and json.loads(old[0]) != body:
                raise ValueError("Conflicting repeated measurement identity; use a new study/assay correction record")
            store.db.execute("INSERT OR IGNORE INTO experiments VALUES(?,?,?,?,?)", (key, r.construct_id, identity(r.context.model_dump()), canonical(body), time.time()))
            keys.append(key)
    return {"ids": keys, "received": len(records), "distinct": len(set(keys)), "invalid_for_efficacy": sum(r.quality != "valid" for r in parsed)}


def analyze_feedback(store, run_id, brief=None, payload=None):
    run = store.get(run_id)
    brief = brief or Brief.model_validate(run["brief"])
    payload = payload or run["payload"]
    context_id = identity(brief.context.model_dump())
    ids = {c["id"] for c in payload.get("candidates", [])}
    # Include a linked prior task only within exactly the same biological/measurement context.
    if brief.parent_run:
        parent = store.get(brief.parent_run)
        ids.update(c["id"] for c in parent["payload"].get("candidates", []))
    all_records = [json.loads(r[0]) for r in store.db.execute("SELECT body FROM experiments WHERE context_id=? ORDER BY created", (context_id,))]
    records = [r for r in all_records if r["construct_id"] in ids]
    valid = [r for r in records if r["quality"] == "valid" and r["dataset_role"] == "experimental"]
    groups = defaultdict(list)
    for r in valid:
        group = (r["metric"], r["unit"], r["time_hours"], r["batch"], r["study"], r["assay"], r["control_id"])
        groups[group].append(r)
    comparisons = []
    for group, rows in groups.items():
        controls = [r["value"] for r in rows if r["construct_id"] == r["control_id"]]
        for key in sorted({r["construct_id"] for r in rows}):
            vals = [r["value"] for r in rows if r["construct_id"] == key]
            comparisons.append({"construct_id": key, "metric": group[0], "unit": group[1], "time_hours": group[2],
                                "batch": group[3], "study": group[4], "assay": group[5], "control_id": group[6],
                                "mean": float(np.mean(vals)), "n": len(vals), "sample_sd": float(np.std(vals, ddof=1)) if len(vals)>1 else None,
                                "baseline_delta": float(np.mean(vals)-np.mean(controls)) if controls else None,
                                "comparable": bool(controls), "evidence": "measured_this_context"})
    trajectories = []
    curves = defaultdict(list)
    for r in valid:
        if r["metric"] in {"protein_concentration","rna_abundance"}:
            curves[(r["construct_id"],r["metric"],r["unit"],r["batch"],r["study"],r["assay"],r["replicate"],r["control_id"])].append(r)
    for group,rows in curves.items():
        rows=sorted(rows,key=lambda r:r["time_hours"])
        if len(rows)<2:continue
        t=np.array([r["time_hours"] for r in rows]);y=np.array([r["value"] for r in rows])
        window=brief.context.time_window_hours
        covers=window is None or (float(t[0])<=window[0] and float(t[-1])>=window[1])
        # Integrate only the registered window when bracketed by observations; never extrapolate.
        if window and covers:
            grid=np.unique(np.concatenate(([window[0]],t[(t>window[0])&(t<window[1])],[window[1]])))
            yy=np.interp(grid,t,y)
        else:grid,yy=t,y
        trajectories.append({"construct_id":group[0],"metric":group[1],"unit":group[2],"batch":group[3],"study":group[4],"assay":group[5],"replicate":group[6],"control_id":group[7],
                             "observed_peak":float(y.max()),"observed_peak_time_hours":float(t[int(y.argmax())]),
                             "auc":float(np.trapezoid(yy,grid)),"auc_unit":group[2]+" * h","integration_window_hours":[float(grid[0]),float(grid[-1])],
                             "requested_window_covered":covers,"method":"trapezoidal observed/interpolated values; no extrapolation",
                             "half_life_hours":None,"functional_duration_hours":None,"reason":"RNA decay and functional duration need their own identifiable measurements/models"})
    diagnostics = []
    evaluators = payload.get("evaluations", [])
    for metric in {r["metric"] for r in valid}:
        matched = []
        for e in evaluators:
            pred = next((m for m in e["metrics"] if m["metric"] == metric and m["status"] == "ok"), None)
            matched_rows = [r for r in comparisons if r["construct_id"] == e["construct_id"] and r["metric"] == metric and r["comparable"]]
            if pred and matched_rows:
                # Never average incompatible batches/assays into a calibration target.
                for row in matched_rows:
                    matched.append({"construct_id": e["construct_id"], "prediction": pred["value"], "prediction_unit": pred["unit"], "observed": row["mean"], "observed_unit": row["unit"], "source": pred["source"], "group": (row["batch"],row["study"],row["assay"],row["time_hours"],row["unit"])})
        strata = defaultdict(list)
        for row in matched: strata[row["group"]].append(row)
        for group, rows in strata.items():
            rho = None
            if len(rows)>=3 and len({r["prediction"] for r in rows})>1 and len({r["observed"] for r in rows})>1:
                from scipy.stats import spearmanr
                rho = float(spearmanr([r["prediction"] for r in rows], [r["observed"] for r in rows]).statistic)
            diagnostics.append({"metric": metric, "group": group, "n_constructs": len(rows), "spearman": rho,
                                "trust": "downweight_in_this_context" if rho is not None and rho<0 else "rank_support_in_this_context" if rho is not None and rho>0.5 else "insufficient_evidence",
                                "records": rows, "calibrated": False, "reason": "Rank association only; no calibrated confidence interval or absolute output claim"})
    # Select by the first declared primary endpoint in comparable measured strata.
    primary = next((o for o in brief.objectives if o.role == "primary"), brief.objectives[0])
    usable = [r for r in comparisons if r["metric"] == primary.metric and r["comparable"]]
    next_batch = []
    if usable:
        first = usable[0]
        stratum = [r for r in usable if all(r[k] == first[k] for k in ("batch","study","assay","time_hours","unit","control_id"))]
        key = (lambda r: -abs(r["mean"]-primary.target)) if primary.direction == "target" else (lambda r: r["mean"] * (1 if primary.direction == "maximize" else -1))
        winner = max(stratum, key=key)
        next_batch.append({"id": winner["construct_id"], "role": "measured_followup", "reason": "Best observed primary endpoint in one matched control/batch/time stratum; retain paired comparisons"})
    measured_ids = {r["construct_id"] for r in valid}
    for item in payload.get("portfolio", {}).get("selected", []):
        if item["id"] not in {x["id"] for x in next_batch}:
            next_batch.append({**item, "reason": "Preserve control" if item["role"] == "control" else "Explore an unmeasured candidate" if item["id"] not in measured_ids else "Replicate to resolve uncertainty"})
    result = {"context_id": context_id, "records": len(records), "valid_experimental_records": len(valid),
              "excluded": [{"construct_id": r["construct_id"], "quality": r["quality"], "dataset_role": r["dataset_role"]} for r in records if r not in valid],
              "comparisons": comparisons, "trajectories":trajectories,"model_diagnostics": diagnostics, "next_batch": next_batch[:brief.budget.shortlist],
              "attribution": "Changes in sequence/configuration/delivery are not assigned solely to sequence; identical context and controls required",
              "raw_predictions_preserved": True, "experimental_success": False}
    key = store.artifact(result)
    store.event(run_id, "feedback.analyzed", {"artifact": key, "records": len(valid), "next_batch": result["next_batch"]})
    return result


def fit_calibration(records):
    """Explicit held-out group protocol; never uses test observations to fit or select."""
    if len(records)<12:
        raise ValueError("Need at least 12 grouped prediction/measurement pairs")
    required = {"prediction", "observed", "group", "split", "context_id", "metric", "unit"}
    if any(not required <= r.keys() for r in records): raise ValueError("Incomplete calibration records")
    signature = {(r["context_id"],r["metric"],r["unit"]) for r in records}
    if len(signature)!=1: raise ValueError("Calibration must use one exact context, endpoint and unit")
    groups = {s:{r["group"] for r in records if r["split"]==s} for s in ("train","calibration","test")}
    if any(not g for g in groups.values()) or any(groups[a]&groups[b] for a,b in [("train","calibration"),("train","test"),("calibration","test")]):
        raise ValueError("Protein/family/batch/source groups must be disjoint across splits")
    arrays = {s:np.array([[r["prediction"],r["observed"]] for r in records if r["split"]==s],dtype=float) for s in groups}
    if not all(np.isfinite(x).all() and len(x)>=3 for x in arrays.values()): raise ValueError("Each split needs >=3 finite pairs")
    train = arrays["train"]
    if np.std(train[:,0])==0: raise ValueError("Constant calibration predictor")
    slope, offset = np.linalg.lstsq(np.column_stack((train[:,0],np.ones(len(train)))),train[:,1],rcond=None)[0]
    residual = np.abs(arrays["calibration"][:,1]-(slope*arrays["calibration"][:,0]+offset))
    # Finite-sample conformal quantile; very small calibration sets produce an unbounded interval.
    k = int(np.ceil((len(residual)+1)*0.9))
    radius = float(np.sort(residual)[k-1]) if k<=len(residual) else None
    test = arrays["test"]; errors = np.abs(test[:,1]-(slope*test[:,0]+offset))
    return {"slope": float(slope), "offset": float(offset), "interval_radius_90": radius,
            "interval_status": "finite" if radius is not None else "unbounded_small_calibration_set", "test_mae": float(errors.mean()),
            "test_coverage": float(np.mean(errors<=radius)) if radius is not None else None,
            "split_groups": {k:sorted(v) for k,v in groups.items()}, "context": list(signature)[0],
            "assumption": "Exchangeable calibration/test observations within registered context; group transfer not guaranteed", "training_data_hash": identity(records)}
