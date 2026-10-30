"""Measured peptide-target pKd regression with a separate conformal calibration split."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from pydantic import Field
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from .benchmark import features
from .contracts import AA, Contract, Sequence, Target, ToolResult, Unavailable
from .models import model_root


class AffinityRequest(Contract):
    sequences: list[Sequence] = Field(min_length=1, max_length=128)
    target_sequence: Target
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")


def vector(sequence, target):
    return features(sequence) + [target.count(a)/len(target) for a in AA]


def train(csv_text, model_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", model_id):
        raise ValueError("Invalid model id")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not 12 <= len(rows) <= 4000:
        raise ValueError("Provide 12..4000 measured records")
    if not {"sequence", "target_sequence", "pkd", "split", "group"}.issubset(rows[0]):
        raise ValueError("Required columns: sequence,target_sequence,pkd,split,group")
    groups, pairs = {}, set()
    for r in rows:
        AffinityRequest(sequences=[r["sequence"]], target_sequence=r["target_sequence"], model_id=model_id)
        value = float(r["pkd"])
        if not np.isfinite(value) or not -3 <= value <= 20:
            raise ValueError("pKd must be finite and expressed as -log10(Kd in mol/L)")
        if r["split"] not in {"train", "calibration", "test"} or not r["group"]:
            raise ValueError("Require train/calibration/test and nonempty homology/target groups")
        if r["group"] in groups and groups[r["group"]] != r["split"]:
            raise ValueError("Group leakage across splits")
        groups[r["group"]] = r["split"]
        pair = r["sequence"], r["target_sequence"]
        if pair in pairs: raise ValueError("Duplicate peptide-target pair")
        pairs.add(pair)
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "calibration", "test")}
    if min(map(len, splits.values())) < 4:
        raise ValueError("Each split requires at least four records")
    x = {s: np.array([vector(r["sequence"], r["target_sequence"]) for r in part]) for s, part in splits.items()}
    y = {s: np.array([float(r["pkd"]) for r in part]) for s, part in splits.items()}
    scaler = StandardScaler().fit(x["train"])
    model = Ridge(alpha=10.0).fit(scaler.transform(x["train"]), y["train"])
    residual = np.abs(y["calibration"] - model.predict(scaler.transform(x["calibration"])))
    # Finite-sample split-conformal rank; require enough data for a finite 80% interval.
    radius = float(np.sort(residual)[int(np.ceil((len(residual)+1)*0.8))-1])
    prediction = model.predict(scaler.transform(x["test"]))
    model_data = {"id": model_id, "method": "AAC/descriptors Ridge + split-conformal", "endpoint": "pKd",
                  "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                  "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
                  "interval_radius": radius, "nominal_coverage": 0.8,
                  "dataset_sha256": hashlib.sha256(csv_text.encode()).hexdigest(),
                  "counts": {s: len(p) for s, p in splits.items()},
                  "test_rmse": float(root_mean_squared_error(y["test"], prediction)),
                  "test_mae": float(mean_absolute_error(y["test"], prediction)),
                  "test_coverage": float(np.mean(np.abs(y["test"]-prediction) <= radius)),
                  "domain": "canonical linear peptides; supplied assay distribution only",
                  "train_targets": sorted({r["target_sequence"] for r in splits["train"]})}
    root = model_root() / "calibrations"; root.mkdir(parents=True, exist_ok=True)
    path = root / (model_id + ".json")
    if path.exists(): raise ValueError("Model id already exists; use a new versioned id")
    path.write_text(json.dumps(model_data, allow_nan=False))
    return model_data


def predict(request: AffinityRequest):
    path = model_root() / "calibrations" / (request.model_id + ".json")
    if not path.exists(): raise Unavailable("Calibrated model missing; supply measured pKd train/calibration/test data")
    m = json.loads(path.read_text())
    results = []
    for seq in request.sequences:
        x = (np.asarray(vector(seq, request.target_sequence))-m["mean"])/m["scale"]
        pkd = float(x @ m["coef"] + m["intercept"])
        results.append({"sequence": seq, "predicted_pkd": pkd,
                        "interval_80": [pkd-m["interval_radius"], pkd+m["interval_radius"]]})
    return ToolResult(tool="affinity.calibrated", status="ok", trust="PREDICTED", data={"candidates": results},
                      provenance={k: m[k] for k in ("id", "method", "dataset_sha256", "counts", "test_rmse", "test_coverage")},
                      warnings=["Intervals rely on exchangeability with supplied calibration data; shifted targets can invalidate coverage.",
                                "User-supplied labels are not independently verified experimental measurements."])
