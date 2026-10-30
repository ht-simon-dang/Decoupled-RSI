"""Condition-matched experimental property models and gated batch selection.

This is an explicit ridge/conformal baseline, not pretrained LaMBO-2 weights.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .config import data_root, digest, encode
from .contracts import AntigenState, Binder, Contract, Identifier, Product
from .delivery import public_run

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class Domain(Contract):
    endpoint: str = Field(min_length=1, max_length=100)
    unit: str = Field(min_length=1, max_length=100)
    conditions: dict[str, str | float | bool] = Field(min_length=1)
    antigen: AntigenState
    product: Product


class TrainingRow(Contract):
    id: Identifier
    binder: Binder
    lineage: Identifier
    split: Literal["train", "calibration", "test"]
    status: Literal["measured", "censored", "technical_failure"]
    value: float | None = None
    source: str = Field(min_length=1)
    measurement_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def value_present(self):
        if self.status == "measured" and self.value is None:
            raise ValueError("Measured training rows require numeric values")
        if self.status == "technical_failure" and self.value is not None:
            raise ValueError("Technical failure cannot contain a property value")
        return self


class TrainingRequest(Contract):
    model_id: Identifier
    domain: Domain
    rows: list[TrainingRow] = Field(min_length=16, max_length=10000)


class Objective(Contract):
    model_id: Identifier
    direction: Literal["min", "max"]


class PredictionRequest(Contract):
    objectives: list[Objective] = Field(min_length=1, max_length=5)
    count: int = Field(default=4, ge=1, le=16)


def features(binder):
    chains = {c.role: c.sequence for c in binder.chains}
    values = []
    for role in ("VH", "VL", "VHH"):
        sequence = chains.get(role, "")
        values.extend(sequence.count(aa) / max(1, len(sequence)) for aa in AMINO_ACIDS)
        values.append(len(sequence) / 150)
    return np.asarray(values, dtype=float)


def model_file(model_id):
    # Use the same identifier validator as training, including path separators.
    validated = Objective(model_id=model_id, direction="min")
    return data_root() / "property_models" / (validated.model_id + ".json")


def fit(request):
    request = TrainingRequest.model_validate(request)
    path = model_file(request.model_id)
    if path.exists():
        raise ValueError("Model version already exists; sealed models cannot be overwritten")
    if len({r.id for r in request.rows}) != len(request.rows):
        raise ValueError("Training record ids must be unique")
    seen_lineages, seen_sequences, seen_measurements = {}, {}, {}
    for row in request.rows:
        if row.binder.product != request.domain.product:
            raise ValueError("All training measurements must use the declared final product format")
        identity = digest(sorted((c.role, c.sequence) for c in row.binder.chains))
        for table, key in (
            (seen_lineages, row.lineage),
            (seen_sequences, identity),
            (seen_measurements, row.measurement_id),
        ):
            if key in table and table[key] != row.split:
                raise ValueError("Lineage, sequence or original measurement leakage across frozen splits")
            table[key] = row.split
    # Retain failures/censoring in the dataset fingerprint and counts; do not relabel them.
    pools = {
        split: [r for r in request.rows if r.status == "measured" and r.split == split]
        for split in ("train", "calibration", "test")
    }
    if len(pools["train"]) < 8 or len(pools["calibration"]) < 4 or len(pools["test"]) < 4:
        raise ValueError("Need >=8 train, >=4 calibration and >=4 held-out test measured rows")
    for split, pool in pools.items():
        if len({r.lineage for r in pool}) < (4 if split == "train" else 2):
            raise ValueError("Too few independent lineages in a split")
    x = np.stack([features(r.binder) for r in pools["train"]])
    y = np.array([r.value for r in pools["train"]])
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1

    def matrix(rows):
        normalized = (np.stack([features(r.binder) for r in rows]) - mean) / scale
        return np.column_stack([np.ones(len(rows)), normalized])

    design = matrix(pools["train"])
    penalty = np.eye(design.shape[1]) * 10
    penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    calibration_errors = np.abs(
        matrix(pools["calibration"]) @ weights - np.array([r.value for r in pools["calibration"]])
    )
    # Compute one residual per lineage to avoid counting replicate rows as independent calibration units.
    group_errors = {}
    for row, error in zip(pools["calibration"], calibration_errors, strict=True):
        group_errors[row.lineage] = max(group_errors.get(row.lineage, 0), float(error))
    if len(group_errors) < 4:
        raise ValueError("80% conformal calibration requires at least four independent calibration lineages")
    k = math.ceil((len(group_errors) + 1) * 0.8)
    radius = sorted(group_errors.values())[k - 1]
    truth = np.array([r.value for r in pools["test"]])
    prediction = matrix(pools["test"]) @ weights
    dataset = request.model_dump()
    result = {
        "model_id": request.model_id,
        "method": "paired-chain composition ridge (alpha=10) + grouped split conformal",
        "feature_version": "chain-composition-v1",
        "domain": request.domain.model_dump(),
        "dataset_sha256": digest(dataset),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "interval_radius": radius,
        "nominal_coverage": 0.8,
        "counts": {s: len(p) for s, p in pools.items()},
        "excluded": {s: sum(r.status == s for r in request.rows) for s in ("censored", "technical_failure")},
        "test": {
            "mae": float(np.abs(prediction - truth).mean()),
            "mean_baseline_mae": float(np.abs(y.mean() - truth).mean()),
            "interval_coverage": float((np.abs(prediction - truth) <= radius).mean()),
            "lineages": len({r.lineage for r in pools["test"]}),
            "rows": len(truth),
        },
        "limitations": [
            "Within-target, condition-matched baseline only",
            "No clinical or cross-antigen generalization claim",
            "Frozen test data is evaluation-only; no automatic workflow/model promotion",
        ],
        "split_manifest": [
            {
                "id": r.id,
                "lineage": r.lineage,
                "split": r.split,
                "status": r.status,
                "measurement_id": r.measurement_id,
            }
            for r in request.rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent requests from replacing a frozen model version.
    with path.open("x") as handle:
        handle.write(encode({"model": result, "sha256": digest(result)}))
    dataset_path = path.with_suffix(".dataset.json")
    with dataset_path.open("x") as handle:
        handle.write(encode(dataset))
    return {
        k: result[k]
        for k in ("model_id", "method", "dataset_sha256", "counts", "excluded", "test", "limitations")
    }


def read_model(model_id):
    import json

    saved = json.loads(model_file(model_id).read_text())
    if (
        digest(saved["model"]) != saved["sha256"]
        or saved["model"].get("feature_version") != "chain-composition-v1"
    ):
        raise ValueError("Property model integrity or feature-version mismatch")
    return saved["model"]


def predict_batch(run, request):
    request = PredictionRequest.model_validate(request)
    public = public_run(run)
    if not public["candidates"]:
        raise ValueError("Active learning can only nominate independently evaluated, deliverable candidates")
    models = [(read_model(o.model_id), o.direction) for o in request.objectives]
    rows = []
    for candidate in public["candidates"]:
        binder = Binder.model_validate(candidate["binder"])
        estimates, objective_values = [], []
        for model, direction in models:
            domain = Domain.model_validate(model["domain"])
            if binder.product != domain.product or domain.antigen.model_dump() not in run["brief"]["states"]:
                raise ValueError("Prediction domain differs in antigen state or final product format")
            x = (features(binder) - np.array(model["mean"])) / np.array(model["scale"])
            mean = float(np.r_[1, x] @ np.array(model["weights"]))
            radius = model["interval_radius"]
            estimates.append(
                {
                    "model_id": model["model_id"],
                    "endpoint": domain.endpoint,
                    "unit": domain.unit,
                    "conditions": domain.conditions,
                    "status": "predicted",
                    "mean": mean,
                    "lower": mean - radius,
                    "upper": mean + radius,
                    "nominal_coverage": model["nominal_coverage"],
                }
            )
            objective_values.append(mean if direction == "max" else -mean)
        rows.append(
            {"candidate_id": candidate["id"], "predictions": estimates, "objectives": objective_values}
        )
    for row in rows:
        row["pareto"] = not any(
            all(a >= b for a, b in zip(other["objectives"], row["objectives"], strict=True))
            and any(a > b for a, b in zip(other["objectives"], row["objectives"], strict=True))
            for other in rows
        )
    # Portfolio already enforces distinct paired-sequence families. Alternate the Pareto front
    # with unexplored families for information gain; intervals alone are not efficacy scores.
    frontier = [r for r in rows if r["pareto"]]
    alternatives = [r for r in rows if not r["pareto"]]
    ordered = []
    while frontier or alternatives:
        if frontier:
            ordered.append({**frontier.pop(0), "role": "predicted_performance"})
        if alternatives:
            ordered.append({**alternatives.pop(0), "role": "family_exploration"})
    limit = min(request.count, run["brief"]["budget"]["experiment_slots"])
    for row in ordered:
        row.pop("objectives", None)
    return {
        "selected": ordered[:limit],
        "policy": "Independent gates, condition-matched multi-objective Pareto front and family exploration",
        "experimental_gain": None,
        "limitation": "Predictions prioritize measurements; they do not create experimental evidence",
    }
