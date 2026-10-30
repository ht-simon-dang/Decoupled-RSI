import copy

import pytest

from antibody_agent.learning import fit, predict_batch, read_model


def dataset(brief, binder):
    rows = []
    for i, aa in enumerate("ACDEFGHIKLMNPQRS"):
        b = binder.model_dump()
        b["chains"][0]["sequence"] = aa + b["chains"][0]["sequence"][1:]
        rows.append(
            {
                "id": "row" + str(i),
                "binder": b,
                "lineage": "lineage" + str(i),
                "split": "train" if i < 8 else "calibration" if i < 12 else "test",
                "status": "measured",
                "value": float(i),
                "source": "synthetic test",
                "measurement_id": "m" + str(i),
            }
        )
    return {
        "model_id": "test-v1",
        "domain": {
            "endpoint": "synthetic_test",
            "unit": "test",
            "conditions": {"assay": "synthetic"},
            "antigen": brief.states[0].model_dump(),
            "product": binder.product.model_dump(),
        },
        "rows": rows,
    }


def test_condition_matched_fit_and_gated_selection(brief, binder, completed):
    request = dataset(brief, binder)
    result = fit(request)
    assert result["counts"] == {"train": 8, "calibration": 4, "test": 4}
    model = read_model("test-v1")
    assert model["interval_radius"] >= 0
    prediction = predict_batch(completed, {"objectives": [{"model_id": "test-v1", "direction": "min"}]})
    assert len(prediction["selected"]) == 1
    assert prediction["selected"][0]["predictions"][0]["status"] == "predicted"
    with pytest.raises(ValueError, match="overwritten"):
        fit(request)
    completed["state"] = "blocked"
    with pytest.raises(ValueError, match="independently evaluated"):
        predict_batch(completed, {"objectives": [{"model_id": "test-v1", "direction": "min"}]})


def test_training_leakage_and_failures_do_not_create_labels(brief, binder):
    request = dataset(brief, binder)
    request["rows"][8]["lineage"] = request["rows"][0]["lineage"]
    with pytest.raises(ValueError, match="leakage"):
        fit(request)
    request = dataset(brief, binder)
    request["rows"][8]["status"] = "censored"
    with pytest.raises(ValueError, match="measured rows"):
        fit(request)


def test_test_labels_do_not_affect_fitted_weights(brief, binder):
    first = dataset(brief, binder)
    fit(first)
    second = copy.deepcopy(first)
    second["model_id"] = "independent-test-check"
    for row in second["rows"]:
        if row["split"] == "test":
            row["value"] += 10000.0
    fit(second)
    a, b = read_model(first["model_id"]), read_model(second["model_id"])
    assert a["weights"] == b["weights"]
    assert a["interval_radius"] == b["interval_radius"]
    assert a["test"]["mae"] != b["test"]["mae"]
