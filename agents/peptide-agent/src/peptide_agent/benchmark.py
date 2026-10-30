"""User-labelled, held-out binary endpoints; never fit on generated pseudo-labels."""
import csv

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import descriptors
from .contracts import AA, Analyze


def features(sequence):
    d = descriptors(sequence)
    return [sequence.count(a) / len(sequence) for a in AA] + [d[k] for k in
        ("length", "molecular_weight_da", "charge_at_ph", "gravy", "aromaticity", "entropy_bits")]


def benchmark(path, seed=1729):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 8 <= len(rows) <= 10000:
        raise ValueError("Provide 8..10000 labelled rows")
    if not {"sequence", "label", "split", "group"}.issubset(rows[0]):
        raise ValueError("CSV requires sequence,label,split,group")
    seen, groups = {}, {}
    for row in rows:
        Analyze(sequences=[row["sequence"]])
        if row["label"] not in {"0", "1"} or row["split"] not in {"train", "test"} or not row["group"]:
            raise ValueError("Require binary label, train/test split and nonempty homology/target group")
        if row["sequence"] in seen:
            raise ValueError("Duplicate sequence; deduplicate before benchmarking")
        seen[row["sequence"]] = row["split"]
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("Group leakage across train/test")
        groups[row["group"]] = row["split"]
    train, test = ([r for r in rows if r["split"] == split] for split in ("train", "test"))
    if any({r["label"] for r in part} != {"0", "1"} for part in (train, test)):
        raise ValueError("Both train and test need both classes")
    x, xt = (np.array([features(r["sequence"]) for r in part]) for part in (train, test))
    y, yt = (np.array([int(r["label"]) for r in part]) for part in (train, test))
    models = {"dummy_prior": DummyClassifier(strategy="prior"),
              "aac_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
              "aac_random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                           min_samples_leaf=2, n_jobs=1, random_state=seed)}
    results = {}
    for name, model in models.items():
        model.fit(x, y)
        p = model.predict_proba(xt)[:, 1]
        pred = (p >= 0.5).astype(int)
        results[name] = {"roc_auc": roc_auc_score(yt, p), "average_precision": average_precision_score(yt, p),
                         "mcc": matthews_corrcoef(yt, pred), "accuracy": accuracy_score(yt, pred)}
    return {"trust": "CALCULATED", "train_count": len(train), "test_count": len(test), "seed": seed,
            "features": "20 amino-acid frequencies + 6 sequence descriptors", "metrics": results,
            "limitations": ["Groups must be supplied from target/homology clustering; group semantics are not inferred.",
                            "Single held-out split; no hyperparameter tuning or claim of benchmark leadership."]}
