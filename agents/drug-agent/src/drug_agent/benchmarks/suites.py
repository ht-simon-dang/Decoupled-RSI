"""Scientific sanity benchmarks (demonstration-only, offline).

These are labelled demonstration-only: they exercise the baseline model on the
synthetic fixture and prove the scientific *guards* work (a model must beat
trivial baselines; label-scrambling must destroy signal; duplicate leakage must
be detectable). They are NOT validated benchmark claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.connectors import ChemblLikeConnector
from ..tools.rdkit_tool import Cheminformatics
from ..tools.sklearn_tool import BaselineActivityModel
from ..util import stable_hash_int


def _load_pairs() -> list[tuple[str, float]]:
    snap = ChemblLikeConnector().load()
    out = []
    for r in snap.records:
        smi = r.get("smiles")
        pv = r.get("pchembl_value")
        if smi and pv:
            out.append((smi, float(pv)))
    return out


def _scaffold_split(pairs: list[tuple[str, float]]):
    chem = Cheminformatics()
    scaf = [chem.scaffold(s) or s for s, _ in pairs]
    test_sc = {sc for sc in set(scaf) if stable_hash_int(sc, modulo=3) == 0}
    tr = [(s, v) for (s, v), sc in zip(pairs, scaf, strict=True) if sc not in test_sc]
    te = [(s, v) for (s, v), sc in zip(pairs, scaf, strict=True) if sc in test_sc]
    return (tr, te) if tr and te else (pairs[: len(pairs) * 2 // 3], pairs[len(pairs) * 2 // 3 :])


def scientific_suite() -> dict[str, Any]:
    pairs = _load_pairs()
    results: dict[str, Any] = {"note": "demonstration-only; synthetic labels", "checks": {}}
    if len(pairs) < 8:
        results["checks"]["data"] = {"ok": False, "reason": "insufficient data"}
        results["summary"] = "insufficient data"
        return results

    train, test = _scaffold_split(pairs)
    model = BaselineActivityModel(seed=1729).fit([s for s, _ in train], [v for _, v in train])
    y = np.array([v for _, v in test])
    yhat = np.array([model.predict_one(s).value for s, _ in test])
    ymean = float(np.mean([v for _, v in train]))
    mae = float(np.mean(np.abs(y - yhat)))
    mae_const = float(np.mean(np.abs(y - ymean)))

    # Y-scramble negative control: shuffle labels -> signal should collapse.
    rng = np.random.default_rng(0)
    yperm = np.array([v for _, v in train])
    yperm = rng.permutation(yperm)
    m2 = BaselineActivityModel(seed=1729).fit([s for s, _ in train], list(yperm))
    yhat2 = np.array([m2.predict_one(s).value for s, _ in test])
    mae_scramble = float(np.mean(np.abs(y - yhat2)))

    # Duplicate-leakage detector: identical structures across splits.
    train_ik = {Cheminformatics().standardize(s).inchikey for s, _ in train}
    test_ik = {Cheminformatics().standardize(s).inchikey for s, _ in test}
    leak = len(train_ik & test_ik - {None})

    results["checks"] = {
        "beats_constant_baseline": {
            "ok": mae < mae_const,
            "mae": round(mae, 4),
            "mae_constant": round(mae_const, 4),
        },
        "y_scramble_destroys_signal": {
            "ok": mae_scramble >= mae,
            "mae_scrambled": round(mae_scramble, 4),
            "mae_true": round(mae, 4),
        },
        "no_duplicate_leakage": {"ok": leak == 0, "n_leaked": leak},
    }
    all_ok = all(c["ok"] for c in results["checks"].values())
    results["all_passed"] = all_ok
    results["summary"] = (
        "all scientific-sanity checks passed" if all_ok else "one or more sanity checks failed"
    )
    return results


def run_suite(name: str) -> dict[str, Any]:
    if name in {"scientific", "sanity"}:
        return scientific_suite()
    return {"error": f"unknown suite {name}", "available": ["scientific"]}
