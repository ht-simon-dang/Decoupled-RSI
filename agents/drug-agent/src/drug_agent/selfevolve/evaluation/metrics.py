"""The kernel's own measurements of a finished run.

Deliberately **not** a call into :mod:`drug_agent.agents.llm.evaluation`. That
module is part of the system under test: a candidate is allowed to change it,
and a judge a candidate can rewrite is not a judge. Everything here is computed
from the raw run — scorecards, portfolio, candidate SMILES — by code inside the
integrity manifest.

Two tiers, and the distinction is reported on every record rather than blurred:

``independent``
    Recomputed here from the shortlist SMILES with RDKit, using RDKit's own
    PAINS catalogue rather than the platform's alert list. QED, MW, cLogP, TPSA,
    alert-free fraction, pairwise Tanimoto distance, and hard-constraint pass
    rate against the brief's own constraints. If the platform's scorecards were
    wrong or tampered with, these would disagree with them — and the kernel
    records both, so the disagreement is visible.

``platform``
    Values only the platform can produce: a RandomForest potency prediction, its
    applicability domain, the perturbation restoration score, the heuristic SA
    score. The kernel cannot recompute an ML prediction without becoming a
    second copy of the model, so it reads these and says that it did.

Gold criteria are scored over both tiers, but the *guard* against a tampered
arm is that the independent tier exists and is cross-checked.
"""

from __future__ import annotations

import math
import statistics
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _rdkit() -> Any:
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import (  # type: ignore
        QED,
        Crippen,
        Descriptors,
        rdFingerprintGenerator,  # type: ignore
        rdMolDescriptors,
    )
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return {
        "Chem": Chem,
        "QED": QED,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "rdMolDescriptors": rdMolDescriptors,
        "catalog": catalog,
        "fpgen": gen,
    }


def _endpoint(scorecard: dict[str, Any], name: str) -> float | None:
    for e in scorecard.get("endpoints", []):
        if e.get("endpoint") == name and not e.get("missing") and e.get("raw_value") is not None:
            return float(e["raw_value"])
    return None


def _applicability(scorecard: dict[str, Any], name: str = "pIC50") -> str | None:
    for e in scorecard.get("endpoints", []):
        if e.get("endpoint") == name:
            return e.get("applicability")
    return None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None


def independent_chemistry(smiles: list[str], constraints: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute, from SMILES alone, everything RDKit can tell us.

    ``constraints`` are the brief's own hard constraints, re-evaluated here so
    the pass rate does not depend on the platform having filtered correctly.
    """
    rd = _rdkit()
    Chem = rd["Chem"]
    mols = []
    unparsed: list[str] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            unparsed.append(smi)
        else:
            mols.append((smi, mol))
    if not mols:
        return {
            "n": 0,
            "n_unparsed": len(unparsed),
            "note": "no parseable shortlist SMILES",
        }

    qed, mw, logp, tpsa, hbd, hba, alerts = [], [], [], [], [], [], []
    for _smi, mol in mols:
        qed.append(float(rd["QED"].qed(mol)))
        mw.append(float(rd["Descriptors"].MolWt(mol)))
        logp.append(float(rd["Crippen"].MolLogP(mol)))
        tpsa.append(float(rd["rdMolDescriptors"].CalcTPSA(mol)))
        hbd.append(float(rd["rdMolDescriptors"].CalcNumHBD(mol)))
        hba.append(float(rd["rdMolDescriptors"].CalcNumHBA(mol)))
        alerts.append(1.0 if rd["catalog"].HasMatch(mol) else 0.0)

    fps = [rd["fpgen"].GetFingerprint(m) for _s, m in mols]
    distances: list[float] = []
    from rdkit import DataStructs  # type: ignore

    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            distances.append(1.0 - float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

    props_by_index = [
        {
            "descriptors": {
                "mw": mw[i],
                "logp": logp[i],
                "tpsa": tpsa[i],
                "hbd": hbd[i],
                "hba": hba[i],
                "qed": qed[i],
            },
            "n_structural_alerts": alerts[i],
        }
        for i in range(len(mols))
    ]
    n_pass = sum(1 for props in props_by_index if _passes(props, constraints))

    return {
        "n": len(mols),
        "n_unparsed": len(unparsed),
        "qed_median": _median(qed),
        "mw_median": _median(mw),
        "logp_median": _median(logp),
        "tpsa_median": _median(tpsa),
        "pains_free_fraction": round(1.0 - (sum(alerts) / len(alerts)), 4),
        "mean_pairwise_distance": round(statistics.mean(distances), 4) if distances else None,
        "min_pairwise_distance": round(min(distances), 4) if distances else None,
        "constraint_pass_rate": round(n_pass / len(mols), 4),
        "n_constraints": len(constraints),
    }


_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _passes(props: dict[str, Any], constraints: list[dict[str, Any]]) -> bool:
    for con in constraints:
        key = str(con.get("property") or "")
        cur: Any = props
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None:
            return False
        op = _OPS.get(str(con.get("op") or "<="))
        if op is None:
            return False
        try:
            if not op(cur, con.get("value")):
                return False
        except TypeError:
            return False
    return True


def kernel_metrics(run: Any, requested_shortlist: int, brief: dict[str, Any]) -> dict[str, Any]:
    """Every number the gold rubric is scored against, computed here.

    ``run`` is a :class:`drug_agent.llm.bridge.BackendRun`.
    """
    portfolio = run.portfolio or {}
    scorecards = list(run.scorecards)
    by_id = {sc.get("candidate_id"): sc for sc in scorecards}
    members = portfolio.get("shortlist", []) or []
    shortlist_ids = [m.get("candidate_id") for m in members]
    short = [by_id[i] for i in shortlist_ids if i in by_id]
    smiles = [str(m.get("canonical_smiles") or "") for m in members]

    def short_vals(name: str) -> list[float]:
        return [v for sc in short if (v := _endpoint(sc, name)) is not None]

    n_all = len(scorecards)
    n_pass = sum(1 for sc in scorecards if sc.get("hard_constraints_passed"))
    domains = [d for sc in short if (d := _applicability(sc)) is not None]

    independent = independent_chemistry(smiles, brief.get("hard_constraints") or [])

    platform = {
        "n_candidates": n_all,
        "n_passed_filters": n_pass,
        "hard_constraint_pass_rate": round(n_pass / n_all, 4) if n_all else 0.0,
        "shortlist_size": len(shortlist_ids),
        "requested_shortlist": requested_shortlist,
        "shortlist_filled": round(len(shortlist_ids) / requested_shortlist, 4)
        if requested_shortlist
        else 0.0,
        "max_pIC50": round(max(short_vals("pIC50")), 4) if short_vals("pIC50") else None,
        "median_pIC50": _median(short_vals("pIC50")),
        "qed_median": _median(short_vals("qed")),
        "sa_median": _median(short_vals("sa_score")),
        "herg_median": _median(short_vals("hERG")),
        "restoration_median": _median(short_vals("restoration_score")),
        "in_domain_fraction": round(sum(1 for d in domains if d == "IN_DOMAIN") / len(domains), 4)
        if domains
        else 0.0,
        "mean_pairwise_distance": portfolio.get("diversity"),
        "n_scaffold_clusters": float(portfolio.get("n_clusters", 0) or 0),
        "n_pareto": float(portfolio.get("n_pareto", 0) or 0),
        "state": run.state,
    }

    merged: dict[str, Any] = dict(platform)
    # Independent recomputations win under their own names, so a gold criterion
    # can be pointed at the tier it should be scored on.
    merged.update(
        {
            "independent_qed_median": independent.get("qed_median"),
            "independent_pains_free_fraction": independent.get("pains_free_fraction"),
            "independent_mean_pairwise_distance": independent.get("mean_pairwise_distance"),
            "independent_constraint_pass_rate": independent.get("constraint_pass_rate"),
            "independent_n_shortlist": float(independent.get("n", 0)),
            "independent_n_unparsed": float(independent.get("n_unparsed", 0)),
        }
    )
    return {"metrics": merged, "independent": independent, "platform": platform}


def cross_check(metrics: dict[str, Any], *, tolerance: float = 0.15) -> list[str]:
    """Where the platform and the kernel disagree about the same quantity.

    A non-empty list is not automatically foul play — QED from a different RDKit
    version drifts, and PAINS is not the platform's alert catalogue — but it is
    always worth putting on the record, and a large disagreement on a quantity
    that should be identical is exactly the signature of a tampered arm.
    """
    notes: list[str] = []
    pairs = [
        ("qed_median", "independent_qed_median"),
        ("mean_pairwise_distance", "independent_mean_pairwise_distance"),
    ]
    for platform_key, own_key in pairs:
        a, b = metrics.get(platform_key), metrics.get(own_key)
        if a is None or b is None:
            continue
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            continue
        denom = max(abs(float(a)), 1e-9)
        if abs(float(a) - float(b)) / denom > tolerance:
            notes.append(
                f"{platform_key}={a} but the kernel independently computed {b} (>{tolerance:.0%} apart)"
            )
    return notes


__all__ = ["cross_check", "independent_chemistry", "kernel_metrics"]
