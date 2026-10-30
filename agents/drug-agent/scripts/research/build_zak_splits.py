"""Freeze the ZAK family-level partitions and the protected exclusion registry.

Offline. It reads the standardized rows Step 5 wrote and produces four things:

``split_manifest.json``
    The frozen, hashable description of the split — counts, family edge rules,
    the cutoff actually used, the feasibility verdict and the Track A primary
    mode. No measured label, no timestamp, so two builds of one split are one
    artifact.
``protected_exclusion_registry.json``
    Identifiers only: the held-out compounds, every source alias they were
    published under, their connectivity keys and their family ids. Written with
    owner-only permissions under the *gitignored* protected directory, never
    into source control.
``split_membership.json``
    The full per-compound membership, beside the registry for the same reason.
``mode_selection.json``
    Why regression rather than ranking, decided on **development** counts.

Three things it deliberately does not do
----------------------------------------
It does not reduce the similarity cutoff, shrink the test share, or merge strata
when the preflight guard is tight. If the guard fails the manifest says
``INSUFFICIENT_TEST_DATA`` and the programme takes the grouped-cross-validation
path — see Appendix A §3.2.

It does not choose the weighting after looking at a result. The allocation
balances on the count of *exactly measured* primary-stratum molecules first and
the total compound count second, because that is the quantity the preflight
guard is defined on, and the weighting is fixed here, before any candidate
search, with both alternatives reported side by side in ``weighting_evidence``.

It does not decide the split on a per-stratum basis. Families are computed over
**every** compound in the dataset and allocated once, so a compound's partition
applies to every endpoint it was ever measured in — Appendix A §3.2's "related
measurements in other endpoints also follow the compound's partition".

Usage::

    python scripts/research/build_zak_splits.py
    python scripts/research/build_zak_splits.py --seed 41
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.selfevolve.target.splits import (
    EXCLUSION_BOUNDARY_NOTE,
    PREFLIGHT_GUARD,
    SIMILARITY_THRESHOLD,
    build_exclusion_registry,
    build_split_manifest,
    cross_partition_max_similarity,
    label_class,
    partition_families,
    preflight_feasibility,
    probe_protected_access,
    write_protected_registry,
)
from drug_agent.storage.artifacts import LocalArtifactStore
from drug_agent.util import utcnow_iso

DEFAULT_DERIVED = Path(".external/research/zak-c9r/derived")
DEFAULT_PROTECTED = Path(".external/research/zak-c9r/protected")
DEFAULT_STORE = Path(".drugagent/research/artifacts")
PRODUCED_BY = "scripts/research/build_zak_splits.py"

#: Appendix A §3.2's rule, applied to the measured strata table: the largest
#: scientifically compatible quantitative stratum by independent-molecule count,
#: tie-broken by stratum id. Recorded here so the choice is auditable rather than
#: a constant someone typed.
PRIMARY_STRATUM_RULE = "largest independent-molecule count, tie-broken by stratum id"


def _select_primary_stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, set[str]] = {}
    exact: dict[str, set[str]] = {}
    for row in rows:
        per.setdefault(row["stratum_id"], set()).add(row["inchikey"])
        if row["exact"]:
            exact.setdefault(row["stratum_id"], set()).add(row["inchikey"])
    ranked = sorted(per, key=lambda s: (-len(per[s]), s))
    chosen = ranked[0]
    return {
        "stratum_id": chosen,
        "rule": PRIMARY_STRATUM_RULE,
        "n_molecules": len(per[chosen]),
        "n_exact_molecules": len(exact.get(chosen, ())),
        "ranking": [
            {"stratum_id": s, "n_molecules": len(per[s]), "n_exact_molecules": len(exact.get(s, ()))}
            for s in ranked
        ],
    }


def _compound_view(rows: list[dict[str, Any]], primary: str) -> list[dict[str, Any]]:
    """One partitionable record per independent compound, with its aliases."""
    view: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = view.setdefault(
            row["inchikey"],
            {
                "molecule_id": row["inchikey"],
                "connectivity_key": row["connectivity_key"],
                "smiles_variants": set(),
                "aliases": set(),
                "n_exact_primary": 0,
                "n_primary": 0,
                "_primary_classes": set(),
                "_strata": set(),
                "_documents": set(),
            },
        )
        entry["smiles_variants"].add(row["molecule"]["standard_smiles"])
        entry["aliases"].add(row["molecule_id"])
        for source in row.get("sources", ()) or ():
            if source.get("molecule_id"):
                entry["aliases"].add(source["molecule_id"])
        entry["_strata"].add(row["stratum_id"])
        entry["_documents"].add(row["document_key"])
        if row["stratum_id"] == primary:
            entry["n_primary"] = 1
            if row["exact"]:
                entry["n_exact_primary"] = 1
            entry["_primary_classes"].add(label_class(row["value_M"], row["relation"]))

    out: list[dict[str, Any]] = []
    for entry in view.values():
        classes = entry.pop("_primary_classes")
        # A compound is a class only when every conclusive measurement agrees.
        # Disagreement is `inconclusive`, never a majority vote, and a compound
        # with no primary-stratum measurement has no class at all.
        conclusive = sorted(c for c in classes if c in ("positive", "negative"))
        if len(conclusive) == 1:
            entry["label_class"] = conclusive[0]
        elif conclusive:
            entry["label_class"] = "inconclusive"
        elif "intermediate" in classes:
            entry["label_class"] = "intermediate"
        elif classes:
            entry["label_class"] = "inconclusive"
        else:
            entry["label_class"] = None
        entry["n_strata"] = len(entry.pop("_strata"))
        entry["n_documents"] = len(entry.pop("_documents"))
        entry["smiles_variants"] = sorted(entry["smiles_variants"])
        entry["aliases"] = sorted(entry["aliases"])
        out.append(entry)
    return sorted(out, key=lambda e: e["molecule_id"])


def _composition(rows: list[dict[str, Any]], membership: dict[str, str]) -> dict[str, Any]:
    """Per-study and per-stratum composition, as counts only."""
    by_partition: dict[str, dict[str, Counter]] = {}
    for row in rows:
        side = membership.get(row["inchikey"])
        if side is None:
            continue
        bucket = by_partition.setdefault(side, {"strata": Counter(), "documents": Counter()})
        bucket["strata"][row["stratum_id"]] += 1
        bucket["documents"][row["document_key"]] += 1
    return {
        side: {
            "n_rows": sum(bucket["strata"].values()),
            "strata": dict(sorted(bucket["strata"].items())),
            "n_documents": len(bucket["documents"]),
            "documents": dict(sorted(bucket["documents"].items())),
        }
        for side, bucket in sorted(by_partition.items())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--derived", default=str(DEFAULT_DERIVED))
    parser.add_argument("--protected", default=str(DEFAULT_PROTECTED))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)

    derived = Path(args.derived)
    measurements_path = derived / "measurements.json"
    if not measurements_path.is_file():
        print(
            f"no standardized rows at {measurements_path}; run "
            "scripts/research/build_zak_target_data.py first",
            file=sys.stderr,
        )
        return 2
    rows = json.loads(measurements_path.read_text())
    target_manifest = json.loads((derived / "target_manifest.json").read_text())
    index = json.loads((derived / "artifact_index.json").read_text())

    primary = _select_primary_stratum(rows)
    compounds = _compound_view(rows, primary["stratum_id"])

    # The declared weighting, and the alternative, computed together so the
    # choice is evidence rather than a claim. Both are produced before any
    # candidate search and neither is re-run afterwards.
    weighted = partition_families(compounds, args.seed, weight_keys=("n_exact_primary", "n_primary"))
    unweighted = partition_families(compounds, args.seed)

    exact_rows = [c for c in compounds if c["n_exact_primary"]]
    primary_rows = [c for c in compounds if c["n_primary"]]

    weighting_evidence = {
        "declared": "n_exact_primary, then n_primary, then n_molecules",
        "rationale": (
            "the preflight guard is defined on unique compounds in the held-out partition, and "
            "continuous regression can only use exactly measured ones, so that is the quantity "
            "the allocation balances first. Fixed here, before any candidate search."
        ),
        "weighted": {
            "test_exact_compounds": preflight_feasibility(exact_rows, weighted, mode="regression")[
                "n_compounds"
            ],
            "test_exact_families": preflight_feasibility(exact_rows, weighted, mode="regression")[
                "n_families"
            ],
            "achieved_fractions": weighted["achieved_fractions"],
        },
        "compound_count_only": {
            "test_exact_compounds": preflight_feasibility(exact_rows, unweighted, mode="regression")[
                "n_compounds"
            ],
            "test_exact_families": preflight_feasibility(exact_rows, unweighted, mode="regression")[
                "n_families"
            ],
            "achieved_fractions": unweighted["achieved_fractions"],
        },
    }

    partition = weighted
    membership = partition["membership"]

    regression = preflight_feasibility(exact_rows, partition, mode="regression")
    ranking = preflight_feasibility(primary_rows, partition, mode="ranking")

    # Mode selection on DEVELOPMENT counts only. The test-partition guard above
    # is Appendix A's feasibility check, not the evidence for this decision.
    development_regression = preflight_feasibility(
        exact_rows, partition, mode="regression", partition_name="development"
    )
    development_ranking = preflight_feasibility(
        primary_rows, partition, mode="ranking", partition_name="development"
    )
    primary_mode = "regression" if development_regression["status"] == "FEASIBLE" else None
    if primary_mode is None and development_ranking["status"] == "FEASIBLE":
        primary_mode = "ranking"
    mode_selection = {
        "decided_on": "development partition counts only",
        "primary_mode": primary_mode,
        "regression_development": development_regression,
        "ranking_development": development_ranking,
        "rule": (
            "Appendix A §3.2: default to continuous prediction when enough exact measurements "
            "exist; choose activity ranking only when meaningful positive/negative labels exist"
        ),
        "note": (
            "Ranking is rejected on measured development counts, not on preference: the primary "
            "stratum is almost entirely inactive under the provisional <=100 nM / >=1000 nM "
            "policy. Untested molecules were never converted into negatives."
        ),
    }

    similarity = cross_partition_max_similarity(
        [{"compound_key": c["molecule_id"], "smiles_variants": c["smiles_variants"]} for c in compounds],
        membership,
    )
    registry = build_exclusion_registry(compounds, partition, measurement_rows=rows)
    composition = _composition(rows, membership)
    composition["label_classes"] = {
        side: dict(
            sorted(
                Counter(
                    c["label_class"] or "unclassified"
                    for c in primary_rows
                    if membership[c["molecule_id"]] == side
                ).items()
            )
        )
        for side in ("train", "development", "test")
    }

    manifest = build_split_manifest(
        partition=partition,
        feasibility=regression if primary_mode == "regression" else ranking,
        target_manifest_hash=target_manifest["manifest_hash"],
        data_hash=index["measurement_rows"]["content_hash"],
        stratum_id=primary["stratum_id"],
        primary_mode=primary_mode,
        similarity_report=similarity,
        exclusion_registry=registry,
        composition=composition,
        notes=[
            f"primary stratum selected by: {primary['rule']}",
            (
                "the primary stratum's assay_type is F and its bao_format is BAO_0000019 — a "
                "FUNCTIONAL assay format carrying a Ki endpoint. It is a legitimate stratum and "
                "it is NOT a biochemical Ki; nothing merges it with the B-typed strata."
            ),
            f"regression feasibility (exact measurements only): {regression['status']}",
            f"ranking feasibility (all primary-stratum molecules): {ranking['status']}",
            EXCLUSION_BOUNDARY_NOTE,
        ],
    )

    protected_root = Path(args.protected)
    written = write_protected_registry(protected_root, registry)
    probe = probe_protected_access(written["path"])
    membership_path = protected_root / "split_membership.json"
    membership_path.write_text(
        json.dumps(
            {
                "split_hash": partition["split_hash"],
                "seed": partition["seed"],
                "membership": membership,
                "family_of": partition["family_of"],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    membership_path.chmod(0o600)
    (protected_root / "mode_selection.json").write_text(
        json.dumps(mode_selection, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    store = LocalArtifactStore(args.store)
    manifest_ref = store.put_json(manifest, kind="split_manifest", produced_by=PRODUCED_BY)
    registry_ref = store.put_json(registry, kind="exclusion_registry", produced_by=PRODUCED_BY)

    derived.mkdir(parents=True, exist_ok=True)
    (derived / "split_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    summary = {
        "built_at": utcnow_iso(),
        "seed": args.seed,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "preflight_guard": dict(PREFLIGHT_GUARD),
        "primary_stratum": primary,
        "primary_mode": primary_mode,
        "weighting_evidence": weighting_evidence,
        "split_hash": partition["split_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "split_manifest": manifest_ref.model_dump(mode="json"),
        "exclusion_registry": registry_ref.model_dump(mode="json"),
        "counts": partition["counts"],
        "n_families": partition["n_families"],
        "largest_family": partition["largest_family"],
        "infeasible_reasons": partition["infeasible_reasons"],
        "regression_feasibility": regression,
        "ranking_feasibility": ranking,
        "mode_selection": mode_selection,
        "cross_partition_similarity": similarity,
        "protected": {**written, "membership": str(membership_path)},
        "access_probe": probe,
    }
    (derived / "split_index.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
