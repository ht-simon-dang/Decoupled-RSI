#!/usr/bin/env python3
"""Recount a curated PFM evaluation snapshot using only Python's standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


GATES = {
    "geometry", "identity", "novelty", "properties", "safety",
    "structure_score", "synthetic_accessibility",
}
HERE = Path(__file__).resolve().parent


def replay() -> dict:
    data_path = HERE / "recorded_evaluations.json"
    change_path = HERE / "candidate_change.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    change = json.loads(change_path.read_text(encoding="utf-8"))
    if data.get("example_kind") != "recorded_evaluation_replay":
        raise ValueError("Unrecognized example kind")
    if data.get("schema_version") != 1 or change.get("schema_version") != 1:
        raise ValueError("Unsupported example schema version")

    constants = change["recorded_constants"]
    records = {}
    for record in data["records"]:
        cycle = record["cycle"]
        if cycle in records:
            raise ValueError(f"Duplicate cycle: {cycle}")
        evaluation = record["evaluation"]
        if evaluation["status"] != "MEASURED":
            raise ValueError(f"{cycle}: the source evaluation was not measured")
        if evaluation["metric"] != "qualifying_scaffolds":
            raise ValueError(f"{cycle}: unexpected metric")
        if evaluation["evaluated_partition"] != "development":
            raise ValueError(f"{cycle}: unexpected evaluation partition")
        if record["weights_changed"] is not False or record["optimiser_steps"] != 0:
            raise ValueError(f"{cycle}: recorded frozen-weight conditions differ")
        if record["checkpoint_sha256"] != constants["checkpoint_sha256"]:
            raise ValueError(f"{cycle}: checkpoint identity differs")
        if evaluation["evaluator_hash"] != constants["evaluator_hash"]:
            raise ValueError(f"{cycle}: evaluator identity differs")
        if record["qualification_hash"] != constants["qualification_hash"]:
            raise ValueError(f"{cycle}: qualification identity differs")

        sampling = record["sampling"]
        rows = record["rows"]
        if len(rows) != sampling["n_submitted"] or len(rows) != constants["submission_budget"]:
            raise ValueError(f"{cycle}: evaluation rows do not match the submission budget")
        if sampling["n_generated_ceiling"] != constants["draw_ceiling"]:
            raise ValueError(f"{cycle}: recorded draw ceiling differs")

        scaffolds = set()
        for row in rows:
            gates = row["gates"]
            if set(gates) != GATES or any(type(value) is not bool for value in gates.values()):
                raise ValueError(f"{cycle}: missing, extra or non-boolean gate decisions")
            scaffold = row["scaffold_sha256"]
            if not isinstance(scaffold, str) or re.fullmatch(r"[0-9a-f]{64}", scaffold) is None:
                raise ValueError(f"{cycle}: invalid scaffold digest")
            if all(gates.values()):
                scaffolds.add(scaffold)

        count = len(scaffolds)
        if count != evaluation["value"]:
            raise ValueError(f"{cycle}: recounted {count}, recorded {evaluation['value']}")
        records[cycle] = {
            "cycle": cycle,
            "decoder": record["decoder"],
            "seed": record["seed"],
            "n_submitted": len(rows),
            "n_generated": sampling["n_generated"],
            "n_distinct": sampling["n_distinct"],
            "n_cheap_gate_survivors": sampling["n_survivors"],
            "qualifying_scaffolds": count,
        }

    pairs = []
    used_cycles = set()
    for pair in change["paired_cycles"]:
        stock = records[pair["stock"]]
        shaped = records[pair["shaped"]]
        if stock["decoder"] != "stock" or shaped["decoder"] != "shaped":
            raise ValueError("A pair must compare stock and shaped decoders")
        if stock["seed"] != shaped["seed"] or stock["seed"] != pair["seed"]:
            raise ValueError("A pair must use the same recorded seed")
        if used_cycles.intersection({pair["stock"], pair["shaped"]}):
            raise ValueError("A cycle appears in more than one pair")
        used_cycles.update({pair["stock"], pair["shaped"]})
        pairs.append({
            "seed": pair["seed"],
            "stock": stock,
            "shaped": shaped,
            "difference": shaped["qualifying_scaffolds"] - stock["qualifying_scaffolds"],
        })
    if used_cycles != set(records):
        raise ValueError("Some recorded cycles are not paired")

    return {
        "example_kind": data["example_kind"],
        "classification": data["classification"],
        "recorded_evaluations_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "candidate_change_sha256": hashlib.sha256(change_path.read_bytes()).hexdigest(),
        "evaluation_rows": sum(record["n_submitted"] for record in records.values()),
        "pairs": pairs,
        "scope": "Recounts stored gate decisions; no PFM generation or scientific scoring is rerun.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the replay summary as JSON")
    args = parser.parse_args()
    try:
        result = replay()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print("PFM recorded-evaluation replay (development partition; exploratory)")
    print(f"Recounted {result['evaluation_rows']} stored evaluation rows.")
    for pair in result["pairs"]:
        stock, shaped = pair["stock"], pair["shaped"]
        print(
            f"seed {pair['seed']}: {stock['cycle']} stock {stock['qualifying_scaffolds']}"
            f" -> {shaped['cycle']} shaped {shaped['qualifying_scaffolds']}"
            f" qualifying scaffolds (difference {pair['difference']:+d};"
            f" {stock['n_submitted']} submitted in each arm)"
        )
    print(result["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
