"""Established sequence descriptors and transparent structural baselines."""
from __future__ import annotations

import io
import math
from collections import Counter
from itertools import combinations

import Bio
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial import cKDTree

from .contracts import Analyze, Structure, ToolResult


def descriptors(sequence: str, ph: float = 7.4) -> dict:
    p = ProteinAnalysis(sequence)
    counts = Counter(sequence)
    return {
        "sequence": sequence,
        "length": len(sequence),
        "molecular_weight_da": p.molecular_weight(),
        "isoelectric_point": p.isoelectric_point(),
        "charge_at_ph": p.charge_at_pH(ph),
        "ph": ph,
        "gravy": p.gravy(),
        "aromaticity": p.aromaticity(),
        "instability_index": p.instability_index(),
        "cysteines": counts["C"],
        "entropy_bits": -sum((c / len(sequence)) * math.log2(c / len(sequence)) for c in counts.values()),
        "trypsin_sites": [i + 1 for i, aa in enumerate(sequence[:-1]) if aa in "KR" and sequence[i + 1] != "P"],
    }


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def diversity(sequences: list[str]) -> dict:
    unique = sorted(set(sequences))
    distances = [edit_distance(a, b) / max(len(a), len(b)) for a, b in combinations(unique, 2)]
    return {"count": len(sequences), "unique_count": len(unique),
            "unique_fraction": len(unique) / len(sequences),
            "mean_normalized_edit_distance": float(np.mean(distances)) if distances else None}


def analyze(request: Analyze) -> ToolResult:
    from .chemistry import properties
    rows = [{**descriptors(s, request.ph), **properties(s, request.topology, request.ph)} for s in request.sequences]
    return ToolResult(tool="sequence.analyze", status="ok", trust="CALCULATED",
                      data={"candidates": rows,
                            "diversity": diversity(request.sequences)},
                      provenance={"algorithm": "Biopython ProtParam; RDKit peptide graph; side-chain pKa cyclic charge; Levenshtein", "biopython": Bio.__version__, "rdkit": __import__("rdkit").__version__},
                      warnings=["Canonical L-residues, reduced cysteines; topology is explicit. Cyclic pI may be undefined.",
                                "Instability index is protein-derived, not validated peptide serum half-life.",
                                "Trypsin sites are a sequence rule, not a proteolysis rate or toxicity prediction."])


def structure(request: Structure) -> ToolResult:
    if request.receptor_chain == request.peptide_chain:
        raise ValueError("Receptor and peptide chains must differ")
    model = PDBParser(QUIET=True).get_structure("complex", io.StringIO(request.pdb))[0]
    def coords(chain):
        atoms = [a.coord for r in model[chain] if r.id[0] == " " for a in r
                 if a.element not in {"H", "D"}]
        if not atoms:
            raise ValueError("Chain has no protein heavy atoms")
        points = np.asarray(atoms)
        if not np.isfinite(points).all():
            raise ValueError("Non-finite PDB coordinates")
        return points
    receptor, peptide = coords(request.receptor_chain), coords(request.peptide_chain)
    tree = cKDTree(receptor)
    contacts = tree.query_ball_point(peptide, r=5.0, return_length=True)
    clashes = tree.query_ball_point(peptide, r=2.0, return_length=True)
    return ToolResult(tool="structure.contacts", status="ok", trust="CALCULATED",
                      data={"heavy_atom_contacts_5A": int(sum(contacts)),
                            "heavy_atom_clashes_2A": int(sum(clashes)),
                            "min_interchain_distance_A": float(tree.query(peptide)[0].min())},
                      provenance={"algorithm": "scipy cKDTree; first PDB model; protein heavy atoms"},
                      warnings=["Distance cutoffs are geometric heuristics, not binding free energy or DockQ."])


def rank(candidates: list[dict], constraints, limit: int) -> dict:
    accepted, rejected = [], []
    for row in {c["sequence"]: c for c in candidates}.values():
        reasons = []
        if abs(row["charge_at_ph"]) > constraints.max_abs_charge:
            reasons.append("absolute charge exceeds brief")
        if row["gravy"] > constraints.max_gravy:
            reasons.append("GRAVY exceeds brief")
        if row["cysteines"] > constraints.max_cysteines:
            reasons.append("cysteine count exceeds brief")
        (rejected if reasons else accepted).append(dict(row, reasons=reasons))
    # Pareto layers on transparent developability heuristics; no affinity score.
    def objective(c):
        return abs(c["charge_at_ph"]), c["gravy"], -c["entropy_bits"]
    def dominates(a, b):
        return all(x <= y for x, y in zip(objective(a), objective(b))) and any(
            x < y for x, y in zip(objective(a), objective(b)))
    pool = sorted(accepted, key=lambda c: c["sequence"])
    ordered, layer = [], 0
    while pool:
        front = [c for c in pool if not any(dominates(o, c) for o in pool)]
        while front:
            pick = max(front, key=lambda c: min(
                edit_distance(c["sequence"], o["sequence"]) / max(c["length"], o["length"])
                for o in ordered) if ordered else c["entropy_bits"])
            pick["pareto_layer"] = layer
            ordered.append(pick)
            front.remove(pick)
            pool.remove(pick)
        layer += 1
    return {"shortlist": ordered[:limit], "accepted": ordered, "rejected": rejected,
            "ranking_trust": "INFERRED", "objectives": ["min abs(charge)", "min GRAVY", "max entropy"],
            "limitation": "Developability heuristic prioritization only; affinity, toxicity and efficacy unknown."}
