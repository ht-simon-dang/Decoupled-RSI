"""Coordinate checks on actual predicted structures, not generated confidence."""

from __future__ import annotations

import math
from io import StringIO

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def coordinates(pdb):
    structure = PDBParser(QUIET=True).get_structure("prediction", StringIO(pdb))
    model = next(structure.get_models())
    result = {}
    for chain in model:
        rows = []
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue
            xyz = tuple(float(x) for x in residue["CA"].coord)
            if not all(math.isfinite(x) for x in xyz):
                raise ValueError("Nonfinite structure coordinates")
            rows.append(
                {
                    "aa": seq1(residue.resname),
                    "xyz": xyz,
                    "residue": str(residue.id[1]) + residue.id[2].strip(),
                }
            )
        if rows:
            result[chain.id] = rows
    if not result:
        raise ValueError("Structure contains no protein CA coordinates")
    return result


def audit(pdb, binder, state=None):
    chains = coordinates(pdb)
    expected = {"H" if c.role in {"VH", "VHH"} else "L": c.sequence for c in binder.chains}
    if state is not None:
        expected["A"] = state.sequence
    for chain, sequence in expected.items():
        if "".join(r["aa"] for r in chains.get(chain, [])) != sequence:
            raise ValueError(f"Predicted chain {chain} does not exactly match the requested sequence")
    if set(chains) != set(expected):
        raise ValueError("Unexpected chains in evaluated structure")
    atoms = [(chain, i, row["xyz"]) for chain, rows in chains.items() for i, row in enumerate(rows)]
    # Spatial cells avoid a quadratic all-pairs loop for full antigen contexts.
    cells = {}
    clashes = 0
    for chain, index, xyz in atoms:
        cell = tuple(math.floor(v / 2.5) for v in xyz)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other_chain, other_index, other in cells.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), []
                    ):
                        if chain == other_chain and abs(index - other_index) <= 1:
                            continue
                        if math.dist(xyz, other) < 2.5:
                            clashes += 1
        cells.setdefault(cell, []).append((chain, index, xyz))
    contacts = []
    if state is not None:
        antibody = [row["xyz"] for c in ("H", "L") for row in chains.get(c, [])]
        contacts = [
            i
            for i, row in enumerate(chains["A"], 1)
            if any(math.dist(row["xyz"], other) <= 8.0 for other in antibody)
        ]
    return {
        "sequence_verified": True,
        "ca_clashes": clashes,
        "antigen_contacts": contacts,
        "geometry_method": "CA distance: nonlocal clashes <2.5 A; antigen contacts <=8 A; no all-atom claim",
    }
