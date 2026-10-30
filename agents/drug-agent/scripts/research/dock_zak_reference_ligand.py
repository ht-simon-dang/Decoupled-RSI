#!/usr/bin/env python
"""Redock the ZAK reference ligand C9R into 6JUU with every available backend.

This is the acceptance evidence for repair R1 Part B, and it is a *self-docking
sanity check on the install*, not a benchmark and not a calibration. It asks one
question of each docking backend: given the receptor Step 5 prepared and the box
the deposited ligand occupies, does the engine put the ligand back where the
crystallographer found it?

It is deliberately not a unit test. It needs the real binaries, a GPU for GNINA's
CNN rescoring, and ~25 s, and it reads files under ``.external/`` that are not in
the repository. ``tests/research/test_docking_backends.py`` holds the parts that
can run anywhere.

Inputs, all from Step 5's snapshot:

``.external/research/zak-c9r/rcsb/6JUU.pdb``
    The deposited entry. The receptor is its ATOM records — protein only, no
    waters and no ligand — so the engine cannot see the answer it is being asked
    to reproduce.
``.external/research/zak-c9r/derived/target_manifest.json``
    Where the reference ligand's SMILES and the structure's identity come from,
    so the ligand docked is the ligand the manifest says is in the structure.

Usage::

    PYTHONPATH=$PWD/src python scripts/research/dock_zak_reference_ligand.py
    PYTHONPATH=$PWD/src python scripts/research/dock_zak_reference_ligand.py --json out.json

Exit status is 0 when at least one backend docked, 1 when none could.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / ".external" / "research" / "zak-c9r"
ENTRY_PDB = SNAPSHOT / "rcsb" / "6JUU.pdb"
MANIFEST = SNAPSHOT / "derived" / "target_manifest.json"
POCKET = SNAPSHOT / "derived" / "zak_6juu_C9R_pocket10.pdb"

#: The deposited ligand's three-letter component id, from the manifest.
LIGAND_COMP_ID = "C9R"

#: Padding added to the reference ligand's own extent to make the search box. The
#: ligand is 18 Å long in one direction, so a cube would be either too small for
#: it or far larger than the site in the other two.
BOX_PAD = 6.0

#: Fixed so two runs of this script are comparable. Vina and GNINA both take it.
SEED = 42


def _atom_xyz(lines: list[str]) -> list[tuple[float, float, float]]:
    return [(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])) for ln in lines]


def _centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    n = float(len(points))
    return tuple(sum(p[i] for p in points) / n for i in range(3))  # type: ignore[return-value]


def _extent(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(  # type: ignore[return-value]
        max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)
    )


def prepare_inputs(workdir: Path) -> dict[str, Any]:
    """Receptor PDB, reference-ligand SDF, and the box they imply."""
    if not ENTRY_PDB.is_file():
        raise SystemExit(f"{ENTRY_PDB} is missing; run scripts/research/fetch_zak_sources.py first")
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    reference = manifest.get("reference_ligand", {})
    smiles = reference.get("standard_smiles")
    if not smiles:
        raise SystemExit(f"{MANIFEST} carries no reference_ligand.standard_smiles")

    text = ENTRY_PDB.read_text().splitlines()
    protein = [ln for ln in text if ln.startswith("ATOM")]
    ligand = [ln for ln in text if ln.startswith("HETATM") and ln[17:20].strip() == LIGAND_COMP_ID]
    if not protein or not ligand:
        raise SystemExit(f"{ENTRY_PDB} has {len(protein)} ATOM and {len(ligand)} {LIGAND_COMP_ID} records")

    receptor = workdir / "receptor.pdb"
    receptor.write_text("\n".join(protein) + "\nEND\n")
    crystal = workdir / f"{LIGAND_COMP_ID}.pdb"
    crystal.write_text("\n".join(ligand) + "\nEND\n")

    points = _atom_xyz(ligand)
    center = _centroid(points)
    size = tuple(round(e + BOX_PAD, 3) for e in _extent(points))
    return {
        "receptor_pdb": receptor,
        "crystal_pdb": crystal,
        "smiles": smiles,
        "center": tuple(round(c, 3) for c in center),
        "size": size,
        "receptor_id": f"{manifest.get('structure', {}).get('pdb_id', '6JUU')}:"
        f"{manifest.get('structure', {}).get('chain_id', 'A')}",
        "manifest_hash": manifest.get("manifest_hash"),
        "pocket_pdb": str(POCKET) if POCKET.is_file() else None,
        "n_protein_atoms": len(protein),
        "n_ligand_atoms": len(ligand),
    }


def rmsd_to_crystal(pose_ref: str, crystal_pdb: Path, smiles: str) -> float | None:
    """Symmetry-corrected in-place RMSD of a docked pose from the crystal ligand.

    In place: no superposition. The engine was given the receptor and a box, not
    the answer, so agreement with the deposited coordinates is the measurement.
    Bond orders are assigned from the manifest's SMILES on both sides, because a
    PDB ligand and a docked pose both arrive without them and
    :func:`rdkit.Chem.rdMolAlign.CalcRMS` needs matching graphs to enumerate the
    symmetry it is correcting for.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign

    RDLogger.DisableLog("rdApp.*")
    template = Chem.MolFromSmiles(smiles)
    crystal = Chem.MolFromPDBFile(str(crystal_pdb))
    if template is None or crystal is None:
        return None
    # A Pose carries a mol block whose leading (empty) title line was stripped by
    # the contract's whitespace handling; restoring it is what makes it parse.
    block = pose_ref if pose_ref.startswith("\n") else f"\n{pose_ref}"
    docked = Chem.MolFromMolBlock(block)
    if docked is None:
        return None
    try:
        ref = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(crystal))
        probe = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(docked))
        return round(float(rdMolAlign.CalcRMS(probe, ref)), 3)
    except (ValueError, RuntimeError):
        return None


def run_backend(name: str, adapter: Any, inputs: dict[str, Any], n_poses: int) -> dict[str, Any]:
    """Dock once with *adapter* and report what came back, or why nothing did."""
    record: dict[str, Any] = {"backend": name, "available": bool(adapter.available())}
    if not adapter.available():
        record["status"] = "UNAVAILABLE"
        record["reason"] = adapter.profile.unavailable_reason
        return record
    kwargs: dict[str, Any] = {
        "candidate_id": "zak_c9r_redock",
        "smiles": inputs["smiles"],
        "receptor_pdb": inputs["receptor_pdb"],
        "receptor_id": inputs["receptor_id"],
        "center": inputs["center"],
        "size": inputs["size"],
        "n_poses": n_poses,
        "seed": SEED,
        "exhaustiveness": 8,
    }
    started = time.time()
    try:
        poses = adapter.dock(**kwargs)
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.time() - started, 2)
        return record
    record["status"] = "SUCCEEDED"
    record["seconds"] = round(time.time() - started, 2)
    record["engine_version"] = poses[0].engine_version if poses else None
    record["n_poses"] = len(poses)
    record["poses"] = [
        {
            "rank": p.rank,
            "score": p.score,
            "score_unit": p.score_unit,
            "engine": p.engine,
            "trust_label": str(p.trust_label),
            "qc_flags": list(p.qc_flags),
            "rmsd_to_crystal_A": rmsd_to_crystal(p.pose_ref, inputs["crystal_pdb"], inputs["smiles"]),
        }
        for p in poses
    ]
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    parser.add_argument("--n-poses", type=int, default=9)
    parser.add_argument("--only", default="", help="run one backend only: vina or gnina")
    args = parser.parse_args(argv)

    import tempfile

    from drug_agent.tools.structure.gnina_tool import GninaAdapter
    from drug_agent.tools.vina_tool import DockingAdapter

    with tempfile.TemporaryDirectory(prefix="zak_redock_") as tmp:
        workdir = Path(tmp)
        inputs = prepare_inputs(workdir)
        wanted = {"vina": DockingAdapter(), "gnina": GninaAdapter()}
        if args.only:
            wanted = {k: v for k, v in wanted.items() if k == args.only}
            if not wanted:
                raise SystemExit(f"unknown backend {args.only!r}; this script drives vina, gnina")
        report: dict[str, Any] = {
            "target": "ZAK / MAP3K20 (UniProt Q9NYL2), PDB 6JUU chain A",
            "ligand": f"{LIGAND_COMP_ID} (paper compound 6r), the deposited ligand",
            "smiles": inputs["smiles"],
            "receptor": "ATOM records of .external/research/zak-c9r/rcsb/6JUU.pdb",
            "n_protein_atoms": inputs["n_protein_atoms"],
            "n_ligand_atoms": inputs["n_ligand_atoms"],
            "target_manifest_hash": inputs["manifest_hash"],
            "box_center": list(inputs["center"]),
            "box_size": list(inputs["size"]),
            "seed": SEED,
            "backends": [run_backend(n, a, inputs, args.n_poses) for n, a in wanted.items()],
        }

    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text)
    print(text)
    docked = [b for b in report["backends"] if b.get("status") == "SUCCEEDED"]
    return 0 if docked else 1


if __name__ == "__main__":
    sys.exit(main())
