#!/usr/bin/env python
"""Write the ZAK reference ligand C9R as an SDF with real bond orders.

Two of the pocket-conditioned generators in this catalogue refuse to sample
without a *reference ligand*, and neither will take the pocket file the rest of
them use:

``molecule.generate.molpilot``
    needs ``pocket['ref_ligand_sdf']``. Upstream's ``sample_for_pocket.py`` uses
    it twice — to define the site and, under ``sample_num_atoms='ref'``, to decide
    how many atoms to sample.
``molecule.generate.pocketxmol``
    needs a numeric ``pocket_center``, which is the same ligand's centroid.

Both inputs exist on this host already, but only as coordinates inside the
deposited entry. A PDB ``HETATM`` block carries no bond orders, so handing one
straight to a generator gives it a molecule with every bond single and every
aromatic ring broken. The template is therefore the chemical-component
definition's own SMILES, taken from ``target_manifest.json`` (which took it from
the RCSB component record, not from the paper's compound table) and applied with
:func:`rdkit.Chem.AllChem.AssignBondOrdersFromTemplate`.

This writes a *derived* artifact under the gitignored snapshot, exactly like the
pocket cut it sits beside. Nothing here is a measurement and nothing is fitted:
the coordinates are the deposited ones and the graph is the deposited component's.

Usage::

    PYTHONPATH=$PWD/src python scripts/research/build_zak_reference_ligand.py
    PYTHONPATH=$PWD/src python scripts/research/build_zak_reference_ligand.py --json out.json

Exit status is 0 when the SDF was written or already matched, 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / ".external" / "research" / "zak-c9r"
ENTRY_PDB = SNAPSHOT / "rcsb" / "6JUU.pdb"
MANIFEST = SNAPSHOT / "derived" / "target_manifest.json"
OUTPUT_SDF = SNAPSHOT / "derived" / "zak_6juu_C9R_ligand.sdf"

#: The deposited ligand's three-letter component id.
LIGAND_COMP_ID = "C9R"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build(output: Path = OUTPUT_SDF) -> dict[str, Any]:
    """Write the reference ligand SDF and return what went into it."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    if not ENTRY_PDB.is_file():
        raise SystemExit(f"{ENTRY_PDB} is missing; run scripts/research/fetch_zak_sources.py first")
    if not MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST} is missing; run scripts/research/build_zak_target_data.py first")

    manifest = json.loads(MANIFEST.read_text())
    smiles = (manifest.get("reference_ligand") or {}).get("standard_smiles")
    if not smiles:
        raise SystemExit(f"{MANIFEST} carries no reference_ligand.standard_smiles")

    lines = [
        ln
        for ln in ENTRY_PDB.read_text().splitlines()
        if ln.startswith("HETATM") and ln[17:20].strip() == LIGAND_COMP_ID
    ]
    if not lines:
        raise SystemExit(f"{ENTRY_PDB} holds no {LIGAND_COMP_ID} HETATM records")

    block = "\n".join(lines) + "\nEND\n"
    crystal = Chem.MolFromPDBBlock(block, removeHs=False)
    template = Chem.MolFromSmiles(smiles)
    if crystal is None or template is None:
        raise SystemExit("RDKit could not read the deposited ligand or its component SMILES")

    # Bond orders come from the component definition; coordinates stay the
    # deposited ones. A failure here is loud on purpose: a silently single-bonded
    # reference ligand would make every downstream sample wrong in a way no
    # validity check would catch.
    ligand = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(crystal))
    Chem.SanitizeMol(ligand)
    ligand.SetProp("_Name", f"{LIGAND_COMP_ID} from 6JUU (deposited coordinates)")

    conf = ligand.GetConformer()
    n = ligand.GetNumAtoms()
    center = [round(sum(conf.GetAtomPosition(i)[axis] for i in range(n)) / n, 3) for axis in range(3)]

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    try:
        writer.write(ligand)
    finally:
        writer.close()

    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "component_id": LIGAND_COMP_ID,
        "template_smiles": smiles,
        "written_smiles": Chem.MolToSmiles(Chem.RemoveHs(ligand)),
        "n_heavy_atoms": Chem.RemoveHs(ligand).GetNumAtoms(),
        "n_hetatm_records": len(lines),
        "centroid": center,
        "source_entry": str(ENTRY_PDB),
        "source_entry_sha256": sha256_file(ENTRY_PDB),
        "manifest_hash": manifest.get("manifest_hash"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_SDF)
    parser.add_argument("--json", type=Path, default=None, help="also write this record as JSON")
    args = parser.parse_args(argv)

    record = build(args.output)
    text = json.dumps(record, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
