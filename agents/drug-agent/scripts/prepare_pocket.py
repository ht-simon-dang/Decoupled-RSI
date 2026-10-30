"""Cut a binding-site pocket out of a PDB structure, ready for PFM sampling.

Structure-conditioned generators size the ligand from the *pocket's* spatial
extent, so handing PFM a whole protein produces meaningless molecules. Upstream
ships ``scripts/data_preparation/clean_pocket.py`` for this, but it is hardcoded
to the authors' filenames and a single literal coordinate, so it cannot be used
for a new target. This is the equivalent tool, parameterized.

It runs in the **drug-agent** environment (standard library only) — no PFM, no
torch — so a pocket can be prepared before any of that is installed.

Two ways to say where the site is::

    # 1. Around a co-crystallized ligand (preferred — it *is* the binding site)
    python scripts/prepare_pocket.py 3I28.pdb --ligand 34N -o seh_pocket.pdb

    # 2. Around explicit coordinates
    python scripts/prepare_pocket.py 3I28.pdb --center 12.5,-3.1,20.4 -o pocket.pdb

Selection matches the CrossDocked "pocket10" convention PFM was trained on: every
standard amino-acid residue with **any heavy atom within `--radius` Å** of the
site is kept, whole. Waters, ions, and hetero-groups are dropped — PFM's
``PDBProtein`` parser only reads ``ATOM`` records and raises on non-standard
residue names.

The resulting path goes into a brief as ``generation.pocket.pdb_path``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: PFM's PDBProtein raises KeyError on anything outside this set.
# A 20-residue table reads better as a grid than as one name per line.
# fmt: off
STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
}
# fmt: on


def _parse_atom_line(line: str) -> dict | None:
    """Parse one fixed-column PDB ATOM/HETATM record."""
    tag = line[0:6].strip()
    if tag not in ("ATOM", "HETATM"):
        return None
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None
    element = line[76:78].strip().capitalize() or line[12:16].strip()[:1]
    return {
        "line": line.rstrip("\n"),
        "tag": tag,
        "name": line[12:16].strip(),
        "altloc": line[16:17].strip(),
        "res_name": line[17:20].strip(),
        "chain": line[21:22].strip(),
        "res_seq": line[22:27].strip(),  # res id + insertion code
        "xyz": (x, y, z),
        "element": element,
    }


def _dist2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def extract_pocket(
    pdb_text: str,
    *,
    ligand: str | None = None,
    center: tuple[float, float, float] | None = None,
    radius: float = 10.0,
    chain: str | None = None,
) -> tuple[str, dict]:
    """Return ``(pocket_pdb_text, info)`` for the site given by ligand or center."""
    atoms = []
    for line in pdb_text.splitlines():
        if line[0:6].strip() == "ENDMDL":
            break  # first model only
        parsed = _parse_atom_line(line)
        if parsed is not None:
            atoms.append(parsed)
    if not atoms:
        raise ValueError("no ATOM/HETATM records found — is this a PDB file?")

    # --- locate the site --------------------------------------------------
    if ligand:
        site_points = [
            a["xyz"]
            for a in atoms
            if a["tag"] == "HETATM"
            and a["res_name"].upper() == ligand.upper()
            and a["element"] != "H"
            and (chain is None or a["chain"] == chain)
        ]
        if not site_points:
            present = sorted(
                {a["res_name"] for a in atoms if a["tag"] == "HETATM" and a["res_name"] != "HOH"}
            )
            raise ValueError(
                f"ligand {ligand!r} not found. HETATM residues present: {', '.join(present) or 'none'}"
            )
    elif center:
        site_points = [center]
    else:
        raise ValueError("give either --ligand or --center")

    site_center = tuple(sum(p[i] for p in site_points) / len(site_points) for i in range(3))

    # --- select whole residues with any heavy atom in range ---------------
    r2 = radius * radius
    by_residue: dict[tuple[str, str, str], list[dict]] = {}
    for a in atoms:
        if a["tag"] != "ATOM" or a["res_name"] not in STANDARD_AA:
            continue
        if a["altloc"] not in ("", "A"):  # keep one conformer
            continue
        by_residue.setdefault((a["chain"], a["res_seq"], a["res_name"]), []).append(a)

    selected = []
    for key, res_atoms in by_residue.items():
        for a in res_atoms:
            if a["element"] == "H":
                continue
            if any(_dist2(a["xyz"], p) <= r2 for p in site_points):
                selected.append((key, res_atoms))
                break
    # Deterministic order: chain, then residue number.
    selected.sort(key=lambda item: (item[0][0], _res_sort_key(item[0][1])))

    lines = ["HEADER    POCKET", "COMPND    POCKET"]
    n_atoms = 0
    for _key, res_atoms in selected:
        for a in res_atoms:
            lines.append(a["line"])
            n_atoms += 1
    lines.append("END")

    info = {
        "n_residues": len(selected),
        "n_atoms": n_atoms,
        "center": [round(c, 3) for c in site_center],
        "radius": radius,
        "site": f"ligand {ligand}" if ligand else "explicit center",
    }
    return "\n".join(lines) + "\n", info


def _res_sort_key(res_seq: str) -> tuple[int, str]:
    digits = "".join(ch for ch in res_seq if ch.isdigit() or ch == "-")
    return (int(digits) if digits else 0, res_seq)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", type=Path, help="input PDB file (whole structure)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output pocket PDB")
    ap.add_argument("--ligand", help="residue name of a co-crystallized ligand, e.g. 34N")
    ap.add_argument("--center", help="site centre as 'x,y,z' (Angstrom)")
    ap.add_argument("--chain", help="restrict the ligand search to one chain")
    ap.add_argument("--radius", type=float, default=10.0, help="pocket radius in Angstrom (default 10)")
    args = ap.parse_args()

    center = None
    if args.center:
        parts = [p for p in args.center.replace(" ", "").split(",") if p]
        if len(parts) != 3:
            ap.error("--center must be 'x,y,z'")
        center = tuple(float(p) for p in parts)  # type: ignore[assignment]

    try:
        text, info = extract_pocket(
            args.pdb.read_text(),
            ligand=args.ligand,
            center=center,
            radius=args.radius,
            chain=args.chain,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"{args.out}: {info['n_residues']} residues / {info['n_atoms']} atoms "
        f"within {info['radius']} A of {info['site']} "
        f"(centre {info['center'][0]}, {info['center'][1]}, {info['center'][2]})"
    )
    if info["n_atoms"] > 3000:
        print("warning: that is large for a pocket — consider a smaller --radius", file=sys.stderr)
    if info["n_atoms"] < 50:
        print("warning: very few atoms — check the ligand/centre is right", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Re-exported for tests and programmatic use.
__all__ = ["STANDARD_AA", "extract_pocket"]
