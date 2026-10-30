"""Fixed-column PDB reading and pocket cutting, in the package, standard library only.

Every detector in this package ends the same way: it has found *where* a site is —
a set of alpha-sphere centres, a predicted centroid — and something downstream
needs a **prepared pocket PDB** at that location, because
:class:`~drug_agent.contracts.brief.PocketSpec` is a path and the pocket-conditioned
generators size their molecules from the atoms in it. Detection that stops at three
floats is detection that no generator can use.

``scripts/prepare_pocket.py`` already implements exactly that cut, and this module
is deliberately its twin rather than an improvement on it: same selection rule
(whole standard residues with any heavy atom within *radius* of a site point), same
altloc handling, same deterministic chain/residue ordering, same
``HEADER``/``COMPND``/``ATOM``/``END`` output shape. Two reasons it is duplicated
rather than imported:

  * ``scripts/`` is not packaged. ``pyproject.toml`` ships ``src/drug_agent`` plus
    ``system_prompts`` and nothing else, so a wheel install has no
    ``scripts.prepare_pocket`` to import, and an adapter that depended on one would
    be AVAILABLE in a checkout and broken in an install.
  * The script is a CLI over one file; the adapters need the pieces separately —
    parse once, cut three times, and report the residues that were kept.

The twinning is checked rather than asserted: cutting ``.external/pdb/3I28.pdb``
at 10 A around the 31 heavy atoms of its co-crystallised ligand ``34N`` produces
exactly the 752 ``ATOM`` records of ``.external/pockets/seh_3i28_pocket10.pdb``, the
file the script wrote. ``tests/test_pocket_tools.py`` pins that equality, so a change
here that quietly altered the selection rule would stop reproducing the pocket every
generator in this repository has been run against.

The one addition is provenance. A pocket written from a *prediction* is not the
same artifact as a pocket cut around a crystallographic ligand, and six months
later the only thing that can say which is the file itself, so
:func:`write_pocket_pdb` stamps ``REMARK`` lines naming the detector, its version,
the rank, the score and the centre. ``REMARK`` is skipped by every PDB reader in
this stack — PFM's ``PDBProtein`` reads only ``ATOM`` records — so the annotation
costs nothing at the far end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: Residue names a pocket file may contain. PFM's ``PDBProtein`` raises ``KeyError``
#: on anything outside this set, so a pocket carrying a modified residue, an ion or
#: a water is a pocket that crashes the generator it was cut for.
# A 20-residue table reads better as a grid than as one name per line.
# fmt: off
STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
})
# fmt: on

#: Above this many atoms the cut has stopped being a pocket and become a domain.
#: Same threshold ``scripts/prepare_pocket.py`` warns at.
LARGE_POCKET_ATOMS = 3000

#: Below this the site is probably wrong — a centre in solvent, or a radius typo.
SMALL_POCKET_ATOMS = 50

Point = tuple[float, float, float]


@dataclass(frozen=True)
class Atom:
    """One ``ATOM``/``HETATM`` record, with its source line kept verbatim.

    The line is carried because a pocket file is written by *copying* the selected
    records, never by re-formatting them: re-emitting fixed-column PDB is a way to
    silently lose occupancies, insertion codes and element columns that the
    downstream parser does read.
    """

    tag: str
    name: str
    alt_loc: str
    res_name: str
    chain: str
    res_seq: str
    element: str
    xyz: Point
    line: str

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return (self.chain, self.res_seq, self.res_name)

    @property
    def is_hydrogen(self) -> bool:
        return self.element == "H"


@dataclass(frozen=True)
class PocketCut:
    """The result of cutting one site out of a structure."""

    text: str
    n_residues: int
    n_atoms: int
    residues: tuple[str, ...]
    center: Point
    radius: float

    @property
    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.n_atoms > LARGE_POCKET_ATOMS:
            out.append(
                f"pocket has {self.n_atoms} atoms, which is large for a binding site — "
                f"a smaller radius than {self.radius:g} A would tighten it"
            )
        if self.n_atoms < SMALL_POCKET_ATOMS:
            out.append(
                f"pocket has only {self.n_atoms} atoms — check the detected centre is "
                f"on the protein and not in solvent"
            )
        return tuple(out)


def parse_atom_line(line: str) -> Atom | None:
    """Parse one fixed-column ``ATOM``/``HETATM`` record, or ``None``.

    Column positions, not whitespace splitting: a PDB with a four-character atom
    name and no space before the residue name tokenises wrongly and coordinates
    slide by one field, which shows up later as a pocket centre tens of Angstrom
    from the protein.
    """
    tag = line[0:6].strip()
    if tag not in ("ATOM", "HETATM"):
        return None
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None
    element = line[76:78].strip().capitalize() or line[12:16].strip()[:1]
    return Atom(
        tag=tag,
        name=line[12:16].strip(),
        alt_loc=line[16:17].strip(),
        res_name=line[17:20].strip(),
        chain=line[21:22].strip(),
        res_seq=line[22:27].strip(),  # residue number plus insertion code
        element=element,
        xyz=(x, y, z),
        line=line.rstrip("\n"),
    )


def parse_atoms(pdb_text: str) -> list[Atom]:
    """Every atom of the **first model** of *pdb_text*.

    NMR ensembles and some crystal depositions carry several ``MODEL`` blocks with
    identical residue numbering. Reading all of them would double every atom and
    move every centroid, so parsing stops at the first ``ENDMDL`` exactly as
    ``scripts/prepare_pocket.py`` does.
    """
    atoms: list[Atom] = []
    for line in pdb_text.splitlines():
        if line[0:6].strip() == "ENDMDL":
            break
        parsed = parse_atom_line(line)
        if parsed is not None:
            atoms.append(parsed)
    return atoms


def parse_points(text: str) -> list[Point]:
    """Coordinates of every ``ATOM``/``HETATM`` record, without residue semantics.

    For fpocket's ``pocket<N>_vert.pqr``, whose records are alpha-sphere centres
    written as pseudo-atoms of residue ``STP`` — geometry, not chemistry.
    """
    return [atom.xyz for atom in parse_atoms(text)]


def centroid(points: list[Point]) -> Point:
    """Arithmetic mean of *points*. Raises on an empty set rather than returning 0."""
    if not points:
        raise ValueError("cannot take the centroid of no points")
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def distance(a: Point, b: Point) -> float:
    """Euclidean distance in Angstrom."""
    return math.dist(a, b)


def bounding_size(points: list[Point], *, pad: float = 0.0) -> tuple[float, float, float]:
    """Axis-aligned extent of *points* plus ``2 * pad``, for a docking search box."""
    if not points:
        raise ValueError("cannot size a box around no points")
    return tuple(  # type: ignore[return-value]
        (max(p[i] for p in points) - min(p[i] for p in points)) + 2.0 * pad for i in range(3)
    )


def _residue_sort_key(res_seq: str) -> tuple[int, str]:
    digits = "".join(ch for ch in res_seq if ch.isdigit() or ch == "-")
    return (int(digits) if digits else 0, res_seq)


def cut_pocket(atoms: list[Atom], *, site_points: list[Point], radius: float = 10.0) -> PocketCut:
    """Whole standard residues with any heavy atom within *radius* of a site point.

    This is the CrossDocked ``pocket10`` convention the structure-conditioned
    generators in this repository were trained on, and it is the same rule the
    hand-prepared ``.external/pockets/seh_3i28_pocket10.pdb`` was cut with — which
    is what makes a detected pocket comparable with it rather than merely adjacent
    to it.

    *site_points* is whatever the detector actually produced: fpocket's alpha-sphere
    centres, or a single predicted centroid. Passing the points rather than a centre
    matters for a cavity that is long — a 10 A ball around the centre of an 17 A
    channel misses both ends of it.
    """
    if not site_points:
        raise ValueError("cut_pocket needs at least one site point")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    by_residue: dict[tuple[str, str, str], list[Atom]] = {}
    for atom in atoms:
        if atom.tag != "ATOM" or atom.res_name not in STANDARD_AA:
            continue
        if atom.alt_loc not in ("", "A"):  # one conformer, the first
            continue
        by_residue.setdefault(atom.residue_key, []).append(atom)

    r2 = radius * radius
    selected: list[tuple[tuple[str, str, str], list[Atom]]] = []
    for key, residue_atoms in by_residue.items():
        for atom in residue_atoms:
            if atom.is_hydrogen:
                continue
            if any(_sq_distance(atom.xyz, point) <= r2 for point in site_points):
                selected.append((key, residue_atoms))
                break
    selected.sort(key=lambda item: (item[0][0], _residue_sort_key(item[0][1])))

    lines: list[str] = []
    for _key, residue_atoms in selected:
        lines.extend(atom.line for atom in residue_atoms)

    return PocketCut(
        text="\n".join(lines),
        n_residues=len(selected),
        n_atoms=len(lines),
        residues=tuple(f"{key[0]}_{key[1]}" for key, _ in selected),
        center=centroid(site_points),
        radius=radius,
    )


def _sq_distance(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def write_pocket_pdb(cut: PocketCut, *, provenance: dict[str, Any]) -> str:
    """Render *cut* as a PDB file body, with its origin recorded in ``REMARK`` lines.

    The header is ``scripts/prepare_pocket.py``'s, unchanged, so a file from either
    producer parses identically. The ``REMARK`` block is the difference that matters
    downstream: a predicted pocket and a ligand-derived pocket are the same file
    format and very much not the same evidence, and this is where a reader — or a
    report — finds out which one is in their hands.
    """
    lines = ["HEADER    POCKET", "COMPND    POCKET"]
    for key, value in provenance.items():
        lines.append(f"REMARK   1 DRUGAGENT {key.upper()}: {value}")
    lines.append(f"REMARK   1 DRUGAGENT N_RESIDUES: {cut.n_residues}")
    lines.append(f"REMARK   1 DRUGAGENT N_ATOMS: {cut.n_atoms}")
    if cut.text:
        lines.append(cut.text)
    lines.append("END")
    return "\n".join(lines) + "\n"


__all__ = [
    "Atom",
    "LARGE_POCKET_ATOMS",
    "Point",
    "PocketCut",
    "SMALL_POCKET_ATOMS",
    "STANDARD_AA",
    "bounding_size",
    "centroid",
    "cut_pocket",
    "distance",
    "parse_atom_line",
    "parse_atoms",
    "parse_points",
    "write_pocket_pdb",
]
