"""A docked pose in, the exact ``(SDF, fragment_atom_indices)`` pair PocketXMol wants out.

PocketXMol's conditioned tasks — ``growing``, the five ``linking`` variants,
``optimize_partial`` — are the same denoiser as de-novo design told that some of
the atoms are already known. The telling is a partition:
``task.transform.preset_partition.grouped_node_p1`` names the atoms that are
*given* (upstream's "part1"), and with ``part1_pert: {fixed: 1}`` those atoms keep
both their element and their position while every other atom in the sample is
generated.

**The indices are the whole contract, and they are silent when wrong.** They are
0-based indices into the input SDF *as the sampler reads it*, and the sampler does
not read it the way a casual caller would. ``process/utils_process.py::get_pocmol_data``
does ``Chem.SDMolSupplier(path)`` — which removes hydrogens by default — and then
``Chem.RemoveAllHs`` on top. A pose written with its 18 explicit hydrogens still
attached therefore loses them before the partition is applied, every index above
the first hydrogen shifts, and the run produces a chemically plausible molecule
built around the wrong atoms with no error anywhere. The adapter's own
``known_failure_modes`` says exactly this. This module exists so that failure mode
cannot happen without raising.

Two properties make that guarantee:

1. **Nothing is assumed about atom order.** Every index this module returns has
   been checked against the file that was actually written, re-read through the
   sampler's own code path, atom by atom, on element *and* 3D coordinate.
   :func:`verify_conditioning_roundtrip` is the check, it is called on every
   :func:`prepare_conditioning`, and it is exported so campaign code can assert it
   per molecule rather than trust a returned list.
2. **The geometry comes from the docked pose, never from a fresh embedding.** The
   accepted molecules already have real conformers placed in this pocket by the
   docking run (``.research/pxm-prefilter/<run>/scoring/poses/<id>.sdf``, mirrored
   as ``pose_molblock`` in ``molecules.json``). Re-embedding with ETKDG would
   throw away the one thing that makes the fixed part1 worth fixing: a scaffold
   sitting where the scaffold actually binds.

**The SDF design: scaffold-only, not whole-molecule-plus-indices.**

There are two ways to hand PocketXMol a scaffold to grow from.

*(a) Write the scaffold substructure as its own molecule*, so the file contains
only the atoms to be held fixed and ``grouped_node_p1`` is every atom in it.

*(b) Write the whole parent molecule* and pass indices selecting the scaffold, so
the non-scaffold atoms become part2.

This module implements **(a)**, for four reasons:

* It is the shape upstream validated. ``configs/sample/examples/growing_fixed_frag.yml``
  points ``data.input_ligand`` at a fragment-only SDF of seven atoms and sets
  ``grouped_node_p1: [[0, 1, 2, 3, 4, 5, 6]]`` — all of them. Design (b) is the
  shape upstream uses for *linking* (``linking_fixed_frags.yml``: a 38-atom
  ligand, two fragment groups, the 16 linker atoms regenerated), which is a
  different task with a different meaning.
* The two designs do not mean the same thing. Under (b) the parent's side chains
  are part2 and get re-drawn at the parent's own size — that is R-group
  *redesign*, closer to ``optimize_partial``. Under (a) the molecule is built
  outward from the scaffold and its size comes from
  ``transforms.variable_mol_size``, which is what "growing" means.
* It removes a failure mode rather than documenting one. The driver mirrors the
  indices into ``transforms.variable_mol_size.not_remove``
  (``scripts/external/pocketxmol_sample.py::_apply_partition``); under (a) that is
  every atom in the file, so the variable-size transform can only add atoms and
  can never delete part of the scaffold it was told to keep. Under (b) a reorder
  on write/read silently re-partitions the molecule into a different given set.
* The file is smaller and contains no structure that is not being conditioned on.

Choosing (a) does not make verification optional, and this module does not treat
it that way. The written indices are still checked positionally — index *j* of the
returned group must be the *j*-th atom that was asked for — so a permutation is
caught even though a permutation of a total index set would leave the *set*
correct. That keeps the verifier honest enough to serve design (b) too, should a
linking campaign ever need it.

Nothing here prints, logs or returns a SMILES, a scaffold SMILES or a MOL block.
The metadata is counts, a checksum and a formula-free description of size.

**The leading underscore is load-bearing.** Every non-underscored module under
``src/drug_agent/tools/`` is expected to bind a capability and to appear in
``CAPABILITY_MODULES``, which ``tests/test_pools.py::test_every_tool_module_is_named_for_what_it_binds``
enforces. This module binds none — it prepares an *input* for the PocketXMol
adapter — and ``CAPABILITY_MODULES`` lives in
``selfevolve/evaluation/conformance.py``, which is inside the sealed kernel, so
registering a non-capability there would move ``kernel_digest()``. The underscore
is the convention already used for exactly this case by ``tools/pockets/_pdb.py``.
Renaming this file without an underscore breaks that test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

__all__ = [
    "COORDINATE_TOLERANCE_ANGSTROM",
    "ConditioningPlan",
    "EmptyScaffoldError",
    "RoundTripMismatchError",
    "ScaffoldConditioningError",
    "extract_fixed_substructure",
    "prepare_conditioning",
    "read_as_sampler",
    "scaffold_atom_indices",
    "verify_conditioning_roundtrip",
    "write_conditioning_sdf",
]


class ScaffoldConditioningError(ValueError):
    """A conditioning input could not be turned into a partition that is safe to run."""


class EmptyScaffoldError(ScaffoldConditioningError):
    """The molecule has no ring system, so its Murcko scaffold is empty.

    Refused rather than returned. An empty ``grouped_node_p1`` is de-novo
    generation wearing a conditioning label: nothing is held fixed, the run costs
    the same, and its output would be recorded as scaffold-conditioned.
    """


class RoundTripMismatchError(ScaffoldConditioningError):
    """The written SDF does not read back as the atoms the indices claim.

    This is the silent failure the module exists to make loud. It is raised, never
    warned, because the only thing downstream of a wrong partition is a batch of
    plausible molecules grown around the wrong atoms.
    """


#: Atom coordinates in a V2000 SDF are written ``%10.4f``, so a value can move by
#: at most 5e-5 Å per axis (under 1e-4 Å in 3D) through a write/read cycle. A
#: tolerance three orders of magnitude below the shortest chemical bond is
#: therefore loose enough never to fire on formatting and far too tight to let a
#: different atom pass as the intended one.
COORDINATE_TOLERANCE_ANGSTROM = 1e-3

#: Atom property used to carry a parent index through RDKit's Murcko decomposition.
#: ``GetScaffoldForMol`` builds its result by deleting atoms from a copy, so the
#: surviving ``Atom`` objects keep any non-computed property that was set on them.
#: That is an exact mapping back to the parent and needs no substructure match,
#: which matters: a symmetric scaffold can match its own parent several ways.
_PARENT_INDEX_PROP = "_drugagent_parent_idx"


def _rdkit() -> tuple[Any, Any]:
    """RDKit's ``Chem`` and ``MurckoScaffold``, imported late.

    Deferred so importing this module costs nothing in a process that only wants
    the exception types — the tools package is imported by the registry at
    start-up, and RDKit is not free.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return Chem, MurckoScaffold


# --- reading ---------------------------------------------------------------
def read_as_sampler(sdf_path: str | Path) -> Any:
    """Read *sdf_path* exactly the way PocketXMol's sampler reads its input ligand.

    Not "with RDKit" — *the same way*. ``get_pocmol_data`` takes the first record
    from ``Chem.SDMolSupplier(path)`` (whose ``removeHs`` defaults to true) and
    then applies ``Chem.RemoveAllHs``, which also strips the handful of hydrogens
    ``RemoveHs`` protects. Any check that reads the file differently is checking a
    molecule the model will never see, and the indices it blesses can still be
    wrong.
    """
    Chem, _ = _rdkit()
    path = Path(sdf_path)
    if not path.is_file():
        raise ScaffoldConditioningError(f"conditioning SDF does not exist: {path}")
    supplier = Chem.SDMolSupplier(str(path))
    records = list(supplier)
    if not records or records[0] is None:
        raise ScaffoldConditioningError(
            f"{path} did not yield a parseable molecule through Chem.SDMolSupplier, which is "
            "the reader PocketXMol uses; the sampler would fail on it too."
        )
    if len(records) > 1:
        raise ScaffoldConditioningError(
            f"{path} holds {len(records)} records and PocketXMol reads only the first "
            "(process/utils_process.py::get_pocmol_data). A conditioning file must hold "
            "exactly one molecule so there is no question which one was fixed."
        )
    return Chem.RemoveAllHs(records[0])


def _require_3d(mol: Any, what: str) -> Any:
    """The conformer of *mol*, refusing anything that is not a real 3D pose."""
    if mol is None:
        raise ScaffoldConditioningError(f"{what} is None")
    if mol.GetNumAtoms() == 0:
        raise ScaffoldConditioningError(f"{what} has no atoms")
    if mol.GetNumConformers() == 0:
        raise ScaffoldConditioningError(
            f"{what} carries no conformer. Conditioning geometry must come from the docked "
            "pose; there is nothing to hold fixed without coordinates, and embedding one "
            "here would discard the placement the docking run established."
        )
    conf = mol.GetConformer()
    if not conf.Is3D():
        raise ScaffoldConditioningError(
            f"{what} carries a 2D conformer. A depiction is not a pose: PocketXMol would fix "
            "the scaffold in a plane inside the pocket."
        )
    return conf


def _refuse_explicit_hydrogens(mol: Any, what: str) -> None:
    """Refuse a molecule carrying explicit hydrogens as atoms.

    The sampler strips them (``SDMolSupplier`` + ``RemoveAllHs``) and renumbers
    everything after them. Indices computed over a hydrogen-bearing molecule are
    therefore wrong *for the run* while looking perfectly correct here, which is
    the exact shape of the failure this module refuses to let through.
    """
    n_h = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
    if n_h:
        raise ScaffoldConditioningError(
            f"{what} carries {n_h} explicit hydrogen atom(s). PocketXMol reads its input "
            "ligand through Chem.SDMolSupplier followed by Chem.RemoveAllHs, so every index "
            "after the first hydrogen would shift between this call and the sample. Strip "
            "them with Chem.RemoveAllHs before computing indices."
        )


# --- the scaffold ----------------------------------------------------------
def scaffold_atom_indices(mol: Any) -> tuple[int, ...]:
    """Indices, into *mol* as given, of the atoms in its Bemis-Murcko scaffold.

    The mapping is exact rather than inferred. Each atom of a copy is tagged with
    its own index, ``MurckoScaffold.GetScaffoldForMol`` deletes the side chains
    from that copy, and the tags that survive *are* the answer. The obvious
    alternative — take the scaffold and substructure-match it back onto the parent
    — is ambiguous on any symmetric scaffold and can fail outright when removing a
    substituent changes an atom's implicit hydrogen count.

    Raises :class:`EmptyScaffoldError` for an acyclic molecule: Murcko's framework
    is rings plus the linkers between them, so a molecule with no ring has an
    empty one, and an empty fixed set is not conditioning.
    """
    Chem, MurckoScaffold = _rdkit()
    if mol is None:
        raise ScaffoldConditioningError("scaffold_atom_indices() was given None")
    if mol.GetNumAtoms() == 0:
        raise ScaffoldConditioningError("scaffold_atom_indices() was given an empty molecule")
    _refuse_explicit_hydrogens(mol, "the molecule passed to scaffold_atom_indices()")

    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetIntProp(_PARENT_INDEX_PROP, atom.GetIdx())

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(tagged)
    except Exception as exc:  # RDKit raises several unrelated types here
        raise ScaffoldConditioningError(
            f"RDKit could not compute a Murcko scaffold: {type(exc).__name__}: {exc}"
        ) from exc

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        raise EmptyScaffoldError(
            f"the molecule has no ring system, so its Bemis-Murcko scaffold is empty "
            f"({mol.GetNumAtoms()} heavy atoms, {mol.GetRingInfo().NumRings()} rings). "
            "Refusing: an empty part1 means "
            "nothing is held fixed, which is de-novo generation recorded as scaffold "
            "conditioning. Condition on a different molecule, or pick the fixed atoms "
            "explicitly and call write_conditioning_sdf() with them."
        )

    indices: list[int] = []
    for atom in scaffold.GetAtoms():
        if not atom.HasProp(_PARENT_INDEX_PROP):
            raise ScaffoldConditioningError(
                "RDKit's Murcko decomposition did not carry the parent-index tags through, so "
                "the scaffold cannot be mapped back onto the pose. Refusing rather than "
                "falling back to a substructure match, which is ambiguous under symmetry."
            )
        indices.append(int(atom.GetIntProp(_PARENT_INDEX_PROP)))

    unique = sorted(set(indices))
    if len(unique) != len(indices):
        raise ScaffoldConditioningError(
            "the Murcko decomposition mapped two scaffold atoms onto the same parent atom"
        )
    n_atoms = mol.GetNumAtoms()
    if unique[0] < 0 or unique[-1] >= n_atoms:
        raise ScaffoldConditioningError(
            f"scaffold indices {unique[0]}..{unique[-1]} fall outside the parent's "
            f"0..{n_atoms - 1} atom range"
        )
    return tuple(unique)


# --- the substructure that gets written ------------------------------------
def extract_fixed_substructure(mol: Any, keep_indices: list[int] | tuple[int, ...]) -> Any:
    """A standalone molecule holding only *keep_indices*, with their pose coordinates.

    Built by deleting every other atom from an ``RWMol`` copy, highest index
    first, so the surviving atoms come out in ascending *keep_indices* order and
    RDKit rewrites the conformer as it goes — the kept coordinates are the pose's
    own numbers, not a re-embedding.

    Cutting a substituent off an aromatic ring leaves an atom that was satisfied
    by its neighbour and is not any more; on three of the 107 accepted ZAK poses
    that produced a bare ``Can't kekulize`` from the sanitiser. Every kept atom
    that loses a heavy neighbour therefore has that neighbour's place accounted
    for before the deletion: an aromatic atom gains one explicit hydrogen per lost
    bond (the pyrrole-nitrogen repair), and every such atom is released to
    perceive its own implicit hydrogens again. With that, all 107 sanitise and
    each one's canonical form matches the scaffold RDKit's own
    ``GetScaffoldForMol`` returns.
    """
    Chem, _ = _rdkit()
    _require_3d(mol, "the molecule passed to extract_fixed_substructure()")
    keep = _validated_index_group(keep_indices, mol.GetNumAtoms())

    lost_neighbours = {
        idx: sum(1 for nb in mol.GetAtomWithIdx(idx).GetNeighbors() if nb.GetIdx() not in keep)
        for idx in keep
    }

    rw = Chem.RWMol(mol)
    for idx, n_lost in lost_neighbours.items():
        if not n_lost:
            continue
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetIsAromatic():
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + n_lost)
        atom.SetNoImplicit(False)
    for idx in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        rw.RemoveAtom(idx)

    frag = rw.GetMol()
    frag.ClearComputedProps()
    frag.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(frag)
    except Exception as exc:
        raise ScaffoldConditioningError(
            f"the selected {len(keep)} atoms do not form a molecule RDKit will sanitise "
            f"({type(exc).__name__}: {exc}). PocketXMol would fail to read it, so it is "
            "refused here rather than written."
        ) from exc
    if frag.GetNumAtoms() != len(keep):
        raise ScaffoldConditioningError(
            f"substructure extraction produced {frag.GetNumAtoms()} atoms for {len(keep)} requested indices"
        )
    return frag


def _validated_index_group(indices: Any, n_atoms: int) -> set[int]:
    """*indices* as a set, refusing anything that is not 0-based and in range."""
    try:
        values = [int(i) for i in indices]
    except (TypeError, ValueError) as exc:
        raise ScaffoldConditioningError(f"atom indices must be integers: {exc}") from exc
    if not values:
        raise ScaffoldConditioningError(
            "no atom indices were given. An empty fixed set is de-novo generation, not "
            "conditioning; see EmptyScaffoldError."
        )
    out = set(values)
    if len(out) != len(values):
        raise ScaffoldConditioningError(
            f"atom indices contain {len(values) - len(out)} duplicate(s); PocketXMol's "
            "grouped_node_p1 names each given atom once"
        )
    bad = sorted(i for i in out if i < 0 or i >= n_atoms)
    if bad:
        raise ScaffoldConditioningError(
            f"atom indices {bad} are outside the molecule's 0..{n_atoms - 1} range. Indices "
            "are 0-based into the SDF as RDKit reads it."
        )
    return out


def write_conditioning_sdf(
    mol: Any,
    keep_indices: list[int] | tuple[int, ...],
    out_path: str | Path,
) -> tuple[int, ...]:
    """Write the fixed atoms of *mol* to *out_path* and return their indices in that file.

    Design (a) of the two described in the module docstring: the file holds the
    substructure alone, so the returned indices are every atom in it. They are
    returned in the order *keep_indices* was given, which is the order the atoms
    were written, and they are returned only after
    :func:`verify_conditioning_roundtrip` has re-read the file through the
    sampler's own code path and confirmed atom by atom that they are the atoms
    that were asked for.

    The molecule is written without explicit hydrogens because the sampler removes
    them; writing them would leave this function's indices correct for the file
    and wrong for the run.
    """
    Chem, _ = _rdkit()
    _require_3d(mol, "the molecule passed to write_conditioning_sdf()")
    _refuse_explicit_hydrogens(mol, "the molecule passed to write_conditioning_sdf()")
    keep = tuple(int(i) for i in keep_indices)
    _validated_index_group(keep, mol.GetNumAtoms())

    frag = extract_fixed_substructure(mol, keep)
    # Deletion ran highest-index-first, so the fragment's atoms are in ascending
    # parent order. The caller's order is honoured by mapping back through it.
    ascending = sorted(set(keep))
    position_of = {parent_idx: j for j, parent_idx in enumerate(ascending)}
    written = tuple(position_of[i] for i in keep)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(frag, str(path))

    verify_conditioning_roundtrip(mol, keep, path, written)
    return written


# --- the check that makes the indices worth returning ----------------------
def verify_conditioning_roundtrip(
    mol: Any,
    keep_indices: list[int] | tuple[int, ...],
    sdf_path: str | Path,
    written_indices: list[int] | tuple[int, ...],
    *,
    tolerance: float = COORDINATE_TOLERANCE_ANGSTROM,
) -> dict[str, Any]:
    """Confirm *written_indices* select, in *sdf_path*, exactly the atoms *keep_indices* select in *mol*.

    Atom order is not guaranteed to survive a write/read cycle, so this asserts it
    instead of assuming it, and asserts it *positionally*: the atom at
    ``written_indices[j]`` in the re-read file must be the atom at
    ``keep_indices[j]`` in the pose, matched on atomic number, formal charge and
    3D coordinate. A set-based check would pass a permutation, and a permutation
    is precisely how a conditioning run ends up growing from the wrong atoms.

    The file is re-read with :func:`read_as_sampler`, so what is verified is the
    molecule PocketXMol will actually partition — not a more forgiving parse of
    the same bytes.

    Returns a small dict of what was checked (counts and the worst coordinate
    deviation seen) so a caller can record the evidence; raises
    :class:`RoundTripMismatchError` on any disagreement.
    """
    source_conf = _require_3d(mol, "the molecule passed to verify_conditioning_roundtrip()")
    keep = [int(i) for i in keep_indices]
    written = [int(i) for i in written_indices]
    _validated_index_group(keep, mol.GetNumAtoms())

    reread = read_as_sampler(sdf_path)
    n_written = reread.GetNumAtoms()
    if len(written) != len(keep):
        raise RoundTripMismatchError(
            f"{len(written)} written indices were offered for {len(keep)} fixed atoms"
        )
    _validated_index_group(written, n_written)
    if n_written != len(written):
        raise RoundTripMismatchError(
            f"{Path(sdf_path).name} reads back as {n_written} atoms through PocketXMol's own "
            f"reader but the partition names {len(written)}. Under the scaffold-only design "
            "every atom in the file is fixed, so a difference means the file gained or lost "
            "atoms on the way out — hydrogens are the usual culprit."
        )

    target_conf = _require_3d(reread, f"the molecule re-read from {Path(sdf_path).name}")
    worst = 0.0
    worst_at = -1
    for j, (parent_idx, file_idx) in enumerate(zip(keep, written, strict=True)):
        want = mol.GetAtomWithIdx(parent_idx)
        got = reread.GetAtomWithIdx(file_idx)
        if want.GetAtomicNum() != got.GetAtomicNum():
            raise RoundTripMismatchError(
                f"atom {j} of the partition: the pose's atom {parent_idx} has atomic number "
                f"{want.GetAtomicNum()} but index {file_idx} of {Path(sdf_path).name} has "
                f"{got.GetAtomicNum()}. The written indices do not name the intended atoms; "
                "running this partition would fix the wrong part of the molecule."
            )
        if want.GetFormalCharge() != got.GetFormalCharge():
            raise RoundTripMismatchError(
                f"atom {j} of the partition: formal charge {want.GetFormalCharge()} in the "
                f"pose against {got.GetFormalCharge()} at index {file_idx} of "
                f"{Path(sdf_path).name}"
            )
        a = source_conf.GetAtomPosition(parent_idx)
        b = target_conf.GetAtomPosition(file_idx)
        dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        if dist > worst:
            worst, worst_at = dist, j
        if dist > tolerance:
            raise RoundTripMismatchError(
                f"atom {j} of the partition moved {dist:.4f} A between the pose's atom "
                f"{parent_idx} and index {file_idx} of {Path(sdf_path).name}, over the "
                f"{tolerance} A tolerance. Either the atom order changed on write/read or "
                "the coordinates are not the docked ones; both make the fixed part1 a "
                "different thing from the pose that was accepted."
            )

    return {
        "n_fixed_atoms": len(keep),
        "n_atoms_in_file": n_written,
        "max_coordinate_deviation_angstrom": round(worst, 6),
        "max_deviation_at_partition_position": worst_at,
        "coordinate_tolerance_angstrom": tolerance,
        "reader": "Chem.SDMolSupplier + Chem.RemoveAllHs (PocketXMol's own path)",
    }


# --- end to end ------------------------------------------------------------
class ConditioningPlan(NamedTuple):
    """What a conditioned PocketXMol run needs, and the evidence it is right.

    Unpacks as ``(sdf_path, fragment_atom_indices, metadata)``.
    ``fragment_atom_indices`` is already grouped per fragment, which is the shape
    ``PocketXMolAdapter(fragment_atom_indices=...)`` and upstream's
    ``grouped_node_p1`` both take.
    """

    sdf_path: Path
    fragment_atom_indices: list[list[int]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PoseSource:
    """Where the pose came from, for the metadata. No structure, only provenance."""

    kind: str
    reference: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _load_pose(pose: Any) -> tuple[Any, _PoseSource]:
    """A pose molecule from a Mol, an SDF path, or a MOL block string."""
    Chem, _ = _rdkit()
    if pose is None:
        raise ScaffoldConditioningError("prepare_conditioning() was given no pose")
    if hasattr(pose, "GetNumAtoms"):
        return Chem.RemoveAllHs(Chem.Mol(pose)), _PoseSource("rdkit_mol")
    if isinstance(pose, (str, Path)):
        text = str(pose)
        looks_like_block = "\n" in text and ("V2000" in text or "V3000" in text)
        if looks_like_block:
            mol = Chem.MolFromMolBlock(text)
            if mol is None:
                raise ScaffoldConditioningError(
                    "the MOL block given as the pose did not parse; pose_molblock in "
                    "molecules.json is the usual source and it is expected to parse"
                )
            return Chem.RemoveAllHs(mol), _PoseSource("molblock")
        return read_as_sampler(text), _PoseSource("sdf_file", reference=str(Path(text)))
    raise ScaffoldConditioningError(
        f"prepare_conditioning() cannot read a pose of type {type(pose).__name__}; pass an "
        "RDKit Mol, a path to a pose SDF, or a MOL block string"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_conditioning(
    pose: Any,
    out_path: str | Path,
    *,
    molecule_id: str = "",
    keep_indices: list[int] | tuple[int, ...] | None = None,
    tolerance: float = COORDINATE_TOLERANCE_ANGSTROM,
) -> ConditioningPlan:
    """A docked pose in; the SDF, the partition and the evidence out.

    *pose* is an RDKit ``Mol``, a path to a pose SDF
    (``.research/pxm-prefilter/<run>/scoring/poses/<id>.sdf``) or a MOL block
    (``molecules.json``'s ``pose_molblock``). Hydrogens are removed on the way in
    with ``Chem.RemoveAllHs``, which is what the sampler does, so the indices this
    returns are indices into the molecule the model will see.

    *keep_indices* overrides the Murcko scaffold for a caller who has already
    decided which atoms to fix — a BRICS fragment, a matched-pair core. Left
    ``None``, the fixed set is the Bemis-Murcko scaffold of the pose.

    The returned plan is verified: the file has been written, re-read through
    PocketXMol's own reader and checked atom by atom before this returns, and the
    same check is exported as :func:`verify_conditioning_roundtrip` so a campaign
    can re-assert it at dispatch time.
    """
    mol, source = _load_pose(pose)
    _require_3d(mol, "the pose given to prepare_conditioning()")
    _refuse_explicit_hydrogens(mol, "the pose given to prepare_conditioning()")

    if keep_indices is None:
        fixed = scaffold_atom_indices(mol)
        selection = "bemis_murcko_scaffold"
    else:
        fixed = tuple(int(i) for i in keep_indices)
        _validated_index_group(fixed, mol.GetNumAtoms())
        selection = "caller_supplied"

    path = Path(out_path)
    written = write_conditioning_sdf(mol, fixed, path)
    evidence = verify_conditioning_roundtrip(mol, fixed, path, written, tolerance=tolerance)

    n_parent = mol.GetNumAtoms()
    n_fixed = len(fixed)
    metadata: dict[str, Any] = {
        "molecule_id": molecule_id,
        "selection": selection,
        "sdf_design": "substructure_only",
        "n_parent_atoms": n_parent,
        "n_scaffold_atoms": n_fixed,
        # Atoms of the pose that are NOT in the conditioning file. Under the
        # scaffold-only design they are simply absent, not "regenerated": how many
        # atoms the sample actually grows is set by transforms.variable_mol_size
        # (the adapter's mol_size_mean / mol_size_std), which can take the result
        # past the parent's own size. This number is the decoration that was cut,
        # and on the accepted ZAK poses it is usually 0-2 — worth reading before
        # concluding that a scaffold-growing campaign has room to explore.
        "n_parent_atoms_dropped": n_parent - n_fixed,
        "scaffold_atom_fraction": round(n_fixed / n_parent, 4),
        "parent_atom_indices_fixed": [int(i) for i in fixed],
        "sdf_path": str(path),
        "sdf_sha256": _sha256(path),
        "pose_source": source.kind,
        "pose_reference": source.reference,
        "roundtrip_verified": True,
        "roundtrip": evidence,
    }
    return ConditioningPlan(path, [[int(i) for i in written]], metadata)
