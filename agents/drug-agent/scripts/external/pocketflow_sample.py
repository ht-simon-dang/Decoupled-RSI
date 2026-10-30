"""Sample ligands inside one pocket with PocketFlow — the drug-agent ⇄ PocketFlow bridge.

This file is executed inside whichever interpreter carries PocketFlow's
dependencies. It imports nothing from ``drug_agent``: the two sides talk over a
JSON request/response pair on disk, so a CUDA fault inside the autoregressive
sampler cannot reach the caller.

**Which upstream entry point this reproduces.** ``main_generate.py`` is the only
generation entry point in the checkout, and its ``__main__`` block (lines 39-102)
is a straight-line script with no callable wrapper. This driver reproduces that
block statement for statement — the parse, the six transforms, the checkpoint
load, the :class:`Generate` construction — replacing only the argparse front-end
and the file-writing back-end. Every call below carries the ``main_generate.py``
line it was taken from, at ``PocketFlow`` commit a31a5a03f7d1.

**Why ``Generate.run`` and not ``Generate.generate``.** ``generate()``
(``pocket_flow/generate.py`` lines 348-387) writes SMILES and SDF to files under
a timestamped directory and returns ``None``, so its output could only be
recovered by scraping a directory it names itself. Its per-sample body is
``out = self.run(data.clone().detach())`` (lines 357-358), which is what this
driver calls directly. That also sidesteps an upstream bug: ``mol`` is only bound
inside ``if out:`` (line 360-361) but is read unconditionally at line 362, so a
first sample that fails raises ``NameError`` and destroys the whole batch. Here a
failed sample is counted and reported as data.

**Three incompatibilities this driver detects and refuses rather than papers
over.** All three were measured on this machine (aarch64, PyG 2.8.0.post1,
torch 2.14.0+cu130); see :mod:`drug_agent.tools.generation.pocketflow_tool` for
the full account. Each is checked up front so the failure names the cause instead
of surfacing as an ``ImportError`` from three frames inside upstream:

1. ``tensorboard`` — ``pocket_flow/utils/train.py`` line 11 does
   ``from torch.utils import tensorboard`` at module scope, and
   ``pocket_flow/__init__.py`` line 1 reaches it via ``from .utils import *``.
   The package cannot be imported at all without it, on the generation path
   included.
2. ``torch_geometric.utils.subgraph`` — ``pocket_flow/utils/transform_utils.py``
   line 5 imports it. PyG moved that module to ``torch_geometric.utils._subgraph``
   in 2.5; upstream's ``environment.yml`` pins ``torch-geometric==2.3.1``.
3. ``pyg-lib`` — PyG 2.8 routes ``knn``, ``knn_graph``, ``radius`` and
   ``radius_graph`` through pyg-lib and raises ``ImportError`` for each when
   ``torch_geometric.typing.WITH_KNN`` / ``WITH_RADIUS`` are false. PocketFlow
   calls all four (``pocket_flow/generate.py`` line 2;
   ``pocket_flow/utils/transform_utils.py`` lines 242 and 256).

**CUDA is not optional.** ``Generate.run`` builds its two prior distributions with
``torch.zeros([...]).cuda(data.cpx_pos.device)`` (``pocket_flow/generate.py``
lines 246-253), unconditionally, so a CPU device raises inside the sampler. The
``-d cpu`` flag ``main_generate.py`` line 24 advertises does not work.

Request JSON:
    pocket_pdb          path to a prepared POCKET pdb (a ~10 A region, not a protein)
    checkpoint          path to a PocketFlow .pt (ships in the checkout under ckpt/)
    n_molecules         how many samples to attempt
    device              "cuda:N" — see the CUDA note above
    seed                RNG seed (upstream seeds nothing; this driver's addition)
    max_atom_num        hard cap on atoms per molecule
    atom_temperature    atom-flow sampling temperature
    bond_temperature    bond-flow sampling temperature
    focus_threshold     probability floor for a focal atom
    choose_max          take the highest-probability focal atom instead of sampling
    min_dist_inter_mol  protein-ligand distance floor for the FIRST atom, in Angstrom
    bond_length_range   [lo, hi] bond-length window in Angstrom
    max_double_in_6ring double bonds tolerated in a six-ring before repair
    out_dir             directory to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]         one entry per *attempt*, failures included — a sample that
                        terminated with no valid molecule is reported as an empty
                        ``smiles`` with ``qc_flags=["generation_failed"]`` so the
                        caller's validity denominator is the number attempted, not
                        the number that happened to survive.
    n_requested, n_sampled, n_generation_failures, n_pocket_atoms,
    n_surface_atoms, device, seed, checkpoint, max_atom_num, warnings[], errors[]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path bootstrap above, deliberately.
import _driver

#: PocketFlow's nine ligand elements: C, N, O, F, P, S, Cl, Br, I. Fixed by the
#: checkpoint, not configurable — ``main_generate.py`` lines 62-63 pass this exact
#: list to both the ligand featurizer and the focal masker, and the atom flow's
#: output dimension is ``len(atom_type_map)``.
_ATOM_TYPE_MAP = [6, 7, 8, 9, 15, 16, 17, 35, 53]

#: Above this the input is a whole protein rather than a cut pocket. PocketFlow
#: places its first atom relative to the *protein* surface atoms it was given
#: (``pocket_flow/generate.py`` lines 66-74), so a full structure does not error —
#: it just anchors generation somewhere arbitrary.
_POCKET_ATOM_WARN = 3000


def _preflight(device):
    """Refuse, with the cause named, before any upstream import can obscure it.

    Each check corresponds to a numbered blocker in this module's docstring. The
    messages are the operator's only diagnostic when this driver is run by hand.
    """
    import importlib.util

    if not str(device).startswith("cuda"):
        raise RuntimeError(
            f"PocketFlow requires a CUDA device; got device={device!r}. "
            "pocket_flow/generate.py lines 246-253 call .cuda() unconditionally when "
            "building the atom and bond priors, so the sampler raises on CPU."
        )
    if importlib.util.find_spec("tensorboard") is None:
        raise RuntimeError(
            "PocketFlow cannot be imported: pocket_flow/utils/train.py line 11 does "
            "`from torch.utils import tensorboard` at module scope and pocket_flow/__init__.py "
            "line 1 reaches it, but tensorboard is not installed in this interpreter."
        )
    for note in _apply_pyg_compat_shims():
        print(f"[pocketflow] {note}", file=sys.stderr)


def _apply_pyg_compat_shims():
    """Restore the two PyTorch Geometric spellings PocketFlow was written against.

    Upstream's ``environment.yml`` pins ``torch-geometric==2.3.1`` and a
    ``linux_x86_64``-only ``pyg_lib`` wheel. Neither is usable here: this host is
    aarch64, and pyg-lib publishes no aarch64 build at any version. Two things
    PocketFlow imports moved in later PyG and both are on the sampling path.

    1. ``pocket_flow/utils/transform_utils.py:5`` does ``from
       torch_geometric.utils.subgraph import subgraph``. PyG 2.5 made that module
       private (``_subgraph``) and kept re-exporting the *function* from
       ``torch_geometric.utils``. Aliasing the old module name onto the same
       function is exact, not approximate.
    2. ``pocket_flow/generate.py:2`` and ``transform_utils.py:3,6`` import ``knn``,
       ``radius``, ``knn_graph`` and ``radius_graph``. Those used to be thin
       re-exports of ``torch_cluster``; PyG >= 2.7 raises ``ImportError: requires
       'pyg-lib>=0.6.0'`` for each instead. ``torch_cluster`` is compiled and
       working in this environment, on CPU and CUDA, so routing them back gives
       PocketFlow the functions it was trained and published with.

    This is the same shim ``pocket2mol_sample.py`` applies for the same reason.
    Each is applied only if the modern spelling actually fails, and every one
    applied is reported, so a run is never silently adapted.
    """
    import types

    import torch
    import torch_geometric.nn
    import torch_geometric.nn.pool
    import torch_geometric.utils as tg_utils

    notes = []

    try:
        import torch_geometric.utils.subgraph
    except ImportError:
        shim = types.ModuleType("torch_geometric.utils.subgraph")
        shim.subgraph = tg_utils.subgraph
        sys.modules["torch_geometric.utils.subgraph"] = shim
        notes.append(
            "re-created the module 'torch_geometric.utils.subgraph' as an alias for the "
            "function torch_geometric.utils.subgraph (later PyG made the module private "
            "and kept the function; the function itself is unchanged)"
        )

    try:
        torch_geometric.nn.pool.knn_graph(torch.zeros(2, 3), 1)
    except ImportError:
        from torch_cluster import knn, knn_graph, radius, radius_graph

        for mod in (torch_geometric.nn, torch_geometric.nn.pool):
            mod.knn = knn
            mod.knn_graph = knn_graph
            mod.radius = radius
            mod.radius_graph = radius_graph
        notes.append(
            "routed knn, knn_graph, radius and radius_graph back to torch_cluster, which "
            "is what they were in the PyG version PocketFlow was published against; "
            "PyG >= 2.7 routes them through pyg-lib, which has no aarch64 build"
        )

    return notes


def _build_pocket_data(pocket_pdb, torch, upstream):
    """Parse a pocket PDB into the transformed ``ComplexData`` the sampler expects.

    This is ``main_generate.py`` lines 53-74 unchanged in order and arguments. The
    ligand half is deliberately empty (``Ligand.empty_dict()``, line 54) because the
    ligand is what is being generated; ``mask_node`` with an empty index (line 72)
    is what marks every ligand atom as still-to-be-placed. Upstream's line 73,
    ``focal_masker.run(data)``, is commented out in the checkout, so the
    ``FocalMaker`` built at line 63 is never applied and is not built here either.
    """
    protein = upstream["Protein"](str(pocket_pdb))
    # get_surf=True is main_generate.py line 53. It does NOT invoke pymol: the
    # surface flag is read off each ATOM record's trailing `inner`/`surf` token by
    # ParseFile.py line 61, a non-standard column that upstream's own
    # SplitPocket._split_pocket_with_surface_atoms writes. A pocket without that
    # column parses fine and yields an all-false mask.
    pro_dict = protein.get_atom_dict(removeHs=True, get_surf=True)
    n_atoms = len(pro_dict["element"])
    if n_atoms == 0:
        raise ValueError(f"no ATOM records parsed from {pocket_pdb} — is it a PDB pocket file?")
    n_surf = int(pro_dict["surface_mask"].sum()) if "surface_mask" in pro_dict else 0

    lig_dict = upstream["Ligand"].empty_dict()  # line 54
    data = upstream["ComplexData"].from_protein_ligand_dicts(  # lines 55-58
        protein_dict=upstream["torchify_dict"](pro_dict),
        ligand_dict=upstream["torchify_dict"](lig_dict),
    )

    protein_featurizer = upstream["FeaturizeProteinAtom"]()  # line 61
    ligand_featurizer = upstream["FeaturizeLigandAtom"](atomic_numbers=_ATOM_TYPE_MAP)  # line 62
    atom_composer = upstream["AtomComposer"](  # line 64
        knn=16, num_workers=16, for_gen=True, use_protein_bond=True
    )

    data = upstream["RefineData"]()(data)  # line 67
    data = upstream["LigandCountNeighbors"]()(data)  # line 68
    data = protein_featurizer(data)  # line 69
    data = ligand_featurizer(data)  # line 70
    node4mask = torch.arange(data.ligand_pos.size(0))  # line 71
    data = upstream["mask_node"](  # line 72
        data, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.0
    )
    data = atom_composer.run(data)  # line 74
    return data, atom_composer, n_atoms, n_surf


def run(request):
    """Load the checkpoint, sample the pocket one molecule at a time, report counts."""
    device = str(request.get("device") or "cuda:0")
    _preflight(device)

    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    # The upstream import block, main_generate.py lines 3-5. The checkout root is
    # on PYTHONPATH and is the subprocess cwd, so `pocket_flow` resolves there.
    from pocket_flow import Generate, PocketFlow
    from pocket_flow.utils import (
        AtomComposer,
        ComplexData,
        FeaturizeLigandAtom,
        FeaturizeProteinAtom,
        Ligand,
        LigandCountNeighbors,
        Protein,
        RefineData,
        mask_node,
        torchify_dict,
    )

    upstream = {
        "AtomComposer": AtomComposer,
        "ComplexData": ComplexData,
        "FeaturizeLigandAtom": FeaturizeLigandAtom,
        "FeaturizeProteinAtom": FeaturizeProteinAtom,
        "Ligand": Ligand,
        "LigandCountNeighbors": LigandCountNeighbors,
        "Protein": Protein,
        "RefineData": RefineData,
        "mask_node": mask_node,
        "torchify_dict": torchify_dict,
    }

    warnings = []
    pocket_pdb = Path(str(request["pocket_pdb"]))
    if not pocket_pdb.is_file():
        raise FileNotFoundError(f"pocket PDB not found: {pocket_pdb}")
    checkpoint = Path(str(request["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PocketFlow checkpoint not found: {checkpoint}")

    seed = int(request.get("seed", 0))
    # Upstream seeds nothing — main_generate.py has no manual_seed and Generate
    # draws from the global RNG. Seeding here makes a run reproducible without
    # touching the algorithm.
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    data, atom_composer, n_pocket_atoms, n_surface_atoms = _build_pocket_data(pocket_pdb, torch, upstream)
    if n_pocket_atoms > _POCKET_ATOM_WARN:
        warnings.append(
            f"{n_pocket_atoms} atoms in {pocket_pdb.name}: this looks like a whole protein "
            "rather than a cut pocket, and PocketFlow will anchor generation arbitrarily."
        )
    if n_surface_atoms == 0:
        warnings.append(
            f"{pocket_pdb.name} carries no `surf` column, so every atom is marked buried. "
            "Upstream's own pockets (test_samples/test_pocket10/*-surf.pdb) carry it, written by "
            "SplitPocket._split_pocket_with_surface_atoms. With choose_max=True the focal atom is "
            "the argmax and the mask is unused; with choose_max=False generation cannot start."
        )

    # main_generate.py lines 79-83. weights_only=False is required: the checkpoint
    # stores an easydict.EasyDict under 'config' alongside the tensors, and line 81
    # reads it to build the model. torch >= 2.6 defaults weights_only=True and
    # refuses to unpickle it.
    try:
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        ckpt = torch.load(str(checkpoint), map_location=device)
    model = PocketFlow(ckpt["config"]).to(device)  # line 82
    model.load_state_dict(ckpt["model"])  # line 83

    bond_range = request.get("bond_length_range") or [1.0, 2.0]
    generator = Generate(  # lines 89-92
        model,
        atom_composer.run,
        temperature=[
            float(request.get("atom_temperature", 1.0)),
            float(request.get("bond_temperature", 1.0)),
        ],
        atom_type_map=_ATOM_TYPE_MAP,
        num_bond_type=4,
        max_atom_num=int(request.get("max_atom_num", 35)),
        focus_threshold=float(request.get("focus_threshold", 0.5)),
        max_double_in_6ring=int(request.get("max_double_in_6ring", 0)),
        min_dist_inter_mol=float(request.get("min_dist_inter_mol", 3.0)),
        bond_length_range=(float(bond_range[0]), float(bond_range[1])),
        choose_max=bool(request.get("choose_max", True)),
        device=device,
    )

    n_requested = int(request.get("n_molecules", 10))
    molecules = []
    n_failures = 0
    for sample_idx in range(n_requested):
        # Generate.generate's own per-sample body, pocket_flow/generate.py lines
        # 357-358. run() returns (modified_mol, raw_mol) or None.
        try:
            out = generator.run(data.clone().detach())
        # Broad on purpose: one bad sample must not end the batch, and the
        # sampler reaches CUDA, RDKit and torch_cluster, each with its own types.
        except Exception as exc:
            n_failures += 1
            if len(warnings) < 40:
                warnings.append(f"sample {sample_idx} raised {type(exc).__name__}: {exc}")
            molecules.append(
                {"smiles": "", "sdf": None, "sample_index": sample_idx, "qc_flags": ["sampler_raised"]}
            )
            continue

        mol = out[0] if out else None
        if mol is None:
            # Upstream's own failure mode: run() caught a MolReconsError from
            # data2mol and returned None (generate.py lines 342-346), or the focal
            # search terminated with an empty molecule.
            n_failures += 1
            molecules.append(
                {
                    "smiles": "",
                    "sdf": None,
                    "sample_index": sample_idx,
                    "qc_flags": ["generation_failed"],
                }
            )
            continue

        smiles = Chem.MolToSmiles(mol)
        flags = []
        # Kept and flagged rather than dropped: a molecule that came back as two
        # disconnected pieces is a measurement of the sampler, and deleting it
        # would make the completeness rate unrecoverable downstream.
        if "." in smiles:
            flags.append("fragmented_multi_component")
        molecules.append(
            {
                "smiles": smiles,
                "sdf": Chem.MolToMolBlock(mol),
                "sample_index": sample_idx,
                "n_atoms": int(mol.GetNumAtoms()),
                "qc_flags": flags,
            }
        )

    return {
        "molecules": molecules,
        "n_requested": n_requested,
        "n_sampled": len(molecules) - n_failures,
        "n_generation_failures": n_failures,
        "n_pocket_atoms": n_pocket_atoms,
        "n_surface_atoms": n_surface_atoms,
        "device": device,
        "seed": seed,
        "checkpoint": checkpoint.name,
        "max_atom_num": int(request.get("max_atom_num", 35)),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
