"""Sample ligands for one pocket with TargetDiff — the drug-agent ⇄ TargetDiff bridge.

This file is executed inside whichever interpreter carries TargetDiff's
dependencies. It imports nothing from ``drug_agent``: the two sides talk over a
JSON request/response pair on disk, so a CUDA fault or an OOM inside the diffusion
sampler cannot reach the caller.

**Why not call ``scripts/sample_diffusion.py``.** That script is the test-set
entry point. Its ``__main__`` block calls ``get_dataset(...)`` and then indexes
``test_set[args.data_id]``, so it needs the preprocessed
``crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`` and can only sample
pockets already in the CrossDocked split. Drug discovery on a new target needs the
opposite: one arbitrary pocket PDB and no dataset. Upstream's own answer to that is
``scripts/sample_for_pocket.py``, and this driver reproduces its body call for
call, replacing only the argparse front-end and the ``torch.save`` back-end.

The sampler itself is still upstream's: ``sample_diffusion_ligand`` is imported
from ``scripts/sample_diffusion.py`` rather than reimplemented, which is also why
``lmdb`` is a hard runtime dependency of this path — importing that module runs
``from datasets import get_dataset`` at line 16, and ``datasets/__init__.py``
imports ``pl_pair_dataset``, which imports ``lmdb`` at module scope.

Every upstream call below carries the file and line it was taken from, at
``targetdiff`` commit 142f1eb71784.

Request JSON:
    pocket_pdb      path to a prepared POCKET pdb (a 10 A region, not a protein)
    checkpoint      path to pretrained_diffusion.pt
    config          path to configs/sampling.yml
    n_molecules     total molecules to sample
    batch_size      molecules per sampler call
    seed            RNG seed, overriding config.sample.seed
    device          e.g. "cuda:0" / "cpu"
    num_steps       optional override for config.sample.num_steps
    out_dir         directory to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]     one entry per *sample*, including the failures — a
                    reconstruction failure is reported as an empty ``smiles`` with
                    ``qc_flags=["reconstruction_failed"]`` so that the caller's
                    validity denominator is the number sampled, not the number
                    that happened to survive.
    n_requested, n_sampled, n_reconstruction_failures, n_fragmented,
    n_pocket_atoms, device, seed, ligand_atom_mode, num_steps, warnings[], errors[]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path bootstrap above, deliberately.
import _driver

#: Above this, the input is a whole protein rather than a cut pocket. TargetDiff
#: sizes the ligand from the pocket's own extent (``atom_num.get_space_size`` takes
#: the median of the ten largest pairwise distances), so a full structure silently
#: produces a meaningless atom count rather than an error.
_POCKET_ATOM_WARN = 3000


def _apply_pyg_compat_shims():
    """Route knn/radius back to torch_cluster, the way TargetDiff was published.

    TargetDiff calls ``knn_graph`` on its sampling path (``models/molopt_score_model.py``
    via ``torch_geometric.nn.pool``). Those functions used to be thin re-exports of
    ``torch_cluster``; PyG >= 2.7 routes them through ``pyg-lib`` and raises
    ``ImportError: 'knn_graph' requires 'pyg-lib>=0.6.0'`` instead.

    pyg-lib publishes no aarch64 build at any version, so on this architecture the
    reroute is not a convenience — it is the only way the model runs at all.
    ``torch_cluster`` is compiled and working here on CPU and CUDA, and it is what
    these functions were when TargetDiff was trained, so this restores the published
    behaviour rather than approximating it.

    The same shim is applied by ``pocket2mol_sample.py`` and ``pocketflow_sample.py``
    for the same reason. Applied only when the modern spelling actually fails, and
    reported when it is, so a run is never silently adapted.
    """
    import torch
    import torch_geometric.nn
    import torch_geometric.nn.pool

    try:
        torch_geometric.nn.pool.knn_graph(torch.zeros(2, 3), 1)
    except ImportError:
        from torch_cluster import knn, knn_graph, radius, radius_graph

        for mod in (torch_geometric.nn, torch_geometric.nn.pool):
            mod.knn = knn
            mod.knn_graph = knn_graph
            mod.radius = radius
            mod.radius_graph = radius_graph
        return [
            "routed knn, knn_graph, radius and radius_graph back to torch_cluster, which "
            "is what they were in the PyG version TargetDiff was published against; "
            "PyG >= 2.7 routes them through pyg-lib, which has no aarch64 build"
        ]
    return []


def _torch_load(torch, path, device):
    """``torch.load`` across the 2.6 ``weights_only`` default change.

    TargetDiff checkpoints are not plain tensor dicts: ``sample_for_pocket.py``
    line 50 reads ``ckpt['config']``, an ``easydict.EasyDict`` pickled with the
    weights, and both the model geometry and ``ligand_atom_mode`` are taken from
    it. Unpickling that needs the pre-2.6 behaviour.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=device)


def _build_pocket_data(pocket_pdb, torch, trans, ProteinLigandData, torchify_dict):
    """Parse a pocket PDB into the ``ProteinLigandData`` the sampler expects.

    This is ``pdb_to_pocket_data`` from ``scripts/sample_for_pocket.py`` lines
    19-33, verbatim in structure: ``PDBProtein(...).to_dict_atom()``, then the
    empty-ligand dict whose five keys and dtypes are what
    ``ProteinLigandData.from_protein_ligand_dicts`` requires, then the
    ``Compose([protein_featurizer])`` transform from lines 53-58. The ligand
    featurizer is deliberately *not* in the Compose: line 57 applies the protein
    featurizer alone, because the ligand is what is being generated.
    """
    from utils.data import PDBProtein

    protein = PDBProtein(str(pocket_pdb))
    pocket_dict = protein.to_dict_atom()
    n_atoms = len(protein.pos)
    if n_atoms == 0:
        raise ValueError(f"no ATOM records parsed from {pocket_pdb} — is it a PDB pocket file?")

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            "element": torch.empty([0], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0], dtype=torch.long),
        },
    )
    return data, n_atoms


def run(request):
    """Load the checkpoint, sample the pocket, rebuild molecules, report counts."""
    for note in _apply_pyg_compat_shims():
        print(f"[targetdiff] {note}", file=sys.stderr)
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger
    from torch_geometric.transforms import Compose

    RDLogger.DisableLog("rdApp.*")

    # The upstream import block, scripts/sample_for_pocket.py lines 6-15. The
    # checkout root is on PYTHONPATH and is the subprocess cwd, so `utils`,
    # `models`, `datasets` and `scripts` resolve as namespace packages — none of
    # them ships an __init__.py.
    import utils.misc as misc
    import utils.transforms as trans
    from datasets.pl_data import ProteinLigandData, torchify_dict
    from models.molopt_score_model import ScorePosNet3D
    from scripts.sample_diffusion import sample_diffusion_ligand
    from utils import reconstruct

    warnings = []

    device = request.get("device") or "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append("CUDA unavailable in the TargetDiff environment; falling back to CPU (very slow)")
        device = "cpu"

    # config: sample_for_pocket.py line 44. Carries model.checkpoint and the whole
    # `sample` block (seed, num_samples, num_steps, pos_only, center_pos_mode,
    # sample_num_atoms) — see configs/sampling.yml.
    config = misc.load_config(request["config"])
    seed = int(request.get("seed", config.sample.seed))
    misc.seed_all(seed)  # line 46

    checkpoint = request["checkpoint"]
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"TargetDiff checkpoint not found at {checkpoint}. The checkout ships code only; "
            "pretrained_diffusion.pt comes from the Google Drive folder linked under "
            "'Trained model checkpoint' in the upstream README."
        )
    ckpt = _torch_load(torch, checkpoint, device)  # line 49

    # Feature dimensions, lines 53-55. The protein featurizer's width is fixed by
    # the checkout; the ligand's depends on ligand_atom_mode, which is read off the
    # *checkpoint's* training config rather than assumed, because it decides how
    # many atom-type classes the diffusion head emits.
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    # Model, lines 61-66. `strict` mirrors upstream exactly: a sampling config that
    # names a train_config is a fine-tuned/property-guided variant whose head does
    # not match, and only then is a partial load accepted.
    model = ScorePosNet3D(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict="train_config" not in config.model)

    # Pocket, lines 69-70.
    data, n_pocket_atoms = _build_pocket_data(
        request["pocket_pdb"], torch, trans, ProteinLigandData, torchify_dict
    )
    data = Compose([protein_featurizer])(data)
    if n_pocket_atoms > _POCKET_ATOM_WARN:
        warnings.append(
            f"pocket has {n_pocket_atoms} atoms, which looks like a whole protein rather than a "
            "10 A pocket; TargetDiff sizes the ligand from the pocket extent, so the atom-count "
            "prior will be wrong. Cut a pocket first (scripts/data_preparation/extract_pockets.py)."
        )

    n_requested = int(request.get("n_molecules", 10))
    batch_size = max(1, int(request.get("batch_size", 10)))
    num_steps = request.get("num_steps") or config.sample.num_steps

    # Sampling, lines 75-81. Keyword names are upstream's own, defined at
    # scripts/sample_diffusion.py lines 31-33.
    sampled = sample_diffusion_ligand(
        model,
        data,
        n_requested,
        batch_size=batch_size,
        device=device,
        num_steps=num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
    )
    # sample_diffusion.py line 117 returns (pos, v, pos_traj, v_traj, v0_traj,
    # vt_traj, time_list). Only the endpoint is needed here; the four trajectory
    # lists hold num_steps x num_atoms arrays per sample and are dropped.
    all_pred_pos, all_pred_v = sampled[0], sampled[1]

    # Reconstruction, lines 97-114. This is the step that decides what fraction of
    # a diffusion run is usable at all, and it is a *bond perceiver over a point
    # cloud* (OpenBabel's connect_the_dots + PerceiveBondOrders), not a chemical
    # model. Upstream passes `pred_aromatic` positionally while leaving
    # `basic_mode` at its default True, which makes reconstruct_from_generated set
    # indicators=None and discard the aromaticity flags; that is what both
    # sample_for_pocket.py line 102 and evaluate_diffusion.py line 92 do, and it is
    # reproduced rather than corrected so these samples are comparable to the
    # numbers the paper reports.
    molecules = []
    n_recon_failures = 0
    n_fragmented = 0
    for sample_idx in range(len(all_pred_pos)):
        pred_pos = all_pred_pos[sample_idx]
        pred_v = all_pred_v[sample_idx]
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=ligand_atom_mode)  # line 99
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=ligand_atom_mode)  # line 101
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except Exception as exc:
            # Upstream catches only MolReconsError (line 104), but the perceiver
            # underneath is OpenBabel and RDKit sanitization, both of which raise
            # their own types. A reconstruction failure is data about the model,
            # not a malfunction of this driver, so all of them are counted.
            n_recon_failures += 1
            if n_recon_failures <= 3:
                warnings.append(f"reconstruction failure (sample {sample_idx}): {exc}")
            molecules.append(
                {
                    "smiles": "",
                    "sdf": None,
                    "sample_index": sample_idx,
                    "n_atoms": int(np.asarray(pred_pos).shape[0]),
                    "qc_flags": ["reconstruction_failed"],
                }
            )
            continue

        # Upstream drops fragmented output entirely (line 108). It is kept here,
        # flagged, because "the point cloud rebuilt into two disconnected pieces"
        # is a measurement of the reconstruction step and deleting it would make
        # the completeness rate unrecoverable downstream.
        flags = []
        if "." in smiles:
            n_fragmented += 1
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
        "n_sampled": len(all_pred_pos),
        "n_reconstruction_failures": n_recon_failures,
        "n_fragmented": n_fragmented,
        "n_pocket_atoms": n_pocket_atoms,
        "device": device,
        "seed": seed,
        "ligand_atom_mode": ligand_atom_mode,
        "num_steps": int(num_steps),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
