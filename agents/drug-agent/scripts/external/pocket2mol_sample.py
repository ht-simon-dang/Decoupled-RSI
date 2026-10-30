"""Sample ligands inside one pocket with Pocket2Mol — the drug-agent ⇄ Pocket2Mol bridge.

This file is executed in whichever interpreter has Pocket2Mol's dependencies. It
imports nothing from ``drug_agent``: the two sides talk over a JSON request /
response pair on disk, so a CUDA fault inside an autoregressive beam search
cannot reach the caller. Kept syntactically conservative (no ``zip(strict=)``, no
walrus in the upstream-mirroring loop) because an operator may point
``DRUGAGENT_POCKET2MOL_PYTHON`` at the Python 3.8 environment
``.external/Pocket2Mol/env_cuda113.yml`` builds.

**Which upstream entry point this is.** ``sample.py`` iterates the CrossDocked
test *split* by integer ``--data_id`` and therefore needs the preprocessed
dataset; it cannot sample a new target. ``sample_for_pdb.py`` is the entry point
for an arbitrary pocket, and its ``__main__`` block is the algorithm reproduced
below, call for call. Every upstream symbol used here is imported from the
checkout rather than reimplemented, and each argument below names the upstream
file and line it came from.

Request JSON:
    checkout        Pocket2Mol checkout root (also on PYTHONPATH via the adapter)
    pocket_pdb      path to a prepared POCKET pdb
    center          [x, y, z] centre of the sampling box, or null for the
                    centroid of the pocket file's ATOM/HETATM records
    bbox_size       side length of the cubic box in Angstrom
                    (``sample_for_pdb.py`` argparse default: 23.0)
    config          path to configs/sample_for_pdb.yml
    checkpoint      path to the trained checkpoint (config model.checkpoint)
    n_molecules     how many finished molecules to stop at
    beam_size       beam width override, else config.sample.beam_size
    max_steps       step cap override, else config.sample.max_steps
    seed            RNG seed, else config.sample.seed
    device          e.g. "cuda:0" / "cpu"
    out_dir         directory to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]     {smiles, sdf, sample_index, n_atoms, qc_flags[]} — one entry
                    per *terminal* beam sample, including the ones that failed
                    reconstruction (empty ``smiles``, ``reconstruction_failed``)
                    and the ones upstream rejects as multi-fragment. Emitting the
                    failures is what makes the caller's ``validity`` a real
                    fraction instead of 1.0 by construction.
    n_finished, n_duplicate, n_fragmented, n_reconstruction_failures
    n_terminal, n_steps, n_pocket_atoms, center, bbox_size, device, seed
    compat_notes[]  version shims that had to be applied to import upstream
    warnings[], errors[]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: How many per-sample reconstruction failures are echoed into ``warnings``.
#: The full count is always reported; only the messages are capped.
_MAX_REPORTED_FAILURES = 3


def _apply_pyg_compat_shims() -> list[str]:
    """Restore the two PyTorch Geometric spellings Pocket2Mol was written against.

    Pocket2Mol is pinned to PyG 2.0 (``README.md``, "Dependency" table). Two
    things it imports moved in later PyG and both are on the sampling path, so
    without these the checkout cannot even be imported here:

    1. ``utils/transforms.py:16`` does ``from torch_geometric.utils.subgraph
       import subgraph``. PyG made that module private (``_subgraph``) and
       re-exports the *function* from ``torch_geometric.utils``. Aliasing the old
       module name onto the same function is exact, not approximate.
    2. ``utils/transforms.py:14,17`` and ``models/common.py:7`` import ``knn``,
       ``knn_graph`` and ``radius`` from ``torch_geometric.nn``. Those used to be
       thin re-exports of ``torch_cluster``; PyG >= 2.7 raises
       ``ImportError: 'knn_graph' requires 'pyg-lib>=0.6.0'`` at call time
       instead. Routing them back to ``torch_cluster`` restores the functions
       Pocket2Mol was trained and published with.

    Each shim is applied only if the modern spelling actually fails, and every
    one applied is reported to the caller so a run is never silently adapted.
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
        shim.subgraph = tg_utils.subgraph  # type: ignore[attr-defined]
        sys.modules["torch_geometric.utils.subgraph"] = shim
        notes.append(
            "re-created the module 'torch_geometric.utils.subgraph' as an alias for the "
            "function torch_geometric.utils.subgraph (later PyG made the module private "
            "and kept the function; the function itself is unchanged)"
        )

    try:
        torch_geometric.nn.pool.knn_graph(torch.zeros(2, 3), 1)
    except ImportError:
        from torch_cluster import knn, knn_graph, radius

        for module in (torch_geometric.nn, torch_geometric.nn.pool):
            module.knn = knn
            module.knn_graph = knn_graph
            module.radius = radius
        notes.append(
            "routed torch_geometric.nn knn/knn_graph/radius back to torch_cluster "
            "(this PyG requires pyg-lib for them; torch_cluster is what PyG 2.0 called)"
        )
    return notes


def _pocket_centroid(pocket_pdb: str) -> list[float]:
    """Centroid of the pocket file's ATOM/HETATM records, in PDB column order.

    ``sample_for_pdb.py`` takes ``--center`` as a required piece of chemistry the
    operator supplies. When a caller has a *cut* pocket file and no centre, the
    centroid of that file is the box centre the file itself implies; deriving it
    here keeps the fallback in one auditable place rather than in the adapter.
    """
    xs = []
    ys = []
    zs = []
    with open(pocket_pdb) as handle:  # noqa: PTH123 - Path.open needs 3.6+, kept plain for 3.8 parity
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                # PDB fixed columns 31-38 / 39-46 / 47-54, per the wwPDB format.
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
    if not xs:
        raise ValueError(f"no ATOM/HETATM records in {pocket_pdb} — is it a PDB pocket file?")
    n = float(len(xs))
    return [sum(xs) / n, sum(ys) / n, sum(zs) / n]


def _torch_load(torch, path: str, device: str):
    """``torch.load`` that survives the torch 2.6 ``weights_only=True`` default.

    Pocket2Mol's checkpoint is not a bare state dict: ``sample_for_pdb.py:117``
    reads ``ckpt['config'].model`` — an ``easydict.EasyDict`` — to build the
    network before ``ckpt['model']`` is loaded into it. Unpickling that needs the
    pre-2.6 behaviour.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=device)


def _record(sample_index: int, smiles: str, mol, chem, qc_flags: list[str]) -> dict:
    """One terminal beam sample, in the shape the adapter parses."""
    block = ""
    n_atoms = 0
    if mol is not None:
        n_atoms = int(mol.GetNumAtoms())
        try:
            block = chem.MolToMolBlock(mol)
        except Exception:  # a molblock failure is not a sampling failure
            qc_flags = [*qc_flags, "molblock_unwritable"]
    return {
        "smiles": smiles,
        "sdf": block,
        "sample_index": sample_index,
        "n_atoms": n_atoms,
        "qc_flags": qc_flags,
    }


def run(request: dict) -> dict:
    """Run Pocket2Mol's ``sample_for_pdb`` algorithm over one pocket."""
    compat_notes = _apply_pyg_compat_shims()

    checkout = request.get("checkout")
    if checkout and checkout not in sys.path:
        sys.path.insert(0, checkout)

    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    # Upstream symbols, all from the checkout. `sample_for_pdb.py:11` does
    # `from sample import *`, so `sample.py`'s star-exports (get_init, get_next,
    # logp_to_rank_prob, STATUS_*, the transforms, MolReconsError,
    # reconstruct_from_generated_with_edges, load_config, seed_all) arrive with it.
    from sample import (
        STATUS_FINISHED,
        STATUS_RUNNING,
        AtomComposer,
        ContrastiveSample,
        FeaturizeLigandAtom,
        FeaturizeProteinAtom,
        LigandCountNeighbors,
        LigandMaskAll,
        MaskFillModelVN,
        RefineData,
        get_init,
        get_next,
        load_config,
        logp_to_rank_prob,
        reconstruct_from_generated_with_edges,
        seed_all,
        transform_data,
    )
    from sample_for_pdb import pdb_to_pocket_data
    from torch_geometric.transforms import Compose

    warnings = []
    device = request.get("device") or "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append("CUDA unavailable in the Pocket2Mol environment; falling back to CPU (slow)")
        device = "cpu"

    # `sample_for_pdb.py:81,83` — load_config, then seed_all(config.sample.seed).
    config = load_config(request["config"])
    seed = int(request.get("seed") or config.sample.seed)
    seed_all(seed)

    checkpoint = request.get("checkpoint") or config.model.checkpoint
    n_molecules = int(request.get("n_molecules") or config.sample.num_samples)
    beam_size = int(request.get("beam_size") or config.sample.beam_size)
    max_steps = int(request.get("max_steps") or config.sample.max_steps)
    threshold = config.sample.threshold

    # `sample_for_pdb.py:98-108` — the featurizers and the transform pipeline.
    # `ContrastiveSample(num_real=0, num_fake=0)` is constructed only for its
    # `num_elements` (7: C,N,O,F,P,S,Cl), which is the model's class count.
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    contrastive_sampler = ContrastiveSample(num_real=0, num_fake=0)
    transform = Compose(
        [
            RefineData(),
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            LigandMaskAll(),
        ]
    )

    # `sample_for_pdb.py:110` — pdb_to_pocket_data(pdb_path, center, bbox_size).
    pocket_pdb = request["pocket_pdb"]
    center = request.get("center") or _pocket_centroid(pocket_pdb)
    bbox_size = float(request.get("bbox_size") or 23.0)
    data = pdb_to_pocket_data(pocket_pdb, center, bbox_size)
    n_pocket_atoms = int(data.protein_pos.size(0))
    data = transform(data)

    # `sample_for_pdb.py:115-123` — the architecture comes out of the checkpoint,
    # so a missing checkpoint is not only missing weights: the network cannot be
    # constructed at all.
    ckpt = _torch_load(torch, checkpoint, device)
    model = MaskFillModelVN(
        ckpt["config"].model,
        num_classes=contrastive_sampler.num_elements,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        num_bond_types=3,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    # `sample_for_pdb.py:138-139` — AtomComposer's knn comes from the trained
    # encoder, never from a local default.
    pool_finished = []
    pool_smiles = set()
    records = []
    n_duplicate = 0
    n_fragmented = 0
    n_failures = 0
    sample_index = 0

    atom_composer = AtomComposer(
        protein_featurizer.feature_dim,
        ligand_featurizer.feature_dim,
        model.config.encoder.knn,
    )
    data = transform_data(data, atom_composer)
    queue = get_init(
        data.to(device),
        model=model,
        transform=atom_composer,
        threshold=threshold,
    )
    if len(queue) > beam_size:
        queue = queue[:beam_size]

    # `sample_for_pdb.py:162-210` — the beam loop, reproduced call for call. The
    # one addition is the empty-queue break: upstream's
    # `logp_to_rank_prob(np.array([...]))` divides by a zero-length sum when every
    # beam terminates in the same step, which is a crash rather than a result.
    n_steps = 0
    while len(pool_finished) < n_molecules and n_steps < max_steps and queue:
        n_steps += 1
        queue_tmp = []
        queue_weight = []
        for item in queue:
            nexts = []
            data_next_list = get_next(
                item.to(device),
                model=model,
                transform=atom_composer,
                threshold=threshold,
            )
            for data_next in data_next_list:
                if data_next.status == STATUS_FINISHED:
                    index = sample_index
                    sample_index += 1
                    try:
                        rdmol = reconstruct_from_generated_with_edges(data_next)
                        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(rdmol)))
                    # Upstream raises its own `MolReconsError` here; the broad
                    # clause also catches the RDKit sanitization errors that escape
                    # from underneath it. One unbuildable point cloud is a datum
                    # about the model, not a reason to lose the whole beam.
                    except Exception as exc:
                        n_failures += 1
                        if n_failures <= _MAX_REPORTED_FAILURES:
                            warnings.append(f"reconstruction failure (sample {index}): {exc}")
                        records.append(_record(index, "", None, Chem, ["reconstruction_failed"]))
                        continue
                    if smiles in pool_smiles:
                        # Upstream discards these; they are still real samples and
                        # counting them is what keeps uniqueness honest.
                        n_duplicate += 1
                        records.append(_record(index, smiles, rdmol, Chem, []))
                    elif "." in smiles:
                        # `sample_for_pdb.py:190` rejects multi-fragment output.
                        n_fragmented += 1
                        records.append(_record(index, smiles, rdmol, Chem, ["fragmented_multi_component"]))
                    else:
                        pool_finished.append(data_next)
                        pool_smiles.add(smiles)
                        records.append(_record(index, smiles, rdmol, Chem, []))
                elif data_next.status == STATUS_RUNNING:
                    nexts.append(data_next)

            queue_tmp += nexts
            if len(nexts) > 0:
                queue_weight += [1.0 / len(nexts)] * len(nexts)

        if not queue_tmp:
            break
        # `sample_for_pdb.py:207-209` — rank by summed log-probability, then draw
        # the next beam without replacement.
        prob = logp_to_rank_prob(np.array([p.average_logp[2:] for p in queue_tmp]), queue_weight)
        n_tmp = len(queue_tmp)
        next_idx = np.random.choice(np.arange(n_tmp), p=prob, size=min(beam_size, n_tmp), replace=False)
        queue = [queue_tmp[idx] for idx in next_idx]

    if not records:
        warnings.append(
            f"Pocket2Mol terminated with no molecules: the beam emptied after {n_steps} "
            "step(s). Check that the box centre and size actually cover the pocket."
        )

    return {
        "molecules": records,
        "n_requested": n_molecules,
        "n_terminal": len(records),
        "n_finished": len(pool_finished),
        "n_duplicate": n_duplicate,
        "n_fragmented": n_fragmented,
        "n_reconstruction_failures": n_failures,
        "n_steps": n_steps,
        "n_pocket_atoms": n_pocket_atoms,
        "center": [float(v) for v in center],
        "bbox_size": bbox_size,
        "beam_size": beam_size,
        "device": device,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "compat_notes": compat_notes,
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
