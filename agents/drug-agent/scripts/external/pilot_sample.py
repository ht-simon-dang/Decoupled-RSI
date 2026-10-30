"""Sample ligands for one pocket with PILOT — the drug-agent ⇄ e3moldiffusion bridge.

Runs inside whichever interpreter has PILOT's dependencies. It imports nothing from
``drug_agent`` and nothing outside the standard library except the upstream package
itself, which reaches it on ``PYTHONPATH`` (``ExternalBackend.child_env`` puts the
checkout root there, exactly as upstream's own launcher does with
``export PYTHONPATH=./pilot_folder``).

**Which upstream.** ``pfizer-opensource/e3moldiffusion`` at ``f5e9bb8a77b6``. The
family has variants; the argument names below are not portable across them.

**Where each call came from.** Every step mirrors ``evaluate()`` in
``experiments/generate_ligands_multi_batch.py``, which is the function
``scripts/generate_ligands_multi_pdb_file.sl`` invokes. What is dropped is that
script's SLURM job-array sharding, its qvina2 docking filter, its PoseBusters and
PoseCheck bookkeeping and its pickle dumps — none of which is sampling. What is kept
is the model construction and the sampling call, unchanged.

Why not shell out to ``generate_ligands_multi_batch.py``? It writes SDF and pickle
files into a save directory and prints its results, so the caller would have to parse
a directory tree. This driver calls the same two functions that script calls and
returns the molecules directly.

Usage (the adapter does this for you)::

    PYTHONPATH=$PILOT_ROOT python scripts/external/pilot_sample.py request.json

Request JSON:
    model_path       PILOT diffusion checkpoint (upstream's --model-path)
    dataset_root     processed dataset root, or null to take it from the checkpoint
                     (upstream's --dataset-root: "If not set it will be taken from
                     the model ckpt")
    test_dir         directory holding <pdb>.pdb, <stem>.sdf and <stem>.txt as
                     written by experiments/data/ligand/process_pdb.py
    ligand_stem      the <stem> in that directory to sample against
    n_molecules      how many raw samples to return
    batch_size       molecules per model call (upstream's --batch-size)
    max_sample_iter  how many sampling rounds before giving up (--max-sample-iter)
    seed             RNG seed; upstream exposes none, so this driver seeds torch,
                     numpy and random itself
    device           e.g. "cuda:0" / "cpu"
    guidance         importance-sampling settings, one key per generate_ligands
                     argument; {"objectives": []} means unguided
    out_dir          where result.json goes

Response JSON (``out_dir/result.json``):
    molecules[]     {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_requested, n_sampled, n_reconstruction_failures, guidance{}, warnings[], errors[]
"""

from __future__ import annotations

import sys
from pathlib import Path

# _driver sits next to this file, not on the upstream environment's path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver


class _DotDict(dict):
    """Attribute access over a dict, returning ``None`` for absent keys.

    Copied from ``generate_ligands_multi_batch.py``'s own ``dotdict``, because both
    ``hparams`` and ``args`` are consumed with attribute syntax throughout
    ``experiments/utils.py`` and a plain ``argparse.Namespace`` would raise on the
    handful of attributes the upstream code reads opportunistically.
    """

    __getattr__ = dict.get  # type: ignore[assignment]
    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]


def _load_hparams(torch, model_path, dataset_root):
    """The checkpoint's own hyper-parameters, with the overrides upstream applies.

    Source: ``evaluate()`` in ``experiments/generate_ligands_multi_batch.py``, the
    block from ``hparams = torch.load(args.model_path)["hyper_parameters"]`` down to
    ``hparams.gpus = 1``, plus the four backward-compatibility defaults it sets just
    before ``Trainer.load_from_checkpoint``.
    """
    hparams = torch.load(model_path, map_location="cpu")["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams["num_charge_classes"] = 6
    if dataset_root:
        hparams["dataset_root"] = dataset_root
    hparams = _DotDict(hparams)
    hparams.load_ckpt_from_pretrained = None
    hparams.store_intermediate_coords = False
    hparams.load_ckpt = None
    hparams.gpus = 1
    # Same four keys, same defaults, same reason: checkpoints predating these options
    # do not carry them and load_from_checkpoint passes them through positionally.
    for key in (
        "use_centroid_context_embed",
        "use_latent_encoder",
        "use_scaffold_latent_embed",
        "flow_matching",
    ):
        if key not in hparams:
            hparams[key] = False
    return hparams


def _datamodule_class(hparams):
    """The loader upstream selects for this checkpoint's dataset.

    Source: the ``if hparams.dataset == "crossdocked" / "kiba" / "kinodata"`` cascade
    in ``evaluate()``. The adaptive/non-adaptive split is upstream's, and so is the
    exception for an unrecognised dataset.
    """
    dataset = hparams.dataset
    if dataset in ("crossdocked", "kiba"):
        if hparams.use_adaptive_loader:
            from experiments.data.ligand.ligand_dataset_adaptive import (
                LigandPocketDataModule,
            )
        else:
            from experiments.data.ligand.ligand_dataset_nonadaptive import (
                LigandPocketDataModule,
            )
        return LigandPocketDataModule
    if dataset == "kinodata":
        from experiments.data.ligand.kino_dataset_nonadaptive import (
            LigandPocketDataModule,
        )

        return LigandPocketDataModule
    raise ValueError(
        f"PILOT checkpoint declares dataset={dataset!r}; upstream evaluate() supports "
        "only crossdocked, kiba and kinodata"
    )


def _property_distributions(datamodule, hparams):
    """``(prop_norm, prop_dist)`` for a context-conditioned checkpoint, else ``(None, None)``.

    Source: the ``prop_norm, prop_dist = None, None`` block in ``evaluate()``. The
    condition is reproduced clause for clause rather than tidied up — it decides
    whether the checkpoint's context embedding gets a normaliser, and getting it
    wrong silently changes what the model conditions on.
    """
    from experiments.data.distributions import DistributionProperty

    needs = (
        len(hparams.properties_list) > 0
        and hparams.context_mapping
        and not hparams.use_centroid_context_embed
    ) or (
        hparams.property_training and hparams.regression_property not in ("sa_score", "docking_score", "ic50")
    )
    if not needs:
        return None, None
    prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
    prop_dist = DistributionProperty(datamodule, hparams.properties_list)
    prop_dist.set_normalizer(prop_norm)
    return prop_norm, prop_dist


def _build_model(torch, np, request, hparams):
    """Load PILOT's Lightning module the way ``evaluate()`` loads it.

    Source: ``evaluate()``, from ``datamodule = DataModule(hparams)`` through
    ``model = model.eval()``. ``dataset_info``, ``histogram`` and ``smiles_list`` are
    all required constructor arguments of ``Trainer.load_from_checkpoint`` — this is
    why a processed dataset root is not optional.
    """
    from experiments.data.data_info import GeneralInfos as DataInfos

    datamodule = _datamodule_class(hparams)(hparams)
    dataset_info = DataInfos(datamodule, hparams)
    histogram_path = Path(hparams.dataset_root) / "size_distribution.npy"
    if not histogram_path.is_file():
        raise FileNotFoundError(
            f"PILOT needs the processed dataset's ligand-size histogram at "
            f"{histogram_path}; build it with "
            "experiments/data/ligand/process_crossdocked.py, or pass dataset_root"
        )
    histogram = np.load(histogram_path).tolist()
    test_smiles = list(datamodule.test_dataset.smiles)
    prop_norm, prop_dist = _property_distributions(datamodule, hparams)

    if hparams.additional_feats:
        # Upstream reaches for experiments/deprecated/ here. That package is not in
        # the pinned checkout, so such a checkpoint cannot be loaded from this clone.
        raise RuntimeError(
            "this PILOT checkpoint sets additional_feats, which upstream loads from "
            "experiments.deprecated.diffusion_discrete_pocket_addfeats; that package "
            "is absent from e3moldiffusion at f5e9bb8a77b6"
        )
    from experiments.diffusion_discrete_pocket import Trainer

    device = request.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"PILOT was asked for device {device!r} but torch.cuda.is_available() is False")
    model = Trainer.load_from_checkpoint(
        request["model_path"],
        dataset_info=dataset_info,
        smiles_list=test_smiles,
        histogram=histogram,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        use_centroid_context_embed=hparams.use_centroid_context_embed,
        use_latent_encoder=hparams.use_latent_encoder,
        use_scaffold_latent_embed=hparams.use_scaffold_latent_embed,
        flow_matching=hparams.flow_matching,
        load_ckpt=None,
        run_evaluation=True,
        strict=False,
    ).to(device)
    return model.eval(), dataset_info, device


def _pocket_residues(test_dir, ligand_stem):
    """The Biopython residues ``prepare_pocket_cutoff`` featurises, plus the ref ligand.

    Source: the per-``sdf_file`` block in ``evaluate()`` — the PDB name is the ligand
    stem up to the first ``_``, the residue ids come from ``<stem>.txt`` split on
    whitespace, and each id is ``chain:resnum`` indexed into a ``PDBParser`` model.
    """
    from Bio.PDB import PDBParser

    sdf_file = Path(test_dir) / f"{ligand_stem}.sdf"
    pdb_name = ligand_stem.split("_")[0]
    pdb_file = Path(test_dir) / f"{pdb_name}.pdb"
    txt_file = Path(test_dir) / f"{ligand_stem}.txt"
    for path in (sdf_file, pdb_file, txt_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"PILOT needs {path}; process_pdb.py writes the .pdb, .sdf and .txt triple together"
            )
    resi_list = txt_file.read_text().split()
    if not resi_list:
        raise ValueError(f"{txt_file} is empty; it must list the pocket residues as chain:resnum")
    structure = PDBParser(QUIET=True).get_structure("", str(pdb_file))[0]
    residues = [structure[rid.split(":")[0]][(" ", int(rid.split(":")[1]), " ")] for rid in resi_list]
    return residues, sdf_file


def _sampling_args(request, guidance):
    """The ``args`` namespace ``prepare_data_and_generate_ligands`` reads.

    Every key is an argument of ``get_args()`` in
    ``experiments/generate_ligands_multi_batch.py``, spelled with underscores as
    argparse stores it, and every value that is not taken from the request is that
    parser's own default. The guidance keys arrive already named after upstream's
    flags, so nothing is translated here.
    """
    args = _DotDict(
        {
            # sampling shape
            "batch_size": int(request.get("batch_size") or 10),
            "prior_n_atoms": request.get("prior_n_atoms") or "conditional",
            "n_nodes_bias": int(request.get("n_nodes_bias") or 0),
            "vary_n_nodes": bool(request.get("vary_n_nodes", False)),
            "fix_n_nodes": bool(request.get("fix_n_nodes", False)),
            # reverse process
            "ddim": bool(request.get("ddim", False)),
            "eta_ddim": float(request.get("eta_ddim") or 1.0),
            # reconstruction: upstream's parser defaults, both False. build_obabel_mol
            # rebuilds bonds from coordinates alone and is the noisier of the two.
            "build_obabel_mol": bool(request.get("build_obabel_mol", False)),
            "sanitize": bool(request.get("sanitize", False)),
            "relax_mol": bool(request.get("relax_mol", False)),
            "max_relax_iter": int(request.get("max_relax_iter") or 200),
            # features this driver does not use, pinned to their parser defaults
            "encode_ligands": False,
            "use_lipinski_context": False,
            "inpainting": False,
            "keep_ids": None,
            "context_fixed": None,
            "latent_gamma": 1.0,
            "emd_ot": False,
            "clash_guidance": False,
            "clash_guidance_scale": 0.1,
            "clash_guidance_start": None,
            "clash_guidance_end": None,
            "importance_gradient_guidance": False,
            "ckpts_ensemble": [],
            # classifier guidance is a separate mechanism from importance sampling
            # and this adapter does not expose it; parser defaults throughout.
            "property_classifier_guidance": False,
            "property_classifier_guidance_complex": False,
            "property_classifier_self_guidance": False,
            "classifier_guidance_scale": 1e-1,
            "classifier_guidance_kind": None,
            "classifier_guidance_period": None,
            # importance sampling off, at get_args()'s own defaults. The guidance
            # dict below overwrites exactly these keys when guidance was requested,
            # so an unguided call cannot leave a stale flag set.
            "sa_importance_sampling": False,
            "ckpt_sa_model": None,
            "sa_tau": 0.1,
            "sa_every_importance_t": 5,
            "sa_importance_sampling_start": None,
            "sa_importance_sampling_end": None,
            "property_importance_sampling": False,
            "ckpt_property_model": None,
            "property_tau": 0.1,
            "property_every_importance_t": 5,
            "property_importance_sampling_start": None,
            "property_importance_sampling_end": None,
            "minimize_property": False,
            "joint_importance_sampling": False,
            "property_normalization": False,
        }
    )
    args.update(guidance)
    return args


def _records_from_molecules(Chem, molecules, offset):
    """Turn upstream ``Molecule`` objects into the adapter's record dicts.

    ``Molecule.rdkit_mol`` is the *reconstructed* graph — ``build_molecule()`` from
    the predicted bond matrix, or ``build_molecule_openbabel()`` from coordinates.
    Nothing here repairs it: a molecule that cannot be written to SMILES is counted
    as a reconstruction failure and reported, because that count is the number a
    reader needs to judge the batch.
    """
    records = []
    failures = 0
    for i, molecule in enumerate(molecules):
        rdmol = getattr(molecule, "rdkit_mol", None)
        if rdmol is None:
            failures += 1
            continue
        try:
            smiles = Chem.MolToSmiles(rdmol)
            sdf = Chem.MolToMolBlock(rdmol)
        except Exception:
            # RDKit raises bare exceptions on a graph the reconstruction broke.
            failures += 1
            continue
        if not smiles:
            failures += 1
            continue
        qc_flags = []
        if "." in smiles:
            qc_flags.append("multi_fragment")
        records.append(
            {
                "smiles": smiles,
                "sdf": sdf,
                "sample_index": offset + i,
                "n_atoms": int(rdmol.GetNumAtoms()),
                "qc_flags": qc_flags,
            }
        )
    return records, failures


def run(request):
    """Load PILOT, sample one pocket, and return the raw records."""
    import random

    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    seed = int(request.get("seed") or 0)
    # Upstream exposes no seed for sampling, so reproducibility is this driver's job.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    guidance = dict(request.get("guidance") or {})
    objectives = list(guidance.pop("objectives", []) or [])
    guidance.pop("mode", None)
    warnings = []
    if objectives:
        if guidance.get("sa_importance_sampling") and not guidance.get("ckpt_sa_model"):
            # reverse_sampling() falls back to self.model when sa_model is None, so
            # this is a real change in what did the guiding, not a no-op.
            warnings.append(
                "sa guidance requested with no ckpt_sa_model: the diffusion model's "
                "own property head will score the trajectory"
            )
        if guidance.get("property_importance_sampling") and not guidance.get("ckpt_property_model"):
            warnings.append(
                "property guidance requested with no ckpt_property_model: the "
                "diffusion model's own property head will score the trajectory"
            )

    hparams = _load_hparams(torch, request["model_path"], request.get("dataset_root"))
    model, dataset_info, device = _build_model(torch, np, request, hparams)
    residues, sdf_file = _pocket_residues(request["test_dir"], request["ligand_stem"])
    args = _sampling_args(request, guidance)

    from experiments.utils import prepare_data_and_generate_ligands

    wanted = int(request.get("n_molecules") or 0)
    max_iter = int(request.get("max_sample_iter") or 20)
    records = []
    failures = 0
    rounds = 0
    while len(records) < wanted and rounds < max_iter:
        rounds += 1
        molecules = prepare_data_and_generate_ligands(
            model,
            residues,
            sdf_file,
            dataset_info,
            hparams=hparams,
            args=args,
            device=device,
            embedding_dict=None,
            residues_10A=None,
        )
        if not molecules:
            warnings.append(f"sampling round {rounds} returned no molecules")
            continue
        batch_records, batch_failures = _records_from_molecules(Chem, molecules, len(records))
        records.extend(batch_records)
        failures += batch_failures

    if len(records) < wanted:
        warnings.append(f"reached {rounds} sampling rounds with {len(records)} of {wanted} molecules")

    return {
        "molecules": records[:wanted],
        "n_requested": wanted,
        "n_sampled": len(records),
        "n_reconstruction_failures": failures,
        "n_rounds": rounds,
        "device": device,
        "guidance": {"objectives": objectives, "guided": bool(objectives)},
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
