"""Sample ligands for a single pocket with PFM — the drug-agent ⇄ PFM bridge.

This file is executed in whichever interpreter has PFM's dependencies — usually
drug-agent's own (they co-install fine), occasionally a separate environment with
a different CUDA build. It is deliberately dependency-free with respect to
``drug_agent``: the two sides talk over a JSON request/response pair on disk, so
either arrangement works and a CUDA fault here cannot reach the caller.

Kept deliberately conservative (no ``zip(strict=)``, percent-formatting) so it
still runs under the older interpreters some PFM installs pin.

Why not call ``scripts/sample_fm.py`` directly? That script iterates the
CrossDocked test *split*, so it needs the full preprocessed dataset and can only
sample pockets that are already in it. Drug discovery on a new target needs the
opposite: one arbitrary pocket PDB, no dataset. So this driver reuses PFM's own
model, prior, sampler (``sample_ode``) and reconstruction code, and only replaces
the dataset front-end with a direct ``PDBProtein`` parse.

Usage (the adapter does this for you)::

    PYTHONPATH=$PFM_ROOT:$PFM_ROOT/scripts \\
        python scripts/pfm_sample_pocket.py request.json

Request JSON:
    pocket_pdb      path to a prepared POCKET pdb (not a whole protein)
    n_molecules     total molecules to sample
    batch_size      molecules per sampler call
    seed            RNG seed
    device          e.g. "cuda:0" / "cpu"
    checkpoint      PFM checkpoint (model_params/pfm.pth)
    training_yml    PFM training config (model_params/training.yml)
    x_predictor     optional override for the perturbation predictors
    h_predictor     optional override
    out_dir         directory to write result.json (+ molecules.sdf)

Response JSON (``out_dir/result.json``):
    molecules[]     {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_requested, n_sampled, n_reconstruction_failures, warnings[], errors[]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _fail(out_dir, message, detail=""):
    """Write a machine-readable failure and exit non-zero."""
    payload = {
        "molecules": [],
        "n_requested": 0,
        "n_sampled": 0,
        "n_reconstruction_failures": 0,
        "warnings": [],
        "errors": [message] + ([detail] if detail else []),
    }
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "result.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError:
        pass
    sys.stderr.write(message + "\n" + detail + "\n")
    sys.exit(2)


def _torch_load(torch, path, device):
    """``torch.load`` that works across the 2.6 ``weights_only`` default change."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=device)


def build_pocket_data(pocket_pdb, trans, ProteinLigandData):
    """Parse a pocket PDB into the ProteinLigandData ``sample_ode`` expects.

    Mirrors what PFM's dataset pipeline produces for the protein half of a pair
    (``PDBProtein.to_dict_atom`` → ``torchify_dict`` → ``FeaturizeProteinAtom``),
    but builds the tensors explicitly so it does not depend on ``np.long`` /
    ``np.bool``, which newer numpy removed.
    """
    import numpy as np
    import torch
    from utils.data import PDBProtein

    protein = PDBProtein(pocket_pdb)
    if len(protein.pos) == 0:
        raise ValueError("no ATOM records parsed from %s — is it a PDB pocket file?" % pocket_pdb)

    data = ProteinLigandData(
        protein_pos=torch.from_numpy(np.array(protein.pos, dtype=np.float32)),
        protein_element=torch.from_numpy(np.array(protein.element, dtype=np.int64)),
        protein_is_backbone=torch.from_numpy(np.array(protein.is_backbone, dtype=bool)),
        protein_atom_to_aa_type=torch.from_numpy(np.array(protein.atom_to_aa_type, dtype=np.int64)),
    )
    data = trans.FeaturizeProteinAtom()(data)
    return data, len(protein.pos)


def main():
    parser = argparse.ArgumentParser(description="Sample PFM ligands for one pocket.")
    parser.add_argument("request", help="path to the request JSON")
    args = parser.parse_args()

    with open(args.request) as fh:
        req = json.load(fh)
    out_dir = req.get("out_dir") or "./pfm_out"

    try:
        import numpy as np
        import torch
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        # --- PFM imports (PYTHONPATH must contain $PFM_ROOT and $PFM_ROOT/scripts)
        from datasets.pl_data import ProteinLigandData
        from egnn.DotGat import DotPre_FM
        from flow_matching_utils import CFM_CD_PERTURB
        from prior_distributions import PositionFeaturePrior_PFM
        from sampling import sample_ode
        from utils import misc
        from utils import transforms as trans
        from utils.reconstruct import reconstruct_from_generated
    except ImportError as exc:
        _fail(
            out_dir,
            "cannot import PFM/torch inside the configured interpreter (%s)" % sys.executable,
            "%s\nPYTHONPATH=%s" % (exc, os.environ.get("PYTHONPATH", "")),
        )
        return

    warnings = []
    try:
        device = req.get("device") or "cuda:0"
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.append("CUDA unavailable in the PFM environment; falling back to CPU (slow)")
            device = "cpu"

        seed = int(req.get("seed", 2024))
        misc.seed_all(seed)

        training_config = misc.load_config(req["training_yml"])
        ligand_atom_mode = training_config.data.transform.ligand_atom_mode
        # sample_ode hard-codes 12 logit classes and returns argmax+1, i.e. an
        # index into the 13-entry add_aromatic map (0 = hydrogen, never emitted).
        if ligand_atom_mode != "add_aromatic":
            warnings.append(
                "training config declares ligand_atom_mode=%r; the sampler assumes "
                "'add_aromatic' (12 emitted classes) — atom types may be misdecoded" % ligand_atom_mode
            )

        # --- pocket -------------------------------------------------------
        data, n_pocket_atoms = build_pocket_data(req["pocket_pdb"], trans, ProteinLigandData)
        if n_pocket_atoms > 3000:
            warnings.append(
                "pocket has %d atoms, which looks like a whole protein rather than a "
                "pocket; PFM sizes the ligand from the pocket extent, so results will "
                "be poor. Cut a pocket first (see PFM scripts/data_preparation/clean_pocket.py)."
                % n_pocket_atoms
            )

        # --- model --------------------------------------------------------
        protein_feat_dim = int(data.protein_atom_feature.size(-1))
        net_dynamics = DotPre_FM(training_config.model, protein_feat_dim, 12)
        x_bar_predictor = DotPre_FM(training_config.modelp, protein_feat_dim, 12)
        h_bar_predictor = DotPre_FM(training_config.modelp, protein_feat_dim, 12)
        prior = PositionFeaturePrior_PFM(n_dim=3, in_node_nf=12)
        flow = CFM_CD_PERTURB(training_config.train.sigma_pos).to(device)

        # PFM's checkpoints carry optimizer/scheduler state, not just tensors, so
        # they need the pre-2.6 unpickling behaviour. torch.load has accepted
        # weights_only since 1.13, so this is safe on older pinned torches too.
        ckpt = _torch_load(torch, req["checkpoint"], device)
        x_pre_path = req.get("x_predictor") or training_config.data.x_predictor
        h_pre_path = req.get("h_predictor") or training_config.data.h_predictor
        for label, path in (("x_predictor", x_pre_path), ("h_predictor", h_pre_path)):
            if not os.path.isfile(path):
                raise IOError(
                    "%s not found at %s — PFM's shipped training.yml points at the "
                    "authors' absolute paths; set DRUGAGENT_PFM_%s." % (label, path, label.upper())
                )
        x_pre_ckpt = _torch_load(torch, x_pre_path, device)
        h_pre_ckpt = _torch_load(torch, h_pre_path, device)
        x_bar_predictor.load_state_dict(x_pre_ckpt["net_dynamics_state_dict"])
        h_bar_predictor.load_state_dict(h_pre_ckpt["net_dynamics_state_dict"])
        net_dynamics.load_state_dict(ckpt["net_dynamics_state_dict"])
        flow.load_state_dict(ckpt["FM_state_dict"])

        x_bar_predictor = x_bar_predictor.to(device).eval()
        h_bar_predictor = h_bar_predictor.to(device).eval()
        net_dynamics = net_dynamics.to(device).eval()
        x_para = flow.x_perturb.detach()
        h_para = flow.h_perturb.detach()

        # --- sample -------------------------------------------------------
        n_requested = int(req.get("n_molecules", 10))
        batch_size = max(1, int(req.get("batch_size", 10)))
        molecules = []
        n_sampled = 0
        n_failures = 0
        sample_index = 0

        while n_sampled < n_requested:
            num_mole = min(batch_size, n_requested - n_sampled)
            with torch.no_grad():
                ligand_pos, ligand_v = sample_ode(
                    device,
                    data,
                    x_bar_predictor,
                    h_bar_predictor,
                    x_para,
                    h_para,
                    prior,
                    net_dynamics,
                    sample_num_atoms="prior",
                    num_mole=num_mole,
                )
            n_sampled += num_mole

            for pos, types in zip(ligand_pos, ligand_v):
                index = sample_index
                sample_index += 1
                try:
                    idx = np.asarray(types).astype(int)
                    atomic_nums = trans.get_atomic_number_from_index(idx, mode=ligand_atom_mode)
                    aromatic = trans.is_aromatic_from_index(idx, mode=ligand_atom_mode)
                    mol = reconstruct_from_generated(
                        np.asarray(pos, dtype=np.float64).tolist(),
                        atomic_nums,
                        aromatic,
                        basic_mode=(ligand_atom_mode == "basic"),
                    )
                    smiles = Chem.MolToSmiles(mol)
                    if not smiles or Chem.MolFromSmiles(smiles) is None:
                        raise ValueError("reconstructed molecule does not round-trip through SMILES")
                    molecules.append(
                        {
                            "smiles": smiles,
                            "sdf": Chem.MolToMolBlock(mol),
                            "sample_index": index,
                            "n_atoms": int(mol.GetNumAtoms()),
                            "qc_flags": [],
                        }
                    )
                except Exception as exc:  # a fraction always fails; that is the validity metric
                    n_failures += 1
                    if n_failures <= 3:
                        warnings.append("reconstruction failure (sample %d): %s" % (index, exc))

        # --- write --------------------------------------------------------
        os.makedirs(out_dir, exist_ok=True)
        sdf_path = os.path.join(out_dir, "molecules.sdf")
        writer = Chem.SDWriter(sdf_path)
        for item in molecules:
            block = Chem.MolFromMolBlock(item["sdf"], sanitize=False)
            if block is not None:
                writer.write(block)
        writer.close()

        payload = {
            "molecules": molecules,
            "n_requested": n_requested,
            "n_sampled": n_sampled,
            "n_reconstruction_failures": n_failures,
            "n_pocket_atoms": n_pocket_atoms,
            "device": device,
            "seed": seed,
            "ligand_atom_mode": ligand_atom_mode,
            "sdf_path": sdf_path,
            "warnings": warnings,
            "errors": [],
        }
        with open(os.path.join(out_dir, "result.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        sys.stderr.write(
            "PFM: %d/%d molecules reconstructed (%d failures)\n" % (len(molecules), n_sampled, n_failures)
        )
    except Exception as exc:  # always report structurally, never crash opaquely
        _fail(out_dir, "PFM sampling failed: %s" % exc, traceback.format_exc())


if __name__ == "__main__":
    main()
