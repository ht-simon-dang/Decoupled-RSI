"""Sample ligands from ShEPhERD against a *reference ligand* — the drug-agent ⇄ ShEPhERD bridge.

This runs inside whichever interpreter has ShEPhERD's dependencies. It imports
nothing from ``drug_agent``; the two sides talk over a JSON request/response pair
on disk, so a CUDA fault here cannot reach the caller.

**ShEPhERD is not a pocket generator.** It conditions on the shape, electrostatic
potential and pharmacophores of a *molecule*, so this driver refuses to run
without a reference SMILES. There is no pocket argument, because upstream has no
pocket argument: every conditional entry point in ``examples/`` starts from an
RDKit ``Mol`` with a conformer.

Which repository this needs: only ``.external/shepherd``. Its README states that
the parts of ``shepherd-score`` needed to sample were copied into
``src/shepherd/shepherd_score_utils/``, and this driver imports them from there.
``.external/shepherd-score`` supplies the *evaluation* suite — 3D similarity,
alignment, post-relaxation validity — which is scoring, not generation, and is
never imported here.

The conditioning pipeline, and where each step comes from in the checkout:

  * ``embed_conformer_from_smiles`` — ``src/shepherd/shepherd_score_utils/conformer_generation.py``
    (``AddHs`` → ETKDG → MMFF94). ShEPhERD diffuses explicit hydrogens, so the
    reference must carry them.
  * centering the conformer on the origin — required, per
    ``examples/conditional_generation.ipynb`` cell 5 and
    ``examples/paper_experiments/run_inference_pdb.py``.
  * ``get_atomic_vdw_radii`` / ``get_molecular_surface`` / ``get_electrostatics``
    — ``src/shepherd/shepherd_score_utils/generate_point_cloud.py``. The surface
    is Open3D ball-pivoting plus Poisson-disk sampling; there is no pure-NumPy
    path to it upstream.
  * ``get_pharmacophores`` — ``src/shepherd/shepherd_score_utils/pharm_utils/pharmacophore.py``.
  * ``generate`` — ``src/shepherd/inference/sampler.py``, re-exported by
    ``src/shepherd/inference/__init__.py``.
  * ``create_rdkit_molecule`` — ``src/shepherd/extract.py``.

**Partial charges.** ``examples/conditional_generation.ipynb`` obtains reference
partial charges from xTB. xTB is a separate binary that is not part of the
ShEPhERD package, so this driver calls upstream's other supported route,
``get_electrostatics(mol, points)``, whose own docstring reads "Gets partial
charges from MMFF or Gasteiger". That is a real electrostatic potential from real
charges, but they are MMFF94 charges, not xTB charges, and the difference is
reported in ``warnings`` on every run rather than hidden.

Request JSON:
    reference_smiles   SMILES of the reference ligand (required, non-empty)
    n_molecules        total molecules to sample
    batch_size         molecules per ``generate`` call
    seed               RNG seed
    device             "cuda" / "cuda:0" / "cpu"
    checkpoint         path to a ShEPhERD .ckpt
    model_type         label recorded on the result (e.g. "mosesaq")
    num_steps          denoising steps (upstream default 400)
    n_atoms            optional override for N_x1; defaults to the reference's
                       atom count including hydrogens
    embed_attempts     ETKDG attempts for the reference conformer
    out_dir            directory to write result.json (+ molecules.sdf)

Response JSON (``out_dir/result.json``):
    molecules[]        {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_requested, n_sampled, n_reconstruction_failures
    reference_smiles, n_reference_atoms, n_surface_points, n_pharmacophores
    device, seed, num_steps, model_type, sdf_path, warnings[], errors[]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver  # must follow the sys.path insertion above


def _build_reference(request, params, warnings):
    """Reference conformer plus the three interaction profiles ShEPhERD inpaints.

    Returns ``(mol, surface, electrostatics, pharm_types, pharm_pos, pharm_direction)``.
    Raises on anything that would leave a conditioning target empty, because an
    empty target silently degrades conditional generation into unconditional
    generation, which is the one outcome this tool must never produce.
    """
    import numpy as np
    from shepherd.shepherd_score_utils.conformer_generation import (
        embed_conformer_from_smiles,
        update_mol_coordinates,
    )
    from shepherd.shepherd_score_utils.generate_point_cloud import (
        get_atomic_vdw_radii,
        get_electrostatics,
        get_molecular_surface,
    )
    from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

    smiles = str(request.get("reference_smiles") or "").strip()
    if not smiles:
        raise ValueError(
            "ShEPhERD conditions on a reference ligand's shape, electrostatics and "
            "pharmacophores. No reference_smiles was supplied, and there is no pocket "
            "argument that could stand in for one."
        )

    mol = embed_conformer_from_smiles(
        smiles,
        attempts=int(request.get("embed_attempts", 200)),
        # MMFF94 relaxation, so the MMFF partial charges below are taken on a
        # geometry MMFF itself considers reasonable.
        MMFF_optimize=True,
    )
    if mol is None:
        raise ValueError(f"could not embed a 3D conformer for reference SMILES {smiles!r}")

    # Centering is required by the sampler: center_of_mass is passed as zeros.
    positions = np.array(mol.GetConformer().GetPositions())
    mol = update_mol_coordinates(mol, positions - positions.mean(axis=0))

    centers = mol.GetConformer().GetPositions()
    radii = get_atomic_vdw_radii(mol)
    # Argument names and the num_samples_per_atom=20 value are taken verbatim from
    # examples/paper_experiments/run_inference_pdb.py.
    surface = get_molecular_surface(
        centers,
        radii,
        params["dataset"]["x3"]["num_points"],
        probe_radius=params["dataset"]["probe_radius"],
        num_samples_per_atom=20,
    )
    n_points = int(params["dataset"]["x3"]["num_points"])
    if surface.shape[0] != n_points:
        warnings.append(
            f"surface sampling returned {surface.shape[0]} points, not the {n_points} "
            "the checkpoint was trained on"
        )

    electrostatics = get_electrostatics(mol, surface)
    warnings.append(
        "reference electrostatics were computed from MMFF94 partial charges via "
        "shepherd_score_utils.get_electrostatics; the published protocol uses xTB "
        "charges, which are not available in this environment"
    )

    pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
        mol,
        multi_vector=params["dataset"]["x4"]["multivectors"],
        check_access=params["dataset"]["x4"]["check_accessibility"],
    )
    if len(pharm_types) == 0:
        raise ValueError(
            f"no pharmacophores were extracted from reference {smiles!r}; ShEPhERD's x4 "
            "channel would have nothing to inpaint"
        )
    return mol, surface, electrostatics, pharm_types, pharm_pos, pharm_direction


def run(request):
    """Load the checkpoint, condition on the reference ligand, sample, reconstruct."""
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    from shepherd import load_model
    from shepherd.extract import create_rdkit_molecule
    from shepherd.inference import generate

    warnings: list[str] = []

    device = str(request.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append("CUDA is unavailable in the ShEPhERD environment; falling back to CPU (slow)")
        device = "cpu"

    seed = int(request.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = str(request["checkpoint"])
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"ShEPhERD checkpoint not found at {checkpoint}")
    # load_model's local_checkpoint_path branch (src/shepherd/model_loader.py) short-
    # circuits before checkpoint_manager, so nothing here reaches HuggingFace.
    model_pl = load_model(device=device, local_checkpoint_path=checkpoint)
    model_pl.to(torch.device(device))
    model_pl.model.device = torch.device(device)
    params = model_pl.params

    mol, surface, electrostatics, pharm_types, pharm_pos, pharm_direction = _build_reference(
        request, params, warnings
    )

    n_requested = int(request.get("n_molecules", 10))
    batch_size = max(1, int(request.get("batch_size", 10)))
    num_steps = int(request.get("num_steps", 400))
    n_atoms = int(request.get("n_atoms") or mol.GetNumAtoms())

    molecules = []
    n_sampled = 0
    n_failures = 0
    sample_index = 0
    while n_sampled < n_requested:
        this_batch = min(batch_size, n_requested - n_sampled)
        # Keyword names and the inpainting flags are those of
        # shepherd.inference.sampler.generate, as called in
        # examples/conditional_generation.ipynb cell 9.
        samples = generate(
            model_pl=model_pl,
            batch_size=this_batch,
            N_x1=n_atoms,
            N_x4=len(pharm_types),
            unconditional=False,
            inpaint_x3_pos=True,
            inpaint_x3_x=True,
            inpaint_x4_pos=True,
            inpaint_x4_direction=True,
            inpaint_x4_type=True,
            center_of_mass=np.zeros(3),
            surface=surface,
            electrostatics=electrostatics,
            pharm_types=pharm_types,
            pharm_pos=pharm_pos,
            pharm_direction=pharm_direction,
            num_steps=num_steps,
            verbose=False,
        )
        n_sampled += this_batch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        for sample in samples:
            index = sample_index
            sample_index += 1
            # create_rdkit_molecule writes an XYZ block and calls
            # rdDetermineBonds.DetermineBonds over the generated point cloud; it
            # returns None when bond perception fails at every trial charge. A
            # fraction always fails, and that fraction is the validity metric.
            rdmol = create_rdkit_molecule(sample)
            if rdmol is None:
                n_failures += 1
                continue
            try:
                smiles = Chem.MolToSmiles(rdmol)
                block = Chem.MolToMolBlock(rdmol)
            except Exception as exc:  # a bad graph is data, not a crash
                n_failures += 1
                if n_failures <= 3:
                    warnings.append(f"reconstruction failure (sample {index}): {exc}")
                continue
            molecules.append(
                {
                    "smiles": smiles,
                    "sdf": block,
                    "sample_index": index,
                    "n_atoms": int(rdmol.GetNumAtoms()),
                    "qc_flags": [],
                }
            )

    out_dir = Path(request["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / "molecules.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for item in molecules:
        block = Chem.MolFromMolBlock(item["sdf"], sanitize=False)
        if block is not None:
            writer.write(block)
    writer.close()

    sys.stderr.write(
        f"ShEPhERD: {len(molecules)}/{n_sampled} molecules reconstructed "
        f"({n_failures} bond-perception failures)\n"
    )
    return {
        "molecules": molecules,
        "n_requested": n_requested,
        "n_sampled": n_sampled,
        "n_reconstruction_failures": n_failures,
        "reference_smiles": str(request.get("reference_smiles") or ""),
        "n_reference_atoms": int(mol.GetNumAtoms()),
        "n_surface_points": int(surface.shape[0]),
        "n_pharmacophores": len(pharm_types),
        "n_atoms_sampled": n_atoms,
        "device": device,
        "seed": seed,
        "num_steps": num_steps,
        "model_type": str(request.get("model_type") or ""),
        "sdf_path": str(sdf_path),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
