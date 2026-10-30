"""Sample unconditional all-atom molecules with FlowMol3 — the drug-agent ⇄ FlowMol bridge.

This runs inside whichever interpreter has FlowMol's dependencies (DGL, PyTorch
Lightning, einops, torch_scatter, wandb), which is not the interpreter the platform
runs in. It imports nothing from ``drug_agent``; the two sides talk over a JSON
request/response pair on disk, so a DGL or CUDA fault here cannot reach the caller.

FlowMol3 is **unconditional**. There is no pocket argument and no reference ligand
argument, because the model has neither. Anything the caller knows about a target is
discarded before this script is reached, and the adapter says so on the batch.

Every upstream call below was taken from the checkout, not from memory:

``FlowMol.load_from_checkpoint(ckpt)``
    ``test.py`` line 76 (``checkpoint_file = args.model_dir / 'checkpoints' /
    'last.ckpt'``) and line 82 (``model = FlowMol.load_from_checkpoint(checkpoint_file)``), and
    ``flowmol/__init__.py`` in ``load_pretrained``, which builds the identical path.
    ``config.yaml`` beside the checkpoint is read by ``test.py`` only for the
    ``--metrics`` path, so plain sampling does not need it.

``model.sample_random_sizes(batch_size, device=..., n_timesteps=...)``
    ``test.py`` line 110, dispatching to
    ``flowmol/models/flowmol.py:473``. The number of atoms per molecule is drawn
    from the training-set histogram (``sample_n_atoms``, same file line 468), so no
    size argument is invented here. ``n_timesteps`` reaches ``sample`` through
    ``**kwargs``; upstream's default is 250 (``test.py`` argparse, and
    ``FlowMol.default_n_timesteps``).

``SampledMolecule.rdkit_mol``
    ``flowmol/analysis/molecule_builder.py:70``. Note that
    ``molecule_builder.build_molecule`` (same file, line 268) returns an
    **unsanitized** RWMol and returns ``None`` on a kekulization failure.

SMILES extraction copies ``SampleAnalyzer.compute_validity``
(``flowmol/analysis/metrics.py:170-200``) step for step — fragment split with
``sanitizeFrags=False``, keep the largest fragment, ``SanitizeMol``, then
``MolToSmiles`` — so that "valid" here means what it means in the FlowMol3 paper
rather than something this bridge invented. A molecule that fails any of those
steps is returned with an empty ``smiles`` and a reason, never dropped silently:
the caller counts it against validity, which is the number that exposes a bad
checkpoint or a bad reconstruction.

Request JSON:
    flowmol_root    checkout root, prepended to ``sys.path`` (optional; PYTHONPATH
                    normally carries it already)
    model_dir       trained-model directory, e.g.
                    ``flowmol/trained_models/flowmol3`` (holds ``checkpoints/last.ckpt``)
    n_molecules     how many molecules to sample in total
    batch_size      molecules per ``sample_random_sizes`` call
    n_timesteps     Euler integration steps (upstream default 250)
    seed            RNG seed, passed to ``pytorch_lightning.seed_everything``
    device          "cuda:0" or "cpu"; falls back to cpu when CUDA is absent
    out_dir         where to write result.json

Response JSON:
    molecules[]     {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_requested, n_sampled, n_build_failures, n_disconnected, n_sanitize_failures
    device, n_timesteps, checkpoint, warnings[], errors[]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver  # the sys.path insert above is what makes this importable


def _extract(rdmol, chem):
    """One sampled molecule's SMILES and SDF, or a reason it has neither.

    Mirrors ``SampleAnalyzer.compute_validity`` in
    ``flowmol/analysis/metrics.py``: split fragments without sanitizing, keep the
    largest, sanitize that, then canonicalize. Hydrogens are explicit in FlowMol's
    output because ``H`` is a generated atom type, so the SMILES is reported with
    them removed while the SDF block keeps every generated atom and its coordinate.
    """
    if rdmol is None:
        # build_molecule returns None when RDKit cannot kekulize the generated
        # bond assignment (flowmol/analysis/molecule_builder.py:268).
        return "", None, ["flowmol_build_molecule_returned_none"], 0, False

    frags = chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
    disconnected = len(frags) > 1
    largest = max(frags, default=rdmol, key=lambda m: m.GetNumAtoms())
    n_atoms = largest.GetNumAtoms()
    chem.SanitizeMol(largest)

    try:
        smiles = chem.MolToSmiles(chem.RemoveHs(largest))
    except Exception:  # RemoveHs is a convenience, not the result
        smiles = chem.MolToSmiles(largest)
    try:
        sdf = chem.MolToMolBlock(largest)
    except Exception:  # upstream writes with kekulize=False for exactly this reason
        sdf = chem.MolToMolBlock(largest, kekulize=False)
    return smiles, sdf, [], n_atoms, disconnected


def run(request):
    """Load the FlowMol3 checkpoint and sample ``n_molecules`` from it."""
    root = request.get("flowmol_root")
    if root:
        sys.path.insert(0, str(root))

    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    # test.py:5-6 — the two upstream names a sampling run needs.
    from flowmol.models.flowmol import FlowMol
    from pytorch_lightning import seed_everything

    warnings = []
    n_requested = int(request.get("n_molecules", 10))
    batch_size = max(1, int(request.get("batch_size", 10)))
    n_timesteps = int(request.get("n_timesteps", 250))
    seed = request.get("seed")

    model_dir = Path(request["model_dir"])
    # flowmol/__init__.py load_pretrained, and test.py:84 — the same path in both.
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"FlowMol3 checkpoint {checkpoint} does not exist. Download it with: wget -r -np -nH "
            "--cut-dirs=3 --reject 'index.html*' -P flowmol/trained_models/ "
            "https://bits.csb.pitt.edu/files/FlowMol/trained_models_v3.1/flowmol3/"
        )

    if seed is not None:
        seed_everything(int(seed))  # test.py:71

    # test.py:91 — cuda:0 when it is really there, cpu otherwise.
    want = str(request.get("device") or "cuda:0")
    if want.startswith("cuda") and not torch.cuda.is_available():
        warnings.append(f"CUDA is not available in this interpreter; sampling on cpu instead of {want}")
        want = "cpu"
    device = torch.device(want)

    model = FlowMol.load_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    molecules = []
    n_sampled = 0
    n_build_failures = 0
    n_disconnected = 0
    n_sanitize_failures = 0

    n_batches = math.ceil(n_requested / batch_size)  # test.py:100
    for _ in range(n_batches):
        n_needed = n_requested - n_sampled
        if n_needed <= 0:
            break
        # flowmol/models/flowmol.py:473 — sizes drawn from the training histogram.
        batch = model.sample_random_sizes(
            min(n_needed, batch_size),
            device=device,
            n_timesteps=n_timesteps,
        )
        for sampled in batch:
            idx = n_sampled
            n_sampled += 1
            try:
                smiles, sdf, flags, n_atoms, disconnected = _extract(sampled.rdkit_mol, Chem)
            except Exception as exc:  # a bad reconstruction is data, not a crash
                n_sanitize_failures += 1
                molecules.append(
                    {
                        "smiles": "",
                        "sdf": None,
                        "sample_index": idx,
                        "n_atoms": 0,
                        "qc_flags": [f"rdkit_rejected_reconstruction: {type(exc).__name__}: {exc}"],
                    }
                )
                continue
            if not smiles:
                n_build_failures += 1
            if disconnected:
                n_disconnected += 1
            molecules.append(
                {
                    "smiles": smiles,
                    "sdf": sdf,
                    "sample_index": idx,
                    "n_atoms": n_atoms,
                    "qc_flags": flags,
                }
            )

    if n_disconnected:
        warnings.append(
            f"{n_disconnected} of {n_sampled} samples came back as more than one fragment; "
            "the largest fragment was kept, as upstream's own validity metric does"
        )

    return {
        "molecules": molecules,
        "n_requested": n_requested,
        "n_sampled": n_sampled,
        "n_build_failures": n_build_failures,
        "n_disconnected": n_disconnected,
        "n_sanitize_failures": n_sanitize_failures,
        "device": str(device),
        "n_timesteps": n_timesteps,
        "checkpoint": str(checkpoint),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
