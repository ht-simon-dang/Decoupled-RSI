"""Sample ligands for one pocket with DrugFlow — the drug-agent ⇄ DrugFlow bridge.

This file is executed by the interpreter that has DrugFlow's dependencies, which
upstream builds from ``.external/DrugFlow/environment.yaml`` (Python 3.11.8,
torch 2.2.1+cu121, pytorch-lightning 2.2.1, pyg 2.5.1, rdkit 2023.09.6, plus
``webdataset``, ``ProDy``, ``fcd``, ``useful_rdkit_utils``, ``prolif`` and
``posebusters`` from the pip section). It imports nothing from ``drug_agent``:
only the standard library and DrugFlow itself, exactly as
``scripts/external/_driver.py`` requires.

**What it mirrors.** ``src/generate.py`` is DrugFlow's single-target entry point
and it is a ``__main__`` script with no importable function, so this driver
reproduces its body call for call. Every step below cites the upstream line it
came from:

    ``src/generate.py:78-79``    ``utils.set_deterministic`` + ``disable_rdkit_logging``
    ``src/generate.py:88``       batch size is clamped to the sample count
    ``src/generate.py:93``       ``DrugFlow.load_from_checkpoint(..., strict=False)``
    ``src/generate.py:94-99``    ``model.datadir``, ``setup(stage='generation')``,
                                 ``batch_size``/``eval_batch_size``, ``eval().to(device)``
    ``src/generate.py:100-101``  ``model.T`` override when ``n_steps`` is given
    ``src/generate.py:107-117``  ``molecule_size`` parsing: an int, or ``uniform_L_R``
    ``src/generate.py:120-129``  ``PDBParser`` + ``SDMolSupplier`` + ``process_raw_pair``
    ``src/generate.py:130-137``  one pocket repeated ``batch_size`` times through
                                 ``ProcessedLigandPocketDataset.collate_fn``
    ``src/generate.py:150-160``  ``model.sample(new_data, n_samples=1, ...)``

**What it adds, and why.** ``src/generate.py`` writes its samples with
``utils.write_sdf_file`` (line 200), and ``Chem.SDWriter`` does not carry *atom*
properties — so running the upstream script would silently discard the thing that
distinguishes DrugFlow from every other pocket-conditioned generator. When the
checkpoint was trained with ``predict_confidence: True``
(``configs/training/drugflow_ood.yml:45``), ``src/model/lightning.py:1249-1256``
writes two per-atom double properties onto each generated molecule:

    ``sigma_x``     ``sqrt(∫ σ_vel² dt)`` accumulated over the sampling trajectory
                    (``lightning.py:1003`` accumulates, ``:1085`` takes the root)
    ``entropy_h``   ``∫ H(logits_h) dt``, the categorical entropy of the atom-type
                    prediction in nats (``lightning.py:1004``). The ``dt`` weights
                    sum to 1 over ``t ∈ [0, 1]``, so this is a trajectory average.

This driver reads those atom properties and averages them per molecule with the
same ``mean`` reduction DrugFlow's own ``src/sample_and_evaluate.py:93-95`` uses,
then writes the per-molecule values back as *molecule* properties before
serialising — which is also what ``sample_and_evaluate.py:95`` does — so the
numbers survive into the SDF block the caller stores.

Nothing is computed here that DrugFlow does not compute: the mean, the max and
the atom count are reductions over values the model emitted.

Request JSON:
    root                     DrugFlow checkout root, prepended to ``sys.path``
    protein_pdb              input protein PDB
    ref_ligand_sdf           reference ligand SDF that defines the pocket
    checkpoint               ``.ckpt`` file to load
    datadir                  directory holding ``size_distribution.npy``
    n_molecules              total molecules to sample
    batch_size               molecules per ``model.sample`` call
    seed                     RNG seed
    device                   e.g. ``cuda:0`` / ``cpu``
    n_steps                  denoising steps; null keeps the checkpoint's own T
    molecule_size            null, ``"20"`` or ``"15,25"`` (upstream's spelling)
    pocket_distance_cutoff   Å cutoff around the reference ligand
    max_batches              hard cap on sampler calls; defaults to twice the
                             number a healthy run needs, so a degenerate batch
                             cannot spin for the whole timeout
    out_dir                  where ``result.json`` goes (injected by the caller)

Response JSON (``out_dir/result.json``):
    molecules[]              {smiles, sdf, sample_index, n_atoms, qc_flags[],
                              uncertainty{...} or null}
    n_requested, n_sampled, n_unparseable, n_batches
    predict_confidence       whether the loaded checkpoint has the confidence head
    checkpoint, device, n_steps, batch_size, seed
    engine_version           DrugFlow has no ``__version__``; this is the commit
                             the caller said it launched, echoed back
    warnings[], errors[]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path insert above: the shared driver contract lives beside this file.
import _driver

#: The per-atom double properties DrugFlow's confidence head writes onto every
#: generated molecule. Named in ``src/model/lightning.py:1252-1253``.
_UNCERTAINTY_PROPS: tuple[str, ...] = ("sigma_x", "entropy_h")

#: Upstream's own default, ``src/generate.py:67``.
_DEFAULT_POCKET_CUTOFF = 8.0


def _molecule_size_spec(raw: Any) -> Any:
    """Parse ``--molecule_size`` the way ``src/generate.py:107-117`` parses it.

    ``None`` means "sample the size from the CrossDocked histogram", a bare
    integer fixes it, and ``"15,25"`` becomes the ``uniform_15_25`` string
    ``DrugFlow.parse_num_nodes_spec`` understands.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"molecule_size must be an integer or 'left,right', got {raw!r}")
    return f"uniform_{int(parts[0])}_{int(parts[1])}"


def _atom_uncertainty(mol: Any) -> dict[str, float] | None:
    """Per-molecule uncertainty, reduced from the per-atom properties.

    Returns ``None`` when the checkpoint carried no confidence head, which is the
    honest answer for ``checkpoints/drugflow.ckpt``
    (``configs/training/drugflow.yml:44`` sets ``predict_confidence: False``).
    The ``mean`` is the same reduction ``src/sample_and_evaluate.py:95`` applies.
    """
    atoms = list(mol.GetAtoms())
    if not atoms:
        return None
    out: dict[str, float] = {}
    for prop in _UNCERTAINTY_PROPS:
        values = [a.GetDoubleProp(prop) for a in atoms if a.HasProp(prop)]
        if not values:
            continue
        out[f"{prop}_mean"] = float(sum(values) / len(values))
        out[f"{prop}_max"] = float(max(values))
    if not out:
        return None
    out["n_atoms_scored"] = float(len(atoms))
    return out


def _sdf_block(mol: Any, uncertainty: dict[str, float] | None) -> str:
    """One molecule as SDF text, carrying the uncertainty as SD data fields.

    The molecules come out of ``build_molecule`` unsanitized, so kekulisation can
    fail; ``src/utils.py:120-123`` handles exactly that by retrying with
    kekulisation off, and so does this.
    """
    import io

    from rdkit import Chem

    tagged = Chem.Mol(mol)
    for key, value in (uncertainty or {}).items():
        tagged.SetDoubleProp(key, float(value))
    try:
        buffer = io.StringIO()
        writer = Chem.SDWriter(buffer)
        writer.write(tagged)
        writer.close()
        text = buffer.getvalue()
        if text.strip():
            return text
    except (RuntimeError, ValueError, TypeError):
        pass
    # Last resort: the bare mol block. It loses the SD tags, which is why the
    # caller also receives `uncertainty` as a field of its own.
    return Chem.MolToMolBlock(tagged, kekulize=False)


def _smiles(mol: Any) -> str:
    """SMILES by DrugFlow's own route — ``RepresentationEvaluator``, metrics.py:151."""
    from rdkit import Chem

    try:
        return Chem.MolToSmiles(mol) or ""
    except (RuntimeError, ValueError):
        return ""


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Load one DrugFlow checkpoint and sample molecules for one pocket."""
    root = str(request.get("root") or "")
    if root:
        # src/generate.py:14-15 does the same for its own basedir, and `src` is a
        # namespace package with no __init__.py, so the root must be on sys.path.
        sys.path.insert(0, root)

    import warnings as _warnings

    _warnings.filterwarnings("ignore")  # src/generate.py:16

    from functools import partial

    from Bio.PDB import PDBParser
    from rdkit import Chem
    from src import utils
    from src.data.data_utils import TensorDict, process_raw_pair
    from src.data.dataset import ProcessedLigandPocketDataset
    from src.model.lightning import DrugFlow
    from torch.utils.data import DataLoader

    warnings: list[str] = []

    protein_pdb = str(request.get("protein_pdb") or "")
    ref_ligand_sdf = str(request.get("ref_ligand_sdf") or "")
    checkpoint = str(request.get("checkpoint") or "")
    for label, path in (
        ("protein_pdb", protein_pdb),
        ("ref_ligand_sdf", ref_ligand_sdf),
        ("checkpoint", checkpoint),
    ):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"{label} not found at {path!r}")

    n_molecules = int(request.get("n_molecules") or 10)
    batch_size = int(request.get("batch_size") or 32)
    seed = int(request.get("seed") or 42)
    device = str(request.get("device") or "cuda:0")
    n_steps = request.get("n_steps")
    n_steps = int(n_steps) if n_steps else None
    cutoff = float(request.get("pocket_distance_cutoff") or _DEFAULT_POCKET_CUTOFF)
    molecule_size = _molecule_size_spec(request.get("molecule_size"))

    utils.set_deterministic(seed=seed)  # src/generate.py:78
    utils.disable_rdkit_logging()  # src/generate.py:79

    datadir = Path(request.get("datadir") or Path(root, "src", "default"))
    if not (datadir / "size_distribution.npy").is_file():
        # Checked unconditionally, unlike src/generate.py:81-85, which only checks
        # it when no molecule size was given: `setup_sampling` (lightning.py:412)
        # np.load()s the histogram on every path, so the narrower guard just moves
        # the same failure into a stack trace.
        raise FileNotFoundError(
            f"{datadir / 'size_distribution.npy'} is missing, so "
            "DrugFlow.setup(stage='generation') cannot build its size distribution"
        )

    batch_size = min(batch_size, n_molecules)  # src/generate.py:88

    # Each pass of the loop below is exactly one `model.sample` call yielding
    # `batch_size` molecules, so the number of calls needed is known. Twice that,
    # floored at four, is the cap: it cannot bite a healthy run and it stops a
    # sampler that returns nothing from spinning for the whole timeout budget.
    needed = -(-n_molecules // batch_size) if batch_size else 1
    max_batches = int(request.get("max_batches") or max(4, needed * 2))

    model = DrugFlow.load_from_checkpoint(  # src/generate.py:93
        checkpoint, map_location=device, strict=False
    )
    model.datadir = datadir  # src/generate.py:94-95
    model.setup(stage="generation")  # src/generate.py:97
    model.batch_size = model.eval_batch_size = batch_size  # src/generate.py:98
    model.eval().to(device)  # src/generate.py:99
    if n_steps is not None:
        model.T = n_steps  # src/generate.py:100-101

    predict_confidence = bool(getattr(model, "predict_confidence", False))
    if not predict_confidence:
        warnings.append(
            f"{Path(checkpoint).name} was trained with predict_confidence=False, so DrugFlow "
            "emits no sigma_x/entropy_h and every molecule is flagged no_confidence_head"
        )

    # --- pocket, exactly as src/generate.py:120-137 builds it -----------------
    pdb_model = PDBParser(QUIET=True).get_structure("", protein_pdb)[0]
    rdmol = Chem.SDMolSupplier(ref_ligand_sdf)[0]
    if rdmol is None:
        raise ValueError(f"RDKit could not read a reference ligand from {ref_ligand_sdf}")

    ligand, pocket = process_raw_pair(
        pdb_model,
        rdmol,
        dist_cutoff=cutoff,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=protein_pdb if model.dynamics.add_nma_feat else None,
    )
    ligand["name"] = "ligand"
    # data_utils.py:385 — every pocket representation sets `size` to a one-element
    # tensor holding the residue count, so this access is safe for all of them.
    n_pocket_residues = int(pocket["size"].item())
    if n_pocket_residues == 0:
        raise ValueError(
            f"no standard residues within {cutoff} A of the reference ligand — "
            f"is {Path(ref_ligand_sdf).name} the ligand of {Path(protein_pdb).name}?"
        )

    dataset = [{"ligand": ligand, "pocket": pocket} for _ in range(batch_size)]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
        pin_memory=True,
    )

    # --- sampling loop, src/generate.py:148-197 without --filter --------------
    molecules: list[dict[str, Any]] = []
    n_sampled = 0
    n_unparseable = 0
    n_batches = 0
    while n_sampled < n_molecules and n_batches < max_batches:
        for data in dataloader:
            new_data = {
                "ligand": TensorDict(**data["ligand"]).to(device),
                "pocket": TensorDict(**data["pocket"]).to(device),
            }
            rdmols, _rdpockets, _names = model.sample(
                new_data, n_samples=1, timesteps=n_steps, num_nodes=molecule_size
            )
            n_batches += 1
            for mol in rdmols:
                index = n_sampled
                n_sampled += 1
                qc_flags: list[str] = []
                if mol is None:
                    n_unparseable += 1
                    molecules.append(
                        {
                            "smiles": "",
                            "sdf": None,
                            "sample_index": index,
                            "n_atoms": 0,
                            "qc_flags": ["build_molecule_returned_none"],
                            "uncertainty": None,
                        }
                    )
                    continue
                uncertainty = _atom_uncertainty(mol)
                if uncertainty is None:
                    qc_flags.append("no_confidence_head")
                smiles = _smiles(mol)
                if not smiles:
                    n_unparseable += 1
                    qc_flags.append("smiles_write_failed")
                molecules.append(
                    {
                        "smiles": smiles,
                        "sdf": _sdf_block(mol, uncertainty),
                        "sample_index": index,
                        "n_atoms": int(mol.GetNumAtoms()),
                        "qc_flags": qc_flags,
                        "uncertainty": uncertainty,
                    }
                )
            if n_sampled >= n_molecules:
                break

    if n_sampled < n_molecules:
        warnings.append(
            f"stopped after {n_batches} sampler calls with {n_sampled}/{n_molecules} molecules; "
            "raise max_batches if this was not a degenerate pocket"
        )

    return {
        "molecules": molecules[:n_molecules],
        "n_requested": n_molecules,
        "n_sampled": n_sampled,
        "n_unparseable": n_unparseable,
        "n_batches": n_batches,
        "n_pocket_residues": n_pocket_residues,
        "predict_confidence": predict_confidence,
        "checkpoint": checkpoint,
        "device": device,
        "n_steps": n_steps if n_steps is not None else int(getattr(model, "T_sampling", 0)),
        "batch_size": batch_size,
        "seed": seed,
        "engine_version": str(request.get("upstream_ref") or "unknown"),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
