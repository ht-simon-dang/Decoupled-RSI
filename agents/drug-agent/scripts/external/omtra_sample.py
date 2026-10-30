"""Sample pocket-conditioned ligands from OMTRA — the drug-agent ⇄ OMTRA bridge.

This runs inside OMTRA's own environment (Python 3.11, torch 2.4.0+cu121,
dgl 2.4.0+cu121 per ``requirements-cuda.txt``), never in drug-agent's. It imports
nothing from ``drug_agent`` and nothing outside the standard library except OMTRA
itself, so the two sides can hold irreconcilable pins and still talk.

**Which entry point, and why that one.** OMTRA's packaging declares one console
script — ``[project.scripts] omtra = "cli:main"`` in ``pyproject.toml`` — and
``cli.main`` does two things: it dispatches the ``mol2pharm`` subcommand, and
otherwise it calls ``cli.run_sample(args)``. This driver calls ``run_sample``
directly, with an argument namespace produced by OMTRA's **own**
``cli.create_parser()``, so every default (``n_timesteps=250``, ``bbox_length=23.0``,
``eps=0.01``, ``n_lig_atom_margin=0.15``, ``dataset='pharmit'``) is upstream's and
not this file's invention. ``run_sample`` is the layer that resolves the checkpoint
through ``omtra.utils.checkpoints.get_checkpoint_path_for_task``, turns the input
files into DGL graphs through ``omtra.utils.file_to_graph.create_conditional_graphs_from_files``,
performs the ``n_samples``/``n_replicates`` swap that makes ``--n_samples`` mean
"replicates of this one system", and then hands off to ``routines.sample.main``.
Reimplementing any of that here would be guessing at an interface that already
exists.

**Which task.** ``fixed_protein_ligand_denovo_condensed`` — pocket-conditioned de
novo generation. In ``omtra/tasks/tasks_condensed.py`` that task fixes
``['protein_identity', 'protein_structure']`` and generates
``['ligand_identity_condensed', 'ligand_structure']``: the protein is held, the
ligand's atoms, bonds and coordinates are sampled. OMTRA also ships
``rigid_docking_condensed``, which shares the *same* checkpoint
(``mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt``) but fixes
``ligand_identity_condensed`` and generates only ``ligand_structure``. That is
docking, not generation, and this driver never selects it — a molecule generator
that quietly returned poses of a molecule it was given would be lying about what
it made.

**Seeding.** OMTRA's CLI exposes no seed: neither ``cli.create_parser`` nor
``routines.sample.parse_args`` defines one, and ``routines.sample.main`` never
calls a seeding function. This driver therefore seeds ``random``, ``numpy`` and
``torch`` itself immediately before ``run_sample``. That is the only reproducibility
on offer and it is partial — DGL and cuDNN kernels are not made deterministic by
it, so repeated runs at one seed are similar, not identical. The payload reports
``seed_applied`` so the caller can say so rather than imply more.

Usage (the adapter does this for you)::

    PYTHONPATH=$OMTRA_ROOT python scripts/external/omtra_sample.py request.json

Request JSON:
    protein_file    path to the pocket/protein PDB or CIF (cli.py --protein_file)
    checkpoint      explicit checkpoint path (cli.py --checkpoint); optional
    task            OMTRA task name (cli.py --task); defaults to the de novo task
    n_molecules     replicates of this one system (cli.py --n_samples)
    n_timesteps     integration steps (cli.py --n_timesteps)
    pocket_center   [x, y, z] (cli.py --pocket_center, nargs=3); optional — when
                    absent the driver derives it from the protein file's own atoms
    bbox_length     box edge in Angstrom (cli.py --bbox_length)
    stochastic      bool (cli.py --stochastic_sampling)
    noise_scaler    float (cli.py --noise_scaler)
    seed            int, applied by this driver because OMTRA has no --seed
    out_dir         where result.json goes

Response JSON (``out_dir/result.json``):
    molecules[]     {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_written       molecules OMTRA wrote to gen_ligands.sdf
    n_sanitizable   of those, how many RDKit could sanitize
    n_allene        of those written, how many match [*]=[#6]=[*]
    n_small_ring    of those written, how many carry a 3- or 4-membered ring
    sdf_paths[]     the gen_ligands.sdf files that were read
    pocket_center   the centre actually used, and pocket_center_source says where
                    it came from ("request" or "protein_centroid")
    warnings[], errors[]
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver  # imported after the sys.path line above, which is what makes it findable

#: OMTRA's pocket-conditioned de novo task. Registered in
#: ``omtra/tasks/tasks_condensed.py`` and keyed in
#: ``omtra/utils/checkpoints.py::TASK_TO_CHECKPOINT``.
DENOVO_TASK = "fixed_protein_ligand_denovo_condensed"

#: SMARTS for a cumulated diene. Two double bonds on one sp carbon is the
#: signature artefact of a generated bond block, and OMTRA's bond vocabulary
#: (``omtra/constants/__init__.py``: ``[None, SINGLE, DOUBLE, TRIPLE]``) permits
#: it with nothing downstream rejecting it before the SDF is written.
ALLENE_SMARTS = "[*]=[#6]=[*]"


def _read_generated(sdf_path, chem):
    """Every molecule in one ``gen_ligands.sdf``, sanitized on a copy only.

    ``sanitize=False`` on the supplier is deliberate. OMTRA writes this file from
    ``omtra.eval.system.SampledSystem.build_molecule``, which assembles an
    ``RWMol`` from generated atom types, charges and bond orders and never calls
    ``Chem.SanitizeMol`` (sanitization appears only on the ``--metrics`` path, in
    ``omtra/eval/utils.py`` and ``omtra/eval/evals.py``). Reading with sanitization
    on would silently drop the invalid molecules, and how many there are is the
    validity metric.
    """
    records = []
    supplier = chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            records.append({"mol": None, "smiles": "", "sdf": "", "n_atoms": 0})
            continue
        try:
            block = chem.MolToMolBlock(mol, kekulize=False)
        except Exception:  # an unsanitized mol may refuse to serialise
            block = ""
        smiles = ""
        probe = chem.Mol(mol)
        try:
            chem.SanitizeMol(probe)
            candidate = chem.MolToSmiles(probe)
            # Round-trip, because MolToSmiles can emit a string RDKit will not read
            # back; the base adapter's validity count is exactly that read-back.
            if candidate and chem.MolFromSmiles(candidate) is not None:
                smiles = candidate
        except Exception:  # failing to sanitize IS the measurement
            smiles = ""
        records.append({"mol": mol, "smiles": smiles, "sdf": block, "n_atoms": int(mol.GetNumAtoms())})
    return records


def _reconstruction_counts(records, chem):
    """How many written molecules carry an allene, and how many a 3- or 4-ring.

    Computed on the raw molecule with ``UpdatePropertyCache(strict=False)`` and
    a symmetrized SSSR, because ring info is not populated on a mol that was never
    sanitized and ``GetRingInfo`` would otherwise raise.
    """
    allene_query = chem.MolFromSmarts(ALLENE_SMARTS)
    n_allene = 0
    n_small_ring = 0
    for record in records:
        mol = record["mol"]
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            chem.GetSymmSSSR(mol)
        except Exception:  # a malformed graph counts as neither
            continue
        try:
            if allene_query is not None and mol.HasSubstructMatch(allene_query):
                n_allene += 1
        except Exception:  # a malformed graph simply does not count
            pass
        try:
            if any(len(ring) <= 4 for ring in mol.GetRingInfo().AtomRings()):
                n_small_ring += 1
        except Exception:  # a malformed graph simply does not count
            pass
    return n_allene, n_small_ring


def _protein_centroid(protein_file):
    """The mean coordinate of a protein file's heavy atoms, as [x, y, z].

    ``cli._validate_task_inputs`` refuses a protein-conditioned task with no pocket
    definition, so a centre must exist. This mirrors OMTRA's own reader —
    ``omtra/utils/file_to_graph.py::load_protein_biotite`` reads a ``.pdb`` through
    ``biotite.structure.io.pdb`` and a ``.cif`` through ``pdbx.CIFFile``, then drops
    HOH, H and D — and then applies the same rule ``cli.run_sample`` applies for
    ``--pocket_residues``: the mean of the retained atoms' coordinates.
    """
    import biotite.structure.io.pdb as biotite_pdb
    from biotite.structure.io.pdbx import CIFFile, get_structure

    path = Path(protein_file)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        structure = biotite_pdb.PDBFile.read(str(path)).get_structure(model=1)
    elif suffix == ".cif":
        structure = get_structure(CIFFile.read(str(path)), model=1, include_bonds=False)
    else:
        raise ValueError(f"unsupported protein format {suffix!r}; OMTRA reads .pdb and .cif")
    structure = structure[structure.res_name != "HOH"]
    structure = structure[structure.element != "H"]
    structure = structure[structure.element != "D"]
    if structure.coord.size == 0:
        raise ValueError(f"{path} has no non-water heavy atoms to centre a pocket on")
    return [float(c) for c in structure.coord.mean(axis=0)]


def _build_argv(request, sample_dir):
    """The exact argv ``cli.create_parser()`` would see on the command line.

    Every flag below is read off ``cli.py::create_parser``; nothing is invented.
    """
    argv = [
        "--task",
        str(request.get("task") or DENOVO_TASK),
        "--n_samples",
        str(int(request.get("n_molecules", 10))),
        "--n_timesteps",
        str(int(request.get("n_timesteps", 250))),
        "--output_dir",
        str(sample_dir),
        "--protein_file",
        str(request["protein_file"]),
    ]
    checkpoint = request.get("checkpoint")
    if checkpoint:
        argv += ["--checkpoint", str(checkpoint)]
    # cli.py: pocket_group.add_argument("--pocket_center", type=float, nargs=3), and
    # _validate_task_inputs rejects a protein-conditioned task that has no pocket
    # definition at all, so this is always supplied.
    center = request["pocket_center"]
    argv += ["--pocket_center", *[str(float(c)) for c in center]]
    argv += ["--bbox_length", str(float(request.get("bbox_length", 23.0)))]
    if request.get("stochastic"):
        argv += ["--stochastic_sampling", "--noise_scaler", str(float(request.get("noise_scaler", 1.0)))]
    return argv


def run(request):
    # OMTRA takes no device argument: upstream's routines/sample.py decides for
    # itself with `device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`. So a
    # caller that asks for CPU can only be honoured by making the GPU invisible,
    # and that has to happen before torch is imported here -- CUDA reads
    # CUDA_VISIBLE_DEVICES once, when the context is created.
    #
    # This matters beyond taste. OMTRA pins torch 2.4.0+cu121, whose fatbins stop
    # at sm_90. On a newer card `torch.cuda.is_available()` still returns True --
    # the driver is present and graphs move to cuda:0 without complaint -- and the
    # first real kernel dies with "no kernel image is available for execution on
    # the device", minutes into a sample. Honouring an explicit `device: cpu`
    # turns that crash into the slow-but-real CPU run the warning below describes.
    if str(request.get("device") or "").startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    warnings = []
    out_dir = Path(request["out_dir"])
    sample_dir = out_dir / "omtra_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    protein_file = request.get("protein_file")
    if not protein_file or not Path(protein_file).is_file():
        raise FileNotFoundError(f"protein_file {protein_file!r} does not exist")

    # OMTRA picks its own device: routines/sample.py does
    # `device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`. Say so rather
    # than let a caller read a five-hour CPU run as a hang.
    if not torch.cuda.is_available():
        warnings.append(
            "no CUDA in the OMTRA environment; routines/sample.py falls back to CPU, "
            "where 250-step flow-matching integration is impractically slow"
        )

    center = request.get("pocket_center")
    center_source = "request"
    if not center:
        center = _protein_centroid(protein_file)
        center_source = "protein_centroid"
        warnings.append(
            f"no pocket centre was supplied; centred the box on the mean of {protein_file}'s "
            f"non-water heavy atoms at {[round(c, 3) for c in center]}"
        )
    request = dict(request, pocket_center=center)

    seed = int(request.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # OMTRA loads its own checkpoint deep inside routines/sample.py, so the
    # ``weights_only=False`` that pocket2mol_sample._torch_load and
    # pocketflow_sample apply at their own call sites cannot be applied here --
    # the call is upstream's. Default it instead, before upstream is imported.
    #
    # Needed only since this environment moved to torch >= 2.6 to reach sm_120:
    # torch 2.6 flipped ``torch.load`` to ``weights_only=True``, and OMTRA's
    # checkpoint pickles omegaconf ``ListConfig``/``DictConfig`` hyper-parameters,
    # so the load raises "Unsupported global: GLOBAL omegaconf.listconfig.ListConfig".
    # The alternative, add_safe_globals, has to name every class the checkpoint
    # happens to contain and breaks again on the next one.
    #
    # This is a *local* checkpoint fetched by us from OMTRA's own published URL and
    # size-verified on download, not arbitrary input, which is what makes relaxing
    # the guard acceptable here. It is scoped to this driver's process.
    # The override is unconditional, not a setdefault: OMTRA reaches the checkpoint
    # through PyTorch Lightning, and lightning_fabric.utilities.cloud_io._load
    # passes ``weights_only=True`` EXPLICITLY, so a default would never be consulted.
    _real_torch_load = torch.load

    def _torch_load_trusted(*args, **kwargs):
        kwargs["weights_only"] = False
        return _real_torch_load(*args, **kwargs)

    torch.load = _torch_load_trusted

    import cli  # OMTRA's entry point: pyproject.toml [project.scripts] omtra = "cli:main"

    args = cli.create_parser().parse_args(_build_argv(request, sample_dir))
    try:
        cli.run_sample(args)
    except SystemExit as exc:
        # cli.run_sample calls sys.exit(1) on a missing checkpoint or an
        # unsatisfied task input. Turn that into a structured failure instead of
        # a bare non-zero exit with no result.json.
        raise RuntimeError(
            f"cli.run_sample exited with status {exc.code} — the checkpoint was not "
            "found or the task's required inputs were not supplied"
        ) from exc

    # routines/sample.py writes one gen_ligands.sdf per system, under
    # `output_dir / f"sys_{sys_id}_gt"`. One input file means one system.
    sdf_paths = sorted(sample_dir.glob("sys_*_gt/gen_ligands.sdf"))
    if not sdf_paths:
        raise FileNotFoundError(
            f"OMTRA wrote no sys_*_gt/gen_ligands.sdf under {sample_dir}; "
            f"'ligand_structure' may not be in this task's groups_generated"
        )

    records = []
    for path in sdf_paths:
        records.extend(_read_generated(path, Chem))
    n_allene, n_small_ring = _reconstruction_counts(records, Chem)

    molecules = [
        {
            "smiles": record["smiles"],
            "sdf": record["sdf"],
            "sample_index": index,
            "n_atoms": record["n_atoms"],
            "qc_flags": [],
        }
        for index, record in enumerate(records)
    ]
    n_sanitizable = sum(1 for record in records if record["smiles"])

    return {
        "molecules": molecules,
        "n_written": len(records),
        "n_sanitizable": n_sanitizable,
        "n_allene": n_allene,
        "n_small_ring": n_small_ring,
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "n_timesteps": int(args.n_timesteps),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": seed,
        "seed_applied": "random, numpy, torch (OMTRA's CLI exposes no --seed)",
        "sdf_paths": [str(p) for p in sdf_paths],
        "pocket_center": center,
        "pocket_center_source": center_source,
        "bbox_length": float(args.bbox_length),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
