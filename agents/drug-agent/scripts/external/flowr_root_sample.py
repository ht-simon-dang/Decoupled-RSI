"""Sample ligands with FLOWR.root — the drug-agent ⇄ flowr_root bridge.

Runs inside the ``flowr_root`` environment (conda ``flowr_root``, built from the
checkout's ``environment.yml``), never inside drug-agent's. It imports nothing from
``drug_agent``: the two sides talk over a JSON request/response pair on disk, so a
CUDA fault or an OOM inside FLOWR cannot reach the caller.

**How the upstream entry point is invoked.** Not re-implemented. This driver sets
``sys.argv`` to exactly the flags upstream's own SLURM scripts use and then calls
the upstream module's own ``get_args()`` and ``evaluate()``. Every default —
``--integration_steps``, ``--bucket_cost_scale``, ``--solver``,
``--cat_sampling_noise_level`` and the forty others — therefore comes from the
upstream parser rather than from a hand-copied dict that can drift from it.

Where each flag comes from:

* the flag set and their values: ``scripts/generate_pdb.sl`` (pocket mode) and
  ``scripts/generate_sdf.sl`` (ligand-only mode);
* ``--arch`` / ``--pocket_type`` are ``required=True`` in
  ``flowr/gen/generate_from_pdb.py::get_args``; ``--arch`` is required in
  ``flowr/gen/generate_from_sdf_mol.py::get_args`` and its choices there are
  ``flowr`` / ``transformer``;
* ``--pocket_cutoff 7`` is the README's documented default for cutting a pocket
  out of a protein PDB;
* ``--sample_n_molecules_per_mol`` (not ``--sample_n_molecules``) because
  ``flowr/gen/utils.py::load_data_from_sdf_mol`` asserts exactly one of the two is
  set, and per-mol is what turns one reference into N samples;
* the output SDF names are the ones ``evaluate()`` writes:
  ``samples_<pdb stem>.sdf`` in ``generate_from_pdb.py``, and
  ``samples_<sdf stem>_unprocessed.sdf`` in ``generate_from_sdf_mol.py``.

**What is deliberately not passed.** ``--filter_valid_unique``,
``--filter_diversity`` and the PoseBusters/strain post-processing flags. The
calling adapter computes validity, uniqueness and the heavy-atom window itself, in
one place, for every generator it owns; letting each upstream apply its own filters
first would make those numbers incomparable. Note that ``evaluate()`` still
sanitises its own output on the ``not args.filter_valid_unique`` branch, so the
returned set is post-sanitisation either way — a warning saying so is attached to
every payload rather than left for a reader to infer.

Request JSON:
    mode            "pocket" | "ligand"
    entry_point     upstream module to run, e.g. "flowr.gen.generate_from_pdb"
    arch            value for --arch ("pocket" | "flowr")
    checkpoint      absolute path to the .ckpt for this mode
    protein_pdb     protein/pocket PDB (pocket mode only)
    reference_sdf   reference ligand; --ligand_file (pocket) or --sdf_path (ligand)
    n_molecules     molecules to sample
    batch_cost      value for --batch_cost
    seed            RNG seed
    device          e.g. "cuda:0"
    num_workers     dataloader workers (optional, default 4)
    out_dir         where to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]     {smiles, sdf, sample_index, n_atoms, qc_flags[], affinity{}}
    n_requested, n_returned, mode, checkpoint, sdf_path, run_seconds,
    warnings[], errors[]
"""

from __future__ import annotations

import importlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: Affinity-head properties FLOWR.root writes onto each generated molecule
#: (``flowr/gen/generate.py``). Read back verbatim; the adapter does not turn them
#: into a Prediction.
_AFFINITY_PROPS = ("pic50", "pki", "pkd", "pec50")

#: Upstream sanitises inside ``evaluate()`` regardless of ``--filter_valid_unique``
#: (see ``flowr/gen/generate_from_pdb.py``), so the caller's validity metric is
#: measured over an already-cleaned set. Said out loud, on every batch.
_SANITISED_UPSTREAM = (
    "FLOWR.root's evaluate() calls smolRD.sanitize_list(..., sanitize=True) on its own "
    "output before writing the SDF, so generation_validity here is measured over molecules "
    "that already passed upstream's sanitiser and is not the model's raw reconstruction "
    "validity."
)


def _build_argv(request):
    """The exact command line, mirroring upstream's own generate_*.sl scripts."""
    mode = request["mode"]
    save_dir = str(Path(request["out_dir"]) / "samples")
    n = int(request.get("n_molecules", 10))
    batch_cost = max(1, int(request.get("batch_cost", 10)))
    workers = max(0, int(request.get("num_workers", 4)))
    seed = int(request.get("seed", 42))
    # Flag/value pairs stay on one line, as upstream's own get_args() does, so this
    # reads against scripts/generate_pdb.sl and scripts/generate_sdf.sl line for line.
    # fmt: off
    common = [
        "--ckpt_path", str(request["checkpoint"]),
        "--save_dir", save_dir,
        "--gpus", "1",
        "--mp_index", "0",
        "--num_workers", str(workers),
        "--batch_cost", str(batch_cost),
        "--seed", str(seed),
        "--categorical_strategy", "uniform-sample",
        "--ode_sampling_strategy", "linear",
    ]
    if mode == "pocket":
        # scripts/generate_pdb.sl: cut a 7 Å pocket around the reference ligand,
        # 0.1 coordinate noise, up to 30 resampling iterations.
        argv = [
            *common,
            "--pdb_file", str(request["protein_pdb"]),
            "--ligand_file", str(request["reference_sdf"]),
            "--arch", str(request.get("arch", "pocket")),
            "--pocket_type", "holo",
            "--cut_pocket",
            "--pocket_cutoff", "7",
            "--max_sample_iter", "30",
            "--coord_noise_scale", "0.1",
            "--sample_n_molecules_per_target", str(n),
        ]
    else:
        # scripts/generate_sdf.sl: one reference ligand (index 0), 0.2 coordinate
        # noise, up to 20 resampling iterations.
        argv = [
            *common,
            "--sdf_path", str(request["reference_sdf"]),
            "--ligand_idx", "0",
            "--arch", str(request.get("arch", "flowr")),
            "--max_sample_iter", "20",
            "--coord_noise_scale", "0.2",
            "--sample_n_molecules_per_mol", str(n),
        ]
    # fmt: on
    return argv, save_dir


def _stage_reference(request, save_dir):
    """Copy the reference ligand into the scratch directory before it is parsed.

    ``flowr/data/dataset.py::GeometricMolSDFDataset`` writes a ``sdf_mol_data.smol``
    cache *beside the SDF it was given* and reuses it on the next run purely on the
    strength of that file existing — it never checks that it came from this SDF. A
    reference sitting in a shared or read-only directory would therefore either fail
    to write or, worse, silently sample from somebody else's molecule. Staging a copy
    per run makes the cache private to the run.
    """
    src = Path(request["reference_sdf"])
    dest_dir = Path(save_dir) / "reference"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copyfile(str(src), str(dest))
    return dest


def _read_samples(sdf_path):
    """Parse what ``evaluate()`` wrote, keeping the affinity properties verbatim."""
    from rdkit import Chem

    molecules = []
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
    for index, mol in enumerate(supplier):
        if mol is None:
            # Upstream already sanitised; an unreadable record here is a write-side
            # failure, recorded as a flag rather than dropped silently.
            molecules.append(
                {
                    "smiles": "",
                    "sdf": None,
                    "sample_index": index,
                    "n_atoms": 0,
                    "qc_flags": ["sdf_record_unreadable"],
                    "affinity": {},
                }
            )
            continue
        affinity = {}
        for prop in _AFFINITY_PROPS:
            if mol.HasProp(prop):
                try:
                    affinity[prop] = float(mol.GetProp(prop))
                except ValueError:
                    affinity[prop] = mol.GetProp(prop)
        molecules.append(
            {
                "smiles": Chem.MolToSmiles(mol),
                "sdf": Chem.MolToMolBlock(mol),
                "sample_index": index,
                "n_atoms": int(mol.GetNumAtoms()),
                "qc_flags": [],
                "affinity": affinity,
            }
        )
    return molecules


def run(request):
    import torch
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    mode = request.get("mode")
    if mode not in ("pocket", "ligand"):
        raise ValueError(f"request mode must be 'pocket' or 'ligand', got {mode!r}")

    checkpoint = Path(request["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"FLOWR.root checkpoint not found at {checkpoint}. The checkouts carry code only; "
            "the released checkpoints are downloaded separately from the Google Drive folder "
            "linked in the FLOWR.root README."
        )
    if not torch.cuda.is_available():
        # flowr/gen/generate_from_pdb.py and generate_from_sdf_mol.py both call
        # model.to("cuda") unconditionally. Failing here names the reason; failing
        # there produces a bare torch device assertion.
        raise RuntimeError(
            "FLOWR.root requires CUDA: upstream's evaluate() calls model.to('cuda') "
            f"unconditionally. torch.cuda.is_available() is False in {sys.executable}."
        )

    argv, save_dir = _build_argv(request)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if mode == "ligand":
        staged = _stage_reference(request, save_dir)
        argv[argv.index("--sdf_path") + 1] = str(staged)
        target_stem = staged.stem
        out_sdf = Path(save_dir) / f"samples_{target_stem}_unprocessed.sdf"
    else:
        target_stem = Path(request["protein_pdb"]).stem
        out_sdf = Path(save_dir) / f"samples_{target_stem}.sdf"

    entry_point = request.get("entry_point") or (
        "flowr.gen.generate_from_pdb" if mode == "pocket" else "flowr.gen.generate_from_sdf_mol"
    )
    module = importlib.import_module(entry_point)

    saved_argv = sys.argv
    started = time.time()
    try:
        # Upstream's own parser, so every default this driver does not name is the
        # one the upstream script would have used.
        sys.argv = [entry_point, *argv]
        args = module.get_args()
        module.evaluate(args)
    finally:
        sys.argv = saved_argv
    run_seconds = time.time() - started

    warnings = [_SANITISED_UPSTREAM]
    if not out_sdf.is_file():
        raise FileNotFoundError(
            f"FLOWR.root finished in {run_seconds:.1f}s but wrote no {out_sdf}; "
            "the run produced nothing to read."
        )
    molecules = _read_samples(out_sdf)
    n_requested = int(request.get("n_molecules", 10))
    if len(molecules) < n_requested:
        warnings.append(
            f"FLOWR.root returned {len(molecules)} of {n_requested} requested molecules; "
            "upstream stops at --max_sample_iter."
        )

    return {
        "molecules": molecules,
        "n_requested": n_requested,
        "n_returned": len(molecules),
        "mode": mode,
        "arch": request.get("arch"),
        "entry_point": entry_point,
        "checkpoint": str(checkpoint),
        "sdf_path": str(out_sdf),
        "run_seconds": round(run_seconds, 3),
        "device": request.get("device"),
        "seed": int(request.get("seed", 42)),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
