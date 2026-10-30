"""Sample ligands for one pocket with MolPilot — the drug-agent ⇄ MolCRAFT bridge.

Runs inside whichever interpreter has MolPilot's dependencies. Imports nothing
from ``drug_agent``; the only non-stdlib imports are ``yaml`` and ``rdkit``, both
of which are MolPilot's own dependencies (``core/config/config.py`` line 1 and
``sample_for_pocket.py`` line 38), and both are imported lazily so a missing one
becomes a structured failure rather than a traceback.

**Which upstream entry point this drives.** ``MolPilot/sample_for_pocket.py`` at
MolCRAFT ``743dfb5198c4``. It is a ``__main__`` script with no importable
function — unlike ``MolCRAFT/sample_for_pocket.py``, which exposes ``call()`` at
line 98 — so it is launched as a subprocess with the argv upstream documents in
``MolPilot/README.md`` under "Sampling from pdb file"::

    python sample_for_pocket.py --protein_path ${PDB_PATH} --ligand_path ${SDF_PATH} \\
        --time_scheduler_path ${TIME_SCHEDULER} --num_samples ${NUM_MOLS_PER_POCKET}

plus ``--output_dir`` and ``--res_path``, which the script declares
``required=True`` (lines 253-254) although the README omits them.

**Why this driver writes a config instead of only passing flags.** Three facts
from the checkout, in this order:

1. ``sample_for_pocket.py`` line 275 does
   ``tr_cfg = Config(cfg.accounting.dump_config_path)`` and takes the network
   architecture from it. Sampling therefore needs the YAML a *training run*
   dumped, not just a sampling config. Upstream publishes that dump as
   ``molpilot_config.yaml`` on https://huggingface.co/GenSI/MolPilot.
2. That dump is a plain resolved YAML with no ``!SUB`` nodes, and
   ``core/config/config.py`` substitutes command-line values only into ``!SUB``
   nodes. Passing it as ``--config_file`` makes ``--protein_path``,
   ``--ligand_path``, ``--num_samples``, ``--sample_steps``, ``--eval_batch_size``
   and ``--time_scheduler_path`` inert. The real values have to be *in* the file.
3. Its ``accounting.*`` keys point at ``/sharefs/qiukeyue/project/logs/...``, the
   authors' cluster. Line 355 globs ``accounting.checkpoint_dir`` for
   ``*mol_stable*`` to find the weights — note that ``--ckpt_path`` (line 180) is
   declared and never read — so that directory has to exist locally and hold the
   checkpoint under a name carrying ``mol_stable`` and ``val_loss``.

So the driver materialises a scratch run directory, copies the published dump
into it with ``accounting`` and ``evaluation`` rewritten, links the checkpoint
into ``<rundir>/checkpoints/`` under its published basename, and passes that one
file as both ``--config_file`` and (by construction) ``dump_config_path``, which
makes ``cfg.dynamics`` and ``tr_cfg.dynamics`` the same block. The documented
flags are passed as well, unchanged, so the command matches upstream's; they are
inert for the reason above, which is exactly why the config carries the values.

Nothing about the model is reimplemented here: the schedule, the sampler, the
reconstruction and the SDF writing are all upstream's
(``core/models/sbdd4train_timewarp.py`` line 451 writes each molecule).

Request JSON:
    molpilot_dir     absolute path of the ``MolPilot`` folder in the checkout
    protein_path     whole receptor PDB (not a pre-cut pocket: upstream clips it)
    ligand_path      reference ligand SDF that defines the pocket and atom count
    checkpoint       molpilot_epoch26-val_loss...-mol_stable...-complete....ckpt
    time_scheduler   optimized_path_rect_closest.pt (the VOS schedule tensor)
    train_config     molpilot_config.yaml (the dumped training config)
    n_molecules      molecules to sample for this pocket
    batch_size       evaluation.batch_size
    sample_steps     evaluation.sample_steps; the schedule tensor's shape is
                     asserted to be (sample_steps + 1, 2)
    seed             RNG seed
    device           e.g. "cuda:0"
    skip_eval        skip MolPilot's own Vina/PoseBusters pass
    out_dir          where to write result.json

Response JSON (``<out_dir>/result.json``):
    molecules[]      {smiles, sdf, qc_flags[]}
    n_requested, n_sdf, warnings[], errors[]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: The dumped checkpoint name must survive two upstream string operations:
#: ``glob("*mol_stable*")`` at sample_for_pocket.py:355 and the sort key
#: ``float(x.split("val_loss")[-1][:4])`` at line 357.
_CKPT_GLOB_TOKEN = "mol_stable"
_CKPT_SORT_TOKEN = "val_loss"

#: exp_name/revision only name the scratch run directory. They are written into
#: the config rather than derived from ``!SUB``, so their values are cosmetic.
_EXP_NAME = "drugagent"
_REVISION = "molpilot_sample"


def _require_file(path_s, what):
    """Return *path_s* as a Path, or raise naming what is missing and why."""
    if not path_s:
        raise ValueError(f"request is missing {what}")
    path = Path(path_s)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found at {path}")
    return path


def _link_or_copy(src, dst):
    """Symlink *src* to *dst*, falling back to a copy on a filesystem that cannot."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(str(src), str(dst))


def _localised_config(train_config, rundir, request, protein, ligand, scheduler):
    """The published training dump with ``accounting`` and ``evaluation`` rewritten.

    Only those two blocks change. ``dynamics`` is left exactly as trained — it is
    what the checkpoint's weights were shaped by — and ``data`` is left pointing
    at the authors' CrossDock paths because ``get_dataloader_from_pdb``
    (sample_for_pocket.py lines 46-95) builds the test set from the PDB and never
    opens ``data.path``; only ``data.atom_decoder``, ``normalizer_dict``,
    ``colors_dic``, ``radius_dic`` and ``transform.ligand_atom_mode`` are read,
    and those are values, not paths.
    """
    import yaml

    with train_config.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{train_config} is not a YAML mapping")

    cfg["accounting"] = {
        "logdir": str(rundir),
        "wandb_logdir": str(rundir),
        # Read back by sample_for_pocket.py:275 as the training config. Pointing it
        # at the very file being passed as --config_file keeps cfg.dynamics and
        # tr_cfg.dynamics identical, which is what the checkpoint expects.
        "dump_config_path": str(rundir / "config.yaml"),
        # Globbed for the weights at line 355.
        "checkpoint_dir": str(rundir / "checkpoints"),
        "generated_mol_dir": str(rundir / "generated_mol"),
        # DockingTestCallback.on_test_start (validation_callback.py:2166) writes here.
        "test_outputs_dir": str(rundir / "test_outputs"),
    }

    evaluation = dict(cfg.get("evaluation") or {})
    evaluation["protein_path"] = str(protein)
    evaluation["ligand_path"] = str(ligand)
    # test_set = [transform(data)] * cfg.evaluation.num_samples (line 81): this is
    # the molecule count, and the script resets num_samples to 1 straight after.
    evaluation["num_samples"] = int(request.get("n_molecules") or 10)
    evaluation["batch_size"] = int(request.get("batch_size") or 10)
    evaluation["sample_steps"] = int(request.get("sample_steps") or 100)
    evaluation["time_scheduler_path"] = str(scheduler)
    cfg["evaluation"] = evaluation

    cfg["seed"] = int(request.get("seed") or 0)
    cfg["test_only"] = True
    cfg["no_wandb"] = True
    cfg["debug"] = False
    cfg["empty_folder"] = False  # true would rmtree accounting.logdir (line 258)
    cfg["exp_name"] = _EXP_NAME
    cfg["revision"] = _REVISION
    # best_ckpt selects the sort at line 357, which splits the basename on
    # "val_loss"; the published checkpoint name carries that token.
    cfg["best_ckpt"] = _CKPT_SORT_TOKEN
    # validation_callback.py:2059 — skip MolPilot's own Vina/PoseBusters scoring.
    # It does not remove the imports, only the run: this platform docks and busts
    # with its own tools rather than letting a generator grade itself.
    cfg["skip_eval"] = bool(request.get("skip_eval", True))

    rundir.mkdir(parents=True, exist_ok=True)
    config_path = rundir / "config.yaml"
    with config_path.open("w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False)
    return config_path


def _child_env(molpilot_dir, device):
    """The environment the upstream script runs in."""
    env = dict(os.environ)
    parts = [str(molpilot_dir)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # get_logger() builds a WandbLogger unconditionally (line 96). offline=no_wandb
    # already keeps it local; this makes the no-egress guarantee independent of the
    # config being read correctly.
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    if device and device.startswith("cuda") and ":" in device:
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return env


def _sdf_sort_key(path):
    """Order ``{batch_idx}_{i}_{idx}.sdf`` numerically, not lexically."""
    parts = path.stem.split("_")
    try:
        return (0, tuple(int(p) for p in parts))
    except ValueError:
        return (1, path.stem)


def _read_molecules(sdf_dir):
    """Every SDF upstream wrote, as ``(records, warnings)``.

    A molecule RDKit cannot sanitize is kept with a qc flag rather than dropped:
    the base adapter computes validity from the raw count, so silently discarding
    the failures here would inflate it.

    ``removeHs=False`` keeps the hydrogens upstream wrote, because the SDF block
    is the 3D pose and losing them would change it. The SMILES that comes out
    therefore carries explicit ``[H]``; that is deliberate and harmless, since
    :meth:`GenerativeToolAdapter._screen` re-parses and canonicalises every SMILES
    itself and RDKit's SMILES parser absorbs them.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    records = []
    warnings = []
    for path in sorted(sdf_dir.glob("*.sdf"), key=_sdf_sort_key):
        block = path.read_text()
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
        flags = []
        if mol is None:
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            flags.append("rdkit_sanitize_failed")
        if mol is None:
            warnings.append(f"{path.name} could not be parsed as a molblock at all")
            records.append({"smiles": "", "sdf": block, "qc_flags": ["unparseable_molblock"]})
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception as exc:
            warnings.append(f"{path.name}: MolToSmiles failed: {exc}")
            smiles = ""
            flags.append("smiles_write_failed")
        records.append({"smiles": smiles, "sdf": block, "qc_flags": flags})
    return records, warnings


def run(request):
    molpilot_dir = _require_file(
        Path(request.get("molpilot_dir") or ".") / "sample_for_pocket.py",
        "MolPilot/sample_for_pocket.py",
    ).parent
    protein = _require_file(request.get("protein_path"), "protein PDB (protein_path)")
    ligand = _require_file(request.get("ligand_path"), "reference ligand SDF (ligand_path)")
    checkpoint = _require_file(request.get("checkpoint"), "MolPilot checkpoint (checkpoint)")
    scheduler = _require_file(request.get("time_scheduler"), "VOS schedule tensor (time_scheduler)")
    train_config = _require_file(request.get("train_config"), "dumped training config (train_config)")

    name = checkpoint.name
    if _CKPT_GLOB_TOKEN not in name or _CKPT_SORT_TOKEN not in name:
        raise ValueError(
            f"checkpoint basename {name!r} must contain {_CKPT_GLOB_TOKEN!r} and "
            f"{_CKPT_SORT_TOKEN!r}: sample_for_pocket.py finds the weights by globbing "
            f"'*{_CKPT_GLOB_TOKEN}*' (line 355) and sorts the matches on the digits after "
            f"'{_CKPT_SORT_TOKEN}' (line 357), so a renamed checkpoint is not found or not parsed"
        )

    out_dir = Path(request["out_dir"])
    rundir = out_dir / "run"
    sdf_dir = out_dir / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "molpilot_metrics.json"

    _link_or_copy(checkpoint, rundir / "checkpoints" / name)
    config_path = _localised_config(train_config, rundir, request, protein, ligand, scheduler)

    argv = [
        sys.executable,
        "sample_for_pocket.py",
        "--config_file",
        str(config_path),
        # required=True at lines 253-254; the only two values the script reads from
        # argv after the config is loaded (lines 376-377).
        "--output_dir",
        str(sdf_dir),
        "--res_path",
        str(res_path),
        # From MolPilot/README.md's documented invocation. Inert against a dumped
        # config (no !SUB nodes to substitute into) — the config above carries the
        # real values — but passed so the command is upstream's, not a variant.
        "--protein_path",
        str(protein),
        "--ligand_path",
        str(ligand),
        "--time_scheduler_path",
        str(scheduler),
        "--num_samples",
        str(int(request.get("n_molecules") or 10)),
        "--sample_steps",
        str(int(request.get("sample_steps") or 100)),
        "--eval_batch_size",
        str(int(request.get("batch_size") or 10)),
        "--seed",
        str(int(request.get("seed") or 0)),
        "--exp_name",
        _EXP_NAME,
        "--revision",
        _REVISION,
        "--test_only",
        "--no_wandb",
    ]

    log_path = out_dir / "sample_for_pocket.log"
    with log_path.open("wb") as log:
        proc = subprocess.run(
            argv,
            cwd=str(molpilot_dir),
            env=_child_env(molpilot_dir, request.get("device")),
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    records, warnings = _read_molecules(sdf_dir)
    tail = "\n".join(log_path.read_text(errors="replace").strip().splitlines()[-20:])
    if not records:
        # Nothing on disk is a failed run, whatever the exit code said. Raising here
        # lets _driver.main write a structured failure the adapter re-raises as
        # GenerationBackendError, instead of an empty batch that reads as a healthy
        # zero three artifacts downstream.
        raise RuntimeError(
            f"sample_for_pocket.py wrote no .sdf into {sdf_dir} (exit {proc.returncode}):\n{tail}"
        )
    if proc.returncode != 0:
        # Molecules exist but the script died afterwards — the likely culprit is its
        # own post-hoc Metrics pass (lines 386-400), where PoseCheck() is constructed
        # outside the try. The samples are still upstream's; the failure is recorded.
        warnings.append(
            f"sample_for_pocket.py exited {proc.returncode} after writing {len(records)} molecules:\n{tail}"
        )
    return {
        "molecules": records,
        "n_requested": int(request.get("n_molecules") or 0),
        "n_sdf": len(records),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
