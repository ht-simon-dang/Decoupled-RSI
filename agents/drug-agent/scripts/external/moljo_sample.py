"""Sample optimised ligands for one pocket with MolJO — the drug-agent ⇄ MolJO bridge.

MolJO lives inside the MolCRAFT monorepo (``GenSI-THUAIR/MolCRAFT``) as the
``MolJO/`` subtree; there is no separate ``GenSI-THUAIR/MolJO`` repository. Its
pocket entry point is ``MolJO/sample_for_pocket_guided.py``, and everything this
driver does is taken from that file and the two it depends on:

``MolJO/sample_for_pocket_guided.py``
    ``get_dataloader_from_pdb`` (lines 39-87) — the protein/reference-ligand front
    end, imported and called here rather than re-implemented — and the ``__main__``
    block (lines 112-313), whose sequence this driver reproduces: build the config
    from ``configs/test_opt.yaml``, seed, load the *dumped training* config over it,
    construct ``BFNTrainLoop``, attach the callbacks, load one property-predictor
    per objective, ``configure_classifiers``, then ``trainer.test``.

``MolJO/core/config/config.py``
    ``Config(config_file, **kwargs)`` — a YAML loader with ``!SUB`` substitution
    from the argparse namespace, ``!ENV`` from the environment, and ``!PATHJOIN``.
    Every ``!SUB ${name}`` in ``configs/test_opt.yaml`` must therefore appear as a
    key of the dict passed to ``Config``; ``ARGPARSE_DEFAULTS`` below is that dict,
    copied field by field from the entry point's ``add_argument`` defaults.

``MolJO/core/callbacks/basic.py``
    ``NormalizerCallback`` divides ``protein_pos``/``ligand_pos`` by
    ``data.normalizer_dict.pos`` in ``on_test_batch_start``, and ``EMACallback``
    swaps the checkpoint's EMA weights in at ``on_test_start``. Both change the
    molecules that come out, so both are kept.

**Why the ``__main__`` block is reproduced instead of executed.** Line 286 of
``sample_for_pocket_guided.py`` hard-codes ``work_dir =
'/sharefs/qiukeyue/project/BFN4MolOpt'`` and loads every property-predictor
checkpoint from under it, so that script cannot run on any machine but the
authors' cluster. Its sibling ``sample_guided.py`` (line 221) sets ``work_dir =
'.'`` for the same load, which agrees with ``MolJO/README.md`` ("please put the
checkpoints under the ``pretrained`` folder"). This driver follows the README and
``sample_guided.py``: predictors are read from ``<moljo_root>/pretrained``.

**Which upstream callbacks are dropped, and why that is safe.** The entry point
also attaches ``CondMolGenValidationCallback``, ``MolVisualizationCallback``,
``ReconValidationCallback``, ``DockingTestCallback`` and ``ModelCheckpoint``. Of
those only ``DockingTestCallback`` touches test output, and what it does there
(``core/callbacks/validation_callback.py`` lines 578-620) is run AutoDock Vina
over the samples and write scored SDFs. Generation must not depend on the
evaluator, and this platform docks with its own capability, so this driver
collects the very same ``out_data_list`` that callback collects — via the same
``on_test_batch_end`` hook — and writes SMILES and mol blocks without scoring
them. Nothing dropped alters the sampled coordinates or atom types.

Usage (the adapter does this for you)::

    PYTHONPATH=$MOLCRAFT_ROOT:$MOLCRAFT_ROOT/MolJO \\
        python scripts/external/moljo_sample.py request.json

Request JSON:
    moljo_root        ``<MolCRAFT checkout>/MolJO``
    config_file       sampling config, normally ``configs/test_opt.yaml``
    protein_path      receptor PDB (whole protein or pocket; MolJO clips it)
    ligand_path       reference ligand SDF — REQUIRED, see below
    pretrained_dir    directory holding the property-predictor checkpoints
    backbone_ckpt     optional explicit backbone checkpoint
    checkpoint_dir    directory globbed for the backbone when it is not given
    train_config      the training run's dumped ``config.yaml``
    n_molecules       molecules to sample (``evaluation.num_samples``)
    batch_size        pockets per forward pass (``evaluation.batch_size``)
    seed, device, sample_steps, sample_num_atoms
    objective         ``vina`` / ``sa`` / ``qed`` joined by ``_``, e.g. ``vina_sa``
    guide_mode, pos_grad_weight, type_grad_weight
    out_dir           where ``result.json`` is written

``ligand_path`` is not optional. ``get_dataloader_from_pdb`` asserts on it and
uses it twice: to clip the pocket to residues within ``dynamics.net_config.r_max``
(10 Å) of the reference, and as the ligand half of the ``ProteinLigandData`` pair
the sampler starts from. MolJO optimises an existing molecule; it is not an
unconditional de-novo sampler and cannot be run as one.

Response JSON (``out_dir/result.json``):
    molecules[]       {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_requested, n_sampled, n_reconstruction_failures
    backbone_ckpt, objective, warnings[], errors[]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: Every ``add_argument`` default in ``MolJO/sample_for_pocket_guided.py`` lines
#: 116-176, copied verbatim. ``Config`` substitutes each ``!SUB ${name}`` in
#: ``configs/test_opt.yaml`` from this dict, so a missing key is a hard failure at
#: config-parse time rather than a silently wrong sampler setting.
ARGPARSE_DEFAULTS = {
    "config_file": "configs/test_opt.yaml",
    "exp_name": "debug",
    "revision": "default",
    "debug": False,
    "resume": False,
    "wandb_resume_id": None,
    "empty_folder": False,
    "test_only": True,
    "seed": 1234,
    "no_wandb": True,
    "logging_level": "warning",
    "random_rot": False,
    "pos_noise_std": 0,
    "pos_normalizer": 2.0,
    "batch_size": 8,
    "epochs": 10,
    "v_loss_weight": 1,
    "lr": 5e-4,
    "scheduler": "plateau",
    "weight_decay": 0,
    "max_grad_norm": "Q",
    "sigma1_coord": 0.03,
    "beta1": 1.5,
    "t_min": 0.0001,
    "use_discrete_t": False,
    "discrete_steps": 1000,
    "destination_prediction": False,
    "sampling_strategy": "end_back_pmf",
    "time_emb_mode": "simple",
    "time_emb_dim": 1,
    "pos_init_mode": "zero",
    "num_samples": 10,
    "sample_steps": 200,
    "sample_num_atoms": "prior",
    "visual_chain": False,
    "protein_path": None,
    "ligand_path": None,
    "last_ckpt": False,
    "docking_mode": "vina_score",
    "save_traj": False,
    # `sample_for_pocket_guided.py` defaults this to None, which makes
    # `cfg.evaluation.objective.split('_')` raise. `sample_guided.py` line 195 and
    # MolJO/README.md's sampling recipe both use vina_sa, so that is the default here.
    "objective": "vina_sa",
    "guide_mode": "param_naive",
    # README.md sampling recipe: --pos_grad_weight 50 --type_grad_weight 50.
    # The entry point's own argparse defaults are 1.0, which is not the published setting.
    "pos_grad_weight": 50.0,
    "type_grad_weight": 50.0,
    "interpolate_coef": 0.5,
}

#: Objective token -> property-predictor checkpoint stem, from
#: ``sample_for_pocket_guided.py`` lines 293-306 (``vina`` reads ``affinity``).
OBJECTIVE_STEMS = {"vina": "affinity", "qed": "qed", "sa": "sa"}


def _require_file(path, what):
    """Return *path* as a resolved :class:`Path`, or explain exactly what is absent."""
    if not path:
        raise RuntimeError(f"MolJO needs {what} and none was supplied")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise RuntimeError(f"MolJO {what} not found at {resolved}")
    return resolved.resolve()


def _find_backbone(explicit, checkpoint_dir):
    """The backbone checkpoint, chosen the way the entry point chooses it.

    ``sample_for_pocket_guided.py`` lines 276-283 globs ``*mol_stable*`` in
    ``cfg.accounting.checkpoint_dir`` and falls back to ``*complete*``, sorting on
    the metric embedded in the filename by the training ``ModelCheckpoint``. Both
    patterns are kept so a checkpoint saved by upstream's own trainer is found
    under either name.
    """
    if explicit:
        return _require_file(explicit, "backbone checkpoint")
    if not checkpoint_dir:
        raise RuntimeError("MolJO needs either backbone_ckpt or checkpoint_dir")
    ckpt_dir = Path(checkpoint_dir).expanduser()
    for pattern, key in (("*mol_stable*", "mol_stable"), ("*complete*", "complete")):
        found = sorted(ckpt_dir.glob(pattern))
        if found:
            return sorted(found, key=lambda p: _metric_of(p.name, key))[-1].resolve()
    raise RuntimeError(
        f"no MolJO backbone checkpoint in {ckpt_dir}: nothing matches '*mol_stable*' "
        "or '*complete*'. MolJO/README.md links the backbone checkpoint from Google "
        "Drive; download it into that directory."
    )


def _metric_of(name, key):
    """The float the training ``ModelCheckpoint`` wrote after *key* in a filename."""
    try:
        return float(name.split(key)[-1][:4])
    except (IndexError, ValueError):
        return float("-inf")


def _load_classifiers(objective, pretrained_dir, cfg, Config, ClassifierTrainLoop):
    """One property predictor per objective token, as the entry point loads them.

    Mirrors ``sample_for_pocket_guided.py`` lines 285-310, including the ``postfix``
    that a ``classifier.input_type`` in the training config appends to the
    checkpoint stem (``_noisy`` for ``sample``, ``_x0`` for ``data``).
    """
    postfix = ""
    classifier_cfg = getattr(cfg, "classifier", None)
    input_type = getattr(classifier_cfg, "input_type", None) if classifier_cfg is not None else None
    if input_type == "sample":
        postfix = "_noisy"
    elif input_type == "data":
        postfix = "_x0"

    objectives = str(objective).split("_")
    modules = []
    for token in objectives:
        stem = OBJECTIVE_STEMS.get(token)
        if stem is None:
            supported = ", ".join(sorted(OBJECTIVE_STEMS))
            raise RuntimeError(
                f"MolJO objective {token!r} is not supported by this checkout; "
                f"sample_for_pocket_guided.py implements {supported}"
            )
        obj_yaml = _require_file(
            pretrained_dir / f"{stem}{postfix}.yaml",
            f"{token} property-predictor config",
        )
        obj_ckpt = _require_file(
            pretrained_dir / f"{stem}{postfix}.ckpt",
            f"{token} property-predictor checkpoint",
        )
        module = ClassifierTrainLoop.load_from_checkpoint(str(obj_ckpt), config=Config(str(obj_yaml)))
        module.eval()
        modules.append(module.dynamics)
    return objectives, modules


def _collector(Callback):
    """A callback that keeps what ``test_step`` returned, and nothing else.

    The two hooks are the ones ``DockingTestCallback`` uses
    (``core/callbacks/validation_callback.py`` lines 557-578), so the list this
    collects is the same ``out_data_list`` upstream would have docked and written.
    """

    class _Collect(Callback):
        def __init__(self):
            super().__init__()
            self.outputs = []

        def on_test_start(self, trainer, pl_module):
            self.outputs = []

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.outputs.extend(outputs)

    return _Collect()


def _records_from(graphs, Chem):
    """Turn the sampled graphs into ``{smiles, sdf, ...}``, counting what failed.

    ``train_loop.shared_sampling_step`` (line 519) sets ``out_batch.mol`` to
    ``reconstruct.reconstruct_from_generated(...)`` per graph, or ``None`` when
    ``MolReconsError`` was raised. A ``None`` is a reconstruction failure, which is
    a fact about the sample, not a malfunction — it is counted and reported.
    """
    records = []
    failures = 0
    for index, graph in enumerate(graphs):
        mol = getattr(graph, "mol", None)
        if mol is None:
            failures += 1
            continue
        flags = []
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:  # an unsanitizable graph is data about the sample, not a crash
            smiles = ""
            flags.append("moljo_smiles_failed")
        try:
            block = Chem.MolToMolBlock(mol)
        except Exception:  # same: a mol block that will not write is a flagged sample
            block = None
            flags.append("moljo_molblock_failed")
        records.append(
            {
                "smiles": smiles,
                "sdf": block,
                "sample_index": index,
                "n_atoms": int(mol.GetNumAtoms()),
                "qc_flags": flags,
            }
        )
    return records, failures


def run(request):
    """Sample ``n_molecules`` MolJO ligands for one pocket and reference ligand."""
    moljo_root = Path(request["moljo_root"]).expanduser().resolve()
    if not (moljo_root / "sample_for_pocket_guided.py").is_file():
        raise RuntimeError(f"{moljo_root} is not a MolJO checkout (no sample_for_pocket_guided.py)")

    # `Config` resolves `configs/test_opt.yaml` and `accounting.logdir` relative to
    # the process cwd, and `core.*` must be importable — both are what upstream's
    # `cd MolJO && python sample_...` gives it.
    os.chdir(str(moljo_root))
    if str(moljo_root) not in sys.path:
        sys.path.insert(0, str(moljo_root))
    # configs/test_opt.yaml line 1 is `!ENV "${LOGNAME}_bfn_molopt"`. LOGNAME is not
    # in the adapter's environment allowlist, and an unset one silently yields the
    # project name "_bfn_molopt".
    os.environ.setdefault("LOGNAME", os.environ.get("USER", "drugagent"))

    # Inputs and weights are resolved before anything heavy is imported, so a
    # missing checkpoint is reported in milliseconds instead of after torch,
    # Lightning and MolJO's evaluation stack have all been loaded.
    out_dir = Path(request["out_dir"])
    device = str(request.get("device") or "cuda:0")
    n_molecules = int(request.get("n_molecules", 10))
    protein_path = _require_file(request.get("protein_path"), "receptor PDB")
    ligand_path = _require_file(request.get("ligand_path"), "reference ligand SDF")
    config_file = _require_file(
        request.get("config_file") or (moljo_root / "configs/test_opt.yaml"), "sampling config"
    )
    pretrained_dir = Path(request.get("pretrained_dir") or (moljo_root / "pretrained")).expanduser()
    train_config = _require_file(
        request.get("train_config") or (pretrained_dir / "config.yaml"),
        "dumped training config (accounting.dump_config_path)",
    )
    backbone_ckpt = _find_backbone(
        request.get("backbone_ckpt"), request.get("checkpoint_dir") or pretrained_dir
    )

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import Callback
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    # `sample_for_pocket_guided` is the real entry point: imported, not copied, so
    # `get_dataloader_from_pdb` is upstream's own parsing, clipping and featurising.
    import sample_for_pocket_guided as upstream
    from core.callbacks.basic import EMACallback, NormalizerCallback, RecoverCallback
    from core.config.config import Config
    from core.models.train_loop import BFNTrainLoop, ClassifierTrainLoop

    args = dict(ARGPARSE_DEFAULTS)
    args["config_file"] = str(config_file)
    args["protein_path"] = str(protein_path)
    args["ligand_path"] = str(ligand_path)
    args["num_samples"] = n_molecules
    args["seed"] = int(request.get("seed", ARGPARSE_DEFAULTS["seed"]))
    args["sample_steps"] = int(request.get("sample_steps", ARGPARSE_DEFAULTS["sample_steps"]))
    args["sample_num_atoms"] = str(request.get("sample_num_atoms") or ARGPARSE_DEFAULTS["sample_num_atoms"])
    args["objective"] = str(request.get("objective") or ARGPARSE_DEFAULTS["objective"])
    args["guide_mode"] = str(request.get("guide_mode") or ARGPARSE_DEFAULTS["guide_mode"])
    args["pos_grad_weight"] = float(request.get("pos_grad_weight", ARGPARSE_DEFAULTS["pos_grad_weight"]))
    args["type_grad_weight"] = float(request.get("type_grad_weight", ARGPARSE_DEFAULTS["type_grad_weight"]))

    cfg = Config(**args)
    seed_everything(cfg.seed)

    # `accounting.*` is `!PATHJOIN [!ENV "${HOME}/project/logs/", ...]`
    # (configs/test_opt.yaml lines 53-59), i.e. a training run's own directory. This
    # is a sampling run against checkpoints the operator placed somewhere else, so
    # every accounting path is repointed before anything reads one.
    cfg.accounting.logdir = str(out_dir)
    cfg.accounting.wandb_logdir = str(out_dir / "wandb")
    cfg.accounting.checkpoint_dir = str(backbone_ckpt.parent)
    cfg.accounting.dump_config_path = str(train_config)
    cfg.accounting.generated_mol_dir = str(out_dir / "generated_mol")
    cfg.accounting.test_outputs_dir = str(out_dir / "test_outputs")
    # evaluation.batch_size is a literal 3 in configs/test_opt.yaml, not a !SUB, so
    # it is the one sampler knob that has to be set on the object.
    cfg.evaluation.batch_size = max(1, int(request.get("batch_size", 3)))

    # sample_for_pocket_guided.py line 196: builds the loader and, as a side effect,
    # sets the protein/ligand feature dims and resets evaluation.num_samples to 1.
    test_loader = upstream.get_dataloader_from_pdb(cfg)

    # Lines 199-212: the dumped training config is authoritative for `dynamics` and
    # `data`; only these fields carry over from the sampling config.
    tr_cfg = Config(cfg.accounting.dump_config_path)
    tr_cfg.test_only = cfg.test_only
    tr_cfg.evaluation = cfg.evaluation
    tr_cfg.visual = cfg.visual
    tr_cfg.accounting = cfg.accounting
    tr_cfg.dynamics.beta1 = cfg.dynamics.beta1
    tr_cfg.dynamics.sigma1_coord = cfg.dynamics.sigma1_coord
    tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
    tr_cfg.seed = cfg.seed
    cfg = tr_cfg
    if not hasattr(cfg.train, "max_grad_norm"):
        cfg.train.max_grad_norm = "Q"

    model = BFNTrainLoop(config=cfg)
    collector = _collector(Callback)
    on_gpu = device.startswith("cuda")
    trainer = Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        accelerator="gpu" if on_gpu else "cpu",
        devices=1,
        # Upstream logs to Weights & Biases. A driver must not phone home.
        logger=False,
        num_sanity_val_steps=0,
        inference_mode=not cfg.test_only,
        callbacks=[
            RecoverCallback(
                latest_ckpt=str(Path(cfg.accounting.checkpoint_dir) / "last.ckpt"),
                resume=cfg.train.resume,
                recover_trigger_loss=1e7,
            ),
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            # Upstream hard-codes ema_device="cuda"; that is a crash on a CPU box.
            EMACallback(decay=cfg.train.ema_decay, ema_device="cuda" if on_gpu else "cpu"),
            collector,
        ],
    )

    objectives, classifiers = _load_classifiers(
        cfg.evaluation.objective, pretrained_dir, cfg, Config, ClassifierTrainLoop
    )
    model.configure_classifiers(
        classifiers,
        objectives,
        cfg.evaluation.guide_mode,
        cfg.evaluation.pos_grad_weight,
        cfg.evaluation.type_grad_weight,
    )
    trainer.test(model, dataloaders=test_loader, ckpt_path=str(backbone_ckpt))

    records, failures = _records_from(collector.outputs, Chem)
    warnings = []
    if failures:
        warnings.append(
            f"MolJO: {failures} of {len(collector.outputs)} samples failed bond "
            "reconstruction (MolReconsError)"
        )
    return {
        "molecules": records,
        "n_requested": n_molecules,
        "n_sampled": len(collector.outputs),
        "n_reconstruction_failures": failures,
        "backbone_ckpt": str(backbone_ckpt),
        "objective": "_".join(objectives),
        "guide_mode": str(cfg.evaluation.guide_mode),
        "pos_grad_weight": float(cfg.evaluation.pos_grad_weight),
        "type_grad_weight": float(cfg.evaluation.type_grad_weight),
        "sample_steps": int(cfg.evaluation.sample_steps),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
