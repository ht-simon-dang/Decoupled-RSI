"""Run one PocketXMol task on user data — the drug-agent ⇄ PocketXMol bridge.

This file is executed inside whichever interpreter carries PocketXMol's
dependencies (upstream's own conda environment is named ``pxm``). It imports
nothing from ``drug_agent``: the two sides talk over a JSON request/response pair
on disk, so a CUDA fault, an OOM in the denoiser or an openbabel segfault inside
bond perception cannot reach the caller.

**Which upstream entry point, and why.** PocketXMol ships four sampling scripts and
only one of them takes user data:

``scripts/sample_use.py``
    The user-data entry point, and the one this driver calls. Its ``get_input_data``
    (line 75) takes a ``protein_path`` plus an optional ``input_ligand``, cuts the
    pocket itself with ``process.utils_process.extract_pocket``, and wraps the
    result in ``utils.dataset.UseDataset``. No dataset, no LMDB, any PDB.
``scripts/sample_use_pdb.py``
    The same, for peptide outputs (writes PDB rather than SDF).
``scripts/sample_drug3d.py``
    The benchmark entry point. Its config carries ``data.dataset.dbs[].lmdb_path``
    (see ``configs/sample/test/denovo_geom/base.yml``), so it can only sample
    entries already in the processed GEOM/CSD/MOAD LMDBs.
``scripts/sample_pdb.py``
    The benchmark entry point for peptides, likewise LMDB-backed.

``sample_use.py`` has no importable ``main``: everything lives under
``if __name__ == '__main__'`` (line 155). It is therefore launched as a subprocess
with exactly the arguments its ``argparse`` block declares —
``--config_task``, ``--config_model``, ``--outdir``, ``--device``, ``--batch_size``,
``--num_workers`` — rather than reimplemented here. A reimplementation would have to
duplicate the transform stack, the noiser, the auto-regressive refinement loop and
the reconstruction pool, and would drift from upstream at the first commit.

**How the task is specified.** PocketXMol is one checkpoint behind many tasks, and
the task is a *config file*, not a flag. This driver therefore starts from one of
upstream's own example configs under ``configs/sample/examples/`` — which is where
upstream defines what each task means, down to the noise schedule — and overrides
only the input fields: the protein, the ligand, the pocket, the fragment partition
and the sampling counts. The noise block, the level schedule and the transform
names are never synthesized here, because a hand-written noise schedule would be a
different model.

**Two upstream behaviours worth knowing before reading a result.**

1. The seed is not the seed you passed. ``sample_use.py`` line 173 computes
   ``seed = config.sample.seed + sum(ord(c) for c in outdir) + sum(ord(c) for c in
   config_task)``, so the same nominal seed gives a different run from a different
   output directory. The effective value is recorded by upstream as
   ``sample.complete_seed`` in the config it copies into its log directory, and this
   driver reads it back and returns it as ``complete_seed``.
2. Reconstruction is where these models fail. ``sample_use.py`` tags every sample
   in ``gen_info.csv`` (lines 306-345): an empty tag is a molecule, ``incomp`` means the
   reconstructed SMILES contained a ``.`` (disconnected fragments), and ``bad``
   means ``utils.reconstruct.reconstruct_from_generated_with_edges`` raised
   ``MolReconsError``. All three are returned, so the caller's validity denominator
   is the number of samples drawn, not the number that happened to survive.

Every upstream call below carries the file and line it was taken from, at
``PocketXMol`` commit 65488cf635c8.

Request JSON:
    root                checkout root (also the cwd upstream's relative paths need)
    base_config         task config to start from, relative to *root*, e.g.
                        ``configs/sample/examples/sbdd.yml``
    config_model        model config carrying the checkpoint path, relative to
                        *root*; upstream's default is ``configs/sample/pxm.yml``
    protein_path        absolute path to a protein or pocket PDB
    input_ligand        SDF path, PDB path or SMILES; null for de-novo SBDD
    pocket_coord        [x, y, z] pocket centre for extraction, or null
    ref_ligand_path     alternative to pocket_coord: cut around this ligand
    pocket_radius       radius in Angstrom for pocket extraction
    center              [x, y, z] denoising space centre, or null to leave the
                        base config's own centring alone
    fragment_groups     [[atom indices], ...] for the maskfill tasks
    anchor_groups       [[atom indices], ...] allowed connecting atoms, or null
    mol_size_mean       override the generated heavy-atom count mean, or null
    mol_size_std        override its standard deviation, or null
    n_molecules         total molecules to sample
    batch_size          molecules per forward pass
    seed                nominal seed (see caveat 1 above)
    device              e.g. "cuda:0" / "cpu"
    data_id, pdbid      provenance strings echoed into gen_info.csv
    timeout_seconds     budget for the sampling subprocess
    out_dir             directory to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]         one entry per *sample*, failures included:
                        {smiles, sdf, sample_index, tag, qc_flags[], cfd_traj,
                        cfd_pos, cfd_node, cfd_edge}
    n_requested, n_sampled, n_success, n_incomplete, n_reconstruction_failures,
    task, base_config, checkpoint, complete_seed, log_dir, device,
    warnings[], errors[]
"""

from __future__ import annotations

import contextlib
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path bootstrap above, deliberately.
import _driver

#: PyYAML is not the standard library, but it is a hard dependency of the upstream
#: package this driver runs beside: ``utils/misc.py`` line 8 imports it, and
#: ``load_config``/``save_config`` (lines 76-99) are how every PocketXMol script
#: reads and writes a config. The dump options below are copied from
#: ``save_config`` so the file this driver writes round-trips through upstream's
#: own loader unchanged.
import yaml

#: How much of the adapter's budget the sampling subprocess may use. The remainder
#: covers config assembly, reading ``gen_info.csv`` and reading the SDF files back,
#: and leaves room to write a structured failure instead of being killed mid-write.
_CHILD_BUDGET_FRACTION = 0.92

#: Lines of the child's stderr carried into a structured failure.
_ERROR_TAIL_LINES = 25

#: Tag values ``sample_use.py`` writes into ``gen_info.csv`` (lines 306-345), and
#: the qc flag each becomes. An empty tag is a molecule and carries no flag.
_TAG_FLAGS = {
    "incomp": "pocketxmol_disconnected_fragments",
    "bad": "pocketxmol_reconstruction_failed",
}


def _load_yaml(path):
    """Read a YAML config as plain dicts. Upstream's ``load_config``, minus EasyDict."""
    with Path(path).open() as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return loaded


def _dump_yaml(config, path):
    """Write a config the way ``utils/misc.py::save_config`` (line 98) writes one."""
    with Path(path).open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)


def _section(config, name):
    """The named top-level block, created empty if the base config has none."""
    block = config.get(name)
    if not isinstance(block, dict):
        block = {}
        config[name] = block
    return block


def _apply_inputs(config, request, warnings):
    """Overwrite the base config's ``data`` block with this request's structures.

    Field names are upstream's, from ``sample_use.py::get_input_data`` (line 75) and
    the reference table in ``docs/sample_provided_data.md`` section 3.2.
    """
    data = _section(config, "data")
    data["protein_path"] = request["protein_path"]
    # is_pep must be set explicitly: sample_use.py line 237 infers it from
    # `data_cfg.input_ligand.endswith('.pdb')`, which raises AttributeError when
    # there is no input ligand at all (the de-novo SBDD case).
    data["is_pep"] = False

    input_ligand = request.get("input_ligand")
    if input_ligand:
        data["input_ligand"] = input_ligand
    else:
        data.pop("input_ligand", None)

    # pocket_args: ref_ligand_path and pocket_coord are mutually exclusive upstream
    # (docs/sample_provided_data.md section 3.2). With neither, get_input_data line
    # 128 falls back to using input_ligand as the reference, which needs a file.
    pocket_args = {"radius": request.get("pocket_radius") or 10}
    if request.get("ref_ligand_path"):
        pocket_args["ref_ligand_path"] = request["ref_ligand_path"]
    elif request.get("pocket_coord"):
        pocket_args["pocket_coord"] = [float(v) for v in request["pocket_coord"]]
    elif not (input_ligand or "").endswith((".sdf", ".pdb")):
        raise ValueError(
            "no pocket_coord, no ref_ligand_path and no SDF/PDB input_ligand: "
            "sample_use.py::get_input_data has nothing to cut a pocket around"
        )
    data["pocket_args"] = pocket_args

    pocmol_args = _section(data, "pocmol_args")
    pocmol_args["data_id"] = request.get("data_id") or "drugagent"
    pocmol_args["pdbid"] = request.get("pdbid") or ""

    transforms = _section(config, "transforms")
    center = request.get("center")
    if center:
        # featurizer_pocket.center is the denoising space centre (sbdd.yml line 22,
        # dock_smallmol.yml line 27). It is distinct from the pocket-extraction
        # centre above, and setting one does not set the other.
        _section(transforms, "featurizer_pocket")["center"] = [float(v) for v in center]
        featurizer = transforms.get("featurizer")
        if isinstance(featurizer, dict) and featurizer.pop("mol_as_pocket_center", None):
            # linking/growing/opt configs centre on the input ligand instead; an
            # explicit centre replaces that rather than fighting it.
            warnings.append(
                "explicit pocket centre overrode the base config's transforms.featurizer.mol_as_pocket_center"
            )
    else:
        # A stale centre is the one silent catastrophe in this file: the example
        # configs hard-code the coordinates of *their* protein, and leaving one in
        # place would denoise this request's molecule inside a box somewhere else
        # entirely, with no error. Deleting it is well defined — FeaturizePocket
        # (utils/transforms.py line 703) then centres on the mean position of the
        # extracted pocket atoms.
        featurizer_pocket = transforms.get("featurizer_pocket")
        if isinstance(featurizer_pocket, dict) and featurizer_pocket.pop("center", None) is not None:
            warnings.append(
                "no denoising centre was given, so the base config's example coordinates "
                "were removed; FeaturizePocket falls back to the extracted pocket's centroid"
            )


def _apply_partition(config, request, warnings):
    """Set the part1/part2 split the maskfill tasks generate around.

    ``grouped_node_p1`` names the atoms that are *given* (the fragments); everything
    else in the molecule is generated. ``transforms.variable_mol_size.not_remove``
    must list the same atoms, per the comment on ``linking_fixed_frags.yml`` line 31
    — otherwise the variable-size transform is free to delete the fragment the task
    was supposed to keep.
    """
    groups = request.get("fragment_groups")
    task = _section(config, "task")
    transform = _section(task, "transform")
    partition = transform.get("preset_partition")
    if not groups:
        if isinstance(partition, dict) and partition.get("grouped_node_p1"):
            raise ValueError(
                f"task {request.get('task', '')!r} partitions the input molecule but no "
                "fragment_groups were given; the base config's example indices would be "
                "applied to a different ligand"
            )
        return

    if not isinstance(partition, dict):
        raise ValueError(
            f"task {request.get('task', '')!r} has no task.transform.preset_partition block, "
            "so fragment_groups cannot be applied"
        )
    clean = [[int(i) for i in group] for group in groups if group]
    if not clean:
        raise ValueError("fragment_groups contained no atom indices")
    partition["grouped_node_p1"] = clean
    partition.pop("node_p2", None)

    settings = _section(transform, "settings")
    anchors = request.get("anchor_groups")
    if anchors:
        partition["grouped_anchor_p1"] = [[int(i) for i in group] for group in anchors]
        settings["known_anchor"] = {"all": 1}
    else:
        # known_anchor: all requires grouped_anchor_p1 (docs section 3.4.3). Dropping
        # the anchors without dropping the setting would leave the config referring
        # to connecting atoms that are no longer there.
        if partition.pop("grouped_anchor_p1", None) is not None:
            warnings.append(
                "no anchor_groups given, so the base config's connecting-atom "
                "constraint was replaced by known_anchor: none"
            )
        settings["known_anchor"] = {"none": 1}

    var_size = _section(config, "transforms").get("variable_mol_size")
    if isinstance(var_size, dict):
        flat = sorted({idx for group in clean for idx in group})
        var_size["not_remove"] = flat


def _apply_sampling(config, request, warnings):
    """Counts, seed and generated-size overrides."""
    sample = _section(config, "sample")
    sample["num_mols"] = int(request["n_molecules"])
    sample["batch_size"] = int(request["batch_size"])
    sample["seed"] = int(request["seed"])
    # Trajectories are debugging output: one SDF per saved molecule per trajectory
    # component. Nothing downstream reads them, so they are switched off.
    sample["save_traj_prob"] = 0.0
    # num_repeats multiplies num_mols in the outer loop of sample_use.py line 277;
    # the count is expressed once, through num_mols.
    sample.pop("num_repeats", None)

    mean = request.get("mol_size_mean")
    std = request.get("mol_size_std")
    if mean is None and std is None:
        return
    var_size = _section(config, "transforms").get("variable_mol_size")
    if not isinstance(var_size, dict):
        warnings.append(
            "mol_size override ignored: this task's config has no "
            "transforms.variable_mol_size block, so the molecule size comes from the input"
        )
        return
    # strategy: mol_atoms_based draws the atom count from Normal(coef * n_input +
    # bias, ...) — see sbdd.yml lines 26-32. coef is 0 in every example config, so
    # bias alone is the mean.
    distri = _section(var_size, "num_atoms_distri")
    if mean is not None:
        _section(distri, "mean")["bias"] = float(mean)
    if std is not None:
        _section(distri, "std")["bias"] = float(std)


def _rdkit_six_shim_dir(workdir):
    """A PYTHONPATH entry that restores ``rdkit.six`` for the sampling subprocess.

    ``utils/sascorer.py:22`` does ``from rdkit.six.moves import cPickle``.
    ``rdkit.six`` was a vendored copy of the ``six`` compatibility library that
    RDKit dropped years ago, so on any modern RDKit that import is a hard
    ``ModuleNotFoundError`` before a single molecule is sampled.

    The substitution is exact rather than approximate: in Python 3 ``cPickle`` *is*
    ``pickle``, which is what ``six.moves`` aliased it to. Nothing about scoring
    changes.

    It is installed through ``sitecustomize.py`` in a throwaway directory rather
    than by writing into the installed ``rdkit`` package, because the subprocess
    imports it long before any code here could patch ``sys.modules``, and because
    mutating a shared site-packages to make one model run would be a much worse
    trade than a directory that vanishes with the run.
    """
    import pickle
    import types  # noqa: F401 - documents what the generated shim builds

    shim_dir = workdir / "_compat"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(
        "import sys, types, pickle\n"
        "try:\n"
        "    import rdkit.six  # noqa: F401\n"
        "except ImportError:\n"
        "    six = types.ModuleType('rdkit.six')\n"
        "    moves = types.ModuleType('rdkit.six.moves')\n"
        "    moves.cPickle = pickle\n"
        "    six.moves = moves\n"
        "    six.iteritems = lambda d, **kw: iter(d.items(**kw))\n"
        "    six.iterkeys = lambda d, **kw: iter(d.keys(**kw))\n"
        "    six.itervalues = lambda d, **kw: iter(d.values(**kw))\n"
        "    six.string_types = (str,)\n"
        "    six.integer_types = (int,)\n"
        "    six.text_type = str\n"
        "    six.binary_type = bytes\n"
        "    six.PY2, six.PY3 = False, True\n"
        "    sys.modules['rdkit.six'] = six\n"
        "    sys.modules['rdkit.six.moves'] = moves\n"
        # Same pyg-lib reroute the other drivers apply, delivered the same way:
        # sample_use.py is a subprocess, so it has to be installed at interpreter
        # start rather than patched into this process.
        "try:\n"
        "    import torch, torch_geometric.nn, torch_geometric.nn.pool as _p\n"
        "    try:\n"
        "        _p.knn_graph(torch.zeros(2, 3), 1)\n"
        "    except ImportError:\n"
        "        from torch_cluster import knn, knn_graph, radius, radius_graph\n"
        "        for _m in (torch_geometric.nn, _p):\n"
        "            _m.knn, _m.knn_graph = knn, knn_graph\n"
        "            _m.radius, _m.radius_graph = radius, radius_graph\n"
        "except Exception:\n"
        "    pass\n"
    )
    assert pickle is not None
    return shim_dir


def _newest_log_dir(outdir):
    """The run directory ``get_new_log_dir`` (utils/misc.py line 132) just made."""
    candidates = [p for p in Path(outdir).iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _tail(text, lines=_ERROR_TAIL_LINES):
    kept = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return "\n".join(kept[-lines:]) if kept else "(no output)"


def _read_molecules(log_dir, warnings):
    """Parse ``gen_info.csv`` and the SDF directory beside it.

    The SDF directory is ``<log_dir>/<log_dir name>_SDF``, built at
    ``sample_use.py`` line 199 as ``os.path.basename(log_dir) + '_SDF'``; the
    ``filename`` column of ``gen_info.csv`` names a file inside it.
    """
    df_path = log_dir / "gen_info.csv"
    if not df_path.is_file():
        raise OSError(f"PocketXMol wrote no {df_path} — sampling produced no batches at all")
    sdf_dir = log_dir / (log_dir.name + "_SDF")

    molecules = []
    counts = {"": 0, "incomp": 0, "bad": 0}
    with df_path.open(newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh)):
            tag = (row.get("tag") or "").strip()
            counts[tag] = counts.get(tag, 0) + 1
            record = {
                "smiles": (row.get("smiles") or "").strip(),
                "sample_index": index,
                "tag": tag,
                "qc_flags": [_TAG_FLAGS[tag]] if tag in _TAG_FLAGS else [],
            }
            filename = (row.get("filename") or "").strip()
            if filename:
                sdf_path = sdf_dir / filename
                if sdf_path.is_file():
                    # A 'bad' sample's file is the raw generated point cloud written
                    # as an SDF string (sample_use.py line 393), not a sanitized
                    # molecule. It is kept because it is the only evidence of what
                    # the model actually emitted.
                    record["sdf"] = sdf_path.read_text()
                else:
                    warnings.append(f"gen_info.csv names {filename} but it is not there")
            for key in ("cfd_traj", "cfd_pos", "cfd_node", "cfd_edge"):
                raw = (row.get(key) or "").strip()
                if raw:
                    with contextlib.suppress(ValueError):
                        record[key] = float(raw)
            molecules.append(record)
    return molecules, counts


def _complete_seed(log_dir, config_name, warnings):
    """The seed upstream actually used, from the config it copied into *log_dir*.

    ``sample_use.py`` line 175 sets ``config.sample.complete_seed`` and line 195
    saves the config under the task config's own basename. Reading it back is the
    only way to record what was really seeded — see caveat 1 in the module docstring.
    """
    saved = log_dir / config_name
    if not saved.is_file():
        warnings.append(f"upstream saved no {config_name}, so the effective seed is unknown")
        return None
    try:
        return _load_yaml(saved).get("sample", {}).get("complete_seed")
    except (OSError, ValueError) as exc:
        warnings.append(f"could not read the effective seed from {saved}: {exc}")
        return None


def run(request):
    warnings = []
    root = Path(request["root"])
    out_dir = Path(request["out_dir"])

    base_config = root / request["base_config"]
    if not base_config.is_file():
        raise OSError(
            f"PocketXMol task config {base_config} is missing from the checkout; the task "
            "vocabulary this driver was written against is the one at commit 65488cf635c8"
        )
    config_model = root / request["config_model"]
    if not config_model.is_file():
        raise OSError(f"PocketXMol model config {config_model} is missing from the checkout")

    # The checkpoint path lives in the model config (configs/sample/pxm.yml is two
    # lines: model.checkpoint). sample_use.py line 177 torch.loads it, and line 179
    # then lists the sibling train_config/ directory, so both must be there.
    model_cfg = _load_yaml(config_model)
    checkpoint = model_cfg.get("model", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError(f"{config_model} carries no model.checkpoint")
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.is_file():
        raise OSError(
            f"PocketXMol checkpoint {ckpt_path} does not exist. Download model_weights.tar.gz "
            f"from https://zenodo.org/records/17801271 and extract it in {root} "
            "(docs/setup.md, 'Data & Model Weights')"
        )
    train_cfg_dir = ckpt_path.parent.parent / "train_config"
    if not train_cfg_dir.is_dir():
        raise OSError(
            f"PocketXMol training config directory {train_cfg_dir} does not exist; "
            "sample_use.py line 179 lists it to rebuild the model geometry, and it ships "
            "inside model_weights.tar.gz alongside the checkpoint"
        )

    config = _load_yaml(base_config)
    _apply_inputs(config, request, warnings)
    _apply_partition(config, request, warnings)
    _apply_sampling(config, request, warnings)

    # Written under the task's own name: sample_use.py line 169 builds the log
    # directory prefix from this basename, and line 173 folds it into the seed.
    config_name = f"{request['task']}.yml"
    config_path = out_dir / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _dump_yaml(config, config_path)

    sample_root = out_dir / "pxm"
    sample_root.mkdir(parents=True, exist_ok=True)

    # Exactly the arguments sample_use.py's argparse block declares (lines 157-163).
    # num_workers is pinned to 0 rather than left at -1: the default reads
    # train_config.train.num_workers and forks DataLoader workers inside what is
    # already a subprocess of a subprocess, where a crashed worker is invisible.
    argv = [
        sys.executable,
        str(root / "scripts" / "sample_use.py"),
        "--config_task",
        str(config_path),
        "--config_model",
        str(config_model),
        "--outdir",
        str(sample_root),
        "--device",
        request.get("device") or "cuda:0",
        "--batch_size",
        str(int(request["batch_size"])),
        "--num_workers",
        "0",
    ]
    budget = float(request.get("timeout_seconds") or 1800.0) * _CHILD_BUDGET_FRACTION
    started = time.time()
    try:
        child_env = dict(os.environ)
        shim_dir = _rdkit_six_shim_dir(Path(request["out_dir"]).parent)
        child_env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), child_env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        proc = subprocess.run(
            argv,
            cwd=str(root),  # sample_use.py line 39 does sys.path.append('.')
            env=child_env,
            capture_output=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"PocketXMol sampling exceeded its {budget:.0f}s share of the budget and was terminated"
        ) from exc
    elapsed = time.time() - started
    stderr = proc.stderr.decode("utf-8", "replace")

    log_dir = _newest_log_dir(sample_root)
    if proc.returncode != 0:
        raise RuntimeError(
            f"scripts/sample_use.py exited {proc.returncode} after {elapsed:.1f}s:\n{_tail(stderr)}"
        )
    if log_dir is None:
        raise RuntimeError(
            f"scripts/sample_use.py exited 0 but created no run directory under "
            f"{sample_root}:\n{_tail(stderr)}"
        )

    molecules, counts = _read_molecules(log_dir, warnings)
    n_sampled = len(molecules)
    if n_sampled < int(request["n_molecules"]):
        warnings.append(
            f"asked for {int(request['n_molecules'])} samples, gen_info.csv holds "
            f"{n_sampled} — the sampler stopped early"
        )

    return {
        "molecules": molecules,
        "n_requested": int(request["n_molecules"]),
        "n_sampled": n_sampled,
        "n_success": counts.get("", 0),
        "n_incomplete": counts.get("incomp", 0),
        "n_reconstruction_failures": counts.get("bad", 0),
        "task": request["task"],
        "base_config": request["base_config"],
        "checkpoint": str(ckpt_path),
        "complete_seed": _complete_seed(log_dir, config_name, warnings),
        "log_dir": str(log_dir),
        "device": request.get("device") or "cuda:0",
        "sampling_seconds": round(elapsed, 3),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
