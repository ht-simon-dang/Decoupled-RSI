"""Run DiffDock-L blind docking inside DiffDock's own environment.

This is the driver half of :mod:`drug_agent.tools.structure.diffdock_tool`. It runs
in whichever interpreter has DiffDock's dependencies — upstream pins Python 3.9.18,
``torch==1.13.1+cu117``, ``e3nn``, ``fair-esm`` and ``prody`` in its
``environment.yml`` — so it imports nothing from ``drug_agent`` and nothing outside
the standard library except DiffDock itself.

**It calls the supported entry point, not a reimplementation.** ``inference.py``
exposes ``get_parser()`` and ``main(args)``, and the README's documented invocation
is::

    python -m inference --config default_inference_args.yaml
        --protein_ligand_csv data/protein_ligand_example.csv --out_dir results/...

so this driver builds exactly those three arguments and calls ``inference.main``.
Every sampling knob — the noise schedule, the temperatures, ``actual_steps`` — is
left at the values ``default_inference_args.yaml`` publishes for DiffDock-L.

**Why the knobs go through the config file and not the command line.**
``inference.main`` merges the YAML *over* the parsed namespace (``arg_dict[key] =
value`` for every key in the config), so a ``--samples_per_complex`` passed on the
command line is silently overwritten by the file's value. The only way to change
one of those keys and have it stick is to change the file, so the driver writes a
copy of ``default_inference_args.yaml`` with the requested overrides applied and
points ``--config`` at that.

**Why the working directory is the checkout.** ``utils/so3.py`` and
``utils/torus.py`` cache their truncated series to ``.so3_*.npy``, ``.p.npy`` and
``.score.npy`` in the *current directory* as an import side effect. Running from
the checkout means that precomputation — which upstream's README calls "a couple of
minutes" and its Dockerfile bakes in with ``python utils/precompute_series.py`` —
happens once rather than once per call.

**Why it refuses to download weights.** ``inference.main`` fetches
``diffdock_models.zip`` from the DiffDock GitHub releases when ``model_dir`` does
not exist. Pulling a 124 MB archive in the middle of a docking call, inside a
subprocess with a timeout, is not a thing a driver should do quietly, so the
weights are checked first and a missing one is a structured failure naming the
exact path.

Request JSON:
    diffdock_root         path to the DiffDock checkout
    receptor_pdb          path to the receptor ``.pdb`` (whole protein; DiffDock is blind)
    ligands[]             ``{complex_name, candidate_id, ligand_description}``, where
                          ``ligand_description`` is a SMILES or a file RDKit can read
    samples_per_complex   poses to sample per ligand (optional)
    inference_steps       denoising steps (optional)
    actual_steps          steps actually performed (optional)
    batch_size            poses per forward pass (optional)
    config                override for ``default_inference_args.yaml`` (optional)
    loglevel              DiffDock's own log level (optional, default WARNING)
    out_dir               where to write ``result.json``

Response JSON (``out_dir/result.json``):
    complexes[]           ``{complex_name, candidate_id, poses[], errors[]}`` where each
                          pose is ``{rank, confidence, sdf}`` — ``sdf`` is the file
                          DiffDock wrote, verbatim
    n_complexes           how many ligands were requested
    n_poses               how many pose files were found across all of them
    samples_per_complex   the value actually in force after the config merge
    model_release         the directory name of the weight release used, e.g. ``v1.1``
    device                ``cuda`` or ``cpu``, as DiffDock itself resolved it
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: DiffDock names every written pose ``rank<N>_confidence<C>.sdf`` — the
#: ``write_mol_with_coords`` loop at the end of ``inference.py``. ``C`` is formatted
#: with ``:.2f``, so two decimals is the whole precision the supported entry point
#: exposes. The same loop also writes a bare ``rank1.sdf`` duplicate of the top
#: pose; this pattern deliberately does not match it, so it is not counted twice.
_POSE_NAME = re.compile(r"^rank(\d+)_confidence(-?\d+(?:\.\d+)?)\.sdf$")

#: Keys of ``default_inference_args.yaml`` a request may override. Everything else
#: in that file is the sampling schedule the DiffDock-L authors tuned, and is
#: copied through untouched.
_OVERRIDABLE = (
    "samples_per_complex",
    "inference_steps",
    "actual_steps",
    "batch_size",
    "model_dir",
    "confidence_model_dir",
    "ckpt",
    "confidence_ckpt",
)

#: The two model directories, each with the config key naming its checkpoint.
_MODEL_DIRS = (("model_dir", "ckpt"), ("confidence_model_dir", "confidence_ckpt"))


def _absolute(root, raw):
    """Resolve a config path against the checkout, since the config ships ``./`` paths."""
    path = Path(str(raw))
    return str(path if path.is_absolute() else (root / path).resolve())


def _build_config(root, request, work):
    """Write ``default_inference_args.yaml`` plus this request's overrides, and return it."""
    import yaml

    base = Path(request["config"]) if request.get("config") else root / "default_inference_args.yaml"
    if not base.is_file():
        raise FileNotFoundError(f"DiffDock inference config not found at {base}")
    with base.open() as fh:
        config = yaml.safe_load(fh) or {}

    for key in _OVERRIDABLE:
        value = request.get(key)
        if value is not None:
            config[key] = value
    for key, _ckpt in _MODEL_DIRS:
        if config.get(key):
            config[key] = _absolute(root, config[key])

    path = work / "inference_args.yaml"
    with path.open("w") as fh:
        yaml.safe_dump(config, fh)
    return config, path


def _require_weights(config):
    """Fail with the exact paths before ``inference.main`` decides to download them."""
    missing = []
    for dir_key, ckpt_key in _MODEL_DIRS:
        model_dir = config.get(dir_key)
        if not model_dir:
            missing.append(f"<{dir_key} unset in the inference config>")
            continue
        checkpoint = config.get(ckpt_key)
        if not checkpoint:
            missing.append(f"<{ckpt_key} unset in the inference config>")
            continue
        for rel in ("model_parameters.yml", str(checkpoint)):
            candidate = Path(model_dir) / rel
            if not candidate.is_file():
                missing.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            "DiffDock model weights are missing: "
            + ", ".join(missing)
            + ". inference.py would download diffdock_models.zip from the DiffDock GitHub "
            "releases at this point; this driver will not fetch 124 MB inside a docking call. "
            "Extract the release once into <checkout>/workdir/."
        )


def _write_csv(path, receptor, ligands):
    """The four-column complex table ``--protein_ligand_csv`` documents.

    ``protein_sequence`` is left empty on every row: an empty cell reaches
    ``inference.main`` as NaN and ``set_nones`` turns it into ``None``, which is
    what keeps ESMFold out of the run. A receptor path is always supplied here, so
    there is nothing to fold.
    """
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["complex_name", "protein_path", "ligand_description", "protein_sequence"])
        for ligand in ligands:
            writer.writerow([ligand["complex_name"], receptor, ligand["ligand_description"], ""])


def _collect(pred_dir, complex_name):
    """Every pose DiffDock wrote for one complex, in its own rank order."""
    found = []
    directory = pred_dir / complex_name
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        match = _POSE_NAME.match(path.name)
        if match is None:
            continue
        found.append(
            {
                "rank": int(match.group(1)),
                "confidence": float(match.group(2)),
                "sdf": path.read_text(),
            }
        )
    found.sort(key=lambda item: item["rank"])
    return found


def run(request):
    """Dock every requested ligand against one receptor and return the pose files."""
    root = Path(request["diffdock_root"]).resolve()
    receptor = Path(request["receptor_pdb"]).resolve()
    out_dir = Path(request["out_dir"]).resolve()
    if not receptor.is_file():
        raise FileNotFoundError(f"receptor PDB not found: {receptor}")
    ligands = list(request.get("ligands") or [])
    if not ligands:
        raise ValueError("request carries no ligands")

    work = out_dir / "run"
    work.mkdir(parents=True, exist_ok=True)

    # See the module docstring: utils.so3 and utils.torus cache to the cwd on import.
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    config, config_path = _build_config(root, request, work)
    _require_weights(config)

    csv_path = work / "protein_ligand.csv"
    _write_csv(csv_path, str(receptor), ligands)
    pred_dir = work / "predictions"

    import inference
    import torch

    args = inference.get_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--protein_ligand_csv",
            str(csv_path),
            "--out_dir",
            str(pred_dir),
            "--loglevel",
            str(request.get("loglevel") or "WARNING"),
        ]
    )
    inference.main(args)

    complexes = []
    n_poses = 0
    for ligand in ligands:
        poses = _collect(pred_dir, ligand["complex_name"])
        n_poses += len(poses)
        complexes.append(
            {
                "complex_name": ligand["complex_name"],
                "candidate_id": ligand.get("candidate_id", ""),
                "poses": poses,
                "errors": [] if poses else ["diffdock_wrote_no_pose_for_this_ligand"],
            }
        )

    if n_poses == 0:
        # inference.main catches per-complex exceptions, counts them and returns
        # normally, so an empty output directory is the only signal that every
        # complex failed. The reason is on stderr, which the adapter tails.
        raise RuntimeError(
            f"DiffDock wrote no rank*_confidence*.sdf for any of the {len(ligands)} requested "
            "complexes. inference.py swallows per-complex failures, so the cause is in the "
            "driver's stderr above."
        )

    return {
        "complexes": complexes,
        "n_complexes": len(ligands),
        "n_poses": n_poses,
        "samples_per_complex": int(config.get("samples_per_complex") or 0),
        "model_release": Path(str(config["model_dir"])).parent.name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    _driver.main(run)
