"""FLOWR — sample ligands into a pocket from ONE pinned, published checkpoint.

Runs inside the upstream ``flowr_root`` environment (its ``environment.yml`` pins python
3.12, torch 2.5.1, lightning 2.6, pymol-bundle, tensordict, biotite, prolif), so it imports
nothing from ``drug_agent``: the two sides talk over the request/result JSON pair described
in :mod:`scripts.external._driver`.

Everything this script calls was read off the checkout at ``86a25f08161d``; each borrowed
fact names its source:

* entry point — ``flowr/gen/generate_from_pdb.py``, whose ``__main__`` block (line 701) is
  exactly ``args = get_args(); evaluate(args)``. This driver does the same, filling
  ``sys.argv`` so upstream's own parser supplies all ~100 defaults. Hand-building the
  Namespace would silently drift from the parser on the next upstream commit.
* flags — ``scripts/generate_pdb.sl``, upstream's own submission script: ``--arch pocket``,
  ``--pocket_type holo``, ``--cut_pocket``, ``--pocket_cutoff 7``, ``--ckpt_path``,
  ``--batch_cost``, ``--sample_n_molecules_per_target``, ``--categorical_strategy
  uniform-sample``. That script passes ``--ode_sampling_strategy "$sampling_strategy"`` with
  the variable never assigned, so the value used here is the parser's own default,
  ``DEFAULT_ODE_SAMPLING_STRATEGY = "linear"``.
* output — ``evaluate()`` writes ``<save_dir>/samples_<pdb stem>.sdf``
  (``generate_from_pdb.py:448``) plus a ``.pt`` pickle of the same molecules. The SDF is what
  this driver reads back.
* checkpoint loading — ``flowr/scriptutil.py:1042`` ``load_model``: ``torch.load`` at line
  1048, ``hyper_parameters`` at 1052, then a ``LigandGenerator``/``PocketEncoder`` built by
  hard-indexing that dict. The preflight below checks those keys *before* the run, so a
  wrong-family checkpoint fails in one readable sentence instead of a ``KeyError`` inside a
  constructor.

**The pin.** This capability is FLOWR, the published model; the checkout is FLOWR.root's
code, which also loads FLOWR.root checkpoints. The two must never be confused, so the driver
refuses to build a model until the checkpoint's SHA-256 equals the digest written beside it,
and returns that digest with the samples.

Request JSON:
    checkout            path to the flowr_root checkout (also on PYTHONPATH)
    checkpoint          the pinned .ckpt
    checkpoint_pin      a file whose first token is that checkpoint's expected sha256
    pdb_file            protein/pocket PDB or CIF
    ligand_file         reference ligand (SDF/MOL/PDB); the pocket is cut around it
    n_molecules         molecules to sample for this target
    batch_cost          upstream's batch size knob
    seed, pocket_cutoff, integration_steps, max_sample_iter
    out_dir             injected by the adapter; the sampling run writes below it

Result JSON:
    molecules[]         {smiles, sdf, properties{}, qc_flags[]}
    checkpoint{}        {path, sha256, size_bytes, hparams{}}
    warnings[], errors[]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: Read the checkpoint in 4 MiB blocks: these files are gigabytes.
_HASH_CHUNK = 4 * 1024 * 1024

#: Hyper-parameters that ``--arch pocket`` generation hard-indexes, so a checkpoint missing
#: any of them cannot be loaded by this release at all. Sources: ``flowr/scriptutil.py``
#: 1213-1272 (the PocketEncoder and LigandGenerator constructor arguments),
#: ``flowr/gen/utils.py`` 403-480 (load_util), and ``flowr/gen/generate_from_pdb.py`` 99-104
#: (remove_hs / remove_aromaticity).
REQUIRED_HPARAMS: tuple[str, ...] = (
    "pocket-fixed_equi",
    "pocket-d_equi",
    "pocket-d_inv",
    "pocket-n_layers",
    "d_equi",
    "d_inv",
    "d_message",
    "d_message_ff",
    "d_edge",
    "n_layers",
    "n_attn_heads",
    "emb_size",
    "add_feats",
    "predict_interactions",
    "flow_interactions",
    "predict_affinity",
    "predict_docking_score",
    "use_rbf",
    "use_sphcs",
    "use_distances",
    "use_crossproducts",
    "use_fourier_time_embed",
    "use_lig_pocket_rbf",
    "self_cond",
    "coord_skip_connect",
    "coord_scale",
    "remove_hs",
    "remove_aromaticity",
    "val-ligand-prior-type-noise",
    "val-ligand-prior-bond-noise",
    "val-ligand-type-interpolation",
)

#: The subset written back with the samples. Anyone comparing two FLOWR batches needs to see,
#: without loading either checkpoint, whether they came from the same model.
FINGERPRINT_KEYS: tuple[str, ...] = (
    "predict_affinity",
    "predict_docking_score",
    "flow_interactions",
    "predict_interactions",
    "add_feats",
    "d_equi",
    "d_inv",
    "n_layers",
    "coord_scale",
    "remove_hs",
    "dataset",
    "lora_finetuning",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pin(path: Path) -> str:
    """The expected digest, accepting ``sha256sum`` output verbatim."""
    text = path.read_text(encoding="utf-8").strip()
    token = text.split()[0].lower() if text else ""
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise RuntimeError(
            f"{path} does not hold a sha256 digest (first token {token[:80]!r}); write it "
            "with `sha256sum flowr.ckpt > flowr.ckpt.sha256`"
        )
    return token


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], bool]:
    """The checkpoint, plus whether it loads under ``weights_only=True``.

    Upstream calls ``torch.load`` bare (``flowr/scriptutil.py:1048``); torch >= 2.6 defaults
    that to ``weights_only=True`` and refuses a checkpoint whose hyper-parameters pickle a
    non-primitive object. Trying the strict form first costs nothing when it works and turns
    that into a warning the operator can act on instead of a crash inside ``load_model``.
    """
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True), True
    except TypeError:  # a torch too old to know the argument at all
        return torch.load(str(path), map_location="cpu"), False
    except Exception:  # every strict-load failure means the same thing: fall back explicitly
        return torch.load(str(path), map_location="cpu", weights_only=False), False


def _resolve_pocket_noise(hparams: dict[str, Any]) -> str:
    """``pocket_noise`` the way ``load_model`` resolves it (flowr/scriptutil.py:1062-1067)."""
    return str(hparams.get("pocket_noise") or hparams.get("train-pocket-noise") or "fix")


def _jsonable(value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _preflight(checkpoint: Path, pin_file: Path) -> dict[str, Any]:
    """Prove this is the pinned checkpoint and that this release can load it.

    Returns the identity block that travels back with the samples. Every check here is a
    refusal, never a repair: a checkpoint that fails one of them is a different model.
    """
    if not checkpoint.is_file():
        raise RuntimeError(f"pinned FLOWR checkpoint {checkpoint} does not exist")
    if not pin_file.is_file():
        raise RuntimeError(
            f"no digest pin at {pin_file}, so the checkpoint cannot be identified; FLOWR and "
            "FLOWR.root checkpoints load through the same code and must not be confused"
        )
    expected = _read_pin(pin_file)
    digest = _sha256(checkpoint)
    if digest != expected:
        raise RuntimeError(
            f"checkpoint {checkpoint} has sha256 {digest} but {pin_file} pins {expected} — "
            "refusing to sample from an unidentified FLOWR-family checkpoint"
        )

    blob, weights_only_ok = _load_checkpoint(checkpoint)
    hparams = blob.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise RuntimeError(
            f"checkpoint {checkpoint} carries no 'hyper_parameters' mapping; load_model reads "
            "it at flowr/scriptutil.py:1052"
        )
    absent = ", ".join(key for key in REQUIRED_HPARAMS if key not in hparams)
    if absent:
        raise RuntimeError(
            f"checkpoint {checkpoint} is not loadable by this release: its hyper_parameters "
            f"lack {absent}, which flowr/scriptutil.py and flowr/gen/utils.py index directly"
        )
    pocket_noise = _resolve_pocket_noise(hparams)
    if pocket_noise not in ("fix", "random"):
        raise RuntimeError(
            f"checkpoint {checkpoint} was trained with pocket_noise={pocket_noise!r}, which "
            "needs --arch pocket_flex; this release raises NotImplementedError for that "
            "architecture (flowr/scriptutil.py:1272)"
        )
    coord_scale = hparams["coord_scale"]
    if float(coord_scale) != 1.0:
        raise RuntimeError(
            f"checkpoint {checkpoint} has coord_scale={coord_scale!r}; flowr/gen/utils.py:413 "
            "binds coord_std only when it is 1.0, so load_util would raise UnboundLocalError"
        )
    interpolation = hparams["val-ligand-type-interpolation"]
    if str(interpolation) != "unmask":
        raise RuntimeError(
            f"checkpoint {checkpoint} has val-ligand-type-interpolation={interpolation!r}; the "
            "uniform-sample categorical strategy asserts 'unmask' at flowr/gen/utils.py:479"
        )

    fingerprint = {key: _jsonable(hparams[key]) for key in FINGERPRINT_KEYS if key in hparams}
    fingerprint["pocket_noise"] = pocket_noise
    for key in ("epoch", "global_step", "pytorch-lightning_version"):
        if key in blob:
            fingerprint[key] = _jsonable(blob[key])
    identity = {
        "path": str(checkpoint),
        "sha256": digest,
        "size_bytes": checkpoint.stat().st_size,
        "hparams": fingerprint,
        "weights_only_loadable": weights_only_ok,
    }
    del blob
    return identity


def _argv(request: dict[str, Any], save_dir: Path) -> list[str]:
    """Upstream's own command line, taken from ``scripts/generate_pdb.sl``.

    Upstream's post-processing filters (``--filter_valid_unique``, ``--filter_diversity``,
    ``--filter_pb_valid``) are deliberately absent: the platform computes validity and
    uniqueness itself from the raw output, and a backend that filtered first would make those
    numbers describe survivors instead of samples.
    """
    return [
        "generate_from_pdb",
        "--pdb_file",
        str(request["pdb_file"]),
        "--ligand_file",
        str(request["ligand_file"]),
        "--arch",
        "pocket",
        "--pocket_type",
        "holo",
        "--cut_pocket",
        "--pocket_cutoff",
        str(float(request.get("pocket_cutoff", 7.0))),
        "--ckpt_path",
        str(request["checkpoint"]),
        "--save_dir",
        str(save_dir),
        "--gpus",
        "1",
        # No DataLoader worker processes: one target, one small dataset, and every forked
        # child is another thing a timeout has to reap off the GPU.
        "--num_workers",
        "0",
        "--batch_cost",
        str(int(request.get("batch_cost", 10))),
        "--sample_n_molecules_per_target",
        str(int(request["n_molecules"])),
        "--max_sample_iter",
        str(int(request.get("max_sample_iter", 20))),
        "--integration_steps",
        str(int(request.get("integration_steps", 100))),
        "--categorical_strategy",
        "uniform-sample",
        "--ode_sampling_strategy",
        "linear",
        "--coord_noise_scale",
        "0.0",
        "--seed",
        str(int(request.get("seed", 42))),
    ]


def _read_sdf(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The generated ligands, one record per SDF entry, unreadable ones kept as evidence."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for mol in supplier:
        if mol is None:
            records.append({"smiles": "", "qc_flags": ["flowr_sdf_record_unreadable"]})
            continue
        record: dict[str, Any] = {"smiles": Chem.MolToSmiles(mol), "qc_flags": []}
        try:
            record["sdf"] = Chem.MolToMolBlock(mol)
        except (ValueError, RuntimeError) as exc:
            record["qc_flags"].append("no_3d_block")
            warnings.append(f"could not write a molblock for one sample: {exc}")
        try:
            record["properties"] = dict(mol.GetPropsAsDict())
        except (ValueError, RuntimeError):
            record["properties"] = {}
        records.append(record)
    return records, warnings


def run(request: dict[str, Any]) -> dict[str, Any]:
    checkout = str(request["checkout"])
    if checkout not in sys.path:
        sys.path.insert(0, checkout)

    identity = _preflight(Path(str(request["checkpoint"])), Path(str(request["checkpoint_pin"])))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FLOWR sampling is GPU-only: flowr/gen/generate_from_pdb.py:82 calls "
            "model.to('cuda') unconditionally, and torch.cuda.is_available() is False in "
            f"{sys.executable}"
        )

    from flowr.gen import generate_from_pdb

    save_dir = Path(str(request["out_dir"])) / "flowr_samples"
    save_dir.mkdir(parents=True, exist_ok=True)

    saved, sys.argv = sys.argv, _argv(request, save_dir)
    try:
        # get_args() rather than a hand-built Namespace: it is upstream's own parser, so every
        # default this driver does not set stays whatever the checkout says it is.
        args = generate_from_pdb.get_args()
    finally:
        sys.argv = saved
    generate_from_pdb.evaluate(args)

    target_stem = Path(str(request["pdb_file"])).stem
    sdf_path = save_dir / f"samples_{target_stem}.sdf"
    if not sdf_path.is_file():
        matches = sorted(save_dir.glob("samples_*.sdf"))
        if not matches:
            raise RuntimeError(f"FLOWR finished but wrote no SDF under {save_dir} (expected {sdf_path})")
        sdf_path = matches[0]

    molecules, warnings = _read_sdf(sdf_path)
    if not identity["weights_only_loadable"]:
        warnings.append(
            "the checkpoint needs weights_only=False; upstream's bare torch.load at "
            "flowr/scriptutil.py:1048 refuses it on torch >= 2.6"
        )
    return {
        "molecules": molecules,
        "checkpoint": identity,
        "n_requested": int(request["n_molecules"]),
        "sdf_path": str(sdf_path),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
