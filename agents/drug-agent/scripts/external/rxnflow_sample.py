"""Sample synthesis-constrained molecules with RxnFlow — the drug-agent ⇄ RxnFlow bridge.

This file runs inside whichever interpreter has ``rxnflow`` installed. RxnFlow pins
``torch==2.5.1`` and ``numpy==1.26.4`` and builds ``torch-scatter``/``torch-sparse``/
``torch-cluster`` against that torch, while this platform runs torch 2.14 and numpy
2.5, so the two are expected to live in different environments and the JSON
request/response pair on disk is the whole interface. It imports nothing from
``drug_agent``.

**Which upstream entry point this is.** It is ``scripts/sampling_zeroshot.py``
(RxnFlow a39c7aebd45f), line for line in the order that script does it:
``init_empty(Config())`` → ``config.env_dir`` / ``config.algo.num_from_policy`` /
``config.algo.action_subsampling.sampling_ratio`` → ``ProxySampler(config,
model_path, device)`` from ``rxnflow.tasks.multi_pocket`` →
``sampler.update_temperature(...)`` → ``sampler.set_pocket(protein, center,
ref_ligand)`` → ``sampler.sample(n, calc_reward=...)``. That is the only load-and-
sample path RxnFlow publishes: every other script in ``scripts/`` trains a
GFlowNet against a reward first.

**Why the checkpoint is a path and never a name.** ``rxnflow.utils.download``
resolves a name such as ``qvina-unif-0-64`` by pulling it from Google Drive with
``gdown``. A driver launched by the platform must not reach the network, so the
caller resolves the file and this script fails with the exact ``gdown`` command
when it is absent.

**Why ``parse_temperature`` is inlined.** Upstream keeps it in
``scripts/_utils.py`` and reaches it by running from inside ``scripts/``. Copying
the six lines is safer than putting a foreign ``scripts`` directory on
``sys.path`` beside this file's own ``_driver``.

Request JSON:
    env_dir            synthesis environment directory; ``SynthesisEnv`` reads
                       template.txt, building_block.smi, bb_mask.npy,
                       bb_fp_2_1024.npy and bb_desc.npy from it (required)
    checkpoint         pocket-conditional GFlowNet checkpoint, a ``.pt`` file
                       carrying ``cfg`` and ``models_state_dict`` (required)
    protein_pdb        protein or pocket PDB; residues within 20 A of the centre
                       are kept by ``generate_protein_data`` (required)
    center             pocket centre [x, y, z]; required unless ref_ligand is given
    ref_ligand         .sdf/.mol2/.pdb whose centroid becomes the centre
    n_molecules        how many samples to return
    batch_size         molecules per policy call (``algo.num_from_policy``)
    subsampling_ratio  ``algo.action_subsampling.sampling_ratio``
    temperature        e.g. "uniform-16-64"; must match the checkpoint's family
    seed               RNG seed
    device             "cuda" | "cpu"
    calc_reward        also score each sample with the TacoGFN proxy
    out_dir            directory to write result.json (injected by the caller)

Response JSON (``out_dir/result.json``):
    molecules[]        {smiles, sample_index, route[], info{}}
    route[]            {step, smiles, action} — the reaction trajectory that built
                       the molecule, from ``SynthesisEnvContext.read_traj``
    n_requested, n_sampled, sampling_seconds, device, seed, temperature,
    checkpoint, env_dir, warnings[], errors[]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _driver  # the shared driver contract lives beside this file

#: Google Drive ids from ``src/rxnflow/utils/download.py``. Quoted here so the
#: failure message names the fetch instead of describing it.
_DRIVE_IDS = {
    "qvina-unif-0-64_20250512.pt": "13ATBmJIgIhU3Kclwut0yssjaciPCieEV",
    "qed-unif-0-64_20250512.pt": "1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi",
}

#: The five files ``SynthesisEnv.__init__`` opens under ``env_dir``. Checked here
#: so a half-built environment is a sentence rather than a bare FileNotFoundError
#: three frames inside upstream.
_ENV_FILES = (
    "template.txt",
    "building_block.smi",
    "bb_mask.npy",
    "bb_fp_2_1024.npy",
    "bb_desc.npy",
)


def parse_temperature(temperature_param: str) -> tuple[str, list[float]]:
    """Split "uniform-16-64" into ("uniform", [16.0, 64.0]).

    Copied verbatim from ``.external/RxnFlow/scripts/_utils.py``.
    """
    temperature_info = temperature_param.split("-")
    sample_dist = temperature_info[0]
    dist_params = list(map(float, temperature_info[1:]))
    assert sample_dist in ("constant", "uniform", "loguniform", "gamma", "beta")
    if sample_dist == "constant":
        assert len(dist_params) == 1, "constant temperature requires only one parameter; e.g. constant-32"
    else:
        assert len(dist_params) == 2, f"{sample_dist} temperature requires two parameters."
    return sample_dist, dist_params


def _jsonable(value: Any) -> Any:
    """Anything the sampler puts in ``info`` or a trajectory, as JSON.

    ``info`` carries torch scalars (``beta``) and, with ``calc_reward``, tensors
    the reward function wrote (``reward_vina``, ``reward_qed``). Coercing them
    here keeps ``json.dumps`` from falling back to ``str`` and silently shipping
    ``tensor(0.71)`` as a number's text.
    """
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except (ValueError, RuntimeError):
            pass
    return str(value)


def _read_route(traj: Any) -> list[dict[str, Any]]:
    """The reaction trajectory as steps.

    ``RxnFlowSampler.iter`` has already passed the raw trajectory through
    ``SynthesisEnvContext.read_traj``, which yields ``(smiles, action_repr)``
    pairs where ``action_repr`` is ``("FirstBlock", block)``, ``("UniRxn",
    template)``, ``("BiRxn", template, block)`` or ``("Stop",)``. Each step is
    therefore the *state* plus the action taken from it — the SMARTS and the
    building block that made the next state.
    """
    steps: list[dict[str, Any]] = []
    for index, entry in enumerate(traj or []):
        try:
            smiles, action = entry
        except (TypeError, ValueError):
            steps.append({"step": index, "smiles": None, "action": _jsonable(entry)})
            continue
        steps.append({"step": index, "smiles": str(smiles), "action": [str(a) for a in action]})
    return steps


def _check_inputs(request: dict[str, Any]) -> tuple[str, str, str]:
    """Resolve and validate the three paths RxnFlow cannot start without."""
    env_dir = str(request.get("env_dir") or "")
    checkpoint = str(request.get("checkpoint") or "")
    protein = str(request.get("protein_pdb") or "")
    if not env_dir:
        raise ValueError("request carries no env_dir; RxnFlow cannot build a SynthesisEnv without one")
    if not checkpoint:
        raise ValueError("request carries no checkpoint path")
    if not protein:
        raise ValueError("request carries no protein_pdb; the pocket-conditional model needs a structure")

    missing = [name for name in _ENV_FILES if not (Path(env_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"synthesis environment {env_dir} is incomplete, missing {', '.join(missing)}. Build one "
            f"with `python data/scripts/b_create_env.py -b <blocks.smi[.gz]> "
            f"-t data/templates/real.txt -o {env_dir}`"
        )
    if not Path(checkpoint).is_file():
        name = Path(checkpoint).name
        drive_id = _DRIVE_IDS.get(name)
        fetch = f" Fetch it with `gdown {drive_id} -O {checkpoint}`." if drive_id else ""
        raise FileNotFoundError(f"GFlowNet checkpoint {checkpoint} does not exist.{fetch}")
    if not Path(protein).is_file():
        raise FileNotFoundError(f"protein structure {protein} does not exist")
    return env_dir, checkpoint, protein


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Zero-shot pocket-conditional sampling, exactly as ``sampling_zeroshot.py`` does it."""
    env_dir, checkpoint, protein = _check_inputs(request)

    # Imported after the paths are checked so a missing file is a sentence rather
    # than thirty seconds of torch import followed by one.
    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.multi_pocket import ProxySampler

    warnings: list[str] = []
    device = str(request.get("device") or "cpu")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            warnings.append("CUDA is unavailable in the RxnFlow environment; sampling on CPU instead")
            device = "cpu"
    # ProxySampler compares `self.device == "cuda"` upstream, so the bare word is
    # what sampling_zeroshot.py passes; "cuda:0" would silently skip its seeding.
    device = "cuda" if device.startswith("cuda") else "cpu"

    n_molecules = int(request.get("n_molecules") or 100)
    batch_size = max(1, int(request.get("batch_size") or 64))
    seed = int(request.get("seed") or 1)
    temperature = str(request.get("temperature") or "uniform-16-64")
    calc_reward = bool(request.get("calc_reward"))

    center = request.get("center")
    ref_ligand = request.get("ref_ligand")
    if center is not None:
        center = tuple(float(c) for c in center)
        if len(center) != 3:
            raise ValueError(f"center must be three floats, got {len(center)}")
        if ref_ligand:
            warnings.append("both center and ref_ligand were given; RxnFlow ignores the reference ligand")
    elif not ref_ligand:
        raise ValueError("one of center or ref_ligand is required to place the pocket")
    elif not Path(str(ref_ligand)).is_file():
        raise FileNotFoundError(f"reference ligand {ref_ligand} does not exist")

    # Config: only these four fields are overridden. Everything else — the model
    # dimensions, the temperature family, the reward objectives, the proxy key —
    # comes from the checkpoint's own stored `cfg`, which RxnFlowSampler.__init__
    # merges this on top of. Overriding more than sampling_zeroshot.py does would
    # risk tripping its check_gfn_condition assertions.
    config = init_empty(Config())
    config.seed = seed
    config.env_dir = env_dir
    config.algo.num_from_policy = batch_size
    config.algo.action_subsampling.sampling_ratio = float(request.get("subsampling_ratio") or 0.1)

    sampler = ProxySampler(config, checkpoint, device)
    sample_dist, dist_params = parse_temperature(temperature)
    sampler.update_temperature(sample_dist, dist_params)
    sampler.set_pocket(protein, center, ref_ligand)

    tick = time.time()
    res = sampler.sample(n_molecules, calc_reward=calc_reward)
    sampling_seconds = time.time() - tick

    molecules = []
    for index, sample in enumerate(res):
        molecules.append(
            {
                "smiles": sample.get("smiles"),
                "sample_index": index,
                "route": _read_route(sample.get("traj")),
                "info": _jsonable(sample.get("info") or {}),
            }
        )
    if len(molecules) < n_molecules:
        warnings.append(
            f"RxnFlow returned {len(molecules)} of {n_molecules} requested; the sampler drops "
            "trajectories it marks invalid"
        )

    sys.stderr.write(f"RxnFlow: {len(molecules)} molecules in {sampling_seconds:.1f}s on {device}\n")
    return {
        "molecules": molecules,
        "n_requested": n_molecules,
        "n_sampled": len(molecules),
        "sampling_seconds": round(sampling_seconds, 3),
        "device": device,
        "seed": seed,
        "temperature": temperature,
        "checkpoint": checkpoint,
        "env_dir": env_dir,
        "calc_reward": calc_reward,
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
