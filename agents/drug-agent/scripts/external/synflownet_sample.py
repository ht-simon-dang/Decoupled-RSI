"""Sample molecules-with-routes from a trained SynFlowNet policy — the drug-agent bridge.

Runs inside whichever interpreter has ``synflownet`` (``drug-agent-synflownet``),
imports nothing from ``drug_agent``, and talks to the adapter through a JSON
request/response pair on disk. It performs **no network I/O**: the policy comes
from a local ``model_state.pt``, the action space from
``src/synflownet/data/building_blocks/enamine_bbs.txt`` and
``src/synflownet/data/templates/hb.txt`` inside the clone, and the sEH reward
proxy from ``src/synflownet/models/cache/bengio2021flow_proxy.pkl.gz``, which
upstream vendors as a git blob.

**Which entry points this calls, and where each came from.** Nothing is invented;
each call is the one upstream's own sampling path makes.

``ReactionTrainer(config)``
    ``tasks/reactions_task.py::main``. Building the trainer is how the reaction
    environment, the context, the task and the sampler get constructed together;
    there is no lighter constructor, and half-building them by hand would risk a
    context whose action indices do not match the checkpoint's output layer.

``model.load_state_dict(state["models_state_dict"][0])``
    ``online_trainer.py::StandardOnlineTrainer.load_from_checkpoint``, which does
    exactly this after ``Config(**state["cfg"])``.

``task.sample_conditional_information(n, 0)["encoding"]`` then
``algo.create_training_data_from_own_samples(model, n, cond_info, 0.0)``
    ``data/data_source.py::do_sample_model_n_times``, lines 133-135. This is the
    call the final-generation loop makes, with ``random_action_prob`` at 0.

``task.compute_obj_properties(mols, traj_lens)``
    ``tasks/reactions_task.py::ReactionTask``. Optional here, and reported as
    ``reward_seh_proxy`` on the raw record rather than as an observable — it is
    the 2021 ligand-based MPNN's estimate of an sEH binding score, not a docking
    and not a measurement.

**Three upstream behaviours this driver has to defend against**, all found by
running it rather than by reading it:

``load_from_checkpoint would delete the checkpoint``
    ``GFNTrainer.setup`` does ``shutil.rmtree(cfg.log_dir)`` when the stored
    config has ``overwrite_existing_exp=True``, and a training run's stored
    config points ``log_dir`` at the directory the checkpoint is *in*. Rebuilding
    a trainer from a checkpoint with its own config would therefore erase the
    thing being loaded. ``log_dir`` is redirected into the driver's scratch
    directory before the trainer is constructed, and never points at the
    checkpoint.

``torch.load defaults to weights_only=True from torch 2.6``
    The checkpoint stores an OmegaConf ``DictConfig`` beside the tensors, which
    the restricted unpickler refuses. It is loaded with ``weights_only=False``
    because this file is produced by this repository's own
    ``scripts/external/synflownet_train.py``; a checkpoint from anywhere else is
    an untrusted pickle and should not be pointed at this driver.

``the last sampled action is not applied to the molecule``
    ``SynthesisSampler.sample_from_model`` samples an action at every ``t`` up to
    ``max_len``, but at ``t == max_len - 1`` it sets ``done`` and continues the
    loop *before* ``graphs[i] = g`` — so the final reaction is drawn, recorded in the
    trajectory, and its product thrown away. With upstream's ``max_len = 3`` a
    returned molecule is therefore a first building block plus **one** applied
    reaction, with a second reaction sitting in the trajectory that never touched
    it. Training is consistent with this, because the reward is computed on the
    same truncated state — but a route that listed that action would be claiming a
    step the molecule does not contain. Each step here therefore carries
    ``state_before``, ``state_after`` and ``applied``, and ``applied`` is decided
    by whether the state actually changed rather than by counting actions. A
    trajectory the sampler flagged ``is_valid=False`` keeps the flag rather than
    being silently dropped.

Request JSON:
    checkpoint       absolute path to a ``model_state.pt`` written by synflownet_train.py
    n_molecules      how many trajectories to sample in total
    batch_size       trajectories per sampler call
    seed             RNG seed for torch/numpy/random
    device           "cuda" / "cuda:0" / "cpu"
    random_action_prob  exploration noise; 0.0 samples the policy itself
    compute_reward   score each molecule with the checkpoint's own reward proxy
    out_dir          where to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]      {smiles, sample_index, is_valid, n_steps, route[], reward_seh_proxy}
    n_requested, n_sampled, checkpoint, train_card, reward, beta,
    n_building_blocks, n_templates, n_bimolecular_rxns, n_unimolecular_rxns,
    device, torch_version, seconds_setup, seconds_sample, warnings[], errors[]

Each ``route`` entry is one forward action:
    {step, action, state_before, state_after, applied, building_block, template,
     rxn_index, bb_index}
``action`` is upstream's own ``GraphActionType`` name — ``AddFirstReactant``,
``ReactUni``, ``ReactBi`` or ``Stop`` — and ``applied`` says whether it changed
the molecule.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _driver

#: Upstream's own name for "sample the policy, do not explore".
DEFAULT_RANDOM_ACTION_PROB = 0.0


def _resolve_device(torch, requested, warnings):
    """The device to run on, falling back to CPU with a warning rather than crashing."""
    device = requested or "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append(
            "CUDA is not available in this interpreter, so SynFlowNet sampled on CPU; "
            "the graph transformer and the sEH proxy both run there, only slower."
        )
        return "cpu"
    return device


def _seed_everything(torch, seed):
    """Seed every generator the sampling path draws from."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_state(torch, checkpoint):
    """The checkpoint dict, or a message an operator can act on.

    ``weights_only=False`` is deliberate and is safe only because this file is
    written by ``scripts/external/synflownet_train.py`` on this machine — see the
    module docstring.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"no SynFlowNet policy at {path}. SynFlowNet publishes no pretrained checkpoint; "
            "produce one with `python scripts/external/synflownet_setup.py --train`."
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("models_state_dict", "cfg"):
        if key not in state:
            raise ValueError(f"{path} is not a SynFlowNet checkpoint: it has no {key!r}")
    return state


def _build_trainer(state, *, device, scratch_dir, warnings):
    """Rebuild the trainer the checkpoint was trained by, pointed somewhere safe.

    The stored config's ``log_dir`` is the directory the checkpoint lives in, and
    upstream's ``setup()`` deletes it. Redirecting it is not a preference.
    """
    from synflownet.config import Config
    from synflownet.tasks.reactions_task import ReactionTrainer

    config = Config(**state["cfg"])
    config.log_dir = str(scratch_dir)
    config.overwrite_existing_exp = True
    config.device = device
    config.num_workers = 0
    config.num_training_steps = 0
    config.num_final_gen_steps = 0
    config.validate_every = 0
    trial = ReactionTrainer(config)
    trial.model.load_state_dict(state["models_state_dict"][0])
    trial.model.to(device)
    trial.model.eval()
    if state.get("step") is not None:
        warnings.append(f"policy loaded at training step {state['step']}")
    return trial


def _route_from_traj(ctx, traj):
    """Turn one sampled trajectory into the synthesis route that produced it.

    The route is the whole reason this generator exists, so it is reconstructed
    from the action objects the sampler returned rather than re-derived from the
    product — a retro-guess about how a molecule *could* be made is a different
    claim from the pathway the policy actually walked.

    ``applied`` is the load-bearing field. It is computed from the states, not
    from the action type: an action counts as applied only when the state after it
    differs from the state before, which is what makes the ``max_len`` truncation
    in the module docstring visible instead of silently inflating the route.
    """
    states = [ctx.object_to_log_repr(graph) for graph, _ in traj]
    steps = []
    for index, (_graph, action) in enumerate(traj):
        name = getattr(action.action, "name", str(action.action))
        before = states[index]
        after = states[index + 1] if index + 1 < len(states) else None
        entry = {
            "step": index,
            "action": name,
            "state_before": before,
            "state_after": after,
            "applied": bool(after is not None and after != before and name != "Stop"),
        }
        if name in {"AddFirstReactant", "AddReactant"}:
            entry["bb_index"] = int(action.bb)
            entry["building_block"] = ctx.building_blocks[int(action.bb)]
            # The first action turns the empty graph into a block, so "changed"
            # is the right test for it too, and `before` is the empty string.
        elif name == "ReactUni":
            entry["rxn_index"] = int(action.rxn)
            entry["template"] = ctx.unimolecular_reactions[int(action.rxn)].template
        elif name == "ReactBi":
            entry["rxn_index"] = int(action.rxn)
            entry["template"] = ctx.bimolecular_reactions[int(action.rxn)].template
            entry["bb_index"] = int(action.bb)
            entry["building_block"] = ctx.building_blocks[int(action.bb)]
        steps.append(entry)
    return steps


def _score(trial, mols, traj_lens, warnings):
    """The checkpoint's own reward for these molecules, or ``None`` per molecule.

    A failure here is a warning, never an error: the reward is diagnostic context
    on a raw record, and refusing to return molecules because a proxy would not
    load would throw away the thing that was asked for.
    """
    import torch

    try:
        with torch.no_grad():
            props, is_valid = trial.task.compute_obj_properties(mols, torch.tensor(traj_lens))
    # A broad except on purpose: the reward is optional context on a raw record.
    except Exception as exc:
        warnings.append(f"reward proxy did not score this batch ({type(exc).__name__}: {exc})")
        return [None] * len(mols)
    flat = props.reshape(-1).tolist()
    scores, cursor = [], 0
    for ok in is_valid.tolist():
        if ok and cursor < len(flat):
            scores.append(float(flat[cursor]))
            cursor += 1
        else:
            scores.append(None)
    return scores


def _sample_batch(trial, *, n, device, random_action_prob, compute_reward, warnings):
    """One sampler call, returned as a list of raw records."""
    import torch
    from rdkit import Chem

    ctx = trial.ctx
    cond_info = trial.task.sample_conditional_information(n, 0)
    with torch.no_grad():
        trajs = trial.algo.create_training_data_from_own_samples(
            trial.model, n, cond_info["encoding"].to(device), random_action_prob
        )
    mols, traj_lens, routes, valids = [], [], [], []
    for traj in trajs:
        steps = _route_from_traj(ctx, traj["traj"])
        try:
            mol = ctx.graph_to_obj(traj["result"])
        # A broad except on purpose: a broken product is a record, not a crash.
        except Exception as exc:
            warnings.append(f"a sampled trajectory produced no molecule ({type(exc).__name__}: {exc})")
            mol = None
        mols.append(mol)
        traj_lens.append(len(traj["traj"]))
        routes.append(steps)
        valids.append(bool(traj.get("is_valid", True)))
    kept = [(m, n) for m, n in zip(mols, traj_lens, strict=True) if m is not None]
    scores = (
        _score(trial, [m for m, _ in kept], [n for _, n in kept], warnings)
        if compute_reward
        else [None] * len(kept)
    )
    records, cursor = [], 0
    for mol, steps, valid in zip(mols, routes, valids, strict=True):
        smiles = Chem.MolToSmiles(mol) if mol is not None else ""
        score = None
        if mol is not None:
            score = scores[cursor] if cursor < len(scores) else None
            cursor += 1
        records.append(
            {
                "smiles": smiles,
                "is_valid": valid,
                "n_steps": len(steps),
                "route": steps,
                "reward_seh_proxy": score,
            }
        )
    return records


def run(request):
    """Sample ``n_molecules`` trajectories from a trained policy."""
    import json

    import torch

    warnings, errors = [], []
    out_dir = Path(request["out_dir"])
    device = _resolve_device(torch, request.get("device"), warnings)
    seed = int(request.get("seed", 0))
    _seed_everything(torch, seed)
    torch.set_grad_enabled(False)

    checkpoint = Path(request["checkpoint"]).expanduser()
    state = _load_state(torch, checkpoint)

    started = time.perf_counter()
    scratch = out_dir / "trainer_scratch"
    trial = _build_trainer(state, device=device, scratch_dir=scratch, warnings=warnings)
    seconds_setup = time.perf_counter() - started

    n_requested = int(request.get("n_molecules", 10))
    batch_size = max(1, int(request.get("batch_size", 10)))
    random_action_prob = float(request.get("random_action_prob", DEFAULT_RANDOM_ACTION_PROB))
    # The reward is diagnostic context, not the product, so a caller may turn it
    # off. It is on by default because a GFlowNet sample is uninterpretable
    # without the quantity the policy was trained to sample in proportion to.
    compute_reward = bool(request.get("compute_reward", True))

    started = time.perf_counter()
    records = []
    while len(records) < n_requested:
        take = min(batch_size, n_requested - len(records))
        records.extend(
            _sample_batch(
                trial,
                n=take,
                device=device,
                random_action_prob=random_action_prob,
                compute_reward=compute_reward,
                warnings=warnings,
            )
        )
    seconds_sample = time.perf_counter() - started
    for index, record in enumerate(records):
        record["sample_index"] = index

    card_path = checkpoint.parent / "train_card.json"
    try:
        card = json.loads(card_path.read_text()) if card_path.is_file() else None
    except (OSError, ValueError):
        card = None
        warnings.append(f"could not read the training card beside the checkpoint at {card_path}")

    ctx = trial.ctx
    return {
        "molecules": records,
        "n_requested": n_requested,
        "n_sampled": len(records),
        "checkpoint": str(checkpoint),
        "train_card": card,
        "reward": str(trial.cfg.reward),
        "beta": list(trial.cfg.cond.temperature.dist_params),
        "n_building_blocks": int(ctx.num_building_blocks),
        "n_templates": len(ctx.reactions),
        "n_bimolecular_rxns": int(ctx.num_bimolecular_rxns),
        "n_unimolecular_rxns": int(ctx.num_unimolecular_rxns),
        "device": device,
        "torch_version": torch.__version__,
        "seconds_setup": round(seconds_setup, 2),
        "seconds_sample": round(seconds_sample, 2),
        "warnings": warnings,
        "errors": errors,
    }


if __name__ == "__main__":
    _driver.main(run)
