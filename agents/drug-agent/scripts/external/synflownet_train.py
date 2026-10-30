#!/usr/bin/env python3
"""Train the one artifact SynFlowNet does not publish: a policy for one reward.

SynFlowNet is a GFlowNet, and a GFlowNet is not a prior you load and sample. It is
a sampler *for one reward function*, and upstream ships no trained one — neither
``mirunacrt/synflownet`` nor ``recursionpharma/synflownet-boltz`` has a GitHub
release, a ``weights/`` directory or a HuggingFace repo, and the paper's
"source code is available at ..." is the whole of what was released. So the
checkpoint that ``molecule.generate.synflownet`` reads has to be produced here,
once, by running upstream's own trainer. This script is that step, and it is
deliberately separate from :mod:`synflownet_sample`: sampling is a runtime
operation that must never touch the network or a GPU-week; training is a setup
operation that is expected to cost hours.

**It runs upstream's configuration, not a convenient one.** Every value below is
copied from ``src/synflownet/tasks/reactions_task.py::main`` at the pinned
commit, and the paper's sEH experiment (arXiv:2405.01155, Sec. 4) reports the
same: 5,000 training steps, batch 64, ``max_len`` 3, reward exponent β = 32, the
sEH binding proxy of Bengio et al. 2021 as the reward. Three settings differ from
``main()`` and **none of them touches the learned policy**:

``validate_every`` 500 → 0
    Validation draws batches under ``torch.no_grad`` and updates nothing. It is
    off because it costs a quarter of the wall clock and its metrics are not what
    this repository ranks on.
``num_final_gen_steps`` 100 → 0
    Upstream generates a final sample set into a SQLite log at the end of
    training. This repository samples through ``scripts/external/synflownet_sample.py``
    instead, so the same molecules would be produced twice.
``checkpoint_every`` None → 250
    Upstream checkpoints only at validation, which with validation off means only
    at the very end. 250 keeps a five-hour run resumable, and ``--resume`` is what
    makes that word mean something: it rebuilds the trainer from the checkpoint's
    *own* stored config, so the action space and every hyperparameter are the ones
    the policy was trained with, restores the model, the target network and the
    optimizer, and fast-forwards the learning-rate schedules by the steps already
    taken. Two things a resume cannot restore and the card therefore records: the
    Adam moments of the separate ``logZ`` optimizer, which upstream's
    ``_get_state`` never saves, and the sampler's RNG stream.

Anything else the caller wants changed is a flag, and every flag is written into
``train_card.json`` beside the checkpoint, so a checkpoint can always be read
back to the settings that made it.

**The reward is the point, and it is local.** ``seh_reaction`` scores each
molecule with the pretrained MPNN from *Flow Network based Generative Models for
Non-Iterative Diverse Candidate Generation* (Bengio et al., NeurIPS 2021), which
upstream vendors into the clone as
``src/synflownet/models/cache/bengio2021flow_proxy.pkl.gz`` — 625,026 parameters,
loaded from that file, no download. sEH is soluble epoxide hydrolase, which is
this repository's own target (EPHX2 / UniProt P34913), so the released proxy and
the platform's brief are pointed at the same protein. That is a coincidence worth
naming rather than relying on: the proxy is a 2021 ligand-based model, not a
docking of these molecules into 3I28.

Usage (inside the SynFlowNet environment, e.g. ``drug-agent-synflownet``)::

    python scripts/external/synflownet_train.py --log-dir .external/synflownet/checkpoints/seh_reaction

The result is ``<log-dir>/model_state.pt`` plus ``<log-dir>/train_card.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess  # nosec B404 - fixed argv, used only to record the checkout sha
import sys
import time
from pathlib import Path

#: Upstream's ``main()`` defaults, which are also the paper's sEH settings.
UPSTREAM_TRAINING_STEPS = 5000
UPSTREAM_BATCH_SIZE = 64
UPSTREAM_MAX_LEN = 3
UPSTREAM_BETA = 32.0
UPSTREAM_LEARNING_RATE = 1e-4
UPSTREAM_LR_DECAY = 2000
UPSTREAM_SAMPLING_TAU = 0.99
UPSTREAM_FINGERPRINT = "morgan_1024"

#: Rewards ``ReactionTask`` implements. Only ``seh_reaction`` and ``seh_qed`` are
#: offline: ``gsk``/``drd2`` reach PyTDC, which downloads an oracle on first use,
#: and ``vina`` needs a compiled QuickVina2-GPU binary that is not in this repo.
OFFLINE_REWARDS = ("seh_reaction", "seh_qed")


def _checkout_sha(root: Path) -> str:
    """The commit the trainer actually ran from, or ``"unknown"``."""
    try:
        out = subprocess.run(  # nosec B603 - fixed argv, no shell
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def build_config(args: argparse.Namespace):
    """Upstream's ``main()`` configuration, with the three documented deviations."""
    import torch
    from synflownet.config import Config, init_empty

    cfg = Config()
    config = init_empty(cfg)
    config.reward = args.reward
    config.print_every = args.print_every
    config.log_dir = str(args.log_dir)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = args.overwrite
    config.num_training_steps = args.steps
    config.validate_every = 0
    config.num_final_gen_steps = 0
    config.checkpoint_every = args.checkpoint_every
    config.num_workers = args.num_workers
    config.opt.learning_rate = UPSTREAM_LEARNING_RATE
    config.opt.lr_decay = UPSTREAM_LR_DECAY
    config.algo.sampling_tau = UPSTREAM_SAMPLING_TAU
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [args.beta]
    config.model.graph_transformer.continuous_action_embs = True
    config.model.graph_transformer.fingerprint_type = UPSTREAM_FINGERPRINT
    config.algo.max_len = UPSTREAM_MAX_LEN
    config.algo.tb.backward_policy = "MaxLikelihood"
    config.algo.tb.do_parameterize_p_b = True
    config.algo.tb.do_sample_p_b = False
    config.replay.use = False
    config.algo.synthesis_cost_as_bck_reward = False
    config.algo.tb.reinforce_loss_multiplier = 1.0
    config.algo.tb.bck_entropy_loss_multiplier = 1.0
    config.algo.num_from_policy = args.batch_size
    config.algo.num_from_buffer_for_pb = args.batch_size
    config.algo.strict_forward_policy = False
    config.algo.strict_bck_masking = False
    config.algo.tb.do_correct_idempotent = False
    return config


def resume(args: argparse.Namespace) -> tuple[object, int]:
    """Rebuild the trainer from ``<log-dir>/model_state.pt`` and carry on.

    The config comes from the checkpoint rather than from the flags, because a
    GFlowNet policy is only meaningful against the action space it was trained
    over and re-deriving that from arguments is how the two quietly diverge.

    ``log_dir`` is redirected to a scratch directory for construction only:
    upstream's ``GFNTrainer.setup`` calls ``shutil.rmtree`` on its log dir when
    ``overwrite_existing_exp`` is set, and the checkpoint being resumed lives
    inside it. It is pointed back afterwards, so saves land where they belong.
    """
    import tempfile

    import torch
    from synflownet.config import Config
    from synflownet.tasks.reactions_task import ReactionTrainer

    checkpoint = args.log_dir / "model_state.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--resume needs a checkpoint at {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    started_at = int(state["step"])

    scratch = Path(tempfile.mkdtemp(prefix="synflownet_resume_"))
    config = Config(**state["cfg"])
    config.log_dir = str(scratch)
    config.overwrite_existing_exp = True
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.num_training_steps = args.steps
    config.start_at_step = started_at
    config.checkpoint_every = args.checkpoint_every
    config.print_every = args.print_every
    config.num_workers = args.num_workers
    config.validate_every = 0
    config.num_final_gen_steps = 0

    trial = ReactionTrainer(config)
    trial.model.load_state_dict(state["models_state_dict"][0])
    if "sampling_model_state_dict" in state and trial.sampling_model is not trial.model:
        trial.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])
    trial.opt.load_state_dict(state["optimizer_state_dict"])
    # The checkpoint is read with map_location="cpu" so it loads on a machine with
    # no GPU, which leaves Adam's moments on the CPU while `run()` moves the
    # parameters to CUDA. torch then raises "found at least two devices" on the
    # first optimizer step, tens of seconds into what looks like a healthy resume.
    device = torch.device(config.device)
    for slot in trial.opt.state.values():
        for key, value in slot.items():
            if torch.is_tensor(value):
                slot[key] = value.to(device)
    # The schedules are recreated fresh by setup(), so without this the learning
    # rate would jump back to its day-one value at the resume point.
    for _ in range(started_at):
        trial.lr_sched.step()
        trial.lr_sched_Z.step()
    trial.cfg.log_dir = str(args.log_dir)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"resumed {checkpoint} at step {started_at}", flush=True)
    return trial, started_at


def write_card(args: argparse.Namespace, *, seconds: float, checkout: Path, n_blocks: int) -> Path:
    """Record what produced this checkpoint, beside the checkpoint.

    A GFlowNet checkpoint is meaningless without its action space and its reward:
    sampled against a different building-block file it indexes into a different
    catalogue, and its samples are a different distribution wearing the same name.
    The card is what lets the sampler check that it is not doing that.
    """
    card = {
        "artifact": "synflownet_policy",
        "upstream": "https://github.com/mirunacrt/synflownet",
        "upstream_sha": _checkout_sha(checkout),
        "trained_by": "scripts/external/synflownet_train.py",
        "reward": args.reward,
        "reward_model": (
            "bengio2021flow sEH binding proxy, loaded from "
            "src/synflownet/models/cache/bengio2021flow_proxy.pkl.gz in the clone"
        ),
        "num_training_steps": args.steps,
        "batch_size": args.batch_size,
        "max_len": UPSTREAM_MAX_LEN,
        "beta": args.beta,
        "fingerprint_type": UPSTREAM_FINGERPRINT,
        "building_blocks_file": "src/synflownet/data/building_blocks/enamine_bbs.txt",
        "n_building_blocks": n_blocks,
        "templates_file": "src/synflownet/data/templates/hb.txt",
        "deviations_from_upstream_main": {
            "validate_every": "500 -> 0 (evaluation only; does not update the policy)",
            "num_final_gen_steps": "100 -> 0 (sampling is done by synflownet_sample.py)",
            "checkpoint_every": "None -> 250 (upstream checkpoints only at validation)",
        },
        "resumed_from_step": args.resumed_from,
        "resume_caveats": (
            None
            if not args.resumed_from
            else (
                "restarted from a 250-step checkpoint: the logZ optimizer's Adam moments "
                "(never saved by upstream's _get_state) and the sampler's RNG stream begin "
                "again at the resume point; the model, the target network, the main "
                "optimizer and both learning-rate schedules are carried over"
            )
        ),
        "steps_this_run": args.steps - args.resumed_from,
        "train_seconds_this_run": round(seconds, 1),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        card["torch"] = torch.__version__
        card["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            card["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a hard dependency of the trainer
        pass
    path = Path(args.log_dir) / "train_card.json"
    path.write_text(json.dumps(card, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-dir", required=True, type=Path, help="where model_state.pt is written")
    parser.add_argument("--reward", default="seh_reaction", choices=OFFLINE_REWARDS)
    parser.add_argument("--steps", type=int, default=UPSTREAM_TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=UPSTREAM_BATCH_SIZE)
    parser.add_argument("--beta", type=float, default=UPSTREAM_BETA)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from <log-dir>/model_state.pt instead of starting over",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing --log-dir; upstream's trainer refuses to write into one",
    )
    args = parser.parse_args(argv)

    # Upstream's `GFNTrainer.setup` refuses an existing log dir unless told to
    # delete it, so the directory is created by the trainer, never here.
    args.log_dir = args.log_dir.expanduser().resolve()
    args.resumed_from = 0
    if args.log_dir.exists() and not args.overwrite and not args.resume:
        sys.stderr.write(
            f"{args.log_dir} already exists; pass --overwrite to retrain into it, or point "
            "--log-dir somewhere else. A checkpoint is not overwritten by accident.\n"
        )
        return 2
    args.log_dir.parent.mkdir(parents=True, exist_ok=True)

    from synflownet.envs.synthesis_building_env import BUILDING_BLOCKS, PRECOMPUTED_BB_MASKS, TEMPLATES

    if BUILDING_BLOCKS is None or PRECOMPUTED_BB_MASKS is None:
        sys.stderr.write(
            "the building-block catalogue or its precomputed masks are missing from the clone; "
            "run `python scripts/external/synflownet_setup.py --steps data masks` first\n"
        )
        return 2
    import synflownet

    checkout = Path(synflownet.__file__).resolve().parents[2]
    print(
        f"action space: {len(TEMPLATES)} templates, {len(BUILDING_BLOCKS)} building blocks, "
        f"masks {PRECOMPUTED_BB_MASKS.shape}",
        flush=True,
    )

    started = time.perf_counter()
    if args.resume:
        trial, args.resumed_from = resume(args)
    else:
        from synflownet.tasks.reactions_task import ReactionTrainer

        trial = ReactionTrainer(build_config(args))
    print(
        f"trainer built in {time.perf_counter() - started:.1f}s; training to step {args.steps} "
        f"from {args.resumed_from}",
        flush=True,
    )
    trial.run()
    seconds = time.perf_counter() - started

    checkpoint = args.log_dir / "model_state.pt"
    if not checkpoint.is_file():
        sys.stderr.write(f"training finished but wrote no checkpoint at {checkpoint}\n")
        return 1
    card = write_card(args, seconds=seconds, checkout=checkout, n_blocks=len(BUILDING_BLOCKS))
    print(f"checkpoint {checkpoint} ({checkpoint.stat().st_size:,} bytes)")
    print(f"card       {card}")
    ran = max(args.steps - args.resumed_from, 1)
    print(f"seconds    {seconds:.1f} for {ran} steps ({seconds / ran:.3f} s/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
