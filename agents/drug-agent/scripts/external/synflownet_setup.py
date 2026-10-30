#!/usr/bin/env python3
"""Make SynFlowNet real on this machine: clone, environment, catalogue, masks, policy.

SynFlowNet is the one generator in this catalogue whose missing artifact cannot be
downloaded, because upstream never published it. ``mirunacrt/synflownet`` has no
GitHub release, no ``weights/`` directory and no HuggingFace repository, and
neither does Recursion's ``synflownet-boltz`` fork; a GFlowNet is a sampler *for
one reward*, and the reward is the user's choice. So this script is the whole of
"install SynFlowNet", and its last step is a training run rather than a download.

Everything that touches the network happens here and happens once. Sampling —
``scripts/external/synflownet_sample.py``, driven by
``drug_agent.tools.generation.synflownet_tool`` — reads only local files and makes
no request of any kind, which is what lets the platform run with
``allow_network: false``.

    python scripts/external/synflownet_setup.py --list
    python scripts/external/synflownet_setup.py             # everything but the training
    python scripts/external/synflownet_setup.py --train     # ... and the 4.7 GPU-hour run
    python scripts/external/synflownet_setup.py --only data masks

Every step is idempotent and says what it found rather than redoing it.

**The five steps, and what each is actually for.**

``clone``
    ``git clone`` at the sha pinned in ``CHECKOUTS.tsv``. Upstream renames entry
    points; a floating clone is how an adapter starts failing for reasons that
    look like this repository's fault.

``env``
    A dedicated ``drug-agent-synflownet`` conda environment. SynFlowNet pins
    ``torch==2.1.2`` with PyG wheels for that torch, and torch 2.1's SASS stops at
    sm_90 — it cannot launch a kernel on this host's sm_120 card at all. The pin
    is therefore raised to ``torch 2.7.1+cu128``, whose ``torch_scatter`` /
    ``torch_sparse`` / ``torch_cluster`` wheels exist prebuilt at data.pyg.org and
    were verified here to run real CUDA kernels on sm_120. Two dependencies are
    installed that upstream's ``pyproject.toml`` does not name: ``setuptools<81``,
    because PyTDC imports the removed ``pkg_resources`` at module scope, and
    ``gemmi``, which meeko imports and which ``reactions_task.py`` reaches through
    ``utils/gpu_vina.py`` on the way in.

``data``
    The building-block catalogue, the artifact that gated ReaSyn and does not gate
    this. Upstream's ``ReactionTaskConfig`` wants ``enamine_bbs.txt`` and its
    README sends you to Enamine "upon request"; Recursion's MIT-licensed fork
    publishes an Enamine-derived, template-matching, sanitised list as an ordinary
    git blob. It is fetched from a pinned commit, verified on size and sha256,
    de-duplicated, and then — this is the part that matters — re-sanitised
    **locally** with upstream's own ``sanitize_building_blocks.py``, because the
    published file is canonical under whichever RDKit wrote it and not under the
    one here. Ninety blocks differ. The step ends by asserting the catalogue is
    round-trip closed, which is the property training silently depends on. A data
    card is written beside it recording source, commit, hashes and counts.

``masks``
    Upstream's ``precompute_bb_masks.py``: for each of the 92 bi-molecular
    templates and each reactant position, which blocks match. ~2.5 minutes and a
    330 MB pickle, and the environment refuses to build without it.

``train``
    ``scripts/external/synflownet_train.py`` at upstream's own settings. Measured
    on this host: 68.8 s to build the trainer, then 3.41 s per step at batch 64
    over a full run, so the 5,000 steps upstream and the paper both use are
    4.7 GPU-hours and 320,000 scored trajectories. It writes ``model_state.pt``
    every 250 steps and ``train_card.json`` only on completion — which is the
    artifact the adapter actually requires, so a run killed at hour one leaves the
    capability UNAVAILABLE rather than half-trained and claiming otherwise. Run
    this step again and it resumes from the last save instead of starting over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess  # nosec B404 - fixed argv lists, no shell, no user interpolation
import sys
import urllib.request
from pathlib import Path

REPO = next(
    p
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "pyproject.toml").is_file()
)
EXTERNAL = REPO / ".external"
CHECKOUT = EXTERNAL / "synflownet"

#: The upstream repository and the commit the driver and this script were written
#: against. Mirrors the row in ``CHECKOUTS.tsv``; ``make external-clone`` clones at
#: HEAD, and this script pins.
UPSTREAM_URL = "https://github.com/mirunacrt/synflownet"
UPSTREAM_SHA = "574f1e148f42e0c79877318fa9d84d2552cf5025"

#: Recursion's MIT-licensed fork, which publishes the Enamine-derived catalogue
#: that the original repository only describes. Pinned for the same reason.
BLOCKS_REPO = "https://github.com/recursionpharma/synflownet-boltz"
BLOCKS_SHA = "547dea3972a68c5a27389240976db6f2d9c02a20"
BLOCKS_PATH = "synflownet/data/building_blocks/enamine_bbs_hartenfeller_matching_sanitized.txt"
BLOCKS_URL = f"https://raw.githubusercontent.com/recursionpharma/synflownet-boltz/{BLOCKS_SHA}/{BLOCKS_PATH}"
#: Read from the server on 2026-09-08. A mismatch means upstream republished, not
#: that the download is fine and the check is fussy.
BLOCKS_SIZE = 5_874_496
BLOCKS_SHA256 = "bfdb9b2692d5a49f2af4a5974f3d557d29a699d71eefaf750fbb3442d2825d9f"

ENV_NAME = "drug-agent-synflownet"
PYTHON_VERSION = "3.10"
TORCH_VERSION = "2.7.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PYG_FIND_LINKS = f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu128.html"

BB_DIR = CHECKOUT / "src" / "synflownet" / "data" / "building_blocks"
BLOCKS_FILE = BB_DIR / "enamine_bbs.txt"
MASKS_FILE = BB_DIR / "precomputed_bb_masks_enamine_bbs.pkl"
DATA_CARD = BB_DIR / "enamine_bbs_data_card.json"
CHECKPOINT_DIR = CHECKOUT / "checkpoints" / "seh_reaction"

STEPS = ("clone", "env", "data", "masks", "train")
UA = {"User-Agent": "drug-agent/0.1 (+synflownet_setup.py)"}


def _run(argv: list[str], *, cwd: Path | None = None) -> None:
    """Run *argv*, echoing it first and raising on failure."""
    print("  $ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=str(cwd) if cwd else None, check=True)  # nosec B603 - fixed argv, no shell


def _conda() -> str:
    """The conda executable, or a message naming what to install."""
    exe = shutil.which("conda") or shutil.which("mamba")
    if exe is None:
        raise SystemExit("no `conda` on PATH; this project's environments are conda environments")
    return exe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --- steps ----------------------------------------------------------------
def step_clone() -> None:
    """Clone SynFlowNet at the pinned commit."""
    if (CHECKOUT / ".git").is_dir():
        head = subprocess.run(  # nosec B603 - fixed argv, no shell
            ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        print(f"  = checkout present at {CHECKOUT} ({head[:12]})")
        if head != UPSTREAM_SHA:
            print(f"  ! HEAD is not the pinned {UPSTREAM_SHA[:12]}; the driver was written against that one")
        return
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--quiet", UPSTREAM_URL, str(CHECKOUT)])
    _run(["git", "-C", str(CHECKOUT), "checkout", "--quiet", UPSTREAM_SHA])
    print(f"  + cloned {UPSTREAM_URL} at {UPSTREAM_SHA[:12]}")


def step_env() -> None:
    """Create ``drug-agent-synflownet`` and install the stack that runs on this GPU."""
    conda = _conda()
    envs = subprocess.run(  # nosec B603 - fixed argv, no shell
        [conda, "env", "list"], capture_output=True, text=True, check=False
    ).stdout
    if f"/{ENV_NAME}\n" in envs or f"/{ENV_NAME} " in envs or f"{ENV_NAME} " in envs:
        print(f"  = conda environment {ENV_NAME} exists")
    else:
        _run([conda, "create", "-y", "-n", ENV_NAME, f"python={PYTHON_VERSION}"])
    pip = [conda, "run", "--no-capture-output", "-n", ENV_NAME, "pip", "install", "-q"]
    # torch first and from PyTorch's own index: the cu128 build is the one whose
    # SASS covers sm_120, and the PyG extension wheels below are compiled against
    # exactly this version.
    _run([*pip, f"torch=={TORCH_VERSION}", "--index-url", TORCH_INDEX])
    _run([*pip, "numpy<2", "torch_scatter", "torch_sparse", "torch_cluster", "-f", PYG_FIND_LINKS])
    _run(
        [
            *pip,
            "torch-geometric==2.4.0",
            "rdkit",
            "tables",
            "scipy",
            "networkx",
            "tensorboard",
            "cvxopt",
            "pyarrow",
            "gitpython",
            "botorch",
            "pyro-ppl",
            "gpytorch",
            "omegaconf>=2.3",
            "wandb",
            "pandas",
            "PyTDC==1.0.7",
            "meeko",
            "pyyaml",
            "tqdm",
            # Not in upstream's dependency list, and both are hard imports here:
            # PyTDC reaches `pkg_resources`, meeko reaches `gemmi`.
            "setuptools<81",
            "gemmi",
        ]
    )
    _run([*pip, "-e", str(CHECKOUT), "--no-deps"])
    _run(
        [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            ENV_NAME,
            "python",
            "-c",
            "import synflownet.tasks.reactions_task",
        ]
    )
    print(f"  + {ENV_NAME} imports the sampling path")


def _download(url: str, dest: Path, *, expect_size: int, expect_sha256: str) -> None:
    """Fetch *url* to *dest*, verified on size and digest. Idempotent."""
    if dest.is_file() and dest.stat().st_size == expect_size and _sha256(dest) == expect_sha256:
        print(f"  = {dest.name} already present ({expect_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  + {dest.name} <- {url}")
    # https, and the URL is a module constant pinned to a commit, not caller input.
    request = urllib.request.Request(url, headers=dict(UA))
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as fh:  # nosec B310
        shutil.copyfileobj(response, fh)
    size, digest = part.stat().st_size, _sha256(part)
    if size != expect_size or digest != expect_sha256:
        head = part.open("rb").read(16)
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"{url} returned {size:,} bytes / sha256 {digest}, expected {expect_size:,} / "
            f"{expect_sha256}. First bytes were {head!r}. Upstream may have republished; "
            "check before assuming a bad download."
        )
    part.replace(dest)


def step_data() -> None:
    """Fetch, de-duplicate and locally re-sanitise the building-block catalogue."""
    raw = BB_DIR / "enamine_bbs_hartenfeller_matching_sanitized.txt"
    _download(BLOCKS_URL, raw, expect_size=BLOCKS_SIZE, expect_sha256=BLOCKS_SHA256)

    lines = raw.read_text().splitlines()
    unique = sorted({line.strip() for line in lines if line.strip()})
    print(f"  · {len(lines):,} lines -> {len(unique):,} unique building blocks")
    BLOCKS_FILE.write_text("\n".join(unique) + "\n")

    # Upstream's own pipeline step, run against the *local* RDKit. The published
    # file is canonical under whichever RDKit produced it; ninety of these blocks
    # round-trip to a different string here, and for those the environment masks
    # the only legal backward action, which kills training with `Invalid log_p_B`.
    conda = _conda()
    sanitized = BB_DIR / "enamine_bbs_local_sanitized.txt"
    _run(
        [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            ENV_NAME,
            "python",
            str(BB_DIR / "sanitize_building_blocks.py"),
            "--building_blocks_filename",
            str(BLOCKS_FILE),
            "--output_filename",
            str(sanitized),
        ]
    )
    after = sorted({line.strip() for line in sanitized.read_text().splitlines() if line.strip()})
    # A set difference, not a positional one: both lists are sorted, so comparing
    # index by index counts every block the rewrite shifted rather than the ones it
    # actually changed, and reports thousands where there are ninety.
    rewritten = len(set(unique) - set(after))
    BLOCKS_FILE.write_text("\n".join(after) + "\n")
    sanitized.unlink(missing_ok=True)
    print(f"  · local sanitisation rewrote {rewritten} of {len(after):,} blocks")

    _run(
        [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            ENV_NAME,
            "python",
            "-c",
            "from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*');"
            "from synflownet.envs.synthesis_building_env import ReactionTemplateEnvContext as C;"
            "ctx=C();"
            "bad=[s for s in ctx.building_blocks if Chem.MolToSmiles("
            "ctx.graph_to_obj(ctx.obj_to_graph(Chem.MolFromSmiles(s)))) not in ctx.building_blocks_set];"
            "print(f'round-trip misses: {len(bad)}');"
            "assert not bad, bad[:5]",
        ]
    )
    _write_data_card(after, raw)


def _write_data_card(blocks: list[str], raw: Path) -> None:
    """Record where the catalogue came from, using the repository's snapshot machinery."""
    sys.path.insert(0, str(REPO / "src"))
    # Imported here rather than at module scope: every other step runs without
    # `drug_agent` importable, and only this one needs the snapshot machinery.
    from drug_agent.data.snapshot import DataSnapshot

    snapshot = DataSnapshot(
        name="synflownet_building_blocks",
        source=f"{BLOCKS_REPO}@{BLOCKS_SHA}:{BLOCKS_PATH}",
        kind="building_block_catalogue",
        records=[{"smiles": s} for s in blocks],
        license=(
            "redistributed under MIT by recursionpharma/synflownet-boltz; derived from "
            "Enamine's building-block catalogue, whose own terms apply to the compounds"
        ),
        metadata={
            "upstream_repo": BLOCKS_REPO,
            "upstream_commit": BLOCKS_SHA,
            "upstream_path": BLOCKS_PATH,
            "download_url": BLOCKS_URL,
            "downloaded_sha256": _sha256(raw),
            "downloaded_bytes": raw.stat().st_size,
            "n_lines_as_published": len(raw.read_text().splitlines()),
            "n_unique": len(blocks),
            "local_sha256": _sha256(BLOCKS_FILE),
            "post_processing": (
                "de-duplicated, then re-sanitised with upstream's "
                "sanitize_building_blocks.py against the local RDKit, then asserted "
                "round-trip closed under graph_to_obj(obj_to_graph(mol))"
            ),
            "templates": "src/synflownet/data/templates/hb.txt (105 Hartenfeller templates)",
        },
    )
    DATA_CARD.write_text(json.dumps(snapshot.data_card(), indent=2) + "\n")
    print(f"  + data card {DATA_CARD} ({snapshot.snapshot_id})")


def step_masks() -> None:
    """Precompute which building blocks match which template reactant position."""
    if MASKS_FILE.is_file():
        print(f"  = {MASKS_FILE.name} present ({MASKS_FILE.stat().st_size:,} bytes)")
        return
    conda = _conda()
    _run(
        [conda, "run", "--no-capture-output", "-n", ENV_NAME, "python", "precompute_bb_masks.py"],
        cwd=BB_DIR,
    )


def step_train(steps: int) -> None:
    """Train the policy upstream never published. ~4.7 GPU-hours at the default.

    Idempotent in the way a five-hour step has to be: a finished run is left
    alone, and a run that died part way is *resumed* from its last 250-step save
    rather than started over. The card is the discriminator — it is written only
    on completion, which is also why the adapter requires it.
    """
    if (CHECKPOINT_DIR / "train_card.json").is_file():
        print(f"  = policy present at {CHECKPOINT_DIR / 'model_state.pt'}")
        return
    conda = _conda()
    argv = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        ENV_NAME,
        "python",
        str(REPO / "scripts" / "external" / "synflownet_train.py"),
        "--log-dir",
        str(CHECKPOINT_DIR),
        "--steps",
        str(steps),
    ]
    if (CHECKPOINT_DIR / "model_state.pt").is_file():
        print("  · a checkpoint with no card: an interrupted run, resuming it")
        argv.append("--resume")
    else:
        argv.append("--overwrite")
    _run(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="print the steps and what each costs")
    parser.add_argument("--only", nargs="+", choices=STEPS, help="run just these steps")
    parser.add_argument("--train", action="store_true", help="also run the reward-training step")
    parser.add_argument("--steps", type=int, default=5000, help="training steps (upstream default)")
    args = parser.parse_args(argv)

    costs = {
        "clone": "~23 MB, pinned at 574f1e148f42",
        "env": f"conda env {ENV_NAME}: python {PYTHON_VERSION}, torch {TORCH_VERSION}+cu128",
        "data": f"{BLOCKS_SIZE:,} bytes from synflownet-boltz, then ~2.5 min of local RDKit",
        "masks": "~2.5 min, writes a 330 MB pickle",
        "train": "~4.7 GPU-hours: 5,000 steps x 64 trajectories against the sEH proxy",
    }
    if args.list:
        for name in STEPS:
            print(f"{name:8s} {costs[name]}")
        return 0

    selected = list(args.only) if args.only else [s for s in STEPS if s != "train" or args.train]
    for name in selected:
        print(f"[{name}] {costs[name]}")
        if name == "train":
            step_train(args.steps)
        else:
            {"clone": step_clone, "env": step_env, "data": step_data, "masks": step_masks}[name]()
    print("\nVerify: drugagent tools | grep molecule.generate.synflownet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
