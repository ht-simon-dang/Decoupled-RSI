#!/usr/bin/env python3
"""Fetch the pocket-detection backends once, so detection never needs the network.

This is the *setup* half of the rule the whole pockets package is built on: setup
may use the network, the running agent may not. Everything below downloads,
unpacks or installs a real upstream artifact and then writes a provenance record
beside it; after that, ``drug_agent.tools.pockets`` reads local files only and the
platform can run with egress denied.

    python scripts/external/fetch_pockets.py --list
    python scripts/external/fetch_pockets.py                  # all three
    python scripts/external/fetch_pockets.py --only fpocket   # ~30 s, no download
    python scripts/external/fetch_pockets.py --only p2rank    # 276 MB

What each one gets, and what it is worth:

``fpocket``   conda-forge 4.2.3 into its own environment, ``drug-agent-fpocket``.
              A 1.8 MB C binary, no weights, no GPU; this is the one that makes
              ``pocket.detect.fpocket`` genuinely AVAILABLE. Its own environment
              because the nine verified environments are not to be re-solved for it.
``p2rank``    the 2.5.1 release archive, which carries the JAR *and* the trained
              forests under ``models/``. Needs a JRE (Java 17-24) that this script
              does not install — it reports whether one is present.
``vnegnn``    the MIT source at a pinned commit. Cloning it does **not** make the
              capability available: upstream publishes no checkpoint, so
              ``pocket.detect.vnegnn`` stays UNAVAILABLE and says why. The clone is
              fetched anyway so that the refusal can name files that exist.

Every download is verified on size and on leading bytes before it is unpacked,
because the failure that costs the most time is not an error — it is an HTML error
page saved under a ``.tar.gz`` name, which only surfaces much later.

**The provenance record.** Each artifact directory gets a
``drugagent_provenance.json`` naming the source URL, the release or commit, a
SHA-256 per file recorded, and the fetch timestamp.
:meth:`drug_agent.tools.pockets.PocketToolAdapter.artifact_snapshot` reads it back
into a :class:`~drug_agent.data.snapshot.DataSnapshot`, whose ``snapshot_id`` a run
can cite. A cached copy of genuinely-fetched upstream bytes is real data; what makes
it real rather than merely present is that this file says where it came from.

Standard library only, and it imports nothing from ``drug_agent`` — same rule as
every other script in this directory, so it runs before the package is installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
EXTERNAL = Path(os.environ.get("DRUGAGENT_EXTERNAL_DIR", REPO / ".external"))

#: Written by every fetcher; read by ``PocketToolAdapter.artifact_snapshot``.
PROVENANCE_FILENAME = "drugagent_provenance.json"

GZIP_MAGIC = b"\x1f\x8b"
UA = {"User-Agent": "drug-agent/0.1 (+fetch_pockets.py)"}

# --- fpocket ---------------------------------------------------------------
FPOCKET_ENV = "drug-agent-fpocket"
FPOCKET_CHANNEL = "conda-forge"
FPOCKET_VERSION = "4.2.3"

# --- P2Rank ----------------------------------------------------------------
P2RANK_RELEASE = "2.5.1"
P2RANK_URL = (
    f"https://github.com/rdk/p2rank/releases/download/{P2RANK_RELEASE}/p2rank_{P2RANK_RELEASE}.tar.gz"
)
P2RANK_BYTES = 275_625_956
P2RANK_DIR = EXTERNAL / "p2rank"
#: Recorded in the provenance file with a digest each. The launcher, the JAR, and
#: the two trained forests the adapter's two modes load.
P2RANK_RECORDED = (
    "prank",
    "bin/p2rank.jar",
    "models/default/model.zst",
    "models/default_rescore/model.zst",
    "config/default.groovy",
    "config/default_rescore.groovy",
)

# --- VN-EGNN ---------------------------------------------------------------
VNEGNN_URL = "https://github.com/ml-jku/vnegnn"
VNEGNN_REF = "1d1f922cb9f015d4793e7804cbc9b8b248c68498"
VNEGNN_DIR = EXTERNAL / "vnegnn"
VNEGNN_RECORDED = (
    "src/models/vnegnn/vnegnn.py",
    "configs/model/vnegnn.yaml",
    "configs/eval.yaml",
    "src/eval.py",
    "src/datasets/binding_dataset.py",
)


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _run(argv: list[str], *, what: str) -> None:
    print(f"  $ {' '.join(argv)}")
    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"{what} failed with exit {proc.returncode}")


def _conda() -> str:
    """The conda executable, from the environment that launched this script."""
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not exe:
        raise SystemExit("no conda executable found: set CONDA_EXE or put conda on PATH")
    return exe


def write_provenance(
    directory: Path,
    *,
    name: str,
    source: str,
    kind: str,
    license_text: str,
    version: str,
    relpaths: tuple[str, ...],
    extra: dict[str, object] | None = None,
) -> Path:
    """Record where these bytes came from, beside the bytes.

    The digests are per file rather than per directory: a release archive that
    unpacked halfway leaves a directory that exists, and a record naming four files
    that are individually hashed is the difference between "this is the 2.5.1
    release" and "something is here".
    """
    files = [
        {"path": rel, "bytes": (directory / rel).stat().st_size, "digest": _sha256(directory / rel)}
        for rel in relpaths
        if (directory / rel).is_file()
    ]
    payload: dict[str, object] = {
        "name": name,
        "source": source,
        "kind": kind,
        "license": license_text,
        "version": version,
        "fetched_at": _utcnow(),
        "fetched_by": "scripts/external/fetch_pockets.py",
        "files": files,
    }
    payload.update(extra or {})
    record = directory / PROVENANCE_FILENAME
    record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  = provenance written to {record.relative_to(REPO) if record.is_relative_to(REPO) else record}")
    return record


def download(url: str, dest: Path, *, expect_size: int, magic: bytes) -> Path:
    """Download *url* to *dest*, verified on size and leading bytes. Idempotent.

    Staged through ``.part`` and resumed with a ``Range`` request, because 276 MB
    over a flaky link should not restart from zero, and a half-written file under
    the final name would pass an ``is_file()`` check forever.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expect_size:
        print(f"  = {dest.name} already present ({expect_size:,} bytes)")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > expect_size:
        part.unlink()
        have = 0
    print(f"  + {dest.name} <- {url}")

    req = urllib.request.Request(url, headers=dict(UA))
    if have:
        print(f"    resuming at {have:,} bytes")
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req, timeout=120) as response, part.open("ab" if have else "wb") as fh:
        done = have
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            print(f"\r    {done:,} / {expect_size:,} bytes", end="", file=sys.stderr)
    print(file=sys.stderr)

    size = part.stat().st_size
    if size != expect_size:
        raise SystemExit(f"{dest.name}: expected {expect_size:,} bytes, got {size:,}")
    with part.open("rb") as fh:
        head = fh.read(len(magic))
    if head != magic:
        raise SystemExit(
            f"{dest.name}: leading bytes are {head!r}, expected {magic!r} — "
            f"the server probably returned an error page rather than the archive"
        )
    part.replace(dest)
    return dest


# --- the three backends ----------------------------------------------------
def fetch_fpocket() -> None:
    """Create ``drug-agent-fpocket`` with conda-forge's fpocket, and prove it runs.

    A separate environment is not fastidiousness: the nine verified environments on
    this host are a measured configuration, and adding a C binary to one of them
    means re-solving it. The adapter finds this environment's ``bin/fpocket``
    without it ever being on the platform's PATH.
    """
    print(f"fpocket -> conda env {FPOCKET_ENV} ({FPOCKET_CHANNEL}, {FPOCKET_VERSION} at time of writing)")
    conda = _conda()
    _run([conda, "create", "-y", "-n", FPOCKET_ENV, "-c", FPOCKET_CHANNEL, "fpocket"], what="conda create")
    binary = Path(conda).parent.parent / "envs" / FPOCKET_ENV / "bin" / "fpocket"
    if not binary.is_file():
        raise SystemExit(f"conda reported success but {binary} is not there")
    # fpocket has no --version; with no input it prints its banner and exits non-zero.
    proc = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    banner = next((ln for ln in (proc.stdout + proc.stderr).splitlines() if "fpocket" in ln), "")
    print(f"  = {binary}")
    print(f"  = banner: {banner.strip() or '(none printed)'}")


def fetch_p2rank() -> None:
    """Download and unpack the P2Rank release, then record what it is."""
    print(f"P2Rank {P2RANK_RELEASE} -> {P2RANK_DIR}")
    archive = EXTERNAL / f"p2rank_{P2RANK_RELEASE}.tar.gz"
    download(P2RANK_URL, archive, expect_size=P2RANK_BYTES, magic=GZIP_MAGIC)
    digest = _sha256(archive)

    staging = EXTERNAL / f".p2rank_unpack_{P2RANK_RELEASE}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    print(f"  + unpacking {archive.name}")
    with tarfile.open(archive) as tar:
        # `filter="data"` refuses absolute paths, ".." members, symlinks out of the
        # tree and device nodes. The archive is from a release page over TLS, but an
        # extractor that trusts member paths is a hazard regardless of the source.
        tar.extractall(staging, filter="data")
    unpacked = staging / f"p2rank_{P2RANK_RELEASE}"
    if not (unpacked / "prank").is_file():
        raise SystemExit(f"archive did not contain {unpacked.name}/prank — did the layout change?")

    shutil.rmtree(P2RANK_DIR, ignore_errors=True)
    unpacked.replace(P2RANK_DIR)
    shutil.rmtree(staging, ignore_errors=True)
    (P2RANK_DIR / "prank").chmod(0o755)

    write_provenance(
        P2RANK_DIR,
        name=f"p2rank-{P2RANK_RELEASE}",
        source=P2RANK_URL,
        kind="model_release",
        license_text="P2Rank MIT (Krivak & Hoksza, J. Cheminformatics 2018, 10:39)",
        version=P2RANK_RELEASE,
        relpaths=P2RANK_RECORDED,
        extra={
            "archive_bytes": P2RANK_BYTES,
            "archive_digest": digest,
            "models": ["default (predict)", "default_rescore (PRANK rescoring)"],
        },
    )
    java = shutil.which("java")
    if java:
        print(f"  = java found at {java}; P2Rank 2.5.1 supports Java 17-24")
    else:
        print("  ! no java on PATH — P2Rank will report UNAVAILABLE until a JRE is installed:")
        print("      sudo apt-get install -y openjdk-21-jre-headless")


def fetch_vnegnn() -> None:
    """Clone the VN-EGNN source. This does **not** make the capability available.

    Said plainly here because it is the only fetcher in this file whose success
    leaves a tool still refusing: upstream publishes no checkpoint, and the
    adapter's ``unavailable_reason`` names exactly that. The clone is worth doing
    because the refusal then cites files an operator can open.
    """
    print(f"VN-EGNN -> {VNEGNN_DIR} (source only; no checkpoint is published)")
    if not (VNEGNN_DIR / ".git").is_dir():
        VNEGNN_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", VNEGNN_URL, str(VNEGNN_DIR)], what="git clone")
    _run(["git", "-C", str(VNEGNN_DIR), "checkout", "--quiet", VNEGNN_REF], what="git checkout")

    weights = sorted(
        str(p.relative_to(VNEGNN_DIR))
        for p in VNEGNN_DIR.rglob("*")
        if p.suffix in (".ckpt", ".pt", ".pth", ".safetensors") and ".git" not in p.parts
    )
    write_provenance(
        VNEGNN_DIR,
        name="vnegnn-source",
        source=VNEGNN_URL,
        kind="source_checkout",
        license_text="VN-EGNN MIT (Sestak et al., arXiv:2404.07194)",
        version=VNEGNN_REF,
        relpaths=VNEGNN_RECORDED,
        extra={
            "commit": VNEGNN_REF,
            "checkpoints_found": weights,
            "note": (
                "No checkpoint is published. configs/eval.yaml pins wandb_run_id "
                "pt8wsnz9 and src/eval.py resolves the checkpoint from that private "
                "Weights & Biases run, so pocket.detect.vnegnn stays UNAVAILABLE."
            ),
        },
    )
    print(f"  = checkpoints found in the checkout: {weights or 'none — as expected'}")


FETCHERS = {
    "fpocket": fetch_fpocket,
    "p2rank": fetch_p2rank,
    "vnegnn": fetch_vnegnn,
}

DESCRIPTIONS = {
    "fpocket": f"conda-forge fpocket {FPOCKET_VERSION} into the {FPOCKET_ENV} env (no download)",
    "p2rank": f"P2Rank {P2RANK_RELEASE} release archive, {P2RANK_BYTES:,} bytes",
    "vnegnn": f"VN-EGNN source at {VNEGNN_REF[:12]} (~11 MB; no checkpoint exists)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FETCHERS), action="append", help="fetch just this one")
    ap.add_argument("--list", action="store_true", help="show what each target fetches and exit")
    args = ap.parse_args()

    if args.list:
        for name in sorted(FETCHERS):
            print(f"{name:10s} {DESCRIPTIONS[name]}")
        return 0

    targets = args.only or sorted(FETCHERS)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for name in targets:
        FETCHERS[name]()
        print()
    print("Verify with: drugagent tools | grep pocket.detect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
