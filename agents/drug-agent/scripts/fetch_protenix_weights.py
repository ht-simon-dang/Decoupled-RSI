"""Fetch Protenix's checkpoint and data caches once, so inference never has to.

``structure.predict.protenix`` runs offline. Upstream does not: ``runner.inference.
download_inference_cache`` reaches ByteDance's object store on the first
prediction and pulls ~2.1 GB while a tool call is already in flight. The driver
therefore refuses to download, and this script is where the download happens
instead — deliberately, once, with a record of what arrived.

    export DRUGAGENT_ALLOW_NETWORK=1
    export DRUGAGENT_EGRESS_ALLOWLIST=protenix.tos-cn-beijing.volces.com
    python scripts/fetch_protenix_weights.py                     # the default model
    python scripts/fetch_protenix_weights.py --model protenix_mini_default_v0.5.0
    python scripts/fetch_protenix_weights.py --list

**The URLs are not written here.** They are read out of the installed package's own
``protenix.web_service.dependency_url.URL`` table, by asking the Protenix
interpreter for it. A URL copied into this repository would be a second source of
truth that upstream could silently outgrow; asking the installed version means the
files fetched are always the ones that version expects, and the recorded
provenance names the exact ``protenix`` release that chose them.

Everything goes through :class:`~drug_agent.security.egress.EgressPolicy`, which
denies by default, and lands under ``PROTENIX_ROOT_DIR`` (default ``~/.protenix``)
in the layout upstream's own loader reads:

    <root>/checkpoint/<model_name>.pt
    <root>/common/{components.cif, components.cif.rdkit_mol.pkl,
                   clusters-by-entity-40.txt, obsolete_release_date.csv}

A ``protenix_weights.json`` data card is written beside them: URL, byte count,
SHA-256, fetch timestamp and the Protenix version that named each artifact. It is
provenance, not configuration — nothing reads a model path out of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from drug_agent.data.snapshot import DataSnapshot  # noqa: E402
from drug_agent.security.egress import EgressPolicy, NetworkDeniedError  # noqa: E402
from drug_agent.util import utcnow_iso  # noqa: E402

#: Upstream's default inference model, and what the adapter probes for.
DEFAULT_MODEL = "protenix_base_default_v1.0.0"

#: The four caches ``runner.inference.download_inference_cache`` requires for any
#: prediction, in upstream's own order. Template and RNA-MSA caches are extra and
#: are only fetched with ``--templates``.
DATA_CACHES: tuple[str, ...] = (
    "ccd_components_file",
    "ccd_components_rdkit_mol_file",
    "pdb_cluster_file",
    "obsolete_release_data_csv",
)

TEMPLATE_CACHES: tuple[str, ...] = ("obsolete_pdbs_path", "release_dates_path")

#: A torch checkpoint is a zip archive. Checking the magic separates "downloaded"
#: from "saved an HTML error page under a .pt name", which otherwise only surfaces
#: as a confusing traceback inside torch.load minutes into a prediction.
TORCH_ZIP_MAGIC = b"PK\x03\x04"

CARD_NAME = "protenix_weights.json"
CHUNK = 1 << 20
TIMEOUT_S = 300.0
USER_AGENT = "drug-agent/1.0 (structure.predict.protenix weight setup)"


def protenix_python(explicit: str | None) -> str:
    """The interpreter Protenix is installed into."""
    if explicit:
        return explicit
    env = os.environ.get("DRUGAGENT_PROTENIX_PYTHON")
    if env:
        return env
    conda = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX") or ""
    roots = [
        Path(conda).parent / "envs",
        Path.home() / "anaconda3" / "envs",
        Path.home() / "miniconda3" / "envs",
    ]
    for root in roots:
        candidate = root / "drug-agent-protenix" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def upstream_urls(python: str) -> tuple[dict[str, str], str]:
    """Ask the installed Protenix for its own download table and version."""
    code = (
        "import json;"
        "from protenix.web_service.dependency_url import URL;"
        "from protenix.version import __version__;"
        "print(json.dumps({'urls': URL, 'version': __version__}))"
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or exc
        raise SystemExit(
            f"cannot read the Protenix URL table from {python}: {detail}\n"
            "Install it first with `make protenix-install`."
        ) from exc
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    return dict(payload["urls"]), str(payload["version"])


def head_size(url: str) -> int | None:
    """Content-Length, so a truncated download is caught rather than loaded."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = response.headers.get("Content-Length")
    except (urllib.error.URLError, OSError):
        return None
    return int(raw) if raw and raw.isdigit() else None


def download(url: str, destination: Path, *, magic: bytes | None) -> dict[str, Any]:
    """Fetch one artifact, staged through ``.part`` and verified before it lands."""
    expected = head_size(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, part.open("wb") as handle:
        status = int(response.status)
        first = response.read(CHUNK)
        if magic and not first.startswith(magic):
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"{url} did not return a torch checkpoint (first bytes {first[:16]!r}); "
                "the server most likely answered with an error page"
            )
        handle.write(first)
        digest.update(first)
        while chunk := response.read(CHUNK):
            handle.write(chunk)
            digest.update(chunk)
    n_bytes = part.stat().st_size
    if expected is not None and n_bytes != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{url} truncated: got {n_bytes} bytes, Content-Length said {expected}")
    shutil.move(str(part), str(destination))
    return {
        "url": url,
        "file": str(destination),
        "http_status": status,
        "n_bytes": n_bytes,
        "sha256": digest.hexdigest(),
        "fetched_at": utcnow_iso(),
        "origin": "downloaded",
    }


def record_existing(url: str, destination: Path) -> dict[str, Any]:
    """Hash a file that is already in place, without claiming to have fetched it."""
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return {
        "url": url,
        "file": str(destination),
        "http_status": None,
        "n_bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
        "fetched_at": None,
        "origin": "already_present",
    }


def plan(
    urls: dict[str, str], root: Path, models: list[str], *, templates: bool
) -> list[tuple[str, str, Path]]:
    """``(artifact name, url, destination)`` for everything this run should ensure."""
    wanted = list(DATA_CACHES) + (list(TEMPLATE_CACHES) if templates else [])
    items: list[tuple[str, str, Path]] = []
    for name in wanted:
        url = urls.get(name)
        if not url:
            raise SystemExit(f"the installed Protenix has no URL for {name!r}")
        items.append((name, url, root / "common" / Path(url).name))
    for model in models:
        url = urls.get(model)
        if not url:
            known = ", ".join(sorted(k for k in urls if k.startswith("protenix")))
            raise SystemExit(f"unknown Protenix model {model!r}; the installed version offers {known}")
        items.append((model, url, root / "checkpoint" / f"{model}.pt"))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="PROTENIX_ROOT_DIR (default ~/.protenix)")
    parser.add_argument("--model", action="append", default=[], help=f"model name (default {DEFAULT_MODEL})")
    parser.add_argument("--templates", action="store_true", help="also fetch the template-search caches")
    parser.add_argument("--protenix-python", default=None, help="interpreter Protenix is installed into")
    parser.add_argument(
        "--list", action="store_true", help="print the installed version's URL table and exit"
    )
    parser.add_argument("--force", action="store_true", help="re-download artifacts already present")
    args = parser.parse_args(argv)

    python = protenix_python(args.protenix_python)
    urls, version = upstream_urls(python)
    if args.list:
        print(f"protenix {version} ({python})")
        for name in sorted(urls):
            print(f"  {name:44s} {urls[name]}")
        return 0

    root = (args.root or Path(os.environ.get("PROTENIX_ROOT_DIR", "~/.protenix"))).expanduser()
    models = args.model or [DEFAULT_MODEL]
    policy = EgressPolicy.from_env()
    records: list[dict[str, Any]] = []
    failures = 0

    print(f"protenix {version} -> {root}")
    for name, url, destination in plan(urls, root, models, templates=args.templates):
        if destination.is_file() and not args.force:
            record = record_existing(url, destination)
            print(f"  = {name}: already present ({record['n_bytes']:,} bytes)")
        else:
            try:
                policy.check(url)
            except NetworkDeniedError as exc:
                print(exc, file=sys.stderr)
                return 2
            print(f"  + {name}: {url}")
            try:
                record = download(url, destination, magic=TORCH_ZIP_MAGIC if name in models else None)
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                print(f"    failed: {exc}", file=sys.stderr)
                failures += 1
                continue
            print(f"    {record['n_bytes']:,} bytes -> {destination}")
        record["artifact"] = name
        records.append(record)

    snapshot = DataSnapshot(
        name="protenix_inference_artifacts",
        source="protenix.web_service.dependency_url.URL",
        kind="model_weights",
        records=records,
        license="Protenix is Apache-2.0; the checkpoints carry ByteDance's model terms",
        metadata={
            "protenix_version": version,
            "protenix_python": python,
            "root": str(root),
            "models": models,
            "templates": args.templates,
            "note": "provenance only; the adapter probes the filesystem, never this card",
        },
    )
    card = root / CARD_NAME
    card.write_text(json.dumps({**snapshot.data_card(), "records": records}, indent=2) + "\n")
    print(f"card: {card} ({len(records)} artifacts)")
    print("Verify: drugagent tools | grep structure.predict.protenix")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
