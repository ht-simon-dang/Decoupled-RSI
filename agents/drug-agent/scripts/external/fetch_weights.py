#!/usr/bin/env python3
"""Fetch the generator/docking artifacts into the exact paths the adapters read.

This is the headless twin of ``notebooks/fetch_generator_weights.ipynb``. The
notebook documents *why* each artifact is where it is; this script is what runs
without a Jupyter kernel, which ``environment.yml`` does not ship.

Every download is verified on size and on leading bytes before it is moved into
place, because the failure mode here is not an error -- it is a 2 KB HTML page
saved under a ``.pt`` name, which only surfaces later as a confusing
``torch.load`` traceback. Downloads stage through ``.part`` and resume.

    python scripts/external/fetch_weights.py --list
    python scripts/external/fetch_weights.py                # the five cheap ones
    python scripts/external/fetch_weights.py --only gnina   # 2.0 GB, x86-64 only

Sizes were read from the servers on 2026-09-08. If one no longer matches, check
whether upstream republished before assuming a bad download; the error prints the
leading bytes so the two are distinguishable.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
EXTERNAL = Path(os.environ.get("DRUGAGENT_EXTERNAL_DIR", REPO / ".external"))

ELF = b"\x7fELF"
TORCH_ZIP = b"PK\x03\x04"
GZIP = b"\x1f\x8b"

UA = {"User-Agent": "drug-agent/0.1 (+fetch_weights.py)"}


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n / 1:,.0f} {unit}"
        n /= 1024
    return str(n)


def fetch(url: str, dest: Path, *, expect_size: int, magic: bytes | None = None) -> Path:
    """Download *url* to *dest*, verified on size and magic. Idempotent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expect_size:
        print(f"  = {dest.name} already present ({expect_size:,} bytes)")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > expect_size:
        part.unlink()
        have = 0
    print(f"  + {dest.name} <- {url[:78]}{'...' if len(url) > 78 else ''}")

    req = urllib.request.Request(url, headers=dict(UA))
    if have:
        print(f"    resuming at {have:,} bytes")
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req, timeout=120) as r, part.open("ab" if have else "wb") as fh:
        done = have
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if expect_size:
                pct = 100 * done / expect_size
                print(f"\r    {done:,} / {expect_size:,} bytes ({pct:5.1f}%)", end="", flush=True)
        print()

    size = part.stat().st_size
    head = part.open("rb").read(8)
    if size != expect_size:
        raise RuntimeError(
            f"{dest.name}: expected {expect_size:,} bytes, got {size:,}. First bytes: {head!r}"
        )
    if magic and not head.startswith(magic):
        raise RuntimeError(f"{dest.name}: expected magic {magic!r}, got {head!r} (an HTML page?)")
    part.replace(dest)
    print(f"    verified {size:,} bytes -> {dest.relative_to(REPO)}")
    return dest


# --- the artifacts -----------------------------------------------------------
# name -> (description, target path relative to EXTERNAL, callable)


def _pocket2mol() -> Path:
    # The Drive *folder* link serves HTML; the file id scraped from it downloads
    # unauthenticated, but only from drive.usercontent.google.com. The older
    # drive.google.com/uc host returns the virus-scan interstitial even with confirm=t.
    return fetch(
        "https://drive.usercontent.google.com/download"
        "?id=1WaoEj9RDG4VEcyHEmgsjbh958txm1W6x&export=download&confirm=t",
        EXTERNAL / "Pocket2Mol" / "ckpt" / "pretrained_Pocket2Mol.pt",
        expect_size=44_921_289,
        magic=TORCH_ZIP,
    )


def _targetdiff() -> Path:
    # Upstream's own Drive folder now 404s. This public MIT-licensed HuggingFace
    # mirror serves the same checkpoint, under weight/ rather than pretrained_models/.
    return fetch(
        "https://huggingface.co/OneScience-Group/TargetDiff/resolve/main/weight/pretrained_diffusion.pt",
        EXTERNAL / "targetdiff" / "pretrained_models" / "pretrained_diffusion.pt",
        expect_size=33_092_213,
        magic=TORCH_ZIP,
    )


def _pocketxmol() -> Path:
    pxm = EXTERNAL / "PocketXMol"
    ckpt = pxm / "data" / "trained_models" / "pxm" / "checkpoints" / "pocketxmol.ckpt"
    if ckpt.is_file():
        print(f"  = {ckpt.relative_to(EXTERNAL)} already unpacked")
        return ckpt
    tarball = fetch(
        "https://zenodo.org/records/17801271/files/model_weights.tar.gz",
        pxm / "model_weights.tar.gz",
        expect_size=640_300_872,
        magic=GZIP,
    )
    print("  unpacking...")
    with tarfile.open(tarball) as tf:
        # Refuse any member that would escape the checkout before extracting any.
        for member in tf.getmembers():
            target = (pxm / member.name).resolve()
            if not str(target).startswith(str(pxm.resolve())):
                raise RuntimeError(f"tarball member escapes the checkout: {member.name}")
        tf.extractall(pxm)
    tarball.unlink()  # 611 MB of duplicate once unpacked
    if not ckpt.is_file():
        found = sorted(pxm.rglob("*.ckpt"))
        print(f"  expected {ckpt.relative_to(REPO)} and it is not there.")
        for p in found[:10]:
            print(f"    unpacked instead: {p.relative_to(pxm)}")
    return ckpt


def _reinvent4() -> Path:
    # Zenodo publishes the unconditional prior as reinvent_pubchem.prior; REINVENT4's
    # own configs/sampling.toml reads priors/reinvent.prior. A rename on download.
    return fetch(
        "https://zenodo.org/records/20701824/files/reinvent_pubchem.prior",
        EXTERNAL / "REINVENT4" / "priors" / "reinvent.prior",
        expect_size=23_528_840,
        magic=TORCH_ZIP,
    )


def _gnina() -> Path:
    # The single release asset is an x86-64 ELF; the adapter finds it with
    # shutil.which("gnina"), so it lands in the active env's bin/ and is made
    # executable. Refuses on any other architecture rather than downloading 2 GB
    # that the kernel will not load.
    host = platform.machine()
    if host not in ("x86_64", "AMD64"):
        raise RuntimeError(
            f"gnina.cuda12.8.static is an x86-64 ELF and this host is {host}; "
            "it must be built from source here. See gnina_tool.py."
        )
    staged = EXTERNAL / "gnina-bin" / "gnina"
    fetch(
        "https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static",
        staged,
        expect_size=2_056_131_000,
        magic=ELF,
    )
    staged.chmod(0o755)
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        link = Path(prefix) / "bin" / "gnina"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(staged)
        print(f"    linked {link} -> {staged}")
    else:
        print(f"    CONDA_PREFIX unset; add to PATH yourself: {staged.parent}")
    return staged


ARTIFACTS = {
    "pocket2mol": ("Pocket2Mol checkpoint, 42.8 MB", _pocket2mol),
    "targetdiff": ("TargetDiff checkpoint, 31.6 MB", _targetdiff),
    "pocketxmol": ("PocketXMol weights tarball, 611 MB", _pocketxmol),
    "reinvent4": ("REINVENT4 unconditional prior, 23.5 MB", _reinvent4),
    "gnina": ("GNINA v1.3.3 static binary, 2.0 GB (x86-64 only)", _gnina),
}

DEFAULT = ["pocket2mol", "targetdiff", "pocketxmol", "reinvent4"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="fetch just these artifacts")
    ap.add_argument("--all", action="store_true", help="fetch everything, including gnina")
    ap.add_argument("--list", action="store_true", help="list the artifacts and exit")
    args = ap.parse_args()

    if args.list:
        for name, (desc, _) in ARTIFACTS.items():
            mark = "*" if name in DEFAULT else " "
            print(f" {mark} {name:12} {desc}")
        print("\n * = fetched by default (no argument). --all adds the rest.")
        return 0

    names = args.only if args.only else (list(ARTIFACTS) if args.all else DEFAULT)
    unknown = [n for n in names if n not in ARTIFACTS]
    if unknown:
        print(f"unknown artifact(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(f"repo:     {REPO}")
    print(f"external: {EXTERNAL}")
    if not EXTERNAL.is_dir():
        print("  (run `make external-clone` first)")

    failed = []
    for name in names:
        desc, fn = ARTIFACTS[name]
        print(f"\n--- {name}: {desc}")
        try:
            fn()
        except Exception as exc:  # report and continue to the next artifact
            print(f"  ! {name} FAILED: {exc}")
            failed.append(name)

    print("\n" + "=" * 72)
    ok = [n for n in names if n not in failed]
    print(f"fetched: {', '.join(ok) if ok else '(none)'}")
    if failed:
        print(f"failed:  {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
