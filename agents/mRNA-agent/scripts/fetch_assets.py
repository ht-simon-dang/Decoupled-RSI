"""Pinned, quota checked scientific assets; no upstream weights are redistributed."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import requests
from mrna_agent.storage import data_root, quota, used_bytes, MAX_BYTES

ROOT = data_root()


def fetch(url, path, expected_size=None, sha256=None):
    path = Path(path)
    if path.is_file() and (expected_size is None or path.stat().st_size == expected_size):
        if sha256 is None or hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == sha256:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        size = expected_size or int(response.headers.get("Content-Length", 0))
        if not size:
            size = 1_000_000_000
        with quota(size + 10_000_000):
            part = path.with_suffix(path.suffix + ".partial")
            written = 0
            try:
                with part.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        written += len(chunk)
                        if written > size:
                            raise ValueError("Response exceeds declared download budget")
                        handle.write(chunk)
                if expected_size and written != expected_size:
                    raise ValueError("Truncated download")
                digest = hashlib.file_digest(part.open("rb"), "sha256").hexdigest()
                if sha256 and digest != sha256:
                    raise ValueError("Asset checksum mismatch")
                part.replace(path)
            finally:
                part.unlink(missing_ok=True)
    print(json.dumps({"asset": str(path.relative_to(ROOT)), "bytes": written, "sha256": digest}), flush=True)
    return path


def lfs(repo, name, patterns):
    rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    for pattern in patterns:
        for path in repo.glob(pattern):
            if path.stat().st_size > 300:
                continue
            lines = path.read_text().splitlines()
            if not lines or not lines[0].startswith("version https://git-lfs"):
                continue
            digest = lines[1].split(":")[-1]
            size = int(lines[2].split()[1])
            fetch(f"https://media.githubusercontent.com/media/{name}/{rev}/{path.relative_to(repo)}", path, size, digest)


def main():
    external = ROOT / "external"
    lfs(external / "GEMORNA", "RainaBio/GEMORNA", ["checkpoints/*.pt", "vocab/*.pkl", "src/shared/*.so"])
    lfs(external / "parade", "autosome-ru/parade", ["predictor/*/saved_models/*.ckpt"])
    fetch("https://drive.usercontent.google.com/download?id=1HTAWKGajvAwRxq2rvIbxJ6i8yQZEQhKc&export=download&confirm=t",external / "Smart5UTR/models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5",401684224,"01a5f1546d4ce0c767fafce64f5795161c3c30db7d0ddd33e22dfef02425f3e6")
    archive = ROOT / "models/ribonn-weights.zip"
    if not archive.is_file():
        download_ribonn(archive)
    if hashlib.file_digest(archive.open("rb"),"md5").hexdigest() != "dd088ee2feffcda6ce26124a471a140f":
        raise ValueError("RiboNN archive differs from the official Zenodo record checksum")
    with zipfile.ZipFile(archive) as z:
        size = sum(f.file_size for f in z.infolist())
        with quota(size):
            for member in z.infolist():
                target = (ROOT / "models/ribonn" / member.filename).resolve()
                if not target.is_relative_to(ROOT / "models/ribonn"):
                    raise ValueError("Unsafe archive path")
            z.extractall(ROOT / "models/ribonn")
    previous=ROOT / "assets.json"
    manifest = json.loads(previous.read_text()) if previous.exists() else {}
    for repo in external.iterdir():
        if (repo / ".git").exists():
            manifest[repo.name] = {"revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()}
    paths = list((ROOT / "models").rglob("*.pth")) + list(external.rglob("*.pt")) + list(external.glob("parade/predictor/*/saved_models/*.ckpt"))
    for pattern in ["external/Smart5UTR/models/**/*.h5","external/Smart5UTR/models/**/*.scaler","external/GEMORNA/vocab/*.pkl","external/GEMORNA/src/shared/*.so","models/ribonn/*/runs.csv","external/LinearDesign/bin/*","external/LinearDesign/codon_usage_freq_table_human.csv"]:
        paths.extend(ROOT.glob(pattern))
    for path in paths:
        manifest[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    (ROOT / "assets.json").write_text(json.dumps(manifest, indent=2))
    print("Total bytes", used_bytes(), "limit", MAX_BYTES)


def download_ribonn(path):
    """Zenodo occasionally drops long responses. Resume bounded byte ranges."""
    size = 207395960
    part = path.with_suffix(".partial")
    url = "https://zenodo.org/records/17258709/files/weights.zip?download=1"
    with quota(size + 1_000_000):
        for attempt in range(12):
            start = part.stat().st_size if part.exists() else 0
            if start == size:
                part.replace(path)
                return
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-"}, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    append = response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Wrong response byte range")
                    written = start if append else 0
                    with part.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_content(1 << 20):
                            written += len(chunk)
                            if written > size: raise ValueError("Archive exceeds expected size")
                            handle.write(chunk)
            except requests.RequestException as exc:
                print("Retrying resumable Zenodo download", attempt, type(exc).__name__, flush=True)
        raise RuntimeError("RiboNN download retries exhausted; partial file retained")


if __name__ == "__main__":
    main()
