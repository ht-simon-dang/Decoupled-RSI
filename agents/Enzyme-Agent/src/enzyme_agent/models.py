"""Pinned model inventory and quota-aware explicit downloads; inference is offline."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import requests

from .config import atomic_json, check_space, data_root, quota_lock, usage, MAX_BYTES
from .contracts import Unavailable

MODELS = {
    "qwen": {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "role": "planner", "license": "Apache-2.0"},
    "esm3": {"repo": "biohub/esm3-sm-open-v1", "revision": "47f0545b2b6daf26a93439a3cd610f4f7f3d5478", "role": "generator", "license": "MIT"},
    "esmfold2": {"repo": "biohub/ESMFold2-Fast", "revision": "45fe8656f5b3ef493c17fcf9abe9a2968902e712", "role": "independent_fold", "license": "MIT"},
    "zymctrl": {"repo": "AI4PD/ZymCTRL", "revision": "3c532ef172b9cd2e95238baadf5167ebb89fbc32", "role": "generator", "license": "Apache-2.0"},
}
MPNN_REVISION = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
EXTERNAL = {
    "rfdiffusion2": ("generator", "https://github.com/RosettaCommons/RFdiffusion2", "Requires motif atom constraints and separately validated runtime"),
    "placer": ("local_conformation", "https://github.com/baker-laboratory/PLACER", "Requires a protein/ligand complex; local conformations are not rates"),
    "catpred": ("kinetics", "https://github.com/maranasgroup/CatPred", "Requires small-molecule compatibility and calibrated domain"),
    "catapro": ("kinetics", "https://github.com/zchwang/CataPro", "Requires small-molecule compatibility and calibrated domain"),
    "ezspecificity": ("substrate_compatibility", "https://github.com/antecede/EZSpecificity", "Requires structures and license/input validation"),
    "netsolp": ("solubility", "https://services.healthtech.dtu.dk/services/NetSolP-1.0/", "E. coli domain only"),
    "netmhciipan": ("immunogenicity_clue", "https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/", "Requires licensed installation and explicit HLA panel"),
    "progen3": ("generator", "https://github.com/Profluent-AI/progen3", "Separate noncommercial weight terms"),
    "genzyme": ("research_generator", "https://github.com/WillHua127/GENzyme", "Research hypothesis; separate noncommercial terms"),
}


def model_path(name, require=True):
    if name == "qwen":
        override = os.environ.get("ENZYME_QWEN_DIR")
        shared_root = Path(__file__).resolve().parents[3] / "peptide-agent/.peptide-agent/models"
        shared_root = Path(os.environ.get("PEPTIDE_MODEL_DIR", str(shared_root))).expanduser()
        shared = shared_root / "qwen" / MODELS[name]["revision"]
        # src/enzyme_agent -> Enzyme-Agent -> agents (shared sibling packages)
        path = Path(override) if override else shared
        if not override and not path.exists():
            path = data_root() / "models" / name / MODELS[name]["revision"]
    elif name in MODELS:
        path = data_root() / "models" / name / MODELS[name]["revision"]
    else:
        path = data_root() / "models" / name
    if require and not path.exists():
        raise Unavailable(f"Missing {name}; run enzyme-agent download {name}")
    return path


def download_file(url, destination, expected_size=None, sha256=None):
    destination = Path(destination)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        if not sha256 or file_sha(destination) == sha256:
            return file_sha(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with quota_lock():
        temporary.unlink(missing_ok=True)
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            size = expected_size or int(response.headers.get("content-length", 0))
            check_space(size)
            count, hasher = 0, hashlib.sha256()
            try:
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        check_space(len(chunk))
                        count += len(chunk)
                        if expected_size and count > expected_size:
                            raise ValueError("Download exceeds declared size")
                        handle.write(chunk)
                        hasher.update(chunk)
                if expected_size and count != expected_size:
                    raise ValueError("Incomplete model download")
                if sha256 and hasher.hexdigest() != sha256:
                    raise ValueError("Model SHA-256 mismatch")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    return hasher.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name):
    if name in {'rfdiffusion3','thermompnn'}:
        from .extended_models import download_extended
        return download_extended(name)
    if name in {"ligandmpnn", "proteinmpnn"}:
        return download_mpnn()
    if name not in MODELS:
        raise ValueError("No automatic downloader for this model")
    info = MODELS[name]
    path = model_path(name, require=False)
    if name == "qwen" and path.is_dir():
        return {**info, "path": str(path), "shared": True}
    response = requests.get(f"https://huggingface.co/api/models/{info['repo']}/revision/{info['revision']}", params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    siblings = [f for f in response.json()["siblings"] if not f["rfilename"].endswith((".png", ".py", ".md")) and not f["rfilename"].startswith(".")]
    check_space(sum(f.get("size", 0) for f in siblings if not (path / f["rfilename"]).exists()))
    manifest = {**info, "files": {}}
    for f in siblings:
        filename = f["rfilename"]
        print(f"Downloading {name}/{filename}", flush=True)
        sha = f.get("lfs", {}).get("sha256")
        checksum = download_file(f"https://huggingface.co/{info['repo']}/resolve/{info['revision']}/{filename}", path / filename, f.get("size"), sha)
        manifest["files"][filename] = {"sha256": checksum, "size": (path / filename).stat().st_size}
    atomic_json(path / "manifest.json", manifest)
    return manifest


def download_mpnn():
    # Resolve once; subsequent installations keep the manifest's immutable commit.
    base = data_root() / "models/ligandmpnn"
    manifest_file = base / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    revision = manifest.get("revision")
    if not revision:
        revision = MPNN_REVISION
    manifest.update(repo="dauparas/LigandMPNN", revision=revision, license="MIT", files={})
    for name in ("run.py", "model_utils.py", "data_utils.py", "sc_utils.py", "LICENSE"):
        sha = download_file(f"https://raw.githubusercontent.com/dauparas/LigandMPNN/{revision}/{name}", base / "source" / name)
        manifest["files"][name] = sha
    # Side-chain packing imports OpenFold helpers even with packing disabled.
    r = requests.get(f"https://api.github.com/repos/dauparas/LigandMPNN/git/trees/{revision}?recursive=1", timeout=60)
    r.raise_for_status()
    for item in r.json()["tree"]:
        if item["path"].startswith("openfold/") and item["type"] == "blob" and item["path"].endswith(".py"):
            name = item["path"]
            manifest["files"][name] = download_file(f"https://raw.githubusercontent.com/dauparas/LigandMPNN/{revision}/{name}", base / "source" / name, item.get("size"))
    for name in ("ligandmpnn_v_32_010_25.pt", "proteinmpnn_v_48_020.pt"):
        manifest["files"][name] = download_file("https://files.ipd.uw.edu/pub/ligandmpnn/" + name, base / name)
    atomic_json(manifest_file, manifest)
    return manifest


def capabilities():
    models = []
    for name, info in MODELS.items():
        p = model_path(name, require=False)
        ready = False
        if (p/'manifest.json').is_file():
            manifest=json.loads((p/'manifest.json').read_text())
            ready=all((p/f).is_file() and (p/f).stat().st_size==info['size'] for f,info in manifest.get('files',{}).items())
        elif name=='qwen' and (p/'model.safetensors.index.json').is_file():
            index=json.loads((p/'model.safetensors.index.json').read_text())
            ready=(p/'config.json').is_file() and all((p/f).is_file() for f in set(index['weight_map'].values()))
        models.append({"name": name, **info, "status": "ready" if ready else "needs_download", "path": str(p)})
    for name in ("ligandmpnn", "proteinmpnn"):
        p = data_root() / "models/ligandmpnn/manifest.json"
        info = json.loads(p.read_text()) if p.exists() else {}
        filename='ligandmpnn_v_32_010_25.pt' if name=='ligandmpnn' else 'proteinmpnn_v_48_020.pt'
        ready=p.exists() and (p.parent/filename).is_file() and (p.parent/'source/run.py').is_file() and importlib.util.find_spec('prody')
        models.append({"name": name, "role": "generator", "status": "ready" if ready else "needs_install", "revision": info.get("revision"), "license": "MIT"})
    for name,role in [('rfdiffusion3','constrained_all_atom_generator'),('thermompnn','point_mutation_stability')]:
        p=data_root()/'models'/name
        manifest=json.loads((p/'manifest.json').read_text()) if (p/'manifest.json').exists() else {}
        validation=json.loads((p/'runtime-validation.json').read_text()) if (p/'runtime-validation.json').exists() else {}
        checkpoint_ok=(p/'rfd3_latest.ckpt').is_file() if name=='rfdiffusion3' else (p/'source/models/thermoMPNN_default.pt').is_file()
        runtime_ok=(data_root()/'foundry-env/bin/rfd3').is_file() if name=='rfdiffusion3' else importlib.util.find_spec('omegaconf') is not None
        ready=bool(checkpoint_ok and runtime_ok and manifest and validation.get('status')=='passed' and validation.get('revision')==manifest.get('revision'))
        models.append({'name':name,'role':role,'status':'ready' if ready else 'needs_validation' if manifest else 'needs_install',
            'revision':manifest.get('revision'),'license':manifest.get('license'),'validation':validation,
            'reason':'Constrained reference-structure redesign; new catalytic mechanisms require evidenced atom constraints' if name=='rfdiffusion3' else 'Point mutations in reference background; no multi-mutant additivity'})
    for name, (role, source, reason) in EXTERNAL.items():
        models.append({"name": name, "role": role, "status": "unavailable", "source": source, "reason": reason})
    return {"models": models, "storage": {"used_bytes": usage(), "limit_bytes": MAX_BYTES}, "scientific_scope": "Computational prioritization; experimental validation is separate"}


if __name__ == "__main__":
    import sys
    for model in sys.argv[1:]:
        download(model)
