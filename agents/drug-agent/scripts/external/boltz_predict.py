"""Predict one protein-ligand complex with Boltz-2 — the drug-agent ⇄ boltz bridge.

This runs inside the ``drug-agent-boltz`` environment (Python 3.11.16, boltz 2.2.1,
torch 2.14.0+cu130), not the platform's, so it imports nothing from ``drug_agent``.
The two sides talk over a JSON request/response pair on disk.

**It calls upstream's own entry point.** ``boltz.main.predict`` is a
``click.Command``; ``predict.callback`` is the undecorated function behind it, with
the same defaults click would apply. Reimplementing the pipeline — MSA handling,
CCD loading, the manifest, the two Lightning ``Trainer.predict`` passes — would be
a second implementation to keep in step with the first, so this driver only does
what the CLI cannot: build the YAML, choose the arguments, and read the results
back as structures rather than as files on disk.

**It refuses to download.** ``predict`` calls ``download_boltz2(cache)`` before it
does anything else, which fetches 6.2 GB from ``model-gateway.boltz.bio`` when the
cache is cold. A tool call is not the place to discover that, so the cache is
checked first and a cold cache is a structured failure naming the missing files.
``allow_download`` exists for an operator running this driver by hand with network
access; the adapter always sends it false.

Request JSON (``argv[1]``):
    name                    job name; becomes the YAML stem and boltz's record id
    sequences[]             {kind: protein|dna|rna|ligand, id, sequence|smiles|ccd,
                            msa, cyclic, modifications}
    constraints[]           passed verbatim into the YAML (pocket / contact / bond)
    affinity_binder         chain id of the small molecule to score, or null
    cache                   absolute Boltz cache directory
    results_dir             where boltz writes; defaults under out_dir (temporary)
    allow_download          permit the 6.2 GB first-run fetch (default false)
    diffusion_samples, recycling_steps, sampling_steps, step_scale, seed,
    output_format, accelerator, devices, num_workers, use_msa_server,
    msa_server_url, use_potentials, no_kernels, method, override,
    affinity_mw_correction, sampling_steps_affinity, diffusion_samples_affinity,
    max_parallel_samples, subsample_msa, num_subsampled_msa, preprocessing_threads
    out_dir                 injected by the caller; result.json goes here

Response JSON (``<out_dir>/result.json``):
    samples[]               {model_index, format, path, coordinates, confidence{}}
    affinity                the affinity_*.json object, or null
    affinity_requested      whether a binder chain was declared at all
    engine_version          boltz.__version__ as installed
    results_dir, yaml_path, cache, no_kernels, warnings[]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _driver

#: What ``boltz.main.download_boltz2`` puts in the cache. ``mols`` is the directory
#: ``mols.tar`` is extracted into; upstream tests it with ``.exists()``, so a
#: half-extracted cache reads as present here exactly as it does there.
STRUCTURE_CACHE_ENTRIES = ("boltz2_conf.ckpt", "mols")
AFFINITY_CACHE_ENTRY = "boltz2_aff.ckpt"

#: Polymer kinds that carry ``sequence``; everything else must be a ligand.
POLYMER_KINDS = ("protein", "dna", "rna")


def missing_cache_entries(cache: Path, *, need_affinity: bool) -> list[str]:
    """Which Boltz-2 cache entries are absent, in the order upstream fetches them."""
    names = list(STRUCTURE_CACHE_ENTRIES)
    if need_affinity:
        names.append(AFFINITY_CACHE_ENTRY)
    return [name for name in names if not (cache / name).exists()]


def build_yaml_document(request: dict[str, Any]) -> dict[str, Any]:
    """Assemble the Boltz YAML schema documented in ``docs/prediction.md``."""
    sequences: list[dict[str, Any]] = []
    for entry in request.get("sequences", []):
        kind = entry.get("kind")
        body: dict[str, Any] = {"id": entry["id"]}
        if kind in POLYMER_KINDS:
            body["sequence"] = entry["sequence"]
            if entry.get("msa"):
                body["msa"] = entry["msa"]
            if entry.get("cyclic"):
                body["cyclic"] = True
            if entry.get("modifications"):
                body["modifications"] = entry["modifications"]
        elif kind == "ligand":
            if entry.get("smiles"):
                body["smiles"] = entry["smiles"]
            elif entry.get("ccd"):
                body["ccd"] = entry["ccd"]
            else:
                msg = f"ligand chain {entry.get('id')!r} has neither smiles nor ccd"
                raise ValueError(msg)
        else:
            msg = f"unsupported entity kind {kind!r}; expected one of protein, dna, rna, ligand"
            raise ValueError(msg)
        sequences.append({kind: body})

    if not sequences:
        msg = "request carries no sequences"
        raise ValueError(msg)

    document: dict[str, Any] = {"version": 1, "sequences": sequences}
    if request.get("constraints"):
        document["constraints"] = request["constraints"]
    if request.get("affinity_binder"):
        document["properties"] = [{"affinity": {"binder": request["affinity_binder"]}}]
    return document


def read_json(path: Path) -> dict[str, Any]:
    """Parse one of boltz's small JSON side-files."""
    import json

    with path.open() as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def collect_samples(pred_dir: Path, name: str, suffix: str) -> list[dict[str, Any]]:
    """Read every ``<name>_model_<i>.<suffix>`` with its confidence side-file.

    The coordinates are carried back as text rather than as a path because the
    adapter's scratch directory is removed as soon as the driver exits — a path
    would name a file that no longer exists by the time a Pose is built.
    """
    samples: list[dict[str, Any]] = []
    for path in pred_dir.glob(f"{name}_model_*{suffix}"):
        stem = path.stem
        index_text = stem.rsplit("_model_", 1)[-1]
        if not index_text.isdigit():
            continue
        index = int(index_text)
        confidence_path = pred_dir / f"confidence_{name}_model_{index}.json"
        samples.append(
            {
                "model_index": index,
                "format": suffix.lstrip("."),
                "path": str(path),
                "coordinates": path.read_text(),
                "confidence": read_json(confidence_path) if confidence_path.is_file() else {},
            }
        )
    samples.sort(key=lambda sample: sample["model_index"])
    return samples


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run one Boltz-2 prediction and return its structures, confidence and affinity."""
    # pyyaml is a declared boltz dependency and boltz.data.parse.yaml imports it,
    # so it is present wherever boltz itself can run.
    import boltz
    import yaml
    from boltz.main import predict

    out_dir = Path(request["out_dir"])
    name = str(request.get("name") or "complex")
    cache = Path(request["cache"]).expanduser()
    need_affinity = bool(request.get("affinity_binder"))

    absent = missing_cache_entries(cache, need_affinity=need_affinity)
    if absent and not request.get("allow_download"):
        msg = (
            f"the Boltz-2 cache {cache} is missing {', '.join(absent)}. "
            "boltz.main.download_boltz2 would fetch 6.2 GB from model-gateway.boltz.bio "
            "on this call; run that download deliberately instead, or re-send this "
            "request with allow_download=true."
        )
        raise RuntimeError(msg)

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(build_yaml_document(request), sort_keys=False))

    results_dir = Path(request.get("results_dir") or (out_dir / "boltz"))
    results_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    output_format = str(request.get("output_format") or "mmcif")
    no_kernels = bool(request.get("no_kernels", True))

    predict.callback(
        data=str(yaml_path),
        out_dir=str(results_dir),
        cache=str(cache),
        devices=int(request.get("devices", 1)),
        accelerator=str(request.get("accelerator") or "gpu"),
        recycling_steps=int(request.get("recycling_steps", 3)),
        sampling_steps=int(request.get("sampling_steps", 200)),
        diffusion_samples=int(request.get("diffusion_samples", 1)),
        sampling_steps_affinity=int(request.get("sampling_steps_affinity", 200)),
        diffusion_samples_affinity=int(request.get("diffusion_samples_affinity", 5)),
        max_parallel_samples=request.get("max_parallel_samples"),
        step_scale=request.get("step_scale"),
        output_format=output_format,
        num_workers=int(request.get("num_workers", 2)),
        override=bool(request.get("override", True)),
        seed=request.get("seed"),
        use_msa_server=bool(request.get("use_msa_server", False)),
        msa_server_url=str(request.get("msa_server_url") or "https://api.colabfold.com"),
        msa_pairing_strategy=str(request.get("msa_pairing_strategy") or "greedy"),
        use_potentials=bool(request.get("use_potentials", False)),
        model="boltz2",
        method=request.get("method"),
        affinity_mw_correction=bool(request.get("affinity_mw_correction", False)),
        preprocessing_threads=int(request.get("preprocessing_threads", 1)),
        max_msa_seqs=int(request.get("max_msa_seqs", 8192)),
        subsample_msa=bool(request.get("subsample_msa", True)),
        num_subsampled_msa=int(request.get("num_subsampled_msa", 1024)),
        no_kernels=no_kernels,
    )

    pred_dir = results_dir / f"boltz_results_{name}" / "predictions" / name
    warnings: list[str] = []
    if not pred_dir.is_dir():
        msg = f"boltz wrote no prediction directory at {pred_dir}"
        raise RuntimeError(msg)

    suffix = ".pdb" if output_format == "pdb" else ".cif"
    samples = collect_samples(pred_dir, name, suffix)
    if not samples:
        msg = f"boltz wrote no {suffix} model into {pred_dir}"
        raise RuntimeError(msg)

    affinity: dict[str, Any] | None = None
    affinity_path = pred_dir / f"affinity_{name}.json"
    if need_affinity:
        if affinity_path.is_file():
            affinity = read_json(affinity_path)
        else:
            warnings.append(
                f"affinity was requested but {affinity_path.name} was not written; the "
                "affinity module reported a failed example"
            )

    return {
        "name": name,
        "engine_version": getattr(boltz, "__version__", "unknown"),
        "samples": samples,
        "affinity": affinity,
        "affinity_requested": need_affinity,
        "results_dir": str(results_dir),
        "predictions_dir": str(pred_dir),
        "yaml_path": str(yaml_path),
        "cache": str(cache),
        "no_kernels": no_kernels,
        "warnings": warnings,
    }


if __name__ == "__main__":
    _driver.main(run)
