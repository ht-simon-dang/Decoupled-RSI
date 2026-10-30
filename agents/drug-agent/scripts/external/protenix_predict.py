"""Predict one complex with Protenix — the drug-agent ⇄ protenix bridge.

This runs inside the ``drug-agent-protenix`` environment (Python 3.11, protenix
2.0.0, torch 2.7.1+cu128), not the platform's, so it imports nothing from
``drug_agent``. The two sides talk over a JSON request/response pair on disk.

**It calls upstream's own entry points**, one level below the CLI:
``runner.batch_inference.get_default_runner`` builds the configured model,
``preprocess_input`` prepares the input JSON, and ``runner.inference.infer_predict``
runs the samples. The CLI wrapper around those three is ``inference_jsons``, and
it is deliberately not used: it catches every exception per input file and only
logs a warning, so a failed prediction would return exit 0 with no output and the
adapter would have to guess why. Here a failure raises and becomes a structured
error the adapter can report.

**It refuses to download.** ``get_default_runner`` calls ``download_inference_cache``,
which fetches ~2.1 GB from ``protenix.tos-cn-beijing.volces.com`` when the cache
is cold — the checkpoint plus the CCD component definitions. A tool call is not
the place to discover that, so the artifacts are checked first and a cold cache is
a structured failure naming each missing path. ``allow_download`` exists for an
operator running this driver by hand with network access; the adapter always sends
it false.

**PROTENIX_ROOT_DIR is set before the config modules are imported**, because
``configs.configs_inference`` and ``configs.configs_data`` read it at import time
to build every path they expose. Importing them first and assigning afterwards
would silently leave the checkpoint directory pointing at ``$HOME``.

**LAYERNORM_TYPE is an import-time switch.**
``protenix.model.triangular.layers`` reads it at module import, not per call, so it
has to be in the environment before the first protenix import — which is why it is
set here beside ``PROTENIX_ROOT_DIR`` rather than passed as a config field like the
other two kernel choices. On this host the fused path returns NaN; the adapter
therefore sends ``torch``, and this driver records what it was given.

**MSA.** ``use_msa`` false is upstream's single-sequence mode: it skips the search
entirely and warns that accuracy degrades. True makes ``preprocess_input`` post the
sequence to Protenix's MSA server, which is network I/O, so the adapter only sets
it when an operator asked for it — or supplies precomputed MSA paths in the chain
entry, in which case upstream's ``need_msa_search`` finds nothing to do and no
request is made.

Request JSON (``argv[1]``):
    name                    job name; becomes the sample name in every output path
    sequences[]             {kind: protein|dna|rna, sequence, count, msa}
    ligands[]               {ligand: SMILES | "CCD_XXX" | "FILE_/path.sdf", count}
    root_dir                PROTENIX_ROOT_DIR — where checkpoint/ and common/ live
    model_name              checkpoint stem, e.g. protenix_base_default_v1.0.0
    seeds[], n_cycle, n_step, n_sample, dtype, use_msa, msa_server_mode,
    trimul_kernel, triatt_kernel, layernorm_type, enable_cache, enable_fusion,
    enable_tf32, need_atom_confidence, allow_download, results_dir
    out_dir                 injected by the caller; result.json goes here

Response JSON (``<out_dir>/result.json``):
    samples[]               {rank, seed, format, path, coordinates, confidence{},
                            n_atom_rows, n_nonfinite_rows, nonfinite}
    engine_version          protenix.version.__version__ as installed
    model_name, root_dir, results_dir, n_token, seconds, use_msa, warnings[]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _driver

#: The four data caches ``download_inference_cache`` requires for any prediction,
#: as ``(config key, path under the root)``. Names are upstream's own, and the
#: basenames are asserted equal to the URL basenames by upstream itself.
DATA_CACHE_FILES = (
    "common/components.cif",
    "common/components.cif.rdkit_mol.pkl",
    "common/clusters-by-entity-40.txt",
    "common/obsolete_release_date.csv",
)

#: Confidence fields that are one number per sample. The chain and chain-pair
#: matrices in the same file are carried through untouched under their own keys
#: rather than being flattened into these.
SCALAR_CONFIDENCE = (
    "plddt",
    "gpde",
    "ptm",
    "iptm",
    "has_clash",
    "disorder",
    "num_recycles",
    "ranking_score",
    "pb_ranking_score",
    "has_vdw_pl_clash",
    "pb_ranking_score_vdw_penalized",
)

#: ``<name>_summary_confidence_sample_<rank>.json`` — upstream's dumper naming.
SUMMARY_RE = re.compile(r"^(?P<name>.+)_summary_confidence_sample_(?P<rank>\d+)\.json$")

POLYMER_KINDS = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}


def missing_artifacts(root: Path, model_name: str) -> list[str]:
    """Every inference artifact absent from *root*, as absolute paths."""
    wanted = [root / rel for rel in DATA_CACHE_FILES]
    wanted.append(root / "checkpoint" / f"{model_name}.pt")
    return [str(path) for path in wanted if not path.exists()]


def build_input(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Assemble Protenix's input JSON: a list of jobs, each a list of chains."""
    sequences: list[dict[str, Any]] = []
    for entry in request.get("sequences", []):
        kind = str(entry.get("kind") or "protein")
        key = POLYMER_KINDS.get(kind)
        if key is None:
            msg = f"unsupported entity kind {kind!r}; expected one of {sorted(POLYMER_KINDS)}"
            raise ValueError(msg)
        chain: dict[str, Any] = {"sequence": entry["sequence"], "count": int(entry.get("count", 1))}
        if entry.get("msa"):
            chain["msa"] = entry["msa"]
        sequences.append({key: chain})
    for entry in request.get("ligands", []):
        ligand = entry.get("ligand")
        if not ligand:
            msg = "a ligand entry needs `ligand`: a SMILES, 'CCD_<code>' or 'FILE_<path>'"
            raise ValueError(msg)
        sequences.append({"ligand": {"ligand": ligand, "count": int(entry.get("count", 1))}})
    if not sequences:
        msg = "request carries no sequences and no ligands"
        raise ValueError(msg)
    return [{"sequences": sequences, "name": str(request.get("name") or "complex")}]


#: How an mmCIF row spells a coordinate that is not a number. Protenix writes
#: these verbatim when the forward pass produced NaN, and the file is otherwise
#: a perfectly well-formed structure — which is exactly why it has to be caught.
NON_FINITE_TOKENS = frozenset({"nan", "-nan", "inf", "-inf", "+inf"})


def coordinate_health(text: str) -> tuple[int, int]:
    """``(atom rows, rows carrying a non-finite coordinate)`` in one mmCIF model.

    A prediction whose numerics blew up still writes a complete file: every atom,
    every residue name, every chain, with ``nan`` where the coordinates should be.
    Nothing downstream would notice — a Pose would be built, ranked and reported —
    so the driver reads the file back and counts.
    """
    rows = 0
    bad = 0
    for line in text.splitlines():
        if not line.startswith(("ATOM ", "HETATM")):
            continue
        rows += 1
        if any(field.lower() in NON_FINITE_TOKENS for field in line.split()):
            bad += 1
    return rows, bad


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def collect_samples(results_dir: Path) -> list[dict[str, Any]]:
    """Pair every predicted structure with the confidence file written beside it.

    The coordinates are carried back as text rather than as a path because the
    adapter's scratch directory is removed as soon as the driver exits — a path
    would name a file that no longer exists by the time a Pose is built.
    """
    samples: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.rglob("*_summary_confidence_sample_*.json")):
        match = SUMMARY_RE.match(summary_path.name)
        if match is None:
            continue
        name, rank = match.group("name"), int(match.group("rank"))
        structure_path = summary_path.parent / f"{name}_sample_{rank}.cif"
        seed_dir = summary_path.parent.parent.name
        text = structure_path.read_text() if structure_path.is_file() else None
        n_rows, n_bad = coordinate_health(text) if text else (0, 0)
        # A model that is entirely NaN is not a structure. It is withheld rather
        # than returned with a caveat, because a caveat is a field somebody has to
        # read and coordinates are a field everybody uses.
        unusable = n_rows > 0 and n_bad == n_rows
        samples.append(
            {
                "rank": rank,
                "seed": int(seed_dir.removeprefix("seed_")) if seed_dir.startswith("seed_") else None,
                "format": "mmcif",
                "path": str(structure_path),
                "coordinates": None if unusable else text,
                "n_atom_rows": n_rows,
                "n_nonfinite_rows": n_bad,
                "nonfinite": unusable,
                "confidence": read_json(summary_path),
            }
        )
    samples.sort(key=lambda sample: (sample["seed"] if sample["seed"] is not None else 0, sample["rank"]))
    return samples


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run one Protenix prediction and return its structures and confidence."""
    out_dir = Path(request["out_dir"])
    root = Path(request.get("root_dir") or "~/.protenix").expanduser()
    model_name = str(request.get("model_name") or "protenix_base_default_v1.0.0")

    absent = missing_artifacts(root, model_name)
    if absent and not request.get("allow_download"):
        msg = (
            f"the Protenix artifact cache {root} is missing {', '.join(absent)}. "
            "runner.inference.download_inference_cache would fetch ~2.1 GB from "
            "protenix.tos-cn-beijing.volces.com on this call; run that download "
            "deliberately with `python scripts/fetch_protenix_weights.py --model "
            f"{model_name}`, or re-send this request with allow_download=true."
        )
        raise RuntimeError(msg)

    # Before the config modules are imported: they read this at import time.
    os.environ["PROTENIX_ROOT_DIR"] = str(root)
    root.mkdir(parents=True, exist_ok=True)

    # DeepSpeed JIT-compiles its CUDA ops and shells out to `ninja` to do it, which
    # it looks for on PATH. The platform hands a driver its own PATH (the launching
    # environment's), so this interpreter's sibling binaries are not on it and the
    # build fails with "Ninja is required to load C++ extensions" several minutes
    # into a prediction. Putting this environment's bin directory first is what
    # makes a foreign-interpreter driver see the tools installed beside it.
    bin_dir = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join([bin_dir, os.environ.get("PATH", "")]).rstrip(os.pathsep)

    # The third kernel switch, and the only one upstream exposes as an environment
    # variable rather than a config field: anything other than "fast_layernorm"
    # makes `InferenceRunner` skip the JIT-compiled fused LayerNorm. Read here so
    # all three sit in the request beside each other.
    os.environ["LAYERNORM_TYPE"] = str(request.get("layernorm_type") or "fast_layernorm")

    from configs.configs_inference import inference_configs
    from protenix.version import __version__ as protenix_version
    from runner.batch_inference import get_default_runner, preprocess_input
    from runner.inference import infer_predict

    results_dir = Path(request.get("results_dir") or (out_dir / "protenix"))
    results_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    name = str(request.get("name") or "complex")
    input_path = input_dir / f"{name}.json"
    input_path.write_text(json.dumps(build_input(request), indent=2))

    use_msa = bool(request.get("use_msa", False))
    # `inference_jsons` sets this before building the runner; the dump directory is
    # baked into the runner's configs, so assigning it afterwards has no effect.
    inference_configs["dump_dir"] = str(results_dir)

    started = time.perf_counter()
    runner = get_default_runner(
        seeds=[int(s) for s in request.get("seeds") or [101]],
        n_cycle=int(request.get("n_cycle", 10)),
        n_step=int(request.get("n_step", 200)),
        n_sample=int(request.get("n_sample", 5)),
        dtype=str(request.get("dtype") or "bf16"),
        model_name=model_name,
        use_msa=use_msa,
        trimul_kernel=str(request.get("trimul_kernel") or "cuequivariance"),
        triatt_kernel=str(request.get("triatt_kernel") or "cuequivariance"),
        enable_cache=bool(request.get("enable_cache", True)),
        enable_fusion=bool(request.get("enable_fusion", True)),
        enable_tf32=bool(request.get("enable_tf32", True)),
        use_template=False,
        use_rna_msa=False,
        use_seeds_in_json=False,
        need_atom_confidence=bool(request.get("need_atom_confidence", False)),
    )
    configs = runner.configs
    configs["input_json_path"] = preprocess_input(
        str(input_path),
        out_dir=str(results_dir),
        use_msa=use_msa,
        use_template=False,
        use_rna_msa=False,
        msa_server_mode=str(request.get("msa_server_mode") or "protenix"),
    )
    infer_predict(runner, configs)
    seconds = time.perf_counter() - started

    warnings: list[str] = []
    samples = collect_samples(results_dir)
    if not samples:
        error_dir = Path(getattr(runner, "error_dir", results_dir))
        detail = ""
        error_file = error_dir / "error.txt"
        if error_file.is_file():
            detail = " " + error_file.read_text().strip()[-2000:]
        msg = f"Protenix wrote no prediction into {results_dir}.{detail}"
        raise RuntimeError(msg)
    if all(sample["nonfinite"] for sample in samples):
        msg = (
            f"Protenix ran to completion but every one of its {len(samples)} sample(s) is "
            "entirely non-finite: the .cif carries `nan` for every atom. The forward pass "
            "produced NaN rather than coordinates, so there is no structure to return. "
            "Retry with trimul_kernel='torch' and triatt_kernel='torch' to take the "
            "pure-torch path instead of the cuequivariance kernels, which is the usual "
            "cause on a GPU whose architecture the shipped kernels were not built for."
        )
        raise RuntimeError(msg)
    for sample in samples:
        if sample["nonfinite"]:
            warnings.append(
                f"sample {sample['rank']} was withheld: all {sample['n_atom_rows']} atom rows "
                "carry non-finite coordinates"
            )
        elif sample["n_nonfinite_rows"]:
            warnings.append(
                f"sample {sample['rank']} has {sample['n_nonfinite_rows']} of "
                f"{sample['n_atom_rows']} atom rows with non-finite coordinates"
            )
        elif sample["coordinates"] is None:
            warnings.append(f"sample {sample['rank']}: a confidence file with no .cif beside it")

    return {
        "name": name,
        "engine_version": protenix_version,
        "model_name": model_name,
        "samples": samples,
        "root_dir": str(root),
        "results_dir": str(results_dir),
        "input_json": str(input_path),
        "use_msa": use_msa,
        "seconds": seconds,
        "kernels": {
            "trimul": str(request.get("trimul_kernel") or "cuequivariance"),
            "triatt": str(request.get("triatt_kernel") or "cuequivariance"),
            "layernorm": os.environ["LAYERNORM_TYPE"],
        },
        "scalar_confidence_keys": list(SCALAR_CONFIDENCE),
        "warnings": warnings,
    }


if __name__ == "__main__":
    _driver.main(run)
