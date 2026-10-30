"""Project seed molecules into SynFormer's synthesizable space — the drug-agent ⇄ SynFormer bridge.

This file is executed inside whichever interpreter has SynFormer's dependencies
(torch, omegaconf, einops, rdkit, pandas, scikit-learn), which is not necessarily
the platform's own. It therefore imports nothing from ``drug_agent`` and nothing
outside the standard library except SynFormer itself, and talks to the caller
through the ``request.json`` / ``result.json`` pair described in ``_driver``.

**Which upstream entry point, and where each argument came from.** Everything
below was read out of the pinned checkout (``wenhao-gao/synformer`` at
``bef02bc4481adf3ed792090fc6169331d4d0553a``), not inferred:

``synformer.sampler.analog.parallel.run_sampling_one_cpu``
    The CPU sampler, defined in ``synformer/sampler/analog/parallel.py``. It is
    the only entry point that takes the building-block artefacts explicitly —
    ``mat_path`` (``matrix.pkl``) and ``fpi_path`` (``fpindex.pkl``) — instead of
    reading them out of the checkpoint. It loads the checkpoint with
    ``torch.load(..., map_location="cpu")``, rebuilds the architecture from the
    checkpoint's own ``hyper_parameters.config`` via ``OmegaConf.create``, and
    strips the six-character ``"model."`` prefix off every state-dict key —
    the identical load ``Worker.run`` performs. Its knobs
    (``search_width`` → ``StatePool.factor``, ``exhaustiveness`` →
    ``StatePool.max_active_states``, ``time_limit``, ``max_evolve_steps``,
    ``max_results``) are passed through unchanged.

``synformer.sampler.analog.parallel.run_parallel_sampling_return_smiles``
    The GPU sampler from the same file: one ``Worker`` process per GPU, each
    pinned with ``CUDA_VISIBLE_DEVICES`` and each reading ``config.chem.fpindex``
    and ``config.chem.rxn_matrix`` out of the checkpoint as paths **relative to
    the working directory**. The adapter runs this driver with the checkout as
    cwd, which is what upstream's own README invocation assumes. It returns the
    concatenated per-target dataframe rather than writing a CSV, which is why it
    is used here in preference to ``run_parallel_sampling``.

``synformer.chem.mol.Molecule``
    The input wrapper both samplers expect; ``StatePool`` encodes it and scores
    every product by Tanimoto similarity back to it.

**Why not ``scripts/sample.py``.** That script is a two-line shim onto
``synformer.sampler.analog.cli:main``, and line 3 of ``cli.py`` is
``from parallel import run_parallel_sampling`` — a bare module import that
resolves only when ``synformer/sampler/analog`` is itself on ``sys.path``.
Invoked the way the README documents it raises ``ModuleNotFoundError: No module
named 'parallel'``. The CLI also only ever writes a CSV. Importing ``parallel``
through its real package path is the same code without the broken import.

**Output columns.** ``StatePool.get_dataframe`` emits ``target``, ``smiles``,
``score`` (Morgan/Tanimoto similarity of the product to the seed), ``synthesis``
(the action string: building-block SMILES and ``R<index>`` reaction tokens joined
by ``;``) and ``num_steps`` (``Stack.count_reactions()``), plus ``scf_sim``,
``pharm2d_sim`` and ``rdkit_sim`` on the top rows only. Those three are dropped
here rather than reported for some molecules and not others.

Request JSON:
    input_smiles        seed molecules to project, as SMILES
    model_path          SynFormer checkpoint (``data/trained_weights/default.ckpt``)
    fpindex_path        ``fpindex.pkl``; used on the CPU path, ignored on the GPU path
    rxn_matrix_path     ``matrix.pkl``; same
    device              "cpu" or "cuda[:N]"
    search_width        StatePool.factor            (upstream CLI default 24)
    exhaustiveness      StatePool.max_active_states (upstream CLI default 64)
    time_limit          seconds per seed molecule   (upstream CLI default 180)
    max_evolve_steps    Worker default 12
    max_results         rows kept per seed molecule
    sort_by_scores      StatePool.sort_by_score
    seed                RNG seed for torch/numpy/random
    out_dir             where to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]         {smiles, target, route, num_steps, score, sample_index}
    n_targets, n_targets_parsed, n_targets_with_results, n_rows
    device, torch_version, warnings[], errors[]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: Columns ``StatePool.get_dataframe`` fills for every row. The similarity extras
#: (``scf_sim``/``pharm2d_sim``/``rdkit_sim``) are computed for the top
#: ``num_calc_extra_metrics`` rows only, so they are not reported at all.
_ROW_COLUMNS = ("target", "smiles", "score", "synthesis", "num_steps")


def _require_file(label: str, raw: Any) -> Path:
    """Resolve a request path, failing with the exact name that is missing."""
    if not raw:
        raise FileNotFoundError(f"{label} was not supplied in the request")
    path = Path(str(raw)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist at {path}")
    return path


def _seed_everything(seed: int) -> None:
    """Seed the three generators SynFormer's sampling actually draws from.

    ``Synformer.predict`` draws its next token with ``torch.multinomial`` and the
    diffusion fingerprint head samples noise, so the analog sampler is stochastic
    even with a fixed beam width. Upstream seeds nothing; this at least pins the
    single-process CPU path.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rows_from_frame(frame: Any) -> list[dict[str, Any]]:
    """Coerce one target's dataframe into JSON-safe rows, dropping unusable ones."""
    rows: list[dict[str, Any]] = []
    if frame is None or len(frame) == 0:
        return rows
    missing = [col for col in _ROW_COLUMNS if col not in frame.columns]
    if missing:
        raise RuntimeError(
            "SynFormer returned a dataframe without " + ", ".join(missing) + " — "
            "the checkout's StatePool.get_dataframe contract has changed"
        )
    for record in frame[list(_ROW_COLUMNS)].to_dict(orient="records"):
        smiles = str(record.get("smiles") or "").strip()
        if not smiles:
            continue
        rows.append(
            {
                "smiles": smiles,
                "target": str(record.get("target") or ""),
                "route": str(record.get("synthesis") or ""),
                "num_steps": int(record.get("num_steps") or 0),
                "score": float(record.get("score") or 0.0),
            }
        )
    return rows


def _sample_cpu(request: dict[str, Any], molecules: list[Any], model_path: Path) -> list[dict[str, Any]]:
    """One seed molecule at a time through ``run_sampling_one_cpu``."""
    from synformer.sampler.analog.parallel import run_sampling_one_cpu

    fpindex_path = _require_file("fpindex.pkl", request.get("fpindex_path"))
    rxn_matrix_path = _require_file("matrix.pkl", request.get("rxn_matrix_path"))
    frames: list[Any] = []
    for mol in molecules:
        frames.append(
            run_sampling_one_cpu(
                input=mol,
                model_path=model_path,
                mat_path=rxn_matrix_path,
                fpi_path=fpindex_path,
                search_width=int(request.get("search_width", 24)),
                exhaustiveness=int(request.get("exhaustiveness", 64)),
                time_limit=int(request.get("time_limit", 180)),
                max_results=int(request.get("max_results", 100)),
                max_evolve_steps=int(request.get("max_evolve_steps", 12)),
                sort_by_scores=bool(request.get("sort_by_scores", True)),
            )
        )
    rows: list[dict[str, Any]] = []
    for frame in frames:
        rows.extend(_rows_from_frame(frame))
    return rows


def _sample_gpu(
    request: dict[str, Any], molecules: list[Any], model_path: Path, num_gpus: int
) -> list[dict[str, Any]]:
    """Every seed molecule through the multiprocess ``Worker`` pool."""
    from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles

    frame = run_parallel_sampling_return_smiles(
        input=molecules,
        model_path=model_path,
        search_width=int(request.get("search_width", 24)),
        exhaustiveness=int(request.get("exhaustiveness", 64)),
        num_gpus=num_gpus,
        num_workers_per_gpu=int(request.get("num_workers_per_gpu", 1)),
        time_limit=int(request.get("time_limit", 180)),
        sort_by_scores=bool(request.get("sort_by_scores", True)),
    )
    rows = _rows_from_frame(frame)
    limit = int(request.get("max_results", 100))
    kept: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = kept.setdefault(row["target"], [])
        if len(bucket) < limit:
            bucket.append(row)
    return [row for bucket in kept.values() for row in bucket]


def run(request: dict[str, Any]) -> dict[str, Any]:
    import torch
    from synformer.chem.mol import Molecule

    smiles_in = [str(s).strip() for s in (request.get("input_smiles") or []) if str(s).strip()]
    if not smiles_in:
        raise ValueError(
            "SynFormer's analog sampler projects a seed molecule into synthesizable "
            "space; request.input_smiles was empty, so there is nothing to project"
        )

    model_path = _require_file("SynFormer checkpoint", request.get("model_path"))
    _seed_everything(int(request.get("seed", 0)))

    warnings: list[str] = []
    errors: list[str] = []
    device = str(request.get("device") or "cpu")

    # `Molecule.__init__` only stores the string — RDKit is not consulted until
    # `_rdmol` is touched deep inside `featurize_simple`, so an unparseable seed
    # would otherwise surface as an AttributeError on None several frames down.
    molecules: list[Any] = []
    accepted: list[str] = []
    for smiles in smiles_in:
        mol = Molecule(smiles)
        if not mol.is_valid:
            errors.append(f"seed SMILES did not parse under RDKit: {smiles}")
            continue
        molecules.append(mol)
        accepted.append(smiles)
    if not molecules:
        raise ValueError("none of the seed SMILES parsed under RDKit; there is nothing to project")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device!r} was requested but torch.cuda.is_available() is False; "
                "set DRUGAGENT_SYNFORMER_DEVICE=cpu to use the CPU sampler instead"
            )
        num_gpus = max(1, torch.cuda.device_count())
        warnings.append(
            "the GPU sampler runs one Worker process per GPU and each re-loads the "
            "checkpoint independently; the seed set here does not reach those children, "
            "so a fixed seed does not reproduce a GPU run"
        )
        warnings.append(
            "on the GPU path the building-block index is read from the checkpoint's own "
            "config.chem.fpindex / config.chem.rxn_matrix, resolved against the working "
            "directory; fpindex_path and rxn_matrix_path in this request are ignored"
        )
        rows = _sample_gpu(request, molecules, model_path, num_gpus)
    else:
        rows = _sample_cpu(request, molecules, model_path)

    produced = {row["target"] for row in rows}
    for smiles in accepted:
        if smiles not in produced:
            errors.append(f"no synthesizable analogue reached a single-product stack for {smiles}")

    for index, row in enumerate(rows):
        row["sample_index"] = index

    return {
        "molecules": rows,
        "n_targets": len(smiles_in),
        "n_targets_parsed": len(accepted),
        "n_targets_with_results": len(produced),
        "n_rows": len(rows),
        "device": device,
        "torch_version": str(torch.__version__),
        "warnings": warnings,
        "errors": errors,
    }


if __name__ == "__main__":
    _driver.main(run)
