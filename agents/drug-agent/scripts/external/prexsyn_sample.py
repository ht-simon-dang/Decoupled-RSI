"""Sample synthesizable molecules with PrexSyn — the drug-agent ⇄ PrexSyn bridge.

Runs inside whichever interpreter has ``prexsyn`` and its C++ extension
``prexsyn_engine``. It imports nothing from ``drug_agent``; the two sides talk over a
JSON request/response pair on disk, so a segfault in the extension cannot reach the
caller.

**Which entry points this calls, and where each came from.** Nothing here is invented:
every call below is the one upstream's own runnable example makes.

``AllInOneLoader(config_path)`` / ``.model().to(device).eval()`` / ``.detokenizer()``
    ``scripts/examples/projection.py`` lines 33-36, and the same three lines in
    ``docs/getting-started/import.md``. The loader reads the released YAML, resolves
    the checkpoint as that YAML's path with a ``.ckpt`` suffix
    (``prexsyn/shortcuts/loader.py``), and downloads the checkpoint and the
    chemical-space cache from the ``remote:`` URLs in the YAML if they are missing.

``MoleculeProjector(model=, detokenizer=, descriptor=, num_samples=, batch_size_limit=)``
    ``scripts/examples/projection.py`` lines 38-43. ``batch_size_limit`` defaults to 64
    in ``prexsyn/shortcuts/projector.py``; the adapter passes it explicitly.

``projector.many(smiles_list)`` → ``result.items`` sorted by descending Tanimoto
    ``prexsyn/shortcuts/projector.py::MoleculeProjector._aggregate_results``, which
    sorts on ``prexsyn/utils/metrics.py::tanimoto_similarity``. Each item carries
    ``.molecule``, ``.synthesis``, ``.similarity`` and ``.get_tree()``.

``projector.desc(genotypes)`` with sparse random boolean fingerprints
    ``prexsyn/shortcuts/genetic.py::initialize``, which builds
    ``np.random.rand(n, *descriptor_function.size()) < 0.02`` and then keeps
    ``result.best()`` per genotype in ``::hatch``. That is the unconditional path here.

``torch.set_grad_enabled(False)``, ``torch.manual_seed`` / ``np.random.seed``
    The first is ``scripts/examples/projection.py`` line 30. The second pair is what
    ``docs/customization/sampling.md`` names to make a run repeatable: "The process is
    stochastic unless you seed NumPy and PyTorch."

**Working directory.** The chemical-space cache path inside the released YAML is
relative (``data/chemical_spaces/...``), and upstream's installation notes say to run
from the repository root. ``ExternalBackend.run_json`` already launches this driver
with ``cwd`` set to the checkout root, so those relative paths resolve. ``config_path``
itself arrives absolute.

PrexSyn's own ``pyproject.toml`` declares ``requires-python = ">=3.11"`` and its
``.python-version`` pins 3.13, so this driver is written for a modern interpreter —
unlike ``scripts/pfm_sample_pocket.py``, which stays 3.8-compatible because PFM's
environment is.

Request JSON:
    config_path        absolute path to a PrexSyn model YAML
    context_smiles     reference ligands to project; empty means unconditional
    n_molecules        how many products to return
    num_samples        pathways sampled per conditioning descriptor
    batch_size_limit   descriptors per model call
    descriptor         "ecfp4" or "fcfp4" — must be registered by the loaded config
    device             "cuda" / "cuda:0" / "cpu"
    seed               RNG seed for torch and numpy
    oversample_factor  unconditional path only; genotypes drawn per requested molecule
    out_dir            where to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]        {smiles, route, similarity, query_smiles, sample_index}
    mode               "projection" or "unconditional"
    n_requested, n_sampled, model_name, prexsyn_version, engine_version, device,
    descriptor, seconds_model, seconds_detokenize, warnings[], errors[]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _driver

#: Upstream's own oversampling default when hatching random genotypes
#: (``prexsyn/shortcuts/genetic.py::initialize``, ``oversample_factor=2``). Some
#: genotypes decode to no product at all, which is why upstream draws extra.
DEFAULT_OVERSAMPLE_FACTOR = 2

#: Bit density of a random ECFP4 genotype, from the same function:
#: ``np.random.rand(...) < 0.02``.
RANDOM_GENOTYPE_DENSITY = 0.02


def _resolve_device(torch, requested, warnings):
    """The device to load the model onto, falling back to CPU with a warning."""
    device = requested or "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append(
            "CUDA is not available in this interpreter, so PrexSyn ran on CPU; "
            "upstream calls CPU inference substantially slower."
        )
        return "cpu"
    return device


def _check_descriptor(loader, descriptor):
    """Fail loudly if the loaded checkpoint was never trained on this descriptor."""
    registered = [str(item.type) for item in loader.config().descriptors]
    if descriptor not in registered:
        offered = ", ".join(registered) or "none"
        raise ValueError(
            f"descriptor {descriptor!r} is not registered by {loader.config_path()}; "
            f"this config offers {offered}"
        )


def _valid_queries(smiles_list, warnings):
    """Drop references RDKit cannot parse, recording each as a warning, not an error."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    kept = []
    for smiles in smiles_list:
        text = str(smiles or "").strip()
        if not text:
            continue
        if Chem.MolFromSmiles(text) is None:
            warnings.append(f"PrexSyn skipped an unparseable reference SMILES: {text!r}")
            continue
        kept.append(text)
    return kept


def _project(projector, queries, n_molecules, warnings):
    """Chemical-space projection: rank each reference's products by similarity."""
    per_query = max(1, -(-int(n_molecules) // len(queries)))  # ceiling division
    batched = projector.many(queries)
    molecules = []
    for query_index, result in enumerate(batched.results):
        if len(result) == 0:
            warnings.append(
                f"PrexSyn produced no product for reference {queries[query_index]!r}: every "
                f"sampled pathway was incomplete or its reactions did not apply."
            )
            continue
        # `_aggregate_results` already sorted items by descending Tanimoto similarity.
        for item in result.items[:per_query]:
            molecules.append(
                {
                    "smiles": item.molecule.smiles(),
                    "route": item.get_tree(),
                    "similarity": float(item.similarity),
                    "query_smiles": queries[query_index],
                    "sample_index": len(molecules),
                }
            )
    return molecules, batched.time


def _unconditional(projector, n_molecules, oversample_factor, warnings):
    """Draw sparse random genotypes and keep each one's best product, as `hatch` does."""
    import numpy as np

    descriptor_function = projector.descriptor_function
    if descriptor_function.dtype() != np.dtype(bool):
        raise ValueError(
            f"the unconditional path needs a boolean descriptor; "
            f"{projector.descriptor_name!r} is {descriptor_function.dtype()}, and upstream's "
            f"genetic sampler raises NotImplementedError for anything else"
        )
    n_genotypes = max(1, int(n_molecules) * max(1, int(oversample_factor)))
    genotypes = np.random.rand(n_genotypes, *descriptor_function.size()) < RANDOM_GENOTYPE_DENSITY
    batched = projector.desc(genotypes)

    molecules = []
    n_empty = 0
    for result in batched.results:
        best = result.best()
        if best is None:
            n_empty += 1
            continue
        molecules.append(
            {
                "smiles": best.molecule.smiles(),
                "route": best.get_tree(),
                "similarity": float(best.similarity),
                "query_smiles": None,
                "sample_index": len(molecules),
            }
        )
    if n_empty:
        warnings.append(
            f"{n_empty} of {n_genotypes} random genotypes decoded to no product; upstream "
            f"oversamples by {oversample_factor}x for exactly this reason."
        )
    return molecules, batched.time


def run(request):
    """Load the released model and sample, either from references or from the prior."""
    import numpy as np
    import torch

    # scripts/examples/projection.py line 30 — inference only, no autograd graph.
    torch.set_grad_enabled(False)

    import prexsyn
    import prexsyn_engine
    from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector

    warnings = []
    config_path = request.get("config_path")
    if not config_path:
        raise ValueError("request carries no config_path (a PrexSyn model YAML)")
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"PrexSyn model config not found at {config_path}")

    seed = int(request.get("seed") or 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    descriptor = str(request.get("descriptor") or "ecfp4")
    n_molecules = max(1, int(request.get("n_molecules") or 1))
    num_samples = max(1, int(request.get("num_samples") or 16))
    batch_size_limit = max(1, int(request.get("batch_size_limit") or 64))
    oversample_factor = max(1, int(request.get("oversample_factor") or DEFAULT_OVERSAMPLE_FACTOR))

    loader = AllInOneLoader(config_path)
    _check_descriptor(loader, descriptor)
    device = _resolve_device(torch, request.get("device"), warnings)
    model = loader.model().to(device).eval()
    projector = MoleculeProjector(
        model=model,
        detokenizer=loader.detokenizer(),
        descriptor=descriptor,
        num_samples=num_samples,
        batch_size_limit=batch_size_limit,
    )

    queries = _valid_queries(request.get("context_smiles") or [], warnings)
    if queries:
        mode = "projection"
        molecules, timing = _project(projector, queries, n_molecules, warnings)
    else:
        mode = "unconditional"
        warnings.append(
            "no reference ligand was supplied, so PrexSyn sampled from sparse random "
            "ECFP4 genotypes: this is a draw from the model's prior over synthesizable "
            "space, not a projection of anything."
        )
        molecules, timing = _unconditional(projector, n_molecules, oversample_factor, warnings)

    return {
        "molecules": molecules,
        "mode": mode,
        "n_requested": n_molecules,
        "n_sampled": len(molecules),
        "num_samples": num_samples,
        "descriptor": descriptor,
        "device": str(device),
        "model_name": str(loader.config().note.name),
        "config_path": str(loader.config_path()),
        "prexsyn_version": getattr(prexsyn, "__version__", "unknown"),
        "engine_version": getattr(prexsyn_engine, "__version__", "unknown"),
        "seconds_model": float(timing.model),
        "seconds_detokenize": float(timing.detok),
        "warnings": warnings,
        "errors": [],
    }


if __name__ == "__main__":
    _driver.main(run)
