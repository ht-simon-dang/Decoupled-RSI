"""Project molecules onto synthesizable space with ReaSyn — the drug-agent ⇄ ReaSyn bridge.

This file is executed **inside ReaSyn's own interpreter** (upstream ``env.yml``
builds a conda env named ``reasyn`` on Python 3.10 / torch 2.7), so it imports
nothing from ``drug_agent`` and nothing outside the standard library except the
upstream package itself.

Every upstream fact below was read out of the checkout at
``NVIDIA-Digital-Bio/reasyn@8a0b02178e90``, not guessed:

``run_sampling_one``
    ``reasyn/sampler/parallel.py``. The single-molecule form of what
    ``scripts/sample.py`` drives through ``run_parallel_sampling``. It is used
    here in preference to the parallel path because the parallel path forks a
    worker pool per GPU, writes its own CSV and prints its own summary, whereas
    this one loads the two models in-process and returns the result DataFrame.
    ``scripts/sample_one.py`` is upstream's own demonstration of exactly this
    call (note: that script as committed passes ``num_editflow_samples`` twice
    and would raise ``SyntaxError``; the keyword set used here is the one
    ``run_sampling_one`` actually declares).

``model_path``
    ``run_sampling_one`` asserts ``isinstance(model_path, list) and len(...) == 2``.
    Order is not free: ``reasyn/sampler/sampler.py::Sampler.__init__`` unpacks
    ``self.model, self.model_editflow = model`` and then asserts
    ``model_type == 'autoregressive'`` for the first and ``'editflow'`` for the
    second. So it is [AR checkpoint, Edit-Bridge checkpoint], which is also the
    order of the comma-separated ``-m`` string in ``scripts/sample.py``.

``config``
    There is no inference config file. ``run_sampling_one`` reads the config out
    of the checkpoint — ``OmegaConf.create(ckpt["hyper_parameters"]["config"])``
    — and builds ``ReaSyn(config.model)``. ``configs/train_ar.yml`` is a training
    config; passing it at inference time would be wrong.

``cwd``
    The building-block index and reaction matrix are unpickled from
    ``config.chem.fpindex`` and ``config.chem.rxn_matrix``, which are the
    *relative* paths ``data/processed/comp_2048/{fpindex,matrix}.pkl``. Upstream
    resolves them against the process working directory — ``scripts/sample.py``
    opens with ``sys.path.append('.')`` for the same reason. This driver
    therefore chdirs into the checkout root before importing anything from it.

``output``
    ``reasyn/sampler/sampler.py::Sampler.get_dataframe`` returns one row per
    distinct product with columns ``target``, ``smiles``, ``score`` (Morgan
    Tanimoto similarity of the product to the input), ``synthesis``,
    ``num_steps``; the top ten rows additionally carry ``scf_sim``,
    ``pharm2d_sim`` and ``rdkit_sim``. ``run_sampling_one`` appends ``time``.
    ``synthesis`` is ``reasyn/chem/stack.py::Stack.get_action_string`` — a
    ``;``-joined sequence of building-block SMILES and ``R<n>`` reaction-template
    indices, bottom-up.

``settings``
    The defaults below are upstream's **synthesizable hit expansion** command from
    README.md — ``--search_width 12 --exhaustiveness 128 --num_cycles 12`` with
    ``scripts/sample.py``'s own ``--num_editflow_samples/--num_editflow_steps``
    defaults of 100 — not the lighter ``scripts/sample.py`` defaults of 4 and 8.
    Hit expansion is the task this bridge performs: find synthesizable analogues
    of a known-active molecule.

    That README command also passes ``--no_exact_break``, which
    ``run_sampling_one`` cannot express — it builds ``sampler_opt`` without
    ``exact_break``, so ``Sampler.__init__``'s default of ``True`` applies and the
    search stops early on any input it reconstructs exactly. The consequence is
    stated rather than hidden: for an input that is already synthesizable, this
    bridge returns fewer analogues than upstream's CLI would.

Request JSON:
    root                    path of the ReaSyn checkout (chdir + sys.path target)
    smiles[]                input molecules to project (required, non-empty)
    model_paths[]           [AR checkpoint, Edit-Bridge checkpoint]
    device                  e.g. "cuda:0" / "cpu"
    seed                    RNG seed, applied driver-side (upstream takes none)
    search_width            -> run_sampling_one(search_width=...)   (sampler "factor")
    exhaustiveness          -> run_sampling_one(exhaustiveness=...) (max_active_states)
    num_cycles              refinement cycles; each is bottom-up, top-down, edit
    num_editflow_samples    Edit-Bridge samples per editing pass
    num_editflow_steps      Edit-Bridge integration steps
    max_evolve_steps        autoregressive steps per direction
    max_results             rows kept per input molecule
    time_limit_seconds      -> run_sampling_one(time_limit=...), per input molecule
    add_bb_path             optional extra fpindex pickle to union into the index

Response JSON (``out_dir/result.json``):
    molecules[]  {smiles, target, route, num_steps, similarity_to_input,
                  exact_reconstruction, qc_flags[], sample_index}
    n_inputs, n_projected, warnings[], errors[]

**The output molecules are not the input molecules.** ``score`` is a similarity,
not a guarantee of identity; unless ``exact_reconstruction`` is true the product
is a different compound with different properties. Nothing downstream may carry a
score across that boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The bootstrap must precede the import: _driver sits beside this file, and this
# script is launched by absolute path from a working directory that is the ReaSyn
# checkout, so it is not otherwise importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver


def _seed_everything(seed: int) -> None:
    """Seed the three generators ReaSyn's search actually draws from.

    Upstream exposes no seed argument: ``Sampler.evolve`` picks parent states with
    ``numpy.random.choice`` (``reasyn/sampler/sampler.py``), samples tokens with
    ``torch.multinomial``, and ``reasyn/chem/stack.py`` uses ``random`` for the
    initial stack. Seeding here is therefore a driver-side act, and it does not
    make the sampler deterministic — CUDA reductions and the worker scheduling in
    the upstream parallel path are both free to reorder.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical(smiles: str) -> str:
    """Canonical, isomerism-stripped SMILES — the form ReaSyn compares on.

    Mirrors ``reasyn/chem/mol.py::Molecule.csmiles``
    (``Chem.MolToSmiles(..., canonical=True, isomericSmiles=False)``), so that
    "did the projection land back on the input" is decided the same way upstream
    decides its reconstruction rate.
    """
    from rdkit import Chem

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        return ""
    return Chem.MolToSmiles(rdmol, canonical=True, isomericSmiles=False)


def _rows_to_records(frame, target_smiles, offset):
    """Turn one ``get_dataframe`` result into records the adapter can screen."""
    records = []
    target_canon = _canonical(target_smiles)
    for i, row in enumerate(frame.to_dict(orient="records")):
        product = str(row.get("smiles") or "")
        if not product:
            continue
        exact = bool(target_canon) and _canonical(product) == target_canon
        flags = ["reasyn_exact_reconstruction" if exact else "reasyn_projected_analogue"]
        if not row.get("num_steps"):
            # A zero-reaction "route" is a bare building block, not a synthesis.
            flags.append("reasyn_no_reaction_step")
        records.append(
            {
                "smiles": product,
                "target": target_smiles,
                "route": str(row.get("synthesis") or ""),
                "num_steps": int(row.get("num_steps") or 0),
                "similarity_to_input": float(row.get("score") or 0.0),
                "exact_reconstruction": exact,
                "qc_flags": flags,
                "sample_index": offset + i,
            }
        )
    return records


def run(request):
    """Project every requested molecule and return the products with their routes."""
    root = request.get("root")
    if not root:
        raise ValueError("request is missing 'root' (the ReaSyn checkout path)")
    # Upstream resolves data/processed/... relative to the working directory; see
    # the module docstring and scripts/sample.py's `sys.path.append('.')`.
    os.chdir(root)
    sys.path.insert(0, str(root))

    inputs = [s for s in (request.get("smiles") or []) if s]
    if not inputs:
        raise ValueError("ReaSyn projects an input molecule; 'smiles' was empty")

    model_paths = [str(p) for p in (request.get("model_paths") or [])]
    if len(model_paths) != 2:
        raise ValueError(
            f"ReaSyn needs exactly two checkpoints, [autoregressive, edit-bridge]; got {len(model_paths)}"
        )
    for path in model_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"ReaSyn checkpoint not found: {path}")

    _seed_everything(int(request.get("seed") or 0))

    # Imported after the chdir so the pickled index paths inside the checkpoint
    # config resolve. `reasyn.sampler.parallel` calls mp.set_start_method('spawn')
    # at import time, which is why it is imported once, here, and not per molecule.
    from reasyn.chem.mol import Molecule
    from reasyn.sampler.parallel import run_sampling_one

    warnings = []
    errors = []
    records = []
    n_projected = 0
    for mol_smiles in inputs:
        try:
            frame = run_sampling_one(
                input=Molecule(mol_smiles),
                model_path=[Path(p) for p in model_paths],
                # Defaults are README.md's hit-expansion settings, not the
                # lighter scripts/sample.py CLI defaults. See the module docstring.
                search_width=int(request.get("search_width") or 12),
                exhaustiveness=int(request.get("exhaustiveness") or 128),
                max_evolve_steps=int(request.get("max_evolve_steps") or 8),
                max_results=int(request.get("max_results") or 10),
                time_limit=int(request.get("time_limit_seconds") or 600),
                add_bb_path=request.get("add_bb_path") or None,
                device=str(request.get("device") or "cuda:0"),
                num_cycles=int(request.get("num_cycles") or 12),
                num_editflow_samples=int(request.get("num_editflow_samples") or 100),
                num_editflow_steps=int(request.get("num_editflow_steps") or 100),
            )
        except Exception as exc:  # one bad input must not lose the whole batch
            errors.append(f"{mol_smiles}: {type(exc).__name__}: {exc}")
            continue
        if frame is None or len(frame) == 0:
            warnings.append(f"ReaSyn found no synthetic pathway for {mol_smiles}")
            continue
        n_projected += 1
        records.extend(_rows_to_records(frame, mol_smiles, len(records)))

    return {
        "molecules": records,
        "n_inputs": len(inputs),
        "n_projected": n_projected,
        "warnings": warnings,
        "errors": errors,
    }


if __name__ == "__main__":
    _driver.main(run)
