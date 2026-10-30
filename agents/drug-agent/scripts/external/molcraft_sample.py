"""Sample ligands for one pocket with MolCRAFT — the drug-agent ⇄ MolCRAFT bridge.

Runs inside whichever interpreter has MolCRAFT's dependencies. It imports nothing
from ``drug_agent``: the two sides talk over a JSON request/response pair on disk,
so a CUDA fault or a Lightning teardown here cannot reach the caller.

**The entry point.** ``MolCRAFT/sample_for_pocket.py`` exposes ``call()``, and that
is what runs — not a reimplementation of it. Everything this driver does around
that call is forced by how ``call()`` is written, and each step names its source:

``chdir into the project directory``
    ``call()`` line 102 opens ``'./checkpoints/config.yaml'`` and its ``ckpt_path``
    default is ``'./checkpoints/last.ckpt'``. Both are relative to the process
    working directory, so the driver moves there rather than passing paths that
    would be ignored.
``PROJECT_SUBDIR on sys.path``
    MolCRAFT, MolJO and MolPilot are three sibling directories in one repository,
    each with its own top-level ``core`` package. Importing ``core.models`` from the
    repository root would resolve to whichever sibling came first on the path.
``read ./output``
    ``DockingTestCallback.on_test_epoch_end`` empties ``OUT_DIR`` (``'./output'``,
    ``core/callbacks/validation_callback_for_sample.py:49``) and writes ``<idx>.sdf``
    for every sample that reconstructed, sanitized and came out as one fragment.
    The directory is wiped at the start of each test epoch, so it is read after every
    ``call()`` and before the next one.
``one call() per batch``
    ``call()`` takes ``num_samples`` but no batch size; the per-forward batch is
    ``cfg.evaluation.batch_size`` inside the checkpoint's config. Chunking the request
    is the only way the caller's ``batch_size`` means anything, at the cost of one
    checkpoint load per chunk.

Request JSON:
    project_dir        the MolCRAFT project directory inside the series checkout
    protein_path       protein PDB (whole structure or pocket)
    ligand_path        reference ligand SDF; MolCRAFT clips the site from it
    ckpt_path          MolCRAFT/checkpoints/last.ckpt
    num_samples        molecules to draw in total
    batch_size         molecules per call() invocation
    sample_steps       BFN sampling steps
    sample_num_atoms   'prior' (size from pocket volume) or 'ref'
    beta1, sigma1_coord, sampling_strategy, seed
    out_dir            where to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]        {smiles, sdf, sample_index, n_atoms, qc_flags[]}
    n_requested        samples asked of the model — the denominator for validity
    n_written          SDF files MolCRAFT actually produced
    warnings[]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver


def _load_entry_point(project_dir: Path):
    """Import MolCRAFT's ``call()`` and its output directory from the checkout."""
    sys.path.insert(0, str(project_dir))
    # Not cosmetic: call() resolves its config and checkpoint relative to cwd.
    os.chdir(project_dir)
    import sample_for_pocket
    from core.callbacks.validation_callback_for_sample import OUT_DIR

    return sample_for_pocket, Path(OUT_DIR)


def _sdf_order(path: Path) -> tuple[int, str]:
    """Sort ``0.sdf``, ``1.sdf`` … ``10.sdf`` numerically, anything else by name."""
    stem = path.stem
    return (int(stem), "") if stem.isdigit() else (1 << 30, stem)


def _collect(out_dir: Path, start_index: int, warnings: list) -> list:
    """Every SDF MolCRAFT wrote in this batch, as records the adapter can screen."""
    from rdkit import Chem

    records = []
    if not out_dir.is_dir():
        warnings.append(f"MolCRAFT wrote no {out_dir} directory for this batch")
        return records
    for offset, path in enumerate(sorted(out_dir.glob("*.sdf"), key=_sdf_order)):
        block = path.read_text()
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
        if mol is None:
            records.append(
                {
                    "smiles": "",
                    "sdf": block,
                    "sample_index": start_index + offset,
                    "n_atoms": 0,
                    "qc_flags": ["sdf_unparseable"],
                }
            )
            continue
        records.append(
            {
                "smiles": Chem.MolToSmiles(mol),
                "sdf": block,
                "sample_index": start_index + offset,
                "n_atoms": mol.GetNumHeavyAtoms(),
                "qc_flags": [],
            }
        )
    return records


def run(request: dict) -> dict:
    """Draw ``num_samples`` ligands for one pocket, in ``batch_size`` chunks."""
    project_dir = Path(request["project_dir"]).expanduser().resolve()
    protein = str(Path(request["protein_path"]).expanduser().resolve())
    ligand = str(Path(request["ligand_path"]).expanduser().resolve())
    ckpt = str(Path(request["ckpt_path"]).expanduser().resolve())

    config = project_dir / "checkpoints" / "config.yaml"
    for required in (Path(ckpt), config):
        if not required.is_file():
            raise FileNotFoundError(
                f"MolCRAFT needs {required}; sample_for_pocket.call() reads the checkpoint "
                "through Trainer.test(ckpt_path=...) and its training config from "
                "./checkpoints/config.yaml"
            )

    total = max(1, int(request.get("num_samples") or 10))
    chunk = max(1, int(request.get("batch_size") or total))
    warnings: list = []

    sample_for_pocket, out_dir = _load_entry_point(project_dir)

    molecules: list = []
    drawn = 0
    while drawn < total:
        n = min(chunk, total - drawn)
        sample_for_pocket.call(
            protein,
            ligand,
            ckpt_path=ckpt,
            num_samples=n,
            sample_steps=int(request.get("sample_steps") or 100),
            sample_num_atoms=str(request.get("sample_num_atoms") or "prior"),
            beta1=float(request.get("beta1", 1.5)),
            sigma1_coord=float(request.get("sigma1_coord", 0.03)),
            sampling_strategy=str(request.get("sampling_strategy") or "end_back"),
            seed=int(request.get("seed", 1234)),
        )
        # Read before the next call(): on_test_epoch_end removes OUT_DIR first.
        molecules.extend(_collect(out_dir, drawn, warnings))
        drawn += n

    return {
        "molecules": molecules,
        "n_requested": total,
        "n_written": len(molecules),
        "warnings": warnings,
    }


if __name__ == "__main__":
    _driver.main(run)
