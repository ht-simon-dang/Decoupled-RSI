"""Uni-Mol and Uni-Mol2 in their own interpreter: represent, fine-tune, predict.

This script runs inside the ``drug-agent-unimol`` conda environment and imports
nothing from ``drug_agent``. It exists because ``unimol_tools`` pins numpy 2.2.6
and the platform environment carries numpy 2.5.2, so the two cannot share an
interpreter.

Three verbs, selected by ``request["task"]``, because Uni-Mol is a representation
model rather than a property oracle and the difference has to survive the wire:

``represent``
    ``UniMolRepr.get_repr`` — a pooled ``[CLS]`` embedding per molecule, plus the
    per-atom embeddings and the conformer coordinates the encoder actually saw.
``fit``
    ``MolTrain.fit`` — k-fold fine-tuning of the pretrained encoder on caller
    supplied labels, written to ``model_dir``.
``predict``
    ``MolPredict.predict`` — inference with a model directory ``fit`` produced.
    It refuses without one, because there is no pretrained ADMET head to fall
    back on.

**Weights are never downloaded here.** ``UNIMOL_WEIGHT_DIR`` is pointed at the
directory the adapter has already verified, and ``HF_HUB_OFFLINE`` is set before
``unimol_tools`` is imported. Upstream's ``weights/weighthub.py`` silently fetches
a 190 MB checkpoint from Hugging Face on first use; inside a metered run that
would read as a slow model rather than as a missing file.

**The namespace-package shadow.** ``ExternalBackend.child_env`` puts the Uni-Mol
checkout on ``PYTHONPATH``, and the checkout contains ``unimol_tools/`` — a
directory with no ``__init__.py``, so PEP 420 records it as a namespace portion.
The installed regular package still wins the import, but only because a regular
package beats a recorded namespace portion, which is a rule too subtle to rely on
for a package this one cannot run without. The shadow is removed explicitly.

Request keys (``out_dir`` is injected by the caller):
    task            "represent" | "fit" | "predict"
    smiles          list of SMILES, already screened by the caller with RDKit
    weight_dir      directory holding the pretrained checkpoints and dictionary
    device          "cuda:0" / "cpu"
    model_name      "unimolv1" | "unimolv2"
    model_size      Uni-Mol2 size token, e.g. "84m"
    batch_size      inference/training batch size
    remove_hs       whether hydrogens are dropped before encoding
    max_atoms       atom budget; upstream randomly subsets molecules above it
    return_atomic   represent: include per-atom embeddings in the payload
    targets         fit: one label per SMILES
    target_name     fit: column name for the label
    model_dir       fit: where to write the model; predict: where to read it
    ml_task         fit: "regression" | "classification" | ...
    epochs, kfold, learning_rate    fit hyper-parameters
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver


def _drop_namespace_shadow() -> None:
    """Remove ``sys.path`` entries whose ``unimol_tools/`` has no ``__init__.py``.

    The Uni-Mol checkout ships ``unimol_tools/`` as a *packaging* directory
    (setup.py, MANIFEST.in, and the real package one level down). On PYTHONPATH it
    becomes a PEP 420 namespace portion with the same name as the installed
    package.
    """
    for entry in list(sys.path):
        try:
            candidate = Path(entry) / "unimol_tools"
            shadow = candidate.is_dir() and not (candidate / "__init__.py").is_file()
        except OSError:
            continue
        if shadow:
            sys.path.remove(entry)


def _prepare_environment(request: dict) -> None:
    """Point Uni-Mol at the verified weight directory and forbid downloads."""
    weight_dir = request.get("weight_dir")
    if weight_dir:
        os.environ["UNIMOL_WEIGHT_DIR"] = str(weight_dir)
    # The adapter has already asserted the checkpoints are on disk. Anything the
    # library would fetch now is a bug in that check, and it should say so rather
    # than spend the run's budget on a download.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    device = str(request.get("device") or "cpu")
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        # Uni-Mol's Trainer pins itself to cuda:0 whenever DDP is off, so a device
        # index can only be honoured by restricting what the process can see.
        if index and index != "0":
            os.environ["CUDA_VISIBLE_DEVICES"] = index
    _drop_namespace_shadow()


def _use_cuda(request: dict) -> bool:
    return not str(request.get("device") or "cpu").startswith("cpu")


def _model_params(request: dict) -> dict:
    """The ``model_name`` / ``model_size`` pair, validated against what exists."""
    model_name = str(request.get("model_name") or "unimolv1")
    if model_name not in ("unimolv1", "unimolv2"):
        raise ValueError(f"model_name must be unimolv1 or unimolv2, not {model_name!r}")
    return {"model_name": model_name, "model_size": str(request.get("model_size") or "84m")}


def _atom_symbols(raw) -> list[str]:
    """Normalise one molecule's atom labels to strings, dropping the ``[SEP]`` row.

    Uni-Mol v1 returns element symbols with a trailing ``[SEP]`` token that has an
    embedding and a coordinate but is not an atom; Uni-Mol2 returns atomic numbers
    for heavy atoms only. Both become a list of strings the same length as the
    rows that describe real atoms.
    """
    symbols = [str(s) for s in list(raw)]
    while symbols and symbols[-1] == "[SEP]":
        symbols.pop()
    return symbols


def _qc_flags(symbols: list[str], coords, n_kept: int, max_atoms: int) -> list[str]:
    """Recoverable problems with the conformer the encoder was actually given.

    Upstream ``inner_smi2coords`` never raises for a molecule ETKDG cannot embed.
    It falls back to ``Compute2DCoords`` — every z exactly 0.0 — and, if that also
    fails, to an all-zero coordinate array, announcing it with a ``print`` that a
    caller never sees. Both produce a confident 512-dimensional embedding of a
    conformer that does not exist, so both are detected here and reported as data.
    """
    import numpy as np

    flags: list[str] = []
    array = np.asarray(coords, dtype=float)[:n_kept] if n_kept else np.zeros((0, 3))
    if array.size == 0:
        flags.append("no_coordinates_returned")
        return flags
    if not np.any(array):
        flags.append("conformer_generation_failed")
    elif np.all(array[:, 2] == 0.0):
        # Compute2DCoords is the documented fallback when EmbedMolecule returns -1.
        flags.append("two_dimensional_coordinates")
    if len(symbols) >= max_atoms:
        # ConformerGen subsets with np.random.choice above max_atoms, so the encoder
        # sees a random fragment and the embedding is not reproducible.
        flags.append("atom_count_at_max_atoms_limit")
    return flags


def _represent(request: dict) -> dict:
    """Pooled and per-atom Uni-Mol embeddings for a list of SMILES."""
    import numpy as np
    from unimol_tools import UniMolRepr

    smiles = list(request.get("smiles") or [])
    if not smiles:
        raise ValueError("represent needs a non-empty 'smiles' list")
    max_atoms = int(request.get("max_atoms") or 256)
    params = _model_params(request)
    encoder = UniMolRepr(
        data_type="molecule",
        batch_size=int(request.get("batch_size") or 32),
        remove_hs=bool(request.get("remove_hs", False)),
        use_cuda=_use_cuda(request),
        max_atoms=max_atoms,
        **params,
    )
    # Atomic representations are always requested: the coordinates they come with
    # are the only evidence of whether a real 3D conformer was built. They are
    # returned to the caller only when asked for.
    output = encoder.get_repr(smiles, return_atomic_reprs=True)
    cls_repr = np.asarray(output["cls_repr"], dtype=np.float32)
    return_atomic = bool(request.get("return_atomic", False))

    molecules = []
    for index, smi in enumerate(smiles):
        symbols = _atom_symbols(output["atomic_symbol"][index])
        coords = output["atomic_coords"][index]
        atomic = np.asarray(output["atomic_reprs"][index], dtype=np.float32)
        record = {
            "index": index,
            "smiles": smi,
            "embedding": [float(v) for v in cls_repr[index]],
            "n_atoms": len(symbols),
            "qc_flags": _qc_flags(symbols, coords, len(symbols), max_atoms),
        }
        if return_atomic:
            record["atom_symbols"] = symbols
            record["atomic_embeddings"] = [[float(v) for v in row] for row in atomic[: len(symbols)]]
        molecules.append(record)

    return {
        "molecules": molecules,
        "n_input": len(smiles),
        "embedding_dim": int(cls_repr.shape[1]) if cls_repr.ndim == 2 else 0,
        "model_name": params["model_name"],
        "model_size": params["model_size"],
    }


def _fit(request: dict) -> dict:
    """Fine-tune the pretrained encoder on caller-supplied labels."""
    import joblib
    import pandas as pd
    from unimol_tools import MolTrain

    smiles = list(request.get("smiles") or [])
    targets = list(request.get("targets") or [])
    if len(smiles) != len(targets):
        raise ValueError(f"fit got {len(smiles)} SMILES and {len(targets)} targets")
    if not smiles:
        raise ValueError("fit needs a non-empty 'smiles' list")
    model_dir = request.get("model_dir")
    if not model_dir:
        raise ValueError("fit needs an absolute 'model_dir' to write the model into")
    target_name = str(request.get("target_name") or "TARGET")
    frame = pd.DataFrame({"SMILES": smiles, target_name: targets})

    trainer = MolTrain(
        task=str(request.get("ml_task") or "regression"),
        data_type="molecule",
        epochs=int(request.get("epochs") or 10),
        learning_rate=float(request.get("learning_rate") or 1e-4),
        batch_size=int(request.get("batch_size") or 16),
        kfold=int(request.get("kfold") or 5),
        save_path=str(model_dir),
        remove_hs=bool(request.get("remove_hs", False)),
        target_cols=[target_name],
        use_cuda=_use_cuda(request),
        **_model_params(request),
    )
    trainer.fit(frame)

    # MolTrain.fit returns None; the cross-validated scores it computed are joblib
    # dumped by NNModel.run into metric.result, which is the only place they exist.
    metrics_path = Path(str(model_dir)) / "metric.result"
    cv_metrics: dict[str, float] = {}
    if metrics_path.is_file():
        for key, value in dict(joblib.load(metrics_path)).items():
            try:
                cv_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return {
        "model_dir": str(model_dir),
        "ml_task": str(request.get("ml_task") or "regression"),
        "target_cols": [target_name],
        "n_train": len(smiles),
        "kfold": int(request.get("kfold") or 5),
        "epochs": int(request.get("epochs") or 10),
        "cv_metrics": cv_metrics,
        "artifacts": sorted(p.name for p in Path(str(model_dir)).iterdir()),
    }


def _predict(request: dict) -> dict:
    """Predict with a model directory ``fit`` wrote. There is no pretrained head."""
    import numpy as np
    import pandas as pd
    from unimol_tools import MolPredict

    model_dir = request.get("model_dir")
    if not model_dir:
        raise ValueError("predict needs a 'model_dir' produced by the fit task")
    if not (Path(str(model_dir)) / "config.yaml").is_file():
        raise ValueError(
            f"{model_dir}/config.yaml is missing, so this is not a Uni-Mol model directory; "
            "run the fit task first — Uni-Mol ships no pretrained property head"
        )
    smiles = list(request.get("smiles") or [])
    if not smiles:
        raise ValueError("predict needs a non-empty 'smiles' list")

    predictor = MolPredict(load_model=str(model_dir))
    raw = np.asarray(predictor.predict(pd.DataFrame({"SMILES": smiles})), dtype=float)
    matrix = raw.reshape(len(smiles), -1)
    columns = [str(c) for c in predictor.target_cols]
    if matrix.shape[1] != len(columns):
        # multiclass: one column per class, not one per target.
        columns = [f"{columns[0]}_class{i}" for i in range(matrix.shape[1])]
    return {
        "columns": columns,
        "ml_task": str(predictor.task),
        "predictions": [[float(v) for v in row] for row in matrix],
        "n_input": len(smiles),
        "model_dir": str(model_dir),
    }


_TASKS = {"represent": _represent, "fit": _fit, "predict": _predict}


def run(request: dict) -> dict:
    task = str(request.get("task") or "represent")
    handler = _TASKS.get(task)
    if handler is None:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_TASKS)}")
    _prepare_environment(request)
    payload = handler(request)
    payload["task"] = task
    return payload


if __name__ == "__main__":
    _driver.main(run)
