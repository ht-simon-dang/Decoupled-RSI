"""Chemprop — a directed message-passing network fitted to one endpoint's own labels.

Chemprop is not an oracle. It ships no pretrained ADMET panel and predicts nothing
until somebody trains it on labelled data for the endpoint they care about. That
single fact shapes this whole adapter, because it means the tool has **two
independent readiness gates** and collapsing them would be a lie in either
direction:

  1. *Is the library here?* — ``chemprop``, ``torch``, ``lightning``,
     ``torchmetrics``, ``astartes`` and the rest of the run path import, or they do
     not. This is what :meth:`available` and the capability card's ``status``
     answer, and it is the gate :meth:`~drug_agent.tools.base.ToolAdapter.require_available`
     enforces on every operation.
  2. *Is there a trained checkpoint for this endpoint?* — a separate question with
     a separate answer per endpoint. :meth:`train` is what makes it true;
     :meth:`checkpoint_for` reports it; :meth:`predict` raises
     :class:`ChempropNotTrained` when it is false.

An AVAILABLE card here therefore means "this machine can fit a Chemprop model",
never "this machine can predict your endpoint". A tool that reported the second
because it had verified the first would be the most expensive kind of wrong: the
number that came back would be structurally indistinguishable from a measurement.

**What it does now.**

``train(endpoint, smiles, y)``
    Fits a real :class:`chemprop.models.MPNN` — ``BondMessagePassing`` →
    ``MeanAggregation`` → ``RegressionFFN`` — end to end with the Lightning trainer,
    on the SMILES and labels supplied. Targets are z-scored with
    :meth:`chemprop.data.MoleculeDataset.normalize_targets` and the scaler is baked
    into the predictor as an ``UnscaleTransform``, so a saved checkpoint predicts in
    the endpoint's own units without the caller having to carry a scaler around.
    Saves through :func:`chemprop.models.save_model`, which writes the
    hyper-parameters, the state dict and the output column names into one ``.pt``.
    Returns a :class:`ChempropCheckpoint`:

    ``path``                     the ``.pt`` written, and a ``.json`` sidecar beside it
    ``n_train`` / ``n_val``      molecules that survived RDKit parsing, per split
    ``val_rmse`` / ``val_mae``   held-out error **in the endpoint's own units**
    ``val_r2``                   held-out coefficient of determination, dimensionless
    ``epochs``                   epochs actually run
    ``seconds``                  wall time of the fit

    The held-out numbers are computed here rather than read off the trainer.
    Upstream's ``on_validation_model_eval`` deliberately puts
    ``predictor.output_transform`` back into training mode, so the ``val/rmse``
    Lightning logs is in *standardised* units against normalised targets. Reporting
    that as an endpoint error would misstate it by a factor of the label standard
    deviation, so this adapter runs its own prediction pass over the raw held-out
    targets instead.

``predict(smiles, endpoint=... | checkpoint=...)``
    Loads a checkpoint with :func:`chemprop.models.load_model` and returns one
    :class:`~drug_agent.tools.properties.PropertyPanel` per input SMILES. Endpoint
    names come from the checkpoint's own ``output_columns``, so a value can always
    be traced to the training run that produced it. A SMILES RDKit cannot parse gets
    an empty panel carrying ``qc_flags=["unparseable_smiles"]`` — data, not an
    exception, because one bad string is a fact about that molecule and the rest of
    the batch is still valid.

``checkpoint_for`` / ``trained_endpoints``
    The second gate, readable without training anything.

**Measured on this machine.** x86-64, chemprop 2.2.4, torch 2.14.0+cu130, lightning
2.6.5, CPU accelerator. 40 SMILES from ``fixtures/data/compound_library.csv``,
labelled with RDKit ``Crippen.MolLogP`` — a synthetic label, but an exactly
checkable one, so the fit either learns it or visibly does not. Label mean 2.189,
standard deviation 1.464. Every fifth molecule (8 of 40) was held out entirely; the
remaining 32 went to ``train(val_fraction=0.25, epochs=60, batch_size=16,
hidden_dim=128, depth=3, ffn_hidden_dim=128, seed=1729)``, which split them again
into 24 fit / 8 internal validation.

  * fit: 60 epochs, 1.1-2.9 s over three runs here, CPU; internal validation RMSE
    1.031, MAE 0.807, R² 0.707, all in log-P units and identical across every run —
    and identical, digit for digit, to what the aarch64 host reported, which is the
    point of a seeded fit on a deterministic label. Only the seconds moved (3.9 s and
    6.1 s there);
  * checkpoint written: 290,649 bytes plus a ``demo_logp.json`` sidecar;
  * reloaded through :func:`chemprop.models.load_model` and predicted the 8 truly
    held-out molecules: RMSE 0.561, MAE 0.401, R² 0.779, against a held-out label
    standard deviation of 1.192;
  * predicting twice from the reloaded checkpoint agreed to 0.0 absolute difference;
  * ``predict`` before any ``train`` raised :class:`ChempropNotTrained`, and
    ``"not-a-smiles"`` came back as an empty panel with
    ``qc_flags=["unparseable_smiles"]`` rather than a number.

**Compare it against the cheap baseline before believing it earned its cost.** A
message-passing network is not automatically better than a Morgan fingerprint into
a RandomForest, and at the dataset sizes typical of a real project — hundreds of
compounds, not hundreds of thousands — it frequently is not. This repository
already ships that baseline as
:class:`~drug_agent.tools.sklearn_tool.BaselineActivityModel`, which fits in
milliseconds on CPU and reports both an ensemble-std uncertainty and a
nearest-neighbour applicability domain, neither of which this adapter provides.
Fitted on the same 32 molecules and scored on the same 8, it reached RMSE 0.941,
MAE 0.752, R² 0.377 — worse than the network here. That decides nothing: eight
molecules of a synthetic label is a demonstration that the comparison runs, not
evidence about any endpoint. Run it on the real endpoint, on the real split, and
keep the network only if it wins there.

**What it deliberately does not do.**

  * *No uncertainty.* ``RegressionFFN`` emits a point estimate. Chemprop ships
    ``MveFFN``, ``EvidentialFFN`` and the ``chemprop.uncertainty`` calibrators;
    none is wired here, so no panel carries an interval and none should be invented
    downstream.
  * *No applicability domain.* Nothing checks whether a molecule resembles the
    training set. The network will return a confident float for anything RDKit can
    parse. The panel's ``limitation`` says so, and setting applicability is the
    specialist agent's job.
  * *No pretrained weights.* There is no shipped checkpoint for any endpoint, and
    this adapter downloads none. Before :meth:`train` has run, :meth:`predict`
    raises.
  * *No hyper-parameter search, no ensembling, no transfer learning.* Upstream's
    ``chemprop hpopt`` and its foundation-model fine-tuning path are not exposed.
  * *No classification, multi-component or atom/bond targets.* Single-task
    regression over single molecules only. Chemprop supports all of those; this
    adapter does not claim to.

**Why chemprop 2.2.4 and not 2.3.x.** Verified on this machine, not assumed, and
**re-verified on x86-64** because the first version of this paragraph read as an
architecture story and is not one. chemprop 2.3.0 declares ``Requires-Dist:
cuik_molmaker_pin`` and, unlike 2.2.4, imports it *unconditionally* at module scope
— ``chemprop/featurizers/molgraph/molecule.py`` line 4 is a bare ``import
cuik_molmaker``, where 2.2.4 has the same import guarded inside a ``try`` at line 16.

``cuik_molmaker`` is **not** installed in this environment, and PyPI does publish a
wheel for this host: ``pip download cuik-molmaker-pin --only-binary=:all:`` here
takes ``cuik_molmaker_pin-2026.3.5-py312-none-manylinux_2_24_x86_64.whl`` (328 KB,
2026-09-09). Unpacking it and running ``readelf -d`` on the
``cuik_molmaker_cpp.cpython-312-x86_64-linux-gnu.so`` it carries shows the same 79
auditwheel-mangled RDKit sonames the aarch64 wheel had — of the form
``libRDKit<Lib>-<hash>.so.1``, e.g. ``libRDKitAbbreviations-addd88ce.so.1``, with a
per-architecture hash. Those names exist only inside the *pip* RDKit wheel the
extension was built and repaired against. This environment's RDKit is the
conda-forge build 2026.03.6, which provides the same libraries under their unmangled
names (``libRDKitAbbreviations.so.1``), so the dynamic loader would find nothing and
``import cuik_molmaker`` would raise ``ImportError:
libRDKitAbbreviations-<hash>.so.1: cannot open shared object file``, exactly as it
did on aarch64. Under 2.2.4 that failure is contained — the ``try`` swallows it and
the pure-Python ``SimpleMoleculeMolGraphFeaturizer`` is used, which is the featurizer
this adapter runs. Under 2.3.x it would make ``import chemprop`` fail outright.
Pinning 2.2.4 is therefore a constraint of this environment's *RDKit provenance* —
conda-forge rather than pip — and holds on both architectures.

Chemprop is pure Python over torch, Lightning and RDKit, so like PoseBusters it
needs no checkout, no weights and no separate interpreter: it imports in the
``drug-agent`` environment or it does not. When it does not, this capability is
UNAVAILABLE and predicts nothing. There is no surrogate — under
:mod:`drug_agent.tools.base` a fabricated property value is exactly the failure this
package exists to refuse.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import PropertyModelError, PropertyPanel, PropertyToolAdapter

#: This adapter's contract with Chemprop, not Chemprop's own version.
CHEMPROP_ADAPTER_VERSION = "1.0.0"

#: Suffix of the file :func:`chemprop.models.save_model` writes.
CHECKPOINT_SUFFIX = ".pt"
#: Suffix of the provenance sidecar written beside it. Plain JSON on purpose: what
#: a checkpoint was fitted on has to stay readable without torch.
SIDECAR_SUFFIX = ".json"

SPEC = ExternalSpec(
    name="Chemprop",
    env_prefix="CHEMPROP",
    selection_modules=("chemprop",),
    # Everything the fit/predict path imports at module scope, read out of the
    # installed tree rather than guessed. `chemprop.data.__init__` re-exports
    # `splitting`, which imports `astartes`; `datasets` imports `multiprocess`
    # (the dill fork, not stdlib `multiprocessing`) and `psutil`; `nn.metrics`
    # imports `torchmetrics`; `nn.transforms` imports `sklearn`. Each of those is a
    # separate distribution that a half-finished install really does lose, and
    # probing only `chemprop` would report AVAILABLE and then fail at call time.
    runtime_modules=(
        "chemprop",
        "torch",
        "lightning",
        "torchmetrics",
        "astartes",
        "multiprocess",
        "psutil",
        "packaging",
        "rdkit",
        "sklearn",
        "numpy",
    ),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/chemprop/chemprop",
    upstream_ref="v2.2.4",
    license="Chemprop MIT",
    install_hint='conda run -n drug-agent pip install "chemprop==2.2.4"',
    # It trains and predicts on CPU. A GPU is an accelerator, not a requirement,
    # so claiming one would make the card ask for hardware the tool does not need.
    needs_gpu=False,
)


class ChempropTrainingError(PropertyModelError):
    """Chemprop was installed, the fit started, and it did not produce a model."""


class ChempropNotTrained(PropertyModelError):
    """Prediction was asked for an endpoint that has no trained checkpoint.

    Deliberately *not* :class:`~drug_agent.tools.base.ToolUnavailable`. The backend
    is present and working — ``train`` would run right now. What is missing is
    labelled data for this endpoint, which no install fixes, so conflating the two
    would send an operator to the wrong remedy. What it shares with
    ``ToolUnavailable`` is the part that matters: it raises rather than returning a
    default, because a placeholder property value is indistinguishable from a
    prediction three artifacts downstream.
    """


@dataclass
class ChempropCheckpoint:
    """One trained model: where it lives, what it was fitted on, how well it held up.

    ``val_*`` are in the endpoint's own units and are ``None`` when the fit used no
    held-out split — an honest absence rather than a zero that would read as a
    perfect model.
    """

    path: Path
    endpoint: str
    n_train: int = 0
    n_val: int = 0
    val_rmse: float | None = None
    val_mae: float | None = None
    val_r2: float | None = None
    epochs: int = 0
    seconds: float = 0.0
    device: str = "cpu"
    chemprop_version: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(SIDECAR_SUFFIX)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "endpoint": self.endpoint,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "val_rmse": self.val_rmse,
            "val_mae": self.val_mae,
            "val_r2": self.val_r2,
            "epochs": self.epochs,
            "seconds": round(self.seconds, 3),
            "device": self.device,
            "chemprop_version": self.chemprop_version,
            "adapter_version": CHEMPROP_ADAPTER_VERSION,
            "hyperparameters": self.hyperparameters,
            "qc_flags": self.qc_flags,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats."""
        out: dict[str, float] = {
            "chemprop_n_train": float(self.n_train),
            "chemprop_n_val": float(self.n_val),
            "chemprop_epochs": float(self.epochs),
            "chemprop_fit_seconds": float(self.seconds),
        }
        for key in ("val_rmse", "val_mae", "val_r2"):
            value = getattr(self, key)
            if value is not None and math.isfinite(value):
                out[f"chemprop_{key}"] = float(value)
        return out


class ChempropAdapter(PropertyToolAdapter):
    """Task-specific molecular property learning: fit an MPNN, then predict with it."""

    capability_id = "property.predict.chemprop"
    spec = SPEC
    engine = "chemprop"
    version = CHEMPROP_ADAPTER_VERSION
    modality = "task-specific regression, trained on the caller's own labels"

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s)
        # Path arithmetic only. No mkdir, no stat, no glob: `tool_pool()` builds
        # every adapter on every call, and the directory is created by `train`,
        # which is the only operation that has a reason to.
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            home = getattr(settings, "home", None) if settings is not None else None
            self.checkpoint_dir = Path(home or ".drugagent") / "chemprop"

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Task-specific property model (Chemprop D-MPNN)",
            description=(
                "Fits a directed message-passing neural network to the caller's own labelled "
                "SMILES and predicts that endpoint for new molecules. Ships no pretrained "
                "weights and no endpoint panel: an AVAILABLE card means this machine can "
                "train a model, not that any endpoint has one. Point estimates only — no "
                "uncertainty and no applicability domain."
            ),
            input_schema_ref="list[str]",
            output_schema_ref="PropertyPanel",
            # Weight initialisation, dropout and shuffling are all stochastic. The
            # seed is honoured, but two fits are not bit-identical across devices.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no trained checkpoint for the requested endpoint — predict() raises "
                "ChempropNotTrained; this is the normal state before train() has run",
                "too few labelled molecules to fit anything that generalises; the held-out "
                "error is reported and is usually the tell",
                "no uncertainty and no applicability domain: a molecule unlike the training "
                "set still gets a confident number",
                "SMILES RDKit cannot parse are screened out and reported as qc_flags",
                "a fit that outruns its time budget stops at the epoch boundary and the "
                "checkpoint is whatever the last epoch produced",
                "checkpoints are torch pickles: a chemprop or torch upgrade can make an old one unloadable",
            ],
            validation_checks=[
                "predict() with no checkpoint raises rather than returning a value",
                "endpoint names are read from the checkpoint's own output_columns",
                "held-out error is computed in the endpoint's own units, not standardised",
                "unparseable SMILES yield an empty panel with a qc_flag, never a number",
            ],
        )

    # --- the second gate ---------------------------------------------------
    @staticmethod
    def slug(endpoint: str) -> str:
        """Filesystem-safe stem for an endpoint name."""
        return re.sub(r"[^0-9A-Za-z._-]+", "_", endpoint).strip("_") or "endpoint"

    def checkpoint_for(self, endpoint: str) -> Path | None:
        """The trained checkpoint for *endpoint*, or ``None`` if there is not one."""
        path = self.checkpoint_dir / f"{self.slug(endpoint)}{CHECKPOINT_SUFFIX}"
        return path if path.is_file() else None

    def trained_endpoints(self) -> list[str]:
        """Endpoints this machine has a checkpoint for. Empty is the usual answer."""
        if not self.checkpoint_dir.is_dir():
            return []
        return sorted(p.stem for p in self.checkpoint_dir.glob(f"*{CHECKPOINT_SUFFIX}"))

    # --- operations --------------------------------------------------------
    def train(
        self,
        *,
        endpoint: str,
        smiles: list[str],
        y: list[float],
        epochs: int = 50,
        val_fraction: float = 0.2,
        batch_size: int = 32,
        hidden_dim: int = 300,
        depth: int = 3,
        ffn_hidden_dim: int = 300,
        dropout: float = 0.0,
        seed: int = 1729,
        output_path: str | Path | None = None,
    ) -> ChempropCheckpoint:
        """Fit an MPNN on *smiles*/*y* and write a checkpoint for *endpoint*."""
        self.require_available()
        with measure(self.capability_id, "train", backend=self.backend) as metered:
            ckpt = self._train(
                endpoint=endpoint,
                smiles=smiles,
                y=y,
                epochs=epochs,
                val_fraction=val_fraction,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                depth=depth,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                seed=seed,
                output_path=output_path,
            )
            for key, value in ckpt.metrics_for_ledger().items():
                metered[key] = value
        return ckpt

    def predict(
        self,
        smiles_list: list[str],
        *,
        endpoint: str | None = None,
        checkpoint: str | Path | None = None,
        candidate_ids: list[str] | None = None,
        batch_size: int = 64,
    ) -> list[PropertyPanel]:
        """Predict with a trained checkpoint. Raises if there is not one.

        Give either *endpoint* — looked up under :attr:`checkpoint_dir` — or an
        explicit *checkpoint* path.
        """
        self.require_available()
        with measure(self.capability_id, "predict", backend=self.backend) as metered:
            panels = self._predict(
                smiles_list=smiles_list,
                endpoint=endpoint,
                checkpoint=checkpoint,
                candidate_ids=candidate_ids,
                batch_size=batch_size,
            )
            for key, value in self.coverage(panels).items():
                metered[key] = value
        return panels

    # --- real backend ------------------------------------------------------
    def _accelerator(self) -> tuple[str, Any, str]:
        """Lightning ``(accelerator, devices)`` plus the device string for the record."""
        import torch

        device = str(self.ext.device or "cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            _, _, index = device.partition(":")
            return "gpu", [int(index or 0)], device
        return "cpu", 1, "cpu"

    def _trainer(self, *, epochs: int, seed: int) -> Any:
        """A Lightning trainer wired the way this adapter needs it.

        ``logger``/``enable_checkpointing`` are off because this adapter owns where
        the model is written; ``max_time`` is how the capability's timeout is
        actually enforced, since the fit runs in this process and there is no
        subprocess for :class:`~drug_agent.tools._ext.backend.ExternalBackend` to
        reap.
        """
        from datetime import timedelta

        from lightning import pytorch as pl

        accelerator, devices, _ = self._accelerator()
        return pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=epochs,
            max_time=timedelta(seconds=self.timeout_s),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )

    def _screened_pairs(self, smiles: list[str], y: list[float]) -> tuple[list[str], list[float], list[str]]:
        """Drop the rows RDKit cannot parse, keeping SMILES and labels aligned."""
        if len(smiles) != len(y):
            raise ChempropTrainingError(f"{len(smiles)} SMILES but {len(y)} labels; they must pair up")
        _, rejected = self.screen(smiles)
        keep_smiles: list[str] = []
        keep_y: list[float] = []
        flags: list[str] = []
        for idx, (smi, target) in enumerate(zip(smiles, y, strict=True)):
            value = self.numeric(target)
            if idx in rejected:
                flags.append(f"dropped_unparseable_smiles[{idx}]")
            elif value is None:
                flags.append(f"dropped_non_finite_label[{idx}]")
            else:
                keep_smiles.append(smi)
                keep_y.append(value)
        return keep_smiles, keep_y, flags

    def _train(
        self,
        *,
        endpoint: str,
        smiles: list[str],
        y: list[float],
        epochs: int,
        val_fraction: float,
        batch_size: int,
        hidden_dim: int,
        depth: int,
        ffn_hidden_dim: int,
        dropout: float,
        seed: int,
        output_path: str | Path | None,
    ) -> ChempropCheckpoint:
        # Imported here rather than at module scope: `tool_pool()` constructs every
        # adapter on every call, and importing torch + lightning in a constructor is
        # what turns a capability listing into a multi-second operation.
        import numpy as np
        from chemprop import data as cpdata
        from chemprop import models as cpmodels
        from chemprop import nn as cpnn
        from lightning import pytorch as pl

        keep_smiles, keep_y, flags = self._screened_pairs(smiles, y)
        if len(keep_smiles) < 4:
            raise ChempropTrainingError(
                f"only {len(keep_smiles)} usable molecules for endpoint {endpoint!r}; "
                "a message-passing network needs at least 4 to fit at all, and far more to "
                "generalise — compare against the Morgan/RandomForest baseline in "
                "tools/sklearn_tool.py before training one on data this small"
            )

        pl.seed_everything(seed, workers=True)
        points = [
            cpdata.MoleculeDatapoint.from_smi(smi, np.array([target], dtype=float))
            for smi, target in zip(keep_smiles, keep_y, strict=True)
        ]

        val_points: list[Any] = []
        want_val = val_fraction > 0.0 and round(len(points) * val_fraction) >= 1
        if want_val:
            train_idx, val_idx, _ = cpdata.make_split_indices(
                points, "random", (1.0 - val_fraction, val_fraction, 0.0), seed=seed
            )
            train_split, val_split, _ = cpdata.split_data_by_indices(points, train_idx, val_idx, None)
            train_points = list(train_split[0])
            val_points = list(val_split[0])
        else:
            train_points = list(points)
            if val_fraction > 0.0:
                flags.append("val_split_too_small_to_hold_out")

        train_ds = cpdata.MoleculeDataset(train_points)
        scaler = train_ds.normalize_targets()
        # The scaler rides inside the predictor, so the checkpoint predicts in the
        # endpoint's own units and no caller has to remember to unscale.
        output_transform = cpnn.UnscaleTransform.from_standard_scaler(scaler)

        message_passing = cpnn.BondMessagePassing(d_h=hidden_dim, depth=depth, dropout=dropout)
        predictor = cpnn.RegressionFFN(
            n_tasks=1,
            input_dim=message_passing.output_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            output_transform=output_transform,
        )
        model = cpmodels.MPNN(
            message_passing,
            cpnn.MeanAggregation(),
            predictor,
            batch_norm=False,
            metrics=[cpnn.RMSE(), cpnn.MAE()],
        )

        # drop_last=False explicitly: `build_dataloader` would otherwise silently
        # discard a trailing batch of one, and with batch_norm off there is no
        # reason to throw a labelled molecule away.
        train_loader = cpdata.build_dataloader(
            train_ds, batch_size=batch_size, num_workers=0, shuffle=True, seed=seed, drop_last=False
        )
        val_loader = None
        if val_points:
            val_ds = cpdata.MoleculeDataset(list(val_points))
            val_ds.normalize_targets(scaler)
            val_loader = cpdata.build_dataloader(
                val_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False
            )

        trainer = self._trainer(epochs=epochs, seed=seed)
        started = time.perf_counter()
        try:
            trainer.fit(model, train_loader, val_loader)
        # Broad on purpose: lightning and chemprop raise bare exceptions, and a failed
        # fit is 'the backend ran and failed', which is this module's own error type.
        except Exception as exc:
            raise ChempropTrainingError(f"Chemprop fit failed for endpoint {endpoint!r}: {exc}") from exc
        seconds = time.perf_counter() - started

        path = (
            Path(output_path)
            if output_path
            else (self.checkpoint_dir / f"{self.slug(endpoint)}{CHECKPOINT_SUFFIX}")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        cpmodels.save_model(path, model, output_columns=[endpoint])

        _, _, device = self._accelerator()
        ckpt = ChempropCheckpoint(
            path=path,
            endpoint=endpoint,
            n_train=len(train_points),
            n_val=len(val_points),
            epochs=int(trainer.current_epoch),
            seconds=seconds,
            device=device,
            chemprop_version=self._chemprop_version(),
            hyperparameters={
                "hidden_dim": hidden_dim,
                "depth": depth,
                "ffn_hidden_dim": ffn_hidden_dim,
                "dropout": dropout,
                "batch_size": batch_size,
                "max_epochs": epochs,
                "val_fraction": val_fraction,
                "seed": seed,
            },
            qc_flags=flags,
        )
        if val_points:
            truth = np.asarray([float(p.y[0]) for p in val_points], dtype=float)
            preds = self._forward(model, [p.mol for p in val_points], batch_size=batch_size)
            self._fill_val_metrics(ckpt, truth, preds[:, 0])
        ckpt.sidecar_path.write_text(json.dumps(ckpt.to_dict(), indent=2))
        return ckpt

    @staticmethod
    def _fill_val_metrics(ckpt: ChempropCheckpoint, truth: Any, pred: Any) -> None:
        """Held-out error in the endpoint's own units.

        Not read off the trainer: upstream puts ``output_transform`` back into
        training mode during validation, so its ``val/rmse`` is standardised.
        """
        import numpy as np

        residual = np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)
        ckpt.val_rmse = float(np.sqrt(np.mean(residual**2)))
        ckpt.val_mae = float(np.mean(np.abs(residual)))
        variance = float(np.var(truth))
        # R² is undefined when every held-out label is identical; say so rather than
        # dividing by zero and shipping an infinity.
        ckpt.val_r2 = float(1.0 - np.mean(residual**2) / variance) if variance > 0 else None
        if ckpt.val_r2 is None:
            ckpt.qc_flags.append("val_r2_undefined_zero_label_variance")

    def _forward(self, model: Any, mols: list[Any], *, batch_size: int) -> Any:
        """Run a loaded model over RDKit mols and return an ``(n, n_tasks)`` array."""
        import torch
        from chemprop import data as cpdata
        from chemprop.conf import LIGHTNING_26_COMPAT_ARGS

        dataset = cpdata.MoleculeDataset([cpdata.MoleculeDatapoint(mol) for mol in mols])
        loader = cpdata.build_dataloader(
            dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False
        )
        accelerator, devices, _ = self._accelerator()
        from lightning import pytorch as pl

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        # LIGHTNING_26_COMPAT_ARGS is chemprop's own shim for lightning >= 2.6, which
        # is the version installed here; upstream's CLI passes it the same way.
        batches: Any = trainer.predict(model, loader, **LIGHTNING_26_COMPAT_ARGS)
        if not batches:
            raise ChempropTrainingError("Chemprop returned no predictions for a non-empty batch")
        return torch.cat([b.detach().cpu() for b in batches]).numpy()

    def _predict(
        self,
        *,
        smiles_list: list[str],
        endpoint: str | None,
        checkpoint: str | Path | None,
        candidate_ids: list[str] | None,
        batch_size: int,
    ) -> list[PropertyPanel]:
        from chemprop import models as cpmodels
        from chemprop.models.utils import load_output_columns
        from rdkit import Chem

        path = self._resolve_checkpoint(endpoint=endpoint, checkpoint=checkpoint)
        ids = candidate_ids or [""] * len(smiles_list)
        if len(ids) != len(smiles_list):
            raise PropertyModelError(f"{len(smiles_list)} SMILES but {len(ids)} candidate ids")
        _, rejected = self.screen(smiles_list)
        keep_idx = [i for i in range(len(smiles_list)) if i not in rejected]
        panels: list[PropertyPanel] = [
            self.empty_panel(ids[i], smiles_list[i], rejected.get(i, "unparseable_smiles"))
            for i in range(len(smiles_list))
        ]
        if not keep_idx:
            return panels

        try:
            model = cpmodels.load_model(path)
            columns = load_output_columns(path)
        # torch.load over a pickle raises almost anything when the file is stale,
        # truncated, or written by an incompatible chemprop.
        except Exception as exc:
            raise ChempropTrainingError(f"could not load Chemprop checkpoint {path}: {exc}") from exc
        model.eval()

        names = self._endpoint_names(columns, n_tasks=int(model.n_tasks), fallback=endpoint)
        limitation = self._limitation(path, names)
        values = self._forward(
            model, [Chem.MolFromSmiles(smiles_list[i]) for i in keep_idx], batch_size=batch_size
        )
        for row, idx in enumerate(keep_idx):
            endpoints: dict[str, float] = {}
            flags: list[str] = []
            for col, name in enumerate(names):
                value = self.numeric(values[row][col])
                if value is None:
                    flags.append(f"non_finite_prediction[{name}]")
                else:
                    endpoints[name] = value
            panels[idx] = PropertyPanel(
                candidate_id=ids[idx],
                smiles=smiles_list[idx],
                endpoints=endpoints,
                model_id=self.capability_id,
                model_version=self.version,
                qc_flags=flags,
                limitation=limitation,
            )
        return panels

    def _resolve_checkpoint(self, *, endpoint: str | None, checkpoint: str | Path | None) -> Path:
        """The checkpoint to predict with, or a loud refusal naming what is missing."""
        if checkpoint is not None:
            path = Path(checkpoint)
            if not path.is_file():
                raise ChempropNotTrained(
                    f"no Chemprop checkpoint at {path}. Chemprop predicts nothing until it is "
                    f"trained; call train(endpoint=..., smiles=..., y=...) with labelled data."
                )
            return path
        if not endpoint:
            raise ChempropNotTrained("predict() needs either an endpoint name or a checkpoint path")
        found = self.checkpoint_for(endpoint)
        if found is None:
            trained = ", ".join(self.trained_endpoints()) or "none"
            raise ChempropNotTrained(
                f"no trained Chemprop checkpoint for endpoint {endpoint!r}: expected "
                f"{self.checkpoint_dir / f'{self.slug(endpoint)}{CHECKPOINT_SUFFIX}'}. "
                f"Trained endpoints here: {trained}. Chemprop ships no pretrained weights, so "
                f"until train() has run on labelled data for this endpoint there is nothing to "
                f"predict with — and there is no surrogate value."
            )
        return found

    @staticmethod
    def _endpoint_names(columns: Any, *, n_tasks: int, fallback: str | None) -> list[str]:
        """Task names from the checkpoint's own ``output_columns`` where it has them."""
        if isinstance(columns, list) and len(columns) == n_tasks and all(isinstance(c, str) for c in columns):
            return [str(c) for c in columns]
        base = fallback or "chemprop_task"
        return [base] if n_tasks == 1 else [f"{base}_{i}" for i in range(n_tasks)]

    def _limitation(self, path: Path, names: list[str]) -> str:
        """What the caller must know before treating these numbers as evidence."""
        sidecar = path.with_suffix(SIDECAR_SUFFIX)
        provenance = ""
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            meta = None
        if isinstance(meta, dict):
            rmse = meta.get("val_rmse")
            provenance = f" Fitted on {meta.get('n_train', '?')} molecules" + (
                f"; held-out RMSE {float(rmse):.3f} in the endpoint's own units." if rmse else "."
            )
        return (
            f"Chemprop point estimate for {', '.join(names)} from checkpoint {path.name}."
            f"{provenance} No uncertainty and no applicability domain: nothing here checks "
            f"whether this molecule resembles the training set, and a message-passing network "
            f"should be compared against the Morgan-fingerprint baseline in tools/sklearn_tool.py "
            f"before it is trusted over one."
        )

    @staticmethod
    def _chemprop_version() -> str:
        import chemprop

        return str(getattr(chemprop, "__version__", "unknown"))


__all__ = [
    "CHECKPOINT_SUFFIX",
    "CHEMPROP_ADAPTER_VERSION",
    "SIDECAR_SUFFIX",
    "SPEC",
    "ChempropAdapter",
    "ChempropCheckpoint",
    "ChempropNotTrained",
    "ChempropTrainingError",
]
