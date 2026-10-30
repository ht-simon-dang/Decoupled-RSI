"""Uni-Mol — a pretrained 3D molecular encoder, and the property models built on it.

**What it does now.** Runs the real ``unimol_tools`` package in the
``drug-agent-unimol`` conda environment against pretrained checkpoints held under
``.external/Uni-Mol/weights``. Three operations, deliberately separate:

    ``represent``   RDKit ETKDG conformer → Uni-Mol SE(3) transformer → a pooled
                    ``[CLS]`` embedding per molecule, and optionally the per-atom
                    embeddings and the coordinates the encoder was given.
    ``fit``         k-fold fine-tuning of that encoder on labels the caller
                    supplies, written to a model directory.
    ``predict``     inference with a model directory ``fit`` produced.

Re-measured on this machine on 2026-09-09 (x86-64, NVIDIA RTX PRO 6000 Blackwell
Workstation Edition, ``unimol_tools`` 0.1.6, torch 2.14.0, weights
``mol_pre_all_h_220816.pt``): aspirin, ethanol and benzene encoded with ``unimolv1``
return a ``(3, 512)`` float32 embedding with L2 norms 27.60, 27.08 and 27.12 — the
same three numbers the aarch64 host reported, to two decimals. Aspirin's atomic
block is ``(22, 512)`` upstream — 13 heavy atoms, 8 hydrogens and one ``[SEP]`` row
that is not an atom — and this adapter returns the 21 real-atom rows. Re-encoding
the same SMILES in the same process reproduces the embedding to 0.0 absolute
difference, because upstream ``inner_smi2coords`` embeds with ``randomSeed=42``. A
1-epoch, 2-fold regression fine-tune over 20 molecules took 5.5-6.3 s of driver wall
time — interpreter start, weight load, conformers and both folds — and wrote
``config.yaml``, ``cv.data``, ``metric.result``, ``model_0.pth``, ``model_1.pth``,
``target_scaler.ss`` and ``unimol_conformers_train.sdf``; predicting with that
directory took 4.3 s, and encoding three molecules 3.7-3.9 s, most of which is the
190 MB checkpoint load a driver pays on every call.

The ``unimolv2`` figures are **not** re-verified here. On the aarch64 host the same
molecules at ``model_size="84m"`` returned a ``(2, 768)`` embedding over heavy atoms
only, with no ``[SEP]`` row and atomic numbers rather than element symbols. On this
host ``represent(model_name="unimolv2")`` raises ``PropertyModelError`` naming
``.external/Uni-Mol/weights/modelzoo/84M/checkpoint.pt`` as absent, which is the
refusal working: the 84m checkpoint was never fetched here.

Observables, with keys and units:

    ``embedding``            list[float]; the ``[CLS]`` vector. 512 for
                             ``unimolv1``, 768 for ``unimolv2`` at ``84m``.
                             Dimensionless — an encoder activation, not a quantity.
    ``atomic_embeddings``    list[list[float]]; one row per atom, same width.
    ``n_atoms``              int; rows that describe real atoms.
    ``qc_flags``             str[]; see the failure modes below.
    ``cv_metrics``           fit only: upstream's own cross-validated scores, keyed
                             by its names (``mse``, ``mae``, ``pearsonr``,
                             ``spearmanr``, ``r2`` for regression), in the units of
                             the caller's labels.
    ``endpoints``            predict only: one entry per target column, in the
                             units of the labels ``fit`` was given.

**What it deliberately does not do.** It does not predict ADMET, toxicity, potency
or any other endpoint out of the box, and :meth:`predict` raises without a model
directory. Uni-Mol is a *representation* model: the released checkpoints are
pretrained on conformers with a masked-atom and coordinate-denoising objective and
carry no property head. Reading an embedding as a prediction is a category error,
which is why the operations have different names, different return types and no
shared code path — ``represent`` returns
:class:`MolecularRepresentation`, ``predict`` returns
:class:`~drug_agent.tools.properties.PropertyPanel`, and nothing converts one into
the other. It also does not rank, filter or standardise: a representation is an
input to a model, not a verdict about a molecule.

**Where the numbers can be wrong, and how that surfaces.** Upstream's conformer
step never raises. ``inner_smi2coords`` calls ``AllChem.EmbedMolecule``; when that
returns -1 in the default ``fast`` mode it falls back to ``Compute2DCoords``, and
if that also fails it substitutes an all-zero coordinate array and announces it
with a bare ``print``. Either way a 3D encoder receives coordinates that are not a
conformer and returns a confident vector. The driver therefore inspects the
coordinates it gets back and flags ``two_dimensional_coordinates`` (every z exactly
0.0) and ``conformer_generation_failed`` (every coordinate zero) as ``qc_flags`` on
that molecule. Above ``max_atoms`` — 256 by default — upstream selects a random
subset of atoms with ``np.random.choice`` off the global RNG, so the embedding is
both incomplete and irreproducible; reaching the limit is flagged as
``atom_count_at_max_atoms_limit``. SMILES RDKit cannot parse are rejected before
the driver starts, because ``inner_smi2coords`` calls ``AddHs`` on the parse result
outside its own ``try``.

**Weights.** They were not present when this adapter was written and are not
bundled with the checkout, which carries code only. They were fetched once with
upstream's own ``unimol_tools.weights.weight_download`` from the
``dptech/Uni-Mol-Models`` Hugging Face repository into ``.external/Uni-Mol/weights``:
``mol_pre_all_h_220816.pt`` (190,540,187 bytes) and ``mol.dict.txt`` (91 bytes).
Uni-Mol2's ``modelzoo/84M/checkpoint.pt`` (336,632,021 bytes) came from the
separate ``dptech/Uni-Mol2`` repository into the same tree. Those two v1 files are
this capability's ``weight_relpaths``, so a machine without them reports
UNAVAILABLE and names them rather than downloading 190 MB inside a metered run;
the Uni-Mol2 checkpoint is checked per call instead, since it is an alternative to
the default model rather than a precondition for the tool. The driver sets
``HF_HUB_OFFLINE`` before importing ``unimol_tools`` so no operation can quietly
reach the network.

**Why a driver and not an import.** ``unimol_tools`` pins numpy 2.2.6 and the
platform environment runs numpy 2.5.2, so the two cannot share an interpreter.
The backend is selected by probing ``drug-agent-unimol`` for ``unimol_tools`` and
is launched through ``scripts/external/unimol_repr.py``, which imports nothing
from this package.

**Version drift, stated plainly.** The pinned checkout is Uni-Mol at
``90f52c41299a``; the environment has ``unimol_tools`` 0.1.6, which is newer — it
adds ``pretrained_model_path``, ``max_atoms`` and an OmegaConf ``cfg`` argument to
``UniMolRepr`` that the checkout's ``predictor.py`` does not have. The driver is
written against the installed package, because that is what runs. The checkout is
kept as the reference source and as the home for the weights, and its
``sentinel_files`` are the modules the driver's behaviour was read out of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import PropertyModelError, PropertyPanel, PropertyToolAdapter

#: This adapter's contract with Uni-Mol, not Uni-Mol's own version.
UNIMOL_ADAPTER_VERSION = "1.0.0"

#: Uni-Mol2 checkpoints live in a different Hugging Face repository from the v1
#: ones and are an alternative to the default model rather than a precondition for
#: the tool, so they are checked per call instead of in ``weight_relpaths``.
UNIMOL2_WEIGHT_RELPATHS: dict[str, str] = {
    "84m": "weights/modelzoo/84M/checkpoint.pt",
    "164m": "weights/modelzoo/164M/checkpoint.pt",
    "310m": "weights/modelzoo/310M/checkpoint.pt",
    "570m": "weights/modelzoo/570M/checkpoint.pt",
    "1.1b": "weights/modelzoo/1.1B/checkpoint.pt",
}

SPEC = ExternalSpec(
    name="Uni-Mol",
    env_prefix="UNIMOL",
    checkout_relpath=".external/Uni-Mol",
    # The driver imports the *installed* unimol_tools, but these are the modules
    # its behaviour was read out of — the conformer fallbacks, the weight
    # resolution and the get_repr contract all live in these three files.
    sentinel_files=(
        "unimol_tools/unimol_tools/predictor.py",
        "unimol_tools/unimol_tools/models/unimol.py",
        "unimol_tools/unimol_tools/data/conformer.py",
    ),
    driver_name="unimol_repr.py",
    selection_modules=("unimol_tools",),
    # What a half-built environment actually loses. unimol_tools imports omegaconf
    # at module scope and joblib inside train/predict, and neither is pulled in by
    # torch or rdkit, so probing the headline dependency alone would report
    # AVAILABLE and then fail at call time.
    runtime_modules=(
        "unimol_tools",
        "torch",
        "rdkit",
        "numpy",
        "pandas",
        "joblib",
        "omegaconf",
        "huggingface_hub",
    ),
    weight_relpaths=("weights/mol_pre_all_h_220816.pt", "weights/mol.dict.txt"),
    conda_envs=("drug-agent-unimol",),
    upstream_url="https://github.com/deepmodeling/Uni-Mol",
    upstream_ref="90f52c41299a",
    license=(
        "Uni-Mol MIT (LICENSE at the checkout root); pretrained weights from "
        "dptech/Uni-Mol-Models and dptech/Uni-Mol2 on Hugging Face"
    ),
    install_hint=(
        "create the drug-agent-unimol env with unimol_tools, then fetch the weights: "
        'python -c "from unimol_tools.weights import weight_download as d; '
        "[d(f, '.external/Uni-Mol/weights') for f in "
        "('mol_pre_all_h_220816.pt', 'mol.dict.txt')]\""
    ),
    # A 15-layer, 512-wide encoder with 64 attention heads. It runs on CPU and is
    # correct there; fine-tuning it at a usable speed is a GPU job.
    needs_gpu=True,
)


@dataclass
class MolecularRepresentation:
    """One molecule's Uni-Mol embedding, and what the encoder actually saw.

    Not a :class:`~drug_agent.contracts.predictions.Prediction` and not a
    :class:`~drug_agent.tools.properties.PropertyPanel`: nothing here is a
    predicted quantity with a unit. It is an encoder activation, useful as an input
    to a model and meaningless as an output on its own.
    """

    candidate_id: str = ""
    smiles: str = ""
    #: The pooled ``[CLS]`` vector. Dimensionless.
    embedding: list[float] = field(default_factory=list)
    #: One row per real atom, same width as ``embedding``. Empty unless requested.
    atomic_embeddings: list[list[float]] = field(default_factory=list)
    #: Element symbols (``unimolv1``) or atomic numbers as strings (``unimolv2``).
    atom_symbols: list[str] = field(default_factory=list)
    n_atoms: int = 0
    model_id: str = ""
    model_version: str = "0.0.0"
    #: Which pretrained encoder produced this — ``unimolv1`` or ``unimolv2``.
    model_name: str = ""
    qc_flags: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        """Width of the embedding. 0 when nothing was encoded."""
        return len(self.embedding)

    @property
    def ok(self) -> bool:
        """Whether this vector describes a real 3D conformer of this molecule."""
        if not self.embedding:
            return False
        broken = {"unparseable_smiles", "conformer_generation_failed", "no_coordinates_returned"}
        return not broken.intersection(self.qc_flags)


@dataclass
class FineTunedModel:
    """A model directory ``fit`` wrote, and the scores upstream computed for it."""

    model_dir: str = ""
    ml_task: str = ""
    target_cols: list[str] = field(default_factory=list)
    n_train: int = 0
    kfold: int = 0
    epochs: int = 0
    #: Upstream's own cross-validated metrics, under its own names.
    cv_metrics: dict[str, float] = field(default_factory=dict)
    #: Files written into ``model_dir``, as the driver found them.
    artifacts: list[str] = field(default_factory=list)
    model_name: str = ""

    @property
    def ok(self) -> bool:
        """Whether the directory holds a model :meth:`UniMolAdapter.predict` can load."""
        return bool(self.model_dir) and "config.yaml" in self.artifacts


class UniMolAdapter(PropertyToolAdapter):
    """3D molecular representation, and property models fine-tuned on top of it."""

    capability_id = "property.represent.unimol"
    spec = SPEC
    engine = "unimol"
    version = UNIMOL_ADAPTER_VERSION
    produces_nothing = "computes no representation and predicts nothing"
    modality = "pretrained 3D molecular representation"

    #: Defaults for the encoder, overridable per call.
    default_model_name = "unimolv1"
    default_model_size = "84m"
    default_max_atoms = 256

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="3D molecular representation (Uni-Mol / Uni-Mol2)",
            description=(
                "Pretrained SE(3) transformer over an RDKit ETKDG conformer: a pooled "
                "[CLS] embedding per molecule, with per-atom embeddings on request. "
                "Fine-tuning and prediction are separate operations and need caller-supplied "
                "labels — Uni-Mol ships no pretrained property head."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="MolecularRepresentation",
            # The conformer seed is fixed at 42 upstream and the encoder is run in
            # eval mode, so the same SMILES gives the same vector — verified to 0.0
            # absolute difference. Fine-tuning is stochastic, and so is any molecule
            # above max_atoms, which upstream subsets with the global numpy RNG.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "ETKDG fails and upstream silently substitutes 2D coordinates "
                "(flagged two_dimensional_coordinates)",
                "conformer generation fails entirely and coordinates are all zero "
                "(flagged conformer_generation_failed)",
                "molecules above max_atoms are randomly subset by upstream, so the "
                "embedding is incomplete and irreproducible",
                "predict without a model directory from fit: there is no pretrained "
                "property head to fall back on",
                "Uni-Mol2 sizes other than the one whose checkpoint is on disk",
                "fine-tuning on a handful of molecules overfits; the cross-validated "
                "scores are reported unmodified so this is visible",
            ],
            validation_checks=[
                "every embedding has the same width within a batch",
                "atom count and atomic embedding rows agree",
                "unparseable SMILES are rejected before the encoder sees them",
                "coordinates are inspected for the 2D and all-zero fallbacks",
            ],
        )

    # --- operations --------------------------------------------------------
    def represent(
        self,
        *,
        candidate_ids: list[str],
        smiles_list: list[str],
        return_atomic: bool = False,
        model_name: str | None = None,
        model_size: str | None = None,
        batch_size: int = 32,
        remove_hs: bool = False,
        max_atoms: int | None = None,
    ) -> list[MolecularRepresentation]:
        """Encode molecules. Returns one representation per input, in input order.

        A molecule RDKit cannot parse gets an empty representation carrying
        ``unparseable_smiles`` rather than stopping the batch.
        """
        self.require_available()
        with measure(self.capability_id, "represent", backend=self.backend) as metered:
            reps, seconds = self._represent(
                candidate_ids=candidate_ids,
                smiles_list=smiles_list,
                return_atomic=return_atomic,
                model_name=model_name,
                model_size=model_size,
                batch_size=batch_size,
                remove_hs=remove_hs,
                max_atoms=max_atoms,
            )
            metered.update(self._represent_metrics(reps, seconds))
        return reps

    def fit(
        self,
        *,
        smiles_list: list[str],
        targets: list[float],
        model_dir: str | Path,
        ml_task: str = "regression",
        target_name: str = "TARGET",
        epochs: int = 10,
        kfold: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        model_name: str | None = None,
        model_size: str | None = None,
        remove_hs: bool = False,
    ) -> FineTunedModel:
        """Fine-tune the pretrained encoder on *targets* and write it to *model_dir*.

        This is the only way to get a prediction out of Uni-Mol. The labels are the
        caller's; the cross-validated scores come back unmodified so a model fitted
        on too little data reports that itself.
        """
        self.require_available()
        with measure(self.capability_id, "fit", backend=self.backend) as metered:
            model, seconds = self._fit(
                smiles_list=smiles_list,
                targets=targets,
                model_dir=model_dir,
                ml_task=ml_task,
                target_name=target_name,
                epochs=epochs,
                kfold=kfold,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_name=model_name,
                model_size=model_size,
                remove_hs=remove_hs,
            )
            metered["unimol_n_train"] = float(model.n_train)
            metered["unimol_kfold"] = float(model.kfold)
            metered["unimol_epochs"] = float(model.epochs)
            metered["unimol_driver_seconds"] = float(seconds)
            for key, value in model.cv_metrics.items():
                if math.isfinite(value):
                    metered[f"unimol_cv_{key}"] = float(value)
        return model

    def predict(
        self,
        *,
        model_dir: str | Path,
        candidate_ids: list[str],
        smiles_list: list[str],
    ) -> list[PropertyPanel]:
        """Predict with a model *fit* produced. Raises without one.

        Deliberately separate from :meth:`represent`. Uni-Mol's released checkpoints
        have no property head, so there is no pretrained panel this could return and
        no default endpoint it could name.
        """
        self.require_available()
        with measure(self.capability_id, "predict", backend=self.backend) as metered:
            panels, seconds = self._predict(
                model_dir=model_dir,
                candidate_ids=candidate_ids,
                smiles_list=smiles_list,
            )
            metered.update(self.coverage(panels))
            metered["unimol_driver_seconds"] = float(seconds)
        return panels

    # --- real backend ------------------------------------------------------
    def _encoder_request(
        self,
        *,
        model_name: str | None,
        model_size: str | None,
        remove_hs: bool,
        batch_size: int,
    ) -> dict[str, Any]:
        """The encoder half of a driver request, with the requested weights checked."""
        name = model_name or self.default_model_name
        size = (model_size or self.default_model_size).lower()
        if name not in ("unimolv1", "unimolv2"):
            raise PropertyModelError(f"model_name must be unimolv1 or unimolv2, not {name!r}")
        if name == "unimolv2":
            self._require_unimol2_checkpoint(size)
        return {
            "model_name": name,
            "model_size": size,
            "remove_hs": bool(remove_hs),
            "batch_size": int(batch_size),
            "weight_dir": str(self._weight_dir()),
            "device": self.ext.device,
        }

    def _weight_dir(self) -> Path:
        """Where the verified checkpoints live. ``available()`` has already run."""
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() precedes every call
            raise PropertyModelError("Uni-Mol checkout is not resolved")
        return Path(root) / "weights"

    def _require_unimol2_checkpoint(self, size: str) -> None:
        """Refuse a Uni-Mol2 size whose checkpoint is not on disk, naming the file.

        Uni-Mol2's checkpoints are an alternative to the default encoder, so they are
        not in ``weight_relpaths`` — the tool is AVAILABLE without them. Asking for
        one that is absent is a failed call, not an unavailable capability, and the
        driver would otherwise discover it after paying for process start-up.
        """
        relpath = UNIMOL2_WEIGHT_RELPATHS.get(size)
        if relpath is None:
            raise PropertyModelError(
                f"unknown Uni-Mol2 model_size {size!r}; expected one of {sorted(UNIMOL2_WEIGHT_RELPATHS)}"
            )
        root = self.ext.root
        if root is None or not (Path(root) / relpath).is_file():
            raise PropertyModelError(
                f"Uni-Mol2 {size} checkpoint not found at {Path(root or '.') / relpath}; fetch it with "
                f'`python -c "from unimol_tools.weights import weight_download_v2 as d; '
                f"d('{relpath.removeprefix('weights/')}', '.external/Uni-Mol/weights')\"`"
            )

    def _represent(
        self,
        *,
        candidate_ids: list[str],
        smiles_list: list[str],
        return_atomic: bool,
        model_name: str | None,
        model_size: str | None,
        batch_size: int,
        remove_hs: bool,
        max_atoms: int | None,
    ) -> tuple[list[MolecularRepresentation], float]:
        if len(candidate_ids) != len(smiles_list):
            raise PropertyModelError(f"got {len(candidate_ids)} candidate ids for {len(smiles_list)} SMILES")
        limit = int(max_atoms or self.default_max_atoms)
        keep, rejected = self.screen(smiles_list)
        reps = [
            self._empty_representation(cid, smi, rejected.get(idx, ""))
            for idx, (cid, smi) in enumerate(zip(candidate_ids, smiles_list, strict=True))
        ]
        if not keep:
            return reps, 0.0

        request = {
            "task": "represent",
            "smiles": keep,
            "return_atomic": bool(return_atomic),
            "max_atoms": limit,
            **self._encoder_request(
                model_name=model_name,
                model_size=model_size,
                remove_hs=remove_hs,
                batch_size=batch_size,
            ),
        }
        run = self.run_driver(request, error_cls=PropertyModelError)
        payload = run.payload
        molecules = payload.get("molecules") or []
        if len(molecules) != len(keep):
            raise PropertyModelError(
                f"Uni-Mol returned {len(molecules)} representations for {len(keep)} molecules"
            )

        # The driver was given only the molecules that survived screening, so map
        # its results back onto the original positions rather than assuming they
        # line up.
        positions = [idx for idx in range(len(smiles_list)) if idx not in rejected]
        for position, record in zip(positions, molecules, strict=True):
            reps[position] = self._representation(
                candidate_id=candidate_ids[position],
                smiles=smiles_list[position],
                record=record,
                model_name=str(payload.get("model_name") or ""),
            )
        return reps, float(run.seconds)

    def _empty_representation(self, candidate_id: str, smiles: str, reason: str) -> MolecularRepresentation:
        """A representation that encoded nothing, and says why."""
        return MolecularRepresentation(
            candidate_id=candidate_id,
            smiles=smiles,
            model_id=self.capability_id,
            model_version=self.version,
            qc_flags=[reason] if reason else [],
        )

    def _representation(
        self,
        *,
        candidate_id: str,
        smiles: str,
        record: dict[str, Any],
        model_name: str,
    ) -> MolecularRepresentation:
        embedding = [float(v) for v in record.get("embedding") or []]
        atomic = [[float(v) for v in row] for row in record.get("atomic_embeddings") or []]
        return MolecularRepresentation(
            candidate_id=candidate_id,
            smiles=smiles,
            embedding=embedding,
            atomic_embeddings=atomic,
            atom_symbols=[str(s) for s in record.get("atom_symbols") or []],
            n_atoms=int(record.get("n_atoms") or 0),
            model_id=self.capability_id,
            model_version=self.version,
            model_name=model_name,
            qc_flags=[str(f) for f in record.get("qc_flags") or []],
        )

    @staticmethod
    def _represent_metrics(reps: list[MolecularRepresentation], seconds: float) -> dict[str, float]:
        """Ledger metrics: how much of the batch got a usable vector, and how wide."""
        if not reps:
            return {"unimol_driver_seconds": float(seconds)}
        usable = [r for r in reps if r.ok]
        dims = {r.dim for r in usable}
        return {
            "unimol_n_molecules": float(len(reps)),
            "unimol_coverage": float(len(usable) / len(reps)),
            "unimol_embedding_dim": float(next(iter(dims)) if len(dims) == 1 else 0),
            "unimol_n_flagged": float(sum(1 for r in reps if r.qc_flags)),
            "unimol_driver_seconds": float(seconds),
        }

    def _fit(
        self,
        *,
        smiles_list: list[str],
        targets: list[float],
        model_dir: str | Path,
        ml_task: str,
        target_name: str,
        epochs: int,
        kfold: int,
        batch_size: int,
        learning_rate: float,
        model_name: str | None,
        model_size: str | None,
        remove_hs: bool,
    ) -> tuple[FineTunedModel, float]:
        if len(smiles_list) != len(targets):
            raise PropertyModelError(f"got {len(smiles_list)} SMILES for {len(targets)} targets")
        keep, rejected = self.screen(smiles_list)
        if not keep:
            raise PropertyModelError("no parseable SMILES in the training set")
        labels = [float(t) for idx, t in enumerate(targets) if idx not in rejected]
        # An absolute path, because the driver's working directory is the Uni-Mol
        # checkout and MolTrain writes its whole model tree into save_path.
        destination = Path(model_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        request = {
            "task": "fit",
            "smiles": keep,
            "targets": labels,
            "target_name": target_name,
            "model_dir": str(destination),
            "ml_task": ml_task,
            "epochs": int(epochs),
            "kfold": int(kfold),
            "learning_rate": float(learning_rate),
            **self._encoder_request(
                model_name=model_name,
                model_size=model_size,
                remove_hs=remove_hs,
                batch_size=batch_size,
            ),
        }
        run = self.run_driver(request, error_cls=PropertyModelError)
        payload = run.payload
        metrics = {
            str(k): float(v)
            for k, v in (payload.get("cv_metrics") or {}).items()
            if self.numeric(v) is not None
        }
        model = FineTunedModel(
            model_dir=str(payload.get("model_dir") or destination),
            ml_task=str(payload.get("ml_task") or ml_task),
            target_cols=[str(c) for c in payload.get("target_cols") or [target_name]],
            n_train=int(payload.get("n_train") or len(keep)),
            kfold=int(payload.get("kfold") or kfold),
            epochs=int(payload.get("epochs") or epochs),
            cv_metrics=metrics,
            artifacts=[str(a) for a in payload.get("artifacts") or []],
            model_name=str(request["model_name"]),
        )
        return model, float(run.seconds)

    def _predict(
        self,
        *,
        model_dir: str | Path,
        candidate_ids: list[str],
        smiles_list: list[str],
    ) -> tuple[list[PropertyPanel], float]:
        if len(candidate_ids) != len(smiles_list):
            raise PropertyModelError(f"got {len(candidate_ids)} candidate ids for {len(smiles_list)} SMILES")
        source = Path(model_dir).expanduser().resolve()
        if not (source / "config.yaml").is_file():
            raise PropertyModelError(
                f"{source} is not a Uni-Mol model directory (no config.yaml). Uni-Mol ships no "
                "pretrained property head, so predict needs a directory that fit produced."
            )
        keep, rejected = self.screen(smiles_list)
        panels = [
            self.empty_panel(cid, smi, rejected.get(idx, "no_prediction"))
            for idx, (cid, smi) in enumerate(zip(candidate_ids, smiles_list, strict=True))
        ]
        if not keep:
            return panels, 0.0

        request = {
            "task": "predict",
            "model_dir": str(source),
            "smiles": keep,
            "weight_dir": str(self._weight_dir()),
            "device": self.ext.device,
        }
        run = self.run_driver(request, error_cls=PropertyModelError)
        payload = run.payload
        columns = [str(c) for c in payload.get("columns") or []]
        rows = payload.get("predictions") or []
        if len(rows) != len(keep):
            raise PropertyModelError(f"Uni-Mol returned {len(rows)} predictions for {len(keep)} molecules")

        limitation = (
            "Fine-tuned by this run on caller-supplied labels; the applicability domain is "
            "whatever that training set covered and has not been checked here."
        )
        positions = [idx for idx in range(len(smiles_list)) if idx not in rejected]
        for position, row in zip(positions, rows, strict=True):
            endpoints = {
                name: value
                for name, raw in zip(columns, row, strict=False)
                if (value := self.numeric(raw)) is not None
            }
            panels[position] = PropertyPanel(
                candidate_id=candidate_ids[position],
                smiles=smiles_list[position],
                endpoints=endpoints,
                model_id=self.capability_id,
                model_version=self.version,
                qc_flags=[] if endpoints else ["no_finite_prediction"],
                limitation=limitation,
            )
        return panels, float(run.seconds)


__all__ = [
    "SPEC",
    "UNIMOL2_WEIGHT_RELPATHS",
    "UNIMOL_ADAPTER_VERSION",
    "FineTunedModel",
    "MolecularRepresentation",
    "UniMolAdapter",
]
