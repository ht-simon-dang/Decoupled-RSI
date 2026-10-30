"""FlowDock — flexible protein-ligand docking that also predicts a binding affinity.

FlowDock is a geometric flow-matching model for generative protein-ligand docking
(BioinfoMachineLearning/FlowDock, arXiv:2412.10966, ISMB 2025). It integrates a
flow from an ESMFold, harmonic or Gaussian prior to a holo complex, so the
receptor moves with the ligand rather than staying rigid, and it carries two
auxiliary heads that report on what it produced: a **confidence** head (plDDT)
and an **affinity** head.

**The two heads are two numbers and stay two numbers.** ``multi_pose_sampling``
in ``flowdock/utils/sampling_utils.py`` writes them into the filename of every
pose it saves — ``lig_rank{rank}_plddt{confidence:.7f}_affinity{pK:.7f}.sdf`` —
and this adapter reads them back separately under two documented keys:

    ``flowdock_confidence_plddt_ligand``   mean predicted plDDT over the ligand
                                           atoms. ``flowdock/models/components/
                                           flowdock.py`` builds it as
                                           ``cumsum(softmax(plddt_gram_head(...))
                                           [:, :, :4]).mean(-1)``, a mean of
                                           cumulative bin probabilities, so it
                                           lies in [0, 1]. Higher is better. This
                                           is also FlowDock's own ranking
                                           quantity when
                                           ``rank_outputs_by_confidence=true``
                                           and ``plddt_ranking_type=ligand``,
                                           which is why it, and not the affinity,
                                           is what lands in ``Pose.score``.
    ``flowdock_affinity_pk``               the affinity head's output. Its
                                           training target is PDBBind's
                                           ``-logKd/Ki`` column — see
                                           ``parse_pdbbind_binding_affinity_data_file``
                                           in ``flowdock/utils/data_utils.py``,
                                           which reads ``pK_value =
                                           float(columns[3])`` from
                                           ``data/pdbbind/pdbbind_binding_affinity_data/
                                           binding_affinity_values.txt``, whose
                                           header names that column
                                           ``-logKd/Ki``. So the unit is pK, a
                                           dimensionless −log₁₀(M).

**The affinity head is a prediction, not a measurement.** It is a neural network
output on the same scale as an experimental pK and it must never be recorded as
one. Nothing in this adapter promotes it: it is not written into ``Pose.score``,
it is not averaged with the confidence, and it never becomes a ``Prediction`` —
the specialist agents own that decision, with the trust label it deserves.
Merging the two would also be a category error in the other direction: a confident
pose and a potent ligand are different claims, and a single blended number cannot
say which one failed.

**Why this capability is UNAVAILABLE on this machine.**

1. **No weights.** ``flowdock/sample.py`` asserts ``os.path.exists(cfg.ckpt_path)``
   before it does anything else, and the checkpoint the README's inference command
   names is ``checkpoints/esmfold_prior_paper_weights-EMA.ckpt``. The clone at
   ``.external/FlowDock`` has no ``checkpoints/`` directory at all — the weights
   ship separately, in ``flowdock_checkpoints.tar.gz`` on Zenodo record 15066450.
2. **Three of the run path's imports exist in no environment here.** ``openfold``
   — the ``amorehead/openfold`` fork pinned in
   ``environments/flowdock_environment.yaml`` — is imported at
   ``flowdock/utils/data_utils.py`` line 23; ``pytorch3d`` at
   ``flowdock/models/components/flowdock.py`` line 11; ``esm`` (fair-esm) at
   ``flowdock/models/flowdock_fm_module.py`` line 3. All three are installed from
   git in the upstream environment file and all three build native extensions.
3. **The rest of the Hydra/Lightning stack is absent too.** ``hydra``,
   ``omegaconf``, ``rootutils``, ``beartype``, ``lovely_tensors``, ``mendeleev``
   and ``msgpack_numpy`` are imported on the sampling path and are not installed
   in the ``drug-agent`` environment.
4. **The pinned environment does not describe this platform.**
   ``environments/flowdock_environment.yaml`` pins
   ``pytorch=2.2.1=py3.10_cuda11.8_cudnn8.7.0_0``,
   ``pytorch-scatter=2.1.2=py310_torch_2.2.0_cu118`` and
   ``pyg=2.5.2=py310_torch_2.2.0_cu118`` — CUDA 11.8 builds for linux-64. This
   host **is** linux-64, so the channel matches; what does not match is the GPU. It
   reports compute capability 12.0 (``torch.cuda.get_device_capability(0)`` →
   ``(12, 0)``, NVIDIA RTX PRO 6000 Blackwell Workstation Edition), which CUDA 11.8
   predates and cannot emit code for. ``mamba env create -f`` therefore still does
   not give a working environment here; the work is rebuilding openfold's and
   pytorch3d's CUDA extensions against this machine's CUDA 13 toolchain for a torch
   the upstream never pinned. The move to x86-64 shortened this blocker by one
   clause and did not clear it.

None of these is a claim that FlowDock is unrunnable in principle — the code is
pure Python and the blockers are an install and a download. That is why they are
probed rather than asserted: :data:`SPEC` names the modules and the weight file,
and the moment they exist this capability turns AVAILABLE on its own.

There is also a live network dependency once it does. With the default
``prior_type: esmfold``, ``on_predict_start`` in
``flowdock/models/flowdock_fm_module.py`` calls
``esm.pretrained.load_model_and_alphabet_hub("esm2_t33_650M_UR50D")`` every run,
and — unless an ``input_template`` PDB is supplied whose sequence matches the
receptor — ``esm.pretrained.esmfold_v1()`` as well. Both come from the torch hub
cache, so an air-gapped host needs them warmed first.

**What the platform does instead: nothing.** Under
:mod:`drug_agent.tools.base` there is no surrogate docking backend and no
approximate affinity. :meth:`FlowDockAdapter.dock` raises
:class:`~drug_agent.tools.base.ToolUnavailable`, no :class:`Pose` is produced, and
the run continues with this arm visibly absent. An invented pK on FlowDock's
scale would be the worst possible fabrication in this package, because it is
numerically indistinguishable from a measured one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.structure import Pose
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import StructureError, StructureToolAdapter

#: This adapter's contract with FlowDock, not FlowDock's own version.
FLOWDOCK_ADAPTER_VERSION = "1.0.0"

#: Named output 1 — FlowDock's confidence head, the mean predicted plDDT over the
#: ligand's atoms, in [0, 1], higher better. Also lands in ``Pose.score``.
CONFIDENCE_KEY = "flowdock_confidence_plddt_ligand"

#: Named output 2 — FlowDock's affinity head, a **predicted** pK (−log₁₀ M) on
#: PDBBind's ``-logKd/Ki`` scale. Never a measurement, never blended into the score.
AFFINITY_KEY = "flowdock_affinity_pk"

#: What ``Pose.score_unit`` says, so no consumer can mistake a plDDT for an energy.
CONFIDENCE_UNIT = "plddt_ligand"

#: The checkpoint ``flowdock/sample.py`` asserts on, relative to the checkout.
CHECKPOINT_RELPATH = "checkpoints/esmfold_prior_paper_weights-EMA.ckpt"

SPEC = ExternalSpec(
    name="FlowDock",
    env_prefix="FLOWDOCK",
    checkout_relpath=".external/FlowDock",
    # Every one of these is on the sampling path: the entry point, the config it
    # composes, the LightningModule the config instantiates, and the module that
    # writes the plDDT/affinity filenames this adapter parses.
    sentinel_files=(
        "flowdock/sample.py",
        "flowdock/models/flowdock_fm_module.py",
        "flowdock/utils/sampling_utils.py",
        "configs/sample.yaml",
        "configs/model/flowdock_fm.yaml",
    ),
    driver_name="flowdock_predict.py",
    # `openfold` is the amorehead fork FlowDock pins and nothing else in this
    # repository uses, which makes it the cheapest honest answer to "is this the
    # FlowDock environment".
    selection_modules=("openfold",),
    # Read off the upstream import statements, not guessed: hydra/lightning and
    # the numeric stack that `flowdock/sample.py` pulls in transitively, then the
    # three git-installed packages a half-finished install is actually missing.
    runtime_modules=(
        "hydra",
        "omegaconf",
        "rootutils",
        "beartype",
        "lovely_tensors",
        "mendeleev",
        "msgpack_numpy",
        "torch",
        "torch_scatter",
        "torchmetrics",
        "lightning",
        "esm",
        "openfold",
        "pytorch3d",
    ),
    weight_relpaths=(CHECKPOINT_RELPATH,),
    conda_envs=("drug-agent-flowdock", "drug-agent"),
    upstream_url="https://github.com/BioinfoMachineLearning/FlowDock",
    upstream_ref="c233fd8be67f",
    license="FlowDock MIT (LICENSE, Copyright (c) 2024 BioinfoMachineLearning)",
    # Named rather than wrapped in a `make` target that does not exist: the two
    # real actions are building an environment the upstream lockfile does not
    # describe for this GPU (its CUDA 11.8 pins predate sm_120), and fetching
    # weights the clone does not carry.
    install_hint=(
        "mamba env create -f .external/FlowDock/environments/flowdock_environment.yaml, "
        "then untar flowdock_checkpoints.tar.gz from zenodo.org/records/15066450 into "
        ".external/FlowDock/checkpoints/"
    ),
    needs_gpu=True,
)


class FlowDockError(StructureError):
    """FlowDock was installed, it ran, and it did not return a usable complex."""


def named_outputs(pose: Pose) -> dict[str, float]:
    """Read FlowDock's two heads back off one pose, by name.

    The typed :class:`~drug_agent.contracts.structure.Pose` has room for exactly
    one score, so the second head travels in ``qc_flags`` as ``key=value`` —
    the same carriage :meth:`StructureToolAdapter.poses_from_sdf` already uses for
    engine properties. This is the reader for it, so downstream code asks for
    ``flowdock_affinity_pk`` by name rather than parsing strings itself.
    """
    out: dict[str, float] = {}
    for flag in pose.qc_flags:
        key, sep, raw = flag.partition("=")
        if not sep or key not in (CONFIDENCE_KEY, AFFINITY_KEY):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            out[key] = value
    return out


class FlowDockAdapter(StructureToolAdapter):
    """Flexible protein-ligand docking with separate confidence and affinity heads."""

    capability_id = "structure.dock.flowdock"
    spec = SPEC
    engine = "flowdock"
    version = FLOWDOCK_ADAPTER_VERSION
    produces_nothing = "docks nothing"

    #: The score on a returned pose is a confidence, not an energy, and higher is
    #: better — so the base class must not sort it like a docking score.
    score_unit = CONFIDENCE_UNIT
    lower_is_better = False
    # The score is the plDDT confidence head's output — a network prediction,
    # not a deterministic computation over coordinates.
    trust_label = TrustLabel.PREDICTED

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # Sampling loads ESM-2 650M and integrates a 40-step ODE per batch
        # segment, so the default budget is an hour rather than the shared 1800 s.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 3600.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Flexible protein-ligand docking (FlowDock)",
            description=(
                "Geometric flow matching from an ESMFold, harmonic or Gaussian prior to a "
                "holo protein-ligand complex, with two auxiliary heads reported separately: "
                f"a confidence plDDT in [0, 1] ({CONFIDENCE_KEY}, also the pose score) and a "
                f"predicted binding affinity in pK on PDBBind's -logKd/Ki scale ({AFFINITY_KEY}). "
                "The affinity is a model prediction, never a measurement."
            ),
            input_schema_ref="Pose",
            output_schema_ref="Pose",
            # A stochastic sampler. `sampler=VDODE` with `sampler_eta=1.0` and the
            # ODE start time are fixed by this adapter, and `seed` is forwarded to
            # `L.seed_everything(..., workers=True)`, but GPU reductions are not
            # bitwise reproducible and n_samples poses differ from one another by
            # construction.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint absent — flowdock/sample.py asserts on cfg.ckpt_path before loading",
                "receptor PDB whose parsed sequence does not match the supplied input_template, "
                "which upstream logs and then skips, producing no poses",
                "ESM-2 and ESMFold weights not in the torch hub cache on an air-gapped host",
                "GPU out of memory on a long receptor; lower chunk_size or set esmfold_chunk_size",
                "SMILES RDKit cannot parse, so FlowDock builds no ligand graph",
            ],
            validation_checks=[
                "every pose carries the engine name 'flowdock', never 'mock'",
                f"{CONFIDENCE_KEY} and {AFFINITY_KEY} are reported as separate named outputs",
                "Pose.score is the confidence plDDT and Pose.score_unit says so",
                "poses are ranked by confidence, higher first",
            ],
        )

    # --- operations --------------------------------------------------------
    def dock(
        self,
        *,
        candidate_id: str,
        smiles: str,
        receptor_pdb: str | Path,
        receptor_id: str | None = None,
        template_pdb: str | Path | None = None,
        n_samples: int = 5,
        chunk_size: int | None = None,
        num_steps: int = 40,
        sampler: str = "VDODE",
        sampler_eta: float = 1.0,
        prior_type: str = "esmfold",
        seed: int | None = None,
    ) -> list[Pose]:
        """Sample *n_samples* complexes for one ligand against one receptor.

        *smiles* is passed to FlowDock verbatim — its ``SamplingDataset`` splits
        multi-fragment ligands on ``|`` and reads the graph itself, so there is no
        need to embed a starting conformer the way a rigid docking engine wants
        one. *template_pdb*, when given, is used as the starting protein template
        and skips the ESMFold prior; its sequence must match *receptor_pdb*.
        """
        self.require_available()
        with measure(self.capability_id, "dock", backend=self.backend) as metered:
            poses = self._dock(
                candidate_id=candidate_id,
                smiles=smiles,
                receptor_pdb=receptor_pdb,
                receptor_id=receptor_id,
                template_pdb=template_pdb,
                n_samples=n_samples,
                chunk_size=chunk_size,
                num_steps=num_steps,
                sampler=sampler,
                sampler_eta=sampler_eta,
                prior_type=prior_type,
                seed=seed,
            )
            for key, value in self._ledger_metrics(poses).items():
                metered[key] = value
        return poses

    def estimate_auxiliary(
        self,
        *,
        candidate_id: str,
        ligand_sdf: str | Path,
        receptor_pdb: str | Path,
        receptor_id: str | None = None,
        template_pdb: str | Path | None = None,
        prior_type: str = "esmfold",
        seed: int | None = None,
    ) -> list[Pose]:
        """Score coordinates that already exist, instead of sampling new ones.

        This is FlowDock's ``auxiliary_estimation_only=true`` path: it keeps the
        input ligand's 3D coordinates (upstream forces ``discard_sdf_coords`` off)
        and runs only the confidence and affinity heads over them. That makes
        FlowDock usable as a judge of a pose some other engine produced, which is
        the one arrangement in which its affinity head is worth having on its own.
        *ligand_sdf* must be an SDF path — a SMILES has no coordinates to score.
        """
        self.require_available()
        path = Path(ligand_sdf)
        if path.suffix.lower() != ".sdf":
            raise FlowDockError(f"auxiliary estimation needs ligand coordinates; {path.name} is not an .sdf")
        with measure(self.capability_id, "estimate_auxiliary", backend=self.backend) as metered:
            poses = self._dock(
                candidate_id=candidate_id,
                smiles=str(path),
                receptor_pdb=receptor_pdb,
                receptor_id=receptor_id,
                template_pdb=template_pdb,
                n_samples=1,
                chunk_size=1,
                num_steps=1,
                sampler="VDODE",
                sampler_eta=1.0,
                prior_type=prior_type,
                seed=seed,
                auxiliary_estimation_only=True,
            )
            for key, value in self._ledger_metrics(poses).items():
                metered[key] = value
        return poses

    # --- real backend ------------------------------------------------------
    def _dock(
        self,
        *,
        candidate_id: str,
        smiles: str,
        receptor_pdb: str | Path,
        receptor_id: str | None,
        template_pdb: str | Path | None,
        n_samples: int,
        chunk_size: int | None,
        num_steps: int,
        sampler: str,
        sampler_eta: float,
        prior_type: str,
        seed: int | None,
        auxiliary_estimation_only: bool = False,
    ) -> list[Pose]:
        receptor = Path(receptor_pdb)
        if not receptor.is_file():
            raise FlowDockError(f"receptor PDB not found: {receptor}")
        if template_pdb is not None and not Path(template_pdb).is_file():
            raise FlowDockError(f"template PDB not found: {template_pdb}")
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() has already run
            raise FlowDockError("FlowDock checkout disappeared between probe and call")

        request: dict[str, Any] = {
            "checkout_root": str(root),
            "ckpt_path": str(root / CHECKPOINT_RELPATH),
            "input_receptor": str(receptor.resolve()),
            "input_ligand": smiles,
            "input_template": str(Path(template_pdb).resolve()) if template_pdb else None,
            "sample_id": candidate_id or "sample",
            "n_samples": int(n_samples),
            "chunk_size": int(chunk_size or n_samples),
            "num_steps": int(num_steps),
            "sampler": sampler,
            "sampler_eta": float(sampler_eta),
            "start_time": "1.0",
            "prior_type": prior_type,
            "esmfold_chunk_size": None,
            "auxiliary_estimation_only": bool(auxiliary_estimation_only),
            "device": self.ext.device,
            "seed": seed,
        }
        run = self.run_driver(request, error_cls=FlowDockError, seed=seed)
        return self._poses_from_payload(
            run.payload,
            candidate_id=candidate_id,
            receptor_id=receptor_id,
        )

    def _poses_from_payload(
        self,
        payload: dict[str, Any],
        *,
        candidate_id: str,
        receptor_id: str | None,
    ) -> list[Pose]:
        """Turn the driver's records into ranked :class:`Pose` objects.

        A record with no confidence still becomes a pose: the coordinates FlowDock
        wrote are real whether or not the auxiliary heads reported on them, and
        dropping it would hide that the model produced a structure. The absence is
        a ``qc_flag``, in the same shape as every other recoverable per-molecule
        failure in this package.
        """
        poses: list[Pose] = []
        for record in payload.get("poses") or []:
            confidence = self._finite(record.get("confidence_plddt_ligand"))
            affinity = self._finite(record.get("affinity_pk"))
            flags: list[str] = []
            if confidence is None:
                flags.append(f"missing_{CONFIDENCE_KEY}")
            else:
                flags.append(f"{CONFIDENCE_KEY}={confidence:.7f}")
            if affinity is None:
                flags.append(f"missing_{AFFINITY_KEY}")
            else:
                flags.append(f"{AFFINITY_KEY}={affinity:.7f}")
            poses.append(
                Pose(
                    candidate_id=candidate_id,
                    receptor_id=receptor_id,
                    rank=int(record.get("rank") or 0),
                    score=confidence,
                    score_unit=self.score_unit,
                    pose_ref=record.get("sdf"),
                    engine=self.engine,
                    engine_version=self.spec.upstream_ref,
                    trust_label=self.trust_label,
                    qc_flags=flags,
                )
            )
        ranked = self.rank(poses)
        # Run-level warnings ride on the top pose, after ranking, so they are on
        # the object a caller is certain to look at.
        for warning in payload.get("warnings") or []:
            if ranked:
                ranked[0].qc_flags.append(f"flowdock_warning={warning}")
        return ranked

    @staticmethod
    def _finite(value: Any) -> float | None:
        """A float, or ``None`` for anything that is not a usable number."""
        if value is None or isinstance(value, bool):
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def _ledger_metrics(poses: list[Pose]) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        Both heads are metered, separately and under their own keys. A collapse in
        confidence and a drift in predicted affinity are different failures — a
        broken receptor shows up in the first, a mis-loaded checkpoint in the
        second — and a single blended metric would hide whichever moved less.
        """
        out: dict[str, float] = {"flowdock_n_poses": float(len(poses))}
        if not poses:
            return out
        top = named_outputs(poses[0])
        if CONFIDENCE_KEY in top:
            out[CONFIDENCE_KEY] = top[CONFIDENCE_KEY]
        if AFFINITY_KEY in top:
            out[AFFINITY_KEY] = top[AFFINITY_KEY]
        scored = [pose.score for pose in poses if pose.score is not None]
        if scored:
            out["flowdock_confidence_mean"] = float(sum(scored) / len(scored))
        return out


__all__ = [
    "AFFINITY_KEY",
    "CHECKPOINT_RELPATH",
    "CONFIDENCE_KEY",
    "CONFIDENCE_UNIT",
    "FLOWDOCK_ADAPTER_VERSION",
    "SPEC",
    "FlowDockAdapter",
    "FlowDockError",
    "named_outputs",
]
