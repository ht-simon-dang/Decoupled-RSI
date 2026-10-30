"""Boltz-2 — predict the protein-ligand complex, then score the ligand against it.

Boltz-2 is a diffusion model over biomolecular structure with a second head trained
on binding data. One invocation therefore answers two different questions, and this
adapter keeps them apart because they are measured on different scales and mean
different things.

**What it computes.** From a protein sequence and a ligand (SMILES or CCD code) it
returns the predicted complex as ``list[Pose]``, one per diffusion sample, plus the
affinity head's output as a separate field. The observables, with upstream's own
keys and units:

    ``Pose.score``                      ``confidence_score``, 0-1, higher is better
                                        (upstream: ``0.8 * complex_plddt + 0.2 *
                                        iptm``, ``ptm`` for single chains)
    ``Pose.pose_ref``                   the predicted complex, mmCIF or PDB text
    ``BoltzComplexPrediction.confidence``
                                        ``ptm``, ``iptm``, ``ligand_iptm``,
                                        ``protein_iptm``, ``complex_plddt``,
                                        ``complex_iplddt`` (all 0-1, higher better),
                                        ``complex_pde``, ``complex_ipde`` (Å, lower
                                        better)
    ``affinity.pred_value``             ``log10(IC50)`` for an IC50 in µM — lower is
                                        a stronger predicted binder
    ``affinity.probability_binary``     P(this ligand is a binder), 0-1
    ``affinity.pred_value_1/_2``,       the two ensemble members behind the two
    ``affinity.probability_binary_1/_2``  fields above

**Why the affinity is not the pose score.** ``Pose.score`` carries the structural
confidence and nothing else. A Boltz-2 affinity is not a docking energy: it is a
learned ``log10(IC50/µM)``, it has no sign convention in common with a GNINA or Vina
kcal/mol, and upstream is explicit that its two affinity outputs answer different
questions — ``probability_binary`` separates binders from decoys at the hit stage,
``pred_value`` compares *actives* during optimisation and is not meaningful on
inactives (upstream ``docs/prediction.md``). Writing either number into
``Pose.score`` would make it sortable against docking scores three artifacts
downstream, where nothing could tell them apart. It lives in ``.affinity``.

**What blocks it here.** The code and the environment are ready; the weights are
not. Concretely, one blocker:

  1. The Boltz-2 checkpoint cache is empty. ``boltz.main.get_cache_path()`` returns
     ``~/.boltz`` by default (when ``BOLTZ_CACHE`` is unset) and that
     directory does not exist, so all three entries ``boltz.main.download_boltz2``
     requires are absent: ``boltz2_conf.ckpt``, ``boltz2_aff.ckpt`` and ``mols/``.
     Populating it is a 6.20 GB download from ``model-gateway.boltz.bio``
     (``huggingface.co/boltz-community`` is upstream's fallback): 2.29 GB for
     ``boltz2_conf.ckpt``, 2.06 GB for ``boltz2_aff.ckpt`` and 1.86 GB for
     ``mols.tar``, which is then extracted to ``mols/`` beside it — sizes read from
     the HTTP ``content-length`` of those URLs on 2026-09-07. Nothing in this
     package downloads them implicitly: ``boltz.main.predict`` would fetch them on
     its first call, so the driver checks the cache first and refuses instead.

Everything else was verified present: conda env ``drug-agent-boltz`` imports
``boltz.main`` cleanly at boltz 2.2.1 under Python 3.11.16, with torch 2.14.0+cu130
reporting ``cuda.is_available() == True`` on an NVIDIA GB10, numpy 1.26.4,
pytorch_lightning 2.5.0, rdkit 2026.03.6 and numba 0.61.0. The installed 2.2.1
matches the pinned checkout ``.external/boltz`` at ``b1ebfc46ecf5``, which is what
the driver was written against.

**Two further requirements a populated cache does not remove.** Neither is a
blocker today, and both decide whether a call can succeed:

  * *An MSA.* ``boltz predict`` refuses a protein chain with no MSA unless
    ``--use_msa_server`` is set, which posts the sequence to
    ``https://api.colabfold.com``. This adapter never reaches the network by
    default: pass ``msa=<path to .a3m/.csv>``, or ``msa="empty"`` for upstream's
    single-sequence mode (which upstream advises against, as it costs accuracy), or
    set ``use_msa_server=True`` deliberately.
  * *Kernels.* ``cuequivariance_torch`` and ``trifast`` are not installed in
    ``drug-agent-boltz``, and Boltz imports them lazily inside the triangular
    update and attention forward passes. Running with kernels on would raise
    ``ImportError`` mid-forward, so the adapter probes for them and sends
    ``no_kernels=True`` when they are absent — upstream's own documented fallback.

**What it deliberately does not do.** It does not dock into a prepared receptor: it
folds the complex from sequence, so there is no receptor PDB and no search box, and
a pocket can only be *suggested*, via ``pocket_residues`` (upstream's pocket
constraint, default 6 Å). It exposes one protein chain and one ligand; multimers,
nucleic acids, covalent bonds, modified residues and templates are all in the YAML
schema the driver builds but are not surfaced as adapter arguments. It does not
validate the poses it returns — PoseBusters does that, and the producer must not be
its own judge. It never converts ``pred_value`` into an experimental affinity;
``ic50_micromolar`` and ``kcal_per_mol`` are upstream's own documented arithmetic on
the same predicted number, offered for readability and not as measurements.

Until the cache is populated this capability is UNAVAILABLE, every operation raises
:class:`~drug_agent.tools.base.ToolUnavailable`, and the platform predicts no
complexes and no affinities. There is no surrogate: a fabricated confidence would be
indistinguishable from a real one by the time it reached a scorecard, and a
fabricated affinity would be worse.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.structure import Pose
from ...observability.toolmeter import measure
from .._ext.adapter import NO_BACKEND
from .._ext.probe import missing_modules
from .._ext.spec import ExternalSpec
from . import StructureError, StructureToolAdapter

#: This adapter's contract with Boltz-2, not Boltz-2's own version.
BOLTZ2_ADAPTER_VERSION = "1.0.0"

#: What ``boltz.main.download_boltz2`` puts in the cache, with the size each URL
#: reported as HTTP ``content-length`` on 2026-09-07. ``mols`` is the directory
#: ``mols.tar`` is extracted into; upstream tests for it with ``.exists()``.
BOLTZ2_CACHE_ENTRIES: tuple[tuple[str, int], ...] = (
    ("boltz2_conf.ckpt", 2_286_561_469),
    ("boltz2_aff.ckpt", 2_062_139_170),
    ("mols", 1_855_662_080),
)

#: Kernel packages Boltz imports lazily inside its triangular ops. Absent here, so
#: the driver is told to take the pure-torch path instead of failing mid-forward.
_KERNEL_MODULES: tuple[str, ...] = ("cuequivariance_torch", "trifast")

#: Chain ids this adapter assigns in the YAML it builds. Fixed, because the pocket
#: constraint and the affinity binder both have to name them.
PROTEIN_CHAIN = "A"
LIGAND_CHAIN = "B"

#: Scalar fields of upstream's ``confidence_*.json``. The nested ``chains_ptm`` and
#: ``pair_chains_iptm`` maps are deliberately not flattened.
_CONFIDENCE_SCALARS: tuple[str, ...] = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)

#: Confidence numbers carried onto each Pose as ``key=value`` flags, so a pose that
#: travels alone still says how much of its confidence came from the interface.
_POSE_FLAG_KEYS: tuple[str, ...] = ("ptm", "iptm", "ligand_iptm", "complex_plddt")

#: Affinity fields, as ``(upstream key, dataclass attribute)``.
_AFFINITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("affinity_pred_value", "pred_value"),
    ("affinity_probability_binary", "probability_binary"),
    ("affinity_pred_value1", "pred_value_1"),
    ("affinity_probability_binary1", "probability_binary_1"),
    ("affinity_pred_value2", "pred_value_2"),
    ("affinity_probability_binary2", "probability_binary_2"),
)

SPEC = ExternalSpec(
    name="Boltz-2",
    env_prefix="BOLTZ",
    driver_name="boltz_predict.py",
    selection_modules=("boltz",),
    # What a broken install loses, read off upstream's imports rather than guessed:
    # boltz.main imports torch and pytorch_lightning directly; numba is imported by
    # boltz.data.feature.featurizerv2 and gemmi by boltz.data.parse.mmcif, both on
    # the run path, and both are compiled wheels that a partial install skips.
    runtime_modules=("boltz", "torch", "pytorch_lightning", "rdkit", "numba", "gemmi"),
    conda_envs=("drug-agent-boltz",),
    upstream_url="https://github.com/jwohlwend/boltz",
    upstream_ref="b1ebfc46ecf5",
    license="Boltz MIT License (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro",
    install_hint=(
        'conda run -n drug-agent-boltz python -c "from pathlib import Path; '
        "from boltz.main import download_boltz2; c = Path('~/.boltz').expanduser(); "
        'c.mkdir(parents=True, exist_ok=True); download_boltz2(c)"'
    ),
    needs_gpu=True,
)


class BoltzPredictionError(StructureError):
    """Boltz-2 was installed with its weights, it ran, and it produced no usable complex."""


@dataclass
class BoltzAffinityPrediction:
    """The affinity head's output for one ligand against one predicted complex.

    ``pred_value`` is ``log10(IC50)`` for an IC50 in micromolar, so lower is a
    stronger predicted binder, and it is only meaningful *between actives*.
    ``probability_binary`` is the separate binder-versus-decoy head. Upstream trains
    the two on largely different data and says so; they are kept as two numbers here
    for the same reason.
    """

    candidate_id: str = ""
    pred_value: float | None = None
    probability_binary: float | None = None
    pred_value_1: float | None = None
    probability_binary_1: float | None = None
    pred_value_2: float | None = None
    probability_binary_2: float | None = None
    mw_correction: bool = False
    engine: str = ""
    engine_version: str = ""
    qc_flags: list[str] = field(default_factory=list)

    @property
    def ic50_micromolar(self) -> float | None:
        """``10 ** pred_value`` — the IC50 the prediction implies, in µM.

        The inverse of upstream's stated encoding, not a second prediction.
        """
        if self.pred_value is None or not math.isfinite(self.pred_value):
            return None
        return float(10.0**self.pred_value)

    @property
    def kcal_per_mol(self) -> float | None:
        """``(6 - pred_value) * 1.364``, upstream's own conversion.

        ``docs/prediction.md`` gives this formula verbatim and calls the result
        "pIC50 in kcal/mol". It is arithmetic on the predicted number, carrying
        exactly the predicted number's error, and it is not comparable with a
        docking score in kcal/mol.
        """
        if self.pred_value is None or not math.isfinite(self.pred_value):
            return None
        return float((6.0 - self.pred_value) * 1.364)


@dataclass
class BoltzComplexPrediction:
    """One Boltz-2 run: the predicted complex, its confidence, and the affinity."""

    candidate_id: str = ""
    poses: list[Pose] = field(default_factory=list)
    affinity: BoltzAffinityPrediction | None = None
    #: Scalar confidence fields of the best-ranked model.
    confidence: dict[str, float] = field(default_factory=dict)
    engine_version: str = ""
    #: Where boltz's own output tree was left, when the caller asked for one kept.
    results_dir: str | None = None
    used_kernels: bool = False
    qc_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def confidence_score(self) -> float | None:
        """The best model's aggregated confidence, 0-1, higher is better."""
        return self.confidence.get("confidence_score")

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        A collapse in ``ligand_iptm`` is how a mis-specified ligand chain or a bad
        MSA announces itself long before anyone looks at the coordinates, so the
        interface confidences are metered on every call alongside the affinity.
        """
        out: dict[str, float] = {"boltz2_n_poses": float(len(self.poses))}
        for key in ("confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt"):
            value = self.confidence.get(key)
            if value is not None and math.isfinite(value):
                out[f"boltz2_{key}"] = float(value)
        if self.affinity is not None:
            if self.affinity.pred_value is not None:
                out["boltz2_affinity_pred_value"] = float(self.affinity.pred_value)
            if self.affinity.probability_binary is not None:
                out["boltz2_affinity_probability_binary"] = float(self.affinity.probability_binary)
        return out


class Boltz2Adapter(StructureToolAdapter):
    """Complex structure prediction and a binding-affinity head, from sequence."""

    capability_id = "structure.complex.boltz2"
    spec = SPEC
    engine = "boltz2"
    version = BOLTZ2_ADAPTER_VERSION
    produces_nothing = "predicts no complexes"
    #: Not a docking energy. The score on a Boltz-2 pose is structural confidence,
    #: and it is not comparable with a GNINA or Vina score under any transform.
    score_unit = "confidence_score (0-1)"
    lower_is_better = False

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        # A single complex is minutes of GPU work at the default 200 sampling steps,
        # and the affinity pass adds a second Trainer.predict on top of it.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 3600.0)
        self.cache_dir = self._resolve_cache_dir(settings, cache_dir)
        self.missing_weights = [
            name for name, _ in BOLTZ2_CACHE_ENTRIES if not (self.cache_dir / name).exists()
        ]
        if self.missing_weights:
            # The weights live outside any checkout, so ``weight_relpaths`` cannot
            # express them and the backend's probe never sees them. Appending here
            # keeps available(), the card and the raised exception on one answer.
            self._missing = [*self._missing, self._weights_blocker()]
            self.backend = NO_BACKEND
        self._kernels_absent: list[str] | None = None

    # --- configuration -----------------------------------------------------
    def _resolve_cache_dir(self, settings: Any, override: str | Path | None) -> Path:
        """Where Boltz keeps its checkpoints: explicit, then upstream's own variable.

        ``ExternalBackend`` resolves only ``DRUGAGENT_*`` names, and ``BOLTZ_CACHE``
        is upstream's, read by ``boltz.main.get_cache_path``. Honouring it here is
        what stops the card claiming weights are missing from a directory Boltz
        itself would never look in.
        """
        if override:
            return Path(override).expanduser()
        configured = getattr(settings, "boltz_cache", None) if settings is not None else None
        if configured:
            return Path(str(configured)).expanduser()
        env = os.environ.get("BOLTZ_CACHE")
        if env:
            return Path(env).expanduser()
        return Path("~/.boltz").expanduser()

    def _weights_blocker(self) -> str:
        """Name the absent checkpoints and what fetching them costs."""
        sizes = ", ".join(
            f"{name} ({size / 1e9:.2f} GB)"
            for name, size in BOLTZ2_CACHE_ENTRIES
            if name in self.missing_weights
        )
        total = sum(size for name, size in BOLTZ2_CACHE_ENTRIES if name in self.missing_weights)
        return (
            f"the Boltz-2 checkpoint cache {self.cache_dir} is missing {sizes} — "
            f"{total / 1e9:.2f} GB that boltz.main.download_boltz2 fetches from "
            "model-gateway.boltz.bio on first use (huggingface.co/boltz-community is its "
            "fallback), and which this package never downloads implicitly"
        )

    @property
    def unavailable_reason(self) -> str | None:
        """The whole reason, including blockers the backend's own probe cannot see.

        Mirrors ``ExternalBackend.unavailable_reason`` word for word so the card and
        the ``ToolUnavailable`` raised at call time stay one string, not two.
        """
        if not self._missing:
            return None
        hint = f" Install it with `{self.spec.install_hint}`." if self.spec.install_hint else ""
        return (
            f"{self.spec.name} is not usable here, so this capability is UNAVAILABLE and "
            f"{self.produces_nothing} — there is no surrogate backend.{hint} "
            f"Reason: {'; '.join(self._missing)}."
        )

    def use_kernels(self) -> bool:
        """Whether the cuequivariance/trifast fast paths can be taken at all.

        Boltz imports them *inside* the triangular forward passes, so an install
        without them fails deep in a prediction rather than at load. Probed once,
        lazily: ``tool_pool()`` builds this adapter on every listing.
        """
        if self._kernels_absent is None:
            self._kernels_absent = missing_modules(self.ext.python, _KERNEL_MODULES)
        return not self._kernels_absent

    # --- profile -----------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        requirements = dict(self.ext.runtime_requirements())
        requirements["weights_cache"] = str(self.cache_dir)
        requirements["missing_weights"] = list(self.missing_weights)
        return self.build_profile(
            title="Complex structure prediction and binding affinity (Boltz-2)",
            description=(
                "Folds a protein-ligand complex from sequence with a diffusion model and "
                "returns one Pose per diffusion sample, scored by structural confidence "
                "(0-1). A separate trained head reports the ligand's predicted binding "
                "affinity as log10(IC50/uM) and its probability of being a binder; neither "
                "is written into the pose score, because neither is a docking energy."
            ),
            output_schema_ref="BoltzComplexPrediction",
            runtime_requirements=requirements,
            # Diffusion sampling. Reproducible only when the caller passes a seed,
            # which upstream leaves unset by default.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint cache empty; the first run would download 6.20 GB",
                "a protein chain with no MSA and no --use_msa_server is refused by boltz",
                "the MSA server path posts the sequence to api.colabfold.com",
                "cuequivariance_torch/trifast absent, so the kernel fast paths cannot be used",
                "affinity is limited to one small-molecule chain of at most 128 atoms",
                "affinity_pred_value is meaningless on inactives; it compares actives only",
                "GPU memory: a large complex at 200 sampling steps can exhaust the device",
            ],
            validation_checks=[
                "every returned pose carries mmCIF or PDB coordinates",
                "pose score is the structural confidence, never the affinity",
                "affinity is present only when a ligand chain was declared the binder",
            ],
        )

    # --- operations --------------------------------------------------------
    def predict(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        ligand_smiles: str | None = None,
        ligand_ccd: str | None = None,
        receptor_id: str | None = None,
        msa: str | Path | None = None,
        use_msa_server: bool = False,
        with_affinity: bool = True,
        pocket_residues: Sequence[int] | None = None,
        pocket_max_distance: float | None = None,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        seed: int | None = None,
        output_format: str = "mmcif",
        results_dir: str | Path | None = None,
    ) -> BoltzComplexPrediction:
        """Predict one complex, with the affinity head unless ``with_affinity`` is false."""
        self.require_available()
        with measure(self.capability_id, "predict", backend=self.backend) as metered:
            prediction = self._predict(
                candidate_id=candidate_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles,
                ligand_ccd=ligand_ccd,
                receptor_id=receptor_id,
                msa=msa,
                use_msa_server=use_msa_server,
                with_affinity=with_affinity,
                pocket_residues=pocket_residues,
                pocket_max_distance=pocket_max_distance,
                diffusion_samples=diffusion_samples,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                seed=seed,
                output_format=output_format,
                results_dir=results_dir,
            )
            for key, value in prediction.metrics_for_ledger().items():
                metered[key] = value
        return prediction

    def predict_complex(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        ligand_smiles: str | None = None,
        ligand_ccd: str | None = None,
        receptor_id: str | None = None,
        msa: str | Path | None = None,
        use_msa_server: bool = False,
        pocket_residues: Sequence[int] | None = None,
        diffusion_samples: int = 1,
        seed: int | None = None,
        output_format: str = "mmcif",
        results_dir: str | Path | None = None,
    ) -> list[Pose]:
        """The structure alone, ranked by confidence. Skips the affinity pass entirely."""
        self.require_available()
        with measure(self.capability_id, "predict_complex", backend=self.backend) as metered:
            prediction = self._predict(
                candidate_id=candidate_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles,
                ligand_ccd=ligand_ccd,
                receptor_id=receptor_id,
                msa=msa,
                use_msa_server=use_msa_server,
                with_affinity=False,
                pocket_residues=pocket_residues,
                pocket_max_distance=None,
                diffusion_samples=diffusion_samples,
                recycling_steps=3,
                sampling_steps=200,
                seed=seed,
                output_format=output_format,
                results_dir=results_dir,
            )
            for key, value in prediction.metrics_for_ledger().items():
                metered[key] = value
        return prediction.poses

    def predict_affinity(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        ligand_smiles: str | None = None,
        ligand_ccd: str | None = None,
        msa: str | Path | None = None,
        use_msa_server: bool = False,
        pocket_residues: Sequence[int] | None = None,
        diffusion_samples: int = 1,
        seed: int | None = None,
        results_dir: str | Path | None = None,
    ) -> BoltzAffinityPrediction:
        """The affinity head's verdict on one ligand.

        The complex is predicted regardless — the affinity module reads the structure
        the diffusion model just produced — so this costs what :meth:`predict` costs
        and simply discards the coordinates. Call :meth:`predict` if you want both.
        """
        self.require_available()
        with measure(self.capability_id, "predict_affinity", backend=self.backend) as metered:
            prediction = self._predict(
                candidate_id=candidate_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles,
                ligand_ccd=ligand_ccd,
                receptor_id=None,
                msa=msa,
                use_msa_server=use_msa_server,
                with_affinity=True,
                pocket_residues=pocket_residues,
                pocket_max_distance=None,
                diffusion_samples=diffusion_samples,
                recycling_steps=3,
                sampling_steps=200,
                seed=seed,
                output_format="mmcif",
                results_dir=results_dir,
            )
            for key, value in prediction.metrics_for_ledger().items():
                metered[key] = value
        if prediction.affinity is None:
            raise BoltzPredictionError(
                f"Boltz-2 predicted a complex for {candidate_id!r} but wrote no affinity file; "
                f"the affinity module reported a failed example ({'; '.join(prediction.warnings)})"
            )
        return prediction.affinity

    # --- real backend ------------------------------------------------------
    def _predict(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        ligand_smiles: str | None,
        ligand_ccd: str | None,
        receptor_id: str | None,
        msa: str | Path | None,
        use_msa_server: bool,
        with_affinity: bool,
        pocket_residues: Sequence[int] | None,
        pocket_max_distance: float | None,
        diffusion_samples: int,
        recycling_steps: int,
        sampling_steps: int,
        seed: int | None,
        output_format: str,
        results_dir: str | Path | None,
    ) -> BoltzComplexPrediction:
        request = self._build_request(
            candidate_id=candidate_id,
            protein_sequence=protein_sequence,
            ligand_smiles=ligand_smiles,
            ligand_ccd=ligand_ccd,
            msa=msa,
            use_msa_server=use_msa_server,
            with_affinity=with_affinity,
            pocket_residues=pocket_residues,
            pocket_max_distance=pocket_max_distance,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            seed=seed,
            output_format=output_format,
            results_dir=results_dir,
        )
        run = self.run_driver(request, error_cls=BoltzPredictionError, seed=seed)
        return self._parse(
            run.payload,
            candidate_id=candidate_id,
            receptor_id=receptor_id,
            kept_results_dir=str(results_dir) if results_dir else None,
        )

    def _build_request(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        ligand_smiles: str | None,
        ligand_ccd: str | None,
        msa: str | Path | None,
        use_msa_server: bool,
        with_affinity: bool,
        pocket_residues: Sequence[int] | None,
        pocket_max_distance: float | None,
        diffusion_samples: int,
        recycling_steps: int,
        sampling_steps: int,
        seed: int | None,
        output_format: str,
        results_dir: str | Path | None,
    ) -> dict[str, Any]:
        """Validate the inputs and assemble the driver request.

        The checks here are the ones worth making *before* committing a GPU to a
        prediction that would fail on the same grounds twenty minutes in.
        """
        if not protein_sequence.strip():
            raise BoltzPredictionError("protein_sequence is empty; Boltz-2 folds from sequence")
        if bool(ligand_smiles) == bool(ligand_ccd):
            raise BoltzPredictionError(
                "supply exactly one of ligand_smiles or ligand_ccd "
                f"(got smiles={ligand_smiles!r}, ccd={ligand_ccd!r})"
            )
        if msa is None and not use_msa_server:
            raise BoltzPredictionError(
                "boltz refuses a protein chain with no MSA: pass msa=<path to .a3m/.csv>, "
                'or msa="empty" for upstream\'s single-sequence mode, or set '
                "use_msa_server=True to post the sequence to api.colabfold.com"
            )
        if output_format not in ("mmcif", "pdb"):
            raise BoltzPredictionError(f"output_format must be 'mmcif' or 'pdb', got {output_format!r}")

        protein: dict[str, Any] = {
            "kind": "protein",
            "id": PROTEIN_CHAIN,
            "sequence": protein_sequence.strip(),
        }
        if msa is not None:
            protein["msa"] = str(msa)
        ligand: dict[str, Any] = {"kind": "ligand", "id": LIGAND_CHAIN}
        if ligand_smiles:
            ligand["smiles"] = ligand_smiles
        else:
            ligand["ccd"] = ligand_ccd

        constraints: list[dict[str, Any]] = []
        if pocket_residues:
            pocket: dict[str, Any] = {
                "binder": LIGAND_CHAIN,
                "contacts": [[PROTEIN_CHAIN, int(res)] for res in pocket_residues],
            }
            if pocket_max_distance is not None:
                pocket["max_distance"] = float(pocket_max_distance)
            constraints.append({"pocket": pocket})

        return {
            # boltz names its record after the input file's stem, so the job name
            # decides every output path the driver then reads back.
            "name": _job_name(candidate_id),
            "sequences": [protein, ligand],
            "constraints": constraints,
            "affinity_binder": LIGAND_CHAIN if with_affinity else None,
            "cache": str(self.cache_dir),
            "results_dir": str(results_dir) if results_dir else None,
            "allow_download": False,
            "diffusion_samples": int(diffusion_samples),
            "recycling_steps": int(recycling_steps),
            "sampling_steps": int(sampling_steps),
            "seed": seed,
            "output_format": output_format,
            "accelerator": "gpu" if self.ext.device.startswith("cuda") else "cpu",
            "use_msa_server": bool(use_msa_server),
            "no_kernels": not self.use_kernels(),
            "override": True,
        }

    def _parse(
        self,
        payload: dict[str, Any],
        *,
        candidate_id: str,
        receptor_id: str | None,
        kept_results_dir: str | None,
    ) -> BoltzComplexPrediction:
        """Turn the driver's payload into poses, confidence and an affinity."""
        samples = payload.get("samples") or []
        engine_version = str(payload.get("engine_version") or "unknown")
        warnings = [str(w) for w in payload.get("warnings") or []]

        poses: list[Pose] = []
        # ``rank()`` renumbers ``Pose.rank``, so the link back to the diffusion
        # sample it came from has to be held here rather than read off the pose.
        confidence_of: dict[int, dict[str, float]] = {}
        for sample in samples:
            confidence = _scalars(sample.get("confidence") or {})
            flags = [f"{key}={confidence[key]:.4f}" for key in _POSE_FLAG_KEYS if key in confidence]
            coordinates = sample.get("coordinates")
            if not coordinates:
                flags.append("no_coordinates_returned")
            if "confidence_score" not in confidence:
                flags.append("missing_confidence_score")
            pose = Pose(
                candidate_id=candidate_id,
                receptor_id=receptor_id,
                rank=int(sample.get("model_index", len(poses))),
                score=confidence.get("confidence_score"),
                score_unit=self.score_unit,
                pose_ref=coordinates,
                engine=self.engine,
                engine_version=engine_version,
                # A diffusion model's output is a prediction, not a calculation.
                trust_label=TrustLabel.PREDICTED,
                qc_flags=flags,
            )
            confidence_of[id(pose)] = confidence
            poses.append(pose)

        poses = self.rank(poses)
        # The best-ranked model's numbers, not those of whichever file was read first.
        best_confidence = confidence_of.get(id(poses[0]), {}) if poses else {}

        affinity: BoltzAffinityPrediction | None = None
        raw_affinity = payload.get("affinity")
        if isinstance(raw_affinity, dict):
            affinity = BoltzAffinityPrediction(
                candidate_id=candidate_id,
                engine=self.engine,
                engine_version=engine_version,
            )
            for key, attr in _AFFINITY_FIELDS:
                value = raw_affinity.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    setattr(affinity, attr, float(value))
            if affinity.pred_value is None:
                affinity.qc_flags.append("missing_affinity_pred_value")
        affinity_requested = bool(payload.get("affinity_requested"))

        return BoltzComplexPrediction(
            candidate_id=candidate_id,
            poses=poses,
            affinity=affinity,
            confidence=best_confidence,
            engine_version=engine_version,
            results_dir=kept_results_dir,
            used_kernels=not bool(payload.get("no_kernels", True)),
            qc_flags=["affinity_not_written"] if affinity_requested and affinity is None else [],
            warnings=warnings,
        )


def _job_name(candidate_id: str) -> str:
    """A filesystem-safe stem for one job; boltz derives its record id from it."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in candidate_id.strip())
    return safe[:64] or "complex"


def _scalars(confidence: dict[str, Any]) -> dict[str, float]:
    """The finite scalar confidence fields, ignoring the nested per-chain maps."""
    out: dict[str, float] = {}
    for key in _CONFIDENCE_SCALARS:
        value = confidence.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fvalue = float(value)
            if math.isfinite(fvalue):
                out[key] = fvalue
    return out


__all__ = [
    "BOLTZ2_ADAPTER_VERSION",
    "BOLTZ2_CACHE_ENTRIES",
    "SPEC",
    "Boltz2Adapter",
    "BoltzAffinityPrediction",
    "BoltzComplexPrediction",
    "BoltzPredictionError",
]
