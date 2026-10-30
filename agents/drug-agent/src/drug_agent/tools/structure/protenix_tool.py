"""Protenix — ByteDance's open reproduction of AlphaFold3, folding a complex from sequence.

Protenix is a trainable, fully open re-implementation of the AlphaFold3
architecture: a pairformer trunk, a diffusion module over all-atom coordinates,
and a confidence head. Given a protein sequence and a ligand it predicts the
complex — the same question Boltz-2 answers, from a different codebase and a
different training run, which is exactly why both are here. Two independent
predictors that agree are evidence; one predictor that is confident is not.

**What it computes.** From one or more polymer chains and zero or more ligands it
returns ``list[Pose]``, one per diffusion sample per seed, each carrying the
sample's confidence. The observables, with upstream's own keys and units:

    ``Pose.score``            ``ranking_score``, upstream's own ordering statistic,
                              ``0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash``,
                              higher is better and it is not bounded to 0-1
    ``Pose.pose_ref``         the predicted complex as mmCIF text
    ``confidence['plddt']``   mean atom pLDDT, 0-100, higher is better
    ``confidence['ptm']``     predicted TM-score of the whole complex, 0-1
    ``confidence['iptm']``    the interface pTM — the number that says whether the
                              *ligand placement* is believed, not just the fold
    ``confidence['gpde']``    global predicted distance error, lower is better
    ``confidence['has_clash']``  1.0 when upstream's clash test fired

Per-chain and per-chain-pair matrices (``chain_plddt``, ``chain_pair_iptm``, ...)
come back untouched in ``ProtenixPrediction.chain_confidence`` rather than being
flattened into the scalars, because averaging an interface matrix into one number
is how a confidently-wrong ligand pose hides behind a well-folded protein.

**Why the score is not comparable with anything else in this package.** A
``ranking_score`` is not a docking energy, not a Boltz-2 ``confidence_score``, and
not a Boltz-2 affinity. It has no kcal/mol, it is not bounded, and the
``-100*has_clash`` term makes a clashing sample's value an outlier by construction
rather than a slightly worse score. It is recorded with its own ``score_unit`` and
its own ``engine``, and nothing in this package converts between engines.

**Setup reaches the network; running never does.** Upstream's
``runner.inference.download_inference_cache`` fetches the checkpoint and the CCD
component definitions on the first prediction — 2.13 GB, mid-tool-call, from
``protenix.tos-cn-beijing.volces.com``. The driver refuses to do that:
``scripts/fetch_protenix_weights.py`` is where it happens, once, deliberately,
reading the URLs out of the installed package's own table so this repository never
holds a second copy of them. At runtime the driver checks the five artifacts and
fails with their absolute paths if any is missing.

**MSA, and what it costs.** Protenix's accuracy assumes an MSA, and upstream's own
default fetches one from its MSA server. That is network I/O inside a tool call,
so this adapter defaults ``use_msa`` to **false** — upstream's single-sequence
mode, which upstream itself warns degrades accuracy significantly. Two ways to get
the accuracy back without opening egress mid-run: pass ``msa=<dir of precomputed
alignments>``, in which case upstream's ``need_msa_search`` finds nothing to do; or
set ``use_msa=True`` deliberately, with the platform's egress policy opened, and
accept that the sequence leaves the machine. That second path is checked against
:class:`~drug_agent.security.egress.EgressPolicy` *here*, before the driver is
launched: the search happens inside a subprocess whose sockets the policy cannot
see, so a check on this side is the only one there is, and without it the
platform's deny-by-default posture would simply not apply to this tool. The choice
is recorded on every prediction as a qc flag, because a single-sequence Protenix
number and an MSA-backed one are not the same measurement. Measured on this host,
on the same input — the 555-residue sEH construct from ``.external/pdb/3I28.pdb``
with its co-crystal ligand 34N as a CCD code, ``protenix_base_default_v1.0.0``,
seed 101 — single-sequence gives pLDDT 39.3, pTM 0.326, ipTM 0.303, ranking_score
0.308; with the MSA server it gives pLDDT 94.4, pTM 0.818, ipTM 0.983,
ranking_score 0.950. Same model, same weights, same seed: the flag is the
difference between a guess and a prediction.

**The one default this adapter overrides, and how that was established.**
Upstream selects a JIT-compiled fused LayerNorm (``LAYERNORM_TYPE=fast_layernorm``,
read at import time by ``protenix.model.triangular.layers``). On this host —
NVIDIA RTX PRO 6000 Blackwell, sm_120, torch 2.7.1+cu128, nvcc 13.1 — that kernel
compiles, runs, and returns NaN: the model writes a complete, well-formed mmCIF
with ``nan`` in every coordinate column and a confidence file whose ``plddt``,
``ptm``, ``iptm`` and ``ranking_score`` are all NaN. Nothing raises. The cause was
isolated by bisection over the six execution switches, on a 58-token peptide-ATP
complex with ``protenix_mini_default_v0.5.0``: with ``LAYERNORM_TYPE=torch`` the
prediction is finite with the cuequivariance triangle kernels, the fused diffusion
transformer, the diffusion cache and TF32 all left at upstream's defaults; turning
``fast_layernorm`` back on alone returns it to all-NaN. So exactly one default is
overridden, it is overridden in one named place, and every prediction records the
three kernel choices it was computed with.

That failure is also why the driver reads its own output back: a run that produces
NaN coordinates exits zero, and a Pose built from it would be indistinguishable
from a real one. The driver counts non-finite atom rows, withholds any sample that
is entirely NaN, and raises if none survives.

**What it deliberately does not do.** It does not dock into a prepared receptor —
it folds the complex, so there is no receptor PDB and no search box. It predicts no
binding affinity: Protenix has no affinity head, and the confidence of a pose is
not a statement about how tightly it binds. It does not validate its own poses;
PoseBusters does that, and the producer must not be its own judge. And it does not
fall back to a smaller model, a cheaper sampling schedule or a cached answer when
the requested checkpoint is absent: the capability reports UNAVAILABLE, names the
file, and predicts nothing.
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
from ...security.egress import EgressPolicy
from .._ext.adapter import NO_BACKEND
from .._ext.spec import ExternalSpec
from . import StructureError, StructureToolAdapter

#: This adapter's contract with Protenix, not Protenix's own version.
PROTENIX_ADAPTER_VERSION = "1.0.0"

#: Upstream's default inference checkpoint, and the one the readiness probe looks
#: for. A caller may ask for another; the driver refuses if it is not cached.
DEFAULT_MODEL = "protenix_base_default_v1.0.0"

#: What every prediction needs on disk, relative to ``PROTENIX_ROOT_DIR``, with the
#: size each URL reported as HTTP ``content-length`` on 2026-09-09. The four data
#: caches are ``download_inference_cache``'s unconditional list; the checkpoint is
#: appended per model because which one is needed depends on the request.
PROTENIX_DATA_CACHE: tuple[tuple[str, int], ...] = (
    ("common/components.cif", 490_777_362),
    ("common/components.cif.rdkit_mol.pkl", 142_498_117),
    ("common/clusters-by-entity-40.txt", 21_699_572),
    ("common/obsolete_release_date.csv", 134_716),
)

#: Checkpoint sizes, same source and date. Absent entries are still probed for
#: existence; only the "what will this cost" arithmetic needs the number.
PROTENIX_CHECKPOINT_BYTES: dict[str, int] = {
    "protenix_base_default_v1.0.0": 1_475_950_125,
    "protenix_mini_default_v0.5.0": 537_049_294,
}

#: One number per sample in upstream's ``*_summary_confidence_sample_*.json``.
_SCALAR_CONFIDENCE: tuple[str, ...] = (
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

#: Confidence numbers carried onto each Pose as ``key=value`` flags, so a pose that
#: travels alone still says how much of its confidence came from the interface.
_POSE_FLAG_KEYS: tuple[str, ...] = ("plddt", "ptm", "iptm", "has_clash")

#: The three kernel switches Protenix exposes, and what this adapter sends by
#: default. Two are upstream's own; the third is not, and the difference was
#: measured rather than assumed — see "The one default this adapter overrides" in
#: the module docstring. All three are named in ``runtime_requirements`` so a
#: prediction can always be traced to the kernels that computed it.
DEFAULT_TRIMUL_KERNEL = "cuequivariance"
DEFAULT_TRIATT_KERNEL = "cuequivariance"
#: NOT upstream's default, which is ``fast_layernorm``. That path JIT-compiles
#: ``fast_layer_norm_cuda_v2`` and returns NaN for every atom on this GPU.
DEFAULT_LAYERNORM_TYPE = "torch"

#: Three further execution switches upstream turns on by default: the fused
#: diffusion-transformer path, the diffusion module's shared-variable cache, and
#: TF32 matmuls. They change how the forward pass is computed, not what it is
#: asked to compute, so they belong beside the kernel selection rather than in the
#: sampling schedule — and like it, they are named in ``runtime_requirements``.
DEFAULT_ENABLE_FUSION = True
DEFAULT_ENABLE_CACHE = True
DEFAULT_ENABLE_TF32 = True

#: Where an MSA search would go, per upstream's ``msa_server_mode``. Named here so
#: the platform's egress policy can be asked about the exact host *before* the
#: driver is launched: a subprocess's sockets are outside ``EgressPolicy``'s reach,
#: so the check has to happen on this side or not at all.
MSA_SERVER_URLS: dict[str, str] = {
    "protenix": "https://protenix-server.com/api/msa",
    "colabfold": "https://api.colabfold.com",
}

#: Sampling schedules upstream recommends per model family (its ``--use_default_params``).
#: Applied only when the caller asks for them, and never silently: a mini-model
#: schedule on the base model would quietly produce a much cheaper, worse answer.
DEFAULT_SCHEDULES: dict[str, tuple[int, int]] = {
    "base": (10, 200),
    "mini": (4, 5),
    "tiny": (4, 5),
}

SPEC = ExternalSpec(
    name="Protenix",
    env_prefix="PROTENIX",
    driver_name="protenix_predict.py",
    selection_modules=("protenix",),
    # What a broken install loses, read off upstream's imports rather than guessed.
    # `runner` and `configs` are top-level modules the protenix wheel installs and
    # the driver imports by name; `biotite` builds the AtomArray every output cif
    # is written from; `cuequivariance_torch` backs the default triangle kernels,
    # and unlike Boltz's optional kernels it is a hard dependency here, imported at
    # model construction rather than lazily inside a forward pass.
    runtime_modules=(
        "protenix",
        "runner",
        "configs",
        "torch",
        "biotite",
        "rdkit",
        "gemmi",
        "ml_collections",
        "cuequivariance_torch",
    ),
    conda_envs=("drug-agent-protenix",),
    upstream_url="https://github.com/bytedance/Protenix",
    upstream_ref="v2.0.0 (PyPI protenix 2.0.0)",
    license="Protenix Apache License 2.0 (c) 2024 ByteDance; checkpoints carry ByteDance's model terms",
    install_hint="make protenix-install && make protenix-weights",
    needs_gpu=True,
)


class ProtenixPredictionError(StructureError):
    """Protenix was installed with its weights, it ran, and it produced no usable complex."""


@dataclass
class ProtenixPrediction:
    """One Protenix run: the predicted complex and the confidence head's verdict."""

    candidate_id: str = ""
    poses: list[Pose] = field(default_factory=list)
    #: Scalar confidence of the best-ranked sample.
    confidence: dict[str, float] = field(default_factory=dict)
    #: Per-chain and per-chain-pair matrices of the best-ranked sample, verbatim.
    chain_confidence: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    engine_version: str = ""
    used_msa: bool = False
    n_samples: int = 0
    seconds: float | None = None
    #: Where Protenix's own output tree was left, when the caller asked for one kept.
    results_dir: str | None = None
    qc_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ranking_score(self) -> float | None:
        """The best sample's ``ranking_score``. Higher is better; not bounded to 0-1."""
        return self.confidence.get("ranking_score")

    @property
    def plddt(self) -> float | None:
        """Mean atom pLDDT of the best sample, 0-100."""
        return self.confidence.get("plddt")

    @property
    def iptm(self) -> float | None:
        """Interface pTM of the best sample — the ligand-placement confidence, 0-1."""
        return self.confidence.get("iptm")

    @property
    def has_clash(self) -> bool:
        """Whether upstream's clash test fired on the best sample."""
        return bool(self.confidence.get("has_clash", 0.0))

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        A collapse in ``iptm`` is how a mis-specified ligand, a truncated sequence
        or a single-sequence run announces itself long before anyone looks at the
        coordinates, so the interface confidence is metered on every call alongside
        the ranking score and the clash flag.
        """
        out: dict[str, float] = {"protenix_n_poses": float(len(self.poses))}
        for key in ("ranking_score", "plddt", "ptm", "iptm", "gpde", "has_clash"):
            value = self.confidence.get(key)
            if value is not None and math.isfinite(value):
                out[f"protenix_{key}"] = float(value)
        if self.seconds is not None:
            out["protenix_seconds"] = float(self.seconds)
        return out


class ProtenixAdapter(StructureToolAdapter):
    """All-atom complex structure prediction from sequence, with a confidence head."""

    capability_id = "structure.predict.protenix"
    spec = SPEC
    engine = "protenix"
    version = PROTENIX_ADAPTER_VERSION
    produces_nothing = "predicts no complexes"
    #: Not a docking energy and not a Boltz-2 confidence. See the module docstring.
    score_unit = "ranking_score (0.8*ipTM + 0.2*pTM - 100*has_clash)"
    lower_is_better = False
    #: A diffusion model's output is a prediction, not a calculation.
    trust_label = TrustLabel.PREDICTED

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        root_dir: str | Path | None = None,
        model_name: str | None = None,
        trimul_kernel: str | None = None,
        triatt_kernel: str | None = None,
        layernorm_type: str | None = None,
        enable_fusion: bool | None = None,
        enable_cache: bool | None = None,
        enable_tf32: bool | None = None,
    ) -> None:
        # A single complex is minutes of GPU work at the default schedule (10
        # recycles, 200 diffusion steps, 5 samples), and it scales with sequence
        # length squared through the pairformer.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 3600.0)
        self.model_name = model_name or _configured(settings, "protenix_model") or DEFAULT_MODEL
        self.trimul_kernel = trimul_kernel or DEFAULT_TRIMUL_KERNEL
        self.triatt_kernel = triatt_kernel or DEFAULT_TRIATT_KERNEL
        self.layernorm_type = layernorm_type or DEFAULT_LAYERNORM_TYPE
        self.enable_fusion = DEFAULT_ENABLE_FUSION if enable_fusion is None else bool(enable_fusion)
        self.enable_cache = DEFAULT_ENABLE_CACHE if enable_cache is None else bool(enable_cache)
        self.enable_tf32 = DEFAULT_ENABLE_TF32 if enable_tf32 is None else bool(enable_tf32)
        # The only network the run path can reach is an MSA search, and it happens
        # inside the driver subprocess where EgressPolicy cannot see it. Resolve the
        # policy here, from settings when there are any and from the environment
        # otherwise, and consult it before a request that would search is built.
        self.egress = (
            EgressPolicy(
                allow_network=bool(getattr(settings, "allow_network", False)),
                allowlist=list(getattr(settings, "egress_allowlist", None) or []),
            )
            if settings is not None
            else EgressPolicy.from_env()
        )
        self.root_dir = self._resolve_root_dir(settings, root_dir)
        self.missing_artifacts = [rel for rel, _ in PROTENIX_DATA_CACHE if not (self.root_dir / rel).exists()]
        checkpoint = f"checkpoint/{self.model_name}.pt"
        if not (self.root_dir / checkpoint).is_file():
            self.missing_artifacts.append(checkpoint)
        if self.missing_artifacts:
            # The artifacts live outside any checkout, so ``weight_relpaths`` cannot
            # express them and the backend's probe never sees them. Appending here
            # keeps available(), the card and the raised exception on one answer.
            self._missing = [*self._missing, self._artifacts_blocker()]
            self.backend = NO_BACKEND

    # --- configuration -----------------------------------------------------
    def _resolve_root_dir(self, settings: Any, override: str | Path | None) -> Path:
        """Where Protenix keeps its checkpoint and CCD caches.

        Precedence: an explicit argument, then the platform's own
        ``external_tools['protenix']['root']`` / ``DRUGAGENT_PROTENIX_ROOT`` (which
        ``ExternalBackend`` has already resolved), then upstream's own
        ``PROTENIX_ROOT_DIR``, then ``~/.protenix``. Honouring upstream's variable
        is what stops the card claiming artifacts are missing from a directory
        Protenix itself would never look in — ``configs.configs_inference`` reads it
        at import time to build ``load_checkpoint_dir``.
        """
        if override:
            return Path(override).expanduser()
        if self.ext.root is not None:
            return self.ext.root
        configured = _configured(settings, "protenix_root")
        if configured:
            return Path(configured).expanduser()
        env = os.environ.get("PROTENIX_ROOT_DIR")
        if env:
            return Path(env).expanduser()
        return Path("~/.protenix").expanduser()

    def _artifacts_blocker(self) -> str:
        """Name the absent artifacts and what fetching them costs."""
        sizes = dict(PROTENIX_DATA_CACHE)
        sizes[f"checkpoint/{self.model_name}.pt"] = PROTENIX_CHECKPOINT_BYTES.get(self.model_name, 0)
        listed = ", ".join(
            f"{self.root_dir / rel}" + (f" ({sizes[rel] / 1e9:.2f} GB)" if sizes.get(rel) else "")
            for rel in self.missing_artifacts
        )
        total = sum(sizes.get(rel, 0) for rel in self.missing_artifacts)
        return (
            f"the Protenix artifact cache {self.root_dir} is missing {listed} — "
            f"{total / 1e9:.2f} GB that runner.inference.download_inference_cache fetches from "
            "protenix.tos-cn-beijing.volces.com on first use, and which this package never "
            "downloads implicitly; run `python scripts/fetch_protenix_weights.py --model "
            f"{self.model_name}` with egress open"
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

    def default_schedule(self) -> tuple[int, int]:
        """``(n_cycle, n_step)`` upstream recommends for the configured model."""
        family = self.model_name.split("_")[1] if "_" in self.model_name else "base"
        return DEFAULT_SCHEDULES.get(family, DEFAULT_SCHEDULES["base"])

    # --- profile -----------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        requirements = dict(self.ext.runtime_requirements())
        requirements["artifact_root"] = str(self.root_dir)
        requirements["model_name"] = self.model_name
        requirements["missing_artifacts"] = list(self.missing_artifacts)
        requirements["network_at_runtime"] = False
        requirements["kernels"] = {
            "trimul": self.trimul_kernel,
            "triatt": self.triatt_kernel,
            "layernorm": self.layernorm_type,
            "efficient_fusion": self.enable_fusion,
            "diffusion_cache": self.enable_cache,
            "tf32": self.enable_tf32,
        }
        return self.build_profile(
            title="All-atom complex structure prediction (Protenix)",
            description=(
                "Folds a protein-ligand complex from sequence with an open AlphaFold3 "
                "reproduction and returns one Pose per diffusion sample, scored by "
                "upstream's ranking_score (0.8*ipTM + 0.2*pTM - 100*has_clash). pLDDT, "
                "pTM, ipTM and the per-chain confidence matrices come back beside it. "
                "It predicts no binding affinity, and its score is not comparable with "
                "a docking energy or with Boltz-2's confidence."
            ),
            output_schema_ref="ProtenixPrediction",
            runtime_requirements=requirements,
            # Diffusion sampling. Reproducible for a fixed seed and schedule; the
            # driver always sends an explicit seed list, defaulting to upstream's.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "artifact cache empty; the first run would download 2.13 GB",
                (
                    "single-sequence mode is the offline default and it costs accuracy: "
                    "measured pLDDT 39.3 / ipTM 0.303 against 94.4 / 0.983 with an MSA, "
                    "same complex, same weights, same seed"
                ),
                "use_msa=True posts the sequence to Protenix's MSA server",
                "GPU memory: cost grows with sequence length squared through the pairformer",
                "the v1.0.0 base model refuses inputs above 2560 tokens",
                "a SMILES that RDKit cannot embed is rejected before the model sees it",
                "cuequivariance kernels need a CUDA device; there is no CPU fallback path here",
            ],
            validation_checks=[
                "every returned pose carries mmCIF coordinates",
                "pose score is upstream's ranking_score, never pLDDT or an affinity",
                "each pose records whether it was predicted with an MSA",
                "a clashing sample keeps its -100 penalty rather than being dropped",
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
        extra_chains: Sequence[dict[str, Any]] | None = None,
        msa: str | Path | None = None,
        use_msa: bool = False,
        msa_server_mode: str = "protenix",
        model_name: str | None = None,
        seeds: Sequence[int] | None = None,
        n_cycle: int | None = None,
        n_step: int | None = None,
        n_sample: int = 5,
        dtype: str = "bf16",
        need_atom_confidence: bool = False,
        results_dir: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> ProtenixPrediction:
        """Predict one complex and return its poses with the confidence head's numbers."""
        self.require_available()
        with measure(self.capability_id, "predict", backend=self.backend) as metered:
            prediction = self._predict(
                candidate_id=candidate_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles,
                ligand_ccd=ligand_ccd,
                receptor_id=receptor_id,
                extra_chains=extra_chains,
                msa=msa,
                use_msa=use_msa,
                msa_server_mode=msa_server_mode,
                model_name=model_name,
                seeds=seeds,
                n_cycle=n_cycle,
                n_step=n_step,
                n_sample=n_sample,
                dtype=dtype,
                need_atom_confidence=need_atom_confidence,
                results_dir=results_dir,
                timeout_s=timeout_s,
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
        use_msa: bool = False,
        msa_server_mode: str = "protenix",
        seeds: Sequence[int] | None = None,
        n_sample: int = 5,
        results_dir: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> list[Pose]:
        """The structure alone, ranked by ``ranking_score``. Same cost as :meth:`predict`."""
        return self.predict(
            candidate_id=candidate_id,
            protein_sequence=protein_sequence,
            ligand_smiles=ligand_smiles,
            ligand_ccd=ligand_ccd,
            receptor_id=receptor_id,
            msa=msa,
            use_msa=use_msa,
            msa_server_mode=msa_server_mode,
            seeds=seeds,
            n_sample=n_sample,
            results_dir=results_dir,
            timeout_s=timeout_s,
        ).poses

    def fold(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        msa: str | Path | None = None,
        use_msa: bool = False,
        msa_server_mode: str = "protenix",
        seeds: Sequence[int] | None = None,
        n_sample: int = 1,
        timeout_s: float | None = None,
    ) -> ProtenixPrediction:
        """The apo protein alone — no ligand chain at all.

        Distinct from :meth:`predict` with a null ligand because the question is
        different: this is "what does the receptor look like", and its ``iptm`` is
        meaningless where a complex's is the whole point.
        """
        return self.predict(
            candidate_id=candidate_id,
            protein_sequence=protein_sequence,
            msa=msa,
            use_msa=use_msa,
            msa_server_mode=msa_server_mode,
            seeds=seeds,
            n_sample=n_sample,
            timeout_s=timeout_s,
        )

    # --- real backend ------------------------------------------------------
    def _predict(
        self,
        *,
        candidate_id: str,
        protein_sequence: str,
        ligand_smiles: str | None,
        ligand_ccd: str | None,
        receptor_id: str | None,
        extra_chains: Sequence[dict[str, Any]] | None,
        msa: str | Path | None,
        use_msa: bool,
        msa_server_mode: str,
        model_name: str | None,
        seeds: Sequence[int] | None,
        n_cycle: int | None,
        n_step: int | None,
        n_sample: int,
        dtype: str,
        need_atom_confidence: bool,
        results_dir: str | Path | None,
        timeout_s: float | None,
    ) -> ProtenixPrediction:
        request = self._build_request(
            candidate_id=candidate_id,
            protein_sequence=protein_sequence,
            ligand_smiles=ligand_smiles,
            ligand_ccd=ligand_ccd,
            extra_chains=extra_chains,
            msa=msa,
            use_msa=use_msa,
            msa_server_mode=msa_server_mode,
            model_name=model_name,
            seeds=seeds,
            n_cycle=n_cycle,
            n_step=n_step,
            n_sample=n_sample,
            dtype=dtype,
            need_atom_confidence=need_atom_confidence,
            results_dir=results_dir,
        )
        seed_list = list(request["seeds"])
        run = self.run_driver(
            request,
            error_cls=ProtenixPredictionError,
            timeout_s=timeout_s,
            seed=seed_list[0] if seed_list else None,
        )
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
        extra_chains: Sequence[dict[str, Any]] | None,
        msa: str | Path | None,
        use_msa: bool,
        model_name: str | None,
        seeds: Sequence[int] | None,
        n_cycle: int | None,
        n_step: int | None,
        n_sample: int,
        dtype: str,
        need_atom_confidence: bool,
        results_dir: str | Path | None,
        msa_server_mode: str = "protenix",
    ) -> dict[str, Any]:
        """Validate the inputs and assemble the driver request.

        The checks here are the ones worth making *before* committing a GPU to a
        prediction that would fail on the same grounds twenty minutes in.
        """
        if not protein_sequence.strip():
            raise ProtenixPredictionError("protein_sequence is empty; Protenix folds from sequence")
        if ligand_smiles and ligand_ccd:
            raise ProtenixPredictionError(
                "supply at most one of ligand_smiles or ligand_ccd "
                f"(got smiles={ligand_smiles!r}, ccd={ligand_ccd!r})"
            )
        if n_sample < 1:
            raise ProtenixPredictionError(f"n_sample must be at least 1, got {n_sample}")
        selected = model_name or self.model_name
        if selected != self.model_name and not (self.root_dir / f"checkpoint/{selected}.pt").is_file():
            raise ProtenixPredictionError(
                f"checkpoint {self.root_dir / 'checkpoint' / (selected + '.pt')} is not cached; "
                f"fetch it with `python scripts/fetch_protenix_weights.py --model {selected}`"
            )

        chain: dict[str, Any] = {"kind": "protein", "sequence": protein_sequence.strip(), "count": 1}
        if msa is not None:
            chain["msa"] = str(msa)
        sequences: list[dict[str, Any]] = [chain, *(dict(c) for c in extra_chains or ())]

        ligands: list[dict[str, Any]] = []
        if ligand_smiles:
            ligands.append({"ligand": ligand_smiles, "count": 1})
        elif ligand_ccd:
            ligands.append({"ligand": f"CCD_{ligand_ccd.strip().upper()}", "count": 1})

        # An MSA search is the one thing on the run path that leaves the machine,
        # and it happens in a subprocess. Ask the policy here, where it can still
        # refuse, rather than discovering the request in a packet capture. A
        # precomputed alignment searches nothing and is never checked.
        searches = bool(use_msa) and msa is None
        if searches:
            server = MSA_SERVER_URLS.get(msa_server_mode)
            if server is None:
                raise ProtenixPredictionError(
                    f"unknown msa_server_mode {msa_server_mode!r}; expected one of {sorted(MSA_SERVER_URLS)}"
                )
            self.egress.check(server)

        cycle, step = self.default_schedule()
        return {
            "name": _job_name(candidate_id),
            "sequences": sequences,
            "ligands": ligands,
            "root_dir": str(self.root_dir),
            "model_name": selected,
            "seeds": [int(s) for s in (seeds or (101,))],
            "n_cycle": int(n_cycle if n_cycle is not None else cycle),
            "n_step": int(n_step if n_step is not None else step),
            "n_sample": int(n_sample),
            "dtype": dtype,
            "trimul_kernel": self.trimul_kernel,
            "triatt_kernel": self.triatt_kernel,
            "layernorm_type": self.layernorm_type,
            "enable_fusion": self.enable_fusion,
            "enable_cache": self.enable_cache,
            "enable_tf32": self.enable_tf32,
            # False is the offline default. True makes preprocess_input post the
            # sequence to Protenix's MSA server; a precomputed `msa` path avoids
            # both, because upstream then finds nothing to search for.
            "use_msa": bool(use_msa) or msa is not None,
            "msa_server_mode": msa_server_mode,
            "need_atom_confidence": bool(need_atom_confidence),
            "results_dir": str(results_dir) if results_dir else None,
            "allow_download": False,
        }

    def _parse(
        self,
        payload: dict[str, Any],
        *,
        candidate_id: str,
        receptor_id: str | None,
        kept_results_dir: str | None,
    ) -> ProtenixPrediction:
        """Turn the driver's payload into ranked poses and confidence."""
        samples = payload.get("samples") or []
        engine_version = str(payload.get("engine_version") or "unknown")
        used_msa = bool(payload.get("use_msa"))
        warnings = [str(w) for w in payload.get("warnings") or []]

        poses: list[Pose] = []
        # ``rank()`` renumbers ``Pose.rank``, so the link back to the diffusion
        # sample it came from has to be held here rather than read off the pose.
        confidence_of: dict[int, dict[str, Any]] = {}
        for sample in samples:
            raw = sample.get("confidence") or {}
            scalars = _scalars(raw)
            flags = [f"{key}={scalars[key]:.4f}" for key in _POSE_FLAG_KEYS if key in scalars]
            flags.append("msa=used" if used_msa else "msa=single_sequence")
            coordinates = sample.get("coordinates")
            if not coordinates:
                flags.append("no_coordinates_returned")
            if "ranking_score" not in scalars:
                flags.append("missing_ranking_score")
            if scalars.get("has_clash"):
                flags.append("protenix_clash_detected")
            pose = Pose(
                candidate_id=candidate_id,
                receptor_id=receptor_id,
                rank=int(sample.get("rank", len(poses))),
                score=scalars.get("ranking_score"),
                score_unit=self.score_unit,
                pose_ref=coordinates,
                engine=self.engine,
                engine_version=engine_version,
                trust_label=self.trust_label,
                qc_flags=flags,
            )
            confidence_of[id(pose)] = raw
            poses.append(pose)

        poses = self.rank(poses)
        # The best-ranked sample's numbers, not those of whichever file was read first.
        best_raw = confidence_of.get(id(poses[0]), {}) if poses else {}

        return ProtenixPrediction(
            candidate_id=candidate_id,
            poses=poses,
            confidence=_scalars(best_raw),
            chain_confidence={k: v for k, v in best_raw.items() if k not in _SCALAR_CONFIDENCE},
            model_name=str(payload.get("model_name") or self.model_name),
            engine_version=engine_version,
            used_msa=used_msa,
            n_samples=len(samples),
            seconds=_float_or_none(payload.get("seconds")),
            results_dir=kept_results_dir,
            qc_flags=[] if used_msa else ["single_sequence_mode"],
            warnings=warnings,
        )


def _configured(settings: Any, attr: str) -> str | None:
    """One flat settings attribute, or ``None``. Kept in one place for both lookups."""
    value = getattr(settings, attr, None) if settings is not None else None
    return str(value) if value else None


def _job_name(candidate_id: str) -> str:
    """A filesystem-safe stem for one job; Protenix names every output path from it."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in candidate_id.strip())
    return safe[:64] or "complex"


def _scalars(confidence: dict[str, Any]) -> dict[str, float]:
    """The finite scalar confidence fields, ignoring the per-chain matrices."""
    out: dict[str, float] = {}
    for key in _SCALAR_CONFIDENCE:
        value = _float_or_none(confidence.get(key))
        if value is not None:
            out[key] = value
    return out


def _float_or_none(value: Any) -> float | None:
    """A finite float, or ``None``.

    Booleans are converted rather than rejected: upstream serialises ``has_clash``
    as a JSON boolean, and dropping it would lose the one flag that says a sample's
    atoms overlap. ``NaN`` becomes ``None`` — the confidence head does produce NaN
    (a single-sequence run is one way), and a NaN that reached ``Pose.score`` would
    sort unpredictably and print as a number.
    """
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = [
    "DEFAULT_ENABLE_CACHE",
    "DEFAULT_ENABLE_FUSION",
    "DEFAULT_ENABLE_TF32",
    "DEFAULT_LAYERNORM_TYPE",
    "DEFAULT_MODEL",
    "DEFAULT_SCHEDULES",
    "DEFAULT_TRIATT_KERNEL",
    "DEFAULT_TRIMUL_KERNEL",
    "PROTENIX_ADAPTER_VERSION",
    "PROTENIX_CHECKPOINT_BYTES",
    "MSA_SERVER_URLS",
    "PROTENIX_DATA_CACHE",
    "SPEC",
    "ProtenixAdapter",
    "ProtenixPrediction",
    "ProtenixPredictionError",
]
