"""PILOT — pocket-conditioned equivariant diffusion steered by importance sampling.

PILOT is Pfizer's structure-based generator: a discrete-and-continuous equivariant
denoising diffusion model that samples ligand atom positions, types, charges and
bonds inside a protein pocket, and then re-weights the diffusion trajectory towards
molecules a second network scores well. That re-weighting is the whole point of the
model. Unguided, it is one more pocket-conditioned diffusion sampler. Guided, it is
a multi-objective sampler that trades a little pocket fit for synthetic accessibility
or predicted potency. The two are different experiments and this adapter refuses to
let them share a label.

**Which upstream.** ``e3moldiffusion`` is a family. The same code name appears under
several owners with different heads, different property models and different
defaults, so "PILOT" alone does not identify a model. This adapter is written
against ``pfizer-opensource/e3moldiffusion`` at commit ``f5e9bb8a77b6`` — the
repository whose README is titled *"PILOT: Equivariant diffusion for pocket
conditioned de novo ligand generation with multi-objective guidance via importance
sampling"* and whose paper is Cremer et al., *Chem. Sci.* **15**, 14954-14967 (2024),
DOI 10.1039/D4SC03523B. ``SPEC.upstream_ref`` pins that commit and every driver
argument below was read out of that tree. A different variant would need a different
driver: the argument names, the ``Trainer`` class path and the property-head wiring
are not stable across forks.

**Guidance is a recorded parameter, not a default.** :meth:`generate` takes an
explicit ``guidance`` argument. ``None`` means plain pocket-conditioned diffusion.
A :class:`PilotGuidance` names the objectives — ``sa_score``, ``docking_score``,
``ic50`` — and the checkpoints that score them. Either way the batch config carries
``guidance`` and a boolean ``guided``, always written, never inferred, so an
unguided sample can never be read back as a guided one. It records checkpoint paths
and not only objective names for a specific reason: in
``experiments/diffusion_discrete_pocket.py``, ``reverse_sampling`` selects
``self.model`` when ``sa_model is None``, so asking for SA guidance without
``ckpt_sa_model`` does not fail — it silently guides with whatever property head the
diffusion checkpoint itself carries. Which head did the steering is a fact about the
result, so it goes in the record.

**This capability is UNAVAILABLE on this machine and generates nothing.** The
blockers, each verified against the checkout:

1. **No diffusion checkpoint.** The run path is
   ``torch.load(model_path)["hyper_parameters"]`` followed by
   ``Trainer.load_from_checkpoint(model_path, ...)``
   (``experiments/generate_ligands_multi_batch.py``, ``evaluate()``). ``find`` over
   ``.external/e3moldiffusion`` returns no ``*.ckpt`` at all: the clone is code only.
   The file that would have to exist is
   ``.external/e3moldiffusion/checkpoints/best_valid.ckpt`` — ``best_valid.ckpt`` is
   the upstream README's own name for it ("model-path: Set the path to the trained
   model (normally save_dir/best_valid.ckpt)"). It cannot be downloaded: under
   *Model checkpoints* the README says, verbatim, "Currently, we provide the model
   weights upon request. Please contact us via email", and gives
   ``julian.cremer@pfizer.com`` and ``tuan.le@pfizer.com``. There is no URL to fetch
   and no mirror in this repository.

2. **``torch_sparse`` is not installed.** ``importlib.util.find_spec("torch_sparse")``
   returns ``None`` in the ``drug-agent`` environment. It is not optional here:
   ``experiments/utils.py`` opens with ``from torch_sparse import coalesce``,
   ``experiments/molecule_utils.py`` does the same, and
   ``e3moldiffusion/coordsatomsbonds.py`` does ``from torch_sparse import
   SparseTensor`` — all three are imported before any sampling begins.
   ``torch_scatter`` and ``torch_cluster``, from the same PyG extension family, *are*
   present on this machine; ``torch_sparse`` is a compiled extension that was never
   built against this environment's torch.

3. **No processed dataset root.** ``evaluate()`` cannot reach the checkpoint until it
   has built a ``LigandPocketDataModule``, derived ``GeneralInfos`` from it, loaded
   ``os.path.join(hparams.dataset_root, "size_distribution.npy")`` and read
   ``datamodule.test_dataset.smiles`` — all three are constructor arguments to
   ``Trainer.load_from_checkpoint``. The checked-in ``configs/diffusion_crossdocked.yaml``
   points ``dataset_root`` at an upstream machine's private dataset directory,
   an author's scratch filesystem. Building it means downloading CrossDocked2020 and
   running ``experiments/data/ligand/process_crossdocked.py``.

4. **A pocket PDB is not a PILOT input.** ``prepare_pocket_cutoff`` takes Biopython
   residue objects selected by a residue-id list, so the model wants a *triple*:
   ``<pdb>.pdb``, a reference ``<ligand>.sdf`` for the pocket geometry and ligand-size
   prior, and ``<ligand>.txt`` holding space-separated ``chain:resnum`` ids, all
   produced by ``experiments/data/ligand/process_pdb.py``. This repository's
   ``.external/pockets/seh_3i28_pocket10.pdb`` is a cut pocket with no companion
   ``.txt`` or reference ``.sdf``, so it is not directly usable even once 1-3 are
   fixed.

**What the platform does instead.** Nothing. ``available()`` is false, ``backend`` is
``"none"``, and :meth:`generate` raises
:class:`~drug_agent.tools.base.ToolUnavailable` through ``require_available()``
before any request is built. There is no surrogate sampler, no cached batch and no
stand-in molecule. Per :mod:`drug_agent.tools.base` that is the only alternative to
running the real thing.

**Read validity and ring statistics before believing anything.** PILOT is a
point-cloud model: the network emits coordinates and discrete types, and
``experiments/molecule_utils.Molecule`` rebuilds the graph — ``build_molecule()``
from the predicted bond matrix, ``build_molecule_openbabel()`` from coordinates
alone. Reconstruction, not chemistry, is where these models fail, and the failure is
invisible in a SMILES list. The measured baseline in this repository is PFM: of 40
generated molecules, 21 contained an allene and 21 a four-membered ring, leaving 2
chemically ordinary molecules, and all 40 fell outside the activity model's
applicability domain. Before reading a PILOT batch as a result, look at
``generation_validity``, ``generation_uniqueness``, and the allene and small-ring
rates over the returned SMILES. A guided run that improves a predicted objective
while its ring statistics collapse has not found better molecules; it has found
better-scoring reconstruction artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...contracts.brief import PocketSpec
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with PILOT, not PILOT's own version.
PILOT_ADAPTER_VERSION = "1.0.0"

#: Objectives PILOT's importance sampler can steer towards. These are the values
#: ``experiments/diffusion_discrete_pocket.py`` branches on (``kind == "sa_score"``,
#: ``kind == "docking_score" or kind == "ic50"``) and the values the shipped configs
#: put in ``regression_property`` (``[sa_score]`` for CrossDocked, ``[sa_score,
#: ic50]`` for Kinodata-3D). Anything else is rejected rather than passed through.
GUIDANCE_OBJECTIVES: tuple[str, ...] = ("sa_score", "docking_score", "ic50")

#: The objective served by PILOT's SA channel. The other two ride the property
#: channel, which is why the two channels have separate schedules and checkpoints.
_SA_OBJECTIVE = "sa_score"

SPEC = ExternalSpec(
    name="PILOT",
    env_prefix="PILOT",
    checkout_relpath=".external/e3moldiffusion",
    # Every one of these is imported by the sampling path, not merely present:
    # `Trainer` lives in diffusion_discrete_pocket, `prepare_data_and_generate_ligands`
    # in experiments/utils, the reconstruction in experiments/molecule_utils, the
    # `GeneralInfos` the checkpoint needs in experiments/data/data_info, and the
    # network itself in e3moldiffusion/coordsatomsbonds.
    sentinel_files=(
        "experiments/diffusion_discrete_pocket.py",
        "experiments/utils.py",
        "experiments/molecule_utils.py",
        "experiments/data/data_info.py",
        "e3moldiffusion/coordsatomsbonds.py",
    ),
    driver_name="pilot_sample.py",
    # Narrow enough to identify the environment PILOT was installed into; readiness
    # then rejects that environment module by module below.
    selection_modules=("torch", "torch_geometric"),
    # What a broken install actually loses, read off the imports on the run path.
    # `torch_sparse` is the one that is missing here, and it is a compiled extension
    # that has to be built against the installed torch — the same failure PFM's spec
    # exists to catch. `e3moldiffusion` itself is deliberately absent: it is never
    # installed into site-packages, it is put on PYTHONPATH from the checkout, and
    # the sentinel files above are what prove it is there.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "pytorch_lightning",
        "torchmetrics",
        "Bio",
        "openbabel",
        "rdkit",
    ),
    # README, "Model checkpoints": weights are released by email on request, so this
    # is the path the operator would drop the reply into. `best_valid.ckpt` is
    # upstream's own name for the file every sampling script passes to --model-path.
    weight_relpaths=("checkpoints/best_valid.ckpt",),
    conda_envs=("e3mol", "drug-agent"),
    upstream_url="https://github.com/pfizer-opensource/e3moldiffusion",
    upstream_ref="f5e9bb8a77b6",
    license=(
        "Apache-2.0 (LICENSE at the checkout root). The model weights are not "
        "covered by that file and are not present: upstream releases them by email "
        "on request."
    ),
    install_hint=(
        "make external-clone; then request the PILOT checkpoints from the addresses "
        "in the upstream README and place best_valid.ckpt at "
        ".external/e3moldiffusion/checkpoints/"
    ),
    needs_gpu=True,
)


@dataclass(frozen=True)
class PilotGuidance:
    """Which objectives steer the diffusion trajectory, and with what schedule.

    The mechanism, read off ``reverse_sampling`` in
    ``experiments/diffusion_discrete_pocket.py``: at selected timesteps the property
    network scores every partially denoised molecule in the batch, the scores become
    weights via ``(score / tau).softmax(dim=0)``, and the batch is then resampled
    with ``torch.multinomial(weights, num_samples=len(weights), replacement=True)``.
    It is importance resampling, not a gradient — and because the resampling is with
    replacement, a low ``tau`` collapses the batch onto a few trajectories. Diversity
    is the cost of guidance, and ``generation_uniqueness`` is where that cost shows.

    There are two independent channels, and upstream gives each its own checkpoint,
    temperature, window and stride:

    ``sa`` (``--sa-importance-sampling``)
        Synthetic accessibility. Always maximised.
    ``property`` (``--property-importance-sampling``)
        Whatever the property checkpoint regresses — ``docking_score`` or ``ic50``.
        ``maximize`` picks the direction; upstream spells it as the negated
        ``--minimize-property``.

    ``joint`` is upstream's ``--joint-importance-sampling``, which re-weights on both
    channels together rather than alternating them.

    The defaults for the windows and temperatures are the ones written into
    ``scripts/generate_ligands_multi_pdb_file.sl``: ``start 0``, ``end 200``,
    ``every 5``, ``tau 0.1``. A window is expressed in diffusion timesteps, counted
    the way upstream counts them.
    """

    #: A subset of :data:`GUIDANCE_OBJECTIVES`. Empty means unguided.
    objectives: tuple[str, ...] = ()
    #: Property network scoring ``sa_score``. See the module docstring on why leaving
    #: this ``None`` does not disable guidance upstream.
    sa_checkpoint: str | None = None
    #: Property network scoring ``docking_score`` / ``ic50``.
    property_checkpoint: str | None = None
    sa_tau: float = 0.1
    sa_every_t: int = 5
    sa_start: int = 0
    sa_end: int = 200
    property_tau: float = 0.1
    property_every_t: int = 5
    property_start: int = 0
    property_end: int = 200
    #: Direction for the property channel. Upstream's flag is ``--minimize-property``.
    maximize_property: bool = True
    joint: bool = False
    normalize_property: bool = False

    def __post_init__(self) -> None:
        unknown = [obj for obj in self.objectives if obj not in GUIDANCE_OBJECTIVES]
        if unknown:
            raise ValueError(
                f"unknown PILOT guidance objective(s) {unknown}; "
                f"upstream branches only on {list(GUIDANCE_OBJECTIVES)}"
            )

    @property
    def guides_sa(self) -> bool:
        return _SA_OBJECTIVE in self.objectives

    @property
    def guides_property(self) -> bool:
        return any(obj != _SA_OBJECTIVE for obj in self.objectives)

    def as_request(self) -> dict[str, Any]:
        """The driver-side view: upstream's own flag names, one per key.

        Each key here maps onto exactly one argument of ``generate_ligands`` in
        ``experiments/diffusion_discrete_pocket.py``, so the driver never has to
        invent a translation.
        """
        return {
            "objectives": list(self.objectives),
            "sa_importance_sampling": self.guides_sa,
            "ckpt_sa_model": self.sa_checkpoint,
            "sa_tau": float(self.sa_tau),
            "sa_every_importance_t": int(self.sa_every_t),
            "sa_importance_sampling_start": int(self.sa_start),
            "sa_importance_sampling_end": int(self.sa_end),
            "property_importance_sampling": self.guides_property,
            "ckpt_property_model": self.property_checkpoint,
            "property_tau": float(self.property_tau),
            "property_every_importance_t": int(self.property_every_t),
            "property_importance_sampling_start": int(self.property_start),
            "property_importance_sampling_end": int(self.property_end),
            "minimize_property": not self.maximize_property,
            "joint_importance_sampling": bool(self.joint),
            "property_normalization": bool(self.normalize_property),
        }


#: What ``config["guidance"]`` says when :meth:`PilotAdapter.generate` was called
#: without a :class:`PilotGuidance`. Written explicitly on every batch so that
#: "unguided" is a recorded fact rather than the absence of a key.
_UNGUIDED: dict[str, Any] = {"objectives": [], "mode": "unguided"}


@dataclass(frozen=True)
class _PocketInputs:
    """The file triple PILOT actually consumes for one pocket."""

    test_dir: Path
    ligand_stem: str
    pdb_path: Path
    sdf_path: Path
    residues_txt: Path


class PilotAdapter(GenerativeToolAdapter):
    """Pocket-conditioned diffusion with multi-objective importance-sampling guidance."""

    capability_id = "molecule.generate.pilot"
    spec = SPEC
    engine = "pilot"
    version = PILOT_ADAPTER_VERSION
    conditioning = "pocket"
    synthesis_aware = False

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s)
        # Optional, and normally left unset: upstream's --dataset-root is documented
        # as "If not set it will be taken from the model ckpt". The override exists
        # because a checkpoint trained on an author's scratch filesystem carries a
        # dataset_root that does not exist anywhere else.
        self.dataset_root = str(dataset_root) if dataset_root else None
        self._guidance: PilotGuidance | None = None
        self._driver_seconds: float | None = None

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned diffusion with property guidance (PILOT)",
            description=(
                "Equivariant denoising diffusion over ligand coordinates, atom types, "
                "charges and bonds inside a protein pocket, optionally steered by "
                "trajectory importance sampling towards synthetic accessibility and a "
                "predicted potency objective. Proposes molecules; scores nothing."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "bond perception over the generated point cloud invents allenes and "
                "four-membered rings; the SMILES parse cleanly and the chemistry is wrong",
                "reconstruction can also fail outright, returning fragments or a "
                "disconnected multi-component molecule for one sample",
                "guidance requested with no property checkpoint does not raise: "
                "reverse_sampling falls back to the diffusion model's own property head",
                "guidance resamples the batch with replacement, so a low tau collapses "
                "it onto a few trajectories and uniqueness falls",
                "the ligand-size prior is sampled from the training size distribution, "
                "so a pocket unlike CrossDocked2020 gets molecules of the wrong size",
                "the pocket must arrive as a pdb/sdf/txt triple from process_pdb.py; "
                "a bare pocket PDB has no residue-id list and cannot be featurised",
                "batch_size drives GPU memory directly — upstream's own guidance is "
                "40-50 molecules per batch on a V100",
            ],
            validation_checks=[
                "validity and uniqueness are computed by the domain base from raw "
                "sample counts, never reported by the backend",
                "every returned molecule is RDKit-parseable and inside the heavy-atom window",
                "batch config records guided/unguided, the objective list and the "
                "checkpoint that scored each channel",
                "batch config records the pinned upstream commit the driver was written against",
            ],
        )

    # --- operations --------------------------------------------------------
    def generate(  # type: ignore[override]
        self,
        *,
        n_molecules: int,
        pocket: PocketSpec | dict[str, Any] | None = None,
        context_smiles: list[str] | None = None,
        seed: int = 0,
        batch_size: int = 10,
        min_heavy_atoms: int = 10,
        max_heavy_atoms: int = 70,
        keep_invalid: bool = False,
        max_runtime_seconds: float | None = None,
        guidance: PilotGuidance | None = None,
    ) -> GenerationBatch:
        """Sample ligands for ``pocket``, guided by ``guidance`` or not at all.

        ``guidance=None`` is plain pocket-conditioned diffusion — the unguided arm of
        the comparison PILOT exists to support. Whichever it was, the returned
        ``batch.config`` carries ``guided`` and ``guidance``, so the two arms stay
        distinguishable once the batch has travelled downstream and only its config
        survives.
        """
        self._guidance = guidance
        try:
            batch = super().generate(
                n_molecules=n_molecules,
                pocket=pocket,
                context_smiles=context_smiles,
                seed=seed,
                batch_size=batch_size,
                min_heavy_atoms=min_heavy_atoms,
                max_heavy_atoms=max_heavy_atoms,
                keep_invalid=keep_invalid,
                max_runtime_seconds=max_runtime_seconds,
            )
        finally:
            self._guidance = None
        # Stamped here rather than inside the base's config builder because the base
        # is shared by nineteen generators and importance-sampling guidance is PILOT's
        # alone. Both keys are always written: a missing key would make "unguided" and
        # "nobody recorded it" the same reading.
        batch.config["guided"] = guidance is not None and bool(guidance.objectives)
        batch.config["guidance"] = guidance.as_request() if guidance else dict(_UNGUIDED)
        return batch

    # --- real backend ------------------------------------------------------
    def _sample(
        self,
        *,
        n_molecules: int,
        pocket: dict[str, Any],
        context_smiles: list[str],
        seed: int,
        batch_size: int,
        timeout_s: float,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Run the PILOT driver once and hand its records back to the base class."""
        inputs = self._pocket_inputs(pocket)
        guidance = self._guidance
        request: dict[str, Any] = {
            "model_path": str(self._checkpoint()),
            "dataset_root": self.dataset_root,
            "test_dir": str(inputs.test_dir),
            "ligand_stem": inputs.ligand_stem,
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            # Upstream's --max-sample-iter: how many times it may re-sample the pocket
            # before giving up on reaching the requested count.
            "max_sample_iter": 20,
            "seed": int(seed),
            "device": self.ext.device,
            "guidance": guidance.as_request() if guidance else dict(_UNGUIDED),
        }
        if context_smiles:
            # PILOT conditions on a pocket, not on a reference SMILES. Saying so is
            # better than quietly dropping the argument.
            warning = f"PILOT is pocket-conditioned; {len(context_smiles)} context SMILES were ignored"
        else:
            warning = ""

        run = self.run_driver(request, error_cls=GenerationBackendError, seed=seed, timeout_s=timeout_s)
        self._driver_seconds = float(run.seconds)
        payload = run.payload
        records = [rec for rec in (payload.get("molecules") or []) if isinstance(rec, dict)]
        warnings = [str(w) for w in (payload.get("warnings") or [])]
        errors = [str(e) for e in (payload.get("errors") or [])]
        if warning:
            warnings.append(warning)
        failures = int(payload.get("n_reconstruction_failures") or 0)
        if failures:
            warnings.append(f"{failures} samples could not be rebuilt into an RDKit molecule")
        return records, warnings, errors

    def _meter(self, metered: dict[str, float], batch: GenerationBatch, *, n_molecules: int) -> None:
        """The base's counts, plus how long the subprocess itself really took.

        Separating driver wall time from the adapter's own is the only way to tell
        "the diffusion model was slow" from "we were slow assembling its output" once
        a run is over and only the ledger is left.
        """
        super()._meter(metered, batch, n_molecules=n_molecules)
        if self._driver_seconds is not None:
            metered["pilot_driver_seconds"] = float(self._driver_seconds)
        self._driver_seconds = None

    # --- inputs ------------------------------------------------------------
    def _checkpoint(self) -> Path:
        """The diffusion checkpoint every sampling script passes to ``--model-path``."""
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() has already refused
            raise GenerationBackendError("PILOT checkout is not resolved")
        return Path(root) / SPEC.weight_relpaths[0]

    def _pocket_inputs(self, pocket: dict[str, Any]) -> _PocketInputs:
        """Resolve the pdb/sdf/txt triple ``prepare_pocket_cutoff`` needs.

        ``experiments/generate_ligands_multi_batch.py`` derives all three from one
        directory: it globs ``<test_dir>/*.sdf``, splits the stem on ``_`` to get the
        PDB name, and opens ``<stem>.txt`` for the residue ids. This mirrors that
        derivation from a :class:`PocketSpec` whose ``pdb_path`` points into such a
        directory, and says exactly which file is missing when one is.
        """
        raw = pocket.get("pdb_path")
        if not raw:
            raise ValueError(
                "PILOT needs pocket.pdb_path pointing at a PDB inside a directory "
                "prepared by experiments/data/ligand/process_pdb.py"
            )
        pdb_path = Path(raw)
        test_dir = pdb_path.parent
        pdb_name = pdb_path.stem
        candidates = sorted(test_dir.glob(f"{pdb_name}_*.sdf"))
        if not candidates:
            raise ValueError(
                f"no reference ligand {pdb_name}_*.sdf beside {pdb_path}; PILOT takes the "
                "pocket geometry and its ligand-size prior from that file "
                "(experiments/data/ligand/process_pdb.py writes it)"
            )
        sdf_path = candidates[0]
        residues_txt = test_dir / f"{sdf_path.stem}.txt"
        if not residues_txt.is_file():
            raise ValueError(
                f"no pocket residue list at {residues_txt}; it holds the space-separated "
                "chain:resnum ids prepare_pocket_cutoff selects on"
            )
        return _PocketInputs(
            test_dir=test_dir,
            ligand_stem=sdf_path.stem,
            pdb_path=pdb_path,
            sdf_path=sdf_path,
            residues_txt=residues_txt,
        )


__all__ = [
    "GUIDANCE_OBJECTIVES",
    "PILOT_ADAPTER_VERSION",
    "SPEC",
    "PilotAdapter",
    "PilotGuidance",
]
