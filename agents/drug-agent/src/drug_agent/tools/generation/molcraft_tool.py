"""MolCRAFT — pocket-conditioned 3D generation in a continuous parameter space.

MolCRAFT (ICML'24) is a structure-based generator built on a Bayesian flow
network rather than a diffusion process. Where a diffusion model denoises samples,
a BFN updates the *parameters* of the atom distributions — coordinates and types
alike — so the continuous and the discrete modality are carried in one continuous
space instead of two coupled ones. Its own README claims "reference-level Vina
Scores" and "more than 10x speedup" from the variance-reduced sampler; those are
the upstream project's numbers, measured on CrossDocked2020, not measurements made
here.

**One repository, three methods.** ``.external/MolCRAFT`` is the MolCRAFT *series*
checkout: the same clone carries MolCRAFT, MolJO and MolPilot, and the catalogue
treats each as a distinct method with a distinct checkpoint. This adapter binds
MolCRAFT alone, identified by the entry point and config it runs:

``MolCRAFT``
    ``MolCRAFT/sample_for_pocket.py``, function ``call()`` (line 98). It builds
    ``core.models.sbdd_train_loop.SBDDTrainLoop`` over
    ``core.models.bfn4sbdd.BFN4SBDDScoreModel``, reads its configuration from
    ``./checkpoints/config.yaml`` (line 102) — a dump of
    ``MolCRAFT/configs/default.yaml`` — and loads ``MolCRAFT/checkpoints/last.ckpt``.
``MolJO``
    ``MolJO/sample_for_pocket_guided.py`` with ``MolJO/configs/test_opt.yaml``,
    ``core.models.train_loop.BFNTrainLoop`` plus ``ClassifierTrainLoop``. It needs
    the Vina-score and SA property-predictor checkpoints under ``MolJO/pretrained/``
    in addition to the backbone, because it is gradient-guided optimisation, not
    unconditional pocket sampling.
``MolPilot``
    ``MolPilot/sample_for_pocket.py``, whose ``--config_file`` defaults to
    ``configs/molpilot_config.yaml`` (the clone ships ``crossdock_train_test.yaml``
    and two ``posebusters_*`` configs instead). It builds
    ``core.models.sbdd4train_timewarp.SBDD4Train`` and loads a
    ``--time_scheduler_path`` tensor alongside ``--ckpt_path`` (line 371). Its
    schedule is the method; a MolCRAFT checkpoint has no schedule to load.

None of the three checkpoints substitutes for another, and running MolCRAFT's
entry point against MolJO's or MolPilot's weights is a state-dict mismatch, not a
degraded result.

**Why it cannot run here.** Four blockers, each verified against the checkout at
``743dfb5198c4``:

1. ``.external/MolCRAFT/MolCRAFT/checkpoints/last.ckpt`` does not exist. The clone
   is code-only. Upstream ships the file two ways: ``MolCRAFT/scripts.mk`` target
   ``checkpoint`` fetches Google Drive id ``1a1laBFYRNqaMpcS3Id0L0R6XoLEk4gDG``
   into ``checkpoints/last.ckpt``, and ``MolCRAFT/README.md`` links
   ``1TcUQM7Lw1klH2wOVBu20cTsvBTcC1WKu`` as "the pretrained MolCRAFT checkpoint".
   The series README also carries a Hugging Face badge for
   ``https://huggingface.co/GenSI/MolCRAFT/``.
2. ``.external/MolCRAFT/MolCRAFT/checkpoints/config.yaml`` does not exist and
   nothing in the checkout creates it. ``sample_for_pocket.py`` line 102 reads that
   exact relative path before anything else; ``train_bfn.py`` writes its config
   dump to ``logs/<project>/<exp>/<revision>/config.yaml`` instead, and the
   ``checkpoint`` target in ``scripts.mk`` downloads only the weights. The file has
   to be placed beside ``last.ckpt`` by hand, which is why it is listed as a
   required artefact rather than assumed to arrive with the download.
3. The interpreter cannot import seven modules the run path imports at module
   scope, each named here with the upstream file that imports it: ``posecheck``
   (``sample_for_pocket.py:37``, ``core/evaluation/metrics.py:17``), ``vina``
   (``core/evaluation/docking_vina.py:4``), ``AutoDockTools``
   (``core/evaluation/docking_vina.py:9``), ``absl`` (``core/callbacks/basic.py:12``),
   ``fire`` (``core/config/config.py:4``), ``overrides``
   (``core/callbacks/basic.py:8``) and ``wandb``
   (``core/callbacks/validation_callback.py:16``). These are import-time costs of
   reaching the sampler, not of docking: the test callback never calls
   ``CondMolGenMetric.compute_*``. ``vina`` in particular is declared in no MolCRAFT
   dependency file — neither ``docker/asset/requirements.txt`` nor
   ``docker/environment.yml`` mentions it — yet the import is unconditional.
4. MolCRAFT clips its own pocket, so a pocket PDB alone is not a sufficient input.
   ``get_dataloader_from_pdb`` asserts both ``protein_path`` and ``ligand_path``
   (line 43) and derives the site from the reference ligand's residue neighbourhood
   at ``r_max``. :class:`~drug_agent.contracts.brief.PocketSpec` has no field for a
   reference ligand, so the pocket dict must carry one under ``ref_ligand_sdf``
   (``reference_ligand_sdf``, ``ligand_path`` and ``ligand_sdf_path`` are accepted
   spellings). A batch requested without one comes back empty with that reason in
   ``errors``.

**What the driver does when those are fixed.** ``scripts/external/molcraft_sample.py``
changes into the MolCRAFT project directory — mandatory, because ``call()``'s config
path is relative — and calls ``sample_for_pocket.call()`` once per batch, then reads
the SDF files the ``DockingTestCallback`` writes into ``./output`` (``OUT_DIR``,
``core/callbacks/validation_callback_for_sample.py:49``). Two properties of that path
are carried into the batch rather than smoothed over:

* ``on_test_epoch_end`` writes only the samples that survive
  ``reconstruct_from_generated``, ``Chem.SanitizeMol`` and a fragment check, so the
  files on disk are the survivors, not the samples. Counting only files would make
  ``generation_validity`` 1.0 by construction. The adapter therefore pads the record
  list up to the number of samples requested, each pad flagged
  ``upstream_reconstruction_failed``, so ``n_raw`` is the real denominator and
  validity is survivors over samples.
* ``call()`` calls ``seed_everything(cfg.seed)`` at line 103 with the seed baked into
  the checkpoint's own config, and assigns its ``seed`` argument at line 116, after
  which nothing reads it. The requested seed is recorded on the batch config and a
  warning says it did not reach the sampler. Sampling is stochastic regardless, which
  is why ``determinism`` is STOCHASTIC.

Device selection has no lever either: ``call()`` constructs ``pl.Trainer(devices=1)``
with Lightning's default accelerator, so ``DRUGAGENT_MOLCRAFT_DEVICE`` cannot steer
it and ``CUDA_VISIBLE_DEVICES`` — which the driver environment allowlist passes
through — is the only control.

**What the platform does instead.** Nothing. The capability reports UNAVAILABLE,
``generate`` raises :class:`~drug_agent.tools.base.ToolUnavailable`, and no molecule
is produced under MolCRAFT's name. Under :mod:`drug_agent.tools.base` there is no
surrogate sampler and no fallback generator: a pocket with no model is an absent arm
of the run, visibly absent, not an arm filled with plausible SMILES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with MolCRAFT, not MolCRAFT's own version.
MOLCRAFT_ADAPTER_VERSION = "1.0.0"

#: The MolCRAFT project inside the series checkout. MolJO and MolPilot are sibling
#: directories with their own ``core`` packages, so this is also what goes on
#: PYTHONPATH: importing ``core.models.sbdd_train_loop`` from the repository root
#: would resolve to whichever sibling landed on the path first.
PROJECT_SUBDIR = "MolCRAFT"

#: Pocket-dict keys searched for the reference ligand MolCRAFT clips its site from.
_LIGAND_KEYS: tuple[str, ...] = (
    "ref_ligand_sdf",
    "reference_ligand_sdf",
    "ligand_path",
    "ligand_sdf_path",
)

SPEC = ExternalSpec(
    name="MolCRAFT",
    env_prefix="MOLCRAFT",
    checkout_relpath=".external/MolCRAFT",
    # Every one of these is imported by the path the driver runs, and every one
    # exists in the clone. The first is the entry point; the next three are the
    # modules that make this MolCRAFT rather than MolJO or MolPilot.
    sentinel_files=(
        "MolCRAFT/sample_for_pocket.py",
        "MolCRAFT/core/models/sbdd_train_loop.py",
        "MolCRAFT/core/models/bfn4sbdd.py",
        "MolCRAFT/core/callbacks/validation_callback_for_sample.py",
        "MolCRAFT/core/config/config.py",
        "MolCRAFT/configs/default.yaml",
    ),
    driver_name="molcraft_sample.py",
    # The stack that identifies a MolCRAFT environment: Lightning drives the
    # sampler, torch_geometric carries the protein-ligand graph. An environment
    # with torch alone is a different tool's environment.
    selection_modules=("torch", "pytorch_lightning", "torch_geometric"),
    # Read off the import statements of the run path, not off requirements.txt.
    # `torch_scatter` compiles from source with no wheel and is what a half-built
    # install loses first; `vina` and `AutoDockTools` are imported by
    # core/evaluation/docking_vina.py even when nothing docks.
    runtime_modules=(
        "torch",
        "pytorch_lightning",
        "torch_geometric",
        "torch_scatter",
        "rdkit",
        "openbabel",
        "meeko",
        "easydict",
        "matplotlib",
        "absl",
        "fire",
        "overrides",
        "wandb",
        "posecheck",
        "vina",
        "AutoDockTools",
    ),
    # Both are read by sample_for_pocket.call(): the weights at line 147 through
    # Trainer.test(ckpt_path=...), the config at line 102 through Config(...).
    weight_relpaths=(
        "MolCRAFT/checkpoints/last.ckpt",
        "MolCRAFT/checkpoints/config.yaml",
    ),
    conda_envs=("drug-agent", "pfm"),
    upstream_url="https://github.com/GenSI-THUAIR/MolCRAFT",
    upstream_ref="743dfb5198c4",
    license="MolCRAFT CC-BY-NC-SA-4.0 (non-commercial; see .external/MolCRAFT/LICENSE)",
    # Kept to one line: the probe repeats it once per failure, and the modules a
    # broken install is missing are already named by the probe's own message.
    install_hint=(
        "make external-clone, then put last.ckpt and config.yaml in "
        ".external/MolCRAFT/MolCRAFT/checkpoints/ (MolCRAFT/scripts.mk target `checkpoint`)"
    ),
    needs_gpu=True,
)


class MolCraftAdapter(GenerativeToolAdapter):
    """Pocket-conditioned 3D sampling with MolCRAFT's Bayesian flow network."""

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Also needs a reference ligand: upstream's loader asserts both paths."""
        return ("pocket_pdb", "ref_ligand_sdf")

    capability_id = "molecule.generate.molcraft"
    spec = SPEC
    engine = "molcraft"
    version = MOLCRAFT_ADAPTER_VERSION
    conditioning = "pocket"
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned 3D generation (MolCRAFT)",
            description=(
                "Samples ligand atom positions and types inside a binding site with a Bayesian "
                "flow network operating on distribution parameters rather than on samples, then "
                "rebuilds a molecular graph from the point cloud. Requires a protein PDB and the "
                "reference ligand it clips the pocket from. Proposes molecules; scores none."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "bond perception over generated coordinates invents allenes and small rings; "
                "reconstruct_from_generated drops those samples, so a batch returns fewer "
                "molecules than requested",
                "sample_num_atoms='prior' draws the atom count from the pocket volume "
                "(core/evaluation/utils/atom_num.py), so an over-wide pocket PDB yields "
                "oversized molecules",
                "get_dataloader_from_pdb asserts a reference ligand as well as a protein; a "
                "pocket with no ligand file cannot be sampled at all",
                "call() re-seeds from the checkpoint's own config.yaml, so the requested seed "
                "does not change the sample",
                "the batch dimension is cfg.evaluation.batch_size from the checkpoint config "
                "(100 in configs/default.yaml), so a large request can exhaust GPU memory "
                "inside Lightning rather than in this process",
            ],
            validation_checks=[
                "every returned molecule is parsed by RDKit here and de-duplicated canonically",
                "n_raw counts every sample requested, including those upstream's reconstruction "
                "dropped, so generation_validity is survivors over samples",
                "heavy-atom count is inside the requested [min_heavy_atoms, max_heavy_atoms] window",
                "the checkpoint loaded is MolCRAFT/checkpoints/last.ckpt, never MolJO's "
                "property-predictor bundle or MolPilot's time-scheduler tensor",
            ],
        )

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
        """Run MolCRAFT's ``call()`` in its own interpreter and read back ``./output``."""
        warnings: list[str] = []
        errors: list[str] = []

        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() rejects this first
            return [], warnings, ["MolCRAFT checkout did not resolve"]

        protein = self._existing(pocket.get("pdb_path"))
        ligand = self._reference_ligand(pocket)
        if protein is None:
            errors.append("pocket.pdb_path is missing or is not a file; MolCRAFT needs a protein PDB")
        if ligand is None:
            errors.append(
                "no reference ligand in the pocket spec: MolCRAFT's get_dataloader_from_pdb clips "
                "the site from a reference ligand and asserts both paths. Supply one under "
                + "/".join(_LIGAND_KEYS)
            )
        if errors:
            return [], warnings, errors

        if context_smiles:
            warnings.append("MolCRAFT is pocket-conditioned only; the supplied context_smiles were not used.")
        warnings.append(
            "MolCRAFT seeds from the checkpoint's own config.yaml (sample_for_pocket.py:103) and "
            f"assigns its seed argument afterwards, so seed={seed} did not reach the sampler."
        )

        request = {
            "project_dir": str(root / PROJECT_SUBDIR),
            "protein_path": str(protein),
            "ligand_path": str(ligand),
            "ckpt_path": str(root / PROJECT_SUBDIR / "checkpoints" / "last.ckpt"),
            "num_samples": int(n_molecules),
            "batch_size": int(batch_size),
            # call()'s own defaults, kept explicit so the request records what ran.
            # `end_back` is call()'s default and a real branch of the sampler
            # (core/models/bfn4sbdd.py:604); train_bfn.py's CLI defaults to
            # `end_back_pmf` instead, and the two are different estimators, so the
            # entry point being called decides which one is right here.
            "sample_steps": 100,
            "sample_num_atoms": "prior",
            "beta1": 1.5,
            "sigma1_coord": 0.03,
            "sampling_strategy": "end_back",
            "seed": int(seed),
        }
        # run_json rather than run_driver: run_driver pins the budget to
        # self.timeout_s, and a brief's max_runtime_seconds has to be able to cap a
        # sampler that runs for minutes per batch.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            extra_pythonpath=(PROJECT_SUBDIR,),
            seed=seed,
        )
        return self._records(run.payload, n_molecules=n_molecules, warnings=warnings, errors=errors)

    # --- parsing -----------------------------------------------------------
    def _records(
        self,
        payload: dict[str, Any],
        *,
        n_molecules: int,
        warnings: list[str],
        errors: list[str],
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Driver payload to raw records, padded to the number of samples requested.

        The padding is the honest part. MolCRAFT writes an SDF only for a sample
        that reconstructed, sanitized and came out as a single fragment, so the
        files are survivors. One record per *requested* sample keeps ``n_raw`` the
        real denominator; the pads carry no SMILES and no coordinates, only the
        fact that the model produced nothing usable for that draw.
        """
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        errors.extend(str(e) for e in payload.get("errors") or [])

        records: list[dict[str, Any]] = []
        for item in payload.get("molecules") or []:
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "smiles": str(item.get("smiles") or ""),
                    "sdf": item.get("sdf"),
                    "qc_flags": [str(f) for f in item.get("qc_flags") or []],
                }
            )
        requested = int(payload.get("n_requested") or n_molecules)
        dropped = max(0, requested - len(records))
        records.extend({"smiles": "", "qc_flags": ["upstream_reconstruction_failed"]} for _ in range(dropped))
        if dropped:
            warnings.append(
                f"MolCRAFT reconstructed {len(records) - dropped} of {requested} samples; "
                f"{dropped} were dropped by reconstruct_from_generated or sanitization."
            )
        return records, warnings, errors

    @staticmethod
    def _existing(value: Any) -> Path | None:
        """*value* as a path, if it names a file that is really there."""
        if not value:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_file() else None

    @classmethod
    def _reference_ligand(cls, pocket: dict[str, Any]) -> Path | None:
        """The reference ligand SDF, under whichever of the accepted keys carries it."""
        for key in _LIGAND_KEYS:
            found = cls._existing(pocket.get(key))
            if found is not None:
                return found
        return None


__all__ = ["MOLCRAFT_ADAPTER_VERSION", "PROJECT_SUBDIR", "SPEC", "MolCraftAdapter"]
