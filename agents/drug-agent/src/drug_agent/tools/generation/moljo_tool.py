"""MolJO — gradient-guided joint optimisation of a ligand that already exists.

MolJO ("Empower Structure-Based Molecule Optimization with Gradient Guided
Bayesian Flow Networks", ICML 2025, arXiv:2411.13280) is not a de-novo sampler.
It takes a reference ligand sitting in a pocket and pushes it, through the
Bayesian-flow parameter space MolCRAFT established, along the gradients of one or
more property predictors — Vina affinity, SA, QED — guiding continuous
coordinates and discrete atom types jointly. The published recipe in
``MolJO/README.md`` is ``--objective vina_sa --guide_mode param_naive
--pos_grad_weight 50 --type_grad_weight 50 --sample_steps 200 --sample_num_atoms
prior``.

**Those are optimisation settings, and numbers produced under them must never be
quoted as unconditional de-novo performance.** Two of the three default
objectives are the same quantities a screening pipeline later uses to judge a
molecule: a run guided by a Vina-affinity predictor at weight 50 has been steered
towards good Vina scores, so its Vina scores are not independent evidence about
it. Reporting such a batch beside an unguided generator's output compares a model
that was told the answer with one that was not.

There is no separate ``GenSI-THUAIR/MolJO`` repository. MolJO ships as the
``MolJO/`` subtree of the MolCRAFT monorepo, alongside MolCRAFT and MolPilot, and
that is why ``SPEC.checkout_relpath`` is ``.external/MolCRAFT`` while every
sentinel and weight path below is prefixed ``MolJO/``. The entry point is
``MolJO/sample_for_pocket_guided.py``; ``scripts/external/moljo_sample.py``
imports it and reproduces its ``__main__`` sequence.

**Why this capability is UNAVAILABLE here.** Five blockers, each verified against
the checkout at commit ``743dfb5198c44ca0cec180822bd54258f9d386f1``:

1. **No property-predictor weights.** The default ``vina_sa`` objective is split
   into tokens and each token loads a checkpoint plus its config —
   ``sample_for_pocket_guided.py`` lines 293-302 read
   ``pretrained/affinity{postfix}.yaml``, ``pretrained/affinity{postfix}.ckpt``,
   ``pretrained/sa{postfix}.yaml`` and ``pretrained/sa{postfix}.ckpt``, with
   ``postfix`` empty because ``configs/test_opt.yaml`` declares no ``classifier``
   section. None of those four files exists: ``.external/MolCRAFT/MolJO/pretrained``
   is not present at all. ``MolJO/README.md`` links them from a Google Drive
   folder and says to put them under ``pretrained/``.

2. **No backbone weights.** The BFN backbone is a separate download.
   ``sample_for_pocket_guided.py`` lines 276-283 locate it by globbing
   ``cfg.accounting.checkpoint_dir`` for ``*mol_stable*``, falling back to
   ``*complete*``, and ``configs/test_opt.yaml`` lines 53-57 define that directory
   as ``${HOME}/project/logs/${LOGNAME}_bfn_molopt/<exp_name>/<revision>/checkpoints``
   — a training run's own output directory, not anything inside the checkout. A
   search of the whole clone for ``*.ckpt``, ``*.pt`` and ``*.pth`` returns zero
   files. The clone carries code only.

3. **No dumped training config.** Line 199 loads ``Config(cfg.accounting.dump_config_path)``
   — ``<logdir>/config.yaml`` — and lines 200-209 keep only ``evaluation``,
   ``visual``, ``accounting``, ``beta1``, ``sigma1_coord``, ``sampling_strategy``
   and ``seed`` from the sampling config. The whole ``dynamics`` and ``data``
   blocks, including the network shape the checkpoint's tensors must match, come
   from that dumped file. A bare ``.ckpt`` is therefore not enough to sample; the
   training run's ``config.yaml`` has to travel with it.

4. **Upstream's pocket script is not runnable off its authors' cluster.** Line
   286 of ``sample_for_pocket_guided.py`` sets ``work_dir =
   '/sharefs/qiukeyue/project/BFN4MolOpt'`` and loads every property predictor
   from beneath it. Its sibling ``sample_guided.py`` line 221 sets ``work_dir =
   '.'`` for the identical load, which is what ``README.md`` documents. The driver
   follows ``sample_guided.py`` and the README, and reads predictors from
   ``<checkout>/MolJO/pretrained``.

5. **Missing Python modules in the ``drug-agent`` environment.** Importing MolJO's
   own modules with the checkout on ``PYTHONPATH`` fails immediately:
   ``core.config.config`` on ``fire``, ``core.datasets.utils`` on ``absl``,
   ``core.callbacks.basic`` on ``overrides``, and ``core.callbacks.validation_callback``
   on ``wandb``. Past ``wandb`` the same import chain reaches
   ``core/evaluation/docking_vina.py``, which imports ``vina``, ``meeko``,
   ``AutoDockTools`` and ``posecheck`` at module scope; of those, ``vina``,
   ``AutoDockTools`` and ``posecheck`` are also absent here. These are installable
   and are listed in ``runtime_modules`` so the probe names them one by one, but
   installing them fixes nothing on its own: blockers 1 to 3 are the weights.

**What a working install would compute.** One reference ligand plus one receptor
PDB in, ``n_molecules`` optimised ligands out, each with 3D coordinates.
``get_dataloader_from_pdb`` clips the receptor to residues within
``dynamics.net_config.r_max`` — 10 Å in ``configs/test_opt.yaml`` — of the
reference ligand, featurises both halves, and repeats the pair ``num_samples``
times so each copy is sampled once. The sampler is the same
``BFN4SBDDScoreModel`` MolCRAFT uses (a 9-layer, 128-dimensional, 16-head
``UniTransformerO2TwoUpdateGeneral`` over a 32-nearest-neighbour graph, per
``configs/test_opt.yaml``), with the classifier gradients added at each of the
200 steps. Observables returned per molecule:

    ``smiles``          canonical SMILES, re-derived by RDKit in the domain base
    ``sdf``             mol block with the generated coordinates
    ``n_atoms``         heavy-atom count of the reconstructed graph
    ``qc_flags``        ``moljo_smiles_failed`` / ``moljo_molblock_failed``

**What it deliberately does not do.** It does not dock, score or rank. Upstream's
``DockingTestCallback`` runs AutoDock Vina over its own samples at test end
(``core/callbacks/validation_callback.py`` lines 578-620); the driver drops that
callback and keeps the three that change the molecules — ``NormalizerCallback``,
``EMACallback`` and ``RecoverCallback``. A generator that scores itself with the
same quantity it optimised is not evidence, and this platform's docking
capability is a separate tool for exactly that reason.

It also does not accept ``context_smiles``. MolJO needs the reference ligand's
3D coordinates, so the reference arrives as an SDF path on the pocket mapping
(``reference_ligand_sdf``, ``ligand_path`` or ``ligand_sdf``); a SMILES string
cannot stand in for it. ``get_dataloader_from_pdb`` asserts on that path.

**What the platform does instead.** Nothing. Under :mod:`drug_agent.tools.base`
there is no surrogate: :meth:`generate` raises
:class:`~drug_agent.tools.base.ToolUnavailable` naming all five blockers, the run
proceeds with this arm visibly absent, and structure-based proposals come from
whichever generator in this package actually has a backend — PFM is the one that
ships its own weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with MolJO, not MolJO's own version.
MOLJO_ADAPTER_VERSION = "1.0.0"

SPEC = ExternalSpec(
    name="MolJO",
    env_prefix="MOLJO",
    # MolJO has no repository of its own; it is a subtree of the MolCRAFT monorepo.
    checkout_relpath=".external/MolCRAFT",
    sentinel_files=(
        "MolJO/sample_for_pocket_guided.py",
        "MolJO/configs/test_opt.yaml",
        "MolJO/core/config/config.py",
        "MolJO/core/models/train_loop.py",
        "MolJO/core/models/bfn4sbdd.py",
        "MolJO/core/datasets/utils.py",
        "MolJO/core/callbacks/basic.py",
    ),
    driver_name="moljo_sample.py",
    # Torch plus Lightning is what identifies an environment MolJO was installed
    # into; readiness below then rejects a half-built one module by module.
    selection_modules=("torch", "pytorch_lightning"),
    # Everything the entry point's import chain pulls in. `fire` (core/config),
    # `absl` (core/datasets, core/callbacks), `overrides` (EMACallback) and `wandb`
    # (validation_callback) are the four that fail first here; `vina`, `meeko`,
    # `AutoDockTools` and `posecheck` come in via core/evaluation/docking_vina.py.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "pytorch_lightning",
        "rdkit",
        "openbabel",
        "sklearn",
        "pytz",
        "absl",
        "fire",
        "overrides",
        "easydict",
        "imageio",
        "wandb",
        "vina",
        "meeko",
        "AutoDockTools",
        "posecheck",
    ),
    # The four files the default `vina_sa` objective loads, named exactly as
    # sample_for_pocket_guided.py lines 293-302 build them. The backbone checkpoint
    # and the training run's dumped config.yaml are separate downloads that upstream
    # locates outside the checkout; install_hint names them.
    weight_relpaths=(
        "MolJO/pretrained/affinity.ckpt",
        "MolJO/pretrained/affinity.yaml",
        "MolJO/pretrained/sa.ckpt",
        "MolJO/pretrained/sa.yaml",
    ),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/GenSI-THUAIR/MolCRAFT",
    upstream_ref="743dfb5198c4",
    license="MolCRAFT/MolJO CC BY-NC-SA 4.0 (non-commercial; weights not redistributed)",
    # Repeated once per missing weight in the unavailable reason, so it stays one
    # line — but it names all three artefacts, because the backbone and the dumped
    # training config are separate downloads no weight_relpath can express.
    install_hint=(
        "see MolJO/README.md: backbone ckpt, its training config.yaml, and the "
        "affinity/sa predictors go in .external/MolCRAFT/MolJO/pretrained/"
    ),
    needs_gpu=True,
)


class MolJoSamplingError(GenerationBackendError):
    """MolJO was installed, sampling started, and no usable molecules came back."""


class MolJoAdapter(GenerativeToolAdapter):
    """Gradient-guided optimisation of a reference ligand inside its pocket."""

    capability_id = "molecule.generate.moljo"
    spec = SPEC
    engine = "moljo"
    version = MOLJO_ADAPTER_VERSION
    conditioning = "pocket"
    # MolJO generates atoms, not reactions: a proposal arrives without a route.
    synthesis_aware = False

    #: MolJO/README.md's published sampling recipe, verbatim. These are guidance
    #: settings: `objective` names the property predictors whose gradients steer the
    #: sample, and the two weights say how hard. Changing them changes what the
    #: batch means, so they are declared here rather than buried in the driver.
    objective = "vina_sa"
    guide_mode = "param_naive"
    pos_grad_weight = 50.0
    type_grad_weight = 50.0
    sample_steps = 200
    sample_num_atoms = "prior"

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Structure-based molecule optimisation (MolJO)",
            description=(
                "Gradient-guided joint optimisation of an existing ligand in its pocket, "
                "over the MolCRAFT Bayesian-flow backbone. Property-predictor gradients "
                "(Vina affinity, SA) steer coordinates and atom types together. This is "
                "optimisation from a reference ligand, not unconditional de-novo design, "
                "and its output must not be reported as de-novo performance."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            # A Bayesian flow sampled from a prior: a seed pins one run, the model
            # is a sampler.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no reference ligand SDF: get_dataloader_from_pdb asserts on "
                "evaluation.ligand_path and cannot clip the 10 A pocket without it",
                "bond perception over the generated point cloud raises MolReconsError; "
                "those samples are counted as reconstruction failures, never emitted",
                "the backbone is found by filename glob (*mol_stable*, then *complete*), "
                "so a renamed checkpoint is reported absent rather than loaded",
                "a property-predictor checkpoint trained against different dynamics "
                "loads without complaint and then guides the wrong quantity",
                "guidance weights of 50 are strong: on a pocket unlike CrossDocked they "
                "drive samples into strained geometry that reconstruction then rejects",
                "EMACallback swaps the checkpoint's EMA weights in at on_test_start; "
                "dropping it silently samples the raw non-EMA weights instead",
                "CUDA out of memory at large evaluation.batch_size",
            ],
            validation_checks=[
                "every objective token has both a .ckpt and a .yaml under pretrained/ before sampling starts",
                "the dumped training config.yaml is present, since dynamics and data "
                "come from it rather than from configs/test_opt.yaml",
                "each emitted SMILES is re-parsed by RDKit in the domain base, which "
                "computes validity and uniqueness from the raw counts",
                "the heavy-atom window is applied before a molecule is kept",
                "reconstruction failures are reported as a warning and are never emitted as molecules",
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
        """Run the driver inside MolJO's interpreter and return its raw records."""
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() precedes this
            raise MolJoSamplingError("MolJO checkout disappeared between probe and call")
        moljo_root = root / "MolJO"

        receptor = self._existing(pocket.get("pdb_path"), "receptor PDB (pocket.pdb_path)")
        reference = self._existing(
            pocket.get("reference_ligand_sdf") or pocket.get("ligand_path") or pocket.get("ligand_sdf"),
            "reference ligand SDF (pocket.reference_ligand_sdf)",
        )

        warnings: list[str] = []
        if context_smiles:
            warnings.append(
                f"MolJO ignored {len(context_smiles)} context SMILES: it optimises the 3D "
                "reference ligand named by pocket.reference_ligand_sdf, and a SMILES string "
                "carries no coordinates to start from"
            )

        request = {
            "moljo_root": str(moljo_root),
            "config_file": str(moljo_root / "configs" / "test_opt.yaml"),
            "protein_path": str(receptor),
            "ligand_path": str(reference),
            "pretrained_dir": str(moljo_root / "pretrained"),
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device": self.ext.device,
            "objective": self.objective,
            "guide_mode": self.guide_mode,
            "pos_grad_weight": float(self.pos_grad_weight),
            "type_grad_weight": float(self.type_grad_weight),
            "sample_steps": int(self.sample_steps),
            "sample_num_atoms": self.sample_num_atoms,
        }
        # run_json rather than run_driver: the caller's max_runtime_seconds has to
        # reach the subprocess, and run_driver pins the budget to self.timeout_s.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=MolJoSamplingError,
            # `core.*` is importable only from inside MolJO/, and the driver chdirs
            # there for the config's relative paths.
            extra_pythonpath=("MolJO",),
            seed=seed,
        )
        payload = run.payload
        records = [item for item in (payload.get("molecules") or []) if isinstance(item, dict)]
        warnings.extend(str(item) for item in (payload.get("warnings") or []))
        errors = [str(item) for item in (payload.get("errors") or [])]
        if not records and not errors:
            errors.append(
                f"MolJO returned no molecules from {payload.get('n_sampled', 0)} samples "
                f"({payload.get('n_reconstruction_failures', 0)} reconstruction failures)"
            )
        return records, warnings, errors

    @staticmethod
    def _existing(value: Any, what: str) -> Path:
        """The path *value* names, or a domain error saying which input was absent."""
        if not value:
            raise MolJoSamplingError(f"MolJO requires a {what}; none was supplied")
        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise MolJoSamplingError(f"MolJO {what} does not exist: {path}")
        return path.resolve()


__all__ = ["MOLJO_ADAPTER_VERSION", "SPEC", "MolJoAdapter", "MolJoSamplingError"]
