"""MolPilot — the same BFN generator as MolCRAFT, run on a VLB-optimal noise schedule.

MolPilot is the third method in the MolCRAFT series and it does not have its own
repository: ``AlgoMole/MolPilot`` and ``GenSI-THUAIR/MolPilot`` both return 404,
and the code lives in the ``MolPilot/`` subdirectory of ``AlgoMole/MolCRAFT``
(cloned here at ``.external/MolCRAFT``, pinned to ``743dfb5198c4``). The three
siblings in that checkout are three distinct methods with three distinct
checkpoints, and the difference is not cosmetic:

``MolCRAFT/``
    A Bayesian Flow Network over a *unified continuous parameter space* for atom
    positions and types. Entry point ``MolCRAFT/sample_for_pocket.py``, model
    ``core/models/sbdd_train_loop.py::SBDDTrainLoop``. Atom types and coordinates
    share one noise level, advanced together on a linear schedule.

``MolJO/``
    Structure-based molecule *optimisation*, not generation: gradient guidance
    from a property classifier is injected into a trained BFN's sampler. Entry
    points ``MolJO/sample_for_pocket_guided.py`` and ``MolJO/train_classifier.py``.
    It needs a guidance model in addition to a generator checkpoint.

``MolPilot/``
    A *time-decoupled* BFN — separate noise levels for the continuous 3D modality
    and the discrete 2D modality, including explicit bond types
    (``num_bond_classes: 4`` in ``MolPilot/configs/crossdock_train_test.yaml``,
    which the two siblings' configs do not carry) — sampled along a schedule
    chosen by optimising the variational lower bound as a path integral (VOS).
    The schedule is a **file, not a formula**: ``sample_for_pocket.py`` lines
    368-374 load it with ``torch.load`` and assign it to ``model.time_scheduler``,
    and ``core/models/sbdd4train_timewarp.py`` lines 549-555 use it as the
    per-modality ``t_steps`` when it is set and fall back to the linear
    ``1 - linspace(1, 0, sample_steps + 1)`` when it is ``None``. Run MolPilot's
    checkpoint without its schedule file and the VOS contribution — the entire
    published difference from MolCRAFT — is silently switched off, while sampling
    still succeeds and still writes molecules. That is the failure mode this
    adapter is most concerned with, because it produces plausible output.

Upstream's own headline for MolPilot is a 95.9% PoseBusters passing rate on
CrossDock (``.external/MolCRAFT/README.md``, "Key Contributions"). That number is
upstream's, measured by upstream, on CrossDock test pockets, with the schedule
file loaded. It is quoted here for provenance and is not evidence about any
molecule this platform has produced.

**Why this capability is AVAILABLE here, and why it was not before.** Five blockers
were recorded on the aarch64 host. Four were artifacts and an environment, and have
been cleared. The fifth was the architecture — and the way it was *written* is the
finding worth keeping, so it is restated below rather than deleted.

1. **The generative weights had to be fetched, and are now present.** The clone is
   code only. The file that has to exist, and does, is
   ``.external/MolCRAFT/MolPilot/checkpoints/molpilot_epoch26-val_loss5.42-mol_stable0.48-complete0.83.ckpt``
   — that exact basename, because ``sample_for_pocket.py`` ignores its own
   ``--ckpt_path`` flag (declared at line 180, never read) and instead globs
   ``cfg.accounting.checkpoint_dir`` for ``*mol_stable*`` at line 355, then sorts
   the matches by the digits after ``val_loss`` at line 357. Upstream publishes it
   at https://huggingface.co/GenSI/MolPilot; the model index at
   ``huggingface.co/api/models/GenSI/MolPilot`` lists it as an LFS file of
   49,191,007 bytes.

2. **The VOS schedule tensor, likewise.** ``optimized_path_rect_closest.pt``
   (2,223 bytes in the same model index) is the only artifact that makes this
   MolPilot rather than a bond-aware MolCRAFT. It is consumed as
   ``--time_scheduler_path`` and asserted at ``sbdd4train_timewarp.py`` line 552
   to have shape ``(sample_steps + 1, 2)``, so it is tied to the sampling step
   count upstream documents (``--sample_steps 100``). This adapter treats it as a
   required weight, not an option, and finds it at
   ``.external/MolCRAFT/MolPilot/checkpoints/optimized_path_rect_closest.pt``.

3. **The dumped training config, likewise — and it needs rewriting on the way in.**
   ``sample_for_pocket.py`` line 275
   does ``tr_cfg = Config(cfg.accounting.dump_config_path)`` and takes
   ``cfg.dynamics`` from it, so the network architecture comes from a YAML file a
   training run wrote, not from the sampling config. Upstream ships that dump as
   ``molpilot_config.yaml`` (3,524 bytes). It is not usable as-is: its
   ``accounting.*`` keys point at ``/sharefs/qiukeyue/project/logs/...`` and its
   ``data.path`` at ``/sharefs/share/sbdd_data/...``, absolute paths on the
   authors' cluster. The driver therefore rewrites the ``accounting`` and
   ``evaluation`` blocks into a scratch run directory — necessary, not cosmetic,
   because a *dumped* config carries no ``!SUB`` tags, and
   ``core/config/config.py`` only substitutes command-line values into ``!SUB``
   nodes. Passing ``--num_samples`` or ``--protein_path`` alongside a dumped
   config changes nothing at all.

4. **Ten of MolPilot's imports are missing from ``drug-agent``, which is why it
   does not run there.** Probed with :mod:`drug_agent.tools._ext.probe`, the
   ``drug-agent`` environment cannot import ``torch_sparse``
   (``core/models/uni_transformer_edge.py`` line 13), ``absl``
   (``core/models/bfn4sbdd.py`` line 1), ``wandb``
   (``sbdd4train_timewarp.py`` line 6), ``fire`` (``core/config/config.py``
   line 4), ``overrides`` (``core/callbacks/basic.py`` line 8), ``imageio``
   (``core/evaluation/visualization.py`` line 6), ``torchdiffeq``
   (``core/models/bfn_base.py`` line 5), ``posecheck``, ``vina`` or
   ``AutoDockTools`` (``core/evaluation/docking_vina.py`` lines 12, 4 and 9).
   None of these is optional: every one is a module-scope import reached from
   ``sample_for_pocket.py``, so the script fails before it parses a PDB. All ten are
   present in the dedicated MolPilot environment this adapter is pointed at.

5. **``vina`` had no aarch64 wheel — and the way that was recorded made MolPilot
   permanently UNAVAILABLE on every host, including the ones that can run it.**
   ``vina`` sits on MolPilot's import path only because the evaluation half of the
   script is wired into the sampling half; upstream's ``skip_eval`` switch
   (``core/callbacks/validation_callback.py`` line 2059) skips *running* it but not
   *importing* it. On aarch64 that was a real wall: PyPI's release index for ``vina``
   1.2.7 lists ten binary wheels, all ``manylinux_2_17_x86_64`` or
   ``musllinux_1_2_x86_64``, plus an sdist, so ``vina`` there meant compiling AutoDock
   Vina and Boost from source.

   That fact was written into :data:`SPEC` as an **unconditional**
   ``platform_blockers`` tuple. ``ExternalBackend.problems`` returns those verbatim
   and unconditionally, so the reason "this host is aarch64 and vina publishes no
   aarch64 wheel" would have been reported on an x86-64 host too, with vina installed
   and working. MolPilot could not have become AVAILABLE anywhere. It is now computed
   from ``platform.machine()`` — see :data:`_VINA_WHEEL_BLOCKERS`, which copies the
   pattern ``omtra_tool._DGL_WHEEL_BLOCKERS`` already used. On this host ``pip install
   vina`` takes the manylinux wheel; ``vina`` 1.2.7 imports in the MolPilot
   environment. **The lesson generalises past this file: a hard-coded platform blocker
   is a claim about every machine, not about the one it was written on.**

**Where it runs.** Not in ``drug-agent`` — the ten imports in blocker 4 are why.
MolPilot runs in its own environment, named to this adapter through
``external_tools.molpilot.python`` in the active profile; ``configs/local.yaml``
pins it here.

**What the platform does if any of that goes away.** Nothing. ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` through ``require_available()``,
the capability card reports UNAVAILABLE with the blockers above, and no molecule
is attributed to MolPilot. There is no surrogate schedule, no fallback to
MolCRAFT's checkpoint and no linear-schedule "approximation" — the last would be
the most dangerous of the three, because it would run, produce molecules, and
label them with the name of a method that did not generate them.

**How to read MolPilot output.** This is a
pocket-conditioned 3D generator: the network emits atom positions and types and a
bond perceiver rebuilds the molecular graph. MolPilot predicts bond types
directly, which is exactly why upstream can claim a PoseBusters rate, but the
reconstruction hazard does not disappear. The measured baseline on this
repository is PFM, the one pocket-conditioned generator whose weights are present:
of 40 generated molecules, 21 contained an allene and 21 a four-membered ring,
leaving 2 chemically ordinary molecules, and all 40 fell outside the activity
model's applicability domain. So before believing anything about a MolPilot
batch, look at ``generation_validity`` and ``generation_uniqueness`` on the
metered call, then count allenes (``C=C=C``) and four-membered rings in the
result. A high validity with a high allene rate is a reconstruction artefact, not
chemistry, and it looks identical to success in every downstream artifact.

Every molecule that would leave here carries ``TrustLabel.PROPOSED`` and is
screened by the same RDKit, activity, safety and synthesis pipeline as any other
candidate, per :mod:`drug_agent.tools.generation`.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with MolPilot, not MolPilot's own version.
MOLPILOT_ADAPTER_VERSION = "1.0.0"

#: The driver executed inside the MolPilot environment.
DRIVER_NAME = "molpilot_sample.py"

#: Where this adapter expects the three downloaded artifacts, relative to the
#: MolCRAFT checkout root. ``MolPilot/checkpoints/`` is upstream's own default:
#: ``sample_for_pocket.py`` line 180 declares ``--ckpt_path`` as
#: ``checkpoints/molpilot_epoch26-...ckpt``, relative to the ``MolPilot`` folder.
#: The schedule and the dumped training config have no upstream default path at
#: all, so they are co-located there by this adapter's convention.
CHECKPOINT_RELPATH = "MolPilot/checkpoints/molpilot_epoch26-val_loss5.42-mol_stable0.48-complete0.83.ckpt"
TIME_SCHEDULER_RELPATH = "MolPilot/checkpoints/optimized_path_rect_closest.pt"

#: The ``vina`` import barrier, which is a property of the *host architecture* and
#: not of this project. MolPilot reaches ``vina`` through a module-scope import in
#: ``core/evaluation/docking_vina.py``, so the sampler cannot start without it; on
#: aarch64 PyPI offers only an sdist, which means compiling AutoDock Vina and Boost.
#: On x86-64 ``pip install vina`` takes a manylinux wheel and the barrier is simply
#: absent, so this must be computed rather than asserted: a blocker listed here is
#: returned unconditionally by ``ExternalBackend.problems`` and makes the capability
#: UNAVAILABLE on *every* host, which kept MolPilot dark on machines that can run it.
#: Same shape as ``omtra_tool._DGL_WHEEL_BLOCKERS``. Verified 2026-09-08 on x86-64:
#: vina 1.2.7 installs from a wheel and MolPilot samples.
_VINA_WHEEL_BLOCKERS: tuple[str, ...] = (
    ()
    if platform.machine() in {"x86_64", "AMD64"}
    else (
        f"this host is {platform.machine()} and vina publishes no aarch64 wheel "
        "(PyPI vina 1.2.7 ships manylinux/musllinux x86_64 only); MolPilot imports it "
        "at module scope in core/evaluation/docking_vina.py, so sample_for_pocket.py "
        "cannot start here until AutoDock Vina is built from its sdist",
    )
)
TRAIN_CONFIG_RELPATH = "MolPilot/checkpoints/molpilot_config.yaml"

#: The ``!SUB``-templated sampling config shipped in the checkout. Used only as a
#: sentinel: it proves the checkout is MolPilot's rather than a sibling's, since
#: it is the config carrying ``num_bond_classes: 4``. The driver does not pass it
#: — the dumped training config above is a complete resolved config on its own,
#: and mixing a templated file with a resolved one would need two ``Config``
#: loaders with different substitution rules.
SAMPLING_CONFIG_RELPATH = "MolPilot/configs/crossdock_train_test.yaml"

SPEC = ExternalSpec(
    name="MolPilot",
    env_prefix="MOLPILOT",
    # MolPilot has no repository of its own; it is a folder in MolCRAFT.
    checkout_relpath=".external/MolCRAFT",
    # Files the run path really imports, checked as files. The first is the entry
    # point; the rest are the modules it pulls in that distinguish MolPilot from
    # its two siblings in the same checkout.
    sentinel_files=(
        "MolPilot/sample_for_pocket.py",
        "MolPilot/core/models/sbdd4train_timewarp.py",
        "MolPilot/core/models/bfn4sbdd.py",
        "MolPilot/core/models/uni_transformer_edge.py",
        "MolPilot/core/config/config.py",
        SAMPLING_CONFIG_RELPATH,
    ),
    driver_name=DRIVER_NAME,
    # Narrow probe: which interpreter MolPilot was installed into at all.
    #
    # `("torch", "pytorch_lightning")` was not narrow — the project's own
    # `drug-agent` environment has both, so selection stopped at `sys.executable`
    # before any conda environment was considered, and MolPilot then reported
    # UNAVAILABLE for the nine packages `drug-agent` was never going to have.
    # `torchdiffeq` is the discriminator: MolPilot's time-warped ODE sampler is the
    # only thing in this catalogue that imports it, so an interpreter that has it is
    # a MolPilot environment and one that does not is not.
    selection_modules=("torch", "torchdiffeq"),
    # Everything a half-built install loses, each discovered by reading the
    # module-scope import that reaches it from sample_for_pocket.py. torch_scatter
    # and torch_sparse lead because they compile from source with no wheel — the
    # same reason PFM probes them by name.
    runtime_modules=(
        "torch",
        "torch_scatter",
        "torch_sparse",
        "torch_geometric",
        "pytorch_lightning",
        "torchdiffeq",
        "absl",
        "wandb",
        "fire",
        "overrides",
        "imageio",
        "easydict",
        "Bio",
        "openbabel",
        "meeko",
        "posebusters",
        "posecheck",
        "vina",
        "AutoDockTools",
    ),
    weight_relpaths=(
        CHECKPOINT_RELPATH,
        TIME_SCHEDULER_RELPATH,
        TRAIN_CONFIG_RELPATH,
    ),
    conda_envs=("drug-agent-molpilot",),
    upstream_url="https://github.com/AlgoMole/MolCRAFT",
    upstream_ref="743dfb5198c4",
    license=(
        "MolCRAFT code: CC BY-NC-SA 4.0 (.external/MolCRAFT/LICENSE and "
        "MolPilot/LICENSE). Weights: CC BY-NC-SA 2.0 (huggingface.co/GenSI/MolPilot). "
        "Non-commercial only; not downloaded here."
    ),
    install_hint=(
        "make external-clone && huggingface-cli download GenSI/MolPilot "
        "molpilot_epoch26-val_loss5.42-mol_stable0.48-complete0.83.ckpt "
        "optimized_path_rect_closest.pt molpilot_config.yaml "
        "--local-dir .external/MolCRAFT/MolPilot/checkpoints"
    ),
    needs_gpu=True,
    platform_blockers=_VINA_WHEEL_BLOCKERS,
)


class MolPilotAdapter(GenerativeToolAdapter):
    """Pocket-conditioned 3D generation with MolPilot's VLB-optimal noise schedule."""

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Also needs a reference ligand: upstream's loader asserts both paths."""
        return ("pocket_pdb", "ref_ligand_sdf")

    capability_id = "molecule.generate.molpilot"
    spec = SPEC
    engine = "molpilot"
    version = MOLPILOT_ADAPTER_VERSION
    conditioning = "pocket"
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="De-novo pocket-conditioned generation (MolPilot, VOS schedule)",
            description=(
                "Samples 3D ligands inside a protein binding site with the MolPilot "
                "time-decoupled Bayesian Flow Network, advancing the continuous "
                "coordinate modality and the discrete atom/bond modality on separate, "
                "VLB-optimal noise schedules loaded from a fitted schedule tensor. "
                "Needs the whole protein PDB and a reference ligand, because the entry "
                "point clips the pocket to a 10 A shell around that ligand and, with "
                "sample_num_atoms='ref', takes the atom count from it too."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                # Each of these is a specific line in the checkout, not a generic worry.
                "the schedule tensor is missing, so sbdd4train_timewarp.py:553 falls back "
                "to a linear schedule and the run is MolCRAFT-with-bonds wearing MolPilot's "
                "name — it succeeds, which is what makes it dangerous",
                "the schedule tensor was fitted for a different step count: "
                "sbdd4train_timewarp.py:552 asserts its shape is (sample_steps + 1, 2)",
                "bond perception over a generated point cloud yields allenes and "
                "four-membered rings; the measured PFM baseline in this repository is 21 "
                "allenes and 21 four-membered rings in 40 molecules",
                "sample_for_pocket.py ignores --ckpt_path (line 180) and globs "
                "accounting.checkpoint_dir for '*mol_stable*' (line 355), so a checkpoint "
                "renamed without 'mol_stable' and 'val_loss' in it is silently not found",
                "a dumped training config has no !SUB nodes, so every sampling flag passed "
                "on the command line beside it is inert (core/config/config.py)",
                "no reference ligand: get_dataloader_from_pdb asserts both protein_path and "
                "ligand_path, and the pocket is defined by the ligand, not by a centre",
                "vina, posecheck and AutoDockTools are imported at module scope by the "
                "evaluation half and abort the sampling half when absent",
                "CUDA OOM at large --eval_batch_size; upstream's documented default is 100",
            ],
            validation_checks=[
                "every returned SMILES is re-parsed by RDKit here; validity and uniqueness "
                "are computed by the base class from raw counts, never reported by MolPilot",
                "the schedule tensor is a required weight, so an UNAVAILABLE card is raised "
                "rather than a linear-schedule run mislabelled as MolPilot",
                "the checkpoint basename is checked for the 'mol_stable' and 'val_loss' "
                "substrings the upstream glob and sort depend on",
                "molecules carry TrustLabel.PROPOSED and pocket_id from the request",
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
        """Run ``MolPilot/sample_for_pocket.py`` once and return its molecules.

        ``context_smiles`` is ignored: MolPilot conditions on a pocket and a
        reference ligand's *geometry*, and has no ligand-SMILES conditioning path.
        Silently accepting a context that cannot influence the samples would be a
        lie about what the model was given, so it is reported as a warning.
        """
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() precedes this
            raise GenerationBackendError("MolPilot checkout not resolved")

        protein_pdb = pocket.get("protein_pdb_path") or pocket.get("pdb_path")
        ref_ligand = pocket.get("ref_ligand_sdf") or pocket.get("reference_ligand_sdf")
        if not protein_pdb or not Path(protein_pdb).is_file():
            raise GenerationBackendError(
                f"MolPilot needs a protein PDB; generation.pocket.pdb_path={protein_pdb!r} "
                "is not a file. Unlike PFM this is the whole receptor, not a pre-cut pocket: "
                "get_dataloader_from_pdb clips the 10 A shell itself."
            )
        if not ref_ligand or not Path(ref_ligand).is_file():
            raise GenerationBackendError(
                "MolPilot needs a reference ligand SDF to define the pocket and, under "
                "sample_num_atoms='ref', the number of atoms to sample. PocketSpec carries "
                "no such field, so pass it as pocket['ref_ligand_sdf']; "
                f"got {ref_ligand!r}."
            )

        warnings: list[str] = []
        if context_smiles:
            warnings.append(
                f"MolPilot ignores context_smiles ({len(context_smiles)} supplied): it "
                "conditions on pocket geometry and a reference ligand, not on SMILES."
            )

        request = {
            "molpilot_dir": str(root / "MolPilot"),
            "protein_path": str(Path(protein_pdb).resolve()),
            "ligand_path": str(Path(ref_ligand).resolve()),
            "checkpoint": str(root / CHECKPOINT_RELPATH),
            "time_scheduler": str(root / TIME_SCHEDULER_RELPATH),
            "train_config": str(root / TRAIN_CONFIG_RELPATH),
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            # Upstream's documented sampling command uses --sample_steps 100, and the
            # schedule tensor's shape assertion is tied to it. Not a free parameter.
            "sample_steps": 100,
            "seed": int(seed),
            "device": self.ext.device,
            # Skip MolPilot's own Vina/PoseBusters pass (validation_callback.py:2059).
            # This platform docks and busts with its own tools, and the generator must
            # not be the judge of what it generated.
            "skip_eval": True,
        }
        run = self.run_driver(
            request,
            error_cls=GenerationBackendError,
            timeout_s=timeout_s,
            # PYTHONPATH gets <checkout>/MolPilot so `import core...` resolves; the
            # driver chdirs there itself because upstream's config paths are relative.
            extra_pythonpath=("MolPilot",),
            seed=seed,
        )
        payload = run.payload
        records = [dict(item) for item in payload.get("molecules") or []]
        warnings += [str(w) for w in payload.get("warnings") or []]
        errors = [str(e) for e in payload.get("errors") or []]
        return records, warnings, errors


__all__ = [
    "CHECKPOINT_RELPATH",
    "DRIVER_NAME",
    "MOLPILOT_ADAPTER_VERSION",
    "SPEC",
    "TIME_SCHEDULER_RELPATH",
    "TRAIN_CONFIG_RELPATH",
    "MolPilotAdapter",
]
