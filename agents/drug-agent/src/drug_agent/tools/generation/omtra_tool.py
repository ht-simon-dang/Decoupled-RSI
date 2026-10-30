"""OMTRA — one flow-matching model that generates a ligand in a pocket, and also docks one.

OMTRA (*One Model To Rule them All*, gnina group, https://arxiv.org/abs/2512.05080)
is a multitask flow-matching model over protein-ligand systems. The same network
and, for the protein-conditioned tasks, the same checkpoint serve de novo design,
rigid docking, conformer generation and pharmacophore-conditioned variants of
each; what changes is which modality groups a task holds fixed and which it
generates. That breadth is the reason this adapter has to be explicit about one
thing above all others: **which task it invokes.**

**The task this adapter runs.** ``fixed_protein_ligand_denovo_condensed``. In
``omtra/tasks/tasks_condensed.py`` it fixes ``['protein_identity',
'protein_structure']`` and generates ``['ligand_identity_condensed',
'ligand_structure']`` — the protein is held rigid and the ligand's atom types,
formal charges, bond orders and 3D coordinates are all sampled. That is de novo
generation, and it is what a generation capability may claim.

**The task this adapter refuses to run.** ``rigid_docking_condensed``. It loads
the *same* checkpoint (``omtra/utils/checkpoints.py`` maps both to
``mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt``) but fixes
``ligand_identity_condensed`` and generates only ``ligand_structure``: it places a
molecule it was given. A generator that returned those poses would be reporting a
ligand it did not invent as one it did, and no downstream consumer could tell.
Both task names are therefore written onto every batch's ``config`` —
``omtra_task`` and ``omtra_docking_task_not_used`` — so the record says which
half of the model produced the molecules.

**Correcting the catalogue.** The research catalogue's OMTRA row says the direct
repository fetch failed during that review and that checkpoint availability had to
be verified before adoption. Both halves are now settled, and in opposite
directions. The fetch succeeded: ``.external/OMTRA`` is a real clone at
``745127dabfa1`` (a merge commit dated 2026-04-10) carrying 294 tracked files, 149
of them Python, including ``cli.py``, ``routines/sample.py``, the whole ``omtra``
package and the web application. The checkpoints are *not* in it — ``git ls-files``
finds zero ``.ckpt`` files and no ``omtra/trained_models`` path — but they are
published and reachable, which the readme's "Model Weights" section and a live
check both confirm. The row should read "code cloned; weights hosted separately at
bits.csb.pitt.edu and not downloaded", not "fetch failed".

**Why this capability is AVAILABLE on this machine, and what it took.** On the
aarch64 host it was UNAVAILABLE for a reason that x86-64 removes, and for a second
reason that x86-64 does *not* remove. Both are recorded, because the second one is
the reusable finding.

1. **The checkpoint had to be downloaded, and now is.**
   ``omtra/utils/checkpoints.py::TASK_TO_CHECKPOINT`` maps
   ``fixed_protein_ligand_denovo_condensed`` to
   ``mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt``, and
   ``cli.run_sample`` looks for it under ``Path(omtra_root()) / "omtra/trained_models/"``.
   Upstream serves it: the readme gives a ``wget -r`` over
   ``https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/`` (267 MiB). It is
   present at
   ``.external/OMTRA/omtra/trained_models/mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt``.

2. **DGL installs here, because the wheel index is x86-64 only.**
   ``omtra/models/omtra.py`` imports ``dgl`` at module scope and 31 files in the
   checkout import it, so nothing in the sampling path survives without it.
   ``requirements-cuda.txt`` pins ``dgl==2.4.0+cu121`` against
   ``https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html``, which publishes
   ``manylinux1_x86_64`` wheels only — ten of them, cp38 through cp312, none for
   aarch64. That is what blocked the aarch64 host, where PyPI's newest ``dgl`` is
   2.1.0 and CPU-only. Here the pinned wheel installs unchanged. The blocker stays in
   the spec, computed from ``platform.machine()`` rather than asserted, so the card
   still tells the truth on a host where the wheel does not exist.

3. **DGL's CUDA tag was never the constraint — torch's SASS was.** Installing the
   pinned stack gave a working import and then ``CUDA error: no kernel image is
   available for execution on the device`` on the first real kernel. The obvious
   reading — "DGL is built for cu121, this card is sm_120, and no prebuilt DGL exists
   above cu124" — is **wrong**, and it cost a day. ``libdgl.so`` does not link
   ``libtorch`` at all; DGL reaches torch through DLPack, so its CUDA tag never had to
   match the card. What actually failed is torch: ``torch 2.4.0+cu121``'s
   ``libtorch_cuda.so`` carries SASS to sm_90 and **zero PTX**, so there is nothing to
   JIT for sm_120. Installing a torch built with native sm_120 SASS — 2.7.1+cu128 or
   2.8.0+cu128 — makes real DGL CUDA kernels run against **DGL at its own original
   pin, unchanged**. See ADR-0014 for the measurement error that produced the wrong
   conclusion.

4. **OMTRA additionally needed a ``torch.load`` fix.** Reaching sm_120 requires torch
   ≥ 2.7, and torch 2.6 flipped ``torch.load``'s ``weights_only`` default to ``True``.
   OMTRA's checkpoint pickles ``omegaconf.listconfig.ListConfig``, so loading it
   through PyTorch Lightning raises ``UnpicklingError: Weights only load failed ...
   Unsupported global: GLOBAL omegaconf.listconfig.ListConfig``. Lightning passes
   ``weights_only`` **explicitly**, so setting the default back is not enough: only an
   unconditional override in the driver clears it.

5. **The interpreter is not ``drug-agent``.** ``dgl``, ``hydra``, ``omegaconf``,
   ``einops``, ``biotite``, ``peppr`` and ``wandb`` are absent from it. OMTRA runs in
   its own environment, named to this adapter through ``external_tools.omtra.python``
   in the active profile — ``configs/local.yaml`` pins it here, with
   ``device: cuda:0``.

**Measured here on 2026-09-09.** Four molecules for the sEH pocket
(``.external/pockets/seh_3i28_pocket10.pdb``, centre ``(74.101, -9.434, 67.295)``,
radius 11.5 Å, seed 7, 250 timesteps) on ``cuda:0``: 55.1 s wall, ``n_raw`` 4,
``n_valid`` 1, ``n_unique`` 1, one molecule kept. The reconstruction audit reported
3 of 4 matching the allene SMARTS and 2 of 4 carrying a 3- or 4-membered ring. Read
that as the warning below says to read it: the sampler ran, the reconstruction is
where the batch was lost, and a working install is not a good batch.

**What the platform does if the backend goes away.** Nothing. ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` carrying the whole reason above,
this capability produces no molecules, and no other generator is substituted for
it. A run whose brief asked for OMTRA candidates proceeds with that arm visibly
empty. Under :mod:`drug_agent.tools.base` there is no third status.

**What it would compute, and how to read it.** ``generate()`` returns the same
:class:`~drug_agent.tools.pfm_tool.GenerationBatch` every generator here returns —
``validity``, ``uniqueness``, ``n_kept`` and per-molecule
:class:`~drug_agent.tools.pfm_tool.GeneratedMolecule` records with an SDF block —
computed by :class:`~drug_agent.tools.generation.GenerativeToolAdapter`, never by
the backend.

Read validity and the reconstruction rates before reading anything else. OMTRA
does not perceive bonds from a point cloud the way TargetDiff-lineage models do;
its bond orders are a generated modality, drawn from the four-symbol vocabulary in
``omtra/constants/__init__.py`` (``[None, SINGLE, DOUBLE, TRIPLE]`` — there is no
aromatic type, so output is always Kekulé). What it shares with them is the
consequence: ``omtra.eval.system.SampledSystem.build_molecule`` assembles an
``RWMol`` from those generated bonds and never calls ``Chem.SanitizeMol``
(sanitization appears only on the ``--metrics`` path), so an impossible valence is
written to ``gen_ligands.sdf`` unchallenged and is only caught downstream. The
driver therefore counts, over every molecule OMTRA wrote, how many match
``[*]=[#6]=[*]`` and how many carry a 3- or 4-membered ring, and this adapter
reports both as batch warnings.

The measured baseline on this repository is PFM's, and it is the reason those
counts exist: of 40 PFM molecules, 21 contained an allene and 21 a four-membered
ring, leaving 2 chemically ordinary molecules, and all 40 fell outside the
activity model's applicability domain. Those are PFM's numbers under distance-based
bond perception; they are not a prediction about OMTRA, whose mechanism differs.
They are the reason to look — and on this machine OMTRA's own first batch says the
same thing: 4 sampled, 1 sanitizable, 3 of 4 carrying an allene centre. One batch of
four is not a rate; it is a warning that the checkpoint loading is the easy half.
"""

from __future__ import annotations

import platform
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with OMTRA, not OMTRA's own version (its
#: ``pyproject.toml`` declares ``version = "0.0.0"``).
OMTRA_ADAPTER_VERSION = "1.0.0"

#: The task ``_sample`` invokes: pocket-conditioned de novo generation.
#: ``omtra/tasks/tasks_condensed.py`` — fixes protein identity and structure,
#: generates ligand identity and structure.
OMTRA_DENOVO_TASK = "fixed_protein_ligand_denovo_condensed"

#: The task ``_sample`` deliberately never invokes. Same checkpoint, but it fixes
#: the ligand's identity and generates only its structure: that is docking.
OMTRA_DOCKING_TASK = "rigid_docking_condensed"

#: Checkpoint for the protein-conditioned tasks, relative to the checkout root.
#: From ``omtra/utils/checkpoints.py::TASK_TO_CHECKPOINT`` joined onto the
#: ``Path(omtra_root()) / "omtra/trained_models/"`` that ``cli.run_sample`` searches.
OMTRA_CHECKPOINT_RELPATH = "omtra/trained_models/mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt"

#: Integration steps. ``cli.py::create_parser`` defaults ``--n_timesteps`` to 250.
OMTRA_N_TIMESTEPS = 250

#: Box edge when a pocket centre is supplied. ``cli.py`` defaults ``--bbox_length``
#: to 23.0 Angstrom; a :class:`PocketSpec` radius overrides it as ``2 * radius``.
OMTRA_DEFAULT_BBOX_LENGTH = 23.0

#: True of *this* machine, not of OMTRA. Computed rather than hard-coded so the
#: spec does not lie on a host where the pinned wheel exists.
_DGL_WHEEL_BLOCKERS: tuple[str, ...] = (
    ()
    if platform.machine() in {"x86_64", "AMD64"}
    else (
        f"this host is {platform.machine()} and OMTRA's requirements-cuda.txt pins "
        "dgl==2.4.0+cu121 from https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html, an index "
        "that publishes manylinux1_x86_64 wheels only (ten listed, cp38-cp312, none aarch64); "
        "PyPI's newest dgl is 2.1.0, CPU-only on aarch64, so the pinned CUDA stack "
        "cannot be built here",
    )
)

SPEC = ExternalSpec(
    name="OMTRA",
    env_prefix="OMTRA",
    checkout_relpath=".external/OMTRA",
    # Every one of these is imported by the sampling path and all six exist in the
    # clone at 745127dabfa1: the console entry point, the sampler it hands off to,
    # the task/checkpoint tables, the pocket graph builder and the SDF writer.
    sentinel_files=(
        "cli.py",
        "routines/sample.py",
        "omtra/utils/checkpoints.py",
        "omtra/utils/file_to_graph.py",
        "omtra/models/omtra.py",
        "omtra/eval/system.py",
    ),
    driver_name="omtra_sample.py",
    # dgl identifies the environment OMTRA was built into: nothing else in this
    # catalogue needs it, and OMTRA's build_env.sh installs it first.
    selection_modules=("dgl",),
    # Discovered by walking the imports from cli.py through routines/sample.py,
    # omtra/load/quick.py and omtra/models/omtra.py, not guessed. dgl, torch_scatter
    # and torch_cluster head the list because they are what a half-finished install
    # leaves out: the first ships wheels for x86-64 only (nothing for aarch64) and the
    # other two compile from source on every platform.
    runtime_modules=(
        "torch",
        "dgl",
        "torch_scatter",
        "torch_cluster",
        "hydra",
        "omegaconf",
        "pytorch_lightning",
        "einops",
        "biotite",
        "zarr",
        "peppr",
        "useful_rdkit_utils",
        "wandb",
        "rdkit",
        "numpy",
        "scipy",
        "pandas",
    ),
    weight_relpaths=(OMTRA_CHECKPOINT_RELPATH,),
    # Same measured split as FlowMol3: `drug-agent-omtra` (torch 2.4.0+cu121) imports
    # everything and fails at the first kernel launch on this sm_120 card, while
    # `drug-agent-omtra-gpu` (torch 2.8.0+cu128) runs. The order here is a preference,
    # not the decision — `ExternalBackend._resolve_python` decides on an executed
    # kernel, so this list stays correct on a host where the plain env is the good one.
    conda_envs=("drug-agent-omtra-gpu", "drug-agent-omtra"),
    upstream_url="https://github.com/gnina/OMTRA",
    upstream_ref="745127dabfa1",
    license="Apache-2.0 (LICENSE in the checkout); weights served separately from bits.csb.pitt.edu",
    install_hint=(
        "make external-clone && conda create -y -n omtra python=3.11 && "
        "conda run -n omtra bash .external/OMTRA/build_env.sh && "
        "wget -r -np -nH --cut-dirs=3 -R 'index.html*' "
        "-P .external/OMTRA/omtra/trained_models "
        "https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/"
    ),
    needs_gpu=True,
    platform_blockers=_DGL_WHEEL_BLOCKERS,
)


class OmtraGenerationError(GenerationBackendError):
    """OMTRA was installed, sampling was attempted, and no usable batch came back.

    Also raised when the request itself cannot be turned into an OMTRA invocation
    — no pocket PDB, say. That is not a per-molecule failure to be recorded as a
    flag; it means the call could not be made at all.
    """


class OmtraAdapter(GenerativeToolAdapter):
    """Pocket-conditioned de novo generation with OMTRA's protein-conditioned checkpoint."""

    capability_id = "molecule.generate.omtra"
    spec = SPEC
    engine = "omtra"
    version = OMTRA_ADAPTER_VERSION
    conditioning = "pocket"
    #: OMTRA samples atoms and bonds, not reactions. There is no route attached to
    #: anything it proposes.
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned de-novo generation (OMTRA)",
            description=(
                "Flow-matching generation of a ligand's atom types, charges, bond orders and 3D "
                f"coordinates inside a fixed protein pocket, via OMTRA's {OMTRA_DENOVO_TASK} task. "
                f"The same checkpoint also serves {OMTRA_DOCKING_TASK}; this capability never "
                "invokes it, because docking a supplied molecule is not generation."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint absent: cli.run_sample prints 'No checkpoint found for task' and exits 1, "
                "because omtra/trained_models/ is not in the clone",
                "generated bond block yields an impossible valence; build_molecule never calls "
                "SanitizeMol, so the molecule reaches gen_ligands.sdf and fails only at RDKit parse",
                "cumulated dienes and 3-/4-membered rings from a sampled bond block, counted here "
                "rather than hidden, because a collapse in either is how a wrong checkpoint shows",
                "a whole-protein PDB passed where a pocket was meant: the bounding box then spans "
                "the protein and the sampled ligand size is meaningless",
                "no CUDA in the OMTRA environment: routines/sample.py silently falls back to CPU, "
                "where 250 integration steps take impractically long rather than failing",
                "torch older than 2.7 in the OMTRA environment: its libtorch_cuda.so carries SASS "
                "only to sm_90 and no PTX, so on an sm_120 card every kernel raises 'no kernel "
                "image is available for execution on the device' — a torch problem, not a DGL one",
                "torch 2.6+ defaults torch.load to weights_only=True and Lightning passes it "
                "explicitly, so OMTRA's omegaconf-pickling checkpoint raises UnpicklingError "
                "unless the driver overrides it unconditionally",
                "no seed exists upstream — neither cli.py nor routines/sample.py defines one — so "
                "the driver's own seeding does not make DGL and cuDNN kernels reproducible",
            ],
            validation_checks=[
                "the invoked task is recorded on the batch config as omtra_task",
                "validity and uniqueness are computed by the base adapter from raw sample counts, "
                "never reported by the backend",
                "allene and small-ring rates over every molecule OMTRA wrote are surfaced as warnings",
                "every molecule leaves as TrustLabel.PROPOSED",
            ],
        )

    # --- batch record ------------------------------------------------------
    def _generate(self, metered: dict[str, float], **kwargs: Any) -> GenerationBatch:
        """Stamp which half of the multitask model produced this batch.

        The base builds ``config`` before ``_sample`` runs, so the task names are
        written on afterwards. They belong on the record rather than only in this
        docstring: OMTRA's generation and docking tasks share a checkpoint, and six
        months later "which one made these molecules" must be answerable from the
        artifact, not from reading the adapter.
        """
        batch = super()._generate(metered, **kwargs)
        batch.config["omtra_task"] = OMTRA_DENOVO_TASK
        batch.config["omtra_task_fixes"] = ["protein_identity", "protein_structure"]
        batch.config["omtra_task_generates"] = ["ligand_identity_condensed", "ligand_structure"]
        batch.config["omtra_docking_task_not_used"] = OMTRA_DOCKING_TASK
        batch.config["omtra_checkpoint_relpath"] = OMTRA_CHECKPOINT_RELPATH
        batch.config["omtra_n_timesteps"] = OMTRA_N_TIMESTEPS
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
        """One driver call: ``cli.run_sample`` on the de novo task, then its SDF.

        ``batch_size`` is accepted and not forwarded. OMTRA has no batch argument
        — ``cli.run_sample`` rewrites ``--n_samples`` into ``n_replicates`` of a
        single system and ``routines.sample.main`` passes the whole count to
        ``model.sample`` in one call — so splitting the request here would change
        the number of forward passes without changing the result, and pretending
        the knob exists would be worse than saying it does not.
        """
        pdb_path = pocket.get("pdb_path")
        if not pdb_path:
            raise OmtraGenerationError(
                f"{self.capability_id} needs a protein structure: PocketSpec.pdb_path is unset, "
                "and cli.py's --protein_file has no default"
            )
        if self.ext.root is None:  # pragma: no cover - require_available() precedes this
            raise OmtraGenerationError("the OMTRA checkout disappeared between probe and call")

        warnings: list[str] = []
        if context_smiles:
            warnings.append(
                f"{len(context_smiles)} context SMILES were supplied and ignored: "
                f"{OMTRA_DENOVO_TASK} conditions on the pocket only. A ligand would be "
                f"consumed by {OMTRA_DOCKING_TASK} via --ligand_file, which is docking, "
                "not generation, and this capability does not invoke it"
            )

        request: dict[str, Any] = {
            "protein_file": str(pdb_path),
            "checkpoint": str(self.ext.root / OMTRA_CHECKPOINT_RELPATH),
            "task": OMTRA_DENOVO_TASK,
            "n_molecules": int(n_molecules),
            "n_timesteps": OMTRA_N_TIMESTEPS,
            "seed": int(seed),
            # Carried even though upstream's routines/sample.py chooses its own
            # device: the driver uses it to hide the GPU before importing torch,
            # which is the only way to hold OMTRA to a CPU run. That is not
            # hypothetical here — OMTRA pins torch 2.4.0+cu121, whose kernels stop
            # at sm_90, so on a newer card `torch.cuda.is_available()` is True and
            # the first kernel launch fails instead. Every sibling driver is sent
            # its device the same way.
            "device": self.ext.device,
        }
        radius = pocket.get("radius")
        request["bbox_length"] = 2.0 * float(radius) if radius else OMTRA_DEFAULT_BBOX_LENGTH
        center = pocket.get("center")
        if center:
            request["pocket_center"] = [float(c) for c in center]
        else:
            # cli._validate_task_inputs refuses a protein-conditioned task with no
            # pocket definition at all, so the driver derives one from the protein
            # file's own atoms — the rule cli.run_sample itself applies when the
            # pocket was given as residues. It reports the coordinates it used.
            warnings.append(
                "PocketSpec.center is unset, so the driver centres OMTRA's bounding box on the "
                f"mean heavy-atom coordinate of {pdb_path}. Supply a centre, and a pocket PDB "
                "rather than a whole protein, or the box will not be the site"
            )

        run = self.ext.run_json(request, timeout_s=timeout_s, error_cls=OmtraGenerationError, seed=seed)
        payload = run.payload
        records = list(payload.get("molecules") or [])
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        warnings.extend(self._reconstruction_warnings(payload, len(records)))
        errors = [str(e) for e in payload.get("errors") or []]
        return records, warnings, errors

    @staticmethod
    def _reconstruction_warnings(payload: dict[str, Any], n_records: int) -> list[str]:
        """Report the counts that say whether the sampled bond blocks were sane.

        These are warnings, not per-molecule ``qc_flags``, and the distinction is
        deliberate: the base adapter drops any molecule carrying a flag unless
        ``keep_invalid``, so flagging every allene would turn an observation into a
        silent filter. A reader is told the rate and decides.
        """
        written = int(payload.get("n_written") or n_records)
        if not written:
            return []
        out: list[str] = []
        for key, label in (
            ("n_allene", "match the allene SMARTS [*]=[#6]=[*]"),
            ("n_small_ring", "carry a 3- or 4-membered ring"),
        ):
            count = payload.get(key)
            if count is None:
                continue
            out.append(
                f"OMTRA reconstruction: {int(count)}/{written} molecules {label} ({int(count) / written:.0%})"
            )
        sanitizable = payload.get("n_sanitizable")
        if sanitizable is not None:
            out.append(
                f"OMTRA reconstruction: {int(sanitizable)}/{written} molecules survived "
                "Chem.SanitizeMol; build_molecule does not sanitize before writing the SDF"
            )
        return out


__all__ = [
    "OMTRA_ADAPTER_VERSION",
    "OMTRA_CHECKPOINT_RELPATH",
    "OMTRA_DENOVO_TASK",
    "OMTRA_DOCKING_TASK",
    "SPEC",
    "OmtraAdapter",
    "OmtraGenerationError",
]
