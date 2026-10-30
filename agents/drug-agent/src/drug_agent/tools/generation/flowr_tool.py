"""FLOWR — the published pocket-conditioned flow-matching generator, pinned to one checkpoint.

FLOWR and FLOWR.root are two different models that ship from one repository, which is why
this package binds them as two capabilities over one checkout. FLOWR is the model published
in Nature Computational Science (``s43588-026-00998-8``): flow matching over ligand atom
positions, types and bonds inside a binding site, with interaction-aware and fragment-aware
conditioning. FLOWR.root is the later foundation model that adds a joint affinity head and
ships its own checkpoints. The Python in ``.external/flowr_root`` is FLOWR.root's; what makes
a run *FLOWR* rather than *FLOWR.root* is the checkpoint it loads, and nothing else.

The catalogue's warning — do not silently treat different FLOWR checkpoints as identical —
is therefore not a footnote here, it is the design. :data:`SPEC` pins exactly one file,
``ckpts/flowr.ckpt``, and requires a digest declared beside it in ``ckpts/flowr.ckpt.sha256``;
:mod:`scripts.external.flowr_sample` re-hashes the checkpoint before it builds a model and
refuses to sample when the two disagree; and the digest it did match, together with the
checkpoint's own hyper-parameter fingerprint, is written into the batch's ``warnings`` so a
batch of molecules can never be read back without knowing which model made it. An adapter
that accepted whatever ``.ckpt`` it found would make ``molecule.generate.flowr`` and
``molecule.generate.flowr_root`` two names for the same unknown thing.

**What it would compute.** One ``generate`` operation, sampling ligands into a prepared
pocket and returning a
:class:`~drug_agent.tools.pfm_tool.GenerationBatch` of ``TrustLabel.PROPOSED`` molecules:

    ``molecules[].smiles``        canonical SMILES of each sampled ligand
    ``molecules[].sdf_block``     its generated 3D pose, in the pocket's frame
    ``n_raw`` / ``n_valid``       samples written by FLOWR, and how many RDKit parsed
    ``n_unique``                  distinct canonical structures among those
    ``generation_validity``       ``n_valid / n_raw``, computed here, never by the backend
    ``generation_uniqueness``     ``n_unique / n_valid``
    ``warnings[]``                the checkpoint digest and hparams fingerprint of the run

Upstream's own ``--filter_valid_unique``, ``--filter_diversity`` and PoseBusters filters are
deliberately **not** passed. The base class computes validity and uniqueness from raw counts,
and a backend that pre-filters its own output would make those two numbers describe a
survivor set rather than a sample. What upstream does unconditionally — dropping
reconstruction failures with ``sanitize_list`` before writing its SDF
(``flowr/gen/generate_from_pdb.py:260``) — cannot be turned off, so a collapse in
reconstruction quality shows up as ``n_kept`` falling short of ``n_requested``, not as
invalid SMILES. Predicted affinities that a checkpoint's affinity head writes into the SDF
are carried in the driver payload and consumed by nothing: a generator scoring its own
molecules is not evidence, and the activity plane judges them like any other candidate.

**Why it cannot run on this machine.** Six blockers, each verified against the checkout at
``86a25f08161d``:

1. **The checkpoint is absent.** ``.external/flowr_root/ckpts/flowr.ckpt`` does not exist.
   The clone is code-only by construction — the checkout's own ``.gitignore`` excludes
   ``**/*.ckpt`` — and the FLOWR.root README (``Checkpoints``) offers only four files, all
   from one Google Drive folder: ``flowr_root_v2.2.ckpt``, ``flowr_root_v2.ckpt``,
   ``flowr_root_spindr_base.ckpt`` and ``flowr_root_v2_mol.ckpt``. Every one of those is a
   FLOWR.root checkpoint and belongs to the other capability. Nothing in this checkout names
   the published FLOWR checkpoint's filename or its download location; the only trace of the
   published model here is the arXiv badge ``2504.10564`` on line 3 of ``README.md``. So the
   path this capability requires is the platform's pin, not a name upstream ships, and it has
   to be filled by hand from the FLOWR authors' own release.
2. **The pin is absent.** ``ckpts/flowr.ckpt.sha256`` does not exist either, and without it
   the tool stays UNAVAILABLE even once a checkpoint is dropped in. That is deliberate:
   an unpinned checkpoint is the failure mode the catalogue warns about.
3. **The environment is not built.** There is no ``flowr_root`` conda environment on this
   host, and the ``drug-agent`` environment cannot import the run path. Measured:
   ``PYTHONPATH=.external/flowr_root python -c "import flowr.gen.generate_from_pdb"`` fails at
   ``flowr/gen/utils.py:14: from pymol import cmd`` with ``ModuleNotFoundError: No module
   named 'pymol'``. Also absent from that interpreter, all imported at module scope on the
   same path: ``tensordict``, ``biotite``, ``hydride``, ``pdbinf``, ``posecheck``,
   ``biopandas``. Upstream's ``environment.yml`` sources two of these from channels that are
   not open packages — ``schrodinger::pymol-bundle=3.1.6.1`` and ``openeye-toolkits``.
4. **The pinned upstream stack does not match this host, and the reason is the GPU
   rather than the CPU.** ``environment.yml`` pins ``pytorch::pytorch=2.5.1`` with
   ``pytorch-cuda=12.4``. Measured here on 2026-09-09: ``x86_64``, torch
   ``2.14.0+cu130``, CUDA 13.0, device ``NVIDIA RTX PRO 6000 Blackwell Workstation
   Edition`` of compute capability ``(12, 0)`` — an sm_120 Blackwell part that postdates
   the CUDA 12.4 toolkit that file pins, whose torch build carries SASS only to sm_90
   and no PTX to JIT from. The upstream environment therefore still has to be rebuilt
   around this host's toolchain rather than created from the file as written; moving to
   x86-64 did not change that.
5. **Sampling is GPU-only by construction.** ``evaluate()`` moves the model with
   ``model.to("cuda")`` at ``flowr/gen/generate_from_pdb.py:82``, unconditionally. There is no
   CPU sampling path to fall back to, and the README asks for "at least 40GB VRAM" for
   inference.
6. **The typed pocket cannot express what FLOWR needs.**
   :class:`~drug_agent.contracts.brief.PocketSpec` carries a pocket PDB and a radius;
   ``process_complex`` (``flowr/data/preprocess_pdbs.py:906``) cuts the pocket around a
   *reference ligand* and needs an SDF/MOL/PDB for it — or a residue list through
   ``--res_txt_file``, which this adapter does not wire. So the ligand is read from
   ``pocket["reference_ligand_sdf"]`` and refused when absent, because a pocket cut with no
   reference is a different experiment, not a default.

**What the platform does instead.** Nothing, visibly. ``molecule.generate.flowr`` reports
UNAVAILABLE with the reason above, ``generate`` raises
:class:`~drug_agent.tools.base.ToolUnavailable`, and no molecule is produced. There is no
surrogate: a FLOWR.root checkpoint is not a stand-in for FLOWR, an unpinned checkpoint is not
a stand-in for the pinned one, and a sampler with no weights has nothing to say about a
pocket. The run continues with this generator absent and the absence recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with FLOWR, not FLOWR's own version.
FLOWR_ADAPTER_VERSION = "1.0.0"

#: The one checkpoint this capability will load, relative to the checkout root.
#: ``ckpts/`` is upstream's own convention for a checkpoint directory inside the
#: repository (``flowr_vis/README.md``: "Put at least one .ckpt file in the ckpts/
#: directory at the project root"); the *filename* is this platform's pin, chosen to
#: be unmistakably not ``flowr_root.ckpt``.
FLOWR_CHECKPOINT_RELPATH = "ckpts/flowr.ckpt"

#: The declared SHA-256 of that checkpoint, written beside it — the output of
#: ``sha256sum flowr.ckpt`` is accepted verbatim. Its absence keeps the capability
#: UNAVAILABLE, because a FLOWR run that cannot say which checkpoint produced it is
#: exactly what the catalogue warns against.
FLOWR_CHECKPOINT_PIN_RELPATH = "ckpts/flowr.ckpt.sha256"

#: Pocket cutoff in Angstrom used to cut the pocket around the reference ligand.
#: Upstream's README and ``scripts/generate_pdb.sl`` both use 7 Å; the argparse default
#: in ``generate_from_pdb.py`` is 6.0, so the value is always passed explicitly rather
#: than inherited from whichever of the two a reader happened to look at.
FLOWR_POCKET_CUTOFF_A = 7.0

#: Integration steps for the ODE sampler — ``DEFAULT_INTEGRATION_STEPS`` in
#: ``flowr/gen/generate_from_pdb.py`` and the value ``scripts/generate_pdb.sl`` uses.
FLOWR_INTEGRATION_STEPS = 100

#: How many times the sampling loop may go round before giving up on reaching
#: ``n_molecules`` (``--max_sample_iter``, upstream default 20).
FLOWR_MAX_SAMPLE_ITER = 20

SPEC = ExternalSpec(
    name="FLOWR (published checkpoint)",
    env_prefix="FLOWR",
    # The same clone as molecule.generate.flowr_root. Sharing the checkout is correct —
    # it is one codebase — and is safe only because the two capabilities pin different
    # weights, which is the entire point of them being two capabilities.
    checkout_relpath=".external/flowr_root",
    sentinel_files=(
        "flowr/gen/generate_from_pdb.py",
        "flowr/gen/generate.py",
        "flowr/gen/utils.py",
        "flowr/scriptutil.py",
        "flowr/models/fm_pocket.py",
        "flowr/models/pocket.py",
        "flowr/util/pocket.py",
    ),
    driver_name="flowr_sample.py",
    # torch plus lightning identifies the environment FLOWR was installed into; the long
    # list below is what a half-built one loses.
    selection_modules=("torch", "lightning"),
    # Every module the sampling path imports at module scope, read off the import
    # statements rather than guessed: flowr/gen/utils.py (pymol, lightning, yaml),
    # flowr/util/pocket.py (biotite, MDAnalysis, prolif, tensordict), flowr/util/rdkit.py
    # (biotite), flowr/util/functional.py (hydride, pdbinf), flowr/util/metrics.py
    # (posecheck, biopandas, dill), flowr/data/dataset.py (lmdb), flowr/scriptutil.py
    # (openbabel, torchmetrics). pymol is the one that actually fails first here.
    runtime_modules=(
        "torch",
        # FLOWR's run path imports it; leaving it out let the card report
        # AVAILABLE on an interpreter that cannot construct the model.
        "torch_geometric",
        "lightning",
        "torchmetrics",
        "rdkit",
        "numpy",
        "scipy",
        "yaml",
        "tensordict",
        "lmdb",
        "biotite",
        "MDAnalysis",
        "prolif",
        "pymol",
        "hydride",
        "pdbinf",
        "posecheck",
        "biopandas",
        "dill",
        "openbabel",
    ),
    # Both entries are required. The digest is not an optional extra: it is what
    # distinguishes this capability's checkpoint from any other FLOWR-family file.
    weight_relpaths=(FLOWR_CHECKPOINT_RELPATH, FLOWR_CHECKPOINT_PIN_RELPATH),
    conda_envs=("flowr_root", "drug-agent"),
    upstream_url="https://github.com/jule-c/flowr_root",
    upstream_ref="86a25f08161d",
    license="MIT (LICENSE); the flowr_vis/ app is under a separate source-available licence",
    # Short on purpose: this string is repeated once per unmet requirement inside the
    # unavailable reason, and the long version of the story is the module docstring.
    install_hint=(
        "make external-clone; conda env create -f .external/flowr_root/environment.yml; "
        "put the published FLOWR checkpoint at .external/flowr_root/ckpts/flowr.ckpt and run "
        "`sha256sum flowr.ckpt > flowr.ckpt.sha256` beside it"
    ),
    needs_gpu=True,
)


class FlowrCheckpointError(GenerationBackendError):
    """The pinned FLOWR checkpoint is missing, unpinned, mismatched, or unloadable here.

    Separate from :class:`~drug_agent.tools.base.ToolUnavailable`, which says the backend
    was never there. This one says a checkpoint was found and rejected — the digest did not
    match the pin, or its ``hyper_parameters`` are not ones this release's model constructor
    can consume. Refusing loudly is the point: silently sampling from the wrong FLOWR
    checkpoint would produce molecules nobody could attribute.
    """


class FlowrAdapter(GenerativeToolAdapter):
    """Pocket-conditioned 3D ligand generation with the published FLOWR checkpoint."""

    capability_id = "molecule.generate.flowr"
    spec = SPEC
    engine = "flowr"
    version = FLOWR_ADAPTER_VERSION
    conditioning = "pocket"
    #: Atoms and bonds, not reactions. FLOWR proposes structures with no route attached.
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned 3D generation (FLOWR, published checkpoint)",
            description=(
                "Flow matching over ligand atom positions, types and bonds inside a prepared "
                "binding site, as published in Nature Computational Science. Pinned to one "
                "checkpoint by SHA-256 and refuses to sample from any other, so it can never be "
                "confused with molecule.generate.flowr_root, which shares the same checkout."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            # A flow-matching sampler. The seed is forwarded to L.seed_everything, but the
            # run is a sampler either way and must never be described as reproducible science.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint SHA-256 does not match ckpts/flowr.ckpt.sha256 — a FLOWR.root "
                "checkpoint put in FLOWR's place is refused, never sampled",
                "checkpoint hyper_parameters lack a key this release's constructor indexes "
                "(pocket-d_equi, add_feats, predict_affinity, ...), so load_model raises KeyError",
                "apo-trained checkpoint: it needs --arch pocket_flex, which this release raises "
                "NotImplementedError for at flowr/scriptutil.py:1272",
                "hparams['coord_scale'] != 1.0 leaves coord_std unbound at flowr/gen/utils.py:413",
                "no reference ligand in the pocket spec, so process_complex cannot cut the pocket",
                "torch >= 2.6 defaults torch.load to weights_only=True while upstream calls it "
                "bare at flowr/scriptutil.py:1048, so a checkpoint pickling non-primitive "
                "hyper-parameters raises UnpicklingError before the model is built",
                "GPU-only: evaluate() calls model.to('cuda') unconditionally "
                "(flowr/gen/generate_from_pdb.py:82)",
                "point-cloud reconstruction failures are dropped by upstream before its SDF is "
                "written, so they appear as n_kept short of n_requested rather than as invalid SMILES",
            ],
            validation_checks=[
                "the checkpoint's SHA-256 equals the pin in ckpts/flowr.ckpt.sha256, checked "
                "before any model is built",
                "the checkpoint's hyper_parameters carry every key this release hard-indexes, "
                "and pocket_noise resolves to a holo-pocket model",
                "validity and uniqueness are computed here from the raw SDF, never taken from "
                "upstream's own filters, which are left off",
                "the checkpoint digest and hparams fingerprint are recorded on every batch",
                "every molecule leaves as TrustLabel.PROPOSED, screened by the same pipeline as "
                "any other candidate",
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
        """Run one pinned-checkpoint FLOWR sampling job and parse what it wrote."""
        root = self.ext.root
        if root is None:  # require_available() has already run; this is the type narrowing
            raise FlowrCheckpointError("no FLOWR checkout resolved")

        pocket_pdb = self._required_file(
            pocket.get("pdb_path"),
            "a prepared pocket or protein PDB at generation.pocket.pdb_path",
        )
        reference_ligand = self._required_file(
            pocket.get("reference_ligand_sdf") or pocket.get("ligand_sdf"),
            "a reference ligand (SDF/MOL/PDB) at generation.pocket.reference_ligand_sdf — "
            "FLOWR cuts the pocket around it (flowr/data/preprocess_pdbs.py:906)",
        )

        request = {
            "checkout": str(root),
            "checkpoint": str(root / FLOWR_CHECKPOINT_RELPATH),
            "checkpoint_pin": str(root / FLOWR_CHECKPOINT_PIN_RELPATH),
            "pdb_file": str(pocket_pdb),
            "ligand_file": str(reference_ligand),
            "n_molecules": int(n_molecules),
            "batch_cost": max(1, int(batch_size)),
            "seed": int(seed),
            "pocket_cutoff": FLOWR_POCKET_CUTOFF_A,
            "integration_steps": FLOWR_INTEGRATION_STEPS,
            "max_sample_iter": FLOWR_MAX_SAMPLE_ITER,
        }
        # Not run_driver(): that spends self.timeout_s, and the brief's
        # generation.max_runtime_seconds arrives here as timeout_s. The budget the caller
        # asked for is the one the subprocess gets.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=FlowrCheckpointError,
            seed=seed,
        )
        payload = run.payload
        records = list(payload.get("molecules") or [])
        warnings = [self._provenance(payload)]
        warnings += [str(w) for w in (payload.get("warnings") or [])]
        if context_smiles:
            warnings.append(
                f"{len(context_smiles)} context SMILES were ignored: this capability samples "
                "de novo into the pocket and does not condition on ligand SMILES"
            )
        radius = pocket.get("radius")
        if radius is not None and float(radius) != FLOWR_POCKET_CUTOFF_A:
            warnings.append(
                f"pocket.radius={float(radius):.1f} A was not used; the pocket was cut at "
                f"{FLOWR_POCKET_CUTOFF_A:.1f} A, the cutoff upstream's own generate_pdb.sl uses"
            )
        warnings.append(
            "upstream drops reconstruction failures before writing its SDF, so n_raw counts "
            "molecules FLOWR kept, not attempts it made"
        )
        return records, warnings, [str(e) for e in (payload.get("errors") or [])]

    def _provenance(self, payload: dict[str, Any]) -> str:
        """The one line that says which FLOWR made these molecules.

        The base class builds ``GenerationBatch.config`` before ``_sample`` runs, so the
        warnings list is the channel that reaches the batch. That is not a workaround: a
        batch whose provenance is missing *should* read as something to look at.
        """
        info = payload.get("checkpoint") or {}
        digest = str(info.get("sha256") or "unknown")
        fingerprint = info.get("hparams") or {}
        parts = ", ".join(f"{k}={v}" for k, v in sorted(fingerprint.items()))
        return (
            f"FLOWR checkpoint {Path(str(info.get('path') or FLOWR_CHECKPOINT_RELPATH)).name} "
            f"sha256={digest} matched the pin in {FLOWR_CHECKPOINT_PIN_RELPATH}"
            + (f"; hparams {parts}" if parts else "")
        )

    @staticmethod
    def _required_file(value: Any, what: str) -> Path:
        """A path that must really be a file, or a refusal that says what is wanted."""
        if not value:
            raise FlowrCheckpointError(f"FLOWR needs {what}; nothing was supplied")
        path = Path(str(value))
        if not path.is_file():
            raise FlowrCheckpointError(f"FLOWR needs {what}; {path} is not a readable file")
        return path.resolve()


__all__ = [
    "FLOWR_ADAPTER_VERSION",
    "FLOWR_CHECKPOINT_PIN_RELPATH",
    "FLOWR_CHECKPOINT_RELPATH",
    "SPEC",
    "FlowrAdapter",
    "FlowrCheckpointError",
]
