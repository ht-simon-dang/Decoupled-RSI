"""FLOWR.root — one foundation model, two conditioning modes, and neither may be read as the other.

FLOWR.root (*A flow matching based foundation model for joint multi-purpose
structure-aware 3D ligand generation and affinity prediction*, arXiv:2510.02578,
https://github.com/jule-c/flowr_root, pinned at ``86a25f08161d``) integrates a
flow-matching ODE over ligand atom positions, element types, charges,
hybridisation and bonds, and ships an affinity head alongside the generator. The
release is two models behind one name, and this adapter keeps them apart because
conflating them is the one error that cannot be spotted downstream — a SMILES
string carries no record of what it was conditioned on.

``mode="pocket"`` — the default
    Runs ``flowr.gen.generate_from_pdb`` with ``--arch pocket``: a protein PDB and
    a reference ligand are turned into a pocket by ``process_complex`` (7 Å heavy-atom
    cutoff, upstream's documented default), and molecules are sampled inside it.
    ``conditioning`` is ``"pocket"``, ``engine`` is ``flowr_root``, and the pinned
    checkpoint is ``ckpts/flowr_root_v2.ckpt``.

``mode="ligand"``
    Runs ``flowr.gen.generate_from_sdf_mol`` with ``--arch flowr``: no protein is
    read at all, and generation is conditioned only on a reference molecule from an
    SDF. ``conditioning`` is ``"ligand"`` and ``engine`` becomes ``flowr_root_mol``,
    so the mode is stamped onto the batch, onto ``batch.backend`` and onto every
    molecule's ``origin``, not only into a config field somebody has to remember to
    read. The pinned checkpoint is ``ckpts/flowr_root_v2_mol.ckpt`` — the README is
    explicit that the pocket checkpoints are not usable for this mode.

The two checkpoints are pinned per mode rather than left to the caller, so the
readiness probe rejects a deployment that has one and not the other instead of
discovering it at sample time.

**Why this capability is UNAVAILABLE on this machine.** Three blockers, none of
them soft:

1. **No checkpoint.** ``.external/flowr_root`` is a code-only clone: there is no
   ``ckpts/`` directory in it at all. Both weight files this adapter pins —
   ``.external/flowr_root/ckpts/flowr_root_v2.ckpt`` for pocket mode and
   ``.external/flowr_root/ckpts/flowr_root_v2_mol.ckpt`` for ligand mode — are
   absent. Upstream never ships them in the repository (its ``.gitignore`` carries
   ``**/*.ckpt``); the README's *Checkpoints* section says every checkpoint is
   downloaded from the authors' Google Drive folder
   ``https://drive.google.com/drive/u/0/folders/1NWpzTY-BG_9C4zXZndWlKwdu7UJNCYj8``.
   The ``ckpts/`` directory name is upstream's own: ``flowr_vis/worker.py`` sets
   ``CKPTS_DIR = ROOT_DIR / "ckpts"`` with ``ROOT_DIR`` the checkout root.
2. **No ``flowr_root`` environment.** ``mamba env create -f environment.yml`` builds
   a conda environment named ``flowr_root``; no such environment exists here, so the
   readiness probe falls back to ``drug-agent``, which cannot import ``pymol``,
   ``biotite``, ``hydride``, ``pdbinf``, ``tensordict`` or ``biopandas``. This is not
   a soft dependency: ``flowr/gen/utils.py`` does ``from pymol import cmd`` at module
   scope, and both entry points import that module, so neither can even be imported
   without PyMOL present.
3. **The pinned environment now resolves, and this blocker is retracted.**
   ``environment.yml`` pins ``schrodinger::pymol-bundle=3.1.6.1``, which has no
   ``linux-aarch64`` build — that was blocker 3 on the previous host. Re-checked here
   on 2026-09-09: ``conda search -c schrodinger --platform linux-64 pymol-bundle``
   lists ``3.1.6.1``, so on this x86-64 host the file can be created as written. The
   aarch64 substitution that would have been needed — ``conda-forge::pymol-open-source``
   3.1.0, which does publish ``linux-aarch64`` builds and supplies the same ``pymol``
   module — is recorded here because it is the fix on that architecture, not because
   it is needed on this one. Blockers 1 and 2 are what remain, and both are downloads
   and an install, not a machine.

Because of those, this capability reports UNAVAILABLE, ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable`, and the platform generates
nothing from FLOWR.root. There is no surrogate: no BRICS recombination, no
smaller model standing in, no cached sample set replayed as though it had just
been generated.

**What it would compute once the weights are present.** ``generate()`` returns a
:class:`~drug_agent.tools.pfm_tool.GenerationBatch` assembled by
:class:`~drug_agent.tools.generation.GenerativeToolAdapter` from what the upstream
entry point actually wrote:

    ``molecules[].smiles``        canonical SMILES, RDKit-parsed and de-duplicated here
    ``molecules[].sdf_block``     the generated 3D conformer, as FLOWR.root wrote it
    ``molecules[].origin``        ``flowr_root`` or ``flowr_root_mol`` — the mode
    ``n_raw`` / ``n_valid``       counts over the driver's records, computed by the base
    ``generation_uniqueness``     distinct canonical structures / valid, from those counts
    ``config.conditioning``       ``"pocket"`` or ``"ligand"``
    ``config.flowr_checkpoint``   the checkpoint file the sample was drawn from
    ``config.flowr_entry_point``  the upstream module that was executed

**What it deliberately does not do.**

*It does not report the model's raw reconstruction validity.* Upstream's
``evaluate()`` sanitises its own output before writing the SDF — in
``flowr/gen/generate_from_pdb.py`` the ``smolRD.sanitize_list(..., sanitize=True)``
call runs on both branches of ``if not args.filter_valid_unique``. So the samples
that reach this adapter have already had unparseable ones dropped, and
``generation_validity`` measured over them is close to 1.0 by construction. That is
a property of the entry point, not a measurement of the model, and the driver
attaches a warning saying so to every batch. ``--filter_valid_unique`` and
``--filter_diversity`` are deliberately *not* passed, so de-duplication, the
heavy-atom window and the metrics stay this package's, computed from one set of
counts for all generators.

*It does not surface the affinity head.* FLOWR.root writes ``pic50``, ``pki``,
``pkd`` and ``pec50`` onto each generated molecule
(``flowr/gen/generate.py``). The driver reads them back and returns them verbatim,
and this adapter drops them: a generator does not get to author its own evidence,
and constructing a :class:`~drug_agent.contracts.predictions.Prediction` belongs to
the specialist agents. A molecule leaves here as ``TrustLabel.PROPOSED`` with no
affinity attached, and is judged by the same pipeline as every other candidate.

*It does not invent a reference ligand.* Both upstream entry points require one —
``--ligand_file`` for the pocket path, a required ``--sdf_path`` for the ligand
path — and in pocket mode it also defines where the pocket is cut. Embedding a
conformer from a SMILES string would put that reference at an arbitrary position
with no relation to the binding site, so when no reference file is supplied this
adapter raises :class:`GenerationBackendError` naming exactly what is missing
rather than sampling against a fiction.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with FLOWR.root, not FLOWR.root's own version.
FLOWR_ROOT_ADAPTER_VERSION = "1.0.0"

#: Protein-conditioned generation: ``flowr.gen.generate_from_pdb``, ``--arch pocket``.
MODE_POCKET = "pocket"
#: Ligand-only generation: ``flowr.gen.generate_from_sdf_mol``, ``--arch flowr``.
MODE_LIGAND = "ligand"

#: Checkpoint, entry-point module and ``--arch`` value for each mode. The paths are
#: relative to the checkout because that is where upstream's own tooling looks:
#: ``flowr_vis/worker.py`` computes ``CKPTS_DIR = ROOT_DIR / "ckpts"``. The file names
#: are the README's, and the README is explicit that the ligand-only mode needs
#: ``flowr_root_v2_mol.ckpt`` — a pocket checkpoint is a different model, not a
#: degraded one.
_MODES: dict[str, dict[str, str]] = {
    MODE_POCKET: {
        "checkpoint": "ckpts/flowr_root_v2.ckpt",
        "entry_point": "flowr.gen.generate_from_pdb",
        "arch": "pocket",
        "engine": "flowr_root",
        "conditioning": "pocket",
    },
    MODE_LIGAND: {
        "checkpoint": "ckpts/flowr_root_v2_mol.ckpt",
        "entry_point": "flowr.gen.generate_from_sdf_mol",
        "arch": "flowr",
        "engine": "flowr_root_mol",
        "conditioning": "ligand",
    },
}

SPEC = ExternalSpec(
    name="FLOWR.root",
    env_prefix="FLOWR_ROOT",
    checkout_relpath=".external/flowr_root",
    # Every one of these is imported by the run path: the two entry points, the
    # sampler they call, the loader that reads the checkpoint, and the pocket model
    # the checkpoint's state dict is loaded into.
    sentinel_files=(
        "flowr/gen/generate_from_pdb.py",
        "flowr/gen/generate_from_sdf_mol.py",
        "flowr/gen/generate.py",
        "flowr/gen/utils.py",
        "flowr/scriptutil.py",
        "flowr/models/pocket.py",
    ),
    driver_name="flowr_root_sample.py",
    # Narrow, and deliberately satisfiable by a half-built environment: selection
    # only has to identify *which* interpreter the backend was installed into, and
    # readiness then rejects it module by module with a reason.
    selection_modules=("torch", "lightning"),
    # What a broken install actually loses, read off the import statements on the
    # path from `flowr.gen.generate_from_pdb` down. `pymol` is first among equals:
    # `flowr/gen/utils.py` imports it at module scope, so it gates the import of the
    # entry point itself. The vendored `posebusters` and `posecheck` packages sit in
    # the checkout rather than site-packages and are deliberately not probed here —
    # the probe globs site-packages, and listing them would report a false absence.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "lightning",
        "torchmetrics",
        "tensordict",
        "pymol",
        "openbabel",
        "biotite",
        "hydride",
        "pdbinf",
        "prolif",
        "MDAnalysis",
        "biopandas",
        "lmdb",
        "dill",
    ),
    # Pocket mode's checkpoint. Ligand mode swaps this per instance — see
    # FlowrRootAdapter.__init__.
    weight_relpaths=(_MODES[MODE_POCKET]["checkpoint"],),
    conda_envs=("flowr_root", "drug-agent"),
    upstream_url="https://github.com/jule-c/flowr_root",
    upstream_ref="86a25f08161d",
    # Two licences in one checkout, and only one of them covers what runs here.
    license=(
        "MIT (LICENSE, Copyright (c) 2025-2026 Julian Cremer) covers the FLOWR.root model "
        "code — the `flowr/` package, which is all this adapter imports and runs. The "
        "FLOWR.ui visualisation app in `flowr_vis/` is NOT MIT: it carries a separate "
        "source-available licence (flowr_vis/LICENSE) permitting internal, "
        "non-redistributive use only. This adapter uses no part of `flowr_vis/`. "
        "Model weights are distributed separately by the authors and carry their own terms."
    ),
    install_hint=(
        "make external-clone, then download flowr_root_v2.ckpt (pocket) and "
        "flowr_root_v2_mol.ckpt (ligand-only) from the Google Drive folder linked in the "
        "FLOWR.root README into .external/flowr_root/ckpts/, and build the flowr_root conda "
        "environment from .external/flowr_root/environment.yml"
    ),
    needs_gpu=True,
)


class FlowrRootAdapter(GenerativeToolAdapter):
    """FLOWR.root sampling, in exactly one conditioning mode per instance."""

    capability_id = "molecule.generate.flowr_root"
    spec = SPEC
    engine = "flowr_root"
    version = FLOWR_ROOT_ADAPTER_VERSION
    conditioning = MODE_POCKET
    #: FLOWR.root samples atoms and bonds, not reactions. It reports no route.
    synthesis_aware = False

    def __init__(
        self,
        *,
        mode: str = MODE_POCKET,
        reference_ligand: str | Path | None = None,
        settings: Any = None,
        timeout_s: float | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(
                f"unknown FLOWR.root mode {mode!r}; expected one of {sorted(_MODES)}. "
                "The mode decides which checkpoint is loaded and what the samples are "
                "conditioned on, so there is no default that covers both."
            )
        cfg = _MODES[mode]
        self.mode = mode
        # Instance attributes, set before super().__init__ reads them. `conditioning`
        # is what `_generate` writes into the batch config, and `engine` is what lands
        # on `batch.backend` and on every molecule's `origin` — so a ligand-only sample
        # cannot be read back as pocket-conditioned at any of the three places a reader
        # might look.
        self.conditioning = cfg["conditioning"]
        self.engine = cfg["engine"]
        self.entry_point = cfg["entry_point"]
        self.arch = cfg["arch"]
        self.checkpoint_relpath = cfg["checkpoint"]
        # The frozen module-level SPEC pins pocket mode's checkpoint; ligand mode needs
        # a different file, and requiring both would report a deployment that has the
        # right one as unusable.
        self.spec = SPEC if mode == MODE_POCKET else replace(SPEC, weight_relpaths=(cfg["checkpoint"],))
        self.reference_ligand = Path(reference_ligand) if reference_ligand else None
        super().__init__(settings=settings, timeout_s=timeout_s)

    # --- capability --------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        pocket_mode = self.mode == MODE_POCKET
        return self.build_profile(
            title=f"De-novo 3D molecule generation, {self.conditioning}-conditioned (FLOWR.root)",
            description=(
                "Flow-matching foundation model sampling ligand coordinates, element types, "
                "charges, hybridisation and bonds. "
                + (
                    "Pocket-conditioned: a protein PDB and a reference ligand define a binding "
                    "site and molecules are generated inside it."
                    if pocket_mode
                    else "Ligand-only: no protein is read; generation is conditioned on a "
                    "reference molecule from an SDF."
                )
                + " Proposes molecules; scores nothing."
            ),
            input_schema_ref="PocketSpec" if pocket_mode else "Molecule",
            output_schema_ref="Candidate",
            # A flow-matching ODE integrated from a random prior. A seed pins one run;
            # the model is a sampler.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint absent — the released weights are a separate Google Drive download, "
                "not part of the clone",
                "wrong checkpoint for the mode: flowr_root_v2_mol.ckpt is ligand-only and the "
                "pocket checkpoints are not interchangeable with it",
                "flowr/gen/utils.py imports pymol at module scope, so a missing PyMOL fails the "
                "entry point at import time rather than at sample time",
                "upstream's evaluate() calls model.to('cuda') unconditionally, so a CPU-only host "
                "raises instead of running slowly",
                "the README states the joint generation+affinity model is not fully converged and "
                "that clash counts can be high on out-of-distribution pockets",
                "GeometricMolSDFDataset caches the parsed reference as sdf_mol_data.smol beside "
                "the SDF, so a stale cache silently replaces a new reference (ligand mode)",
                "max_sample_iter reached before the requested count, returning a short batch",
            ],
            validation_checks=[
                "the checkpoint pinned for this mode exists before the capability reports AVAILABLE",
                "the mode is recorded on the batch config, on batch.backend and on every molecule's origin",
                "every returned SMILES is re-parsed and re-canonicalised by RDKit here, not "
                "trusted from the driver",
                "validity and uniqueness are computed from the driver's own record counts",
                "affinity-head values are read but never turned into a Prediction",
            ],
        )

    # --- generation --------------------------------------------------------
    def _generate(self, metered: dict[str, float], **kwargs: Any) -> GenerationBatch:
        """Assemble the batch, then stamp which model actually produced it.

        The base records ``conditioning``; these three fields make the record
        auditable rather than merely present — the exact checkpoint file, the exact
        upstream module, and the ``--arch`` it was run with.
        """
        batch = super()._generate(metered, **kwargs)
        batch.config["flowr_mode"] = self.mode
        batch.config["flowr_checkpoint"] = self.checkpoint_relpath
        batch.config["flowr_entry_point"] = self.entry_point
        batch.config["flowr_arch"] = self.arch
        return batch

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
        """Run the upstream entry point once and return its records unscreened."""
        root = self.ext.root
        if root is None:  # unreachable via generate(); require_available() ran first
            raise GenerationBackendError("FLOWR.root checkout could not be resolved")

        reference = self._resolve_reference_ligand(pocket)
        request: dict[str, Any] = {
            "mode": self.mode,
            "arch": self.arch,
            "entry_point": self.entry_point,
            "checkpoint": str(root / self.checkpoint_relpath),
            "reference_sdf": str(reference),
            "n_molecules": int(n_molecules),
            "batch_cost": int(batch_size),
            "seed": int(seed),
            "device": self.ext.device,
        }
        if self.mode == MODE_POCKET:
            request["protein_pdb"] = str(self._resolve_protein_pdb(pocket))

        # Not `run_driver`: that helper hard-wires ``self.timeout_s``, and a caller's
        # ``max_runtime_seconds`` has to be the budget the child is actually killed on.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            seed=seed,
        )
        payload = run.payload
        records = [item for item in (payload.get("molecules") or []) if isinstance(item, dict)]
        warnings = [str(w) for w in (payload.get("warnings") or [])]
        errors = [str(e) for e in (payload.get("errors") or [])]
        if not records and not errors:
            errors.append(
                f"FLOWR.root ran for {run.seconds:.1f}s and wrote no molecules to "
                f"{payload.get('sdf_path') or 'its output SDF'}"
            )
        return records, warnings, errors

    # --- inputs ------------------------------------------------------------
    def _resolve_protein_pdb(self, pocket: dict[str, Any]) -> Path:
        """The structure ``generate_from_pdb`` cuts the pocket out of."""
        raw = pocket.get("pdb_path")
        if not raw:
            raise GenerationBackendError(
                "pocket-conditioned FLOWR.root needs PocketSpec.pdb_path, a protein or pocket "
                "PDB file; none was supplied. Use mode='ligand' to generate without a protein."
            )
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise GenerationBackendError(f"pocket PDB {path} does not exist")
        return path

    def _resolve_reference_ligand(self, pocket: dict[str, Any]) -> Path:
        """The reference molecule both entry points require.

        Three sources, in precedence order: the ``ext`` namespace of a
        :class:`~drug_agent.contracts.brief.PocketSpec` (contracts are
        ``extra="forbid"``, so ``ext`` is where a caller is allowed to put this), a
        plain ``ligand_path`` key when the caller passed a raw dict, and the
        ``reference_ligand`` this adapter was constructed with.
        """
        ext = pocket.get("ext")
        candidates = [
            ext.get("ligand_path") if isinstance(ext, dict) else None,
            pocket.get("ligand_path"),
            self.reference_ligand,
        ]
        raw = next((c for c in candidates if c), None)
        if raw is None:
            raise GenerationBackendError(
                "FLOWR.root needs a reference ligand file: generate_from_pdb passes it as "
                "--ligand_file to cut the pocket, and generate_from_sdf_mol requires it as "
                "--sdf_path. Supply it as FlowrRootAdapter(reference_ligand=...) or as "
                "PocketSpec.ext['ligand_path']. It is not synthesised from a SMILES string, "
                "because an arbitrary embedded conformer has no relation to the binding site."
            )
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise GenerationBackendError(f"FLOWR.root reference ligand {path} does not exist")
        return path


__all__ = [
    "FLOWR_ROOT_ADAPTER_VERSION",
    "MODE_LIGAND",
    "MODE_POCKET",
    "SPEC",
    "FlowrRootAdapter",
]
