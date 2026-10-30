"""ShEPhERD — bioisosteres of a *reference ligand*, not occupants of a pocket.

ShEPhERD is a joint diffusion model over a molecule's 3D graph and three
interaction profiles: its solvent-accessible surface (x2/x3 shape), the
electrostatic potential sampled on that surface (x3), and its pharmacophores with
their direction vectors (x4). Conditional generation is inpainting: the surface,
the electrostatics and the pharmacophores of a *known molecule* are held fixed
while the atoms are denoised, so what comes out is a different chemical graph that
presents the same interaction profile. That is bioisosterism, stated as a
generative objective.

**It is a ligand-conditioned model and must never be catalogued as a pocket
generator.** Nothing in the upstream checkout accepts a receptor. Every conditional
entry point — ``examples/conditional_generation.ipynb``,
``examples/paper_experiments/run_inference_pdb.py``,
``run_inference_natural_products.py`` — begins with an RDKit ``Mol`` carrying a
conformer, and ``shepherd.inference.sampler.generate`` takes ``surface``,
``electrostatics``, ``pharm_types``, ``pharm_pos`` and ``pharm_direction``, all of
which are extracted from that molecule. Its own ``run_inference_pdb.py`` reads PDB
*ligands* out of ``data/conformers/pdb/molblock_charges_pdb_lowestenergy.pkl``; the
protein never enters. :attr:`conditioning` is therefore ``"ligand"``, and
:meth:`_sample` raises rather than run when it is handed a pocket and no reference
— an unconditioned sample dressed up as a targeted one is the failure this refusal
exists to prevent.

**Which of the two repositories this needs.** Only ``.external/shepherd``. The
upstream README states that the parts of ``shepherd-score`` needed to train or
sample were copied into ``src/shepherd/shepherd_score_utils/``, and the driver
imports the surface, electrostatics and pharmacophore functions from there.
``.external/shepherd-score`` (pinned at ``20ebed761e05``) is the evaluation half —
3D similarity scoring, SE(3) alignment, post-relaxation validity — which is
judgement, not generation, and is never imported on this path.

**Why it is UNAVAILABLE here.** Four concrete blockers, each verified against the
checkout at ``1707fddc3b87`` or against this interpreter:

1. **The weights are not present.**
   ``.external/shepherd/data/shepherd_chkpts/`` contains exactly one file,
   ``README.md``. The run path needs
   ``data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt`` —
   the filename is ``MODEL_CHECKPOINTS['mosesaq']['filename']`` in
   ``src/shepherd/checkpoint_manager.py``. Both READMEs record that on
   3 September 2025 the weights were stripped from git history with
   ``git-filter-repo`` and moved to the HuggingFace repository
   ``kabeywar/shepherd``; the pre-conversion PyTorch Lightning 1.2 originals
   remain on the authors' Dropbox. A clone therefore carries code only.

2. **``open3d`` is not installed, and PyPI has nothing to install here.**
   It is not an optional visualisation dependency: the 75 x3 surface points come
   from ``get_molecular_surface`` in
   ``src/shepherd/shepherd_score_utils/generate_point_cloud.py``, which builds a
   triangle mesh with Open3D's ball-pivoting reconstruction and samples it with
   ``sample_points_poisson_disk`` (``import open3d as o3d`` at line 15). There is
   no NumPy fallback upstream. **This blocker changed shape on x86-64 and did not
   go away.** On the aarch64 host pip answered ``Could not find a version that
   satisfies the requirement open3d (from versions: none)`` for every target tried,
   because Open3D publishes x86-64 Linux wheels and no aarch64 wheel at all. Re-run
   here on 2026-09-09, ``pip index versions open3d`` on this interpreter (CPython
   3.12, Linux x86-64) resolves ``open3d 0.19.0``. So the *install route* exists now;
   what has not happened is the install. It is still not in any interpreter on this
   host. :attr:`ExternalSpec.platform_blockers` was left empty for exactly this
   reason — the aarch64 gap was an install problem wearing an architecture costume,
   and hard-coding it would have made ShEPhERD permanently UNAVAILABLE here too.

3. **``e3nn`` is not installed, and it is the first wall.** Importing *any*
   ``shepherd`` submodule runs ``src/shepherd/__init__.py``, and the chain
   ``__init__`` → ``model_loader`` → ``lightning_module`` → ``model/model.py`` →
   ``model/equiformer_v2_encoder.py`` → ``.../equiformer_v2/transformer_block.py``
   → ``.../equiformer_v2/drop.py`` ends at line 9, ``from e3nn import o3``. Run
   here, that chain raises ``ModuleNotFoundError: No module named 'e3nn'``, which
   is what the driver reports when launched in this environment. conda-forge
   carries e3nn 0.6.0 as a noarch build.

4. **``huggingface_hub`` is not installed, and a purely local checkpoint does not
   avoid it.** ``src/shepherd/model_loader.py`` line 9 imports
   ``checkpoint_manager``, whose own line 8 is
   ``from huggingface_hub import hf_hub_download, HfApi`` — at module scope, above
   any branch. ``load_model(local_checkpoint_path=...)`` never contacts the Hub,
   but the module that offers that argument cannot be imported without the
   library. The dependency is on the import graph, not on whether a download
   happens.

Everything else the run path needs is already here: torch 2.14 with working CUDA,
``torch_geometric``, ``torch_scatter``, ``pytorch_lightning``, ``rdkit`` and
``scipy``. ``xtb`` is absent but is not required — see below.

**What the platform does instead: nothing.** Under :mod:`drug_agent.tools.base`
this capability reports :attr:`CapabilityStatus.UNAVAILABLE` and every call raises
:class:`~drug_agent.tools.base.ToolUnavailable`. No unconditioned sample, no
substitute generator, no cached molecules. A run configured for ShEPhERD on a
machine in this state produces no ShEPhERD candidates and says so.

**The reconstruction hazard, which applies here too.** ShEPhERD's sampler does
emit bond types — ``_extract_generated_samples`` in
``src/shepherd/inference/steps.py`` returns ``x1['bonds']`` alongside
``x1['atoms']`` and ``x1['positions']`` — but the reconstruction routine upstream
provides, ``create_rdkit_molecule`` in ``src/shepherd/extract.py``, discards them.
It writes an XYZ block from the atomic numbers and coordinates and calls
``rdDetermineBonds.DetermineBonds``, retrying at formal charges 0, +1, −1, +2 and
−2 and returning ``None`` if all five fail. So the graph is still perceived from a
generated point cloud, with the failure modes that implies. The measured PFM
baseline in this repository is the standing warning: of 40 generated molecules 21
contained an allene and 21 a four-membered ring, leaving 2 chemically ordinary
molecules, and all 40 fell outside the activity model's applicability domain.
Read ``generation_validity`` first, then ``shepherd_allene_rate`` and
``shepherd_small_ring_rate``, before believing anything a batch from this tool
says about the model. Those two rates are computed here on every run, over the
molecules that were kept, and they are metered rather than used as a filter: a
strained ring is a real fact about a proposal, not a malfunction, and dropping the
molecules would hide the rate that diagnoses the reconstruction.

**What a working install would compute**, per :meth:`generate`:

    ``molecules[].smiles``       canonical SMILES of each reconstructed sample
    ``molecules[].sdf_block``    the 3D conformer ShEPhERD generated, as a molblock
    ``validity``                 fraction of raw samples RDKit could parse
    ``uniqueness``               fraction of those that were distinct
    ``shepherd_allene_rate``     fraction of kept molecules matching ``C=C=C``
    ``shepherd_small_ring_rate`` fraction with a three- or four-membered ring
    ``runtime_seconds``          wall clock for the whole batch

**What it deliberately does not do.** It does not dock, score or rank; it does not
report 3D similarity to the reference — that is ``shepherd-score``'s
``ConditionalEvalPipeline``, deliberately kept in the other repository so the
generator and its judge are not the same code. It does not relax the generated
conformer, so an SDF block here is a diffusion output, not a minimum. It does not
use xTB partial charges: the published protocol takes reference charges from xTB,
xTB is not installed here, and the driver instead calls upstream's own
``get_electrostatics``, whose docstring reads "Gets partial charges from MMFF or
Gasteiger". That is a real electrostatic potential from real charges, computed by
upstream code, and it is not the paper's; every batch carries a warning saying so.
Every molecule leaves as :attr:`TrustLabel.PROPOSED`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with ShEPhERD, not ShEPhERD's own version.
SHEPHERD_ADAPTER_VERSION = "1.0.0"

#: Checkpoint filenames, copied verbatim from ``MODEL_CHECKPOINTS`` in
#: ``src/shepherd/checkpoint_manager.py`` at ``1707fddc3b87``. ``mosesaq`` is the
#: full P(x1,x3,x4) model — shape, electrostatics and pharmacophores — and is the
#: only one this adapter samples from; the three GDB-17 models condition on
#: strictly less and were trained on molecules of at most 17 heavy atoms.
CHECKPOINT_FILENAMES: dict[str, str] = {
    "mosesaq": "x1x3x4_diffusion_mosesaq_20240824_submission.ckpt",
    "gdb_x2": "x1x2_diffusion_gdb17_20240824_submission.ckpt",
    "gdb_x3": "x1x3_diffusion_gdb17_20240824_submission.ckpt",
    "gdb_x4": "x1x4_diffusion_gdb17_20240824_submission.ckpt",
}

#: Denoising steps. ``shepherd.inference.sampler.generate`` defaults to 400 and
#: the notebooks pass 400 explicitly; fewer steps is a supported speed/quality
#: trade-off added in the November 2025 changelog entry.
DEFAULT_NUM_STEPS = 400

#: A cumulated diene. Bond perception over a generated point cloud produces these
#: at a rate that reflects the reconstruction, not the chemistry the model learned.
_ALLENE_SMARTS = "[#6]=[#6]=[#6]"

SPEC = ExternalSpec(
    name="ShEPhERD",
    env_prefix="SHEPHERD",
    checkout_relpath=".external/shepherd",
    # Every one of these is imported on the sampling path: the loader, the
    # Lightning module the checkpoint deserialises into, the sampler that owns
    # `generate`, the point-cloud reconstruction, and the two vendored
    # shepherd-score modules that build the conditioning targets.
    sentinel_files=(
        "src/shepherd/model_loader.py",
        "src/shepherd/lightning_module.py",
        "src/shepherd/inference/sampler.py",
        "src/shepherd/extract.py",
        "src/shepherd/shepherd_score_utils/generate_point_cloud.py",
        "src/shepherd/shepherd_score_utils/pharm_utils/pharmacophore.py",
    ),
    driver_name="shepherd_sample.py",
    # Narrow enough to identify a torch environment, so a half-built one is still
    # selected and then rejected module by module by `runtime_modules`.
    selection_modules=("torch", "torch_geometric"),
    # Discovered by reading upstream imports, not guessed: `e3nn` from
    # model/model.py and model/equiformer_operations.py; `open3d` from
    # shepherd_score_utils/generate_point_cloud.py; `huggingface_hub` from
    # checkpoint_manager.py, which src/shepherd/__init__.py imports
    # unconditionally; `torch_scatter` from model/model.py.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "pytorch_lightning",
        "e3nn",
        "open3d",
        "huggingface_hub",
        "rdkit",
        "scipy",
    ),
    weight_relpaths=(f"data/shepherd_chkpts/{CHECKPOINT_FILENAMES['mosesaq']}",),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/coleygroup/shepherd",
    upstream_ref="1707fddc3b87",
    license="MIT (Coley Research Group @ MIT, 2024); weights not distributed with the code",
    install_hint=(
        "conda install -n drug-agent -c conda-forge open3d e3nn huggingface_hub "
        "&& hf download kabeywar/shepherd "
        "x1x3x4_diffusion_mosesaq_20240824_submission.ckpt "
        "--local-dir .external/shepherd/data/shepherd_chkpts"
    ),
    needs_gpu=True,
)


class ShepherdConditioningError(ValueError):
    """ShEPhERD was asked to generate without a reference ligand.

    Deliberately not a :class:`~drug_agent.tools.generation.GenerationBackendError`:
    nothing ran and nothing failed. The caller asked a ligand-conditioned model for
    pocket-conditioned generation, or for no conditioning at all, and the only
    honest answer is to refuse. Falling through to an unconditional sample would
    put molecules into the pipeline labelled as though they had been designed
    against a target.
    """


class ShepherdAdapter(GenerativeToolAdapter):
    """Ligand-conditioned 3D generation against a reference molecule's shape, ESP and pharmacophores."""

    capability_id = "molecule.generate.shepherd"
    spec = SPEC
    engine = "shepherd"
    version = SHEPHERD_ADAPTER_VERSION
    conditioning = "ligand"
    synthesis_aware = False

    #: Which checkpoint to sample from. Only ``mosesaq`` carries all three
    #: conditioning channels this adapter inpaints.
    model_type = "mosesaq"
    #: Denoising steps handed to ``generate``.
    num_steps = DEFAULT_NUM_STEPS

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Ligand-conditioned 3D generation (ShEPhERD)",
            description=(
                "Diffusion generation of 3D molecules conditioned on a reference ligand's "
                "molecular surface, surface electrostatic potential and directional "
                "pharmacophores, by inpainting those three channels while the atoms are "
                "denoised. Proposes bioisosteres of a known molecule. It conditions on a "
                "ligand, never on a pocket, and refuses to run without one."
            ),
            input_schema_ref="GenerationSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no reference ligand supplied; a pocket cannot substitute for one and the "
                "adapter raises rather than sample unconditioned",
                "bond perception fails on the generated point cloud — create_rdkit_molecule "
                "tries formal charges 0, +1, -1, +2, -2 and returns nothing if all five fail",
                "reconstruction yields allenes and three- or four-membered rings at a rate "
                "set by the point cloud, not by the chemistry the model learned",
                "reference conformer cannot be embedded by ETKDG, so there is no surface to condition on",
                "reference yields zero pharmacophores, leaving the x4 channel with nothing to inpaint",
                "reference electrostatics come from MMFF94 charges here, not the xTB charges "
                "the published protocol uses",
                "N_x1 fixes the atom count for a whole batch, so one call explores one molecular size",
                "400 denoising steps on GPU; a large batch will exceed a short timeout",
            ],
            validation_checks=[
                "a reference SMILES is present and RDKit-parseable before any sampling starts",
                "the batch config records conditioning='ligand', never 'pocket'",
                "validity and uniqueness are computed from raw sample counts by the domain base",
                "allene and small-ring rates are measured on every batch and metered, not filtered",
                "every molecule is emitted as TrustLabel.PROPOSED",
            ],
        )

    # --- real backend ------------------------------------------------------
    def _checkpoint_path(self) -> Path:
        """The checkpoint the driver loads. Only reachable once readiness passed."""
        root = self.ext.root
        if root is None:  # pragma: no cover - readiness already proved otherwise
            raise GenerationBackendError("ShEPhERD checkout disappeared between probe and call")
        return Path(root) / "data" / "shepherd_chkpts" / CHECKPOINT_FILENAMES[self.model_type]

    @staticmethod
    def _reference_smiles(pocket: dict[str, Any], context_smiles: list[str]) -> str:
        """The reference ligand, or a refusal that names what was supplied instead.

        This is the one place the ligand/pocket distinction is enforced, and it is
        enforced before the driver is launched so the message is about the request
        rather than about a crash inside somebody else's environment.
        """
        for smiles in context_smiles:
            if smiles and smiles.strip():
                return smiles.strip()
        if pocket:
            where = pocket.get("pdb_path") or pocket.get("receptor_id") or pocket.get("pdb_id") or "a pocket"
            raise ShepherdConditioningError(
                f"molecule.generate.shepherd was given {where} and no reference ligand. "
                "ShEPhERD conditions on a molecule's shape, electrostatics and pharmacophores, "
                "not on a binding site; upstream has no receptor argument. Pass the reference "
                "ligand as context_smiles, or use a pocket-conditioned generator instead. "
                "Sampling unconditioned here would produce molecules that look targeted and "
                "are not."
            )
        raise ShepherdConditioningError(
            "molecule.generate.shepherd requires a reference ligand in context_smiles. "
            "ShEPhERD is a conditional model; running it unconditioned is a different "
            "experiment and is not what this capability claims to do."
        )

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
        """Run the driver inside ShEPhERD's interpreter and return its raw samples."""
        reference = self._reference_smiles(pocket, context_smiles)
        request = {
            "reference_smiles": reference,
            "n_molecules": int(n_molecules),
            "batch_size": max(1, int(batch_size)),
            "seed": int(seed),
            "device": self.ext.device,
            "checkpoint": str(self._checkpoint_path()),
            "model_type": self.model_type,
            "num_steps": int(self.num_steps),
        }
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            # The checkout is a src-layout package, so `import shepherd` needs
            # <root>/src on the path; <root> itself is already added by child_env.
            extra_pythonpath=("src",),
            seed=seed,
        )
        payload = run.payload
        records = [dict(item) for item in payload.get("molecules") or []]
        warnings = [str(w) for w in payload.get("warnings") or []]
        errors = [str(e) for e in payload.get("errors") or []]

        n_failures = int(payload.get("n_reconstruction_failures") or 0)
        if n_failures:
            warnings.append(
                f"{n_failures} of {payload.get('n_sampled', 0)} samples failed bond perception "
                "in create_rdkit_molecule and were dropped before validity was computed"
            )
        n_pharm = payload.get("n_pharmacophores")
        if n_pharm is not None:
            warnings.append(
                f"conditioned on {reference} — {payload.get('n_reference_atoms', '?')} atoms, "
                f"{payload.get('n_surface_points', '?')} surface points, {n_pharm} pharmacophores"
            )
        return records, warnings, errors

    # --- metering ----------------------------------------------------------
    def _meter(self, metered: dict[str, float], batch: GenerationBatch, *, n_molecules: int) -> None:
        """The base's counts, plus the two rates that diagnose the reconstruction.

        ``create_rdkit_molecule`` perceives bonds geometrically, so a batch can be
        perfectly "valid" by RDKit's standard and still be mostly allenes and
        strained rings. Validity alone does not show that; these do.
        """
        super()._meter(metered, batch, n_molecules=n_molecules)
        from rdkit import Chem

        kept = batch.molecules
        if not kept:
            return
        allene = Chem.MolFromSmarts(_ALLENE_SMARTS)
        n_allene = 0
        n_small_ring = 0
        n_parsed = 0
        for molecule in kept:
            rdmol = Chem.MolFromSmiles(molecule.smiles)
            if rdmol is None:
                continue
            n_parsed += 1
            if allene is not None and rdmol.HasSubstructMatch(allene):
                n_allene += 1
            if any(len(ring) <= 4 for ring in rdmol.GetRingInfo().AtomRings()):
                n_small_ring += 1
        if not n_parsed:
            return
        metered["shepherd_allene_rate"] = float(n_allene) / float(n_parsed)
        metered["shepherd_small_ring_rate"] = float(n_small_ring) / float(n_parsed)


__all__ = [
    "CHECKPOINT_FILENAMES",
    "DEFAULT_NUM_STEPS",
    "SHEPHERD_ADAPTER_VERSION",
    "SPEC",
    "ShepherdAdapter",
    "ShepherdConditioningError",
]
