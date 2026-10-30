"""TargetDiff — the target-conditioned diffusion baseline the rest of this shelf is measured against.

TargetDiff (*3D Equivariant Diffusion for Target-Aware Molecule Generation and
Affinity Prediction*, ICLR 2023) samples ligand atom **positions and types** inside
a protein pocket by running an SE(3)-equivariant denoising diffusion process, then
hands the resulting point cloud to a bond perceiver that rebuilds a molecular
graph. It is the reference implementation of that recipe and the direct ancestor of
:mod:`drug_agent.tools.pfm_tool`, which replaces the diffusion process with
perturbed flow matching and keeps everything else — the pocket featurizer, the
atom-count prior, the reconstruction step. That shared lineage is the reason this
adapter exists: TargetDiff is the **control**. A result from any pocket-conditioned
generator on this shelf that is not also shown against TargetDiff is a result
without a baseline.

**Three claims that must not be merged.** A single headline number for a
pocket-conditioned generator conflates three separate questions, and the catalogue
requires them reported apart:

1. *Generation quality* — what fraction of samples became a parseable, connected,
   chemically ordinary molecule. This is as much a property of the **reconstruction
   step** as of the model. ``generation_validity`` and ``generation_uniqueness``,
   plus the allene and small-ring rates below, answer it.
2. *Pose plausibility* — whether the emitted coordinates are a physically possible
   bound conformation. TargetDiff does not answer this and this adapter does not
   claim to; :mod:`drug_agent.tools.structure.posebusters_tool` is the judge, and
   the producer and the judge are deliberately different code.
3. *Score optimisation* — whether a docking or affinity score improves. That is a
   property of the scoring function at least as much as of the molecule, and a
   generator that is sampled against the same score that later grades it has
   proved nothing.

A molecule can be perfectly valid, sit in an impossible pose, and score well. All
three statements can hold at once, which is exactly why one number cannot carry
them.

**The reconstruction hazard, first.** Before believing anything about this model,
look at validity and at the ring and allene rates. Bond perception over a generated
point cloud (OpenBabel ``connect_the_dots`` plus ``PerceiveBondOrders``, in
``utils/reconstruct.py``) invents cumulated double bonds and four-membered rings at
a rate that has nothing to do with how well the model learned the pocket. The
measured baseline for the sibling model in this repository is the warning: of 40
PFM molecules, 21 contained an allene and 21 a four-membered ring, leaving 2
chemically ordinary molecules, and all 40 fell outside the activity model's
applicability domain. A high validity figure over reconstructed molecules is not
evidence against this; it is the wrong denominator. This adapter therefore reports
validity over the number **sampled**, counts every reconstruction failure as a
sample, and meters ``targetdiff_allene_rate`` and
``targetdiff_small_ring_rate`` on every call.

**Why this capability is UNAVAILABLE here.** Exactly one thing is missing, and it
is not the platform:

1. **The trained weights are absent.** ``configs/sampling.yml`` names
   ``./pretrained_models/pretrained_diffusion.pt``; the checkout's
   ``pretrained_models/.gitignore`` contains ``*`` and ``!.gitignore``, so the
   directory ships empty and the clone is code only. Upstream distributes the file
   from the Google Drive folder linked under "Trained model checkpoint" in
   ``.external/targetdiff/README.md``
   (``https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz``).
   Until that file exists, ``torch.load`` raises ``FileNotFoundError`` and there is
   nothing to sample from.

Nothing else blocks it, and that was verified rather than assumed. In the
``drug-agent`` environment on this machine (x86-64, Python 3.12.14, torch
2.14.0+cu130, CUDA available) the entire upstream run path imports —
``utils.misc``, ``utils.transforms``, ``datasets.pl_data``,
``models.molopt_score_model``, ``scripts.sample_diffusion``, ``utils.data`` and
``utils.reconstruct``; ``.external/pockets/seh_3i28_pocket10.pdb`` parses to 752
protein atoms with a protein feature width of 27 and a 13-class ``add_aromatic``
ligand map; ``utils.evaluation.atom_num.get_space_size`` returns 37.27 for that
pocket; and ``ScorePosNet3D``, built from the checkout's own
``configs/training.yml``, instantiates on ``cuda:0`` with 2 839 692 parameters.
Running ``scripts/external/targetdiff_sample.py`` against that pocket proceeds
through config load, seeding and the checkpoint check, and stops with a structured
``FileNotFoundError`` naming the missing ``.pt``. The driver is written against the
real entry point and is ready; only the weights are not here.

**What the platform does instead.** Nothing. Under :mod:`drug_agent.tools.base`
there is no surrogate: ``generate`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` and the capability card reports
UNAVAILABLE naming the exact ``.pt`` path. A fabricated molecule from a control
would be worse than no control, because the whole point of a baseline is that
another model's numbers are compared against it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: Version of *this adapter's* contract with TargetDiff, not of TargetDiff's weights.
TARGETDIFF_ADAPTER_VERSION = "1.0.0"

#: The checkpoint ``configs/sampling.yml`` points at, relative to the checkout root.
#: Named here as well as in :data:`SPEC` because it is the single blocker and the
#: unavailable reason has to be able to print it.
CHECKPOINT_RELPATH = "pretrained_models/pretrained_diffusion.pt"

#: The sampling config the driver loads. Supplies ``sample.num_steps`` (1000),
#: ``sample.seed`` (2021), ``sample.pos_only`` (False), ``sample.center_pos_mode``
#: ("protein") and ``sample.sample_num_atoms`` ("prior").
SAMPLING_CONFIG_RELPATH = "configs/sampling.yml"

#: A carbon with exactly two connections and two double bonds — the allene,
#: ketene and cumulene centre a bond perceiver invents when it has to guess bond
#: orders from interatomic distances alone. Verified against RDKit: it matches
#: ``C=C=C`` and ``O=C=O`` and does not match ``CC#CC`` or acetanilide.
ALLENE_SMARTS = "[CX2](=*)=*"

SPEC = ExternalSpec(
    name="TargetDiff",
    env_prefix="TARGETDIFF",
    checkout_relpath=".external/targetdiff",
    # Every one of these is imported by the run path, not merely present: the first
    # five are the modules scripts/sample_for_pocket.py imports at lines 6-15, and
    # the sixth is the config the driver loads. A stale directory that has lost any
    # of them is not a TargetDiff checkout and must not let the platform claim the
    # model.
    sentinel_files=(
        "scripts/sample_for_pocket.py",
        "scripts/sample_diffusion.py",
        "models/molopt_score_model.py",
        "utils/transforms.py",
        "utils/reconstruct.py",
        "datasets/pl_data.py",
        SAMPLING_CONFIG_RELPATH,
    ),
    driver_name="targetdiff_sample.py",
    selection_modules=("torch", "torch_geometric"),
    # Discovered by reading the upstream imports, and each one is a module a broken
    # install really loses. `torch_scatter` compiles from source with no wheel on
    # any platform and is imported by models/molopt_score_model.py line 5 and
    # datasets/pl_data.py line 2. `lmdb` is not optional despite this path using no
    # dataset: importing scripts/sample_diffusion.py runs `from datasets import
    # get_dataset` at line 16, and datasets/__init__.py imports pl_pair_dataset,
    # which imports lmdb at module scope. `easydict` is what utils/misc.py wraps
    # every config in, and `openbabel` is the bond perceiver underneath
    # utils/reconstruct.py — conda-only, and the piece most likely to be absent.
    runtime_modules=(
        "torch",
        "torch_geometric",
        # Imported at module scope by the driver, by the declared sentinel
        # utils/reconstruct.py, and by utils/data.py — a probe that omitted it
        # would pass on an interpreter that cannot rebuild a molecule at all.
        "rdkit",
        "torch_scatter",
        "lmdb",
        "easydict",
        "openbabel",
    ),
    # The single blocker on this machine. Checked with `exists`, so the reason
    # names the absolute path an operator has to fill.
    weight_relpaths=(CHECKPOINT_RELPATH,),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/guanjq/targetdiff",
    upstream_ref="142f1eb71784",
    # Upstream spells the file LICIENCE; the text is the MIT licence, 2023 Jiaqi Guan.
    license="TargetDiff MIT (LICIENCE in the checkout)",
    install_hint=(
        "make targetdiff-clone, then download pretrained_diffusion.pt into "
        ".external/targetdiff/pretrained_models/ from the Google Drive folder linked under "
        "'Trained model checkpoint' in .external/targetdiff/README.md "
        "(https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz)"
    ),
    needs_gpu=True,
)


class TargetDiffAdapter(GenerativeToolAdapter):
    """Pocket-conditioned 3D diffusion sampling with TargetDiff. The baseline, or nothing."""

    capability_id = "molecule.generate.targetdiff"
    spec = SPEC
    engine = "targetdiff"
    version = TARGETDIFF_ADAPTER_VERSION
    conditioning = "pocket"
    synthesis_aware = False

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        num_steps: int | None = None,
    ) -> None:
        # 3600 s rather than the shared 1800 s default: the shipped config runs
        # 1000 denoising steps for every batch, and a budget that expires mid-run
        # reports a timeout where the honest answer is "this model is slow".
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 3600.0)
        #: Override for ``sample.num_steps``. ``None`` keeps the checkout's own
        #: value, which is what the published numbers were produced with.
        self.num_steps = int(num_steps) if num_steps else None

    # --- capability --------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned de-novo generation (TargetDiff)",
            description=(
                "SE(3)-equivariant denoising diffusion over ligand atom positions and types "
                "inside a binding pocket, followed by bond perception to rebuild a molecular "
                "graph. The reference target-conditioned diffusion baseline and the lineage "
                "PFM belongs to. Proposes molecules; scores nothing and validates no pose."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            # A diffusion ODE started from Gaussian noise, with the ligand's atom
            # count drawn from a pocket-size prior. A seed pins one run; the model
            # is stochastic.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint absent — the clone is code only (pretrained_models/.gitignore is '*')",
                "bond perception over the point cloud invents allenes and four-membered rings; "
                "the measured PFM baseline in this repo is 21 of 40 for each",
                "reconstruction returns two disconnected fragments, which upstream discards "
                "silently (sample_for_pocket.py line 108) and this adapter flags instead",
                "a whole-protein PDB instead of a 10 A pocket mis-sizes the ligand, because "
                "atom_num.get_space_size takes the median of the ten largest pairwise distances",
                "ligand_atom_mode in the checkpoint disagreeing with the 13-class add_aromatic map",
                "CUDA out of memory at large batch_size; every batch runs 1000 denoising steps",
                "generated molecules landing outside a downstream QSAR model's applicability "
                "domain, where its predictions are not evidence",
            ],
            validation_checks=[
                "validity is computed over the number of samples, not the number that "
                "survived reconstruction — failures are returned as samples",
                "SMILES parses under RDKit; duplicates removed by canonical SMILES",
                "heavy-atom count within the configured window",
                "allene and <=4-membered-ring rates metered on every call, because a "
                "reconstruction artefact is not a chemistry result",
                "multi-component reconstructions flagged rather than dropped",
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
        """Run ``scripts/external/targetdiff_sample.py`` and parse what it wrote.

        ``context_smiles`` is accepted and ignored: TargetDiff conditions on the
        pocket alone, and silently treating a reference ligand as conditioning
        would misdescribe what the model was given.
        """
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() already refused
            raise GenerationBackendError("TargetDiff checkout is not resolvable")

        pocket_pdb = pocket.get("pdb_path")
        if not pocket_pdb or not Path(pocket_pdb).is_file():
            raise GenerationBackendError(
                f"TargetDiff requires a prepared pocket PDB; generation.pocket.pdb_path="
                f"{pocket_pdb!r} is missing or unreadable. Cut one with "
                "`python scripts/prepare_pocket.py <structure>.pdb --ligand <code> -o pocket.pdb`."
            )

        request: dict[str, Any] = {
            "pocket_pdb": str(Path(pocket_pdb).resolve()),
            "checkpoint": str(root / CHECKPOINT_RELPATH),
            "config": str(root / SAMPLING_CONFIG_RELPATH),
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device": self.ext.device,
            "num_steps": self.num_steps,
        }
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            seed=seed,
        )
        payload = run.payload

        warnings = [str(w) for w in payload.get("warnings", [])]
        n_failures = int(payload.get("n_reconstruction_failures") or 0)
        n_sampled = int(payload.get("n_sampled") or 0)
        if n_failures:
            warnings.append(
                f"{n_failures} of {n_sampled} generated point clouds could not be reconstructed "
                "into a molecule; they are counted as samples so validity keeps its denominator"
            )
        if payload.get("n_fragmented"):
            warnings.append(
                f"{int(payload['n_fragmented'])} of {n_sampled} reconstructions came back as two "
                "or more disconnected fragments and are flagged, not silently dropped"
            )
        molecules = [dict(item) for item in payload.get("molecules", [])]
        return molecules, warnings, [str(e) for e in payload.get("errors", [])]

    # --- metering ----------------------------------------------------------
    def _meter(self, metered: dict[str, float], batch: GenerationBatch, *, n_molecules: int) -> None:
        """Add the reconstruction diagnostics to the shared generation metrics.

        Validity and uniqueness alone cannot distinguish a model that learned the
        pocket from a bond perceiver guessing bond orders out of distances, because
        an allene-bearing molecule parses perfectly well and is counted as valid.
        These two rates are what makes that visible in the ledger, so a run whose
        chemistry collapsed is legible without re-reading the SDF.
        """
        super()._meter(metered, batch, n_molecules=n_molecules)
        metered.update(self._reconstruction_diagnostics(batch))

    def _reconstruction_diagnostics(self, batch: GenerationBatch) -> dict[str, float]:
        """Fraction of the kept molecules carrying a bond-perception artefact.

        Measured over the molecules this tool actually hands back, which is the
        population a downstream consumer would use, not the raw samples.
        """
        # Imported here, not at module scope: `tool_pool()` builds every adapter on
        # every call, and thirty module-level chemistry imports turn a listing into
        # a multi-second operation.
        from rdkit import Chem

        smiles = [m.smiles for m in batch.molecules if m.smiles]
        if not smiles:
            return {}
        allene = Chem.MolFromSmarts(ALLENE_SMARTS)
        n_allene = 0
        n_small_ring = 0
        n_parsed = 0
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_parsed += 1
            if allene is not None and mol.HasSubstructMatch(allene):
                n_allene += 1
            if any(len(ring) <= 4 for ring in mol.GetRingInfo().AtomRings()):
                n_small_ring += 1
        if not n_parsed:
            return {}
        return {
            "targetdiff_allene_rate": round(n_allene / n_parsed, 4),
            "targetdiff_small_ring_rate": round(n_small_ring / n_parsed, 4),
            "targetdiff_n_scored_for_artefacts": float(n_parsed),
        }


__all__ = [
    "ALLENE_SMARTS",
    "CHECKPOINT_RELPATH",
    "SAMPLING_CONFIG_RELPATH",
    "SPEC",
    "TARGETDIFF_ADAPTER_VERSION",
    "TargetDiffAdapter",
]
