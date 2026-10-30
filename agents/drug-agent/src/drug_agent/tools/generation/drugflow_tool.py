"""DrugFlow — pocket-conditioned 3D generation that also says how sure it is.

DrugFlow (Schneuing, Igashov et al., *Multi-domain distribution learning for de
novo drug design*, ICLR 2025; arXiv:2508.17815) samples a ligand inside a protein
binding site by integrating **continuous flow matching** over atom coordinates
jointly with **discrete Markov bridges** over atom and bond types. Its own
abstract states the part that matters for a platform that has to decide whether
to believe a generated molecule: "We endow DrugFlow with an uncertainty estimate
that is able to detect out-of-distribution samples."

That estimate is not a wrapper around the sampler. It is a head trained inside it,
and this adapter exists to carry its output rather than throw it away.

**The uncertainty, exactly.** When a checkpoint is trained with
``predict_confidence: True`` (``configs/training/drugflow_ood.yml:45``), the
dynamics network emits a per-atom ``uncertainty_vel = softplus(...)``
(``src/model/dynamics_hetero.py:802``) and the sampler accumulates two quantities
along the whole generative trajectory (``src/model/lightning.py:1003-1004``):

    ``sigma_x``     ``sqrt(∫ σ_vel² dt)`` — how uncertain the model was about
                    where it was pushing this atom, rooted at ``lightning.py:1085``
    ``entropy_h``   ``∫ H(logits_h) dt`` — the categorical entropy, in nats, of
                    the atom-type prediction

Both land as RDKit *atom* double properties on the generated molecule
(``lightning.py:1249-1256``), and DrugFlow's own pipeline reduces them per
molecule with a mean over atoms (``src/sample_and_evaluate.py:93-95``). This
adapter does the same reduction, keeps the maximum alongside it, and reports the
pair under a named ``uncertainty`` field on every record — *and* as SD data fields
inside the stored SDF block, because ``GeneratedMolecule`` has no numeric slot of
its own and a number that lives only in a log is a number that is gone. A
checkpoint without the confidence head produces no such properties, and its
molecules are flagged ``no_confidence_head`` rather than given a default: absence
of an uncertainty is not low uncertainty.

**Why this capability is UNAVAILABLE on this machine.** Two blockers, both
verified here against the checkout at ``ed6841677ccf``:

1. **No weights.** ``.external/DrugFlow`` holds 59 files, 9.2 MB, and zero
   ``.ckpt``/``.pt``/``.pth`` files — ``checkpoints/`` does not exist as a
   directory. The file this adapter needs is
   ``.external/DrugFlow/checkpoints/drugflow_ood.ckpt``, the *DrugFlow +
   confidence head* checkpoint, which the upstream README §Models publishes at
   ``https://zenodo.org/records/14919171/files/drugflow_ood.ckpt``. The plain
   ``drugflow.ckpt`` from the same record would sample molecules but sets
   ``predict_confidence: False`` (``configs/training/drugflow.yml:44``), so it
   cannot honour this capability's contract; the OOD checkpoint is therefore the
   one named in ``weight_relpaths``.
2. **Three missing modules in every interpreter here.** Importing
   ``src.model.lightning`` — the module holding the ``DrugFlow``
   ``LightningModule`` the driver loads — fails in the ``drug-agent`` environment
   at ``src/data/dataset.py:5``, ``ModuleNotFoundError: No module named
   'webdataset'``. Behind it are two more: ``prody``, imported unconditionally by
   ``src/data/normal_modes.py:3`` (reached from ``src/data/data_utils.py:20``),
   and ``fcd``, imported by ``src/sbdd_metrics/metrics.py:13`` (reached from
   ``src/model/lightning.py:36``). No ``drugflow`` or ``sbdd`` conda environment
   exists on this host; upstream's ``environment.yaml`` builds one pinned to
   Python 3.11.8, torch 2.2.1+cu121 and pyg 2.5.1.

Neither blocker is architectural, which is why moving to x86-64 did not clear
either of them. This is x86-64 with a working CUDA torch; nothing in DrugFlow is an
architecture-specific binary — ``gnina`` is, but the driver never invokes it, because
it samples molecules and computes no docking score. Both blockers are downloads.

**A correction to the catalogue record.** The research catalogue
(``drug_design_agent_research_catalogue.md``, DrugFlow row) records that "direct
repository content fetch did not succeed in this review" and asks that the
installation and weight-licence checks stay open. The fetch has since succeeded:
the checkout is present at commit ``ed6841677ccf`` (LPDI-EPFL/DrugFlow, merge of
PR #9, 1 Sep 2025), 37 Python files, and it contains the training configs, the
sampling entry point ``src/generate.py``, the full metric suite, and the KRAS
example pair ``examples/kras.pdb`` + ``examples/kras_ref_ligand.sdf``. The
licence half of that open question resolves cleanly for the **code**: the
repository ships an MIT ``LICENSE``, © 2025 Arne Schneuing, Ilia Igashov, Adrian
Dobbelstein. It does **not** resolve for the **weights**: the checkpoints live on
Zenodo record 14919171, nothing in the checkout states their licence, and this
host has no network access to check, so the weight-licence question stays open and
is recorded as such rather than assumed to inherit MIT.

**What the platform does instead.** Nothing. This capability reports UNAVAILABLE,
``generate()`` raises :class:`~drug_agent.tools.base.ToolUnavailable` naming the
checkpoint path above, and the run proceeds with this generator visibly absent.
There is no surrogate: a molecule attributed to DrugFlow that DrugFlow did not
sample would carry a fabricated ``sigma_x`` too, and an invented confidence is
worse than an invented structure — it invites trust in the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: Version of *this adapter's* contract with DrugFlow, not of DrugFlow's weights.
DRUGFLOW_ADAPTER_VERSION = "1.0.0"

#: The checkpoint this adapter is written against, relative to the checkout root.
#: The *confidence-head* variant deliberately: the plain ``drugflow.ckpt`` sets
#: ``predict_confidence: False`` and would silently drop the uncertainty this
#: capability promises.
DRUGFLOW_CHECKPOINT_RELPATH = "checkpoints/drugflow_ood.ckpt"

#: Upstream's own default cutoff for choosing pocket residues around the
#: reference ligand, ``src/generate.py:67``. ``PocketSpec.radius`` is deliberately
#: *not* mapped onto it: that field is a radius about a centre, this is a
#: per-residue distance to any reference-ligand atom, and silently equating the
#: two would change which residues the model sees.
DRUGFLOW_POCKET_CUTOFF_A = 8.0

#: Keys a caller may use to name the reference ligand SDF in the pocket dict.
#: ``PocketSpec`` has no field for one, and DrugFlow cannot define a pocket
#: without it, so a plain dict is the way a caller supplies it.
_REF_LIGAND_KEYS: tuple[str, ...] = (
    "ref_ligand_path",
    "ref_ligand_sdf",
    "ligand_sdf",
    "reference_ligand",
)

SPEC = ExternalSpec(
    name="DrugFlow",
    env_prefix="DRUGFLOW",
    checkout_relpath=".external/DrugFlow",
    # Every one of these is on the run path: the LightningModule the driver loads,
    # the pocket builder it calls, the collate function its DataLoader uses, the
    # graph builder that writes the uncertainty onto the atoms, and the size
    # histogram `setup(stage='generation')` np.load()s before anything samples.
    sentinel_files=(
        "src/model/lightning.py",
        "src/data/data_utils.py",
        "src/data/dataset.py",
        "src/data/molecule_builder.py",
        "src/default/size_distribution.npy",
    ),
    driver_name="drugflow_sample.py",
    # Narrow enough to identify the environment DrugFlow was installed into.
    # Readiness then rejects a half-built one module by module below.
    selection_modules=("torch", "pytorch_lightning"),
    # What a broken install actually loses, read off the upstream import
    # statements rather than guessed: `webdataset` (dataset.py:5), `prody`
    # (normal_modes.py:3), `fcd` (sbdd_metrics/metrics.py:13),
    # `useful_rdkit_utils` (metrics.py:21), `prolif` (interactions.py:2),
    # `posebusters` (metrics.py:14), `Bio` (generate.py:8), plus the two compiled
    # extensions that have no wheel and are what a partial install leaves out.
    runtime_modules=(
        "torch",
        "pytorch_lightning",
        "torch_scatter",
        "torch_geometric",
        "webdataset",
        "prody",
        "fcd",
        "useful_rdkit_utils",
        "prolif",
        "posebusters",
        "Bio",
    ),
    weight_relpaths=(DRUGFLOW_CHECKPOINT_RELPATH,),
    conda_envs=("drugflow", "sbdd", "drug-agent"),
    upstream_url="https://github.com/LPDI-EPFL/DrugFlow",
    upstream_ref="ed6841677ccf",
    license=(
        "MIT (LICENSE, (c) 2025 Arne Schneuing, Ilia Igashov, Adrian Dobbelstein) for the code, "
        "present at .external/DrugFlow. The checkpoints are distributed separately on Zenodo "
        "record 14919171 and the checkout states no licence for them; unverified here."
    ),
    install_hint=(
        "make external-clone && conda env create -f .external/DrugFlow/environment.yaml -n drugflow "
        "&& wget -P .external/DrugFlow/checkpoints "
        "https://zenodo.org/records/14919171/files/drugflow_ood.ckpt"
    ),
    needs_gpu=True,
)


class DrugFlowSamplingError(GenerationBackendError):
    """DrugFlow was installed, the driver started, and sampling did not finish."""


class DrugFlowAdapter(GenerativeToolAdapter):
    """Pocket-conditioned 3D generation with a per-molecule uncertainty estimate."""

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Also needs a reference ligand: upstream's loader asserts both paths."""
        return ("pocket_pdb", "ref_ligand_sdf")

    capability_id = "molecule.generate.drugflow"
    spec = SPEC
    engine = "drugflow"
    version = DRUGFLOW_ADAPTER_VERSION
    conditioning = "pocket"
    #: Atoms and bonds, not reactions. DrugFlow proposes a structure and no route.
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned 3D generation with uncertainty (DrugFlow)",
            description=(
                "Samples ligand coordinates, atom types and bond types inside a binding site by "
                "integrating continuous flow matching jointly with discrete Markov bridges. With "
                "the confidence-head checkpoint, every molecule also carries the model's own "
                "trajectory-accumulated positional uncertainty (sigma_x) and atom-type entropy "
                "(entropy_h), reduced per molecule the way upstream reduces them."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "the plain drugflow.ckpt sets predict_confidence: False, so no sigma_x or "
                "entropy_h is emitted and every molecule is flagged no_confidence_head",
                "build_molecule assembles bonds from argmax logits without sanitizing, so a "
                "sample can be chemically impossible and RDKit rejects it — counted as invalid, "
                "never repaired",
                "the reference ligand defines the pocket: a ligand whose atoms do not sit in the "
                "supplied PDB leaves no residue within the 8 A cutoff and the driver refuses "
                "rather than conditioning on an empty pocket",
                "molecule size is drawn from src/default/size_distribution.npy, the CrossDocked "
                "training histogram, so heavy-atom counts follow CrossDocked and the caller's "
                "min/max window filters afterwards instead of steering the model",
                "sampling is one ODE integration per batch on a GPU; a large n_molecules with a "
                "small batch_size multiplies checkpoint-resident wall time",
                "the uncertainty is the model's estimate of its own trajectory, not a calibrated "
                "probability that the molecule binds",
            ],
            validation_checks=[
                "every SMILES is re-parsed by RDKit here; validity and uniqueness come from those "
                "counts, never from anything DrugFlow reports about itself",
                "sigma_x and entropy_h are read from the per-atom double properties the sampler "
                "wrote, and averaged per molecule exactly as src/sample_and_evaluate.py does",
                "a molecule with no confidence properties is flagged no_confidence_head rather "
                "than assigned a default uncertainty",
                "the driver refuses a pocket with zero residues inside the cutoff",
            ],
        )

    # --- the one hook a generator supplies ---------------------------------
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
        """Run the DrugFlow driver once and return its raw records.

        Screening, de-duplication and the validity/uniqueness arithmetic all
        belong to :class:`~drug_agent.tools.generation.GenerativeToolAdapter`;
        what happens here is the call and the reading of its reply.

        Three DrugFlow-specific knobs ride in the *pocket* dict, because
        ``generate()``'s signature is shared by nineteen backends and none of them
        may widen it: the reference-ligand SDF (required — see
        :data:`_REF_LIGAND_KEYS`), ``n_steps`` to override the checkpoint's own
        integration length, and ``molecule_size`` in upstream's spelling
        (``"20"`` or ``"15,25"``). A :class:`PocketSpec` dump carries none of
        them, so a caller who wants them passes a plain dict.
        """
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() precedes this
            raise DrugFlowSamplingError("DrugFlow checkout is not resolved")

        protein_pdb = pocket.get("pdb_path")
        if not protein_pdb:
            raise DrugFlowSamplingError("DrugFlow conditions on a structure: the pocket carries no pdb_path")
        ref_ligand = next((pocket[k] for k in _REF_LIGAND_KEYS if pocket.get(k)), None)
        if not ref_ligand:
            raise DrugFlowSamplingError(
                "DrugFlow defines its pocket by the residues within "
                f"{DRUGFLOW_POCKET_CUTOFF_A:.0f} A of a reference ligand, and the pocket names "
                f"none — supply one of {', '.join(_REF_LIGAND_KEYS)} as a path to an SDF"
            )

        warnings: list[str] = []
        if context_smiles:
            warnings.append(
                f"{len(context_smiles)} context SMILES were supplied and ignored: DrugFlow is "
                "pocket-conditioned and takes no ligand context"
            )

        request = {
            "root": str(root),
            "protein_pdb": str(protein_pdb),
            "ref_ligand_sdf": str(ref_ligand),
            "checkpoint": str(root / DRUGFLOW_CHECKPOINT_RELPATH),
            "datadir": str(root / "src" / "default"),
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device": self.ext.device,
            "n_steps": pocket.get("n_steps"),
            "molecule_size": pocket.get("molecule_size"),
            "pocket_distance_cutoff": float(pocket.get("pocket_distance_cutoff") or DRUGFLOW_POCKET_CUTOFF_A),
            "upstream_ref": self.spec.upstream_ref,
        }
        # `ExternalBackend.run_json` rather than `run_driver`, because the caller's
        # `max_runtime_seconds` reached `_sample` as `timeout_s` and honouring it is
        # the point of the budget; `run_driver` would substitute the adapter default.
        run = self.ext.run_json(request, timeout_s=timeout_s, error_cls=DrugFlowSamplingError, seed=seed)
        return self._records(run.payload, warnings)

    # --- reading the driver's reply ----------------------------------------
    def _records(
        self, payload: dict[str, Any], warnings: list[str]
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Turn the driver payload into base-class records, keeping the uncertainty."""
        warnings = warnings + [str(w) for w in (payload.get("warnings") or [])]
        errors = [str(e) for e in (payload.get("errors") or [])]
        records: list[dict[str, Any]] = []
        for item in payload.get("molecules") or []:
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "smiles": item.get("smiles") or "",
                    "sdf": item.get("sdf"),
                    "qc_flags": [str(f) for f in (item.get("qc_flags") or [])],
                    # The named per-molecule field. The base class does not read it,
                    # which is why `_screen` below lifts it onto the batch as well.
                    "uncertainty": item.get("uncertainty") or None,
                    "n_atoms": item.get("n_atoms"),
                }
            )
        if not payload.get("predict_confidence", True):
            warnings.append(
                f"the loaded checkpoint {Path(str(payload.get('checkpoint') or '')).name} has no "
                "confidence head, so no molecule carries sigma_x or entropy_h"
            )
        return records, warnings, errors

    # --- keeping the uncertainty attached to the batch ----------------------
    def _screen(
        self,
        batch: GenerationBatch,
        raw: list[dict[str, Any]],
        *,
        pocket: dict[str, Any],
        min_heavy_atoms: int,
        max_heavy_atoms: int,
        keep_invalid: bool,
        n_molecules: int,
    ) -> None:
        """Screen exactly as the base does, then re-attach the uncertainty.

        The base owns validity and uniqueness and this override changes neither —
        it calls straight through. What it adds is the one thing the shared
        ``GeneratedMolecule`` cannot hold: the per-molecule ``sigma_x`` and
        ``entropy_h``, indexed by the ``sample_index`` the base already preserves,
        so the batch carries them in a machine-readable form rather than only as
        text inside each SDF block.
        """
        super()._screen(
            batch,
            raw,
            pocket=pocket,
            min_heavy_atoms=min_heavy_atoms,
            max_heavy_atoms=max_heavy_atoms,
            keep_invalid=keep_invalid,
            n_molecules=n_molecules,
        )
        kept = {mol.sample_index for mol in batch.molecules}
        table = {
            str(idx): {str(k): float(v) for k, v in (item.get("uncertainty") or {}).items()}
            for idx, item in enumerate(raw)
            if idx in kept and item.get("uncertainty")
        }
        batch.config["drugflow_uncertainty"] = table

    def _meter(self, metered: dict[str, float], batch: GenerationBatch, *, n_molecules: int) -> None:
        """Meter the base's counts, plus the batch's mean uncertainty.

        A checkpoint drifting out of distribution shows up here before it shows up
        in validity: the molecules stay parseable and ``sigma_x`` climbs.
        """
        super()._meter(metered, batch, n_molecules=n_molecules)
        table = batch.config.get("drugflow_uncertainty") or {}
        if not table:
            return
        metered["drugflow_n_with_uncertainty"] = float(len(table))
        for key in ("sigma_x_mean", "entropy_h_mean"):
            values = [row[key] for row in table.values() if key in row]
            if values:
                metered[f"drugflow_{key}"] = float(sum(values) / len(values))


__all__ = [
    "DRUGFLOW_ADAPTER_VERSION",
    "DRUGFLOW_CHECKPOINT_RELPATH",
    "DRUGFLOW_POCKET_CUTOFF_A",
    "SPEC",
    "DrugFlowAdapter",
    "DrugFlowSamplingError",
]
