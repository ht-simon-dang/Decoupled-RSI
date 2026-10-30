"""DiffDock-L — blind docking by reverse diffusion, ranked by a confidence, not an energy.

DiffDock does not search a box. It treats docking as generative modelling over the
ligand's translation, rotation and torsion angles, denoises a randomly-placed
ligand against the whole receptor for twenty steps, and then asks a separate
confidence model to judge each of the sampled poses. The two models are distinct
checkpoints and only the second one produces a number.

**What that number is.** The confidence model is a classifier: ``inference.py``
reads ``confidence_args.rmsd_classification_cutoff`` and takes column zero when it
is a list, so the head is trained to say "is this pose within the cutoff RMSD of
the true one". Its output is used unnormalised, which makes the score unitless
log-odds, unbounded in both directions, and **higher is better**. DiffDock's own
README gives the working guideline: ``c > 0`` high confidence, ``-1.5 < c < 0``
moderate, ``c < -1.5`` low, for a drug-like ligand against a one- or two-chain
protein in a near-bound conformation. It is not a binding energy, and upstream's
FAQ says so directly. This adapter therefore sets ``score_unit`` to
``diffdock_confidence`` with the direction spelled out in the string, and
``lower_is_better = False``. A confidence read as kcal/mol would invert every
ranking it touched and would put a thermodynamic quantity nobody computed into the
object store.

**What it would compute here.** One :class:`~drug_agent.contracts.structure.Pose`
per sampled pose, with:

    ``score``          the DiffDock confidence, two decimals
    ``score_unit``     ``diffdock_confidence (unitless log-odds; higher is better)``
    ``rank``           0-based, re-sorted by confidence descending
    ``pose_ref``       the pose coordinates as a molblock
    ``engine``         ``diffdock``
    ``engine_version`` the pinned commit plus the weight release, e.g. ``85c49b60d3e0+v1.1``
    ``qc_flags``       ``diffdock_rank=<n>``, DiffDock's own 1-based rank

Two decimals is not a choice this adapter made. ``inference.py`` writes each pose
to ``rank<N>_confidence<C>.sdf`` with ``C`` formatted ``:.2f`` and puts no data
fields inside the SDF at all, so the file name is the only place the supported
entry point publishes the score. The driver reads it from there and the adapter
stamps it onto the molecule before parsing.

**What it deliberately does not do.** It takes no box, no pocket centre and no
reference ligand — blind docking is the whole claim of the method, and pretending
to accept a site would misdescribe the search. It predicts no affinity. It moves no
receptor side chain: ``InferenceDataset`` builds the receptor graph once, from the
PDB as given, and holds it rigid. And it does not judge its own geometry — a
DiffDock pose is a candidate for :mod:`~drug_agent.tools.structure.posebusters_tool`
to fail, not a pose that has already passed anything.

**Why this capability is UNAVAILABLE on this machine.** Four blockers, each
verified against the checkout at ``.external/DiffDock`` (commit ``85c49b60d3e0``)
and against the conda environments installed here:

1. **No weights.** ``.external/DiffDock/workdir/`` does not exist, and
   ``find .external/DiffDock -name '*.pt'`` returns nothing: the clone carries code
   only. ``default_inference_args.yaml`` names four files that must exist —
   ``workdir/v1.1/score_model/model_parameters.yml``,
   ``workdir/v1.1/score_model/best_ema_inference_epoch_model.pt``,
   ``workdir/v1.1/confidence_model/model_parameters.yml`` and
   ``workdir/v1.1/confidence_model/best_model_epoch75.pt``. ``inference.main``
   would download ``diffdock_models.zip`` from the DiffDock GitHub releases on
   first use; the driver refuses to pull the 124 MB archive inside a docking call.

2. **``e3nn`` is absent** from ``drug-agent``, ``drug-agent-boltz`` and ``pfm``.
   The score model is built from e3nn tensor products — ``models/aa_model.py`` line
   1 is ``from e3nn import o3``, and ``models/tensor_layers.py`` imports
   ``e3nn.o3.TensorProduct`` and ``e3nn.nn.BatchNorm`` — so ``utils.utils.get_model``
   cannot construct either checkpoint's architecture without it.

3. **``esm`` (fair-esm) is absent.** ``utils/inference_utils.py`` line 2 is
   ``from esm import FastaBatchedDataset, pretrained``, and ``inference.py``
   constructs ``InferenceDataset(..., lm_embeddings=True)``, so every receptor is
   embedded with ESM-2 650M (``model_location = "esm2_t33_650M_UR50D"``) before a
   graph exists. Installing the package is necessary but not sufficient: that
   checkpoint is fetched on first use and is not in this machine's torch hub cache
   — ``~/.cache/torch/hub/checkpoints/`` holds one file, ``resnet152-394f9c45.pth``.

4. **``prody`` is absent.** ``utils/inference_utils.get_sequences_from_pdbfile``
   parses the receptor with ``prody.parsePDB``, and ``datasets/parse_chi.py`` uses
   it again for the residue one-hot encoding. The receptor never reaches the model
   without it.

The geometric-deep-learning stack that usually blocks this class of model is *not*
the problem here: ``torch`` 2.14.0+cu130, ``torch_geometric``, ``torch_cluster``
and ``torch_scatter`` all import in ``drug-agent``, and CUDA works. Nor is the
upstream pin the route back: ``environment.yml`` asks for Python 3.9.18,
``torch==1.13.1+cu117`` and ``pt113cu117`` wheels for torch-scatter/cluster/sparse,
while this host is x86-64 with a compute-capability 12.0 GPU (``NVIDIA RTX PRO 6000
Blackwell Workstation Edition``), which CUDA 11.7 does not target. The architecture
changed and this blocker did not: the pin is too old for this GPU either way. The
realistic path is DiffDock's code on the newer torch already installed here, with
``e3nn``, ``fair-esm`` and ``prody`` added and
the weight release extracted — and then the ``torch.load`` calls in ``inference.py``
become the next thing to check, because they pass no ``weights_only`` and this
machine's torch refuses such a load with ``UnpicklingError: Weights only load
failed`` for any checkpoint carrying a non-tensor object.

**What the platform does instead: nothing.** ``structure.dock.diffdock`` reports
UNAVAILABLE, ``dock`` and ``dock_batch`` raise
:class:`~drug_agent.tools.base.ToolUnavailable`, and the run continues with this arm
visibly absent. There is no surrogate confidence, because a fabricated one would be
indistinguishable from a real one three artifacts downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.structure import Pose
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import StructureError, StructureToolAdapter

#: This adapter's contract with DiffDock, not DiffDock's own version. DiffDock
#: publishes no ``__version__``; the honest version of a checkout is its commit,
#: which travels on every pose as part of ``engine_version``.
DIFFDOCK_ADAPTER_VERSION = "1.0.0"

#: SDF property the adapter writes the confidence into before the domain base
#: parses it. It is this package's tag, not an upstream one: DiffDock's output SDFs
#: carry no data fields, and the confidence exists only in the file name.
CONFIDENCE_PROP = "diffdock_confidence"

#: Poses per ligand. Matches ``samples_per_complex`` in ``default_inference_args.yaml``.
DEFAULT_SAMPLES_PER_COMPLEX = 10

SPEC = ExternalSpec(
    name="DiffDock",
    env_prefix="DIFFDOCK",
    checkout_relpath=".external/DiffDock",
    # Every one of these is imported by the run path: inference.py is the entry
    # point, the YAML is the config it merges, and the other four are the modules
    # main() reaches through — InferenceDataset, sampling(), write_mol_with_coords
    # and the score model itself.
    sentinel_files=(
        "inference.py",
        "default_inference_args.yaml",
        "utils/inference_utils.py",
        "utils/sampling.py",
        "datasets/process_mols.py",
        "models/aa_model.py",
    ),
    driver_name="diffdock_predict.py",
    # e3nn identifies a DiffDock environment without being satisfied by any of the
    # general-purpose ones here; readiness then rejects a half-built env module by
    # module rather than crashing mid-run.
    selection_modules=("e3nn",),
    # What a broken install actually loses, read off the upstream imports:
    # models/*.py need e3nn, torch_cluster and torch_scatter; utils/inference_utils.py
    # needs esm and prody; datasets/process_mols.py needs Bio and torch_cluster;
    # utils/torsion.py needs networkx; inference.py itself needs pandas and yaml.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_cluster",
        "torch_scatter",
        "e3nn",
        "esm",
        "prody",
        "Bio",
        "networkx",
        "rdkit",
        "pandas",
        "yaml",
    ),
    # The exact four paths default_inference_args.yaml names. The clone ships none
    # of them; they come from the diffdock_models.zip release asset.
    weight_relpaths=(
        "workdir/v1.1/score_model/model_parameters.yml",
        "workdir/v1.1/score_model/best_ema_inference_epoch_model.pt",
        "workdir/v1.1/confidence_model/model_parameters.yml",
        "workdir/v1.1/confidence_model/best_model_epoch75.pt",
    ),
    conda_envs=("drug-agent-diffdock", "diffdock", "drug-agent"),
    upstream_url="https://github.com/gcorso/DiffDock",
    upstream_ref="85c49b60d3e0",
    license="MIT — DiffDock LICENSE, Copyright (c) 2022 Gabriele Corso, Hannes Stärk, Bowen Jing",
    install_hint="make diffdock-install",
    needs_gpu=True,
)


class DiffDockError(StructureError):
    """DiffDock was installed, it ran, and it did not produce a usable pose.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`, which means the
    checkout, the weights or the interpreter were never there.
    """


class DiffDockAdapter(StructureToolAdapter):
    """Diffusion-based blind docking over a whole receptor, ranked by confidence."""

    capability_id = "structure.dock.diffdock"
    spec = SPEC
    engine = "diffdock"
    version = DIFFDOCK_ADAPTER_VERSION
    produces_nothing = "docks nothing"

    #: Not an energy, and the string says so wherever a pose is read. See the
    #: module docstring: this is the confidence model's unnormalised classifier
    #: output for "within the RMSD cutoff", so it is unitless and unbounded.
    score_unit = "diffdock_confidence (unitless log-odds; higher is better)"
    #: Higher confidence is a better pose, which inverts the docking-energy default.
    lower_is_better = False
    # The score is the confidence model's classifier output, not an energy, so
    # a pose from here is a prediction. `poses_from_sdf` would otherwise stamp
    # CALCULATED, which is right for GNINA and wrong for a diffusion sampler.
    trust_label = TrustLabel.PREDICTED

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # An hour, not the half hour the base defaults to: the first call on a
        # machine also downloads the ESM-2 650M checkpoint and precomputes the
        # SO(2)/SO(3) series, which upstream's README calls "a couple of minutes"
        # on top of twenty denoising steps for every sampled pose.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 3600.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Blind docking by reverse diffusion (DiffDock-L)",
            description=(
                "Generative blind docking: denoises a randomly placed ligand over translation, "
                "rotation and torsions against the whole receptor, then scores each sampled pose "
                "with a separate confidence model. The score is a unitless confidence, higher is "
                "better, and is not a binding energy."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="Pose",
            # Reverse diffusion from a randomised starting position, with noise at
            # every step but the last. Two runs of the same ligand and receptor
            # give different coordinates and different confidences.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "ligand SMILES that RDKit cannot embed: InferenceDataset marks the complex "
                "unsuccessful and inference.py skips it, so that candidate comes back with no poses",
                "receptor PDB that ProDy cannot parse, or one with no CA atoms, fails the same way",
                "blind docking finds a surface pocket rather than the intended site; the confidence "
                "does not distinguish the two",
                "confidence is comparable across poses of one complex, much less so across "
                "different receptors or conformations (upstream README says so explicitly)",
                "large ligands, large complexes and apo conformations shift the confidence scale "
                "down, so the README's guideline thresholds no longer apply",
                "inference.py calls torch.load with no weights_only, which torch 2.6+ resolves to a "
                "weights-only unpickle and refuses for a checkpoint carrying non-tensor objects",
                "receptor is held rigid; no side-chain movement is modelled",
            ],
            validation_checks=[
                "score_unit names a confidence and never kcal/mol",
                "lower_is_better is False, so ranking sorts confidence descending",
                "every returned pose carries engine='diffdock', never 'mock'",
                "poses are re-ranked from 0 by confidence, not by DiffDock's file order",
            ],
        )

    # --- operations --------------------------------------------------------
    def dock(
        self,
        *,
        candidate_id: str,
        smiles: str,
        receptor_pdb: str | Path,
        receptor_id: str | None = None,
        n_poses: int = DEFAULT_SAMPLES_PER_COMPLEX,
        inference_steps: int | None = None,
        batch_size: int | None = None,
    ) -> list[Pose]:
        """Dock one ligand against one receptor. Returns poses ranked by confidence.

        An empty list means DiffDock ran and produced nothing for this ligand —
        usually an embedding failure — which is a fact about the molecule, not a
        malfunction of the engine, and so is not an exception.
        """
        self.require_available()
        with measure(self.capability_id, "dock", backend=self.backend) as metered:
            results = self._dock(
                ligands=[(candidate_id, smiles)],
                receptor_pdb=receptor_pdb,
                receptor_id=receptor_id,
                n_poses=n_poses,
                inference_steps=inference_steps,
                batch_size=batch_size,
            )
            poses = results.get(candidate_id, [])
            for key, value in _ledger_metrics(poses).items():
                metered[key] = value
        return poses

    def dock_batch(
        self,
        ligands: list[tuple[str, str]],
        *,
        receptor_pdb: str | Path,
        receptor_id: str | None = None,
        n_poses: int = DEFAULT_SAMPLES_PER_COMPLEX,
        inference_steps: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, list[Pose]]:
        """Dock several ligands, each ``(candidate_id, smiles)``, against one receptor.

        One driver call, because the expensive parts — loading the score model, the
        confidence model and ESM-2, and embedding the receptor — are paid once per
        receptor rather than once per ligand. A ligand DiffDock could not handle
        maps to an empty list rather than removing its key.
        """
        self.require_available()
        with measure(self.capability_id, "dock_batch", backend=self.backend) as metered:
            results = self._dock(
                ligands=ligands,
                receptor_pdb=receptor_pdb,
                receptor_id=receptor_id,
                n_poses=n_poses,
                inference_steps=inference_steps,
                batch_size=batch_size,
            )
            flat = [pose for poses in results.values() for pose in poses]
            for key, value in _ledger_metrics(flat).items():
                metered[key] = value
            metered["diffdock_n_ligands"] = float(len(ligands))
            metered["diffdock_n_ligands_docked"] = float(sum(1 for v in results.values() if v))
        return results

    # --- real backend ------------------------------------------------------
    def _dock(
        self,
        *,
        ligands: list[tuple[str, str]],
        receptor_pdb: str | Path,
        receptor_id: str | None,
        n_poses: int,
        inference_steps: int | None,
        batch_size: int | None,
    ) -> dict[str, list[Pose]]:
        if not ligands:
            return {}
        receptor = Path(receptor_pdb)
        if not receptor.is_file():
            raise DiffDockError(f"receptor PDB not found: {receptor}")

        request: dict[str, Any] = {
            "diffdock_root": str(self.ext.root),
            "receptor_pdb": str(receptor.resolve()),
            # DiffDock uses complex_name as a directory name, so it must be
            # filesystem-safe; candidate ids are carried alongside, not as the name.
            "ligands": [
                {"complex_name": f"complex_{index}", "candidate_id": cid, "ligand_description": smiles}
                for index, (cid, smiles) in enumerate(ligands)
            ],
            "samples_per_complex": int(n_poses),
        }
        if inference_steps is not None:
            # actual_steps is what the sampler really runs; leaving it at the
            # config's 19 while raising inference_steps would silently ignore
            # the request.
            request["inference_steps"] = int(inference_steps)
            request["actual_steps"] = int(inference_steps)
        if batch_size is not None:
            request["batch_size"] = int(batch_size)

        run = self.run_driver(request, error_cls=DiffDockError)
        payload = run.payload
        release = str(payload.get("model_release") or "")
        engine_version = f"{SPEC.upstream_ref}+{release}" if release else SPEC.upstream_ref

        results: dict[str, list[Pose]] = {cid: [] for cid, _ in ligands}
        with self.scratch("diffdock") as work:
            for entry in payload.get("complexes", []):
                candidate_id = str(entry.get("candidate_id") or "")
                name = str(entry.get("complex_name") or "complex")
                sdf_path = work / f"{name}.sdf"
                written = self._write_scored_sdf(list(entry.get("poses") or []), sdf_path)
                results[candidate_id] = (
                    self.poses_from_sdf(
                        sdf_path,
                        candidate_id=candidate_id,
                        receptor_id=receptor_id,
                        score_prop=CONFIDENCE_PROP,
                        engine_version=engine_version,
                        extra_props=("diffdock_rank",),
                    )
                    if written
                    else []
                )
        return results

    @staticmethod
    def _write_scored_sdf(poses: list[dict[str, Any]], path: Path) -> int:
        """Stamp each pose's confidence onto the molecule and write one SDF.

        DiffDock publishes the confidence in the *file name* and writes no SDF data
        fields, so the number has to be attached here before the domain base can
        read it as a score. A pose that will not sanitize is still written, with
        ring perception forced, because the coordinates are real and dropping the
        pose would hide that DiffDock produced one.
        """
        from rdkit import Chem

        writer = Chem.SDWriter(str(path))
        written = 0
        try:
            for item in poses:
                block = str(item.get("sdf") or "")
                mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
                if mol is None:
                    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
                    if mol is not None:
                        Chem.FastFindRings(mol)
                if mol is None:
                    continue
                confidence = item.get("confidence")
                if confidence is not None:
                    mol.SetProp(CONFIDENCE_PROP, f"{float(confidence):.2f}")
                rank = item.get("rank")
                if rank is not None:
                    mol.SetProp("diffdock_rank", str(int(rank)))
                writer.write(mol)
                written += 1
        finally:
            writer.close()
        return written


def _ledger_metrics(poses: list[Pose]) -> dict[str, float]:
    """The numbers worth carrying across runs, as floats.

    A collapse in the top confidence, or a sudden drop in poses returned, is how a
    wrong receptor, a broken ligand writer or a half-loaded checkpoint announces
    itself. ``max`` rather than ``min`` because higher confidence is better here.
    """
    out: dict[str, float] = {"diffdock_n_poses": float(len(poses))}
    scores = [pose.score for pose in poses if pose.score is not None]
    if scores:
        out["diffdock_top_confidence"] = float(max(scores))
        out["diffdock_mean_confidence"] = float(sum(scores) / len(scores))
    return out


__all__ = [
    "CONFIDENCE_PROP",
    "DEFAULT_SAMPLES_PER_COMPLEX",
    "DIFFDOCK_ADAPTER_VERSION",
    "SPEC",
    "DiffDockAdapter",
    "DiffDockError",
]
