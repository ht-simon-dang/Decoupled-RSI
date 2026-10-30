"""Pocket2Mol — autoregressive 3D generation inside a binding site, kept as a control.

Pocket2Mol (Peng et al., ICML 2022, arXiv:2205.07249) grows a ligand one atom at a
time inside a protein pocket: an equivariant graph network predicts a frontier
atom, a position, an element and the bonds to the existing fragment, and a beam
search keeps the highest-log-probability partial molecules until each terminates.
It is not a diffusion or flow model — the graph is built by the network as it
goes, not perceived from a finished point cloud — which is why it is worth having
beside the flow-matching generators rather than instead of them.

**It is here as a control, not as a recommendation.** Pocket2Mol appears in the
UTHealth structure-based-design benchmark this project compares against, so it is
the baseline whose numbers the newer models' numbers are supposed to beat.
Including it is a statement about comparability and nothing else. It is a 2022
model and no claim is made here that it is the current state of the art.

**Read validity and the ring statistics before anything else.** Every
pocket-conditioned generator in this package emits atoms in 3D and depends on a
bond step to turn them into a molecule, and that step is where the chemistry goes
wrong in ways that have nothing to do with how well the model learned the pocket.
The measured baseline in this repository is PFM: of 40 generated molecules, 21
contained an allene and 21 a four-membered ring, leaving 2 chemically ordinary
molecules, and all 40 fell outside the activity model's applicability domain.
That number is PFM's, not Pocket2Mol's — no Pocket2Mol sample has been generated
on this machine, because there is no checkpoint to generate one with. The
instruction it implies is general: before reading a single affinity prediction
over this tool's output, look at ``generation_validity``, at the fraction of
molecules containing an allene (``C=C=C``), and at the fraction containing a
four-membered ring. A model that collapses announces itself there first.

**Why this capability is UNAVAILABLE on this machine.**

1. The checkpoint is absent. ``configs/sample_for_pdb.yml`` sets
   ``model.checkpoint: ./ckpt/pretrained_Pocket2Mol.pt``; the checkout's
   ``ckpt/`` directory contains only ``README.md`` and a ``.gitignore`` whose
   body is ``*``, so a clone can never carry it. Upstream's
   ``ckpt/README.md`` directs the reader to a Google Drive folder,
   https://drive.google.com/drive/folders/1KfdOczjUPITPhIvCuBmnj4xFTV-iI2xB.
2. That same file is also the architecture. ``sample_for_pdb.py`` builds the
   network from ``ckpt['config'].model`` *before* loading ``ckpt['model']`` into
   it, so the missing file is not only missing weights — there is no shipped
   config from which the model could be constructed at all, and no partial run
   is possible.

Nothing else blocks it, and the rest was checked rather than assumed.

**What was verified here, and what it says about the environment.** The task of
deciding whether Pocket2Mol's older torch/PyG generation can coexist with this
environment's torch 2.14.0+cu130 and PyTorch Geometric 2.8.0.post1 was answered by
running it, not by reading the pin table. Upstream's ``README.md`` states the
tested environment as Python 3.8.12, PyTorch 1.10.1, CUDA 11.3.1, PyG 2.0.0,
RDKit 2022.03 and BioPython 1.79. On this machine, in the ``drug-agent``
environment, the checkout needs exactly two version shims and then works:

* ``utils/transforms.py`` imports ``from torch_geometric.utils.subgraph import
  subgraph``; that module is private in PyG 2.8 and the import raises
  ``ModuleNotFoundError``. The function is still exported from
  ``torch_geometric.utils``, so the driver aliases the old module name onto it.
* ``utils/transforms.py`` and ``models/common.py`` take ``knn``, ``knn_graph``
  and ``radius`` from ``torch_geometric.nn``. In PyG 2.8 all three raise
  ``ImportError: 'knn_graph' requires 'pyg-lib>=0.6.0'`` when called, and
  ``pyg_lib`` is not installed here. They were re-exports of ``torch_cluster`` in
  the PyG 2.0 that Pocket2Mol was published against; ``torch_cluster`` 1.6.3 is
  installed and its ``knn_graph`` runs, so the driver routes them back to it.

With those two shims applied, ``sample.py`` and ``sample_for_pdb.py`` both import
under torch 2.14.0+cu130, and the entire pocket-preparation path executes: on
``.external/pockets/seh_3i28_pocket10.pdb`` (752 ATOM/HETATM records), a 23 Å
cubic box centred on the file's centroid selects 485 protein atoms, the
featurizers report protein and ligand feature dimensions of 27 and 13, and
``AtomComposer`` builds a composed graph of 485 nodes with a
``compose_knn_edge_index`` of shape ``(2, 23280)``. So Pocket2Mol does **not**
need its own interpreter: it needs its checkpoint. The ``pfm`` environment, the
obvious alternative, is not one — it has no BioPython, and ``sample_for_pdb.py``
parses the pocket with ``Bio.PDB``.

**What it would do, given the checkpoint.** Sample ``n_molecules`` ligands inside
the box implied by ``pocket.pdb_path``, ``pocket.center`` and ``pocket.radius``,
and return them as a ``GenerationBatch`` whose ``validity`` and ``uniqueness``
are computed by :class:`~drug_agent.tools.generation.GenerativeToolAdapter` from
real counts. The driver returns every *terminal* beam sample, including the ones
whose graph could not be rebuilt — those arrive with an empty SMILES and a
``reconstruction_failed`` flag — because a driver that returned only its
successes would make ``validity`` 1.0 by construction, which is the kind of
number this package exists to refuse.

**What it deliberately does not do.** It does not score, dock, rank or filter for
drug-likeness. Its molecules leave here as ``TrustLabel.PROPOSED`` and become
evidence only after the same RDKit, activity, safety and synthesis pipeline that
judges every other candidate has judged them. ``batch_size`` is not used: a beam
search grows one population rather than independent batches, so the beam width
comes from the shipped ``configs/sample_for_pdb.yml`` (300) and is not silently
reinterpreted from a caller's batching preference. ``context_smiles`` is not used
either — Pocket2Mol conditions on the pocket alone, and a ligand-conditioned
request is answered with a warning rather than with a molecule that pretends the
context was honoured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: Version of *this adapter's* contract with Pocket2Mol, not of its weights.
POCKET2MOL_ADAPTER_VERSION = "1.0.0"

#: The checkpoint ``configs/sample_for_pdb.yml`` names in ``model.checkpoint``,
#: relative to the checkout root. It carries the architecture as well as the
#: weights, so it is probed as a hard requirement rather than as an optimisation.
POCKET2MOL_CHECKPOINT_RELPATH = "ckpt/pretrained_Pocket2Mol.pt"

#: Upstream's own sampling configuration for an arbitrary pocket, as shipped.
POCKET2MOL_CONFIG_RELPATH = "configs/sample_for_pdb.yml"

#: The cubic box side length ``sample_for_pdb.py``'s argparse defaults to, used
#: when a :class:`~drug_agent.contracts.brief.PocketSpec` carries no radius.
POCKET2MOL_DEFAULT_BBOX_SIZE = 23.0

SPEC = ExternalSpec(
    name="Pocket2Mol",
    env_prefix="POCKET2MOL",
    checkout_relpath=".external/Pocket2Mol",
    # Every one of these is a file the sampling path really imports or reads:
    # `sample_for_pdb.py` does `from sample import *`, `sample.py` pulls
    # `models/maskfill.py`, `utils/transforms.py` and `utils/reconstruct.py`, and
    # the config is what names the checkpoint and the beam settings.
    sentinel_files=(
        "sample.py",
        "sample_for_pdb.py",
        "models/maskfill.py",
        "utils/transforms.py",
        "utils/reconstruct.py",
        POCKET2MOL_CONFIG_RELPATH,
    ),
    driver_name="pocket2mol_sample.py",
    selection_modules=("torch", "torch_geometric"),
    # Discovered by reading the upstream import statements, not guessed. `lmdb`
    # looks wrong for PDB sampling and is not: `sample.py:18` does
    # `from utils.datasets import get_dataset`, which imports
    # `utils/datasets/pl.py`, which imports lmdb at module scope — so a build
    # without it loses `sample_for_pdb.py` too. `Bio` is `sample_for_pdb.py`'s own
    # PDB parser, and is exactly what the `pfm` environment is missing.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_cluster",
        "rdkit",
        "easydict",
        "yaml",
        "tqdm",
        "numpy",
        "Bio",
        "lmdb",
    ),
    weight_relpaths=(POCKET2MOL_CHECKPOINT_RELPATH,),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/pengxingang/Pocket2Mol",
    upstream_ref="836a0c4ce487",
    license="Pocket2Mol MIT (Copyright (c) 2022 Xingang Peng)",
    install_hint=(
        "make external-clone, then download pretrained_Pocket2Mol.pt from "
        "https://drive.google.com/drive/folders/1KfdOczjUPITPhIvCuBmnj4xFTV-iI2xB "
        "into .external/Pocket2Mol/ckpt/"
    ),
    needs_gpu=True,
)


class Pocket2MolAdapter(GenerativeToolAdapter):
    """Autoregressive pocket-conditioned 3D molecule generation (Pocket2Mol)."""

    capability_id = "molecule.generate.pocket2mol"
    spec = SPEC
    engine = "pocket2mol"
    version = POCKET2MOL_ADAPTER_VERSION
    conditioning = "pocket"
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned 3D generation (Pocket2Mol)",
            description=(
                "Autoregressive de-novo generation inside a binding site: an equivariant "
                "graph network places one atom at a time — frontier, position, element, "
                "bonds — under a beam search over partial molecules. Carried as the "
                "established 2022 baseline that the UTHealth structure-based-design "
                "benchmark reports, not as a claim about the current state of the art. "
                "Proposes molecules; scores nothing."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "checkpoint absent: ckpt/pretrained_Pocket2Mol.pt carries the architecture as "
                "well as the weights, so without it the network cannot be constructed at all",
                "bond perception over generated coordinates invents allenes and four-membered "
                "rings; read the ring and allene rates before any downstream score",
                "multi-fragment output: upstream rejects any SMILES containing '.', and an "
                "oversized box produces many of them",
                "beam collapse: every beam terminates in the first step and the run returns "
                "nothing, usually a box centre that misses the pocket",
                "PyTorch Geometric drift: the checkout is pinned to PyG 2.0 and needs the "
                "driver's subgraph and knn/knn_graph/radius shims on PyG >= 2.7",
                "a whole-protein PDB passed as the pocket, which makes the box meaningless",
            ],
            validation_checks=[
                "every returned SMILES is re-parsed and re-canonicalized by RDKit in the base "
                "adapter; a generator is never allowed to report its own validity",
                "validity is computed over all terminal beam samples, including the "
                "reconstruction failures the driver returns with an empty SMILES",
                "multi-fragment samples are flagged and excluded, never silently dropped",
                "the checkpoint path probed at construction is the same path handed to the driver",
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
        """Run the driver over one pocket and return its terminal beam samples."""
        warnings: list[str] = []
        if context_smiles:
            warnings.append(
                f"Pocket2Mol conditions on the pocket alone; {len(context_smiles)} "
                "context SMILES were ignored"
            )
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() already refused
            return [], warnings, ["Pocket2Mol checkout could not be located"]

        pocket_pdb = pocket.get("pdb_path")
        if not pocket_pdb or not Path(pocket_pdb).is_file():
            return (
                [],
                warnings,
                [f"Pocket2Mol needs a prepared pocket PDB; pocket.pdb_path={pocket_pdb!r} is not a file"],
            )

        radius = pocket.get("radius")
        # PocketSpec states a radius; sample_for_pdb.py takes the side length of a
        # cube. Twice the radius is the box that contains that sphere; with no
        # radius, upstream's own 23 Å default stands.
        bbox_size = 2.0 * float(radius) if radius else POCKET2MOL_DEFAULT_BBOX_SIZE
        request = {
            "checkout": str(root),
            "pocket_pdb": str(pocket_pdb),
            "center": pocket.get("center"),
            "bbox_size": bbox_size,
            "config": str(root / POCKET2MOL_CONFIG_RELPATH),
            # The same relpath the spec probes, so the readiness check and the run
            # can never disagree about which file is the checkpoint.
            "checkpoint": str(root / POCKET2MOL_CHECKPOINT_RELPATH),
            "n_molecules": int(n_molecules),
            # Left unset on purpose: a beam search has no batch, and the beam width
            # belongs to the model's published config, not to a caller's batching.
            "beam_size": None,
            "max_steps": None,
            "seed": int(seed),
            "device": self.ext.device,
        }
        # `run_json` directly rather than `run_driver`, because the caller's
        # max_runtime_seconds arrives here as `timeout_s` and `run_driver` would
        # substitute the adapter-wide budget for it.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            seed=seed,
        )
        payload = run.payload
        records = list(payload.get("molecules") or [])
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        # A shimmed run is still a real run, but the operator is told which
        # spellings had to be adapted rather than discovering it from a diff.
        warnings.extend(f"PyG compatibility shim: {n}" for n in payload.get("compat_notes") or [])
        n_failed = int(payload.get("n_reconstruction_failures") or 0)
        if n_failed:
            warnings.append(
                f"{n_failed} of {len(records)} terminal samples could not be rebuilt into a "
                "molecule; this is counted in validity, not hidden"
            )
        errors = [str(e) for e in payload.get("errors") or []]
        return records, warnings, errors


__all__ = [
    "POCKET2MOL_ADAPTER_VERSION",
    "POCKET2MOL_CHECKPOINT_RELPATH",
    "POCKET2MOL_CONFIG_RELPATH",
    "POCKET2MOL_DEFAULT_BBOX_SIZE",
    "SPEC",
    "Pocket2MolAdapter",
]
