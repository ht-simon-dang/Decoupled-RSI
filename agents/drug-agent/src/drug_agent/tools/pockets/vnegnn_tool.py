"""VN-EGNN — an E(3)-equivariant GNN whose weights this deployment cannot obtain.

VN-EGNN (Sestak, Schneckenreiter, Brandstetter, Hochreiter, Mayr & Klambauer,
arXiv:2404.07194) identifies binding sites by adding *virtual nodes* to an
E(n)-equivariant graph neural network over the protein's residue graph. The virtual
nodes are initialised on a sphere around the protein, carry their own message-passing
round, and are pulled by the network towards the sites; where a virtual node settles
is the prediction. It is a genuinely different third opinion from the two detectors
beside it in this package — fpocket asks where the geometry is concave, P2Rank asks
which surface points look ligandable, and this asks a learned equivariant field where
a ligand centre would be — which is exactly why it is worth naming even while it
cannot run.

**This capability is UNAVAILABLE here, and no part of it is approximated.** Under
:mod:`drug_agent.tools.base` there is no substitute: ``detect()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` and the platform gets no VN-EGNN
pockets at all. Reporting fpocket's answer under this name, or emitting a plausible
centroid, would be the precise failure ADR-0011 exists to have removed — a pocket is
an object, it becomes a ``PocketSpec.pdb_path``, a generator samples inside it, and
three artifacts downstream nothing can tell a predicted site from an invented one.

The checkout is real and complete. ``git clone`` of ``ml-jku/vnegnn`` at
``1d1f922cb9f0`` is 11 MB of MIT-licensed source with the model
(``src/models/vnegnn/vnegnn.py``), its Hydra config (``configs/model/vnegnn.yaml``)
and two example structures. What is missing is everything needed to turn that into a
prediction, and both blockers were read out of this checkout rather than assumed.

1. **There is no published checkpoint, and the code's only route to one is a private
   Weights & Biases run.** ``find`` over the checkout returns no ``.ckpt``, ``.pt``,
   ``.pth`` or ``.safetensors`` file. ``configs/eval.yaml`` line 28 sets
   ``ckpt_path: null`` and line 16 pins ``wandb_run_id: pt8wsnz9``, with
   ``wandb_project`` and ``wandb_entity`` read from the environment. ``src/eval.py``
   line 127 calls ``load_run_config_from_wb(entity, project, run_id)``, which is
   ``wandb.Api().run(f"{entity}/{project}/{run_id}")`` in
   ``src/utils/logging_utils.py``; line 137 then derives the checkpoint as
   ``load_ckpt_path(ckpt_dir=cfg_wandb.callbacks.model_checkpoint.dirpath)``, and
   ``load_ckpt_path`` (``src/utils/utils.py`` line 157) ``os.listdir``\\ s that
   directory. So the weights are addressed by a filesystem path recorded inside the
   authors' own W&B run: obtaining them needs network access, credentials for that
   entity, and a machine that is not this one. The README's claim that the repository
   "contains all code, instructions and model weights" is not true of the tree as
   published, and the Zenodo record its badge points at is the processed *dataset*.

2. **There is no single-structure inference path, so even a checkpoint would not be
   enough.** ``src/eval.py`` has no ``--pdb`` input; it runs
   ``trainer.predict(dataloaders=datamodule.test_dataloader()[i])`` over a
   ``BindingDataset`` built from a preprocessed directory, and that dataset loads
   ``np.load(path / "embeddings.npz")["residue_embeddings"]`` per protein
   (``src/datasets/binding_dataset.py`` line 87) — precomputed ESM-2 embeddings that
   ``scripts/generate_esm_embeddings.py`` produces in a separate pass. Running this
   on ``.external/pdb/3I28.pdb`` would mean generating those embeddings, building the
   PyG dataset, and writing the inference entry point upstream does not ship.

**Why there is no driver script in ``scripts/external/``.** Every other external
backend in this repository has one, and writing one here would mean inventing the
inference interface described above against a checkpoint whose ``state_dict`` nobody
can load — hundreds of lines that cannot be executed, cannot be tested, and would
encode guesses as though they were the upstream contract. The refusal is the honest
artifact; a driver is work for the day the checkpoint exists, and this docstring is
the specification for it.

**What would make this real, in order.** (a) A published checkpoint — upstream
releasing one, or the authors' W&B artifact being made public — placed at
``.external/vnegnn/checkpoints/vnegnn.ckpt``. (b) An environment: upstream wants
Python 3.9+, PyTorch 2.1+ (2.7+ recommended), PyTorch Geometric 2.4+ with
``torch-scatter``/``torch-sparse``/``torch-cluster`` built for the installed torch,
plus ``hydra-core``, ``omegaconf``, ``rootutils``, ``einops``, ``lightning`` and
``wandb``; on this host that means ``torch>=2.7.1+cu128``, because a torch pinned
below 2.7 carries SASS only to sm_90 and will not launch a kernel on this card. (c) A
single-structure inference path plus the ESM-2 embedding step. The readiness probe
below reports (a) and (b) precisely; (c) is stated here because no probe can detect a
missing entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import DEFAULT_N_PREPARED, DEFAULT_POCKET_RADIUS, PocketDetection, PocketToolAdapter

#: This adapter's contract with VN-EGNN, not VN-EGNN's own version.
VNEGNN_ADAPTER_VERSION = "1.0.0"

#: Where a published checkpoint would go, and the path the reason names. Chosen
#: rather than found: upstream's own route is a W&B run directory, so there is no
#: upstream-blessed location to adopt.
VNEGNN_CHECKPOINT_RELPATH = "checkpoints/vnegnn.ckpt"

#: The conda environment a working install would live in. Not created by any script
#: here, because an environment without a checkpoint would still detect nothing.
VNEGNN_ENV = "drug-agent-vnegnn"

#: The command that fetches everything obtainable. It is deliberately *not* a fix:
#: it clones the code, and the checkpoint the code needs does not exist to fetch.
#: The blockers below say so; this stays a bare command because the reason string
#: wraps it in "Install it with `...`" and a sentence there would read as one.
INSTALL_HINT = "python scripts/external/fetch_pockets.py --only vnegnn"

#: The facts that make this UNAVAILABLE regardless of what is installed, each read
#: out of the pinned checkout. Reported alongside the ordinary discovery failures.
_UPSTREAM_BLOCKERS: tuple[str, ...] = (
    "ml-jku/vnegnn publishes no checkpoint: there is no .ckpt/.pt/.pth in the tree at "
    "1d1f922cb9f0, configs/eval.yaml sets `ckpt_path: null` and pins `wandb_run_id: pt8wsnz9`, "
    "and src/eval.py:137 resolves the checkpoint from that private Weights & Biases run's "
    "`callbacks.model_checkpoint.dirpath` — a path on the authors' training machine, reachable "
    "only with network access and credentials for their W&B entity",
    "upstream ships no single-structure inference entry point: src/eval.py predicts over a "
    "BindingDataset whose per-protein records include precomputed ESM-2 embeddings "
    "(src/datasets/binding_dataset.py:87 reads embeddings.npz['residue_embeddings']), so a PDB "
    "path is not an input this code accepts even with weights present",
)


SPEC = ExternalSpec(
    name="VN-EGNN",
    env_prefix="VNEGNN",
    checkout_relpath=".external/vnegnn",
    # The model, its Hydra config, the evaluation entry point and the dataset that
    # defines the input features. A tree without these is not the code these
    # blockers were read out of.
    sentinel_files=(
        "src/models/vnegnn/vnegnn.py",
        "configs/model/vnegnn.yaml",
        "src/eval.py",
        "src/datasets/binding_dataset.py",
    ),
    # No driver: see the module docstring. Writing one would mean inventing an
    # inference interface upstream does not have, against weights that do not exist.
    driver_name=None,
    # `torch_geometric` identifies the environment a PyG install went into;
    # `runtime_modules` is what the run path would actually import, including the
    # compiled extensions that have no wheel for an arbitrary torch/CUDA pair and
    # the Hydra stack the config system needs.
    selection_modules=("torch_geometric",),
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_cluster",
        "lightning",
        "hydra",
        "omegaconf",
        "rootutils",
        "einops",
    ),
    weight_relpaths=(VNEGNN_CHECKPOINT_RELPATH,),
    conda_envs=(VNEGNN_ENV,),
    binary=None,
    upstream_url="https://github.com/ml-jku/vnegnn",
    upstream_ref="1d1f922cb9f015d4793e7804cbc9b8b248c68498",
    license="VN-EGNN MIT (Sestak et al., arXiv:2404.07194)",
    install_hint=INSTALL_HINT,
    needs_gpu=True,
    platform_blockers=_UPSTREAM_BLOCKERS,
)


class VnEgnnAdapter(PocketToolAdapter):
    """Binding-site identification by an equivariant GNN with virtual nodes."""

    capability_id = "pocket.detect.vnegnn"
    spec = SPEC
    engine = "vnegnn"
    version = VNEGNN_ADAPTER_VERSION
    score_kind = "vnegnn_confidence"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 900.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Binding-site identification with virtual nodes (VN-EGNN)",
            description=(
                "An E(3)-equivariant GNN whose virtual nodes migrate to the binding sites of a "
                "protein's residue graph. UNAVAILABLE on this deployment: upstream publishes no "
                "checkpoint and no single-structure inference path, so this capability produces "
                "no pockets rather than a plausible-looking centroid."
            ),
            input_schema_ref="ProteinStructure",
            output_schema_ref="PocketSpec",
            # Virtual nodes are initialised on a sphere around the protein and the
            # published protocol averages several initialisations, so the prediction
            # is not a function of the structure alone even at fixed weights.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no published checkpoint: the only path to weights is a private W&B run",
                "no single-structure inference entry point; src/eval.py evaluates a prepared dataset",
                "the residue graph needs precomputed ESM-2 embeddings from a separate pass",
                "torch below 2.7 has no sm_120 SASS and cannot launch a kernel on this GPU",
                "torch-scatter/torch-cluster have no wheel for an arbitrary torch/CUDA pair",
            ],
            validation_checks=[
                "produces no pockets while UNAVAILABLE; no other detector's output is relabelled",
                "the unavailable reason names the checkpoint path and the missing inference path",
            ],
        )

    # --- operations ---------------------------------------------------------
    def detect(
        self,
        structure_pdb: str | Path,
        *,
        out_dir: str | Path | None = None,
        n_prepared: int = DEFAULT_N_PREPARED,
        radius: float = DEFAULT_POCKET_RADIUS,
        n_virtual_nodes: int = 8,
        seed: int = 0,
    ) -> PocketDetection:
        """Would detect pockets in *structure_pdb*; refuses while there is no backend.

        The signature is the same as the working detectors' so that a caller — an
        agent choosing a detector, a benchmark comparing three of them — is written
        against one shape, and the refusal happens at the single enforcement point in
        :meth:`~drug_agent.tools.base.ToolAdapter.require_available` rather than at
        some later import.

        ``n_virtual_nodes`` is upstream's ``num_global_nodes``, which ``src/eval.py``
        passes as 8; it is named here because it is the parameter that decides how
        many sites the method can report at all.
        """
        return self.unsupported()


__all__ = [
    "INSTALL_HINT",
    "SPEC",
    "VNEGNN_ADAPTER_VERSION",
    "VNEGNN_CHECKPOINT_RELPATH",
    "VNEGNN_ENV",
    "VnEgnnAdapter",
]
