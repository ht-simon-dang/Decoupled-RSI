"""RxnFlow — a GFlowNet that samples reaction pathways, so the route arrives with the molecule.

RxnFlow (*Generative Flows on Synthetic Pathway for Drug Design*,
https://github.com/SeonghwanSeo/RxnFlow) does not place atoms. It picks a
building block, then a reaction template, then another block, and the molecule is
whatever that pathway makes. The action space in the pinned checkout is 109
reaction templates (``data/templates/real.txt``, from the Enamine REAL protocol)
or 71 hand-built ones (``data/templates/hb_edited.txt``, 13 uni-molecular and 58
bi-molecular), crossed with a building-block catalogue. Every proposal therefore
carries the trajectory that produced it: ``("FirstBlock", block)``, then
``("UniRxn", smarts)`` or ``("BiRxn", smarts, block)`` at each step, read back
through ``SynthesisEnvContext.read_traj``.

**A GFlowNet is trained against a reward. That is the cost.** This is the
difference that matters when RxnFlow is put beside a diffusion prior, and
reporting it as a load-and-sample model would understate it by orders of
magnitude. A pocket-conditioned diffusion model learns a distribution over
ligands once and then samples any pocket for free. A GFlowNet learns to sample in
proportion to *one* reward function, so a new objective is a new training run,
not a new checkpoint load. Upstream's own defaults price it:
``scripts/train_pocket_conditional.py`` defaults to ``-n 50_000`` iterations at
``--batch_size 64``, and ``ProxyTrainer_MultiPocket.set_default_hps`` sets the
same ``num_training_steps = 50_000`` — 3.2 million sampled molecules, each scored
by the reward, before the first useful sample. ``scripts/opt_unidock.py`` defaults
to 1,000 iterations of 64 molecules, i.e. 64,000 UniDock docking runs, for a
single target.

The one exception, and the only path this adapter drives, is
``scripts/sampling_zeroshot.py``: the released pocket-conditional checkpoint was
trained across the 15,201 CrossDocked2020 training pockets named in
``data/README.md``, which amortises that cost over pockets and lets a *new* pocket
be sampled with no training iterations. It buys a new target, not a new
objective — the reward is fixed at what it was trained on, the TacoGFN QuickVina
proxy crossed with QED (``config.task.pocket_conditional.proxy =
("TacoGFN_Reward", "QVina", "ZINCDock15M")``). Wanting a different objective
means paying the 3.2 million evaluations.

**Why this capability is UNAVAILABLE on this machine.** Four independent blockers,
each verified by reading the checkout at ``.external/RxnFlow`` (a39c7aebd45f) and
probing the interpreters here:

1. **No GFlowNet checkpoint.** ``.external/RxnFlow/weights/`` contains one file,
   ``README.md``. The pocket-conditional checkpoint this adapter loads would have
   to exist at ``.external/RxnFlow/weights/qvina-unif-0-64_20250512.pt``;
   ``src/rxnflow/utils/download.py`` maps the name ``qvina-unif-0-64`` to Google
   Drive id ``13ATBmJIgIhU3Kclwut0yssjaciPCieEV`` and writes it under
   ``./weights/``, fetched with ``gdown``. (The non-pocket QED checkpoint is
   ``1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi`` → ``weights/qed-unif-0-64_20250512.pt``.)
   The clone is code-only; neither is present.

2. **No synthesis environment.** ``SynthesisEnv.__init__``
   (``src/rxnflow/envs/env.py``) opens exactly five files under ``env_dir``:
   ``template.txt``, ``building_block.smi``, ``bb_mask.npy``,
   ``bb_fp_2_1024.npy`` and ``bb_desc.npy``. ``.external/RxnFlow/data/envs/``
   contains only ``.gitkeep``. Building one is
   ``python data/scripts/b_create_env.py -b <blocks.smi[.gz]> -t
   data/templates/real.txt -o data/envs/<name>``, which precomputes a Morgan +
   MACCS fingerprint and eight RDKit descriptors per block and a template
   reactivity mask. The clone ships the templates and one 10,000-line debug block
   file (``data/building_blocks/zincfrag_10k.smi.gz``); the 200k ZINCFrag set is a
   separate Drive download (``16N8Xyxr9a-CifjIofgdH3ssFukC4Eh_V``) and the Enamine
   catalogue the paper used is available from Enamine on request only.

3. **RxnFlow is installed in no interpreter here.** ``importlib.util.find_spec``
   reports no ``rxnflow`` and no ``gflownet`` in any of ``drug-agent``,
   ``drug-agent-aizynth``, ``drug-agent-boltz``, ``drug-agent-unimol``,
   ``drug-agent-vina``, ``pfm`` or ``base``. Within ``drug-agent`` the run path's
   other imports are split: ``torch``, ``torch_geometric``, ``torch_scatter``,
   ``torch_cluster``, ``rdkit``, ``numpy`` and ``Bio`` resolve; ``omegaconf``,
   ``botorch``, ``wandb``, ``tensorboard`` and ``git`` do not. Those last four are
   not optional here — ``rxnflow/base/__init__.py`` imports ``.trainer``, which
   imports ``wandb`` and (through ``base/gflownet/online_trainer.py``) ``git`` and
   ``torch.utils.tensorboard.writer``, and ``rxnflow/base/task.py`` reaches
   ``gflownet/utils/metrics.py``, which imports ``botorch``. Importing
   ``RxnFlowSampler`` pulls in the whole trainer stack whether or not anything is
   trained.

4. **The pocket-conditional reward needs PharmacoNet.**
   ``rxnflow/tasks/multi_pocket.py`` opens with ``from pmnet_appl import
   BaseProxy, get_docking_proxy``, so the module this adapter's driver imports
   fails at import without it. It is the ``pmnet`` extra
   (``pharmaconet @ git+https://github.com/SeonghwanSeo/PharmacoNet.git``), and
   ``get_docking_proxy`` downloads its own TacoGFN proxy weights at first use.

Installing is a real cost rather than an impossibility, and the shape of it is
worth stating: RxnFlow pins ``torch==2.5.1`` and ``numpy==1.26.4``, against this
environment's torch 2.14 and numpy 2.5, and its install line points
``torch-scatter``/``torch-sparse``/``torch-cluster`` at
``data.pyg.org/whl/torch-2.5.1+cu121.html``, which publishes x86-64 wheels — so on
this x86-64 host they install rather than compile, which is one cost the port
removed. A separate conda environment is
therefore the expected arrangement, which is why sampling goes through
``scripts/external/rxnflow_sample.py`` rather than an in-process import.

Until all four are fixed this capability reports ``UNAVAILABLE`` and generates
nothing. There is no surrogate: a BRICS recombination or a random walk over the
shipped 10k debug blocks would produce molecules with routes attached, and a
route that no trained policy proposed is a fabricated claim about chemistry
wearing the evidence of a real one.

**What it would do once real.** ``generate()`` returns a
:class:`~drug_agent.tools.pfm_tool.GenerationBatch` of 2D molecules, validity and
uniqueness computed here by RDKit from the raw sample list, and each molecule's
reaction pathway kept in :attr:`RxnFlowAdapter.last_routes`. Upstream's README
reports 0.06 s/molecule on CPU and 0.04 s/molecule on GPU for zero-shot sampling;
no throughput has been measured on this machine, because nothing has run.

**What it deliberately does not do.** It emits no coordinates — a molecule here is
a SMILES and a pathway, never a pose, and nothing in this module writes an
``sdf_block``. It does not verify the route: a template that fires in RDKit is a
hypothesis about chemistry, and confirming it is the retrosynthesis planner's
job, not the generator's. It does not dock: the ``reward_vina`` a sample may carry
is the TacoGFN *proxy*'s estimate of a QuickVina score, and it is left in the raw
sample info rather than promoted to an observable, because a proxy for a docking
score for a binding affinity is three inferences away from a measurement.
"""

from __future__ import annotations

from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: Version of *this adapter's* contract with RxnFlow, not of RxnFlow's weights.
RXNFLOW_ADAPTER_VERSION = "1.0.0"

#: Where ``sampling_zeroshot.py`` looks for the synthesis environment by default
#: (``--env_dir``), relative to the checkout root.
DEFAULT_ENV_RELPATH = "data/envs/catalog"

#: The file ``rxnflow.utils.download`` writes for the name ``qvina-unif-0-64``,
#: which is ``sampling_zeroshot.py``'s default ``--model_path``.
DEFAULT_CHECKPOINT_RELPATH = "weights/qvina-unif-0-64_20250512.pt"

#: Temperature setting ``sampling_zeroshot.py`` defaults to. It must stay in the
#: same family as the checkpoint's own, or ``RxnFlowSampler.update_temperature``
#: refuses it.
DEFAULT_TEMPERATURE = "uniform-16-64"

#: ``algo.action_subsampling.sampling_ratio``. Upstream calls this a
#: memory-variance trade-off and defaults zero-shot sampling to 0.1.
DEFAULT_SUBSAMPLING_RATIO = 0.1

SPEC = ExternalSpec(
    name="RxnFlow",
    env_prefix="RXNFLOW",
    checkout_relpath=".external/RxnFlow",
    # Modules the sampling path really imports, not directory markers: the
    # sampler, the reaction environment, the pocket-conditional task and model,
    # and the vendored GFlowNet graph env underneath them all.
    sentinel_files=(
        "src/rxnflow/base/generator.py",
        "src/rxnflow/envs/env.py",
        "src/rxnflow/tasks/multi_pocket.py",
        "src/rxnflow/appl/pocket_conditional/model.py",
        "src/gflownet/envs/graph_building_env.py",
    ),
    driver_name="rxnflow_sample.py",
    selection_modules=("rxnflow",),
    # Every one of these is on the import path of `from rxnflow.tasks.multi_pocket
    # import ProxySampler`, checked against the upstream import statements rather
    # than the dependency list: `wandb`, `git` and `tensorboard` come in through
    # `rxnflow/base/__init__.py` -> `.trainer`, `botorch` through
    # `gflownet/utils/metrics.py`, `Bio` through the pocket featuriser, and
    # `pmnet_appl` at the top of `tasks/multi_pocket.py` itself. `torch_sparse` is
    # a declared dependency but only `gflownet/models/mxmnet.py` imports it, which
    # this path never reaches, so it is deliberately absent.
    runtime_modules=(
        "rxnflow",
        "gflownet",
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_cluster",
        "omegaconf",
        "botorch",
        "wandb",
        "tensorboard",
        "git",
        "pmnet_appl",
        "Bio",
        "rdkit",
        "numpy",
    ),
    # Neither is in the clone. The first is the trained policy; the second is the
    # action space it was trained over, and a GFlowNet checkpoint is meaningless
    # against a building-block library other than its own.
    weight_relpaths=(
        DEFAULT_CHECKPOINT_RELPATH,
        f"{DEFAULT_ENV_RELPATH}/building_block.smi",
    ),
    conda_envs=("drug-agent-rxnflow", "drug-agent"),
    upstream_url="https://github.com/SeonghwanSeo/RxnFlow",
    upstream_ref="a39c7aebd45f",
    license=(
        "RxnFlow MIT (LICENSE, (c) 2024 Seonghwan Seo); reaction templates and "
        "building-block catalogues carry their own terms"
    ),
    install_hint="make rxnflow-install",
    needs_gpu=True,
)


class RxnFlowAdapter(GenerativeToolAdapter):
    """Pocket-conditional synthesis-pathway generation with a pretrained GFlowNet."""

    capability_id = "molecule.generate.rxnflow"
    spec = SPEC
    engine = "rxnflow"
    version = RXNFLOW_ADAPTER_VERSION
    produces_nothing = "generates nothing"
    #: The released checkpoint is conditioned on a protein pocket, and the driver
    #: refuses to sample without one. Never read this as ligand-conditioned.
    conditioning = "pocket"
    #: Generation happens in reaction/building-block space, so every molecule
    #: arrives with the pathway that built it.
    synthesis_aware = True

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        env_dir: str | None = None,
        checkpoint: str | None = None,
        subsampling_ratio: float = DEFAULT_SUBSAMPLING_RATIO,
        temperature: str = DEFAULT_TEMPERATURE,
        calc_reward: bool = False,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s)
        root = self.ext.root
        # Path arithmetic only. `tool_pool()` builds every adapter on every call,
        # so nothing here may touch the filesystem; the probe in ExternalBackend
        # has already decided whether these files exist.
        self.env_dir = str(env_dir) if env_dir else (str(root / DEFAULT_ENV_RELPATH) if root else "")
        self.checkpoint = (
            str(checkpoint) if checkpoint else (str(root / DEFAULT_CHECKPOINT_RELPATH) if root else "")
        )
        self.subsampling_ratio = float(subsampling_ratio)
        self.temperature = str(temperature)
        self.calc_reward = bool(calc_reward)
        #: Reaction pathways from the most recent :meth:`generate`, one entry per
        #: raw sample: ``{"smiles", "sample_index", "n_steps", "steps"}``.
        #: ``GenerationBatch`` has no route field and inventing a parallel result
        #: model to carry one would fork the generation contract, so the pathway
        #: lives here, beside the batch, for a caller that wants it.
        self.last_routes: list[dict[str, Any]] = []

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Synthesis-pathway generation (RxnFlow GFlowNet)",
            description=(
                "Samples molecules by composing reaction templates over a building-block "
                "catalogue with a pretrained pocket-conditional GFlowNet, returning each "
                "molecule with the reaction trajectory that built it. 2D only: no "
                "coordinates, no pose. The policy is trained against one fixed reward "
                "(a QuickVina proxy crossed with QED); a different objective is a new "
                "training run, not a new checkpoint."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "the policy samples for the reward it was trained on; a new objective costs a "
                "training run — upstream defaults are 50,000 iterations x 64 molecules",
                "RxnFlowSampler.check_gfn_condition asserts the temperature family, focus type "
                "and preference type match the checkpoint's, so a mismatched config raises "
                "before a single molecule is sampled",
                "update_temperature refuses a checkpoint trained with sample_dist='constant', "
                "and refuses to cross between uniform/loguniform and any other family",
                "SynthesisEnv loads one 1190-byte fingerprint row per building block, so a "
                "1M-block catalogue is over a gigabyte of resident array before sampling starts",
                "action_subsampling.sampling_ratio below the 0.1 default shows the policy a "
                "smaller random slice of the catalogue each step, which upstream describes as a "
                "memory-variance trade-off",
                "a checkpoint paired with a building-block library other than the one it was "
                "trained over indexes into a different action space and its samples are noise",
                "the pocket featuriser keeps residues within 20 A of the centre, so a centre "
                "placed off the binding site yields a pocket graph of the wrong residues",
                "a generated route is a template that fires in RDKit, not a synthesis that works",
            ],
            validation_checks=[
                "every SMILES is re-parsed by RDKit here; validity and uniqueness come from "
                "those counts, never from the model",
                "each molecule keeps the reaction trajectory that built it; one that arrives "
                "without a pathway is flagged rather than silently accepted",
                "sampling refuses to start without a pocket structure, since the checkpoint is "
                "pocket-conditional and would otherwise be sampled outside its conditioning",
                "the checkpoint path, the synthesis environment and the temperature setting are "
                "recorded on the batch config",
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
        """Run the driver and hand the base class raw records.

        Screening, de-duplication and the metrics belong to
        :class:`~drug_agent.tools.generation.GenerativeToolAdapter`, so this only
        turns a pocket into a request and a payload into records.
        """
        warnings: list[str] = []
        protein = pocket.get("pdb_path")
        center = pocket.get("center")
        if not protein:
            raise GenerationBackendError(
                "RxnFlow's released checkpoint is pocket-conditional and its sampler needs a "
                "structure: PocketSpec.pdb_path is empty."
            )
        if not center:
            raise GenerationBackendError(
                "RxnFlow places the pocket from a centre: PocketSpec.center is empty and no "
                "reference ligand was supplied."
            )
        if context_smiles:
            warnings.append(
                f"RxnFlow's sampler takes no ligand context; {len(context_smiles)} context SMILES ignored"
            )

        request = {
            "env_dir": self.env_dir,
            "checkpoint": self.checkpoint,
            "protein_pdb": str(protein),
            "center": [float(c) for c in center],
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "subsampling_ratio": self.subsampling_ratio,
            "temperature": self.temperature,
            "seed": int(seed),
            "device": self.ext.device,
            "calc_reward": self.calc_reward,
        }
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            # `rxnflow` and `gflownet` live under `src/` in the checkout, so an
            # environment that has the dependencies but never ran `pip install -e .`
            # still resolves them.
            extra_pythonpath=("src",),
            seed=seed,
        )
        payload = run.payload
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        errors = [str(e) for e in payload.get("errors") or []]

        records: list[dict[str, Any]] = []
        self.last_routes = []
        for index, item in enumerate(payload.get("molecules") or []):
            if not isinstance(item, dict):
                errors.append(f"driver returned a {type(item).__name__} where a molecule was expected")
                continue
            steps = [s for s in (item.get("route") or []) if isinstance(s, dict)]
            smiles = str(item.get("smiles") or "")
            flags: list[str] = []
            if not steps:
                # A pathway is the whole point of this generator; a molecule with
                # none is still a molecule, but the claim "synthesis-aware" does
                # not hold for it and the flag says so.
                flags.append("no_reaction_route")
            records.append({"smiles": smiles, "qc_flags": flags, "route": steps})
            self.last_routes.append(
                {
                    "smiles": smiles,
                    "sample_index": int(item.get("sample_index", index)),
                    "n_steps": len(steps),
                    "steps": steps,
                }
            )
        if not records and not errors:
            errors.append("RxnFlow's driver returned no molecules and reported no error")
        return records, warnings, errors


__all__ = [
    "DEFAULT_CHECKPOINT_RELPATH",
    "DEFAULT_ENV_RELPATH",
    "DEFAULT_SUBSAMPLING_RATIO",
    "DEFAULT_TEMPERATURE",
    "RXNFLOW_ADAPTER_VERSION",
    "SPEC",
    "RxnFlowAdapter",
]
