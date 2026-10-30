"""SynFlowNet — a GFlowNet whose action space is a synthesis, so the route is the molecule.

SynFlowNet (Cretu, Harris, Igashov, Schneuing, Segler, Correia, Roy, Bengio and
Liò, *Design of Diverse and Novel Molecules with Synthesis Constraints*, ICLR
2025, arXiv:2405.01155, https://github.com/mirunacrt/synflownet) never places an
atom. Its MDP starts from an empty graph, the first action picks a purchasable
building block, and every later action fires an RDKit reaction template against a
second block — so the trajectory *is* the synthesis and the molecule is whatever
that synthesis produces. This is the generator to reach for when "can we actually
make it?" is a constraint on the search rather than a filter applied afterwards:
there is no post-hoc retrosynthesis step to fail, because a route was never
optional.

The action space in the pinned checkout is 105 Hartenfeller-derived templates
(``src/synflownet/data/templates/hb.txt``, 13 uni-molecular and 92 bi-molecular
— the split this adapter reports back on every batch) crossed with a
building-block catalogue. What comes back per sample is
``AddFirstReactant(block)`` and then ``ReactUni(template)`` or
``ReactBi(template, block)``, each step carrying the state before it, the state
after it, and whether it was applied.

**That last field is not bookkeeping, and it is the correction this integration
turned up.** ``SynthesisSampler.sample_from_model`` draws an action at every step
up to ``max_len``, but at the final one it marks the trajectory done and skips the
assignment that would store the new graph — so the last reaction is sampled,
recorded, and its product discarded. With upstream's own ``max_len = 3`` a
returned molecule is a first building block plus **one** applied reaction, with a
second reaction sitting in the trajectory that never touched it. Training is
self-consistent, because the reward is computed on the same truncated state; a
route reader is not, and counting trajectory actions would have this adapter
reporting two-step syntheses for one-step molecules. ``applied`` is therefore
decided by comparing the states either side of each action, never by counting
them, and ``n_reactions`` on every route counts only the applied ones.

**A GFlowNet is trained against a reward. That is the whole cost, and it was
measured here rather than estimated.** ADR-0013 recorded the same lesson for
RxnFlow and it holds harder for SynFlowNet, because SynFlowNet has no amortised
escape hatch: RxnFlow at least released a pocket-conditional checkpoint trained
over 15,201 CrossDocked pockets, so a *new pocket* is free. SynFlowNet's policy is
tied to one scalar reward and nothing else, so a new objective is a new training
run.

*Is there a released checkpoint?* No — checked, not assumed. ``mirunacrt/synflownet``
(12 commits, pinned at ``574f1e148f42``) has **zero GitHub releases**, no
``weights/`` or ``ckpt/`` path anywhere in its tree, and a README whose only
instruction is to train. Recursion's fork ``recursionpharma/synflownet-boltz``
likewise has zero releases and no checkpoint blob. A HuggingFace model search for
``synflownet`` and for ``synflow`` returns an empty list. The paper says "source
code is available at ..." and stops there. There is nothing to download.

*So what does producing one cost?* Measured over a full run on this machine, with
upstream's own ``main()`` configuration and the full catalogue below: **68.8 s to
build the trainer, then 3.41 s per training step at batch 64** on an RTX PRO 6000
(sm_120, torch 2.7.1+cu128). A twenty-step burst reads 2.86 s/step and is the
wrong number to quote — the rate settles higher once trajectories stop failing
early. Upstream's default and the paper's sEH setting are both 5,000 steps, so one
policy is **4.7 GPU-hours and 320,000 sampled trajectories**, each scored by the
reward. That is the figure to put beside a diffusion prior's "load and sample",
and it is per objective, not per target.

*And it worked.* The policy this deployment samples from was trained here on
2026-09-09: mean sampled reward rose 0.41 → 0.92 and the trajectory-balance loss
fell 673 → 2.1 over the 5,000 steps. Sampling 40 molecules through
:meth:`generate` then took 80 s wall — 11 s of it sampling, the rest rebuilding
the 223,974-block context — and gave **n_valid 40/40, n_unique 40/40, n_kept
40/40** at the default 10-70 heavy-atom window: 20 to 44 heavy atoms, median 31,
every one with an applied reaction. The contrast with the pocket-conditioned 3D
generators is the method's point rather than a detail — there is no bond
perception here to invent an allene, because there is no point cloud.

**The Enamine question, which gated ReaSyn, does not gate this one.** Upstream's
``ReactionTaskConfig`` names ``enamine_bbs.txt`` and its
``data/building_blocks/README.md`` sends you to Enamine's catalogue "upon
request". But Recursion's MIT-licensed fork publishes
``synflownet/data/building_blocks/enamine_bbs_hartenfeller_matching_sanitized.txt``
as an ordinary 5,874,496-byte git blob — 240,278 lines, 223,974 unique after the
de-duplication upstream's own pipeline prescribes. Its ``hartenfeller.txt`` is
byte-identical (sha256 ``5ddf9be7…``) to the pinned repo's ``hb.txt``, so the two
halves of the action space agree. For scale: the paper uses 10,000 blocks in most
experiments and 221,181 in its scaling study, so this is the large setting.

**One trap that only running it finds, recorded because it will recur.** That
published file is canonical under the RDKit that wrote it, not under the RDKit
here (2023.09.6). Ninety of the 223,974 blocks — organostannanes and similar,
written with explicit ``[CH2]`` / ``[C]`` atoms — round-trip through
``graph_to_obj(obj_to_graph(mol))`` to a *different* string. The environment's
``BckRemoveFirstReactant`` mask is a membership test of that round-tripped SMILES
against the raw file's strings, so for those blocks the only legal backward action
is masked, ``log_p_B`` is ``-inf``, and upstream raises ``ValueError: Invalid
log_p_B`` roughly forty steps into training. Upstream's own
``sanitize_building_blocks.py`` is the fix and it is a step that has to be run
*locally*, against the local RDKit, rather than inherited from whoever published
the file. ``scripts/external/synflownet_setup.py`` runs it and then asserts the
catalogue is round-trip closed — 0 misses over all 223,974 blocks — before
precomputing the template/block compatibility masks.

**Setup may use the network; sampling never does.** The setup script clones at the
pinned commit, fetches the catalogue, sanitises it, precomputes the masks and runs
the training, all of which needs the network exactly once. Sampling reads only
local files: the policy from ``checkpoints/<reward>/model_state.pt``, the action
space from the two files above, and the reward proxy from
``src/synflownet/models/cache/bengio2021flow_proxy.pkl.gz`` — 625,026 parameters
that upstream vendors as a git blob, so even the reward needs no download.

**The reward is sEH, which is this repository's own target, and that is a
coincidence worth naming rather than leaning on.** ``seh_reaction`` scores each
molecule with the pretrained MPNN from Bengio et al. (NeurIPS 2021) for soluble
epoxide hydrolase — EPHX2, UniProt P34913, the protein the demo brief is written
against. It is a 2021 ligand-based proxy, not a docking of these molecules into
3I28 and not an assay, so its value rides on the raw sample record as
``reward_seh_proxy`` and is never promoted to an observable or a prediction. A
proxy for an affinity is not a measurement of one.

**What this adapter deliberately does not claim.** It is not pocket-conditioned:
the released task takes no structure at all, ``conditioning`` is ``"none"``, and a
``PocketSpec`` handed to ``generate()`` is recorded on the batch config and
otherwise ignored — labelling it "pocket" would be the exact mislabelling the
package docstring forbids. It emits no coordinates: a molecule here is a SMILES
and a pathway, never a pose, and nothing in this module writes an ``sdf_block``.
And it does not verify the route: a template that fires in RDKit says the reacting
groups were present, not that the reaction works, that the intermediate survives,
or that the block is still in stock. That is a hypothesis about chemistry in
exactly the sense a generated pose is a hypothesis about binding.

**When it refuses.** Three artifacts must exist inside the checkout, and the
readiness probe names whichever is missing by full path: the building-block
catalogue, its precomputed masks, and the trained policy. The first two are a
download and three minutes of RDKit; the third is the five hours above. Until the
policy is there this capability reports UNAVAILABLE and generates nothing, and
there is no surrogate — a BRICS recombination or a random walk over the templates
would emit molecules with routes attached, and a route no trained policy proposed
is a fabricated claim about chemistry wearing the evidence of a real one.
"""

from __future__ import annotations

from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: Version of *this adapter's* contract with SynFlowNet, not of SynFlowNet's
#: weights — there are no published weights to version.
SYNFLOWNET_ADAPTER_VERSION = "1.0.0"

#: The reward the shipped policy is trained against. It is part of the artifact's
#: identity, not a runtime option: a checkpoint samples for the reward it was
#: trained on and for no other.
DEFAULT_REWARD = "seh_reaction"

#: Where ``scripts/external/synflownet_train.py`` writes the policy, relative to
#: the checkout root. One directory per reward, because two rewards are two
#: different samplers that happen to share an architecture.
CHECKPOINT_RELPATH = f"checkpoints/{DEFAULT_REWARD}/model_state.pt"

#: The card the training script writes *after* the run completes, naming the
#: reward, the step count, the catalogue and the machine. It is a required
#: artifact, not documentation: upstream checkpoints every 250 steps, so an
#: interrupted five-hour run leaves a loadable ``model_state.pt`` with no record
#: of how far it got. Requiring the card is what stops a half-trained policy
#: from reporting AVAILABLE and sampling as though it were the finished one.
TRAIN_CARD_RELPATH = f"checkpoints/{DEFAULT_REWARD}/train_card.json"

#: The action space, at the paths ``envs/synthesis_building_env.py`` reads at
#: *module import time* from ``ReactionTaskConfig``. Neither is in a code-only
#: clone; both are produced by the setup script.
BUILDING_BLOCKS_RELPATH = "src/synflownet/data/building_blocks/enamine_bbs.txt"
BB_MASKS_RELPATH = "src/synflownet/data/building_blocks/precomputed_bb_masks_enamine_bbs.pkl"

#: ``SynthesisSampler.sample_from_model``'s ``random_action_prob``. Zero samples
#: the policy itself; anything above it is exploration noise, which is a training
#: device and not something a caller asking for proposals wants.
DEFAULT_RANDOM_ACTION_PROB = 0.0

SPEC = ExternalSpec(
    name="SynFlowNet",
    env_prefix="SYNFLOWNET",
    checkout_relpath=".external/synflownet",
    # Modules the sampling path really imports: the trainer that rebuilds the
    # context, the reaction environment, the synthesis sampler, the policy network
    # and the vendored sEH proxy that supplies the reward.
    sentinel_files=(
        "src/synflownet/tasks/reactions_task.py",
        "src/synflownet/envs/synthesis_building_env.py",
        "src/synflownet/algo/reaction_sampling.py",
        "src/synflownet/models/graph_transformer.py",
        "src/synflownet/models/cache/bengio2021flow_proxy.pkl.gz",
        "src/synflownet/data/templates/hb.txt",
    ),
    driver_name="synflownet_sample.py",
    # Deliberately not ``("synflownet",)``. SynFlowNet has to be installed
    # *editable*, because ``envs/synthesis_building_env.py`` loads the catalogue,
    # the masks and the templates at module-import time from paths relative to its
    # own ``__file__`` — a non-editable copy in ``site-packages`` would look for
    # them there, where they are not. An editable install leaves nothing under
    # ``site-packages/synflownet/``, only a ``.pth`` and a dist-info, so the
    # interpreter probe (a site-packages glob, never an import) cannot see it and
    # would report the tool permanently missing on a machine where it works.
    # ``torch_sparse`` and ``tdc`` are installed normally and identify this
    # environment between them; the *package itself* is verified by
    # ``sentinel_files`` against the checkout, which is a stricter check anyway,
    # and reaches the driver through ``extra_pythonpath=("src",)``.
    selection_modules=("torch_sparse", "tdc"),
    # Read off the module-scope import statements of every file on the path from
    # `import synflownet.tasks.reactions_task` to a sampled trajectory, not off
    # the dependency list. `synflownet` itself is absent for the reason above.
    # The surprising ones are real: `tdc` and `meeko` are
    # imported at the top of `reactions_task.py` and `utils/gpu_vina.py` for the
    # PyTDC-oracle and Vina rewards, so a run that uses neither still cannot
    # start without them; `gemmi` is meeko's own import and find_spec on meeko
    # alone would not catch it; `git` and `tensorboard` come in through
    # `online_trainer.py` and `trainer.py`; `botorch` through `utils/metrics.py`.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "rdkit",
        "numpy",
        "scipy",
        "networkx",
        "omegaconf",
        "yaml",
        "wandb",
        "tensorboard",
        "git",
        "botorch",
        "sklearn",
        "tdc",
        "meeko",
        "gemmi",
    ),
    # In dependency order, which is also the order they get cheaper to explain:
    # the catalogue is a download, the masks are three minutes of RDKit over it,
    # the policy is a five-hour training run against one reward, and the card is
    # that run having actually finished.
    weight_relpaths=(
        BUILDING_BLOCKS_RELPATH,
        BB_MASKS_RELPATH,
        CHECKPOINT_RELPATH,
        TRAIN_CARD_RELPATH,
    ),
    conda_envs=("drug-agent-synflownet", "drug-agent"),
    upstream_url="https://github.com/mirunacrt/synflownet",
    upstream_ref="574f1e148f42",
    license=(
        "SynFlowNet MIT (LICENSE, Recursion Pharmaceuticals). The building-block "
        "catalogue is Enamine-derived and redistributed under MIT by "
        "recursionpharma/synflownet-boltz; the sEH reward proxy is Bengio et al. 2021, "
        "vendored in the clone. Enamine's own catalogue terms apply to the compounds"
    ),
    # Repeated verbatim once per missing artifact, so it names the one command
    # that produces all three rather than describing any of them.
    install_hint="python scripts/external/synflownet_setup.py --train",
    # Upstream trains and samples on CUDA by default and the policy is a graph
    # transformer over a 223,974-way action space; CPU works and is much slower.
    needs_gpu=True,
)


class SynFlowNetAdapter(GenerativeToolAdapter):
    """Molecules sampled as synthesis routes by a reward-trained GFlowNet policy."""

    capability_id = "molecule.generate.synflownet"
    spec = SPEC
    engine = "synflownet"
    version = SYNFLOWNET_ADAPTER_VERSION
    produces_nothing = "generates nothing"
    #: The released task conditions on a temperature and a reward, never on a
    #: structure. Reading this as pocket-conditioned would be a claim the model
    #: cannot make.
    conditioning = "none"
    #: Generation happens in reaction/building-block space, so every molecule
    #: arrives with the pathway that built it.
    synthesis_aware = True

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        checkpoint: str | None = None,
        random_action_prob: float = DEFAULT_RANDOM_ACTION_PROB,
        compute_reward: bool = True,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s)
        root = self.ext.root
        # Path arithmetic only. `tool_pool()` builds every adapter on every call,
        # so nothing here may touch the filesystem; the probe in ExternalBackend
        # has already decided whether these files exist.
        self.checkpoint = str(checkpoint) if checkpoint else (str(root / CHECKPOINT_RELPATH) if root else "")
        self.random_action_prob = float(random_action_prob)
        self.compute_reward = bool(compute_reward)
        #: Synthesis routes from the most recent :meth:`generate`, one entry per
        #: raw sample: ``{"smiles", "sample_index", "n_steps", "reward_seh_proxy",
        #: "steps"}``. ``GenerationBatch`` has no route field and inventing a
        #: parallel result model to carry one would fork the generation contract,
        #: so the route lives here, beside the batch, for a caller that wants it.
        self.last_routes: list[dict[str, Any]] = []
        #: Provenance of the policy that produced the last batch — the training
        #: card written beside the checkpoint. A GFlowNet sample is only
        #: interpretable next to the reward it was trained for.
        self.last_train_card: dict[str, Any] | None = None

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Synthesis-route generation (SynFlowNet GFlowNet)",
            description=(
                "Samples molecules by composing 105 reaction templates over a 223,974-block "
                "Enamine-derived catalogue with a GFlowNet policy, returning each molecule "
                "with the reaction trajectory that built it. 2D only: no coordinates, no "
                "pose, no pocket conditioning. The policy samples in proportion to one fixed "
                "reward it was trained against; a different objective is a new ~5-hour "
                "training run, not a new checkpoint load."
            ),
            # GenerationSpec, not PocketSpec: nothing here conditions on a
            # structure, and a pocket field on the input contract would invite
            # exactly the misreading the module docstring rules out.
            input_schema_ref="GenerationSpec",
            output_schema_ref="Candidate",
            # A GFlowNet is a sampler by construction: actions are drawn from a
            # hierarchical categorical at every step. A seed pins one run.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no policy exists to load — upstream publishes no checkpoint at all, so the "
                "first sample costs a full reward-training run (5,000 steps x 64 "
                "trajectories, measured at 3.41 s/step here)",
                "a training run interrupted part way leaves a loadable checkpoint from the "
                "last 250-step save; the tool stays UNAVAILABLE because the training card "
                "that certifies the run is written only when it finishes",
                "the policy samples for the reward it was trained on and for no other; the "
                "training card beside the checkpoint is the only record of which that was",
                "a checkpoint paired with a building-block file other than the one it was "
                "trained over indexes into a different action space, and its samples are a "
                "different distribution wearing the same name",
                "a catalogue whose SMILES are not round-trip canonical under the local RDKit "
                "masks the only legal backward action for those blocks, and training dies "
                "with 'Invalid log_p_B: -inf' tens of steps in",
                "upstream's own load_from_checkpoint rebuilds a trainer from the checkpoint's "
                "config, whose log_dir is the checkpoint's own directory and which setup() "
                "deletes; the driver redirects it rather than inheriting it",
                "max_len counts sampled actions, not applied ones: upstream discards the "
                "product of the last action, so upstream's max_len=3 yields a block plus one "
                "applied reaction and nothing longer can be proposed",
                "a trajectory whose reaction fails to reverse is flagged invalid by the "
                "sampler and still returned, because a silently dropped sample would inflate "
                "the validity this pool meters",
                "a generated route is a template that fires in RDKit, not a synthesis that "
                "works, and the sEH reward is a 2021 ligand-based proxy, not an assay",
            ],
            validation_checks=[
                "every SMILES is re-parsed by RDKit here; validity and uniqueness come from "
                "those counts, never from the model",
                "each molecule keeps the reaction trajectory that built it, with the "
                "building block, the template SMARTS and the state either side of every "
                "step; one that arrives without a pathway is flagged rather than accepted",
                "a step counts as part of the route only when the state before it differs "
                "from the state after it, so an action upstream sampled but never applied is "
                "reported as such instead of inflating the synthesis",
                "a trajectory the sampler itself marked invalid keeps that flag through "
                "screening instead of being dropped",
                "the checkpoint path, the reward it was trained on, the temperature it was "
                "trained at and the size of the action space are recorded on the batch config",
                "the reward proxy's score rides on the raw record as reward_seh_proxy and is "
                "never emitted as an observable or a prediction",
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
        turns a request into a payload and a payload into records.
        """
        warnings: list[str] = []
        if pocket:
            warnings.append(
                "a pocket was supplied but SynFlowNet's released task is unconditional; it is "
                "recorded on the batch config and did not influence sampling"
            )
        if context_smiles:
            warnings.append(
                f"SynFlowNet's sampler takes no ligand context; {len(context_smiles)} context SMILES ignored"
            )

        request = {
            "checkpoint": self.checkpoint,
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device": self.ext.device,
            "random_action_prob": self.random_action_prob,
            "compute_reward": self.compute_reward,
        }
        run = self.run_driver(
            request,
            error_cls=GenerationBackendError,
            timeout_s=timeout_s,
            # `synflownet` lives under `src/` in the checkout, so an environment
            # with the dependencies but no `pip install -e .` still resolves it.
            extra_pythonpath=("src",),
            seed=seed,
        )
        payload = run.payload
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        errors = [str(e) for e in payload.get("errors") or []]
        self.last_train_card = (
            payload.get("train_card") if isinstance(payload.get("train_card"), dict) else None
        )

        records: list[dict[str, Any]] = []
        self.last_routes = []
        for index, item in enumerate(payload.get("molecules") or []):
            if not isinstance(item, dict):
                errors.append(f"driver returned a {type(item).__name__} where a molecule was expected")
                continue
            records.append(self._record(item, index))
        if not records and not errors:
            errors.append("SynFlowNet's driver returned no molecules and reported no error")
        return records, warnings, errors

    def _record(self, item: dict[str, Any], index: int) -> dict[str, Any]:
        """One driver row as a raw sample record, with its route and its QC flags.

        The route rides beside the batch rather than on it because
        ``GeneratedMolecule`` has no field for one. Stating that here, rather than
        dropping the pathway silently, is what stops a caller reading a molecule
        from this tool as though a route travelled with it.
        """
        steps = [s for s in (item.get("route") or []) if isinstance(s, dict)]
        smiles = str(item.get("smiles") or "")
        flags: list[str] = []
        # Applied, not merely sampled — see the module docstring. Counting actions
        # here would report a two-step synthesis for a one-step molecule on every
        # sample upstream's max_len truncates.
        reactions = [s for s in steps if s.get("action") in {"ReactUni", "ReactBi"} and s.get("applied")]
        sampled = [s for s in steps if s.get("action") in {"ReactUni", "ReactBi"}]
        # A pathway is the whole point of this generator. A molecule without one
        # is still a molecule, but "synthesis-aware" does not hold for it.
        if not steps:
            flags.append("no_reaction_route")
        elif not reactions:
            # A block and a Stop, or a reaction whose product was discarded:
            # a catalogue lookup, not a synthesis.
            flags.append("no_reaction_steps")
        if not item.get("is_valid", True):
            flags.append("sampler_marked_invalid")
        self.last_routes.append(
            {
                "smiles": smiles,
                "sample_index": int(item.get("sample_index", index)),
                "n_steps": len(steps),
                "n_reactions": len(reactions),
                "n_reactions_sampled": len(sampled),
                "reward_seh_proxy": item.get("reward_seh_proxy"),
                "steps": steps,
            }
        )
        return {"smiles": smiles, "qc_flags": flags, "route": steps}


__all__ = [
    "BB_MASKS_RELPATH",
    "BUILDING_BLOCKS_RELPATH",
    "CHECKPOINT_RELPATH",
    "DEFAULT_RANDOM_ACTION_PROB",
    "DEFAULT_REWARD",
    "SPEC",
    "SYNFLOWNET_ADAPTER_VERSION",
    "TRAIN_CARD_RELPATH",
    "SynFlowNetAdapter",
]
