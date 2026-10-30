"""SynFormer — propose a molecule by proposing the reactions that would build it.

SynFormer (Gao, Luo and Coley, *Generative artificial intelligence for navigating
synthesizable chemical space*, arXiv:2410.03494) does not sample atoms. It samples
a **post-order action sequence over a synthesis stack**: push a building block,
apply a reaction template, repeat, and read the molecule off the top of the stack
when the stack has collapsed to one product. The molecule is therefore whatever
the route produces, and a route always exists for it by construction. That is the
whole point of the method and also the whole of its risk.

**A generated route is not proof of laboratory success.** What ``synthesis`` names
is a sequence of reaction *templates* — 115 SMARTS transforms in
``data/rxn_templates/comprehensive.txt`` at the pinned commit, against a decoder
configured for 120 reaction classes — matched against catalogue building blocks.
A template match says the reacting groups are present. It says nothing about
yield, about selectivity when three matching groups are present, about protecting
groups, about whether the intermediate survives the next step, or about whether
the supplier still stocks the block. The route is a hypothesis about chemistry in
exactly the sense a generated pose is a hypothesis about binding, and it is
labelled ``TrustLabel.PROPOSED`` like every other proposal in this package.

**Conditioning is on a ligand, never on a pocket.** The sampler this adapter
drives is the *analog* sampler in ``synformer/sampler/analog/``: ``StatePool``
takes one ``Molecule``, encodes it, and scores every partial route by fingerprint
similarity back to it. It projects a seed molecule into synthesizable space. It
cannot be pointed at a binding site, so ``conditioning`` is ``"ligand"`` and a
``PocketSpec`` handed to ``generate()`` is recorded on the batch config and
otherwise ignored. The decoder-only configuration in
``configs/dev_deconly_diffusion_tfmer.yml`` (``encoder_type: none``) would sample
unconditionally, but no unconditional inference script ships in the checkout —
``scripts/sample.py``, ``scripts/sample_naive.py`` and ``scripts/molopt.py`` all
require an input molecule.

**Why this capability is UNAVAILABLE on this machine.** Not because of the
architecture — it was UNAVAILABLE on aarch64 and is UNAVAILABLE on x86-64 for the
same three reasons. The checkout is real and complete as *code* — 656 KB excluding
``.git``, every module the sampler imports present — and ``import synformer.chem.mol``
succeeds here under Python 3.12.14 / torch 2.14.0+cu130 / rdkit 2026.03.6, pulling in
the featurizer, the tokenizer and the RDKit layer without complaint. SynFormer is
pure Python over PyTorch and RDKit and has no architecture-specific component.
The three blockers are data and packaging.

1. **The checkpoint is absent.** ``.external/synformer/data/trained_weights/``
   holds one file, a ``.gitignore`` reading ``*`` / ``!.gitignore``, so the
   weights were never in the clone and were never meant to be: upstream's
   ``.gitignore`` excludes them deliberately.
   ``synformer/sampler/analog/cli.py`` defaults ``--model-path`` to
   ``data/trained_weights/default.ckpt``; the README's own usage example names
   ``data/trained_weights/original_default.ckpt``. Upstream's README, section
   *Trained Models*, says to download them from
   https://huggingface.co/whgao/synformer into ``data/trained_weights``. Without
   that file there is no architecture to rebuild: both samplers read the model
   config out of the checkpoint's ``hyper_parameters.config``, so the checkpoint
   is the config as well as the weights.

2. **The building-block index is absent, and is gated on a licence, not a
   download.** ``StatePool`` cannot take a step without a ``FingerprintIndex``
   and a ``ReactantReactionMatrix``; the shipped config
   ``configs/dev_smiles_diffusion.yml`` places them at
   ``data/processed/comp_2048/fpindex.pkl`` and
   ``data/processed/comp_2048/matrix.pkl``. ``.external/synformer/data/processed/``
   does not exist. Upstream distributes the preprocessed pair on the same
   HuggingFace repository but states that it is derived from Enamine's
   building-block catalogue, which is "available only upon request", and directs
   users to request the *US Stock* catalogue from Enamine and place it in
   ``data/building_blocks``. That directory holds a ``README.md`` repeating the
   instruction and nothing else. This blocker is not fixed by more disk or more
   bandwidth.

3. **This platform's default interpreter is missing two of the sampler's
   imports.** ``import synformer.sampler.analog.parallel`` in the ``drug-agent``
   environment stops at ``ModuleNotFoundError: No module named 'omegaconf'`` —
   ``parallel.py`` needs it to rebuild the config out of the checkpoint. ``einops``
   is absent for the same reason one step further in, imported by
   ``synformer/models/transformer/graph_transformer.py`` and
   ``rotary_embedding.py``. This is the one blocker here that a ``conda install``
   fixes; it is reported rather than smoothed over because a readiness probe that
   hid it would move the failure from startup to sample time.

**What the platform does instead.** Nothing. ``generate()`` calls
``require_available()`` first and raises
:class:`~drug_agent.tools.base.ToolUnavailable` with the whole reason, and no
other backend stands in. There is no surrogate route generator and no
approximation of one: a fabricated synthesis is the most quietly damaging object
this package could emit, because unlike a fabricated score it reads as a
laboratory instruction.

**What it would compute once installed.** For each seed SMILES, a table of
synthesizable products with, per row: the product ``smiles``; ``route``, the
action string from ``Stack.get_action_string()`` — building-block SMILES and
``R<index>`` reaction tokens joined by ``;``; ``num_steps`` from
``Stack.count_reactions()``; and ``score``, the 4096-bit Morgan/Tanimoto
similarity of the product back to the seed. Upstream's README reports 30 s to
2 min per molecule on CPU and a few seconds to 30 s on an NVIDIA RTX 4090; no
timing is claimed here because none was measured here.
"""

from __future__ import annotations

import math
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with SynFormer, not SynFormer's own version (0.1.3).
SYNFORMER_ADAPTER_VERSION = "1.0.0"

#: The checkpoint ``synformer/sampler/analog/cli.py`` defaults ``--model-path`` to.
#: The README's usage example names ``original_default.ckpt`` in the same
#: directory; either satisfies the tool, and this is the one an unconfigured run
#: would reach for.
CHECKPOINT_RELPATH = "data/trained_weights/default.ckpt"

#: The preprocessed building-block artefacts, at the paths
#: ``configs/dev_smiles_diffusion.yml`` declares for the 2048-bit model.
FPINDEX_RELPATH = "data/processed/comp_2048/fpindex.pkl"
RXN_MATRIX_RELPATH = "data/processed/comp_2048/matrix.pkl"

#: ``--search-width`` in ``cli.py``, passed to ``StatePool`` as ``factor``: how
#: many reactant/reaction continuations each active state expands into.
DEFAULT_SEARCH_WIDTH = 24

#: ``--exhaustiveness`` in ``cli.py``, passed as ``max_active_states``: how many
#: partial routes survive each round of sorting.
DEFAULT_EXHAUSTIVENESS = 64

#: ``--time-limit`` in ``cli.py``, in seconds, applied per seed molecule.
DEFAULT_TIME_LIMIT_S = 180

#: ``Worker.__init__``'s ``max_evolve_steps`` — the number of ``StatePool.evolve``
#: rounds before the search gives up on a seed.
DEFAULT_MAX_EVOLVE_STEPS = 12

#: Fraction of the adapter's own budget the driver is allowed to spend on
#: sampling, leaving room for checkpoint load and index unpickling.
_BUDGET_FRACTION = 0.9

SPEC = ExternalSpec(
    name="SynFormer",
    env_prefix="SYNFORMER",
    checkout_relpath=".external/synformer",
    # Every one of these is on the analog sampler's import path and every one is
    # present in the clone, so a failure here means a truncated or wrong checkout,
    # never a missing weight.
    sentinel_files=(
        "synformer/sampler/analog/parallel.py",
        "synformer/sampler/analog/state_pool.py",
        "synformer/models/synformer.py",
        "synformer/chem/fpindex.py",
        "synformer/chem/matrix.py",
        "synformer/chem/mol.py",
        "data/rxn_templates/comprehensive.txt",
    ),
    driver_name="synformer_sample.py",
    # torch identifies a deep-learning environment; omegaconf is what separates
    # one that could host SynFormer from one that merely has torch, since
    # `parallel.py` needs it to rebuild the config out of the checkpoint.
    selection_modules=("torch", "omegaconf"),
    # Read off the import statements on the sampling path, not off
    # requirements.txt: einops comes from the transformer blocks, sklearn and
    # joblib from `chem/fpindex.py`, pandas from `state_pool.get_dataframe`.
    runtime_modules=(
        "torch",
        "omegaconf",
        "rdkit",
        "einops",
        "numpy",
        "pandas",
        "sklearn",
        "joblib",
        "tqdm",
    ),
    # The checkpoint carries the model config as well as the weights, and the two
    # pickles are the building-block search space; none of the three is in a
    # code-only clone.
    weight_relpaths=(CHECKPOINT_RELPATH, FPINDEX_RELPATH, RXN_MATRIX_RELPATH),
    conda_envs=("synformer", "drug-agent"),
    upstream_url="https://github.com/wenhao-gao/synformer",
    upstream_ref="bef02bc4481a",
    license=(
        "Apache-2.0 (upstream LICENSE). The preprocessed building-block index is derived "
        "from Enamine's US Stock catalogue, which upstream states is available only on "
        "request and for research use only"
    ),
    # Deliberately short: the readiness probe repeats this hint once per missing
    # file, and the checkout-relative paths are already named by those messages.
    # The licence gate and the two README sections are spelled out in the module
    # docstring, which is where an operator reads the whole story.
    install_hint=(
        "fetch default.ckpt, fpindex.pkl and matrix.pkl from "
        "https://huggingface.co/whgao/synformer (the index needs Enamine's US Stock "
        "catalogue, obtained on request from Enamine)"
    ),
    # Upstream documents CPU inference as supported but slow. GPU is a speed
    # decision here, not a capability one, so the card does not demand one.
    needs_gpu=False,
)


class SynFormerAdapter(GenerativeToolAdapter):
    """Synthesis-aware generation: a molecule and the route that would make it."""

    capability_id = "molecule.generate.synformer"
    spec = SPEC
    engine = "synformer"
    version = SYNFORMER_ADAPTER_VERSION
    #: The analog sampler encodes one seed molecule and scores routes by
    #: similarity to it. It has no pocket input of any kind.
    conditioning = "ligand"
    #: Generation happens in reaction/building-block space, so every product
    #: arrives with an action string.
    synthesis_aware = True

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Synthesizable analogue generation (SynFormer)",
            description=(
                "Projects a seed molecule into synthesizable chemical space by sampling a "
                "post-order sequence of building-block pushes and reaction-template "
                "applications, then reading the product off the synthesis stack. Every "
                "product carries the route that would build it; the route is a template "
                "match against a catalogue, not evidence that the synthesis works."
            ),
            input_schema_ref="GenerationSpec",
            output_schema_ref="Candidate",
            # `Synformer.predict` draws the next action with `torch.multinomial`
            # and the fingerprint head is a diffusion sampler, so two runs at the
            # same beam width differ.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no seed molecule supplied — the analog sampler has nothing to encode and "
                "cannot generate unconditionally with the shipped inference scripts",
                "no route collapses the stack to a single product within max_evolve_steps, "
                "so a seed yields zero rows and is reported as an error rather than silence",
                "the per-seed time limit expires mid-evolution: StatePool.evolve breaks out "
                "of its expansion loop and returns only the routes finished by then",
                "the fingerprint index was built at a different Morgan bit width than the "
                "checkpoint's decoder expects, so reactant retrieval returns blocks the "
                "model was never trained against",
                "the projection reproduces the seed molecule exactly instead of proposing an "
                "analogue; upstream measures this as its 'reconstruction rate'",
                "a product is a bare building block with zero reactions, which is a catalogue "
                "lookup rather than a synthesis proposal",
                "on the GPU path each Worker is a separate process that re-seeds nothing, so a "
                "fixed seed does not reproduce a run",
            ],
            validation_checks=[
                "every product SMILES parses under RDKit and is canonicalised before de-duplication",
                "heavy-atom count within the configured window",
                "a product identical to its seed is flagged reconstructed_input, not counted as new",
                "a route with zero reaction steps is flagged no_reaction_steps",
                "validity and uniqueness are computed by the domain base from the raw sample "
                "list, never reported by SynFormer itself",
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
        """Project every seed in *context_smiles* and return the products as records.

        ``pocket`` is accepted and ignored: this is a ligand-conditioned model and
        silently treating a pocket as conditioning would be the exact mislabelling
        the domain docstring forbids. ``batch_size`` is likewise unused — the
        analog sampler has no batch dimension over seeds; its parallelism is
        ``search_width`` within one seed and one worker process per GPU across
        seeds.
        """
        if not context_smiles:
            raise ValueError(
                f"{self.capability_id} is ligand-conditioned: generate() needs at least one "
                "seed molecule in context_smiles. SynFormer's shipped inference scripts all "
                "project an input molecule; none of them samples unconditionally."
            )
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() already refused
            raise GenerationBackendError("SynFormer checkout disappeared between probe and call")

        seeds = list(dict.fromkeys(s.strip() for s in context_smiles if s and s.strip()))
        budget = max(1.0, timeout_s * _BUDGET_FRACTION / max(1, len(seeds)))
        request = {
            "input_smiles": seeds,
            "model_path": str(root / CHECKPOINT_RELPATH),
            "fpindex_path": str(root / FPINDEX_RELPATH),
            "rxn_matrix_path": str(root / RXN_MATRIX_RELPATH),
            "device": self.ext.device,
            "search_width": DEFAULT_SEARCH_WIDTH,
            "exhaustiveness": DEFAULT_EXHAUSTIVENESS,
            # Never longer than upstream's tested per-molecule limit, and never
            # longer than this call's own share of the budget, so the driver
            # cannot be the thing that trips the subprocess timeout.
            "time_limit": int(min(DEFAULT_TIME_LIMIT_S, math.floor(budget))),
            "max_evolve_steps": DEFAULT_MAX_EVOLVE_STEPS,
            # Slicing an already-computed dataframe costs nothing, so ask each
            # seed for the full quota and let the base class de-duplicate across
            # seeds before truncating.
            "max_results": int(n_molecules),
            "sort_by_scores": True,
            "seed": int(seed),
        }
        run = self.run_driver(request, error_cls=GenerationBackendError, seed=seed, timeout_s=timeout_s)
        payload = run.payload
        records = [self._record(item, index) for index, item in enumerate(payload.get("molecules") or [])]
        warnings = [str(w) for w in (payload.get("warnings") or [])]
        errors = [str(e) for e in (payload.get("errors") or [])]
        return records, warnings, errors

    def _record(self, item: dict[str, Any], index: int) -> dict[str, Any]:
        """One driver row as a raw sample record, with its route and its QC flags.

        ``route`` rides on the record because a synthesis-aware generator's output
        is the route as much as the molecule. The shared ``GeneratedMolecule`` has
        no route field, so what reaches a candidate is the SMILES plus these
        flags; the route is stated here rather than dropped silently so that no
        caller reads a molecule from this tool as if a route travelled with it.
        """
        smiles = str(item.get("smiles") or "")
        target = str(item.get("target") or "")
        num_steps = int(item.get("num_steps") or 0)
        flags: list[str] = []
        if num_steps <= 0:
            flags.append("no_reaction_steps")
        if target and smiles == target:
            flags.append("reconstructed_input")
        return {
            "smiles": smiles,
            "sdf": None,
            "qc_flags": flags,
            "route": str(item.get("route") or ""),
            "route_num_steps": num_steps,
            "seed_smiles": target,
            "seed_similarity": float(item.get("score") or 0.0),
            "sample_index": int(item.get("sample_index", index)),
        }


__all__ = [
    "CHECKPOINT_RELPATH",
    "FPINDEX_RELPATH",
    "RXN_MATRIX_RELPATH",
    "SPEC",
    "SYNFORMER_ADAPTER_VERSION",
    "SynFormerAdapter",
]
