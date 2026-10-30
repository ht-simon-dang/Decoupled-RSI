"""REINVENT4 — de-novo SMILES sampled token by token from a trained prior.

REINVENT4 (AstraZeneca; https://github.com/MolecularAI/REINVENT4, pinned here at
``ee0d56f4a074``, upstream version string ``4.8.24``) is the mature end of this
catalogue. One command line covers five run modes — ``reinvent/runmodes/__init__.py``
exports ``run_sampling``, ``run_staged_learning``, ``run_transfer_learning``,
``run_scoring`` and ``run_enumeration`` — over the seven generator names in
``runmodes/samplers/run_sampling.py:HEADERS``: the unconditional ``Reinvent``
prior, ``Libinvent`` and ``Linkinvent`` with their transformer re-implementations,
``Mol2Mol`` and ``Pepinvent``.

It writes SMILES a token at a time instead of placing atoms in a pocket, so the
failure mode that dominates the rest of this package is simply absent here: there
is no point cloud, therefore no bond perception, therefore none of the allenes and
four-membered rings that the measured PFM baseline in
:mod:`drug_agent.tools.generation` records. What it has instead is a prior that
knows nothing whatsoever about a target. Unconditional sampling reproduces the
chemistry of the prior's training distribution; steering it at a target is what
the staged reinforcement-learning run mode is for, and that is a different,
scored, much longer computation than this adapter performs.

**Which mode this adapter binds.** The unconditional ``Reinvent`` prior, through
``run_type = "sampling"``. ``conditioning`` is accordingly ``"none"``, and it has
to stay that way. LibInvent (R-groups onto a scaffold), LinkInvent (a linker
between two warheads), Mol2Mol (analogues of an input molecule) and PepInvent
(peptide fillers) all read seed structures from ``parameters.smiles_file`` and are
therefore ``"ligand"``-conditioned. They are not reachable from this class,
because ``conditioning`` is a class-level statement about what every batch from it
means, and one class cannot honestly carry both answers. They also emit different
columns — ``HEADERS`` gives ``Reinvent`` three and the fragment generators five —
so the right shape for them is a second adapter over a second driver, not a flag.

**Why ``synthesis_aware`` is False.** The package docstring groups REINVENT4 with
the synthesis-aware generators, and for the framework as a whole that is fair:
``reinvent_plugins/components/SAScore``, ``.../synthsense`` and
``comp_reaction_filter.py`` let a reinforcement-learning run *score* how makeable
a molecule looks. Scoring makeability is not planning a route. The sampling run
mode emits SMILES and a negative log-likelihood and nothing else, so no record
this adapter produces ever carries a ``route`` key, and claiming otherwise would
put an empty promise of a synthesis in front of a chemist.

**Scoring dependencies are not trusted here.** The research catalogue's entry for
REINVENT4 warns, verbatim: "Verify each optional scoring dependency; some paths
involve proprietary chemistry tools." That is concrete in the checkout.
``reinvent_plugins/components/OpenEye/comp_rocs.py`` is a ROCS shape-similarity
component built on the OpenEye toolkits, which ``pyproject.toml`` declares as the
optional ``openeye`` extra served from ``https://pypi.anaconda.org/OpenEye/simple``
and which requires a commercial licence — the upstream README says plainly that
some tests need an ``OE_LICENSE``. ``comp_chemprop.py``, ``comp_chemprop2.py``,
``comp_qptuna.py``, ``comp_dockstream.py``, ``comp_icolos.py`` and ``comp_mmp.py``
each drag in a backend of their own. The sampling run mode used here touches none
of them: it loads a prior and samples. So this tool verifies, trusts and reports
no scoring component at all, and any later adapter that switches REINVENT4's
scoring on must check each component's backend individually before a number it
produces is allowed out.

**Whether it runs on this machine — measured at Step 8, 2026-09-12.**

It does. This block used to say it could not, and that statement is now false in
both of its parts; it is corrected here rather than left to mislead a later
worker into recording REINVENT4 as an unavailable competitor.

1. **The prior is present.** ``.external/REINVENT4/priors/reinvent.prior`` exists
   (23,528,840 bytes, sha256 ``fe8cd1678452ad292a8f93e97cb19a85959b729e17113e157180d5e69ae89ef3``).
   ``reinvent/utils/prior_registry.py`` computes ``PRIOR_BASE =
   pathlib.Path(reinvent.__file__).parents[1] / "priors"`` unless
   ``REINVENT_PRIOR_BASE`` is set, and ``configs/sampling.toml`` names
   ``priors/reinvent.prior`` as the ``model_file`` for de-novo sampling, so that
   is the file the sampling run mode loads. Upstream publishes the public priors
   on Zenodo: https://doi.org/10.5281/zenodo.15641296.

2. **The import set resolves.** ``available()`` is True and a real sampling run
   completes. Step 8 executed it through this adapter: 16 requested, 16 rows
   written, 15 valid and 15 unique after RDKit screening, 3.7 s on ``cpu``, with
   REINVENT4 reporting its own version as ``4.8.24``.

**What it reports.** The driver runs the real ``reinvent.Reinvent:main`` over a
generated ``sampling`` config and reads back the CSV that ``run_sampling``
writes; the base class in :mod:`drug_agent.tools.generation` does the screening.

    ``molecules[].smiles``       canonical SMILES, canonicalised by RDKit here
    ``molecules[].qc_flags``     heavy-atom window violations, unparseable SMILES
    ``n_raw``                    rows REINVENT4 wrote, equal to ``num_smiles``
    ``n_valid`` / ``n_unique``   recomputed with RDKit, never read off upstream
    ``generation_validity``      ``n_valid / n_raw``, dimensionless
    ``generation_uniqueness``    ``n_unique / n_valid``, dimensionless
    ``runtime_seconds``          wall time of the whole call

**Training is reachable, and the generation path knows which weights it loaded.**
This block used to say that it was not: that ``target/baselines.py`` probed the
two verbs separately and the verified answer for ``fit`` was no. That is no
longer true. :meth:`Reinvent4Adapter.fit` drives ``run_transfer_learning``
through ``scripts/external/reinvent4_tl.py``, the same way :meth:`_sample` drives
``run_sampling``, and reports:

    ``checkpoint``          absolute path of the fine-tuned weights
    ``sha256``              sha256 of the checkpoint's bytes
    ``input_model_sha256``  sha256 of the prior it started from
    ``n_training_smiles``   REINVENT4's own count of what it trained on, read from
                            its log — not the row count of the input file
    ``weights_sha256``      sha256 of the checkpoint's tensors alone
    ``input_model_weights_sha256``  and of the prior's
    ``n_smiles_file_rows``  the rows the file offered; ``fit`` refuses to return
                            at all unless ``n_training_smiles`` equals it
    ``seed``                the seed the run was set to
    ``epochs``              epochs asked for
    ``wall_seconds``        wall time of the ``main()`` call
    ``gpu_seconds``         measured GPU seconds, or ``None`` when no GPU
                            telemetry was available

**The adaptation is reproducible, and the training set reconciles.** Two
properties that a first working ``fit`` did not have, and that a digest recorded
as evidence is worthless without.

*Reproducible — but not in the file, and not because of the seed.* Every save
goes through ``reinvent/models/meta_data.py:update_model_data``, which appends
``time.time()`` to the checkpoint's metadata (line 64) and hashes it into
``hash_id``. A wall-clock reading is therefore inside every checkpoint REINVENT4
writes, so ``sha256`` differs between two runs that produced bit-identical
weights, and no sha256 of one of these files can ever be regenerated — by
anybody, at any seed. ``weights_sha256`` covers the tensors and nothing else and
does reproduce: measured on 2026-09-17, four one-epoch fits of the same twelve
molecules gave four different ``sha256`` and one single ``weights_sha256``.

``fit`` passes a seed on every path all the same —
:data:`REINVENT4_TL_DEFAULT_SEED` when the caller names none — to
``reinvent.utils.set_seed`` and, separately, to ``PYTHONHASHSEED`` in the
driver's environment, which ``set_seed`` cannot set for itself. It is honest
about what that buys: for the ``Reinvent`` generator, nothing yet. The training
``DataLoader`` shuffles with a fresh ``torch.Generator()`` at the fixed default
seed ``67280421310721``, which ``torch.manual_seed`` does not touch, and
``randomize_smiles`` draws from RDKit's own default-seeded generator rather than
from ``random`` or ``numpy``. Two fits at seed 42 and two at seed 7 landed on the
identical weights. The seed is the correct thing to send, it is what any future
upstream will honour, and it is not the reason anything here reproduces.

**So a TL arm's two campaigns are not two independent draws of the adaptation.**
They share one checkpoint's weights, bit for bit; they differ only in what
happens downstream of it. That sentence is here, and in :meth:`fit`'s own
docstring, and in ``known_failure_modes``, because "we varied the TL seed" is
otherwise exactly the claim a later reader will make of a campaign table that
lists two seeds beside one arm.

*Reconciled*: ``read_smiles_csv_file`` drops a row upstream's standardiser
refuses with ``if not smiles: continue`` — no exception, exit 0, nothing in the
payload — so a file of 775 trains 700 and reports 700 as though that answered the
question. ``fit`` now compares ``n_training_smiles`` against the file's own row
count and raises on any disagreement, and
:func:`~drug_agent.selfevolve.target.tl_data.build_training_smiles` pre-filters
for the whole of what the standardiser drops, not only the tokeniser, so the
equality is something the input is built to satisfy rather than something the run
is hoped to have.

The second way that equality can break is not a filter at all.
``read_smiles_csv_file`` is called with ``remove_duplicates=True``, so two rows
that standardise to one structure are one training molecule — and with stereo
stripped, an enantiomeric pair, an amine and its protonated form, an acid and its
sodium salt and a cis/trans alkene pair each are one structure. That collapse is
also a coin flip, because the de-duplication key is the string after
``randomize_smiles``. ``build_training_smiles`` therefore de-duplicates on the
standardised form before the file is written and reports
``n_duplicates_collapsed``; after that the row count is the training-set size by
construction rather than by luck. The guard in ``fit`` is unchanged — it is what
found this — and it is the writer that was brought into agreement with it.

``run_staged_learning`` — reinforcement learning against a scoring function — is
still not bound here. That is a scored, much longer computation with its own
component backends, and ``fit`` does not pretend to be it.

The dangerous half of that change is not the training. It is that
:meth:`_prior_path` used to hardcode the parent prior with no way to say
otherwise, so a working ``fit`` on its own would have produced a real fine-tuned
checkpoint that **nothing ever sampled from** — every subsequent batch would have
come from the untouched parent, with no error, no warning and numbers that look
exactly like the incumbent's. Two things prevent that failure from being silent:
``generate(checkpoint=...)`` threads an override all the way to the request's
``model_file``, and every batch records ``config["sample_provenance"]`` naming the
checkpoint that was actually loaded and the sha256 of its bytes. A run that fell
back to the parent says so in its own record, after the fact, without anybody
having to have suspected it.

**What it deliberately does not do.** It does not carry the per-molecule negative
log-likelihood REINVENT4 reports: :class:`~drug_agent.tools.pfm_tool.GeneratedMolecule`
has no field for a model score, and encoding a float into a ``qc_flags`` string to
smuggle it past the contract is exactly the kind of number that reads as a
measurement three artifacts downstream. It does not score, dock, filter on
properties, or run reinforcement learning. It does not accept the platform's
``pocket`` or ``context_smiles`` as conditioning — both are recorded, and their
presence is reported as a batch warning, because a caller who passed a pocket to
an unconditional prior has misunderstood what they will get back.

**What the platform does when the prior is absent.** Nothing, loudly.
``available()`` becomes False, the capability card reports UNAVAILABLE and names
``priors/reinvent.prior`` outright, and ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` through ``require_available()``.
There is no substitute prior, no cached sample stream and no fallback generator,
per :mod:`drug_agent.tools.base`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import measure
from .._ext.backend import ExternalBackend
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with REINVENT4, not REINVENT4's own version.
REINVENT4_ADAPTER_VERSION = "1.0.0"

#: The de-novo prior the ``sampling`` run mode loads, relative to the checkout
#: root. ``reinvent/utils/prior_registry.py`` resolves the registry key
#: ``.reinvent`` to this same path, and ``configs/sampling.toml`` names it as the
#: default ``model_file``.
REINVENT4_PRIOR_RELPATH = "priors/reinvent.prior"

#: The transfer-learning driver, alongside ``reinvent4_sample.py``. It is a second
#: driver rather than a flag on the first because the two run modes take different
#: configuration blocks, write different artifacts and fail in different ways;
#: ``SPEC.driver_name`` names the sampling one, so ``fit`` launches through a
#: copy of the spec that names this one instead.
REINVENT4_TL_DRIVER = "reinvent4_tl.py"

#: The seed :meth:`Reinvent4Adapter.fit` trains under when a caller names none.
#:
#: It is a *default*, not a fallback: an unseeded transfer-learning run produces a
#: checkpoint whose bytes cannot be regenerated, and a sha256 that addresses
#: unregenerable bytes is not evidence of anything. 42 is the same value every
#: other unspecified seed in this program takes — ``actions.py`` reads
#: ``sampling.get("seed", 42)`` and ``preparation.get("seed", 42)`` — so the
#: adaptation path and the sampling path that loads its checkpoint agree by
#: default instead of by coincidence. Nothing about it is re-rolled: a caller who
#: pins a seed gets that seed, and a caller who pins none gets this one, forever.
REINVENT4_TL_DEFAULT_SEED = 42

#: ``numpy.random.seed``'s range, which ``reinvent.utils.helpers.set_seed`` calls
#: unguarded. Checked here so a bad seed costs a function call rather than a
#: process launch and a 23 MB model load.
REINVENT4_SEED_MAX = 2**32 - 1

#: Every third-party package ``import reinvent.Reinvent`` pulls in through an
#: *unconditional module-level* import, found by walking the checkout's import
#: graph rather than by reading ``pyproject.toml`` — the dependency list there
#: includes packages the sampling path never touches (``polars``, ``apted``,
#: ``Pillow``) and would therefore over-report breakage. The last four are the
#: ones a half-finished install actually loses here: ``reinvent/runmodes/__init__.py``
#: imports all five run modes eagerly, so sampling drags in the TensorBoard
#: reporter (``tensorboard``), its molecule-grid images (``torchvision``), the
#: Mol2Mol matched-pair preprocessing (``mmpdblib``) and the scoring transforms
#: (``pumas``) whether or not the run uses them.
REINVENT4_RUNTIME_MODULES: tuple[str, ...] = (
    "torch",
    "rdkit",
    "numpy",
    "pandas",
    "scipy",
    "pydantic",
    "dotenv",
    "yaml",
    "molvs",
    "xxhash",
    "tqdm",
    "requests",
    "tensorboard",
    "torchvision",
    "mmpdblib",
    "pumas",
)

SPEC = ExternalSpec(
    name="REINVENT4",
    env_prefix="REINVENT4",
    checkout_relpath=".external/REINVENT4",
    # Files the sampling run path really imports: the entry point the driver
    # calls, the run mode it dispatches to, and the prior resolver that decides
    # where a checkpoint lives.
    sentinel_files=(
        "reinvent/Reinvent.py",
        "reinvent/runmodes/samplers/run_sampling.py",
        "reinvent/utils/prior_registry.py",
    ),
    driver_name="reinvent4_sample.py",
    # `reinvent` is the one import that says which environment upstream's
    # install.py was run into. It is deliberately absent from runtime_modules:
    # the driver receives the checkout on PYTHONPATH, so the package itself comes
    # from the clone, and sentinel_files already prove the clone is real. What a
    # broken install loses is the dependency set, not the source.
    selection_modules=("reinvent",),
    runtime_modules=REINVENT4_RUNTIME_MODULES,
    weight_relpaths=(REINVENT4_PRIOR_RELPATH,),
    conda_envs=("reinvent4", "drug-agent-reinvent4", "drug-agent"),
    upstream_url="https://github.com/MolecularAI/REINVENT4",
    upstream_ref="ee0d56f4a074",
    license="REINVENT4 Apache-2.0",
    # Repeated verbatim into every line of the unavailable reason, so it stays
    # short and still names where upstream publishes the priors.
    install_hint="make reinvent4-install (priors: https://doi.org/10.5281/zenodo.15641296)",
    # Upstream's README: a GPU is "not strictly necessary but strongly
    # recommended", and sampling from the recurrent prior is its cheapest mode.
    # Set DRUGAGENT_REINVENT4_DEVICE to move it onto CUDA.
    needs_gpu=False,
)


class Reinvent4Adapter(GenerativeToolAdapter):
    """Unconditional de-novo SMILES sampling from REINVENT4's ``Reinvent`` prior."""

    capability_id = "molecule.generate.reinvent4"
    spec = SPEC
    engine = "reinvent4"
    version = REINVENT4_ADAPTER_VERSION
    #: The Reinvent prior is unconditional. The scaffold, linker, analogue and
    #: peptide modes are "ligand" and are not bound by this class — see the module
    #: docstring.
    conditioning = "none"
    #: REINVENT4 can score synthesizability; it plans no route and emits none.
    synthesis_aware = False

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="De-novo molecule generation (REINVENT4 prior)",
            description=(
                "Unconditional de-novo SMILES sampled from REINVENT4's released Reinvent "
                "prior through its 'sampling' run mode. Proposes molecules; scores, docks "
                "and optimises nothing, and does not condition on a pocket or a reference "
                "ligand."
            ),
            # GenerationSpec, not PocketSpec: its fields map one-to-one onto
            # generate() and its `pocket` is optional, which is the only honest
            # input contract for a generator that conditions on nothing.
            input_schema_ref="GenerationSpec",
            output_schema_ref="Candidate",
            # A multinomial sample from an RNN. A seed pins one run; the model is
            # a sampler, not a function.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "priors/reinvent.prior absent — the clone is code-only and upstream ships "
                "priors separately, on Zenodo",
                "a prior of the wrong type silently changes the generator: create_adapter "
                "reads model_type out of the torch pickle, so a Mol2Mol or LibInvent file "
                "would sample a ligand-conditioned model under a card that says 'none'",
                "the prior's own chemistry is the ceiling — an unconditional sample knows "
                "nothing about the target and is not steered by reinforcement learning here",
                "torch.load of the .prior runs with weights_only=False (create_adapter), so "
                "an untrusted prior file executes arbitrary code on load",
                "REINVENT4's optional scoring components need backends this tool never "
                "verifies, including the proprietary OpenEye toolkits behind comp_rocs.py",
                "run_sampling writes its CSV by joining fields with commas and no quoting, "
                "so a malformed row is possible in principle and is counted, not guessed at",
                "upstream's standardiser drops anything outside C/N/O/F/S/Cl/Br, outside "
                "2 < heavy atoms < 70, carrying [C-]#[N+]/[OH+]/[SH], or that RDKit cannot "
                "parse, before the tokeniser ever sees it, and read_smiles_csv_file skips it "
                "with 'if not smiles: continue' — exit 0, nothing in the payload (measured: "
                "51 rows offered, 50 trained, no warning). fit() now refuses to return when "
                "n_training_smiles differs from the file's row count, so this is a raise "
                "rather than a smaller number nobody compared",
                "fit() aborts on the first out-of-vocabulary token when standardisation is "
                "off, because config_parse.validate_tokens raises ValueError and nothing "
                "catches it; tl_data.build_training_smiles pre-filters for exactly this",
                "a fine-tuned checkpoint that nothing samples from is invisible without "
                "config['sample_provenance'] — the weights are real, the batch is the "
                "incumbent's, and no error is raised anywhere",
                "no sha256 of a REINVENT4 checkpoint file is reproducible: every save "
                "appends time.time() to the model's metadata and hashes it into hash_id "
                "(models/meta_data.py:64), so two runs with bit-identical weights address "
                "differently. Cite fit()'s weights_sha256 when a digest has to mean 'the same "
                "generator', and read a file sha256 as 'these exact bytes on this disk'",
                "the TL seed gives reproducibility, NOT seed-sensitivity: four fits at seeds "
                "42, 42, 7, 7 produced one single weights digest, because upstream builds a "
                "fresh torch.Generator at runmodes/TL/learning.py:151 and randomises with "
                "MolToSmiles(doRandom=True) at chemistry/conversions.py:181, neither of which "
                "set_seed reaches. A TL arm's two campaigns are therefore NOT independent "
                "draws of the adaptation — they share one checkpoint's weights bit for bit and "
                "differ only downstream of it, so their spread measures sampling and scoring "
                "and says nothing about the variability of the fine-tuning. 'We varied the TL "
                "seed' is not a claim this adapter's output supports",
                "PYTHONHASHSEED is not a second seeding mechanism: _ext/backend.py:child_env "
                "inherits it from the parent through the env allowlist and then setdefaults "
                "the run's seed onto it, so a parent that exports PYTHONHASHSEED silently wins "
                "and fit()'s seed never reaches the driver. Unset in this program's parents, "
                "so harmless here, and recorded so it is not relied on elsewhere",
                "a training SMILES file with two rows that standardise to one structure is a "
                "file whose row count is not the training-set size: read_smiles_csv_file is "
                "called with remove_duplicates=True (run_transfer_learning.py:80) and collapses "
                "them, and because it compares the string AFTER randomize_smiles the collapse "
                "fires or not by chance — measured at 0.014-0.055 per drug-sized pair and 1.0 "
                "for a symmetric one, deterministic within a process and unpredictable across "
                "files. fit() then refuses AFTER the GPU is spent. Stereo is stripped before "
                "the comparison, so enantiomers, an amine and its protonated form, an acid and "
                "its salt, and cis/trans alkenes are each one molecule; "
                "tl_data.build_training_smiles de-duplicates on that same standardised form "
                "before writing and reports n_duplicates_collapsed",
            ],
            validation_checks=[
                "the CSV header REINVENT4 wrote is matched against run_sampling.HEADERS, and "
                "a model_type other than 'Reinvent' fails the batch rather than mislabelling "
                "its conditioning",
                "unique_molecules is forced false upstream so n_raw is the full sample "
                "stream and validity has an honest denominator",
                "validity and uniqueness are recomputed here with RDKit; upstream's own "
                "SmilesState is reported as provenance and never used as the count",
                "rows whose field count disagrees with the header are counted and reported, "
                "never silently dropped",
                "every batch records the sha256 of the checkpoint that was actually loaded, "
                "whether or not an override was asked for, so a sampling call that fell back "
                "to the parent prior is detectable from the record afterwards",
                "fit() refuses a checkpoint whose weights digest equals the prior's: that is "
                "a run in which no gradient step changed anything, and it would otherwise "
                "sample indistinguishably from the incumbent. The byte comparison it used to "
                "rely on cannot detect this — meta_data.py:64 stamps time.time() into every "
                "save, so the files always differ",
                "fit() trains under a seed on every path — the caller's, or "
                "REINVENT4_TL_DEFAULT_SEED — and refuses seed 0, which upstream's "
                "'args.seed or ...' would silently turn back into an unseeded run. Measured: "
                "the seed does not yet reach the weights of a Reinvent TL run, so "
                "weights_sha256 and not the seed is what makes a checkpoint attributable",
                "fit() reconciles REINVENT4's own n_training_smiles against the row count of "
                "the file it was handed and raises on any gap, so a training set the "
                "standardiser silently shrank cannot be reported as an adaptation",
            ],
        )

    # --- real backend ------------------------------------------------------
    def _prior_path(self, checkpoint: str | Path | None = None) -> Path:
        """The checkpoint file the sampling run mode loads.

        With no *checkpoint* this is the parent prior and nothing about the call
        has changed. With one it is that file, checked for existence **here**
        rather than inside the driver: ``create_adapter`` would raise
        ``RuntimeError: model file ... is not accessible`` three processes away,
        and a fine-tuned checkpoint that silently does not exist is exactly the
        failure that would otherwise be reported as an ordinary incumbent-looking
        batch.
        """
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise GenerationBackendError(
                    f"checkpoint {path} does not exist, so this batch would have been "
                    "sampled from the parent prior while claiming to be adapted"
                )
            return path.resolve()
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() refuses first
            raise GenerationBackendError("REINVENT4 checkout could not be resolved")
        return Path(root) / REINVENT4_PRIOR_RELPATH

    #: sha256 by path, so a 23 MB prior is hashed once per process rather than
    #: once per batch. Keyed on ``(path, size, mtime_ns)`` so a checkpoint
    #: rewritten in place — which is exactly what a TL run does — is re-hashed.
    _SHA_CACHE: dict[tuple[str, int, int], str] = {}

    @classmethod
    def _sha256(cls, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        cached = cls._SHA_CACHE.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        cls._SHA_CACHE[key] = digest.hexdigest()
        return cls._SHA_CACHE[key]

    def _sample(
        self,
        *,
        n_molecules: int,
        pocket: dict[str, Any],
        context_smiles: list[str],
        seed: int,
        batch_size: int,
        timeout_s: float,
        checkpoint: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Run REINVENT4's ``sampling`` run mode once and return its raw rows."""
        warnings: list[str] = []
        errors: list[str] = []

        if pocket:
            warnings.append(
                "a pocket was supplied but REINVENT4's Reinvent prior is unconditional; it "
                "is recorded on the batch config and did not influence sampling"
            )
        if context_smiles:
            warnings.append(
                f"{len(context_smiles)} context SMILES were ignored: this adapter binds the "
                "unconditional Reinvent prior, not Mol2Mol/LibInvent/LinkInvent, which are "
                "the modes that read seed structures"
            )
        if batch_size and batch_size < n_molecules:
            warnings.append(
                f"batch_size={batch_size} was not forwarded: run_sampling sets the sampler's "
                f"batch size to num_smiles and chunks internally at SAMPLE_BATCH_SIZE=1000 "
                f"(runmodes/samplers/params.py), so all {n_molecules} were requested at once"
            )

        model_file = self._prior_path(checkpoint)
        parent_prior = self._prior_path()
        is_parent = model_file.resolve() == parent_prior.resolve()
        # Requirement of the whole change: the batch records which weights were
        # loaded, always — not only when an override was asked for. A silent
        # fall-back to the parent prior is only detectable after the fact if the
        # default path is stamped too.
        self._sample_provenance = {
            "checkpoint": str(model_file),
            "checkpoint_sha256": self._sha256(model_file),
            "checkpoint_requested": str(checkpoint) if checkpoint is not None else None,
            "is_parent_prior": is_parent,
            "parent_prior": str(parent_prior),
        }
        if checkpoint is not None and is_parent:
            warnings.append(
                f"checkpoint={checkpoint} resolves to the parent prior, so this batch is "
                "the incumbent generator and must not be reported as adapted"
            )

        request = {
            "model_file": str(model_file),
            "num_smiles": int(n_molecules),
            "device": self.ext.device,
            "seed": int(seed),
            # See the driver: filtering upstream would destroy the denominator
            # the platform needs to compute validity.
            "unique_molecules": False,
            "randomize_smiles": True,
            "temperature": 1.0,
        }
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            seed=seed,
        )
        payload = run.payload

        model_type = str(payload.get("model_type") or "")
        if model_type != "Reinvent":
            headers = payload.get("headers") or []
            errors.append(
                f"{model_file} loaded as generator {model_type or 'unknown'} "
                f"(CSV header {headers}), not the unconditional Reinvent prior; this "
                "capability reports conditioning='none' and refuses to relabel a "
                "ligand-conditioned batch"
            )
            return [], warnings, errors

        rows = payload.get("molecules") or []
        malformed = int(payload.get("n_malformed_rows") or 0)
        if malformed:
            warnings.append(
                f"{malformed} CSV rows written by REINVENT4 did not match its own header and "
                "were dropped before screening"
            )
        upstream_version = str(payload.get("reinvent_version") or "unknown")
        warnings.append(
            f"REINVENT4 {upstream_version} sampled {len(rows)} rows on device "
            f"{payload.get('device')} in {run.seconds:.1f}s; "
            f"{self._state_summary(rows)}. Validity and uniqueness below are recomputed "
            "here with RDKit and do not use these counts."
        )
        warnings.append(
            f"weights loaded: {model_file.name} sha256 "
            f"{self._sample_provenance['checkpoint_sha256']} "
            f"({'the parent prior' if is_parent else 'a fine-tuned checkpoint'})"
        )

        records: list[dict[str, Any]] = [{"smiles": str(row.get("smiles") or "")} for row in rows]
        return records, warnings, errors

    # --- transfer learning -------------------------------------------------
    def _tl_backend(self) -> ExternalBackend:
        """The same checkout and interpreter, launching the TL driver instead.

        ``ExternalBackend`` resolves one driver per spec, and ``SPEC`` names the
        sampling one. Rather than teach the launcher about two, ``fit`` builds a
        second backend from a copy of the spec whose ``driver_name`` is the TL
        script; everything else — root, interpreter, device, readiness — is
        resolved by exactly the same code, so ``fit`` cannot end up pointed at a
        different checkout than ``generate``. Built once and cached: readiness
        resolution spawns an import check per interpreter.
        """
        backend = getattr(self, "_tl_backend_cache", None)
        if backend is None:
            backend = ExternalBackend(
                dataclasses.replace(self.spec, driver_name=REINVENT4_TL_DRIVER),
                getattr(self.ext, "_settings", None),
            )
            self._tl_backend_cache = backend
        return backend

    def _resolve_seed(self, seed: int | None) -> int:
        """The seed this run trains under, refusing the two values that lie.

        ``0`` is not seed zero. ``reinvent/Reinvent.py:120`` resolves the seed as
        ``args.seed or getattr(val_config, "seed", None)``, so a falsy 0 seeds
        nothing and the run is as unreproducible as an unseeded one while every
        record of it says ``seed: 0``. Out-of-range is refused for the duller
        reason that ``set_seed`` calls ``numpy.random.seed`` unguarded, and
        finding that out after the model has loaded wastes the whole run.
        """
        resolved = REINVENT4_TL_DEFAULT_SEED if seed is None else int(seed)
        if resolved == 0:
            raise GenerationBackendError(
                "seed 0 is refused: reinvent/Reinvent.py:120 resolves the seed as "
                "'args.seed or ...', so 0 is falsy and seeds nothing, producing an "
                "unreproducible checkpoint under a record that claims a seed. Pass any "
                f"seed in 1..{REINVENT4_SEED_MAX}, or omit it for "
                f"{REINVENT4_TL_DEFAULT_SEED}"
            )
        if not 0 < resolved <= REINVENT4_SEED_MAX:
            raise GenerationBackendError(
                f"seed must be in 1..{REINVENT4_SEED_MAX} (the range numpy.random.seed "
                f"accepts, which reinvent.utils.set_seed calls unguarded), got {resolved}"
            )
        return resolved

    def fit(
        self,
        *,
        smiles_file: Path,
        output_model_file: Path,
        num_epochs: int,
        batch_size: int,
        input_model_file: Path | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Transfer-learn a REINVENT4 prior onto *smiles_file*.

        ``input_model_file=None`` means the parent prior, which is the only
        starting point a first adaptation has. Any other value is a checkpoint a
        previous ``fit`` wrote, so a campaign can adapt in stages.

        ``smiles_file`` must be tab-delimited with no header and SMILES in column
        0, and every molecule in it must be inside the prior's vocabulary —
        ``validate_tokens`` raises on the first token that is not and nothing
        upstream catches it, so one bad compound costs the whole run.
        :func:`drug_agent.selfevolve.target.tl_data.build_training_smiles` is what
        writes a file that satisfies both.

        **The run is seeded, always** — ``seed=None`` means
        :data:`REINVENT4_TL_DEFAULT_SEED`, not "unseeded" — **and that is not why
        the result is reproducible.** Both statements matter, and the second one
        is the one a reader is likely to get wrong. The seed is sent to the two
        places a caller can send it: ``reinvent.utils.set_seed`` through the
        driver's request, and ``PYTHONHASHSEED`` in the driver's environment,
        which ``set_seed`` cannot set for itself because it runs inside an
        interpreter that already started. The second of those is weaker than it
        looks and is not a second mechanism to rely on:
        ``_ext/backend.py:child_env`` inherits ``PYTHONHASHSEED`` from this
        process through its allowlist and then *setdefaults* the seed onto it, so
        a parent that already exports ``PYTHONHASHSEED`` silently keeps its own
        value and the seed here never reaches the child. Harmless as this runs —
        the parent does not set it — but it is one place, not two, whenever it is
        set. Measured on 2026-09-17, neither reaches
        the weights for the ``Reinvent`` generator: the training ``DataLoader``
        is built with a fresh ``torch.Generator(device=...)``
        (``runmodes/TL/learning.py:151``) whose fixed default seed
        ``torch.manual_seed`` does not touch, and ``randomize_smiles`` reaches
        ``MolToSmiles(..., doRandom=True)``
        (``chemistry/conversions.py:181``), which draws from RDKit's own
        generator rather than ``random`` or ``numpy``. Four fits — at seeds 42,
        42, 7 and 7 — produced one single weights digest between them.
        The seed is still sent because it is the only correct thing to send and
        because it is what will carry the run when any of that changes upstream —
        but it is never the evidence, and this docstring will not let it be
        mistaken for the evidence. Nothing in the vendored checkout is patched to
        change this: upstream is not ours to edit, and manufacturing
        seed-sensitivity by editing it would be a worse answer than stating
        plainly what the number is.

        **Which makes the seed a reproducibility control and not a replicate,
        and the difference decides what a campaign table may claim.** Same
        inputs, same weights — that is what this program needs and what
        ``weights_sha256`` records. It is *not* seed-sensitivity: two seeds do
        not give two samples of the adaptation, they give the same adaptation
        twice. **A TL arm's two campaigns are therefore not independent draws of
        the adaptation.** Both load one checkpoint whose tensors are identical,
        and they differ only in what happens downstream of it — the sampling,
        the scoring, the selection. Their spread measures that downstream, and
        nothing about the variability of the fine-tuning. Anyone writing "we
        varied the TL seed" over such a pair would be reporting a variance that
        was never measured.

        **What is reproducible is ``weights_sha256``, and what is not is
        ``sha256``.** ``reinvent/models/meta_data.py:64`` appends ``time.time()``
        to the checkpoint's metadata on every save and hashes it into
        ``hash_id``, so the *file* carries a wall-clock reading and its sha256 can
        never be regenerated by anybody at any seed. The four different digests
        that made this a finding have that cause, not a missing seed, and no seed
        could have fixed them. ``weights_sha256`` covers the tensors alone,
        excludes the metadata, answers the question the digest is actually asked —
        will this generator sample the same way — and reproduces exactly.

        **What it trained on is reconciled, not reported.** ``read_smiles_csv_file``
        drops a row the standardiser refuses with ``if not smiles: continue``, no
        exception and no non-zero exit, so a file of 775 can train 700 and report
        700 as if that were the answer to "how many did you offer". This method
        therefore compares REINVENT4's own ``n_training_smiles`` against the row
        count of the file it was handed and **raises** when they disagree, naming
        how many the standardiser rejected. A training set that shrank is either a
        finding about the input or a finding about the filter; it is never a
        detail.

        Returns:
            ``checkpoint`` (absolute path to the weights that were written),
            ``sha256`` of its bytes, ``n_training_smiles`` (REINVENT4's own count
            of what it trained on, not the row count of the file), ``epochs``,
            ``wall_seconds``, ``gpu_seconds`` (``None`` when no GPU telemetry was
            available — never a zero standing in for an unmeasured number),
            ``input_model_sha256`` (the prior it started from),
            ``input_model_weights_sha256`` (that prior's tensors alone),
            ``weights_sha256`` (this checkpoint's tensors alone — the digest that
            reproduces, where ``sha256`` cannot),
            ``n_smiles_file_rows`` (the rows the file offered, which
            ``n_training_smiles`` is guaranteed to equal by the check above) and
            ``seed``, the value the run was set to.

        Raises:
            ToolUnavailable: REINVENT4 is not installed here.
            GenerationBackendError: the training file or the input checkpoint is
                missing, the seed is one upstream would silently drop, the run
                mode failed, the checkpoint is byte-identical to its prior, or
                REINVENT4 trained on a different number of molecules than the file
                offered.
        """
        self.require_available()

        smiles_path = Path(smiles_file).expanduser()
        if not smiles_path.is_file():
            raise GenerationBackendError(f"training SMILES file {smiles_path} does not exist")
        prior_path = self._prior_path(input_model_file)
        output_path = Path(output_model_file).expanduser().resolve()
        if int(num_epochs) < 1:
            raise GenerationBackendError(f"num_epochs must be >= 1, got {num_epochs}")
        if int(batch_size) < 1:
            raise GenerationBackendError(f"batch_size must be >= 1, got {batch_size}")
        resolved_seed = self._resolve_seed(seed)

        request = {
            "input_model_file": str(prior_path),
            "output_model_file": str(output_path),
            "smiles_file": str(smiles_path.resolve()),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "device": str(device or self.ext.device),
            # Without this the checkpoint's bytes are unrepeatable and its sha256
            # addresses nothing anybody can produce again.
            "seed": resolved_seed,
            # info is not cosmetic here: the driver reads REINVENT4's own
            # "Read N input SMILES" line to report what actually trained.
            "log_level": "info",
        }

        started = time.perf_counter()
        with measure(self.capability_id, "fit", backend=self.backend) as metered:
            run = self._tl_backend().run_json(
                request,
                timeout_s=self.timeout_s,
                error_cls=GenerationBackendError,
                # The same seed again, for a different mechanism: child_env turns
                # it into PYTHONHASHSEED, which has to be in the environment
                # before the interpreter starts and so cannot be set by set_seed.
                seed=resolved_seed,
            )
            payload = run.payload
            metered["n_training_smiles"] = float(payload.get("n_training_smiles") or 0)
            metered["epochs"] = float(payload.get("epochs") or 0)
        wall = float(payload.get("wall_seconds") or (time.perf_counter() - started))

        checkpoint = Path(str(payload["checkpoint"]))
        if not checkpoint.is_file():  # pragma: no cover - the driver checks first
            raise GenerationBackendError(f"REINVENT4 reported a checkpoint at {checkpoint} that is not there")
        written_sha = str(payload["checkpoint_sha256"])
        written_weights = payload.get("checkpoint_weights_sha256")
        prior_weights = payload.get("input_model_weights_sha256")
        # The weights, not the file. Comparing the files cannot answer this
        # question and never could: update_model_data appends time.time() to
        # metadata["updates"] on every save (meta_data.py:64), so the output
        # differs from the input in bytes even when not one parameter moved, and
        # the byte comparison below has never been capable of firing. It is kept
        # because it is still true and still cheap; the digest comparison is the
        # one that detects a run in which no gradient step changed anything — a
        # checkpoint that would sample indistinguishably from the incumbent while
        # every record called it adapted.
        if written_weights is not None and written_weights == prior_weights:
            raise GenerationBackendError(
                f"the checkpoint written to {checkpoint} carries the same weights as the "
                f"prior it started from (weights sha256 {written_weights}); no gradient step "
                f"changed anything, and sampling from it would re-measure the incumbent under "
                f"an adapted generator's name"
            )
        if written_sha == str(payload["input_model_sha256"]):  # pragma: no cover - see above
            raise GenerationBackendError(
                f"the checkpoint written to {checkpoint} is byte-identical to the prior it "
                f"started from (sha256 {written_sha}); no training happened"
            )

        n_training_smiles = int(payload["n_training_smiles"])
        n_file_rows = int(payload["n_smiles_file_rows"])
        if n_training_smiles != n_file_rows:
            rejected = payload.get("n_standardizer_rejected")
            attributed = (
                f"upstream's standardiser logged {rejected} distinct SMILES as invalid"
                if rejected
                else "upstream's standardiser logged no rejection, so the difference is "
                "duplicate collapse in read_smiles_csv_file (remove_duplicates=True) or a "
                "row it skipped"
            )
            raise GenerationBackendError(
                f"REINVENT4 trained on {n_training_smiles} molecules from a file offering "
                f"{n_file_rows} rows, so {abs(n_file_rows - n_training_smiles)} "
                f"{'went missing' if n_file_rows > n_training_smiles else 'appeared from nowhere'}. "
                f"read_smiles_csv_file drops a refused row with 'if not smiles: continue' — no "
                f"exception, exit 0, and the shortfall is invisible in every number this run "
                f"would otherwise report. {attributed}. The checkpoint at {checkpoint} was "
                f"written and is NOT reported: an adaptation whose training set cannot be "
                f"reconciled against the set that was offered is unattributable. Pre-filter the "
                f"input with tl_data.build_training_smiles, which now screens for exactly what "
                f"the standardiser drops, or investigate why these rows are refused"
            )

        return {
            "checkpoint": str(checkpoint.resolve()),
            "sha256": written_sha,
            "weights_sha256": (None if written_weights is None else str(written_weights)),
            "n_training_smiles": n_training_smiles,
            "n_smiles_file_rows": n_file_rows,
            "seed": resolved_seed,
            "epochs": int(payload["epochs"]),
            "wall_seconds": wall,
            "gpu_seconds": (None if payload.get("gpu_seconds") is None else float(payload["gpu_seconds"])),
            "input_model_sha256": str(payload["input_model_sha256"]),
            "input_model_weights_sha256": (None if prior_weights is None else str(prior_weights)),
        }

    @staticmethod
    def _state_summary(rows: list[dict[str, Any]]) -> str:
        """How REINVENT4 itself labelled the rows, as provenance only.

        ``SmilesState`` (0 INVALID, 1 VALID, 2 DUPLICATE) is assigned by
        ``samplers/sampler.py:validate_smiles``. It is reported and never
        substituted for the platform's own counts: the domain base is explicit
        that a generator does not get to report its own validity.
        """
        names = {0: "INVALID", 1: "VALID", 2: "DUPLICATE"}
        counts: dict[str, int] = {}
        for row in rows:
            state = row.get("state")
            # The driver writes None when a row's SMILES_state did not parse as an
            # integer, which is a real thing to report rather than round to VALID.
            label = names.get(state, "UNLABELLED") if isinstance(state, int) else "UNLABELLED"
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return "it labelled no rows"
        listed = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
        return f"it labelled them {listed} (its own SmilesState)"


__all__ = [
    "REINVENT4_ADAPTER_VERSION",
    "REINVENT4_PRIOR_RELPATH",
    "REINVENT4_RUNTIME_MODULES",
    "REINVENT4_SEED_MAX",
    "REINVENT4_TL_DEFAULT_SEED",
    "REINVENT4_TL_DRIVER",
    "SPEC",
    "Reinvent4Adapter",
]
