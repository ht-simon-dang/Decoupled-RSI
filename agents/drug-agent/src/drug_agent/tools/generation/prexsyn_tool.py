"""PrexSyn — a synthesis program sampled from a fingerprint, evaluated into a molecule.

PrexSyn (*Efficient and Programmable Exploration of Synthesizable Chemical Space*,
arXiv:2512.00384, https://github.com/luost26/prexsyn) is a decoder-only transformer
that autoregressively emits a **postfix notation of synthesis**: a stack program in
which a building-block token pushes a purchasable compound and a reaction token pops
its reactants and pushes the products, conditioned on an ECFP4 or FCFP4 fingerprint.
The molecule is whatever that program evaluates to, so the route is not reconstructed
after the fact — it is the thing that was generated, and the product is a by-product
of executing it. That is what ``synthesis_aware`` means here, and it is the whole
difference from the pocket-conditioned 3D generators in this package, which emit a
point cloud and then have to guess the bonds.

It conditions on a *descriptor*, never on a pocket. ``conditioning`` is therefore
``"ligand"``, and a caller that passes a :class:`PocketSpec` gets a warning on the
batch rather than a silently mislabelled pocket-conditioned result.

**Which PrexSyn this is.** ``.external/prexsyn`` is branch ``main`` at commit
3522e9747538, and ``prexsyn/__init__.py`` there declares ``__version__ = "1.1.3"``.
That is **v1**, not the paper's ``dev-v0`` branch, and the two are not interchangeable.
``docs/reproducibility/version-differences.md`` in the checkout lists composite
property queries (AND / OR / NOT) and property-conditioned generation as *deprecated*
in v1, and ``prexsyn/factory/descriptor_registry.py`` registers exactly two
descriptors, ``ecfp4`` and ``fcfp4``. Anything this adapter would produce therefore
belongs to the v1 projection / descriptor path, and comparing it against a paper table
that reports a property-query result would be comparing two different systems. The
checkpoint is not where the difference lies: the released config
``data/trained_models/enamine2310_rxn115_202511.yml`` describes its own weights as
"Model released in November 2025. This is the model used in the paper." The feature
set is.

**What it would compute here.** One driver call per ``generate``, returning records the
domain base screens and de-duplicates like every other generator's:

    ``smiles``        the product of one sampled synthesis program
    ``route``         that program as a nested dict — ``SMILES`` at every node,
                      ``BuildingBlock`` identifiers at the leaves, ``Reaction`` names
                      and ``Precursors`` at the internal nodes
                      (``prexsyn.utils.syndag.SynthesisDAG.to_dict``)
    ``similarity``    Tanimoto of the product's descriptor against the query's,
                      unitless in [0, 1], computed upstream in
                      ``prexsyn/utils/metrics.py``
    ``query_smiles``  which reference ligand this product was projected from

With ``context_smiles`` the mode is *chemical-space projection*: each reference is
fingerprinted, ``num_samples`` pathways are sampled from it, and the products are
ranked by similarity to the reference. Without ``context_smiles`` there is nothing to
condition on, so the driver instead draws sparse random ECFP4 genotypes exactly as
``prexsyn/shortcuts/genetic.py::initialize`` does and keeps the best product of each —
an unconditional draw from the model's prior over synthesizable space, and labelled as
such on the batch config.

**What it deliberately does not do.** It does not score, dock, rank against a target,
or claim the route works. A generated synthesis is a hypothesis about chemistry in
exactly the way a generated pose is a hypothesis about binding: the products leave
here as ``TrustLabel.PROPOSED`` and are judged by the same downstream pipeline as any
other candidate. It also does not run the property-conditioned or composite-query
generation described in the paper, because v1 does not implement them.

**Why it runs on this machine, and what stopped it on the last one.**

1. ``prexsyn_engine`` — the C++ extension that holds the chemical space, applies the
   reactions and detokenizes the sampled programs — installs from a wheel here.
   ``prexsyn`` 1.1.3 requires ``prexsyn-engine>=1.1.1,<1.2.0``, and every
   ``prexsyn_engine`` 1.1.1 distribution on PyPI is ``manylinux_2_27_x86_64``,
   ``macosx_15_0_arm64`` or ``win_amd64``. This host is Linux **x86-64**, so ``pip``
   takes ``prexsyn_engine-1.1.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl``
   and lands ``prexsyn_engine.cpython-312-x86_64-linux-gnu.so`` in site-packages.
   On the aarch64 host the same command answered ``ERROR: Could not find a version
   that satisfies the requirement prexsyn-engine==1.1.1 (from versions: 0.1.1.dev1
   ... 0.1.1.dev5)`` — five obsolete pure-Python pre-releases, not the engine.
   Upstream says as much in ``docs/getting-started/installation.md``: "The published
   PrexSyn Engine wheels currently target Linux x86-64."

2. The source build was therefore never needed here, and it is worth recording what
   it would have cost on a host without the wheel. ``prexsyn_engine-1.1.1.tar.gz``
   builds through ``scikit-build-core``; its ``CMakeLists.txt`` needs CMake ≥ 3.28, a
   C++20 compiler and OpenMP, and unless both ``USE_SYSTEM_BOOST`` and
   ``USE_SYSTEM_RDKIT`` are supplied it first runs ``third_party/build.py``, which
   builds Boost 1.86.0 and RDKit ``Release_2025_09_5`` from source, then fetches
   pybind11 3.0.2, spdlog 1.17.0, nlohmann/json 3.12.0 and csv-parser 2.5.1. Nothing
   in that chain is x86-only — the AVX2 flags are already guarded by a
   ``CMAKE_SYSTEM_PROCESSOR`` test — so the aarch64 route was a long build, not a
   dead end.

3. The weights were absent, because ``.external/prexsyn`` is a code-only shallow clone.
   Two files would have to exist under the checkout root, and this adapter's spec names
   both so the refusal points at them:
   ``.external/prexsyn/data/trained_models/enamine2310_rxn115_202511.ckpt`` and
   ``.external/prexsyn/data/chemical_spaces/enamine2310_rxn115.chemspace``. Upstream
   publishes both at https://huggingface.co/datasets/luost26/prexsyn-data and
   ``AllInOneLoader`` will fetch them from the ``remote:`` URLs already written into
   the released YAML on first use. They are a matched pair:
   ``docs/getting-started/data.md`` warns "Do not mix a checkpoint with another
   chemical-space cache", because building blocks and reactions are addressed by index.

**Measured here on 2026-09-09.** Eight molecules with no reference ligand, seed 7,
on ``cuda:0`` in the dedicated PrexSyn environment: 79.5 s wall, 16 raw samples, 16
RDKit-valid, 16 unique, 8 kept after the shared screen. Each carries the reaction
program PrexSyn actually executed — building-block ids, reaction ids and precursors
— because that is what the model emits. With no reference ligand the sampler draws
from sparse random ECFP4 genotypes, which the adapter records as a warning on the
batch: it is a draw from the model's prior over synthesizable space, not a
projection of anything.

**What the platform does if the backend goes away.** Nothing. Under
:mod:`drug_agent.tools.base` this capability reports UNAVAILABLE, ``generate`` raises
:class:`~drug_agent.tools.base.ToolUnavailable`, and the run proceeds with the
synthesis-aware generation arm visibly absent. No surrogate emits a molecule with a
plausible-looking route attached, because a route nobody ran is the most expensive kind
of fiction this package could produce: it survives into the portfolio as evidence that
a compound can be made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: Version of *this adapter's* contract with PrexSyn, not of PrexSyn's weights.
PREXSYN_ADAPTER_VERSION = "1.0.0"

#: The released model configuration shipped in the checkout, relative to its root.
#: Every upstream entry point defaults to it (``scripts/examples/projection.py``,
#: ``scripts/examples/sampling.py``, ``docs/getting-started/import.md``), and both the
#: checkpoint and the chemical-space cache are named *inside* it.
RELEASED_MODEL_CONFIG = "data/trained_models/enamine2310_rxn115_202511.yml"

#: ``MoleculeProjector``'s own default chunk size (``prexsyn/shortcuts/projector.py``).
#: Passed explicitly so the driver never has to guess a number upstream already chose.
PROJECTOR_BATCH_SIZE_LIMIT = 64

SPEC = ExternalSpec(
    name="PrexSyn",
    env_prefix="PREXSYN",
    checkout_relpath=".external/prexsyn",
    # Every one of these is on the run path: the package marker, the two shortcut
    # classes the driver constructs, the sampler the projector drives, the factory that
    # builds the model and detokenizer, and the released config the driver defaults to.
    sentinel_files=(
        "prexsyn/__init__.py",
        "prexsyn/shortcuts/loader.py",
        "prexsyn/shortcuts/projector.py",
        "prexsyn/samplers/basic.py",
        "prexsyn/factory/factory.py",
        RELEASED_MODEL_CONFIG,
    ),
    driver_name="prexsyn_sample.py",
    # The one import that identifies an environment PrexSyn was installed into. It is
    # also the one that cannot be installed here: see blocker 1 in the module docstring.
    selection_modules=("prexsyn_engine",),
    # Read off the import statements on the run path, not guessed. `prexsyn` itself is
    # supplied by the checkout on PYTHONPATH, so it is covered by `sentinel_files`
    # instead. `requests`/`tqdm` come in through `prexsyn/utils/download.py`, which
    # `shortcuts/loader.py` imports unconditionally; `matplotlib`/`PIL`/`pydot` come in
    # through `prexsyn/utils/draw.py`, which `shortcuts/projector.py` imports
    # unconditionally even when no image is ever drawn. `lightning` is deliberately
    # absent: it is a training-only import and losing it costs sampling nothing.
    runtime_modules=(
        "prexsyn_engine",
        "torch",
        "numpy",
        "rdkit",
        "omegaconf",
        "requests",
        "tqdm",
        "matplotlib",
        "PIL",
        "pydot",
    ),
    # A checkpoint and the exact chemical space it was tokenized against. Building
    # blocks and reactions are addressed by index, so the pair cannot be mixed.
    weight_relpaths=(
        "data/trained_models/enamine2310_rxn115_202511.ckpt",
        "data/chemical_spaces/enamine2310_rxn115.chemspace",
    ),
    # `prexsyn-engine` is a compiled wheel that cannot be installed beside this
    # project's own pins, so it lives in its own environment and `drug-agent` will
    # never have it. Naming only the environment it is really in keeps the failure
    # legible: if that env is absent the reason says so, rather than blaming
    # `drug-agent` for missing a package it was never meant to carry.
    conda_envs=("drug-agent-prexsyn",),
    upstream_url="https://github.com/luost26/prexsyn",
    upstream_ref="3522e9747538",
    license="PrexSyn MIT (LICENSE, Copyright (c) 2025 Shitong Luo)",
    # Deliberately terse: `ExternalBackend.unavailable_reason` repeats this hint once per
    # readiness failure, so the full wheel/source-build story lives in the module
    # docstring rather than four times over in one diagnostic line.
    install_hint=(
        "pip install 'prexsyn==1.1.3' into a dedicated env (prexsyn-engine ships wheels for "
        "linux x86-64, macos arm64 and windows only; elsewhere build it from the sdist), then "
        "fetch the weights from huggingface.co/datasets/luost26/prexsyn-data"
    ),
    # CPU inference is supported and documented; upstream's examples default to CUDA
    # and the installation notes call it "substantially slower" without one.
    needs_gpu=True,
)


class PrexSynAdapter(GenerativeToolAdapter):
    """Synthesizable molecules generated as reaction programs over purchasable blocks."""

    capability_id = "molecule.generate.prexsyn"
    spec = SPEC
    engine = "prexsyn"
    version = PREXSYN_ADAPTER_VERSION
    #: A fingerprint, not a binding site. Recorded on every batch so a descriptor-
    #: conditioned proposal can never be read back as a pocket-conditioned one.
    conditioning = "ligand"
    #: The route is the generated object, not an afterthought.
    synthesis_aware = True

    #: The descriptor the released checkpoint is conditioned on. ``ecfp4`` is what
    #: every upstream example passes; ``fcfp4`` is the only other registered choice.
    descriptor = "ecfp4"

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Synthesis-aware molecule generation (PrexSyn v1)",
            description=(
                "Autoregressive generation of postfix synthesis notations conditioned on an "
                "ECFP4/FCFP4 fingerprint: a stack program over purchasable building blocks and "
                "reaction templates, evaluated by PrexSyn Engine into a product molecule and the "
                "route that makes it. Projects reference ligands into synthesizable chemical "
                "space, or draws from the prior when no reference is given. Proposes; scores "
                "nothing; proves no route."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="Candidate",
            # Token types, building blocks and reactions are each drawn by
            # torch.multinomial over a softmax (prexsyn/samplers/basic.py), and the
            # unconditional path additionally draws random genotypes. Seeding torch and
            # numpy makes a run repeatable; it does not make the model deterministic.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "a sampled postfix program is incomplete or its reaction does not apply, so "
                "that sample yields no product at all — upstream states the number of products "
                "can differ from num_samples in either direction",
                "one program can push several products, so a batch skews toward the pathways "
                "with the most outcomes unless the per-query cap is applied",
                "projection returns the nearest reachable molecule, not the query: a target "
                "outside the Enamine/rxn115 chemical space comes back with low Tanimoto "
                "similarity rather than an error",
                "checkpoint and chemical-space cache are index-coupled; pairing a checkpoint "
                "with a differently-ordered building-block library silently decodes to the "
                "wrong molecules instead of failing",
                "the chemical-space cache is loaded whole before the first sample, so a run "
                "that would take seconds can be dominated by that load",
                "the unconditional path needs a boolean descriptor and upstream raises "
                "NotImplementedError for any other kind",
            ],
            validation_checks=[
                "every returned SMILES is re-parsed by RDKit here; validity and uniqueness are "
                "computed by the domain base from raw counts, never reported by the backend",
                "the checkpoint and the chemical-space cache named in the released YAML must "
                "both exist before the capability reports AVAILABLE",
                "the requested descriptor must be one the loaded config actually registers",
                "similarity is carried through as the backend computed it and is never "
                "converted into a score, a rank or a probability",
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
        """Run the PrexSyn driver once and hand its records to the domain base."""
        warnings: list[str] = []
        if pocket:
            warnings.append(
                "PrexSyn conditions on a molecular descriptor, not a binding site; the pocket "
                "was recorded on the batch config and had no effect on what was sampled."
            )
        request = {
            "config_path": str(self._model_config_path()),
            "context_smiles": list(context_smiles),
            "n_molecules": int(n_molecules),
            # `batch_size` in this package means "samples per model call"; in PrexSyn the
            # equivalent knob is how many pathways are sampled per conditioning
            # descriptor, which is what `MoleculeProjector(num_samples=...)` takes.
            "num_samples": max(1, int(batch_size)),
            "batch_size_limit": PROJECTOR_BATCH_SIZE_LIMIT,
            "descriptor": self.descriptor,
            "device": self.ext.device,
            "seed": int(seed),
        }
        # Not `run_driver`: it hard-codes `self.timeout_s`, and `generate` lets a caller
        # set `max_runtime_seconds`. A budget that is advertised and then not enforced is
        # worse than no budget, so the caller's number is the one that reaches the child.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            seed=seed,
        )
        payload = run.payload
        records = [dict(item) for item in payload.get("molecules") or [] if isinstance(item, dict)]
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        errors = [str(e) for e in payload.get("errors") or []]
        if not records and not errors:
            errors.append(
                "PrexSyn ran and returned no product: every sampled postfix program was "
                "incomplete or its reactions did not apply."
            )
        return records, warnings, errors

    def _model_config_path(self) -> Path:
        """The released model YAML inside the checkout.

        Both the checkpoint and the chemical-space cache are named *inside* this file,
        so it is the single thing the driver has to be told; the spec's
        ``weight_relpaths`` check that what it names is really there.
        """
        root = self.ext.root
        return (root / RELEASED_MODEL_CONFIG) if root is not None else Path(RELEASED_MODEL_CONFIG)


__all__ = [
    "PREXSYN_ADAPTER_VERSION",
    "PROJECTOR_BATCH_SIZE_LIMIT",
    "RELEASED_MODEL_CONFIG",
    "SPEC",
    "PrexSynAdapter",
]
