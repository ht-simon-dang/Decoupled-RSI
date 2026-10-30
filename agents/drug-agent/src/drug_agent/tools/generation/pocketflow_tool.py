"""PocketFlow — autoregressive pocket-conditioned generation with valence rules in the loop.

PocketFlow grows a ligand inside a binding site one atom at a time. Each step picks
a focal atom, samples an element from a normalising flow, places it in 3D, then
samples its bonds from a second flow — and *checks the result before keeping it*.
That last clause is the architecture's whole claim, and it is worth stating
precisely because it separates PocketFlow from most of this package's
pocket-conditioned catalogue.

**Why it is in the catalogue.** The authors report prospective, wet-lab validated
results: applied to HAT1 and YTHDC1, two epigenetic-regulation targets, PocketFlow
produced bioactive compounds whose binding modes were confirmed by an X-ray
crystal structure (upstream ``README.md``; Jiang et al., *Nature Machine
Intelligence*, 2024, doi:10.1038/s42256-024-00808-8). Almost nothing else here has
a prospective result at all, and that is a real reason to include an older model
beside newer architectures. It is equally not a property this platform inherits.
That evidence is the authors', on the authors' targets, from the authors' pipeline
of generation plus docking plus synthesis plus assay. Running this checkpoint
against a different pocket produces hypotheses, exactly like every other generator
in this package, and they become evidence only after the same RDKit, activity,
safety and synthesis pipeline that judges every other candidate has judged them.

**How it differs from the point-cloud generators, and where that stops helping.**
The dominant failure of pocket-conditioned 3D generation in this repository is not
chemistry, it is *reconstruction*: the model emits atoms and coordinates, and a
bond perceiver guesses the graph afterwards. The measured PFM baseline here is the
warning — of 40 generated molecules, 21 contained an allene and 21 a four-membered
ring, leaving 2 chemically ordinary molecules, and all 40 fell outside the activity
model's applicability domain. PocketFlow's design attacks exactly that step:

* bonds are *sampled*, not perceived. ``pocket_flow/utils/generate_utils.py``
  ``data2mol`` builds the RDKit molecule from ``ligand_context_bond_index`` and
  ``ligand_context_bond_type`` — the model's own bond table — never from
  interatomic distances;
* every proposed bond set is valence-checked with RDKit ``SANITIZE_PROPERTIES``
  (``check_valency``) and resampled up to 50 times before the atom is abandoned
  (``pocket_flow/generate.py`` lines 184-237);
* five alert SMARTS are rejected outright at each step —
  ``[O]-[O]``, ``[N]-[O,Br,Cl,I,F,P]``, ``[S,P]-[Br,Cl,I,F]``, ``[P]-[O]-[P]``,
  ``[Br,Cl,I,F]-[Br,Cl,I,F]`` (line 311);
* bond lengths are held inside a window (default 1.0-2.0 Å) and the first atom is
  kept at least ``min_dist_inter_mol`` (default 3.0 Å) off the protein;
* a post-hoc ``modify`` pass demotes excess double bonds in six-rings.

None of that is a reason to trust a batch without looking at it. The rules
constrain local valence; they do not constrain the strained macrocycles, spiro
knots and fused small rings that a step-by-step 3D grower can still assemble, and
they say nothing at all about whether the molecule is in any activity model's
applicability domain — the failure that disqualified all 40 PFM samples. **Read
``generation_validity`` and ``generation_uniqueness`` on every batch, and count
allenes and four-membered rings before believing anything about this model.** The
base class computes validity and uniqueness from real counts for exactly this
reason, and a generator is never permitted to report its own.

**Status here: UNAVAILABLE.** Not for the usual reason. Unlike every other
generative clone in ``.external``, PocketFlow **ships its weights**:
``ckpt/ZINC-pretrained-255000.pt`` and ``ckpt/PubChem-pretrained-315000.pt``, both
3,371,778 bytes, are in the checkout. They are real. Loaded on this machine with
``torch.load(..., weights_only=False)``, the ZINC checkpoint yields the keys
``config``, ``model``, ``optimizer``, ``scheduler``, ``iteration``;
``PocketFlow(ckpt["config"])`` builds from it and ``load_state_dict`` accepts every
tensor, giving a 215,406-parameter model. The blockers are all environmental, and
all three were reproduced here:

1. **``tensorboard`` is not installed.** ``pocket_flow/utils/train.py`` line 11
   runs ``from torch.utils import tensorboard`` at module scope, and
   ``pocket_flow/__init__.py`` line 1 reaches it through ``from .utils import *``.
   The package cannot be imported at all — the generation path included — without
   a training-only dependency. Of the environments on this machine, only
   ``drug-agent-unimol`` has tensorboard, and it has neither ``torch_geometric``
   nor ``torch_scatter`` nor ``torch_cluster``.
2. **``torch_geometric.utils.subgraph`` no longer exists.**
   ``pocket_flow/utils/transform_utils.py`` line 5 imports it. PyG moved that
   module to ``torch_geometric.utils._subgraph`` in 2.5; this machine has PyG
   2.8.0.post1 and upstream's ``environment.yml`` pins ``torch-geometric==2.3.1``.
3. **``pyg-lib`` is absent, and the reason recorded here was wrong about
   architecture.** PyG 2.8 routes ``knn``, ``knn_graph``, ``radius`` and
   ``radius_graph`` through pyg-lib and raises ``ImportError("... requires
   'pyg-lib>=0.6.0'")`` for each when ``torch_geometric.typing.WITH_KNN`` and
   ``WITH_RADIUS`` are false — both are false here. PocketFlow calls all four:
   ``pocket_flow/generate.py`` line 2 imports ``knn`` and ``radius``, and
   ``pocket_flow/utils/transform_utils.py`` lines 242 and 256 call ``knn_graph`` and
   ``radius_graph``.

   The earlier note said pyg-lib "publishes no aarch64 build at any version". Checked
   again on 2026-09-09, that is no longer true and the qualifier matters. ``pip
   download pyg-lib`` from PyPI still reports ``No matching distribution found for
   pyg-lib (from versions: none)`` on **any** architecture — pyg-lib is not on PyPI at
   all. It is published per-torch-version at ``data.pyg.org/whl/``, and those indexes
   split: ``torch-2.4.0+cu121``, ``torch-2.6.0+cu124``, ``torch-2.7.0+cu128`` and
   ``torch-2.8.0+cu128`` list ``linux_x86_64`` and ``win_amd64`` only, zero aarch64
   entries; but ``torch-2.14.0+cu130`` — the index matching this environment's torch —
   lists ``pyg_lib-0.9.0+pt214cu130-cp310-abi3`` for manylinux **x86_64 and aarch64**
   both. So the wheel exists for this environment on either architecture and simply is
   not installed. Upstream's own ``environment.yml`` line 296 still pins the old
   x86-64-only artifact by URL: ``pyg_lib-0.2.0+pt113cu117-cp38-cp38-linux_x86_64.whl``.

   None of that changes what this adapter does. The driver reroutes the four geometry
   ops to ``torch_cluster``, which is installed and working, so PocketFlow does not
   need pyg-lib on any host — which is why ``pyg_lib`` is not in ``runtime_modules``.

A fourth constraint is not a blocker but decides how this tool must be configured
if the three above are ever cleared: **PocketFlow is CUDA-only.** ``Generate.run``
builds its atom and bond priors with ``torch.zeros([...]).cuda(...)``
unconditionally (``pocket_flow/generate.py`` lines 246-253), so the ``-d cpu``
option ``main_generate.py`` advertises raises inside the sampler. The spec
therefore declares ``needs_gpu=True`` and the driver refuses a CPU device up
front, with that line number in the message.

**How far it actually got.** In a scratch diagnostic — not a product run, and not
shipped — the three import failures above were bypassed one at a time to find out
whether anything deeper was wrong. The package imported, the ZINC checkpoint
loaded and built the 215,406-parameter model, and upstream's own pocket
``test_samples/test_pocket10/1bvr_C_rec_pocket10-surf.pdb`` parsed to 540 heavy
atoms of which 147 were flagged as surface. Sampling then stopped at the first
``knn_graph`` call with the pyg-lib ``ImportError``. So blocker 3 is the live one,
and no molecule has been generated by this adapter on this machine. No validity
number, uniqueness number or runtime for PocketFlow appears anywhere in this
module, because none has been measured here.

**Pocket file format.** PocketFlow does not read an ordinary pocket PDB. Its
parser takes each atom's surface flag from a trailing ``inner``/``surf`` token
appended to the ATOM record (``pocket_flow/utils/ParseFile.py`` line 61), a
non-standard column written by upstream's
``SplitPocket._split_pocket_with_surface_atoms``. The prepared pocket in this
repository, ``.external/pockets/seh_3i28_pocket10.pdb``, does not carry it. Such a
file still parses — every atom simply reads as buried — and with the default
``choose_max=True`` the focal atom is the argmax and the mask is unused, so
generation proceeds unguided by the surface. The driver emits a warning saying so
rather than failing, because an all-buried mask is a degradation of the
conditioning, not a malfunction. Note that producing a properly annotated pocket
*does* need pymol, which upstream imports in a bare ``try``/``except`` and which is
absent here; nothing on the generation path itself imports it, so pymol is
deliberately not among this spec's ``runtime_modules``.

**What the platform does instead.** Nothing. ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` through ``require_available()``,
naming the three blockers above. There is no surrogate, no fallback sampler and no
degraded mode: a fabricated molecule would enter the pipeline indistinguishable
from a real proposal and would be scored, ranked and reported as one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with PocketFlow, not PocketFlow's own version.
POCKETFLOW_ADAPTER_VERSION = "1.0.0"

#: The two checkpoints the checkout ships, relative to its root. ZINC is what
#: upstream's README invokes; PubChem is the alternative pretraining corpus. Both
#: are 3,371,778 bytes and both are real — see the module docstring.
ZINC_CHECKPOINT = "ckpt/ZINC-pretrained-255000.pt"
PUBCHEM_CHECKPOINT = "ckpt/PubChem-pretrained-315000.pt"

SPEC = ExternalSpec(
    name="PocketFlow",
    env_prefix="POCKETFLOW",
    checkout_relpath=".external/PocketFlow",
    # Files the run path really imports, all present in the checkout: the entry
    # point the driver reproduces, the package root, the sampler, the model, the
    # PDB parser and the transform stack.
    sentinel_files=(
        "main_generate.py",
        "pocket_flow/__init__.py",
        "pocket_flow/generate.py",
        "pocket_flow/gdbp_model/pocket_flow.py",
        "pocket_flow/utils/ParseFile.py",
        "pocket_flow/utils/transform.py",
    ),
    driver_name="pocketflow_sample.py",
    # Narrow: which environment PocketFlow's stack was installed into.
    selection_modules=("torch", "torch_geometric"),
    # Everything the run path loses when the install is half-built. `torch` alone
    # reports AVAILABLE and then crashes at sample time, which is the whole reason
    # this field is separate from selection.
    #
    #   torch_scatter   pocket_flow/gdbp_model/layers.py, net_utils.py
    #   torch_cluster   what PyG 2.3.1 dispatches knn/radius to
    #   pyg_lib         what PyG >= 2.5 dispatches them to instead; ABSENT here,
    #                   and upstream's environment.yml pins an x86-64-only wheel
    #   scipy           pocket_flow/utils/process_raw.py (pdist, squareform)
    #   lmdb            pocket_flow/utils/load_dataset.py, imported by utils/__init__
    #   easydict        gdbp_model, and the pickled `config` inside the checkpoint
    #   tensorboard     pocket_flow/utils/train.py line 11; ABSENT here
    #   tqdm            pocket_flow/utils/train.py, process_raw.py
    #
    # pymol is deliberately NOT here: upstream imports it in a bare try/except and
    # only SplitPocket needs it. Nothing on the generation path does.
    runtime_modules=(
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_cluster",
        # NOT pyg_lib. Upstream's environment.yml pins a linux_x86_64-only wheel, and
        # pyg-lib is not on PyPI on any architecture, so requiring it would make this
        # capability UNAVAILABLE on a host that has not gone to data.pyg.org — for a
        # dependency the model does not actually need. The driver reroutes
        # knn/radius/knn_graph/radius_graph back to torch_cluster, which is what
        # they were in the PyG release PocketFlow was published against — the same
        # shim pocket2mol_sample.py applies. torch_cluster is probed above.
        "rdkit",
        "numpy",
        "scipy",
        "lmdb",
        "easydict",
        "tensorboard",
        "tqdm",
    ),
    # Unlike every other generative clone here, the weights ship with the code.
    # This probe passes; the module docstring records that it was also loaded.
    weight_relpaths=(ZINC_CHECKPOINT,),
    conda_envs=("drug-agent", "pfm"),
    upstream_url="https://github.com/Saoge123/PocketFlow",
    upstream_ref="a31a5a03f7d1",
    license="PocketFlow MIT (Liscense in the checkout root; weights ship with it)",
    install_hint=(
        "conda run -n drug-agent pip install tensorboard, plus a torch-geometric/pyg-lib pair "
        "that provides knn and radius (upstream pins torch-geometric==2.3.1 and an "
        "x86-64-only pyg_lib wheel; pyg-lib is not on PyPI at all, it is published "
        "per-torch-version at data.pyg.org/whl/)"
    ),
    needs_gpu=True,
)


class PocketFlowSamplingError(GenerationBackendError):
    """PocketFlow was installed, sampling started, and it produced nothing usable."""


class PocketFlowAdapter(GenerativeToolAdapter):
    """Autoregressive, valence-checked de-novo generation inside a binding pocket."""

    capability_id = "molecule.generate.pocketflow"
    spec = SPEC
    engine = "pocketflow"
    version = POCKETFLOW_ADAPTER_VERSION
    conditioning = "pocket"
    synthesis_aware = False

    #: Which shipped checkpoint the driver loads. Upstream's README invokes the
    #: ZINC one; ``PUBCHEM_CHECKPOINT`` is the alternative pretraining corpus.
    default_checkpoint = ZINC_CHECKPOINT

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s)
        # A string only. `tool_pool()` builds every adapter on every call, so the
        # constructor stays free of filesystem and network work; whether this file
        # exists is `weight_relpaths`' question, answered once by ExternalBackend.
        self.checkpoint = checkpoint or self.default_checkpoint

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pocket-conditioned de-novo generation (PocketFlow)",
            description=(
                "Grows a ligand inside a binding site one atom at a time: a focal atom is "
                "chosen, an element is sampled from a normalising flow, placed in 3D, and its "
                "bonds sampled from a second flow, with RDKit valence checking and five alert "
                "SMARTS applied at every step. Bonds are sampled rather than perceived from "
                "distances. Returns proposals, never evidence."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            # Two normalising flows sampled per atom, plus multinomial focal and
            # position choices. A seed makes a run reproducible; it does not make
            # the model deterministic.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "pyg-lib absent, so knn/knn_graph/radius/radius_graph all raise before sampling "
                "starts (PyG >= 2.5 routes them through it; it is not on PyPI, only on "
                "data.pyg.org/whl/<torch version>), which is why the driver reroutes to "
                "torch_cluster instead",
                "tensorboard absent, so `import pocket_flow` fails on a training-only dependency",
                "torch_geometric.utils.subgraph moved to ._subgraph in PyG 2.5, which "
                "pocket_flow/utils/transform_utils.py line 5 imports by the old path",
                "CPU device raises inside Generate.run, which calls .cuda() unconditionally",
                "pocket PDB without the trailing inner/surf column leaves every atom marked "
                "buried, so generation proceeds unguided by the protein surface",
                "a sample can terminate with no molecule at all: the focal search finds no atom "
                "with spare valence, or 50 bond resamples fail — counted, not raised",
                "long autoregressive chains can close strained rings that per-step valence "
                "checking does not forbid",
            ],
            validation_checks=[
                "validity and uniqueness are computed by the base class from raw sample counts, "
                "never reported by the backend",
                "failed samples are returned with an empty smiles and a qc_flag, so the validity "
                "denominator is the number attempted",
                "every molecule leaves as TrustLabel.PROPOSED and is screened by the same "
                "pipeline as every other candidate",
                "the checkpoint is loaded from the checkout's own ckpt/ directory, whose files "
                "are verified by weight_relpaths before the tool reports AVAILABLE",
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
        """Run the driver in PocketFlow's own interpreter and return its raw samples.

        Screening, de-duplication and the validity/uniqueness arithmetic all belong
        to :class:`GenerativeToolAdapter`; this returns records exactly as the
        backend produced them, failures included.
        """
        pocket_pdb = pocket.get("pdb_path")
        if not pocket_pdb or not Path(pocket_pdb).is_file():
            raise PocketFlowSamplingError(
                f"PocketFlow is pocket-conditioned and needs a prepared pocket PDB; "
                f"generation.pocket.pdb_path={pocket_pdb!r} is missing or unreadable. "
                "It also wants the trailing inner/surf column upstream's SplitPocket writes — "
                "without it every atom reads as buried and the surface conditioning is lost."
            )
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() has already run
            raise PocketFlowSamplingError("PocketFlow checkout could not be resolved")

        warnings: list[str] = []
        if context_smiles:
            # Stated rather than silently dropped: a caller who passed seeds is
            # entitled to know they had no effect on what came back.
            warnings.append(
                f"PocketFlow is pocket-conditioned only; {len(context_smiles)} context SMILES "
                "were ignored. Use a ligand-conditioned generator to grow from a seed."
            )
        if batch_size and batch_size > 1:
            # Generate.generate loops one molecule at a time (generate.py line 356)
            # and there is no batched entry point. Saying so beats letting an
            # operator tune a parameter that does nothing.
            warnings.append(
                f"batch_size={batch_size} has no effect: PocketFlow samples autoregressively, "
                "one molecule per call, with no batched entry point."
            )

        request = {
            "pocket_pdb": str(Path(pocket_pdb).resolve()),
            "checkpoint": str(root / self.checkpoint),
            "n_molecules": int(n_molecules),
            "device": self.ext.device,
            "seed": int(seed),
            # Upstream's README invocation, which differs from the argparse
            # defaults in main_generate.py for max_atom_num (35 vs 40).
            "max_atom_num": 35,
            "atom_temperature": 1.0,
            "bond_temperature": 1.0,
            "focus_threshold": 0.5,
            "choose_max": True,
            "min_dist_inter_mol": 3.0,
            "bond_length_range": [1.0, 2.0],
            "max_double_in_6ring": 0,
        }
        run = self.run_driver(request, error_cls=PocketFlowSamplingError, timeout_s=timeout_s)
        payload = run.payload

        records = list(payload.get("molecules") or [])
        warnings.extend(str(w) for w in payload.get("warnings") or [])
        failures = int(payload.get("n_generation_failures") or 0)
        if failures:
            warnings.append(
                f"{failures} of {n_molecules} samples terminated without a molecule — the focal "
                "search ran out of atoms with spare valence, or bond resampling hit its limit. "
                "They are returned with an empty smiles so validity counts them."
            )
        return records, warnings, [str(e) for e in payload.get("errors") or []]


__all__ = [
    "POCKETFLOW_ADAPTER_VERSION",
    "PUBCHEM_CHECKPOINT",
    "SPEC",
    "ZINC_CHECKPOINT",
    "PocketFlowAdapter",
    "PocketFlowSamplingError",
]
