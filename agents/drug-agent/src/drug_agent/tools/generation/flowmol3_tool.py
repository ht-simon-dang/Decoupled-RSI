"""FlowMol3 — unconditional all-atom 3D molecule generation, used here as a chemical prior.

FlowMol3 samples atom positions, atom types, formal charges and bond orders jointly
by flow matching, integrating a learned vector field over a fixed number of Euler
steps and then reading a molecule out of the result. It conditions on **nothing**.
There is no pocket argument, no reference ligand, no property target. What it
produces is a draw from the model's approximation of GEOM-Drugs chemistry.

**Why the catalogue keeps it anyway.** It is the control. Every pocket-conditioned
generator in this package is judged on molecules it emitted *for a pocket*, and the
first question about any such number is what the same pipeline would report for
molecules generated with no pocket at all. FlowMol3 answers that question and
nothing else. It is not a substitute for FLOWR, FLOWR.root, PFM or any other
pocket-conditioned model, and scoring it on a pocket-conditioned benchmark —
docking its output into a site it never saw, then comparing that to a conditioned
model's output for the same site — measures nothing. ``conditioning`` is therefore
``"none"`` and is written onto every batch config, so a downstream reader cannot
mistake one for the other. If a pocket is passed to :meth:`generate` it is recorded
as a warning on the batch and discarded, because silently accepting it would be the
same lie one indirection further away.

**The reconstruction hazard applies here too.** FlowMol3 emits a point cloud with
predicted bond orders, and a graph is rebuilt from it; ``build_molecule`` in
``flowmol/analysis/molecule_builder.py`` assembles an RWMol and does not sanitize
it. Bond perception over generated coordinates produces chemistry that has nothing
to do with how well the model learned its training distribution. The measured PFM
baseline in this repository is the reason this warning is on every generator here:
of 40 generated molecules, 21 contained an allene and 21 a four-membered ring,
leaving 2 chemically ordinary molecules, and all 40 fell outside the activity
model's applicability domain. Before believing anything about a FlowMol3 batch,
read ``generation_validity`` on the batch and the reconstruction audit this adapter
attaches to ``warnings`` — the count of cumulated double bonds (allene centres) and
of three- and four-membered rings among the samples that parsed. A model whose
validity is high and whose allene rate is also high has not generated drug-like
chemistry; it has generated a point cloud that RDKit was willing to accept.

**Which FlowMol this is.** The checkout at ``.external/FlowMol``, pinned at
``77cae22174b7``, is FlowMol3 and not an earlier release. Three independent things
in it say so: ``setup.cfg`` declares ``version = 3.1.0``; ``readme.md`` states "The
current version is FlowMol3" and points FlowMol1 and FlowMol2 users at the ``v0.1``
and ``v2.0`` git tags, which are not what is checked out; and
``flowmol/__init__.py`` lists exactly one default pretrained model, ``flowmol3``,
alongside twenty-one ``fm3_*`` ablations, downloading them from
``trained_models_v3.1/``. The upstream preprint is arXiv:2508.12629.

**This capability is AVAILABLE on this machine.** ADR-0013 filed FlowMol3 as the
one *platform*-blocked generator in the catalogue — the only verdict of its kind —
and x86-64 retracted it. What that took is worth stating exactly, because two of the
three original blockers were real and the third was a misdiagnosis.

1. **The weights were a download, and it has been run.**
   ``.external/FlowMol/flowmol/trained_models/`` now holds ``flowmol3/``. The
   checkpoint the run path wants is
   ``flowmol/trained_models/flowmol3/checkpoints/last.ckpt`` — the path is built
   identically in ``flowmol/__init__.py`` (``load_pretrained``) and in ``test.py``
   line 76. Nothing in the clone can produce it: ``.gitattributes`` routes ``*.ckpt``
   through git-lfs and no checkpoint is tracked, so upstream distributes it over HTTP
   instead, from
   ``https://bits.csb.pitt.edu/files/FlowMol/trained_models_v3.1/flowmol3/``
   (``flowmol/trained_models/readme.md``, and ``download_remote_model_dir`` in
   ``flowmol/__init__.py``, which shells out to ``wget -r``), roughly 65 MB.

2. **DGL installs here, and this is the blocker the architecture removed.**
   Every module on the sampling path imports it: ``flowmol/models/flowmol.py:6``,
   ``flowmol/models/vector_field.py:3`` and
   ``flowmol/analysis/molecule_builder.py:4``. Upstream installs it with
   ``mamba install -c dglteam/label/cu121 dgl=2.0.0.cu121=py310_0``
   (``build_env.sh``), and that channel builds ``linux-64`` only — which is why the
   aarch64 host had nothing but the CPU-only PyPI wheel
   ``dgl-2.1.0-cp312-cp312-manylinux2014_aarch64.whl``, whose bundled
   ``libgraphbolt_pytorch_<torch version>.so`` exists for torch 2.0.0-2.2.1 and for
   nothing else. On x86-64 the real DGL builds install.

3. **The GPU was not blocked by DGL. It was blocked by torch, and the first
   diagnosis was wrong.** With DGL installed, the first kernel raised ``CUDA error:
   no kernel image is available for execution on the device``. Because DGL was the
   pinned, unusual dependency — and because no prebuilt DGL exists above cu124 — DGL
   looked like the binding constraint and this tool was pinned to CPU. That was
   wrong. ``libdgl.so`` does not link ``libtorch`` at all; DGL reaches torch through
   DLPack, so DGL's CUDA tag never had to match the card. The real cause is that
   ``torch 2.6.0+cu124``'s ``libtorch_cuda.so`` carries 2,389 SASS cubins topping out
   at sm_90 and **zero PTX**, so an sm_120 card has nothing to run and nothing to JIT.
   ``libdgl.so`` from stock ``dgl 2.5.0+cu124``, by contrast, carries 336 cubins *and*
   42 PTX modules. Installing ``torch 2.7.1+cu128`` — whose arch list ends
   ``sm_120, compute_120`` — makes real DGL CUDA kernels run against **DGL at its own
   original pin**. ADR-0014 records the ``cuobjdump`` mistake that hid this.

4. **Two more imports are missing from the ``drug-agent`` interpreter**, which is why
   FlowMol3 does not run in it. ``flowmol/models/flowmol.py:21`` does ``from einops
   import rearrange``, and ``flowmol/analysis/metrics.py:8`` does ``import wandb`` —
   on the sampling path whether or not metrics are wanted, because ``flowmol.py``
   imports ``SampleAnalyzer`` from that module at import time. FlowMol3 therefore runs
   in its own environment, named through ``external_tools.flowmol3.python`` in the
   active profile; ``configs/local.yaml`` pins it here, with ``device: cuda:0``.

**Measured here on 2026-09-09.** Eight molecules at the default 250 timesteps,
seed 1234, through this adapter: **12.0 s and 12.8 s on ``cuda:0``** across two warm
runs (20.5 s on the first, cold, call), against **204.5 s, 249.6 s and 340.6 s on
``cpu``** across three runs. That is a speedup of at least 16x. The CPU spread is not
noise in the model — the CPU sampler saturates every core, so its wall time tracks
whatever else the machine is doing, while the GPU number does not move. All five runs
returned ``n_raw`` 8, ``n_valid`` 8, 8 kept.

**What the platform does if the backend goes away: nothing.** ``generate()`` calls
``require_available()`` and raises :class:`~drug_agent.tools.base.ToolUnavailable`.
No unconditional prior is substituted, no cached sample set is replayed, and no
control arm is fabricated for the pocket-conditioned models to be compared against.
A run that wanted this control does without it, visibly.

**What it would report if the backend were here.** One
:class:`~drug_agent.tools.pfm_tool.GenerationBatch`, screened by the shared
generator base rather than by this module:

    ``molecules``                  de-duplicated, RDKit-parseable samples, each
                                   ``TrustLabel.PROPOSED``, carrying the generated
                                   3D block in ``sdf_block``
    ``n_raw`` / ``n_valid``        samples returned, and samples RDKit parsed
    ``n_unique``                   distinct canonical SMILES among the valid ones
    ``generation_validity``        ``n_valid / n_raw``, dimensionless
    ``generation_uniqueness``      ``n_unique / n_valid``, dimensionless
    ``runtime_seconds``            wall time of the whole call
    ``warnings``                   the reconstruction audit, the fragment count,
                                   and any discarded conditioning

"Valid" means what it means in the FlowMol3 paper: the driver reproduces
``SampleAnalyzer.compute_validity`` (``flowmol/analysis/metrics.py:170``) —
fragment split, largest fragment, ``SanitizeMol``, ``MolToSmiles`` — so a
disconnected or unsanitizable sample is returned with an empty SMILES and counts
against validity rather than vanishing from the denominator.

**What it deliberately does not do.** It does not dock, score, filter on
drug-likeness, or condition on anything. It has no notion of the brief's target. A
FlowMol3 molecule is a proposal about chemistry and becomes evidence only after the
same activity, safety, structure and synthesis pipeline that judges every other
candidate has judged it.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with FlowMol3, not FlowMol3's own version.
FLOWMOL3_ADAPTER_VERSION = "1.0.0"

#: Upstream's own default, from ``FlowMol.default_n_timesteps`` and ``test.py``'s
#: ``--n_timesteps`` argparse default. Fewer steps is the first thing to try when a
#: run is too slow, and the first thing to suspect when validity collapses.
DEFAULT_N_TIMESTEPS = 250

#: The pretrained model ``flowmol/__init__.py`` loads by default. The other twenty-one
#: names in ``pretrained_model_names`` are ``fm3_*`` ablations of this one.
DEFAULT_MODEL_NAME = "flowmol3"

SPEC = ExternalSpec(
    name="FlowMol3",
    env_prefix="FLOWMOL3",
    checkout_relpath=".external/FlowMol",
    # The first three are modules the sampling path imports outright. The fourth is
    # not a module and is load-bearing anyway: FlowMol's constructor `torch.load`s
    # the training-set atom-count histogram to build its size distribution, and
    # raises FileNotFoundError at flowmol/models/flowmol.py:100 without it. A clone
    # with the code but not the processed-data stubs would otherwise look complete.
    sentinel_files=(
        "flowmol/__init__.py",
        "flowmol/models/flowmol.py",
        "flowmol/analysis/molecule_builder.py",
        "data/geom_full_kekulized/train_data_n_atoms_histogram.pt",
    ),
    driver_name="flowmol3_sample.py",
    # dgl identifies a FlowMol environment: nothing else in this catalogue uses it,
    # and every module on the sampling path imports it.
    selection_modules=("dgl",),
    # What a broken install actually loses, read off the import statements rather
    # than guessed. wandb and posebusters are here because flowmol/models/flowmol.py
    # imports SampleAnalyzer from flowmol/analysis/metrics.py at module scope, so
    # `import flowmol` drags both in even for a run that computes no metrics.
    # torch_ema is deliberately absent: only train.py imports it.
    runtime_modules=(
        "torch",
        "dgl",
        "pytorch_lightning",
        "einops",
        "torch_scatter",
        "rdkit",
        "wandb",
        "posebusters",
        "useful_rdkit_utils",
        "pandas",
        "scipy",
        "yaml",
    ),
    weight_relpaths=("flowmol/trained_models/flowmol3/checkpoints/last.ckpt",),
    # Measured on this host, not guessed. `drug-agent-flowmol3` imports every module
    # above and then dies at the first kernel launch — its torch 2.6.0+cu124 carries
    # SASS only to sm_90 and this card is sm_120 — while `drug-agent-flowmol3-gpu`
    # (torch 2.7.1+cu128) runs real DGL `update_all` SpMM kernels on it. Both are kept:
    # the broken one is not removed by hand, it loses the device probe in
    # `ExternalBackend._resolve_python`, which is what makes the rule work on a host
    # where the plain env is the good one.
    conda_envs=("drug-agent-flowmol3-gpu", "drug-agent-flowmol3"),
    upstream_url="https://github.com/Dunni3/FlowMol",
    upstream_ref="77cae22174b7",
    license="FlowMol MIT (Copyright (c) 2023 University of Pittsburgh); LICENSE present in the clone",
    install_hint=(
        "make external-clone && bash .external/FlowMol/build_env.sh && wget -r -np -nH "
        "--cut-dirs=3 --reject 'index.html*' -P .external/FlowMol/flowmol/trained_models/ "
        "https://bits.csb.pitt.edu/files/FlowMol/trained_models_v3.1/flowmol3/"
    ),
    needs_gpu=True,
)

#: A carbon carrying two double bonds — the allene/cumulene centre that bond
#: perception over a generated point cloud produces at a rate unrelated to how well
#: the model learned its training distribution.
_ALLENE_SMARTS = "*=[#6X2]=*"


def _reconstruction_audit(records: list[dict[str, Any]]) -> list[str]:
    """Count allene centres and strained small rings among the parseable samples.

    A real RDKit measurement over the batch that was actually generated, reported
    as a warning rather than as a per-molecule ``qc_flag``: a flag would make the
    shared screening step *drop* those molecules, and hiding the failure mode is
    the opposite of what this audit is for. The reader is meant to see the rate.
    """
    allene = Chem.MolFromSmarts(_ALLENE_SMARTS)
    parsed = 0
    n_allene = 0
    n_small_ring = 0
    for item in records:
        smiles = item.get("smiles") or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue
        parsed += 1
        if allene is not None and mol.HasSubstructMatch(allene):
            n_allene += 1
        if any(len(ring) <= 4 for ring in mol.GetRingInfo().AtomRings()):
            n_small_ring += 1
    if not parsed:
        return []
    return [
        f"reconstruction audit over {parsed} parseable samples: {n_allene} contain a cumulated "
        f"double bond (allene centre), {n_small_ring} contain a three- or four-membered ring"
    ]


class FlowMol3Adapter(GenerativeToolAdapter):
    """Unconditional all-atom 3D generation with FlowMol3 — the catalogue's control arm."""

    capability_id = "molecule.generate.flowmol3"
    spec = SPEC
    engine = "flowmol3"
    version = FLOWMOL3_ADAPTER_VERSION
    #: Not "pocket". FlowMol3 has no pocket input; see the module docstring.
    conditioning = "none"
    synthesis_aware = False

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        n_timesteps: int = DEFAULT_N_TIMESTEPS,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s)
        self.n_timesteps = int(n_timesteps)
        self.model_name = str(model_name)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Unconditional 3D molecule generation (FlowMol3)",
            description=(
                "Flow matching over atom positions, atom types, formal charges and bond orders, "
                "sampled with no target of any kind. A chemical prior and the control arm for the "
                "pocket-conditioned generators — never a substitute for one."
            ),
            # No input contract: an unconditional model has no input. A PocketSpec
            # handed to generate() is recorded as a warning and discarded.
            input_schema_ref=None,
            output_schema_ref="Candidate",
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no checkpoint: flowmol/trained_models/flowmol3/checkpoints/last.ckpt is not in the "
                "clone and is fetched over HTTP from bits.csb.pitt.edu (~65 MB)",
                "torch older than 2.7 in the FlowMol3 environment: its libtorch_cuda.so stops at "
                "sm_90 SASS and carries no PTX, so on an sm_120 card DGL's kernels raise 'no kernel "
                "image is available for execution on the device' — torch's gap, not DGL's",
                "dgl has no CUDA build for aarch64 in any channel, so on that architecture the "
                "sampler is CPU-only and 250 integration steps are impractically slow",
                "molecule_builder.build_molecule returns an unsanitized RWMol, and None outright when "
                "RDKit cannot kekulize the generated bond assignment",
                "bond perception over generated coordinates yields allenes and three- or four-membered "
                "rings at a rate unrelated to model quality",
                "samples come back as several disconnected fragments; only the largest is kept, as "
                "upstream's own validity metric does",
                "the checkpoint's recorded processed_data_dir does not resolve here, and FlowMol raises "
                "FileNotFoundError for train_data_n_atoms_histogram.pt (flowmol/models/flowmol.py:100)",
                "explicit hydrogens are a generated atom type, so a sample can be 'valid' and still be "
                "chemically absurd once the hydrogens are read",
            ],
            validation_checks=[
                "conditioning is reported as 'none' on every batch config",
                "a pocket or context SMILES passed in is recorded as a warning, never used",
                "validity reproduces SampleAnalyzer.compute_validity: fragment split, largest "
                "fragment, SanitizeMol, MolToSmiles",
                "a sample that fails reconstruction is returned with an empty SMILES and counts "
                "against validity rather than leaving the denominator",
                "every batch carries an allene and small-ring count over the samples that parsed",
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
        warnings: list[str] = []
        if pocket:
            warnings.append(
                "FlowMol3 is unconditional: the pocket "
                f"{pocket.get('receptor_id') or pocket.get('pdb_id') or '(unnamed)'} was discarded and "
                "these molecules were generated with no target. Do not compare them to a "
                "pocket-conditioned model's output for that pocket."
            )
        if context_smiles:
            warnings.append(
                f"FlowMol3 takes no reference ligand: {len(context_smiles)} context SMILES were discarded."
            )

        request = {
            "flowmol_root": str(self.ext.root) if self.ext.root else None,
            "model_dir": str(self.ext.root / "flowmol" / "trained_models" / self.model_name)
            if self.ext.root
            else self.model_name,
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "n_timesteps": self.n_timesteps,
            "seed": int(seed),
            "device": self.ext.device,
        }
        # run_json rather than run_driver: the base hands _sample the caller's
        # per-call budget, and run_driver would silently substitute self.timeout_s.
        run = self.ext.run_json(
            request,
            timeout_s=timeout_s,
            error_cls=GenerationBackendError,
            seed=seed,
        )
        payload = run.payload

        records = [r for r in (payload.get("molecules") or []) if isinstance(r, dict)]
        warnings.extend(str(w) for w in (payload.get("warnings") or []))
        warnings.extend(_reconstruction_audit(records))
        errors = [str(e) for e in (payload.get("errors") or [])]

        n_sampled = int(payload.get("n_sampled") or 0)
        if n_sampled < int(n_molecules):
            warnings.append(f"FlowMol3 returned {n_sampled} samples for a request of {n_molecules}.")
        return records, warnings, errors


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_N_TIMESTEPS",
    "FLOWMOL3_ADAPTER_VERSION",
    "SPEC",
    "FlowMol3Adapter",
]
