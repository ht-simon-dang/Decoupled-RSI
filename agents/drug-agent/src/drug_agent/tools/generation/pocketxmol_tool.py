"""PocketXMol — one checkpoint, thirteen pocket tasks, and the label that keeps them apart.

PocketXMol (Peng et al., *Unified modeling of 3D molecular generation via atomic
interactions with PocketXMol*, Cell 2026, doi:10.1016/j.cell.2026.01.003) is not a
de-novo generator that also docks. It is a single denoiser over atoms — positions,
types and half-edges — whose *task* is supplied at inference time as a noise
schedule and a partition of the molecule into a part that is given and a part that
is generated. De-novo design, fragment linking, fragment growing, ligand
optimisation, docking and conformer generation are the same weights told different
things about which atoms are already known.

**The task is the result's identity, not a setting.** Everything this tool returns
is a SMILES in a batch, and three artifacts downstream a SMILES is just a string. A
linking run returns the two fragments you supplied joined by a generated linker; a
docking run returns the molecule you supplied, unchanged, with new coordinates; an
SBDD run returns a molecule nobody chose. Read as "generated candidates" they are
indistinguishable and two of the three are a lie. So:

* the task is fixed when the adapter is *constructed*, not per call, and an unknown
  or unsupported name is refused with the vocabulary listed rather than defaulted;
* ``batch.config`` carries ``pocketxmol_task``, ``pocketxmol_task_family`` (the
  upstream ``task.name``) and ``pocketxmol_base_config`` (the upstream config file
  the run was built from), so the provenance is the upstream file itself;
* every molecule's ``origin`` is ``pocketxmol.<task>``, not ``pocketxmol``, because
  ``origin`` is the field that survives onto a candidate record.

**The task vocabulary, as the checkout spells it.** Three vocabularies exist in
``.external/PocketXMol`` and they are not the same list.

``task.name`` — the family, consumed by ``utils/dataset.py::UseDataset`` and by the
noiser factory ``utils/sample_noise.py::get_sample_noiser``: ``dock``, ``sbdd``,
``maskfill``, ``pepdesign``, ``conf``, ``denovo``, ``fbdd``, ``custom``, plus the
training-only ``mixed`` and ``dynamic``.

``task.transform.name`` — the registry in ``utils/transforms.py``: ``dock``,
``sbdd``, ``denovo``, ``conf``, ``maskfill``, ``fbdd``, ``pepdesign``, ``linking``,
``growing``, ``ar``, ``ar2``, ``custom``, ``mixed``.

``configs/sample/examples/*.yml`` — the 27 worked configurations that are the
user-facing catalogue, and the vocabulary this adapter binds, because a config is
where upstream actually says what a task means down to its noise schedule.

**The thirteen tasks exposed here**, each the upstream example config named, with
the inputs it really needs:

    ``sbdd``               pocket + centre. De-novo design, auto-regressive
                           refinement (``sbdd.yml``, transform ``ar``).
    ``sbdd_simple``        pocket + centre. De-novo, one shot, no refinement rounds
                           (``sbdd_simple.yml``, transform ``sbdd``).
    ``dock``               pocket + centre + a ligand (SDF *or* SMILES). Pose
                           prediction under isotropic Gaussian noise
                           (``dock_smallmol.yml``).
    ``dock_flexible``      as ``dock``, noise decomposed into translation, rotation
                           and torsion (``dock_smallmol_flex.yml``).
    ``linking``            pocket + a 3D SDF + fragment atom indices. Fragments held
                           exactly at their input poses (``linking_fixed_frags.yml``).
    ``linking_unfixed``    as ``linking``, fragment poses unknown and drawn from the
                           prior (``linking_unfixed_frags_unknown.yml``).
    ``linking_from_input`` as ``linking_unfixed``, but the input poses seed the
                           trajectory (``linking_unfixed_frags_frominput.yml``).
    ``linking_approx``     as ``linking_from_input`` with the part1 noise floor
                           raised from 0.0 to 0.8, i.e. input poses treated as
                           approximate (``linking_unfixed_frags_approx.yml``).
    ``linking_redesign``   as ``linking``, with fragment atom *types* also perturbed
                           (``part1_pert.small``) so the fragments may change
                           (``linking_redesign_frags.yml``).
    ``growing``            pocket + a 3D fragment SDF + its atom indices. Grow from a
                           fixed fragment (``growing_fixed_frag.yml``).
    ``growing_unfixed``    as ``growing``, fragment pose unknown
                           (``growing_unfixed_frag_unknown.yml``).
    ``optimize``           pocket + a 3D SDF. Redesign the whole ligand in place
                           (``opt_mol.yml``).
    ``optimize_partial``   pocket + a 3D SDF + the indices of the part to keep.
                           Redesign the rest (``opt_partial.yml``).

Supplying ``anchor_atom_indices`` alongside fragments switches
``settings.known_anchor`` from ``none`` to ``all`` and writes ``grouped_anchor_p1``,
which is exactly the difference between ``linking_fixed_frags.yml`` and
``linking_fixed_frags_connecting.yml``; it is a parameter here rather than a
fourteenth task.

**What is deliberately not exposed.** Twelve of the 27 example configs are peptide
work — ``dock_pep``, ``dock_pep_fix_some``, ``dock_pep_know_some`` and the nine
``pepdesign_*`` — and upstream routes them through ``is_pep=True``, which writes a
PDB and reconstructs with ``reconstruct_pdb_from_generated``. A peptide is not a
small molecule and its SMILES is a polymer string that the standardisation, activity
and safety pipeline downstream of this package would silently mis-handle, so they
are refused by name rather than half-supported. ``conf`` (conformer generation) and
``denovo`` (unconditional 3D generation) are refused for a different, concrete
reason: their only configs live under ``configs/sample/test/`` and read
``data.dataset.dbs[].lmdb_path``, so they run through ``scripts/sample_drug3d.py``
against the processed GEOM/CSD LMDBs from upstream's ``data_test.tar.gz``, not
against user data. Neither refusal is a judgement about the model.

**The reconstruction hazard, which applies to every model in this family.**
PocketXMol emits a point cloud with half-edge logits and a perceiver rebuilds the
graph (``utils/reconstruct.py::reconstruct_from_generated_with_edges``, openbabel
underneath). The failure mode is therefore not chemistry, it is graph
reconstruction, and it is quantitatively large. The measured baseline in this
repository is PFM's: of 40 generated molecules, 21 contained an allene and 21 a
four-membered ring, leaving 2 chemically ordinary molecules, and all 40 fell outside
the activity model's applicability domain. Before believing anything a run of this
tool says about the model, read, in this order:

1. ``generation_validity`` — what fraction of samples RDKit could parse at all;
2. ``pocketxmol_allene_rate`` and ``pocketxmol_small_ring_rate`` — computed here with
   RDKit over the molecules that survived screening, and metered on every call. A
   double-digit rate in either is reconstruction noise, not medicinal chemistry;
3. ``generation_uniqueness`` — a collapse means the sampler, not the pocket.

Upstream's own confidence numbers ride along in the driver payload
(``cfd_traj``, ``cfd_pos``, ``cfd_node``, ``cfd_edge`` from ``gen_info.csv``). They
are the model grading itself and are not a substitute for either metric above.

**This capability is AVAILABLE on this machine, and was UNAVAILABLE when the text
below it was written.** The section that used to stand here said the weights were
absent and that one pip package was missing. Both were true then and are false now;
it is rewritten rather than deleted because a reader who trusts a stale "UNAVAILABLE"
banner writes off the best pocket-conditioned arm in the study.

Verified on this host, 2026-09-18:

* ``data/trained_models/pxm/checkpoints/pocketxmol.ckpt`` is present — 243,496,215
  bytes, ``sha256:76d918122e598a3c53c77727898659f059d217e680834b7d223b0205c2715c3a``
  — and its sibling ``data/trained_models/pxm/train_config/`` is present too, which
  ``scripts/sample_use.py`` needs to rebuild the model geometry.
* ``PeptideBuilder`` imports in the ``drug-agent`` environment.
* :meth:`available` returns ``True`` and the adapter generates for real: 60 raw
  samples in 119.4 s on ``cuda:0``, about 2.0 s per molecule, at 96.7% validity and
  100% uniqueness.
* The version skew flagged below is real but was not a blocker: the checkpoint's
  ``state_dict`` loads under this environment's torch 2.14.0+cu130 despite upstream
  pinning torch 2.0.0. That was genuinely unknowable before the weights existed, and
  it is now measured.

**What this arm is worth, measured.** Scored by the frozen evaluator against the
frozen seven-gate qualification at the registered 150-molecule submission budget,
PocketXMol returns **47 qualifying scaffolds per 150**, against the campaign parent
REINVENT4's **4**. It passes the ``structure_score`` docking gate at 71.3% where
REINVENT4 passes at 5.3%, and it carries 150 distinct Bemis-Murcko scaffolds in 150
molecules. Its rejections are dominated by a lipophilicity limit (``logP <= 5.0``),
not by binding, and molecular weight never breaches at all.

**What is still true from the original text, and worth keeping.** Upstream's pins
are not this environment's — ``environment.yml`` builds Python 3.8 with torch 2.0.0,
PyG 2.3, numpy 1.24 and rdkit 2023.9.3, while this environment is Python 3.12.14 with
torch 2.14.0+cu130, numpy 2.5.2 and rdkit 2026.03.5. Nothing here silently
substitutes: if the adapter cannot run, ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable` and the absence is visible on the
capability card, because sampling from some other generator and labelling it
PocketXMol — or labelling a docking pose as a de-novo candidate — is precisely the
failure the task-on-the-config discipline above exists to prevent. The adapter also
refuses a ``checkpoint=`` argument outright rather than quietly sampling the default
weights under a request that asked for adapted ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import note
from .._ext.spec import ExternalSpec
from ..pfm_tool import GenerationBatch
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with PocketXMol, not PocketXMol's own version.
POCKETXMOL_ADAPTER_VERSION = "1.0.0"

SPEC = ExternalSpec(
    name="PocketXMol",
    env_prefix="POCKETXMOL",
    checkout_relpath=".external/PocketXMol",
    # Files the run path really touches: the entry point the driver launches, the
    # denoiser it builds, the two factories that turn a config into a task, the
    # module that parses user structures, and the model config carrying the
    # checkpoint path. A checkout missing any of them cannot sample.
    sentinel_files=(
        "scripts/sample_use.py",
        "models/maskfill.py",
        "models/sample.py",
        "utils/sample_noise.py",
        "utils/transforms.py",
        "process/utils_process.py",
        "configs/sample/pxm.yml",
    ),
    driver_name="pocketxmol_sample.py",
    # Selection identifies the environment PocketXMol was installed into — upstream's
    # environment.yml builds a PyG stack (pyg 2.3 + torch-scatter) named `pxm`.
    # Readiness below then rejects a half-built one module by module, which is how
    # the missing PeptideBuilder becomes an upfront reason rather than a crash in
    # `import process.utils_process`.
    selection_modules=("torch_geometric", "torch_scatter"),
    # Every module the run path imports at module scope, traced from
    # scripts/sample_use.py outwards. PeptideBuilder leads because it is the one
    # absent here, and it is absent for a reason worth reading: process/utils_process
    # imports it unconditionally, even for small-molecule tasks.
    runtime_modules=(
        "PeptideBuilder",  # process/utils_process.py lines 20-21
        "torch",
        "torch_geometric",  # scripts/sample_use.py, utils/transforms.py
        "torch_scatter",  # utils/sample_noise.py line 6
        "pytorch_lightning",  # scripts/train_pl.py, imported for DataModule
        "easydict",  # config objects throughout
        "lmdb",  # utils/dataset.py line 8, at module scope
        "openbabel",  # utils/reconstruct.py line 31, bond perception
        "networkx",  # utils/graph.py, fragment decomposition
        "Bio",  # process/utils_process.py, utils/reconstruct.py
        "rdkit",
    ),
    # The one file that decides whether this tool can run at all. Its sibling
    # data/trained_models/pxm/train_config/ ships in the same archive and is listed
    # by sample_use.py line 179; the driver checks for it before launching.
    weight_relpaths=("data/trained_models/pxm/checkpoints/pocketxmol.ckpt",),
    conda_envs=("pxm", "pxm_cu128", "drug-agent"),
    upstream_url="https://github.com/pengxingang/PocketXMol",
    upstream_ref="65488cf635c8",
    license="MIT",
    install_hint=(
        "make external-clone, then in .external/PocketXMol fetch the weights per docs/setup.md "
        "section 2 — 'wget -c https://zenodo.org/records/17801271/files/model_weights.tar.gz' "
        "then 'tar -zxvf model_weights.tar.gz' — and 'pip install peptidebuilder==1.1.0' into "
        "the interpreter that will run it (or 'conda env create -f environment.yml' for "
        "upstream's own pxm environment)"
    ),
    needs_gpu=True,
)

#: Upstream's default model config: two lines, the second of which is the
#: checkpoint path. Overridable per adapter for a differently-tuned checkpoint
#: (``configs/sample/pxm_use.yml`` is the other one shipped).
DEFAULT_MODEL_CONFIG = "configs/sample/pxm.yml"


@dataclass(frozen=True)
class PocketXMolTask:
    """One PocketXMol task: the upstream config that defines it, and its inputs."""

    #: The name this adapter is constructed with.
    name: str
    #: Upstream config, relative to the checkout, that the driver starts from.
    config_relpath: str
    #: Upstream's own ``task.name`` — the family the noiser and dataset see.
    family: str
    #: One line, in the terms of the thing being produced.
    summary: str
    #: Whether the task generates around an input molecule.
    needs_ligand: bool = False
    #: Whether that input molecule must carry 3D coordinates (an SDF), as opposed
    #: to being expressible as a SMILES.
    needs_3d_ligand: bool = False
    #: Whether the task partitions the input molecule into given and generated atoms.
    needs_fragments: bool = False
    #: Whether the upstream config carries an explicit denoising centre. When it
    #: does, this adapter demands one: the example coordinates belong to upstream's
    #: protein, and inheriting them would build the molecule in the wrong place with
    #: no error at all.
    needs_center: bool = False


#: The small-molecule tasks reachable with user data through
#: ``scripts/sample_use.py``. Every entry names a config that exists in the checkout
#: at ``65488cf635c8``; the driver refuses if one is missing.
_TASK_LIST: tuple[PocketXMolTask, ...] = (
    PocketXMolTask(
        "sbdd",
        "configs/sample/examples/sbdd.yml",
        "sbdd",
        "de-novo design in the pocket, auto-regressive refinement",
        needs_center=True,
    ),
    PocketXMolTask(
        "sbdd_simple",
        "configs/sample/examples/sbdd_simple.yml",
        "sbdd",
        "de-novo design in the pocket, one shot, no refinement rounds",
        needs_center=True,
    ),
    PocketXMolTask(
        "dock",
        "configs/sample/examples/dock_smallmol.yml",
        "dock",
        "pose prediction for a supplied ligand, isotropic Gaussian noise",
        needs_ligand=True,
        needs_center=True,
    ),
    PocketXMolTask(
        "dock_flexible",
        "configs/sample/examples/dock_smallmol_flex.yml",
        "dock",
        "pose prediction, noise split into translation, rotation and torsion",
        needs_ligand=True,
        needs_center=True,
    ),
    PocketXMolTask(
        "linking",
        "configs/sample/examples/linking_fixed_frags.yml",
        "maskfill",
        "link fragments held exactly at their input poses",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "linking_unfixed",
        "configs/sample/examples/linking_unfixed_frags_unknown.yml",
        "maskfill",
        "link fragments whose poses are unknown and drawn from the prior",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "linking_from_input",
        "configs/sample/examples/linking_unfixed_frags_frominput.yml",
        "maskfill",
        "link fragments free to move, seeded from their input poses",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "linking_approx",
        "configs/sample/examples/linking_unfixed_frags_approx.yml",
        "maskfill",
        "link fragments whose input poses are treated as approximate only",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "linking_redesign",
        "configs/sample/examples/linking_redesign_frags.yml",
        "maskfill",
        "link fragments whose atom types may also change",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "growing",
        "configs/sample/examples/growing_fixed_frag.yml",
        "maskfill",
        "grow a molecule from a fragment held at its input pose",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "growing_unfixed",
        "configs/sample/examples/growing_unfixed_frag_unknown.yml",
        "maskfill",
        "grow from a fragment graph whose pose is unknown",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
    PocketXMolTask(
        "optimize",
        "configs/sample/examples/opt_mol.yml",
        "sbdd",
        "redesign a whole ligand in its own pocket, seeded from its structure",
        needs_ligand=True,
        needs_3d_ligand=True,
    ),
    PocketXMolTask(
        "optimize_partial",
        "configs/sample/examples/opt_partial.yml",
        "maskfill",
        "redesign part of a ligand, keeping the atoms named as fragments",
        needs_ligand=True,
        needs_3d_ligand=True,
        needs_fragments=True,
    ),
)

#: The same list, keyed by name, which is how the adapter resolves one.
TASKS: dict[str, PocketXMolTask] = {task.name: task for task in _TASK_LIST}

#: Tasks that exist upstream and are refused here, with the reason. Enumerated
#: rather than omitted, so "can PocketXMol do X" is answerable from this file.
UNSUPPORTED_TASKS: dict[str, str] = {
    "pepdesign": (
        "peptide design (9 pepdesign_* example configs: denovo, cyclic_denovo, fix_pos, "
        "fix_type, fix_pos_and_type, invfold, invfold_bbflex, sc_pack, opt). Upstream runs "
        "them with is_pep=True, writes a PDB and reconstructs with "
        "reconstruct_pdb_from_generated; a peptide is not a small molecule and this "
        "package's downstream standardisation, activity and safety pipeline would "
        "mis-handle its SMILES"
    ),
    "dock_pep": (
        "peptide docking (dock_pep, dock_pep_fix_some, dock_pep_know_some). Same is_pep "
        "path and the same reason as pepdesign"
    ),
    "conf": (
        "molecular conformer generation. Its only config, "
        "configs/sample/test/conf_geom/base.yml, reads data.dataset.dbs[].lmdb_path and "
        "runs through scripts/sample_drug3d.py against the processed GEOM LMDBs from "
        "upstream's data_test.tar.gz — it is a benchmark path, not a user-data one"
    ),
    "denovo": (
        "unconditional 3D generation with no pocket. Same benchmark-only situation as "
        "conf (configs/sample/test/denovo_geom/base.yml)"
    ),
    "custom": (
        "upstream's escape hatch for hand-written noise groups and partitions "
        "(docs/sample_provided_data.md section 4). It has no example config, so there is "
        "nothing to start from that upstream has validated"
    ),
}

#: SMARTS for a cumulated double bond. Bond perception over a generated point cloud
#: produces these at a rate that has nothing to do with what the model learned; the
#: PFM baseline in this repository found 21 in 40 samples.
_ALLENE_SMARTS = "[*]=[#6]=[*]"

#: Rings at or below this size are the other half of that same signal. Three- and
#: four-membered rings are real chemistry in small numbers and reconstruction noise
#: in large ones, which is why the *rate* is what gets metered, not their presence.
_SMALL_RING_MAX_SIZE = 4


def resolve_task(name: str) -> PocketXMolTask:
    """The task called *name*, or a refusal that names the whole vocabulary.

    Refusing here rather than at sample time is the point: a task chosen wrongly is
    not an error downstream, it is a correct-looking batch of the wrong kind of
    molecule.
    """
    key = (name or "").strip().lower()
    if key in TASKS:
        return TASKS[key]
    for prefix, reason in UNSUPPORTED_TASKS.items():
        if key == prefix or key.startswith(prefix + "_"):
            raise ValueError(
                f"PocketXMol task {name!r} is not exposed by this adapter: {reason}. "
                f"Available: {', '.join(sorted(TASKS))}."
            )
    raise ValueError(
        f"unknown PocketXMol task {name!r}. Available: {', '.join(sorted(TASKS))}. "
        f"Present upstream but refused here: {', '.join(sorted(UNSUPPORTED_TASKS))}."
    )


class PocketXMolAdapter(GenerativeToolAdapter):
    """Atom-level generation, linking, growing, optimisation and docking in one pocket."""

    capability_id = "molecule.generate.pocketxmol"
    spec = SPEC
    engine = "pocketxmol"
    version = POCKETXMOL_ADAPTER_VERSION
    #: Every exposed task conditions on a binding site. Recorded on the batch config
    #: so this can never be read as ligand-only or unconditional generation.
    conditioning = "pocket"
    #: Samples atoms, not reactions. A molecule from here arrives with no route.
    synthesis_aware = False

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """PocketXMol's contract is per-TASK, not per-model.

        The two de-novo tasks this adapter can reach (``sbdd``, ``sbdd_simple``)
        both denoise inside an explicit box, so both need a centre on top of the
        pocket file. Reporting that here is what lets the master agent see, before
        it commits a cycle, that choosing this backend costs it a pocket centre.
        """
        out = ["pocket_pdb"]
        if self.task.needs_center:
            out.append("pocket_center")
        if self.task.needs_ligand:
            out.append("input_ligand")
        if self.task.needs_fragments:
            out.append("fragment_atom_indices")
        return tuple(out)

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        task: str = "sbdd",
        input_ligand: str | Path | None = None,
        fragment_atom_indices: list[list[int]] | None = None,
        anchor_atom_indices: list[list[int]] | None = None,
        mol_size_mean: float | None = None,
        mol_size_std: float | None = None,
        model_config: str = DEFAULT_MODEL_CONFIG,
    ) -> None:
        """Bind one task. The task is fixed here, not per call, and never defaulted silently.

        ``input_ligand`` is the molecule the task generates around: an SDF path for
        linking, growing and optimisation, an SDF path or a SMILES for docking,
        absent for de-novo design. ``fragment_atom_indices`` are the 0-based atom
        indices, grouped per fragment, that the maskfill tasks hold as *given* —
        upstream's ``preset_partition.grouped_node_p1``. ``anchor_atom_indices``
        restricts which of those atoms may form the new bond.
        """
        self.task = resolve_task(task)
        self.input_ligand = str(input_ligand) if input_ligand else None
        self.fragment_atom_indices = [list(g) for g in (fragment_atom_indices or [])]
        self.anchor_atom_indices = [list(g) for g in (anchor_atom_indices or [])]
        self.mol_size_mean = mol_size_mean
        self.mol_size_std = mol_size_std
        self.model_config = model_config
        # The driver payload carries provenance (the effective seed, the checkpoint
        # actually loaded, upstream's own tag counts) that belongs on the batch, and
        # `_sample` cannot reach the batch. It is stashed here for the length of one
        # `_generate`, which clears it first so a failed call cannot leave a previous
        # run's provenance attached to the next one.
        self._last_run: dict[str, Any] | None = None
        super().__init__(settings=settings, timeout_s=timeout_s)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title=f"Pocket-conditioned atom generation, task '{self.task.name}' (PocketXMol)",
            description=(
                "One denoiser over atom positions, types and half-edges, conditioned on a "
                "binding site. The task is supplied as a noise schedule and a partition of "
                f"the molecule, and this adapter is bound to '{self.task.name}': "
                f"{self.task.summary}. Thirteen small-molecule tasks are reachable "
                "(de-novo design, five linking variants, two growing variants, two "
                "optimisation variants, two docking variants); the task is recorded on the "
                "batch and on every molecule so a linking result can never be read as "
                "de-novo generation."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            # A denoising trajectory from a random prior. Upstream folds the output
            # directory and the config filename into the seed (sample_use.py line
            # 173), so a fixed seed does not by itself fix a run.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "reconstruction, not chemistry, is the dominant failure: a bond perceiver "
                "rebuilds the graph from a generated point cloud and invents allenes and "
                "four-membered rings — check pocketxmol_allene_rate and "
                "pocketxmol_small_ring_rate before reading anything else",
                "a sample whose reconstructed SMILES contains '.' is tagged 'incomp' upstream "
                "and arrives flagged pocketxmol_disconnected_fragments, not dropped silently",
                "a sample that raises MolReconsError is tagged 'bad', arrives with an empty "
                "SMILES and the raw generated point cloud as its SDF, and counts against "
                "validity rather than vanishing from the denominator",
                "the denoising centre is separate from the pocket-extraction centre; the "
                "upstream example configs hard-code their own protein's coordinates, so a "
                "task run without an explicit centre has that key removed by the driver "
                "rather than inherited",
                "fragment atom indices are 0-based indices into the input SDF as RDKit reads "
                "it — wrong indices produce a chemically plausible molecule built around the "
                "wrong atoms, with no error",
                "sample_use.py folds the output directory path and the config filename into "
                "the seed, so the same nominal seed is not the same run; the effective value "
                "is returned as pocketxmol_complete_seed",
                "GPU out of memory at large batch_size — upstream's own configs use 50-100 "
                "and its README suggests lowering it first",
                "peptide tasks and the benchmark-only conf/denovo tasks are refused by name "
                "at construction rather than run and mislabelled",
            ],
            validation_checks=[
                "the task is resolved against the upstream config catalogue at construction; "
                "an unknown or unsupported name raises with the vocabulary listed",
                "the task, its upstream family and the upstream config file are written onto "
                "batch.config, and every molecule's origin is 'pocketxmol.<task>'",
                "tasks that need a ligand, 3D coordinates, fragment indices or an explicit "
                "denoising centre refuse without them instead of falling back to the example "
                "config's own inputs",
                "SMILES are re-parsed by RDKit in the domain base, which counts validity and "
                "uniqueness itself — the model does not report its own",
                "allene and small-ring (three- or four-membered) rates are computed here with "
                "RDKit over the screened molecules and metered on every call",
                "the driver verifies the checkpoint and its sibling train_config directory "
                "exist before launching, so a missing weight is a message, not a stack trace",
            ],
        )

    # --- the real backend --------------------------------------------------
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
        """Run this adapter's task once and return upstream's samples, failures included."""
        root = self.ext.root
        if root is None:  # require_available() has already covered this
            raise GenerationBackendError("PocketXMol checkout disappeared between probe and call")

        warnings: list[str] = []
        protein_path = pocket.get("pdb_path")
        # Every refusal below is a GenerationBackendError, not a bare ValueError.
        # `DesignAgent.generate_batch` catches (GenerationError,
        # GenerationBackendError, ToolUnavailable) to degrade the de-novo arm
        # rather than kill the run; a ValueError walked straight past that except
        # clause and out through `source_candidates`, which is how a missing
        # contract field took down three whole cycles instead of one arm.
        if not protein_path:
            raise GenerationBackendError(
                f"{self.capability_id} is pocket-conditioned: PocketSpec.pdb_path must name a "
                "protein or pocket PDB. sample_use.py cuts its own pocket from it, so a whole "
                "structure is acceptable when a centre and radius are given."
            )
        center = pocket.get("center")
        if self.task.needs_center and not center:
            raise GenerationBackendError(
                f"task {self.task.name!r} needs PocketSpec.center: its upstream config "
                f"({self.task.config_relpath}) carries an explicit denoising centre belonging "
                "to upstream's own protein, and this adapter removes rather than inherits it."
            )

        input_ligand = self._resolve_input_ligand(context_smiles, warnings)
        if self.task.needs_fragments and not self.fragment_atom_indices:
            raise GenerationBackendError(
                f"task {self.task.name!r} partitions the input molecule and needs "
                "fragment_atom_indices — 0-based atom indices, grouped per fragment, naming "
                "the atoms that are given rather than generated."
            )
        if self.anchor_atom_indices and not self.fragment_atom_indices:
            raise GenerationBackendError(
                "anchor_atom_indices restricts which fragment atoms may bond and is "
                "meaningless without fragment_atom_indices."
            )

        request = {
            "root": str(root),
            "task": self.task.name,
            "base_config": self.task.config_relpath,
            "config_model": self.model_config,
            "protein_path": str(protein_path),
            "input_ligand": input_ligand,
            "pocket_coord": list(center) if center else None,
            "ref_ligand_path": None,
            "pocket_radius": float(pocket.get("radius") or 10.0),
            "center": list(center) if center else None,
            "fragment_groups": self.fragment_atom_indices or None,
            "anchor_groups": self.anchor_atom_indices or None,
            "mol_size_mean": self.mol_size_mean,
            "mol_size_std": self.mol_size_std,
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device": self.ext.device,
            "data_id": str(pocket.get("receptor_id") or pocket.get("pdb_id") or "drugagent"),
            "pdbid": str(pocket.get("pdb_id") or ""),
            "timeout_seconds": float(timeout_s),
        }
        run = self.run_driver(request, error_cls=GenerationBackendError, seed=seed, timeout_s=timeout_s)
        payload = run.payload
        self._last_run = payload

        records = [r for r in (payload.get("molecules") or []) if isinstance(r, dict)]
        warnings.extend(str(w) for w in (payload.get("warnings") or []))
        errors = [str(e) for e in (payload.get("errors") or [])]

        n_bad = int(payload.get("n_reconstruction_failures") or 0)
        n_incomplete = int(payload.get("n_incomplete") or 0)
        if n_bad or n_incomplete:
            warnings.append(
                f"PocketXMol reconstruction: {n_bad} of {len(records)} samples raised "
                f"MolReconsError and {n_incomplete} came back as disconnected fragments."
            )
        note(
            self.capability_id,
            "generate",
            f"PocketXMol task {self.task.name!r} drew {len(records)} samples in "
            f"{run.seconds:.1f}s of driver wall time",
        )
        return records, warnings, errors

    def _resolve_input_ligand(self, context_smiles: list[str], warnings: list[str]) -> str | None:
        """The molecule this task generates around, from the constructor or the call.

        ``context_smiles`` is the domain base's channel for a reference molecule, and
        upstream's ``data.input_ligand`` accepts a SMILES for docking, so the two
        line up for that task and only that task. Everything else needs coordinates.
        """
        ligand = self.input_ligand
        if ligand is None and context_smiles:
            if self.task.needs_3d_ligand:
                raise GenerationBackendError(
                    f"task {self.task.name!r} needs 3D coordinates for its input molecule, so "
                    "context_smiles cannot supply it. Construct the adapter with "
                    "input_ligand=<path to an SDF>."
                )
            if not self.task.needs_ligand:
                warnings.append(
                    f"task {self.task.name!r} generates de novo and ignored the "
                    f"{len(context_smiles)} molecule(s) in context_smiles."
                )
                return None
            if len(context_smiles) > 1:
                warnings.append(
                    f"PocketXMol takes one input molecule per run; {len(context_smiles)} were "
                    "given and only the first was used."
                )
            ligand = context_smiles[0]
        if ligand is None and self.task.needs_ligand:
            raise GenerationBackendError(
                f"task {self.task.name!r} ({self.task.summary}) generates around an input "
                "molecule and has nothing to work from. Pass input_ligand=<path to an SDF>"
                + ("" if self.task.needs_3d_ligand else ", or a SMILES in context_smiles")
                + "."
            )
        if ligand is not None and not self.task.needs_ligand:
            warnings.append(
                f"task {self.task.name!r} generates de novo; the input_ligand given at "
                "construction was not used as a starting structure."
            )
            return None
        if ligand is not None and self.task.needs_3d_ligand and not str(ligand).endswith(".sdf"):
            raise GenerationBackendError(
                f"task {self.task.name!r} needs an SDF with coordinates; got {ligand!r}. "
                "Upstream's get_pocmol_data reads a .sdf or .pdb path and otherwise treats "
                "the string as a SMILES, which has no conformer to hold fixed."
            )
        return ligand

    # --- task provenance and the reconstruction signal ---------------------
    def _generate(self, metered: dict[str, float], **kwargs: Any) -> GenerationBatch:
        """The base's generation, plus the two things that make a batch readable later.

        The task goes onto the batch config *and* onto every molecule's ``origin``,
        because ``origin`` is the field that survives onto a candidate record while a
        config dict does not. The allene and four-membered-ring rates are metered
        because a collapse into either is how a bad reconstruction, a wrong pocket or
        a mismatched checkpoint announces itself, and none of them show up in
        validity.
        """
        self._last_run = None
        batch = super()._generate(metered, **kwargs)
        batch.config.update(
            {
                "pocketxmol_task": self.task.name,
                "pocketxmol_task_family": self.task.family,
                "pocketxmol_base_config": self.task.config_relpath,
                "pocketxmol_model_config": self.model_config,
                "pocketxmol_input_ligand": self.input_ligand,
                "pocketxmol_fragment_groups": self.fragment_atom_indices or None,
            }
        )
        payload = self._last_run
        if isinstance(payload, dict):
            for key in ("complete_seed", "checkpoint", "log_dir"):
                value = payload.get(key)
                if value is not None:
                    batch.config[f"pocketxmol_{key}"] = value
            for key in ("n_success", "n_incomplete", "n_reconstruction_failures"):
                value = payload.get(key)
                if value is not None:
                    metered[f"pocketxmol_{key}"] = float(value)
        origin = f"{self.engine}.{self.task.name}"
        for molecule in batch.molecules:
            molecule.origin = origin
        metered.update(self._reconstruction_hazards(batch))
        return batch

    def _reconstruction_hazards(self, batch: GenerationBatch) -> dict[str, float]:
        """Allene and small-ring rates over the molecules that survived screening.

        Both are computed here with RDKit, over this run's own output, and neither is
        taken from the model. The denominator is the kept molecules, because that is
        the set a caller would otherwise carry forward believing it to be chemistry.
        """
        from rdkit import Chem

        kept = [m.smiles for m in batch.molecules if m.smiles]
        if not kept:
            return {}
        allene = Chem.MolFromSmarts(_ALLENE_SMARTS)
        n_allene = 0
        n_small_ring = 0
        n_parsed = 0
        for smiles in kept:
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol is None:
                continue
            n_parsed += 1
            if allene is not None and rdmol.HasSubstructMatch(allene):
                n_allene += 1
            rings = rdmol.GetRingInfo().AtomRings()
            if any(len(ring) <= _SMALL_RING_MAX_SIZE for ring in rings):
                n_small_ring += 1
        if not n_parsed:
            return {}
        return {
            "pocketxmol_n_screened": float(n_parsed),
            "pocketxmol_allene_rate": round(n_allene / n_parsed, 4),
            "pocketxmol_small_ring_rate": round(n_small_ring / n_parsed, 4),
        }


__all__ = [
    "DEFAULT_MODEL_CONFIG",
    "POCKETXMOL_ADAPTER_VERSION",
    "SPEC",
    "TASKS",
    "UNSUPPORTED_TASKS",
    "PocketXMolAdapter",
    "PocketXMolTask",
    "resolve_task",
]
