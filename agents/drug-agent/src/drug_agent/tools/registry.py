"""The tool pool — every capability the platform can call, described uniformly.

A *tool* here is anything that turns inputs into scientific output without
deciding what to do next: RDKit standardization, the activity model, docking,
the safety endpoints, the de-novo generator. Agents decide; tools compute.

Two things make this a pool rather than a folder of modules:

  * every tool publishes a :class:`~drug_agent.contracts.capability.CapabilityProfile`
    saying what it is, what it needs, whether the real backend is present, and
    how it fails; and
  * every tool declares its *operations* — the named units of work it performs.
    Those names are the same ones the run journal meters, so "which tools ran and
    how did they do" is answerable in the same vocabulary the pool is described
    in.

:func:`tool_pool` is the one function to call. It instantiates each adapter
against the current settings, so what it returns reflects this machine — a
missing GPU shows up as ``UNAVAILABLE`` here, not as a surprise mid-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..contracts.base import SchemaModel
from ..contracts.capability import CapabilityProfile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..settings.config import Settings


class ToolCard(SchemaModel):
    """One tool's entry in the pool: identity, operations, and live status."""

    tool_id: str
    title: str = ""
    description: str = ""
    #: Named units of work, mapped to what each one does. These are the keys the
    #: journal meters under.
    operations: dict[str, str] = {}
    backend: str = "builtin"
    status: str = "AVAILABLE"
    determinism: str = "DETERMINISTIC"
    cost_class: str = "CHEAP"
    provides_mock: bool = False
    unavailable_reason: str | None = None
    input_contract: str | None = None
    output_contract: str | None = None
    runtime_requirements: dict[str, Any] = {}
    known_failure_modes: list[str] = []
    validation_checks: list[str] = []
    license: str | None = None

    @classmethod
    def from_profile(cls, profile: CapabilityProfile, operations: dict[str, str]) -> ToolCard:
        return cls(
            tool_id=profile.capability_id,
            title=profile.title,
            description=profile.description,
            operations=operations,
            backend=profile.backend,
            status=str(profile.status),
            determinism=str(profile.determinism),
            cost_class=str(profile.cost_class),
            provides_mock=profile.provides_mock,
            unavailable_reason=profile.unavailable_reason,
            input_contract=profile.input_schema_ref,
            output_contract=profile.output_schema_ref,
            runtime_requirements=dict(profile.runtime_requirements),
            known_failure_modes=list(profile.known_failure_modes),
            validation_checks=list(profile.validation_checks),
            license=profile.license,
        )


#: Operations each tool exposes, keyed by ``capability_id``. Declared here rather
#: than on the adapters so the pool reads as one table: the adapters are free to
#: have private helpers without those leaking into the public surface.
TOOL_OPERATIONS: dict[str, dict[str, str]] = {
    "chem.standardize": {
        "standardize": "Parse, sanitize and standardize a SMILES into a Molecule.",
        "fingerprint": "Morgan fingerprint bit vector.",
        "enumerate_analogues": "Deterministic single-point analogue enumeration from a seed.",
    },
    "model.activity": {
        "fit": "Fit the RandomForest activity model on the run's assay snapshot.",
        "predict_one": "Predict potency for one molecule, with applicability domain and uncertainty.",
        "predict_many": "Predict potency for a batch (delegates to predict_one).",
    },
    "molecule.generate.pfm": {
        "generate": "Sample de-novo ligands for a binding pocket with PFM. No surrogate (ADR-0010).",
    },
    "structure.dock": {
        "dock": (
            "Dock a ligand into a receptor pocket with AutoDock Vina, out of process. "
            "No surrogate (ADR-0011): no binary means no poses."
        ),
        "score_pose": "Score coordinates that already exist with the Vina function (--score_only).",
    },
    "structure.simulate": {
        "simulate": (
            "ETKDGv3 conformer ensemble minimised with MMFF94s; reports energy, "
            "strain span and relaxation RMSD."
        ),
    },
    "activity.safc": {
        "score": (
            "Dock each candidate into every conformation of the bundle's MD receptor "
            "ensemble, encode the complexes with the frozen official EquiScore encoder, "
            "and apply the trained target head. A ranking score, not a calibrated "
            "probability; refuses when the head's target is not this deployment's."
        ),
        "describe_ensemble": (
            "The receptor ensemble the score is read off: conformation count, state "
            "arms and their origin structures. Offline, from the bundle's manifest."
        ),
    },
    "safety.assess": {"assess": "Structural alerts plus calibrated toxicity endpoint models."},
    "synth.assess": {"assess": "Transparent RDKit synthetic-accessibility score. No route (ADR-0011)."},
    "perturb.assess": {
        "score_candidate": "Restoration score for a candidate. UNAVAILABLE: raises (ADR-0011).",
    },
    "platform.envelope": {
        "measure": "Probe the real pipeline and report the achievable range of every success metric.",
    },
    # --- the external catalogue -------------------------------------------
    #
    # Target evidence: the layer that comes before any molecule. Both read a
    # snapshot a setup script fetched once from the source that owns the numbers,
    # and neither opens a socket at run time; absent, each names the missing file
    # and the command that would write it.
    "target.evidence.opentargets": {
        "evidence": "Every layer of one target's snapshot: association, tractability, safety, genetics.",
        "associations": "Target-disease association scores with their per-datatype breakdown.",
        "tractability": "Per-modality tractability buckets, true and false, as the release records them.",
        "safety": "Curated safety liabilities, or the explicit fact that the release records none.",
        "genetics": "Fine-mapped credible sets and locus-to-gene scores. Gentropy output, not re-run.",
        "cached_targets": "Which targets have a fetched snapshot on this deployment.",
        "data_card": "Provenance of one target's snapshot: endpoint, query, release, hash, timestamp.",
    },
    "target.essentiality.depmap": {
        "essentiality": "Gene-effect distribution over the release's cell lines, with its dependency count.",
        "gene_effect": "Per-model Chronos gene effect, most dependent lines first, with their lineage.",
        "selectivity": "Which lineages depend on this gene and which do not, off the same matrix.",
        "cached_genes": "Which genes the fetched DepMap subset actually carries on this deployment.",
        "data_card": "Provenance of the subset: figshare article, release, files, hash, timestamp.",
    },
    # Thirty upstream projects behind typed adapters (ADR-0012). Each either runs
    # the real backend or reports UNAVAILABLE and raises; none of them substitutes
    # anything. Whether a given one runs is a fact about the machine, so these
    # descriptions say what the tool computes, never whether it can.
    #
    # De-novo generation. Nineteen models, one operation, three kinds of
    # conditioning — the kind is recorded on every batch so a ligand-conditioned or
    # unconditional sample can never be read as pocket-conditioned.
    "molecule.generate.reinvent4": {
        "generate": "Sample de-novo SMILES from a released REINVENT4 prior. Token-level, not 3D.",
    },
    "molecule.generate.synformer": {
        "generate": "Sample molecules as synthetic routes over building blocks. Route is a hypothesis.",
    },
    "molecule.generate.prexsyn": {
        "generate": "Sample descriptor-conditioned synthesis programs over building blocks.",
    },
    "molecule.generate.reasyn": {
        "generate": "Project a molecule toward a synthesizable neighbour. Projection changes chemistry.",
    },
    "molecule.generate.rxnflow": {
        "generate": "GFlowNet sampling over reaction templates. Needs a reward-trained policy.",
    },
    "molecule.generate.synflownet": {
        "generate": "GFlowNet sampling over reaction templates; each molecule arrives with its route.",
    },
    "molecule.generate.flowr_root": {
        "generate": "Sample ligands with FLOWR.root in one declared conditioning mode.",
    },
    "molecule.generate.flowr": {
        "generate": "Sample pocket-conditioned ligands with the published FLOWR checkpoint, pinned.",
    },
    "molecule.generate.drugflow": {
        "generate": "Sample pocket-conditioned ligands and carry DrugFlow's own uncertainty through.",
    },
    "molecule.generate.molcraft": {
        "generate": "Sample pocket-conditioned 3D ligands in MolCRAFT's continuous parameter space.",
    },
    "molecule.generate.moljo": {
        "generate": "Gradient-guided structure-based optimisation. Not unconditional de-novo design.",
    },
    "molecule.generate.molpilot": {
        "generate": "Sample with MolPilot's VLB-optimal schedule. Without it this is not MolPilot.",
    },
    "molecule.generate.pocketxmol": {
        "generate": "Run one declared PocketXMol task in a pocket. The task is recorded on the batch.",
    },
    "molecule.generate.omtra": {
        "generate": "Run OMTRA's generation task. It also docks; which task ran is recorded.",
    },
    "molecule.generate.shepherd": {
        "generate": "Sample against a reference ligand's shape and electrostatics. Not pocket-conditioned.",
    },
    "molecule.generate.pilot": {
        "generate": "Pocket-conditioned diffusion with declared multi-objective guidance.",
    },
    "molecule.generate.flowmol3": {
        "generate": "Sample unconditional all-atom 3D molecules. A chemical prior and a control.",
    },
    "molecule.generate.pocket2mol": {
        "generate": "Autoregressive pocket-conditioned sampling. An established baseline, not SOTA.",
    },
    "molecule.generate.targetdiff": {
        "generate": "Target-conditioned diffusion. An established baseline in PFM's own lineage.",
    },
    "molecule.generate.pocketflow": {
        "generate": "Pocket-conditioned autoregressive flow with chemical-rule constraints.",
    },
    # Pockets: find the site the pocket-conditioned generators refuse to run
    # without. Each writes a prepared pocket PDB, which is the thing a PocketSpec
    # needs and the thing this pipeline never had a way to produce.
    "pocket.detect.fpocket": {
        "detect": "Voronoi alpha-sphere cavity detection; writes each ranked site as a prepared pocket PDB.",
    },
    "pocket.detect.p2rank": {
        "detect": "Ligandability prediction over solvent-accessible surface points (P2Rank predict).",
        "rescore": "Re-rank pockets another detector found, e.g. an fpocket run (PRANK rescore).",
    },
    "pocket.detect.vnegnn": {
        "detect": "E(3)-equivariant GNN with virtual nodes. UNAVAILABLE: no checkpoint is published.",
    },
    # Structure: one source of measured coordinates, five producers of predicted
    # ones, and one judge of them. The scores are on different scales and are
    # never comparable across tools.
    "structure.experimental": {
        "resolve": "One cached experimental entry by PDB id, parsed from the file itself.",
        "find": "Every cached entry matching a target, ligand-bearing first then best resolution.",
        "best_for_target": "The single best cached structure for a target, or a refusal naming the cache.",
        "receptor": "The same entry as a ProteinStructure. prepared=False: the deposited coordinates.",
        "catalog": "The whole cache as a content-hashed DataSnapshot with its data card.",
    },
    "structure.predict.protenix": {
        "predict": "Predict a complex and return its poses with the confidence head's own numbers.",
        "predict_complex": "The same prediction, as ranked poses alone.",
        "fold": "The apo protein with no ligand chain. A different question, so a different operation.",
    },
    "structure.dock.gnina": {
        "dock": "Dock into a search box with CNN rescoring. Score is minimizedAffinity, kcal/mol.",
        "score_pose": "Score coordinates that already exist, without searching.",
    },
    "structure.dock.diffdock": {
        "dock": "Diffusion blind docking. The score is a confidence, not an energy.",
        "dock_batch": "Dock several ligands against one receptor in a single model load.",
    },
    "structure.dock.flowdock": {
        "dock": "Flexible docking for apo-to-holo movement, with its own confidence.",
        "estimate_auxiliary": "FlowDock's predicted binding affinity, reported apart from the pose.",
    },
    "structure.complex.boltz2": {
        "predict": "Predict a biomolecular complex and, when asked, its affinity head.",
        "predict_complex": "Complex structure only, as ranked poses.",
        "predict_affinity": "The small-molecule affinity head alone. A prediction, not a measurement.",
    },
    "structure.validate.posebusters": {
        "validate": "Physical and chemical plausibility of one pose, check by check.",
        "validate_batch": "The same suite over several poses, reporting the pass rate.",
    },
    # Properties: one fixed panel, one endpoint learner, one representation.
    "property.admet.admetai": {
        "predict_properties": "41 pretrained ADMET endpoints plus 11 RDKit descriptors, kept apart.",
        "assess_safety": "A SafetyProfile from the toxicity endpoints plus RDKit structural alerts.",
        "assess_safety_batch": "The same, for a batch, in one model load.",
    },
    "property.predict.chemprop": {
        "train": "Fit a Chemprop model on supplied labelled data for one endpoint.",
        "predict": "Predict with a trained checkpoint. Refuses when none exists for the endpoint.",
        "checkpoint_for": "Resolve the checkpoint path for an endpoint, or report that there is none.",
        "trained_endpoints": "Which endpoints this deployment actually has a checkpoint for.",
    },
    "property.represent.unimol": {
        "represent": "3D molecular representations. Embeddings, not property predictions.",
        "fit": "Fine-tune a Uni-Mol property head on supplied labelled data.",
        "predict": "Predict with a fine-tuned head. Refuses without one.",
    },
    # Retrosynthesis: the capability ADR-0011 removed, restored by a real engine.
    "synth.route.aizynthfinder": {
        "plan": "Search retrosynthetic routes to purchasable precursors. Solved means in stock.",
        "plan_batch": "The same search for several candidates, reusing one loaded policy.",
    },
    # Free energy and benchmarking.
    "binding.freeenergy.openfe": {
        "probe": "Report which OpenFE, gufe and OpenMM pieces this interpreter can actually load.",
        "plan_network": "Plan an RBFE ligand network over prepared, pre-posed ligands.",
        "run_edges": "Run each edge's complex and solvent legs and close the cycle into a ddG.",
    },
    "benchmark.pose.posebench": {
        "evaluate": "Score one method's saved predictions over one PoseBench dataset.",
        "compare": "Run that evaluation for several methods and return each one's metrics.",
    },
}


def tool_adapters(settings: Settings | None = None) -> list[Any]:
    """Every tool adapter, constructed for *this* machine. The one list.

    There used to be two: this one, describing what the pool *contains*, and a
    second in ``agents/__init__.build_registry`` describing what actually *runs*
    and what the capability matrix, ``drugagent doctor``, ``/capabilities`` and the
    run manifest see. Only one direction was tested — the registry had to be a
    subset of the pool — so a tool added here and forgotten there passed every test
    while being invisible to every operator-facing surface. With seven adapters
    that was a latent bug; with thirty-seven it is a certainty. ``build_registry``
    now consumes this function, and the second list is gone.

    Construction is deliberately cheap: every adapter does filesystem probing only,
    and none imports its backend or spawns a process. That matters because nothing
    caches this — it runs afresh on every ``drugagent tools``, every
    ``GET /api/pools``, and every report render.
    """
    from ..settings.config import Settings
    from . import envelope_tool  # noqa: F401 - imported for symmetry with the cards below
    from .benchmarking.posebench_tool import PoseBenchAdapter
    from .freeenergy.openfe_tool import OpenFEAdapter
    from .generation.drugflow_tool import DrugFlowAdapter
    from .generation.flowmol3_tool import FlowMol3Adapter
    from .generation.flowr_root_tool import FlowrRootAdapter
    from .generation.flowr_tool import FlowrAdapter
    from .generation.molcraft_tool import MolCraftAdapter
    from .generation.moljo_tool import MolJoAdapter
    from .generation.molpilot_tool import MolPilotAdapter
    from .generation.omtra_tool import OmtraAdapter
    from .generation.pilot_tool import PilotAdapter
    from .generation.pocket2mol_tool import Pocket2MolAdapter
    from .generation.pocketflow_tool import PocketFlowAdapter
    from .generation.pocketxmol_tool import PocketXMolAdapter
    from .generation.prexsyn_tool import PrexSynAdapter
    from .generation.reasyn_tool import ReaSynAdapter
    from .generation.reinvent4_tool import Reinvent4Adapter
    from .generation.rxnflow_tool import RxnFlowAdapter
    from .generation.shepherd_tool import ShepherdAdapter
    from .generation.synflownet_tool import SynFlowNetAdapter
    from .generation.synformer_tool import SynFormerAdapter
    from .generation.targetdiff_tool import TargetDiffAdapter
    from .mmff_tool import SimulationAdapter
    from .pfm_tool import PFMGeneratorAdapter
    from .pockets.fpocket_tool import FpocketAdapter
    from .pockets.p2rank_tool import P2RankAdapter
    from .pockets.vnegnn_tool import VnEgnnAdapter
    from .properties.admetai_tool import AdmetAiAdapter
    from .properties.chemprop_tool import ChempropAdapter
    from .properties.unimol_tool import UniMolAdapter
    from .rdkit_tool import Cheminformatics
    from .retrosynthesis.aizynthfinder_tool import AiZynthFinderAdapter
    from .safc_tool import SafcAdapter
    from .safety_tool import SafetyAdapter
    from .singlecell_tool import SingleCellAdapter
    from .structure.boltz2_tool import Boltz2Adapter
    from .structure.diffdock_tool import DiffDockAdapter
    from .structure.experimental_tool import ExperimentalStructureAdapter
    from .structure.flowdock_tool import FlowDockAdapter
    from .structure.gnina_tool import GninaAdapter
    from .structure.posebusters_tool import PoseBustersAdapter
    from .structure.protenix_tool import ProtenixAdapter
    from .synthesis_tool import SynthesisAdapter
    from .target.depmap_tool import DepMapEssentialityAdapter
    from .target.opentargets_tool import OpenTargetsEvidenceAdapter
    from .vina_tool import DockingAdapter

    settings = settings or Settings()
    return [
        # Built in: these compute with nothing but this package.
        Cheminformatics(settings.fingerprint_bits, settings.fingerprint_radius),
        # ``settings.generator_backend`` names *which* de-novo model a run samples
        # from (see :func:`generator_adapters`), not which backend PFM uses — PFM
        # has exactly one. Passing it through here made every non-PFM value a
        # ValueError out of this constructor, which would take the whole pool down.
        PFMGeneratorAdapter(backend="auto", settings=settings),
        DockingAdapter(backend="auto", settings=settings),
        SimulationAdapter(backend="auto"),
        SafetyAdapter(),
        SafcAdapter(settings=settings),
        SynthesisAdapter(backend="rdkit-sa"),
        SingleCellAdapter(),
        # Target evidence: the layer that comes before any molecule.
        OpenTargetsEvidenceAdapter(settings=settings),
        DepMapEssentialityAdapter(settings=settings),
        # De-novo generation (ADR-0012).
        Reinvent4Adapter(settings=settings),
        SynFormerAdapter(settings=settings),
        PrexSynAdapter(settings=settings),
        ReaSynAdapter(settings=settings),
        RxnFlowAdapter(settings=settings),
        SynFlowNetAdapter(settings=settings),
        FlowrRootAdapter(settings=settings),
        FlowrAdapter(settings=settings),
        DrugFlowAdapter(settings=settings),
        MolCraftAdapter(settings=settings),
        MolJoAdapter(settings=settings),
        MolPilotAdapter(settings=settings),
        PocketXMolAdapter(settings=settings),
        OmtraAdapter(settings=settings),
        ShepherdAdapter(settings=settings),
        PilotAdapter(settings=settings),
        FlowMol3Adapter(settings=settings),
        Pocket2MolAdapter(settings=settings),
        TargetDiffAdapter(settings=settings),
        PocketFlowAdapter(settings=settings),
        # Pockets: find the site the generators refuse to run without.
        FpocketAdapter(settings=settings),
        P2RankAdapter(settings=settings),
        VnEgnnAdapter(settings=settings),
        # Structure: one source of measured coordinates, five producers of
        # predicted ones, one judge of them. A measured structure outranks every
        # prediction, so the cache comes first.
        ExperimentalStructureAdapter(settings=settings),
        GninaAdapter(settings=settings),
        DiffDockAdapter(settings=settings),
        FlowDockAdapter(settings=settings),
        Boltz2Adapter(settings=settings),
        ProtenixAdapter(settings=settings),
        PoseBustersAdapter(settings=settings),
        # Properties, retrosynthesis, free energy, benchmarking.
        AdmetAiAdapter(settings=settings),
        ChempropAdapter(settings=settings),
        UniMolAdapter(settings=settings),
        AiZynthFinderAdapter(settings=settings),
        OpenFEAdapter(settings=settings),
        PoseBenchAdapter(settings=settings),
    ]


#: Id prefix shared by every de-novo generator in the pool. The generators are a
#: *family* — one operation, one contract, twenty implementations — so the agent
#: that samples selects over them by id rather than binding one class.
GENERATOR_ID_PREFIX = "molecule.generate."


def generator_tool_ids() -> list[str]:
    """Every de-novo generator id the catalogue declares, sorted.

    Derived from :data:`TOOL_OPERATIONS` rather than restated, so a generator added
    to the pool is one the molecule agent's wiring already knows about. Cheap: it
    reads a dict and constructs nothing.
    """
    return sorted(tool_id for tool_id in TOOL_OPERATIONS if tool_id.startswith(GENERATOR_ID_PREFIX))


def generator_adapters(settings: Settings | None = None) -> dict[str, Any]:
    """Every de-novo generator adapter, constructed for this machine, keyed by id.

    The selection surface behind ``settings.generator_backend``. It filters
    :func:`tool_adapters` instead of holding a second table of generator classes:
    the pool is the one place that knows which generators exist, and a table here
    would be the same drift ``build_registry`` used to have against it.
    """
    out: dict[str, Any] = {}
    for adapter in tool_adapters(settings):
        tool_id = str(getattr(adapter, "capability_id", ""))
        if tool_id.startswith(GENERATOR_ID_PREFIX):
            out[tool_id] = adapter
    return out


def tool_pool(settings: Settings | None = None) -> list[ToolCard]:
    """Describe every tool as configured for *this* machine.

    Adapters are constructed with backend detection enabled, so the returned
    cards report what would really run — including ``UNAVAILABLE`` where a native
    backend or a GPU model is absent, which under ADR-0011 means the capability
    produces nothing rather than a stand-in.
    """
    from . import envelope_tool
    from .sklearn_tool import BaselineActivityModel

    adapters = tool_adapters(settings)
    cards = [
        ToolCard.from_profile(a.profile, TOOL_OPERATIONS.get(a.profile.capability_id, {})) for a in adapters
    ]
    # The activity model is fitted per run rather than being a standing adapter,
    # so it has no CapabilityProfile — but it is a tool, it is metered like one,
    # and leaving it out of the pool would misrepresent what the platform runs.
    cards.append(
        ToolCard(
            tool_id=BaselineActivityModel.capability_id,
            title="Baseline activity model",
            description=(
                "RandomForest on Morgan fingerprints, fitted per run from the run's own "
                "assay snapshot. Reports ensemble-std uncertainty and an applicability domain."
            ),
            operations=TOOL_OPERATIONS[BaselineActivityModel.capability_id],
            backend="sklearn-rf",
            status="AVAILABLE",
            determinism="DETERMINISTIC",
            cost_class="MODERATE",
            provides_mock=False,
            input_contract="Molecule",
            output_contract="Prediction",
            known_failure_modes=[
                "no assay records for the target (nothing to fit)",
                "candidate outside the training domain (prediction is not meaningful)",
            ],
            validation_checks=["applicability domain assigned", "uncertainty finite"],
            license="scikit-learn BSD-3-Clause",
        )
    )
    # The capability envelope is a tool over the *platform* rather than over a
    # molecule: it runs probe pipelines and reports what this deployment can
    # reach. It has no standing adapter instance because a probe is the call.
    cards.append(
        ToolCard.from_profile(envelope_tool.profile(), TOOL_OPERATIONS.get(envelope_tool.CAPABILITY_ID, {}))
    )
    cards.extend(evolved_artifact_cards(settings))
    return sorted(cards, key=lambda c: c.tool_id)


def evolved_artifact_cards(settings: Settings | None = None) -> list[ToolCard]:
    """Cards for the evolved artifacts this deployment has actually adopted.

    A tool the platform can call is a tool the pool describes, and an artifact
    an earlier campaign produced is no exception once an operator has activated
    it. Three things keep this honest rather than flattering:

    * only ``master_active`` artifacts appear. A registered candidate and a
      benchmark-accepted one are evidence, not capabilities, and neither is
      loadable from here;
    * an artifact scoped to a different target manifest than the deployment
      declares is ``UNAVAILABLE`` naming both hashes. Under ADR-0011 an
      unavailable capability produces nothing, which is the correct outcome: a
      tool evolved for another protein is not a degraded version of this one;
    * the card describes what the artifact *is*, never what it achieved. The
      evidence for a gain is the artifact's own ``Comparison`` record.

    Returns an empty list when no registry is configured, which is the default.
    """
    from ..selfevolve.target.registry import master_tool_cards

    out: list[ToolCard] = []
    for card in master_tool_cards(settings):
        out.append(
            ToolCard(
                tool_id=str(card["tool_id"]),
                title=str(card.get("title") or ""),
                description=str(card.get("description") or ""),
                operations={
                    "resolve": "Resolve the pinned artifact and verify every referenced component.",
                    "load": "Materialize its components and load it through its declared loader.",
                },
                backend=str(card.get("backend") or "target-artifact-registry"),
                status=str(card.get("status") or "UNAVAILABLE"),
                determinism="DETERMINISTIC",
                cost_class="CHEAP",
                provides_mock=False,
                unavailable_reason=card.get("unavailable_reason"),
                runtime_requirements={
                    k: card.get(k)
                    for k in (
                        "artifact_id",
                        "version_hash",
                        "kind",
                        "role",
                        "state",
                        "benchmark_accepted",
                        "fixture",
                        "target_manifest_hash",
                        "registry_path",
                    )
                },
                known_failure_modes=[
                    "the artifact is scoped to a different target manifest (UNAVAILABLE, both hashes named)",
                    "a referenced component is no longer in the content-addressed store",
                ],
                validation_checks=[
                    "target manifest compatibility",
                    "artifact contract",
                    "every component content address present in the store",
                ],
            )
        )
    return out


def tool_index(settings: Settings | None = None) -> dict[str, ToolCard]:
    """The pool keyed by tool id."""
    return {c.tool_id: c for c in tool_pool(settings)}
