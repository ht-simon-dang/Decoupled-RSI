"""The agent pool — every agent in the platform, described uniformly.

An *agent* decides something. A tool computes something. The distinction is the
spine of this system, and it is what the pool makes legible: each entry says what
the agent is for, which plane it belongs to, what it consumes and produces, and
which tools it reaches for.

The three planes are :class:`~drug_agent.agents.base.AgentKind`, and each has a
folder: ``platform/``, ``specialists/``, ``llm/``. An agent declares its own
plane on its class, so this module reads ``kind`` off the agent rather than
keeping a second list of which id belongs where.

``PLATFORM`` and ``SPECIALIST`` agents implement
:class:`~drug_agent.agents.base.ScientificAgent` and are executed by the Engine
against typed ``AgentRequest``/``AgentResult`` contracts, so their cards are
introspected from live instances. ``REASONING`` agents are asynchronous and
orchestrate whole programs, and importing one means opening an HTTP client to
Ollama just to render a table — so they are described declaratively instead, in
``_REASONING_CARDS``. ``tests/test_pools.py`` holds those declarations to the
classes in ``llm/`` so the two cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts.base import SchemaModel
from ..tools.registry import generator_tool_ids
from .base import AgentKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..settings.config import Settings
    from .base import ScientificAgent


class AgentCard(SchemaModel):
    """One agent's entry in the pool."""

    agent_id: str
    kind: AgentKind = AgentKind.SPECIALIST
    title: str = ""
    description: str = ""
    version: str = "1.0.0"
    #: Actions this agent accepts, mapped to what each does.
    actions: dict[str, str] = {}
    #: Tool ids from :mod:`drug_agent.tools.registry` this agent calls.
    uses_tools: list[str] = []
    #: Contract names in and out (see ``schemas/``).
    consumes: list[str] = []
    produces: list[str] = []
    #: Whether this agent calls a language model. Exactly the REASONING ones do.
    llm_backed: bool = False
    status: str = "AVAILABLE"
    notes: str = ""


#: Which tools each deterministic agent reaches for. Kept beside the pool rather
#: than on the agents so the wiring is readable in one place — this table is the
#: edge list of the agent/tool graph in ``docs/agent-tool-architecture.md``.
#:
#: An id belongs here when the agent's code path can really call it — not when the
#: name sounds like the agent's domain. Some of these need a run parameter
#: (``property_backends``, ``route_backend``, ``complex_backend``) or run data (a
#: prepared pocket) before they do anything; that is a fact about a run, and the
#: agent skips them with a reason when it is missing. What must never appear here
#: is a tool the agent has no way to invoke at all, because ``reporting/report.py``
#: reads this table to decide which absent tools cost *this run* something.
#:
#: ``platform.envelope`` is deliberately absent: it probes whole pipelines through
#: the Engine and is therefore a caller of these agents, not a tool one of them
#: can reach for. Its one caller is the master agent, and it is declared there.
AGENT_TOOLS: dict[str, list[str]] = {
    "agent.literature": [],
    "agent.discovery": [],
    "agent.orchestrator": [],
    "agent.memory": [],
    # The evaluator is no longer toolless, and none of these makes it a producer:
    # it re-derives identity, judges coordinates, re-scores a pose that already
    # exists, asks whether the best candidates can be made, and asks a second
    # model about the safety endpoints the first one reported. Nothing it writes
    # is a Prediction, so nothing it computes can reach a scorecard or a
    # shortlist. The last three need a run parameter to switch them on.
    "agent.evaluation": [
        "chem.standardize",
        "structure.validate.posebusters",
        "structure.dock.gnina",
        "synth.route.aizynthfinder",
        "property.admet.admetai",
        # SAFC re-ranks molecules that were already submitted, against a receptor
        # ensemble and a head trained elsewhere. It belongs here, with the other
        # second opinions, for two reasons. It is an analysis of a submission
        # rather than a producer of one — and it must stay one: its ZAK head was
        # fitted on 67 ligands of which four sit in this programme's frozen *test*
        # partition, so a SAFC score that could reach a shortlist would put a
        # held-out compound's influence into the selection it is meant to judge.
        # Being an evaluator tool is what makes that structurally impossible.
        "activity.safc",
    ],
    # The layer before any molecule: what is known about the protein.
    "agent.target": ["target.evidence.opentargets", "target.essentiality.depmap"],
    # A structure in, a prepared pocket file out. `structure.experimental` is how
    # a PDB id in the brief becomes coordinates without a network call.
    "agent.pocket": [
        "pocket.detect.p2rank",
        "pocket.detect.fpocket",
        "pocket.detect.vnegnn",
        "structure.experimental",
    ],
    # Every de-novo generator in the catalogue: `settings.generator_backend` picks
    # one and `select_generator` resolves it, so the whole family is reachable
    # rather than the single class the molecule agent used to bind. They live here
    # because this is the agent whose code path calls `generate()`.
    "agent.design": [*generator_tool_ids()],
    # Sourcing, identity and screening. The three property backends predict
    # endpoints RDKit and the baseline model cannot, and they are ranked, which is
    # why they stay with the producer rather than moving to the evaluator.
    "agent.molecule": [
        "chem.standardize",
        "model.activity",
        "property.admet.admetai",
        "property.predict.chemprop",
        "property.represent.unimol",
    ],
    "agent.structure": [
        "structure.simulate",
        "structure.dock",
        "structure.dock.gnina",
        "structure.validate.posebusters",
        "structure.complex.boltz2",
        "structure.predict.protenix",
        "structure.experimental",
    ],
    "agent.synthesis": ["synth.assess", "synth.route.aizynthfinder"],
    "agent.safety": ["safety.assess", "chem.standardize"],
    "agent.perturbation": ["perturb.assess", "chem.standardize"],
}

#: What each deterministic agent puts into the store, by contract name.
AGENT_PRODUCES: dict[str, list[str]] = {
    "agent.literature": ["EvidenceItem"],
    "agent.discovery": ["ExperimentProposal"],
    "agent.orchestrator": ["AgentResult"],
    "agent.memory": ["AgentResult"],
    # Records *about* results — never a Prediction, which is what keeps the judge
    # out of the ranking it grades.
    "agent.evaluation": ["ValidationIssue", "PoseBustersReport"],
    "agent.target": ["EvidenceItem"],
    "agent.pocket": ["PocketSpec"],
    "agent.design": ["GenerationBatch", "Molecule"],
    "agent.molecule": ["Molecule", "Candidate", "Prediction", "Scorecard", "GenerationBatch"],
    "agent.structure": ["Pose", "SimulationResult"],
    "agent.synthesis": ["SynthesisPlan"],
    "agent.safety": ["SafetyProfile"],
    "agent.perturbation": ["PerturbationSignature"],
}

#: One line per deterministic agent about *how it fits*, in the voice the
#: reasoning cards use. The rest of a deterministic card is introspected from the
#: live instance, which is what keeps it from drifting; a note is the one thing
#: no instance can tell you, because it is about the agent's place in the shape of
#: the whole rather than about the agent.
AGENT_NOTES: dict[str, str] = {
    "agent.target": (
        "The first agent in the program and the only one that runs before chemistry exists. It "
        "produces evidence, never molecules, and it reads snapshots fetched by a setup step — a "
        "target this deployment has not fetched yields a refusal naming the command, never a "
        "fixture. Scheduled by `ResearchBrief.assess_target_evidence`, which the master agent "
        "sets: until that field existed nothing in the graph named this agent, so the two tools "
        "below were advertised as reachable and could be reached by no run at all."
    ),
    "agent.pocket": (
        "Turns a structure into a PocketSpec whose pdb_path is a file that exists. Every "
        "pocket-conditioned generator refuses without one, and until this agent there was no code "
        "path that could produce it — only a human with scripts/prepare_pocket.py."
    ),
    "agent.design": (
        "Owns the twenty-one de-novo generators as a family, selected by settings.generator_backend. "
        "It reads the pocket agent's PocketSpec off the blackboard before it reads the brief, so the "
        "site a run detected is the site it samples in. The molecule agent delegates the de-novo arm "
        "of source_candidates here."
    ),
    "agent.evaluation": (
        "The independent validator, and the only agent that computes with tools it has no stake in: "
        "nothing it writes is a Prediction, so nothing it produces can reach a scorecard, a Pareto "
        "front or a shortlist."
    ),
}

#: The LLM plane. Declared rather than introspected, because importing these
#: modules drags in httpx and the whole pipeline bridge. The ids, actions and
#: ``llm_backed`` flags below are checked against the real classes in
#: ``tests/test_pools.py``; the prose is the pool's own.
_REASONING_CARDS: list[AgentCard] = [
    AgentCard(
        agent_id="agent.llm.master",
        kind=AgentKind.REASONING,
        title="Master agent",
        description=(
            "Plans the research program, transcribes the user's instruction into machine-usable "
            "chemical requirements, dispatches the chemical and evaluation agents each cycle, "
            "critiques what comes back against its own success criteria, and decides whether to "
            "accept, refine the requirements, or replan. It never computes a scientific number."
        ),
        actions={
            "plan": "Turn the instruction into a research plan with measurable success criteria.",
            "transcribe": "Turn intent into a ChemicalRequirements object (the generation prompt).",
            "critique": "Judge tool-computed results against the plan's criteria.",
            "refine": "Diagnose a shortfall and rewrite the generation requirements.",
        },
        # The one tool a reasoning agent calls, and only when the runtime policy
        # turns it on: `_envelope_block` runs the probe to tell the planner what
        # this deployment can actually reach before it sets a target.
        uses_tools=["platform.envelope"],
        consumes=["instruction"],
        produces=["TaskPlan", "ChemicalRequirements", "Critique", "RequirementsAdjustment"],
        llm_backed=True,
        notes=(
            "The only component that calls a language model. Its outputs are decisions and "
            "hypotheses, never measurements."
        ),
    ),
    AgentCard(
        agent_id="agent.llm.chemical",
        kind=AgentKind.REASONING,
        title="Chemical agent (LLM plane)",
        description=(
            "Executes one set of chemical requirements by running the full deterministic pipeline, "
            "then summarizes the candidate-level result. Despite living in the LLM plane it calls "
            "no model: it is the boundary where a decision becomes computation."
        ),
        actions={"run": "Run generation + virtual screening for one requirements object."},
        # It runs the whole deterministic pipeline, so it reaches whatever the
        # deterministic agents reach. Derived rather than restated: a tool wired to
        # a specialist and forgotten here would make this card quietly wrong.
        uses_tools=sorted({tool for tools in AGENT_TOOLS.values() for tool in tools}),
        consumes=["ChemicalRequirements"],
        produces=["ChemicalResult"],
        llm_backed=False,
    ),
    AgentCard(
        agent_id="agent.llm.evaluation",
        kind=AgentKind.REASONING,
        title="Evaluation agent (LLM plane)",
        description=(
            "The independent judge. Scores the pipeline's output against the master agent's "
            "success criteria using only numbers the tool plane produced. It computes; it does "
            "not predict, and it does not call a model."
        ),
        actions={"evaluate": "Score a completed run against the plan's success criteria."},
        uses_tools=[],
        consumes=["BackendRun", "SuccessCriterion"],
        produces=["EvaluationReport"],
        llm_backed=False,
        notes="Separated from the chemical agent so nothing both produces a number and grades it.",
    ),
]


def _card_from_agent(agent: ScientificAgent) -> AgentCard:
    profile = agent.describe_capability()
    return AgentCard(
        agent_id=agent.agent_id,
        kind=agent.kind,
        title=profile.title or agent.agent_id,
        description=profile.description,
        version=agent.version,
        actions=dict(agent.actions),
        uses_tools=AGENT_TOOLS.get(agent.agent_id, []),
        consumes=[profile.input_schema_ref] if profile.input_schema_ref else [],
        produces=AGENT_PRODUCES.get(agent.agent_id, []),
        llm_backed=False,
        status=str(profile.status),
        notes=AGENT_NOTES.get(agent.agent_id, ""),
    )


def agent_pool(settings: Settings | None = None, *, include_reasoning: bool = True) -> list[AgentCard]:
    """Describe every agent, deterministic and reasoning alike.

    Deterministic agents are introspected from live instances so their status
    reflects this machine; reasoning agents come from the declared table above.
    """
    from . import build_registry

    registry = build_registry(settings)
    cards = [_card_from_agent(a) for a in registry.agents.values()]
    if include_reasoning:
        cards.extend(_REASONING_CARDS)
    return sorted(cards, key=lambda c: (c.kind, c.agent_id))


def agent_index(settings: Settings | None = None) -> dict[str, AgentCard]:
    """The pool keyed by agent id."""
    return {c.agent_id: c for c in agent_pool(settings)}
