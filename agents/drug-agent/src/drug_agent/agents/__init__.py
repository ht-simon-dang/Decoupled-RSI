"""The agent pool — one folder per plane, one file per agent.

  * ``base.py``      — the :class:`ScientificAgent` interface and
                       :class:`AgentKind`, the three planes below
  * ``platform/``    — cross-cutting roles that shape a run: literature,
                       discovery, orchestration, memory, evaluation
  * ``specialists/`` — scientific roles that own one domain: target, pocket,
                       design, molecule, synthesis, structure, perturbation,
                       safety
  * ``llm/``         — the reasoning agents: master, chemical, evaluation

Each plane's ``__init__`` holds only the abstraction its agents share; every
concrete agent is a class in its own ``*_agent.py``, so the folder listing is
the list of agents.

:func:`build_registry` assembles the deterministic agents and tool adapters the
Engine executes. :mod:`drug_agent.agents.registry` describes the whole pool —
including the reasoning agents — as :class:`~drug_agent.agents.registry.AgentCard`
records, which is what the CLI, the API, and the architecture doc render.
"""

from __future__ import annotations

from ..orchestration.registry import CapabilityRegistry
from ..settings.config import Settings
from ..tools.registry import tool_adapters
from .base import AgentKind, ScientificAgent
from .platform import (
    EvaluationFeedbackAgent,
    LiteratureIntelligenceAgent,
    MemoryKnowledgeAgent,
    OrchestratorAgent,
    PlatformAgent,
    ScientificDiscoveryAgent,
)
from .registry import AgentCard, agent_index, agent_pool
from .specialists import (
    CandidateAssessmentAgent,
    DesignAgent,
    MoleculeAgent,
    PerturbationAgent,
    PocketAgent,
    SafetyAgent,
    SpecialistAgent,
    StructureDynamicsAgent,
    SynthesisAgent,
    TargetEvidenceAgent,
)

__all__ = [
    "AgentCard",
    "AgentKind",
    "CandidateAssessmentAgent",
    "DesignAgent",
    "EvaluationFeedbackAgent",
    "LiteratureIntelligenceAgent",
    "MemoryKnowledgeAgent",
    "MoleculeAgent",
    "OrchestratorAgent",
    "PerturbationAgent",
    "PlatformAgent",
    "PocketAgent",
    "SafetyAgent",
    "ScientificAgent",
    "ScientificDiscoveryAgent",
    "SpecialistAgent",
    "StructureDynamicsAgent",
    "SynthesisAgent",
    "TargetEvidenceAgent",
    "agent_index",
    "agent_pool",
    "build_registry",
]


def _budget_ledger(settings: Settings) -> object | None:
    """The campaign budget ledger this run docks against, if one is configured.

    ``settings.budget_ledger_path`` is empty by default, so nothing changes for
    a run that declares no campaign budget. When it is set, the ledger is opened
    and its docking cap declared, and every docking call in this process is
    reserved against it before the engine is invoked.
    """
    path = str(getattr(settings, "budget_ledger_path", "") or "").strip()
    if not path:
        return None
    from ..selfevolve.target.budget import BudgetLedger

    ledger = BudgetLedger(path)
    cap = float(getattr(settings, "budget_dock_calls", 0.0) or 0.0)
    if cap > 0:
        ledger.declare("dock_calls", cap, unit="calls", note="campaign cap on docking calls")
    return ledger


def build_registry(settings: Settings | None = None) -> CapabilityRegistry:
    """Assemble the full capability registry (agents + tool adapters)."""
    settings = settings or Settings()
    reg = CapabilityRegistry()

    # Platform agents
    for agent in (
        LiteratureIntelligenceAgent(),
        ScientificDiscoveryAgent(),
        OrchestratorAgent(),
        MemoryKnowledgeAgent(),
        # The evaluator holds tools now, and `settings` is how it finds them. Its
        # route audit and independent ADMET panel are the same optional arms the
        # specialists carry and are switched on the same way — by a setting a run
        # can actually set — and they stay off by default for the same reason: a
        # run that did not ask for them must cost what it always cost. Identity
        # re-derivation and the pose audit are unconditional and cost nothing when
        # there are no poses. Pose RE-SCORING stays off: re-scoring a pose with
        # the tool that produced it is not an independent check.
        EvaluationFeedbackAgent(
            settings=settings,
            route_backend=settings.route_backend,
            admet_backend="admetai" if "admetai" in settings.property_backends else "none",
        ),
    ):
        reg.register_agent(agent)

    # Specialist agents.
    #
    # `settings` reaches each of them because their optional arms — the catalogue
    # generator a run samples from, the docking engine, the pose validator, the
    # complex model, the route searcher — are configured the same way every other
    # external tool is, through `settings.external_tools`. The arms that cost a run
    # real time (learned property panels, retrosynthetic search, complex
    # prediction) default to off, so a pipeline that did not ask for them produces
    # and costs exactly what it did before.
    #
    # They are no longer *pinned* off, which they were: this function used to hard-
    # code `property_backends=()`, `route_backend="none"` and
    # `complex_backend="none"`, so five installed tools that `AGENT_TOOLS` promised
    # were reachable could not be reached by anything — not a run parameter, not a
    # profile, not the master agent. Each now reads the setting of the same name.
    #
    # The target, pocket and design agents are OPT-IN rather than absent. None of
    # the three is in the *default* task graph, and each is now schedulable:
    # `agent.design` through the de-novo arm of `source_candidates`, `agent.pocket`
    # through `GenerationSpec.pocket_source == "detect"`, `agent.target` through
    # `ResearchBrief.assess_target_evidence`. Each has a handler in
    # `orchestration/handlers.py`; a brief that asks for none of them builds
    # exactly the graph it always built.
    reg.register_agent(TargetEvidenceAgent(settings=settings))
    reg.register_agent(PocketAgent(settings=settings))
    reg.register_agent(DesignAgent(settings=settings, generator_backend=settings.generator_backend))
    reg.register_agent(
        MoleculeAgent(
            settings=settings,
            generator_backend=settings.generator_backend,
            property_backends=settings.property_backends,
        )
    )
    reg.register_agent(
        SynthesisAgent(
            backend="rdkit-sa",
            settings=settings,
            route_backend=settings.route_backend,
            route_budget=settings.route_budget,
        )
    )
    reg.register_agent(
        StructureDynamicsAgent(
            dock_backend=settings.dock_backend,
            sim_backend="auto",
            settings=settings,
            complex_backend=settings.complex_backend,
            dock_budget=settings.dock_budget,
            budget_ledger=_budget_ledger(settings),
        )
    )
    reg.register_agent(PerturbationAgent())
    reg.register_agent(SafetyAgent(settings=settings))

    # Tool adapters (for the capability matrix / doctor).
    #
    # Taken from `tools.registry.tool_adapters`, not restated here. This used to be
    # a second hard-coded list, and only one direction was ever tested — the
    # registry had to be a subset of the pool — so a tool added to the pool and
    # forgotten here passed every test while being absent from `drugagent doctor`,
    # `drugagent capabilities`, `GET /capabilities`, the dashboard's capability
    # matrix and the manifest's tool_versions. With thirty-seven adapters that stops
    # being a latent bug and becomes a certainty, so the duplication is gone.
    for adapter in tool_adapters(settings):
        reg.register_adapter(adapter)

    return reg
