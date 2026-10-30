"""The platform plane — cross-cutting roles that shape a run.

These agents do not own a scientific domain; they own the shape of the run
itself: what evidence it starts from, what hypotheses it pursues, how far it has
got, what it remembers, and whether what it produced holds up.

  * :class:`~drug_agent.agents.platform.literature_agent.LiteratureIntelligenceAgent`
  * :class:`~drug_agent.agents.platform.discovery_agent.ScientificDiscoveryAgent`
  * :class:`~drug_agent.agents.platform.orchestrator_agent.OrchestratorAgent`
  * :class:`~drug_agent.agents.platform.memory_agent.MemoryKnowledgeAgent`
  * :class:`~drug_agent.agents.platform.evaluation_agent.EvaluationFeedbackAgent`

One file per agent, so the folder listing *is* the plane's contents. This module
holds only the abstraction they share.
"""

from __future__ import annotations

from ..base import AgentKind, ScientificAgent


class PlatformAgent(ScientificAgent):
    """An agent that shapes a run rather than owning a scientific domain.

    Platform agents read and write the run's own record — evidence, hypotheses,
    task states, the knowledge graph, validation issues — rather than computing
    chemistry. They are deterministic and cheap, and none of them calls a model.

    The class carries no behaviour beyond the plane tag: unlike the specialists,
    these five share an *altitude*, not an execution shape, and inventing a
    common ``execute`` for them would be a fiction. What it does buy is a single
    source of truth for the plane — :func:`drug_agent.agents.registry.agent_pool`
    reads ``kind`` off the class instead of matching against a list of ids.
    """

    kind = AgentKind.PLATFORM


from .discovery_agent import ScientificDiscoveryAgent  # noqa: E402
from .evaluation_agent import EvaluationFeedbackAgent  # noqa: E402
from .literature_agent import LiteratureIntelligenceAgent  # noqa: E402
from .memory_agent import MemoryKnowledgeAgent  # noqa: E402
from .orchestrator_agent import OrchestratorAgent  # noqa: E402

__all__ = [
    "EvaluationFeedbackAgent",
    "LiteratureIntelligenceAgent",
    "MemoryKnowledgeAgent",
    "OrchestratorAgent",
    "PlatformAgent",
    "ScientificDiscoveryAgent",
]
