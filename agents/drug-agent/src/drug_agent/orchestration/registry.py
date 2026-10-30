"""Capability registry.

Holds the set of agents and tool adapters available to a run and can emit the
capability matrix (which capabilities are AVAILABLE / MOCKED / UNAVAILABLE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CapabilityStatus

if TYPE_CHECKING:
    from ..agents.base import ScientificAgent
    from ..tools.base import ToolAdapter


class CapabilityRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, ScientificAgent] = {}
        self._adapters: dict[str, ToolAdapter] = {}

    # --- agents ------------------------------------------------------------
    def register_agent(self, agent: ScientificAgent) -> None:
        self._agents[agent.agent_id] = agent

    def get_agent(self, agent_id: str) -> ScientificAgent:
        if agent_id not in self._agents:
            raise KeyError(f"unknown agent: {agent_id}")
        return self._agents[agent_id]

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._agents

    @property
    def agents(self) -> dict[str, ScientificAgent]:
        return dict(self._agents)

    # --- adapters ----------------------------------------------------------
    def register_adapter(self, adapter: ToolAdapter) -> None:
        self._adapters[adapter.profile.capability_id] = adapter

    def get_adapter(self, capability_id: str) -> ToolAdapter:
        if capability_id not in self._adapters:
            raise KeyError(f"unknown capability: {capability_id}")
        return self._adapters[capability_id]

    def has_adapter(self, capability_id: str) -> bool:
        return capability_id in self._adapters

    @property
    def adapters(self) -> dict[str, ToolAdapter]:
        return dict(self._adapters)

    # --- capability matrix -------------------------------------------------
    def profiles(self) -> list[CapabilityProfile]:
        out: list[CapabilityProfile] = []
        for agent in self._agents.values():
            out.append(agent.describe_capability())
        for adapter in self._adapters.values():
            out.append(adapter.profile)
        return sorted(out, key=lambda p: p.capability_id)

    def capability_matrix(self) -> dict[str, str]:
        return {p.capability_id: str(p.status) for p in self.profiles()}

    def unavailable(self) -> list[CapabilityProfile]:
        return [
            p for p in self.profiles() if p.status in {CapabilityStatus.UNAVAILABLE, CapabilityStatus.MOCKED}
        ]
