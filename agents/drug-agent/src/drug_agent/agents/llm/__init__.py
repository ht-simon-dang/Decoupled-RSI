"""The reasoning plane — LLM-backed roles that run whole programs.

  * :class:`~drug_agent.agents.llm.master_agent.MasterAgent` — plans, transcribes
    the user's intent into machine-usable chemical requirements, dispatches the
    other two, critiques what comes back, and decides accept / refine / replan.
  * :class:`~drug_agent.agents.llm.chemical_agent.ChemicalAgent` — generation and
    virtual screening, executed entirely by the deterministic tool plane.
  * :class:`~drug_agent.agents.llm.evaluation_agent.EvaluationAgent` — the
    independent judge, which *computes* rather than predicts.

Three roles, deliberately separated so that no single component both produces a
number and judges it. Only the master agent calls a language model; the other two
are tool-driven, and that boundary is what keeps model-invented numbers out of
the science.

Concrete agents are imported lazily: pulling in the master agent drags in httpx
and the whole pipeline bridge, which the deterministic engine has no need of.
:class:`ReasoningAgent` itself is free of that weight, so the plane's abstraction
can be read without paying for its implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import AgentKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .chemical_agent import ChemicalAgent, ChemicalResult
    from .evaluation_agent import EvaluationAgent, EvaluationReport
    from .master_agent import MasterAgent


class ReasoningAgent:
    """An agent that runs a whole program rather than answering one request.

    Reasoning agents deliberately do *not* implement
    :class:`~drug_agent.agents.base.ScientificAgent`. That interface is built for
    a synchronous agent answering a single typed ``AgentRequest``; these are
    asynchronous, long-lived, and drive many such requests across cycles. Forcing
    them into it would mean a request object that means nothing.

    What they do share is the plane's contract: a stable id, the declared
    actions the pool advertises, and — the reason the plane exists at all — an
    explicit statement of whether this agent calls a language model. Exactly one
    of the three sets :attr:`llm_backed`, and that is enforceable precisely
    because it is declared here rather than inferred.

    Deliberately not an ``ABC``: the three have no method in common to declare
    abstract — the master runs a program, the chemical agent runs a pipeline, the
    evaluator scores a finished run. What they share is what they *declare*, so
    this is a declaration base and nothing more.
    """

    agent_id: str = "agent.llm"
    kind: AgentKind = AgentKind.REASONING
    #: Whether this agent calls a language model.
    llm_backed: bool = False
    #: Actions this agent accepts, mapped to what each does.
    actions: dict[str, str] = {}


__all__ = [
    "ChemicalAgent",
    "ChemicalResult",
    "EvaluationAgent",
    "EvaluationReport",
    "MasterAgent",
    "ReasoningAgent",
]

_LAZY = {
    "MasterAgent": ("master_agent", "MasterAgent"),
    "ChemicalAgent": ("chemical_agent", "ChemicalAgent"),
    "ChemicalResult": ("chemical_agent", "ChemicalResult"),
    "EvaluationAgent": ("evaluation_agent", "EvaluationAgent"),
    "EvaluationReport": ("evaluation_agent", "EvaluationReport"),
}


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), attr)
