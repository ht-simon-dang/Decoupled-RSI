"""LLM runtime: the plumbing that lets a local language model drive the platform.

This package holds everything *about talking to a model* — the Ollama client,
the decoding harness, the prompt library, the structured-output schemas, the
live event stream, the bridge into the scientific pipeline, and the web console
that renders it. The agents that *use* it live in :mod:`drug_agent.agents.llm`.

The split is deliberate: an agent is a role with a job; this is the machinery it
speaks through. Swapping Ollama for another runtime should touch only this
package, and adding a new reasoning agent should touch only ``agents/llm``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0"
