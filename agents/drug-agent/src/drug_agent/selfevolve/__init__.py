"""The self-evolution subsystem — a research loop over the platform's own history.

The platform in ``drug_agent`` serves users: a natural-language request becomes a
ranked candidate portfolio, and every request leaves a durable journal behind.
This package is the *second* kind of intelligence the brief asks for — one that
reads those journals, works out what went wrong, proposes changes to the
user-serving system, and **earns** each change through measurement against a
benchmark it is not allowed to touch.

The spine is one sentence: **a candidate is a hypothesis, and a hypothesis is
only worth promoting if an evaluator the candidate cannot reach says so.**

```
journals ─▶ experience ─▶ analysis ─▶ failure clusters ─▶ hypotheses
                                                             │
                                              candidates ◀───┘
                                                   │
                              ┌────────────────────┴───────────────────┐
                              │      EVALUATION KERNEL (trusted)       │
                              │  gate 0 invariants → gate 1 replay →   │
                              │  gate 2 live benchmark vs gold rubric  │
                              └────────────────────┬───────────────────┘
                                                   ▼
                                      archive ─▶ promotion gate ─▶ human
```

Layout mirrors that flow:

``experience``   journals → typed :class:`~.schemas.Episode` records
``analysis``     metrics, failure detectors, tool attribution, clustering
``hypotheses``   generator → verifier → reviser over a failure cluster
``candidates``   what a change *is*: a runtime overlay, or a file patch
``archive``      lineage, novelty descriptors, quality-diversity cells
``evaluation``   **the trusted kernel** — benchmark, gold rubric, integrity
``promotion``    the gate a candidate must clear to reach production
``backends``     Ollama (cheap controller) · Claude Code CLI (deep worker)
``research``     provenance for external sources, treated as untrusted data
``controller``   the cycle that drives all of the above

Nothing here is imported by the user-serving path. The only production code this
package needs is a small, default-inert runtime policy
(:mod:`drug_agent.llm.policy`), so a deployment that never runs the evolver
behaves exactly as it did before it existed.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Bumped when the on-disk layout of the evolution store changes.
__version__ = "1.1.0"
