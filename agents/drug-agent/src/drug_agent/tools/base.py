"""Tool-adapter base — make it or fail.

A tool adapter wraps a scientific tool or model behind a typed contract and a
:class:`CapabilityProfile`. There is exactly one rule, and it replaces the
"always ship a deterministic mock" guarantee this package used to make:

    **A tool either computes the real thing, or it refuses. It never
    substitutes a fabricated number for one it cannot compute.**

The previous contract said an absent backend should downgrade to ``MOCKED``
rather than fail the build. That kept the build green at the cost of putting
invented numbers into the object store, the ranking plane and the report, where
nothing downstream could tell them from measurements. A ``MOCKED`` badge on a
capability card is not a defence: by the time a pose or a restoration score
reaches a scorecard it is just a float.

So an adapter that cannot do the work reports :attr:`CapabilityStatus.UNAVAILABLE`
with an ``unavailable_reason``, and every operation raises :class:`ToolUnavailable`.
Callers are expected to let that surface — the run continues with that arm
*absent and visibly so*, rather than populated with fiction. This generalises the
rule ADR-0010 already applied to de-novo generation: a deployment without PFM
generates nothing instead of something fake.

Two things this rule does *not* forbid:

  * A real computation over demonstration data. Fitting a genuine model to a
    synthetic fixture is a real computation; the fixture's provenance belongs in
    ``license`` / ``data_restrictions``, not in a status flag.
  * A cheap deterministic method chosen over an expensive one. MMFF94s
    minimisation is not a mock of molecular dynamics — it is a different, real,
    and honestly-named calculation.

What it forbids is deriving a scientific quantity from a hash of its own inputs
and returning it as though it were measured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts.capability import CapabilityProfile


class ToolUnavailable(RuntimeError):
    """Raised when a tool is asked for work it has no real backend to do.

    Carries the capability id and the profile's ``unavailable_reason`` so the
    failure explains itself wherever it surfaces — a log line, a task record, or
    the run report.
    """

    def __init__(self, capability_id: str, reason: str) -> None:
        super().__init__(f"{capability_id} is UNAVAILABLE: {reason}")
        self.capability_id = capability_id
        self.reason = reason


class ToolAdapter(ABC):
    """Base class for tool adapters."""

    @property
    @abstractmethod
    def profile(self) -> CapabilityProfile: ...

    def available(self) -> bool:
        """Whether this tool can really do its work on this machine.

        The default is ``True`` for adapters that are always able to compute —
        an RDKit calculation needs nothing but RDKit. Adapters that depend on a
        native binary, a model checkout, or a dataset override this.
        """
        return True

    def require_available(self) -> None:
        """Raise :class:`ToolUnavailable` unless the real backend can run.

        Every public operation calls this first. That is the single enforcement
        point for the make-it-or-fail rule.
        """
        if not self.available():
            p = self.profile
            raise ToolUnavailable(p.capability_id, p.unavailable_reason or "no real backend configured")

    def health(self) -> dict[str, str]:
        p = self.profile
        return {"capability_id": p.capability_id, "status": str(p.status), "backend": p.backend}
