"""The half of an external tool adapter that is the same for all thirty of them.

Every adapter over an upstream project answers the same four questions in the same
way, and the answers are mechanical: is the backend really here, what does the card
say when it is not, which engine name goes on the profile, and what does an
operation do when there is nothing to run. Writing those thirty times would not
make them thirty different decisions — it would make thirty chances to get one of
them subtly wrong, and the one that matters most is the one that decides whether a
number is real.

:class:`ExternalToolAdapter` therefore owns:

* construction — build an :class:`~drug_agent.tools._ext.backend.ExternalBackend`
  from the module's ``SPEC`` and cache readiness once, doing filesystem work only;
* ``self.backend`` — the engine name when ready, and the literal ``"none"`` when
  not, because a card that reports UNAVAILABLE while naming a backend is a card
  that claims something it cannot do;
* :meth:`available` and the ``status`` / ``unavailable_reason`` /
  ``runtime_requirements`` fields of the profile, all derived from the same cached
  probe, so the card and the exception can never drift apart;
* :meth:`unsupported`, the one-line body of an operation that has no backend.

What it deliberately does **not** own is the profile itself. Each adapter builds
its own :class:`~drug_agent.contracts.capability.CapabilityProfile` through
:meth:`profile_fields`, because ``title``, ``description``, ``determinism``,
``cost_class``, ``known_failure_modes`` and ``validation_checks`` are statements
about a specific piece of science and are worth writing out per tool. A shared
default for those would be a shared vagueness.

The rule from :mod:`drug_agent.tools.base` is unchanged and is enforced here rather
than restated: there is no third status, no fallback path and no constructor
argument that selects a stand-in. Thirty installable backends do not soften
make-it-or-fail; they only mean the refusal now names the command that would fix
it.
"""

from __future__ import annotations

from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus
from ..base import ToolAdapter
from .backend import ExternalBackend, ExternalToolError
from .spec import ExternalSpec

#: The backend string an UNAVAILABLE card must carry. Pinned by
#: ``tests/test_pools.py::test_an_unavailable_tool_says_why_and_names_no_backend``,
#: which checks both directions: no backend when unavailable, a real one when not.
NO_BACKEND = "none"


class ExternalToolAdapter(ToolAdapter):
    """Base for every adapter whose science lives in an upstream repository."""

    #: Stable dotted id, e.g. ``"structure.dock.gnina"``. Declared here as well as
    #: on each subclass because the shared machinery below reads it — ``measure``,
    #: ``build_profile`` and ``require_available`` all key off it — and without the
    #: annotation mypy cannot see that a base-class method may use it.
    capability_id: str
    #: The upstream project this adapter binds. Set on the subclass.
    spec: ExternalSpec
    #: Engine name written onto the profile and onto every result object when the
    #: backend is real. Never ``"mock"``; never ``"none"`` while AVAILABLE.
    engine: str = ""
    #: This adapter's contract version, not the backend's.
    version: str = "1.0.0"
    #: Completes "the platform ..." in the unavailable reason, e.g. "generates nothing".
    produces_nothing: str = "produces nothing"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        self.ext = ExternalBackend(self.spec, settings)
        self._missing = self.ext.missing
        # An unavailable card must name no backend at all. Deriving both from one
        # probe is what keeps the card, the reason and the exception in agreement.
        self.backend = (self.engine or self.spec.name.lower()) if not self._missing else NO_BACKEND
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else getattr(settings, f"{self.spec.env_prefix.lower()}_timeout_seconds", None) or 1800.0
        )

    # --- readiness ---------------------------------------------------------
    def available(self) -> bool:
        """Whether the real backend can run here. Reads the cached probe."""
        return not self._missing

    @property
    def unavailable_reason(self) -> str | None:
        """The whole reason, or ``None`` when the tool is ready."""
        if not self._missing:
            return None
        return self.ext.unavailable_reason(self.produces_nothing)

    def unsupported(self) -> Any:
        """Refuse, via the single enforcement point.

        The body of every operation on a tool with no backend. It routes through
        ``require_available()`` rather than raising directly, so the exception's
        ``reason`` is always the same string the card shows — hand-raising lets the
        two drift, and a test asserts they match.
        """
        self.require_available()
        raise AssertionError("unreachable")  # pragma: no cover

    # --- profile -----------------------------------------------------------
    def profile_fields(self) -> dict[str, Any]:
        """The fields every external tool's card fills the same way.

        Subclasses spread this into their own :class:`CapabilityProfile` and then
        say what is actually true of *their* science.
        """
        return {
            "version": self.version,
            "status": CapabilityStatus.AVAILABLE if not self._missing else CapabilityStatus.UNAVAILABLE,
            "provides_mock": False,
            "backend": self.backend,
            "timeout_seconds": self.timeout_s,
            "runtime_requirements": self.ext.runtime_requirements(),
            "license": self.spec.license,
            "unavailable_reason": self.unavailable_reason,
        }

    def build_profile(self, **fields: Any) -> CapabilityProfile:
        """A profile carrying *fields* over the shared external-tool defaults."""
        merged = self.profile_fields()
        merged.update(fields)
        return CapabilityProfile(capability_id=self.capability_id, **merged)

    # --- launching ---------------------------------------------------------
    def run_driver(
        self,
        request: dict[str, Any],
        *,
        error_cls: type[Exception],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run this tool's driver, raising *error_cls* on any failure.

        *timeout_s* is the caller's budget for this one call, which is not the same
        thing as the tool's configured ceiling. ``generate()`` takes
        ``max_runtime_seconds`` and passes it down; without this parameter that
        argument reached ``_sample`` and stopped there, so a caller asking for a
        thirty-second sample still got the 1800-second default and a run that was
        supposed to be bounded was not. Falls back to the configured ceiling.
        """
        return self.ext.run_json(
            request,
            timeout_s=self.timeout_s if timeout_s is None else float(timeout_s),
            error_cls=error_cls,
            **kwargs,
        )


__all__ = ["NO_BACKEND", "ExternalToolAdapter", "ExternalToolError"]
