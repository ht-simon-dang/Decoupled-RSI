"""Shared machinery for tools whose science lives in someone else's repository.

Most of the catalogue in ``docs/`` is not code this project wrote. It is thirty
upstream projects — generators, docking engines, property models, a retrosynthesis
planner — each with its own dependency pins, its own weights, and in several cases
its own CUDA build that cannot share this interpreter. What this package owns is
the *boundary*: how one of those is located, how the platform decides whether it
can really run, and how it is launched without letting its failures become the
platform's fiction.

Three pieces, one each:

:mod:`~drug_agent.tools._ext.spec`
    :class:`ExternalSpec` — data about an upstream project. Where it would be, what
    proves it is really there, which modules the run path needs, which commit the
    driver was written against.

:mod:`~drug_agent.tools._ext.probe`
    Whether a given interpreter can import given modules, without importing them
    and without spawning it.

:mod:`~drug_agent.tools._ext.backend`
    :class:`ExternalBackend` — resolution, cached readiness, and a driver launch
    that carries JSON over files, bounds its own output, reaps its process group,
    and reports how long the child really took.

This is infrastructure, not a capability: nothing here publishes a
``CapabilityProfile`` or appears in the tool pool. It lives in a subpackage, with a
leading underscore, for exactly that reason — ``tests/test_pools.py`` enumerates
tool modules and demands each one bind a capability, and a shared helper that
pretended to be a tool would have to lie to satisfy it.

The rule these pieces exist to serve is unchanged from
:mod:`drug_agent.tools.base`: a tool computes the real thing or it reports
UNAVAILABLE and refuses. Making thirty backends installable does not soften it —
it only means the refusal now has an install command in it.
"""

from __future__ import annotations

from .backend import DriverRun, ExternalBackend, ExternalToolError
from .probe import can_import, conda_env_pythons, missing_modules
from .spec import ExternalSpec

__all__ = [
    "DriverRun",
    "ExternalBackend",
    "ExternalSpec",
    "ExternalToolError",
    "can_import",
    "conda_env_pythons",
    "missing_modules",
]
