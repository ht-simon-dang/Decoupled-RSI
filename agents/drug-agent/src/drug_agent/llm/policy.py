"""Runtime policy — the seam the self-evolution subsystem writes through.

The evolver proposes changes to how the master agent behaves. Those changes have
to reach the *user-serving* code somehow, and there are only bad ways to do that
except one: a small, explicit, versioned policy object that the production path
reads, whose defaults reproduce today's behaviour byte-for-byte.

That last property is what makes the whole loop honest. With no overlay
configured, every field below is the value the platform already had, the prompt
strings hash to the same version id, and a deployment that never runs the evolver
cannot tell this module exists. So "did the candidate help?" is a question about
the overlay, not about a rewrite that also happened to change five other things.

Where a policy comes from, in order:

1. an explicit :func:`use_policy` context (what the evaluation kernel uses, so a
   candidate arm and a baseline arm can run in one process without leaking);
2. ``DRUGAGENT_EVOLVE_OVERLAY`` — a path to a JSON overlay file;
3. the defaults below.

Every field is *additive or gating*. Nothing here can make the master agent
invent a scientific number, relax a trust label, or bypass the tool plane: the
policy can change what the planner is told and how the platform repairs the
model's output, and that is the whole surface.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

#: Bumped when a field is added or its default changes, so a prompt version id
#: from an old run cannot silently collide with a new one.
POLICY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimePolicy:
    """How the master agent should behave. Defaults == the pre-evolution system."""

    #: Human label for the overlay this came from (a candidate id, usually).
    policy_id: str = "default"
    notes: str = ""

    # --- prompt surface -------------------------------------------------
    #: Text appended to each phase's system prompt. Empty == unchanged prompt.
    plan_prompt_addendum: str = ""
    transcribe_prompt_addendum: str = ""
    critique_prompt_addendum: str = ""
    refine_prompt_addendum: str = ""

    #: Inject a *measured* capability envelope into the planning prompt — what
    #: this deployment has actually been observed to achieve. The numbers come
    #: from :mod:`drug_agent.llm.envelope`, which runs the real pipeline; the
    #: model is given them, it does not produce them.
    capability_envelope: bool = False
    #: Also give the transcriber the envelope (constraint feasibility).
    capability_envelope_transcribe: bool = False

    # --- deterministic repair surface -----------------------------------
    #: Replace efficacy criteria that cannot apply to the program's goal kind
    #: (``max_pIC50`` on a phenotype-rescue goal) with the one that can. Off by
    #: default because it changes what a plan means, and that is a decision the
    #: evaluation has to earn.
    goal_aware_criteria: bool = False
    #: On refinement, carry forward objectives/constraints the model did not
    #: mention in its own ``changes`` list. Guards the observed failure where an
    #: "adjustment" silently emptied both lists.
    refine_carry_forward: bool = False
    #: Cap how far one refinement may move the requirement set (fraction of
    #: objectives+constraints that may disappear). 1.0 == no cap.
    refine_max_drop_fraction: float = 1.0

    def prompt_version(self) -> str:
        """A stable id for *the prompts this policy produces*.

        Recorded into the journal so a later analysis can ask which prompt
        version generated which episode without keeping a copy of the prompts.
        """
        payload = {
            "schema": POLICY_SCHEMA_VERSION,
            "plan": self.plan_prompt_addendum,
            "transcribe": self.transcribe_prompt_addendum,
            "critique": self.critique_prompt_addendum,
            "refine": self.refine_prompt_addendum,
            "envelope": self.capability_envelope,
            "envelope_transcribe": self.capability_envelope_transcribe,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"p{POLICY_SCHEMA_VERSION}-{digest}"

    def behaviour_version(self) -> str:
        """Id covering prompts *and* deterministic repair behaviour."""
        payload = {k: v for k, v in asdict(self).items() if k not in ("policy_id", "notes")}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"b{POLICY_SCHEMA_VERSION}-{digest}"

    def is_default(self) -> bool:
        return self.behaviour_version() == DEFAULT_POLICY.behaviour_version()

    def public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prompt_version"] = self.prompt_version()
        d["behaviour_version"] = self.behaviour_version()
        return d

    @classmethod
    def from_overlay(cls, overlay: dict[str, Any] | None) -> RuntimePolicy:
        """Build from an overlay payload, ignoring keys this version does not know.

        Unknown keys are dropped rather than raising: an overlay written by a
        newer evolver must not brick an older deployment, and a candidate that
        depends on a field this build lacks will simply fail to show an effect —
        which the benchmark will report honestly.
        """
        if not overlay:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in overlay.items():
            if key not in known:
                continue
            kwargs[key] = value
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()


DEFAULT_POLICY = RuntimePolicy()

_ACTIVE: ContextVar[RuntimePolicy | None] = ContextVar("drug_agent_runtime_policy", default=None)
_ENV_CACHE: dict[str, RuntimePolicy] = {}


def load_overlay_file(path: str | Path) -> RuntimePolicy:
    """Read an overlay JSON file. A malformed file degrades to the default."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_POLICY
    if not isinstance(raw, dict):
        return DEFAULT_POLICY
    # Accept either a bare overlay or a candidate record carrying one.
    overlay = raw.get("overlay") if "overlay" in raw else raw
    policy = RuntimePolicy.from_overlay(overlay if isinstance(overlay, dict) else None)
    label = raw.get("candidate_id") or raw.get("policy_id")
    if label and policy.policy_id == "default":
        policy = RuntimePolicy(**{**asdict(policy), "policy_id": str(label)})
    return policy


def active_policy() -> RuntimePolicy:
    """The policy in force right now."""
    explicit = _ACTIVE.get()
    if explicit is not None:
        return explicit
    env = os.environ.get("DRUGAGENT_EVOLVE_OVERLAY", "")
    if not env:
        return DEFAULT_POLICY
    if env not in _ENV_CACHE:
        _ENV_CACHE[env] = load_overlay_file(env)
    return _ENV_CACHE[env]


@contextlib.contextmanager
def use_policy(policy: RuntimePolicy | dict[str, Any] | None) -> Iterator[RuntimePolicy]:
    """Bind a policy for the duration of the block.

    The evaluation kernel runs a baseline arm and a candidate arm back to back in
    one process; a ``ContextVar`` is what keeps the second from inheriting the
    first. ``None`` binds the default explicitly, which is not the same as
    leaving it unbound — it also masks ``DRUGAGENT_EVOLVE_OVERLAY``, so a
    baseline arm really is the baseline.
    """
    if isinstance(policy, dict):
        policy = RuntimePolicy.from_overlay(policy)
    resolved = policy or DEFAULT_POLICY
    token = _ACTIVE.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE.reset(token)


def reset_env_cache() -> None:
    """Forget cached overlay files (tests, and after an overlay is rewritten)."""
    _ENV_CACHE.clear()


__all__ = [
    "DEFAULT_POLICY",
    "POLICY_SCHEMA_VERSION",
    "RuntimePolicy",
    "active_policy",
    "load_overlay_file",
    "reset_env_cache",
    "use_policy",
]
