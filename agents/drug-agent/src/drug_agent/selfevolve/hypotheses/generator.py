"""Turning a failure cluster into a falsifiable, implementable hypothesis.

The loop is Aletheia's — generate, verify, revise — with one amendment the brief
insists on and the results justify: **the verifier is told that
``INSUFFICIENT_EVIDENCE`` is a good answer.** A verifier that must produce a
verdict every cycle stops being a verifier and becomes a rubber stamp, and at
five episodes "we cannot tell from this data" is frequently the true answer.

Context economy is the other constraint. Claude Max has usage limits, and the
raw journals are ~1 MB of events per request. So the hierarchy is:

    raw journals  →  normalized episodes  →  computed findings  →  one cluster,
    with its statistics and two verbatim excerpts  →  the model

The model is handed numbers and named episode ids, not logs. It may ask for the
original trace: :func:`context_for_cluster` exposes ``excerpt`` for exactly that,
and every claim carries the episode id it came from, so a summary can always be
walked back to the bytes.

The output is not free-form advice. It is a
:class:`~drug_agent.selfevolve.schemas.Hypothesis` naming one of the built-in
mutation operators (or a prompt addendum), because a proposal that cannot be
turned into something the kernel can score is not a proposal, it is an opinion.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..analysis.clustering import surface_of
from ..backends import BackendRegistry
from ..candidates.library import BUILDERS_BY_SURFACE, CANDIDATE_BUILDERS
from ..config import EvolveConfig
from ..schemas import Candidate, Episode, FailureCluster, Finding, Hypothesis

HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "mechanism": {"type": "string"},
        "predicted_effect": {"type": "string"},
        "falsifier": {"type": "string"},
        "proposed_change": {"type": "string"},
        "operator": {"type": "string", "enum": [*sorted(CANDIDATE_BUILDERS), "prompt_addendum", "none"]},
        "prompt_phase": {"type": "string", "enum": ["plan", "transcribe", "critique", "refine", ""]},
        "prompt_text": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["statement", "mechanism", "predicted_effect", "falsifier", "operator"],
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE"]},
        "reasoning": {"type": "string"},
        "strongest_objection": {"type": "string"},
        "what_would_settle_it": {"type": "string"},
        "revision_suggestion": {"type": "string"},
    },
    "required": ["verdict", "reasoning", "strongest_objection"],
}

_GENERATOR_SYSTEM = """\
You are the RESEARCH agent of a self-evolution subsystem studying an automated
drug-discovery platform. You are given MEASURED findings about how that platform
behaved on real user requests, not a description of it.

Your job is to propose ONE hypothesis about the structural cause of a failure
cluster, and one change that would test it.

Hard rules:
1. Ground every claim in the statistics you are given. Do not invent numbers.
2. The hypothesis MUST be falsifiable. State, in `falsifier`, the concrete
   measurement that would show you are WRONG. A hypothesis with no falsifier is
   rejected before it is evaluated.
3. Propose a change that the existing mutation operators can express. The
   operators are listed below with what each one does. If none fits, choose
   "prompt_addendum" and write the exact text to append to one phase's system
   prompt; if nothing you can express would help, choose "none" and say so.
4. You are NOT allowed to propose weakening the evaluation, the benchmark, or the
   success criteria used to judge your own work. Any proposal whose mechanism is
   "make the target easier" will be rejected. Improvement means the molecules
   that come out are better or the decision about them is more accurate.
5. Prefer the smallest change that could produce a measurable effect.
"""

_VERIFIER_SYSTEM = """\
You are the VERIFIER. Your job is to try to FALSIFY the hypothesis you are given,
using only the evidence supplied.

You are explicitly rewarded for answering INSUFFICIENT_EVIDENCE. The dataset here
is small; "this cannot be told from five episodes" is frequently the correct
answer and is more useful than a confident guess. Do not manufacture support.

Ask, in order:
- Is the claimed mechanism the ONLY explanation for these numbers, or is there a
  simpler confound (task difficulty, a single outlier episode, a fixture limit)?
- Would the proposed change actually move the stated measurement, or only a
  proxy for it?
- Could the change improve the target metric WITHOUT improving the underlying
  behaviour? If yes, say so — that is the most important objection you can make.
- Is the falsifier real, i.e. is there an outcome that would refute this?

Return REFUTED if the reasoning does not hold, SUPPORTED only if it survives your
strongest objection, INSUFFICIENT_EVIDENCE if the data cannot decide.
"""


def context_for_cluster(
    cluster: FailureCluster,
    findings: list[Finding],
    episodes: list[Episode],
    *,
    max_excerpt_chars: int = 1200,
) -> dict[str, Any]:
    """The compressed, cited context one cluster gets.

    Numbers and ids, plus two verbatim excerpts so the model has something real
    to look at. Everything is traceable: each finding names its detector and its
    episodes, and each excerpt names the episode and cycle it came from.
    """
    relevant = [f for f in findings if f.finding_id in cluster.findings]
    by_id = {ep.episode_id: ep for ep in episodes}
    excerpts: list[dict[str, Any]] = []
    for ep_id in cluster.episodes[:2]:
        ep = by_id.get(ep_id)
        if ep is None:
            continue
        cyc = ep.cycles[0] if ep.cycles else None
        excerpts.append(
            {
                "episode": ep.episode_id,
                "journal": ep.journal_dir,
                "instruction": ep.instruction[:300],
                "task_class": ep.task_class,
                "criteria": [(c.get("metric"), c.get("target")) for c in ep.criteria],
                "cycle_1": {
                    "n_criteria_met": cyc.n_criteria_met if cyc else None,
                    "n_criteria_total": cyc.n_criteria_total if cyc else None,
                    "metrics": cyc.metrics if cyc else {},
                    "verdict": cyc.verdict if cyc else None,
                    "diagnosis": (cyc.diagnosis[:max_excerpt_chars] if cyc else ""),
                    "notes": (cyc.notes if cyc else []),
                },
            }
        )
    return {
        "cluster": {
            "id": cluster.cluster_id,
            "label": cluster.label,
            "surface": cluster.surface,
            "severity": str(cluster.severity),
            "summary": cluster.summary,
        },
        "findings": [
            {
                "id": f.finding_id,
                "title": f.title,
                "detector": f.detector,
                "severity": str(f.severity),
                "affected": f"{f.n_affected}/{f.n_applicable}",
                "inference": f.inference,
                "stats": _trim(f.stats),
                "evidence_episodes": f.evidence,
                "detail": f.detail,
            }
            for f in relevant
        ],
        "excerpts": excerpts,
        "operators": {
            name: builder().hypothesis[:220] for name, builder in sorted(CANDIDATE_BUILDERS.items())
        },
        "operators_for_this_surface": BUILDERS_BY_SURFACE.get(cluster.surface, []),
        "corpus": {
            "n_episodes": len(episodes),
            "note": (
                "This is the entire historical record available. Any claim needing more than "
                f"{len(episodes)} episodes cannot be supported here — say so instead."
            ),
        },
    }


def _trim(stats: dict[str, Any], *, limit: int = 8) -> dict[str, Any]:
    """Keep statistics small enough to send without losing the numbers."""
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, list):
            out[key] = value[:limit]
            if len(value) > limit:
                out[f"{key}_truncated"] = f"{len(value) - limit} more not shown"
        elif isinstance(value, dict):
            out[key] = dict(list(value.items())[:limit])
        else:
            out[key] = value
    return out


class HypothesisEngine:
    """Generate → verify → revise, with a model or without one."""

    def __init__(self, cfg: EvolveConfig | None = None, registry: BackendRegistry | None = None) -> None:
        self.cfg = cfg or EvolveConfig()
        self.registry = registry or BackendRegistry(self.cfg)
        self.calls: list[dict[str, Any]] = []

    # -- generation ------------------------------------------------------
    def generate(self, context: dict[str, Any]) -> Hypothesis:
        cluster_id = context["cluster"]["id"]
        user = (
            "Failure cluster to explain:\n"
            + json.dumps(context, indent=2, default=str)[:24000]
            + "\n\nPropose ONE hypothesis and the change that would test it."
        )
        result = self.registry.complete(
            "deep_researcher", system=_GENERATOR_SYSTEM, user=user, schema=HYPOTHESIS_SCHEMA
        )
        self._note("deep_researcher", result)
        payload = _payload(result)
        if payload is None:
            return self._fallback(context, reason=result.error or "no structured output")
        return Hypothesis(
            hypothesis_id=f"hyp_{uuid.uuid4().hex[:10]}",
            cluster_id=cluster_id,
            statement=str(payload.get("statement") or ""),
            mechanism=str(payload.get("mechanism") or ""),
            predicted_effect=str(payload.get("predicted_effect") or ""),
            falsifier=str(payload.get("falsifier") or ""),
            proposed_change=json.dumps(
                {
                    "operator": payload.get("operator"),
                    "prompt_phase": payload.get("prompt_phase", ""),
                    "prompt_text": payload.get("prompt_text", ""),
                    "confidence": payload.get("confidence", ""),
                }
            ),
            candidate_class="PlanningPatch",
            author=f"{result.backend}:{result.model}",
            evidence=list(context["cluster"].get("episodes") or [])
            or [e for f in context["findings"] for e in f["evidence_episodes"]],
        )

    # -- verification ----------------------------------------------------
    def verify(self, hypothesis: Hypothesis, context: dict[str, Any]) -> Hypothesis:
        user = (
            "Hypothesis to falsify:\n"
            + json.dumps(
                {
                    "statement": hypothesis.statement,
                    "mechanism": hypothesis.mechanism,
                    "predicted_effect": hypothesis.predicted_effect,
                    "falsifier": hypothesis.falsifier,
                    "proposed_change": hypothesis.proposed_change,
                },
                indent=2,
            )
            + "\n\nEvidence available:\n"
            + json.dumps(context, indent=2, default=str)[:20000]
        )
        result = self.registry.complete("critic", system=_VERIFIER_SYSTEM, user=user, schema=VERDICT_SCHEMA)
        self._note("critic", result)
        payload = _payload(result)
        if payload is None:
            hypothesis.verdict = "INSUFFICIENT_EVIDENCE"
            hypothesis.verifier_notes = (
                f"the verifier could not be reached ({result.error or 'no output'}), so this "
                "hypothesis is unverified — recorded, not treated as supported"
            )
            return hypothesis
        hypothesis.verdict = str(payload.get("verdict") or "INSUFFICIENT_EVIDENCE")
        hypothesis.verifier_notes = json.dumps(
            {
                "reasoning": payload.get("reasoning"),
                "strongest_objection": payload.get("strongest_objection"),
                "what_would_settle_it": payload.get("what_would_settle_it"),
                "revision_suggestion": payload.get("revision_suggestion"),
            }
        )
        return hypothesis

    # -- revision --------------------------------------------------------
    def revise(self, hypothesis: Hypothesis, context: dict[str, Any]) -> Hypothesis:
        """One revision pass after a REFUTED verdict. One, not a search."""
        notes = hypothesis.verifier_notes
        user = (
            "Your hypothesis was refuted. Revise it or abandon it.\n\n"
            f"Original:\n{hypothesis.statement}\n\nMechanism:\n{hypothesis.mechanism}\n\n"
            f"The verifier said:\n{notes}\n\n"
            "Evidence:\n" + json.dumps(context, indent=2, default=str)[:16000] + "\n\n"
            'If the objection is fatal, set operator to "none" and say why in `statement`.'
        )
        result = self.registry.complete(
            "deep_researcher", system=_GENERATOR_SYSTEM, user=user, schema=HYPOTHESIS_SCHEMA
        )
        self._note("deep_researcher", result)
        payload = _payload(result)
        if payload is None:
            return hypothesis
        revised = Hypothesis(
            hypothesis_id=f"hyp_{uuid.uuid4().hex[:10]}",
            cluster_id=hypothesis.cluster_id,
            statement=str(payload.get("statement") or ""),
            mechanism=str(payload.get("mechanism") or ""),
            predicted_effect=str(payload.get("predicted_effect") or ""),
            falsifier=str(payload.get("falsifier") or ""),
            proposed_change=json.dumps(
                {
                    "operator": payload.get("operator"),
                    "prompt_phase": payload.get("prompt_phase", ""),
                    "prompt_text": payload.get("prompt_text", ""),
                }
            ),
            revision_of=hypothesis.hypothesis_id,
            author=f"{result.backend}:{result.model}",
            evidence=list(hypothesis.evidence),
        )
        return revised

    # -- turning a hypothesis into something evaluable --------------------
    def to_candidate(
        self, hypothesis: Hypothesis, *, parent: Candidate | None = None, generation: int = 0
    ) -> Candidate | None:
        """Build the candidate a hypothesis names, or ``None`` if it names nothing."""
        from ..candidates.library import build_candidate, build_prompt_addendum

        try:
            proposal = json.loads(hypothesis.proposed_change or "{}")
        except json.JSONDecodeError:
            proposal = {}
        operator = str(proposal.get("operator") or "none")
        common = {
            "parent_id": parent.candidate_id if parent else None,
            "generation": generation,
            "hypothesis_id": hypothesis.hypothesis_id,
            "evidence": list(hypothesis.evidence),
            "author": hypothesis.author,
        }
        if operator in CANDIDATE_BUILDERS:
            cand = build_candidate(operator, **common)
            cand.hypothesis = hypothesis.statement or cand.hypothesis
            return cand
        if operator == "prompt_addendum":
            text = str(proposal.get("prompt_text") or "").strip()
            phase = str(proposal.get("prompt_phase") or "plan")
            if not text:
                return None
            try:
                return build_prompt_addendum(
                    phase=phase,
                    text=text,
                    hypothesis=hypothesis.statement,
                    addressed=[hypothesis.cluster_id],
                    **common,
                )
            except ValueError:
                return None
        return None

    # -- fallback --------------------------------------------------------
    def _fallback(self, context: dict[str, Any], *, reason: str) -> Hypothesis:
        """When no model is reachable, the loop still proceeds — deterministically.

        The surface→operator table already encodes what the detectors found, so
        a cluster can be turned into a candidate with no model at all. This is
        not a hidden model: the hypothesis records that it was written by the
        deterministic fallback, and the archive shows that.
        """
        surface = context["cluster"]["surface"]
        options = BUILDERS_BY_SURFACE.get(surface, [])
        operator = options[0] if options else "none"
        return Hypothesis(
            hypothesis_id=f"hyp_{uuid.uuid4().hex[:10]}",
            cluster_id=context["cluster"]["id"],
            statement=(
                f"[deterministic fallback] The measured findings on surface {surface!r} are "
                f"addressed by the {operator!r} operator."
            ),
            mechanism=context["cluster"]["summary"],
            predicted_effect="see the operator's declared expected_benefit",
            falsifier="the benchmark comparison shows no improvement in decision_accuracy",
            proposed_change=json.dumps({"operator": operator}),
            author=f"deterministic-fallback ({reason})",
            evidence=[e for f in context["findings"] for e in f["evidence_episodes"]],
        )

    def _note(self, role: str, result: Any) -> None:
        self.calls.append(
            {
                "role": role,
                "backend": result.backend,
                "model": result.model,
                "ok": result.ok,
                "error": result.error,
                "elapsed_s": result.elapsed_s,
                "output_tokens": result.output_tokens,
                "cost_estimate_usd": result.cost_estimate_usd,
                "escalation_declined": (result.raw or {}).get("escalation_declined"),
                "ts": time.time(),
            }
        )


def _payload(result: Any) -> dict[str, Any] | None:
    """Structured output if there is any, else a JSON object parsed from the text."""
    if isinstance(result.parsed, dict):
        return result.parsed
    text = (result.text or "").strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


__all__ = ["HYPOTHESIS_SCHEMA", "VERDICT_SCHEMA", "HypothesisEngine", "context_for_cluster", "surface_of"]
