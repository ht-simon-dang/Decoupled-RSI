"""What the loop proposed, folded into one method per distinct idea.

The raw record is deliberately granular: a hypothesis per cycle, a candidate per
admitted change, an evaluation record per arm, a decision per comparison. That is
right for auditing and wrong for reading — six cycles produced seven hypotheses
that between them describe **four** distinct methods, because the archive kept
refusing to re-evaluate an idea it had already tested.

This module folds them back together. A *method* is one distinct change, keyed by
the candidate's novelty descriptor (the surfaces it touches and their values, with
prose collapsed) — the same key the archive uses to refuse a duplicate. Every
hypothesis that arrived at that method is attached to it, with its own verdict, so
"three separate cycles independently proposed this" is visible rather than being
three rows that look unrelated.

Nothing here recomputes or re-judges. It reads what the loop wrote and arranges
it; the verdicts are the gate's, and where the gate never ran, it says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import EvolveConfig

#: A human name for each overlay surface, so a method reads as a method rather
#: than as a config key.
SURFACE_TITLE: dict[str, str] = {
    "capability_envelope": "Publish the deployment's measured capability envelope to the planner",
    "capability_envelope_transcribe": "Publish the envelope to the transcriber too",
    "goal_aware_criteria": "Drop efficacy criteria the program's goal kind cannot produce",
    "refine_carry_forward": "Carry forward requirements the refiner never said it was changing",
    "refine_max_drop_fraction": "Cap how much of a requirement set one refinement may remove",
    "plan_prompt_addendum": "Append guidance to the planning prompt",
    "transcribe_prompt_addendum": "Append guidance to the transcription prompt",
    "critique_prompt_addendum": "Append guidance to the critique prompt",
    "refine_prompt_addendum": "Append guidance to the refinement prompt",
}


@dataclass
class Method:
    """One distinct change the loop proposed, with everything known about it."""

    method_id: str
    title: str
    novelty: str
    candidate_class: str = ""
    cell: str = ""
    surfaces: list[str] = field(default_factory=list)
    overlay: dict[str, Any] = field(default_factory=dict)
    #: Every candidate that is this method (the archive keeps only the first).
    candidates: list[dict[str, Any]] = field(default_factory=list)
    #: Every hypothesis that arrived at it, with its verifier verdict.
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    proposed_by: list[str] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    expected_benefit: str = ""
    declared_regressions: list[str] = field(default_factory=list)
    #: What each gate said, or None where a gate never ran for this method.
    gate0: dict[str, Any] | None = None
    gate1: dict[str, Any] | None = None
    gate2: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] | None = None
    status: str = "DRAFT"
    outcome: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return d


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


#: Surfaces that qualify another surface rather than standing on their own, so a
#: composition reads by what it does, not by every knob it sets.
_MODIFIER_SURFACES = frozenset({"refine_max_drop_fraction", "capability_envelope_transcribe"})


def _title_for(overlay: dict[str, Any], candidate_class: str) -> str:
    """A readable name for a method, distinct across compositions.

    Three of the methods here set ``capability_envelope``; naming them all after
    that one surface would collapse three different experiments into one line.
    So the title names every load-bearing surface, and falls back to the author's
    own ``policy_id`` when the overlay says nothing nameable.
    """
    surfaces = [
        k for k in overlay if k not in ("policy_id", "notes", "rationale") and k not in _MODIFIER_SURFACES
    ]
    named = [SURFACE_TITLE[s] for s in surfaces if s in SURFACE_TITLE]
    if not named:
        label = str(overlay.get("policy_id") or "").strip()
        return f"{candidate_class}: {label}" if label else f"{candidate_class} (no named surface)"
    if len(named) == 1:
        return named[0]
    head, *rest = named
    return head + " + " + " + ".join(_short(r) for r in rest)


def _short(title: str) -> str:
    """The distinguishing half of a surface title, for composing into one line."""
    return title[0].lower() + title[1:] if title else title


def summarize_methods(cfg: EvolveConfig | None = None) -> dict[str, Any]:
    """Fold hypotheses, candidates, gates and decisions into one row per method."""
    from .archive import Archive
    from .evaluation.kernel import EvaluationKernel

    cfg = cfg or EvolveConfig()
    store = cfg.resolved_store_dir()
    arc = Archive(cfg)
    kernel = EvaluationKernel(cfg)

    candidates = arc.all()
    records = kernel.list_records()
    cycles = _read_cycles(store / "cycles.jsonl")
    hypotheses = _read_hypotheses(store / "hypotheses")

    # candidate id -> the hypothesis that produced it, from the cycle log
    hyp_by_candidate: dict[str, str] = {}
    skipped_to: dict[str, str] = {}
    for row in cycles:
        cid, hid = row.get("candidate_id"), row.get("hypothesis_id")
        if cid and hid:
            hyp_by_candidate[cid] = hid
        if row.get("decision") == "SKIPPED_NOT_NOVEL" and cid:
            # "identical to cand_x" — the reason names the method it duplicated.
            reason = str(row.get("reason") or "")
            marker = "identical to "
            if marker in reason:
                skipped_to[cid] = reason.split(marker, 1)[1].split(" ", 1)[0].strip()

    methods: dict[str, Method] = {}

    def slot(cand: Any) -> Method:
        key = cand.novelty_descriptor()
        if key not in methods:
            methods[key] = Method(
                method_id=cand.candidate_id,
                title=_title_for(cand.overlay, str(cand.candidate_class)),
                novelty=key,
                candidate_class=str(cand.candidate_class),
                cell=cand.cell(),
                surfaces=cand.surfaces(),
                overlay=dict(cand.overlay),
            )
        return methods[key]

    for cand in candidates:
        m = slot(cand)
        m.candidates.append(
            {
                "candidate_id": cand.candidate_id,
                "generation": cand.generation,
                "parent_id": cand.parent_id,
                "status": str(cand.status),
                "author": cand.author,
                "created_at": cand.created_at,
                "notes": cand.notes,
            }
        )
        if cand.author and cand.author not in m.proposed_by:
            m.proposed_by.append(cand.author)
        for fm in cand.failure_modes_addressed:
            if fm not in m.addresses:
                m.addresses.append(fm)
        m.expected_benefit = m.expected_benefit or cand.expected_benefit
        for r in cand.possible_regressions:
            if r not in m.declared_regressions:
                m.declared_regressions.append(r)

        # gates, per candidate id
        m.gate0 = m.gate0 or _read_json(store / f"gate0_{cand.candidate_id}.json")
        m.gate1 = m.gate1 or _read_json(store / f"replay_{cand.candidate_id}.json")
        for rec in records:
            if rec.candidate_id == cand.candidate_id and rec.arm != "baseline":
                m.gate2.append(
                    {
                        "record_id": rec.record_id,
                        "arm": rec.arm,
                        "kernel_verified": rec.kernel_verified,
                        "aggregates": rec.aggregates,
                    }
                )
        m.decision = m.decision or _read_json(store / "decisions" / f"{cand.candidate_id}.json")
        # The most advanced status any of this method's candidates reached.
        m.status = _best_status(m.status, str(cand.status))

    # attach hypotheses — directly, through a skipped duplicate, and back up any
    # revision chain, so a hypothesis the verifier refuted is filed with the
    # revision that replaced it rather than looking like an orphan.
    def _chain(hyp_id: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        cur: str | None = hyp_id
        while cur and cur not in seen and cur in hypotheses:
            seen.add(cur)
            out.append(cur)
            cur = hypotheses[cur].get("revision_of")
        return list(reversed(out))  # oldest first — the reasoning as it developed

    for cand_id, hyp_id in hyp_by_candidate.items():
        target = skipped_to.get(cand_id, cand_id)
        for m in methods.values():
            if not any(c["candidate_id"] == target for c in m.candidates):
                continue
            for hid in _chain(hyp_id):
                if any(h["hypothesis_id"] == hid for h in m.hypotheses):
                    continue
                hyp = hypotheses[hid]
                m.hypotheses.append(
                    {
                        "hypothesis_id": hid,
                        "verdict": hyp.get("verdict"),
                        "author": hyp.get("author"),
                        "revision_of": hyp.get("revision_of"),
                        "statement": hyp.get("statement", ""),
                        "mechanism": hyp.get("mechanism", ""),
                        "predicted_effect": hyp.get("predicted_effect", ""),
                        "falsifier": hyp.get("falsifier", ""),
                        "arrived_via": cand_id if cand_id != target else None,
                    }
                )
                if hyp.get("author") and hyp["author"] not in m.proposed_by:
                    m.proposed_by.append(hyp["author"])
            break

    for m in methods.values():
        m.outcome = _outcome_line(m)

    ordered = sorted(
        methods.values(),
        key=lambda m: (-_status_rank(m.status), -len(m.gate2), m.method_id),
    )
    # Hypotheses that never became a method at all.
    attached = {h["hypothesis_id"] for m in methods.values() for h in m.hypotheses}
    orphans = [
        {"hypothesis_id": k, **{f: v.get(f) for f in ("verdict", "author", "statement")}}
        for k, v in hypotheses.items()
        if k not in attached
    ]

    return {
        "n_methods": len(ordered),
        "n_hypotheses": len(hypotheses),
        "n_candidates": len(candidates),
        "n_cycles": len(cycles),
        "n_arms": len(records),
        "methods": [m.as_dict() for m in ordered],
        "unattached_hypotheses": orphans,
        "store": str(store),
    }


_STATUS_ORDER = ["DRAFT", "QUARANTINED", "REJECTED", "ARCHIVED", "EVALUATED", "PROMOTED"]


def _status_rank(status: str) -> int:
    return _STATUS_ORDER.index(status) if status in _STATUS_ORDER else 0


def _best_status(a: str, b: str) -> str:
    return a if _status_rank(a) >= _status_rank(b) else b


def _outcome_line(m: Method) -> str:
    """One honest sentence about where this method got to.

    The gate-0 *capture* was only added part-way through the run, so an early
    candidate can have failed gate 0 with no ``gate0_*.json`` beside it — its
    archive note is the only record. Reading both is what keeps this line honest
    rather than optimistic.
    """
    if m.decision:
        failed = [c["check"] for c in m.decision.get("checks", []) if not c["passed"]]
        return (
            f"{m.decision['decision']} — "
            + (f"failed {failed}" if failed else "every gate check passed")
            + (" (human approval required)" if m.decision.get("requires_human") else "")
        )
    if m.gate2:
        return "benchmarked, but never compared against a baseline"
    if m.status == "QUARANTINED":
        return "quarantined at admission — it touched the evaluation kernel"
    notes = " ".join(str(c.get("notes") or "") for c in m.candidates).lower()
    gate0_failed = (m.gate0 is not None and not m.gate0.get("passed")) or "gate 0" in notes
    if gate0_failed:
        captured = " (capture in `gates/`)" if m.gate0 else " (no capture — this predates the fix)"
        return "stopped at gate 0 — the platform's test suite failed, so nothing was measured" + captured
    if m.gate1 and m.gate1.get("applicable"):
        return "gate 1 only — deterministic replay recorded, benchmark not run"
    return "built and admitted; never evaluated"


def _read_cycles(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _read_hypotheses(d: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict) and payload.get("hypothesis_id"):
            out[payload["hypothesis_id"]] = payload
    return out


# ---------------------------------------------------------------------------
def render_methods(summary: dict[str, Any]) -> str:
    """The summary as Markdown — the document a human reads."""
    lines = [
        "# Methods proposed during self-evolution",
        "",
        f"{summary['n_methods']} distinct methods, from {summary['n_hypotheses']} hypotheses across "
        f"{summary['n_cycles']} research cycles and {summary['n_candidates']} candidates. "
        f"{summary['n_arms']} benchmark arms were measured.",
        "",
        "A *method* is one distinct change, keyed the way the archive keys it: the surfaces it "
        "touches and their values, with prose collapsed. Several hypotheses arriving at the same "
        "method are folded into one entry — that convergence is a finding, not a duplicate.",
        "",
        "| # | method | class | status | outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, m in enumerate(summary["methods"], 1):
        lines.append(f"| {i} | {m['title']} | `{m['candidate_class']}` | {m['status']} | {m['outcome']} |")
    if summary["unattached_hypotheses"]:
        lines += [
            "",
            f"{len(summary['unattached_hypotheses'])} hypotheses produced no method "
            "(refuted twice, or naming no implementable operator).",
        ]

    for i, m in enumerate(summary["methods"], 1):
        lines += ["", "---", "", f"## {i}. {m['title']}", ""]
        lines += [
            f"- **class** `{m['candidate_class']}` · **cell** `{m['cell']}` · **status** {m['status']}",
            f"- **surfaces changed** {', '.join('`' + s + '`' for s in m['surfaces']) or '—'}",
            f"- **addresses** {', '.join('`' + a + '`' for a in m['addresses']) or '—'}",
            f"- **proposed by** {', '.join(m['proposed_by']) or '—'}",
        ]
        if m["expected_benefit"]:
            lines += ["", f"**Expected.** {m['expected_benefit']}"]

        if m["hypotheses"]:
            lines += ["", "### The reasoning behind it", ""]
            for h in m["hypotheses"]:
                via = f" (reached again via {h['arrived_via']})" if h.get("arrived_via") else ""
                rev = f", revision of `{h['revision_of']}`" if h.get("revision_of") else ""
                lines += [
                    f"**`{h['hypothesis_id']}`** — verifier said **{h['verdict']}**{rev} · "
                    f"{h['author']}{via}",
                    "",
                    "> " + _wrap(h["statement"]),
                    "",
                ]
                if h.get("falsifier"):
                    lines += ["*How it could be shown wrong:* " + _wrap(h["falsifier"], 600), ""]

        if m["declared_regressions"]:
            lines += ["### Regressions it declared in advance", ""]
            lines += [f"- {r}" for r in m["declared_regressions"]]
            lines += [""]

        lines += ["### What the gates said", ""]
        g0 = m["gate0"]
        if g0:
            for c in g0.get("commands", []):
                lines.append(
                    f"- **gate 0** `{c['label']}` exit {c['exit_status']} "
                    f"({c['elapsed_s']} s) — {'passed' if c['ok'] else '**FAILED**'}"
                )
        else:
            lines.append("- **gate 0** not run for this method")
        g1 = m["gate1"]
        if g1 and g1.get("applicable"):
            for suite in g1.get("suites_run", []):
                base = (g1.get("baseline", {}).get("suites") or {}).get(suite, {})
                cand = (g1.get("candidate", {}).get("suites") or {}).get(suite, {})
                deltas = [
                    f"{k} {base.get(k)} → {cand.get(k)}"
                    for k in sorted(cand)
                    if k.startswith("n_") and k != "n_cases" and base.get(k) != cand.get(k)
                ]
                lines.append(f"- **gate 1** `{suite}` — " + ("; ".join(deltas) if deltas else "no change"))
        elif g1:
            lines.append("- **gate 1** not applicable — this method moves nothing the replay suites measure")
        else:
            lines.append("- **gate 1** not run for this method")
        if m["gate2"]:
            for r in m["gate2"]:
                a = r["aggregates"] or {}
                lines.append(
                    f"- **gate 2** `{r['record_id']}` — gold_score {a.get('gold_score')}, "
                    f"decision_accuracy {a.get('decision_accuracy')}, "
                    f"accepts {a.get('n_accept')}, false accepts {a.get('n_false_accept')} "
                    f"(kernel {'verified' if r['kernel_verified'] else '**VOID**'})"
                )
        else:
            lines.append("- **gate 2** no benchmark arm was run for this method")

        d = m["decision"]
        if d:
            lines += ["", "### Promotion gate", "", "| check | result |", "| --- | --- |"]
            for c in d.get("checks", []):
                lines.append(f"| `{c['check']}` | {'PASS' if c['passed'] else '**FAIL**'} |")
            lines += ["", f"**Decision: {d['decision']}**"]
            for r in d.get("reasons", []):
                lines.append(f"- {r}")
        lines += ["", f"**Outcome.** {m['outcome']}"]
        lines += ["", "```json", json.dumps(m["overlay"], indent=2), "```"]

    if summary["unattached_hypotheses"]:
        lines += ["", "---", "", "## Hypotheses that produced no method", ""]
        for h in summary["unattached_hypotheses"]:
            lines += [f"- **`{h['hypothesis_id']}`** ({h['verdict']}) — " + _wrap(h["statement"], 300)]
    return "\n".join(lines) + "\n"


def _wrap(text: str, limit: int = 900) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + " …"


__all__ = ["Method", "SURFACE_TITLE", "render_methods", "summarize_methods"]
