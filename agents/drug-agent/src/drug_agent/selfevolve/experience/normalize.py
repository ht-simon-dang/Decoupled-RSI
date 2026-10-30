"""Journal → :class:`~drug_agent.selfevolve.schemas.Episode`.

The platform already writes an excellent record of every request; this module
does not replace it, it reads it. The raw journal stays exactly where it is and
remains the citable source — an :class:`Episode` carries ``journal_dir`` so any
claim downstream can be walked back to bytes on disk.

**Nothing here imputes a missing field.** The journal for a request that crashed
before its first cycle has no cycles; the episode gets an empty list, not a
zero. A step whose ``eval_count`` the model did not report gets ``None``, not 0.
Every analysis in this package is written to expect that, because the alternative
— a plausible default — turns "we don't know" into "we measured zero", and the
whole point of the loop is that its conclusions can be trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..schemas import CycleRecord, Episode, Step, ToolUse

_MANIFEST = "manifest.json"
_CHAT = "chat.jsonl"
_EVENTS = "events.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a killed process is not an error
    return out


def _prompt_sha(messages: list[dict[str, str]]) -> str:
    blob = "\n\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in messages or [])
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def task_class_of(requirements: dict[str, Any] | None, instruction: str) -> str:
    """The coarse class every comparison conditions on.

    Tool attribution that ignores task class will conclude that the tools used on
    hard tasks cause failure. This is the cheapest honest way to avoid that, and
    it is *derived* — read off the transcribed requirements when the journal has
    them, and only falls back to keywords in the instruction when it does not.
    """
    if requirements:
        kind = str(requirements.get("kind") or "")
        if kind == "phenotype_based":
            return "phenotype_rescue"
        gen = str(requirements.get("generator") or "library")
        if "pfm" in gen:
            return "target_denovo"
        return "target_library"
    low = (instruction or "").lower()
    if any(w in low for w in ("phenotype", "reverse", "restor", "signature", "endotheli")):
        return "phenotype_rescue"
    if any(w in low for w in ("de-novo", "de novo", "from scratch", "novel chemotype", "new chemotype")):
        return "target_denovo"
    if instruction:
        return "target_library"
    return "unknown"


def _steps_from_chat(chat: list[dict[str, Any]]) -> list[Step]:
    steps: list[Step] = []
    for i, entry in enumerate(chat):
        parsed = entry.get("parsed")
        steps.append(
            Step(
                index=i,
                phase=str(entry.get("phase") or ""),
                cycle=int(entry.get("cycle") or 0),
                attempt=int(entry.get("attempt") or 0),
                mode=str(entry.get("mode") or ""),
                decode=str(entry.get("decode") or ""),
                ok=bool(entry.get("ok", True)),
                error=entry.get("error"),
                elapsed_s=float(entry.get("elapsed_s") or 0.0),
                eval_tokens=entry.get("eval_count"),
                prompt_tokens=entry.get("prompt_eval_count"),
                done_reason=entry.get("done_reason"),
                n_repairs=len(entry.get("repairs") or []),
                thinking_chars=len(entry.get("thinking") or ""),
                content_chars=len(entry.get("content") or ""),
                parsed=parsed if isinstance(parsed, dict) else None,
                prompt_sha256=_prompt_sha(entry.get("messages") or []),
            )
        )
    return steps


def _tools_from_manifest(manifest: dict[str, Any]) -> list[ToolUse]:
    out: list[ToolUse] = []
    for t in (manifest.get("tools") or {}).get("by_tool", []):
        metrics = {k: (v or {}).get("mean") for k, v in (t.get("metrics") or {}).items()}
        out.append(
            ToolUse(
                tool_id=str(t.get("tool_id")),
                operation=str(t.get("operation")),
                backend=t.get("backend"),
                n_calls=int(t.get("n_calls") or 0),
                n_errors=int(t.get("n_errors") or 0),
                total_ms=float(t.get("total_ms") or 0.0),
                mean_ms=t.get("mean_ms"),
                metrics=metrics,
                errors=list(t.get("errors") or []),
            )
        )
    return out


def _cycles_from_events(events: list[dict[str, Any]], manifest: dict[str, Any]) -> list[CycleRecord]:
    """Reconstruct per-cycle records from the typed event stream.

    The manifest's ``outcome.scoreboard`` carries the same numbers, but only the
    event stream has the critique, the diagnosis and the adjustment diff — and
    the gap between "what the model said it changed" and "what actually changed"
    is one of the findings this whole package exists to surface.
    """
    by_cycle: dict[int, CycleRecord] = {}

    def slot(n: int) -> CycleRecord:
        return by_cycle.setdefault(n, CycleRecord(cycle=n))

    for e in events:
        kind = e.get("type")
        cyc = int(e.get("cycle") or 0)
        data = e.get("data") or {}
        if kind == "evaluation_result":
            rec = slot(cyc)
            report = data.get("report") or {}
            rec.n_criteria_met = report.get("n_criteria_met")
            rec.n_criteria_total = report.get("n_criteria_total")
            rec.shortlist_size = report.get("shortlist_size")
            rec.n_candidates = report.get("n_candidates")
            rec.headline_utility = report.get("headline_utility")
            rec.metrics = {
                str(c.get("key")): c.get("value") for c in (report.get("criteria") or []) if c.get("key")
            }
            rec.notes = [str(n) for n in (report.get("notes") or [])]
            # The crash reason, kept as its own field rather than left to be
            # regexed back out of `notes`. A cycle that never reached scoring is
            # a different event from one that scored badly, and the corpus could
            # not previously distinguish them.
            rec.failure_reason = report.get("failure_reason")
            rec.pipeline_state = report.get("state")
        elif kind == "metrics":
            rec = slot(cyc)
            point = data.get("point") or {}
            for key in ("max_pIC50", "hard_constraint_pass_rate", "diversity"):
                if point.get(key) is not None:
                    rec.metrics.setdefault(key, point[key])
            if rec.headline_utility is None:
                rec.headline_utility = point.get("headline_utility")
        elif kind == "critique_ready":
            rec = slot(cyc)
            crit = data.get("critique") or {}
            rec.verdict = crit.get("overall_verdict")
            rec.diagnosis = str(crit.get("diagnosis") or "")
        elif kind == "decision":
            rec = slot(cyc)
            rec.verdict = data.get("verdict") or rec.verdict
        elif kind == "requirements_adjusted":
            rec = slot(cyc)
            rec.adjustment_diff = list(data.get("diff") or [])
            rec.adjustment_changes = list(data.get("changes") or [])

    # Fall back to the manifest scoreboard for cycles the event log lost.
    for point in (manifest.get("outcome") or {}).get("scoreboard", []):
        rec = slot(int(point.get("cycle") or 0))
        if rec.n_criteria_met is None:
            rec.n_criteria_met = point.get("n_criteria_met")
            rec.n_criteria_total = point.get("n_criteria_total")
            rec.headline_utility = point.get("headline_utility")
            rec.shortlist_size = point.get("shortlist_size")
    for run in manifest.get("pipeline_runs") or []:
        rec = slot(int(run.get("cycle") or 0))
        if rec.n_candidates is None:
            rec.n_candidates = run.get("n_candidates")

    return [by_cycle[k] for k in sorted(by_cycle) if k]


def _plan_from_events(events: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict | None]:
    criteria: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    requirements: dict[str, Any] | None = None
    for e in events:
        if e.get("type") == "plan_ready":
            data = e.get("data") or {}
            criteria = list(data.get("criteria") or [])
            unscorable = list(data.get("unscorable") or [])
        elif e.get("type") == "requirements_ready":
            requirements = (e.get("data") or {}).get("requirements")
    return criteria, unscorable, requirements


#: Sources that are the evolver's own experiments rather than user requests.
#: A benchmark run must never be studied as experience: it would let the
#: candidate generator read the holdout instructions it is measured on.
EXCLUDED_SOURCES = frozenset({"selfevolve-benchmark"})

#: Only journals of this kind are experience. A pipeline entry records a brief
#: executed directly through the CLI, the API or a script: real history, but no
#: instruction, no plan and no model exchanges, so normalizing it would add
#: empty episodes to every count the analysis reports.
INCLUDED_KINDS = frozenset({"request", ""})


def normalize_request(journal_dir: str | Path) -> Episode | None:
    """Read one request journal directory into an :class:`Episode`.

    Returns ``None`` when there is no manifest — an in-flight or half-written
    request is not an episode, and pretending otherwise would put a phantom into
    every rate this package computes — or when the journal is one of the
    evolver's own benchmark runs, which are experiments, not experience.
    """
    d = Path(journal_dir)
    man_path = d / _MANIFEST
    if not man_path.is_file():
        return None
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if str(manifest.get("source") or "") in EXCLUDED_SOURCES:
        return None
    # Absent for journals written before the kind was recorded: those are all
    # master-agent requests, which is what the empty string admits.
    if str(manifest.get("kind") or "") not in INCLUDED_KINDS:
        return None

    chat = _read_jsonl(d / _CHAT)
    events = _read_jsonl(d / _EVENTS)
    criteria, unscorable, requirements = _plan_from_events(events)
    config = manifest.get("config") or {}
    instruction = str(manifest.get("instruction") or "")

    # The prompt version only exists for episodes produced after the runtime
    # policy landed. Older episodes honestly report None rather than "default":
    # we know their prompts were the pre-policy ones, but not that they hash to
    # the current default, and the difference matters when comparing arms.
    policy = config.get("policy") or {}
    prompt_version = policy.get("prompt_version") or config.get("prompt_version")

    return Episode(
        episode_id=str(manifest.get("request_id") or d.name),
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        duration_s=manifest.get("duration_s"),
        source=str(manifest.get("source") or "unknown"),
        status=str(manifest.get("status") or "UNKNOWN"),
        instruction=instruction,
        task_class=task_class_of(requirements, instruction),
        model=config.get("model"),
        strategy=config.get("strategy") or config.get("refine_strategy"),
        generator=config.get("generator"),
        max_cycles=config.get("max_cycles"),
        prompt_version=prompt_version,
        criteria=criteria,
        unscorable_criteria=unscorable,
        requirements=requirements,
        steps=_steps_from_chat(chat),
        tools=_tools_from_manifest(manifest),
        cycles=_cycles_from_events(events, manifest),
        outcome=manifest.get("outcome") or {},
        error=manifest.get("error"),
        journal_dir=str(d),
    )


__all__ = ["EXCLUDED_SOURCES", "normalize_request", "task_class_of"]
