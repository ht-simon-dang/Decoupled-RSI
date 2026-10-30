"""Turn the running history into a training set.

The journals already hold, verbatim, every message sent to the model and every
reply that came back. That is a supervised fine-tuning corpus in all but format,
and this module writes the format: one JSON object per exchange, messages in the
conventional ``[{role, content}, …]`` shape, with the metadata needed to filter.

Three decisions worth stating, because each one is a place a training set
usually goes quietly wrong:

**The reasoning channel is kept out of ``content``.** A model that streamed its
thinking produced two things, and splicing them into one assistant turn teaches
a format nobody asked for. ``thinking`` rides alongside, so a dataset builder
chooses.

**Failures are kept, and labelled.** An exchange the decoder could not parse is
exactly what a repair-trained model needs to see, so it is exported with
``ok: false`` rather than dropped. ``--only-ok`` drops them at build time when
the goal is imitation of successes only.

**Nothing is deduplicated or shuffled here.** Both are dataset decisions that
depend on the split, and doing them silently at export time would hide how
little independent data five requests actually are.

The exported records inherit the journals' redaction, and no more: they contain
verbatim user instructions and verbatim model output.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .journal import TRAINING, default_journal_root, iter_request_dirs

_MANIFEST = "manifest.json"
_CHAT = "chat.jsonl"
_DIALOGS = "dialogs.jsonl"
_CARD = "DATASET_CARD.md"

#: Bump when the exported record shape changes.
EXPORT_VERSION = 1


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
                continue
    return out


def iter_dialogs(
    root: Path | str | None = None,
    *,
    only_ok: bool = False,
    phases: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    """One training record per recorded model exchange, oldest request first."""
    base = Path(root) if root is not None else default_journal_root()
    dirs = sorted(
        iter_request_dirs(base),
        key=lambda d: _read_manifest(d).get("started_at") or 0.0,
    )
    for d in dirs:
        manifest = _read_manifest(d)
        request_id = str(manifest.get("request_id") or d.name)
        instruction = manifest.get("instruction") or ""
        config = manifest.get("config") or {}
        for i, ex in enumerate(_read_jsonl(d / _CHAT)):
            if only_ok and not ex.get("ok", True):
                continue
            if phases and str(ex.get("phase") or "") not in phases:
                continue
            messages = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in (ex.get("messages") or [])
                if m.get("content")
            ]
            if not messages:
                continue  # an exchange with no prompt teaches nothing
            messages.append({"role": "assistant", "content": str(ex.get("content") or "")})
            yield {
                "export_version": EXPORT_VERSION,
                "id": f"{request_id}#{i:04d}",
                "request_id": request_id,
                "messages": messages,
                "thinking": ex.get("thinking") or "",
                "phase": ex.get("phase"),
                "cycle": ex.get("cycle"),
                "attempt": ex.get("attempt"),
                "model": ex.get("model") or config.get("model"),
                "meta": {
                    "ok": bool(ex.get("ok", True)),
                    "decode": ex.get("decode"),
                    "mode": ex.get("mode"),
                    "error": ex.get("error"),
                    "elapsed_s": ex.get("elapsed_s"),
                    "eval_count": ex.get("eval_count"),
                    "prompt_eval_count": ex.get("prompt_eval_count"),
                    "done_reason": ex.get("done_reason"),
                    "n_repairs": len(ex.get("repairs") or []),
                    "request_instruction": instruction,
                    "request_status": manifest.get("status"),
                    "request_started_at": manifest.get("started_at"),
                    "strategy": config.get("strategy"),
                    "generator": config.get("generator"),
                    "journal_dir": str(d),
                },
            }


def export_dialogs(
    root: Path | str | None = None,
    *,
    out_dir: Path | str | None = None,
    only_ok: bool = False,
    phases: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write ``training/dialogs.jsonl`` plus a dataset card. Returns the counts."""
    base = Path(root) if root is not None else default_journal_root()
    out = Path(out_dir) if out_dir is not None else base / TRAINING
    out.mkdir(parents=True, exist_ok=True)

    by_phase: dict[str, int] = {}
    by_request: dict[str, int] = {}
    by_model: dict[str, int] = {}
    n_failed = 0
    n_thinking = 0
    n_chars = 0

    path = out / _DIALOGS
    with path.open("w", encoding="utf-8") as fh:
        for rec in iter_dialogs(base, only_ok=only_ok, phases=phases):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            by_phase[str(rec.get("phase"))] = by_phase.get(str(rec.get("phase")), 0) + 1
            by_request[rec["request_id"]] = by_request.get(rec["request_id"], 0) + 1
            by_model[str(rec.get("model"))] = by_model.get(str(rec.get("model")), 0) + 1
            n_failed += 0 if rec["meta"]["ok"] else 1
            n_thinking += 1 if rec["thinking"] else 0
            n_chars += sum(len(m["content"]) for m in rec["messages"])

    stats = {
        "root": str(base),
        "out": str(path),
        "n_dialogs": sum(by_request.values()),
        "n_requests": len(by_request),
        "n_undecodable": n_failed,
        "n_with_thinking": n_thinking,
        "total_chars": n_chars,
        "by_phase": dict(sorted(by_phase.items())),
        "by_model": dict(sorted(by_model.items())),
        "only_ok": only_ok,
        "phases": list(phases) if phases else None,
    }
    (out / _CARD).write_text(_card(stats), encoding="utf-8")
    return stats


def _card(stats: dict[str, Any]) -> str:
    phases = "\n".join(f"| `{k}` | {v} |" for k, v in stats["by_phase"].items()) or "| — | — |"
    models = ", ".join(f"`{k}` ({v})" for k, v in stats["by_model"].items()) or "—"
    return f"""# Dataset card — dialogs.jsonl

Generated {time.strftime("%Y-%m-%d %H:%M")} by `drugagent journal export` from
`{stats["root"]}`. Regenerate rather than edit.

## What it is

{stats["n_dialogs"]} model exchanges drawn from {stats["n_requests"]} requests,
one JSON object per line. Every record is a real exchange that happened while
serving a real request — prompts as they were sent, replies as they came back.

| field | meaning |
| --- | --- |
| `messages` | `[{{role, content}}]` — the prompt as sent, plus the assistant reply |
| `thinking` | the model's reasoning channel, kept **out** of the assistant turn |
| `phase` | which step of the program produced it (plan · transcribe · refine · critique) |
| `meta.ok` / `meta.decode` | whether the reply decoded, and by which path |
| `meta.eval_count` | generated tokens the server reported (may be `null`) |
| `meta.journal_dir` | the journal this record came from — every row is walkable back to bytes |

## Composition

| phase | records |
| --- | --: |
{phases}

- models: {models}
- undecodable replies retained: {stats["n_undecodable"]}
  {"(excluded — built with --only-ok)" if stats["only_ok"] else "(kept, labelled `ok: false`)"}
- records carrying a reasoning channel: {stats["n_with_thinking"]}
- total prompt+reply characters: {stats["total_chars"]:,}

## Limits you must not train through

- **It is small, and it is not independent.** {stats["n_requests"]} requests
  against one deployment, largely one target family. Exchanges from the same
  request share a system prompt and a task; treat the *request* as the unit when
  splitting, never the exchange, or the split leaks.
- **Not deduplicated, not shuffled, not split.** Deliberately — those are
  decisions for whoever builds the training run, and doing them here would hide
  how little independent signal is present.
- **It records what the deployment did, not what was correct.** A reply that
  decoded cleanly is not a reply that was scientifically right. Nothing here has
  been graded; the self-evolution loop's evaluation records are where judged
  outcomes live.
- **Verbatim user and model text.** Secret redaction was applied when the
  journals were written; nothing else was filtered. Review before sharing.
"""


def _read_manifest(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
