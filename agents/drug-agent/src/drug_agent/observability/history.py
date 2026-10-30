"""Housekeeping for the running-history folder.

The journal itself (:mod:`.journal`) is append-only and knows how to write one
request. This module owns the things that make the folder a *repository* rather
than a pile of directories:

* ``INDEX.md`` — every request as a table a person can read, newest first;
* ``README.md`` — what the folder is, written once so the folder explains itself;
* ``rebuild_index`` — regenerate both indexes by scanning the request
  directories, for when the index is lost, hand-edited, or newly migrated;
* ``migrate`` — move journals written under an older root or layout into this
  one, without rewriting a single byte of their contents.

Nothing here ever edits a request's own files. The manifests, chat logs, tool
ledgers and event streams are the record; these are views over it, and any one
of them can be deleted and regenerated.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .journal import (
    REQUESTS,
    TRAINING,
    default_journal_root,
    iter_request_dirs,
    request_dirname,
)

_MANIFEST = "manifest.json"
_INDEX = "index.jsonl"
_INDEX_MD = "INDEX.md"
_README = "README.md"
#: Index rows whose request directory no longer exists are kept, not dropped:
#: a missing payload is a fact about the history, and silently deleting the row
#: would erase the evidence that the request ever ran.
_ORPHANS = "orphaned-index-rows.jsonl"


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


def _row_from_manifest(manifest: dict[str, Any], rel_dir: str) -> dict[str, Any]:
    """The index row for a request, derived from its own manifest.

    Same shape the journal appends live, so a rebuilt index and an accumulated
    one are indistinguishable to every reader.
    """
    llm = manifest.get("llm") or {}
    tools = manifest.get("tools") or {}
    config = manifest.get("config") or {}
    started = manifest.get("started_at")
    finished = manifest.get("finished_at")
    return {
        "request_id": manifest.get("request_id"),
        "kind": manifest.get("kind") or "request",
        "dir": rel_dir,
        "started_at": started,
        "finished_at": finished,
        "duration_s": manifest.get("duration_s"),
        "status": manifest.get("status"),
        "source": manifest.get("source"),
        "instruction": (manifest.get("instruction") or "")[:400],
        "model": config.get("model"),
        "generator": config.get("generator"),
        "strategy": config.get("strategy"),
        "n_exchanges": llm.get("n_exchanges"),
        "n_repairs": llm.get("n_repairs"),
        "n_decode_failures": llm.get("n_decode_failures"),
        "n_tool_calls": tools.get("n_calls"),
        "n_tool_errors": tools.get("n_errors"),
        "tool_ms": tools.get("total_ms"),
        "error": manifest.get("error"),
    }


def scan_rows(root: Path | str | None = None) -> list[dict[str, Any]]:
    """One index row per request directory found under ``root``, newest first."""
    base = Path(root) if root is not None else default_journal_root()
    rows: list[dict[str, Any]] = []
    for d in iter_request_dirs(base):
        try:
            manifest = json.loads((d / _MANIFEST).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            rel = str(d.relative_to(base))
        except ValueError:
            rel = str(d)
        rows.append(_row_from_manifest(manifest, rel))
    rows.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return rows


def rebuild_index(root: Path | str | None = None) -> dict[str, Any]:
    """Regenerate ``index.jsonl`` and ``INDEX.md`` from the requests on disk.

    Index rows that no longer have a directory are moved to
    ``orphaned-index-rows.jsonl`` rather than deleted, so a request whose files
    were removed still leaves a trace of having existed.
    """
    base = Path(root) if root is not None else default_journal_root()
    base.mkdir(parents=True, exist_ok=True)
    rows = scan_rows(base)
    known = {str(r.get("request_id")) for r in rows}

    previous = _read_jsonl(base / _INDEX)
    orphans = {}
    for row in previous:
        rid = str(row.get("request_id"))
        if rid not in known:
            orphans[rid] = row  # later rows win, as in the live index

    # Oldest first on disk: the index is an append-only log, and rebuilding it
    # in arrival order keeps "later rows win" meaningful for anything replaying.
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in reversed(rows))
    (base / _INDEX).write_text(text + ("\n" if text else ""), encoding="utf-8")

    if orphans:
        with (base / _ORPHANS).open("a", encoding="utf-8") as fh:
            for row in orphans.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_human_index(base, rows)
    ensure_readme(base)
    return {
        "root": str(base),
        "n_requests": len(rows),
        "n_orphaned_rows": len(orphans),
        "orphaned": sorted(orphans),
    }


def write_human_index(root: Path | str | None = None, rows: list[dict[str, Any]] | None = None) -> Path:
    """Render ``INDEX.md`` — the table a person reads before opening anything."""
    base = Path(root) if root is not None else default_journal_root()
    if rows is None:
        rows = _read_jsonl(base / _INDEX)
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest[str(row.get("request_id"))] = row
        rows = sorted(latest.values(), key=lambda r: r.get("started_at") or 0, reverse=True)

    ok = sum(1 for r in rows if r.get("status") == "SUCCEEDED")
    n_agent = sum(1 for r in rows if (r.get("kind") or "request") == "request")
    total_s = sum(float(r.get("duration_s") or 0) for r in rows)
    lines = [
        "# Running history — index",
        "",
        f"_{len(rows)} entries · {n_agent} agent requests, {len(rows) - n_agent} direct pipeline"
        f" runs · {ok} succeeded · {total_s / 3600:.1f} h of wall clock ·"
        f" generated {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        "Regenerate with `drugagent journal reindex`. One row per entry — `request`",
        "means someone asked the master agent, `pipeline` means a brief was executed",
        "directly. Links are relative to this folder.",
        "",
        "| when | kind | entry | instruction | status | dur | model | source | exchanges | tool calls |",
        "| --- | --- | --- | --- | --- | --: | --- | --- | --: | --: |",
    ]
    for r in rows:
        started = r.get("started_at")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(started)) if started else "—"
        rid = str(r.get("request_id") or "—")
        rel = str(r.get("dir") or "")
        link = f"[`{rid}`](<{rel}>)" if rel else f"`{rid}`"
        instruction = (r.get("instruction") or "").replace("\n", " ").replace("|", "\\|")
        if len(instruction) > 90:
            instruction = instruction[:87] + "…"
        dur = r.get("duration_s")
        n_ex = r.get("n_exchanges")
        n_tc = r.get("n_tool_calls")
        lines.append(
            f"| {when} | {r.get('kind') or 'request'} | {link} | {instruction} |"
            f" {r.get('status') or '—'} |"
            f" {f'{float(dur):.0f} s' if dur else '—'} | {r.get('model') or '—'} |"
            f" {r.get('source') or '—'} | {n_ex if n_ex is not None else '—'} |"
            f" {n_tc if n_tc is not None else '—'} |"
        )
    orphan_rows = _read_jsonl(base / _ORPHANS)
    if orphan_rows:
        lines += [
            "",
            "## Index rows without files",
            "",
            f"{len(orphan_rows)} request(s) were indexed but their directories are not here —",
            f"see `{_ORPHANS}`. They were written by an earlier run whose payload was",
            "stored elsewhere or removed.",
            "",
        ]
        for row in orphan_rows:
            lines.append(f"- `{row.get('request_id')}` — {(row.get('instruction') or '')[:80]}")
    out = base / _INDEX_MD
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


README_TEXT = """# Running history

Everything this platform has run, kept as the durable record of what was asked,
what the model saw and said, which tools ran, and what came out. This is the
system of record for the self-evolution loop and the source any future training
set would be built from — it is **not** scratch state, and nothing here is
regenerated by re-running the platform.

## Two kinds of entry

| `kind` | what it is | what it holds |
| --- | --- | --- |
| `request` | someone asked the master agent, in words | chat · tools · events · the pipeline runs it drove |
| `pipeline` | a brief run directly: CLI, API, script | tools · the run's legs. No chat: no model was asked |

Both are history. Only `request` entries are *experience*: the self-evolution
loop normalizes those into episodes and ignores the rest, because a pipeline run
has no instruction, no plan and no exchanges to learn from.

A pipeline run that pauses at an approval gate and is later resumed stays **one**
entry — the legs are recorded inside it, not as separate rows.

One consequence to know: in `--deterministic` mode the run id is seeded, so
re-running the same brief writes into the *same* entry rather than making a new
one. That is the same brief being executed again, which is what the entry says;
if you want them separate, run without `--deterministic`.

## Layout

```
running-history/
  INDEX.md          every request as a table, newest first (generated)
  index.jsonl       the same rows, machine-readable (generated)
  requests/
    2026-08-26/
      134120__find-potent-drug-like-seh-inhibitors__mrun_29e7222e10dc/
        summary.md      the entry as prose — start here
        manifest.json   settings, timings, outcome, per-tool totals
        chat.jsonl      every model exchange in full: prompts, reasoning, reply
        tools.jsonl     tool ledgers, one line per scope
        events.jsonl    the typed event stream, replayable
    2026-09-01/
      121655__seh-inhibitor-prioritization-demonstration__run_00000000/
        summary.md · manifest.json · tools.jsonl      (a `pipeline` entry)
  training/         dialogs exported for fine-tuning (generated on request)
```

Directory names are `<HHMMSS>__<slug>__<request_id>`, so the folder sorts
chronologically and reads at a glance. Programs never parse that name: they scan
for `manifest.json` and take the id from inside it.

## Reading it

```bash
drugagent journal list                  # the index, newest first
drugagent journal show <request-id>     # one request
drugagent journal show <request-id> --chat
drugagent journal tools                 # cross-request tool performance
drugagent journal reindex               # rebuild INDEX.md + index.jsonl
```

The self-evolution loop reads exactly these files:

```bash
drugagent self-evolve analyze           # normalized episodes over this folder
```

## Building a training set

```bash
drugagent journal export                # → training/dialogs.jsonl + DATASET_CARD.md
```

One JSON object per model exchange, with the messages as they were sent, the
reply as it came back, the reasoning channel kept separate, and the metadata
needed to filter (phase, decode path, success, tokens, latency).

## Two cautions

**These files contain verbatim user text and verbatim model output.** Secret
redaction is applied on write, but nothing else is filtered: read before sharing.

**Do not delete this folder to clean up.** `make clean-demo` removes `.drugagent`
(run state, caches, artifacts) and deliberately leaves this one alone.
"""


def ensure_readme(root: Path | str | None = None) -> Path:
    """Write the folder's README if it is missing. Never overwrites an edited one."""
    base = Path(root) if root is not None else default_journal_root()
    base.mkdir(parents=True, exist_ok=True)
    out = base / _README
    if not out.exists():
        out.write_text(README_TEXT, encoding="utf-8")
    return out


def migrate(
    source: Path | str,
    root: Path | str | None = None,
    *,
    copy: bool = False,
) -> dict[str, Any]:
    """Move request journals from ``source`` into the dated layout under ``root``.

    Contents are never rewritten — a request directory is relocated whole, so its
    manifest, chat log, ledgers and events keep their bytes and their hashes. A
    request already present under ``root`` is skipped rather than duplicated.
    """
    src = Path(source)
    base = Path(root) if root is not None else default_journal_root()
    base.mkdir(parents=True, exist_ok=True)
    (base / TRAINING).mkdir(exist_ok=True)

    existing = {str(_read_manifest(d).get("request_id")) for d in iter_request_dirs(base)}
    moved: list[dict[str, str]] = []
    skipped: list[str] = []

    for d in iter_request_dirs(src):
        manifest = _read_manifest(d)
        rid = str(manifest.get("request_id") or d.name)
        if rid in existing:
            skipped.append(rid)
            continue
        started = float(manifest.get("started_at") or d.stat().st_mtime)
        instruction = str(manifest.get("instruction") or "")
        day = time.strftime("%Y-%m-%d", time.localtime(started))
        dest = base / REQUESTS / day / request_dirname(rid, started, instruction)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            skipped.append(rid)
            continue
        if copy:
            shutil.copytree(d, dest)
        else:
            shutil.move(str(d), str(dest))
        existing.add(rid)
        moved.append({"request_id": rid, "to": str(dest.relative_to(base))})

    # Carry the source's index rows over before rebuilding, so requests whose
    # payload lives elsewhere (the kernel's benchmark journals, say) survive as
    # orphan rows instead of vanishing from the record.
    src_rows = _read_jsonl(src / _INDEX)
    if src_rows:
        with (base / _INDEX).open("a", encoding="utf-8") as fh:
            for row in src_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = rebuild_index(base)
    stats.update({"source": str(src), "n_moved": len(moved), "moved": moved, "skipped": skipped})
    return stats


def _read_manifest(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
