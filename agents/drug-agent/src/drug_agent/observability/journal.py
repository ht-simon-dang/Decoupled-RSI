"""Per-request run journal — the durable record of what a request actually did.

One *request* is one thing a person asked for: a natural-language instruction
submitted through the console, the CLI, or the API. The journal answers four
questions about it, long after the process has exited:

  1. **What was asked?** — the verbatim instruction plus the resolved settings.
  2. **What did the model see and say?** — every chat exchange in full: the
     system and user messages, the reasoning channel, the raw reply, how the
     reply was decoded, and what it cost.
  3. **Which tools ran, and how did they perform?** — aggregated per tool and
     operation: call counts, latency, failures, and the quality metrics each
     tool reports about its own output.
  4. **What came out?** — the final result, or the error that stopped it.

Layout on disk — ``./running-history`` by default, one directory per request,
grouped by the day it started and named so a person can find one by eye:

    <root>/README.md                  what this folder is and how to read it
    <root>/INDEX.md                   every request as a table, newest first
    <root>/index.jsonl                one line per request; the machine index
    <root>/requests/<YYYY-MM-DD>/<HHMMSS>__<slug>__<request_id>/
        manifest.json                 the request, its settings, its outcome
        chat.jsonl                    full JSON chat log, one exchange per line
        tools.jsonl                   tool-usage ledgers, one scope per line
        events.jsonl                  the typed event stream, replayable
        summary.md                    the same story, for a human
    <root>/training/                  dialogs exported for fine-tuning

The directory name carries the date, the time, a slug of the instruction and the
request id, so `ls` is a usable index and the id is still an exact key. Readers
never parse that name: they resolve requests by scanning for ``manifest.json``
and read the id out of the manifest, which keeps every older *flat* layout
(``<root>/<request_id>/``) readable — the benchmark journals the evaluation
kernel writes included.

JSONL is deliberate: a crashed run still leaves a readable, complete prefix, and
appending never rewrites what is already on disk.

Everything written here passes through the same secret redaction as logs,
manifests and reports. A journal records user text and model output verbatim, so
it is exactly the place a pasted API key would come to rest — and applying one
policy in one place beats remembering to apply it per call site.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..security.redaction import redact, redact_obj
from .toolmeter import ToolLedger, metering

#: Bump when the on-disk layout changes in a way readers must know about.
JOURNAL_VERSION = 1

_MANIFEST = "manifest.json"
_CHAT = "chat.jsonl"
_TOOLS = "tools.jsonl"
_EVENTS = "events.jsonl"
_SUMMARY = "summary.md"
_INDEX = "index.jsonl"

#: What produced an entry. A *request* is something a person asked the master
#: agent for in natural language; a *pipeline* run is a brief executed directly
#: through the CLI, the API or a script. Both are history; only the first is
#: experience the self-evolution loop can learn from, so they are labelled.
KIND_REQUEST = "request"
KIND_PIPELINE = "pipeline"

#: Subdirectory holding the per-request directories, grouped by date.
REQUESTS = "requests"
#: Subdirectory holding exports built for model training.
TRAINING = "training"
#: The default history root, relative to the repository.
HISTORY_DIRNAME = "running-history"


def default_journal_root() -> Path:
    """Where journals go unless configured otherwise.

    ``DRUGAGENT_JOURNAL_DIR`` wins; otherwise the repo-local ``running-history``
    when a repo is recognizable, else ``~/.drugagent/running-history``. It must
    NOT sit under the per-run state home (``.drugagent``): that directory is
    deleted when a workspace is cleaned, and the history has to outlive every run
    it describes — it is the input to the self-evolution loop and to any later
    training set.
    """
    env = os.environ.get("DRUGAGENT_JOURNAL_DIR")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent / HISTORY_DIRNAME
    return Path.home() / ".drugagent" / HISTORY_DIRNAME


def slugify(text: str, *, max_len: int = 48) -> str:
    """A short, filesystem-safe, human-readable stem for an instruction."""
    cleaned = []
    for ch in (text or "").lower():
        if ch.isalnum() and ch.isascii():
            cleaned.append(ch)
        elif cleaned and cleaned[-1] != "-":
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")[:max_len].strip("-")
    return slug or "request"


def request_dirname(request_id: str, started_at: float, instruction: str) -> str:
    """``<HHMMSS>__<slug>__<request_id>`` — sortable, readable, still keyed by id."""
    stamp = time.strftime("%H%M%S", time.localtime(started_at))
    return f"{stamp}__{slugify(instruction)}__{request_id}"


def request_dir(root: Path, request_id: str, started_at: float, instruction: str) -> Path:
    """Where this request's files belong under ``root``."""
    day = time.strftime("%Y-%m-%d", time.localtime(started_at))
    return root / REQUESTS / day / request_dirname(request_id, started_at, instruction)


def iter_request_dirs(root: Path | str | None = None) -> list[Path]:
    """Every request directory under ``root``, in chronological path order.

    Three shapes are recognized, because all three exist on disk somewhere: the
    dated layout this module writes, an undated directory dropped straight into
    ``requests/``, and the older flat ``<root>/<request_id>/`` that the tests and
    the kernel's benchmark journals still use.
    """
    base = Path(root) if root is not None else default_journal_root()
    if not base.is_dir():
        return []
    found: set[Path] = set()
    for pattern in (
        f"*/{_MANIFEST}",
        f"{REQUESTS}/*/{_MANIFEST}",
        f"{REQUESTS}/*/*/{_MANIFEST}",
    ):
        for man in base.glob(pattern):
            if man.is_file():
                found.add(man.parent)
    return sorted(found)


def find_request_dir(request_id: str, root: Path | str | None = None) -> Path | None:
    """Resolve one request id to its directory, whatever the layout."""
    base = Path(root) if root is not None else default_journal_root()
    direct = base / request_id  # the flat layout, and the cheap case
    if (direct / _MANIFEST).is_file():
        return direct
    for d in iter_request_dirs(base):
        if d.name == request_id or d.name.endswith(f"__{request_id}"):
            return d
    for d in iter_request_dirs(base):  # last resort: read the ids
        try:
            manifest = json.loads((d / _MANIFEST).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(manifest.get("request_id")) == request_id:
            return d
    return None


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to something ``json.dumps`` accepts.

    A journal must never be the reason a run fails, so anything exotic degrades
    to its ``repr`` rather than raising.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump(mode="json"))
        except Exception:
            pass
    return repr(obj)


@dataclass
class ChatExchange:
    """One request/response round-trip with the language model."""

    phase: str
    cycle: int = 0
    attempt: int = 0
    model: str = ""
    mode: str = ""  # how the reply was requested: schema | fenced | repair
    messages: list[dict[str, str]] = field(default_factory=list)
    thinking: str = ""
    content: str = ""
    parsed: Any = None
    ok: bool = True
    decode: str = ""  # which decoding path produced `parsed`
    error: str | None = None
    elapsed_s: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    done_reason: str | None = None
    repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "cycle": self.cycle,
            "attempt": self.attempt,
            "model": self.model,
            "mode": self.mode,
            "messages": self.messages,
            "thinking": self.thinking,
            "content": self.content,
            "parsed": _jsonable(self.parsed),
            "ok": self.ok,
            "decode": self.decode,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3),
            "options": _jsonable(self.options),
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "done_reason": self.done_reason,
            "repairs": list(self.repairs),
        }


class RunJournal:
    """Append-only journal for one request. Every method is failure-tolerant."""

    def __init__(
        self,
        request_id: str,
        *,
        root: Path | str | None = None,
        instruction: str = "",
        source: str = "unknown",
        config: dict[str, Any] | None = None,
        kind: str = KIND_REQUEST,
        reopen: bool = False,
    ) -> None:
        self.request_id = request_id
        self.root = Path(root) if root is not None else default_journal_root()
        self.instruction = instruction
        self.source = source
        self.kind = kind
        self.config = _jsonable(config or {})
        self.started_at = time.time()
        self.dir = request_dir(self.root, request_id, self.started_at, instruction)
        self.finished_at: float | None = None
        self.status = "RUNNING"
        self.error: str | None = None
        self.n_chat = 0
        #: Streaming deltas seen but not stored: {type: {"events": n, "chars": n}}.
        self.streamed: dict[str, dict[str, int]] = {}
        self.n_repairs = 0
        self.n_decode_failures = 0
        self.eval_tokens = 0
        self.llm_seconds = 0.0
        self.tool_scopes: list[dict[str, Any]] = []
        self.pipeline_runs: list[dict[str, Any]] = []
        self.outcome: dict[str, Any] = {}
        self._ok = True
        if reopen:
            self._reopen()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._ok = False  # journaling is best-effort; never break the run
        self._write_manifest()

    def _reopen(self) -> None:
        """Continue an existing entry instead of starting a second one.

        A pipeline run is executed in more than one call — start it, pause at an
        approval gate, resume it — and each call is the same unit of work to the
        person who launched it. Reopening keeps one directory per run, with the
        counters and ledgers already recorded folded back in, so the manifest
        stays a total rather than the last leg only.
        """
        found = find_request_dir(self.request_id, self.root)
        if found is None:
            return
        self.dir = found
        try:
            manifest = json.loads((found / _MANIFEST).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.started_at = float(manifest.get("started_at") or self.started_at)
        self.instruction = manifest.get("instruction") or self.instruction
        self.source = manifest.get("source") or self.source
        self.kind = manifest.get("kind") or self.kind
        if manifest.get("config"):
            self.config = {**manifest["config"], **self.config}
        llm = manifest.get("llm") or {}
        self.n_chat = int(llm.get("n_exchanges") or 0)
        self.n_repairs = int(llm.get("n_repairs") or 0)
        self.n_decode_failures = int(llm.get("n_decode_failures") or 0)
        self.eval_tokens = int(llm.get("eval_tokens") or 0)
        self.llm_seconds = float(llm.get("seconds") or 0.0)
        self.streamed = dict(manifest.get("streamed") or {})
        self.pipeline_runs = list(manifest.get("pipeline_runs") or [])
        # The per-tool totals are derived from the raw scopes, so reload those
        # rather than the summary: a folded summary cannot be folded again.
        self.tool_scopes = _read_jsonl(found / _TOOLS)

    # --- low-level ------------------------------------------------------
    def _append(self, name: str, payload: dict[str, Any]) -> None:
        if not self._ok:
            return
        try:
            with (self.dir / name).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(redact_obj(_jsonable(payload)), ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            self._ok = False

    def _write_manifest(self) -> None:
        if not self._ok:
            return
        try:
            (self.dir / _MANIFEST).write_text(
                json.dumps(redact_obj(self.manifest()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            self._ok = False

    def manifest(self) -> dict[str, Any]:
        return {
            "journal_version": JOURNAL_VERSION,
            "request_id": self.request_id,
            "kind": self.kind,
            "source": self.source,
            "instruction": self.instruction,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 3),
            "status": self.status,
            "error": self.error,
            "streamed": self.streamed,
            "llm": {
                "n_exchanges": self.n_chat,
                "n_repairs": self.n_repairs,
                "n_decode_failures": self.n_decode_failures,
                "eval_tokens": self.eval_tokens,
                "seconds": round(self.llm_seconds, 3),
            },
            "tools": self.tool_summary(),
            "pipeline_runs": self.pipeline_runs,
            "outcome": self.outcome,
        }

    # --- recording ------------------------------------------------------
    def record_event(self, event: dict[str, Any]) -> None:
        """Append one typed event from the live stream."""
        self._append(_EVENTS, event)

    def record_event_filtered(self, skip_types: Iterable[str]) -> Callable[[Any], None]:
        """An event sink that counts high-frequency types instead of storing them.

        Streaming token deltas are the case this exists for: they are a UI
        transport detail, they arrive in the thousands, and their full content is
        already recorded per exchange. Counting them keeps the fact that they
        happened — and how much was streamed — without the volume.
        """
        skip = frozenset(skip_types)

        def sink(event: Any) -> None:
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
            kind = str(payload.get("type", ""))
            if kind in skip:
                tally = self.streamed.setdefault(kind, {"events": 0, "chars": 0})
                tally["events"] += 1
                tally["chars"] += len(str((payload.get("data") or {}).get("delta", "")))
                return
            self.record_event(payload)

        return sink

    def record_chat(self, exchange: ChatExchange) -> None:
        """Append one full model exchange — prompts, reasoning, reply, decode."""
        self.n_chat += 1
        self.llm_seconds += exchange.elapsed_s
        if exchange.eval_count:
            self.eval_tokens += int(exchange.eval_count)
        if exchange.attempt > 0:
            self.n_repairs += 1
        if not exchange.ok:
            self.n_decode_failures += 1
        self._append(_CHAT, exchange.to_dict())
        self._write_manifest()

    def record_tools(self, ledger: ToolLedger, *, scope: str, cycle: int = 0) -> None:
        """Append one tool-usage ledger (one pipeline run's worth of tool calls)."""
        entry = {
            "scope": scope,
            "cycle": cycle,
            "ts": time.time(),
            "totals": ledger.totals(),
            "tools": ledger.records(),
        }
        self.tool_scopes.append(entry)
        self._append(_TOOLS, entry)
        self._write_manifest()

    def record_pipeline(self, **info: Any) -> None:
        """Note one backend pipeline execution (there is one per cycle)."""
        self.pipeline_runs.append(_jsonable(info))
        self._write_manifest()

    @contextmanager
    def metered(self, *, scope: str, cycle: int = 0) -> Iterator[ToolLedger]:
        """Meter every instrumented tool call made inside the block."""
        ledger = ToolLedger(label=scope)
        with metering(ledger):
            try:
                yield ledger
            finally:
                if ledger:
                    self.record_tools(ledger, scope=scope, cycle=cycle)

    def finish(
        self,
        status: str,
        *,
        outcome: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.status = status
        self.error = error
        self.finished_at = time.time()
        if outcome:
            self.outcome = _jsonable(outcome)
        self._write_manifest()
        self._write_summary()
        self._append_index()
        self._refresh_human_index()

    def _refresh_human_index(self) -> None:
        """Rewrite ``INDEX.md`` (and seed ``README.md``) after a request lands.

        Cheap — it re-renders the index rows already on disk — and it is what
        makes the folder browsable without running anything.
        """
        if not self._ok:
            return
        try:
            from .history import ensure_readme, write_human_index

            write_human_index(self.root)
            ensure_readme(self.root)
        except Exception:  # a cosmetic index must never fail a finished request
            pass

    # --- derived views --------------------------------------------------
    def tool_summary(self) -> dict[str, Any]:
        """Fold every scope's ledger into one per-tool view for this request."""
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for scope in self.tool_scopes:
            for rec in scope["tools"]:
                key = (rec["tool_id"], rec["operation"])
                agg = by_key.setdefault(
                    key,
                    {
                        "tool_id": rec["tool_id"],
                        "operation": rec["operation"],
                        "backend": rec.get("backend"),
                        "n_calls": 0,
                        "n_errors": 0,
                        "total_ms": 0.0,
                        "max_ms": 0.0,
                        "errors": [],
                        "metrics": {},
                    },
                )
                agg["n_calls"] += rec["n_calls"]
                agg["n_errors"] += rec["n_errors"]
                agg["total_ms"] += rec["total_ms"]
                agg["max_ms"] = max(agg["max_ms"], rec["max_ms"])
                agg["backend"] = agg["backend"] or rec.get("backend")
                for err in rec.get("errors", []):
                    if err not in agg["errors"] and len(agg["errors"]) < 5:
                        agg["errors"].append(err)
                for mkey, m in (rec.get("metrics") or {}).items():
                    slot = agg["metrics"].setdefault(
                        mkey, {"n": 0, "sum": 0.0, "min": m["min"], "max": m["max"]}
                    )
                    slot["n"] += m["n"]
                    slot["sum"] += (m["mean"] or 0.0) * m["n"]
                    slot["min"] = min(slot["min"], m["min"])
                    slot["max"] = max(slot["max"], m["max"])
        out = []
        for agg in sorted(by_key.values(), key=lambda a: (-a["total_ms"], a["tool_id"])):
            agg["total_ms"] = round(agg["total_ms"], 3)
            agg["mean_ms"] = round(agg["total_ms"] / agg["n_calls"], 3) if agg["n_calls"] else None
            agg["metrics"] = {
                k: {
                    "n": v["n"],
                    "mean": round(v["sum"] / v["n"], 6) if v["n"] else None,
                    "min": round(v["min"], 6),
                    "max": round(v["max"], 6),
                }
                for k, v in sorted(agg["metrics"].items())
            }
            out.append(agg)
        return {
            "n_scopes": len(self.tool_scopes),
            "n_calls": sum(a["n_calls"] for a in out),
            "n_errors": sum(a["n_errors"] for a in out),
            "total_ms": round(sum(a["total_ms"] for a in out), 3),
            "by_tool": out,
        }

    def _append_index(self) -> None:
        if not self._ok:
            return
        tools = self.tool_summary()
        try:
            rel_dir = str(self.dir.relative_to(self.root))
        except ValueError:
            rel_dir = str(self.dir)
        row = {
            "request_id": self.request_id,
            "kind": self.kind,
            "dir": rel_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 3),
            "status": self.status,
            "source": self.source,
            "instruction": self.instruction[:400],
            "model": self.config.get("model"),
            "generator": self.config.get("generator"),
            "strategy": self.config.get("strategy"),
            "n_exchanges": self.n_chat,
            "n_repairs": self.n_repairs,
            "n_decode_failures": self.n_decode_failures,
            "n_tool_calls": tools["n_calls"],
            "n_tool_errors": tools["n_errors"],
            "tool_ms": tools["total_ms"],
            "error": self.error,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / _INDEX).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(redact_obj(row), ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            self._ok = False

    def _source_label(self) -> str:
        """The candidate source, spelled so it names the model that really ran.

        The recorded token is "pfm" for the de-novo arm whichever of the eleven
        installed models sampled it, so this summary used to report "candidate
        source pfm" for a run in which PFM never executed.
        """
        from ..llm.schemas import candidate_source_label

        gen = self.config.get("generator")
        if not gen:
            return "—"
        return candidate_source_label(gen, self.config.get("generator_backend"))

    def _write_summary(self) -> None:
        if not self._ok:
            return
        m = self.manifest()
        tools = m["tools"]
        pipeline = self.kind == KIND_PIPELINE
        lines = [
            f"# {'Pipeline run' if pipeline else 'Request'} {self.request_id}",
            "",
            f"- **status** {m['status']}" + (f" — {m['error']}" if m["error"] else ""),
            f"- **source** {m['source']}",
            f"- **duration** {m['duration_s']} s",
        ]
        if pipeline:
            lines.append(
                f"- **profile** {self.config.get('profile', '—')}"
                f" · deterministic {self.config.get('deterministic', '—')}"
                f" · objectives {', '.join(str(o) for o in self.config.get('objectives') or []) or '—'}"
            )
        else:
            lines.append(
                f"- **model** {self.config.get('model', '—')}"
                f" · strategy {self.config.get('strategy', '—')}"
                f" · candidate source {self._source_label()}"
            )
        lines += [
            "",
            "## Brief" if pipeline else "## Instruction",
            "",
            "> " + (self.instruction or "(empty)").replace("\n", "\n> "),
            "",
        ]
        # A pipeline run asks no model, and an empty "0 exchanges" section reads
        # as a failure rather than as what it is: a run that never needed one.
        if m["llm"]["n_exchanges"]:
            lines += [
                "## Language model",
                "",
                f"- {m['llm']['n_exchanges']} exchanges, {m['llm']['eval_tokens']} generated tokens,"
                f" {m['llm']['seconds']} s",
                f"- {m['llm']['n_repairs']} repair attempts,"
                f" {m['llm']['n_decode_failures']} undecodable replies",
                "",
            ]
        lines += [
            "## Tools",
            "",
        ]
        if tools["by_tool"]:
            lines += [
                f"{tools['n_calls']} calls across {len(tools['by_tool'])} operations,"
                f" {tools['n_errors']} errors, {tools['total_ms']} ms total.",
                "",
                "| tool | operation | backend | calls | errors | total ms | mean ms | reported metrics |",
                "| --- | --- | --- | --: | --: | --: | --: | --- |",
            ]
            for t in tools["by_tool"]:
                metrics = ", ".join(f"{k}={v['mean']}" for k, v in t["metrics"].items()) or "—"
                lines.append(
                    f"| `{t['tool_id']}` | {t['operation']} | {t['backend'] or '—'} | {t['n_calls']} |"
                    f" {t['n_errors']} | {t['total_ms']} | {t['mean_ms']} | {metrics} |"
                )
        else:
            lines.append("No instrumented tool calls were recorded.")
        if self.pipeline_runs:
            lines += [
                "",
                "## Pipeline runs",
                "",
                "| leg | run id | state | candidates | shortlist |",
                "| --- | --- | --- | --: | --: |",
            ]
            for p in self.pipeline_runs:
                leg = p.get("phase") or p.get("cycle", "—")
                lines.append(
                    f"| {leg} | `{p.get('run_id', '—')}` | {p.get('state', '—')} |"
                    f" {p.get('n_candidates', '—')} | {p.get('shortlist_size', '—')} |"
                )
        lines += ["", "## Files", ""]
        if not pipeline:
            lines += [
                f"- `{_CHAT}` — full JSON chat log",
                f"- `{_EVENTS}` — the typed event stream",
            ]
        lines += [
            f"- `{_TOOLS}` — tool ledgers, one line per scope",
            f"- `{_MANIFEST}` — machine-readable manifest",
            "",
        ]
        try:
            (self.dir / _SUMMARY).write_text(redact("\n".join(lines)), encoding="utf-8")
        except OSError:
            self._ok = False


# --------------------------------------------------------------------------
# Reading side — query what past requests did.
# --------------------------------------------------------------------------
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


def list_requests(root: Path | str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first index rows. Falls back to scanning manifests if the index is gone."""
    base = Path(root) if root is not None else default_journal_root()
    rows = _read_jsonl(base / _INDEX)
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:  # later rows win, so a re-finished request updates in place
        seen[str(row.get("request_id"))] = row
    if not seen and base.is_dir():
        for d in iter_request_dirs(base):
            man = d / _MANIFEST
            try:
                m = json.loads(man.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                rel_dir = str(d.relative_to(base))
            except ValueError:
                rel_dir = str(d)
            seen[str(m.get("request_id"))] = {
                "request_id": m.get("request_id"),
                "dir": rel_dir,
                "started_at": m.get("started_at"),
                "status": m.get("status"),
                "source": m.get("source"),
                "instruction": (m.get("instruction") or "")[:400],
                "model": (m.get("config") or {}).get("model"),
            }
    ordered = sorted(seen.values(), key=lambda r: r.get("started_at") or 0, reverse=True)
    return ordered[:limit]


def load_request(request_id: str, root: Path | str | None = None) -> dict[str, Any] | None:
    """Everything recorded for one request: manifest + chat + tools + events."""
    base = Path(root) if root is not None else default_journal_root()
    d = find_request_dir(request_id, base)
    if d is None:
        return None
    man = d / _MANIFEST
    if not man.is_file():
        return None
    try:
        manifest = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "manifest": manifest,
        "chat": _read_jsonl(d / _CHAT),
        "tools": _read_jsonl(d / _TOOLS),
        "events": _read_jsonl(d / _EVENTS),
        "summary_path": str(d / _SUMMARY),
        "dir": str(d),
    }


def tool_statistics(root: Path | str | None = None, *, limit: int = 200) -> dict[str, Any]:
    """Cross-request tool performance — the point of keeping journals at all.

    Answers "how has this tool behaved over the last N requests": call volume,
    latency, failure rate, and the mean of every quality metric it reported.
    """
    base = Path(root) if root is not None else default_journal_root()
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    n_requests = 0
    for row in list_requests(base, limit=limit):
        rid = str(row.get("request_id"))
        rel = row.get("dir")
        d = (base / str(rel)) if rel else find_request_dir(rid, base)
        if d is None or not (d / _MANIFEST).is_file():
            d = find_request_dir(rid, base)
        if d is None:
            continue
        man = d / _MANIFEST
        if not man.is_file():
            continue
        try:
            manifest = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n_requests += 1
        for t in (manifest.get("tools") or {}).get("by_tool", []):
            key = (t["tool_id"], t["operation"])
            agg = by_key.setdefault(
                key,
                {
                    "tool_id": t["tool_id"],
                    "operation": t["operation"],
                    "backend": t.get("backend"),
                    "n_requests": 0,
                    "n_calls": 0,
                    "n_errors": 0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                    "errors": [],
                    "metrics": {},
                },
            )
            agg["n_requests"] += 1
            agg["n_calls"] += t.get("n_calls", 0)
            agg["n_errors"] += t.get("n_errors", 0)
            agg["total_ms"] += t.get("total_ms", 0.0)
            agg["max_ms"] = max(agg["max_ms"], t.get("max_ms", 0.0))
            agg["backend"] = agg["backend"] or t.get("backend")
            for err in t.get("errors", []):
                if err not in agg["errors"] and len(agg["errors"]) < 5:
                    agg["errors"].append(err)
            for mkey, m in (t.get("metrics") or {}).items():
                slot = agg["metrics"].setdefault(mkey, {"n": 0, "sum": 0.0, "min": m["min"], "max": m["max"]})
                slot["n"] += m["n"]
                slot["sum"] += (m["mean"] or 0.0) * m["n"]
                slot["min"] = min(slot["min"], m["min"])
                slot["max"] = max(slot["max"], m["max"])
    tools = []
    for agg in sorted(by_key.values(), key=lambda a: -a["total_ms"]):
        agg["total_ms"] = round(agg["total_ms"], 3)
        agg["mean_ms"] = round(agg["total_ms"] / agg["n_calls"], 3) if agg["n_calls"] else None
        agg["error_rate"] = round(agg["n_errors"] / agg["n_calls"], 6) if agg["n_calls"] else None
        agg["metrics"] = {
            k: {
                "n": v["n"],
                "mean": round(v["sum"] / v["n"], 6) if v["n"] else None,
                "min": round(v["min"], 6),
                "max": round(v["max"], 6),
            }
            for k, v in sorted(agg["metrics"].items())
        }
        tools.append(agg)
    return {"n_requests": n_requests, "root": str(base), "tools": tools}
