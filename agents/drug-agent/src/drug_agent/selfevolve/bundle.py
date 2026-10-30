"""Collect every log the loop and the platform wrote into one directory.

The evidence is deliberately scattered while it is being written: request
journals belong to the platform, benchmark journals belong to the kernel and are
kept apart from them on purpose, and the loop's own artefacts land beside the
store they describe. That separation is load-bearing at write time and unhelpful
at read time, which is what this is for.

The bundle is a **copy**, never a move. Nothing here deletes, rewrites or
relocates a source file: the originals stay exactly where the code that produced
them expects to find them, and the bundle is a read-only view you can hand to
someone else. Every file carries its sha256 in ``MANIFEST.json``, so a bundle can
be checked against the store it came from.

``INDEX.md`` is the point of the whole thing — a table of what each log is, what
question it answers, and how to read it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import EvolveConfig

#: The per-request event stream is ~1 MB of streamed token deltas per program and
#: carries nothing the chat log and manifest do not. ``--slim`` drops it.
#: Excluded from a slim bundle: the replayable event stream (large, and derivable
#: from nothing else) and the exported training corpus (a derived view of the
#: chat logs that are already in the bundle).
BULKY = ("events.jsonl", "dialogs.jsonl")


@dataclass
class Entry:
    """One log in the bundle, and what it is for."""

    path: str
    what: str
    question: str
    how: str


#: The index. Order is the order you would read them in.
INDEX: list[Entry] = [
    Entry(
        "METHODS.md",
        "Every distinct method the loop proposed, folded across cycles",
        "What did it actually suggest, and what happened to each one?",
        "Start here. Read top to bottom.",
    ),
    Entry(
        "REPORT.md",
        "The experiment report, if one was written",
        "What was measured, and what may be concluded from it?",
        "The claim taxonomy in §0 says which claims are which.",
    ),
    Entry(
        "cycles.jsonl",
        "The evolver's own experience — one line per research cycle",
        "What did each cycle do, decide, and cost?",
        "`jq -c '{cluster:.cluster_id, decision, wall_s, calls:.backend_calls}' cycles.jsonl`",
    ),
    Entry(
        "hypotheses/",
        "Every hypothesis, its mechanism, its falsifier, and the verifier's verdict",
        "What did the deep model think, and did the critic believe it?",
        "`jq -r '.verdict + \"  \" + .statement' hypotheses/*.json`",
    ),
    Entry(
        "archive/",
        "Candidates with lineage, novelty descriptors, status, and linked records",
        "What was built, from what parent, and where did it end up?",
        "`archive/index.jsonl` is the append-only event log; `cand_*.json` are the candidates.",
    ),
    Entry(
        "records/",
        "Evaluation records — one per benchmark arm",
        "What did the six tasks actually score, task by task?",
        "`jq '.aggregates' records/*.json`; per-task rows are under `.tasks[]`.",
    ),
    Entry(
        "decisions/",
        "Promotion reports — the ten gate checks and the decision",
        "Why was this promoted, rejected, or sent to a human?",
        "Read the `.md`; the `.json` beside it is the same thing machine-readable.",
    ),
    Entry(
        "gates/",
        "Gate 0 captures (argv, exit status, stdout, stderr) and gate 1 replay evidence",
        "Did the invariants pass, and what did the deterministic replay show?",
        "`gate0_*.json` for the test suite, `replay_*.json` for the recorded-reply replay.",
    ),
    Entry(
        "analysis.json",
        "The measured findings, clusters, tool attribution, and the capability envelope",
        "What is measurably wrong with the platform, and on what evidence?",
        "`jq -r '.findings[] | .severity + \"  \" + .finding_id' analysis.json`",
    ),
    Entry(
        "envelope.json",
        "What this deployment has been measured able to achieve",
        "Is a given success target reachable here at all?",
        "`jq '.metrics' envelope.json` — `mocked: true` means the value came from a stand-in.",
    ),
    Entry(
        "request_journals/",
        "The MASTER AGENT's logs — the running history, one directory per user request",
        "What was asked, what did the model say, which tools ran, what came out?",
        "Copied from `running-history/`: `INDEX.md` (the readable table) then, under "
        "`requests/<date>/`, `manifest.json` (settings + outcome) · `chat.jsonl` (every "
        "exchange in full) · `tools.jsonl` (per-cycle ledgers) · `events.jsonl` (the typed "
        "stream) · `summary.md`.",
    ),
    Entry(
        "benchmark_journals/",
        "The SELF-EVOLVE loop's own runs — one directory per benchmark task",
        "What did each arm's programs do, exchange by exchange?",
        "Same layout as a request journal. Kept apart from them so the analyzer never "
        "studies the evolver's own experiments.",
    ),
    Entry(
        "episodes/",
        "Normalized experience — the typed view the analysis reads",
        "What does the loop see when it reads a request journal?",
        "Derived from `request_journals/`; regenerate with `self-evolve analyze --refresh`.",
    ),
    Entry(
        "run_bundles/",
        "Full scientific bundles from CLI/API pipeline runs",
        "What molecules came out, with their scorecards, Pareto front and safety register?",
        "Master-agent cycles do NOT produce these — their pipeline state home is a temp "
        "directory, deleted per cycle, which is what keeps the ablation deterministic. "
        "Their numbers live in the journal's events instead.",
    ),
    Entry(
        "MANIFEST.json",
        "Every file in the bundle with its size and sha256, plus provenance",
        "Is this bundle intact, and where did it come from?",
        "`jq '.environment' MANIFEST.json`",
    ),
]


def collect(
    cfg: EvolveConfig | None = None,
    *,
    output: str | Path | None = None,
    slim: bool = False,
    with_journals: bool = True,
    with_run_bundles: bool = True,
) -> dict[str, Any]:
    """Copy every log into one directory and write the index and manifest."""
    cfg = cfg or EvolveConfig()
    store = cfg.resolved_store_dir()
    out = Path(output) if output else store / "bundles" / time.strftime("evolve_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    notes: list[str] = []

    def copy_file(src: Path, rel: str) -> None:
        if not src.is_file():
            return
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(_stat(dst, out))

    def copy_tree(src: Path, rel: str, *, skip: tuple[str, ...] = ()) -> int:
        if not src.is_dir():
            return 0
        n = 0
        for path in sorted(src.rglob("*")):
            if not path.is_file() or path.name in skip:
                continue
            copy_file(path, f"{rel}/{path.relative_to(src)}")
            n += 1
        return n

    # --- the loop's own artefacts ----------------------------------------
    for name in ("analysis.json", "cycles.jsonl", "replay_evidence.json"):
        copy_file(store / name, name)
    for sub in ("hypotheses", "archive", "records", "decisions", "episodes"):
        copy_tree(store / sub, sub)

    # gate evidence, gathered under one name rather than loose in the store
    for path in sorted(store.glob("gate0_*.json")) + sorted(store.glob("replay_cand_*.json")):
        copy_file(path, f"gates/{path.name}")

    # the envelope lives beside the platform state, not in the evolve store
    copy_file(cfg.repo_root() / ".drugagent" / "envelope.json", "envelope.json")

    # --- journals ---------------------------------------------------------
    skip = BULKY if slim else ()
    if with_journals:
        n_req = copy_tree(cfg.resolved_journal_dir(), "request_journals", skip=skip)
        n_bench = copy_tree(store / "benchmark_journals", "benchmark_journals", skip=skip)
        notes.append(f"{n_req} request-journal files, {n_bench} benchmark-journal files")
        if slim:
            notes.append("slim: per-request events.jsonl omitted (streamed deltas, ~1 MB each)")
    else:
        notes.append("journals omitted by request")

    # --- scientific run bundles -------------------------------------------
    if with_run_bundles:
        runs = cfg.repo_root() / ".drugagent" / "runs"
        n_runs = copy_tree(runs, "run_bundles")
        notes.append(
            f"{n_runs} files from {len(list(runs.glob('run_*'))) if runs.is_dir() else 0} run bundles"
        )

    # --- the documents ----------------------------------------------------
    from .methods import render_methods, summarize_methods

    summary = summarize_methods(cfg)
    (out / "METHODS.md").write_text(render_methods(summary), encoding="utf-8")
    (out / "methods.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    copied.append(_stat(out / "METHODS.md", out))
    copied.append(_stat(out / "methods.json", out))

    report = cfg.repo_root() / "docs" / "self_evolution_experiment.md"
    if report.is_file():
        copy_file(report, "REPORT.md")

    (out / "INDEX.md").write_text(render_index(out, copied, notes, summary), encoding="utf-8")
    copied.append(_stat(out / "INDEX.md", out))

    manifest = {
        "created_at": time.time(),
        "source_store": str(store),
        "source_journals": str(cfg.resolved_journal_dir()),
        "slim": slim,
        "n_files": len(copied),
        "total_bytes": sum(f["bytes"] for f in copied),
        "environment": _environment(cfg),
        "notes": notes,
        "files": sorted(copied, key=lambda f: f["path"]),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "bundle_dir": str(out),
        "n_files": len(copied) + 1,
        "total_bytes": manifest["total_bytes"],
        "n_methods": summary["n_methods"],
        "notes": notes,
    }


def _stat(path: Path, root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _environment(cfg: EvolveConfig) -> dict[str, Any]:
    try:
        from .evaluation.integrity import kernel_digest

        digest = kernel_digest()
    except Exception:
        digest = "unavailable"
    try:
        from ..llm.config import Config

        model = Config().model
    except Exception:
        model = "unknown"
    import platform
    import sys

    import drug_agent

    return {
        "drug_agent_version": getattr(drug_agent, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ollama_model": model,
        "kernel_digest_now": digest,
        "note": (
            "kernel_digest_now is the kernel as it is at bundling time. Each evaluation "
            "record carries the digest it was produced under; compare them before comparing arms."
        ),
    }


def render_index(root: Path, copied: list[dict[str, Any]], notes: list[str], summary: dict[str, Any]) -> str:
    # The manifest is written after the index — it has to hash the index — so it
    # is not in `copied` yet and would otherwise be reported absent.
    present = {f["path"] for f in copied} | {"MANIFEST.json"}

    def exists(rel: str) -> bool:
        if rel.endswith("/"):
            return any(p.startswith(rel) for p in present)
        return rel in present or (root / rel).exists()

    lines = [
        "# Self-evolution log bundle",
        "",
        "Every log the loop and the platform wrote, copied into one place. The originals are "
        "untouched — this is a read-only view.",
        "",
        f"- {summary['n_methods']} distinct methods proposed, from {summary['n_hypotheses']} "
        f"hypotheses across {summary['n_cycles']} research cycles",
        f"- {summary['n_arms']} benchmark arms measured",
        f"- {len(copied) + 1} files, {sum(f['bytes'] for f in copied) / 1e6:.1f} MB",
        "",
        "## What is in here, and what each answers",
        "",
        "| file | what it is | the question it answers |",
        "| --- | --- | --- |",
    ]
    for e in INDEX:
        mark = "" if exists(e.path) else " *(absent — never produced)*"
        lines.append(f"| `{e.path}`{mark} | {e.what} | {e.question} |")
    lines += ["", "## How to read each one", ""]
    for e in INDEX:
        if not exists(e.path):
            continue
        lines += [f"**`{e.path}`** — {e.what}.", "", f"> {e.how}", ""]
    if notes:
        lines += ["## Notes on this bundle", ""] + [f"- {n}" for n in notes] + [""]
    lines += [
        "## What is deliberately *not* here",
        "",
        "- **Telemetry spans.** The Engine writes one JSON span per agent step to stdout "
        '(`{"event": "span", …}`); nothing captures them, so there is no file to collect.',
        "- **Per-cycle pipeline state from master-agent runs.** Each cycle runs in a fresh "
        "temp directory that is deleted when the cycle ends — that isolation is what makes the "
        "naive-retry ablation deterministic. The numbers survive in the journal's "
        "`chemical_result` and `evaluation_result` events; the parquet/SDF bundle does not.",
        "- **Model weights, credentials, and anything outside the project.**",
        "",
    ]
    return "\n".join(lines)


__all__ = ["BULKY", "INDEX", "Entry", "collect", "render_index"]
