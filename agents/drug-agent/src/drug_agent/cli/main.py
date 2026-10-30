"""drugagent CLI.

Human-readable by default; ``--json`` emits machine-readable JSON on stdout
without progress noise. Exit codes: 0 ok, 1 unexpected error, 2 user/validation
error, 3 not found, 4 degraded/blocked.
"""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .. import __version__
from ..orchestration.engine import UnknownGateError
from ..service import build_engine, load_brief
from ..settings.config import load_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Multi-agent computational drug-discovery research prioritization.",
)
console = Console()
err = Console(stderr=True)

EXIT_OK, EXIT_ERROR, EXIT_USER, EXIT_NOT_FOUND, EXIT_DEGRADED = 0, 1, 2, 3, 4

#: Process status for a finished (or paused) run, by run state.
#:
#: ``run``, ``resume`` and ``approve`` all used to fall off the end of the
#: function and exit 0 whatever happened, so ``"state": "FAILED"`` was printed on
#: stdout and every caller — CI, a benchmark harness, a shell script, the
#: research study driver — read success. ``EXIT_ERROR`` and ``EXIT_DEGRADED``
#: were defined here and raised nowhere.
#:
#: A *scientific* outcome is not an execution failure and is not in this table:
#: a run that completes and finds no improvement is a successful execution and
#: exits 0 carrying its decision. What earns a non-zero status is the pipeline
#: not having produced the answer it was asked for.
_RUN_EXIT_CODES = {
    "SUCCEEDED": EXIT_OK,
    "AWAITING_APPROVAL": EXIT_OK,  # paused for a human, on purpose
    "CREATED": EXIT_DEGRADED,  # nothing ran
    "RUNNING": EXIT_DEGRADED,  # stalled: it neither finished nor failed
    "PAUSED": EXIT_DEGRADED,
    "CANCELLED": EXIT_DEGRADED,
    "FAILED": EXIT_ERROR,
}


def run_exit_code(state: Any) -> int:
    """Process status for a run state; an unrecognised state is never a success."""
    return _RUN_EXIT_CODES.get(str(state), EXIT_ERROR)


def _emit(data: Any, json_mode: bool, human: str | None = None) -> None:
    if json_mode:
        console.print_json(_json.dumps(data, default=str))
    elif human is not None:
        console.print(human)


def _engine(profile: str = "local", deterministic: bool = False):
    settings = load_settings(profile=profile, deterministic=deterministic).resolved()
    engine = build_engine(settings)
    engine.history_source = "cli"  # so the running history says where the run came from
    return engine


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"drug-agent {__version__}")


@app.command()
def init(
    profile: str = typer.Option("local", help="Deployment profile."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Initialize the local workspace (state DB + artifact store)."""
    settings = load_settings(profile=profile).resolved()
    _emit(
        {
            "home": str(settings.home),
            "db_url": settings.db_url,
            "artifact_dir": str(settings.artifact_dir),
            "profile": profile,
        },
        json,
        f"[green]Initialized[/green] workspace at {settings.home} (profile={profile}).",
    )


@app.command()
def doctor(json: bool = typer.Option(False, "--json")) -> None:
    """Diagnose environment and capability availability."""
    from ..agents import build_registry

    settings = load_settings().resolved()
    reg = build_registry(settings)
    matrix = reg.capability_matrix()
    checks = {
        "python": sys.version.split()[0],
        "profile": settings.profile,
        "network_allowed": settings.allow_network,
        "llm_provider": settings.llm_provider,
        "capabilities": matrix,
    }
    if json:
        _emit(checks, True)
        return
    table = Table(title="drugagent doctor")
    table.add_column("capability")
    table.add_column("status")
    for cid, status in sorted(matrix.items()):
        color = {"AVAILABLE": "green", "MOCKED": "yellow", "UNAVAILABLE": "red"}.get(status, "white")
        table.add_row(cid, f"[{color}]{status}[/{color}]")
    console.print(table)
    console.print(
        f"python={checks['python']} profile={settings.profile} "
        f"network={'on' if settings.allow_network else 'off (default)'} "
        f"llm={settings.llm_provider}"
    )


@app.command(name="validate-brief")
def validate_brief(
    brief: Path = typer.Argument(..., exists=True), json: bool = typer.Option(False, "--json")
) -> None:
    """Validate a research brief against the schema."""
    try:
        b = load_brief(brief)
    except Exception as exc:
        _emit({"valid": False, "error": str(exc)}, json, f"[red]Invalid brief:[/red] {exc}")
        raise typer.Exit(EXIT_USER) from None
    _emit(
        {"valid": True, "id": b.id, "kind": str(b.kind), "n_objectives": len(b.objectives)},
        json,
        f"[green]Valid[/green] {b.kind} brief '{b.title}' "
        f"({len(b.objectives)} objectives, {len(b.hard_constraints)} constraints).",
    )


@app.command()
def capabilities(
    schemas: bool = typer.Option(False, "--schemas", help="Export JSON schemas."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """List capabilities (and optionally export contract JSON schemas)."""
    from ..agents import build_registry
    from ..contracts import CONTRACT_REGISTRY

    if schemas:
        out = {name: model.json_schema() for name, model in CONTRACT_REGISTRY.items()}
        _emit(out, True)
        return
    reg = build_registry(load_settings().resolved())
    profiles = [p.model_dump(mode="json") for p in reg.profiles()]
    if json:
        _emit(profiles, True)
        return
    table = Table(title="capabilities")
    for col in ("capability_id", "status", "determinism", "cost_class", "backend"):
        table.add_column(col)
    for p in reg.profiles():
        table.add_row(p.capability_id, str(p.status), str(p.determinism), str(p.cost_class), p.backend)
    console.print(table)


@app.command()
def ingest(
    connector: str = typer.Argument(...),
    snapshot: str = typer.Option("default", "--snapshot"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Load a data connector snapshot (fixture-backed; offline)."""
    from ..data.connectors import default_connectors

    conns = default_connectors()
    if connector not in conns:
        _emit(
            {"error": f"unknown connector {connector}", "available": list(conns)},
            json,
            f"[red]Unknown connector[/red] {connector}. Available: {', '.join(conns)}",
        )
        raise typer.Exit(EXIT_USER)
    snap = conns[connector].load()
    _emit(
        snap.data_card(),
        json,
        f"[green]Loaded[/green] {snap.name}: {len(snap.records)} records, snapshot={snap.snapshot_id}",
    )


@app.command()
def run(
    brief: Path = typer.Argument(..., exists=True),
    profile: str = typer.Option("local", "--profile"),
    deterministic: bool = typer.Option(False, "--deterministic"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Execute a brief through the workflow (pauses at approval gates)."""
    try:
        b = load_brief(brief)
    except Exception as exc:
        _emit({"error": str(exc)}, json, f"[red]Invalid brief:[/red] {exc}")
        raise typer.Exit(EXIT_USER) from None
    engine = _engine(profile, deterministic)
    r = engine.create_run(b, deterministic=deterministic, profile=profile)
    if not json:
        err.print(f"[cyan]run[/cyan] {r.id} started (profile={profile}, deterministic={deterministic})")
    final = engine.run(r.id)
    tasks = engine.store.list_tasks(r.id)
    from collections import Counter

    states = Counter(t["state"] for t in tasks)
    payload = {
        "run_id": final.id,
        "state": str(final.state),
        "task_states": dict(states),
        # Task names stay under "awaiting" for compatibility; "pending_gates" is what
        # `drugagent approve` actually takes, and the two are not the same string.
        "awaiting": [t["name"] for t in tasks if t["state"] == "AWAITING_APPROVAL"],
        "pending_gates": _pending_gates(engine, final.id),
    }
    payload["error"] = getattr(final, "error", None)
    payload["exit_code"] = run_exit_code(final.state)
    if str(final.state) == "AWAITING_APPROVAL":
        _emit(payload, json, _paused_hint(final.id, payload["pending_gates"]))
        raise typer.Exit(EXIT_OK)
    _emit(payload, json, _run_outcome(final, dict(states)))
    raise typer.Exit(payload["exit_code"])


@app.command()
def status(run_id: str = typer.Argument(...), json: bool = typer.Option(False, "--json")) -> None:
    """Show run status and task states."""
    engine = _engine()
    data = engine.store.get_run(run_id)
    if not data:
        _emit({"error": "not found"}, json, f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(EXIT_NOT_FOUND)
    tasks = engine.store.list_tasks(run_id)
    if json:
        _emit({"run": data, "tasks": tasks}, True)
        return
    table = Table(title=f"run {run_id} [{data['state']}]")
    for col in ("task", "state", "cache", "error"):
        table.add_column(col)
    for t in tasks:
        table.add_row(t["name"], t["state"], "hit" if t.get("cache_hit") else "", (t.get("error") or "")[:40])
    console.print(table)


@app.command()
def logs(
    run_id: str = typer.Argument(...),
    task: str | None = typer.Option(None, "--task"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the run/task event log."""
    engine = _engine()
    events = engine.store.list_events(run_id, task_id=task)
    if json:
        _emit(events, True)
        return
    for e in events:
        console.print(
            f"[dim]{e.get('ts', '')}[/dim] {e.get('event')} "
            f"{e.get('detail', {}).get('name', '')} "
            f"{e.get('from_state', '')}→{e.get('to_state', '')}"
        )


def _pending_gates(engine: Any, run_id: str) -> list[str]:
    """Gate names blocking tasks that are currently AWAITING_APPROVAL."""
    return sorted(
        {
            t["approval_gate"]
            for t in engine.store.list_tasks(run_id)
            if t["state"] == "AWAITING_APPROVAL" and t.get("approval_gate")
        }
    )


def _paused_hint(run_id: str, gates: list[str]) -> str:
    """A copy-pasteable next step. Names the *gate*, which is what approve takes."""
    if not gates:
        return "Paused for approval, but no gate is recorded — inspect with: drugagent status " + run_id
    listed = ", ".join(repr(g) for g in gates)
    return (
        f"[yellow]Paused[/yellow] for approval at gate {listed}.\n"
        f"Approve with: drugagent approve {run_id} {gates[0]}"
    )


def _run_outcome(final: Any, states: dict[str, int] | None) -> str:
    """One line naming the state, the task tally and the reason it failed."""
    colour = "green" if run_exit_code(final.state) == EXIT_OK else "red"
    tally = f": {states}" if states else ""
    reason = f"\n[dim]{final.error}[/dim]" if getattr(final, "error", None) else ""
    return f"[{colour}]Run {final.state}[/{colour}]{tally}{reason}"


def _gate_error(exc: UnknownGateError, json_out: bool) -> None:
    """Report a bad gate name with the list of names that would have worked."""
    _emit(
        {"error": "unknown_gate", "gate": exc.gate, "known_gates": exc.known},
        json_out,
        f"[red]Unknown gate[/red] {exc.gate!r}. This run has: {', '.join(exc.known) or 'no gates'}."
        "\n[dim]Gates come from the brief's approval_gates, not from task names.[/dim]",
    )
    raise typer.Exit(EXIT_USER) from None


@app.command()
def approve(
    run_id: str = typer.Argument(...),
    gate: str = typer.Argument(...),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Approve a human-approval gate and resume the run."""
    engine = _engine()
    if not engine.store.get_run(run_id):
        raise typer.Exit(EXIT_NOT_FOUND)
    try:
        engine.approve_gate(run_id, gate, by="cli")
    except UnknownGateError as exc:
        _gate_error(exc, json)
    final = engine.resume(run_id)
    payload: dict[str, Any] = {"run_id": run_id, "gate": gate, "state": str(final.state)}
    if str(final.state) == "AWAITING_APPROVAL":
        remaining = _pending_gates(engine, run_id)
        payload["pending_gates"] = remaining
        payload["exit_code"] = EXIT_OK
        _emit(payload, json, f"[green]Approved[/green] {gate}. " + _paused_hint(run_id, remaining))
        return
    payload["error"] = getattr(final, "error", None)
    payload["exit_code"] = run_exit_code(final.state)
    _emit(payload, json, f"[green]Approved[/green] {gate}; " + _run_outcome(final, None))
    raise typer.Exit(payload["exit_code"])


@app.command()
def reject(
    run_id: str = typer.Argument(...),
    gate: str = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Reject a human-approval gate (cancels the run)."""
    engine = _engine()
    try:
        final = engine.reject_gate(run_id, gate, by="cli", reason=reason)
    except UnknownGateError as exc:
        _gate_error(exc, json)
    _emit(
        {"run_id": run_id, "gate": gate, "state": str(final.state), "reason": reason},
        json,
        f"[yellow]Rejected[/yellow] {gate}: {reason}",
    )


@app.command()
def resume(run_id: str = typer.Argument(...), json: bool = typer.Option(False, "--json")) -> None:
    """Resume an interrupted or paused run."""
    engine = _engine()
    if not engine.store.get_run(run_id):
        raise typer.Exit(EXIT_NOT_FOUND)
    final = engine.resume(run_id)
    payload: dict[str, Any] = {"run_id": run_id, "state": str(final.state)}
    if str(final.state) == "AWAITING_APPROVAL":
        payload["pending_gates"] = _pending_gates(engine, run_id)
        payload["exit_code"] = EXIT_OK
        _emit(payload, json, _paused_hint(run_id, payload["pending_gates"]))
        return
    payload["error"] = getattr(final, "error", None)
    payload["exit_code"] = run_exit_code(final.state)
    _emit(payload, json, "[green]Resumed[/green]; " + _run_outcome(final, None))
    raise typer.Exit(payload["exit_code"])


@app.command()
def cancel(run_id: str = typer.Argument(...), json: bool = typer.Option(False, "--json")) -> None:
    """Cancel a run."""
    engine = _engine()
    final = engine.cancel(run_id)
    _emit({"run_id": run_id, "state": str(final.state)}, json, f"[yellow]Cancelled[/yellow] run {run_id}.")


@app.command()
def benchmark(suite: str = typer.Argument("scientific"), json: bool = typer.Option(False, "--json")) -> None:
    """Run a labelled (demonstration-only) benchmark suite."""
    from ..benchmarks.suites import run_suite

    result = run_suite(suite)
    if result.get("error"):
        _emit(result, json, f"[red]{result['error']}[/red]; available: {result.get('available')}")
        raise typer.Exit(EXIT_USER)
    # A check that fails is a *scientific* result and exits 0 carrying it; the
    # suite having been unable to run at all is not.
    if result.get("summary") == "insufficient data":
        _emit(result, json, "[yellow]benchmark not run[/yellow]: insufficient data")
        raise typer.Exit(EXIT_DEGRADED)
    _emit(result, json, f"[cyan]benchmark[/cyan] {suite}: {result.get('summary', '')}")


@app.command()
def compare(
    run_a: str = typer.Argument(...),
    run_b: str = typer.Argument(...),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Compare two runs' shortlists and metrics."""
    engine = _engine()

    def shortlist(rid: str) -> set[str]:
        ports = engine.store.list_objects("portfolio", run_id=rid)
        return {m["candidate_id"] for m in (ports[0]["shortlist"] if ports else [])}

    a, b = shortlist(run_a), shortlist(run_b)
    inter = a & b
    jac = round(len(inter) / max(len(a | b), 1), 3)
    _emit(
        {
            "run_a": run_a,
            "run_b": run_b,
            "jaccard": jac,
            "shared": sorted(inter),
            "only_a": sorted(a - b),
            "only_b": sorted(b - a),
        },
        json,
        f"shortlist Jaccard={jac}; shared={len(inter)}, only_a={len(a - b)}, only_b={len(b - a)}",
    )


@app.command()
def report(
    run_id: str = typer.Argument(...),
    format: str = typer.Option("md", "--format", help="md|html|json"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Print or locate the run report."""
    engine = _engine()
    rep = engine.store.get_object("report", run_id)
    run_data = engine.store.get_run(run_id)
    if not run_data:
        raise typer.Exit(EXIT_NOT_FOUND)
    bundle = Path(engine.settings.runs_dir or ".drugagent/runs") / run_id
    if format == "json":
        data = engine.artifacts.get_json(rep["json_ref"]) if rep else {}
        _emit(data, True)
        return
    fname = "report.html" if format == "html" else "report.md"
    path = bundle / fname
    if path.exists():
        if json:
            _emit({"path": str(path)}, True)
        else:
            console.print(path.read_text())
    else:
        _emit(
            {"error": "report not found; run may be incomplete"},
            json,
            "[red]Report not found[/red] (run may be incomplete).",
        )
        raise typer.Exit(EXIT_NOT_FOUND)


@app.command()
def bundle(run_id: str = typer.Argument(...), json: bool = typer.Option(False, "--json")) -> None:
    """Show the reproducibility bundle location and manifest summary."""
    engine = _engine()
    path = Path(engine.settings.runs_dir or ".drugagent/runs") / run_id
    if not path.exists():
        _emit({"error": "bundle not found"}, json, "[red]Bundle not found.[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    manifest = path / "manifest.json"
    _emit(
        {"bundle_dir": str(path), "manifest": str(manifest), "files": sorted(p.name for p in path.iterdir())},
        json,
        f"[green]Bundle[/green] at {path}",
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(2003, "--port", help="Local UI/API port (default 2003)."),
    profile: str = typer.Option("local", "--profile"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Serve the Web UI + API (dashboard at http://<host>:<port>/)."""
    import os

    import uvicorn

    os.environ.setdefault("DRUGAGENT_PROFILE", profile)
    console.print(
        f"[green]drug-agent console[/green] → http://{host}:{port}/  (API docs at /docs)  profile={profile}"
    )
    uvicorn.run("drug_agent.api.server:app", host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# The pools — what this deployment can do
# ---------------------------------------------------------------------------
@app.command()
def agents(json: bool = typer.Option(False, "--json")) -> None:
    """List the agent pool: every agent, its kind, and the tools it uses."""
    from ..agents.registry import agent_pool

    cards = agent_pool()
    if json:
        _emit([c.model_dump(mode="json") for c in cards], True)
        return
    table = Table(title="agent pool")
    for col in ("agent_id", "kind", "title", "llm", "uses tools"):
        table.add_column(col)
    for card in cards:
        table.add_row(
            card.agent_id,
            str(card.kind),
            card.title,
            "yes" if card.llm_backed else "—",
            ", ".join(card.uses_tools) or "—",
        )
    console.print(table)


@app.command()
def tools(json: bool = typer.Option(False, "--json")) -> None:
    """List the tool pool: every tool, its operations, and its live status."""
    from ..tools.registry import tool_pool

    # Settings must be resolved the way `doctor` resolves them. Calling
    # `tool_pool()` bare builds a plain `Settings()`, which never reads
    # `configs/<profile>.yaml` and therefore never sees `external_tools` — the
    # documented, highest-precedence way to point a catalogue tool at the
    # interpreter that actually holds it. The two commands then disagree about the
    # same machine: `doctor` called four generators AVAILABLE here while `tools`
    # called them UNAVAILABLE. Under-reporting is the mirror of the mock this
    # project refuses to ship, and it is read by the parity check and the handoff
    # baseline, so it must describe the machine as configured.
    cards = tool_pool(load_settings().resolved())
    if json:
        _emit([c.model_dump(mode="json") for c in cards], True)
        return
    table = Table(title="tool pool")
    for col in ("tool_id", "status", "backend", "cost", "operations"):
        table.add_column(col)
    for card in cards:
        table.add_row(
            card.tool_id,
            str(card.status),
            card.backend,
            str(card.cost_class),
            ", ".join(card.operations) or "—",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# The request journal
# ---------------------------------------------------------------------------
journal_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect the running history: per-request journals, indexes, exports.",
)
app.add_typer(journal_app, name="journal")


def _journal_root() -> Path:
    from ..observability.journal import default_journal_root

    return default_journal_root()


@journal_app.command("list")
def journal_list(
    limit: int = typer.Option(20, "--limit"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """List past requests, newest first."""
    from ..observability.journal import list_requests

    root = _journal_root()
    rows = list_requests(root, limit=limit)
    if json:
        _emit({"root": str(root), "requests": rows}, True)
        return
    if not rows:
        console.print(f"No journals under [dim]{root}[/dim].")
        return
    table = Table(title=f"requests ({root})")
    for col in ("request_id", "status", "model", "source", "seconds", "chat", "repairs", "tool calls"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            str(row.get("request_id")),
            str(row.get("status")),
            str(row.get("model") or "—"),
            str(row.get("source") or "—"),
            str(row.get("duration_s") or "—"),
            str(row.get("n_exchanges") or 0),
            str(row.get("n_repairs") or 0),
            str(row.get("n_tool_calls") or 0),
        )
    console.print(table)


@journal_app.command("show")
def journal_show(
    request_id: str = typer.Argument(...),
    chat: bool = typer.Option(False, "--chat", help="Include the full chat log."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Show one request: what was asked, what ran, and how it went."""
    from ..observability.journal import load_request

    data = load_request(request_id, _journal_root())
    if data is None:
        err.print(f"[red]no journal for request {request_id!r}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    if json:
        _emit(data if chat else {**data, "chat": []}, True)
        return
    console.print(Path(data["summary_path"]).read_text())
    if chat:
        for entry in data["chat"]:
            console.print(
                f"\n[bold]{entry['phase']}#{entry['cycle']} attempt {entry['attempt']}[/bold] "
                f"({entry['mode']}, {entry['elapsed_s']}s, decode={entry['decode']}, ok={entry['ok']})"
            )
            console.print(f"[dim]{(entry.get('content') or '')[:1500]}[/dim]")


@journal_app.command("tools")
def journal_tools(
    limit: int = typer.Option(200, "--limit", help="How many recent requests to fold in."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Cross-request tool performance — the point of keeping journals."""
    from ..observability.journal import tool_statistics

    stats = tool_statistics(_journal_root(), limit=limit)
    if json:
        _emit(stats, True)
        return
    if not stats["tools"]:
        console.print(f"No tool usage recorded under [dim]{stats['root']}[/dim].")
        return
    table = Table(title=f"tool performance over {stats['n_requests']} requests")
    for col in ("tool_id", "operation", "backend", "requests", "calls", "errors", "mean ms", "metrics"):
        table.add_column(col)
    for tool in stats["tools"]:
        metrics = ", ".join(f"{k}={v['mean']}" for k, v in tool["metrics"].items()) or "—"
        table.add_row(
            tool["tool_id"],
            tool["operation"],
            str(tool.get("backend") or "—"),
            str(tool["n_requests"]),
            str(tool["n_calls"]),
            str(tool["n_errors"]),
            str(tool["mean_ms"]),
            metrics,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# The LLM console
# ---------------------------------------------------------------------------
CEILING_HELP = (
    "Hard ceiling for THIS RUN, as METRIC=VALUE. Repeatable, and a comma-separated "
    "list is accepted: --ceiling max_pIC50=6.5 --ceiling qed_median=0.8. Implies "
    "--ceiling-mode hard."
)
CEILING_MODE_HELP = (
    "off (default, unchanged behaviour) | hard (enforce the --ceiling values) | "
    "dynamic (show the planner the MEASURED envelope as advice, then pick the best "
    "candidate across all cycles in one maximum-reasoning step)."
)


def _apply_ceiling_options(ceiling: list[str] | None, ceiling_mode: str | None) -> None:
    """Validate the run-level ceiling options and hand them to the console process.

    The channel is the process environment, and deliberately so. ``console`` and
    ``console-headless`` are thin pass-throughs to ``drug_agent.llm.app`` and
    ``.headless``, each of which builds its own :class:`~drug_agent.llm.config.Config`
    from ``LLM_AGENT_*`` variables; putting the ceiling there gives it exactly the
    same one resolution path as every other launch parameter instead of a second,
    parallel one. It is also the property the mode requires: this writes to the
    environment of the process being launched and to nothing else — no config
    file, no profile, no runtime policy — so a ceiling lives for one run and is
    gone.

    Validation happens *here*, before anything starts, so an unknown metric is a
    usage error with the valid keys in it rather than a traceback ten seconds
    into a program.
    """
    import os

    from ..llm.config import CEILING_MODE_HARD, CeilingError, load_config, parse_ceilings

    try:
        ceilings = parse_ceilings(ceiling or [])
        if ceiling_mode is not None:
            os.environ["LLM_AGENT_CEILING_MODE"] = ceiling_mode
        elif ceilings:
            # A stated ceiling is a ceiling meant; without this a bare --ceiling
            # would be accepted, recorded, and silently ignored.
            os.environ["LLM_AGENT_CEILING_MODE"] = CEILING_MODE_HARD
        if ceilings:
            os.environ["LLM_AGENT_CEILINGS"] = ",".join(f"{k}={v}" for k, v in sorted(ceilings.items()))
        # Resolving a config built from what was just written surfaces the
        # contradictions (hard with nothing to enforce, a hard ceiling alongside
        # dynamic) at the command line rather than mid-run — including the ones
        # that come from a `.env` the operator forgot about.
        load_config().resolved_ceiling_mode()
    except CeilingError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_USER) from None


@app.command(
    name="console",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def console_cmd(
    ctx: typer.Context,
    ceiling: list[str] = typer.Option(None, "--ceiling", metavar="METRIC=VALUE", help=CEILING_HELP),
    ceiling_mode: str | None = typer.Option(None, "--ceiling-mode", help=CEILING_MODE_HELP),
) -> None:
    """Launch the LLM master-agent web console (needs a local Ollama).

    Flags (passed through to drug_agent.llm.app):

      --host / --port           where to serve (default 127.0.0.1:2003)
      --model                   Ollama model (default qwen3.8:27b)
      --ollama-host             Ollama endpoint
      --strategy                adaptive | naive (the refinement ablation)
      --max-cycles              refinement cycles per program
      --generator               library | pfm | library+pfm
      --n-generated             de-novo molecules sampled per cycle
      --pfm-pocket              prepared pocket PDB handed to PFM
      --structured              auto | on | off — schema-constrained decoding
      --journal-dir             where per-request journals are written
      --no-journal              disable request journaling
      --open                    open the console in a browser

    Handled here, because they apply to the run rather than to the server:

      --ceiling METRIC=VALUE    a HARD ceiling for this run only, repeatable
      --ceiling-mode            off | hard | dynamic
    """
    from ..llm.app import main as console_main

    _apply_ceiling_options(ceiling, ceiling_mode)
    console_main(list(ctx.args))


@journal_app.command("reindex")
def journal_reindex(json: bool = typer.Option(False, "--json")) -> None:
    """Rebuild `INDEX.md` and `index.jsonl` by scanning the request directories.

    Use after moving journals in by hand, or when an index looks wrong. Nothing
    inside a request directory is touched; both indexes are views over them.
    """
    from ..observability.history import rebuild_index

    stats = rebuild_index(_journal_root())
    if json:
        _emit(stats, True)
        return
    console.print(f"indexed [bold]{stats['n_requests']}[/bold] requests under [dim]{stats['root']}[/dim]")
    if stats["n_orphaned_rows"]:
        console.print(
            f"[yellow]{stats['n_orphaned_rows']} index row(s) have no directory[/yellow] — "
            "kept in orphaned-index-rows.jsonl: " + ", ".join(stats["orphaned"])
        )


@journal_app.command("migrate")
def journal_migrate(
    source: Path = typer.Argument(..., help="An older journal root to move requests out of."),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of moving."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Move journals from an older root into the running-history layout."""
    from ..observability.history import migrate

    if not source.is_dir():
        err.print(f"[red]not a directory: {source}[/red]")
        raise typer.Exit(EXIT_NOT_FOUND)
    stats = migrate(source, _journal_root(), copy=copy)
    if json:
        _emit(stats, True)
        return
    verb = "copied" if copy else "moved"
    console.print(f"{verb} [bold]{stats['n_moved']}[/bold] request(s) into [dim]{stats['root']}[/dim]")
    for entry in stats["moved"]:
        console.print(f"  {entry['request_id']} → {entry['to']}")
    if stats["skipped"]:
        console.print(f"[dim]skipped (already present): {', '.join(stats['skipped'])}[/dim]")
    if stats["n_orphaned_rows"]:
        console.print(
            f"[yellow]{stats['n_orphaned_rows']} index row(s) without files[/yellow] — "
            "kept in orphaned-index-rows.jsonl"
        )


@journal_app.command("export")
def journal_export(
    out: Path = typer.Option(None, "--out", help="Target directory (default: <history>/training)."),
    only_ok: bool = typer.Option(False, "--only-ok", help="Drop exchanges that failed to decode."),
    phase: list[str] = typer.Option(None, "--phase", help="Keep only these phases (repeatable)."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Export the chat dialogs as a training set (JSONL + dataset card).

    One record per model exchange: the messages as sent, the reply as returned,
    the reasoning channel kept separate, and the metadata needed to filter or
    split. Nothing is deduplicated, shuffled or split here — those are decisions
    for whoever builds the training run.
    """
    from ..observability.training import export_dialogs

    stats = export_dialogs(
        _journal_root(),
        out_dir=out,
        only_ok=only_ok,
        phases=tuple(phase) if phase else None,
    )
    if json:
        _emit(stats, True)
        return
    if not stats["n_dialogs"]:
        console.print(f"No chat exchanges found under [dim]{stats['root']}[/dim].")
        return
    console.print(
        f"wrote [bold]{stats['n_dialogs']}[/bold] dialogs from {stats['n_requests']} requests "
        f"→ [dim]{stats['out']}[/dim]"
    )
    table = Table(title="dialogs by phase")
    table.add_column("phase")
    table.add_column("records", justify="right")
    for name, n in stats["by_phase"].items():
        table.add_row(name, str(n))
    console.print(table)
    console.print(
        f"[dim]{stats['n_undecodable']} undecodable replies "
        f"{'excluded' if only_ok else 'retained and labelled'} · "
        f"{stats['n_with_thinking']} carry a reasoning channel[/dim]"
    )


# ---------------------------------------------------------------------------
# Self-evolution
# ---------------------------------------------------------------------------
from ..selfevolve.cli import app as self_evolve_app  # noqa: E402

app.add_typer(
    self_evolve_app,
    name="self-evolve",
    help="Study the platform's own request history and earn improvements to it.",
)


@app.command(
    name="console-headless",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def console_headless(
    ctx: typer.Context,
    ceiling: list[str] = typer.Option(None, "--ceiling", metavar="METRIC=VALUE", help=CEILING_HELP),
    ceiling_mode: str | None = typer.Option(None, "--ceiling-mode", help=CEILING_MODE_HELP),
) -> None:
    """Drive one master-agent program from the terminal (needs a local Ollama).

    Usage:

      drugagent console-headless [--ceiling M=V ...] [--ceiling-mode MODE]
                                 [adaptive|naive] [cycles] "instruction" [generator] [n]

    Example:

      drugagent console-headless adaptive 3 "Find potent, drug-like sEH inhibitors."
      drugagent console-headless adaptive 2 "Design novel sEH chemotypes." library+pfm 40

      # MODE 1 — the operator states what this deployment can reach, up front:
      drugagent console-headless --ceiling max_pIC50=6.5 --ceiling qed_median=0.8 \\
          adaptive 3 "Find potent, drug-like sEH inhibitors."

      # MODE 2 — the master agent is shown the MEASURED envelope as advice and
      # picks the best candidate across every cycle at the end:
      drugagent console-headless --ceiling-mode dynamic adaptive 3 "…"

    Prints the event stream as it happens and the journal path at the end. Run
    the same instruction under both strategies to compare their scoreboards.
    """
    from ..llm.headless import main as headless_main

    _apply_ceiling_options(ceiling, ceiling_mode)
    raise typer.Exit(headless_main(list(ctx.args)))


if __name__ == "__main__":
    app()
