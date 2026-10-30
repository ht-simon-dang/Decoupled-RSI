"""``drugagent self-evolve`` — the operator interface to the evolution loop.

Follows the platform's own CLI conventions: human-readable by default, ``--json``
for machine consumption, and the same exit codes (0 ok, 1 unexpected, 2 user
error, 3 not found, 4 degraded).

The commands map onto the loop's stages, cheapest first, so an operator can stop
at any point and still have something useful:

    doctor      what is reachable, what it would cost, what is guarded
    analyze     read the journals and report measured findings
    envelope    what this deployment can actually achieve
    calibrate   show the gold rubric being calibrated against real runs
    baseline    measure the platform as shipped
    replay      the deterministic gate, on its own
    step        one research cycle
    run         bounded evolution
    methods     every distinct method the loop proposed, and what happened to it
    logs        what logs exist, where they are, and what each answers
    bundle      copy every log into one directory with an index
    archive     the candidate archive with lineage
    evaluate    run one candidate through the benchmark
    compare     paired comparison against the baseline
    report      the promotion report for one candidate
    promote     apply a candidate's overlay to the deployment (needs --i-approve)
    target-campaign  one bounded autonomous campaign against one protein target
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .archive import Archive
from .config import EvolveConfig
from .controller import EvolutionController
from .evaluation.integrity import kernel_digest, kernel_manifest
from .evaluation.kernel import EvaluationKernel
from .promotion import compare as compare_records
from .promotion import decide, render_report

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Self-evolution: study the platform's own history and earn improvements to it.",
)
console = Console()
err = Console(stderr=True)

EXIT_OK, EXIT_ERROR, EXIT_USER, EXIT_NOT_FOUND, EXIT_DEGRADED = 0, 1, 2, 3, 4


def _cfg(config: str | None = None) -> EvolveConfig:
    return EvolveConfig.load(config)


def _emit(data: Any, json_mode: bool, human: str | None = None) -> None:
    if json_mode:
        console.print_json(_json.dumps(data, default=str))
    elif human is not None:
        console.print(human)


@app.command()
def doctor(
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """What the loop can reach, what it may spend, and what is protected."""
    cfg = _cfg(config or None)
    ctl = EvolutionController(cfg)
    status = ctl.registry.status()
    manifest = kernel_manifest()
    payload = {
        "config": cfg.public_dict(),
        "backends": status,
        "budget": ctl.registry.budget(),
        "kernel": {
            "digest": kernel_digest(manifest),
            "guarded_files": sorted(manifest),
            "note": "a candidate that changes any of these is quarantined before it is scored",
        },
        "billing": {
            "policy": (
                "the Claude Code CLI is invoked with the operator's subscription login; "
                "API-billing variables are stripped from the child environment and nothing "
                "is written to the shell configuration"
            ),
            "vars_removed_from_child": status["claude_cli"]["billing_vars_removed_from_child"],
            "vars_present_in_this_shell": status["claude_cli"]["billing_vars_in_parent_env"],
        },
        "journals": str(cfg.resolved_journal_dir()),
        "store": str(cfg.resolved_store_dir()),
        "internet_allowed": cfg.allow_internet,
    }
    if json:
        _emit(payload, True)
        return
    table = Table(title="self-evolve doctor")
    for col in ("component", "status", "detail"):
        table.add_column(col)
    o, c = status["ollama"], status["claude_cli"]
    table.add_row(
        "ollama (cheap controller)",
        "[green]ready[/green]" if o["available"] else "[red]unavailable[/red]",
        f"{o['host']} · {o['model'] or '—'}" + ("" if o["available"] else f" · {o['reason']}"),
    )
    table.add_row(
        "claude CLI (deep worker)",
        "[green]ready[/green]" if c["available"] else "[red]unavailable[/red]",
        f"{c['binary'] or 'not on PATH'}" + ("" if c["available"] else f" · {c['reason']}"),
    )
    table.add_row(
        "billing protection",
        "[green]active[/green]",
        "child env stripped of "
        + ", ".join(status["claude_cli"]["billing_vars_removed_from_child"][:3])
        + ", …"
        + (
            f" (present in this shell: {status['claude_cli']['billing_vars_in_parent_env']})"
            if status["claude_cli"]["billing_vars_in_parent_env"]
            else ""
        ),
    )
    table.add_row(
        "evaluation kernel",
        "[green]sealed[/green]",
        f"{len(manifest)} files · {kernel_digest(manifest)[:16]}…",
    )
    table.add_row("internet", "off" if not cfg.allow_internet else "[yellow]on[/yellow]", "default is off")
    table.add_row("auto-promote", "off" if not cfg.auto_promote else "[yellow]ON[/yellow]", "default is off")
    console.print(table)
    console.print(f"journals: [dim]{cfg.resolved_journal_dir()}[/dim]")
    console.print(f"store:    [dim]{cfg.resolved_store_dir()}[/dim]")


@app.command()
def envelope(
    refresh: bool = typer.Option(False, "--refresh", help="Re-measure instead of using the cache."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """What this deployment can actually achieve, measured by probe runs."""
    from ..llm.config import Config
    from ..tools.envelope_tool import measure_envelope

    env = measure_envelope(Config(), refresh=refresh)
    if json:
        _emit(env.as_dict(), True)
        return
    console.print(env.as_prompt_block())


@app.command()
def calibrate(
    json: bool = typer.Option(False, "--json"),
    require: list[str] = typer.Option(
        [],
        "--require",
        help="Fail unless this reference stratum discriminates (repeatable, e.g. --require target).",
    ),
) -> None:
    """Score hand-written requirement sets against the gold rubric.

    This is the audit trail for the benchmark's thresholds: a competent
    transcription should clear them and a careless one should not.

    **It fails closed.** Exit status is 4 (degraded) unless the calibration is
    valid, so a shell gate reading ``$?`` cannot be told "the rubric
    discriminates" on a deployment where a whole stratum could not be scored.
    Repair R1 made the *record* tri-state for that reason; until Step 7 the
    invocation still exited 0 regardless. ``--require target`` gates on the one
    stratum a target-based program depends on.
    """
    from .evaluation.calibration import calibration_exit_code, run_calibration

    result = run_calibration()
    code, reason = calibration_exit_code(result, required=list(require))
    if json:
        _emit({**result, "exit_code": code, "exit_reason": reason}, True)
        raise typer.Exit(code)
    table = Table(title="gold rubric calibration")
    for col in ("reference set", "status", "gold", "score", "missed", "unavailable"):
        table.add_column(col)
    for row in result["rows"]:
        table.add_row(
            row["name"],
            row["status"],
            f"{row['met']}/{row['total']}",
            "—" if row["score"] is None else f"{row['score']:.3f}",
            ", ".join(row["missed"]) or "—",
            ", ".join(row["unavailable"]) or "—",
        )
    console.print(table)

    # Per stratum, because one blanket boolean over a stratum that could not be
    # scored at all is the misreport this table exists to prevent.
    strata = Table(title="discrimination by reference stratum")
    for col in ("stratum", "discriminates", "why"):
        strata.add_column(col)
    for name, stratum in result["strata"].items():
        decided = stratum["discriminates"]
        strata.add_row(
            name,
            "yes" if decided is True else ("no" if decided is False else "UNDECIDED"),
            stratum["verdict"],
        )
    console.print(strata)

    for row in result["rows"]:
        if row["status"] != "SCORED":
            console.print(f"[yellow]{row['name']}: UNSCORABLE — {row['unscorable_reason']}[/yellow]")
    console.print(f"pass threshold: {result['pass_threshold']}")
    decided = result["discriminates"]
    console.print(
        "discriminates: "
        + ("true" if decided is True else ("false" if decided is False else "UNDECIDED"))
        + (
            f"  (unscorable strata: {', '.join(result['unscorable_strata'])})"
            if result["unscorable_strata"]
            else ""
        )
    )
    console.print(f"[dim]{result['verdict']}[/dim]")
    if code != EXIT_OK:
        err.print(f"[red]calibration is not valid here:[/red] {reason}")
    raise typer.Exit(code)


@app.command()
def analyze(
    config: str = typer.Option("", "--config"),
    refresh: bool = typer.Option(False, "--refresh"),
    no_envelope: bool = typer.Option(False, "--no-envelope", help="Skip the probe runs."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Read the request journals and report what is measurably wrong."""
    ctl = EvolutionController(_cfg(config or None))
    report = ctl.analyze(refresh=refresh, with_envelope=not no_envelope)
    if json:
        _emit(report, True)
        return
    console.print(f"[bold]{report['n_episodes']} episodes[/bold] from {ctl.cfg.resolved_journal_dir()}")
    console.print(
        f"split — discovery: {len(report['split']['discovery'])}, holdout: {len(report['split']['holdout'])}"
    )
    table = Table(title="findings")
    for col in ("severity", "finding", "affected", "inference", "title"):
        table.add_column(col)
    colour = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "cyan", "INFO": "dim"}
    for f in report["findings"]:
        sev = f["severity"]
        table.add_row(
            f"[{colour.get(sev, 'white')}]{sev}[/{colour.get(sev, 'white')}]",
            f["finding_id"],
            f"{f['n_affected']}/{f['n_applicable']}",
            f["inference"],
            f["title"][:70],
        )
    console.print(table)
    ct = Table(title="failure clusters")
    for col in ("cluster", "severity", "surface", "summary"):
        ct.add_column(col)
    for c in report["clusters"]:
        ct.add_row(c["cluster_id"], c["severity"], c["surface"], c["summary"][:80])
    console.print(ct)
    console.print(f"[dim]written to {ctl.cfg.resolved_store_dir() / 'analysis.json'}[/dim]")


@app.command()
def attribution(
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Per-tool statistics, conditioned on task class, plus output utilization."""
    from .analysis.attribution import attribute_tools

    ctl = EvolutionController(_cfg(config or None))
    report = attribute_tools(ctl.store.load())
    if json:
        _emit(report, True)
        return
    table = Table(title=f"tool attribution over {report['n_episodes']} episodes")
    for col in ("tool", "episodes", "calls", "errors", "ms/call", "output used", "backends"):
        table.add_column(col)
    for t in report["tools"]:
        used = t["output_utilization"]
        table.add_row(
            t["tool_id"],
            str(t["n_episodes"]),
            str(t["n_calls"]),
            str(t["n_errors"]),
            str(t["mean_ms_per_call"]),
            f"[red]{used}[/red]" if used == 0.0 else str(used),
            ", ".join(t["backends"]) or "—",
        )
    console.print(table)
    console.print(f"[dim]{report['caveat']}[/dim]")


@app.command()
def baseline(
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Measure the platform as shipped. Everything else is compared against this."""
    ctl = EvolutionController(_cfg(config or None))
    record = ctl.baseline()
    payload = {
        "record_id": record.record_id,
        "ok": record.ok,
        "kernel_verified": record.kernel_verified,
        "aggregates": record.aggregates,
    }
    _emit(payload, json, f"[green]baseline[/green] {record.record_id}: {record.aggregates}")


@app.command()
def replay(
    candidate: str = typer.Argument("", help="Candidate id; omit to replay the default policy."),
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Gate 1: push recorded model replies through a policy. No model is called."""
    from ..llm.policy import DEFAULT_POLICY, RuntimePolicy
    from .evaluation.replay import run_replay

    cfg = _cfg(config or None)
    policy = DEFAULT_POLICY
    if candidate:
        cand = Archive(cfg).resolve(candidate)
        if cand is None:
            err.print(f"[red]no such candidate[/red] {candidate}")
            raise typer.Exit(EXIT_NOT_FOUND)
        policy = RuntimePolicy.from_overlay(cand.overlay)
    baseline_result = run_replay(DEFAULT_POLICY, cfg.resolved_journal_dir())
    candidate_result = run_replay(policy, cfg.resolved_journal_dir())
    payload = {"baseline": baseline_result, "candidate": candidate_result, "policy": policy.public_dict()}
    if json:
        _emit(payload, True)
        return
    table = Table(title="deterministic replay over recorded model output")
    for col in ("suite", "measure", "baseline", "candidate"):
        table.add_column(col)
    for name, suite in candidate_result["suites"].items():
        base = baseline_result["suites"][name]
        for key in sorted(k for k in suite if k.startswith("n_") and k != "n_cases"):
            table.add_row(name, key, str(base.get(key)), str(suite.get(key)))
        table.add_row(name, "n_cases", str(base.get("n_cases")), str(suite.get("n_cases")))
    console.print(table)


@app.command()
def step(
    config: str = typer.Option("", "--config"),
    cluster: str = typer.Option("", "--cluster", help="Force a failure cluster."),
    operator: str = typer.Option("", "--operator", help="Force a mutation operator."),
    parent: str = typer.Option("", "--parent", help="Parent candidate id (lineage)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build and admit, do not evaluate."),
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark", help="Gates 0 and 1 only."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """One research cycle: analyse → hypothesise → verify → build → evaluate → gate."""
    ctl = EvolutionController(_cfg(config or None))
    log = ctl.step(
        cluster_id=cluster or None,
        operator=operator or None,
        parent_id=parent or None,
        dry_run=dry_run,
        skip_benchmark=skip_benchmark,
    )
    payload = _json.loads(log.model_dump_json())
    _emit(
        payload,
        json,
        f"[bold]cycle {log.cycle}[/bold] cluster={log.cluster_id} candidate={log.candidate_id} "
        f"→ [cyan]{log.decision}[/cyan]\n{log.reason}",
    )


@app.command()
def run(
    config: str = typer.Option("", "--config"),
    max_candidates: int = typer.Option(0, "--max-candidates"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Bounded evolution: several cycles, stopping on budget or exhaustion."""
    ctl = EvolutionController(_cfg(config or None))
    logs = ctl.run(max_candidates=max_candidates or None, dry_run=dry_run)
    payload = [_json.loads(log.model_dump_json()) for log in logs]
    if json:
        _emit(payload, True)
        return
    for log in logs:
        console.print(
            f"cycle {log.cycle}: {log.candidate_id or '—'} → [cyan]{log.decision}[/cyan] · {log.reason}"
        )


@app.command(name="archive")
def archive_cmd(
    config: str = typer.Option("", "--config"),
    candidate: str = typer.Option("", "--candidate", help="Show one candidate's lineage."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """The candidate archive: cells, lineage, status."""
    arc = Archive(_cfg(config or None))
    if candidate:
        cand = arc.resolve(candidate)
        if cand is None:
            err.print(f"[red]no such candidate[/red] {candidate}")
            raise typer.Exit(EXIT_NOT_FOUND)
        payload = {
            "candidate": _json.loads(cand.model_dump_json()),
            "lineage": [c.candidate_id for c in arc.lineage(cand.candidate_id)],
            "children": [c.candidate_id for c in arc.children(cand.candidate_id)],
            "records": arc.records_for(cand.candidate_id),
        }
        _emit(payload, json, _json.dumps(payload, indent=2, default=str))
        return
    summary = arc.summary()
    cands = arc.all()
    if json:
        _emit({"summary": summary, "candidates": [_json.loads(c.model_dump_json()) for c in cands]}, True)
        return
    table = Table(title=f"candidate archive ({summary['n_candidates']})")
    for col in ("candidate", "gen", "parent", "cell", "status", "hypothesis"):
        table.add_column(col)
    for c in cands:
        table.add_row(
            c.candidate_id,
            str(c.generation),
            c.parent_id or "—",
            c.cell(),
            str(c.status),
            c.hypothesis[:60],
        )
    console.print(table)


@app.command()
def methods(
    config: str = typer.Option("", "--config"),
    full: bool = typer.Option(False, "--full", help="Print the whole Markdown summary."),
    output: str = typer.Option("", "--output", help="Write the Markdown to a file."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Every distinct method the loop proposed, and what happened to each.

    Folded across cycles: several hypotheses arriving at the same change are one
    method here, because that is how the archive treats them — and because that
    convergence is a finding rather than a duplicate.
    """
    from .methods import render_methods, summarize_methods

    summary = summarize_methods(_cfg(config or None))
    if json:
        _emit(summary, True)
        return
    text = render_methods(summary)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        console.print(f"[green]written[/green] {output}")
        return
    if full:
        console.print(text)
        return
    console.print(
        f"[bold]{summary['n_methods']} methods[/bold] from {summary['n_hypotheses']} hypotheses "
        f"across {summary['n_cycles']} cycles · {summary['n_arms']} benchmark arms measured"
    )
    table = Table(title="methods proposed")
    for col in ("#", "method", "class", "status", "outcome"):
        table.add_column(col)
    for i, m in enumerate(summary["methods"], 1):
        colour = {"PROMOTED": "green", "REJECTED": "yellow", "QUARANTINED": "red"}.get(m["status"], "white")
        table.add_row(
            str(i),
            m["title"],
            m["candidate_class"],
            f"[{colour}]{m['status']}[/{colour}]",
            m["outcome"][:60],
        )
    console.print(table)
    for i, m in enumerate(summary["methods"], 1):
        verdicts = ", ".join(f"{h['hypothesis_id']}:{h['verdict']}" for h in m["hypotheses"]) or "—"
        console.print(f"  [dim]{i}. hypotheses: {verdicts}[/dim]")
    if summary["unattached_hypotheses"]:
        console.print(f"  [dim]{len(summary['unattached_hypotheses'])} hypotheses produced no method[/dim]")
    console.print("\n[dim]--full prints the whole summary; --output writes it to a file.[/dim]")


@app.command()
def logs(
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """What logs exist, where they are, how big, and what each one answers."""
    from .bundle import INDEX

    cfg = _cfg(config or None)
    store = cfg.resolved_store_dir()
    root = cfg.repo_root() / ".drugagent"

    rows: list[dict[str, Any]] = []

    def note(label: str, files: list[Path], shown: str, what: str) -> None:
        rows.append(
            {
                "log": label,
                "path": shown,
                "exists": bool(files),
                "n_files": len(files),
                "bytes": sum(f.stat().st_size for f in files),
                "what": what,
            }
        )

    def note_tree(label: str, d: Path, what: str) -> None:
        note(label, [f for f in d.rglob("*") if f.is_file()] if d.is_dir() else [], str(d), what)

    def note_file(label: str, f: Path, what: str) -> None:
        note(label, [f] if f.is_file() else [], str(f), what)

    def note_glob(pattern: str, base: Path, what: str) -> None:
        note(pattern, sorted(base.glob(pattern)), f"{base}/{pattern}", what)

    note_tree(
        "request journals  (master agent)",
        cfg.resolved_journal_dir(),
        "one directory per user request: manifest · chat · tools · events · summary",
    )
    note_tree(
        "benchmark journals (self-evolve)",
        store / "benchmark_journals",
        "one per benchmark task run; kept apart so the analyzer never studies them",
    )
    note_file(
        "cycles.jsonl       (self-evolve)",
        store / "cycles.jsonl",
        "the evolver's own experience, one line per research cycle",
    )
    note_tree("hypotheses/", store / "hypotheses", "statement · mechanism · falsifier · verdict")
    note_tree("archive/", store / "archive", "candidates, lineage, novelty descriptors, records")
    note_tree("records/", store / "records", "evaluation records — one per benchmark arm")
    note_tree("decisions/", store / "decisions", "promotion reports: ten gate checks and the decision")
    note_glob("gate0_*.json", store, "gate 0 captures: argv, exit status, stdout, stderr")
    note_glob("replay_*.json", store, "gate 1 evidence: recorded replies re-decoded, both arms")
    note_file("analysis.json", store / "analysis.json", "findings, clusters, attribution, envelope")
    note_tree("episodes/", store / "episodes", "normalized experience — what the analysis reads")
    note_file("envelope.json", root / "envelope.json", "what this deployment can be measured to achieve")
    note_tree(
        "runs/  (pipeline bundles)",
        root / "runs",
        "full scientific output of CLI/API runs — master-agent cycles do not produce these",
    )
    note_file("state.sqlite", root / "state.sqlite", "Engine run/task state, objects, events")
    note_tree("artifacts/cas", root / "artifacts" / "cas", "content-addressed artifact blobs")

    if json:
        _emit({"logs": rows, "index": [e.__dict__ for e in INDEX]}, True)
        return
    table = Table(title="logs you can check")
    for col in ("log", "files", "size", "what it answers"):
        table.add_column(col)
    for r in rows:
        nbytes = int(r["bytes"])
        size = f"{nbytes / 1e6:.1f} MB" if nbytes > 1e6 else f"{nbytes / 1e3:.0f} KB"
        label = str(r["log"])
        table.add_row(
            label if r["exists"] else f"[dim]{label}[/dim]",
            str(r["n_files"]) if r["exists"] else "—",
            size if r["exists"] else "—",
            str(r["what"]),
        )
    console.print(table)
    console.print(f"[dim]platform state: {root}[/dim]")
    console.print(f"[dim]evolve store:   {store}[/dim]")
    console.print("\n[dim]`self-evolve bundle` copies all of them into one directory with an index.[/dim]")


@app.command(name="bundle")
def bundle_cmd(
    config: str = typer.Option("", "--config"),
    output: str = typer.Option("", "--output", help="Where to write the bundle."),
    slim: bool = typer.Option(False, "--slim", help="Omit per-request events.jsonl (the bulk)."),
    no_journals: bool = typer.Option(False, "--no-journals", help="Omit the journals entirely."),
    no_runs: bool = typer.Option(False, "--no-runs", help="Omit the scientific run bundles."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Copy every log into one directory, with an index and a hashed manifest.

    A copy, never a move: the originals stay where the code that wrote them
    expects to find them.
    """
    from .bundle import collect

    result = collect(
        _cfg(config or None),
        output=output or None,
        slim=slim,
        with_journals=not no_journals,
        with_run_bundles=not no_runs,
    )
    if json:
        _emit(result, True)
        return
    console.print(
        f"[green]bundle[/green] {result['bundle_dir']}\n"
        f"  {result['n_files']} files · {result['total_bytes'] / 1e6:.1f} MB · "
        f"{result['n_methods']} methods summarized"
    )
    for note in result["notes"]:
        console.print(f"  [dim]{note}[/dim]")
    console.print(f"\n  Start with [bold]{result['bundle_dir']}/INDEX.md[/bold]")


@app.command()
def evaluate(
    candidate: str = typer.Argument(...),
    config: str = typer.Option("", "--config"),
    splits: str = typer.Option("historical,holdout", "--splits"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Run one candidate through gate 0 and the benchmark."""
    from ..llm.policy import RuntimePolicy

    cfg = _cfg(config or None)
    arc = Archive(cfg)
    cand = arc.resolve(candidate)
    if cand is None:
        err.print(f"[red]no such candidate[/red] {candidate}")
        raise typer.Exit(EXIT_NOT_FOUND)
    kernel = EvaluationKernel(cfg)
    problems = kernel.admit(cand)
    if problems:
        _emit({"admitted": False, "problems": problems}, json, f"[red]not admitted:[/red] {problems}")
        raise typer.Exit(EXIT_USER)
    gate0 = kernel.gate0(cand)
    commands = gate0["commands"]
    if not gate0["passed"]:
        _emit(
            {"gate0": gate0["passed"], "cwd": gate0["cwd"], "import_origin": gate0["import_origin"]},
            json,
            f"[red]gate 0 failed in the candidate checkout[/red] {gate0['cwd']}",
        )
        raise typer.Exit(EXIT_USER)
    record = kernel.evaluate_arm(
        arm=cand.candidate_id,
        policy=RuntimePolicy.from_overlay(cand.overlay),
        candidate_id=cand.candidate_id,
        splits=tuple(s.strip() for s in splits.split(",") if s.strip()),
        commands=commands,
    )
    arc.link_record(cand.candidate_id, record)
    _emit(
        {"record_id": record.record_id, "ok": record.ok, "aggregates": record.aggregates},
        json,
        f"[green]{record.record_id}[/green] {record.aggregates}",
    )


@app.command()
def compare(
    candidate: str = typer.Argument(...),
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Paired comparison of a candidate's latest record against the baseline."""
    cfg = _cfg(config or None)
    ctl = EvolutionController(cfg)
    cand = ctl.archive.resolve(candidate)
    if cand is None:
        err.print(f"[red]no such candidate[/red] {candidate}")
        raise typer.Exit(EXIT_NOT_FOUND)
    base = ctl.latest_baseline()
    rec = _latest_record(ctl.kernel, cand.candidate_id)
    if base is None or rec is None:
        err.print("[red]need both a baseline record and a candidate record[/red]")
        raise typer.Exit(EXIT_USER)
    cmp = compare_records(base, rec, cfg)
    if json:
        _emit(cmp, True)
        return
    table = Table(title=f"{cand.candidate_id} vs baseline ({cmp['n_paired_tasks']} paired tasks)")
    for col in ("metric", "baseline", "candidate", "delta", "95% CI", "n+/n−"):
        table.add_column(col)
    for name, row in cmp["metrics"].items():
        paired = row.get("paired") or {}
        table.add_row(
            name,
            str(row["baseline"]),
            str(row["candidate"]),
            str(row["delta"]),
            str(paired.get("ci95") or "—"),
            f"{paired.get('n_positive', 0)}/{paired.get('n_negative', 0)}",
        )
    console.print(table)


@app.command()
def report(
    candidate: str = typer.Argument(...),
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """The promotion report: comparison, gate checks, and the decision."""
    cfg = _cfg(config or None)
    ctl = EvolutionController(cfg)
    cand = ctl.archive.resolve(candidate)
    if cand is None:
        err.print(f"[red]no such candidate[/red] {candidate}")
        raise typer.Exit(EXIT_NOT_FOUND)
    base = ctl.latest_baseline()
    rec = _latest_record(ctl.kernel, cand.candidate_id)
    if base is None or rec is None:
        err.print("[red]need both a baseline record and a candidate record[/red]")
        raise typer.Exit(EXIT_USER)
    decision = decide(cand, base, rec, cfg)
    if json:
        _emit(_json.loads(decision.model_dump_json()), True)
        return
    console.print(render_report(cand, decision))


@app.command()
def promote(
    candidate: str = typer.Argument(...),
    config: str = typer.Option("", "--config"),
    approve: bool = typer.Option(False, "--i-approve", help="Required. This changes the deployment."),
    output: str = typer.Option("", "--output", help="Where to write the overlay."),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Write a candidate's overlay so the deployment picks it up.

    Deliberately not automatic and deliberately not silent: it prints the exact
    environment variable an operator has to set, and it refuses without
    ``--i-approve`` even when every gate check passed.
    """
    cfg = _cfg(config or None)
    ctl = EvolutionController(cfg)
    cand = ctl.archive.resolve(candidate)
    if cand is None:
        err.print(f"[red]no such candidate[/red] {candidate}")
        raise typer.Exit(EXIT_NOT_FOUND)
    base = ctl.latest_baseline()
    rec = _latest_record(ctl.kernel, cand.candidate_id)
    if base is None or rec is None:
        err.print("[red]refusing:[/red] no evaluation evidence for this candidate")
        raise typer.Exit(EXIT_USER)
    decision = decide(cand, base, rec, cfg)
    if decision.decision not in ("PROMOTE", "NEEDS_HUMAN"):
        err.print(f"[red]refusing:[/red] the gate says {decision.decision} — {decision.reasons}")
        raise typer.Exit(EXIT_USER)
    if not approve:
        console.print(render_report(cand, decision))
        console.print(
            "\n[yellow]Not promoted.[/yellow] This changes how the user-serving system behaves. "
            "Re-run with [bold]--i-approve[/bold] once you have read the report above."
        )
        raise typer.Exit(EXIT_OK)
    path = Path(output) if output else cfg.resolved_store_dir() / "active_overlay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps(
            {
                "candidate_id": cand.candidate_id,
                "overlay": cand.overlay,
                "promoted_at": decision.decided_at,
                "evidence": {"baseline": base.record_id, "candidate": rec.record_id},
                "benchmark": f"{rec.benchmark_id}@{rec.benchmark_version}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    from .schemas import CandidateStatus

    ctl.archive.set_status(cand.candidate_id, CandidateStatus.PROMOTED, "promoted by operator")
    _emit(
        {"promoted": cand.candidate_id, "overlay_path": str(path)},
        json,
        f"[green]Promoted[/green] {cand.candidate_id}.\n"
        f"Overlay written to {path}\n\n"
        f"Activate it for the deployment with:\n"
        f"  [bold]export DRUGAGENT_EVOLVE_OVERLAY={path}[/bold]\n\n"
        f"To revert, unset that variable. Nothing else changed.",
    )


@app.command()
def cycles(
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """The evolver's own history — one line per cycle."""
    ctl = EvolutionController(_cfg(config or None))
    rows = ctl.cycles()
    if json:
        _emit(rows, True)
        return
    table = Table(title=f"evolution cycles ({len(rows)})")
    for col in ("cycle", "cluster", "hypothesis", "candidate", "decision", "wall s", "claude", "ollama"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r.get("cycle")),
            str(r.get("cluster_id") or "—"),
            str(r.get("hypothesis_id") or "—"),
            str(r.get("candidate_id") or "—"),
            str(r.get("decision") or "—"),
            str(r.get("wall_s")),
            str((r.get("backend_calls") or {}).get("claude_cli")),
            str((r.get("backend_calls") or {}).get("ollama")),
        )
    console.print(table)


@app.command()
def benchmark(json: bool = typer.Option(False, "--json")) -> None:
    """The benchmark and its gold rubric — what a candidate is actually judged on."""
    from .evaluation.benchmark import describe

    d = describe()
    if json:
        _emit(d, True)
        return
    console.print(f"[bold]{d['benchmark_id']}@{d['version']}[/bold]  fingerprint {d['fingerprint']}")
    for task in d["tasks"]:
        console.print(f"\n[cyan]{task['task_id']}[/cyan] ({task['split']} · {task['origin']})")
        console.print(f"  {task['instruction'][:160]}")
        for g in task["gold"]:
            console.print(f"    · {g['metric']} {g['op']} {g['value']}  [dim]{g['why'][:80]}[/dim]")


@app.command(name="target-campaign")
def target_campaign(
    campaign: str = typer.Option(..., "--campaign", help="path to the Campaign record to run"),
    output: str = typer.Option(..., "--output", help="campaign output root; also the resume point"),
    max_cycles: int = typer.Option(..., "--max-cycles", help="the declared cycle bound"),
    ledger_path: str = typer.Option("", "--ledger", help="default: <output>/budget.sqlite"),
    min_improvement: float = typer.Option(0.0, "--min-improvement"),
    per_cycle_output_tokens: float = typer.Option(0.0, "--per-cycle-output-tokens"),
    stop_on_first_improvement: bool = typer.Option(False, "--stop-on-first-improvement"),
    model: str = typer.Option("", "--model"),
    config: str = typer.Option("", "--config"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Target evolution mode: one bounded autonomous campaign against one target.

    Routes through EvolutionController.run_target_campaign, the entry point the
    research plan names for target mode. The bound is declared here and never
    raised by the run: a campaign that spends its cycles without improving
    terminates NO_IMPROVEMENT and exits 0, because a scientific outcome is not
    an execution failure.

    It makes real, billed Claude calls. Re-running the same --output resumes
    rather than restarts: spent cycles count against the same --max-cycles.
    """
    from .backends import ClaudeCLIBackend
    from .target.budget import BudgetLedger
    from .target.controller import (
        CONTROLLER_ACTIONS,
        ClaudeProposer,
        StopRule,
        check_campaign_registration,
        frozen_incumbent_for_campaign,
    )

    cfg = _cfg(config or None)
    record = _json.loads(Path(campaign).read_text(encoding="utf-8"))
    problems = check_campaign_registration(record, root=cfg.repo_root())
    if problems:
        for problem in problems:
            err.print(f"[red]refused[/red]: {problem}")
        raise typer.Exit(EXIT_USER)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(
        Path(ledger_path) if ledger_path else out / "budget.sqlite",
        campaign_id=str(record["campaign_id"]),
    )
    for resource, limit in sorted(dict(record["budget"]).items()):
        ledger.declare(str(resource), float(limit), note=f"declared by {Path(campaign).name}")

    # The number to beat is read out of the committed baseline record, and WHICH
    # record is chosen by the campaign's own track — through the same
    # FROZEN_INCUMBENT_BY_METRIC table scripts/research/run_campaign.py uses, so
    # this command and that script cannot disagree about it.
    #
    # They could before. This call was frozen_incumbent_and_references(), the
    # Track A loader, with no dispatch at all, while every campaign record on
    # disk is Track B — so a campaign run from this CLI was scored against a
    # quantity from a different experiment, and nothing downstream noticed. The
    # comment that used to sit here asserted the invariant instead of holding it.
    try:
        metric, incumbent, references = frozen_incumbent_for_campaign(record, root=cfg.repo_root())
    except KeyError as exc:
        err.print(f"[red]refused[/red]: {exc.args[0] if exc.args else exc}")
        raise typer.Exit(EXIT_USER) from exc
    except (OSError, ValueError) as exc:
        err.print(f"[red]blocked[/red]: the frozen baseline record is unreadable — {exc}")
        raise typer.Exit(EXIT_DEGRADED) from exc
    console.print(
        f"metric [bold]{metric}[/bold] · incumbent {incumbent['artifact_id']} "
        f"{incumbent['metric']}={incumbent['value']} (read from {incumbent['source']})"
    )

    backend = ClaudeCLIBackend(
        default_model=model or "sonnet",
        cwd=cfg.repo_root(),
        ledger=ledger,
        budget_owner="campaign_proposer",
    )
    usable, why = backend.available()
    if not usable:
        err.print(f"[red]blocked[/red]: the reasoning backend is unavailable — {why}")
        raise typer.Exit(EXIT_DEGRADED)

    ctl = EvolutionController(cfg)
    result = ctl.run_target_campaign(
        record,
        ledger=ledger,
        output_root=out,
        stop_rule=StopRule(
            max_cycles=max_cycles,
            stop_on_first_improvement=stop_on_first_improvement,
            min_improvement=min_improvement,
            per_cycle_output_tokens=per_cycle_output_tokens or None,
        ),
        incumbent=incumbent,
        references=references,
        allowed_actions=list(CONTROLLER_ACTIONS),
        proposer=ClaudeProposer(backend=backend, model=model or None),
    )
    if json:
        _emit(result, True)
    else:
        console.print(
            f"[bold]{result['campaign_id']}[/bold] → [cyan]{result['state']}[/cyan] "
            f"({result['terminal_reason']}) · {result['scientific_decision']}\n"
            f"{result['cycles_completed']}/{result['stop_rule']['max_cycles']} cycles · "
            f"{len(result['candidate_pool'])} candidates in the pool"
        )
    raise typer.Exit(int(result["exit_code"]))


def _latest_record(kernel: EvaluationKernel, candidate_id: str) -> Any:
    records = [r for r in kernel.list_records() if r.candidate_id == candidate_id and r.arm != "baseline"]
    return records[-1] if records else None


__all__ = ["app"]
