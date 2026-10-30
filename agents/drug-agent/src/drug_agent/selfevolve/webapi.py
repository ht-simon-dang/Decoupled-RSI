"""The self-evolution subsystem's web surface, mounted on the console app.

The same operations `drugagent self-evolve …` exposes, over HTTP, so the loop can
be driven from the console that already shows the journals it studies.

Three shapes of endpoint, and the split matters:

``read``
    Instant, side-effect free: status, envelope, findings, benchmark, archive,
    records, cycles, reports. These read files the loop already wrote.

``job``
    Long and blocking — an analysis is seconds, a benchmark arm is half an hour.
    Launched into a background task, progress streamed as Server-Sent Events the
    same way a master-agent run is. One expensive job at a time, because the GPU
    and the local model are single.

``promote``
    The one write that changes what users get. It refuses without an explicit
    ``i_approve`` in the body, exactly as the CLI refuses without ``--i-approve``,
    and it refuses outright if the gate did not clear.

What is deliberately absent: any route that writes into the evaluation kernel.
The web layer can *start* a measurement and *read* its evidence; it cannot touch
the thing that produces the verdict.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import EvolveConfig

#: Jobs that cost real model time. Only one may run at a time.
EXPENSIVE = frozenset({"baseline", "evaluate", "step"})

#: What each job kind costs, shown in the UI before anything is spent.
JOB_COST: dict[str, str] = {
    "analyze": "~10 s · runs three probe pipelines, calls no model",
    "envelope": "~5 s · three probe pipelines, calls no model",
    "calibrate": "~10 s · five reference requirement sets through the real pipeline",
    "replay": "~1 s · recorded model replies re-decoded, calls no model",
    "bundle": "~5 s · copies every log into one directory with an index",
    "step_dry": "~5 min · one hypothesis + verification, no evaluation",
    "step_gates": "~7 min · hypothesis, gate 0 invariants, gate 1 replay",
    "step": "~40 min · the full cascade including a benchmark arm",
    "baseline": "~33 min · six real master-agent programs",
    "evaluate": "~33 min · six real master-agent programs for one candidate",
}


class JobRequest(BaseModel):
    kind: str
    candidate: str | None = None
    cluster: str | None = None
    operator: str | None = None
    parent: str | None = None
    splits: str | None = None


class PromoteRequest(BaseModel):
    #: Mirrors the CLI's ``--i-approve``. Absent or false is a refusal, not a default.
    i_approve: bool = False


@dataclass
class Job:
    """One background operation, with its event log."""

    job_id: str
    kind: str
    label: str
    started_at: float
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "RUNNING"  # RUNNING | SUCCEEDED | FAILED | CANCELLED
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    _queues: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set, repr=False)
    _task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def emit(self, kind: str, message: str, **data: Any) -> None:
        """Append one progress event and fan it out to every listener."""
        event = {
            "seq": len(self.log) + 1,
            "ts": time.time(),
            "job_id": self.job_id,
            "type": kind,
            "message": message,
            **data,
        }
        self.log.append(event)
        for q in list(self._queues):
            q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for event in self.log:  # replay the backlog so a late listener sees it all
            q.put_nowait(event)
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._queues.discard(q)

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "params": self.params,
            "error": self.error,
            "n_events": len(self.log),
        }


class JobRunner:
    """Runs blocking loop operations off the event loop, with progress."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def running(self, *, expensive_only: bool = False) -> list[Job]:
        return [
            j
            for j in self.jobs.values()
            if j.status == "RUNNING" and (not expensive_only or j.kind in EXPENSIVE)
        ]

    def start(self, kind: str, label: str, fn: Callable[[Job], Any], **params: Any) -> Job:
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:10]}",
            kind=kind,
            label=label,
            started_at=time.time(),
            params=params,
        )
        self.jobs[job.job_id] = job
        job.emit("started", label, cost=JOB_COST.get(kind, ""))

        async def _run() -> None:
            try:
                job.result = await asyncio.to_thread(fn, job)
                job.status = "SUCCEEDED"
                job.emit("finished", "done")
            except asyncio.CancelledError:
                job.status = "CANCELLED"
                job.emit("cancelled", "cancelled by the operator")
                raise
            except Exception as exc:  # a failed job is a result, not a crash
                job.status = "FAILED"
                job.error = f"{type(exc).__name__}: {exc}"
                job.emit("failed", job.error)
            finally:
                job.finished_at = time.time()
                job.emit("closed", job.status)

        job._task = asyncio.create_task(_run())
        return job


def _watch_arm_progress(job: Job, cfg: EvolveConfig, n_expected: int) -> Callable[[], None]:
    """Report benchmark-arm progress by watching the journals it writes.

    The kernel has no progress callback and I am not adding one — it is sealed
    for the duration of a measurement. But every completed task appends a line to
    the benchmark journal index, so counting those is honest progress that costs
    nothing and touches nothing.
    """
    index = cfg.resolved_store_dir() / "benchmark_journals" / "index.jsonl"
    seen = {"n": _count_lines(index)}
    stop = {"flag": False}

    def poll() -> None:
        while not stop["flag"]:
            time.sleep(5.0)
            now = _count_lines(index)
            if now > seen["n"]:
                seen["n"] = now
                done = max(0, now - (seen.get("base") or 0))
                job.emit(
                    "progress",
                    f"benchmark task {done}/{n_expected} complete",
                    done=done,
                    total=n_expected,
                )

    seen["base"] = seen["n"]
    import threading

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    return lambda: stop.__setitem__("flag", True)


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


# ---------------------------------------------------------------------------
def create_router(cfg: EvolveConfig | None = None) -> APIRouter:
    """The ``/api/evolve`` router. Every heavy import is deferred to a request."""
    cfg = cfg or EvolveConfig()
    router = APIRouter(prefix="/api/evolve", tags=["self-evolution"])
    runner = JobRunner()

    def _controller() -> Any:
        from .controller import EvolutionController

        return EvolutionController(cfg)

    def _kernel() -> Any:
        from .evaluation.kernel import EvaluationKernel

        return EvaluationKernel(cfg)

    def _archive() -> Any:
        from .archive import Archive

        return Archive(cfg)

    # ---- read ------------------------------------------------------------
    @router.get("/status")
    async def status() -> dict[str, Any]:
        """Backends, budgets, the guarded file set, and billing protection."""
        from .evaluation.integrity import kernel_digest, kernel_manifest

        ctl = await asyncio.to_thread(_controller)
        backends = await asyncio.to_thread(ctl.registry.status)
        manifest = kernel_manifest()
        store = cfg.resolved_store_dir()
        return {
            "config": cfg.public_dict(),
            "backends": backends,
            "budget": ctl.registry.budget(),
            "kernel": {
                "digest": kernel_digest(manifest),
                "n_guarded_files": len(manifest),
                "guarded_files": sorted(manifest),
                "note": "a candidate that changes any of these is quarantined before it is scored",
            },
            "billing": {
                "policy": (
                    "the Claude Code CLI runs on the operator's subscription; API-billing "
                    "variables are stripped from the child environment and nothing is written "
                    "to the shell configuration"
                ),
                "removed_from_child": backends["claude_cli"]["billing_vars_removed_from_child"],
                "present_in_this_shell": backends["claude_cli"]["billing_vars_in_parent_env"],
            },
            "store": str(store),
            "has_baseline": bool(await asyncio.to_thread(ctl.latest_baseline)),
            "auto_promote": cfg.auto_promote,
            "allow_internet": cfg.allow_internet,
            "job_costs": JOB_COST,
        }

    @router.get("/envelope")
    async def envelope(refresh: bool = False) -> dict[str, Any]:
        """What this deployment can actually achieve, measured by probe runs."""
        from ..llm.config import Config
        from ..tools.envelope_tool import measure_envelope

        env = await asyncio.to_thread(measure_envelope, Config(), refresh=refresh)
        return {**env.as_dict(), "prompt_block": env.as_prompt_block()}

    @router.get("/analysis")
    async def analysis(refresh: bool = False) -> dict[str, Any]:
        """Measured findings over the request journals, clustered by surface."""
        cached = cfg.resolved_store_dir() / "analysis.json"
        if not refresh and cached.is_file():
            try:
                return json.loads(cached.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        ctl = await asyncio.to_thread(_controller)
        return await asyncio.to_thread(ctl.analyze, refresh=refresh)

    @router.get("/benchmark")
    async def benchmark() -> dict[str, Any]:
        """The tasks and the gold rubric each is judged on."""
        from .evaluation.benchmark import describe

        return describe()

    @router.get("/records")
    async def records() -> dict[str, Any]:
        """Every evaluation record: the arms that have been run, with aggregates."""
        kernel = await asyncio.to_thread(_kernel)
        rows = await asyncio.to_thread(kernel.list_records)
        return {
            "records": [
                {
                    "record_id": r.record_id,
                    "arm": r.arm,
                    "candidate_id": r.candidate_id,
                    "mode": r.mode,
                    "benchmark": f"{r.benchmark_id}@{r.benchmark_version}",
                    "kernel_digest": r.kernel_digest[:16],
                    "kernel_verified": r.kernel_verified,
                    "ok": r.ok,
                    "error": r.error,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "aggregates": r.aggregates,
                    "n_tasks": len(r.tasks),
                }
                for r in rows
            ]
        }

    @router.get("/records/{record_id}")
    async def record_detail(record_id: str) -> dict[str, Any]:
        kernel = await asyncio.to_thread(_kernel)
        rec = await asyncio.to_thread(kernel.load, record_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no record {record_id!r}")
        return json.loads(rec.model_dump_json())

    @router.get("/archive")
    async def archive() -> dict[str, Any]:
        """The candidate archive: lineage, quality-diversity cells, status."""
        arc = await asyncio.to_thread(_archive)
        cands = await asyncio.to_thread(arc.all)
        return {
            "summary": arc.summary(),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "parent_id": c.parent_id,
                    "generation": c.generation,
                    "candidate_class": str(c.candidate_class),
                    "cell": c.cell(),
                    "status": str(c.status),
                    "hypothesis": c.hypothesis,
                    "author": c.author,
                    "created_at": c.created_at,
                    "overlay": c.overlay,
                    "failure_modes_addressed": c.failure_modes_addressed,
                    "notes": c.notes,
                }
                for c in cands
            ],
        }

    @router.get("/archive/{candidate_id}")
    async def candidate_detail(candidate_id: str) -> dict[str, Any]:
        arc = await asyncio.to_thread(_archive)
        cand = await asyncio.to_thread(arc.resolve, candidate_id)
        if cand is None:
            raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")
        decision_dir = cfg.resolved_store_dir() / "decisions"
        report_md = _read(decision_dir / f"{cand.candidate_id}.md")
        decision = _read_json(decision_dir / f"{cand.candidate_id}.json")
        return {
            "candidate": json.loads(cand.model_dump_json()),
            "lineage": [c.candidate_id for c in arc.lineage(cand.candidate_id)],
            "children": [c.candidate_id for c in arc.children(cand.candidate_id)],
            "records": arc.records_for(cand.candidate_id),
            "decision": decision,
            "report_md": report_md,
            "replay": _read_json(cfg.resolved_store_dir() / f"replay_{cand.candidate_id}.json"),
            "gate0": _read_json(cfg.resolved_store_dir() / f"gate0_{cand.candidate_id}.json"),
        }

    @router.get("/methods")
    async def methods() -> dict[str, Any]:
        """Every distinct method the loop proposed, folded across cycles."""
        from .methods import render_methods, summarize_methods

        summary = await asyncio.to_thread(summarize_methods, cfg)
        return {**summary, "markdown": render_methods(summary)}

    @router.get("/logs")
    async def logs() -> dict[str, Any]:
        """What logs exist, where, how big, and what each one answers."""
        from .bundle import INDEX

        root = cfg.repo_root() / ".drugagent"
        store = cfg.resolved_store_dir()

        def measure(files: list[Path], label: str, shown: str, what: str) -> dict[str, Any]:
            return {
                "log": label,
                "path": shown,
                "exists": bool(files),
                "n_files": len(files),
                "bytes": sum(f.stat().st_size for f in files),
                "what": what,
            }

        def tree(label: str, d: Path, what: str) -> dict[str, Any]:
            files = [f for f in d.rglob("*") if f.is_file()] if d.is_dir() else []
            return measure(files, label, str(d), what)

        def one(label: str, f: Path, what: str) -> dict[str, Any]:
            return measure([f] if f.is_file() else [], label, str(f), what)

        rows = [
            tree(
                "request journals (master agent)",
                cfg.resolved_journal_dir(),
                "one directory per user request: manifest · chat · tools · events · summary",
            ),
            tree(
                "benchmark journals (self-evolve)",
                store / "benchmark_journals",
                "one per benchmark task run; kept apart so the analyzer never studies them",
            ),
            one(
                "cycles.jsonl",
                store / "cycles.jsonl",
                "the evolver's own experience, one line per research cycle",
            ),
            tree("hypotheses/", store / "hypotheses", "statement · mechanism · falsifier · verdict"),
            tree("archive/", store / "archive", "candidates, lineage, novelty descriptors, records"),
            tree("records/", store / "records", "evaluation records — one per benchmark arm"),
            tree("decisions/", store / "decisions", "promotion reports: gate checks and the decision"),
            measure(
                sorted(store.glob("gate0_*.json")),
                "gate0_*.json",
                f"{store}/gate0_*.json",
                "gate 0 captures: argv, exit status, stdout, stderr",
            ),
            measure(
                sorted(store.glob("replay_*.json")),
                "replay_*.json",
                f"{store}/replay_*.json",
                "gate 1 evidence: recorded replies re-decoded, both arms",
            ),
            one("analysis.json", store / "analysis.json", "findings, clusters, attribution, envelope"),
            tree("episodes/", store / "episodes", "normalized experience — what the analysis reads"),
            one("envelope.json", root / "envelope.json", "what this deployment can be measured to achieve"),
            tree(
                "runs/ (pipeline bundles)",
                root / "runs",
                "full scientific output of CLI/API runs — master-agent cycles produce none",
            ),
            one("state.sqlite", root / "state.sqlite", "Engine run/task state, objects, events"),
            tree("artifacts/cas", root / "artifacts" / "cas", "content-addressed artifact blobs"),
        ]
        return {
            "logs": rows,
            "index": [e.__dict__ for e in INDEX],
            "store": str(store),
            "platform_state": str(root),
            "total_bytes": sum(r["bytes"] for r in rows),
        }

    @router.get("/cycles")
    async def cycles() -> dict[str, Any]:
        """The evolver's own experience — one row per research cycle."""
        ctl = await asyncio.to_thread(_controller)
        return {"cycles": await asyncio.to_thread(ctl.cycles)}

    @router.get("/hypotheses")
    async def hypotheses() -> dict[str, Any]:
        d = cfg.resolved_store_dir() / "hypotheses"
        rows = []
        if d.is_dir():
            for path in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                payload = _read_json(path)
                if payload:
                    rows.append(payload)
        return {"hypotheses": rows}

    @router.get("/compare/{candidate_id}")
    async def compare(candidate_id: str) -> dict[str, Any]:
        """Paired comparison of a candidate's latest arm against the baseline."""
        from .promotion import compare as compare_records
        from .promotion import decide, render_report

        ctl = await asyncio.to_thread(_controller)
        arc = ctl.archive
        cand = await asyncio.to_thread(arc.resolve, candidate_id)
        if cand is None:
            raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")
        base = await asyncio.to_thread(ctl.latest_baseline)
        recs = await asyncio.to_thread(ctl.kernel.list_records)
        mine = [r for r in recs if r.candidate_id == cand.candidate_id and r.arm != "baseline"]
        if base is None or not mine:
            raise HTTPException(
                status_code=409,
                detail="needs both a baseline record and an evaluated candidate record",
            )
        rec = mine[-1]
        decision = decide(cand, base, rec, cfg)
        return {
            "comparison": compare_records(base, rec, cfg),
            "decision": json.loads(decision.model_dump_json()),
            "report_md": render_report(cand, decision),
        }

    # ---- jobs ------------------------------------------------------------
    @router.get("/jobs")
    async def list_jobs() -> dict[str, Any]:
        return {"jobs": [j.summary() for j in sorted(runner.jobs.values(), key=lambda j: -j.started_at)]}

    @router.post("/jobs")
    async def create_job(req: JobRequest) -> dict[str, Any]:
        kind = req.kind
        if kind not in JOB_COST:
            raise HTTPException(status_code=400, detail=f"unknown job kind {kind!r}")
        if kind in EXPENSIVE and runner.running(expensive_only=True):
            busy = runner.running(expensive_only=True)[0]
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{busy.kind} is already running ({busy.job_id}). The local model is single, "
                    "so two benchmark arms at once would distort both."
                ),
            )
        fn, label = _job_for(kind, req, cfg, _controller, _kernel, _archive, runner)
        params = {k: v for k, v in req.model_dump(exclude_none=True).items() if k != "kind"}
        job = runner.start(kind, label, fn, **params)
        return {"job_id": job.job_id, "kind": kind, "label": label, "cost": JOB_COST[kind]}

    @router.get("/jobs/{job_id}")
    async def job_detail(job_id: str) -> dict[str, Any]:
        job = runner.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return {**job.summary(), "log": job.log, "result": job.result}

    @router.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = runner.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job._task is not None and not job._task.done():
            job._task.cancel()
        return {"job_id": job_id, "cancelled": True, "note": "the worker thread finishes its current step"}

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = runner.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")

        async def gen() -> AsyncIterator[str]:
            yield "retry: 3000\n\n"
            q = job.subscribe()
            try:
                while True:
                    event = await q.get()
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                    if event["type"] == "closed":
                        break
            finally:
                job.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # ---- promote ---------------------------------------------------------
    @router.post("/promote/{candidate_id}")
    async def promote(candidate_id: str, req: PromoteRequest) -> dict[str, Any]:
        """Write a candidate's overlay so the deployment picks it up.

        Refuses without ``i_approve``, and refuses outright if the gate did not
        clear — the same two refusals the CLI makes, for the same reason.
        """
        from .promotion import decide, render_report
        from .schemas import CandidateStatus

        ctl = await asyncio.to_thread(_controller)
        cand = await asyncio.to_thread(ctl.archive.resolve, candidate_id)
        if cand is None:
            raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")
        base = await asyncio.to_thread(ctl.latest_baseline)
        recs = await asyncio.to_thread(ctl.kernel.list_records)
        mine = [r for r in recs if r.candidate_id == cand.candidate_id and r.arm != "baseline"]
        if base is None or not mine:
            raise HTTPException(status_code=409, detail="no evaluation evidence for this candidate")
        decision = decide(cand, base, mine[-1], cfg)
        if decision.decision not in ("PROMOTE", "NEEDS_HUMAN"):
            raise HTTPException(
                status_code=409,
                detail=f"the gate says {decision.decision} — {decision.reasons}",
            )
        if not req.i_approve:
            return {
                "promoted": False,
                "requires_approval": True,
                "decision": json.loads(decision.model_dump_json()),
                "report_md": render_report(cand, decision),
                "note": (
                    "This changes how the user-serving system behaves. Re-send with "
                    "i_approve=true once you have read the report."
                ),
            }
        path = cfg.resolved_store_dir() / "active_overlay.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "candidate_id": cand.candidate_id,
                    "overlay": cand.overlay,
                    "promoted_at": decision.decided_at,
                    "evidence": {"baseline": base.record_id, "candidate": mine[-1].record_id},
                    "benchmark": f"{mine[-1].benchmark_id}@{mine[-1].benchmark_version}",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        await asyncio.to_thread(
            ctl.archive.set_status, cand.candidate_id, CandidateStatus.PROMOTED, "promoted from the console"
        )
        return {
            "promoted": True,
            "candidate_id": cand.candidate_id,
            "overlay_path": str(path),
            "activate": f"export DRUGAGENT_EVOLVE_OVERLAY={path}",
            "revert": "unset DRUGAGENT_EVOLVE_OVERLAY",
            "note": (
                "The overlay is written but NOT active in this process. Set the environment "
                "variable and restart the console (or the platform) to apply it; unsetting it reverts."
            ),
        }

    return router


# ---------------------------------------------------------------------------
def _job_for(
    kind: str,
    req: JobRequest,
    cfg: EvolveConfig,
    controller: Callable[[], Any],
    kernel: Callable[[], Any],
    archive: Callable[[], Any],
    runner: JobRunner,
) -> tuple[Callable[[Job], Any], str]:
    """Map a job kind onto the blocking function that performs it."""

    if kind == "analyze":

        def run_analyze(job: Job) -> Any:
            job.emit("step", "normalizing request journals and measuring the envelope")
            report = controller().analyze(refresh=True)
            job.emit(
                "step",
                f"{len(report['findings'])} findings over {len(report['clusters'])} clusters "
                f"from {report['n_episodes']} episodes",
            )
            return {k: report[k] for k in ("n_episodes", "findings", "clusters", "split")}

        return run_analyze, "Analyze the request history"

    if kind == "envelope":

        def run_envelope(job: Job) -> Any:
            from ..llm.config import Config
            from ..tools.envelope_tool import measure_envelope

            job.emit("step", "probing the real pipeline (unconstrained · constrained · phenotype)")
            env = measure_envelope(Config(), refresh=True)
            job.emit("step", f"{len(env.metrics)} metrics measured")
            return {**env.as_dict(), "prompt_block": env.as_prompt_block()}

        return run_envelope, "Re-measure the capability envelope"

    if kind == "calibrate":

        def run_calibrate(job: Job) -> Any:
            from .evaluation.calibration import run_calibration

            job.emit("step", "scoring five reference requirement sets against the gold rubric")
            result = run_calibration()
            for name, stratum in result["strata"].items():
                decided = stratum["discriminates"]
                state = "yes" if decided is True else ("no" if decided is False else "UNDECIDED")
                job.emit("step", f"{name} stratum discriminates: {state} — {stratum['verdict']}")
            job.emit("step", result["verdict"])
            return result

        return run_calibrate, "Calibrate the gold rubric"

    if kind == "replay":

        def run_replay(job: Job) -> Any:
            from ..llm.policy import DEFAULT_POLICY, RuntimePolicy
            from .evaluation.replay import run_replay as replay

            policy = DEFAULT_POLICY
            if req.candidate:
                cand = archive().resolve(req.candidate)
                if cand is None:
                    raise ValueError(f"no candidate {req.candidate!r}")
                policy = RuntimePolicy.from_overlay(cand.overlay)
            job.emit("step", "re-decoding recorded model replies — no model is called")
            root = cfg.resolved_journal_dir()
            return {"baseline": replay(DEFAULT_POLICY, root), "candidate": replay(policy, root)}

        return run_replay, f"Gate 1 replay{' · ' + req.candidate if req.candidate else ''}"

    if kind == "bundle":

        def run_bundle(job: Job) -> Any:
            from .bundle import collect

            job.emit("step", "copying every log into one directory — the originals are untouched")
            result = collect(cfg, slim=True)
            job.emit("step", f"{result['n_files']} files · {result['total_bytes'] / 1e6:.1f} MB")
            return result

        return run_bundle, "Bundle every log into one directory"

    if kind in ("step", "step_dry", "step_gates"):

        def run_step(job: Job) -> Any:
            ctl = controller()
            job.emit("step", "analysing, routing to a failure cluster, and asking for a hypothesis")
            stop = None
            if kind == "step":
                stop = _watch_arm_progress(job, cfg, _n_benchmark_tasks(cfg))
            try:
                log = ctl.step(
                    cluster_id=req.cluster or None,
                    operator=req.operator or None,
                    parent_id=req.parent or None,
                    dry_run=(kind == "step_dry"),
                    skip_benchmark=(kind == "step_gates"),
                )
            finally:
                if stop:
                    stop()
            job.emit("step", f"{log.decision} — {log.reason[:300]}")
            return json.loads(log.model_dump_json())

        label = {
            "step_dry": "One cycle (dry run — builds and admits, spends nothing)",
            "step_gates": "One cycle (gates 0 and 1 only)",
            "step": "One full research cycle (includes a benchmark arm)",
        }[kind]
        return run_step, label

    if kind == "baseline":

        def run_baseline(job: Job) -> Any:
            ctl = controller()
            job.emit("step", "gate 0 — running the platform's test suite")
            stop = _watch_arm_progress(job, cfg, _n_benchmark_tasks(cfg))
            try:
                record = ctl.baseline()
            finally:
                stop()
            job.emit("step", f"record {record.record_id} — {json.dumps(record.aggregates)[:300]}")
            return {
                "record_id": record.record_id,
                "ok": record.ok,
                "kernel_verified": record.kernel_verified,
                "aggregates": record.aggregates,
            }

        return run_baseline, "Measure the baseline (six real programs)"

    if kind == "evaluate":

        def run_evaluate(job: Job) -> Any:
            from ..llm.policy import RuntimePolicy

            arc = archive()
            cand = arc.resolve(req.candidate or "")
            if cand is None:
                raise ValueError(f"no candidate {req.candidate!r}")
            k = kernel()
            problems = k.admit(cand)
            if problems:
                raise ValueError("not admitted: " + "; ".join(problems))
            job.emit("step", "gate 0 — running the candidate's test suite in its own checkout")
            gate0 = k.gate0(cand)
            commands = gate0["commands"]
            if not gate0["passed"]:
                raise ValueError(
                    "gate 0 failed in "
                    + str(gate0["cwd"])
                    + ": "
                    + str(
                        gate0["materialization_error"]
                        or gate0["origin_error"]
                        or [c["exit_status"] for c in commands]
                    )
                )
            job.emit("step", "gate 2 — six real programs under the candidate's policy")
            stop = _watch_arm_progress(job, cfg, _n_benchmark_tasks(cfg))
            try:
                record = k.evaluate_arm(
                    arm=cand.candidate_id,
                    policy=RuntimePolicy.from_overlay(cand.overlay),
                    candidate_id=cand.candidate_id,
                    splits=tuple(
                        s.strip() for s in (req.splits or "historical,holdout").split(",") if s.strip()
                    ),
                    commands=commands,
                )
            finally:
                stop()
            arc.link_record(cand.candidate_id, record)
            job.emit("step", f"record {record.record_id}")
            return {
                "record_id": record.record_id,
                "ok": record.ok,
                "kernel_verified": record.kernel_verified,
                "aggregates": record.aggregates,
            }

        return run_evaluate, f"Evaluate {req.candidate} against the benchmark"

    raise HTTPException(status_code=400, detail=f"unknown job kind {kind!r}")


def _n_benchmark_tasks(cfg: EvolveConfig) -> int:
    from .evaluation.benchmark import load_benchmark

    return len(load_benchmark(task_ids=cfg.task_ids or None)) * max(1, cfg.repeats)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


__all__ = ["EXPENSIVE", "JOB_COST", "Job", "JobRunner", "create_router"]
