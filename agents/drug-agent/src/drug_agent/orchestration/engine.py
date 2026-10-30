"""The orchestration engine: a state-machine-driven task-graph executor.

Responsibilities: schedule ready tasks (bounded concurrency), enforce per-task
timeouts and retry policy, honour approval gates, propagate cancellation, resume
after interruption without duplicating completed work, cache task results keyed
by inputs+code+config, and emit an append-only event log. It is deliberately not
a chat loop — every transition is validated against the state machine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from typing import Any

from ..agents.base import AgentError, RetryableAgentError
from ..contracts.brief import ResearchBrief
from ..contracts.enums import ApprovalStatus, RunState, TaskState
from ..contracts.run import ApprovalGate, Run, Task
from ..memory.store import MemoryStore
from ..observability.telemetry import Telemetry, get_logger
from ..observability.toolmeter import carry_metering
from ..settings.config import Settings
from ..storage.artifacts import ArtifactStore, make_artifact_store
from ..storage.db import StateStore
from ..tools.llm_provider_tool import make_provider
from ..util import content_hash, current_id_scope, enable_determinism, utcnow_iso
from .context import RunCancelledError, RunContext
from .handlers import HANDLERS
from .registry import CapabilityRegistry
from .state_machine import assert_run_transition, assert_task_transition, is_terminal_task
from .workflow import build_task_graph

CODE_VERSION = "1.1.0"


class ApprovalPending(Exception):
    """Signals the run must pause for a human approval gate."""

    def __init__(self, gate: str) -> None:
        super().__init__(f"awaiting approval: {gate}")
        self.gate = gate


class UnknownGateError(KeyError):
    """A decision named a gate this run does not have.

    Subclasses ``KeyError`` to match the engine's idiom for unknown identifiers, and
    carries the valid names so callers can tell the operator what to type instead.
    """

    def __init__(self, gate: str, known: list[str]) -> None:
        self.gate = gate
        self.known = known
        super().__init__(f"unknown gate {gate!r}; this run has: {', '.join(known) or 'none'}")


class Engine:
    def __init__(
        self,
        settings: Settings,
        registry: CapabilityRegistry,
        store: StateStore | None = None,
        artifacts: ArtifactStore | None = None,
        record_history: bool | None = None,
    ) -> None:
        self.settings = settings.resolved()
        assert self.settings.db_url and self.settings.artifact_dir  # set by resolved()
        self.registry = registry
        self.store = store or StateStore(self.settings.db_url)
        self.artifacts = artifacts or make_artifact_store(
            self.settings.storage_backend, self.settings.artifact_dir
        )
        self.log = get_logger("engine")
        #: Whether executions are recorded in the running history. Off for the
        #: LLM bridge's nested pipeline runs, which their request journal
        #: already records — see :meth:`_history`.
        self.record_history = self.settings.record_history if record_history is None else record_history
        #: What to record as the *source* of a history entry. Entry points set
        #: it ("cli", "api", "demo"); the default says only that something drove
        #: the pipeline directly, which is still true when nobody sets it.
        self.history_source = "pipeline"
        #: Step 9's campaign budget ledger, when ``settings.budget_ledger_path``
        #: names one. The engine's task cache is the repository's real cache, and
        #: a warm one changes what a run *costs* without changing what it
        #: produces. Recording every lookup -- raw calls, hits, and unique misses
        #: kept apart per task -- is what keeps a warm arm's expensive-call
        #: accounting comparable with a cold arm's (Appendix A §4: cold-start
        #: total cost and warm-start deployment cost are separate tables).
        self.budget_ledger = self._open_budget_ledger()

    def _open_budget_ledger(self) -> Any:
        path = str(getattr(self.settings, "budget_ledger_path", "") or "").strip()
        if not path:
            return None
        try:
            from ..selfevolve.target.budget import BudgetLedger

            return BudgetLedger(path)
        except Exception as exc:  # pragma: no cover - a broken ledger must not kill a run
            self.log.warning("budget ledger at %s is unusable: %s", path, exc)
            return None

    def _record_cache_lookup(self, task: Task, *, hit: bool) -> None:
        """One cache lookup, recorded. Never allowed to fail a run."""
        if self.budget_ledger is None or not task.cache_key:
            return
        try:
            self.budget_ledger.cache_event(task.cache_key, hit=hit, owner=task.name, resource="task_results")
        except Exception as exc:  # pragma: no cover - accounting is not the science
            self.log.warning("could not record a cache lookup for %s: %s", task.name, exc)

    # --- run lifecycle -----------------------------------------------------
    def create_run(
        self, brief: ResearchBrief, *, deterministic: bool | None = None, profile: str | None = None
    ) -> Run:
        det = self.settings.deterministic if deterministic is None else deterministic
        if det:
            enable_determinism()
        self.store.put_object("brief", brief, program_id=brief.program_id)
        run = Run(
            brief_id=brief.id,
            program_id=brief.program_id,
            profile=profile or self.settings.profile,
            deterministic=det,
            seed=self.settings.seed,
            state=RunState.CREATED,
            brief_hash=content_hash(brief.model_dump(mode="json")),
            config_hash=self.settings.config_hash(),
        )
        tasks = build_task_graph(brief, run.id, transient_task_retries=self.settings.transient_task_retries)
        run.task_ids = [t.id for t in tasks]
        gate_names = {t.approval_gate for t in tasks if t.approval_gate}
        run.gates = [ApprovalGate(name=g) for g in sorted(n for n in gate_names if n)]
        # Claim the run identity before anything is written under it. A
        # duplicate ID here means two runs would share one history, which the
        # store refuses rather than resolves — see `StateStore.insert_run`.
        self.store.insert_run(run)
        for t in tasks:
            self.store.save_task(t)
        self.store.save_run(run)
        self._event(
            run,
            "run_created",
            detail={
                "n_tasks": len(tasks),
                "kind": str(brief.kind),
                # What to pass to `enable_determinism(scope=...)` to replay this
                # run's identifiers. Empty outside deterministic mode.
                "id_scope": current_id_scope(),
            },
        )
        return run

    def _context(self, run: Run) -> RunContext:
        memory = MemoryStore(self.store, self.artifacts, program_id=run.program_id)
        provider = make_provider(self.settings.llm_provider, self.settings.llm_model)
        telemetry = Telemetry(run.id, mode=self.settings.telemetry)
        return RunContext(
            settings=self.settings,
            store=self.store,
            artifacts=self.artifacts,
            registry=self.registry,
            provider=provider,
            memory=memory,
            run=run,
            telemetry=telemetry,
        )

    def run(self, run_id: str) -> Run:
        with self._history(run_id, phase="run") as record:
            run = self._drive(run_id)
            record(run)
            return run

    def resume(self, run_id: str) -> Run:
        with self._history(run_id, phase="resume") as record:
            run = self._drive(run_id, resuming=True)
            record(run)
            return run

    # --- the running history ----------------------------------------------
    @contextmanager
    def _history(self, run_id: str, *, phase: str) -> Iterator[Callable[[Run], None]]:
        """Record this execution in the running history, and meter its tools.

        Hooked here rather than in each caller — CLI, API, demo script, service
        helper — because "which entry points remember to log" is exactly the
        question that should not have an answer. One pipeline run is one entry:
        starting it, pausing at a gate and resuming it all fold into the same
        directory, so the history shows units of work rather than invocations.

        Two ways out. ``record_history=False`` is set by the LLM bridge, whose
        pipeline runs are already recorded inside the request journal that drove
        them — journaling them again would double-count them and feed the
        analyzer its own tail. And journaling never fails a run: any error here
        is swallowed, because a lost log entry is not worth a lost run.
        """
        if not self.record_history:
            yield lambda _run: None
            return

        from ..observability.journal import KIND_PIPELINE, RunJournal

        journal: RunJournal | None = None
        try:
            run = self._load_run(run_id)
            brief = self.store.get_object("brief", run.brief_id) or {}
            journal = RunJournal(
                run_id,
                instruction=str(brief.get("title") or f"pipeline run {run_id}"),
                source=self.history_source,
                kind=KIND_PIPELINE,
                reopen=True,
                config={
                    "profile": run.profile,
                    "deterministic": run.deterministic,
                    "brief_id": run.brief_id,
                    "program_id": run.program_id,
                    "seed": run.seed,
                    "objectives": [o.get("name") for o in (brief.get("objectives") or [])],
                },
            )
        except Exception:
            journal = None

        def record(final: Run) -> None:
            if journal is None:
                return
            try:
                tasks = self.store.list_tasks(final.id)
                journal.record_pipeline(
                    phase=phase,
                    run_id=final.id,
                    state=str(final.state),
                    n_tasks=len(tasks),
                    n_failed=sum(1 for t in tasks if str(t["state"]).startswith("FAILED")),
                    n_candidates=len(self.store.list_objects("candidate", run_id=final.id)),
                )
            except Exception:
                pass

        if journal is None:
            yield lambda _run: None
            return
        try:
            with journal.metered(scope=f"pipeline-{phase}"):
                yield record
        except Exception as exc:
            journal.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            final_state = ""
            try:
                loaded = self.store.get_run(run_id)
                final_state = str((loaded or {}).get("state") or "")
            except Exception:
                pass
            # A run paused at an approval gate is not finished, but the entry has
            # to be readable now — `resume` reopens it and overwrites the outcome.
            journal.finish(
                final_state or "UNKNOWN",
                outcome={"run_id": run_id, "state": final_state, "runs_dir": str(self.settings.runs_dir)},
            )

    # --- approval gates ----------------------------------------------------
    def approve_gate(self, run_id: str, gate: str, *, by: str = "user", reason: str | None = None) -> Run:
        run = self._load_run(run_id)
        self._set_gate(run, gate, ApprovalStatus.APPROVED, by, reason)
        self.store.save_run(run)
        self._event(run, "gate_approved", detail={"gate": gate, "by": by})
        return run

    def reject_gate(self, run_id: str, gate: str, *, by: str = "user", reason: str = "") -> Run:
        run = self._load_run(run_id)
        self._set_gate(run, gate, ApprovalStatus.REJECTED, by, reason)
        # Cancel the gate task and its descendants.
        for tdata in self.store.list_tasks(run_id):
            t = Task.model_validate(tdata)
            if t.approval_gate == gate and not is_terminal_task(t.state):
                self._transition_task(run, t, TaskState.CANCELLED, reason="gate_rejected")
        self._transition_run(run, RunState.CANCELLED)
        run.error = f"gate '{gate}' rejected: {reason}"
        self.store.save_run(run)
        self._event(run, "gate_rejected", detail={"gate": gate, "reason": reason})
        return run

    def cancel(self, run_id: str) -> Run:
        run = self._load_run(run_id)
        self.store.put_object("control", {"id": f"ctl_{run_id}", "cancel": True}, run_id=run_id)
        for tdata in self.store.list_tasks(run_id):
            t = Task.model_validate(tdata)
            if not is_terminal_task(t.state):
                self._transition_task(run, t, TaskState.CANCELLED, reason="cancelled")
        if run.state not in {RunState.SUCCEEDED, RunState.FAILED}:
            self._transition_run(run, RunState.CANCELLED)
        self.store.save_run(run)
        self._event(run, "run_cancelled")
        return run

    def _set_gate(self, run: Run, gate: str, status: ApprovalStatus, by: str, reason: str | None) -> None:
        """Record a decision on one of *this run's* gates.

        The gate set is fixed at run creation from the task graph, so a name outside it
        can never unblock a task. Recording it anyway would look like success and do
        nothing — the trap being approving ``await_human_approval`` (the *task* name)
        and watching the run sit in AWAITING_APPROVAL forever.
        """
        known = [g.name for g in run.gates]
        if gate not in known:
            raise UnknownGateError(gate, known)
        for g in run.gates:
            if g.name == gate:
                g.status = status
                g.decided_by = by
                g.decided_at = utcnow_iso()
                g.reason = reason

    # --- core scheduler ----------------------------------------------------
    def _drive(self, run_id: str, resuming: bool = False) -> Run:
        run = self._load_run(run_id)
        if run.deterministic:
            enable_determinism()
        ctx = self._context(run)
        if run.state in {
            RunState.CREATED,
            RunState.AWAITING_APPROVAL,
            RunState.PAUSED,
            RunState.FAILED,
            RunState.RUNNING,
        }:
            self._transition_run(run, RunState.RUNNING)
        run.started_at = run.started_at or utcnow_iso()
        self.store.save_run(run)
        self._event(run, "run_resumed" if resuming else "run_started")

        # Recover tasks left mid-flight by an interrupted process: RUNNING /
        # VALIDATING tasks are reset to READY so they re-run. Completed tasks are
        # untouched, and deterministic re-runs upsert identical artifacts (CAS
        # de-dup), so no completed work is duplicated.
        for tdata in self.store.list_tasks(run_id):
            t = Task.model_validate(tdata)
            if t.state in {TaskState.RUNNING, TaskState.VALIDATING}:
                self._transition_task(run, t, TaskState.FAILED_RETRYABLE, reason="recovered")
                t.attempt = max(0, t.attempt - 1)
                self._transition_task(run, t, TaskState.READY, reason="recovered")

        try:
            while True:
                if self._cancel_requested(run_id):
                    self.cancel(run_id)
                    return self._load_run(run_id)
                tasks = [Task.model_validate(t) for t in self.store.list_tasks(run_id)]
                # Promote approved gate tasks back into the ready pool.
                for t in tasks:
                    if t.state == TaskState.AWAITING_APPROVAL and not self._is_blocking_gate(run, t):
                        self._transition_task(run, t, TaskState.READY, reason="gate_approved")
                        t.state = TaskState.READY
                ready = self._ready_tasks(tasks)
                if not ready:
                    break
                # Handle approval gates before executing.
                pending_gate = self._check_gates(run, ready)
                if pending_gate:
                    self._transition_run(run, RunState.AWAITING_APPROVAL)
                    self.store.save_run(run)
                    self._event(run, "run_paused_for_approval", detail={"gate": pending_gate})
                    return self._load_run(run_id)

                runnable = [t for t in ready if not self._is_blocking_gate(run, t)]
                self._execute_batch(ctx, run, runnable)

            run = self._finalize(run_id)
            return run
        except RunCancelledError:
            return self.cancel(run_id)

    def _ready_tasks(self, tasks: list[Task]) -> list[Task]:
        by_id = {t.id: t for t in tasks}
        ready: list[Task] = []
        for t in tasks:
            if t.state not in {TaskState.PENDING, TaskState.READY, TaskState.FAILED_RETRYABLE}:
                continue
            deps_ok = all(
                by_id.get(d) and by_id[d].state in {TaskState.SUCCEEDED, TaskState.SUPERSEDED}
                for d in t.depends_on
            )
            if deps_ok:
                ready.append(t)
        # stable order for determinism
        return sorted(ready, key=lambda t: (t.created_at, t.name))

    def _check_gates(self, run: Run, ready: list[Task]) -> str | None:
        for t in ready:
            if t.approval_gate and self._is_blocking_gate(run, t):
                # Move to AWAITING_APPROVAL and pause (via READY if needed).
                if t.state == TaskState.PENDING:
                    self._transition_task(run, t, TaskState.READY)
                if t.state != TaskState.AWAITING_APPROVAL:
                    self._transition_task(run, t, TaskState.AWAITING_APPROVAL)
                return t.approval_gate
        return None

    def _is_blocking_gate(self, run: Run, t: Task) -> bool:
        if not t.approval_gate:
            return False
        gate = next((g for g in run.gates if g.name == t.approval_gate), None)
        return gate is None or gate.status != ApprovalStatus.APPROVED

    # --- task execution ----------------------------------------------------
    def _execute_batch(self, ctx: RunContext, run: Run, tasks: list[Task]) -> None:
        if not tasks:
            return
        if self.settings.deterministic or self.settings.max_concurrency <= 1 or len(tasks) == 1:
            for t in tasks:
                self._execute_task(ctx, run, t)
            return
        # Bounded concurrency for independent ready tasks.
        workers = min(self.settings.max_concurrency, len(tasks))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task") as ex:
            # carry_metering: a worker thread starts with an empty context, so a
            # ledger bound by the caller is invisible in here — and invisible in a
            # way that looks like "no tools were used" rather than like an error.
            work = carry_metering(lambda t: self._execute_task(ctx, run, t))
            list(ex.map(work, tasks))

    def _cache_key(self, run: Run, task: Task) -> str:
        # Keyed by inputs + code + config + tool/data versions. Scoped to the run
        # because handlers write run-scoped side effects: cross-run skipping would
        # omit those, so identical reruns rely on determinism (identical artifact
        # hashes), while the cache guards within-run idempotency on resume.
        return content_hash(
            {
                "run_id": run.id,
                "name": task.name,
                "params": task.params,
                "brief_hash": run.brief_hash,
                "config_hash": run.config_hash,
                "code_version": CODE_VERSION,
                "deterministic": run.deterministic,
                "deps": sorted(task.depends_on),
            }
        )

    def _execute_task(self, ctx: RunContext, run: Run, task: Task) -> None:
        # Fresh load to avoid stale state under concurrency.
        task = Task.model_validate(self.store.get_task(task.id))
        if is_terminal_task(task.state):
            return
        task.cache_key = self._cache_key(run, task)

        # Cross-run cache: reuse an identical, deterministic completed result.
        cached = self.store.cache_get(task.cache_key) if run.deterministic else None
        usable = bool(cached) and task.name not in _NEVER_CACHE
        if run.deterministic:
            # Counted whether or not it hit, so raw calls, hits and unique misses
            # stay three separate numbers rather than one derived from another.
            self._record_cache_lookup(task, hit=usable)
        if task.state in {TaskState.PENDING, TaskState.FAILED_RETRYABLE}:
            self._transition_task(run, task, TaskState.READY)

        if usable and cached:
            task.cache_hit = True
            task.result_ref = cached.get("result_ref")
            self._transition_task(run, task, TaskState.RUNNING)
            self._transition_task(run, task, TaskState.SUCCEEDED, cache_hit=True)
            return

        self._transition_task(run, task, TaskState.RUNNING)
        task.started_at = utcnow_iso()
        task.attempt += 1

        handler = HANDLERS.get(task.name)
        if handler is None:
            self._fail(run, task, f"no handler for task {task.name}", final=True)
            return

        try:
            payload = self._run_with_timeout(ctx, task, handler)
        except RunCancelledError:
            raise
        except (AgentError, RetryableAgentError, Exception) as exc:
            # AgentError means the agent decided this is not worth retrying; anything
            # else is treated as transient. FutureTimeout used to be excluded here to
            # keep the engine's own budget non-retryable, but the budget no longer
            # raises (see `_run_with_timeout`), so that clause only reclassified
            # timeouts a *handler* raised from its own futures — a policy nobody
            # chose. Those are transient like any other.
            retryable = isinstance(exc, RetryableAgentError) or not isinstance(exc, AgentError)
            can_retry = retryable and task.attempt < task.retry_policy.max_attempts
            if can_retry:
                self._transition_task(
                    run, task, TaskState.FAILED_RETRYABLE, error=str(exc), error_type=type(exc).__name__
                )
                self._transition_task(run, task, TaskState.READY, reason="retry")
            else:
                self._fail(run, task, str(exc), final=True, error_type=type(exc).__name__)
            return

        # Validate + persist result.
        self._transition_task(run, task, TaskState.VALIDATING)
        ref = self.artifacts.put_json(
            {"task": task.name, "payload": payload}, kind="json", produced_by=task.name, run_id=run.id
        )
        task.result_ref = ref.uri
        task.finished_at = utcnow_iso()
        if run.deterministic and task.name not in _NEVER_CACHE:
            self.store.cache_put(task.cache_key, {"result_ref": ref.uri, "payload": payload}, run.id)
        self._transition_task(run, task, TaskState.SUCCEEDED, detail={"payload": payload})

    def _task_budget(self, task: Task) -> tuple[float, int | None]:
        """This task's time budget, and the batch size it was scaled to (``None`` if flat).

        An explicit ``task.timeout_seconds`` wins. Otherwise the steps named in
        ``settings.per_candidate_budget_tasks`` — the ones that walk the whole
        candidate set — are sized per candidate, with the flat
        ``default_task_timeout_s`` kept as a floor so a small batch still has room
        for imports and model loading. Rationale and the measurements behind the
        default rate are in ``settings/config.py``.
        """
        if task.timeout_seconds:
            return float(task.timeout_seconds), None
        flat = float(self.settings.default_task_timeout_s)
        if task.name not in set(self.settings.per_candidate_budget_tasks):
            return flat, None
        # Counting rows is a few ms next to the seconds-per-candidate work that
        # follows, and it is the only measure of the batch the engine can see:
        # the handler's own view of "how many" lives inside the agent.
        n_candidates = len(self.store.list_objects("candidate", run_id=task.run_id))
        return max(flat, float(self.settings.per_candidate_task_timeout_s) * n_candidates), n_candidates

    def _run_with_timeout(self, ctx: RunContext, task: Task, handler: Any) -> dict[str, Any]:
        budget, batch_size = self._task_budget(task)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="taskrun") as ex:
            # Every task handler runs through here, so this is *the* place the
            # caller's context has to cross the thread boundary.
            fut = ex.submit(carry_metering(handler), ctx, task)
            try:
                return fut.result(timeout=budget)
            except FutureTimeout:
                # Be plain about what this budget is and is not. It CANNOT terminate
                # a handler, and never could: `with ThreadPoolExecutor(...)` exits
                # through shutdown(wait=True) and a RUNNING future is not
                # cancellable, so the block below waits for the worker whatever we
                # do here. Raising bought zero seconds and destroyed finished work —
                # one sEH de-novo run overran by 2.8% (123.5 s against 120 s), and
                # the engine threw away 124 completed MMFF94s minimisations, failed
                # the task non-retryably, and with it the whole run.
                #
                # So an overrun is an OBSERVATION, not a failure: the payload that
                # arrived is returned, and the overrun is recorded loudly enough to
                # act on. A genuinely hung handler still hangs the run; killing one
                # needs a cancellable handler (a subprocess, or a cooperative
                # cancellation token the handler checks) and is out of scope here.
                try:
                    return fut.result()
                finally:
                    self._record_budget_overrun(ctx, task, budget, time.monotonic() - started, batch_size)

    def _record_budget_overrun(
        self, ctx: RunContext, task: Task, budget: float, elapsed: float, batch_size: int | None
    ) -> None:
        """Emit the alarm for a task that ran past its budget but still finished."""
        detail: dict[str, Any] = {
            "task_name": task.name,
            "budget_s": round(budget, 3),
            "elapsed_s": round(elapsed, 3),
            "over_by_s": round(elapsed - budget, 3),
            "batch_size": batch_size,
        }
        self._event(ctx.run, "task_budget_overrun", task_id=task.id, detail=detail)
        self.log.warning("task_budget_overrun", run_id=ctx.run.id, task_id=task.id, **detail)

    def _fail(self, run: Run, task: Task, error: str, *, final: bool, error_type: str | None = None) -> None:
        self._transition_task(
            run,
            task,
            TaskState.FAILED_FINAL if final else TaskState.FAILED_RETRYABLE,
            error=error,
            error_type=error_type,
        )

    # --- transitions + events ---------------------------------------------
    def _transition_task(
        self,
        run: Run,
        task: Task,
        dst: TaskState,
        *,
        error: str | None = None,
        error_type: str | None = None,
        cache_hit: bool = False,
        reason: str | None = None,
        detail: dict | None = None,
    ) -> None:
        src = task.state
        assert_task_transition(src, dst)
        task.state = dst
        if error:
            task.error = error
            task.error_type = error_type
        if cache_hit:
            task.cache_hit = True
        if dst in {TaskState.SUCCEEDED, TaskState.FAILED_FINAL, TaskState.CANCELLED}:
            task.finished_at = task.finished_at or utcnow_iso()
        self.store.save_task(task)
        self._event(
            run,
            "task_transition",
            task_id=task.id,
            from_state=str(src),
            to_state=str(dst),
            detail={
                "name": task.name,
                "reason": reason,
                "error": error,
                "cache_hit": cache_hit,
                **(detail or {}),
            },
        )

    def _transition_run(self, run: Run, dst: RunState) -> None:
        assert_run_transition(run.state, dst)
        run.state = dst

    def _finalize(self, run_id: str) -> Run:
        run = self._load_run(run_id)
        tasks = [Task.model_validate(t) for t in self.store.list_tasks(run_id)]
        states = {t.state for t in tasks}
        if TaskState.FAILED_FINAL in states or TaskState.AWAITING_APPROVAL in states:
            # if a real failure blocks progress
            failed = [t for t in tasks if t.state == TaskState.FAILED_FINAL]
            if failed:
                self._transition_run(run, RunState.FAILED)
                # Name the tasks and their reasons. "one or more tasks failed" was
                # true of every failure and actionable for none: diagnosing it meant
                # re-reading the event log to find out which task and why.
                run.error = f"{len(failed)} task(s) failed: " + "; ".join(
                    f"{t.name} [{t.error_type or 'error'}]: {t.error or 'no reason recorded'}"
                    for t in sorted(failed, key=lambda t: (t.created_at, t.name))
                )
        done = {TaskState.SUCCEEDED, TaskState.SUPERSEDED, TaskState.CANCELLED}
        if run.state == RunState.RUNNING and all(t.state in done for t in tasks):
            self._transition_run(run, RunState.SUCCEEDED)
        elif run.state == RunState.RUNNING:
            # Nothing is runnable and the run did not complete. It used to be
            # left RUNNING *and* stamped finished_at *and* announced as
            # `run_finished`, which is three contradictions in one record: the
            # run reads as in-flight, as over, and as neither failed nor paused,
            # so `drugagent status` and every caller of `resume` saw a state that
            # could not be acted on. Say which of the two it actually is.
            stalled = [t for t in tasks if t.state not in done]
            if any(t.state == TaskState.AWAITING_APPROVAL for t in stalled):
                self._transition_run(run, RunState.AWAITING_APPROVAL)
            else:
                self._transition_run(run, RunState.PAUSED)
                run.error = run.error or (
                    "no task is runnable and the run did not finish; blocked on: "
                    + ", ".join(f"{t.name} [{t.state}]" for t in sorted(stalled, key=lambda t: t.name))
                )
        if run.state in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}:
            run.finished_at = utcnow_iso()
        self.store.save_run(run)
        self._event(run, "run_finished", detail={"state": str(run.state)})
        return run

    def _cancel_requested(self, run_id: str) -> bool:
        ctl = self.store.get_object("control", f"ctl_{run_id}")
        return bool(ctl and ctl.get("cancel"))

    def _load_run(self, run_id: str) -> Run:
        data = self.store.get_run(run_id)
        if not data:
            raise KeyError(f"unknown run: {run_id}")
        return Run.model_validate(data)

    def _event(
        self,
        run: Run,
        event: str,
        *,
        task_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        detail: dict | None = None,
    ) -> None:
        from ..contracts.run import TaskEvent

        self.store.append_event(
            TaskEvent(
                ts=utcnow_iso(),
                run_id=run.id,
                task_id=task_id,
                event=event,
                from_state=from_state,
                to_state=to_state,
                detail=detail or {},
                correlation_id=run.id,
            )
        )


# Tasks whose side effects must always re-run (they emit external artifacts).
_NEVER_CACHE = {"generate_report", "create_reproducibility_bundle"}
