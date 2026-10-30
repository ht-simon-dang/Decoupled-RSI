"""The evaluation kernel — the only thing allowed to say a candidate worked.

Responsibilities, in order:

1. **Admission.** Refuse a candidate that touches the guarded set
   (:mod:`.integrity`). A change to the judge is rejected before it is scored,
   not discovered after.
2. **The cascade.** AlphaEvolve's evaluation cascade, sized for this repository:

   ``gate 0 — invariants``  seconds. The existing test suite plus this
   package's own tests, executed *here*, with the command line, exit status,
   stdout and stderr captured. A model saying "all tests passed" is not
   evidence and is never accepted as any.

   ``gate 1 — replay``  seconds. Recorded model replies from the request journal
   are pushed back through the candidate's deterministic code paths. No model
   is called, so the comparison is exact.

   ``gate 2 — benchmark``  minutes per task. The real master agent runs real
   programs under the arm's policy, and the kernel re-executes and re-measures
   what came out.

3. **Evidence.** One :class:`~drug_agent.selfevolve.schemas.EvaluationRecord`
   per arm, carrying every command run, the environment, the kernel digest
   before and after, and the per-task rows. Records are written by the kernel to
   a directory nothing else in the package writes to.

The digest is taken before the first task and verified after the last. If the
kernel changed in between, the record is marked ``kernel_verified=False`` and
the promotion gate refuses it outright.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ...llm.config import Config
from ...llm.policy import DEFAULT_POLICY, RuntimePolicy
from ..config import EvolveConfig
from ..schemas import Candidate, EvaluationRecord, TaskResult
from ..target.worker import (
    CandidateMaterializationError,
    CandidateOriginError,
    CandidatePatchError,
    MaterializedCandidate,
    candidate_env,
    materialize,
    probe_origin,
    verify_origin,
)
from .benchmark import BENCHMARK_ID, BENCHMARK_VERSION, benchmark_fingerprint, load_benchmark
from .harness import run_task
from .integrity import (
    KernelIntegrityError,
    candidate_touches_kernel,
    diff_manifest,
    kernel_digest,
    kernel_manifest,
)

#: Commands gate 0 runs. Each is executed by the kernel and its exit status kept.
INVARIANT_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("platform regression suite", (sys.executable, "-m", "pytest", "-q", "-x", "tests")),
)


class EvaluationKernel:
    """Runs benchmarks and produces the only evidence the gate will accept."""

    def __init__(
        self,
        cfg: EvolveConfig | None = None,
        *,
        invariant_commands: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ) -> None:
        self.cfg = cfg or EvolveConfig()
        self.records_dir = self.cfg.resolved_store_dir() / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        #: Benchmark programs journal here, deliberately apart from the
        #: platform's request journal — see ``harness.run_task``.
        self.benchmark_journal_dir = self.cfg.resolved_store_dir() / "benchmark_journals"
        self.benchmark_journal_dir.mkdir(parents=True, exist_ok=True)
        #: Gate 0. Overridable so a test can exercise the cascade without
        #: launching a full pytest run inside a pytest run.
        self.invariant_commands = INVARIANT_COMMANDS if invariant_commands is None else invariant_commands

    # -- environment -----------------------------------------------------
    def environment(self) -> dict[str, Any]:
        import drug_agent

        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "drug_agent_version": getattr(drug_agent, "__version__", "unknown"),
            "git_commit": _git("rev-parse", "HEAD", cwd=self.cfg.repo_root()),
            "git_dirty": bool(_git("status", "--porcelain", cwd=self.cfg.repo_root())),
            "ollama_model": Config().model,
            "benchmark": {
                "id": BENCHMARK_ID,
                "version": BENCHMARK_VERSION,
                "fingerprint": benchmark_fingerprint(),
            },
            "kernel_digest": kernel_digest(),
        }

    # -- gate 0 ----------------------------------------------------------
    #: Environment prefixes stripped before gate 0 runs. The controller loads the
    #: repo's ``.env`` and seeds platform variables while analysing (the bridge
    #: does it deliberately, so a pipeline finds its fixtures), and inheriting
    #: those makes the invariant suite non-hermetic: it once failed
    #: ``test_pfm_requires_a_prepared_pocket`` purely because a pocket path the
    #: controller had loaded was visible to the child. A gate whose verdict
    #: depends on what the caller happened to touch is not an invariant.
    HERMETIC_STRIP_PREFIXES = ("DRUGAGENT_", "LLM_AGENT_")
    HERMETIC_STRIP_EXACT = ("OLLAMA_HOST",)

    def hermetic_env(self) -> dict[str, str]:
        """The child's environment, as close to a clean checkout as possible."""
        return {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(self.HERMETIC_STRIP_PREFIXES) and k not in self.HERMETIC_STRIP_EXACT
        }

    def run_command(
        self,
        label: str,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        timeout_s: float = 1800.0,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute one command in a hermetic environment and keep everything.

        The captured output is truncated at both ends rather than the middle:
        a pytest failure puts the useful part at the end, and a crash puts it at
        the start.

        ``env`` replaces the hermetic default outright. A candidate's gate 0
        supplies :func:`~drug_agent.selfevolve.target.worker.candidate_env`,
        which is hermetic *and* points the import path at the candidate — the
        hermetic default still inherits the operator's ``PYTHONPATH`` and
        editable install, which is precisely how gate 0 came to measure the
        wrong tree.
        """
        started = time.time()
        try:
            proc = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                cwd=str(cwd or self.cfg.repo_root()),
                timeout=timeout_s,
                check=False,
                env=self.hermetic_env() if env is None else dict(env),
            )
            out, err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", f"timed out after {timeout_s}s", 124
        except OSError as exc:
            out, err, code = "", f"{type(exc).__name__}: {exc}", 127
        return {
            "label": label,
            "argv": list(argv),
            "cwd": str(cwd or self.cfg.repo_root()),
            "exit_status": code,
            "ok": code == 0,
            "started_at": started,
            "finished_at": time.time(),
            "elapsed_s": round(time.time() - started, 3),
            "stdout_head": out[:2000],
            "stdout_tail": out[-4000:] if len(out) > 2000 else "",
            "stderr_tail": err[-4000:],
        }

    def run_invariants(
        self, *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Gate 0 against an explicit checkout.

        Called with no arguments this measures the operator's working tree,
        which is the right thing for a *baseline* and the wrong thing for a
        candidate. A candidate goes through :meth:`run_candidate_invariants`.
        """
        return [self.run_command(label, argv, cwd=cwd, env=env) for label, argv in self.invariant_commands]

    def run_candidate_invariants(
        self,
        mc: MaterializedCandidate,
        *,
        modules: tuple[str, ...] | None = None,
        timeout_s: float = 1800.0,
        log_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Gate 0 for a materialised candidate: its files, its interpreter, its imports.

        Three things happen that ``run_invariants()`` alone could not do:

        * the commands run **in the candidate checkout**, so a syntax error the
          candidate introduced is executed rather than skipped;
        * they run with the candidate's ``PYTHONPATH`` and interpreter, so the
          parent's editable install cannot answer the import;
        * a fresh subprocess reports where the code actually came from, and a
          gate whose code came from outside the candidate does not pass — it
          fails as an infrastructure fault, which is a different thing from the
          candidate being wrong.

        Returns the capture, never raises: gate 0's job is to produce evidence.
        """
        probe = probe_origin(mc, modules=modules, log_dir=log_dir)
        origin: str
        origin_error: str | None = None
        try:
            origin = verify_origin(mc, probe)
        except CandidateOriginError as exc:
            # Reported, not raised: gate 0's job is to produce evidence, and
            # "the code did not come from the candidate" is evidence.
            origin, origin_error = f"UNVERIFIED: {exc}", f"{type(exc).__name__}: {exc}"

        commands: list[dict[str, Any]] = []
        if origin_error is None:
            env = candidate_env(mc)
            for label, argv in self.invariant_commands:
                argv_c = (
                    (mc.interpreter, *tuple(argv[1:]))
                    if argv and argv[0] == sys.executable and mc.interpreter != sys.executable
                    else tuple(argv)
                )
                commands.append(self.run_command(label, argv_c, cwd=mc.root, timeout_s=timeout_s, env=env))
        return {
            "candidate_id": mc.candidate_id,
            "experiment_id": mc.experiment_id,
            "candidate_revision": mc.candidate_revision,
            "parent_commit": mc.parent_commit,
            "cwd": str(mc.root),
            "import_origin": origin,
            "origin_error": origin_error,
            "origin_probe": probe,
            "commands": commands,
            "passed": origin_error is None and bool(commands) and all(c["ok"] for c in commands),
        }

    def gate0(
        self,
        candidate: Candidate,
        *,
        keep_on_failure: bool = True,
        log_dir: Path | None = None,
        modules: tuple[str, ...] | None = None,
        timeout_s: float = 1800.0,
    ) -> dict[str, Any]:
        """Materialise ``candidate`` and run gate 0 inside it. JSON-ready, never raises.

        The single gate-0 entry point. Every caller — the controller, the CLI's
        ``evaluate`` and the web API's evaluate job — goes through here, because
        the defect being repaired was precisely that each of them called
        :meth:`run_invariants` with no candidate and got a verdict about the
        operator's checkout.

        On failure the workspace is kept by default, so whoever reads the record
        can go and look at the tree that failed.
        """
        experiment_id = f"gate0_{candidate.candidate_id}"
        try:
            mc = materialize(candidate, experiment_id=experiment_id, cfg=self.cfg)
        except (CandidatePatchError, CandidateMaterializationError) as exc:
            # Distinct on purpose: a patch that will not apply is the candidate's
            # own fault; a checkout that could not be created is not, and calling
            # the second one a rejected candidate would be a false negative.
            return {
                "candidate_id": candidate.candidate_id,
                "experiment_id": experiment_id,
                "candidate_revision": None,
                "parent_commit": None,
                "cwd": None,
                "import_origin": None,
                "origin_error": None,
                "origin_probe": {},
                "commands": [],
                "passed": False,
                "materialization_error": f"{type(exc).__name__}: {exc}",
                "invalid_candidate": isinstance(exc, CandidatePatchError),
                "workspace_cleanup": {"removed": False, "reason": "never created"},
            }
        capture = self.run_candidate_invariants(mc, modules=modules, timeout_s=timeout_s, log_dir=log_dir)
        capture["materialization_error"] = None
        capture["invalid_candidate"] = False
        capture["workspace_cleanup"] = mc.cleanup(keep=keep_on_failure and not capture["passed"])
        return capture

    # -- admission -------------------------------------------------------
    def admit(self, candidate: Candidate) -> list[str]:
        """Reasons this candidate may not be evaluated. Empty means admitted."""
        problems: list[str] = []
        touched = candidate_touches_kernel(candidate.files_changed, root=self.cfg.repo_root())
        if touched:
            problems.append(
                "candidate modifies files inside the evaluation kernel's integrity manifest: "
                + ", ".join(touched)
            )
        if candidate.patch and "selfevolve/evaluation/" in candidate.patch:
            problems.append("candidate patch references the evaluation kernel's source tree")
        if not candidate.overlay and not candidate.patch:
            problems.append("candidate carries neither an overlay nor a patch — nothing to evaluate")
        return problems

    # -- gate 2 ----------------------------------------------------------
    def evaluate_arm(
        self,
        *,
        arm: str,
        policy: RuntimePolicy,
        candidate_id: str = "baseline",
        splits: tuple[str, ...] = ("historical", "holdout"),
        task_ids: list[str] | None = None,
        cycles: int | None = None,
        repeats: int | None = None,
        commands: list[dict[str, Any]] | None = None,
    ) -> EvaluationRecord:
        """Run the benchmark under one policy and record what happened."""
        before = kernel_manifest()
        digest_before = kernel_digest(before)
        started = time.time()
        tasks = load_benchmark(splits=splits, task_ids=task_ids or (self.cfg.task_ids or None))
        cycles = cycles if cycles is not None else self.cfg.program_cycles
        repeats = repeats if repeats is not None else self.cfg.repeats

        results: list[TaskResult] = []
        error: str | None = None
        for rep in range(max(1, repeats)):
            for task in tasks:
                if time.time() - started > self.cfg.benchmark_timeout_s:
                    error = (
                        f"arm timed out after {self.cfg.benchmark_timeout_s}s with "
                        f"{len(results)}/{len(tasks) * max(1, repeats)} task runs complete"
                    )
                    break
                res = run_task(
                    task,
                    policy=policy,
                    arm=arm,
                    cycles=cycles,
                    timeout_s=min(2400.0, self.cfg.benchmark_timeout_s),
                    journal_dir=str(self.benchmark_journal_dir),
                )
                if repeats > 1:
                    res.task_id = f"{task.task_id}#r{rep}"
                results.append(res)
            if error:
                break

        after = kernel_manifest()
        digest_after = kernel_digest(after)
        verified = digest_before == digest_after
        if not verified:
            error = "EVALUATION VOID: the kernel changed during the measurement — " + json.dumps(
                diff_manifest(before, after)
            )

        record = EvaluationRecord(
            record_id=f"ev_{uuid.uuid4().hex[:12]}",
            candidate_id=candidate_id,
            arm=arm,
            mode="program",
            benchmark_id=BENCHMARK_ID,
            benchmark_version=BENCHMARK_VERSION,
            kernel_digest=digest_before,
            kernel_verified=verified,
            started_at=started,
            finished_at=time.time(),
            commands=list(commands or []),
            tasks=results,
            aggregates=aggregate(results),
            environment={
                **self.environment(),
                "policy": policy.public_dict(),
                "cycles": cycles,
                "repeats": repeats,
                "splits": list(splits),
            },
            ok=verified and error is None and all(r.ok for r in results),
            error=error,
        )
        self.write(record)
        return record

    # -- persistence -----------------------------------------------------
    def write(self, record: EvaluationRecord) -> Path:
        path = self.records_dir / f"{record.record_id}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, record_id: str) -> EvaluationRecord | None:
        path = self.records_dir / f"{record_id}.json"
        if not path.is_file():
            return None
        try:
            return EvaluationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_records(self) -> list[EvaluationRecord]:
        out: list[EvaluationRecord] = []
        for path in sorted(self.records_dir.glob("ev_*.json")):
            try:
                out.append(EvaluationRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda r: r.started_at)

    def verify_now(self, record: EvaluationRecord) -> None:
        """Re-check a stored record against the kernel as it is right now."""
        if kernel_digest() != record.kernel_digest:
            raise KernelIntegrityError(
                f"record {record.record_id} was produced under a different kernel "
                f"({record.kernel_digest[:16]}… vs {kernel_digest()[:16]}…)"
            )


# ---------------------------------------------------------------------------
def aggregate(results: list[TaskResult]) -> dict[str, Any]:
    """Arm-level numbers. Primary, guarded, and cost, kept separate on purpose.

    * ``decision_accuracy`` is the primary: does the agent's ACCEPT/REFINE agree
      with the gold rubric? It is what the historical record says is broken.
    * ``gold_score`` is the guard. A candidate that wins the primary by making
      its own criteria trivial will drop this, and the gate blocks on it.
    * cost is reported but never traded against quality silently.
    """
    # A failed task run has a *missing* scientific value and a recorded cost; it
    # is never an observation of zero. A non-finite score is a computational
    # fault for the same reason — it is not a number the rubric produced.
    failed = [r for r in results if not r.ok]
    finite = [r for r in results if r.ok and math.isfinite(r.gold_score)]
    nonfinite = len(results) - len(failed) - len(finite)
    ok = finite
    if not ok:
        return {
            "n_tasks": len(results),
            "n_ok": 0,
            "n_failed": len(failed),
            "n_nonfinite": nonfinite,
            "note": "no task completed",
        }
    gold = [r.gold_score for r in ok]
    correct = [0.0 if (r.false_accept or r.false_refine) else 1.0 for r in ok]
    by_split: dict[str, list[float]] = {}
    for r in ok:
        by_split.setdefault(_split_of(r.task_id), []).append(r.gold_score)
    return {
        "n_tasks": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_nonfinite": nonfinite,
        "n_unscorable_criteria": sum(int(r.independent.get("gold_unscorable") or 0) for r in ok),
        "n_invalid_criteria": sum(int(r.independent.get("gold_invalid") or 0) for r in ok),
        "gold_score": round(statistics.mean(gold), 4),
        "gold_score_min": round(min(gold), 4),
        "gold_met_total": sum(r.gold_met for r in ok),
        "gold_criteria_total": sum(r.gold_total for r in ok),
        "decision_accuracy": round(statistics.mean(correct), 4),
        "n_false_accept": sum(1 for r in ok if r.false_accept),
        "n_false_refine": sum(1 for r in ok if r.false_refine),
        "n_accept": sum(1 for r in ok if r.verdict == "ACCEPT"),
        "kind_accuracy": round(
            statistics.mean([1.0 if r.self_reported.get("kind_correct") else 0.0 for r in ok]), 4
        ),
        "infeasible_target_rate": _rate(sum(r.infeasible_targets for r in ok), sum(r.n_targets for r in ok)),
        "mean_llm_calls": round(statistics.mean([float(r.llm_calls) for r in ok]), 4),
        "mean_eval_tokens": round(statistics.mean([float(r.eval_tokens) for r in ok]), 1),
        "mean_wall_s": round(statistics.mean([r.wall_s for r in ok]), 2),
        "total_wall_s": round(sum(r.wall_s for r in ok), 2),
        "cross_check_flags": sum(len(r.independent.get("cross_check") or []) for r in ok),
        "by_split": {k: round(statistics.mean(v), 4) for k, v in sorted(by_split.items())},
    }


def _split_of(task_id: str) -> str:
    base = task_id.split("#", 1)[0]
    for task in load_benchmark():
        if task.task_id == base:
            return task.split
    return "unknown"


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _git(*args: str, cwd: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=str(cwd), check=False, timeout=15
        )
        return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def baseline_policy() -> RuntimePolicy:
    """The arm every candidate is compared against: the platform as shipped."""
    return DEFAULT_POLICY


__all__ = ["INVARIANT_COMMANDS", "EvaluationKernel", "aggregate", "baseline_policy"]
