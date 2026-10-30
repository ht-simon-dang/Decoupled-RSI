"""The evolution controller.

One cycle, in order, with the cheap work first and the expensive work last:

```
episodes ─▶ envelope ─▶ detectors ─▶ clusters
                                        │
                     (cheap controller: which cluster is worth opening?)
                                        ▼
                          generator ─▶ verifier ─▶ [reviser]
                                        │
                                     candidate
                                        │
                     novelty check ──▶ archive ──▶ kernel admission
                                        │
              gate 0 invariants ─▶ gate 1 replay ─▶ gate 2 benchmark
                                        │
                              compare vs baseline ─▶ promotion gate
                                        │
                                    cycle log
```

The division of labour between models is a competence judgement, not a cost one:

* **Qwen, locally**, picks which failure cluster to open. That is a routing
  decision over a table of numbers the detectors already computed, which is what
  a 27B model is good at and what it costs nothing to ask.
* **Claude, through the CLI**, writes the hypothesis and tries to falsify it.
  That is architectural reasoning about a system, where being wrong is expensive
  and hard to notice.
* **Neither** decides whether the candidate worked. The kernel does, and it does
  not call a model at all.

Everything is budgeted and resumable. A cycle that runs out of Claude budget, or
hits a subscription limit, records what it got and stops cleanly — the archive
and the cycle log are the state, so the next invocation picks up from them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...llm.config import Config
from ...llm.policy import DEFAULT_POLICY, RuntimePolicy
from ..analysis import analyze, cluster_findings
from ..analysis.attribution import attribute_tools, utilization_findings
from ..archive import Archive
from ..backends import BackendRegistry
from ..candidates.library import BUILDERS_BY_SURFACE, build_candidate
from ..config import EvolveConfig
from ..evaluation.benchmark import describe as describe_benchmark
from ..evaluation.integrity import kernel_digest
from ..evaluation.kernel import EvaluationKernel
from ..evaluation.replay import applicable_suites, run_replay
from ..experience import ExperienceStore
from ..hypotheses import HypothesisEngine, context_for_cluster
from ..promotion import decide, render_report
from ..schemas import Candidate, CandidateStatus, CycleLog, EvaluationRecord

_CLUSTER_CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cluster_id": {"type": "string"},
        "reason": {"type": "string"},
        "escalate_to_deep_model": {"type": "boolean"},
    },
    "required": ["cluster_id", "reason"],
}

_ROUTER_SYSTEM = """\
You are the scheduler of a self-evolution loop. You are given a list of measured
failure clusters from an automated drug-discovery platform, each with a severity,
how many episodes it affects, and a one-line summary.

Pick exactly ONE cluster to investigate next, and say why in one sentence.

Choose on expected value, not on interest: prefer high severity, high affected
fraction, and a surface where a small change could plausibly help. Do not pick a
cluster that is already marked as addressed. Set escalate_to_deep_model to true
when the cluster needs architectural reasoning rather than a mechanical fix.

Reply with the JSON object only.
"""


class EvolutionController:
    """Runs analysis and evolution cycles, and records everything it did."""

    def __init__(self, cfg: EvolveConfig | None = None, *, kernel: EvaluationKernel | None = None) -> None:
        self.cfg = cfg or EvolveConfig()
        self.store = ExperienceStore(self.cfg)
        self.archive = Archive(self.cfg)
        #: Injectable so a test can exercise the real cascade without launching
        #: a full pytest run inside a pytest run. Never injected in production.
        self.kernel = kernel or EvaluationKernel(self.cfg)
        self.registry = BackendRegistry(self.cfg)
        self.engine = HypothesisEngine(self.cfg, self.registry)
        self.log_path = self.cfg.resolved_store_dir() / "cycles.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------
    def analyze(self, *, refresh: bool = False, with_envelope: bool = True) -> dict[str, Any]:
        """Everything the loop knows about the user-serving system right now."""
        episodes = self.store.load(refresh=refresh)
        envelope: dict[str, Any] = {}
        if with_envelope:
            try:
                from ...tools.envelope_tool import measure_envelope

                envelope = measure_envelope(Config()).as_dict()
            except Exception as exc:
                envelope = {"error": f"{type(exc).__name__}: {exc}"}
        findings = analyze(episodes, envelope=envelope) + utilization_findings(episodes)
        clusters = cluster_findings(findings)
        split = self.store.split(episodes)
        report = {
            "generated_at": time.time(),
            "n_episodes": len(episodes),
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "task_class": e.task_class,
                    "status": e.status,
                    "verdict": e.final_verdict,
                    "criteria_met_rate": e.criteria_met_rate,
                    "llm_calls": e.n_llm_calls,
                    "eval_tokens": e.eval_tokens,
                    "wall_s": e.duration_s,
                    "tool_calls": e.n_tool_calls,
                    "prompt_version": e.prompt_version,
                    "journal": e.journal_dir,
                }
                for e in episodes
            ],
            "split": {k: [e.episode_id for e in v] for k, v in split.items()},
            "envelope": envelope,
            "findings": [json.loads(f.model_dump_json()) for f in findings],
            "clusters": [json.loads(c.model_dump_json()) for c in clusters],
            "attribution": attribute_tools(episodes),
            "benchmark": describe_benchmark(),
            "kernel_digest": kernel_digest(),
        }
        self._write(self.cfg.resolved_store_dir() / "analysis.json", report)
        return report

    # ------------------------------------------------------------------
    # baseline
    # ------------------------------------------------------------------
    def baseline(self, *, splits: tuple[str, ...] = ("historical", "holdout")) -> EvaluationRecord:
        """Measure the platform as shipped. Everything else is compared to this."""
        # The one gate-0 call that is *supposed* to measure the operator's
        # checkout: the baseline arm is the parent. Stated explicitly so it is
        # not mistaken for the defect repaired in `_gate0`.
        commands = self.kernel.run_invariants(cwd=self.cfg.repo_root())
        record = self.kernel.evaluate_arm(
            arm="baseline",
            policy=DEFAULT_POLICY,
            candidate_id="baseline",
            splits=splits,
            commands=commands,
        )
        self._write(self.cfg.resolved_store_dir() / "baseline.json", {"record_id": record.record_id})
        return record

    def latest_baseline(self) -> EvaluationRecord | None:
        pointer = self.cfg.resolved_store_dir() / "baseline.json"
        if pointer.is_file():
            try:
                rid = json.loads(pointer.read_text(encoding="utf-8")).get("record_id")
                rec = self.kernel.load(rid) if rid else None
                if rec is not None:
                    return rec
            except (OSError, ValueError):
                pass
        candidates = [r for r in self.kernel.list_records() if r.arm == "baseline"]
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------
    # one cycle
    # ------------------------------------------------------------------
    def step(
        self,
        *,
        cycle: int = 1,
        cluster_id: str | None = None,
        operator: str | None = None,
        parent_id: str | None = None,
        dry_run: bool | None = None,
        skip_benchmark: bool = False,
    ) -> CycleLog:
        """One research cycle. Returns the log; every artefact is on disk."""
        dry_run = self.cfg.dry_run if dry_run is None else dry_run
        started = time.time()
        log = CycleLog(cycle=cycle, started_at=started)

        report = self.analyze()
        clusters = report["clusters"]
        if not clusters:
            log.reason = "no failure clusters — nothing to evolve"
            return self._finish(log, started)

        chosen = self._choose_cluster(clusters, cluster_id)
        log.cluster_id = chosen["cluster_id"]

        # --- hypothesis: generate → verify → maybe revise ----------------
        from ..schemas import FailureCluster, Finding

        cluster_obj = FailureCluster.model_validate(chosen)
        findings = [Finding.model_validate(f) for f in report["findings"]]
        episodes = self.store.load()
        context = context_for_cluster(cluster_obj, findings, episodes)

        hypothesis = self.engine.generate(context)
        hypothesis = self.engine.verify(hypothesis, context)
        if hypothesis.verdict == "REFUTED":
            revised = self.engine.revise(hypothesis, context)
            revised = self.engine.verify(revised, context)
            self._write_hypothesis(hypothesis)
            hypothesis = revised
        self._write_hypothesis(hypothesis)
        log.hypothesis_id = hypothesis.hypothesis_id

        if hypothesis.verdict == "REFUTED":
            log.decision = "ABANDONED"
            log.reason = (
                "the verifier refuted the hypothesis twice; recorded as a negative result rather "
                "than forced into a candidate"
            )
            return self._finish(log, started)

        # --- candidate ---------------------------------------------------
        parent = self.archive.get(parent_id) if parent_id else None
        candidate = None
        if operator:
            candidate = build_candidate(
                operator,
                parent_id=parent.candidate_id if parent else None,
                generation=(parent.generation + 1) if parent else 0,
                hypothesis_id=hypothesis.hypothesis_id,
                evidence=list(hypothesis.evidence),
                author="operator-override",
            )
        else:
            candidate = self.engine.to_candidate(
                hypothesis, parent=parent, generation=(parent.generation + 1) if parent else 0
            )
        if candidate is None:
            fallbacks = BUILDERS_BY_SURFACE.get(cluster_obj.surface, [])
            if not fallbacks:
                log.decision = "NO_CANDIDATE"
                log.reason = (
                    f"the hypothesis named no implementable operator and no builder exists for "
                    f"surface {cluster_obj.surface!r}"
                )
                return self._finish(log, started)
            candidate = build_candidate(
                fallbacks[0],
                hypothesis_id=hypothesis.hypothesis_id,
                evidence=list(hypothesis.evidence),
                author="surface-fallback",
            )
        candidate.hypothesis_id = hypothesis.hypothesis_id

        novel, why = self.archive.is_novel(candidate)
        if not novel:
            log.candidate_id = candidate.candidate_id
            log.decision = "SKIPPED_NOT_NOVEL"
            log.reason = why
            return self._finish(log, started)

        self.archive.add(candidate)
        log.candidate_id = candidate.candidate_id
        log.files_changed = list(candidate.files_changed)

        # --- admission ---------------------------------------------------
        problems = self.kernel.admit(candidate)
        if problems:
            self.archive.set_status(candidate.candidate_id, CandidateStatus.QUARANTINED, "; ".join(problems))
            log.decision = "QUARANTINED"
            log.reason = "; ".join(problems)
            return self._finish(log, started)

        if dry_run:
            log.decision = "DRY_RUN"
            log.reason = (
                f"candidate {candidate.candidate_id} built and admitted; evaluation skipped (--dry-run)"
            )
            return self._finish(log, started)

        # --- the cascade -------------------------------------------------
        policy = RuntimePolicy.from_overlay(candidate.overlay)
        gate0 = self._gate0(candidate)
        commands = gate0["commands"]
        gate0_ok = bool(gate0["passed"])
        # Keep the whole capture, not just the exit status. A gate-0 failure
        # ends the cycle, and "the test suite failed" without the output is a
        # dead end for whoever reads the log next.
        self._write(self.cfg.resolved_store_dir() / f"gate0_{candidate.candidate_id}.json", gate0)

        replay_result = self._gate1(policy)
        self._write(self.cfg.resolved_store_dir() / f"replay_{candidate.candidate_id}.json", replay_result)

        if not gate0_ok:
            reason = gate0.get("materialization_error") or gate0.get("origin_error")
            if reason and gate0.get("invalid_candidate"):
                # The patch does not apply to the parent. Deterministic, and the
                # candidate's own fault, so it is a rejection.
                self.archive.set_status(
                    candidate.candidate_id, CandidateStatus.REJECTED, f"invalid candidate: {reason}"
                )
                log.decision = "INVALID_CANDIDATE"
                log.reason = f"gate 0: the candidate could not be materialised — {reason}"
                log.records = []
                return self._finish(log, started, commands=commands)
            if reason:
                # The candidate never ran. That is an execution failure, not
                # evidence about the candidate, and must not be recorded as one.
                self.archive.set_status(
                    candidate.candidate_id, CandidateStatus.DRAFT, f"gate 0 could not run: {reason}"
                )
                log.decision = "GATE0_UNAVAILABLE"
                log.reason = (
                    f"gate 0 never measured the candidate — {reason}. Recorded as an execution "
                    "failure, not as evidence about the candidate."
                )
                log.records = []
                return self._finish(log, started, commands=commands)
            self.archive.set_status(
                candidate.candidate_id, CandidateStatus.REJECTED, "gate 0 invariants failed"
            )
            failures = [c for c in commands if not c["ok"]]
            detail = "; ".join(
                f"{c['label']} exit {c['exit_status']}: "
                + (c["stderr_tail"] or c["stdout_tail"] or c["stdout_head"])[-400:].replace("\n", " ")
                for c in failures
            )
            log.decision = "REJECTED"
            log.reason = f"gate 0: the candidate's own test suite failed in {gate0['cwd']} — {detail}"
            log.records = []
            return self._finish(log, started, commands=commands)

        if skip_benchmark:
            log.decision = "GATE1_ONLY"
            log.reason = "benchmark skipped by request; replay evidence recorded"
            return self._finish(log, started, commands=commands)

        baseline = self.latest_baseline()
        if baseline is None:
            log.decision = "NO_BASELINE"
            log.reason = "no baseline record — run `drugagent self-evolve baseline` first"
            return self._finish(log, started, commands=commands)

        record = self.kernel.evaluate_arm(
            arm=candidate.candidate_id,
            policy=policy,
            candidate_id=candidate.candidate_id,
            commands=commands,
        )
        self.archive.link_record(candidate.candidate_id, record)
        log.records = [record.record_id]

        decision = decide(candidate, baseline, record, self.cfg)
        self._write_decision(candidate, decision)
        log.decision = decision.decision
        log.reason = "; ".join(decision.reasons)
        status = {
            "PROMOTE": CandidateStatus.PROMOTED,
            "NEEDS_HUMAN": CandidateStatus.EVALUATED,
            "ARCHIVE": CandidateStatus.ARCHIVED,
            "REJECT": CandidateStatus.REJECTED,
        }.get(decision.decision, CandidateStatus.EVALUATED)
        self.archive.set_status(candidate.candidate_id, status, log.reason)
        return self._finish(log, started, commands=commands)

    # ------------------------------------------------------------------
    # target mode
    # ------------------------------------------------------------------
    def run_target_campaign(
        self,
        campaign: dict[str, Any],
        *,
        ledger: Any,
        output_root: Path | str,
        stop_rule: Any,
        incumbent: dict[str, Any],
        allowed_actions: list[str] | None = None,
        proposer: Any = None,
        references: dict[str, Any] | None = None,
        capability_notes: list[str] | None = None,
        dispatch_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route into **target evolution mode** — the per-target campaign loop.

        Appendix A §7 lists this module as the existing entry point that routes
        to target mode, and the implementation map records target mode as a new
        route through this controller rather than a parallel one. This method is
        that route, and it is deliberately thin: everything scientific lives in
        :mod:`drug_agent.selfevolve.target.controller`, which is where the
        campaign contracts, the budget ledger and the frozen evaluator already
        are.

        The two loops answer different questions and share no state. The cycle
        above studies *this platform's own* journals and evolves the platform;
        this one runs a bounded hypothesis/experiment/evidence campaign against
        one protein target, through Step 10's action dispatcher. Nothing here
        reads the target's held-out partition: the campaign's action menu cannot
        name a final-test API and the controller refuses a menu that does.

        Returns:
            The campaign record: state, terminal reason, scientific decision,
            the complete candidate pool and the nomination or the reason there
            is none.
        """
        from ..target.controller import CONTROLLER_ACTIONS, ClaudeProposer, TargetCampaignController

        controller = TargetCampaignController(
            campaign,
            proposer=proposer if proposer is not None else ClaudeProposer(backend=self.registry.claude),
            ledger=ledger,
            output_root=output_root,
            allowed_actions=allowed_actions or list(CONTROLLER_ACTIONS),
            stop_rule=stop_rule,
            incumbent=incumbent,
            references=references,
            capability_notes=capability_notes or [],
            dispatch_kwargs=dispatch_kwargs,
        )
        record = controller.run()
        # The target campaign keeps its own durable history; this line is the
        # platform loop's own log noting that a target campaign ran through it.
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "mode": "target_campaign",
                        "campaign_id": record["campaign_id"],
                        "state": record["state"],
                        "terminal_reason": record["terminal_reason"],
                        "scientific_decision": record["scientific_decision"],
                        "history_path": record["history_path"],
                        "at": time.time(),
                    },
                    default=str,
                )
                + "\n"
            )
        return record

    def run(self, *, max_candidates: int | None = None, dry_run: bool | None = None) -> list[CycleLog]:
        """Bounded evolution: several cycles, stopping on budget or exhaustion."""
        limit = max_candidates if max_candidates is not None else self.cfg.max_candidates
        logs: list[CycleLog] = []
        for i in range(max(1, limit)):
            log = self.step(cycle=i + 1, dry_run=dry_run)
            logs.append(log)
            if log.decision in ("NO_BASELINE", "QUARANTINED"):
                break
            claude_ok, _ = self.registry.claude.available()
            ollama_ok, _ = self.registry.ollama.available()
            if not claude_ok and not ollama_ok:
                break
        return logs

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _gate0(self, candidate: Candidate) -> dict[str, Any]:
        """Gate 0, measured in the candidate's checkout rather than the operator's.

        Before this the loop called ``kernel.run_invariants()`` with no
        arguments, which measured the operator's working tree — so a candidate
        that broke the platform passed the gate that exists to catch it, and a
        candidate that fixed something got no credit either. The audit recorded
        it as finding 5; :meth:`EvaluationKernel.gate0` is the repair, and this
        is the loop's one call into it.
        """
        return self.kernel.gate0(candidate)

    def _gate1(self, policy: RuntimePolicy) -> dict[str, Any]:
        """Deterministic replay, for both arms, over the suites this policy can move."""
        suites = applicable_suites(policy)
        root = self.cfg.resolved_journal_dir()
        if not suites:
            return {
                "applicable": False,
                "note": "this candidate changes nothing the replay suites measure",
                "suites": {},
            }
        return {
            "applicable": True,
            "suites_run": suites,
            "baseline": run_replay(DEFAULT_POLICY, root, suites=suites),
            "candidate": run_replay(policy, root, suites=suites),
        }

    def _choose_cluster(self, clusters: list[dict[str, Any]], forced: str | None) -> dict[str, Any]:
        """Route to a cluster: the caller's choice, the local model's, or severity."""
        if forced:
            for c in clusters:
                if c["cluster_id"] == forced:
                    return c
        actionable = [c for c in clusters if BUILDERS_BY_SURFACE.get(c["surface"])]
        pool = actionable or clusters
        result = self.registry.complete(
            "cheap_controller",
            system=_ROUTER_SYSTEM,
            user=json.dumps(
                [
                    {
                        "cluster_id": c["cluster_id"],
                        "label": c["label"],
                        "surface": c["surface"],
                        "severity": c["severity"],
                        "n_episodes": len(c["episodes"]),
                        "summary": c["summary"],
                        "has_operator": bool(BUILDERS_BY_SURFACE.get(c["surface"])),
                    }
                    for c in pool
                ],
                indent=2,
            ),
            schema=_CLUSTER_CHOICE_SCHEMA,
        )
        chosen_id = None
        if isinstance(result.parsed, dict):
            chosen_id = result.parsed.get("cluster_id")
        for c in pool:
            if c["cluster_id"] == chosen_id:
                return c
        return pool[0]  # severity order — the deterministic fallback

    def _finish(
        self, log: CycleLog, started: float, commands: list[dict[str, Any]] | None = None
    ) -> CycleLog:
        log.finished_at = time.time()
        log.wall_s = round(log.finished_at - started, 3)
        log.backend_calls = {
            "claude_cli": self.registry.claude.n_calls,
            "ollama": self.registry.ollama.n_calls,
        }
        log.backend_seconds = {
            "claude_cli": round(self.registry.claude.total_seconds, 3),
            "ollama": round(self.registry.ollama.total_seconds, 3),
        }
        payload = json.loads(log.model_dump_json())
        payload["budget"] = self.registry.budget()
        payload["escalations_declined"] = list(self.registry.escalations_declined)
        payload["model_calls"] = self.engine.calls
        if commands:
            payload["gate0"] = [
                {"label": c["label"], "exit_status": c["exit_status"], "elapsed_s": c["elapsed_s"]}
                for c in commands
            ]
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return log

    def _write(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _write_hypothesis(self, hypothesis: Any) -> None:
        d = self.cfg.resolved_store_dir() / "hypotheses"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{hypothesis.hypothesis_id}.json").write_text(
            hypothesis.model_dump_json(indent=2), encoding="utf-8"
        )

    def _write_decision(self, candidate: Candidate, decision: Any) -> None:
        d = self.cfg.resolved_store_dir() / "decisions"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{candidate.candidate_id}.json").write_text(
            decision.model_dump_json(indent=2), encoding="utf-8"
        )
        (d / f"{candidate.candidate_id}.md").write_text(render_report(candidate, decision), encoding="utf-8")

    def cycles(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


__all__ = ["EvolutionController"]
