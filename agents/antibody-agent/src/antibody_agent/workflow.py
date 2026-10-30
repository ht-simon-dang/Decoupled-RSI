"""Plan -> constrained generation -> independent evaluation -> human review."""

from __future__ import annotations

import fcntl
import time

from .config import digest
from .contracts import Binder, Brief, BudgetExceeded, Candidate, Unavailable
from .evaluation import (
    context_id,
    evaluate,
    evaluation_binder,
    evaluator_hash,
    make_candidate,
    select,
    validate_edits,
)
from .gateway import Gateway


def research_card(brief):
    gaps = []
    if brief.direction == "unresolved":
        gaps.append("明确阻断、激动、清除、桥联或递送的作用方向")
    if not brief.states:
        gaps.append("确认靶点蛋白、物种、亚型、天然状态与可达性")
    if not brief.epitopes:
        gaps.append("提供功能表位及其与疾病机制的联系")
    if brief.access_gate.status == "unknown":
        gaps.append("补充抗体能到达目标组织和分子状态的证据")
    if not brief.reference and brief.route not in {"research", "library"}:
        gaps.append("提供配对 VH/VL 或 VHH 框架、编号与编辑范围")
    for s in brief.states:
        if not s.sequence:
            gaps.append(f"{s.id}: 缺少明确抗原序列")
        if not s.full_context:
            gaps.append(f"{s.id}: 需要完整天然抗原背景复核")
    return {
        "disease": brief.disease,
        "patient_subgroup": brief.patient_subgroup,
        "mechanism": brief.mechanism,
        "direction": brief.direction,
        "product": brief.product.model_dump(),
        "targets": [
            {"id": s.id, "entity": s.entity, "state": s.state, "role": s.role, "sources": s.sources}
            for s in brief.states
        ],
        "evidence": [e.model_dump() for e in brief.evidence],
        "gaps": gaps,
        "decision": "Computational research hypothesis; resolve evidence gaps before experimental claims",
    }


class Engine:
    def __init__(self, store, gateway_factory=Gateway):
        self.store, self.gateway_factory = store, gateway_factory

    def run(self, brief=None, *, instruction=None):
        if (brief is None) == (instruction is None):
            raise ValueError("Provide exactly one structured brief or instruction")
        value = Brief.model_validate(brief).model_dump() if brief is not None else {}
        run_id = self.store.create(value, {"instruction": instruction} if instruction is not None else {})
        return self.resume(run_id)

    def resume(self, run_id):
        self.store.get(run_id)
        locks = self.store.root / "locks"
        locks.mkdir(exist_ok=True)
        with (locks / (run_id + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("This run already has an executor") from exc
            return self._resume(run_id)

    def request_resume(self, run_id):
        """Recover a stopped executor without resetting spent time or tool calls."""
        from .store import TERMINAL

        locks = self.store.root / "locks"
        locks.mkdir(exist_ok=True)
        self.store.get(run_id)
        with (locks / (run_id + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Executor is still active or stopping") from exc
            run = self.store.get(run_id)
            if run["state"] in {"complete", "budget_exhausted"}:
                raise ValueError("Completed/exhausted runs require a new evaluation")
            payload = run["payload"]
            if run["state"] not in TERMINAL and run["state"] != "queued":
                payload["elapsed_seconds"] = payload.get("elapsed_seconds", 0) + max(
                    0, time.time() - run["updated"]
                )
            self.store.update(run_id, "queued", payload, force=True)

    def _resume(self, run_id):
        run = self.store.get(run_id)
        if run["state"] == "complete":
            return run
        if run["state"] == "cancelled":
            raise ValueError("Cancelled runs require an explicit resume action")
        payload = run["payload"]
        payload.pop("diagnosis", None)
        started = time.monotonic()
        prior = payload.get("elapsed_seconds", 0)

        def save(state):
            if self.store.get(run_id)["state"] == "cancelled":
                raise InterruptedError("Cancelled")
            payload["elapsed_seconds"] = prior + time.monotonic() - started
            payload["cost"] = self.store.cost(run_id)
            self.store.update(run_id, state, payload)
            self.store.event(run_id, state, {})

        def numbered(binder, gateway, scheme):
            if all(c.numbering and c.scheme == scheme for c in binder.chains):
                return binder
            result = gateway.call("numbering.anarcii", {"binder": binder.model_dump(), "scheme": scheme})
            if result.status != "ok":
                raise Unavailable("Numbering unavailable or failed; generation/evaluation stopped")
            mapped = Binder.model_validate(result.data["binder"])
            from .evaluation import identities

            if identities(mapped) != identities(binder) or any(
                not c.numbering or c.scheme != scheme for c in mapped.chains
            ):
                raise ValueError("Numbering adapter changed identity or returned an incomplete map")
            return mapped

        try:
            if not run["brief"]:
                save("planning")
                gateway = self.gateway_factory(self.store, run_id, 4, started + 240 - prior)
                result = gateway.call("planning.qwen", {"instruction": payload["instruction"]})
                if result.status != "ok":
                    raise Unavailable(
                        "Qwen3.5-0.8B planner unavailable or failed; use a structured brief or configure weights"
                    )
                brief = Brief.model_validate(result.data["brief"])
                payload["planning"] = result.model_dump()
                self.store.set_brief(run_id, brief.model_dump())
            else:
                brief = Brief.model_validate(run["brief"])
            payload["research"] = research_card(brief)
            previous = payload.get("evaluator_hash")
            if previous and previous != evaluator_hash():
                raise ValueError("Evaluator changed: reevaluate in a new run with frozen rules")
            payload["evaluator_hash"] = evaluator_hash()
            if payload.get("brief_hash") and payload["brief_hash"] != digest(brief.model_dump()):
                raise ValueError("Brief changed after freezing; create a new run")
            deadline = started + brief.budget.wall_seconds - prior
            gateway = self.gateway_factory(self.store, run_id, brief.budget.tool_calls, deadline)
            save("evidence_review")
            if brief.route == "research":
                payload["diagnosis"] = "研究档案已建立；缺失证据保留为待决事项。尚未生成候选。"
                save("complete")
                return self.store.get(run_id)
            if any(g.status == "contradicted" for g in (brief.direction_gate, brief.access_gate)):
                payload["diagnosis"] = "机制或可达性前提受到反证；请调整表位、格式或靶点。"
                save("complete")
                return self.store.get(run_id)
            if brief.constraints.allow_length_change:
                raise Unavailable(
                    "Length-changing design requires a validated insertion/deletion mapping adapter"
                )
            if brief.reference:
                brief.reference = numbered(brief.reference, gateway, brief.constraints.scheme)
            # Revalidate after numbering to catch wrong labels before calling a generator.
            brief = Brief.model_validate(brief.model_dump())
            self.store.set_brief(run_id, brief.model_dump())
            payload["brief_hash"] = digest(brief.model_dump())
            parent = (
                make_candidate(brief.reference, "reference", {"source": brief.reference.source})
                if brief.reference
                else None
            )
            generation = payload.setdefault("generation", {})
            save("generating")
            if brief.route == "library":
                if brief.generators != ["library"]:
                    raise ValueError("Library analysis must explicitly use generators=['library']")
                supplied = brief.library or ([brief.reference] if brief.reference else [])
                inputs = [(b, "library", {"source": b.source}) for b in supplied]
            else:
                if not parent or not brief.constraints.editable:
                    raise Unavailable(
                        "Generation requires a numbered framework/reference and explicit editable labels"
                    )
                sources = [g for g in brief.generators if g != "library"]
                if not sources or len(sources) > brief.budget.candidates:
                    raise ValueError("Generation budget must cover each selected generator")
                inputs = []
                for i, source in enumerate(sources):
                    count = brief.budget.candidates // len(sources) + int(
                        i < brief.budget.candidates % len(sources)
                    )
                    result = gateway.call(
                        "generation." + source, {"brief": brief.model_dump(), "count": count}
                    )
                    generation[source] = result.model_dump()
                    if result.status == "ok":
                        binders = result.data.get("binders")
                        if not isinstance(binders, list) or not 1 <= len(binders) <= count:
                            raise ValueError("Generator returned missing/over-budget candidates")
                        for item in binders:
                            binder = Binder.model_validate(item)
                            inputs.append((binder, source, result.provenance))
                    save("generating")
            if not inputs:
                raise Unavailable(
                    "No candidates available from the selected routes; inspect configured model status"
                )
            candidates, seen = [], set()
            payload["candidate_errors"] = []
            for binder, origin, provenance in inputs:
                try:
                    binder = numbered(binder, gateway, brief.constraints.scheme)
                except Unavailable:
                    payload["candidate_errors"].append({"source": origin, "reason": "numbering unavailable"})
                candidate = make_candidate(
                    binder, origin, provenance, parent if origin != "library" else None
                )
                previous_candidate = payload.get("lineage", {}).get(candidate.id)
                if previous_candidate:
                    previous_candidate = Candidate.model_validate(previous_candidate)
                    candidate.origin = previous_candidate.origin
                    candidate.provenance = previous_candidate.provenance
                    candidate.parent_id = previous_candidate.parent_id
                    candidate.mutations = previous_candidate.mutations
                if candidate.id not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.id)
            payload["candidates"] = [c.model_dump() for c in candidates]
            payload["generated_count"] = len(inputs)
            payload["duplicate_count"] = len(inputs) - len(candidates)
            calls = payload.setdefault("evaluation_calls", {})
            assessments = {}
            infrastructure_missing = False
            for candidate in candidates:
                save("evaluating")
                records = calls.setdefault(candidate.id, {})
                if not validate_edits(candidate, brief):
                    # Design structures are never passed as independent prediction templates.
                    binding = evaluation_binder(candidate)
                    records["fold"] = gateway.call(
                        "fold.immunebuilder", {"binder": binding, "seed": brief.seed}
                    ).model_dump()
                    if records["fold"]["status"] == "ok":
                        profile = "tnp" if candidate.binder.chains[0].role == "VHH" else "tap"
                        records["profile"] = gateway.call(
                            "profile." + profile,
                            {"binder": binding, "pdb": records["fold"]["data"]["pdb"], "seed": brief.seed},
                        ).model_dump()
                    for state in brief.states:
                        model = "chai1" if candidate.origin == "boltzgen" else "boltz2"
                        records["complex:" + state.id] = gateway.call(
                            "complex." + model,
                            {
                                "binder": binding,
                                "state": state.model_dump(),
                                "context_id": context_id(candidate, state),
                                "seed": brief.seed,
                            },
                        ).model_dump()
                    infrastructure_missing |= any(r["status"] != "ok" for r in records.values())
                else:
                    infrastructure_missing |= any(not c.numbering for c in candidate.binder.chains)
                assessments[candidate.id] = evaluate(candidate, brief, records)
                payload["assessments"] = assessments
                save("evaluating")
                if time.monotonic() >= deadline:
                    raise BudgetExceeded("Wall-time budget exhausted")
            payload["portfolio"] = select(candidates, assessments, brief)
            payload["evidence_artifact"] = self.store.artifact(
                {
                    "candidates": payload["candidates"],
                    "calls": calls,
                    "assessments": assessments,
                    "brief_hash": payload["brief_hash"],
                }
            )
            payload["diagnosis"] = (
                "独立计算评估完成，合格候选等待 human 审阅。"
                if payload["portfolio"]
                else "没有候选通过当前门槛；根据失败原因调整输入、表位、格式或模型配置。"
            )
            save("blocked" if infrastructure_missing and not payload["portfolio"] else "complete")
        except BudgetExceeded as exc:
            payload["diagnosis"] = str(exc)
            save("budget_exhausted")
        except Unavailable as exc:
            payload["diagnosis"] = str(exc)
            save("blocked")
        except InterruptedError:
            payload["elapsed_seconds"] = prior + time.monotonic() - started
            payload["cost"] = self.store.cost(run_id)
            self.store.update(run_id, "cancelled", payload, force=True)
        except Exception as exc:
            payload["private_error"] = f"{type(exc).__name__}: {exc}"
            payload["diagnosis"] = "任务校验或工具执行失败；原始错误仅保存在本地审计记录中。"
            save("failed")
        return self.store.get(run_id)

    def prepare_reevaluation(self, run_id):
        old = self.store.get(run_id)
        from .store import TERMINAL

        if old["state"] not in TERMINAL:
            raise ValueError("Stop the current run before reevaluation")
        brief = Brief.model_validate(old["brief"])
        binders = [Candidate.model_validate(c).binder for c in old["payload"].get("candidates", [])]
        if not binders:
            raise ValueError("No stored candidates to reevaluate")
        values = brief.model_dump()
        values.update(
            route="library",
            generators=["library"],
            library=[b.model_dump() for b in binders],
            parent_run_id=run_id,
        )
        # A new run has a newly declared budget; the old journal stays immutable.
        return self.store.create(
            Brief.model_validate(values).model_dump(),
            {"lineage": {c["id"]: c for c in old["payload"]["candidates"]}},
        )

    def reevaluate(self, run_id):
        return self.resume(self.prepare_reevaluation(run_id))
