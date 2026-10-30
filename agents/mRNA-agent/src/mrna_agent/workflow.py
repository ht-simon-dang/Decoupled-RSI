"""Durable EMA4D orchestration: plan -> design -> evaluate -> portfolio -> feedback."""
from __future__ import annotations
import time
from .contracts import Brief, Construct, identity
from .generation import make_baseline, generate_round
from .gateway import Gateway, BudgetExhausted
from .evaluation import check_construct, predict_batch, evaluate, portfolio
from .registry import QWEN, manifest


def construct_from_card(card):
    return Construct(**{k: card[k] for k in Construct.model_fields if k in card})


class Engine:
    def __init__(self, store):
        self.store = store

    def run(self, brief=None, *, instruction=None, planner=True):
        if brief and "workflow_version" not in brief.model_fields_set:
            active=self.store.db.execute("SELECT value FROM settings WHERE key='active_workflow'").fetchone()
            if active:brief.workflow_version=active[0]
        run_id = self.store.create(brief.model_dump() if brief else {}, instruction)
        return self.resume(run_id, planner=planner)

    def resume(self, run_id, *, planner=True):
        with self.store.exclusive(run_id):
            run = self.store.get(run_id)
            if run["state"] == "complete": return run
            if run["state"] in {"paused", "cancelled"}:
                raise ValueError("Paused/cancelled run must be explicitly queued before resume")
            payload = run["payload"]
            started = time.monotonic()
            try:
                if not run["brief"]:
                    from .llm import from_instruction
                    self.store.update(run_id, "planning", payload)
                    brief = from_instruction(payload["instruction"], self.store, run_id)
                    self.store.set_brief(run_id, brief.model_dump())
                else: brief = Brief.model_validate(run["brief"])
                if not brief.protein:
                    payload.update(stop_reason="needs_payload", report="Application recorded. A fixed protein or explicitly partitioned mRNA is required before sequence design.")
                    self.store.update(run_id, "blocked", payload)
                    return self.store.get(run_id)
                if "plan" not in payload:
                    self.store.update(run_id, "planning", payload)
                    if planner:
                        from .llm import plan
                        payload["plan"] = {**plan(brief, self.store, run_id), "model": QWEN}
                    else:
                        payload["plan"] = {"focus": "balanced", "priority": "exploration", "rationale": "Explicit fixed-workflow ablation; no LLM planning", "model": None}
                    self.store.update(run_id, "planned", payload)
                # Planner can allocate attention, never relax the locked brief.
                payload["target_sha256"] = identity(brief.model_dump())
                payload.setdefault("baseline", make_baseline(brief).card())
                baseline = construct_from_card(payload["baseline"])
                self.store.add_construct(baseline)
                feedback_parent = baseline
                if brief.mode == "iterate" and brief.parent_run:
                    from .feedback import analyze_feedback
                    parent_run = self.store.get(brief.parent_run)
                    if parent_run["brief"].get("context") == brief.context.model_dump():
                        if "parent_feedback" not in payload:
                            payload["parent_feedback"] = analyze_feedback(self.store,brief.parent_run)
                        winner = next((x for x in payload["parent_feedback"]["next_batch"] if x["role"] == "measured_followup"),None)
                        if winner:
                            proposal = construct_from_card(self.store.construct(winner["id"]))
                            from .contracts import translate
                            if translate(proposal.regions.cds) != brief.protein:
                                raise ValueError("Feedback parent encodes a different protein version")
                            feedback_parent = proposal
                            self.store.event(run_id,"feedback.parent_selected",{"construct_id":proposal.id,"reason":winner["reason"],"source_run":brief.parent_run})
                gateway = Gateway(self.store, run_id, brief, payload.get("elapsed_seconds",0) + time.monotonic()-started)
                payload.setdefault("rounds", [])
                payload.setdefault("failures", [])
                existing = {c["id"]: construct_from_card(c) for c in payload.get("candidates", [])}
                evaluations = {e["construct_id"]: e for e in payload.get("evaluations", [])}
                for round_index in range(brief.budget.rounds):
                    if round_index < len(payload["rounds"]): continue
                    checkpoint = "round_" + str(round_index)
                    if checkpoint not in payload:
                        self.store.update(run_id, "generating", payload)
                        round_brief = brief
                        generation_parent = feedback_parent
                        if round_index and payload.get("portfolio", {}).get("selected"):
                            # Task-local feedback uses prior valid construct as parent without changing locks.
                            pick = next((s["id"] for s in payload["portfolio"]["selected"] if s["id"] != baseline.id), baseline.id)
                            parent = existing.get(pick, baseline)
                            generation_parent = parent
                            round_brief = brief.model_copy(update={"reference": brief.reference or parent.regions})
                            self.store.event(run_id, "design.feedback", {"round": round_index, "parent_id": parent.id, "reason": "Explore around an admissible prior candidate; keep locked objectives/regions"})
                        generated, failures = generate_round(round_brief, generation_parent, gateway, round_index, payload["plan"])
                        if round_index == 0 and all(c.id != baseline.id for c in generated):
                            generated = [baseline] + generated[:max(0,brief.budget.candidates-1)]
                        payload[checkpoint] = [c.card() for c in generated]
                        payload["failures"].extend(failures)
                        self.store.update(run_id, "generated", payload)
                    generated = [construct_from_card(c) for c in payload[checkpoint]]
                    fresh = []
                    for c in generated:
                        if c.id not in existing:
                            existing[c.id] = c
                            self.store.add_construct(c)
                        if c.id not in evaluations: fresh.append(c)
                    payload["candidates"] = [c.card() for c in existing.values()]
                    self.store.update(run_id, "validating", payload)
                    good = [c for c in fresh if all(x["status"] == "pass" for x in check_construct(c, brief))]
                    prediction_key = checkpoint + "_predictions"
                    if prediction_key not in payload:
                        self.store.update(run_id, "predicting", payload)
                        payload[prediction_key] = predict_batch(good, brief, gateway)
                        self.store.update(run_id, "predicted", payload)
                    self.store.update(run_id, "evaluating", payload)
                    for c in fresh:
                        if gateway.remaining() <= 0:
                            raise BudgetExhausted("Compute-time budget exhausted before full-construct evaluation")
                        e = evaluate(c, brief, payload[prediction_key].get(c.id, []))
                        e["changed_regions"] = [r for r in ("utr5","cds","utr3") if getattr(c.regions,r) != getattr(baseline.regions,r)]
                        e["configuration_changed"] = c.molecule != baseline.molecule
                        base_e = evaluations.get(baseline.id)
                        e["baseline_deltas"] = []
                        if base_e:
                            for m in e["metrics"]:
                                other = next((v for v in base_e["metrics"] if v["metric"] == m["metric"] and v["status"] == m["status"] == "ok" and all(v[k] == m[k] for k in ("source","version","unit","scope"))),None)
                                if other:
                                    e["baseline_deltas"].append({"metric":m["metric"],"delta":m["value"]-other["value"],"unit":m["unit"],"baseline_id":baseline.id,"interpretation":"Difference on the same model/feature scale; not experimental benefit"})
                        e["evaluation_id"] = self.store.add_evaluation(run_id, c.id, brief.context.model_dump(), e)
                        evaluations[c.id] = e
                        payload["evaluations"] = list(evaluations.values())
                        self.store.update(run_id, "evaluating", payload)
                    payload["portfolio"] = portfolio(list(existing.values()), list(evaluations.values()), brief)
                    payload["rounds"].append({"round": round_index, "candidates": [c.id for c in generated], "frontier": payload["portfolio"]["frontier"], "tool_calls": gateway.calls})
                    self.store.event(run_id, "portfolio.selected", payload["portfolio"])
                    self.store.update(run_id, "selecting", payload)
                    if brief.mode == "diagnose": break
                if brief.parent_run or brief.mode in {"iterate", "transfer"}:
                    from .feedback import analyze_feedback
                    payload["feedback"] = analyze_feedback(self.store, run_id, brief, payload)
                selected = payload["portfolio"]["selected"]
                tool_failures = [f for f in payload["failures"] if f.get("tool")]
                if len(existing)==1 and brief.mode!="diagnose" and tool_failures and all(f.get("status") in {"out_of_scope","unavailable","error"} for f in tool_failures):
                    reason = "missing_applicable_models"
                elif not selected:
                    reason = "all_constraints_failed" if payload["portfolio"]["rejected"] else "no_high_potential_candidates"
                elif len(existing) < min(brief.budget.candidates, 2) and brief.mode != "diagnose":
                    tool_failures = [f for f in payload["failures"] if f.get("tool")]
                    reason = "missing_applicable_models" if tool_failures and all(f.get("status") in {"out_of_scope","unavailable","error"} for f in tool_failures) else "insufficient_candidates"
                elif brief.mode != "diagnose" and len(selected) < min(brief.budget.shortlist,brief.budget.candidates,brief.budget.synthesis_slots if brief.budget.synthesis_slots is not None else brief.budget.shortlist):
                    reason = "insufficient_candidates"
                elif any(o.stage == "experimental" for o in brief.objectives):
                    reason = "needs_experimental_data"
                else: reason = "recommendations_ready"
                payload["stop_reason"] = reason
                payload.update(versions={"mrna_agent": "0.2.0", "qwen": QWEN, "assets": manifest()},
                               elapsed_seconds=payload.get("elapsed_seconds", 0) + time.monotonic()-started,
                               tool_calls=gateway.calls, report=self.report(brief, payload))
                payload["scientific_tool_seconds"] = sum(e["payload"].get("seconds",0) for e in self.store.events(run_id) if e["kind"] == "tool.finished")
                payload.pop("error", None)
                payload["report_sha256"] = self.store.artifact(payload)
                self.store.update(run_id, "complete", payload)
            except InterruptedError:
                return self.store.get(run_id)
            except BudgetExhausted as exc:
                payload.update(stop_reason="budget_exhausted", error=str(exc))
                if payload.get("evaluations"):
                    payload["portfolio"] = portfolio([construct_from_card(c) for c in payload["candidates"]],payload["evaluations"],brief)
                payload["elapsed_seconds"] = payload.get("elapsed_seconds",0) + time.monotonic()-started
                self.store.update(run_id, "budget_exhausted", payload)
            except Exception as exc:
                payload["error"] = f"{type(exc).__name__}: {exc}"
                payload["elapsed_seconds"] = payload.get("elapsed_seconds",0) + time.monotonic()-started
                self.store.event(run_id, "workflow.error", {"error": payload["error"]})
                self.store.update(run_id, "failed", payload)
            return self.store.get(run_id)

    @staticmethod
    def report(brief, payload):
        return {"recommendation": f"{len(payload['portfolio']['selected'])} high-potential constructs passed evaluation and delivery filter",
                "target": {"protein_version": brief.protein_version, "protein_sha256": identity(brief.protein), "protein_length": len(brief.protein)},
                "context": brief.context.model_dump(), "objectives": [o.model_dump() for o in brief.objectives],
                "baseline": payload["baseline"]["id"], "stop_reason": payload["stop_reason"],
                "unavailable_generation_models": sorted({f["tool"] for f in payload.get("failures",[]) if f.get("tool") and f.get("status")=="unavailable"}),
                "assumptions": brief.assumptions + ["All sequence-only recommendations are computational hypotheses", "Unspecified delivery/dose/chemistry are not inferred"],
                "comparison": "Full assembled constructs evaluated with the same tools and input conventions; reward reuse is labeled",
                "limitations": ["No experimental efficacy or immune-safety claim", "Sequence-intrinsic features do not measure manufacturing dsRNA impurity", "MRL, TE, RNA persistence, chemical stability and protein output are separate endpoints"],
                "next_step": "Measure matched candidates and baseline; import context, batch, controls, replicates and quality alongside raw values"}

    def reevaluate(self, run_id, *, context=None):
        original = self.store.get(run_id)
        brief = Brief.model_validate(original["brief"])
        if context is not None:
            brief.context = context
        brief.budget.rounds = 1
        new_id = self.store.create(brief.model_dump())
        candidates = [construct_from_card(c) for c in original["payload"].get("candidates", [])]
        if not candidates:
            raise ValueError("No completed generation checkpoint to reevaluate")
        payload = {"plan": original["payload"].get("plan", {}), "baseline": original["payload"]["baseline"], "round_0": [c.card() for c in candidates], "replay_of": run_id}
        self.store.update(new_id, "queued", payload)
        return self.resume(new_id, planner=False)
