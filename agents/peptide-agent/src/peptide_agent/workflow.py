"""Durable target -> design -> analyze -> evaluate workflow."""
from __future__ import annotations

import importlib.metadata

from .analysis import rank
from .contracts import Analyze, Brief, Generate, BoltzGenerate, Plan, ToolResult
from .gateway import Gateway
from .store import Store


class Engine:
    def __init__(self, store: Store):
        self.store = store

    def run(self, brief: Brief, skill: str | None = None) -> dict:
        skill = skill or ("analyze-peptide" if brief.source == "library" else "design-peptide")
        Plan(skill=skill, brief=brief)
        return self.resume(self.store.create(brief.model_dump(), skill))

    def resume(self, run_id: str) -> dict:
        run = self.store.get(run_id)
        if run["state"] == "complete":
            return run
        brief = Brief.model_validate(run["brief"])
        gateway = Gateway(self.store, run_id, run["skill"])
        payload = run["payload"]
        try:
            if "sequences" not in payload:
                self.store.update(run_id, "designing", payload)
                if brief.source != "library":
                    schema = Generate if brief.source == "pepmlm" else BoltzGenerate
                    args = schema(**{k: getattr(brief, k) for k in schema.model_fields})
                    result = gateway.call("generation." + brief.source, args.model_dump())
                    if result.status != "ok":
                        return self.stop(run_id, result, payload)
                    payload["generation"] = result.model_dump()
                    payload["sequences"] = Analyze(sequences=result.data["sequences"]).sequences
                else:
                    payload["sequences"] = brief.sequences
                    payload["generation"] = {"source": "user_library", "trust": "PROPOSED"}
                self.store.update(run_id, "generated", payload)
            if "analysis" not in payload:
                self.store.update(run_id, "analyzing", payload)
                result = gateway.call("sequence.analyze", {"sequences": payload["sequences"], "ph": brief.ph,
                                                          "topology": brief.topology})
                if result.status != "ok":
                    return self.stop(run_id, result, payload)
                payload["analysis"] = result.model_dump()
                self.store.update(run_id, "analyzed", payload)
            if brief.assess_toxicity and "toxicity" not in payload:
                self.store.update(run_id, "assessing_toxicity", payload)
                payload["toxicity"] = gateway.call("safety.toxinpred3", {
                    "sequences": payload["sequences"], "topology": brief.topology}).model_dump()
                self.store.update(run_id, "toxicity_assessed", payload)
            if brief.assess_affinity and "affinity" not in payload:
                self.store.update(run_id, "assessing_affinity", payload)
                affinity_results = []
                for complex_ in payload["generation"]["data"].get("complexes", []):
                    result = gateway.call("affinity.prodigy", {"pdb": complex_["pdb"],
                        "receptor_chain": brief.target_chain, "peptide_chain": "B"})
                    affinity_results.append({"sequence": complex_["sequence"], **result.model_dump()})
                payload["affinity"] = affinity_results
                self.store.update(run_id, "affinity_assessed", payload)
            rows = payload["analysis"]["data"]["candidates"]
            # Independent result validation before ranking and publication.
            if [r["sequence"] for r in rows] != payload["sequences"]:
                raise ValueError("Analysis sequence identity/order mismatch")
            if any(r["length"] != len(r["sequence"]) or r["molecular_weight_da"] <= 0 for r in rows):
                raise ValueError("Invalid calculated descriptors")
            payload["portfolio"] = rank(rows, brief.constraints, brief.shortlist)
            payload["unavailable_assessments"] = [name for name in ("toxicity",)
                if name in payload and payload[name]["status"] != "ok"]
            if any(item["status"] != "ok" for item in payload.get("affinity", [])):
                payload["unavailable_assessments"].append("affinity")
            payload["versions"] = {p: importlib.metadata.version(p) for p in
                                   ("peptide-agent", "biopython", "numpy", "torch", "transformers", "mcp")}
            payload.pop("error", None)
            payload["report_sha256"] = self.store.artifact(payload)
            self.store.update(run_id, "complete", payload)
        except Exception as exc:
            payload["error"] = f"{type(exc).__name__}: {exc}"
            self.store.update(run_id, "failed", payload)
        return self.store.get(run_id)

    def stop(self, run_id, result: ToolResult, payload):
        payload["error"] = result.model_dump()
        self.store.update(run_id, "blocked" if result.status == "unavailable" else "failed", payload)
        return self.store.get(run_id)
