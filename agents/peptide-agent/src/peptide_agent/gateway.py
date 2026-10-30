"""One policy boundary shared by orchestrator and MCP. No arbitrary commands or URLs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.resources import files

from .analysis import analyze, structure
from .contracts import Analyze, Generate, Structure, BoltzGenerate, ToolResult, Unavailable
from .calibration import AffinityRequest

SCHEMAS = {"generation.pepmlm": Generate, "generation.boltzgen": BoltzGenerate,
           "sequence.analyze": Analyze, "structure.contacts": Structure,
           "safety.toxinpred3": Analyze, "affinity.prodigy": Structure, "affinity.calibrated": AffinityRequest}
POLICY = json.loads(files("peptide_agent").joinpath("policy.json").read_text())


class Gateway:
    def __init__(self, store, run_id, skill):
        if skill not in POLICY["skills"]:
            raise PermissionError("Unknown skill")
        self.store, self.run_id, self.skill = store, run_id, skill
        self.calls = sum(e["kind"] == "tool.request" for e in store.events(run_id))

    def call(self, tool: str, arguments: dict) -> ToolResult:
        if tool not in POLICY["skills"][self.skill] or self.calls >= POLICY["max_calls"]:
            self.store.event(self.run_id, "tool.denied", {"tool": tool, "skill": self.skill})
            raise PermissionError("Tool not allowed for skill, or call budget exhausted")
        try:
            request = SCHEMAS[tool].model_validate(arguments)
        except ValueError as exc:
            self.store.event(self.run_id, "tool.invalid", {"tool": tool, "error": str(exc)[:2000]})
            raise
        self.calls += 1
        self.store.event(self.run_id, "tool.request", {"tool": tool, "skill": self.skill,
                                                     "input_sha256": self.store.artifact(request.model_dump())})
        try:
            if tool == "generation.pepmlm":
                env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
                proc = subprocess.run([sys.executable, "-m", "peptide_agent.worker"],
                                      input=request.model_dump_json(), text=True, capture_output=True,
                                      timeout=POLICY["generation_timeout_seconds"], env=env, check=True)
                result = ToolResult.model_validate_json(proc.stdout)
            elif tool == "sequence.analyze":
                result = analyze(request)
            elif tool == "generation.boltzgen":
                from .structural import generate_boltz
                result = generate_boltz(request)
            elif tool == "safety.toxinpred3":
                from .toxicity import toxicity
                result = toxicity(request)
            elif tool == "affinity.prodigy":
                from .structural import affinity
                result = affinity(request)
            elif tool == "affinity.calibrated":
                from .calibration import predict
                result = predict(request)
            else:
                result = structure(request)
        except Unavailable as exc:
            result = ToolResult(tool=tool, status="unavailable", trust="UNAVAILABLE", warnings=[str(exc)])
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            result = ToolResult(tool=tool, status="error", trust="UNAVAILABLE", warnings=[str(exc)])
        if result.tool != tool:
            raise ValueError("Tool response identity mismatch")
        digest = self.store.artifact(result.model_dump())
        self.store.event(self.run_id, "tool.result", {"tool": tool, "status": result.status, "sha256": digest})
        return result
