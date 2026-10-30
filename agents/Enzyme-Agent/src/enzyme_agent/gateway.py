import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from pydantic import Field
from .contracts import Brief, Contract, Sequence, ToolResult, Unavailable
from .config import atomic_json, check_space, data_root


class GenerationRequest(Contract):
    brief: Brief
    count: int = Field(ge=1, le=32)


class FoldRequest(Contract):
    sequence: Sequence
    seed: int = Field(ge=0, le=2**31-1)
    steps: int = Field(default=50, ge=5, le=200)
    loops: int = Field(default=3, ge=1, le=5)


class PlanRequest(Contract):
    instruction: str = Field(min_length=1, max_length=12000)


class SearchRequest(Contract):
    ec: str = Field(pattern=r"^\d+\.\d+\.\d+\.\d+$")
    count: int = Field(default=8, ge=1, le=16)


class StructureEvaluationRequest(Contract):
    brief: Brief
    sequence: Sequence
    pdb: str = Field(min_length=1, max_length=4_000_000)


class StabilityRequest(Contract):
    brief: Brief
    sequence: Sequence


SCHEMAS = {"generation."+g: GenerationRequest for g in ("esm3", "ligandmpnn", "proteinmpnn", "zymctrl", "simple", "natural", "rfdiffusion3", "rfdiffusion2")}
SCHEMAS.update({"structure.esmfold2": FoldRequest, "planning.qwen": PlanRequest, "retrieval.natural": SearchRequest})
SCHEMAS.update({'evaluation.structure':StructureEvaluationRequest, 'evaluation.thermompnn':StabilityRequest})


def execute(tool, arguments, timeout=1800):
    if tool not in SCHEMAS:
        raise ValueError("Tool not allowed")
    arguments = SCHEMAS[tool].model_validate(arguments).model_dump()
    if tool=='generation.natural':
        from .retrieval import natural_candidates
        try: return ToolResult(tool=tool,status='ok',data=natural_candidates(Brief.model_validate(arguments['brief']),arguments['count']))
        except Exception as exc:return ToolResult(tool=tool,status='error',error=str(exc))
    if tool == "retrieval.natural":
        from .retrieval import natural_search
        try:
            return ToolResult(tool=tool, status="ok", data=natural_search(**arguments))
        except Exception as exc:
            return ToolResult(tool=tool, status="error", error=str(exc))
    check_space(200_000_000)
    scratch = data_root()/"scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="science-", dir=scratch) as directory:
        root = Path(directory)
        atomic_json(root/"input.json", arguments)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MPLCONFIGDIR=str(root/"mpl"))
        with (root/"worker.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "enzyme_agent.science_worker", tool, str(root/"input.json"), str(root/"result.json")],
                                    stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return ToolResult(tool=tool, status="error", error=f"Scientific worker interrupted or timed out ({timeout}s)")
            except BaseException:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
        if not (root/"result.json").is_file():
            return ToolResult(tool=tool, status="error", error=(root/"worker.log").read_text(errors="replace")[-4000:])
        result = ToolResult.model_validate_json((root/"result.json").read_text())
        if result.status != "ok":
            result.data["worker_log_tail"] = (root/"worker.log").read_text(errors="replace")[-4000:]
        return result


class Gateway:
    def __init__(self, store, run_id, limit=24, deadline=None):
        self.store, self.run_id, self.limit, self.deadline = store, run_id, limit, deadline

    def call(self, tool, args):
        if tool not in SCHEMAS:
            raise ValueError("Unknown tool")
        SCHEMAS[tool].model_validate(args)
        timeout = min(1800, max(1, int(self.deadline-time.monotonic()))) if self.deadline else 1800
        if self.deadline and time.monotonic() >= self.deadline:
            raise RuntimeError("Wall time budget exhausted")
        call = self.store.begin_call(self.run_id, tool, args, self.limit)
        start = time.monotonic()
        result = execute(tool, args, timeout)
        self.store.finish_call(self.run_id, call, result.model_dump(), time.monotonic()-start)
        return result
