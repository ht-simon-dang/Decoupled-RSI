"""Allowlisted tools, isolated processes, deadlines and durable audit records."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import atomic_json, data_root, digest
from .contracts import Binder, Brief, BudgetExceeded, ToolResult, Unavailable
from .registry import MODELS, planner_runtime, runtime

TOOLS = {"generation." + name for name, info in MODELS.items() if info[0] == "generator"}
TOOLS |= {
    "planning.qwen",
    "numbering.anarcii",
    "fold.immunebuilder",
    "complex.boltz2",
    "complex.chai1",
    "profile.tap",
    "profile.tnp",
}


def validate_request(tool, args):
    if tool not in TOOLS:
        raise ValueError("Tool is not allowlisted")
    if tool.startswith("generation."):
        Brief.model_validate(args["brief"])
        if type(args.get("count")) is not int or not 1 <= args["count"] <= 32:
            raise ValueError("Invalid generation count")
    elif tool == "planning.qwen":
        if not isinstance(args.get("instruction"), str) or not 1 <= len(args["instruction"]) <= 12000:
            raise ValueError("Instruction must be 1..12000 characters")
    else:
        Binder.model_validate(args["binder"])


def execute(tool, arguments, timeout, *, config=None, cancelled=lambda: False):
    validate_request(tool, arguments)
    name = tool.split(".")[1]
    try:
        cfg = config if config is not None else (planner_runtime() if name == "qwen" else runtime(name))
    except Unavailable as exc:
        return ToolResult(tool=tool, status="unavailable", error=str(exc))
    scratch = data_root() / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="science-", dir=scratch) as work:
        directory = Path(work)
        source, destination = directory / "request.json", directory / "result.json"
        atomic_json(
            source, {"protocol": "antibody-tools-v1", "tool": tool, "arguments": arguments, "runtime": cfg}
        )
        argv = (
            [*cfg["command"], str(source), str(destination)]
            if cfg.get("command")
            else [
                cfg.get("python", sys.executable),
                "-m",
                "antibody_agent.science_worker",
                tool,
                str(source),
                str(destination),
            ]
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            XDG_CACHE_HOME=str(data_root() / "cache"),
            HF_HOME=str(data_root() / "cache/hf"),
            TORCH_HOME=str(data_root() / "cache/torch"),
            MPLCONFIGDIR=str(directory / "mpl"),
        )
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        started = time.monotonic()

        def log_tail():
            with (directory / "worker.log").open("rb") as handle:
                handle.seek(max(0, (directory / "worker.log").stat().st_size - 8192))
                return handle.read().decode(errors="replace")

        with (directory / "worker.log").open("w") as log:
            try:
                proc = subprocess.Popen(
                    argv, cwd=directory, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
            except OSError as exc:
                return ToolResult(tool=tool, status="unavailable", error=str(exc))
            try:
                while proc.poll() is None:
                    if cancelled():
                        raise InterruptedError("Run cancelled")
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError("Scientific tool exceeded remaining wall-time budget")
                    time.sleep(0.1)
                if proc.returncode != 0:
                    return ToolResult(
                        tool=tool,
                        status="error",
                        error=f"Worker exited with code {proc.returncode}",
                        data={"private_worker_log": log_tail()},
                    )
                if not destination.is_file() or destination.stat().st_size > 40_000_000:
                    return ToolResult(tool=tool, status="error", error="Missing or oversized worker response")
                result = ToolResult.model_validate_json(destination.read_text())
                if result.tool != tool:
                    raise ValueError("Worker response names a different tool")
                if result.status != "ok":
                    result.data["private_worker_log"] = log_tail()
                # Source/version are owned by the registry, never by model-generated text.
                result.provenance = {
                    "model": name,
                    "version": cfg.get("version", cfg.get("revision")),
                    "weights_version": cfg.get("weights_version", cfg.get("revision")),
                    "request_hash": digest(arguments),
                    "seed": arguments.get("seed", arguments.get("brief", {}).get("seed")),
                }
                if result.status == "ok" and tool.startswith(("fold.", "complex.")):
                    from .contracts import AntigenState
                    from .structure import audit

                    pdb = result.data.get("pdb")
                    if not isinstance(pdb, str) or not pdb:
                        raise ValueError("Structural tool must return actual PDB coordinates")
                    binder = Binder.model_validate(arguments["binder"])
                    state = (
                        AntigenState.model_validate(arguments["state"])
                        if tool.startswith("complex.")
                        else None
                    )
                    result.data.update(audit(pdb, binder, state))
                result.provenance["output_sha256"] = digest(result.data)
                return result
            except TimeoutError as exc:
                return ToolResult(tool=tool, status="timeout", error=str(exc))
            finally:
                # Terminate the whole tool process group, including descendants.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


class Gateway:
    def __init__(self, store, run_id, limit, deadline):
        self.store, self.run_id, self.limit, self.deadline = store, run_id, limit, deadline

    def call(self, tool, args):
        validate_request(tool, args)
        name = tool.split(".")[1]
        try:
            cfg = planner_runtime() if name == "qwen" else runtime(name)
        except Unavailable:
            cfg = None
        request_hash = digest({"arguments": args, "runtime": cfg})
        cached = self.store.cached_call(self.run_id, tool, request_hash)
        if cached:
            source = cached.get("provenance", {})
            if source.get("request_hash") != digest(args) or source.get("output_sha256") != digest(
                cached.get("data", {})
            ):
                raise ValueError("Cached tool record failed integrity validation")
            return ToolResult.model_validate(cached)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Wall-time budget exhausted")
        call_id = self.store.begin_call(self.run_id, tool, request_hash, self.limit)
        self.store.event(self.run_id, "tool_started", {"tool": tool})
        started = time.monotonic()
        try:
            result = execute(
                tool,
                args,
                min(1800, remaining),
                config=cfg,
                cancelled=lambda: self.store.get(self.run_id)["state"] == "cancelled",
            )
        except InterruptedError:
            result = ToolResult(tool=tool, status="error", error="Cancelled")
            raise
        except Exception as exc:
            result = ToolResult(tool=tool, status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if "result" in locals():
                self.store.finish_call(call_id, result.model_dump(), time.monotonic() - started)
        self.store.event(self.run_id, "tool_finished", {"tool": tool, "status": result.status})
        return result
