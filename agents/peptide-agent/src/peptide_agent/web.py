"""Local research console with a serialized, cancellable process queue."""
import csv
import io
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import Field, model_validator

from .capabilities import capabilities
from .contracts import Brief, Contract
from .gateway import POLICY, SCHEMAS
from .models import model_root
from .store import Store


class Submission(Contract):
    brief: Brief | None = None
    instruction: str | None = Field(default=None, min_length=1, max_length=12000)

    @model_validator(mode="after")
    def either(self):
        if (self.brief is None) == (self.instruction is None):
            raise ValueError("Provide either a structured brief or an instruction")
        return self


class ToolSubmission(Contract):
    tool: str
    arguments: dict


class CalibrationSubmission(Contract):
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    csv: str = Field(min_length=1, max_length=2_000_000)


class Queue:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.processes = {}
        self.pending = set()

    def submit(self, run_id):
        with self.lock:
            if run_id in self.pending: raise ValueError("Run already scheduled")
            self.pending.add(run_id)
        self.pool.submit(self.execute, run_id)

    def execute(self, run_id):
        store = Store(self.root)
        try:
            with self.lock:
                run = store.get(run_id)
                if run["state"] == "cancelled": return
                store.update(run_id, "running", run["payload"])
                logs = self.root / "logs"; logs.mkdir(exist_ok=True)
                with (logs / f"{run_id}.log").open("w") as log:
                    proc = subprocess.Popen([sys.executable, "-m", "peptide_agent.web_worker", str(self.root), run_id],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, PEPTIDE_MODEL_DIR=str(model_root()), HF_HUB_OFFLINE="1",
                                 TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4"))
                self.processes[run_id] = proc
            try:
                proc.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            with self.lock:
                current = store.get(run_id)
                if current["state"] not in {"complete", "failed", "blocked", "cancelled"}:
                    store.update(run_id, "failed", {**current["payload"], "error": "Worker stopped before completing; inspect log"})
        finally:
            with self.lock:
                self.pending.discard(run_id); self.processes.pop(run_id, None)
            store.db.close()

    def cancel(self, run_id):
        with self.lock:
            store = Store(self.root)
            try:
                run = store.get(run_id)
                if run["state"] in {"complete", "failed", "blocked", "cancelled"}:
                    raise ValueError("Run is already terminal")
                store.update(run_id, "cancelled", run["payload"])
                proc = self.processes.get(run_id)
                if proc and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            finally: store.db.close()

    def close(self):
        for run_id in list(self.pending):
            try: self.cancel(run_id)
            except (ValueError, ProcessLookupError): pass
        self.pool.shutdown(wait=True, cancel_futures=True)


def create_app(root=".peptide-agent"):
    root = Path(root).resolve()
    queue = Queue(root)
    @asynccontextmanager
    async def lifespan(app):
        store = Store(root)
        for run in store.list_runs():
            if run["state"] not in {"complete", "failed", "blocked", "cancelled", "unavailable", "error"}:
                current = store.get(run["id"])
                store.update(run["id"], "blocked", {**current["payload"], "error": "Server restarted; resume this run"})
        store.db.close()
        yield
        queue.close()
    app = FastAPI(title="Peptide Agent", lifespan=lifespan)
    app.state.queue = queue
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin mutation rejected"}, status_code=403)
            if len(await request.body()) > 4_000_000:
                return JSONResponse({"detail": "Request exceeds 4 MB"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    def read_run(run_id):
        store = Store(root)
        try: return {**store.get(run_id), "events": store.events(run_id)}
        except ValueError as exc: raise HTTPException(404, str(exc)) from exc
        finally: store.db.close()

    def launch(brief, skill, payload):
        store = Store(root)
        try:
            run_id = store.create(brief, skill); store.update(run_id, "queued", payload)
            queue.submit(run_id)
            return {"id": run_id, "state": "queued"}
        finally: store.db.close()

    @app.get("/api/capabilities")
    def get_capabilities(): return {**capabilities(), "policy": POLICY}

    @app.get("/api/runs")
    def get_runs():
        store = Store(root)
        try: return store.list_runs()
        finally: store.db.close()

    @app.post("/api/runs", status_code=202)
    def create_run(request: Submission):
        if request.brief:
            skill = "analyze-peptide" if request.brief.source == "library" else "design-peptide"
            return launch(request.brief.model_dump(), skill, {})
        return launch({}, "design-peptide", {"instruction": request.instruction})

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str): return read_run(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        read_run(run_id)
        try: queue.cancel(run_id)
        except ValueError as exc: raise HTTPException(409, str(exc)) from exc
        return read_run(run_id)

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str):
        run = read_run(run_id)
        if run["state"] not in {"blocked", "failed", "cancelled"}:
            raise HTTPException(409, "Only blocked, failed or cancelled runs can resume")
        store = Store(root)
        try:
            if run_id in queue.pending: raise HTTPException(409, "Worker is still stopping")
            store.update(run_id, "queued", run["payload"]); queue.submit(run_id)
        finally: store.db.close()
        return {"id": run_id, "state": "queued"}

    @app.get("/api/runs/{run_id}/log")
    def get_log(run_id: str):
        read_run(run_id)
        path = root / "logs" / f"{run_id}.log"
        return {"log": path.read_text(errors="replace")[-16000:] if path.exists() else ""}

    @app.get("/api/runs/{run_id}/download")
    def download_report(run_id: str):
        return Response(json.dumps(read_run(run_id), indent=2), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="peptide-report.json"'})

    @app.get("/api/runs/{run_id}/complex/{index}")
    def complex_file(run_id: str, index: int):
        run = read_run(run_id)
        complexes = run["payload"].get("generation", {}).get("data", {}).get("complexes", [])
        if not 0 <= index < len(complexes): raise HTTPException(404, "Complex not found")
        return Response(complexes[index]["pdb"], media_type="chemical/x-pdb",
                        headers={"Content-Disposition": f'attachment; filename="complex-{index}.pdb"'})

    @app.post("/api/tools", status_code=202)
    def run_tool(request: ToolSubmission):
        if request.tool not in POLICY["skills"]["analyze-peptide"]:
            raise HTTPException(403, "Only analysis tools are available through this endpoint")
        try: SCHEMAS[request.tool].model_validate(request.arguments)
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc
        return launch({}, "analyze-peptide", {"operation": "tool", **request.model_dump()})

    @app.post("/api/calibrations", status_code=202)
    def calibrate(request: CalibrationSubmission):
        return launch({}, "analyze-peptide", {"operation": "calibration", **request.model_dump()})

    @app.get("/api/calibrations")
    def list_calibrations():
        return [{k: d[k] for k in ("id", "counts", "test_rmse", "test_coverage", "domain")}
                for p in sorted((model_root()/"calibrations").glob("*.json")) for d in [json.loads(p.read_text())]]

    web_root = str(files("peptide_agent").joinpath("web"))
    app.mount("/web", StaticFiles(directory=web_root), name="web")
    @app.get("/")
    def index(): return FileResponse(Path(web_root)/"index.html")
    return app
