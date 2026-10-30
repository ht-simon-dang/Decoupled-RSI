"""Local WebUI. All public routes share delivery.public_run."""

from __future__ import annotations

import fcntl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import Field, model_validator

from .config import data_root
from .contracts import Brief, Contract, Observation
from .delivery import public_run
from .exports import export, structure_export
from .feedback import analyze, ingest, review
from .learning import PredictionRequest, TrainingRequest
from .registry import capabilities
from .store import TERMINAL, Store
from .templates import TEMPLATES, template
from .workflow import Engine


class Submission(Contract):
    brief: Brief | None = None
    instruction: str | None = Field(default=None, min_length=1, max_length=12000)

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.brief is None) == (self.instruction is None):
            raise ValueError("Provide one brief or instruction")
        return self


class FeedbackSubmission(Contract):
    observations: list[Observation] = Field(min_length=1, max_length=128)


class ReviewSubmission(Contract):
    candidate_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: str
    note: str = Field(min_length=1, max_length=2000)


class Queue:
    def __init__(self, root):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="antibody")
        self.lock = threading.Lock()
        self.pending = set()

    def submit(self, run_id):
        with self.lock:
            if run_id in self.pending or len(self.pending) >= 32:
                raise ValueError("Task already queued or local queue is full")
            self.pending.add(run_id)
        self.pool.submit(self.execute, run_id)

    def execute(self, run_id):
        try:
            with Store(self.root) as store:
                if store.get(run_id)["state"] != "cancelled":
                    Engine(store).resume(run_id)
        except Exception:
            with Store(self.root) as store:
                run = store.get(run_id)
                store.update(
                    run_id, "failed", {**run["payload"], "diagnosis": "执行器中断；可在本地检查并恢复任务。"}
                )
        finally:
            with self.lock:
                self.pending.discard(run_id)

    def close(self):
        with self.lock:
            pending = list(self.pending)
        with Store(self.root) as store:
            for run_id in pending:
                run = store.get(run_id)
                if run["state"] not in TERMINAL:
                    store.update(run_id, "cancelled", run["payload"], force=True)
        self.pool.shutdown(wait=True, cancel_futures=True)


def create_app(root=None):
    root = Path(root or data_root() / "runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    queue = Queue(root)

    @asynccontextmanager
    async def lifespan(app):
        with (root / ".server.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another WebUI owns this store") from exc
            with Store(root) as store:
                for item in store.list_runs(10000):
                    if item["state"] not in TERMINAL:
                        run = store.get(item["id"])
                        payload = run["payload"]
                        if item["state"] != "queued":
                            payload["elapsed_seconds"] = payload.get("elapsed_seconds", 0) + max(
                                0, time.time() - run["updated"]
                            )
                        payload["diagnosis"] = "服务上次中断；恢复任务会保留已消耗的预算。"
                        store.update(item["id"], "interrupted", payload)
            try:
                yield
            finally:
                queue.close()

    app = FastAPI(title="EMA4D Antibody Agent", version="0.1.0", lifespan=lifespan)
    app.state.queue = queue
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if (origin and urlsplit(origin).netloc != request.headers.get("host")) or request.headers.get(
                "sec-fetch-site"
            ) == "cross-site":
                return JSONResponse({"detail": "Cross-origin mutation rejected"}, status_code=403)
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 12_000_000:
                    return JSONResponse({"detail": "Request exceeds 12 MB"}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Record not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def read(run_id):
        with Store(root) as store:
            run = store.get(run_id)
            run["events"] = store.events(run_id)
            return run

    def launch(brief, payload=None):
        with Store(root) as store:
            run_id = store.create(brief, payload)
            try:
                queue.submit(run_id)
            except ValueError:
                store.update(run_id, "blocked", {"diagnosis": "本地任务队列已满。"})
                raise
        return {"id": run_id, "state": "queued"}

    @app.get("/api/capabilities")
    def get_capabilities():
        return capabilities()

    @app.get("/api/schema")
    def schema():
        return Brief.model_json_schema()

    @app.get("/api/templates")
    def templates():
        return TEMPLATES

    @app.get("/api/templates/{name}")
    def get_template(name: str):
        return template(name)

    @app.get("/api/runs")
    def history(limit: int = 100):
        if not 1 <= limit <= 500:
            raise ValueError("Limit must be 1..500")
        with Store(root) as store:
            return store.list_runs(limit)

    @app.post("/api/runs", status_code=202)
    def submit(request: Submission):
        return launch(
            request.brief.model_dump() if request.brief else {},
            {"instruction": request.instruction} if request.instruction else {},
        )

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        result = public_run(read(run_id))
        with Store(root) as store:
            allowed = {c["id"] for c in result["candidates"]}
            result["reviews"] = [r for r in store.reviews(run_id) if r["candidate_id"] in allowed]
        return result

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        with Store(root) as store:
            run = store.get(run_id)
            if run["state"] in TERMINAL:
                raise HTTPException(409, "Run is already stopped")
            store.update(run_id, "cancelled", run["payload"], force=True)
        return {"id": run_id, "state": "cancelled"}

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume(run_id: str):
        with queue.lock:
            if run_id in queue.pending:
                raise HTTPException(409, "Executor is still stopping/running")
        with Store(root) as store:
            Engine(store).request_resume(run_id)
            queue.submit(run_id)
        return {"id": run_id, "state": "queued"}

    @app.post("/api/runs/{run_id}/reevaluate", status_code=202)
    def reevaluate(run_id: str):
        with Store(root) as store:
            new_id = Engine(store).prepare_reevaluation(run_id)
            queue.submit(new_id)
        return {"id": new_id, "state": "queued"}

    @app.get("/api/runs/{run_id}/export/{format}")
    def download(run_id: str, format: str):
        content, media = export(read(run_id), format)
        suffix = "md" if format == "report" else format
        return Response(
            content,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="antibody-{run_id}.{suffix}"'},
        )

    @app.get("/api/runs/{run_id}/structures/{candidate_id}")
    def structure(run_id: str, candidate_id: str, state_id: str | None = None):
        content = structure_export(read(run_id), candidate_id, state_id)
        return Response(content, media_type="chemical/x-pdb")

    @app.get("/api/runs/{run_id}/feedback")
    def get_feedback(run_id: str):
        with Store(root) as store:
            return analyze(store, run_id)

    @app.post("/api/runs/{run_id}/feedback")
    def post_feedback(run_id: str, request: FeedbackSubmission):
        with Store(root) as store:
            return ingest(store, run_id, request.observations)

    @app.post("/api/runs/{run_id}/review")
    def post_review(run_id: str, request: ReviewSubmission):
        with Store(root) as store:
            return review(store, run_id, request.candidate_id, request.decision, request.note)

    @app.get("/api/runs/{run_id}/benchmark")
    def benchmark(run_id: str):
        from .benchmark import compare

        return compare(read(run_id))

    @app.post("/api/property-models")
    def train_property_model(request: TrainingRequest):
        from .learning import fit

        return fit(request)

    @app.post("/api/runs/{run_id}/next-batch")
    def next_batch(run_id: str, request: PredictionRequest):
        from .learning import predict_batch

        return predict_batch(read(run_id), request)

    static = files("antibody_agent") / "web"
    app.mount("/web", StaticFiles(directory=str(static)), name="web")

    @app.get("/")
    def index():
        return FileResponse(str(static / "index.html"))

    return app
