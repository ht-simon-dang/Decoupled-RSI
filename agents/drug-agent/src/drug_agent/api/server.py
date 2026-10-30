"""FastAPI schema-first API.

Exposes programs, briefs, runs, tasks, capabilities, candidates, predictions,
evidence, decisions, approvals, benchmarks and reports. Mutations accept an
``Idempotency-Key`` header; every response carries an ``X-Correlation-ID``.
Authorization is a seam (``require_auth``) — a no-op locally, replaceable in
production. OpenAPI is generated automatically at ``/openapi.json``.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..contracts.brief import ResearchBrief
from ..orchestration.engine import UnknownGateError
from ..service import build_engine
from ..settings.config import Settings, load_settings
from ..util import new_id

_STATIC_DIR = _Path(__file__).parent / "static"
_FIXTURE_BRIEF_DIR = _Path("fixtures/briefs")

# Object types the read-only dashboard is allowed to enumerate.
_UI_OBJECT_TYPES = {
    "safety",
    "generation_batch",
    "synthesis",
    "perturbation",
    "pose",
    "simulation",
    "experiment",
    "hypothesis",
    "sensitivity",
    "manifest",
    "evidence",
    "decision",
    "data_card",
    "baseline_metrics",
    "agent_result",
    "scorecard",
    "candidate",
    "prediction",
    "portfolio",
    "report",
}


class ErrorModel(BaseModel):
    code: str
    message: str


class RunCreate(BaseModel):
    brief: dict[str, Any]
    profile: str = "local"
    deterministic: bool = True
    start: bool = True


def require_auth(authorization: str | None = Header(default=None)) -> str:
    """Authorization seam. Local build accepts anonymous; production replaces this."""
    return authorization or "anonymous"


def _origin_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Candidates grouped by sourcing origin (library / seed / analogue / pfm…)."""
    counts: dict[str, int] = {}
    for c in candidates:
        origin = str(c.get("origin") or "unknown")
        counts[origin] = counts.get(origin, 0) + 1
    return dict(sorted(counts.items()))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = (settings or load_settings()).resolved()
    engine = build_engine(settings)
    engine.history_source = "api"  # so the running history says where a run came from
    app = FastAPI(
        title="drug-agent API",
        version="1.1.0",
        description="Multi-agent drug-discovery research prioritization.",
    )
    # In-memory idempotency ledger (a real deployment uses the store).
    idempotency: dict[str, str] = {}

    @app.middleware("http")
    async def correlation(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or new_id("cid")
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(HTTPException)
    async def http_exc(_req: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": str(exc.status_code), "detail": exc.detail}},
        )

    @app.exception_handler(ValueError)
    async def value_exc(_req: Request, exc: ValueError):
        return JSONResponse(
            status_code=422, content={"error": {"code": "validation_error", "detail": str(exc)}}
        )

    S = engine.store

    # --- health / capabilities --------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "profile": settings.profile, "network_allowed": settings.allow_network}

    @app.get("/capabilities")
    def capabilities(_: str = Depends(require_auth)) -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in engine.registry.profiles()]

    @app.get("/schemas")
    def schemas() -> dict[str, Any]:
        from ..contracts import CONTRACT_REGISTRY

        return {n: m.json_schema() for n, m in CONTRACT_REGISTRY.items()}

    # --- programs ----------------------------------------------------------
    @app.post("/programs", status_code=201)
    def create_program(body: dict[str, Any], _: str = Depends(require_auth)) -> dict[str, Any]:
        from ..contracts.brief import ResearchProgram

        prog = ResearchProgram.model_validate(body)
        S.put_object("program", prog, program_id=prog.id)
        return prog.model_dump(mode="json")

    @app.get("/programs/{program_id}")
    def get_program(program_id: str) -> dict[str, Any]:
        obj = S.get_object("program", program_id)
        if not obj:
            raise HTTPException(404, "program not found")
        return obj

    # --- briefs ------------------------------------------------------------
    @app.post("/briefs", status_code=201)
    def create_brief(body: dict[str, Any], _: str = Depends(require_auth)) -> dict[str, Any]:
        brief = ResearchBrief.model_validate(body)  # 422 on invalid
        S.put_object("brief", brief, program_id=brief.program_id)
        return brief.model_dump(mode="json")

    @app.get("/briefs/{brief_id}")
    def get_brief(brief_id: str) -> dict[str, Any]:
        obj = S.get_object("brief", brief_id)
        if not obj:
            raise HTTPException(404, "brief not found")
        return obj

    # --- runs --------------------------------------------------------------
    @app.post("/runs", status_code=201)
    def create_run(
        body: RunCreate,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: str = Depends(require_auth),
    ) -> dict[str, Any]:
        if idempotency_key and idempotency_key in idempotency:
            existing = S.get_run(idempotency[idempotency_key])
            return {"run": existing, "idempotent_replay": True}
        brief = ResearchBrief.model_validate(body.brief)
        run = engine.create_run(brief, deterministic=body.deterministic, profile=body.profile)
        if idempotency_key:
            idempotency[idempotency_key] = run.id
        if body.start:
            run = engine.run(run.id)
        return {"run": run.model_dump(mode="json")}

    @app.get("/runs")
    def list_runs(limit: int = Query(50, le=200), offset: int = 0) -> dict[str, Any]:
        runs = S.list_runs(limit=limit + offset)[offset : offset + limit]
        return {"items": runs, "limit": limit, "offset": offset, "count": len(runs)}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        obj = S.get_run(run_id)
        if not obj:
            raise HTTPException(404, "run not found")
        return obj

    @app.post("/runs/{run_id}/resume")
    def resume_run(run_id: str, _: str = Depends(require_auth)) -> dict[str, Any]:
        if not S.get_run(run_id):
            raise HTTPException(404, "run not found")
        return engine.resume(run_id).model_dump(mode="json")

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str, _: str = Depends(require_auth)) -> dict[str, Any]:
        if not S.get_run(run_id):
            raise HTTPException(404, "run not found")
        return engine.cancel(run_id).model_dump(mode="json")

    @app.get("/runs/{run_id}/tasks")
    def get_tasks(run_id: str) -> dict[str, Any]:
        return {"items": S.list_tasks(run_id)}

    @app.get("/runs/{run_id}/events")
    def get_events(run_id: str, task: str | None = None) -> dict[str, Any]:
        return {"items": S.list_events(run_id, task_id=task)}

    # --- approvals ---------------------------------------------------------
    @app.post("/runs/{run_id}/approvals/{gate}")
    def decide_gate(
        run_id: str, gate: str, body: dict[str, Any], _: str = Depends(require_auth)
    ) -> dict[str, Any]:
        if not S.get_run(run_id):
            raise HTTPException(404, "run not found")
        decision = body.get("decision", "approve")
        try:
            if decision == "approve":
                engine.approve_gate(run_id, gate, by=body.get("by", "api"))
                run = engine.resume(run_id)
            else:
                run = engine.reject_gate(
                    run_id, gate, by=body.get("by", "api"), reason=body.get("reason", "")
                )
        except UnknownGateError as exc:
            # 404 on the addressed resource: this run has no such gate. Without this the
            # decision would be recorded against nothing and answer 200.
            known = ", ".join(exc.known) or "no gates"
            raise HTTPException(404, f"unknown gate {exc.gate!r}; this run has: {known}") from None
        return run.model_dump(mode="json")

    # --- run outputs -------------------------------------------------------
    def _paged(items: list[dict], limit: int, offset: int) -> dict[str, Any]:
        return {
            "items": items[offset : offset + limit],
            "count": len(items),
            "limit": limit,
            "offset": offset,
        }

    @app.get("/runs/{run_id}/candidates")
    def candidates(
        run_id: str, limit: int = Query(50, le=500), offset: int = 0, passed_filters: bool | None = None
    ) -> dict[str, Any]:
        items = S.list_objects("candidate", run_id=run_id)
        if passed_filters is not None:
            items = [c for c in items if bool(c.get("passed_hard_filters")) == passed_filters]
        return _paged(items, limit, offset)

    @app.get("/runs/{run_id}/predictions")
    def predictions(
        run_id: str, endpoint: str | None = None, limit: int = Query(100, le=1000), offset: int = 0
    ) -> dict[str, Any]:
        items = S.list_objects("prediction", run_id=run_id)
        if endpoint:
            items = [p for p in items if p.get("endpoint") == endpoint]
        return _paged(items, limit, offset)

    @app.get("/runs/{run_id}/scorecards")
    def scorecards(run_id: str, limit: int = Query(100, le=1000), offset: int = 0) -> dict[str, Any]:
        return _paged(S.list_objects("scorecard", run_id=run_id), limit, offset)

    @app.get("/runs/{run_id}/portfolio")
    def portfolio(run_id: str) -> dict[str, Any]:
        items = S.list_objects("portfolio", run_id=run_id)
        if not items:
            raise HTTPException(404, "portfolio not available")
        return items[0]

    @app.get("/runs/{run_id}/evidence")
    def evidence(run_id: str, limit: int = Query(100, le=1000), offset: int = 0) -> dict[str, Any]:
        return _paged(S.list_objects("evidence", run_id=run_id), limit, offset)

    @app.get("/runs/{run_id}/decisions")
    def decisions(run_id: str, limit: int = Query(100, le=1000), offset: int = 0) -> dict[str, Any]:
        return _paged(S.list_objects("decision", run_id=run_id), limit, offset)

    @app.get("/runs/{run_id}/report")
    def report(run_id: str, format: str = Query("json", pattern="^(json|md|html)$")):
        rep = S.get_object("report", run_id)
        if not rep:
            raise HTTPException(404, "report not available")
        if format == "json":
            return engine.artifacts.get_json(rep["json_ref"])
        ref = rep["md_ref"] if format == "md" else rep["html_ref"]
        body = engine.artifacts.get_bytes(ref).decode("utf-8")
        return HTMLResponse(body) if format == "html" else PlainTextResponse(body)

    @app.get("/artifacts/{content_hash}")
    def artifact(content_hash: str):
        try:
            data = engine.artifacts.get_bytes(f"sha256:{content_hash}")
        except FileNotFoundError:
            raise HTTPException(404, "artifact not found") from None
        return Response(content=data, media_type="application/octet-stream")

    # --- benchmarks --------------------------------------------------------
    @app.post("/benchmarks/{suite}")
    def benchmark(suite: str, _: str = Depends(require_auth)) -> dict[str, Any]:
        from ..benchmarks.suites import run_suite

        return run_suite(suite)

    # --- generic run objects (read-only, allowlisted) ----------------------
    @app.get("/runs/{run_id}/objects/{otype}")
    def run_objects(run_id: str, otype: str) -> dict[str, Any]:
        if otype not in _UI_OBJECT_TYPES:
            raise HTTPException(400, f"object type not exposed: {otype}")
        return {"items": S.list_objects(otype, run_id=run_id)}

    # --- example briefs (for launching runs from the UI) -------------------
    # Note: mounted at /examples/briefs to avoid colliding with /briefs/{id}.
    @app.get("/examples/briefs")
    def brief_examples() -> dict[str, Any]:
        out = []
        if _FIXTURE_BRIEF_DIR.exists():
            for p in sorted(_FIXTURE_BRIEF_DIR.glob("*.yaml")):
                try:
                    body = yaml.safe_load(p.read_text())
                except Exception:
                    continue
                if not isinstance(body, dict):
                    continue
                out.append(
                    {
                        "name": p.stem,
                        "path": str(p),
                        "title": body.get("title", p.stem),
                        "kind": body.get("kind"),
                        "valid": "objectives" in body,
                        "body": body,
                    }
                )
        return {"items": out}

    # --- consolidated dashboard payload (single round-trip for the UI) -----
    @app.get("/runs/{run_id}/dashboard")
    def dashboard(run_id: str) -> dict[str, Any]:
        run = S.get_run(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        tasks = S.list_tasks(run_id)
        candidates = S.list_objects("candidate", run_id=run_id)
        scorecards = S.list_objects("scorecard", run_id=run_id)
        portfolios = S.list_objects("portfolio", run_id=run_id)
        portfolio = portfolios[0] if portfolios else None
        brief = S.get_object("brief", run["brief_id"]) or {}
        n_pass = sum(1 for c in candidates if c.get("passed_hard_filters"))
        baseline = S.list_objects("baseline_metrics", run_id=run_id)
        manifests = S.list_objects("manifest", run_id=run_id)
        return {
            "run": run,
            "brief": brief,
            "tasks": tasks,
            "events": S.list_events(run_id)[-200:],
            "funnel": {
                "sourced": len(S.list_objects("raw_candidate", run_id=run_id)),
                "standardized": len(candidates),
                "passed_filters": n_pass,
                "pareto": (portfolio or {}).get("n_pareto", 0),
                "shortlist": len((portfolio or {}).get("shortlist", [])),
                "reserve": len((portfolio or {}).get("reserve", [])),
            },
            "generation": S.list_objects("generation_batch", run_id=run_id),
            "origin_counts": _origin_counts(candidates),
            "portfolio": portfolio,
            "objectives": (portfolio or {}).get("objectives", brief.get("objectives", [])),
            "candidates": candidates,
            "scorecards": scorecards,
            "predictions": S.list_objects("prediction", run_id=run_id),
            "safety": S.list_objects("safety", run_id=run_id),
            "synthesis": S.list_objects("synthesis", run_id=run_id),
            "perturbation": S.list_objects("perturbation", run_id=run_id),
            "evidence": S.list_objects("evidence", run_id=run_id),
            "decisions": S.list_objects("decision", run_id=run_id),
            "experiments": S.list_objects("experiment", run_id=run_id),
            "hypotheses": S.list_objects("hypothesis", run_id=run_id),
            "sensitivity": next(iter(S.list_objects("sensitivity", run_id=run_id)), None),
            "baseline": baseline[0] if baseline else None,
            "data_cards": S.list_objects("data_card", run_id=run_id),
            "manifest": manifests[0] if manifests else None,
            "capability_matrix": engine.registry.capability_matrix(),
        }

    # --- static Web UI -----------------------------------------------------
    if _STATIC_DIR.exists():

        @app.get("/", include_in_schema=False)
        def _root() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

        app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    return app


# Module-level app for `uvicorn drug_agent.api.server:app`.
app = create_app()
