"""FastAPI server: serves the offline WebUI and streams the master agent's work
as Server-Sent Events. The heavy scientific pipeline runs in a threadpool inside
the agents, so the event stream stays responsive while the model thinks.

It also exposes the request journal (``/api/journal/...``), which is what makes a
finished run inspectable: the full chat log, the tool ledgers, and a cross-run
view of how each tool has been performing — and, under ``/api/evolve/...``, the
self-evolution loop that studies those same journals. The two belong in one
console because they are two views of one thing: what the platform did, and what
that says about how it should change.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agents.llm.master_agent import MasterAgent, installed_arms, installed_generators
from ..agents.registry import AgentCard, agent_pool
from ..observability import journal as jnl
from ..settings.config import Settings, load_settings
from ..tools.registry import ToolCard, tool_pool
from . import events as ev
from .config import Config, load_config
from .ollama import OllamaClient

WEB_DIR = Path(__file__).resolve().parent / "web"

_PRESETS = [
    {
        "id": "seh_selective",
        "label": "sEH — selective, potent, drug-like",
        "text": (
            "Find selective inhibitors of soluble epoxide hydrolase (sEH / EPHX2). "
            "They should be potent (predicted pIC50 >= 7), drug-like (Lipinski-compliant, "
            "no structural alerts), and synthetically accessible."
        ),
    },
    {
        "id": "seh_novel",
        "label": "sEH — novel scaffolds vs known ureas",
        "text": (
            "Propose novel, structurally diverse sEH inhibitor chemotypes that move away "
            "from the classic urea/amide pharmacophore, while keeping good potency and "
            "drug-likeness. Prioritise scaffold novelty and chemical diversity."
        ),
    },
    {
        "id": "seh_strict",
        "label": "sEH — strict CNS-like property window (hard)",
        "text": (
            "Find sEH inhibitors with a strict CNS-like property window: molecular weight "
            "<= 400, logP <= 3, TPSA <= 90, zero structural alerts, and predicted pIC50 >= 7.5. "
            "This is deliberately demanding."
        ),
    },
    {
        "id": "seh_denovo",
        "label": "sEH — de-novo scaffolds from the PFM generative model",
        "text": (
            "Design genuinely new sEH (EPHX2) inhibitor chemotypes from scratch rather than "
            "re-screening known compounds. Sample de-novo structures for the binding pocket, "
            "then keep only those that are potent, drug-like and free of structural alerts. "
            "Prioritise scaffold novelty; sample generously."
        ),
    },
    {
        "id": "endothelium",
        "label": "Phenotype — rescue endothelial dysfunction",
        "text": (
            "Prioritise small molecules that reverse a dysfunctional vascular-endothelium "
            "phenotype toward the healthy state, favouring high restoration score, good "
            "drug-likeness, and synthetic accessibility."
        ),
    },
]


# ---- the topology of this deployment ------------------------------------
#: Order the planes are rendered in: the reasoning plane is the root of the tree
#: (the master agent is what a request enters through), the platform agents shape
#: a run, the specialists do the pipeline work and hold most of the tool edges.
#: Same taxonomy and same order as the pool table in ``web/app.js``.
_KIND_ORDER = {"REASONING": 0, "PLATFORM": 1, "SPECIALIST": 2}


def _agent_sort_key(card: AgentCard) -> tuple[int, int, str]:
    """Reasoning plane first, and the LLM-backed agent first within its plane.

    The master agent is the root of the graph but sorts last alphabetically among
    the reasoning cards, so a plain id sort would hand the frontend a tree whose
    root is third.
    """
    return (_KIND_ORDER.get(str(card.kind), len(_KIND_ORDER)), 0 if card.llm_backed else 1, card.agent_id)


def _topology(agents: list[AgentCard], tools: list[ToolCard], *, profile: str) -> dict[str, Any]:
    """The agent/tool graph, derived once here rather than twice in the client.

    Everything below is computed from the two pools — ``AgentCard.uses_tools`` is
    the edge list — so wiring an agent to a tool updates the view by itself. It is
    derived server-side because two views on one page (the pool table and the
    topology tree) would otherwise each re-implement the same three rules, and the
    interesting facts are exactly the ones a client is tempted to drop:

      * a tool no agent calls exists and is unreachable. Hiding it would make the
        page claim a smaller, healthier platform than the one that is installed —
        seventeen working tools were wired to no agent at all until recently.
      * a tool that is AVAILABLE but unwired is not "ready to use", and a tool that
        is wired but UNAVAILABLE will refuse mid-run (ADR-0011). Both facts have to
        survive to the node.
      * an agent naming a tool the pool does not hold is a wiring bug, not an edge
        to silently drop, so those are reported rather than filtered out.
    """
    by_id = {t.tool_id: t for t in tools}
    callers: dict[str, list[str]] = {t.tool_id: [] for t in tools}
    edges: list[dict[str, str]] = []
    dangling: list[dict[str, str]] = []
    agent_nodes: list[dict[str, Any]] = []

    for card in sorted(agents, key=_agent_sort_key):
        tool_ids: list[str] = []
        for tool_id in card.uses_tools:
            tool = by_id.get(tool_id)
            if tool is None:
                dangling.append({"agent_id": card.agent_id, "tool_id": tool_id})
                continue
            tool_ids.append(tool_id)
            callers[tool_id].append(card.agent_id)
            edges.append({"agent_id": card.agent_id, "tool_id": tool_id, "tool_status": str(tool.status)})
        agent_nodes.append(
            {
                "id": card.agent_id,
                "kind": str(card.kind),
                "title": card.title or card.agent_id,
                "description": card.description,
                "llm_backed": card.llm_backed,
                "status": str(card.status),
                "notes": card.notes,
                "tool_ids": tool_ids,
                "tool_count": len(tool_ids),
                "available_tool_count": sum(1 for t in tool_ids if str(by_id[t].status) == "AVAILABLE"),
            }
        )

    tool_nodes: list[dict[str, Any]] = []
    for tool_card in tools:
        tool_nodes.append(
            {
                "id": tool_card.tool_id,
                "title": tool_card.title or tool_card.tool_id,
                "description": tool_card.description,
                # The namespace, not a label: `TOOL_DOMAINS` in web/app.js turns it
                # into a heading, and a test holds that map to the live pool.
                "domain": tool_card.tool_id.split(".")[0],
                "status": str(tool_card.status),
                "available": str(tool_card.status) == "AVAILABLE",
                "backend": tool_card.backend,
                # An UNAVAILABLE tool without a reason is a shrug. Every adapter
                # sets one; this is the field the view must show, not hide.
                "unavailable_reason": tool_card.unavailable_reason,
                "cost_class": str(tool_card.cost_class),
                "determinism": str(tool_card.determinism),
                "operations": dict(tool_card.operations),
                "agent_ids": callers[tool_card.tool_id],
                "wired": bool(callers[tool_card.tool_id]),
            }
        )

    unwired = [n["id"] for n in tool_nodes if not n["wired"]]
    kinds: dict[str, int] = {}
    for node in agent_nodes:
        kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
    runnable = [n for n in tool_nodes if n["available"]]
    return {
        "profile": profile,
        "agents": agent_nodes,
        "tools": tool_nodes,
        "edges": edges,
        "unwired_tool_ids": unwired,
        "dangling_edges": dangling,
        "counts": {
            "agents": len(agent_nodes),
            "agents_by_kind": kinds,
            "tools": len(tool_nodes),
            "available": len(runnable),
            "unavailable": len(tool_nodes) - len(runnable),
            "edges": len(edges),
            "wired_tools": len(tool_nodes) - len(unwired),
            "unwired_tools": len(unwired),
            "unwired_available": sum(1 for n in runnable if not n["wired"]),
        },
    }


def _pool_settings(cfg: Config) -> Settings:
    """The settings a run launched from this console would really use.

    `agent_pool()` and `tool_pool()` called bare build a plain `Settings()`, which
    never reads `configs/<profile>.yaml` and therefore never sees `external_tools`
    — the documented, highest-precedence way to point a catalogue tool at the
    interpreter that actually holds it. The console then under-reports the machine
    it is running on: four generators that sample here were shown UNAVAILABLE,
    which is the same false-RED already fixed in `cli/main.py:tools` and in the
    master agent's envelope block. Under-reporting is the mirror of the mock this
    project refuses to ship.

    The profile and determinism come from `cfg` for the same reason: `bridge.py`
    builds a run's settings from exactly those two, so this is a description of the
    deployment the Run button uses, not of a neighbouring one.
    """
    return load_settings(profile=cfg.drug_agent_profile, deterministic=cfg.deterministic).resolved()


class RunRequest(BaseModel):
    instruction: str
    strategy: str | None = None
    max_cycles: int | None = None
    model: str | None = None
    #: Where candidate structures come from. The console does NOT send this: the
    #: master agent chooses the source during transcription. It stays on the API
    #: so a scripted caller can still pin one ("library" | "pfm" | "library+pfm")
    #: for an ablation.
    generator: str | None = None
    #: How many molecules the de-novo arm samples. Only meaningful alongside a
    #: pinned `generator`; otherwise the master agent's own value stands.
    n_generated: int | None = None


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="LLM master agent — drug discovery", version="1.1.0")
    tasks: dict[str, asyncio.Task[Any]] = {}

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        client = OllamaClient(cfg.ollama_host, timeout_s=15.0)
        h = await client.health()
        return {
            "ollama": h,
            "model": cfg.model,
            "model_installed": cfg.model in h["models"] or f"{cfg.model}:latest" in h["models"],
            # `installed` is the console's whole picture of the de-novo arm now
            # that it no longer offers a choice: which models could sample here,
            # rather than which one three buttons happened to name.
            "generator": {
                **cfg.generator_status(),
                "installed": list(installed_generators()),
                "chosen_per_run": cfg.generator == "auto",
            },
            # Which OPTIONAL assessment arms this deployment could run if the
            # master agent asks. Reported for the same reason `installed` is: the
            # console cannot say what a run declined to use unless it knows what
            # was on offer, and these five tools spent their whole life installed,
            # advertised, and switched on by nothing.
            "arms": installed_arms(),
            "config": cfg.public_dict(),
        }

    @app.get("/api/presets")
    async def presets() -> dict[str, Any]:
        return {"presets": _PRESETS, "config": cfg.public_dict()}

    @app.post("/api/runs")
    async def create_run(req: RunRequest) -> dict[str, Any]:
        if not req.instruction.strip():
            raise HTTPException(status_code=400, detail="instruction is required")
        run_id = "mrun_" + uuid.uuid4().hex[:12]
        session = ev.REGISTRY.create(run_id)
        agent = MasterAgent(cfg)
        # Open the journal here, before the task starts, so a run that dies in
        # its first second still leaves a record of having been asked for.
        journal = agent.open_journal(
            session,
            req.instruction,
            source="web",
            model=req.model or cfg.model,
            strategy=req.strategy or cfg.refine_strategy,
            generator=req.generator or cfg.generator,
            n_generated=req.n_generated,
            max_cycles=req.max_cycles or cfg.max_cycles,
        )
        task = asyncio.create_task(
            agent.run(
                session,
                req.instruction,
                strategy=req.strategy,
                max_cycles=req.max_cycles,
                model=req.model,
                generator=req.generator,
                n_generated=req.n_generated,
                journal=journal,
            )
        )
        tasks[run_id] = task

        def _forget(_task: asyncio.Task[Any], rid: str = run_id) -> None:
            tasks.pop(rid, None)

        task.add_done_callback(_forget)
        return {"run_id": run_id, "journal": bool(journal)}

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        session = ev.REGISTRY.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown run")

        async def gen() -> Any:
            yield "retry: 3000\n\n"
            async for event in session.stream():
                yield f"data: {json.dumps(event.to_dict())}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.get("/api/runs/{run_id}/result")
    async def get_result(run_id: str) -> dict[str, Any]:
        session = ev.REGISTRY.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return {"finished": session.finished, "result": session.result, "error": session.error}

    # ---- the pools: what this deployment can actually do ------------------
    @app.get("/api/pools")
    async def pools() -> dict[str, Any]:
        """The agent pool, the tool pool, and the graph between them.

        One endpoint rather than two. `topology` is *derived* from `agents` and
        `tools` — it adds no capability the cards do not already carry — so a
        second route would build both pools a second time on every page load and
        give the console two places for the same availability to be computed, and
        therefore to disagree. The tree view and the pool table read one payload.
        """
        settings = await asyncio.to_thread(_pool_settings, cfg)
        # Constructing thirty-nine adapters probes the filesystem for checkouts,
        # weights and interpreters. That is fast but not free, and this route is
        # hit while a run's events are streaming, so it stays off the event loop.
        agents = await asyncio.to_thread(agent_pool, settings)
        tools = await asyncio.to_thread(tool_pool, settings)
        return {
            "agents": [c.model_dump(mode="json") for c in agents],
            "tools": [c.model_dump(mode="json") for c in tools],
            "topology": _topology(agents, tools, profile=settings.profile),
        }

    # ---- the request journal ----------------------------------------------
    @app.get("/api/journal")
    async def journal_index(limit: int = 50) -> dict[str, Any]:
        """Newest-first list of past requests."""
        root = cfg.resolved_journal_dir()
        return {"root": str(root), "requests": jnl.list_requests(root, limit=limit)}

    @app.get("/api/journal/tools")
    async def journal_tools(limit: int = 200) -> dict[str, Any]:
        """How every tool has performed across the last ``limit`` requests."""
        return jnl.tool_statistics(cfg.resolved_journal_dir(), limit=limit)

    @app.get("/api/journal/{request_id}")
    async def journal_request(request_id: str, include_chat: bool = True) -> dict[str, Any]:
        """Everything recorded for one request."""
        data = jnl.load_request(request_id, cfg.resolved_journal_dir())
        if data is None:
            raise HTTPException(status_code=404, detail=f"no journal for request {request_id!r}")
        if not include_chat:
            data = {**data, "chat": []}
        return data

    # ---- the self-evolution loop ------------------------------------------
    # Mounted, not embedded: the subsystem owns its own routes, and the console
    # is only where they surface. A deployment without the loop's store simply
    # gets empty panels rather than an error.
    try:
        from ..selfevolve.webapi import create_router as create_evolve_router

        app.include_router(create_evolve_router())
    except Exception as exc:  # a broken optional subsystem must not stop the console
        reason = f"{type(exc).__name__}: {exc}"

        @app.get("/api/evolve/status")
        async def evolve_unavailable() -> dict[str, Any]:
            return {"available": False, "reason": reason}

    if WEB_DIR.is_dir():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(str(WEB_DIR / "index.html"))

    return app
