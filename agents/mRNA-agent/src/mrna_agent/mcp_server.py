"""Local stdio MCP facade over the same validated workflow/evidence boundary."""
from mcp.server.fastmcp import FastMCP
from .contracts import Brief
from .storage import Store
from .delivery import public_run


def create_server(root=None):
    server=FastMCP("EMA4D mRNA Design Agent")

    @server.tool()
    def mrna_capabilities() -> dict:
        """Inspect installed models, scope, provenance and disk quota."""
        from .registry import capabilities
        return capabilities()

    @server.tool()
    def mrna_design(brief:dict, fixed_plan:bool=False) -> dict:
        """Design mRNA for an explicitly locked protein with scientific tool validation."""
        from .workflow import Engine
        with Store(root) as store:return public_run(Engine(store).run(Brief.model_validate(brief),planner=not fixed_plan))

    @server.tool()
    def mrna_run_status(run_id:str) -> dict:
        """Read a task, immutable evidence and its execution journal."""
        with Store(root) as store:return public_run({**store.get(run_id),"events":store.events(run_id)})

    @server.tool()
    def mrna_resume(run_id:str) -> dict:
        """Resume existing checkpoints; preserve target, call counts and original scores."""
        from .workflow import Engine
        with Store(root) as store:
            with store.exclusive(run_id):
                run=store.get(run_id)
                if run["state"] not in {"paused","failed","blocked"}:raise ValueError("Run is not resumable")
                store.update(run_id,"queued",run["payload"])
            return public_run(Engine(store).resume(run_id))

    @server.tool()
    def mrna_reevaluate(run_id:str) -> dict:
        """Create a new evaluation record; keep original predictions unchanged."""
        from .workflow import Engine
        with Store(root) as store:return public_run(Engine(store).reevaluate(run_id))

    @server.tool()
    def mrna_import_experiments(records:list[dict]) -> dict:
        """Import raw measured records with construct, context, batch, controls and quality."""
        from .feedback import import_experiments
        with Store(root) as store:return import_experiments(store,records)

    @server.tool()
    def mrna_feedback(run_id:str) -> dict:
        """Analyze comparable observations and recommend the next batch."""
        from .feedback import analyze_feedback
        with Store(root) as store:
            from .delivery import public_feedback
            return public_feedback(analyze_feedback(store,run_id),store.get(run_id))
    return server


def serve(root=None):create_server(root).run(transport="stdio")
