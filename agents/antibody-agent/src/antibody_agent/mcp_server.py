"""Optional MCP transport, using the same evaluation and delivery boundary."""


def serve(root=None):
    from mcp.server.fastmcp import FastMCP

    from .contracts import Brief
    from .delivery import public_run
    from .feedback import ingest
    from .registry import capabilities
    from .store import Store
    from .workflow import Engine

    app = FastMCP("EMA4D Antibody Agent")

    @app.tool()
    def antibody_capabilities() -> dict:
        return capabilities()

    @app.tool()
    def antibody_design(brief: dict) -> dict:
        with Store(root) as store:
            return public_run(Engine(store).run(Brief.model_validate(brief)))

    @app.tool()
    def antibody_instruct(instruction: str) -> dict:
        with Store(root) as store:
            return public_run(Engine(store).run(instruction=instruction))

    @app.tool()
    def antibody_report(run_id: str) -> dict:
        with Store(root) as store:
            return public_run(store.get(run_id))

    @app.tool()
    def antibody_feedback(run_id: str, observations: list[dict]) -> dict:
        with Store(root) as store:
            return ingest(store, run_id, observations)

    @app.tool()
    def antibody_next_batch(run_id: str, objectives: list[dict], count: int = 4) -> dict:
        from .learning import predict_batch

        with Store(root) as store:
            return predict_batch(store.get(run_id), {"objectives": objectives, "count": count})

    app.run(transport="stdio")
