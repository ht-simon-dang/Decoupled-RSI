"""MCP entry points share the same typed, journalled workflow as CLI and WebUI."""
def serve(root=None):
    from mcp.server.fastmcp import FastMCP
    from .contracts import Brief
    from .store import Store
    from .workflow import Engine
    from .models import capabilities
    from .templates import template_brief
    mcp=FastMCP('Enzyme Agent')
    @mcp.tool()
    def enzyme_capabilities()->dict:
        return capabilities()
    @mcp.tool()
    def enzyme_research_template(name:str)->dict:
        return template_brief(name).model_dump()
    @mcp.tool()
    def enzyme_design(brief:dict)->dict:
        validated=Brief.model_validate(brief)
        with Store(root) as store:return Engine(store).run(validated)
    @mcp.tool()
    def enzyme_analyze(brief:dict)->dict:
        validated=Brief.model_validate(brief)
        if validated.generators!=['library']:raise ValueError('Analysis entry point cannot authorize a generator')
        with Store(root) as store:return Engine(store).run(validated)
    @mcp.tool()
    def enzyme_report(run_id:str)->dict:
        with Store(root) as store:return {**store.get(run_id),'events':store.events(run_id)}
    @mcp.tool()
    def enzyme_feedback(run_id:str,observations:list[dict])->dict:
        from .feedback import ingest
        with Store(root) as store:return ingest(store,run_id,observations)
    mcp.run(transport='stdio')
