"""Local stdio MCP transport. Each endpoint fixes its skill; callers cannot elevate it."""
from importlib.resources import files

from mcp.server.fastmcp import FastMCP

from .contracts import Analyze, Generate, BoltzGenerate, Structure
from .calibration import AffinityRequest
from .gateway import Gateway, POLICY
from .store import Store

mcp = FastMCP("peptide-agent")


def invoke(tool, request, skill):
    store = Store()
    try:
        run_id = store.create({"mcp_tool": tool}, skill)
        result = Gateway(store, run_id, skill).call(tool, request.model_dump())
        store.update(run_id, "complete" if result.status == "ok" else result.status, result.model_dump())
        return {"run_id": run_id, **result.model_dump()}
    finally:
        store.db.close()


@mcp.tool(name="sequence_analyze")
def sequence_analyze(request: Analyze) -> dict:
    """Calculate canonical linear/head-to-tail peptide descriptors; no activity/toxicity inference."""
    return invoke("sequence.analyze", request, "analyze-peptide")


@mcp.tool(name="generate_pepmlm")
def generate_pepmlm(request: Generate) -> dict:
    """Generate target-conditioned peptides using locally installed pinned PepMLM weights."""
    return invoke("generation.pepmlm", request, "design-peptide")


@mcp.tool(name="structure_contacts")
def structure_contacts(request: Structure) -> dict:
    """Calculate interchain contacts and geometric clashes from inline PDB content."""
    return invoke("structure.contacts", request, "analyze-peptide")


@mcp.resource("peptide://policy")
def policy() -> dict:
    return POLICY


@mcp.tool(name="generate_boltzgen")
def generate_boltzgen(request: BoltzGenerate) -> dict:
    """Generate linear or head-to-tail cyclic peptides and complexes with pinned BoltzGen."""
    return invoke("generation.boltzgen", request, "design-peptide")


@mcp.tool(name="toxicity_predict")
def toxicity_predict(request: Analyze) -> dict:
    """Published ToxinPred3 ML classifier; cyclic peptides are outside the validated domain."""
    return invoke("safety.toxinpred3", request, "analyze-peptide")


@mcp.tool(name="affinity_prodigy")
def affinity_prodigy(request: Structure) -> dict:
    """PRODIGY empirical structure-based affinity; not a peptide-calibrated experimental Kd."""
    return invoke("affinity.prodigy", request, "analyze-peptide")


@mcp.tool(name="affinity_calibrated")
def affinity_calibrated(request: AffinityRequest) -> dict:
    """Predict pKd with a locally trained, held-out-calibrated peptide-target model."""
    return invoke("affinity.calibrated", request, "analyze-peptide")


@mcp.prompt()
def design_peptide() -> str:
    """Target-conditioned design procedure and evidence boundaries."""
    return files("peptide_agent").joinpath("skills/design-peptide/SKILL.md").read_text()


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
