"""Chemical agent — molecule generation + virtual screening.

Consumes the master agent's ``ChemicalRequirements`` (the transcribed generation
prompt), runs the real ``drug_agent`` pipeline, and summarizes the candidate-level
output. All chemistry is tool-computed (RDKit standardization / descriptors /
alerts + the scikit-learn activity model); nothing here is produced by the LLM.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ...llm.bridge import BackendRun, run_pipeline
from ...llm.config import Config
from ...llm.schemas import ChemicalRequirements
from . import ReasoningAgent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...observability.journal import RunJournal


class ChemicalResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_id: str
    state: str
    n_candidates: int = 0
    n_passed_filters: int = 0
    origin_counts: dict[str, int] = Field(default_factory=dict)
    pool_max_pIC50: float | None = None
    pool_median_pIC50: float | None = None
    n_with_alerts: int = 0
    seeds_used: list[str] = Field(default_factory=list)
    seeds_dropped: list[str] = Field(default_factory=list)
    # De-novo generation, when the run enabled it.
    generator_backend: str | None = None
    #: Retained so reports over journals written before the surrogate's removal
    #: still deserialize; nothing sets it True any more.
    generator_is_surrogate: bool = False
    n_generated: int = 0
    generation_validity: float | None = None
    generation_uniqueness: float | None = None
    generation_warnings: list[str] = Field(default_factory=list)
    #: Why the pipeline stopped, when ``state`` is not SUCCEEDED — the failing
    #: task and the reason it gave. Carried up from ``BackendRun.error`` so the
    #: critique and refinement steps diagnose the actual cause instead of
    #: inferring one from an empty result.
    error: str | None = None

    def summary_line(self) -> str:
        if self.error:
            # The reason leads: a cycle that produced no numbers is a cycle whose
            # only useful content IS the reason, and burying it after four nulls
            # invites the critic to theorise about the nulls instead.
            return (
                f"Pipeline {self.state}: {self.error} "
                f"({self.n_candidates} candidates were built before the run stopped; "
                f"none were scored, so no metric has a value this cycle.)"
            )
        line = (
            f"{self.n_candidates} candidates generated & screened "
            f"({self.n_passed_filters} passed hard filters); "
            f"pool best pIC50 {self.pool_max_pIC50}, median {self.pool_median_pIC50}."
        )
        if self.generator_backend:
            line += (
                f" {self.n_generated} de-novo structures from {self.generator_backend}"
                f" (validity {self.generation_validity}, uniqueness {self.generation_uniqueness})."
            )
        return line


def _count_generated(origin_counts: dict[str, int], gen: dict[str, Any] | None) -> int:
    """How many candidates the de-novo arm actually produced.

    Counted against the arm's OWN origin tag, not the literal string "pfm". Every
    catalogue generator tags its molecules with its engine name, and PocketXMol
    tags them ``<engine>.<task>`` — so a successful PocketXMol run used to report
    ``n_generated=0``. That number goes into the summary the critique and refine
    phases read, which means a working non-PFM generator was indistinguishable
    from one that produced nothing, and the master agent could never learn that a
    catalogue model had worked.

    ``pfm_mock`` is kept for journals written before the surrogate's removal
    (ADR-0010); nothing produces it now.
    """
    tag = str((gen or {}).get("origin_tag") or "pfm")
    total = origin_counts.get("pfm_mock", 0)
    for origin, n in origin_counts.items():
        if origin == tag or origin.startswith(tag + "."):
            total += n
    return total


def _pool_pIC50(scorecards: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for sc in scorecards:
        for e in sc.get("endpoints", []):
            if e.get("endpoint") == "pIC50" and not e.get("missing") and e.get("raw_value") is not None:
                out.append(float(e["raw_value"]))
    return out


class ChemicalAgent(ReasoningAgent):
    """Executes one set of chemical requirements, then summarizes the outcome.

    Despite living in the LLM plane it calls no model: it is the boundary
    where a decision becomes computation. Everything it reports was computed
    by the deterministic pipeline it drives.
    """

    agent_id = "agent.llm.chemical"
    llm_backed = False
    actions = {"run": "Run generation + virtual screening for one requirements object."}

    @staticmethod
    def summarize(run: BackendRun) -> ChemicalResult:
        """Reduce a raw backend run to the candidate-level result."""
        origin_counts: dict[str, int] = {}
        n_alerts = 0
        for c in run.candidates:
            o = str(c.get("origin"))
            origin_counts[o] = origin_counts.get(o, 0) + 1
            if c.get("filter_failures") and any("alert" in str(f).lower() for f in c["filter_failures"]):
                n_alerts += 1
        pool = _pool_pIC50(run.scorecards)
        n_passed = sum(1 for sc in run.scorecards if sc.get("hard_constraints_passed"))
        gen = run.generation[0] if run.generation else None
        gen_metrics = (gen or {}).get("metrics", {})
        return ChemicalResult(
            run_id=run.run_id,
            state=run.state,
            n_candidates=len(run.candidates),
            n_passed_filters=n_passed,
            origin_counts=origin_counts,
            pool_max_pIC50=round(max(pool), 4) if pool else None,
            pool_median_pIC50=round(statistics.median(pool), 4) if pool else None,
            n_with_alerts=n_alerts,
            seeds_used=run.valid_seeds,
            seeds_dropped=run.dropped_seeds,
            generator_backend=(gen or {}).get("backend"),
            generator_is_surrogate=bool((gen or {}).get("is_surrogate")),
            n_generated=_count_generated(origin_counts, gen),
            generation_validity=gen_metrics.get("generation_validity"),
            generation_uniqueness=gen_metrics.get("generation_uniqueness"),
            generation_warnings=list((gen or {}).get("warnings", [])) + list((gen or {}).get("errors", [])),
            error=run.error,
        )

    async def run(
        self,
        req: ChemicalRequirements,
        cfg: Config,
        *,
        journal: RunJournal | None = None,
        cycle: int = 0,
    ) -> tuple[BackendRun, ChemicalResult]:
        """Run generation + screening off the event loop; return (raw run, summary).

        The pipeline is synchronous and CPU/GPU-bound, so it runs in a worker thread
        and the event stream stays live while it works. ``journal`` and ``cycle``
        travel with it so every tool call this cycle makes is metered under the
        right scope.
        """
        run = await asyncio.to_thread(run_pipeline, req, cfg, journal=journal, cycle=cycle)
        return run, self.summarize(run)
