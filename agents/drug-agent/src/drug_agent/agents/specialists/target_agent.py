"""De-novo target — the layer that comes before any molecule.

Every other specialist answers a question about a *compound*: will it bind, can it
be made, is it safe. This one answers the question that comes first — **should
this protein be drugged at all, and what is the evidence?** — and it produces
evidence, never molecules.

Two sources, deliberately not one:

``target.evidence.opentargets``
    Open Targets Platform: disease associations with their per-datatype
    breakdown, per-modality tractability, curated safety liabilities, and
    fine-mapped genetics. What the target is implicated in, and by what kind of
    evidence.
``target.essentiality.depmap``
    DepMap CRISPR knockout: how hard the gene's loss hits the cell lines that were
    screened, against the release's own essential and nonessential controls.
    Whether removing the protein does anything.

A target can score well on the first and flat on the second — sEH/EPHX2 does
exactly that — so neither is reported in place of the other and neither one's
absence is covered by the other. Each arm that cannot answer here says so and
contributes nothing.

**Both read snapshots, and neither opens a socket.** The fetch is a setup step an
operator runs once (``make target-evidence``); the adapters read what it wrote and
report UNAVAILABLE naming the missing file and the fetching command when it is
absent. What this agent must never do is fill the gap: the old
``OpenTargetsLikeConnector`` answered this question from
``fixtures/data/targets.json``, three hand-written associations whose ``source``
field says ``synthetic_demo``, and nothing here reads it, extends it or falls back
to it. A target this deployment has not fetched gets a ``partial`` result naming
the command, which is the whole of ADR-0011 applied to evidence rather than to a
number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest, AgentResult, ValidationIssue
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism
from ...contracts.evidence import EvidenceItem
from ...tools.base import ToolUnavailable
from ...tools.target.depmap_tool import DepMapEssentialityAdapter
from ...tools.target.opentargets_tool import OpenTargetsEvidenceAdapter
from . import SpecialistAgent, unavailable_note

if TYPE_CHECKING:
    from ...contracts.brief import ResearchBrief
    from ...orchestration.context import RunContext

#: How many association/genetics rows one dossier carries. The snapshot holds
#: every row upstream returned — EPHX2 alone has 620 associated diseases — and a
#: dossier is a briefing, not a database dump.
DEFAULT_EVIDENCE_LIMIT = 25


class TargetEvidenceAgent(SpecialistAgent):
    """Establish and evidence a target before any chemistry starts."""

    agent_id = "agent.target"
    version = "1.0.0"
    actions = {
        "assemble_target_evidence": (
            "Association, tractability, safety, genetics and essentiality for one target."
        )
    }

    def __init__(self, *, settings: Any = None) -> None:
        self.settings = settings
        self.platform = OpenTargetsEvidenceAdapter(settings=settings)
        self.essentiality = DepMapEssentialityAdapter(settings=settings)

    def describe_capability(self) -> CapabilityProfile:
        # AVAILABLE while at least one source can answer, and the backend string
        # names which. A dossier from one source is a smaller dossier, not a
        # fabricated one; a dossier from neither is nothing at all.
        available = self.platform.available() or self.essentiality.available()
        reasons = [
            t.profile.unavailable_reason or "unavailable"
            for t in (self.platform, self.essentiality)
            if not t.available()
        ]
        return CapabilityProfile(
            capability_id=self.agent_id,
            version=self.version,
            title="Target evidence",
            description="Assemble association, tractability, safety, genetics and essentiality evidence.",
            input_schema_ref="ResearchBrief",
            output_schema_ref="EvidenceItem",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE if available else CapabilityStatus.UNAVAILABLE,
            backend=f"evidence={self.platform.backend},essentiality={self.essentiality.backend}",
            unavailable_reason=None if available else "; ".join(reasons),
            known_failure_modes=[
                "no snapshot fetched for this target on this deployment",
                "the gene is outside the cached DepMap subset",
            ],
            validation_checks=[
                "every item carries the release it came from",
                "no evidence is produced for a target with no snapshot",
            ],
        )

    # --- execution --------------------------------------------------------
    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        if request.action != "assemble_target_evidence":
            return self._result(
                request,
                status="error",
                validation_issues=[ValidationIssue(code="unknown_action", message=request.action)],
            )
        symbol = self._symbol(request)
        if not symbol:
            note = "no target evidence: the request names no gene symbol and the brief has no target"
            return self._result(request, status="partial", warnings=[note], explanation=note)

        limit = int(request.inputs.get("evidence_limit") or DEFAULT_EVIDENCE_LIMIT)
        warnings: list[str] = []
        items: list[EvidenceItem] = []
        cards: dict[str, Any] = {}

        if not self.platform.available():
            warnings.append(unavailable_note(self.platform))
        else:
            try:
                items.extend(self.platform.evidence(symbol, limit=limit))
                cards["opentargets"] = self.platform.data_card(symbol)
            except ToolUnavailable as exc:
                warnings.append(f"{exc.capability_id} produced NOTHING: {exc.reason}")

        if not self.essentiality.available():
            warnings.append(unavailable_note(self.essentiality))
        else:
            try:
                items.append(self.essentiality.essentiality(symbol))
                cards["depmap"] = self.essentiality.data_card()
            except ToolUnavailable as exc:
                warnings.append(f"{exc.capability_id} produced NOTHING: {exc.reason}")

        by_kind: dict[str, int] = {}
        for item in items:
            item.run_id = ctx.run.id
            item.program_id = ctx.run.program_id
            item.subject = item.subject or symbol
            ctx.memory.add_evidence(item)
            by_kind[item.kind] = by_kind.get(item.kind, 0) + 1

        if cards:
            # The dossier's provenance, kept as one object so a reader can check
            # which release every number above came from without re-deriving it.
            ctx.store.put_object(
                "target_dossier",
                {
                    "id": f"target_{symbol}",
                    "run_id": ctx.run.id,
                    "target": symbol,
                    "n_evidence": len(items),
                    "by_kind": by_kind,
                    "data_cards": cards,
                },
                run_id=ctx.run.id,
            )

        status = "ok" if items else "partial"
        breakdown = ", ".join(f"{n} {kind}" for kind, n in sorted(by_kind.items())) or "none"
        return self._result(
            request,
            status=status,
            payload={"target": symbol, "n_evidence": len(items), "by_kind": by_kind},
            metrics={"n_target_evidence": float(len(items)), "n_evidence_kinds": float(len(by_kind))},
            warnings=warnings,
            explanation=f"Assembled {len(items)} evidence item(s) for {symbol} ({breakdown}).",
        )

    @staticmethod
    def _symbol(request: AgentRequest) -> str:
        """The gene this dossier is about: the caller's, else the brief's target."""
        from ...contracts.brief import ResearchBrief  # local import: avoids a cycle

        explicit = request.inputs.get("target_gene") or request.inputs.get("gene_symbol")
        if explicit:
            return str(explicit).strip()
        raw = request.inputs.get("brief")
        if not raw:
            return ""
        brief: ResearchBrief = ResearchBrief.model_validate(raw)
        if brief.target is None:
            return ""
        return str(brief.target.gene_symbol or brief.target.uniprot or brief.target.name or "").strip()

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        """Every item must name the release it came from, and an absent source must have produced none.

        Read back from the memory store rather than from the result, so a bug in
        :meth:`execute` cannot also silence the check on it. Items are matched by
        ``provenance.method``, which is the capability id of the adapter that made
        them, so this grades this agent's own output and not the literature
        agent's.
        """
        by_tool = {t.capability_id: t for t in (self.platform, self.essentiality)}
        issues: list[ValidationIssue] = []
        for item in ctx.memory.list_evidence(run_id=ctx.run.id):
            provenance = item.get("provenance") or {}
            tool = by_tool.get(str(provenance.get("method", "")))
            if tool is None:
                continue
            if not tool.available():
                issues.append(
                    ValidationIssue(
                        code="unavailable_capability_emitted_output",
                        severity="error",
                        message=f"{tool.capability_id} is UNAVAILABLE here yet {item.get('id')} claims it",
                    )
                )
            if not provenance.get("source") or not provenance.get("snapshot_id"):
                issues.append(
                    ValidationIssue(
                        code="evidence_without_provenance",
                        message=f"{item.get('id')} names no source release or snapshot",
                    )
                )
        return issues
