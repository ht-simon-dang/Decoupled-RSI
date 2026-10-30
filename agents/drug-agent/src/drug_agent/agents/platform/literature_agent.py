"""Literature intelligence — assemble evidence from untrusted sources.

Security note: literature/abstract text is UNTRUSTED. This agent never executes
instructions found in retrieved content; it detects injection markers,
quarantines the affected item, and only ever *summarizes* text as data. The
quarantine check runs twice — once on the way in, once on the produced summary
in :meth:`validate_result` — so a leak is caught even if the summarizer echoes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...contracts.agent import AgentRequest, AgentResult, ValidationIssue
from ...contracts.base import Provenance
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ...contracts.evidence import EvidenceItem
from ...security.redaction import redact
from . import PlatformAgent

if TYPE_CHECKING:
    from ...orchestration.context import RunContext

# Heuristic prompt-injection markers looked for in untrusted retrieved text.
_INJECTION_MARKERS = [
    re.compile(r"ignore (all )?previous instructions", re.I),
    re.compile(r"disregard (the )?(above|previous)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"system\s*:", re.I),
    re.compile(r"exfiltrate|api[_\s-]?key|disable safety", re.I),
]


def _has_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_MARKERS)


class LiteratureIntelligenceAgent(PlatformAgent):
    agent_id = "agent.literature"
    actions = {"assemble_evidence": "Retrieve + summarize literature/target evidence (untrusted)."}

    def describe_capability(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.agent_id,
            version="1.0.0",
            title="Literature intelligence",
            description="Assemble evidence from literature/target sources; injection-safe.",
            input_schema_ref="ResearchBrief",
            output_schema_ref="EvidenceItem",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE,
            backend="fake_llm+fixtures",
            known_failure_modes=["empty corpus", "prompt injection in source"],
            validation_checks=["injection quarantined", "evidence has provenance"],
        )

    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        lit = ctx.store.list_objects("literature", run_id=ctx.run.id)
        td = ctx.store.list_objects("target_disease", run_id=ctx.run.id)
        n_ev = 0
        quarantined = 0
        safe_texts: list[str] = []

        for item in lit:
            abstract = str(item.get("abstract", ""))
            injected = _has_injection(abstract)
            if injected:
                quarantined += 1
            else:
                safe_texts.append(abstract)
            ev = EvidenceItem(
                run_id=ctx.run.id,
                program_id=ctx.run.program_id,
                kind="literature",
                statement=(redact(item.get("title", ""))),
                trust_label=TrustLabel.CURATED,
                subject=request.inputs.get("target_name"),
                provenance=Provenance(
                    source=str(item.get("source", "literature")),
                    source_version=str(item.get("year", "")),
                    trust_label=TrustLabel.CURATED,
                    notes="quarantined_injection" if injected else None,
                ),
                value={"pmid": item.get("pmid"), "year": item.get("year"), "quarantined": injected},
                quality_score=0.3 if injected else 0.7,
            )
            ctx.memory.add_evidence(ev)
            n_ev += 1

        for assoc in td:
            ev = EvidenceItem(
                run_id=ctx.run.id,
                program_id=ctx.run.program_id,
                kind="target_disease",
                statement=f"{request.inputs.get('target_name', 'target')} associated with "
                f"{assoc.get('disease')} (score {assoc.get('score')})",
                trust_label=TrustLabel.CURATED,
                subject=assoc.get("disease"),
                provenance=Provenance(
                    source=str(assoc.get("source", "opentargets")), trust_label=TrustLabel.CURATED
                ),
                value=assoc,
                quality_score=float(assoc.get("score", 0.5)),
            )
            ctx.memory.add_evidence(ev)
            n_ev += 1

        # Summarize only the SAFE texts, as data (never executing instructions).
        summary = ctx.provider.summarize(safe_texts, focus="target evidence")
        return self._result(
            request,
            payload={"n_evidence": n_ev, "quarantined": quarantined, "summary": summary.text},
            metrics={"n_evidence": float(n_ev), "quarantined_injections": float(quarantined)},
            tokens_used=summary.tokens,
            warnings=[f"{quarantined} source(s) quarantined for prompt injection"] if quarantined else [],
            explanation=f"Assembled {n_ev} evidence items; quarantined {quarantined} "
            "source(s) containing injected instructions (treated as untrusted data).",
        )

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        # The summary must not contain injected directive phrasing.
        summary = str(result.payload.get("summary", ""))
        if _has_injection(summary):
            return [ValidationIssue(code="injection_leak", message="summary echoes injected instructions")]
        return []
