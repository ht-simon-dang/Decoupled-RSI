"""The specialist plane — agents that each own one scientific domain.

  * :class:`~drug_agent.agents.specialists.target_agent.TargetEvidenceAgent`
  * :class:`~drug_agent.agents.specialists.pocket_agent.PocketAgent`
  * :class:`~drug_agent.agents.specialists.design_agent.DesignAgent`
  * :class:`~drug_agent.agents.specialists.molecule_agent.MoleculeAgent`
  * :class:`~drug_agent.agents.specialists.synthesis_agent.SynthesisAgent`
  * :class:`~drug_agent.agents.specialists.structure_agent.StructureDynamicsAgent`
  * :class:`~drug_agent.agents.specialists.perturbation_agent.PerturbationAgent`
  * :class:`~drug_agent.agents.specialists.safety_agent.SafetyAgent`

They read in the order a program actually runs: evidence a target, find its site,
design into it, screen what comes back, and assess the survivors. The first three
are the newest and they close the pipeline's oldest gap — nothing here could
establish a target, and nothing could produce the prepared pocket file every
pocket-conditioned generator refuses to run without.

One file per agent, so the folder listing *is* the plane's contents. This module
holds only what they share, in two layers: :class:`SpecialistAgent` for the plane
itself, and :class:`CandidateAssessmentAgent` for the four that assess a
candidate batch one molecule at a time.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..base import AgentKind, ScientificAgent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...contracts.agent import AgentRequest, AgentResult
    from ...contracts.predictions import Prediction
    from ...orchestration.context import RunContext


def unavailable_note(tool: Any) -> str:
    """One line saying that *tool* produced nothing here, and why.

    Every specialist now carries optional tools that a given host may not have,
    and each one has to say the same thing when it is absent. Saying it in one
    place keeps the sentence — and the fact that the reason comes off the
    capability card rather than being written out again — identical everywhere.
    """
    profile = tool.profile
    return f"{profile.capability_id} produced NOTHING: {profile.unavailable_reason or 'unavailable'}"


class SpecialistAgent(ScientificAgent):
    """An agent that owns one scientific domain and does the pipeline work.

    Specialists are where chemistry actually happens. They read the run's
    candidate batch, drive a tool adapter, and write back domain objects plus
    the objective-aligned :class:`Prediction` records the ranking plane consumes.
    """

    kind = AgentKind.SPECIALIST

    def candidates(self, ctx: RunContext) -> list[dict]:
        """This run's candidate batch, in insertion order."""
        return ctx.store.list_objects("candidate", run_id=ctx.run.id)


@dataclass(slots=True)
class Assessment:
    """What assessing one candidate produced.

    ``objects`` are domain records to persist as ``(store_type, obj)`` pairs —
    a SynthesisPlan, a Pose, a SafetyProfile — in the order they should land.
    ``predictions`` are the objective-aligned records derived from them. Keeping
    the two lists separate is what lets the base class store every specialist's
    output the same way while each still writes its own domain object.
    """

    objects: list[tuple[str, Any]] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)


class SkipBatch(Exception):
    """Raised from :meth:`CandidateAssessmentAgent.prepare` to skip the batch.

    Some specialists need a resource before they can assess anything — a
    single-cell dataset, a prepared receptor. When it is missing the honest
    outcome is a ``partial`` result that says so, not a failed run and not a
    batch of silently absent predictions.
    """

    def __init__(self, warning: str, explanation: str) -> None:
        super().__init__(warning)
        self.warning = warning
        self.explanation = explanation


class CandidateAssessmentAgent(SpecialistAgent):
    """A specialist that assesses the candidate batch one molecule at a time.

    Four of the five specialists have the same execution shape: optionally
    prepare a shared resource, then for every candidate call one tool, persist
    what it returned, and emit predictions from it. That loop — including the
    bounded fan-out through :meth:`RunContext.map_candidates` and the uniform
    ``prediction`` writes — lives here exactly once. A subclass supplies only the
    part that is actually its own: :meth:`assess_candidate`.

    :class:`~drug_agent.agents.specialists.molecule_agent.MoleculeAgent` extends
    :class:`SpecialistAgent` directly instead, because sourcing and standardizing
    are not per-candidate assessments and pretending otherwise would cost more
    than it saved.
    """

    #: Thread-name prefix for the fan-out, and the label in run traces.
    batch_label: str = "batch"
    #: Payload/metric key for how many candidates were processed.
    count_key: str = "n_assessed"

    def prepare(self, request: AgentRequest, ctx: RunContext) -> Any:
        """Load anything the whole batch shares. Raise :class:`SkipBatch` to skip."""
        return None

    @abstractmethod
    def assess_candidate(self, candidate: dict, prep: Any, ctx: RunContext) -> Assessment:
        """Assess one candidate. The only method a specialist must write."""

    def batch_metrics(self, prep: Any, assessments: list[Assessment]) -> dict[str, float]:
        """Metrics beyond the candidate count."""
        return {}

    def batch_warnings(self, prep: Any) -> list[str]:
        """Run warnings this batch should carry (e.g. a mocked backend)."""
        return []

    @abstractmethod
    def explain_batch(self, n: int, prep: Any) -> str:
        """One human-readable sentence describing what the batch did."""

    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        try:
            prep = self.prepare(request, ctx)
        except SkipBatch as skip:
            return self._result(
                request, status="partial", warnings=[skip.warning], explanation=skip.explanation
            )

        def work(candidate: dict) -> Assessment:
            assessment = self.assess_candidate(candidate, prep, ctx)
            for store_type, obj in assessment.objects:
                ctx.store.put_object(store_type, obj, run_id=ctx.run.id)
            for prediction in assessment.predictions:
                ctx.store.put_object("prediction", prediction, run_id=ctx.run.id)
            return assessment

        assessments = ctx.map_candidates(work, self.candidates(ctx), label=self.batch_label)
        n = len(assessments)
        return self._result(
            request,
            payload={self.count_key: n},
            metrics={self.count_key: float(n), **self.batch_metrics(prep, assessments)},
            warnings=self.batch_warnings(prep),
            explanation=self.explain_batch(n, prep),
        )


from .design_agent import DesignAgent  # noqa: E402
from .molecule_agent import MoleculeAgent  # noqa: E402
from .perturbation_agent import PerturbationAgent  # noqa: E402
from .pocket_agent import PocketAgent  # noqa: E402
from .safety_agent import SafetyAgent  # noqa: E402
from .structure_agent import StructureDynamicsAgent  # noqa: E402
from .synthesis_agent import SynthesisAgent  # noqa: E402
from .target_agent import TargetEvidenceAgent  # noqa: E402

__all__ = [
    "Assessment",
    "CandidateAssessmentAgent",
    "DesignAgent",
    "MoleculeAgent",
    "PerturbationAgent",
    "PocketAgent",
    "SafetyAgent",
    "SkipBatch",
    "SpecialistAgent",
    "StructureDynamicsAgent",
    "SynthesisAgent",
    "TargetEvidenceAgent",
    "unavailable_note",
]
