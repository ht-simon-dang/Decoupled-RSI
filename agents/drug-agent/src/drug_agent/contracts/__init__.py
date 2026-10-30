"""Canonical, versioned contracts for the drug-agent platform.

All cross-module data crosses one of these Pydantic models. ``CONTRACT_REGISTRY``
maps a stable name -> model class and drives JSON-Schema export
(``drugagent capabilities --schemas`` and ``schemas/`` on disk).
"""

from __future__ import annotations

from .agent import AgentPlan, AgentRequest, AgentResult, PlanStep, ValidationIssue
from .base import (
    SCHEMA_VERSION,
    EvidenceRef,
    IdentifiedModel,
    Provenance,
    Quantity,
    SchemaModel,
)
from .brief import (
    Constraint,
    GenerationSpec,
    Objective,
    PhenotypeSpec,
    PocketSpec,
    ResearchBrief,
    ResearchProgram,
    TargetSpec,
)
from .capability import CapabilityProfile
from .chem import AssayRecord, Candidate, Molecule
from .enums import (
    ApplicabilityDomain,
    ApprovalStatus,
    BriefKind,
    CapabilityStatus,
    CostClass,
    DecisionOutcome,
    Determinism,
    GenerationMethod,
    ObjectiveDirection,
    Qualifier,
    RunState,
    Severity,
    TaskState,
    TrustLabel,
)
from .evidence import ArtifactRef, DecisionRecord, EvidenceItem
from .experiment import ExperimentProposal
from .perturbation import DifferentialGene, PerturbationSignature, QCSummary
from .predictions import Prediction, SafetyEndpoint, SafetyProfile
from .run import (
    ApprovalGate,
    Budget,
    RetryPolicy,
    Run,
    RunManifest,
    Task,
    TaskEvent,
)
from .scorecard import (
    EndpointScore,
    Portfolio,
    PortfolioMember,
    Scorecard,
    SensitivityReport,
    SensitivityRow,
)
from .structure import (
    InteractionFingerprint,
    Pose,
    ProteinStructure,
    SimulationResult,
)
from .synthesis import BuildingBlock, RouteStep, SynthesisPlan, SynthesisRoute

CONTRACT_REGISTRY: dict[str, type[SchemaModel]] = {
    "ResearchBrief": ResearchBrief,
    "ResearchProgram": ResearchProgram,
    "GenerationSpec": GenerationSpec,
    "PocketSpec": PocketSpec,
    "Run": Run,
    "Task": Task,
    "RunManifest": RunManifest,
    "CapabilityProfile": CapabilityProfile,
    "AgentRequest": AgentRequest,
    "AgentPlan": AgentPlan,
    "AgentResult": AgentResult,
    "EvidenceItem": EvidenceItem,
    "ArtifactRef": ArtifactRef,
    "DecisionRecord": DecisionRecord,
    "Molecule": Molecule,
    "Candidate": Candidate,
    "AssayRecord": AssayRecord,
    "Prediction": Prediction,
    "Pose": Pose,
    "ProteinStructure": ProteinStructure,
    "SimulationResult": SimulationResult,
    "SynthesisPlan": SynthesisPlan,
    "PerturbationSignature": PerturbationSignature,
    "SafetyProfile": SafetyProfile,
    "Scorecard": Scorecard,
    "Portfolio": Portfolio,
    "SensitivityReport": SensitivityReport,
    "ExperimentProposal": ExperimentProposal,
}

__all__ = [
    "SCHEMA_VERSION",
    "CONTRACT_REGISTRY",
    "SchemaModel",
    "IdentifiedModel",
    "Provenance",
    "Quantity",
    "EvidenceRef",
    # enums
    "TrustLabel",
    "TaskState",
    "RunState",
    "CostClass",
    "Determinism",
    "Qualifier",
    "ApplicabilityDomain",
    "DecisionOutcome",
    "ApprovalStatus",
    "Severity",
    "BriefKind",
    "GenerationMethod",
    "ObjectiveDirection",
    "CapabilityStatus",
    # brief
    "ResearchBrief",
    "ResearchProgram",
    "TargetSpec",
    "PhenotypeSpec",
    "Objective",
    "Constraint",
    "GenerationSpec",
    "PocketSpec",
    # run
    "Run",
    "Task",
    "TaskEvent",
    "RunManifest",
    "ApprovalGate",
    "Budget",
    "RetryPolicy",
    # capability
    "CapabilityProfile",
    # agent
    "AgentRequest",
    "AgentPlan",
    "AgentResult",
    "PlanStep",
    "ValidationIssue",
    # evidence
    "EvidenceItem",
    "ArtifactRef",
    "DecisionRecord",
    # chem
    "Molecule",
    "Candidate",
    "AssayRecord",
    # predictions
    "Prediction",
    "SafetyProfile",
    "SafetyEndpoint",
    # structure
    "Pose",
    "ProteinStructure",
    "SimulationResult",
    "InteractionFingerprint",
    # synthesis
    "SynthesisPlan",
    "SynthesisRoute",
    "RouteStep",
    "BuildingBlock",
    # perturbation
    "PerturbationSignature",
    "DifferentialGene",
    "QCSummary",
    # scorecard
    "Scorecard",
    "EndpointScore",
    "Portfolio",
    "PortfolioMember",
    "SensitivityReport",
    "SensitivityRow",
    # experiment
    "ExperimentProposal",
]
