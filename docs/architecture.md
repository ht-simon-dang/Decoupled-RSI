# Architecture

[English](architecture.md) | [简体中文](architecture_zh.md)

The Task Agent and Self-Evolve Agent have separate responsibilities and entry
points. Task execution produces evidence. Evolution consumes that evidence,
proposes a candidate, evaluates it, and records whether it is eligible for
adoption. The runtime reads an explicitly selected policy version.

| Responsibility | Implementation |
| --- | --- |
| Task planning and execution | [`agents/`](../agents/drug-agent/src/drug_agent/agents/), [`orchestration/`](../agents/drug-agent/src/drug_agent/orchestration/) |
| Scientific tool interfaces | [`tools/`](../agents/drug-agent/src/drug_agent/tools/) and first-party drivers in [`scripts/external/`](../agents/drug-agent/scripts/external/) |
| Execution experience | [`selfevolve/experience/`](../agents/drug-agent/src/drug_agent/selfevolve/experience/) |
| Failure analysis and proposal | [`analysis/`](../agents/drug-agent/src/drug_agent/selfevolve/analysis/), [`hypotheses/`](../agents/drug-agent/src/drug_agent/selfevolve/hypotheses/) |
| Evolution lifecycle and budgets | [`controller/loop.py`](../agents/drug-agent/src/drug_agent/selfevolve/controller/loop.py), [`config.py`](../agents/drug-agent/src/drug_agent/selfevolve/config.py) |
| Candidate isolation and lineage | [`candidates/`](../agents/drug-agent/src/drug_agent/selfevolve/candidates/), [`archive/`](../agents/drug-agent/src/drug_agent/selfevolve/archive/) |
| Independent evaluation | [`evaluation/`](../agents/drug-agent/src/drug_agent/selfevolve/evaluation/) |
| Adoption decision | [`promotion/`](../agents/drug-agent/src/drug_agent/selfevolve/promotion/) and the `promote` CLI |
| Runtime update interface | [`llm/policy.py`](../agents/drug-agent/src/drug_agent/llm/policy.py) |

## What is decoupled

The evolver can be invoked separately from serving a task. It has its own
configuration, experience, candidate archive, model budgets, and evaluation
records. A proposed update does not immediately replace the Task Agent's policy.
The operator-facing promotion command writes an overlay only after checking
evaluation evidence; the deployment then selects that overlay explicitly.

This lifecycle separation is the architecture proposed by this release. It does
not imply that a background scheduler, hot deployment, or universal cross-agent
update interface has already been implemented.

## Two evolution implementations

The platform loop changes the drug agent's behavior through candidate overlays
and code changes. The separate [`selfevolve/target/`](../agents/drug-agent/src/drug_agent/selfevolve/target/)
implementation organizes per-target scientific campaigns. The PFM example comes
from the latter. These share design ideas but are not interchangeable evidence
for the same experiment.

The peptide, mRNA, and enzyme packages have their own task and evolution-related
workflows. They are included as experimental applications; connecting them to a
shared Self-Evolve Agent requires further integration.

## Agent projects and candidate workspaces

All Agent entrypoints live under [`agents/`](../agents/README.md). Drug Agent
keeps its own `src`, runtime configuration, scripts, and package metadata inside
`agents/drug-agent`. The other Task Agents are siblings, preserving their shared
model-resource discovery.

Candidate workspaces distinguish the Git checkout root from the Drug Agent
project root. Imports and execution use the candidate project; patches and
protected paths are normalized against the project boundary. This allows the
shared repository layout without evaluating the parent's installed code or
letting a repository prefix hide a protected-file edit.

The Self-Evolve Agent has its own invocation directory. Its Python implementation
is still shared through the Drug Agent package; it has not been duplicated or
extracted into a standalone distribution.

## Relationship to Harness²

[Harness²](https://github.com/google-research/harness2) already distinguishes an
execution harness from an improvement harness. Its paper describes task-level
refinement and cross-task experience reuse. This project emphasizes a separately
invoked Self-Evolve Agent and explicit evaluation and adoption boundaries.
Their relative efficiency has not been established by matched ablations here.

