# Self-Evolve Agent

[English](self-evolution.md) | [简体中文](self-evolution_zh.md)

The entrypoint is [`agents/self-evolve-agent/run.sh`](../agents/self-evolve-agent/run.sh),
which runs the platform interface `drugagent self-evolve` from the Drug Agent
project. Its shared source is retained in
[`agents/drug-agent/src/drug_agent/selfevolve/`](../agents/drug-agent/src/drug_agent/selfevolve/) and its default
configuration in [`configs/self_evolve.yaml`](../agents/drug-agent/configs/self_evolve.yaml).

The loop reads locally produced request journals, groups measured failures,
proposes candidate policies or patches, and evaluates candidates independently.
Its archive retains candidate lineage and decisions. Journaling and storage
**code** are included; the original operator's journals and stores are not.
Users create their own records when running the software.

## Available source interfaces

- `doctor` describes configuration and backend availability.
- `analyze` consumes experience from the configured journal directory.
- `baseline`, `step`, and `run` perform evaluation and improvement cycles when
  their required runtime and regression checks are supplied.
- `report` explains a candidate's evaluation and adoption decision.
- `promote` requires evaluation evidence and an explicit operator flag, then
  writes an overlay. The deployment reads `DRUGAGENT_EVOLVE_OVERLAY`.

These commands can produce local state or call configured model backends. They
are not part of the standard-library PFM replay example.

## Gate 0 and the excluded development suite

The existing kernel invokes `python -m pytest -q -x tests` as its regression
gate. This publication intentionally excludes the private `tests/` tree and
development dependencies. Consequently the default full platform evolution loop
cannot pass gate 0 from this snapshot alone.

The release preserves that failure behavior. A deployment must provide a
compatible regression suite and its dependencies, or deliberately integrate its
own invariant checks. A missing suite is not a successful evaluation. The
runtime evaluation kernel, admission rules, and promotion checks remain source
code in this repository because they are part of the framework.

## Per-target campaigns

[`scripts/research/`](../agents/drug-agent/scripts/research/) retains campaign dispatch and target
preparation entry points. Historical campaign registrations, protocol locks,
qualification records, held-out partitions, and external scientific datasets
are not distributed. Fresh campaigns require explicit, compatible input and
evaluation configuration; old campaign commands are not advertised as runnable.

The [PFM example](../examples/pfm_replay/README.md) is a curated replay of
recorded evaluation aggregates. It does not supply the original research store
or the resources needed to repeat molecular generation and scoring.

## Isolation and adoption

Candidate worktrees isolate candidate files and identify the code being
evaluated. Running the worker under the same OS user does not make protected
data inaccessible. Deployment-level access restrictions are separate from the
kernel's integrity checks.

This public copy has its own Git repository, with history starting from the
reviewed source snapshot. It does not include the enclosing source repository's
history. No remote is configured in the local publication copy.
Candidate workspaces now track both the Git checkout root and the nested
`agents/drug-agent` project root. After review, use a standalone checkout with a
path without `:` and a committed baseline before using candidate worktrees.

