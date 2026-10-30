# Task Agents

[English](task-agents.md) | [简体中文](task-agents_zh.md)

The public source snapshot includes four Task Agents. Drug Agent is the primary
implementation. Peptide, mRNA, and Enzyme Agent are experimental, incomplete,
and need further evolution and validation. Their presence does not establish a
common cross-domain evolution interface or comparative performance.

## Drug Agent

The project lives in `agents/drug-agent`. Its Python package is `drug_agent`,
with the `drugagent` command. From the repository root:

```bash
./install.sh core
conda activate drug-agent
make doctor
make demo
```

The fixtures are synthetic demonstration inputs. Optional backends need their
own sources, model weights, environments, and sometimes data preparation. An
unavailable tool remains unavailable rather than substituting fabricated
scientific output. Inspect the capability report for the local installation.

The root Makefile enters the Drug Agent project before dispatching commands.
`make demo` runs the deterministic workflow. `make serve` exposes the local web
console. The LLM console additionally requires its configured model backend.

The runtime records new execution evidence locally. The original project's
request records and trained assets are not part of the publication.

## Experimental agents

Each package has separate dependency constraints and installation instructions:

| Agent | Guide | Source |
| --- | --- | --- |
| Peptide | [README](../agents/peptide-agent/README.md) | [`peptide_agent`](../agents/peptide-agent/src/peptide_agent/) |
| mRNA | [README](../agents/mRNA-agent/README.md) | [`mrna_agent`](../agents/mRNA-agent/src/mrna_agent/) |
| Enzyme | [README](../agents/Enzyme-Agent/README.md) | [`enzyme_agent`](../agents/Enzyme-Agent/src/enzyme_agent/) |

These projects live as siblings under `agents/`. Keep them together when using the default shared peptide model
locations. Configure writable data directories using the documented environment
variables. Backend readiness may require local evidence and assets that are not
distributed here; installing a dependency alone does not establish that a
scientific capability is validated.

## Adapting the architecture

A new Task Agent needs execution feedback, explicit editable surfaces,
repeatable evaluation inputs, and an adoption mechanism. The current code gives
reference implementations of these responsibilities. Extracting a reusable
domain-independent evolution package and connecting all four applications is
future work.

