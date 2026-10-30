# Agents

[English](README.md) | [简体中文](README_zh.md)

All Agent entrypoints are grouped here. Each Task Agent keeps its runtime
configuration, dependency declarations, installation script, and source package
inside its own project directory.

| Agent | Role | Entry |
| --- | --- | --- |
| [Self-Evolve Agent](self-evolve-agent/README.md) | Independent improvement lifecycle | `self-evolve-agent/run.sh` |
| [Drug Agent](drug-agent/README.md) | Primary computational drug-discovery Task Agent | `drugagent` |
| [Peptide Agent](peptide-agent/README.md) | Experimental peptide Task Agent | `peptide-agent` |
| [mRNA Agent](mRNA-agent/README.md) | Experimental mRNA Task Agent | `mrna-agent` |
| [Enzyme Agent](Enzyme-Agent/README.md) | Experimental enzyme Task Agent | `enzyme-agent` |

The Self-Evolve directory is a separate invocation and documentation entrypoint.
Its current Python implementation remains in the Drug Agent package, avoiding
duplicated framework code. The other three Task Agents are incomplete and have
not yet been connected to one shared evolution interface.

Third-party installers and expected asset layouts are indexed in
[THIRD_PARTY.md](../THIRD_PARTY.md).
