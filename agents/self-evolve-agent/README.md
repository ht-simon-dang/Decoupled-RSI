# Self-Evolve Agent

[English](README.md) | [简体中文](README_zh.md)

This directory gives the independent improvement lifecycle its own entrypoint
alongside the Task Agents. Its implementation is currently
[`drug_agent.selfevolve`](../drug-agent/src/drug_agent/selfevolve/), shared with
the Drug Agent package rather than copied into a second package.

```bash
./install.sh --help
./install.sh
./run.sh --help
./run.sh doctor
```

`install.sh` installs the Drug Agent runtime environment. It does not install
every scientific backend or download all model weights. `run.sh` enters the
Drug Agent project and forwards commands to `drugagent self-evolve` in the
configured conda environment.

The loop reads execution experience, proposes candidate changes, evaluates
them independently, and records adoption decisions. It can be invoked
separately from task execution. Automatic cross-agent integration with the
experimental peptide, mRNA, and enzyme packages is still future work.

The default full loop requires the private regression suite at gate 0. That
suite is not published, and this entrypoint does not bypass the gate. Read the
[evolution guide](../../docs/self-evolution.md) before launching a cycle.

Dependencies are the [Drug Agent runtime](../drug-agent/environment.yml), the
configured proposal/reasoning backends in
[`self_evolve.yaml`](../drug-agent/configs/self_evolve.yaml), and the scientific
backends needed for the chosen evaluation. See
[third-party installation](../../THIRD_PARTY.md).
