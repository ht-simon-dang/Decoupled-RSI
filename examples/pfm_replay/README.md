# PFM recorded-evaluation replay

[English](README.md) | [简体中文](README_zh.md)

This small example shows a recorded self-evolution intervention in the drug Task
Agent: an evolution worker changed PFM's decoding procedure, and the parent
evaluator assessed the resulting submissions independently of the worker's own
claims. Two paired seeds are included. All measurements are from the development
partition and remain exploratory.

The replay uses Python's standard library. It recounts qualifying scaffolds from
stored evaluation decisions; it does not call an LLM, download models, regenerate
molecules, or recompute chemical or docking scores. It writes only to stdout by
default and creates no run history.

From the repository root:

```bash
python examples/pfm_replay/replay.py
python examples/pfm_replay/replay.py --json
```

The script resolves its two JSON inputs relative to its own location, so invoking
it by an absolute path also works from any other working directory.

Expected counts:

| Seed | Stock decoder | Shaped decoder | Submitted per arm | Difference |
| --- | ---: | ---: | ---: | ---: |
| 20261001 | c04: 77 | c10: 89 | 150 | +12 |
| 20260929 | c06: 87 | c13: 88 | 150 | +1 |

The metric is `qualifying_scaffolds`: the number of distinct scaffold identities
among submitted rows passing all seven recorded gates. It measures computational
yield under the recorded qualification rules, not biological activity.

## What changed

`candidate_change.json` describes the executed decoder modification: a size- and
atom-type-biased ligand prior, with rejection and resampling of partial structures.
It includes the effective parameters recorded by both shaped-decoder runs.

The four records declare the same PFM checkpoint hash, zero optimizer steps, a
150-molecule submission budget and an 8,000-draw ceiling. Generation stops before
that ceiling when its stopping rule is satisfied. A changed decoder produces a
different molecule pool even at a paired seed. The actual third-party sampler and
checkpoint bytes are not included.

## What is included

- `recorded_evaluations.json`: four curated records, containing 600 rows of gate
  booleans and scaffold digests, plus selected sampling counts and evaluation
  metadata. Each record includes hashes of its original sampling summary and
  independent evaluator record.
- `candidate_change.json`: the recorded decoder parameters, paired-cycle mapping
  and common artifact identities.
- `replay.py`: recounts rows for which every gate is true, deduplicates their
  scaffold digests, checks agreement with the recorded evaluator counts, and
  prints the paired comparison.

Scaffold strings have been replaced with their SHA-256 digests. Molecule strings,
molecule identifiers, numerical endpoint values, original conversations, runtime
paths, full logs and model weights are omitted. The gate names are `geometry`,
`identity`, `novelty`, `properties`, `safety`, `structure_score` and
`synthetic_accessibility`.

The source metadata identifies the selected fields and hashes of the original
records, which are not distributed here. The curated files have these SHA-256
digests:

```text
recorded_evaluations.json  ce408284d447e6dcad27b9cf0def2bd5e7d090c0090b3601747e0e09e8741dc8
candidate_change.json     f151feeebec3ac33d1cf4402ede2dd4dd13cac4a5c0cbbc01de7f5c50c3ed1eb
```

## Interpretation

This is an aggregation replay of recorded decisions. It cannot independently
verify the underlying chemistry, checkpoint identity or evaluator implementation
from the distributed data alone. Reproducing the original experiment requires
the external PFM implementation, weights, target inputs and scientific tools.

The example does not establish that the candidate was adopted or deployed. It
does not compare decoupled and coupled self-evolution, establish a general
efficiency advantage, or replace the ablations planned for this project.
