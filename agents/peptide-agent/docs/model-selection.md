# Model and algorithm selection — checked 2026-10-18

[English](model-selection.md) | [简体中文](model-selection_zh.md)

“Frontier” is task-dependent: sequence-only linear binders, pocket-conditioned full-atom
codesign and macrocycles cannot share an unsupported universal SOTA claim. This project chooses
downloadable sequence and all-atom backends; unreleased research candidates remain separate.

| Model | Primary source | Decision |
| --- | --- | --- |
| Qwen3.5-0.8B | [official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Required local planner; strict brief validation, not a scientific predictor |
| PepMLM-650M | [authors' implementation](https://github.com/programmablebio/pepmlm), [weights](https://huggingface.co/ChatterjeeLab/PepMLM-650M) | Selected, implemented and pinned; target sequence to linear peptide, ordinary Transformers runtime |
| BoltzGen-1 | [official implementation](https://github.com/HannesStark/boltzgen), [weights](https://huggingface.co/boltzgen/boltzgen-1) | Installed 0.3.2; all-atom linear/cyclic design and optional Boltz2 refolding |
| PepGLAD | [official code / released checkpoints](https://github.com/THUNLP-MT/PepGLAD) | Established full-atom pocket-conditioned comparator; not installed; older torch/scatter stack needs separate compatibility validation within the one environment |
| PepTri, ICLR 2026 | [official repository](https://github.com/aigensciences/PepTri) | Recent structural frontier candidate; README describes training and local checkpoint layouts, but no verified ready-to-download checkpoint was found in the inspected release page; not an executable backend here |
| PepFGLD, ICML 2026 | [official repository](https://github.com/NiuDongjiang/PepFGLD) | Inspected README says code/data/pretrained models will be released; cannot claim available inference |
| PepEDiff, 2026 | [official repository](https://github.com/LabJunBMI/PepEDiff-Seq-Only) | New sequence-generation research route; checkpoint/data are linked through SharePoint and not validated locally; deferred |

PepMLM's associated Nature Biotechnology article is linked from its authors' model card.
It is selected for reproducible contemporary binder generation, not because this project has
established superiority over every 2026 method. Its likelihood is not an affinity estimate.
The adapter implements simultaneous masked-span prediction and per-position top-k softmax sampling,
restricting output to the 20 canonical amino acids; this explicit restriction is recorded in provenance.

Pinned model revisions:

- Qwen/Qwen3.5-0.8B: `2fc06364715b967f1860aea9cf38778875588b17`.
- ChatterjeeLab/PepMLM-650M: `898fca941a9057aebdd1a6164b5ee09a1a71780e`.

The Qwen card declares Apache-2.0. PepMLM's card has MIT metadata and also retains
additional usage-related fields in its YAML metadata; inspect the upstream terms before distributing
weights or using generated outputs commercially. This repository neither changes nor grants model rights.
The public HF API reported both selected repositories ungated when checked.

## Computational analysis and evaluation

[Biopython ProtParam](https://biopython.org/docs/latest/api/Bio.SeqUtils.ProtParam.html)
provides established molecular-weight, pI, charge and hydropathy calculations. Its protein instability
index is included with an explicit domain limitation, not promoted as a peptide half-life predictor.
Sequence uniqueness/edit distance and heavy-atom geometry supply transparent baselines without learned claims.

[PepBenchmark](https://github.com/ZGCI-AI4S-Pep/PepBenchmark/) is a current public reference for
peptide property evaluation. Our executable benchmark harness supplies explicit Dummy, Logistic
and RandomForest baselines on user-supplied group-disjoint splits. No PepBenchmark datasets have
been downloaded or scored here; the harness is not claimed to reproduce its leaderboard.

For future structural benchmarking, PepGLAD's PepBench is a separate structural dataset, not the same
project as PepBenchmark. Reference-based DockQ/iRMSD/Fnat should be added only with a native complex and
explicit chain/residue mapping. Boltz/AlphaFold confidence, MD energies, and Rosetta interface scores
must retain their own evidence types and are not replacements for experimental Kd/IC50 or toxicity.

The [DockQ](https://github.com/wallnerlab/DockQ) 2.1.3 dependency solve was checked but not installed:
its `numpy<2.0` requirement conflicts with this environment's NumPy 2 analysis stack. Adding it requires
a tested unified dependency revision, not silently installing another environment or forcing incompatible packages.

## Structural generation, toxicity and affinity integration

BoltzGen uses pinned weights `c1be29e1f82ffcc72264f64b993c43fb4e0d17f0` and inference molecule
archive `c3d36fd276e9caf098c75d4113c6d5eb320b1a4c`. Version 0.3.2 runs within the same Python
runtime. The adapter uses all-atom sequence design without a separate inverse-folding stage.
Output chain identity is checked against upstream design-mask metadata before normalization.

[ToxinPred3](https://github.com/raghavagps/toxinpred3) ML-only weights use revision
`dcdba833cd48cc8a2090ac90f3319fd9bcb87a1c`. Its sklearn 1.2.2 pickle is checksum-verified and
converted to NumPy tree arrays without retraining or changing thresholds, avoiding a second
legacy environment. Runtime never loads user pickles. Features are 20 AAC + 400 overlapping DPC,
both percentages. Threshold 0.38 is the published classifier decision threshold, not clinical calibration.

[PRODIGY](https://github.com/haddocking/prodigy) 2.4.0 supplies empirical complex ΔG/Kd.
The separate local Ridge/conformal calibration API requires measured pKd and group-disjoint
train/calibration/test records. Neither model is advertised as experimentally validating candidates.
