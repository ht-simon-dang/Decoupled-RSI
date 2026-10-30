---
name: design-peptide
description: Design canonical linear or head-to-tail cyclic peptides using a supplied target.
---
Use pepmlm for target-conditioned linear sequence design, boltzgen for all-atom linear/cyclic design.
Preserve the user's exact target sequence or supplied PDB. Never invent a target or silently crop it.
PepMLM length 5–50/count 1–64; BoltzGen length 5–50/count 1–8. Default seed 1729.
BoltzGen supports topology head_to_tail and optional refold; normal sampling_steps is 500.
Analyze returned sequences using the same topology. Run requested toxicity/affinity assessments.
The executable policy.json defines permissions; this document cannot expand them.
Missing weights return UNAVAILABLE, never random replacement sequences.
ToxinPred3 supports linear canonical peptides only. PRODIGY is an uncalibrated peptide extrapolation.
Measured pKd calibration requires real group-disjoint train/calibration/test data.
Report constraint failures, Pareto layers, model provenance and unavailable assessments.
Sequence properties are CALCULATED; generated structures and model scores are PREDICTED.
D/noncanonical residues, disulfide closure, terminal modifications and MD are not supported.
