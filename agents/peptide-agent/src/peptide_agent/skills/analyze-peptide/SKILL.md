---
name: analyze-peptide
description: Analyze supplied canonical linear/cyclic peptides and protein–peptide complexes.
---
Use a library brief and preserve supplied sequences. This skill cannot invoke generators.
Permissions come from policy.json: chemistry, contact geometry, toxicity and affinity tools.
ProtParam linear descriptors assume free termini and reduced cysteine. RDKit builds head-to-tail
bonds for cyclic chemistry; cyclic charge excludes terminal groups. Undefined pI is null.
Instability and linear trypsin heuristics are unavailable for cyclic peptides.
Report charge pH, mass units, topology and algorithm versions.
ToxinPred3 is ML-only and linear-only, and is not a clinical toxicity probability.
For PDB tools require distinct chains. Contacts/clashes describe geometry; PRODIGY estimates
ΔG/Kd but is not calibrated for peptides. Measured pKd models require supplied assay data.
Do not infer safety, efficacy or measured affinity from these computational outputs.
