---
id: shared.objective_vocabulary
phase: transcribe, refine
role: system-fragment
used_by: drug_agent.llm.prompts:transcribe_prompt, :refine_prompt
placeholders: (none)
sha256: fa875f5191457b7335bf20fcb5e466dbda3e13cabc0ae3978ef8106b0e3571c5
---
Objective names understood by the backend (use these exact tokens):
  - "pIC50"            (target potency; direction: maximize; higher = more potent)
  - "qed"             (drug-likeness 0..1; maximize)
  - "sa_score"        (synthetic accessibility 1..10; minimize = easier to make)
  - "hERG"            (cardiac liability proxy; minimize)
  - "restoration_score" (phenotype rescue; maximize) -- ONLY for phenotype programs
Objective direction is exactly "maximize" or "minimize".
