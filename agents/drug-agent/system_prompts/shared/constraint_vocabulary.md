---
id: shared.constraint_vocabulary
phase: transcribe, refine
role: system-fragment
used_by: drug_agent.llm.prompts:transcribe_prompt, :refine_prompt
placeholders: (none)
sha256: 7fd5fbf0d075376157fa6179a0548781b8394e12a7818be848a59a70b3cd704d
---
Hard-constraint property keys (dotted; use these exact tokens) with operators
one of <, <=, >, >=, ==, !=:
  - "descriptors.mw"      molecular weight (Da)      e.g. <= 550
  - "descriptors.logp"    calculated logP            e.g. <= 5.5
  - "descriptors.tpsa"    topological polar SA (A^2) e.g. <= 140
  - "descriptors.hbd"     H-bond donors              e.g. <= 5
  - "descriptors.hba"     H-bond acceptors           e.g. <= 10
  - "descriptors.qed"     QED (0..1)                 e.g. >= 0.5
  - "n_structural_alerts" count of structural alerts e.g. == 0
