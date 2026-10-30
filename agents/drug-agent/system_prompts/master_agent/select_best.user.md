---
id: master_agent.select_best.user
phase: select_best
role: user
used_by: drug_agent.llm.prompts:select_best_prompt
placeholders: plan_summary, criteria_summary, candidates_json
sha256: 8012bf2c0b948a1c14cc642af81914fca33d3c8aabc4d6260a359cd539655988
---
Plan summary:
{{plan_summary}}

How each cycle scored against the success criteria (tool-computed, authoritative):
{{criteria_summary}}

Every candidate from every cycle, with its per-endpoint scores (tool-computed, authoritative; `null` means the tool could not compute it):
{{candidates_json}}

Choose the best candidate across all cycles.
