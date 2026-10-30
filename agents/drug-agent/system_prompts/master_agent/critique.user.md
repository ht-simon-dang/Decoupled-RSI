---
id: master_agent.critique.user
phase: critique
role: user
used_by: drug_agent.llm.prompts:critique_prompt
placeholders: plan_summary, requirements_summary, evaluation_json
sha256: 768caf1fc17225c5725a47da01079d933659246fd6a128982ff4b2cf64c393bf
---
Plan summary:
{{plan_summary}}

Chemical requirements used this cycle:
{{requirements_summary}}

Evaluation report (tool-computed, authoritative):
{{evaluation_json}}

Produce your critique.
