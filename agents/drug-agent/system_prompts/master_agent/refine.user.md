---
id: master_agent.refine.user
phase: refine
role: user
used_by: drug_agent.llm.prompts:refine_prompt
placeholders: requirements_json, critique_summary, evaluation_json
sha256: 6deb841d0ff4ed4e5041c19fff25941ad4be24bb8141bc03e1996dca67633402
---
Current chemical requirements:
{{requirements_json}}

Critique / diagnosis:
{{critique_summary}}

Evaluation report (authoritative numbers):
{{evaluation_json}}

Produce the adjusted requirements.
