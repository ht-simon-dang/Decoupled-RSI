---
id: master_agent.transcribe.user
phase: transcribe
role: user
used_by: drug_agent.llm.prompts:transcribe_prompt
placeholders: instruction, plan_summary
sha256: 9cfbce4cde6e05b6d460db967b09bec0744216587838791165845b99f7ae91d2
---
User instruction:
"""
{{instruction}}
"""

Your plan summary:
{{plan_summary}}

Produce the chemical requirements.
