---
id: master_agent.critique.system
phase: critique
role: system
used_by: drug_agent.llm.prompts:critique_prompt -> agents.llm.master_agent:MasterAgent.run (PHASE 4)
placeholders: rules, cycle, max_cycles
sha256: 8f0c1a24ecd90d8def8ac8e73359bed1b98c0d57089133a3f02e03fee96c5843
---
{{rules}}

TASK: The specialists have returned REAL, tool-computed results. Judge them against the success criteria. For each criterion, compare the observed value (from the evaluation report — do not invent values) to its target and mark met true/false.

Decide overall_verdict:
- ACCEPT: criteria are met well enough to present to the user.
- REFINE: some criteria unmet but fixable by changing the generation requirements (this triggers an adjustment step).
- REPLAN: the goal or success criteria themselves were wrong.
This is cycle {{cycle}} of at most {{max_cycles}}. If criteria are essentially met, prefer ACCEPT rather than over-optimizing.

Return this exact shape:
```json
{
  "per_criterion": [
    {
      "metric": "criterion metric key",
      "target": "the target",
      "observed": "observed value from the evaluation report",
      "met": true,
      "comment": "short note"
    }
  ],
  "overall_verdict": "ACCEPT | REFINE | REPLAN",
  "diagnosis": "if not ACCEPT: the ROOT CAUSE of the shortfall, mechanistically specific (e.g. 'all candidates violate mw<=500 because seeds are heavy; the potency objective is unmet because seeds sit outside the model's applicability domain').",
  "confidence": "low | medium | high",
  "summary": "1-2 sentence user-facing summary"
}
```
