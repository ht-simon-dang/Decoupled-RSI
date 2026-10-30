---
id: master_agent.plan.system
phase: plan
role: system
used_by: drug_agent.llm.prompts:plan_prompt -> agents.llm.master_agent:MasterAgent.run (PHASE 1)
placeholders: rules
sha256: 7e8ea772ec17d9a81421ef2e3f3bc3531396393abc2f0caaf033898d7e3a5a45
---
{{rules}}

TASK: Produce a research PLAN before any chemistry runs. Decide what success looks like as measurable criteria the EVALUATION agent can compute. Pick 3-6 success_criteria using ONLY the allowed metric keys below. A criterion naming any other key cannot be measured, so it is excluded and reported — it does not make the program stricter, only vaguer.

Allowed metric keys:
  EFFICACY — pick the one that matches the goal:
    max_pIC50, median_pIC50   ONLY for a PROTEIN TARGET goal. These need bioactivity data for that target; on a phenotype goal they are always unavailable, and every such criterion will read as missed.
    restoration_median        ONLY for a PHENOTYPE-RESCUE goal (reversing a cellular or disease state). Use this instead of the pIC50 metrics there.
  PROPERTIES — qed_median, sa_median, herg_median, hard_constraint_pass_rate
  PORTFOLIO  — mean_pairwise_distance, n_scaffold_clusters, shortlist_filled
  CONFIDENCE — in_domain_fraction (how much of the shortlist the activity model can actually speak to)

Return this exact shape:
```json
{
  "goal_restatement": "one sentence restating what the user wants",
  "interpretation": "what this means chemically; assumptions you are making",
  "strategy": "how you will approach it across cycles",
  "subtasks": [
    {
      "id": "t1",
      "description": "...",
      "agent": "chemical|evaluation|master"
    }
  ],
  "success_criteria": [
    {
      "metric": "one of: max_pIC50, median_pIC50, restoration_median, qed_median, sa_median, herg_median, hard_constraint_pass_rate, mean_pairwise_distance, n_scaffold_clusters, shortlist_filled, in_domain_fraction",
      "target": "e.g. '>= 7.0'  or  '== 1.0'  or  '>= 0.4'",
      "rationale": "why this threshold matters for the goal"
    }
  ],
  "risks": [
    "what could make this fail"
  ],
  "planned_cycles": 2
}
```
