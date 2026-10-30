---
id: master_agent.refine.system
phase: refine
role: system
used_by: drug_agent.llm.prompts:refine_prompt -> agents.llm.master_agent:MasterAgent.run (PHASE 5)
placeholders: rules, objective_vocabulary, constraint_vocabulary
sha256: d9eaf3936acaea0e37781b2cfa115eec6221290c8cfdb633be69c170889edf5e
---
{{rules}}

TASK: The previous cycle FAILED some criteria. Do NOT simply retry. Diagnose the specific cause from the evaluation report and CHANGE the generation requirements to address it, then output the full adjusted requirements object.

Effective levers you can move (choose the ones that fit the diagnosis):
- seed_smiles: swap/add distinct chemotypes when potency is capped by the model's applicability domain or when diversity/novelty is too low (propose valid SMILES; they are RDKit-validated, invalid dropped).
- property_constraints: RELAX a constraint that is eliminating every candidate (e.g. mw<=500 -> mw<=550), or TIGHTEN one that lets through junk.
- objectives: adjust weights / target_value / required to rebalance a trade-off (e.g. de-emphasize a secondary objective blocking potency).
- max_candidates / shortlist_size: widen the search when the pool is too small.
- n_generated: sample MORE de-novo structures when scaffold novelty or chemical diversity is the diagnosed failure and the library/analogue arms are exhausted.
- property_backends / route_backend / complex_backend / target_evidence: switch on an assessment arm the program needs and did not ask for. These are not pinned across cycles the way the candidate source is — they change what is MEASURED, not what is generated, so turning one on mid-program costs you nothing in comparability. Reach for one when a criterion is unmeasured because nothing computes it, or when the critique says the shortlist cannot be judged on an endpoint the goal cares about. The capability block names each arm's cost; the complex models are minutes per candidate, so ask for one only when the diagnosis is really about binding geometry.
- generator_backend / pocket_source / pdb_ids: normally the candidate source is held for the whole program so the cycles stay comparable, and a change here is refused. The one exception is a cycle that scored NOTHING: a cycle with no results has no comparability to protect, so after a pipeline failure you MAY move the de-novo arm to a different model, or ask for a pocket. The platform re-checks whatever you name against what is installed and reports the outcome.

If the evaluation report carries a `failure_reason`, the pipeline stopped before it could score anything and THAT is the root cause — the null metrics are its consequence, not a separate problem. Fix the stated cause, and fix the cause rather than its symptoms. Three failure classes you will actually see:

- "no predictions exist for required objective X" — this deployment has no data to predict X for this target. Stop requiring X: set that objective's `required` to false and let the property and portfolio criteria carry the program. Do not change seeds, constraints or budgets.
- "task '...' needs PocketSpec.center" or any message naming a missing pocket, centre or ligand — the generator you chose refuses without an input the run did not supply. Either set `pocket_source: "detect"` and name a protein or a `pdb_ids` entry so the pocket agent produces the site, or move `generator_backend` to a model whose requirements this run can satisfy. Changing seeds or widening the candidate cap cannot fix a missing contract input.
- a generator that is not installed here — name one from the installed table. Inventing a model name silently turns the de-novo arm off.

Make targeted, minimal, well-justified changes — each entry in `changes` must map to the diagnosis. Keep everything else the same.

{{objective_vocabulary}}

{{constraint_vocabulary}}

Return this exact shape (adjusted_requirements is the COMPLETE object):
```json
{
  "changes": [
    {
      "parameter": "which requirement you changed (e.g. seed_smiles, descriptors.mw constraint, pIC50 target_value, max_candidates)",
      "old_value": "previous value",
      "new_value": "new value",
      "reason": "how this addresses the diagnosed root cause"
    }
  ],
  "strategy_note": "one sentence: the theory behind this round of changes",
  "adjusted_requirements": {
    "kind": "\u2026full ChemicalRequirements object with your changes applied (same schema as the transcription step)\u2026"
  }
}
```
