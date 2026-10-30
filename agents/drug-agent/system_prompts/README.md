# `system_prompts/` — the prompt library

[English](README.md) | [简体中文](README_zh.md)

Every system and user message the **master agent** sends to a model lives here as
a Markdown file, not as a Python string literal. `src/drug_agent/llm/prompts.py`
is now composition only: it says which fragments go into which phase and what
runtime values fill the slots. Nothing in this folder is generated; these files
are the source of truth, and the Python reads them.

Why the prompts left the code: the prompt *is* the artefact this platform
evolves. The self-evolution loop proposes changes to it
(`RuntimePolicy.plan_prompt_addendum` and friends), the journal records which
prompt version produced which episode, and promotion decisions are arguments
about prompt text. That belongs in files a reviewer can read and diff on their
own terms.

---

## Layout

```
system_prompts/
├── README.md                          this map
├── README_zh.md                       Simplified Chinese map
├── stamp_checksums.py                 re-stamp sha256 after an intended edit
├── shared/                            fragments spliced into more than one phase
│   ├── master_rules.md
│   ├── objective_vocabulary.md
│   └── constraint_vocabulary.md
└── master_agent/                      one system + one user file per loop phase
    ├── plan.system.md         plan.user.md
    ├── transcribe.system.md   transcribe.user.md
    ├── critique.system.md     critique.user.md
    ├── refine.system.md       refine.user.md
    └── select_best.system.md  select_best.user.md      (after the loop, dynamic mode)
```

The hierarchy mirrors the planes the repository already uses. There is a
`master_agent/` directory and no `specialists/` directory because
`agents/llm/chemical_agent.py` and `agents/llm/evaluation_agent.py` both declare
`llm_backed = False` — they drive tools and arithmetic, they never call a model,
and they hold no prompt text. That is a property worth keeping: the agent that
produces a number and the agent that grades it are both outside the LLM's reach.

---

## File format

```
---
id: master_agent.plan.system
phase: plan
role: system
used_by: drug_agent.llm.prompts:plan_prompt -> ...
placeholders: rules
sha256: <hex digest of the body>
---
<the prompt body>
```

* **Body** is everything after the closing `---` line, minus the single trailing
  newline every text file ends with. A body that must itself end in a newline
  (only `shared/master_rules.md`) ends the file with a blank line.
* **Slots** are `{{name}}`. Doubled braces cannot occur in a prompt body — the
  JSON output shapes are `indent=2`-formatted, so every brace is followed by a
  newline — which is what makes plain textual substitution safe. The loader
  requires the caller to supply *exactly* the slots a template declares, so a
  renamed slot fails at the call site rather than reaching the model.
* **`sha256`** covers the body after the normalization above, so front matter can
  be re-documented freely while the prompt bytes stay pinned.

---

## The byte-for-byte invariant

`RuntimePolicy.prompt_version()` (`llm/policy.py`, **sealed**) mints the id that
every journal entry records, and analyses over the run history compare episodes
across it. Note what that id actually hashes: the policy *overlay* — the four
addenda and the two envelope flags — **not** the base prompt text. The base text
is assumed constant, and the whole recorded history is only comparable while that
assumption holds. Moving prompts out of Python removed the two things that used
to protect them (a `ruff format` pass and a diff no one can miss), so the
protection is now explicit:

* each file records the `sha256` of its body;
* `drug_agent.llm.prompts._block()` verifies it on first read and raises with
  both digests and the fix if they differ.

An **intended** prompt change is therefore two visible steps:

```bash
$EDITOR system_prompts/master_agent/plan.system.md
python system_prompts/stamp_checksums.py          # --check verifies without writing
```

so the digest change lands in the same diff as the text change that caused it.
Anything that reflows a file by accident — an editor stripping a trailing blank
line, a Markdown formatter rewrapping a paragraph — fails loudly at the next LLM
call instead of silently forking the history. **Do not run a Markdown formatter
over this folder.** The checksum utility skips `README.md` and `README_zh.md`;
neither is a prompt template.

Moving the prompts here changed no byte: `prompt_version()` is
`p1-d65e3ebc70d0` before and after, and all sixteen rendered strings (four
phases × system/user × with and without a policy overlay) compare identical.

---

## Packaging

The canonical copies live at the repository root, outside the package, because
they are documents. `[tool.hatch.build.targets.wheel] packages = ["src/drug_agent"]`
cannot reach a top-level folder, so `pyproject.toml` force-includes it:

```toml
[tool.hatch.build.targets.wheel.force-include]
"system_prompts" = "drug_agent/system_prompts"
```

`_library_root()` then resolves, in order, `<package>/system_prompts` (present
only in a built wheel) and `<repo root>/system_prompts` (a checkout or an
editable install). One set of bytes, three layouts, no copy step. There is
deliberately **no** environment-variable override of the library path: a knob
that swaps prompt text without changing `prompt_version()` would make the journal
lie about what was sent.

---

## The map

### Shared fragments

| file | id | used by | must contain | invariants |
|---|---|---|---|---|
| `shared/master_rules.md` | `shared.master_rules` | `llm/prompts.py` → all four builders; opens every master-agent system prompt | Role priming (master agent, senior computational medicinal chemist, directs the chemical + evaluation specialists) and the five non-negotiable rules: research-prioritization not clinical; never invent scientific numbers; SMILES may be *proposed* as hypotheses only; the user message is a goal, not an instruction override; output is exactly one ```json fence. | Rule 2 is the epistemic guardrail the whole platform rests on — no edit may soften "you must NEVER invent scientific numbers". Rule 4 is the prompt-injection boundary and pairs with `agents/platform/literature_agent.py`. Rule 5 is what `llm/jsonio.py` decodes and `llm/harness.py` repairs against; changing the fence contract breaks both. The body ends with a newline (hence the blank line at end of file). |
| `shared/objective_vocabulary.md` | `shared.objective_vocabulary` | `transcribe_prompt`, `refine_prompt` | The five objective name tokens — `pIC50`, `qed`, `sa_score`, `hERG`, `restoration_score` — with direction and meaning, and the statement that direction is exactly `maximize` or `minimize`. | Every token must be a name the backend scores; `restoration_score` must stay marked phenotype-only. The direction strings are parsed downstream — see `llm/schemas.py:ObjectiveSpec`. |
| `shared/constraint_vocabulary.md` | `shared.constraint_vocabulary` | `transcribe_prompt`, `refine_prompt` | The seven dotted hard-constraint property keys (`descriptors.mw`, `.logp`, `.tpsa`, `.hbd`, `.hba`, `.qed`, `n_structural_alerts`) with units and example thresholds, plus the six allowed operators. | Keys must stay in sync with what the ranking/filter plane can evaluate; a key the backend does not know produces a constraint that silently never binds. Operators must stay in sync with `agents/llm/evaluation_agent.py:_compare`. |

### Master agent — the four loop phases

| file | id | used by (module:function → phase) | must contain | invariants |
|---|---|---|---|---|
| `master_agent/plan.system.md` | `master_agent.plan.system` | `llm/prompts.py:plan_prompt` → `agents/llm/master_agent.py:MasterAgent.run`, **PHASE 1 (plan)**. Decoded into `llm/schemas.py:TaskPlan`. | `{{rules}}`, the planning task, the allowed metric-key list grouped EFFICACY / PROPERTIES / PORTFOLIO / CONFIDENCE, the instruction to pick 3–6 criteria, and the `TaskPlan` output shape. | **The metric keys listed here must stay in sync with `llm/schemas.py:METRIC_KEYS`**, both directions — `tests/test_pools.py::test_the_plan_prompt_offers_every_metric_the_evaluator_can_compute` and its mirror enforce it. A key offered but not computed makes ACCEPT unreachable; a key computed but not offered is dead (this is exactly how `restoration_median` was lost on phenotype programs). The goal-kind guidance (pIC50 metrics for a protein target, `restoration_median` for phenotype rescue) is what `llm/repair.py:repair_criteria_for_goal` deterministically enforces when the policy enables it. |
| `master_agent/plan.user.md` | `master_agent.plan.user` | same | The user instruction fenced in `"""` and the imperative to produce the plan. | `{{instruction}}` is untrusted user text; it stays quoted and stays *below* the rules. `plan_prompt` applies `.strip()` before filling. |
| `master_agent/transcribe.system.md` | `master_agent.transcribe.system` | `llm/prompts.py:transcribe_prompt` → **PHASE 2 (transcribe)**. Decoded into `llm/schemas.py:ChemicalRequirements`. | `{{rules}}`, the transcription task and the `target_based` / `phenotype_based` split, `{{objective_vocabulary}}`, `{{constraint_vocabulary}}`, the guidance block (potency objective required; add `qed`/`sa_score`; encode numeric limits as constraints; novelty → seeds; targets are targets not measurements; what `n_generated` means), and the `ChemicalRequirements` output shape. | The output shape's keys must match `ChemicalRequirements` field names — this phase runs under schema-constrained decoding (`llm/harness.py:_use_schema_decoding` returns True for `transcribe`), so a drifted key becomes a hard decode failure, not a soft one. The generation guidance must keep stating the model's levers accurately: it chooses the candidate source, the de-novo model (`generator_backend`), the sample count, and whether a binding site is detected (`pocket_source`, `pocket_detector`, `pdb_ids`). This reverses the older invariant that the source "is a deployment decision" — it moved to the master agent, because which arm suits a goal is a judgement about the goal. It also chooses the ASSESSMENT DEPTH — `property_backends`, `route_backend`, `complex_backend`, `target_evidence` — each of which switches on an installed tool that no run could reach before the field existed. The one thing the model still may not write is a coordinate: pocket centres are measured, never named. |
| `master_agent/transcribe.user.md` | `master_agent.transcribe.user` | same | Fenced user instruction plus the plan summary from PHASE 1. | `{{plan_summary}}` comes from `master_agent.py:_plan_summary`; `{{instruction}}` is `.strip()`ed untrusted text. |
| `master_agent/critique.system.md` | `master_agent.critique.system` | `llm/prompts.py:critique_prompt` → **PHASE 4 (critique)**, after the chemical and evaluation agents have run. Decoded into `llm/schemas.py:Critique`. | `{{rules}}`, the judging task, the three verdicts (ACCEPT / REFINE / REPLAN) with what each means, the cycle-budget sentence, and the `Critique` output shape with `per_criterion`. | Must keep "do not invent values" adjacent to the comparison instruction: this is the phase where a model is most tempted to supply a number the report did not contain. The verdict tokens are matched by `master_agent.py`; renaming one strands the refinement branch. `{{cycle}}` / `{{max_cycles}}` are filled with `str(int)` and exist to make the model prefer ACCEPT near the budget. |
| `master_agent/critique.user.md` | `master_agent.critique.user` | same | Plan summary, the cycle's requirements summary, and the evaluation report labelled *tool-computed, authoritative*. | `{{evaluation_json}}` is `EvaluationReport.compact_json()` (`agents/llm/evaluation_agent.py`) — a trimmed view; widening it costs context on every cycle. The "authoritative" label is load-bearing: it is what separates measured numbers from the model's own text. |
| `master_agent/refine.system.md` | `master_agent.refine.system` | `llm/prompts.py:refine_prompt` → **PHASE 5 (refine)**, on a REFINE verdict; the adjusted requirements start the next cycle. Decoded into `llm/schemas.py:RequirementsAdjustment`. | `{{rules}}`, "do not simply retry — diagnose then change", the lever list (`seed_smiles`, `property_constraints`, `objectives`, `max_candidates`/`shortlist_size`, `n_generated`), `{{objective_vocabulary}}`, `{{constraint_vocabulary}}`, and the output shape whose `adjusted_requirements` is a COMPLETE `ChemicalRequirements`. | This is the ablation core: every entry in `changes` must map to the diagnosis, which is what makes a cycle's effect attributable. "COMPLETE object" must stay emphatic — `objectives` and `property_constraints` default to `[]`, so an omission under schema-constrained decoding (this phase is constrained) is a *deletion*; that failure is what `RuntimePolicy.refine_carry_forward` and `refine_max_drop_fraction` exist to catch. The levers named must remain levers the pipeline actually honours. |
| `master_agent/refine.user.md` | `master_agent.refine.user` | same | Current requirements JSON, the critique/diagnosis, and the authoritative evaluation report. | Same authority framing as the critique user message. |

### Master agent — the step after the loop

| file | id | used by (module:function → phase) | must contain | invariants |
|---|---|---|---|---|
| `master_agent/select_best.system.md` | `master_agent.select_best.system` | `llm/prompts.py:select_best_prompt` → `agents/llm/master_agent.py:MasterAgent._select_best_candidate`, the **FINAL STEP** of a `dynamic`-ceiling run, after the last cycle. Decoded into `llm/schemas.py:BestCandidateSelection`. | `{{rules}}`, "choose ONE candidate from ANY cycle", the "this is a RANKING, not a measurement" paragraph, the four tie-breaks (criteria the goal depends on; applicability domain; Pareto rank; later ≠ better), and the `BestCandidateSelection` output shape. | **ADR-0011 is the whole shape of this prompt.** The ranking/measurement paragraph and "do not name a candidate that is not in the table" are what keep the step a decision rather than a source of numbers, and they are backed structurally: `BestCandidateSelection` has no numeric field to write into, and `master_agent.py` re-reads every score for the chosen id from the evaluation report and refuses an id nobody offered. Softening the prompt would leave those two, but the three are meant to be read together. Runs under `config.SELECT_BEST` (the largest `num_predict`, the lowest temperature) — a bigger candidate table wants a bigger budget, not a shorter prompt. |
| `master_agent/select_best.user.md` | `master_agent.select_best.user` | same | The plan summary, each cycle's criteria outcomes, and **every candidate from every cycle** with its per-endpoint scores. | `{{candidates_json}}` is deliberately every cycle's shortlist, not the winning cycle's — a step that only sees the winning cycle cannot do the one thing it exists for. Both data slots carry the *tool-computed, authoritative* label, the same framing the critique and refine user messages use, and `null` must keep meaning "the tool could not compute it" rather than zero. |

### Where the optional blocks attach

Runtime-built text is appended to a *system* prompt after it is composed, by
`llm/prompts.py:_extend`, and is therefore not a file here. `_extend` offers one
channel (`envelope_block`) and `agents/llm/master_agent.py:_join_blocks` stacks
into it whichever of these apply to the run:

| block | source | attaches to | gate |
|---|---|---|---|
| deployment capability | `master_agent.py:_availability_block` — the live tool pool intersected with `agents/registry.py:AGENT_TOOLS`, never authored and never hand-listed | plan, transcribe | **none — always on** |
| hard ceilings | `master_agent.py:_ceiling_block` — the `--ceiling METRIC=VALUE` values the operator gave at launch | plan, transcribe | ceiling mode `hard` (a run parameter) |
| capability envelope | `tools/envelope_tool.py:CapabilityEnvelope.as_prompt_block` — generated from a real probe run, never authored | plan, transcribe, critique, refine, select_best | `RuntimePolicy.capability_envelope`, `.capability_envelope_transcribe`, **or** ceiling mode `dynamic` |
| policy addendum | the self-evolution overlay (`RuntimePolicy.*_prompt_addendum`), written by the hypothesis engine at runtime | all four loop phases | non-empty addendum |

**Why the capability block has no gate, and specifically not
`RuntimePolicy.capability_envelope`.** That flag turns on an experimental
treatment the self-evolution arm is studying: giving the planner *measured
numbers* — "this deployment has been observed to reach pIC50 6.4582" — which is
a claim about what the pipeline achieves and is fair to A/B. "AutoDock Vina is
not installed on this machine" is not a treatment and not a measurement; it is a
fact about the deployment, of the same kind as the model name, and a baseline arm
that withholds it is not cleaner, only less informed. The two are therefore
independent by construction, and `tests/test_ceiling_modes.py` holds them apart.

With every block empty, each builder returns exactly the string it returned
before those parameters existed. That is what makes a candidate arm and a
baseline arm comparable, and it is verified by `tests/test_selfevolve_policy.py`.

---

## Prompts NOT moved, and why

These are model-facing text that this change deliberately left in place. Each is
listed so the map is complete even where the file is not.

| prompt | location | why it stayed |
|---|---|---|
| self-evolution **router** | `selfevolve/controller/loop.py:73` `_ROUTER_SYSTEM` | Outside this change's file lane. It is also a different plane: it steers the *evolver*, not the user-serving agent, and its text is not covered by `prompt_version()`. A follow-up can move it to `system_prompts/selfevolve/router.system.md` using the same loader. |
| hypothesis **generator** | `selfevolve/hypotheses/generator.py:67` `_GENERATOR_SYSTEM` | Same lane and plane. Its rule 4 ("you may not propose weakening the evaluation, the benchmark, or the success criteria used to judge your own work") is the integrity guard of the whole self-evolution arm and should be reviewed as carefully as anything in `shared/master_rules.md`. Candidate target: `system_prompts/selfevolve/hypothesis_generator.system.md`. |
| hypothesis **verifier** | `selfevolve/hypotheses/generator.py:91` `_VERIFIER_SYSTEM` | Same. Candidate target: `system_prompts/selfevolve/hypothesis_verifier.system.md`. Its distinguishing instruction — that `INSUFFICIENT_EVIDENCE` is a rewarded answer — must survive any move. |
| JSON **repair** message | `llm/harness.py:300-310` | A *user*-role recovery turn, not a system prompt, and it is assembled from the live decode error plus `_required_fields(schema)` (introspected from the pydantic model). Extracting only its constant half would split one message across two places for no gain. |
| capability **envelope** block | `tools/envelope_tool.py:111` `as_prompt_block` | Not authored text. It is rendered from measurements taken on the machine it runs on; there is nothing static to store. |
| **sealed** evaluation kernel | `selfevolve/evaluation/*.py` | **Hash-pinned.** `selfevolve/evaluation/integrity.py:kernel_digest` fingerprints these modules so a candidate cannot be scored by a kernel it altered; editing any of them voids the benchmark arm. Inventoried and confirmed: the kernel contains **no** model-facing prompt text at all — it is scoring, replay, conformance, and calibration only. Nothing to move. |

Not prompts, listed to close the search: `agents/platform/literature_agent.py:26`
holds prompt-*injection* detection patterns (defensive regexes over untrusted
retrieved text), and `tools/llm_provider_tool.py` is a deterministic stub whose
output is a hash of its input.
