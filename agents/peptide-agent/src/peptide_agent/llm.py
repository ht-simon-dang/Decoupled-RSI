"""Qwen proposes a typed brief; the model cannot grant permissions or invent measurements."""
from __future__ import annotations

import json
from importlib.resources import files

from .contracts import Plan
from .models import MODELS, model_path


def parse_plan(text: str) -> Plan:
    # Reject markdown, trailing text, tool names, shell commands and unknown fields.
    return Plan.model_validate_json(text.strip())


def verify_sequences(planned: Plan, instruction: str) -> Plan:
    supplied = list(planned.brief.sequences)
    if planned.brief.target_sequence:
        supplied.append(planned.brief.target_sequence)
    if planned.brief.target_pdb and planned.brief.target_pdb not in instruction:
        raise ValueError("Planned PDB must occur verbatim in the user instruction")
    if any(sequence not in instruction for sequence in supplied):
        raise ValueError("Planned sequences must occur verbatim in the user instruction")
    return planned


def plan(instruction: str, store=None, run_id="planner") -> Plan:
    import torch
    from .decoding import prefix_function, tokenizer_data
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if not 1 <= len(instruction) <= 12000:
        raise ValueError("Instruction must be 1..12000 characters")
    path = model_path("qwen")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype="auto").eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    constrained_tokenizer = tokenizer_data(tok)
    skills = "\n".join(files("peptide_agent").joinpath("skills", s, "SKILL.md").read_text()
                       for s in ("design-peptide", "analyze-peptide"))
    system = ("You plan peptide research. Return ONLY one JSON object matching this schema. "
              "Use user-provided sequences verbatim. Never invent a target sequence. "
              "If a required target is missing, return {} so validation fails. "
              "For generation: skill=design-peptide, source=pepmlm, target_sequence=the target, sequences=[]. "
              "For cyclic/head-to-tail design choose source=boltzgen and topology=head_to_tail. "
              "For analyzing supplied peptides: skill=analyze-peptide, source=library, sequences=the peptides. "
              "Never copy a target protein into the peptide sequences array. "
              "Omit fields with defaults unless the user specifies them. "
              "Never output computed scientific measurements. No commands, URLs or extra fields. "
              "Schema: " + json.dumps(Plan.model_json_schema()) + "\nSkills:\n" + skills)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    errors = []
    for attempt in range(2):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] > 8192:
            raise ValueError("Planning context exceeds application limit of 8192 tokens")
        prefix = prefix_function(constrained_tokenizer, Plan.model_json_schema())
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1536, do_sample=False, max_time=90,
                                    prefix_allowed_tokens_fn=prefix)
        reply = tok.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if store:
            store.event(run_id, "llm.reply", {"model": MODELS["qwen"], "attempt": attempt,
                                                 "reply": reply, "instruction": instruction})
        try:
            return verify_sequences(parse_plan(reply), instruction)
        except ValueError as exc:
            errors.append(str(exc))
            messages.extend([{"role": "assistant", "content": reply},
                             {"role": "user", "content": "Invalid JSON/schema. Correct it. " + str(exc)[:1500]}])
    raise ValueError("Qwen plan rejected after two attempts: " + errors[-1])
