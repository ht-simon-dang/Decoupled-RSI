"""Qwen3.5-0.8B selects a typed route; sequences stay in validated user input."""
from __future__ import annotations
import gc
import json
import re
from typing import Literal
from pydantic import Field
from .contracts import Brief, Contract, Context, MolecularConfig, Budget, Objective
from .registry import QWEN, qwen_path


class RoutePlan(Contract):
    focus: Literal["balanced", "cds", "utr", "diagnose"]
    priority: Literal["expression", "duration", "selectivity", "structure", "exploration"]
    rationale: str = Field(min_length=1, max_length=400)


class Intent(Contract):
    mode: Literal["full", "cds", "utr", "diagnose", "transfer"] = "full"
    target_cell: str = Field(default="unspecified", max_length=80)
    species: Literal["human", "mouse", "unspecified"] = "unspecified"
    endpoint: Literal["protein_auc", "protein_peak", "functional_duration", "translation_efficiency"] = "protein_auc"
    modification: Literal["unmodified", "m1Psi", "unspecified"] = "unspecified"
    rationale: str = Field(max_length=300)


def query(prompt, schema, store=None, run_id=None):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from .decoding import tokenizer_data, prefix_function
    path = qwen_path()
    if not (path / "config.json").is_file():
        raise FileNotFoundError("Pinned Qwen3.5-0.8B is unavailable; set MRNA_QWEN_DIR")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True, trust_remote_code=False, dtype="auto").eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    messages = [{"role": "system", "content": "You are the EMA4D mRNA research planner. Return ONLY JSON matching the schema. Never output sequences, measurements, shell commands or tool permissions. Preserve user-specified constraints. Unspecified context must remain unspecified. Schema: " + json.dumps(schema.model_json_schema())}, {"role": "user", "content": prompt}]
    data = tokenizer_data(tokenizer)
    error = ""
    try:
        for attempt in range(2):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            if inputs.input_ids.shape[1] > 8192:
                raise ValueError("Planner input exceeds 8192-token application limit")
            with torch.inference_mode():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=512, max_time=90,
                                        prefix_allowed_tokens_fn=prefix_function(data, schema.model_json_schema()))
            reply = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if store:
                store.event(run_id, "planner.reply", {"model": QWEN, "attempt": attempt, "reply": reply, "prompt": prompt,
                                                     "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": int(tokens.shape[1]-inputs.input_ids.shape[1])})
            try:
                return schema.model_validate_json(reply)
            except ValueError as exc:
                error = str(exc)
                messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Repair this schema error: " + error[:600]}]
        raise ValueError("Planner failed validated JSON after two attempts: " + error)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def plan(brief, store=None, run_id=None):
    metadata = brief.model_dump(exclude={"protein", "reference", "joint_proposals"})
    metadata["protein_length_aa"] = len(brief.protein or "")
    metadata["locked_regions"] = sorted(brief.locks)
    return query("Choose a focus and priority for this locked brief. This cannot change its endpoints or constraints. " + json.dumps(metadata, ensure_ascii=False), RoutePlan, store, run_id).model_dump()


def from_instruction(instruction, store=None, run_id=None):
    if not 1 <= len(instruction) <= 12000:
        raise ValueError("Instruction requires 1..12000 characters")
    # Exact sequence extraction prevents the small language model inventing a protein.
    match = re.search(r"(?:protein|sequence|蛋白(?:序列)?|序列)\s*[:：=]\s*([ACDEFGHIKLMNPQRSTVWY]{2,}\*?)", instruction, re.I)
    if match is None:
        matches = re.findall(r"(?<![A-Za-z])[M][ACDEFGHIKLMNPQRSTVWY]{7,}(?![A-Za-z])", instruction)
        protein = matches[0] if len(matches) == 1 else None
    else:
        protein = match.group(1)
    intent = query("Parse the user's request. target_cell may only be a verbatim cell name/code from the request or unspecified. Do not infer a protein identity. Request: " + instruction, Intent, store, run_id)
    cell = intent.target_cell
    if cell != "unspecified" and cell not in instruction:
        cell = "unspecified"
    if intent.mode != "full" or re.search(r"仅\s*(?:CDS|UTR)|(?:CDS|UTR)[ -]only|诊断|diagnos",instruction,re.I):
        raise ValueError("Partial/diagnostic modes require a structured brief with explicitly partitioned UTR/CDS reference; use the advanced input")
    count_match = re.search(r"(?:count|candidates|候选数|候选)\s*[:：=]?\s*(\d+)", instruction, re.I)
    count = min(64, max(1, int(count_match.group(1)))) if count_match else 8
    # The model cannot override unambiguous endpoint/chemistry wording.
    endpoint = intent.endpoint
    for pattern, name in [(r"累计|\bAUC\b", "protein_auc"), (r"峰值|\bpeak\b", "protein_peak"),
                          (r"持续时间|duration", "functional_duration"), (r"翻译效率|translation efficiency", "translation_efficiency")]:
        if re.search(pattern,instruction,re.I):endpoint=name;break
    modification = "m1Psi" if re.search(r"m1[Ψψ]|m1psi|N1.methyl",instruction,re.I) else "unmodified" if re.search(r"未修饰|unmodified",instruction,re.I) else "unspecified"
    return Brief(name="Instruction design", protein=protein, application=instruction,
                 context=Context(target_cell=cell, species=intent.species),
                 molecule=MolecularConfig(modification=modification), budget=Budget(candidates=count),
                 objectives=[Objective(metric=endpoint, stage="computational" if endpoint == "translation_efficiency" else "experimental")],
                 assumptions=[intent.rationale, "Unspecified cap, delivery, dose and assay remain unknown"])
