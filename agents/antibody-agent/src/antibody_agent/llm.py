"""0.8B intent routing; biological entities are extracted verbatim by code."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import Field

from .contracts import AntigenState, Binder, Brief, Budget, Chain, Contract, Gate, Product, Unavailable
from .registry import QWEN, qwen_path


class RoutingIntent(Contract):
    entry: Literal["disease", "target", "sequence", "complex", "feedback"]
    route: Literal["research", "de_novo", "optimize", "pairing", "humanize", "library"]
    direction: Literal["block", "agonize", "clear", "bridge", "deliver", "unresolved"]
    generator: Literal["rfantibody", "antifold", "igdesign", "lichen", "sapiens", "library"]
    count: int = Field(ge=1, le=16)


def materialize(intent, instruction):
    chains = []
    for role in ("VH", "VL", "VHH"):
        matches = re.findall(
            r"(?<![A-Za-z])" + role + r"\s*[:=：]\s*([ACDEFGHIKLMNPQRSTVWY]{20,3000})(?![A-Z])", instruction
        )
        if len(matches) > 1:
            raise ValueError("Multiple sequences for the same chain; use a structured brief")
        if matches:
            chains.append(Chain(role=role, sequence=matches[0]))
    product = (
        Product(format="VHH", valency=1, fc="none", fc_requirement="none", chain_organization="single VHH")
        if any(c.role == "VHH" for c in chains)
        else Product()
    )
    reference = (
        Binder(
            chains=chains, pairing_source="user_supplied", source="Verbatim user instruction", product=product
        )
        if chains
        else None
    )
    target = re.search(r"(?:target|靶点)\s*[:=：]\s*([^\n;；]+)", instruction, re.I)
    antigen = re.search(
        r"(?:antigen_sequence|抗原序列)\s*[:=：]\s*([ACDEFGHIKLMNPQRSTVWY]{20,3000})(?![A-Z])", instruction
    )
    states = (
        [
            AntigenState(
                id="target",
                entity=target[1].strip()[:200],
                sequence=antigen[1] if antigen else None,
                sources=["user instruction"],
            )
        ]
        if target
        else []
    )
    brief = Brief(
        name="Antibody research · Qwen 0.8B",
        entry=intent.entry,
        route=intent.route,
        disease=instruction[:2000],
        mechanism="User instruction; hypothesis requires evidence",
        direction=intent.direction,
        direction_gate=Gate(
            status="hypothesis", rationale="Qwen routing from the user instruction; not verified"
        ),
        states=states,
        reference=reference,
        product=product,
        generators=[intent.generator],
        budget=Budget(candidates=intent.count, shortlist=min(4, intent.count)),
    )
    if intent.route == "library" and reference:
        brief.library = [reference]
    return {
        "brief": brief.model_dump(),
        "intent": intent.model_dump(),
        "planner": QWEN,
        "limitation": "Only routing is inferred; identifiers/sequences are verbatim and scientific evidence remains unknown",
    }


def plan(instruction):
    if not isinstance(instruction, str) or not 1 <= len(instruction) <= 12000:
        raise ValueError("Instruction must be 1..12000 characters")
    path = qwen_path()
    if not (path / "config.json").is_file():
        raise Unavailable("Qwen3.5-0.8B weights missing; set ANTIBODY_QWEN_DIR or run download-qwen")
    cfg = json.loads((path / "config.json").read_text())
    text_config = cfg.get("text_config", cfg)
    # The expected 0.8B architecture: reject accidental substitution of a larger checkpoint.
    if text_config.get("hidden_size") != 1024 or text_config.get("num_hidden_layers") != 24:
        raise ValueError("Planner checkpoint does not match the pinned Qwen3.5-0.8B architecture")
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from .decoding import prefix_function, tokenizer_data

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype="auto"
    )
    model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    schema = RoutingIntent.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": (
                "Route therapeutic antibody RESEARCH. Return only JSON matching the schema. "
                "Disease/mechanism exploration: route=research. Explicit de novo design: rfantibody. "
                "Structure-constrained optimization: antifold or igdesign. Humanization: sapiens. "
                "Paired chain repair: lichen. Supplied candidate analysis: library. "
                "Never generate sequences, biological identifiers, citations, or scientific scores. "
                "Use count=4 unless specified. Unknown mechanism: direction=unresolved. " + json.dumps(schema)
            ),
        },
        {"role": "user", "content": instruction},
    ]
    enforcement = tokenizer_data(tokenizer)
    for _ in range(2):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] > 8192:
            raise ValueError("Planner context exceeds 8192 tokens")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
                max_time=90,
                prefix_allowed_tokens_fn=prefix_function(enforcement, schema),
            )
        reply = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        try:
            return materialize(RoutingIntent.model_validate_json(reply), instruction)
        except ValueError as exc:
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Correct the schema: " + str(exc)[:600]},
            ]
    raise ValueError("Qwen intent rejected after two validation attempts")
