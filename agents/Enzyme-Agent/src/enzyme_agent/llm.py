"""The same 0.8B Qwen as peptide-agent; proposes intent, never scientific values."""
import json
import re
from typing import Literal
from pydantic import Field
from .contracts import Contract, Brief, Reference, Reaction, Budget, Sequence
from .models import MODELS, model_path
from .templates import template_brief
from .config import data_root


class Intent(Contract):
    action: Literal["research", "design", "handoff_drug"]
    template: Literal["fabry", "pku", "cf_dnase"] | None = None
    disease: str = Field(default="unspecified", max_length=200)
    enzyme_name: str = Field(default="user supplied enzyme", max_length=200)
    sequence: Sequence | None = None
    ec: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+\.\d+$")
    generator: Literal["esm3", "ligandmpnn", "proteinmpnn", "zymctrl", "library"] = "esm3"
    count: int = Field(default=4, ge=1, le=16)
    design_positions: list[int] = Field(default_factory=list, max_length=128)
    bottleneck: Literal["catalysis", "selectivity", "stability", "expression", "processing", "localization", "mechanism"] = "stability"


class RoutingIntent(Contract):
    action: Literal['research','design','handoff_drug']
    template: Literal['fabry','pku','cf_dnase','none']
    generator: Literal['esm3','ligandmpnn','proteinmpnn','zymctrl','library']
    count: int = Field(ge=1,le=16)
    bottleneck: Literal['catalysis','selectivity','stability','expression','processing','localization','mechanism']


def materialize(intent, instruction):
    if intent.sequence and intent.sequence not in instruction:
        raise ValueError("Qwen invented a sequence; only verbatim supplied sequences are accepted")
    if intent.ec and intent.ec not in instruction:
        raise ValueError("EC identifiers must occur verbatim in the instruction or come from the curated template")
    if any(not re.search(r'(?<!\d)'+str(p)+r'(?!\d)',instruction) for p in intent.design_positions):
        raise ValueError("Planned residue indices must occur in the instruction")
    if intent.action == "handoff_drug":
        return {"handoff": "drug-agent", "reason": "Inhibitor/small-molecule design targets an enzyme rather than using an enzyme as therapeutic payload"}
    brief = template_brief(intent.template) if intent.template else Brief(entry="disease", route="research", generators=["library"], disease=intent.disease)
    values = brief.model_dump()
    values.update(bottleneck=intent.bottleneck, design_positions=intent.design_positions,
                  budget=Budget(candidates=intent.count).model_dump())
    if intent.sequence:
        values["reference"] = Reference(name=intent.enzyme_name, sequence=intent.sequence, sources=["user instruction"]).model_dump()
        values["entry"] = "enzyme"
    elif intent.template:
        ref_file = data_root() / "references" / (intent.template + "_reference.json")
        if ref_file.is_file():
            values["reference"] = Reference.model_validate_json(ref_file.read_text()).model_dump()
    if intent.ec:
        values["reaction"]["ec"] = intent.ec
        values["entry"] = "reaction"
    if intent.action == "design":
        values["route"] = "family_expansion" if intent.generator == "zymctrl" else "conservative"
        values["generators"] = [intent.generator]
    return {"brief": Brief.model_validate(values).model_dump(), "intent": intent.model_dump()}


def plan(instruction):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from .decoding import prefix_function, tokenizer_data
    if not 1 <= len(instruction) <= 12000:
        raise ValueError("Instruction must be 1..12000 characters")
    path = model_path("qwen")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True, trust_remote_code=False, dtype="auto").eval().to("cuda" if torch.cuda.is_available() else "cpu")
    schema = RoutingIntent.model_json_schema()
    # Ground biological identifiers at decoding time. A 0.8B planner otherwise
    # tends to fill optional schema examples with invented sequences/EC classes.
    supplied_sequences=re.findall(r'(?<![A-Z])[ACDEFGHIKLMNPQRSTVWY]{20,1500}(?![A-Z])',instruction)
    supplied_ec=re.findall(r'\b\d+\.\d+\.\d+\.\d+\b',instruction)
    if len(set(supplied_sequences))>1 or len(set(supplied_ec))>1:
        raise ValueError('Ambiguous multiple parent sequences/EC numbers; use a structured brief')
    position_match=re.search(r'(?:design_positions|positions|设计位点|位点)\s*[:：=]\s*([0-9,，\s]+)',instruction)
    numbers=[int(n) for n in re.findall(r'\d+',position_match[1])] if position_match else []
    system = ("You plan therapeutic ENZYME research. Return one JSON object matching the schema. "
              "Disease route exploration uses action=research. Explicit generation/optimization uses action=design. "
              "Small molecule enzyme INHIBITORS use action=handoff_drug. "
              "Templates: Fabry/法布雷病=fabry; PKU/苯丙酮尿症=pku; cystic fibrosis/DNase/囊性纤维化=cf_dnase. "
              "Never invent sequences, measurements, EC numbers or protected positions. "
              'For disease-only research an example is {"action":"research","template":"cf_dnase","generator":"library","count":4,"bottleneck":"mechanism"}. '
              "Use template=none for other diseases. Default count=4 unless specified. "
              "Sequence and EC extraction are handled separately; do not output them. "
              "Use esm3 for masked engineering; ligandmpnn for atomic-context redesign; zymctrl for EC family expansion. Schema: " + json.dumps(schema))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    tokenizer = tokenizer_data(tok)
    attempts = []
    for attempt in range(2):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] > 8192:
            raise ValueError("Planning context exceeds 8192 tokens")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1024, max_time=90, do_sample=False,
                                    prefix_allowed_tokens_fn=prefix_function(tokenizer, schema))
        reply = tok.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        attempts.append(reply)
        try:
            routing = RoutingIntent.model_validate_json(reply)
            intent = Intent(action=routing.action,template=None if routing.template=='none' else routing.template,
                generator=routing.generator,count=routing.count,bottleneck=routing.bottleneck,
                sequence=supplied_sequences[0] if supplied_sequences else None,ec=supplied_ec[0] if supplied_ec else None,
                design_positions=numbers,disease=instruction[:200] if routing.template=='none' else 'template')
            data = materialize(intent, instruction)
            return {**data, "provenance": {**MODELS["qwen"], "decoding": "JSON schema constrained; strict semantic validation", "attempts": attempts}}
        except ValueError as exc:
            messages.extend([{"role": "assistant", "content": reply}, {"role": "user", "content": "Correct the schema error: " + str(exc)[:1500]}])
    raise ValueError("Qwen intent rejected after two attempts; replies="+repr(attempts))
