"""Pinned local models. Downloads are an explicit CLI operation, never a tool side effect."""
from __future__ import annotations

import os
from pathlib import Path

from .contracts import AA, Generate, ToolResult, Unavailable

MODELS = {
    "qwen": {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
    "pepmlm": {"repo": "ChatterjeeLab/PepMLM-650M", "revision": "898fca941a9057aebdd1a6164b5ee09a1a71780e"},
}


def model_root() -> Path:
    return Path(os.environ.get("PEPTIDE_MODEL_DIR", ".peptide-agent/models")).resolve()


def model_path(name: str) -> Path:
    path = model_root() / name / MODELS[name]["revision"]
    if not (path / "config.json").is_file() or not any(
        p.suffix in {".bin", ".safetensors"} for p in path.glob("*")):
        raise Unavailable(f"Missing {name} weights. Run: peptide-agent download {name}")
    return path


def download(name: str):
    from huggingface_hub import snapshot_download
    info = MODELS[name]
    return snapshot_download(info["repo"], revision=info["revision"],
                             local_dir=model_root() / name / info["revision"],
                             ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot"])


class PepMLM:
    def __init__(self):
        self.model = self.tokenizer = None

    def generate(self, request: Generate) -> ToolResult:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        path = model_path("pepmlm")
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True,
                                                              trust_remote_code=False).eval()
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        tok, model = self.tokenizer, self.model
        text = request.target_sequence + tok.mask_token * request.length
        inputs = tok(text, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] > model.config.max_position_embeddings:
            raise ValueError("Target + peptide exceeds PepMLM context; explicit domain selection required")
        ids = tok.convert_tokens_to_ids(list(AA))
        if len(set(ids)) != 20 or tok.unk_token_id in ids:
            raise ValueError("Tokenizer does not map the canonical amino acid alphabet")
        positions = (inputs.input_ids[0] == tok.mask_token_id).nonzero().flatten()
        if len(positions) != request.length:
            raise ValueError("PepMLM mask length mismatch")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, positions][:, ids].float().cpu()
        # Upstream span-mask generation: one masked-span forward pass, top-k sampling per position.
        values, indices = logits.topk(request.top_k, dim=-1)
        probs = values.softmax(dim=-1)
        rng = torch.Generator(device="cpu").manual_seed(request.seed)
        sequences = []
        for _ in range(request.count):
            choices = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
            selected = indices.gather(1, choices[:, None]).squeeze(-1)
            sequences.append("".join(AA[i] for i in selected.tolist()))
        return ToolResult(tool="generation.pepmlm", status="ok", trust="PREDICTED",
                          data={"sequences": sequences},
                          provenance={**MODELS["pepmlm"], "seed": request.seed, "top_k": request.top_k,
                                      "device": str(model.device), "canonical_token_filter": True,
                                      "sampling": "simultaneous masked span; top-k softmax"},
                          warnings=["Generated sequences are unvalidated binders; no affinity is inferred."])
