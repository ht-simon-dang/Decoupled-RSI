"""Bounded tools, persistent budgets, explicit failures and pinned asset checks."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contracts import Brief, canonical
from .registry import manifest
from .storage import data_root


class BudgetExhausted(RuntimeError):
    pass


class Gateway:
    def __init__(self, store, run_id, brief: Brief, elapsed_before=0):
        self.store, self.run_id, self.brief = store, run_id, brief
        self.started = time.monotonic()
        events = store.events(run_id)
        self.calls = sum(e["kind"] == "tool.started" for e in events)
        self.prior_seconds = sum(e["payload"].get("seconds", 0) for e in events if e["kind"] == "tool.finished")
        self.prior_seconds = max(self.prior_seconds, elapsed_before)

    def remaining(self):
        return self.brief.budget.seconds - self.prior_seconds - (time.monotonic() - self.started)

    def call(self, tool, arguments):
        if tool not in {"gemorna", "lineardesign", "ribonn", "parade", "smart", "codongpt", "utrgan", "saluki", "utr_lm", "rnadegformer", "mrnabert", "mrnautilus", "degscore"}:
            raise ValueError("Tool is not in fixed scientific allowlist")
        if self.calls >= self.brief.budget.tool_calls or self.remaining() <= 0:
            raise BudgetExhausted("Persistent scientific tool-call/time budget exhausted")
        self.calls += 1
        self.store.event(self.run_id, "tool.started", {"tool": tool, "arguments": arguments, "call": self.calls})
        started = time.monotonic()
        root = data_root()
        try:
            provenance = self.verify_assets(tool)
            if tool == "lineardesign":
                path = root / "external/LinearDesign"
                cmd = [str(path / "bin/LinearDesign_2D"), str(arguments["lambda"]), "0", str(path / "codon_usage_freq_table_human.csv")]
                stdin, cwd = arguments["protein"] + "\n", path
            else:
                interpreter = root / "envs/gemorna/bin/python" if tool in {"gemorna", "smart", "utrgan", "saluki"} else root / "envs/transformers4/bin/python" if tool in {"mrnabert", "mrnautilus"} else Path(sys.executable)
                cmd = [str(interpreter), str(Path(__file__).with_name("scientific_worker.py"))]
                stdin, cwd = canonical({"tool": tool, "arguments": arguments}), root
            output = subprocess.run(cmd, input=stdin, text=True, capture_output=True, cwd=cwd,
                                    timeout=max(1, min(600, self.remaining())),
                                    env=dict(os.environ, MRNA_DATA_DIR=str(root), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1"))
            raw_id = self.store.artifact({"stdout": output.stdout, "stderr": output.stderr, "returncode": output.returncode})
            if tool == "lineardesign":
                import re
                match = re.search(r"mRNA sequence:\s*([ACGU]+)", output.stdout)
                if output.returncode or not match:
                    raise RuntimeError("LinearDesign failed; raw artifact " + raw_id)
                result = {"status": "ok", "data": {"sequences": [{"sequence": match[1]}], "raw_artifact": raw_id}}
            else:
                try:
                    result = json.loads(output.stdout)
                except ValueError as exc:
                    raise RuntimeError("Scientific worker returned invalid JSON; raw artifact " + raw_id) from exc
                if output.returncode and result.get("status") == "ok":
                    raise RuntimeError("Worker exited unsuccessfully")
            result.update(provenance=provenance, raw_artifact=raw_id)
        except FileNotFoundError as exc:
            result = {"status": "unavailable", "reason": str(exc)}
        except subprocess.TimeoutExpired:
            result = {"status": "error", "reason": "Scientific worker exceeded bounded time budget"}
        except Exception as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        result["tool"] = tool
        result["seconds"] = time.monotonic() - started
        self.store.event(self.run_id, "tool.finished", result)
        return result

    def verify_assets(self, tool):
        root = data_root()
        optional={"degscore":"external/DegScore/DegScore.py","codongpt":"models/codongpt/pytorch_model.bin","mrnabert":"models/mrnabert/pytorch_model.bin","mrnautilus":"models/mrnautilus/best.ckpt","utrgan":"external/UTRGAN/models/checkpoint_3000.h5","saluki":"models/saluki/f0_c0/train/model0_best.h5","rnadegformer":"external/RNAdegformer-Webapp/RNA_Inference/best_weights/fold0top1.ckpt","utr_lm":"external/UTR-LM/Scripts/esm/model/esm2_secondarystructure.py"}
        names = {"gemorna": ["external/GEMORNA/checkpoints/gemorna_cds.pt", "external/GEMORNA/checkpoints/gemorna_5utr.pt", "external/GEMORNA/checkpoints/gemorna_3utr.pt"],
                 "ribonn": ["models/ribonn/human/runs.csv"],
                 "parade": ["external/parade/predictor/regression_multiple/saved_models/model-utr5-deltas-epoch=9-step=840.ckpt", "external/parade/predictor/regression_multiple/saved_models/model-utr3-deltas-epoch=9-step=1330.ckpt"],
                 "smart": ["external/Smart5UTR/models/Optimus5-Prime/Optimus5_egfp_m1pseudo2_model_200k.h5"],
                 "lineardesign": ["external/LinearDesign/bin/LinearDesign_2D", "external/LinearDesign/codon_usage_freq_table_human.csv"]}.get(tool, [optional[tool]] if tool in optional else [])
        if tool in optional:
            validation_path=root/"model-validation.json"
            validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
            if not validation.get(tool,{}).get("passed"):
                raise FileNotFoundError("Optional model has not passed real inference admission: "+tool)
        provenance = {}
        records = manifest()
        for name in names:
            path = root / name
            if not path.is_file() or path.stat().st_size < 300:
                raise FileNotFoundError("Missing scientific asset: " + name)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if name in records and records[name].get("sha256") and digest != records[name]["sha256"]:
                raise RuntimeError("Scientific asset checksum changed: " + name)
            provenance[name] = digest
        provenance["adapter"] = "mrna-agent-0.2.0"
        return provenance
