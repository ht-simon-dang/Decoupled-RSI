from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import yaml

from .contracts import Brief, Unavailable
from .models import MODELS, download, model_path
from .store import Store
from .workflow import Engine


def doctor():
    import torch
    from .capabilities import capabilities
    models = {}
    for name in MODELS:
        try:
            models[name] = {"status": "installed", "path": str(model_path(name)), **MODELS[name]}
        except Unavailable as exc:
            models[name] = {"status": "unavailable", "reason": str(exc), **MODELS[name]}
    return {"python": sys.executable, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "models": models,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("peptide-agent", "torch", "transformers", "biopython", "mcp")},
            "capabilities": capabilities(),
            "scope": "canonical linear/head-to-tail peptides; calibrated affinity requires measured training data"}


def main():
    parser = argparse.ArgumentParser(prog="peptide-agent")
    parser.add_argument("--store", default=".peptide-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("mcp")
    sub.add_parser("skills")
    p = sub.add_parser("download")
    p.add_argument("model", choices=[*MODELS, "boltzgen", "toxinpred3", "all"])
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=2004)
    p = sub.add_parser("run")
    p.add_argument("brief", type=Path)
    p = sub.add_parser("chat")
    p.add_argument("instruction")
    p.add_argument("--execute", action="store_true", help="Execute the validated plan")
    for command in ("resume", "report"):
        p = sub.add_parser(command)
        p.add_argument("run_id")
    p = sub.add_parser("benchmark")
    p.add_argument("csv", type=Path)
    args = parser.parse_args()
    store = None
    try:
        if args.command == "serve":
            import uvicorn
            from .web import create_app
            uvicorn.run(create_app(args.store), host="127.0.0.1", port=args.port)
            return
        if args.command == "mcp":
            # MCP has a fixed workspace-local store. CLI --store is not an MCP-controlled path.
            from .mcp_server import main as serve
            serve()
            return
        if args.command == "doctor":
            result = doctor()
        elif args.command == "download":
            from .downloads import download_backend
            result = {name: download_backend(name) for name in ([*MODELS, "boltzgen", "toxinpred3"] if args.model == "all" else [args.model])}
        elif args.command == "skills":
            from .gateway import POLICY
            result = POLICY
        elif args.command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(args.csv)
        else:
            store = Store(args.store)
            engine = Engine(store)
            if args.command == "run":
                result = engine.run(Brief.model_validate(yaml.safe_load(args.brief.read_text())))
            elif args.command == "resume":
                result = engine.resume(args.run_id)
            elif args.command == "report":
                result = {**store.get(args.run_id), "events": store.events(args.run_id)}
            else:
                from .llm import plan
                planned = plan(args.instruction, store)
                result = engine.run(planned.brief, planned.skill) if args.execute else planned.model_dump()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if isinstance(result, dict) and result.get("state") in {"blocked", "failed"}:
            raise SystemExit(2)
    except (ValueError, OSError, Unavailable, PermissionError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    finally:
        if store:
            store.db.close()


if __name__ == "__main__":
    main()
