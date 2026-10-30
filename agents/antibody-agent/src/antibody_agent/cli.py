from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import Brief
from .delivery import public_run
from .store import Store


def load(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="antibody-agent", description="EMA4D antibody research with evaluation before human review"
    )
    parser.add_argument("--root", help="Run journal directory; default ANTIBODY_DATA_DIR/runs")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "history", "schema", "download-qwen", "mcp"):
        subs.add_parser(name)
    p = subs.add_parser("run")
    p.add_argument("brief")
    p = subs.add_parser("instruct")
    p.add_argument("instruction")
    for name in ("show", "resume", "reevaluate", "benchmark"):
        p = subs.add_parser(name)
        p.add_argument("run_id")
    p = subs.add_parser("export")
    p.add_argument("run_id")
    p.add_argument("--format", choices=["json", "fasta", "csv", "report"], default="json")
    p.add_argument("--output")
    p = subs.add_parser("feedback")
    p.add_argument("run_id")
    p.add_argument("--data")
    p = subs.add_parser("train-property")
    p.add_argument("data")
    p = subs.add_parser("next-batch")
    p.add_argument("run_id")
    p.add_argument("data")
    p = subs.add_parser("review")
    p.add_argument("run_id")
    p.add_argument("candidate_id")
    p.add_argument("decision", choices=["accept", "reject", "request_changes"])
    p.add_argument("--note", required=True)
    p = subs.add_parser("serve")
    p.add_argument("--port", type=int, default=2007)
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            import uvicorn

            from .web import create_app

            uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)
            return 0
        if args.command == "mcp":
            from .mcp_server import serve

            serve(args.root)
            return 0
        if args.command == "download-qwen":
            from huggingface_hub import snapshot_download

            from .registry import QWEN, qwen_path

            path = snapshot_download(
                repo_id=QWEN["repo"],
                revision=QWEN["revision"],
                local_dir=qwen_path(),
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
            )
            result = {"path": path, **QWEN}
        elif args.command == "doctor":
            from .registry import capabilities

            result = capabilities()
        elif args.command == "schema":
            result = Brief.model_json_schema()
        elif args.command == "train-property":
            from .learning import fit

            result = fit(load(args.data))
        else:
            from .workflow import Engine

            with Store(args.root) as store:
                if args.command == "run":
                    result = public_run(Engine(store).run(Brief.model_validate(load(args.brief))))
                elif args.command == "instruct":
                    result = public_run(Engine(store).run(instruction=args.instruction))
                elif args.command == "show":
                    run = store.get(args.run_id)
                    run["events"] = store.events(args.run_id)
                    result = public_run(run)
                elif args.command == "resume":
                    Engine(store).request_resume(args.run_id)
                    result = public_run(Engine(store).resume(args.run_id))
                elif args.command == "reevaluate":
                    result = public_run(Engine(store).reevaluate(args.run_id))
                elif args.command == "history":
                    result = store.list_runs()
                elif args.command == "benchmark":
                    from .benchmark import compare

                    result = compare(store.get(args.run_id))
                elif args.command == "export":
                    from .exports import export

                    content, _ = export(store.get(args.run_id), args.format)
                    if args.output:
                        Path(args.output).write_text(content)
                    else:
                        print(content, end="")
                    return 0
                elif args.command == "feedback":
                    from .feedback import analyze, ingest

                    result = (
                        ingest(store, args.run_id, load(args.data)["observations"])
                        if args.data
                        else analyze(store, args.run_id)
                    )
                elif args.command == "review":
                    from .feedback import review

                    result = review(store, args.run_id, args.candidate_id, args.decision, args.note)
                elif args.command == "next-batch":
                    from .learning import predict_batch

                    result = predict_batch(store.get(args.run_id), load(args.data))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return (
            2
            if isinstance(result, dict) and result.get("state") in {"failed", "blocked", "budget_exhausted"}
            else 0
        )
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
