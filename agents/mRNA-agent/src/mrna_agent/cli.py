from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import yaml
from .contracts import Brief
from .storage import Store, data_root


def load(path):
    return yaml.safe_load(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(prog="mrna-agent")
    parser.add_argument("--root", default=None, help="Run journal directory (default MRNA_DATA_DIR/runs)")
    subs=parser.add_subparsers(dest="command",required=True)
    subs.add_parser("doctor")
    p=subs.add_parser("run");p.add_argument("brief");p.add_argument("--fixed-plan",action="store_true")
    p=subs.add_parser("instruct");p.add_argument("instruction")
    p=subs.add_parser("resume");p.add_argument("run_id");p.add_argument("--fixed-plan",action="store_true")
    p=subs.add_parser("reevaluate");p.add_argument("run_id")
    subs.add_parser("history")
    p=subs.add_parser("show");p.add_argument("run_id")
    p=subs.add_parser("export");p.add_argument("run_id");p.add_argument("--format", choices=["json","fasta","csv"],default="json");p.add_argument("--output")
    p=subs.add_parser("feedback");p.add_argument("run_id");p.add_argument("--data")
    p=subs.add_parser("calibrate");p.add_argument("data")
    p=subs.add_parser("benchmark");p.add_argument("--evolve",action="store_true")
    subs.add_parser("rollback")
    p=subs.add_parser("serve");p.add_argument("--port",type=int,default=2005)
    subs.add_parser("mcp")
    args=parser.parse_args()
    if args.command=="serve":
        import uvicorn
        from .web import create_app
        uvicorn.run(create_app(args.root),host="127.0.0.1",port=args.port);return
    if args.command=="mcp":
        from .mcp_server import serve
        serve(args.root);return
    with Store(args.root) as store:
        from .workflow import Engine
        if args.command=="doctor":
            from .registry import capabilities
            result=capabilities()
        elif args.command=="run": result=Engine(store).run(Brief.model_validate(load(args.brief)),planner=not args.fixed_plan)
        elif args.command=="instruct": result=Engine(store).run(instruction=args.instruction)
        elif args.command=="resume":
            old=store.get(args.run_id);store.update(args.run_id,"queued",old["payload"])
            result=Engine(store).resume(args.run_id,planner=not args.fixed_plan)
        elif args.command=="reevaluate": result=Engine(store).reevaluate(args.run_id)
        elif args.command=="history": result=store.list_runs()
        elif args.command=="show": result={**store.get(args.run_id),"events":store.events(args.run_id)}
        elif args.command=="export":
            from .exports import export
            text,_=export(store.get(args.run_id),args.format)
            if args.output:Path(args.output).write_text(text)
            else:print(text)
            return
        elif args.command=="feedback":
            from .feedback import import_experiments, analyze_feedback
            if args.data: import_experiments(store,load(args.data))
            from .delivery import public_feedback
            result=public_feedback(analyze_feedback(store,args.run_id),store.get(args.run_id))
        elif args.command=="calibrate":
            from .feedback import fit_calibration
            result=fit_calibration(load(args.data));store.artifact(result)
        elif args.command=="benchmark":
            from .benchmark import evolve, benchmark
            result=evolve(store) if args.evolve else benchmark(store,versions=("v1",))
        elif args.command=="rollback":
            from .benchmark import rollback
            result=rollback(store)
        if args.command in {"run","instruct","resume","reevaluate","show"}:
            from .delivery import public_run
            result=public_run(result)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        if isinstance(result,dict) and result.get("state") in {"failed","blocked","budget_exhausted"}:
            sys.exit(2)


if __name__=="__main__":main()
