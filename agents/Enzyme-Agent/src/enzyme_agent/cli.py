import argparse
import json
from pathlib import Path
import sys
import yaml
from .config import data_root
from .contracts import Brief
from .store import Store
from .workflow import Engine
from .exports import nominated_fasta, validated_handoffs


def load(path):
    return yaml.safe_load(Path(path).read_text())


def output(value):
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    parser=argparse.ArgumentParser(prog='enzyme-agent')
    parser.add_argument('--store',default=None,help='Run store (default ENZYME_DATA_DIR/runs)')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','feedback','evaluate','calibrate'):
        p=sub.add_parser(name);p.add_argument('file')
        if name=='feedback':p.add_argument('--run-id',required=True)
        if name=='calibrate':
            p.add_argument('--model-id',required=True);p.add_argument('--split-mode',default='cross_family')
    p=sub.add_parser('chat');p.add_argument('instruction')
    for name in ('resume','report','next-round','evolve'):
        p=sub.add_parser(name);p.add_argument('run_id')
        if name=='report': p.add_argument('--format',choices=['json','markdown','fasta','mrna'],default='json')
    p=sub.add_parser('serve');p.add_argument('--port',type=int,default=2006)
    p=sub.add_parser('download');p.add_argument('model',choices=['core','qwen','esm3','ligandmpnn','proteinmpnn','zymctrl','esmfold2','rfdiffusion3','thermompnn'])
    sub.add_parser('doctor');sub.add_parser('mcp');sub.add_parser('list')
    p=sub.add_parser('template');p.add_argument('name',choices=['fabry','pku','cf_dnase'])
    args=parser.parse_args()
    if args.command=='serve':
        import uvicorn
        from .web import create_app
        uvicorn.run(create_app(args.store),host='127.0.0.1',port=args.port)
        return
    if args.command=='download':
        from .models import download
        for model in ['qwen','ligandmpnn','esm3','zymctrl','esmfold2'] if args.model=='core' else [args.model]:
            result=download(model);output({k:v for k,v in result.items() if k!='files'})
        return
    if args.command=='doctor':
        from .models import capabilities
        output(capabilities());return
    if args.command=='mcp':
        from .mcp_server import serve
        serve(args.store);return
    if args.command=='template':
        from .templates import template_brief
        output(template_brief(args.name).model_dump());return
    if args.command=='calibrate':
        from .benchmark import calibrate
        output(calibrate(Path(args.file).read_text(),args.model_id,args.split_mode));return
    if args.command=='evaluate':
        from .benchmark import pipeline_metrics
        output(pipeline_metrics(load(args.file)));return
    with Store(args.store) as store:
        engine=Engine(store)
        if args.command=='list':output(store.list_runs());return
        if args.command=='run':run=engine.run(Brief.model_validate(load(args.file)))
        elif args.command=='chat':run=engine.chat(args.instruction)
        elif args.command=='resume':
            run=store.get(args.run_id)
            if run['state']=='cancelled':store.update(args.run_id,'queued',run['payload'])
            run=engine.resume(args.run_id)
        elif args.command=='feedback':
            from .feedback import ingest
            values=load(args.file);output(ingest(store,args.run_id,values['observations'] if isinstance(values,dict) else values));return
        elif args.command=='next-round':
            from .feedback import next_brief
            output(next_brief(store,args.run_id));return
        elif args.command=='evolve':
            from .evolution import compare_policies
            output(compare_policies(store.get(args.run_id)));return
        elif args.command=='report':
            run=store.get(args.run_id)
            if args.format=='json':output({**run,'events':store.events(args.run_id),'observations':store.observations(args.run_id)})
            elif args.format=='markdown':print(run['payload'].get('report','No report yet'))
            elif args.format=='fasta':print(nominated_fasta(run['payload']))
            else:output(validated_handoffs(run['payload']))
            return
        output({'id':run['id'],'state':run['state'],'error':run['payload'].get('error'),
                'candidates':len(run['payload'].get('candidates',[])),'report_sha256':run['payload'].get('report_sha256')})
        if run['state'] in {'failed','blocked'}:raise SystemExit(1)


if __name__=='__main__':main()
