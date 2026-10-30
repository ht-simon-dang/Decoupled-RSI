"""Local research workbench, process queue, audit API and explicit feedback."""
from __future__ import annotations
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from .exports import nominated_fasta, validated_handoffs
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import Field, model_validator

from .config import data_root
from .contracts import Brief, Contract, Observation, Reference, Sequence
from .models import capabilities
from .store import Store, TERMINAL
from .templates import TEMPLATES, template_brief
from .feedback import ingest, next_brief
from .benchmark import pipeline_metrics
from .evolution import compare_policies


class Submission(Contract):
    brief: Brief | None = None
    instruction: str | None = Field(default=None,min_length=1,max_length=12000)
    @model_validator(mode='after')
    def either(self):
        if (self.brief is None)==(self.instruction is None): raise ValueError('Provide a brief or an instruction')
        return self


class FeedbackSubmission(Contract):
    observations: list[Observation] = Field(min_length=1,max_length=128)


class CalibrationSubmission(Contract):
    model_id: str = Field(pattern=r'^[a-z0-9][a-z0-9_-]{0,63}$')
    csv: str = Field(min_length=1,max_length=2_000_000)
    split_mode: str = 'cross_family'


class PredictionSubmission(Contract):
    model_id: str = Field(pattern=r'^[a-z0-9][a-z0-9_-]{0,63}$')
    sequences: list[Sequence] = Field(min_length=1,max_length=32)
    endpoint: str
    unit: str
    environment_id: str
    conditions_id: str
    substrate_id: str
    count: int = Field(default=4,ge=1,le=16)
    direction: str = 'max'
    costs: list[float] | None = None
    cost_budget: float = Field(default=4.0,gt=0)


class Queue:
    def __init__(self,root):
        self.root=Path(root);self.pool=ThreadPoolExecutor(max_workers=1)
        self.lock=threading.RLock();self.pending=set();self.processes={}

    def submit(self,run_id):
        with self.lock:
            if run_id in self.pending: raise ValueError('Run already scheduled')
            if len(self.pending)>=32: raise ValueError('Local queue is full')
            self.pending.add(run_id)
        self.pool.submit(self.execute,run_id)

    def execute(self,run_id):
        try:
            with Store(self.root) as store:
                with self.lock:
                    run=store.get(run_id)
                    if run['state']=='cancelled': return
                    store.update(run_id,'running',run['payload'])
                    logs=self.root/'logs';logs.mkdir(exist_ok=True)
                    with (logs/f'{run_id}.log').open('w') as log:
                        proc=subprocess.Popen([sys.executable,'-m','enzyme_agent.web_worker',str(self.root),run_id],
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    self.processes[run_id]=proc
                try: proc.wait(timeout=14600)
                except subprocess.TimeoutExpired:
                    self.stop_process(proc)
                current=store.get(run_id)
                if current['state'] not in TERMINAL:
                    store.update(run_id,'failed',{**current['payload'],'error':'Worker stopped before completion; inspect execution log'})
        finally:
            with self.lock:
                self.pending.discard(run_id);self.processes.pop(run_id,None)

    @staticmethod
    def stop_process(proc):
        if proc.poll() is not None: return
        try:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        except ProcessLookupError: pass

    def cancel(self,run_id):
        with self.lock,Store(self.root) as store:
            run=store.get(run_id)
            if run['state'] in TERMINAL: raise ValueError('Run is already stopped')
            store.update(run_id,'cancelled',run['payload'])
            if run_id in self.processes: self.stop_process(self.processes[run_id])

    def close(self):
        for run_id in list(self.pending):
            try: self.cancel(run_id)
            except (ValueError,KeyError): pass
        self.pool.shutdown(wait=True,cancel_futures=True)


def create_app(root=None):
    root=Path(root or data_root()/'runs').resolve();root.mkdir(parents=True,exist_ok=True)
    queue=Queue(root)
    @asynccontextmanager
    async def lifespan(app):
        with (root/'.server.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Another WebUI already owns this run store')
            with Store(root) as store: store.recover()
            yield
            queue.close()
    app=FastAPI(title='Enzyme Agent',lifespan=lifespan)
    app.state.queue=queue
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','[::1]','testserver'])
    @app.middleware('http')
    async def local_boundary(request:Request,call_next):
        if request.method in {'POST','PUT','DELETE','PATCH'}:
            origin=request.headers.get('origin')
            if origin and urlsplit(origin).netloc!=request.headers.get('host'):
                return JSONResponse({'detail':'Cross-origin mutation rejected'},status_code=403)
            if request.headers.get('sec-fetch-site')=='cross-site':
                return JSONResponse({'detail':'Cross-site request rejected'},status_code=403)
            size=0;chunks=[]
            async for chunk in request.stream():
                size+=len(chunk)
                if size>6_000_000: return JSONResponse({'detail':'Request exceeds 6 MB'},status_code=413)
                chunks.append(chunk)
            request._body=b''.join(chunks)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    def read(run_id):
        with Store(root) as store:
            try: return {**store.get(run_id),'events':store.events(run_id),'observations':store.observations(run_id)}
            except KeyError as exc: raise HTTPException(404,'Run not found') from exc

    def launch(brief,payload):
        with Store(root) as store:
            run_id=store.create(brief,payload);store.update(run_id,'queued',payload)
            try: queue.submit(run_id)
            except ValueError as exc:
                store.update(run_id,'blocked',{**payload,'error':str(exc)})
                raise HTTPException(409,str(exc)) from exc
            return {'id':run_id,'state':'queued'}

    @app.get('/api/capabilities')
    def get_capabilities(): return capabilities()

    @app.get('/api/templates')
    def get_templates(): return TEMPLATES

    @app.get('/api/templates/{name}')
    def get_template(name:str):
        if name not in TEMPLATES: raise HTTPException(404,'Unknown template')
        b=template_brief(name).model_dump()
        path=data_root()/'references'/f'{name}_reference.json'
        if path.is_file(): b['reference']=Reference.model_validate_json(path.read_text()).model_dump()
        b['budget']['fold_candidates']=0
        return b

    @app.get('/api/runs')
    def list_runs(limit:int=100,offset:int=0):
        if not 1<=limit<=500 or offset<0: raise HTTPException(422,'Invalid pagination')
        with Store(root) as store: return store.list_runs(limit,offset)

    @app.post('/api/runs',status_code=202)
    def create_run(request:Submission):
        return launch(request.brief.model_dump() if request.brief else {},{} if request.brief else {'instruction':request.instruction})

    @app.get('/api/runs/{run_id}')
    def get_run(run_id:str): return read(run_id)

    @app.post('/api/runs/{run_id}/cancel')
    def cancel(run_id:str):
        read(run_id)
        try: queue.cancel(run_id)
        except ValueError as exc: raise HTTPException(409,str(exc)) from exc
        return read(run_id)

    @app.post('/api/runs/{run_id}/resume',status_code=202)
    def resume(run_id:str):
        with queue.lock:
            r=read(run_id)
            if r['state'] not in {'failed','blocked','cancelled'} or run_id in queue.pending:
                raise HTTPException(409,'Run is active, complete or still stopping')
            with Store(root) as store: store.update(run_id,'queued',r['payload'])
            queue.submit(run_id)
        return {'id':run_id,'state':'queued'}

    @app.get('/api/runs/{run_id}/log')
    def log(run_id:str):
        r=read(run_id);p=root/'logs'/f"{r['id']}.log"
        return {'log':p.read_text(errors='replace')[-16000:] if p.is_file() else ''}

    @app.get('/api/runs/{run_id}/download')
    def download(run_id:str,format:str='json'):
        r=read(run_id)
        if format=='json': body=json.dumps(r,indent=2,ensure_ascii=False);mime='application/json';ext='json'
        elif format=='markdown': body=r['payload'].get('report','Report unavailable');mime='text/markdown';ext='md'
        elif format=='fasta': body=nominated_fasta(r['payload']);mime='text/plain';ext='fasta'
        elif format=='mrna': body=json.dumps(validated_handoffs(r['payload']),indent=2);mime='application/json';ext='mrna.json'
        else: raise HTTPException(422,'Unknown report format')
        return Response(body,media_type=mime,headers={'Content-Disposition':f'attachment; filename="enzyme-{run_id}.{ext}"'})

    @app.get('/api/runs/{run_id}/structures/{candidate_id}')
    def structure(run_id:str,candidate_id:str):
        r=read(run_id);c=next((c for c in r['payload'].get('candidates',[]) if c['id']==candidate_id),None)
        if not c or not c.get('pdb'): raise HTTPException(404,'No independent structure for this candidate')
        return Response(c['pdb'],media_type='chemical/x-pdb',headers={'Content-Disposition':f'attachment; filename="{candidate_id}.pdb"'})

    @app.post('/api/runs/{run_id}/feedback')
    def feedback(run_id:str,request:FeedbackSubmission):
        read(run_id)
        try:
            with Store(root) as store: return ingest(store,run_id,[o.model_dump() for o in request.observations])
        except ValueError as exc: raise HTTPException(422,str(exc)) from exc

    @app.get('/api/runs/{run_id}/feedback-template')
    def feedback_template(run_id:str,candidate_id:str):
        from .evaluation import product_id
        r=read(run_id)
        c=next((c for c in r['payload'].get('candidates',[]) if c['id']==candidate_id),None)
        if not c: raise HTTPException(404,'Candidate not found')
        panel=r['brief'].get('panel',[]);envs=r['brief'].get('environments',[])
        return {'candidate_id':candidate_id,'product_id':product_id(c['product']),
            'endpoint':'REPLACE_WITH_MEASURED_ENDPOINT','layer':'disease_substrate','status':'measured','value':None,'unit':'REPLACE_WITH_UNIT',
            'conditions':{'assay_context':'REPLACE_WITH_EXACT_CONDITIONS'},'substrate_id':panel[0]['id'] if panel else 'REPLACE',
            'environment_id':envs[0]['id'] if envs else 'target','expression_host':'REPLACE','processing_state':'REPLACE',
            'batch':'','replicate':'','source':'','version':'1','reference_control':'','detection':'quantified'}

    @app.get('/api/runs/{run_id}/next-round')
    def next_round(run_id:str):
        read(run_id)
        with Store(root) as store: return next_brief(store,run_id)

    @app.get('/api/runs/{run_id}/evaluation')
    def evaluation(run_id:str): return pipeline_metrics(read(run_id))

    @app.get('/api/runs/{run_id}/evolution')
    def evolution(run_id:str): return compare_policies(read(run_id))

    @app.post('/api/calibrations',status_code=202)
    def calibration(request:CalibrationSubmission):
        if request.split_mode not in {'cross_family','within_family','substrate_transfer','reaction_transfer'}: raise HTTPException(422,'Invalid split mode')
        return launch({'name':'Function proxy · '+request.model_id},{'operation':'calibration',**request.model_dump()})

    @app.get('/api/calibrations')
    def calibrations():
        return [{k:d[k] for k in ('id','endpoint','unit','environment_id','conditions_id','substrates','audit','test','test_interval_coverage')} for p in (data_root()/'calibrations').glob('*.json')
                if not p.name.endswith('.dataset.json') for d in [json.loads(p.read_text())]]

    @app.post('/api/predictions')
    def predictions(request:PredictionSubmission):
        from .benchmark import predict
        from .active_learning import acquire
        try:
            args=request.model_dump()
            options={k:args.pop(k) for k in ('count','direction','costs','cost_budget')}
            values=predict(**args)
            result={'predictions':values,'acquisition':acquire(values,**options)}
            with Store(root) as store:
                run_id=store.create({'name':'Function prediction · '+request.model_id},{'operation':'prediction','request':request.model_dump()})
                sha=store.artifact(result)
                store.update(run_id,'complete',{'operation':'prediction','request':request.model_dump(),'result':result,'result_sha256':sha})
            return {**result,'audit_run_id':run_id,'result_sha256':sha}
        except (ValueError,FileNotFoundError) as exc:raise HTTPException(422,str(exc)) from exc

    webroot=Path(str(files('enzyme_agent').joinpath('web')))
    app.mount('/web',StaticFiles(directory=webroot),name='web')
    @app.get('/')
    def index(): return FileResponse(webroot/'index.html')
    return app
