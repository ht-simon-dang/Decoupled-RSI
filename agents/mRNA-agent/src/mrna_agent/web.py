"""Local master-agent console; serialized process queue, cancellation and evidence API."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import Field, model_validator
from .contracts import Contract, Brief, Context
from .storage import Store, data_root

TERMINAL={"complete","failed","blocked","cancelled","paused","budget_exhausted"}


class Submission(Contract):
    brief: Brief | None=None
    instruction: str | None=Field(default=None,min_length=1,max_length=12000)
    fixed_plan: bool=False

    @model_validator(mode="after")
    def either(self):
        if (self.brief is None)==(self.instruction is None):raise ValueError("Provide a brief or an instruction")
        return self


class Experiments(Contract):
    records:list[dict]=Field(min_length=1,max_length=20000)


class Reevaluate(Contract):
    context:Context|None=None


class Queue:
    def __init__(self,root):
        self.root=root;self.pool=ThreadPoolExecutor(max_workers=1);self.lock=threading.RLock();self.pending=set();self.processes={}

    def submit(self,key):
        with self.lock:
            if key in self.pending:raise ValueError("Run already queued")
            self.pending.add(key)
        self.pool.submit(self.execute,key)

    def execute(self,key):
        try:
            with self.lock,Store(self.root) as store:
                run=store.get(key)
                if run["state"] in {"paused","cancelled"}:return
                store.update(key,"running",run["payload"])
                logs=Path(self.root)/"logs";logs.mkdir(exist_ok=True)
                with (logs/(key+".log")).open("a") as out:
                    proc=subprocess.Popen([sys.executable,"-m","mrna_agent.web_worker",str(self.root),key],stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                                          env=dict(os.environ,MRNA_DATA_DIR=str(data_root()),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1"))
                self.processes[key]=proc
            try:proc.wait(timeout=15000)
            except subprocess.TimeoutExpired:
                self.stop_process(proc)
            with self.lock,Store(self.root) as store:
                current=store.get(key)
                if current["state"] not in TERMINAL:
                    store.update(key,"failed",{**current["payload"],"error":"Worker stopped; resume from completed checkpoints"})
        finally:
            with self.lock:self.pending.discard(key);self.processes.pop(key,None)

    @staticmethod
    def stop_process(proc):
        if proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()

    def stop(self,key,state="paused"):
        with self.lock,Store(self.root) as store:
            current=store.get(key)
            if current["state"] in TERMINAL:raise ValueError("Run is already stopped")
            store.update(key,state,current["payload"])
            proc=self.processes.get(key)
            if proc:self.stop_process(proc)

    def close(self):
        for key in list(self.pending):
            try:self.stop(key)
            except ValueError:pass
        self.pool.shutdown(wait=True,cancel_futures=True)


def create_app(root=None, *, execute_jobs=True):
    from .structures import StructurePreview, protein_for
    from .history import preview_run, candidate_record, rna_topology
    root=Path(root or data_root()/"runs").resolve()
    queue=Queue(root)
    structures=StructurePreview()

    @asynccontextmanager
    async def lifespan(app):
        with Store(root) as store:
            for run in store.list_runs(10000):
                if run["state"] not in TERMINAL:
                    try:
                        with store.exclusive(run["id"]):
                            old=store.get(run["id"])
                            store.update(run["id"],"paused",{**old["payload"],"interruption":"Server restarted; resume checkpoints"})
                    except ValueError:
                        pass  # Another live CLI/server executor owns the run.
        yield
        queue.close()
        structures.close()
    app=FastAPI(title="EMA4D mRNA Master Agent",lifespan=lifespan)
    app.state.queue=queue
    app.state.structures=structures
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","[::1]","testserver"])

    @app.middleware("http")
    async def local_boundary(request:Request,call_next):
        if request.method in {"POST","PUT","DELETE","PATCH"}:
            origin=request.headers.get("origin")
            if origin and urlsplit(origin).netloc!=request.headers.get("host"):
                return JSONResponse({"detail":"Cross-origin mutation rejected"},status_code=403)
            length=request.headers.get("content-length","0")
            if not length.isdigit() or int(length)>4_000_000:
                return JSONResponse({"detail":"Request exceeds 4 MB"},status_code=413)
            body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>4_000_000:return JSONResponse({"detail":"Request exceeds 4 MB"},status_code=413)
            request._body=bytes(body)
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff"
        if request.url.path == "/" or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"]="no-store"
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request,exc):return JSONResponse({"detail":str(exc)},status_code=422)

    def get_run(key):
        with Store(root) as store:
            try:return {**store.get(key),"events":store.events(key)}
            except ValueError as exc:raise HTTPException(404,str(exc)) from exc

    def launch(brief,payload,instruction=None):
        with Store(root) as store:
            key=store.create(brief,instruction)
            store.update(key,"queued",{**store.get(key)["payload"],**payload})
        if execute_jobs:queue.submit(key)
        return {"id":key,"state":"queued"}

    @app.get("/api/health")
    def health():return {"status":"ok","name":"EMA4D mRNA Design Agent","version":"0.2.1","queue_size":len(queue.pending)}

    @app.get("/api/capabilities")
    def capabilities():
        from .registry import capabilities
        return capabilities()

    @app.get("/api/examples")
    def examples():
        path=Path(__file__).with_name("examples")
        if not path.is_dir():path=Path(__file__).resolve().parents[2]/"examples"
        return [{"id":p.stem,"brief":json.loads(p.read_text())} for p in sorted(path.glob("*.json"))]

    @app.get("/api/references")
    def references():
        from .generation import reference_elements
        return reference_elements()

    @app.get("/api/runs")
    def runs():
        with Store(root) as store:return store.list_runs()

    @app.post("/api/runs",status_code=202)
    def create(request:Submission):
        if request.brief and "workflow_version" not in request.brief.model_fields_set:
            with Store(root) as store:
                active=store.db.execute("SELECT value FROM settings WHERE key='active_workflow'").fetchone()
                if active:request.brief.workflow_version=active[0]
        return launch(request.brief.model_dump() if request.brief else {},{"fixed_plan":request.fixed_plan},request.instruction)

    @app.get("/api/runs/{key}")
    def read(key:str):
        from .delivery import public_run
        return public_run(get_run(key))

    @app.get("/api/runs/{key}/preview")
    def preview(key:str):
        return preview_run(get_run(key))

    def preview_candidate(key,candidate):
        try:return candidate_record(get_run(key),candidate)
        except KeyError as exc:raise HTTPException(404,str(exc)) from exc

    @app.get("/api/runs/{key}/preview/candidates/{candidate}/rna")
    def rna(key:str,candidate:str):
        return rna_topology(*preview_candidate(key,candidate))

    @app.get("/api/runs/{key}/preview/candidates/{candidate}/protein")
    def protein_status(key:str,candidate:str):
        card,_=preview_candidate(key,candidate)
        return structures.status(protein_for(card))

    @app.post("/api/runs/{key}/preview/candidates/{candidate}/protein")
    def predict_protein(key:str,candidate:str):
        card,_=preview_candidate(key,candidate)
        result=structures.submit(protein_for(card))
        return JSONResponse(result,status_code=202 if result["status"]=="running" else 200)

    @app.get("/api/runs/{key}/preview/candidates/{candidate}/protein.pdb")
    def protein_pdb(key:str,candidate:str):
        result=protein_status(key,candidate)
        if result["status"]!="ready":raise HTTPException(409,"No validated protein preview available")
        return Response(result["pdb"],media_type="chemical/x-pdb",headers={
            "Content-Disposition":f'attachment; filename="protein-preview-{result["key"][:16]}.pdb"'})

    @app.post("/api/runs/{key}/pause")
    def pause(key:str):
        get_run(key);queue.stop(key);return read(key)

    @app.post("/api/runs/{key}/cancel")
    def cancel(key:str):
        get_run(key);queue.stop(key,"cancelled");return read(key)

    @app.post("/api/runs/{key}/resume",status_code=202)
    def resume(key:str):
        run=get_run(key)
        if run["state"] not in {"paused","failed","blocked"}:raise HTTPException(409,"Only paused, failed or blocked runs can resume; budgets remain fixed")
        with queue.lock:
            if key in queue.pending:raise HTTPException(409,"Worker is still stopping; retry shortly")
            with Store(root) as store:store.update(key,"queued",run["payload"])
            if execute_jobs:queue.submit(key)
        return {"id":key,"state":"queued"}

    @app.post("/api/runs/{key}/reevaluate",status_code=202)
    def reevaluate(key:str,request:Reevaluate):
        run=get_run(key);brief=Brief.model_validate(run["brief"]);p=run["payload"]
        if not p.get("candidates"):raise HTTPException(409,"No generated candidates to reevaluate")
        if request.context:brief.context=request.context
        brief.budget.rounds=1
        return launch(brief.model_dump(),{"fixed_plan":True,"plan":p.get("plan",{}),"baseline":p["baseline"],"round_0":p["candidates"],"replay_of":key})

    @app.get("/api/runs/{key}/export/{format}")
    def download(key:str,format:str):
        from .exports import export
        text,mime=export(get_run(key),format)
        return Response(text,media_type=mime,headers={"Content-Disposition":f'attachment; filename="mrna-{key}.{format}"'})

    @app.post("/api/experiments")
    def experiments(request:Experiments):
        from .feedback import import_experiments
        with Store(root) as store:return import_experiments(store,request.records)

    @app.post("/api/runs/{key}/feedback")
    def feedback(key:str):
        from .feedback import analyze_feedback
        get_run(key)
        with Store(root) as store:
            from .delivery import public_feedback
            return public_feedback(analyze_feedback(store,key),store.get(key))

    @app.post("/api/calibration")
    def calibrate(request:Experiments):
        from .feedback import fit_calibration
        with Store(root) as store:
            result=fit_calibration(request.records);return {**result,"artifact":store.artifact(result)}

    @app.get("/api/evolution")
    def evolution_history():
        from .benchmark import history
        with Store(root) as store:return history(store)

    @app.post("/api/evolution",status_code=202)
    def evolve():return launch({"name":"Decoupled workflow audit"},{"operation":"evolution"})

    @app.post("/api/evolution/rollback")
    def rollback():
        from .benchmark import rollback
        with Store(root) as store:return rollback(store)

    static=Path(__file__).with_name("web")
    app.mount("/web",StaticFiles(directory=static),name="web")
    @app.get("/")
    def index():return FileResponse(static/"index.html")
    return app
