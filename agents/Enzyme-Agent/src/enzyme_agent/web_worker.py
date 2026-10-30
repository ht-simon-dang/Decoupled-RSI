import json
import signal
import sys
from .store import Store
from .workflow import Engine
from .benchmark import calibrate


def terminate(signum,frame):
    raise SystemExit(128+signum)


def main():
    signal.signal(signal.SIGTERM,terminate)
    root,run_id=sys.argv[1:]
    with Store(root) as store:
        run=store.get(run_id)
        if run['payload'].get('operation')=='calibration':
            try:
                p=run['payload'];store.update(run_id,'calibrating',p)
                p['result']=calibrate(p['csv'],p['model_id'],p.get('split_mode','cross_family'))
                p.pop('csv',None);store.update(run_id,'complete',p)
            except Exception as exc:
                store.update(run_id,'failed',{**run['payload'],'error':str(exc)})
        else:
            Engine(store).resume(run_id)


if __name__=='__main__': main()
