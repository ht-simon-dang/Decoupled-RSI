import sys
from .storage import Store


def main():
    root,run_id=sys.argv[1:3]
    with Store(root) as store:
        run=store.get(run_id)
        if run["payload"].get("operation")=="evolution":
            from .benchmark import evolve
            try:
                store.update(run_id,"evolving",run["payload"])
                result=evolve(store,progress=lambda row:store.event(run_id,"benchmark.task",row))
                store.update(run_id,"complete",{"operation":"evolution","result":result})
            except Exception as exc:
                store.update(run_id,"failed",{"operation":"evolution","error":str(exc)})
        else:
            from .workflow import Engine
            Engine(store).resume(run_id,planner=not run["payload"].get("fixed_plan",False))


if __name__=="__main__":main()
