import sys

from .contracts import Brief
from .store import Store
from .workflow import Engine


def main():
    store = Store(sys.argv[1]); run_id = sys.argv[2]
    run = store.get(run_id); payload = run["payload"]
    try:
        if payload.get("operation") == "tool":
            from .gateway import Gateway
            result = Gateway(store, run_id, run["skill"]).call(payload["tool"], payload["arguments"])
            payload["result"] = result.model_dump()
            store.update(run_id, "complete" if result.status == "ok" else "blocked" if result.status == "unavailable" else "failed", payload)
        elif payload.get("operation") == "calibration":
            from .calibration import train
            payload["result"] = train(payload["csv"], payload["model_id"])
            payload.pop("csv", None)
            store.update(run_id, "complete", payload)
        else:
            if payload.get("instruction") and not run["brief"]:
                from .llm import plan
                store.update(run_id, "planning", payload)
                planned = plan(payload["instruction"], store, run_id=run_id)
                store.set_brief(run_id, planned.brief.model_dump(), planned.skill)
            Engine(store).resume(run_id)
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        store.update(run_id, "failed", payload)
    finally:
        store.db.close()


if __name__ == "__main__": main()
