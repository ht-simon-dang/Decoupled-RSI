import sys
import time

import pytest
from fastapi.testclient import TestClient

from antibody_agent.contracts import BudgetExceeded
from antibody_agent.delivery import public_run
from antibody_agent.gateway import Gateway, execute
from antibody_agent.store import Store
from antibody_agent.web import create_app
from antibody_agent.workflow import Engine


def test_research_and_missing_model_fail_closed(tmp_path, brief, monkeypatch):
    with Store(tmp_path / "runs") as store:
        research = Engine(store).run({"route": "research"})
        assert research["state"] == "complete"
        assert public_run(research)["delivery"]["state"] == "research_only"
        result = Engine(store).run(brief)
        assert result["state"] == "blocked"
        assert public_run(result)["candidates"] == []
        used = store.cost(result["id"])["tool_calls"]
        Engine(store).resume(result["id"])
        assert store.cost(result["id"])["tool_calls"] > used


def test_budget_remains_spent_after_resume(tmp_path):
    with Store(tmp_path / "runs") as store:
        rid = store.create({})
        gateway = Gateway(store, rid, 1, time.monotonic() + 20)
        gateway.call("planning.qwen", {"instruction": "test"})
        with pytest.raises(BudgetExceeded):
            Gateway(store, rid, 1, time.monotonic() + 20).call("planning.qwen", {"instruction": "test again"})


def test_gateway_checks_process_exit_and_timeout(tmp_path):
    script = tmp_path / "bridge.py"
    script.write_text(
        "import json,sys\njson.dump({'tool':'planning.qwen','status':'ok','data':{}},open(sys.argv[-1],'w'))\nsys.exit(1)\n"
    )
    config = {"command": [sys.executable, str(script)], "version": "test", "weights_version": "test"}
    assert execute("planning.qwen", {"instruction": "test"}, 5, config=config).status == "error"
    script.write_text("import time\ntime.sleep(10)\n")
    assert execute("planning.qwen", {"instruction": "test"}, 0.15, config=config).status == "timeout"


def test_reevaluation_retains_generator_lineage(completed, tmp_path):
    c = completed["payload"]["candidates"][0]
    c["origin"] = "boltzgen"
    with Store(tmp_path / "runs") as store:
        old = store.create(completed["brief"], completed["payload"])
        store.update(old, "complete", completed["payload"])
        new = Engine(store).prepare_reevaluation(old)
        assert store.get(new)["payload"]["lineage"][c["id"]]["origin"] == "boltzgen"
        assert store.get(new)["brief"]["parent_run_id"] == old


def test_http_queue_export_review_and_local_origin(tmp_path, completed):
    root = tmp_path / "runs"
    app = create_app(root)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/web/app.js").status_code == 200
        assert "0.8B" in client.get("/api/capabilities").text
        assert (
            client.post(
                "/api/runs", json={"brief": {}}, headers={"Origin": "https://untrusted.example"}
            ).status_code
            == 403
        )
        assert client.post("/api/runs", json={"brief": {}, "instruction": "test"}).status_code == 422
        assert client.get("/api/runs/invalid").status_code == 404
        response = client.post("/api/runs", json={"brief": {"route": "research"}})
        assert response.status_code == 202
        rid = response.json()["id"]
        for _ in range(50):
            run = client.get("/api/runs/" + rid).json()
            if run["state"] == "complete":
                break
            time.sleep(0.02)
        assert run["delivery"]["state"] == "research_only"
        with Store(root) as store:
            blocked = store.create(completed["brief"], completed["payload"])
            store.update(blocked, "blocked", completed["payload"])
        cid = completed["payload"]["candidates"][0]["id"]
        sequence = completed["payload"]["candidates"][0]["binder"]["chains"][0]["sequence"]
        for path in ("", "/export/json", "/export/fasta", "/export/report", "/benchmark", "/feedback"):
            assert sequence not in client.get("/api/runs/" + blocked + path).text
        assert client.get(f"/api/runs/{blocked}/structures/{cid}").status_code == 422
        assert (
            client.post(
                f"/api/runs/{blocked}/review",
                json={"candidate_id": cid, "decision": "accept", "note": "test"},
            ).status_code
            == 422
        )
        with Store(root) as store:
            store.update(blocked, "complete", completed["payload"])
        assert sequence in client.get(f"/api/runs/{blocked}/export/fasta").text
        assert (
            client.post(
                f"/api/runs/{blocked}/review",
                json={"candidate_id": cid, "decision": "accept", "note": "test"},
            ).status_code
            == 200
        )


def test_cancel_then_resume_cannot_reset_budget(tmp_path, brief):
    brief.budget.tool_calls = 1
    with Store(tmp_path / "runs") as store:
        run = Engine(store).run(brief)
        assert run["state"] == "budget_exhausted"
        assert store.cost(run["id"])["tool_calls"] == 1
        assert public_run(run)["candidates"] == []
