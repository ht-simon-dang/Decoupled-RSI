import copy
import json

import pytest
from conftest import change_result
from pydantic import ValidationError

from antibody_agent.config import digest
from antibody_agent.contracts import Binder, Brief, Chain, Evidence, Observation
from antibody_agent.delivery import public_run
from antibody_agent.evaluation import context_id, identities, make_candidate
from antibody_agent.exports import export, structure_export
from antibody_agent.feedback import ingest, review
from antibody_agent.store import Store


def test_paired_identity_and_context(binder, brief):
    candidate = make_candidate(binder, "library", {})
    new = binder.model_copy(deep=True)
    new.product.fc = "different-Fc"
    assert identities(new)[0] != candidate.id
    assert identities(new)[1] == candidate.variable_id
    state = brief.states[0].model_copy(deep=True)
    state.state = "other antigen presentation"
    assert context_id(candidate, state) != context_id(candidate, brief.states[0])
    with pytest.raises(ValidationError):
        Binder(
            chains=[Chain(role="VH", sequence=binder.chains[0].sequence)],
            pairing_source="user_supplied",
            source="test",
        )


def test_numbering_and_constraints_are_reversible(binder):
    invalid = binder.model_dump()
    invalid["chains"][0]["numbering"][1]["label"] = "1"
    with pytest.raises(ValidationError):
        Binder.model_validate(invalid)
    invalid = binder.model_dump()
    invalid["chains"][0]["numbering"].pop()
    with pytest.raises(ValidationError):
        Binder.model_validate(invalid)
    with pytest.raises(ValidationError):
        Brief(reference=binder, constraints={"editable": {"VHH": ["1000"]}})


@pytest.mark.parametrize("status", ["unknown", "technical_failure"])
def test_missing_evidence_is_not_a_value(status):
    with pytest.raises(ValidationError):
        Evidence(endpoint="binding", status=status, value=0.0, source="test", version="1")


@pytest.mark.parametrize(
    "stage", ["queued", "generating", "evaluating", "blocked", "failed", "cancelled", "budget_exhausted"]
)
def test_no_delivery_before_successful_pipeline(completed, stage):
    completed["state"] = stage
    raw_sequence = completed["payload"]["candidates"][0]["binder"]["chains"][0]["sequence"]
    assert public_run(completed)["candidates"] == []
    for format_ in ("json", "fasta", "csv", "report"):
        assert raw_sequence not in export(completed, format_)[0]
    with pytest.raises(ValueError):
        structure_export(completed, completed["payload"]["candidates"][0]["id"])


@pytest.mark.parametrize(
    "section,changes",
    [
        ("fold", {"sequence_verified": False}),
        ("fold", {"ca_clashes": 3}),
        ("profile", {"applicable": False}),
        ("profile", {"profile": "tap"}),
        ("profile", {"severe_flags": ["synthetic severe risk"]}),
        ("complex:target", {"templates_used": True}),
        ("complex:target", {"forced_contacts": True}),
        ("complex:target", {"context_id": "b" * 64}),
        ("complex:target", {"antigen_iptm": 0.1}),
        ("complex:target", {"antigen_contacts": []}),
        ("complex:target", {"complex_plddt": 90.0}),
    ],
)
def test_independent_failure_cannot_be_overridden_by_persisted_pass(completed, section, changes):
    altered = change_result(completed, section, changes)
    assert public_run(altered)["candidates"] == []
    assert export(altered, "fasta")[0] == ""


def test_forged_stale_and_reformatted_evaluation_withheld(completed):
    assert len(public_run(completed)["candidates"]) == 1
    for modification in ("hash", "brief", "candidate", "output", "source"):
        run = copy.deepcopy(completed)
        cid = run["payload"]["candidates"][0]["id"]
        if modification == "hash":
            run["payload"]["evaluator_hash"] = "old-evaluator"
        elif modification == "brief":
            run["brief"]["product"]["fc"] = "changed"
        elif modification == "candidate":
            run["payload"]["candidates"][0]["binder"]["product"]["fc"] = "changed"
        elif modification == "output":
            run["payload"]["evaluation_calls"][cid]["complex:target"]["data"]["antigen_iptm"] = 0.99
        else:
            run["payload"]["evaluation_calls"][cid]["complex:target"]["provenance"]["weights_version"] = None
        assert public_run(run)["candidates"] == [], modification


def observation(run):
    cid = run["payload"]["candidates"][0]["id"]
    return {
        "id": "o1",
        "candidate_id": cid,
        "context_id": run["payload"]["assessments"][cid]["contexts"]["target"],
        "endpoint": "binding",
        "status": "measured",
        "value": 1.0,
        "unit": "synthetic",
        "outcome": "positive",
        "conditions": {"assay": "synthetic"},
        "batch": "test",
        "replicate": "1",
        "control": "test",
        "uncertainty": "test",
        "source": "synthetic test",
    }


def test_feedback_atomic_identity_censoring_and_human_review(completed, tmp_path):
    with Store(tmp_path / "runs") as store:
        rid = store.create(completed["brief"], completed["payload"])
        store.update(rid, "complete", completed["payload"])
        one = observation(completed)
        bad = {**one, "id": "o2", "context_id": "f" * 64}
        with pytest.raises(ValueError):
            ingest(store, rid, [one, bad])
        assert store.observation_rows(rid) == []
        result = ingest(store, rid, [one])
        assert result["groups"][0]["positive"] == 1
        assert len(ingest(store, rid, [one])["observations"]) == 1
        with pytest.raises(ValueError):
            ingest(store, rid, [{**one, "value": 2.0}])
        with pytest.raises(ValidationError):
            Observation.model_validate(
                {**one, "status": "technical_failure", "value": None, "outcome": "negative"}
            )
        censored = {**one, "id": "o2", "status": "censored", "censoring": "below", "outcome": "undetermined"}
        result = ingest(store, rid, [censored])
        assert result["groups"][0]["censored"] == 1
        assert result["groups"][0]["resolved"] == 1
        result = review(store, rid, one["candidate_id"], "accept", "synthetic review")
        assert result["decision"] == "accept"
        store.update(rid, "blocked", completed["payload"])
        with pytest.raises(ValueError):
            review(store, rid, one["candidate_id"], "accept", "cannot bypass evaluation")


def test_unknown_native_context_remains_explicit(completed):
    completed["brief"]["states"][0]["full_context"] = False
    completed["payload"]["brief_hash"] = digest(completed["brief"])
    assert public_run(completed)["candidates"] == []
    assert "affinity_probability" not in json.dumps(public_run(completed))
