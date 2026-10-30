"""Synthetic protocol fixtures only; none represents scientific model output."""

import copy
import time

import pytest

from antibody_agent.config import digest
from antibody_agent.contracts import Binder, Brief, ToolResult
from antibody_agent.evaluation import context_id, evaluate, evaluation_binder, evaluator_hash, make_candidate


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIBODY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTIBODY_RUNTIMES", str(tmp_path / "runtimes.json"))


@pytest.fixture
def binder():
    # Deliberately non-biological test sequence, not a candidate or antibody benchmark.
    sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN"
    return Binder.model_validate(
        {
            "chains": [
                {
                    "role": "VHH",
                    "sequence": sequence,
                    "numbering_source": "synthetic test map",
                    "numbering": [
                        {"index": i, "label": str(i), "region": "FR1"} for i in range(1, len(sequence) + 1)
                    ],
                }
            ],
            "pairing_source": "user_supplied",
            "source": "synthetic unit-test fixture",
            "product": {
                "format": "VHH",
                "valency": 1,
                "fc": "none",
                "fc_requirement": "none",
                "chain_organization": "single VHH",
            },
        }
    )


@pytest.fixture
def brief(binder):
    return Brief.model_validate(
        {
            "name": "Synthetic protocol test",
            "entry": "sequence",
            "route": "library",
            "generators": ["library"],
            "direction": "block",
            "mechanism": "synthetic test hypothesis",
            "direction_gate": {"status": "hypothesis"},
            "access_gate": {"status": "hypothesis"},
            "product": binder.product.model_dump(),
            "library": [binder.model_dump()],
            "states": [
                {
                    "id": "target",
                    "entity": "synthetic test target",
                    "sequence": "A" * 30,
                    "full_context": True,
                    "glycan_context": "not_applicable",
                    "membrane_context": "not_applicable",
                }
            ],
            "epitopes": [{"state_id": "target", "residues": [1, 2, 3]}],
        }
    )


def signed(tool, arguments, data):
    return ToolResult(
        tool=tool,
        status="ok",
        data=data,
        provenance={
            "model": tool.split(".")[1],
            "version": "synthetic-test-v1",
            "weights_version": "no-real-weights",
            "request_hash": digest(arguments),
            "output_sha256": digest(data),
        },
    )


def records(candidate, brief):
    binding = evaluation_binder(candidate)
    fold = signed(
        "fold.immunebuilder",
        {"binder": binding, "seed": brief.seed},
        {"pdb": "synthetic test structure", "sequence_verified": True, "ca_clashes": 0},
    )
    profile = signed(
        "profile.tnp",
        {"binder": binding, "pdb": fold.data["pdb"], "seed": brief.seed},
        {"profile": "tnp", "applicable": True, "severe_flags": [], "metrics": {"synthetic_test": 1.0}},
    )
    state = brief.states[0]
    ctx = context_id(candidate, state)
    complex_ = signed(
        "complex.boltz2",
        {"binder": binding, "state": state.model_dump(), "context_id": ctx, "seed": brief.seed},
        {
            "pdb": "synthetic complex",
            "sequence_verified": True,
            "ca_clashes": 0,
            "antigen_contacts": [1, 2, 3],
            "context_id": ctx,
            "antigen_iptm": 0.8,
            "complex_plddt": 0.9,
            "templates_used": False,
            "forced_contacts": False,
        },
    )
    return {
        "fold": fold.model_dump(),
        "profile": profile.model_dump(),
        "complex:target": complex_.model_dump(),
    }


@pytest.fixture
def completed(brief, binder):
    candidate = make_candidate(binder, "library", {"source": "synthetic test"})
    calls = records(candidate, brief)
    assessment = evaluate(candidate, brief, calls)
    assert assessment["status"] == "pass", assessment["reasons"]
    return {
        "id": "a" * 32,
        "state": "complete",
        "created": time.time(),
        "updated": time.time(),
        "brief": brief.model_dump(),
        "payload": {
            "candidates": [candidate.model_dump()],
            "evaluation_calls": {candidate.id: calls},
            "assessments": {candidate.id: assessment},
            "evaluator_hash": evaluator_hash(),
            "brief_hash": digest(brief.model_dump()),
        },
    }


def change_result(run, section, changes):
    value = copy.deepcopy(run)
    candidate = value["payload"]["candidates"][0]
    result = value["payload"]["evaluation_calls"][candidate["id"]][section]
    result["data"].update(changes)
    result["provenance"]["output_sha256"] = digest(result["data"])
    return value
