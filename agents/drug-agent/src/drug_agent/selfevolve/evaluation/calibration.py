"""Calibrating the gold rubric against hand-written requirement sets.

A benchmark threshold asserted without evidence is a number someone liked. These
five reference requirement sets are run through the real pipeline and scored, so
the claim "a competent transcription clears the rubric and a careless one does
not" is something an operator can re-check in ten seconds.

They are hand-written, they live inside the integrity manifest, and no candidate
can change them. They are *not* benchmark tasks — nothing is scored against
them; they only demonstrate that the thresholds discriminate.

**Two facts this module keeps apart, because conflating them is what made it
report a false negative.**

*An absence is not a failure.* Scoring runs through
:func:`~drug_agent.selfevolve.evaluation.harness.score_gold`, the same three-way
``SUCCEEDED`` / ``UNAVAILABLE`` / ``INVALID_INPUT`` scorer the benchmark harness
uses. This module used to run its own loop over ``GoldCriterion.score()``, whose
``ok=None`` for an unproduced metric is falsy, so five metrics that were never
computed were filed as five criteria the reference set *missed*. ``missed`` now
lists only criteria that were really compared and really failed; everything else
is in ``unavailable`` or ``invalid``, with an endpoint record saying which.

*A capability this deployment does not have is not a flawed transcription.* Each
reference set declares a **required** objective, and each required objective has
exactly one producing capability (:data:`OBJECTIVE_CAPABILITY`). When that
capability is ``UNAVAILABLE`` here, the pipeline refuses to rank, the shortlist
is empty, and every downstream metric is an absence — which says nothing at all
about the rubric. Such a set is reported ``UNSCORABLE`` with the missing
capability named, and it is never scored against the threshold.

**Discrimination is therefore reported per stratum, and the blanket flag is
tri-state.** ``strata`` carries one verdict per reference family (``target``,
``phenotype``); ``discriminates`` at the top level is ``True`` only when every
stratum was decided and every stratum discriminated, ``False`` when a stratum
was decided and failed, and ``None`` when some stratum could not be scored on
this deployment at all. ``None`` rather than ``True``-with-a-footnote is
deliberate: the documented gate is ``discriminates == true`` (see
``SELF_EVOLUTION_REPORT.md``), ``None`` is falsy and compares unequal to
``True``, so an existing caller stays closed rather than being handed a pass it
did not earn. A caller that only needs one stratum — the ZAK work is
target-based — reads ``result["strata"]["target"]["discriminates"]``, and
``unscorable_strata`` names what it is choosing not to wait for.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Sequence
from typing import Any

from .benchmark import GoldCriterion

REFERENCE_SETS: dict[str, dict[str, Any]] = {
    "target: competent transcription": {
        "kind": "target_based",
        "target_name": "soluble epoxide hydrolase (sEH)",
        "target_gene": "EPHX2",
        "primary_objective": "pIC50",
        "objectives": [
            {"name": "pIC50", "direction": "maximize", "weight": 1.0, "required": True},
            {"name": "qed", "direction": "maximize", "weight": 0.7},
            {"name": "sa_score", "direction": "minimize", "weight": 0.5},
        ],
        "property_constraints": [
            {"property": "descriptors.mw", "op": "<=", "value": 500},
            {"property": "descriptors.logp", "op": "<=", "value": 5.0},
            {"property": "n_structural_alerts", "op": "==", "value": 0},
        ],
        "shortlist_size": 8,
        "max_candidates": 200,
    },
    "target: over-tight property window": {
        "kind": "target_based",
        "target_name": "sEH",
        "primary_objective": "pIC50",
        "objectives": [{"name": "pIC50", "direction": "maximize", "weight": 1.0, "required": True}],
        "property_constraints": [
            {"property": "descriptors.mw", "op": "<=", "value": 300},
            {"property": "descriptors.logp", "op": "<=", "value": 1.5},
            {"property": "descriptors.tpsa", "op": "<=", "value": 60},
        ],
        "shortlist_size": 8,
        "max_candidates": 200,
    },
    "target: gene symbol lost in transcription": {
        "kind": "target_based",
        "target_name": "an unnamed hydrolase",
        "primary_objective": "pIC50",
        "objectives": [
            {"name": "pIC50", "direction": "maximize", "weight": 1.0, "required": True},
            {"name": "qed", "direction": "maximize", "weight": 0.7},
        ],
        "property_constraints": [{"property": "descriptors.mw", "op": "<=", "value": 500}],
        "shortlist_size": 8,
        "max_candidates": 200,
    },
    "phenotype: competent transcription": {
        "kind": "phenotype_based",
        "phenotype_description": "dysfunctional vascular endothelium",
        "primary_objective": "restoration_score",
        "objectives": [
            {"name": "restoration_score", "direction": "maximize", "weight": 1.0, "required": True},
            {"name": "qed", "direction": "maximize", "weight": 0.6},
            {"name": "sa_score", "direction": "minimize", "weight": 0.4},
        ],
        "property_constraints": [
            {"property": "descriptors.mw", "op": "<=", "value": 500},
            {"property": "n_structural_alerts", "op": "==", "value": 0},
        ],
        "shortlist_size": 8,
        "max_candidates": 200,
    },
    "phenotype: mis-transcribed as target-based": {
        "kind": "target_based",
        "target_name": "vascular endothelium phenotype",
        "primary_objective": "pIC50",
        "objectives": [
            {"name": "pIC50", "direction": "maximize", "weight": 1.0, "required": True},
            {"name": "qed", "direction": "maximize", "weight": 0.6},
        ],
        "property_constraints": [{"property": "descriptors.mw", "op": "<=", "value": 500}],
        "shortlist_size": 8,
        "max_candidates": 200,
    },
}


#: The one capability that produces each objective a reference set may declare
#: ``required``. This is the whole of the "can this deployment score this set at
#: all" question: if the producer is absent the pipeline refuses to rank on a
#: required objective (``integrate_objectives``: *cannot rank: no predictions
#: exist for required objective(s) …*), the shortlist is empty, and the rubric
#: is being shown an absence rather than a portfolio.
#:
#: It mirrors — and is asserted against — ``ENDPOINT_TOOL`` in
#: :mod:`drug_agent.selfevolve.analysis.analyzer`, which maps the *scored
#: endpoints* to the same producers. This map is keyed by objective name because
#: that is what a :class:`~drug_agent.llm.schemas.ChemicalRequirements` carries.
OBJECTIVE_CAPABILITY: dict[str, str] = {
    "pIC50": "model.activity",
    "restoration_score": "perturb.assess",
}

#: A tool is usable only in this state. ``MOCKED`` is deliberately not here:
#: under ADR-0011 nothing in this repository returns a stand-in, and a state
#: that ever appeared would mean the rubric was being shown invented numbers.
_USABLE_TOOL_STATUS = "AVAILABLE"

#: What a reference set's row says about whether it could be scored here.
SCORED = "SCORED"
UNSCORABLE = "UNSCORABLE"


def reference_stratum(name: str) -> str:
    """The reference *family* a set belongs to, from its name prefix.

    Deliberately not the transcribed ``kind``: ``phenotype: mis-transcribed as
    target-based`` is a phenotype brief that was written up as a target program,
    so it carries ``kind="target_based"`` and still belongs with the phenotype
    reference sets — it is the flawed half of that stratum's pair.
    """
    return name.split(":", 1)[0].strip()


def missing_capabilities(
    spec: Mapping[str, Any], capability_status: Mapping[str, tuple[str, str | None]]
) -> list[dict[str, Any]]:
    """Producers this deployment lacks for *spec*'s required objectives.

    ``capability_status`` maps a capability id to ``(status, unavailable_reason)``
    — exactly what :func:`drug_agent.tools.registry.tool_index` reports, passed
    in rather than looked up so this stays a pure function the tests can drive.
    An empty list means the pipeline's inability to produce a portfolio, if it
    could not, is a property of the requirement set and not of the machine.
    """
    required = {
        str(obj.get("name"))
        for obj in spec.get("objectives", ())
        if isinstance(obj, Mapping) and obj.get("required")
    }
    primary = spec.get("primary_objective")
    if primary:
        required.add(str(primary))

    out: list[dict[str, Any]] = []
    for objective in sorted(required):
        capability = OBJECTIVE_CAPABILITY.get(objective)
        if capability is None:
            continue
        status, reason = capability_status.get(capability, ("ABSENT", None))
        if status == _USABLE_TOOL_STATUS:
            continue
        out.append(
            {
                "objective": objective,
                "capability": capability,
                "status": status,
                "reason": reason or f"{capability} is {status} on this deployment and produces nothing",
            }
        )
    return out


def score_reference_set(
    *,
    name: str,
    spec: Mapping[str, Any],
    gold: Sequence[GoldCriterion],
    metrics: Mapping[str, Any],
    execution: Mapping[str, Any],
    missing: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score one reference set, or say why it could not be scored here.

    ``execution`` is the
    :func:`~drug_agent.selfevolve.evaluation.harness.kernel_execution_endpoint`
    record for the pipeline run, so a reader can tell a run that completed and
    shortlisted nothing from a run that never completed.

    ``score`` is ``met / len(gold)`` — the same scale
    :data:`~drug_agent.selfevolve.evaluation.benchmark.GOLD_PASS_THRESHOLD` has
    always been read on, and the same one ``harness.run_task`` reports — or
    ``None`` when the set is ``UNSCORABLE``, so there is no number here for a
    caller to mistake for a rubric result.
    """
    from .harness import score_gold

    detail, met, scored, unscorable, invalid = score_gold(tuple(gold), dict(metrics))
    total = len(gold)
    blocked = bool(missing)
    return {
        "name": name,
        "stratum": reference_stratum(name),
        "kind": spec.get("kind"),
        "competent": "competent" in name,
        "status": UNSCORABLE if blocked else SCORED,
        "met": met,
        "total": total,
        "scored": scored,
        "unscorable": unscorable,
        "invalid": invalid,
        "score": None if blocked else (round(met / total, 4) if total else 0.0),
        # Only criteria that were really compared and really failed. An
        # unproduced metric used to land here and is now in `unavailable`.
        "missed": [d["metric"] for d in detail if d["met"] is False],
        "unavailable": [d["metric"] for d in detail if d["endpoint"]["status"] == "UNAVAILABLE"],
        "invalid_metrics": [d["metric"] for d in detail if d["endpoint"]["status"] == "INVALID_INPUT"],
        "missing_capabilities": [dict(m) for m in missing],
        "execution": dict(execution),
        "unscorable_reason": (
            "; ".join(f"{m['objective']} needs {m['capability']}: {m['reason']}" for m in missing)
            if blocked
            else None
        ),
        "detail": detail,
    }


def summarize_calibration(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Per-stratum verdicts, then one honest tri-state flag over all of them.

    A stratum discriminates when every *scorable* competent set in it clears
    ``threshold`` and every *scorable* flawed set stays below. It decides
    nothing — ``None`` — when it has no scorable competent set left, because the
    claim being tested is "a competent transcription clears the rubric" and
    there is then nothing here that could demonstrate or refute it.
    """
    strata: dict[str, dict[str, Any]] = {}
    for row in rows:
        stratum = strata.setdefault(
            str(row["stratum"]),
            {
                "stratum": row["stratum"],
                "competent": [],
                "flawed": [],
                "unscorable": [],
                "missing_capabilities": [],
            },
        )
        if row["status"] == UNSCORABLE:
            stratum["unscorable"].append(row["name"])
            for entry in row.get("missing_capabilities") or ():
                if entry["capability"] not in stratum["missing_capabilities"]:
                    stratum["missing_capabilities"].append(entry["capability"])
            continue
        stratum["competent" if row["competent"] else "flawed"].append(
            {"name": row["name"], "score": row["score"]}
        )

    for stratum in strata.values():
        competent = stratum["competent"]
        flawed = stratum["flawed"]
        if not competent:
            stratum["discriminates"] = None
            stratum["verdict"] = (
                f"undecided here: no competent reference set in the {stratum['stratum']} stratum "
                f"could be scored on this deployment"
                + (
                    f" ({', '.join(stratum['missing_capabilities'])} unavailable)"
                    if stratum["missing_capabilities"]
                    else ""
                )
            )
            continue
        ok = all(r["score"] >= threshold for r in competent) and all(r["score"] < threshold for r in flawed)
        stratum["discriminates"] = ok
        stratum["verdict"] = (
            f"the rubric separates competent from flawed {stratum['stratum']} reference sets at {threshold}"
            if ok
            else f"the rubric does NOT separate competent from flawed {stratum['stratum']} "
            f"reference sets at {threshold}"
        )

    unscorable_strata = sorted(s for s, v in strata.items() if v["discriminates"] is None)
    decided = [v["discriminates"] for v in strata.values() if v["discriminates"] is not None]
    if unscorable_strata:
        discriminates: bool | None = None
    else:
        discriminates = bool(decided) and all(decided)

    decided_ok = sorted(s for s, v in strata.items() if v["discriminates"] is True)
    decided_bad = sorted(s for s, v in strata.items() if v["discriminates"] is False)
    if discriminates is None:
        gaps = sorted({c for s in unscorable_strata for c in strata[s]["missing_capabilities"]})
        verdict = (
            f"UNDECIDED on this deployment: the {', '.join(unscorable_strata)} stratum "
            f"could not be scored"
            + (f" ({', '.join(gaps)} unavailable)" if gaps else "")
            + (
                f"; the {', '.join(decided_ok)} stratum does discriminate at {threshold}"
                if decided_ok
                else ""
            )
            + (f"; the {', '.join(decided_bad)} stratum does NOT discriminate" if decided_bad else "")
        )
    elif discriminates:
        verdict = "the rubric separates competent transcriptions from flawed ones at this threshold"
    else:
        verdict = (
            "WARNING: the rubric does not cleanly separate competent from flawed reference "
            f"sets in the {', '.join(decided_bad)} stratum"
        )

    return {
        "strata": {k: strata[k] for k in sorted(strata)},
        "unscorable_strata": unscorable_strata,
        "discriminates": discriminates,
        "verdict": verdict,
    }


#: Exit statuses this package's CLI already publishes: 0 ok, 4 degraded.
_EXIT_OK = 0
_EXIT_DEGRADED = 4


def calibration_exit_code(result: Mapping[str, Any], *, required: Sequence[str] = ()) -> tuple[int, str]:
    """The exit status a calibration *invocation* should carry, and why.

    Repair R1 made the record honest — ``discriminates`` is tri-state and
    ``None`` compares unequal to ``True``, so a Python caller checking ``== True``
    fails closed. The **invocation** did not: ``drugagent-selfevolve calibrate``
    printed the undecided verdict and exited 0, so a shell gate, a CI job or a
    ``make`` target reading ``$?`` was told the evaluator discriminates on a
    deployment where a whole stratum could not be scored. That is the other half
    of the same defect, and it is what this function closes.

    ``required`` names the strata a caller actually depends on. The ZAK program
    is target-based, so it passes ``["target"]`` and is not blocked by the
    phenotype stratum having no backend. With no ``required``, the gate is the
    documented one: every stratum decided, and every stratum discriminating.

    Returns ``(exit_code, reason)``. ``None`` is never treated as a pass.
    """
    strata = result.get("strata") or {}
    if required:
        failed: list[str] = []
        for name in required:
            stratum = strata.get(name)
            if stratum is None:
                failed.append(f"{name}: no such reference stratum")
            elif stratum.get("discriminates") is not True:
                verdict = "UNDECIDED" if stratum.get("discriminates") is None else "does not discriminate"
                failed.append(f"{name}: {verdict} — {stratum.get('verdict')}")
        if failed:
            return _EXIT_DEGRADED, "; ".join(failed)
        return _EXIT_OK, f"required strata discriminate: {', '.join(required)}"

    decided = result.get("discriminates")
    if decided is True:
        return _EXIT_OK, str(result.get("verdict") or "the rubric discriminates")
    if decided is False:
        return _EXIT_DEGRADED, str(result.get("verdict") or "the rubric does not discriminate")
    unscorable = ", ".join(result.get("unscorable_strata") or ()) or "unknown"
    return (
        _EXIT_DEGRADED,
        f"UNDECIDED: the {unscorable} stratum could not be scored on this deployment. "
        f"Name the strata you depend on with --require to gate on them instead",
    )


def _capability_status() -> dict[str, tuple[str, str | None]]:
    """Live status of every capability :data:`OBJECTIVE_CAPABILITY` names.

    Reads the real tool pool, so a backend that appears or disappears changes
    this answer without anyone editing a list. A pool that cannot even be built
    is reported as ``ABSENT`` rather than being assumed present — the direction
    that refuses to score, never the one that invents a pass.
    """
    from ...tools.registry import tool_index

    wanted = set(OBJECTIVE_CAPABILITY.values())
    try:
        index = tool_index()
    except Exception as exc:  # a broken optional backend must not fake a pass
        why = f"the tool pool could not be built: {type(exc).__name__}: {exc}"
        return dict.fromkeys(wanted, ("ABSENT", why))
    out: dict[str, tuple[str, str | None]] = {}
    for capability in wanted:
        card = index.get(capability)
        if card is None:
            out[capability] = ("ABSENT", f"{capability} is not registered in the tool pool")
        else:
            out[capability] = (str(card.status), card.unavailable_reason)
    return out


def run_calibration() -> dict[str, Any]:
    """Score every reference set. Takes a minute; calls no model."""
    from ...llm.bridge import requirements_to_brief_dict, run_pipeline
    from ...llm.config import Config
    from ...llm.schemas import ChemicalRequirements
    from .benchmark import GOLD_PASS_THRESHOLD, load_benchmark
    from .harness import kernel_execution_endpoint
    from .metrics import kernel_metrics

    cfg = Config()
    tasks = {t.task_id: t for t in load_benchmark()}
    gold_target = tasks["seh_basic"].gold
    gold_pheno = tasks["endothelium_phenotype"].gold
    capability_status = _capability_status()

    rows: list[dict[str, Any]] = []
    with contextlib.redirect_stdout(io.StringIO()):
        for name, spec in REFERENCE_SETS.items():
            req = ChemicalRequirements.model_validate(spec)
            brief, _seeds, _dropped = requirements_to_brief_dict(req, cfg)
            run = run_pipeline(req, cfg)
            metrics = kernel_metrics(run, req.shortlist_size, brief)["metrics"]
            gold = gold_pheno if name.startswith("phenotype: competent") else gold_target
            rows.append(
                score_reference_set(
                    name=name,
                    spec=spec,
                    gold=gold,
                    metrics=metrics,
                    execution=kernel_execution_endpoint(run),
                    missing=missing_capabilities(spec, capability_status),
                )
            )

    summary = summarize_calibration(rows, GOLD_PASS_THRESHOLD)
    return {
        "pass_threshold": GOLD_PASS_THRESHOLD,
        "rows": rows,
        "capability_status": {
            k: {"status": v[0], "reason": v[1]} for k, v in sorted(capability_status.items())
        },
        **summary,
    }


__all__ = [
    "OBJECTIVE_CAPABILITY",
    "REFERENCE_SETS",
    "SCORED",
    "UNSCORABLE",
    "calibration_exit_code",
    "missing_capabilities",
    "reference_stratum",
    "run_calibration",
    "score_reference_set",
    "summarize_calibration",
]
