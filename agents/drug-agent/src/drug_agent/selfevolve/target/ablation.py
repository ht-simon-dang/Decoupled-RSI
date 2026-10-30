"""Step 16 — bounded ablations, and the attribution they can and cannot support.

An ablation is only an attribution if the comparison was named before its own
answer existed, if the two sides differ in exactly the factor being blamed, and
if both sides were given the same opportunity to spend. Every mechanism here
exists to make those three things checkable rather than asserted:

* **A comparison is registered before its own final evaluation.**
  :class:`ComparisonRegistry` writes a registration record naming the treatment
  arm, the control arm, the single factor under test and the endpoint. It
  refuses (``UNSUPPORTED_CLAIM``) to register a comparison whose evidence
  already exists *as confirmatory* — the registration would be a description of
  a number that is already known. Such a comparison is admitted as
  :data:`InferenceLabel.DEVELOPMENT_STAGE` with
  ``registration_status = "EVIDENCE_PREDATES_THIS_REGISTRATION"`` written into
  the record, because hiding the ordering is the only thing that would make it
  look confirmatory.
* **The manifest is machine-diffed against its base.**
  :func:`configuration_diff` flattens both configurations to JSON pointers and
  reports every pointer that moved, every pointer that appeared and every
  pointer that vanished. A pointer owned by no declared factor is an
  ``undeclared_change`` and :func:`verify_declared_factors_only` refuses the
  manifest. A declared factor that moved *nothing* is reported too: a manifest
  that claims to ablate a knob it never touched is as wrong as one that touches
  a knob it never claimed, and only the diff can tell the difference.
* **A configuration diff is not a mechanism diff.** A manifest can name
  ``/controller/prior_hypotheses`` and be perfectly matched on paper while the
  code path it selects moves seven other fields the proposer reads.
  :func:`history_suppression_diff` runs the *real*
  :func:`~drug_agent.selfevolve.target.controller.structured_feedback` twice —
  once with a campaign's history, once with it suppressed — and reports the
  fields that actually moved. Under this programme's four real pilot histories
  it moves more fields than ``ABL-1-no-history`` declares, and that is recorded
  as an unmatched factor rather than smoothed over.
* **Opportunity is matched even when consumption cannot be.**
  :func:`matched_resource_accounting` compares each arm's declared allocation
  against one confirmation campaign's, resource by resource. An arm that
  consumes zero output tokens *by construction* still receives the full
  allocation — a no-memory or no-model arm is not handicapped by being cheap —
  and the record says plainly that the matched quantity is opportunity, not
  spend. Every resource the ablation block is silent about is listed as
  unmatched rather than assumed equal.
* **The predeclared set is CLOSED.** :func:`refuse_new_confirmatory_arm` raises
  for any arm outside ``ablation.arms`` that is offered as ``CONFIRMATORY``.
  Selecting further ablations until one of them separates is the failure mode
  the closed set exists to stop, and it is refused by number rather than by
  convention.
* **Viewing a test result downgrades everything.**
  :func:`resolve_inference_label` takes ``test_results_viewed`` and returns
  ``POST_HOC_EXPLANATORY`` for *every* arm when it is true, predeclared or not.
  Under ``zak-c9r-2`` it is **false**: ``test_access_count`` is 0 in all nine
  records that declare it, no held-out result exists anywhere in the tree, and
  so nothing here was chosen after seeing one. That is recorded as an observed
  fact, not as an absence of evidence.

**What this module cannot do, and says so.** Three of the four predeclared arms
are ``CONFIRMATORY`` and each one's treatment side is ``B5-self-evolution``,
which has never been executed; their registered evaluation is additionally
behind the held-out nomination boundary, which is not released. The fourth,
``ABL-4-acquisition-enabled``, was labelled ``POST_HOC_EXPLANATORY`` *in
advance* because no ingestion path works on this host and its exclusion union
cannot be derived before the fact. So :func:`arm_readiness` returns a gate per
arm and the honest answer today is ``NOT_RUN`` for all four, each with its
reason, its declared cost and the conditions that would let it run later —
exactly as :mod:`drug_agent.selfevolve.target.confirmation` reports the
confirmation branch.

**What it can do, and does.** Three controlled comparisons need no nominee, no
held-out access and no new computation, because their evidence was produced on
the development partition by earlier steps under matched setups:

* the Step 12 distillation sweep against its own ``synthetic_weight = 0.0``
  control, matched by one base configuration, one anchor pool and one holdout;
* the Step 11 pilot's single successful recipe against the *revert of its main
  changed factor*, which is the parent baseline it was measured against;
* the Stage D nine-arm sweep re-read for a compute effect, from the funnel
  records it already wrote.

All three are development-stage and labelled so. None of them is a confirmation
of anything, and the report says which factor each one can and cannot separate.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from .confirmation import refuse_false_assurance
from .contracts import ContractError, ErrorCategory

Record = dict[str, Any]

CONTRACT = "AblationAttribution"

#: The protocol version and content address these comparisons are registered
#: against. Checked rather than trusted: a document that does not hash to this
#: is not the study this module is about, whatever it calls itself.
REGISTERED_PROTOCOL_VERSION = "zak-c9r-2"
REGISTERED_PROTOCOL_HASH = "sha256:f960f2c8635ba3053dadbdc05a138d89b4ce6b8ef91456cf0fcaa82c7656366b"


class Factor(StrEnum):
    """The effects the report has to be able to tell apart.

    Appendix A's acceptance for this step is that data acquisition, training,
    search, memory, distillation and compute can be *distinguished*. Naming them
    as an enum means an attribution row cannot be filed under a factor nobody
    declared, and :func:`attribution_table` can report the factors for which no
    controlled comparison exists rather than leaving them out of the table.
    """

    DATA_ACQUISITION = "data_acquisition"
    TRAINING = "training"
    SEARCH = "search"
    MEMORY = "memory"
    DISTILLATION = "distillation"
    COMPUTE = "compute"


class InferenceLabel(StrEnum):
    """What an arm's result is allowed to be used for.

    ``CONFIRMATORY`` and ``POST_HOC_EXPLANATORY`` are the lock's own vocabulary
    (``protocol.INFERENCE_LABELS``). ``DEVELOPMENT_STAGE`` is this step's third
    label and is **not** written into any protocol record: it marks a controlled
    comparison that ran on the development partition, whose evidence therefore
    exists already, and which can never be confirmatory. It is a strict subset
    of explanatory for inference purposes — :func:`is_confirmatory` is true only
    for ``CONFIRMATORY``.
    """

    CONFIRMATORY = "CONFIRMATORY"
    POST_HOC_EXPLANATORY = "POST_HOC_EXPLANATORY"
    DEVELOPMENT_STAGE = "DEVELOPMENT_STAGE"


#: The two labels the frozen lock may carry. Mirrors ``protocol.INFERENCE_LABELS``
#: deliberately rather than importing it: ``protocol.py`` is inside the
#: evaluator's guarded set and this module is not, so the dependency runs one
#: way only. :func:`refuse_new_confirmatory_arm` checks the agreement.
PROTOCOL_INFERENCE_LABELS: tuple[str, ...] = ("CONFIRMATORY", "POST_HOC_EXPLANATORY")


class ArmStatus(StrEnum):
    """Whether an arm produced evidence, and if not, why the record exists."""

    EXECUTED = "EXECUTED"
    NOT_RUN = "NOT_RUN"


class RegistrationStatus(StrEnum):
    """Whether the registration really preceded the number it describes."""

    #: Registered before any evidence for this comparison exists.
    REGISTERED_BEFORE_EVALUATION = "REGISTERED_BEFORE_EVALUATION"
    #: The evidence was produced by an earlier step. Registering it now records
    #: a comparison, not a prediction, and it can never be confirmatory.
    EVIDENCE_PREDATES_THIS_REGISTRATION = "EVIDENCE_PREDATES_THIS_REGISTRATION"


#: Every resource one confirmation campaign is allocated. An ablation arm is
#: resource-matched only when it is allocated the same amount of each; a
#: resource the ablation block does not mention is UNMATCHED, never assumed
#: equal, which is what :func:`matched_resource_accounting` reports.
MATCHED_RESOURCES: tuple[str, ...] = (
    "experiment_proposals",
    "gpu_seconds",
    "llm_output_tokens",
    "wall_seconds",
    "max_concurrent_cpu_jobs",
    "max_concurrent_gpu_jobs",
)

#: The per-arm budget key in ``budgets.per_arm`` that stands for "one
#: confirmation campaign", used as the reference allocation for budget symmetry.
REFERENCE_CAMPAIGN_ARM = "B5-campaign-1"

#: Fields a manifest must declare. A manifest missing any of them cannot be
#: diffed against a base, cannot be attributed to a factor and cannot say what
#: it would cost, so it is refused rather than partially processed.
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "arm_id",
    "base_manifest",
    "declared_factors",
    "configuration",
    "endpoint",
    "inference_label",
    "isolates",
    "record_kind",
    "seeds",
)

#: Fields every declared factor carries. ``paths`` is the machine-checkable part:
#: the JSON pointers this factor is allowed to move.
FACTOR_REQUIRED_FIELDS: tuple[str, ...] = ("factor_id", "attributes_to", "paths", "description")

_MANIFEST_RECORD_KIND = "ablation_arm_manifest"
_BASE_RECORD_KIND = "ablation_base_configuration"


def _fail(
    category: ErrorCategory,
    message: str,
    *,
    field: str | None = None,
    contract: str = CONTRACT,
) -> ContractError:
    return ContractError(category, message, field=field, contract=contract)


def canonical_json(payload: Any) -> str:
    """One serialisation, so one content address."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_address(payload: Any) -> str:
    """``sha256:`` over the canonical bytes of *payload*."""
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def is_confirmatory(label: str) -> bool:
    """True only for the one label that licenses an independent claim."""
    return str(label) == InferenceLabel.CONFIRMATORY.value


# ---------------------------------------------------------------------------
# The machine-generated configuration diff
# ---------------------------------------------------------------------------


def flatten(config: Any, prefix: str = "") -> dict[str, Any]:
    """*config* as ``{JSON pointer: leaf}``.

    Lists are indexed rather than compared whole, so re-ordering an action menu
    shows up as the change it is. An empty container is a leaf in its own right:
    ``{"prior_hypotheses": []}`` and an absent ``prior_hypotheses`` are different
    configurations and the diff has to be able to say which one it is looking at.
    """
    out: dict[str, Any] = {}
    if isinstance(config, Mapping):
        if not config:
            out[prefix or "/"] = {}
            return out
        for key, value in config.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            out.update(flatten(value, f"{prefix}/{token}"))
        return out
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        if not config:
            out[prefix or "/"] = []
            return out
        for index, value in enumerate(config):
            out.update(flatten(value, f"{prefix}/{index}"))
        return out
    out[prefix or "/"] = config
    return out


def _owns(factor_paths: Iterable[str], pointer: str) -> bool:
    """Whether a declared factor's paths cover *pointer*, exactly or as a parent."""
    return any(pointer == path or pointer.startswith(path.rstrip("/") + "/") for path in factor_paths)


def _factor_for(declared: Sequence[Mapping[str, Any]], pointer: str) -> Record | None:
    for factor in declared:
        if _owns([str(p) for p in factor.get("paths") or ()], pointer):
            return dict(factor)
    return None


def configuration_diff(
    base: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    declared_factors: Sequence[Mapping[str, Any]],
    base_id: str = "base",
    variant_id: str = "variant",
) -> Record:
    """Every pointer that differs between two configurations, attributed to a factor.

    This is the record the work order calls a *machine-generated configuration
    diff*. Nothing in it is typed by a caller: the pointers come from flattening
    the two documents, the ownership comes from the manifest's own
    ``declared_factors[].paths``, and the verdict is derived from those two.

    The three outcomes that matter:

    ``undeclared_changes``
        a pointer moved that no declared factor owns. The arm is not a clean
        ablation of anything and :func:`verify_declared_factors_only` refuses it.
    ``declared_but_unchanged``
        a declared factor whose paths moved nothing. The manifest claims an
        ablation it did not perform. Reported, and it makes ``matched`` false —
        an arm that changed nothing is not a control, it is a duplicate.
    ``changes_by_factor``
        the attributable part: which factor each moved pointer belongs to, so a
        later reader can see that ``memory`` really is the only thing separating
        the two sides.
    """
    flat_base = flatten(dict(base))
    flat_variant = flatten(dict(variant))
    declared = [dict(f) for f in declared_factors]

    pointers = sorted(set(flat_base) | set(flat_variant))
    changed: list[Record] = []
    added: list[str] = []
    removed: list[str] = []
    for pointer in pointers:
        in_base = pointer in flat_base
        in_variant = pointer in flat_variant
        if in_base and in_variant:
            if flat_base[pointer] != flat_variant[pointer]:
                changed.append(
                    {"pointer": pointer, "base": flat_base[pointer], "variant": flat_variant[pointer]}
                )
        elif in_variant:
            added.append(pointer)
            changed.append({"pointer": pointer, "base": None, "variant": flat_variant[pointer]})
        else:
            removed.append(pointer)
            changed.append({"pointer": pointer, "base": flat_base[pointer], "variant": None})

    changes_by_factor: dict[str, list[str]] = {}
    undeclared: list[Record] = []
    for entry in changed:
        factor = _factor_for(declared, str(entry["pointer"]))
        if factor is None:
            undeclared.append(dict(entry))
            continue
        changes_by_factor.setdefault(str(factor["factor_id"]), []).append(str(entry["pointer"]))

    declared_but_unchanged = [
        str(f["factor_id"]) for f in declared if not changes_by_factor.get(str(f["factor_id"]))
    ]

    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_configuration_diff",
        "generated_by": f"{__name__}.configuration_diff",
        "base_id": base_id,
        "variant_id": variant_id,
        "base_address": content_address(dict(base)),
        "variant_address": content_address(dict(variant)),
        "n_pointers_base": len(flat_base),
        "n_pointers_variant": len(flat_variant),
        "n_changed": len(changed),
        "changed": changed,
        "added_pointers": added,
        "removed_pointers": removed,
        "declared_factors": [str(f["factor_id"]) for f in declared],
        "declared_paths": {str(f["factor_id"]): [str(p) for p in f.get("paths") or ()] for f in declared},
        "changes_by_factor": {k: sorted(v) for k, v in sorted(changes_by_factor.items())},
        "undeclared_changes": undeclared,
        "n_undeclared_changes": len(undeclared),
        "declared_but_unchanged": sorted(declared_but_unchanged),
        "matched": not undeclared and not declared_but_unchanged,
        "note": (
            "pointers are JSON pointers over the flattened configurations. An empty container is "
            "its own leaf, so declaring a field empty and omitting it are different changes"
        ),
    }


def verify_declared_factors_only(
    base: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Record:
    """Refuse an ablation manifest that changes anything it did not declare.

    Raises:
        ContractError: ``UNSUPPORTED_CLAIM`` naming every undeclared pointer, or
            every declared factor that moved nothing. The message names paths,
            never values, so a refusal cannot print a measurement.
    """
    record = validate_ablation_manifest(manifest)
    diff = configuration_diff(
        dict(base.get("configuration") or base),
        dict(record["configuration"]),
        declared_factors=record["declared_factors"],
        base_id=str(base.get("base_id") or base.get("arm_id") or "base"),
        variant_id=str(record["arm_id"]),
    )
    if diff["undeclared_changes"]:
        paths = ", ".join(str(c["pointer"]) for c in diff["undeclared_changes"][:12])
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            f"{record['arm_id']} changes {diff['n_undeclared_changes']} pointer(s) no declared "
            f"factor owns, so a difference measured against it cannot be attributed to the "
            f"declared factor: {paths}",
            field="configuration",
        )
    if diff["declared_but_unchanged"]:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            f"{record['arm_id']} declares factor(s) {diff['declared_but_unchanged']} that move "
            f"nothing against the base. An arm that changed nothing is a duplicate of the base, "
            f"not a control for it",
            field="declared_factors",
        )
    return diff


def validate_ablation_manifest(manifest: Mapping[str, Any]) -> Record:
    """The manifest contract, checked field by field.

    Raises:
        ContractError: ``MISSING_FIELD`` for an absent required key,
            ``EMPTY_VALUE`` for a blank one, ``INVALID_TYPE`` for the wrong
            shape, ``INVALID_ENUM`` for a label or factor outside the registered
            vocabulary.
    """
    if not isinstance(manifest, Mapping):
        raise _fail(ErrorCategory.INVALID_TYPE, "an ablation manifest is a mapping", field="manifest")
    record = dict(manifest)
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in record:
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                f"an ablation manifest declares {field!r}; a manifest that does not cannot be "
                f"diffed, attributed or costed",
                field=field,
            )
    if record["record_kind"] != _MANIFEST_RECORD_KIND:
        raise _fail(
            ErrorCategory.INVALID_ENUM,
            f"record_kind is {_MANIFEST_RECORD_KIND!r}, not {record['record_kind']!r}",
            field="record_kind",
        )
    for field in ("arm_id", "base_manifest", "endpoint", "isolates"):
        if not str(record[field] or "").strip():
            raise _fail(ErrorCategory.EMPTY_VALUE, f"{field} is declared and blank", field=field)
    if record["inference_label"] not in tuple(InferenceLabel):
        raise _fail(
            ErrorCategory.INVALID_ENUM,
            f"{record['inference_label']!r} is not one of {[x.value for x in InferenceLabel]}",
            field="inference_label",
        )
    if not isinstance(record["configuration"], Mapping) or not record["configuration"]:
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            "configuration is a non-empty mapping; there is nothing to diff otherwise",
            field="configuration",
        )
    seeds = record["seeds"]
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise _fail(ErrorCategory.INVALID_TYPE, "seeds is a list", field="seeds")
    factors = record["declared_factors"]
    if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes)) or not factors:
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            "declared_factors is a non-empty list; an arm that declares no factor attributes nothing",
            field="declared_factors",
        )
    seen: set[str] = set()
    for index, factor in enumerate(factors):
        if not isinstance(factor, Mapping):
            raise _fail(
                ErrorCategory.INVALID_TYPE,
                "each declared factor is a mapping",
                field=f"declared_factors[{index}]",
            )
        for field in FACTOR_REQUIRED_FIELDS:
            if field not in factor:
                raise _fail(
                    ErrorCategory.MISSING_FIELD,
                    f"a declared factor carries {field!r}",
                    field=f"declared_factors[{index}].{field}",
                )
        if factor["attributes_to"] not in tuple(Factor):
            raise _fail(
                ErrorCategory.INVALID_ENUM,
                f"{factor['attributes_to']!r} is not one of {[x.value for x in Factor]}",
                field=f"declared_factors[{index}].attributes_to",
            )
        paths = factor["paths"]
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
            raise _fail(
                ErrorCategory.INVALID_TYPE,
                "a declared factor names at least one JSON pointer it is allowed to move",
                field=f"declared_factors[{index}].paths",
            )
        for path in paths:
            if not str(path).startswith("/"):
                raise _fail(
                    ErrorCategory.INVALID_VALUE,
                    f"{path!r} is not a JSON pointer; it must start with '/'",
                    field=f"declared_factors[{index}].paths",
                )
        identifier = str(factor["factor_id"])
        if identifier in seen:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"factor_id {identifier!r} is declared twice; ownership of a pointer would be ambiguous",
                field=f"declared_factors[{index}].factor_id",
            )
        seen.add(identifier)
    return record


def load_manifest(path: Path | str) -> Record:
    """Read and validate one manifest from disk."""
    return validate_ablation_manifest(json.loads(Path(path).read_text()))


def manifest_paths(directory: Path | str) -> list[Path]:
    """Every arm manifest in *directory*, in a stable order, index excluded."""
    root = Path(directory)
    return sorted(p for p in root.glob("*.json") if p.name != "index.json" and not p.name.startswith("base-"))


# ---------------------------------------------------------------------------
# The mechanism diff: what the code path actually moves
# ---------------------------------------------------------------------------

#: Every field of the proposer's structured feedback that is derived from the
#: campaign's history. Transcribed from
#: :func:`~drug_agent.selfevolve.target.controller.structured_feedback`'s return
#: statement, and checked against it by
#: :func:`history_derived_fields_are_still_these`, so a controller change that
#: adds a tenth history-derived field fails the suite instead of quietly
#: widening what "no history" means.
HISTORY_DERIVED_FIELDS: tuple[str, ...] = (
    "binding_constraints",
    "data_additions",
    "failure_categories",
    "hypothesis_repetitions",
    "negative_findings",
    "prior_hypotheses",
    "uncertainty",
)


def history_suppression_diff(
    history: Sequence[Mapping[str, Any]],
    *,
    campaign: Mapping[str, Any] | None = None,
    incumbent: Mapping[str, Any] | None = None,
    metric: str = "mae_pactivity",
    remaining: Mapping[str, Any] | None = None,
    cycle: int = 2,
) -> Record:
    """What suppressing *history* really changes in the proposer's context.

    ``ABL-1-no-history`` is described in the lock as "``prior_hypotheses`` empty
    and ``binding_constraints`` empty, with an identical action menu, identical
    caps and identical plan contract". That is a claim about the code, and this
    function measures it: it builds the real structured feedback twice, with the
    history and without it, and returns the fields that moved.

    It consumes no budget, contacts no model and reads no partition — the
    feedback record is counts, categories and the campaign's own development
    numbers by construction.
    """
    from .controller import structured_feedback

    context: dict[str, Any] = {
        "campaign": dict(
            campaign or {"campaign_id": "reference", "track": "A", "resource_regime": "CONTROLLED"}
        ),
        "incumbent": dict(incumbent or {"artifact_id": "A-B0-parent-rf", "value": None, "metric": metric}),
        "metric": metric,
        "remaining": dict(remaining or {}),
        "cycle": int(cycle),
    }
    with_history = structured_feedback(history=[dict(r) for r in history], **context)
    without_history = structured_feedback(history=[], **context)

    moved = sorted(
        key
        for key in set(with_history) | set(without_history)
        if with_history.get(key) != without_history.get(key)
    )
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_mechanism_diff",
        "generated_by": f"{__name__}.history_suppression_diff",
        "n_history_records": len(history),
        "fields_moved": moved,
        "n_fields_moved": len(moved),
        "history_derived_fields": list(HISTORY_DERIVED_FIELDS),
        "fields_moved_outside_the_declared_two": [
            field for field in moved if field not in ("prior_hypotheses", "binding_constraints")
        ],
        "declared_but_unmoved": [
            field for field in ("prior_hypotheses", "binding_constraints") if field not in moved
        ],
        "note": (
            "measured by running the real controller context builder twice. No model was called, "
            "no budget was charged and no partition was read"
        ),
    }


def history_derived_fields_are_still_these() -> Record:
    """Re-derive the history-derived field set from the live controller.

    A drift check, not a doc string: it builds a feedback record from a history
    containing one of every outcome the controller branches on, and reports any
    field that moves but is not in :data:`HISTORY_DERIVED_FIELDS`.
    """
    probe = [
        {
            "experiment_id": "probe-improvement",
            "cycle": 1,
            "hypothesis": "a probe hypothesis",
            "action": "fit_predictor",
            "outcome": "DEVELOPMENT_IMPROVEMENT",
            "status": "SUCCEEDED",
            "failure_category": "NONE",
            "development_metric_value": 0.5,
            "improvement": 0.01,
            "evaluated_partition": "development",
            "actual_cost": {"llm_output_tokens": 1.0},
            "data_additions": [{"artifact_id": "probe-resource"}],
        },
        {
            "experiment_id": "probe-negative",
            "cycle": 2,
            "hypothesis": "a second probe hypothesis",
            "action": "fit_predictor",
            "outcome": "NO_IMPROVEMENT",
            "status": "SUCCEEDED",
            "failure_category": "CANDIDATE_ERROR",
            "reason": "a probe reason",
            "development_metric_value": 0.6,
            "improvement": -0.1,
            "evaluated_partition": "development",
        },
        {
            "experiment_id": "probe-rejected",
            "cycle": 3,
            "hypothesis": "a third probe hypothesis",
            "action": "acquire_resource",
            "outcome": "PLAN_REJECTED",
            "status": "FAILED",
            "error_category": "INVALID_ENUM",
            "error_field": "plan.action",
            "reason": "a probe rejection",
        },
    ]
    diff = history_suppression_diff(probe)
    unexpected = [f for f in diff["fields_moved"] if f not in HISTORY_DERIVED_FIELDS]
    missing = [f for f in HISTORY_DERIVED_FIELDS if f not in diff["fields_moved"]]
    return {
        "record_kind": "ablation_mechanism_drift_check",
        "fields_moved": diff["fields_moved"],
        "unexpected_fields": unexpected,
        "declared_but_unmoved_under_the_probe": missing,
        "agrees": not unexpected and not missing,
    }


# ---------------------------------------------------------------------------
# Registration: a comparison, named before its own answer
# ---------------------------------------------------------------------------


def _comparison(
    comparison_id: str,
    *,
    factor: Factor,
    treatment: str,
    control: str,
    isolates: str,
    endpoint: str,
    partition: str,
    inference_label: InferenceLabel,
    required_by_work_order: bool,
    work_order_reference: str,
    evidence_exists: bool,
    arm_id: str | None = None,
    prerequisite: str,
) -> Record:
    return {
        "comparison_id": comparison_id,
        "factor": factor.value,
        "treatment": treatment,
        "control": control,
        "isolates": isolates,
        "endpoint": endpoint,
        "partition": partition,
        "inference_label": inference_label.value,
        "required_by_work_order": required_by_work_order,
        "work_order_reference": work_order_reference,
        "evidence_exists": evidence_exists,
        "ablation_arm": arm_id,
        "prerequisite": prerequisite,
    }


#: The comparisons Step 16 registers. The first five are the work order's own
#: list, in its own order. The last three exist because ``ABL-2``, ``ABL-3`` and
#: the compute factor would otherwise have no named comparison, and the
#: acceptance criterion is that every factor can be told apart — including the
#: ones for which the answer is "no controlled comparison exists".
#:
#: Registering a comparison is NOT selecting a new ablation arm: the four
#: predeclared arms are the closed set and :func:`refuse_new_confirmatory_arm`
#: keeps it closed. Nothing below adds a fifth.
REGISTERED_COMPARISONS: tuple[Record, ...] = (
    _comparison(
        "CMP-1-full-loop-vs-one-shot",
        # work order: register these comparisons: full loop versus one-shot
        factor=Factor.SEARCH,
        treatment="B5-self-evolution",
        control="B4-one-shot-proposal",
        isolates="whether iterating beats one proposal given the identical aggregate budget",
        endpoint="mae_pactivity on the registered held-out pool, paired by campaign",
        partition="held_out",
        inference_label=InferenceLabel.CONFIRMATORY,
        required_by_work_order=True,
        work_order_reference="register these comparisons: full loop versus one-shot",
        evidence_exists=False,
        prerequisite="B5-self-evolution executes and the nomination boundary is released",
    ),
    _comparison(
        "CMP-2-full-vs-no-memory",
        # work order: register these comparisons: full versus no memory
        factor=Factor.MEMORY,
        treatment="B5-self-evolution",
        control="ABL-1-no-history",
        isolates="whether the accumulated history, rather than the model, produces the gain",
        endpoint="mae_pactivity on the registered held-out pool, paired by campaign",
        partition="held_out",
        inference_label=InferenceLabel.CONFIRMATORY,
        required_by_work_order=True,
        arm_id="ABL-1-no-history",
        work_order_reference="register these comparisons: full versus no memory",
        evidence_exists=False,
        prerequisite="B5-self-evolution executes; ABL-1 is funded at its declared 75,000 tokens",
    ),
    _comparison(
        "CMP-3-conventional-adaptation-vs-full",
        # work order: register these comparisons: conventional adaptation versus full
        factor=Factor.TRAINING,
        treatment="B5-self-evolution",
        control="A-B2-standard-adaptation-rf",
        isolates="whether the loop beats an ordinary hyperparameter search over the same learner",
        endpoint="mae_pactivity on the registered held-out pool",
        partition="held_out",
        inference_label=InferenceLabel.CONFIRMATORY,
        required_by_work_order=True,
        work_order_reference="register these comparisons: conventional adaptation versus full",
        evidence_exists=False,
        prerequisite="B5-self-evolution executes; the control is already EXECUTED on development",
    ),
    _comparison(
        "CMP-4-zero-synthetic-vs-distillation",
        # work order: register these comparisons: synthetic weight zero versus distillation
        factor=Factor.DISTILLATION,
        treatment="distillation at synthetic_weight > 0",
        control="the same recipe at synthetic_weight = 0.0",
        isolates="what the synthetic teacher rows buy, holding learner, anchor pool and holdout fixed",
        endpoint="student mean absolute error against the frozen teacher's vina_affinity",
        partition="development",
        inference_label=InferenceLabel.DEVELOPMENT_STAGE,
        required_by_work_order=True,
        work_order_reference="register these comparisons: synthetic weight zero versus distillation",
        evidence_exists=True,
        prerequisite="none: Step 12 executed it on the development partition under a matched setup",
    ),
    _comparison(
        "CMP-5-controlled-vs-acquisition-enabled",
        # work order: register these comparisons: controlled resources versus acquisition-enabled with
        # data/model-matched controls
        factor=Factor.DATA_ACQUISITION,
        treatment="ABL-4-acquisition-enabled",
        control="the CONTROLLED regime with data- and model-matched baselines",
        isolates="end-to-end usefulness versus the source of the gain",
        endpoint="mae_pactivity against a baseline retrained on the final acquired data",
        partition="held_out",
        inference_label=InferenceLabel.POST_HOC_EXPLANATORY,
        required_by_work_order=True,
        arm_id="ABL-4-acquisition-enabled",
        work_order_reference=(
            "register these comparisons: controlled resources versus acquisition-enabled with "
            "data/model-matched controls"
        ),
        evidence_exists=False,
        prerequisite="a working ingestion path on this host, and an exclusion union derivable in advance",
    ),
    _comparison(
        "CMP-6-menu-access-vs-controller-choice",
        # work order: compare full versus no-memory/one-shot/conventional adaptation; the predeclared arm
        # ABL-2 would otherwise have no registered comparison
        factor=Factor.SEARCH,
        treatment="B5-self-evolution",
        control="ABL-2-shared-menu-random",
        isolates="whether choosing well beats choosing at random from the same nine-action menu",
        endpoint="mae_pactivity on the registered held-out pool, paired by campaign",
        partition="held_out",
        inference_label=InferenceLabel.CONFIRMATORY,
        required_by_work_order=False,
        arm_id="ABL-2-shared-menu-random",
        work_order_reference=(
            "compare full versus no-memory/one-shot/conventional adaptation; the predeclared arm "
            "ABL-2 would otherwise have no registered comparison"
        ),
        evidence_exists=False,
        prerequisite="B5-self-evolution executes; ABL-2 runs its five seeds 211-215",
    ),
    _comparison(
        "CMP-7-revert-the-main-changed-factor",
        # work order: for the nominated successful recipe, revert its main changed factor while preserving
        # other settings
        factor=Factor.TRAINING,
        treatment="art_zak_c9r_step11_pilot_d_c01_dock_augmented_rf",
        control="A-B0-parent-rf, which IS that recipe with its main changed factor reverted",
        isolates="what the one development-stage success is worth once its single added feature is removed",
        endpoint="mae_pactivity on the 34-compound development pool, one point estimate each",
        partition="development",
        inference_label=InferenceLabel.DEVELOPMENT_STAGE,
        required_by_work_order=False,
        work_order_reference=(
            "for the nominated successful recipe, revert its main changed factor while preserving "
            "other settings"
        ),
        evidence_exists=True,
        prerequisite="none: both sides were scored by the frozen evaluator in Steps 8 and 11",
    ),
    _comparison(
        "CMP-8-compute-against-qualifying-yield",
        # work order: the report can distinguish ... and compute effects without overstating a unique
        # mechanism
        factor=Factor.COMPUTE,
        treatment="the nine Stage D generator arms at their measured cost",
        control="each other, under identical caps, seeds, scorer and exclusion set",
        isolates="whether qualifying yield tracks compute spent rather than generator identity",
        endpoint="qualifying_scaffolds against measured GPU and wall seconds",
        partition="development",
        inference_label=InferenceLabel.DEVELOPMENT_STAGE,
        required_by_work_order=False,
        work_order_reference=(
            "the report can distinguish ... and compute effects without overstating a unique mechanism"
        ),
        evidence_exists=True,
        prerequisite="none: the sweep is complete and the funnel records carry both quantities",
    ),
    _comparison(
        "CMP-9-docking-feature-vs-the-code-that-found-it",
        factor=Factor.TRAINING,
        treatment="B5-self-evolution, the loop that proposed the feature",
        control="ABL-3-docking-feature-only, the same feature with no agent anywhere in the loop",
        isolates="whether any Track A gain is the docking FEATURE or the code that found it",
        endpoint="mae_pactivity on the registered held-out pool",
        partition="held_out",
        inference_label=InferenceLabel.CONFIRMATORY,
        required_by_work_order=False,
        work_order_reference=(
            "parameter-only versus broader-artifact comparison; the predeclared arm ABL-3 would "
            "otherwise have no registered comparison"
        ),
        arm_id="ABL-3-docking-feature-only",
        evidence_exists=False,
        prerequisite="B5-self-evolution executes; ABL-3 is docked and fitted at its seed 42",
    ),
)


def resolve_inference_label(
    declared: str,
    *,
    predeclared: bool,
    test_results_viewed: bool,
) -> Record:
    """The label an arm is actually entitled to, and why.

    Two downgrades, both one-way:

    * an arm outside ``ablation.arms`` is not predeclared and can never be
      ``CONFIRMATORY``, whatever its manifest says;
    * **any** arm evaluated after a held-out result has been seen is
      explanatory, predeclared or not. That is the rule the work order states and
      it applies to the frozen arms too, because the thing that makes a
      preregistration worth anything is that the answer was not available when
      it was written.
    """
    label = str(declared)
    reasons: list[str] = []
    if test_results_viewed and label == InferenceLabel.CONFIRMATORY.value:
        label = InferenceLabel.POST_HOC_EXPLANATORY.value
        reasons.append(
            "a held-out result has been viewed under this lock, so no ablation evaluated "
            "afterwards is independent confirmation of it"
        )
    if not predeclared and label == InferenceLabel.CONFIRMATORY.value:
        label = InferenceLabel.POST_HOC_EXPLANATORY.value
        reasons.append("the arm is not in the frozen ablation.arms set, so it was not predeclared")
    return {
        "declared": str(declared),
        "resolved": label,
        "downgraded": label != str(declared),
        "reasons": reasons,
        "test_results_viewed": bool(test_results_viewed),
        "predeclared": bool(predeclared),
    }


class ComparisonRegistry:
    """The comparisons, written down before their own final evaluation.

    The registry is append-only and refuses a second registration of the same
    comparison id, because re-registering is how a comparison quietly acquires a
    different control after its first answer is known.
    """

    def __init__(self, *, protocol: Mapping[str, Any], registered_at: str | None = None) -> None:
        self._protocol = dict(protocol)
        self._entries: list[Record] = []
        self._registered_at = registered_at or time.strftime("%Y-%m-%dT%H:%M:%S%z")
        arms = (self._protocol.get("ablation") or {}).get("arms") or []
        self._arm_ids = {str(a.get("arm_id")) for a in arms}

    @property
    def entries(self) -> list[Record]:
        return [dict(e) for e in self._entries]

    def register(
        self,
        comparison: Mapping[str, Any],
        *,
        evidence_exists: bool,
        test_results_viewed: bool = False,
    ) -> Record:
        """Record one comparison, with its honest registration status.

        Raises:
            ContractError: ``INVALID_VALUE`` for a repeat registration;
                ``UNSUPPORTED_CLAIM`` when a comparison whose evidence already
                exists is offered as ``CONFIRMATORY``.
        """
        entry = dict(comparison)
        identifier = str(entry.get("comparison_id") or "")
        if not identifier:
            raise _fail(ErrorCategory.EMPTY_IDENTIFIER, "a comparison needs an id", field="comparison_id")
        if any(e["comparison_id"] == identifier for e in self._entries):
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"{identifier} is already registered. Re-registering a comparison is how its control "
                f"changes after its first answer is known",
                field="comparison_id",
            )
        arm = entry.get("ablation_arm")
        predeclared = arm is None or str(arm) in self._arm_ids
        label = resolve_inference_label(
            str(entry.get("inference_label") or InferenceLabel.POST_HOC_EXPLANATORY.value),
            predeclared=predeclared,
            test_results_viewed=test_results_viewed,
        )
        if evidence_exists and is_confirmatory(label["resolved"]):
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"{identifier} is offered as CONFIRMATORY while its evidence already exists. A "
                f"registration written after the number is a description of it, not a prediction",
                field="inference_label",
            )
        status = (
            RegistrationStatus.EVIDENCE_PREDATES_THIS_REGISTRATION
            if evidence_exists
            else RegistrationStatus.REGISTERED_BEFORE_EVALUATION
        )
        entry.update(
            {
                "registration_status": status.value,
                "registered_at": self._registered_at,
                "inference_label": label["resolved"],
                "inference_label_declared": label["declared"],
                "inference_label_downgraded": label["downgraded"],
                "inference_label_reasons": label["reasons"],
                "predeclared_arm": predeclared,
            }
        )
        entry["registration_address"] = content_address(
            {k: v for k, v in sorted(entry.items()) if k != "registration_address"}
        )
        self._entries.append(entry)
        return dict(entry)

    def record(self) -> Record:
        """The committed registration record."""
        entries = self.entries
        return {
            "contract_version": "1.0.0",
            "record_kind": "ablation_comparison_registry",
            "protocol_version": str(self._protocol.get("protocol_version") or ""),
            "registered_at": self._registered_at,
            "n_comparisons": len(entries),
            "n_required_by_work_order": sum(1 for e in entries if e.get("required_by_work_order")),
            "n_registered_before_evaluation": sum(
                1
                for e in entries
                if e["registration_status"] == RegistrationStatus.REGISTERED_BEFORE_EVALUATION.value
            ),
            "comparisons": entries,
            "closed_arm_set": sorted(self._arm_ids),
            "rule": (
                "the four arms in ablation.arms are the closed confirmatory set. Registering a "
                "comparison does not add an arm, and no comparison here creates one"
            ),
        }


def refuse_new_confirmatory_arm(protocol: Mapping[str, Any], arm_id: str, inference_label: str) -> None:
    """Keep the predeclared set closed.

    Raises:
        ContractError: ``UNSUPPORTED_CLAIM`` when an arm outside ``ablation.arms``
            is offered as ``CONFIRMATORY``.
    """
    arms = {str(a.get("arm_id")) for a in (protocol.get("ablation") or {}).get("arms") or []}
    if is_confirmatory(inference_label) and str(arm_id) not in arms:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            f"{arm_id!r} is not one of the predeclared arms {sorted(arms)}. Selecting further "
            f"confirmatory ablations after the fact is how a study keeps looking until something "
            f"separates; anything outside the frozen set is POST_HOC_EXPLANATORY whatever it shows",
            field="arm_id",
        )


# ---------------------------------------------------------------------------
# Matched-resource accounting
# ---------------------------------------------------------------------------


def matched_resource_accounting(protocol: Mapping[str, Any]) -> Record:
    """Each arm's declared allocation against one confirmation campaign's.

    The rule the work order states — *a no-memory method should still be allowed
    to use the full declared compute budget* — is checked here rather than
    promised: ``ABL-1-no-history`` is matched on every resource the ablation
    block declares, and an arm that consumes zero output tokens by construction
    is allocated the full amount anyway so the comparison is matched rather than
    handicapped.

    Two kinds of mismatch are reported and never collapsed:

    * a resource the reference allocation declares and the ablation block does
      not mention at all (``resources_not_declared_by_the_ablation_block``);
    * an arm whose own declared allocation differs from the symmetric figure
      (``arms_not_matched``).
    """
    budgets = dict(protocol.get("budgets") or {})
    ablation = dict(protocol.get("ablation") or {})
    per_arm = dict(budgets.get("per_arm") or {})
    reference = dict(per_arm.get(REFERENCE_CAMPAIGN_ARM) or {})
    declared = dict(ablation.get("budget_per_arm") or {})

    matched_resources: list[Record] = []
    unmatched_resources: list[Record] = []
    absent_resources: list[str] = []
    for resource in MATCHED_RESOURCES:
        if resource not in reference:
            continue
        if resource not in declared:
            absent_resources.append(resource)
            continue
        row = {
            "resource": resource,
            "reference": reference[resource],
            "ablation": declared[resource],
            "matched": reference[resource] == declared[resource],
        }
        (matched_resources if row["matched"] else unmatched_resources).append(row)

    symmetric_tokens = declared.get("llm_output_tokens")
    arm_rows: list[Record] = []
    for arm in ablation.get("arms") or []:
        arm_id = str(arm.get("arm_id"))
        tokens = arm.get("llm_output_tokens")
        consumes_zero = "zero output tokens" in str(arm.get("note") or "")
        arm_rows.append(
            {
                "arm_id": arm_id,
                "inference_label": arm.get("inference_label"),
                "declared_llm_output_tokens": tokens,
                "symmetric_llm_output_tokens": symmetric_tokens,
                "matched_to_the_symmetric_allocation": tokens == symmetric_tokens,
                "consumes_zero_by_construction": consumes_zero,
                "opportunity_note": (
                    "allocated the full symmetric budget even though the arm spends none of it; "
                    "the matched quantity is opportunity, not spend"
                    if consumes_zero and tokens == symmetric_tokens
                    else None
                ),
                "unmatched_reason": (
                    None
                    if tokens == symmetric_tokens
                    else (
                        f"declares {tokens} against the symmetric {symmetric_tokens}. "
                        f"{arm.get('note') or arm.get('description') or ''}"[:400]
                    )
                ),
                "n_seeds": len(arm.get("seeds") or []),
            }
        )

    total_declared = sum(float(r["declared_llm_output_tokens"] or 0.0) for r in arm_rows)
    authorisation = dict(budgets.get("authorisation_state") or {})
    remaining = authorisation.get("remaining_output_tokens")
    totals = dict(budgets.get("totals_required_for_the_locked_design") or {})
    recorded_total = totals.get("llm_output_tokens_step16_ablation")

    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_matched_resources",
        "generated_by": f"{__name__}.matched_resource_accounting",
        "reference_arm": REFERENCE_CAMPAIGN_ARM,
        "reference_allocation": reference,
        "declared_per_arm": declared,
        "resources_matched": matched_resources,
        "resources_unmatched": unmatched_resources,
        "resources_not_declared_by_the_ablation_block": absent_resources,
        "transferable_between_arms": bool(
            budgets.get("unspent_allocation_is_not_transferable_between_arms") is False
        ),
        "arms": arm_rows,
        "arms_not_matched": [r["arm_id"] for r in arm_rows if not r["matched_to_the_symmetric_allocation"]],
        "total_declared_llm_output_tokens": total_declared,
        "total_recorded_in_the_lock": recorded_total,
        "totals_agree": recorded_total is None or float(recorded_total) == total_declared,
        "remaining_authorised_output_tokens": remaining,
        "funded": remaining is not None and float(remaining) >= total_declared,
        "shortfall_output_tokens": (
            None if remaining is None else max(0.0, total_declared - float(remaining))
        ),
        "budget_symmetry_claim": ablation.get("budget_symmetry"),
        "note": (
            "a resource the ablation block does not declare is UNMATCHED, never assumed equal to "
            "the reference. An arm spending nothing of a matched allocation is matched on "
            "opportunity and unmatched on spend, and both are reported"
        ),
    }


# ---------------------------------------------------------------------------
# Per-arm readiness: what would have to be true for an arm to run
# ---------------------------------------------------------------------------


def _condition(identifier: str, *, requires: str, observed: Any, met: bool, unblock: str) -> Record:
    return {
        "condition": identifier,
        "requires": requires,
        "observed": observed,
        "met": bool(met),
        "unblock": unblock,
    }


def arm_readiness(
    protocol: Mapping[str, Any],
    arm_id: str,
    *,
    comparisons: Sequence[Mapping[str, Any]] = REGISTERED_COMPARISONS,
    regeneration_permitted: bool = False,
    test_access_count: int = 0,
    boundary_released: bool = False,
) -> Record:
    """The gate for one ablation arm: named conditions, each derived from the lock.

    Nothing here is a flag check. The treatment side's execution status comes
    from ``comparators``, the partition from the registered comparison, the
    budget from ``budgets.authorisation_state``, the capability from the arm's
    own recorded reason, and the exclusion-set condition from
    ``ablation.exclusion_set_derivation``. ``regeneration_permitted`` is the one
    condition supplied by the operator, because whether this session may re-dock
    or re-generate is not a property of the protocol.
    """
    ablation = dict(protocol.get("ablation") or {})
    arms = {str(a.get("arm_id")): dict(a) for a in ablation.get("arms") or []}
    if arm_id not in arms:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"{arm_id!r} is not a predeclared arm; the frozen set is {sorted(arms)}",
            field="arm_id",
        )
    arm = arms[arm_id]
    comparators = {str(c.get("id")): dict(c) for c in protocol.get("comparators") or []}
    related = [dict(c) for c in comparisons if c.get("ablation_arm") == arm_id]
    treatment_ids = [str(c["treatment"]) for c in related]
    partitions = {str(c["partition"]) for c in related}

    conditions: list[Record] = []

    treatment_status = {t: str(comparators.get(t, {}).get("status") or "UNREGISTERED") for t in treatment_ids}
    treatment_executed = bool(treatment_status) and all(v == "EXECUTED" for v in treatment_status.values())
    conditions.append(
        _condition(
            "TREATMENT_ARM_EXECUTED",
            requires="the arm this ablation is a control for has produced a result to explain",
            observed=treatment_status,
            met=treatment_executed,
            unblock="execute the main research arm under an authorised budget",
        )
    )

    needs_held_out = "held_out" in partitions
    conditions.append(
        _condition(
            "REGISTERED_EVALUATION_REACHABLE",
            requires=(
                "the arm's registered endpoint does not read the held-out partition, or the "
                "nomination boundary has been released"
            ),
            observed={
                "partitions": sorted(partitions),
                "test_access_count": int(test_access_count),
                "boundary_released": bool(boundary_released),
            },
            met=(not needs_held_out) or bool(boundary_released),
            unblock=(
                "complete all five locked campaign nominations, then release the boundary through "
                "the registered independent evaluation harness under operator approval"
            ),
        )
    )

    authorisation = dict((protocol.get("budgets") or {}).get("authorisation_state") or {})
    remaining = authorisation.get("remaining_output_tokens")
    needed = float(arm.get("llm_output_tokens") or 0.0)
    conditions.append(
        _condition(
            "BUDGET_AUTHORISED",
            requires=f"{needed:g} output tokens are available for this arm alone, non-transferable",
            observed={"declared": needed, "remaining_authorised": remaining},
            met=remaining is not None and float(remaining) >= needed,
            unblock="an explicit operator authorisation covering this arm's own allocation",
        )
    )

    reason = str(arm.get("description") or "") + " " + str(arm.get("note") or "")
    capability_blocked = "CANNOT be frozen" in reason or "no working ingestion path" in reason
    conditions.append(
        _condition(
            "CAPABILITY_AVAILABLE",
            requires="every capability the arm's regime needs works on this host",
            observed={
                "arm_records_a_blocking_capability": capability_blocked,
                "detail": (
                    "no working ingestion path on this host; both acquire_resource cycles failed"
                    if capability_blocked
                    else "no blocking capability recorded for this arm"
                ),
            },
            met=not capability_blocked,
            unblock="a working ingestion path, verified by a real acquire_resource cycle",
        )
    )

    derivation = str(ablation.get("exclusion_set_derivation") or "")
    widens = "ACQUISITION_ENABLED" in reason and "WIDENS" in derivation
    conditions.append(
        _condition(
            "EXCLUSION_SET_DERIVABLE_IN_ADVANCE",
            requires="the common exclusion union can be derived and hashed before any arm is evaluated",
            observed={
                "regime_widens_the_union": widens,
                "frozen_union_hash": (
                    "sha256:5ec21d5a4ea1135f7bd8fc5067b4a83f4d7bc0779d0c52f41bf226ee0198c70e"
                ),
            },
            met=not widens,
            unblock="run the acquisition arm first, re-derive the widened union, then re-hash before scoring",
        )
    )

    # A deterministic arm still needs new computation here, and saying so is the
    # point: ABL-3 consumes no tokens and is not blocked by the shortfall, but it
    # needs docking scores for 136 compounds that this tree does not cache. The
    # budget reason and the computation reason are different facts, and a later
    # worker with docking approval has to be able to tell them apart.
    deterministic = "deterministic" in str(arm.get("note") or "")
    conditions.append(
        _condition(
            "EXECUTION_WITHOUT_REGENERATION",
            requires="the arm can produce its evidence without re-generating or re-docking anything",
            observed={
                "regeneration_permitted_in_this_session": bool(regeneration_permitted),
                "arm_requires_new_computation": True,
                "arm_is_deterministic": deterministic,
                "detail": (
                    "the arm needs an AutoDock Vina score for all 102 train-partition and 34 "
                    "development-pool compounds, and no cache of those scores exists in this "
                    "tree: the Step 11 pilot computed them inside a candidate checkout and "
                    "persisted only its pActivity predictions"
                    if deterministic
                    else "the arm runs a campaign loop whose interventions generate, fit and dock"
                ),
            },
            met=bool(regeneration_permitted),
            unblock="operator approval to spend the docking and generation cost this arm needs",
        )
    )

    blocked = [c["condition"] for c in conditions if not c["met"]]
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_arm_readiness",
        "arm_id": arm_id,
        "declared_inference_label": arm.get("inference_label"),
        "isolates": arm.get("isolates"),
        "seeds": list(arm.get("seeds") or []),
        "conditions": conditions,
        "n_conditions": len(conditions),
        "n_blocked": len(blocked),
        "blocked_conditions": blocked,
        "decision": ArmStatus.EXECUTED.value if not blocked else ArmStatus.NOT_RUN.value,
        "comparisons": [str(c["comparison_id"]) for c in related],
    }


#: The resource names an arm's ``declared_cost`` and ``cost_actually_spent`` may
#: use. An ALLOWLIST, for the same reason :data:`COST_RESOURCE_FIELDS` is one in
#: the confirmation module: a budget key the lock adds and this set does not name
#: is refused loudly, while a denylist that falls behind would admit a held-out
#: number silently. ``declared_cost`` is copied out of ``ablation.budget_per_arm``,
#: which is part of the protocol document.
ARM_COST_RESOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "llm_output_tokens",
        "llm_input_tokens",
        "gpu_seconds",
        "cpu_seconds",
        "wall_seconds",
        "experiment_proposals",
        "docking_calls",
        "oracle_calls",
        "rows",
        "usd",
    }
)

#: Where an ``ablation_arm_not_run`` record may carry a magnitude. The same
#: inversion :func:`~.confirmation.refuse_false_assurance` applies to the
#: confirmation branch-status record: counts, protocol-declared costs at
#: registered resource names, and the arm's declared seeds. A number anywhere
#: else — including one arriving through a hand-written readiness document, which
#: is copied into this record verbatim — is refused without being read.
ARM_NOT_RUN_MAGNITUDE_PATHS: frozenset[str] = frozenset(
    {
        "n_blocking_conditions",
        "seeds[]",
        "conditions_for_a_legitimate_later_run[].observed.test_access_count",
        "conditions_for_a_legitimate_later_run[].observed.declared",
        "conditions_for_a_legitimate_later_run[].observed.remaining_authorised",
    }
    | {f"declared_cost.{field}" for field in ARM_COST_RESOURCE_FIELDS}
    | {f"cost_actually_spent.{field}" for field in ARM_COST_RESOURCE_FIELDS}
)


def arm_not_run_record(protocol: Mapping[str, Any], readiness: Mapping[str, Any]) -> Record:
    """The committed statement for an arm that did not run: status, reason, cost, unblock.

    The "conditions under which it could later run" are the gate's own condition
    list rather than a prose restatement of it, so a later worker satisfies the
    thing that is actually checked.

    This record carries ``test_access_count: 0`` inside those conditions, which
    makes it an assertion that the held-out partition was not read — and it is
    assembled from a ``readiness`` mapping that a caller supplies and that reaches
    this module off disk. It had no false-assurance guard at all until this round
    — the module carried no reference to one. It has one now, and it is the
    structural one: see
    :func:`~.confirmation.refuse_false_assurance` and
    :data:`ARM_NOT_RUN_MAGNITUDE_PATHS`.
    """
    arm_id = str(readiness["arm_id"])
    ablation = dict(protocol.get("ablation") or {})
    arms = {str(a.get("arm_id")): dict(a) for a in ablation.get("arms") or []}
    arm = arms.get(arm_id, {})
    blocked = [c for c in readiness["conditions"] if not c["met"]]
    record: Record = {
        "contract_version": "1.0.0",
        "record_kind": "ablation_arm_not_run",
        "arm_id": arm_id,
        "status": ArmStatus.NOT_RUN.value,
        "inference_label": arm.get("inference_label"),
        "isolates": arm.get("isolates"),
        "seeds": list(arm.get("seeds") or []),
        "reason": "; ".join(f"{c['condition']}: {c['requires']} — not met" for c in blocked)
        or "no blocking condition",
        "n_blocking_conditions": len(blocked),
        "declared_cost": {
            "llm_output_tokens": arm.get("llm_output_tokens"),
            **{k: v for k, v in (ablation.get("budget_per_arm") or {}).items() if k != "llm_output_tokens"},
        },
        "cost_actually_spent": {
            "llm_output_tokens": 0.0,
            "gpu_seconds": 0.0,
            "wall_seconds": 0.0,
            "note": "the arm did not start; nothing was charged and nothing was reserved",
        },
        "conditions_for_a_legitimate_later_run": readiness["conditions"],
        "never": [
            "shrinking the seed set to fit the remaining budget",
            "widening a gate, threshold, margin or cap so an arm separates",
            "selecting a further ablation after this one fails to separate",
        ],
    }
    # The last thing that happens, exactly as in `not_run_record`: this record
    # may not be written over a magnitude it was never entitled to carry,
    # however that magnitude reached it.
    refuse_false_assurance(
        record,
        allowed_magnitude_paths=ARM_NOT_RUN_MAGNITUDE_PATHS,
        what=f"the branch-status record for arm {arm_id!r}",
    )
    return record


# ---------------------------------------------------------------------------
# The three comparisons whose evidence already exists
# ---------------------------------------------------------------------------


def distillation_attribution(
    ablation_record: Mapping[str, Any],
    repeats_record: Mapping[str, Any] | None = None,
) -> Record:
    """The ``synthetic_weight = 0.0`` control against the distilled students.

    Reads Step 12's executed records. The matched setup is re-checked here rather
    than believed: one base configuration, one anchor pool and one holdout across
    every arm is what makes the weight the only thing that differs, and if the
    record does not say so the comparison is reported as unmatched.
    """
    raw_matched = dict(ablation_record.get("matched_setup") or {})
    # Counts and addresses only. The holdout family list is a list of molecule
    # identities; it is development chemistry rather than a held-out label, and
    # it still has no business in a committed attribution record.
    matched = {
        "matched": bool(raw_matched.get("matched")),
        "base_config_hash": list(raw_matched.get("base_config_hash") or ()),
        "anchor_pool_hash": list(raw_matched.get("anchor_pool_hash") or ()),
        "n_base_config_hashes": len(raw_matched.get("base_config_hash") or ()),
        "n_anchor_pool_hashes": len(raw_matched.get("anchor_pool_hash") or ()),
        "n_holdout_families": len(raw_matched.get("holdout_families") or ()),
        "n_holdout_rows": raw_matched.get("n_holdout_rows"),
        "why_one_of_each_is_the_point": (
            "one base configuration, one anchor pool and one holdout across every arm is what "
            "makes the synthetic weight the only thing that differs between them"
        ),
    }
    arms = [
        {
            "synthetic_weight": a.get("synthetic_weight"),
            "n_train_rows": a.get("n_train_rows"),
            "n_anchor_rows": a.get("n_anchor_rows"),
            "n_synthetic_rows": a.get("n_synthetic_rows"),
            "base_config_hash": a.get("base_config_hash"),
            "anchor_pool_hash": a.get("anchor_pool_hash"),
        }
        for a in ablation_record.get("arms") or []
    ]
    rows = [dict(r) for r in ablation_record.get("comparison_against_no_distillation") or []]
    across = [dict(r) for r in (repeats_record or {}).get("across_seed_summary") or []]
    control = rows[0].get("control_mae_vs_teacher") if rows else None
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_distillation_attribution",
        "comparison_id": "CMP-4-zero-synthetic-vs-distillation",
        "factor": Factor.DISTILLATION.value,
        "inference_label": InferenceLabel.DEVELOPMENT_STAGE.value,
        "status": ArmStatus.EXECUTED.value,
        "partition": "development",
        "endpoint": "student mean absolute error against the frozen teacher's vina_affinity",
        "control_arm": "synthetic_weight = 0.0, which really trains",
        "control_value": control,
        "matched_setup": matched,
        "matched": bool(matched["matched"]),
        "arms": arms,
        "single_seed_rows": [
            {
                "synthetic_weight": r.get("synthetic_weight"),
                "mae_vs_teacher": r.get("mae_vs_teacher"),
                "control_mae_vs_teacher": r.get("control_mae_vs_teacher"),
                "delta_mae_vs_control": r.get("delta_mae_vs_control"),
                "paired_interval": r.get("paired_interval"),
                "interval_note": r.get("interval_note"),
            }
            for r in rows
        ],
        "across_seed_rows": [
            {
                "synthetic_weight": r.get("synthetic_weight"),
                "n_seeds": r.get("n_seeds"),
                "n_seeds_favouring_distillation": r.get("n_seeds_favouring_distillation"),
                "mean_delta_mae": r.get("mean_delta_mae"),
                "interval": (r.get("across_seed_interval") or {}).get("interval"),
                "excludes_no_improvement": r.get("excludes_no_improvement"),
                "precision_note": r.get("precision_note"),
            }
            for r in across
        ],
        "claim_scope": ablation_record.get("claim_scope"),
        "what_it_cannot_separate": (
            "two things. First the teacher's own cost: a distilled scorer is cheap only after "
            "somebody paid for the docking that labelled it, and the efficiency claim is on Track "
            "C's imitation endpoint, not on potency, binding or the registered primary metric. "
            "Second, TRAINING-SET SIZE from synthetic SIGNAL: the control fits 102 anchor rows and "
            "each distilled arm fits 167, because the synthetic rows arrive with the weight. Three "
            "arms cannot decompose that, and no reading of this comparison may claim the teacher's "
            "labels rather than the extra rows are what moved the error"
        ),
        "unmatched_factors": [
            {
                "factor": "training_set_size",
                "control": 102,
                "treatment": 167,
                "why_it_cannot_be_matched": (
                    "the synthetic rows are the intervention; removing them is what the control IS. "
                    "Matching row counts would mean adding rows the control has no source for"
                ),
            }
        ],
    }


def revert_main_factor_attribution(
    recipe: Mapping[str, Any],
    reverted: Mapping[str, Any],
    *,
    reference: Mapping[str, Any] | None = None,
) -> Record:
    """A development-stage recipe against the revert of its single changed factor.

    The work order asks for the nominated successful recipe's main changed factor
    to be reverted with every other setting preserved. There is no nominee — the
    confirmation is ``NOT_RUN`` — so this is applied to the only recipe in the
    programme that improved anything: the Step 11 pilot's dock-augmented random
    forest. Reverting its one added feature *is* the parent baseline, which was
    already fitted and scored by the same frozen evaluator on the same pool, so
    the revert costs nothing and is matched by construction.
    """
    recipe_value = recipe.get("value")
    reverted_value = reverted.get("value")
    delta = None
    if isinstance(recipe_value, (int, float)) and isinstance(reverted_value, (int, float)):
        delta = float(recipe_value) - float(reverted_value)
    bar = dict(reference or {})
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_revert_attribution",
        "comparison_id": "CMP-7-revert-the-main-changed-factor",
        "factor": Factor.TRAINING.value,
        "inference_label": InferenceLabel.DEVELOPMENT_STAGE.value,
        "status": ArmStatus.EXECUTED.value,
        "partition": "development",
        "endpoint": recipe.get("metric"),
        "orientation": "LOWER_IS_BETTER",
        "recipe": recipe,
        "reverted": reverted,
        "main_changed_factor": "one AutoDock Vina docking score plus a valid/imputed flag, appended "
        "to the parent's Morgan r2/2048 feature matrix",
        "delta": delta,
        "resource_opportunity": {
            "recipe_llm_output_tokens": 51748.0,
            "recipe_wall_seconds": 1510.949,
            "revert_llm_output_tokens": 0.0,
            "revert_wall_seconds": 0.0,
            "matched": False,
            "why_not": (
                "the revert is the parent, which was already fitted in Step 8. Giving the reverted "
                "arm the same 51,748-token opportunity would mean running a second agent campaign, "
                "which is unfunded. The asymmetry is in the recipe's favour and is stated rather "
                "than corrected"
            ),
        },
        "reference_bar": bar,
        "interval": None,
        "decision": "INCONCLUSIVE",
        "decision_reason": (
            "one point estimate per side on one 34-compound development pool, with no uncertainty "
            "interval. Appendix A §5 calls a favourable point estimate without interval support "
            "INCONCLUSIVE whatever its sign"
        ),
        "negative_finding": (
            "both sides are worse than the trivial median constant on the same pool, so neither is "
            "demonstrated target signal. This is reported because it is the raw result, not because "
            "it is the comfortable one"
        )
        if isinstance(bar.get("value"), (int, float))
        and isinstance(recipe_value, (int, float))
        and float(recipe_value) > float(bar["value"])
        else None,
    }


def sweep_compute_attribution(report: Mapping[str, Any]) -> Record:
    """Does qualifying yield track compute spent, or generator identity?

    A re-read of the completed nine-arm sweep's own funnel records. No molecule
    is regenerated, nothing is re-docked and no gate, threshold or cap is
    touched: the qualifying counts are the ones the sweep already reported under
    the frozen seven-gate rule, and the cost is the cost it already measured.
    """
    funnel = [dict(r) for r in report.get("generation_funnel") or []]
    by_arm: dict[str, Record] = {}
    for row in funnel:
        arm = str(row.get("arm"))
        entry = by_arm.setdefault(
            arm,
            {
                "arm": arm,
                "conditioning": row.get("conditioning"),
                "qualifying_scaffolds": 0,
                "gpu_seconds": 0.0,
                "wall_seconds": 0.0,
                "n_seeds": 0,
                "gpu_telemetry": row.get("gpu_telemetry"),
            },
        )
        entry["qualifying_scaffolds"] = int(entry["qualifying_scaffolds"]) + int(
            row.get("qualifying_scaffolds") or 0
        )
        gpu = row.get("gpu_seconds")
        entry["gpu_seconds"] = float(entry["gpu_seconds"]) + (float(gpu) if gpu is not None else 0.0)
        wall = row.get("wall_seconds")
        entry["wall_seconds"] = float(entry["wall_seconds"]) + (float(wall) if wall is not None else 0.0)
        entry["n_seeds"] = int(entry["n_seeds"]) + 1
        if gpu is None:
            entry["gpu_telemetry"] = "PARTIAL"

    rows = sorted(by_arm.values(), key=lambda r: (-int(r["qualifying_scaffolds"]), str(r["arm"])))

    # The check that proves nothing was re-scored: the per-arm counts this
    # re-read produces must be the counts the sweep's own registered primary
    # table already reports. If they disagree, a gate moved and the comparison
    # is void, whatever it shows.
    registered = {
        str(row.get("arm")): int(row.get("qualifying_scaffolds") or 0)
        for stratum in ((report.get("primary") or {}).get("strata") or {}).values()
        for row in stratum
    }
    derived = {str(r["arm"]): int(r["qualifying_scaffolds"]) for r in rows}
    reproduces = bool(registered) and registered == derived
    yields = [int(r["qualifying_scaffolds"]) for r in rows]
    gpus = [float(r["gpu_seconds"]) for r in rows]
    correlation = _spearman(gpus, yields)
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_compute_attribution",
        "comparison_id": "CMP-8-compute-against-qualifying-yield",
        "factor": Factor.COMPUTE.value,
        "inference_label": InferenceLabel.DEVELOPMENT_STAGE.value,
        "status": ArmStatus.EXECUTED.value,
        "partition": "development",
        "n_arms": len(rows),
        "arms": rows,
        "total_gpu_seconds": round(sum(gpus), 3),
        "total_qualifying_scaffolds": sum(yields),
        "spearman_gpu_seconds_vs_qualifying_scaffolds": correlation,
        "perfect_value_if_yield_were_bought_with_compute": 1.0,
        "caps_identical_across_arms": True,
        "what_it_shows": (
            "every arm ran under identical caps, seeds, scorer and exclusion set, so compute is "
            "matched by design rather than by adjustment. A rank correlation near zero means the "
            "spread in qualifying yield is not bought with occupancy"
        ),
        "what_it_cannot_separate": (
            "n = 9 arms with most yields at zero. The correlation is descriptive and no arm-level "
            "claim follows from it; the sweep's own registered test is the Poisson tail under Holm, "
            "not this"
        ),
        "no_regeneration": True,
        "no_gate_changed": True,
        "reproduces_the_registered_primary_table": reproduces,
        "registered_primary_counts": registered,
        "disagreements_with_the_registered_primary_table": sorted(
            arm for arm in set(registered) | set(derived) if registered.get(arm) != derived.get(arm)
        ),
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation with midranks, or ``None`` when it is undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return None
    return round(num / ((dx * dy) ** 0.5), 6)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        midrank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = midrank
        index = stop + 1
    return ranks


# ---------------------------------------------------------------------------
# The attribution table
# ---------------------------------------------------------------------------


def attribution_table(
    registry: Mapping[str, Any],
    *,
    executed: Sequence[Mapping[str, Any]] = (),
    not_run: Sequence[Mapping[str, Any]] = (),
) -> Record:
    """One row per factor: the controlled comparison, its status, what it cannot say.

    The acceptance criterion for this step is that the report can *distinguish*
    six effects. A factor with no controlled comparison is a row saying so, not
    an absent row: the thing a reader must be able to see is which of the six the
    evidence separates and which it does not.
    """
    comparisons = [dict(c) for c in registry.get("comparisons") or []]
    executed_by_id = {str(e.get("comparison_id")): dict(e) for e in executed}
    not_run_by_arm = {str(n.get("arm_id")): dict(n) for n in not_run}

    rows: list[Record] = []
    for factor in Factor:
        related = [c for c in comparisons if c["factor"] == factor.value]
        evidence: list[Record] = []
        for comparison in related:
            result = executed_by_id.get(str(comparison["comparison_id"]))
            arm = comparison.get("ablation_arm")
            blocked = not_run_by_arm.get(str(arm)) if arm else None
            evidence.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "treatment": comparison["treatment"],
                    "control": comparison["control"],
                    "inference_label": comparison["inference_label"],
                    "registration_status": comparison["registration_status"],
                    "status": (ArmStatus.EXECUTED.value if result else ArmStatus.NOT_RUN.value),
                    "partition": comparison["partition"],
                    "result": result,
                    "not_run_reason": (
                        (blocked or {}).get("reason") if blocked else comparison["prerequisite"]
                    ),
                }
            )
        has_executed = any(e["status"] == ArmStatus.EXECUTED.value for e in evidence)
        rows.append(
            {
                "factor": factor.value,
                "n_comparisons": len(evidence),
                "evidence": evidence,
                "separated_by_a_controlled_comparison": has_executed,
                "confirmatory_evidence": any(
                    e["status"] == ArmStatus.EXECUTED.value and is_confirmatory(e["inference_label"])
                    for e in evidence
                ),
            }
        )

    separated = [r["factor"] for r in rows if r["separated_by_a_controlled_comparison"]]
    unseparated = [r["factor"] for r in rows if not r["separated_by_a_controlled_comparison"]]
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_attribution_table",
        "factors": rows,
        "factors_separated_on_development_evidence": separated,
        "factors_with_no_controlled_comparison_executed": unseparated,
        "confirmatory_attribution_available": any(r["confirmatory_evidence"] for r in rows),
        "unique_mechanism_claim": unique_mechanism_statement(separated, unseparated),
    }


def unique_mechanism_statement(separated: Sequence[str], unseparated: Sequence[str]) -> Record:
    """Refuse a unique-mechanism claim that the evidence cannot carry.

    A single separated factor does not become *the* mechanism while five others
    are unmeasured, and this is the sentence that says so in the record itself
    rather than leaving it to a reader's charity.
    """
    supported = bool(separated) and not unseparated
    return {
        "supported": supported,
        "separated": list(separated),
        "unseparated": list(unseparated),
        "statement": (
            "every registered factor has an executed controlled comparison, so a mechanism claim "
            "is attributable"
            if supported
            else (
                f"{len(unseparated)} of {len(separated) + len(unseparated)} factors have no executed "
                f"controlled comparison ({', '.join(unseparated)}), so no unique mechanism may be "
                f"claimed for any observed difference. What is separated is separated on "
                f"development-partition evidence only"
            )
        ),
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def ablation_report(
    protocol: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    resources: Mapping[str, Any],
    readiness: Sequence[Mapping[str, Any]],
    not_run: Sequence[Mapping[str, Any]],
    executed: Sequence[Mapping[str, Any]] = (),
    diffs: Sequence[Mapping[str, Any]] = (),
    mechanism: Mapping[str, Any] | None = None,
    test_access: Mapping[str, Any] | None = None,
    **provenance: Any,
) -> Record:
    """Everything this step can say, with the ordering that makes it readable."""
    table = attribution_table(registry, executed=executed, not_run=not_run)
    unmatched: list[Record] = []
    for diff in diffs:
        if diff.get("undeclared_changes") or diff.get("declared_but_unchanged"):
            unmatched.append(
                {
                    "variant_id": diff.get("variant_id"),
                    "undeclared_changes": [c["pointer"] for c in diff.get("undeclared_changes") or []],
                    "declared_but_unchanged": diff.get("declared_but_unchanged"),
                }
            )
    if mechanism and mechanism.get("fields_moved_outside_the_declared_two"):
        unmatched.append(
            {
                "variant_id": "ABL-1-no-history",
                "kind": "MECHANISM",
                "fields_moved_outside_the_declared_two": mechanism["fields_moved_outside_the_declared_two"],
                "detail": (
                    "the arm's own description names prior_hypotheses and binding_constraints. "
                    "Suppressing the history moves these as well, so the arm ablates more than it "
                    "declares and a difference measured against it cannot be attributed to the two "
                    "named fields alone"
                ),
            }
        )
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_attribution_report",
        "protocol_version": protocol.get("protocol_version"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "test_results_viewed": False,
        "test_results_viewed_evidence": test_access
        or {
            "note": "no test-access scan was supplied to this report",
        },
        "post_hoc_because_a_test_result_was_seen": False,
        "why": (
            "no held-out result exists anywhere in this study. test_access_count is 0 in every "
            "record that declares it, the confirmation branch is NOT_RUN, and the main research arm "
            "B5-self-evolution has never been executed. Nothing here was selected after seeing an "
            "answer, and nothing here is independent confirmation of one either"
        ),
        "registry": registry,
        "matched_resources": resources,
        "readiness": list(readiness),
        "not_run": list(not_run),
        "executed": list(executed),
        "configuration_diffs": list(diffs),
        "mechanism_diff": mechanism,
        "unmatched_factors": unmatched,
        "n_unmatched_factors": len(unmatched),
        "attribution": table,
        "arms_executed": [str(e.get("comparison_id")) for e in executed],
        "arms_not_run": [str(n.get("arm_id")) for n in not_run],
        **provenance,
    }


__all__ = [
    "ARM_COST_RESOURCE_FIELDS",
    "ARM_NOT_RUN_MAGNITUDE_PATHS",
    "ArmStatus",
    "CONTRACT",
    "ComparisonRegistry",
    "FACTOR_REQUIRED_FIELDS",
    "Factor",
    "HISTORY_DERIVED_FIELDS",
    "InferenceLabel",
    "MANIFEST_REQUIRED_FIELDS",
    "MATCHED_RESOURCES",
    "PROTOCOL_INFERENCE_LABELS",
    "REFERENCE_CAMPAIGN_ARM",
    "REGISTERED_COMPARISONS",
    "REGISTERED_PROTOCOL_HASH",
    "REGISTERED_PROTOCOL_VERSION",
    "RegistrationStatus",
    "ablation_report",
    "arm_not_run_record",
    "arm_readiness",
    "attribution_table",
    "canonical_json",
    "configuration_diff",
    "content_address",
    "distillation_attribution",
    "flatten",
    "history_derived_fields_are_still_these",
    "history_suppression_diff",
    "is_confirmatory",
    "load_manifest",
    "manifest_paths",
    "matched_resource_accounting",
    "refuse_new_confirmatory_arm",
    "resolve_inference_label",
    "revert_main_factor_attribution",
    "sweep_compute_attribution",
    "unique_mechanism_statement",
    "validate_ablation_manifest",
    "verify_declared_factors_only",
]
