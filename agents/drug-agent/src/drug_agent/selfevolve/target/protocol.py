"""Step 14 — the frozen study protocol, its canonical hash and its exposure boundary.

A protocol lock is the record that makes a confirmation study *pre*-registered
rather than narrated afterwards. It fixes, before any held-out result exists:
which question is primary, what counts as success, what it is compared against,
how uncertainty is estimated, what each arm may spend, which seeds run, and how
a campaign that produces nothing is counted. Appendix A §5 requires it in
``protocol.lock.json``; the step prompt names the fourteen fields it must carry.

Three properties are load-bearing here, and each one has a refusal behind it:

* **A lock determines the inference, so it cannot out-claim its own arithmetic.**
  ``feasibility.classification`` may read ``CONFIRMATORY`` only when the frozen
  power block reaches the frozen ``min_power`` *at the worst case of its own
  variance interval*. A study that is underpowered is labelled, never relaxed —
  which is this step's acceptance criterion expressed as code rather than as a
  sentence in a document.
* **A lock is a content address, not a file.** :func:`canonical_json` is the one
  serialisation, :func:`protocol_hash` is the address over it, and key order or
  whitespace cannot move it while any scientific field must.
* **A lock cannot be rewritten after a test result is revealed.** Once
  ``evaluation_exposure.test_access_count`` is non-zero, the only admissible
  record is one that declares what it ``supersedes`` and sets
  ``presented_as_original_confirmation`` to false. There is no path that
  produces a freshly frozen "original" confirmation after exposure.

Nothing in this module measures anything. :func:`assess_power` does arithmetic
on numbers a pilot measured and returns a classification with its inputs
attached; :func:`pilot_assessment` reads campaign records that already ran and
summarises them. Both keep an unmeasured quantity unmeasured: an unknown
telemetry value is ``None`` beside an explicit state, never a zero.

**Step 15C adds a second kind of power arithmetic**, because ``zak-c9r-2``
redirects the primary endpoint from a per-campaign regression error to a *count*
of qualifying scaffolds per generator arm per generation seed.
:func:`assess_power`'s paired-t machinery does not transfer to a count, and the
thing that decides a count design is not a standard deviation but whether the
exact test at the locked replication unit can reach ``alpha`` at all.
:func:`assess_count_power` answers that, and a lock may now carry more than one
power block — every one of which is validated, and all of which must reach their
own ``min_power`` before anything in the study may be called confirmatory.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from .contracts import ContractError, ErrorCategory

Record = dict[str, Any]

CONTRACT = "StudyProtocol"

#: Canonical content address, the same shape :mod:`.contracts` accepts.
_CONTENT_ADDRESS = re.compile(r"^[a-z0-9]+:[0-9a-f]{32,128}$")

#: The fourteen fields the step prompt requires in the lock. Every one of them
#: must be present *and* non-empty in a ``confirmation`` protocol.
REQUIRED_LOCK_FIELDS: tuple[str, ...] = (
    "target_hash",
    "data_hash",
    "split_hash",
    "evaluator_hash",
    "baseline_hashes",
    "primary_metric",
    "selection_rule",
    "margins",
    "budgets",
    "campaign_seeds",
    "deployment_seeds",
    "statistics",
    "max_final_nominations",
    "resource_regime",
)

#: The four fields above that are single content addresses.
_HASH_FIELDS: tuple[str, ...] = ("target_hash", "data_hash", "split_hash", "evaluator_hash")

#: Decisions the statistics block freezes. Appendix A §5 asks for the replicate
#: hierarchy, the family-resampling policy, the number of draws, the seed, the
#: interval level, the treatment of a failed/no-nominee campaign and any
#: multiplicity correction; ``power`` carries the feasibility arithmetic.
REQUIRED_STATISTICS_FIELDS: tuple[str, ...] = (
    "replicate_hierarchy",
    "family_resampling",
    "n_bootstrap_draws",
    "bootstrap_seed",
    "interval_level",
    "failed_campaign_treatment",
    "multiplicity_correction",
    "power",
)

#: Numbers the power block must carry for the claim gate to be checkable.
REQUIRED_POWER_FIELDS: tuple[str, ...] = (
    "min_power",
    "locked_n_campaigns",
    "effect_under_test",
    "process_sd_point",
    "process_sd_interval",
    "power_at_locked_n_point",
    "power_at_locked_n_worst_case",
    "n_for_min_power_worst_case",
    "evidence_class",
)

PHASES: tuple[str, ...] = ("pilot", "confirmation")
RESOURCE_REGIMES: tuple[str, ...] = ("CONTROLLED", "ACQUISITION_ENABLED")
FEASIBILITY_CLASSIFICATIONS: tuple[str, ...] = ("CONFIRMATORY", "EXPLORATORY", "INFEASIBLE")
INFERENCE_LABELS: tuple[str, ...] = ("CONFIRMATORY", "POST_HOC_EXPLANATORY")

#: Optional fields a protocol may carry beside the required fourteen. Anything
#: outside this union is ``UNKNOWN_FIELD``: a lock that silently carries an
#: unrecognised key is a lock somebody can add a loophole to.
_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "contract_version",
        "record_kind",
        "phase",
        "protocol_version",
        "target_id",
        "target_alias",
        "primary_track",
        "primary_track_rationale",
        "primary_endpoint",
        "registered_tracks",
        "comparators",
        "comparator_reporting_rule",
        "intervention_menu",
        "qualification_rules",
        "nomination_rule",
        "feasibility",
        "ablation",
        "novelty_exclusion",
        "evaluation_exposure",
        "pilot_seeds",
        "pilot_seeds_note",
        "deployment_seeds_note",
        "resource_regime_note",
        "pilot_evidence",
        "deviations",
        "notes",
        "supersedes",
        "presented_as_original_confirmation",
        "frozen_by",
        "fixture",
    }
)

#: Words that mean "the held-out partition" in this project. A development-only
#: selection rule that names one of them is refused as leakage, because a
#: nomination made on test evidence is not a nomination.
_HELD_OUT_WORDS: tuple[str, ...] = ("test partition", "held-out", "held out", "test labels", "test set")


def _fail(
    category: ErrorCategory, message: str, *, field: str | None = None, missing: Sequence[str] = ()
) -> ContractError:
    return ContractError(category, message, contract=CONTRACT, field=field, missing=missing)


def _is_address(value: Any) -> bool:
    return isinstance(value, str) and bool(_CONTENT_ADDRESS.match(value))


def _is_blank(value: Any) -> bool:
    """Present but carrying nothing. ``0`` and ``False`` are values, not blanks."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_protocol(protocol: Mapping[str, Any]) -> Record:
    """Validate one study protocol and return it as JSON-ready data.

    Raises:
        ValueError: a :class:`~.contracts.ContractError` carrying
            ``MISSING_FIELD`` for an absent required field, ``EMPTY_VALUE`` for
            one present and blank, ``UNKNOWN_FIELD`` for a key no contract
            declares, ``INVALID_ENUM`` for an unregistered phase or regime,
            ``INVALID_REFERENCE`` for a malformed content address,
            ``INVALID_VALUE`` for a seed policy, nomination count or per-arm
            budget the specification forbids, ``TEST_LEAKAGE`` for a selection
            rule that reads the held-out partition or a freeze attempted after
            exposure, and ``UNSUPPORTED_CLAIM`` for an inference label the
            protocol's own arithmetic does not reach.
    """
    if not isinstance(protocol, Mapping):
        raise _fail(ErrorCategory.INVALID_TYPE, f"a protocol is a mapping, not {type(protocol).__name__}")

    # A lock is a file. A protocol carrying a value that cannot be serialised
    # could never be hashed or written, so it is refused here as a contract
    # error rather than escaping as a raw TypeError from json.
    try:
        record: Record = json.loads(json.dumps(dict(protocol), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            f"a protocol must be JSON-serialisable to be hashed and written: {exc}",
        ) from exc

    phase = record.get("phase", "confirmation")
    if phase not in PHASES:
        raise _fail(ErrorCategory.INVALID_ENUM, f"{phase!r} is not one of {list(PHASES)}", field="phase")
    record["phase"] = phase

    unknown = sorted(set(record) - set(REQUIRED_LOCK_FIELDS) - _OPTIONAL_FIELDS)
    if unknown:
        raise _fail(
            ErrorCategory.UNKNOWN_FIELD,
            f"no contract declares {unknown[0]!r}; a lock does not silently carry unrecognised keys",
            field=unknown[0],
            missing=unknown,
        )

    if phase == "pilot":
        # A pilot is not a lock. It may carry any subset of the fourteen, but it
        # may never present itself as the frozen confirmation.
        feasibility = record.get("feasibility")
        if isinstance(feasibility, Mapping) and feasibility.get("classification") == "CONFIRMATORY":
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                "a pilot protocol cannot classify itself as confirmatory",
                field="feasibility.classification",
            )
        return record

    _require_lock_fields(record)
    _check_hashes(record)
    _check_selection_rule(record)
    _check_seeds(record)
    _check_statistics(record)
    _check_margins(record)
    _check_budgets(record)
    _check_nomination_count(record)
    _check_regime(record)
    _check_ablation(record)
    _check_exposure(record)
    _check_feasibility_claim(record)
    return record


def _require_lock_fields(record: Record) -> None:
    missing = [f for f in REQUIRED_LOCK_FIELDS if f not in record]
    if missing:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            f"a confirmation protocol must carry every locked field; {missing[0]!r} is absent",
            field=missing[0],
            missing=missing,
        )
    blank = [f for f in REQUIRED_LOCK_FIELDS if _is_blank(record[f])]
    if blank:
        raise _fail(
            ErrorCategory.EMPTY_VALUE,
            f"{blank[0]!r} is present but carries nothing; a required field stays required when it is null",
            field=blank[0],
            missing=blank,
        )


def _check_hashes(record: Record) -> None:
    for field in _HASH_FIELDS:
        if not _is_address(record[field]):
            raise _fail(
                ErrorCategory.INVALID_REFERENCE,
                f"{record[field]!r} is not a canonical content address",
                field=field,
            )
    baselines = record["baseline_hashes"]
    if not isinstance(baselines, Mapping):
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            "baseline_hashes maps a baseline id to its content address",
            field="baseline_hashes",
        )
    for arm in sorted(baselines):
        if not _is_address(baselines[arm]):
            raise _fail(
                ErrorCategory.INVALID_REFERENCE,
                f"{baselines[arm]!r} is not a canonical content address",
                field=f"baseline_hashes.{arm}",
            )


def _check_selection_rule(record: Record) -> None:
    rule = str(record["selection_rule"])
    lowered = rule.lower()
    for word in _HELD_OUT_WORDS:
        if word in lowered:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"the selection rule names {word!r}; a nomination is made on development evidence only",
                field="selection_rule",
            )


def _check_seeds(record: Record) -> None:
    declared_pilot = record.get("pilot_seeds") or []
    if not isinstance(declared_pilot, Sequence) or isinstance(declared_pilot, (str, bytes)):
        raise _fail(ErrorCategory.INVALID_TYPE, "pilot_seeds is a list of integer seeds", field="pilot_seeds")
    if any(not isinstance(s, int) or isinstance(s, bool) for s in declared_pilot):
        raise _fail(ErrorCategory.INVALID_TYPE, "pilot_seeds holds integer seeds", field="pilot_seeds")
    pilot_seeds = [int(s) for s in declared_pilot]
    for field in ("campaign_seeds", "deployment_seeds"):
        seeds = record[field]
        if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
            raise _fail(ErrorCategory.INVALID_TYPE, f"{field} is a list of integer seeds", field=field)
        if any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
            raise _fail(ErrorCategory.INVALID_TYPE, f"{field} holds integer seeds", field=field)
        if len(set(seeds)) != len(seeds):
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"{field} repeats a seed; a repeated seed is a repeated run, not an independent replicate",
                field=field,
            )
    # Both replicate levels are held to the same policy. ``zak-c9r-2``'s primary
    # replicates at the deployment seed rather than at the campaign, so leaving
    # the deployment seeds outside the disjointness rule would leave the rule
    # protecting the level the study no longer uses.
    for field in ("campaign_seeds", "deployment_seeds"):
        reused = sorted(set(record[field]) & set(pilot_seeds))
        if reused:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"{field} reuses seed {reused[0]}, which was already spent on a pilot; pilot results "
                f"are excluded from confirmation and a reused seed replays one",
                field=field,
                missing=[str(s) for s in reused],
            )


def _power_blocks(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Every power assessment the statistics block declares, by key.

    ``zak-c9r-1`` had exactly one primary endpoint and therefore exactly one
    power block. ``zak-c9r-2`` redirects the primary to a count while keeping the
    superseded Track A regression branch registered, so it carries two — and a
    second block that nothing validates would be a hole big enough to park an
    unsupported claim in. Every ``power``/``power_*`` key is a power block, is
    held to :data:`REQUIRED_POWER_FIELDS`, and must reach its own ``min_power``
    before any of them may be called confirmatory.
    """
    return {k: v for k, v in stats.items() if k == "power" or str(k).startswith("power_")}


def _check_primary_power_block(stats: Mapping[str, Any]) -> None:
    """With more than one power block, the lock must say which one is the primary's."""
    blocks = _power_blocks(stats)
    if len(blocks) < 2:
        return
    named = stats.get("primary_power_block")
    if named is None:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            f"this lock declares {len(blocks)} power assessments {sorted(blocks)}; it must name which "
            f"one belongs to the primary endpoint, or a reader cannot tell which arithmetic the "
            f"inference rests on",
            field="statistics.primary_power_block",
        )
    if named not in blocks:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"statistics.primary_power_block names {named!r}, which is not one of {sorted(blocks)}",
            field="statistics.primary_power_block",
        )


def _check_statistics(record: Record) -> None:
    stats = record["statistics"]
    if not isinstance(stats, Mapping):
        raise _fail(
            ErrorCategory.INVALID_TYPE, "statistics is a mapping of frozen decisions", field="statistics"
        )
    for field in REQUIRED_STATISTICS_FIELDS:
        if field not in stats:
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                f"the statistical method is not frozen while {field!r} is undeclared",
                field=f"statistics.{field}",
            )
        if _is_blank(stats[field]):
            raise _fail(
                ErrorCategory.EMPTY_VALUE,
                f"statistics.{field} is present but carries nothing",
                field=f"statistics.{field}",
            )
    for name, power in _power_blocks(stats).items():
        if not isinstance(power, Mapping):
            raise _fail(
                ErrorCategory.INVALID_TYPE, f"statistics.{name} is a mapping", field=f"statistics.{name}"
            )
        for field in REQUIRED_POWER_FIELDS:
            if field not in power:
                raise _fail(
                    ErrorCategory.MISSING_FIELD,
                    f"the power assessment cannot be checked while {field!r} is undeclared",
                    field=f"statistics.{name}.{field}",
                )
    _check_primary_power_block(stats)
    draws = stats["n_bootstrap_draws"]
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1000:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            "a percentile interval needs at least 1000 draws to be worth reporting",
            field="statistics.n_bootstrap_draws",
        )
    level = stats["interval_level"]
    if not isinstance(level, (int, float)) or not 0.0 < float(level) < 1.0:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            "interval_level is a probability strictly between 0 and 1",
            field="statistics.interval_level",
        )


def _check_margins(record: Record) -> None:
    margins = record["margins"]
    if not isinstance(margins, Mapping):
        raise _fail(ErrorCategory.INVALID_TYPE, "margins is a mapping", field="margins")
    for block in ("benchmark_accepted", "practical_gain"):
        if block not in margins:
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                f"Appendix A §5 requires the two decisions to be separate; {block!r} is undeclared",
                field=f"margins.{block}",
            )


def _check_budgets(record: Record) -> None:
    budgets = record["budgets"]
    if not isinstance(budgets, Mapping):
        raise _fail(ErrorCategory.INVALID_TYPE, "budgets is a mapping", field="budgets")
    per_arm = budgets.get("per_arm")
    if not isinstance(per_arm, Mapping) or not per_arm:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            "budgets.per_arm declares what each arm may spend; confirmation cannot enlarge "
            "one arm's allocation",
            field="budgets.per_arm",
        )
    # ``per_arm`` and every ``*_per_arm`` companion. ``zak-c9r-2`` registers a
    # second family of arms — nine pretrained generators that consume no
    # reasoning budget — and a family whose symmetry nothing checks is a family
    # somebody can enlarge one member of.
    families = [k for k in sorted(budgets) if k == "per_arm" or str(k).endswith("_per_arm")]
    for family in families:
        allocations_by_arm = budgets[family]
        if not isinstance(allocations_by_arm, Mapping) or not allocations_by_arm:
            raise _fail(
                ErrorCategory.INVALID_TYPE,
                f"budgets.{family} maps an arm id to its allocation",
                field=f"budgets.{family}",
            )
        allocations = {
            arm: json.dumps(allocations_by_arm[arm], sort_keys=True) for arm in sorted(allocations_by_arm)
        }
        distinct = sorted(set(allocations.values()))
        if len(distinct) > 1:
            differing = sorted(a for a in allocations if allocations[a] != distinct[0])
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"per-arm allocations in budgets.{family} are not symmetric; {differing[0]!r} differs. "
                f"Appendix A §4: confirmation cannot enlarge only one arm's allocation",
                field=f"budgets.{family}",
                missing=differing,
            )


def _check_nomination_count(record: Record) -> None:
    value = record["max_final_nominations"]
    if value != 1 or isinstance(value, bool):
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"max_final_nominations is 1 per campaign per registered track, not {value!r}",
            field="max_final_nominations",
        )


def _check_regime(record: Record) -> None:
    regime = record["resource_regime"]
    if regime not in RESOURCE_REGIMES:
        raise _fail(
            ErrorCategory.INVALID_ENUM,
            f"{regime!r} is not one of {list(RESOURCE_REGIMES)}",
            field="resource_regime",
        )


def _check_ablation(record: Record) -> None:
    ablation = record.get("ablation")
    if ablation is None:
        return
    if not isinstance(ablation, Mapping):
        raise _fail(ErrorCategory.INVALID_TYPE, "ablation is a mapping", field="ablation")
    arms = ablation.get("arms")
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        raise _fail(ErrorCategory.INVALID_TYPE, "ablation.arms is a list of arms", field="ablation.arms")
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            raise _fail(
                ErrorCategory.INVALID_TYPE, "each ablation arm is a mapping", field=f"ablation.arms[{index}]"
            )
        for field in ("arm_id", "inference_label"):
            if field not in arm:
                raise _fail(
                    ErrorCategory.MISSING_FIELD,
                    f"an ablation arm not labelled now is post-hoc later; {field!r} is undeclared",
                    field=f"ablation.arms[{index}].{field}",
                )
        if arm["inference_label"] not in INFERENCE_LABELS:
            raise _fail(
                ErrorCategory.INVALID_ENUM,
                f"{arm['inference_label']!r} is not one of {list(INFERENCE_LABELS)}",
                field=f"ablation.arms[{index}].inference_label",
            )
    if "exclusion_set_derivation" not in ablation:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            "the exclusion set must be derived by a frozen rule before generation is evaluated",
            field="ablation.exclusion_set_derivation",
        )


def _check_exposure(record: Record) -> None:
    exposure = record.get("evaluation_exposure")
    if exposure is None:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            "a confirmation protocol declares its evaluation exposure; no claim of blindness "
            "is made by silence",
            field="evaluation_exposure",
        )
    if not isinstance(exposure, Mapping):
        raise _fail(
            ErrorCategory.INVALID_TYPE, "evaluation_exposure is a mapping", field="evaluation_exposure"
        )
    for field in ("test_access_count", "revealed_at", "boundary", "sealed", "public_exposure"):
        if field not in exposure:
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                f"evaluation_exposure.{field} is undeclared; unknown exposure is stated, "
                f"never assumed absent",
                field=f"evaluation_exposure.{field}",
            )
    count = exposure["test_access_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            "test_access_count is a non-negative integer",
            field="evaluation_exposure.test_access_count",
        )
    exposed = count > 0 or exposure["revealed_at"] is not None
    supersedes = record.get("supersedes")
    presented_as_original = record.get("presented_as_original_confirmation", True)
    if exposed and not supersedes:
        raise _fail(
            ErrorCategory.TEST_LEAKAGE,
            "a test result has already been revealed under this protocol; the lock is closed. A new "
            "version must declare what it supersedes and carry a new evaluation-exposure declaration",
            field="evaluation_exposure",
        )
    if supersedes is not None:
        if not _is_address(supersedes):
            raise _fail(
                ErrorCategory.INVALID_REFERENCE,
                f"{supersedes!r} is not a canonical content address of the superseded lock",
                field="supersedes",
            )
        if presented_as_original:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                "a superseding protocol version cannot be presented as the original untouched confirmation",
                field="presented_as_original_confirmation",
            )


def _check_feasibility_claim(record: Record) -> None:
    """The gate this step exists for: a label cannot outrun its own arithmetic."""
    feasibility = record.get("feasibility")
    if feasibility is None:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            "a confirmation protocol states whether its own design can support confirmatory inference",
            field="feasibility",
        )
    if not isinstance(feasibility, Mapping):
        raise _fail(ErrorCategory.INVALID_TYPE, "feasibility is a mapping", field="feasibility")
    classification = feasibility.get("classification")
    if classification not in FEASIBILITY_CLASSIFICATIONS:
        raise _fail(
            ErrorCategory.INVALID_ENUM,
            f"{classification!r} is not one of {list(FEASIBILITY_CLASSIFICATIONS)}",
            field="feasibility.classification",
        )
    if classification != "CONFIRMATORY":
        if _is_blank(feasibility.get("reasons")):
            raise _fail(
                ErrorCategory.EMPTY_VALUE,
                "a study downgraded below confirmatory must say why, by name",
                field="feasibility.reasons",
            )
        return
    # Every declared power block, not only ``power``: a study is as confirmatory
    # as its weakest registered assessment, and a second block nobody checks is
    # where an unsupported claim would hide.
    for name, power in _power_blocks(record["statistics"]).items():
        worst = power["power_at_locked_n_worst_case"]
        minimum = power["min_power"]
        if worst is None:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"confirmatory inference is claimed while statistics.{name}'s worst-case power is "
                f"unknown; unknown is not adequate",
                field="feasibility.classification",
            )
        if float(worst) < float(minimum):
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"confirmatory inference is claimed at statistics.{name}'s worst-case power "
                f"{float(worst):.3f} against a locked minimum of {float(minimum):.3f}. An underpowered "
                f"study is labelled, not relaxed",
                field="feasibility.classification",
            )
    if feasibility.get("budget_authorised") is False:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            "confirmatory inference is claimed for a design whose budget is not authorised; a study "
            "that cannot be run is not a confirmed one",
            field="feasibility.classification",
        )


# ---------------------------------------------------------------------------
# canonical serialisation and the content address
# ---------------------------------------------------------------------------
def canonical_json(protocol: Mapping[str, Any]) -> str:
    """The one serialisation a protocol is hashed over.

    Sorted keys, no insignificant whitespace, UTF-8, no NaN. Two records that
    differ only in key order or formatting produce identical bytes, so a lock's
    content address tracks its science and nothing else.
    """
    return json.dumps(
        dict(protocol), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` over :func:`canonical_json`."""
    import hashlib

    return "sha256:" + hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def freeze_protocol(protocol: Mapping[str, Any], path: Path | str, **extra: Any) -> Path:
    """Validate, canonicalise, hash and write ``protocol.lock.json``.

    Validation happens first and the file is written only afterwards, so a
    protocol that does not validate leaves no lock behind for someone to find
    and believe. The lock carries the validated protocol, its content address
    and whatever provenance the caller supplies.

    Raises:
        ValueError: whatever :func:`validate_protocol` raises.
    """
    validated = validate_protocol(protocol)
    lock = {
        "contract_version": "1.0.0",
        "record_kind": "study_protocol_lock",
        "protocol": validated,
        "protocol_hash": protocol_hash(validated),
        "canonical_serialisation": "json, sorted keys, separators (',', ':'), utf-8, no NaN",
        **extra,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return out


def load_lock(path: Path | str) -> Record:
    """Read a lock and re-derive its hash from the protocol it carries.

    Raises:
        ValueError: a ``HASH_MISMATCH`` when the stored address disagrees with
            the protocol beside it — a lock whose hash was edited is not a lock.
    """
    lock: Record = json.loads(Path(path).read_text())
    recomputed = protocol_hash(lock["protocol"])
    if recomputed != lock.get("protocol_hash"):
        raise _fail(
            ErrorCategory.HASH_MISMATCH,
            f"the lock at {path} stores {lock.get('protocol_hash')!r} but its protocol "
            f"hashes to {recomputed!r}",
            field="protocol_hash",
        )
    return lock


# ---------------------------------------------------------------------------
# feasibility and power
# ---------------------------------------------------------------------------
@cache
def _chi2_quantile(p: float, df: int) -> float:
    """Chi-square quantile by bisection on the regularised lower incomplete gamma.

    Written out rather than imported so the variance interval below does not
    make SciPy a hard dependency of the lock.
    """
    lo, hi = 1e-12, 1e4
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if _gammainc_lower(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return hi


def _gammainc_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x), series/continued fraction."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def sd_interval(sd: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """The chi-square confidence interval for a standard deviation estimated on ``n`` points.

    This is the number that decides whether a power estimate means anything. An
    SD from four or five observations is itself an estimate with a very wide
    interval, and reporting the point estimate alone is exactly the
    overstatement Appendix A §5 warns about at small sample sizes.
    """
    df = n - 1
    if df < 1 or sd <= 0.0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    upper_q = _chi2_quantile(1.0 - alpha / 2.0, df)
    lower_q = _chi2_quantile(alpha / 2.0, df)
    return (sd * math.sqrt(df / upper_q), sd * math.sqrt(df / lower_q))


@cache
def _student_t_quantile(p: float, df: int) -> float:
    lo, hi = -100.0, 100.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return hi


def _student_t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return (
        1.0
        - math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + b * math.log(1.0 - x) + a * math.log(x)
        )
        * _beta_cf(b, a, 1.0 - x)
        / b
    )


def _beta_cf(a: float, b: float, x: float) -> float:
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 / max(abs(1.0 + num * d), 1e-300) * (1.0 if 1.0 + num * d >= 0 else -1.0)
        c = 1.0 + num / (c if abs(c) > 1e-300 else 1e-300)
        h *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 / max(abs(1.0 + num * d), 1e-300) * (1.0 if 1.0 + num * d >= 0 else -1.0)
        c = 1.0 + num / (c if abs(c) > 1e-300 else 1e-300)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _chi2_pdf(v: float, df: int) -> float:
    if v <= 0.0:
        return 0.0
    k = df / 2.0
    return math.exp((k - 1.0) * math.log(v) - v / 2.0 - k * math.log(2.0) - math.lgamma(k))


def paired_power(delta: float, sd: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided paired test for ``delta`` at ``sd`` with ``n`` campaigns.

    The replication unit is the campaign, never the molecule: thousands of
    molecules are not independent replications of the evolution controller
    (Appendix A section 5). This is the exact noncentral-t tail, integrated over
    the sampling distribution of the sample standard deviation by Simpson's
    rule, so the feasibility statement carries no numerical dependency of its
    own. It agrees with ``scipy.stats.nct`` to better than 1e-6 over the range
    used here, and ``tests/research/test_protocol_lock.py`` asserts that against
    SciPy when SciPy is installed.
    """
    if sd <= 0.0 or n < 2 or delta <= 0.0:
        return float("nan")
    df = n - 1
    lam = delta / sd * math.sqrt(n)
    crit = _student_t_quantile(1.0 - alpha / 2.0, df)
    hi = df + 14.0 * math.sqrt(2.0 * df) + 60.0
    nodes = 2000  # even, for Simpson
    h = hi / nodes
    total = 0.0
    for i in range(nodes + 1):
        v = i * h
        u = math.sqrt(v / df)
        value = _chi2_pdf(v, df) * (_normal_sf(crit * u - lam) + _normal_cdf(-crit * u - lam))
        weight = 1.0 if i in (0, nodes) else (4.0 if i % 2 else 2.0)
        total += weight * value
    return min(1.0, max(0.0, total * h / 3.0))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def n_for_power(
    delta: float, sd: float, target: float = 0.8, alpha: float = 0.05, cap: int = 400
) -> int | None:
    for n in range(2, cap + 1):
        if paired_power(delta, sd, n, alpha) >= target:
            return n
    return None


def minimum_detectable_effect(sd: float, n: int, target: float = 0.8, alpha: float = 0.05) -> float:
    lo, hi = 0.0, sd * 12.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if paired_power(mid, sd, n, alpha) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def assess_power(
    *,
    observations: Sequence[float],
    comparator_value: float,
    locked_n: int,
    min_power: float = 0.8,
    alpha: float = 0.05,
    orientation: str = "LOWER_IS_BETTER",
    evidence_class: str = "measured",
    note: str = "",
) -> Record:
    """The frozen power block, computed from measured pilot observations.

    ``observations`` are per-campaign (or per-run) values of the primary metric
    produced by the process under test; ``comparator_value`` is the deterministic
    comparator's value on the same pool, so the paired difference's variance is
    the process's own. Every returned number carries the inputs that produced it,
    and the SD interval is reported beside the point estimate because a variance
    estimated on a handful of runs is itself very uncertain.
    """
    values = [float(v) for v in observations]
    if len(values) < 2:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            "a variance estimate needs at least two observations",
            field="statistics.power",
        )
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    delta = (comparator_value - mean) if orientation == "LOWER_IS_BETTER" else (mean - comparator_value)
    lo, hi = sd_interval(sd, len(values))
    point = paired_power(delta, sd, locked_n, alpha) if delta > 0 else float("nan")
    worst = paired_power(delta, hi, locked_n, alpha) if delta > 0 else float("nan")
    best = paired_power(delta, lo, locked_n, alpha) if delta > 0 else float("nan")
    needed = n_for_power(delta, hi, min_power, alpha) if delta > 0 else None
    return {
        "min_power": min_power,
        "alpha": alpha,
        "orientation": orientation,
        "locked_n_campaigns": locked_n,
        "observations": values,
        "n_observations": len(values),
        "comparator_value": comparator_value,
        "process_mean": mean,
        "effect_under_test": delta if delta > 0 else None,
        "effect_note": (
            "the process mean against the deterministic comparator, not the best single run; "
            "selecting the best of several runs is not the expected per-campaign effect"
        ),
        "process_sd_point": sd,
        "process_sd_interval": [lo, hi],
        "process_sd_interval_kind": (
            f"95% chi-square interval on {len(values)} observations, df {len(values) - 1}"
        ),
        "power_at_locked_n_point": None if point != point else point,
        "power_at_locked_n_best_case": None if best != best else best,
        "power_at_locked_n_worst_case": None if worst != worst else worst,
        "n_for_min_power_worst_case": needed,
        "minimum_detectable_effect_at_locked_n_point": minimum_detectable_effect(
            sd, locked_n, min_power, alpha
        ),
        "minimum_detectable_effect_at_locked_n_worst_case": minimum_detectable_effect(
            hi, locked_n, min_power, alpha
        ),
        "instability_note": (
            "the standard deviation is itself estimated from a handful of runs. The power at the "
            "locked n is therefore an interval, not a number, and the worst case is what the "
            "classification is decided on"
        ),
        "evidence_class": evidence_class,
        "note": note,
    }


# ---------------------------------------------------------------------------
# count endpoints — the arithmetic a qualifying-scaffold objective needs
# ---------------------------------------------------------------------------
# The Track A machinery above assumes a continuous per-campaign value with a
# variance. ``zak-c9r-2``'s primary is a *count* of qualifying scaffolds per
# generator arm per generation seed, and none of the paired-t arithmetic
# transfers to it. Two things decide a count design's power and neither is a
# standard deviation:
#
# * **What the count is expected to be.** A count whose expected value is at or
#   near zero carries almost no information per cell, however the arms differ.
# * **Whether the test can reach alpha at all.** An exact test over three
#   replicates has a discrete support: the paired sign test's smallest attainable
#   two-sided p is 2 * 2**-n, which at n = 3 is 0.25. No effect size of any size
#   reaches alpha = 0.05 there, so power is exactly 0 — by the arithmetic of the
#   test, not by an assumption about the effect.
def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam < 0.0 or k < 0:
        return float("nan")
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam), summed from the lower tail."""
    if k <= 0:
        return 1.0
    return max(0.0, 1.0 - sum(poisson_pmf(i, lam) for i in range(k)))


def sign_test_min_two_sided_p(n: int) -> float:
    """The smallest two-sided p an exact paired sign test on ``n`` pairs can produce.

    Every pair contributes one bit, so the most extreme outcome — all ``n``
    differences in the same direction — has two-sided probability ``2 * 2**-n``.
    At n = 3 that is 0.25, which is why three seeds cannot support a confirmatory
    claim at alpha = 0.05 no matter how large the effect is.
    """
    if n < 1:
        return 1.0
    return min(1.0, 2.0 * 0.5**n)


def permutation_min_two_sided_p(n_a: int, n_b: int) -> float:
    """The smallest two-sided p an exact two-sample permutation test can produce."""
    if n_a < 1 or n_b < 1:
        return 1.0
    return min(1.0, 2.0 / math.comb(n_a + n_b, n_a))


def seeds_for_attainable_alpha(alpha: float = 0.05, *, paired: bool = True, cap: int = 200) -> int | None:
    """Replicates needed before ``alpha`` is even *reachable* by an exact test."""
    for n in range(1, cap + 1):
        p = sign_test_min_two_sided_p(n) if paired else permutation_min_two_sided_p(n, n)
        if p <= alpha:
            return n
    return None


def smallest_count_reaching_alpha(lam: float, alpha: float, cap: int = 2000) -> int | None:
    """Smallest observed count whose upper-tail Poisson p-value reaches ``alpha``."""
    for k in range(cap + 1):
        if poisson_sf(k, lam) <= alpha:
            return k
    return None


def clopper_pearson_upper(successes: int, trials: int, level: float = 0.95) -> float:
    """Exact binomial upper bound — the companion of :func:`clopper_pearson_lower`."""
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    alpha = 1.0 - level
    lo, hi = float(successes) / trials, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _betainc(float(successes + 1), float(trials - successes), mid) < 1.0 - alpha / 2.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _probability_strictly_greater(lam_high: float, lam_low: float, cap: int = 200) -> float:
    """P(X > Y) for independent X ~ Poisson(lam_high), Y ~ Poisson(lam_low)."""
    total = 0.0
    for x in range(1, cap):
        px = poisson_pmf(x, lam_high)
        if px < 1e-18 and x > lam_high + 40:
            break
        total += px * sum(poisson_pmf(y, lam_low) for y in range(x))
    return min(1.0, max(0.0, total))


def assess_count_power(
    *,
    qualifying_successes: int,
    molecules_observed: int,
    gate_widening_factor: float,
    molecules_per_cell: int,
    n_seeds: int,
    n_arms: int,
    absolute_threshold: int,
    ceiling_rate_per_molecule: float,
    min_power: float = 0.8,
    alpha: float = 0.05,
    level: float = 0.95,
    evidence_class: str = "measured",
    note: str = "",
) -> Record:
    """The frozen power block for a *count* primary, computed from measured yields.

    Every number below follows from the six measured inputs, so the whole
    assessment is recomputable from the lock alone — the same property the Track
    A block has, expressed for an endpoint that has no variance to estimate.

    The verdict this produces is deliberately in two parts, because they answer
    different questions and only one of them is admissible as process-level
    confidence:

    * **At the locked replication unit** — the generation seed — an exact test
      over ``n_seeds`` replicates has a discrete support. If ``alpha`` is not
      inside that support, the power at the locked n is **0.0 at every effect
      size**, and the returned ``power_at_locked_n_*`` fields say so.
    * **Under a pooled reading** that treats molecules as exchangeable within an
      arm, the design has more resolution — but that reading is per-molecule
      pseudo-replication, which this lock prohibits as process confidence. It is
      returned as ``pooled_pseudo_replicated_reading`` and labelled, never as
      the headline.
    """
    if molecules_observed < 1 or qualifying_successes < 0:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            "a qualification rate needs a positive denominator",
            field="statistics.power",
        )
    observed_rate = qualifying_successes / molecules_observed
    baseline_rate = observed_rate * gate_widening_factor
    lam_cell = baseline_rate * molecules_per_cell
    lam_arm = lam_cell * n_seeds
    n_cells = n_arms * n_seeds

    rate_lo = clopper_pearson_lower(qualifying_successes, molecules_observed, level) * gate_widening_factor
    rate_hi = clopper_pearson_upper(qualifying_successes, molecules_observed, level) * gate_widening_factor
    lam_arm_lo = rate_lo * molecules_per_cell * n_seeds
    lam_arm_hi = rate_hi * molecules_per_cell * n_seeds

    clears = poisson_sf(absolute_threshold, lam_arm)
    any_clears = 1.0 - (1.0 - clears) ** n_arms

    paired_min_p = sign_test_min_two_sided_p(n_seeds)
    unpaired_min_p = permutation_min_two_sided_p(n_seeds, n_seeds)
    attainable = paired_min_p <= alpha

    ceiling_lam_cell = ceiling_rate_per_molecule * molecules_per_cell
    q = _probability_strictly_greater(ceiling_lam_cell, lam_cell)
    seeds_at_ceiling: int | None = None
    for n in range(1, 201):
        if sign_test_min_two_sided_p(n) > alpha:
            continue
        if q**n + (1.0 - q) ** n >= min_power:
            seeds_at_ceiling = n
            break
    power_now = (q**n_seeds + (1.0 - q) ** n_seeds) if attainable else 0.0

    uncorrected = smallest_count_reaching_alpha(lam_arm, alpha)
    corrected = smallest_count_reaching_alpha(lam_arm, alpha / n_arms)
    per_arm_budget = molecules_per_cell * n_seeds
    return {
        "endpoint_kind": "COUNT",
        "orientation": "HIGHER_IS_BETTER",
        "replication_unit": "generation seed",
        "min_power": min_power,
        "alpha": alpha,
        "locked_n_campaigns": n_seeds,
        "n_seeds_per_arm": n_seeds,
        "n_arms": n_arms,
        "n_cells": n_cells,
        "n_unique_molecules_per_cell": molecules_per_cell,
        "n_unique_molecules_per_arm": per_arm_budget,
        "evidence": {
            "qualifying_scaffolds_observed": qualifying_successes,
            "molecules_observed": molecules_observed,
            "observed_rate_per_molecule_under_the_narrower_gate_set": observed_rate,
            "gate_widening_factor": gate_widening_factor,
            "baseline_rate_per_molecule": baseline_rate,
        },
        "expected_count_per_cell": lam_cell,
        "expected_count_per_arm": lam_arm,
        "probability_a_cell_is_exactly_zero": math.exp(-lam_cell),
        "expected_number_of_zero_cells": n_cells * math.exp(-lam_cell),
        "baseline_rate_interval_per_molecule": [rate_lo, rate_hi],
        "expected_count_per_arm_interval": [lam_arm_lo, lam_arm_hi],
        "process_sd_point": math.sqrt(lam_arm),
        "process_sd_interval": [math.sqrt(lam_arm_lo), math.sqrt(lam_arm_hi)],
        "process_sd_interval_kind": (
            "Poisson standard deviation of the per-arm count, evaluated at the exact "
            f"{level:.0%} Clopper-Pearson interval of the observed qualification rate. A count has no "
            "variance of its own to estimate; its spread is fixed by its mean"
        ),
        "effect_under_test": absolute_threshold - lam_arm,
        "effect_note": (
            "the predeclared absolute margin minus what a baseline arm is expected to produce at the "
            "locked budget. It is the gap the design is asked to resolve, not a difference anyone measured"
        ),
        "absolute_threshold_scaffolds": absolute_threshold,
        "probability_one_baseline_arm_clears_the_absolute_threshold": clears,
        "probability_any_of_the_arms_clears_the_absolute_threshold": any_clears,
        "probability_one_baseline_arm_clears_it_at_the_rate_interval": [
            poisson_sf(absolute_threshold, lam_arm_lo),
            poisson_sf(absolute_threshold, lam_arm_hi),
        ],
        "attainability_at_the_locked_replication_unit": {
            "paired_sign_test_min_two_sided_p": paired_min_p,
            "two_sample_permutation_min_two_sided_p": unpaired_min_p,
            "alpha_is_attainable": attainable,
            "n_seeds_for_alpha_paired_sign_test": seeds_for_attainable_alpha(alpha, paired=True),
            "n_seeds_for_alpha_two_sample_permutation": seeds_for_attainable_alpha(alpha, paired=False),
            "what_this_means": (
                "an exact test over the locked replicates has a discrete support. Where alpha falls "
                "outside it, no effect size of any magnitude reaches significance and the power at the "
                "locked n is 0.0 by arithmetic rather than by assumption"
            ),
        },
        "power_at_locked_n_point": power_now,
        "power_at_locked_n_best_case": power_now,
        "power_at_locked_n_worst_case": power_now,
        "power_note": (
            "all three are the same number because the binding constraint is not the effect size or "
            "the rate interval: it is whether alpha is reachable at the locked number of replicates"
        ),
        "n_for_min_power_worst_case": None,
        "n_for_min_power_note": (
            "null, and not a number anybody could quote: at the lower end of the measured rate interval "
            "the arms' expected counts differ by a fraction of one scaffold over the whole locked budget, "
            "so no replicate count separates them for a reason that is about the rate rather than the test"
        ),
        "n_seeds_for_min_power_at_the_known_active_ceiling": seeds_at_ceiling,
        "known_active_ceiling": {
            "rate_per_molecule": ceiling_rate_per_molecule,
            "expected_count_per_cell": ceiling_lam_cell,
            "probability_such_an_arm_beats_a_baseline_cell": q,
            "what_it_is": (
                "the rate at which REAL known-active development controls clear this same gate set. An "
                "arm reaching it would be generating molecules as qualifiable as human-measured actives, "
                "which is a ceiling rather than an expectation"
            ),
        },
        "minimum_detectable_effect_at_locked_n_point": None,
        "minimum_detectable_effect_at_locked_n_worst_case": None,
        "minimum_detectable_effect_note": (
            "undefined at the locked replication unit, because no effect is detectable there at alpha. "
            "The pooled reading below carries the count an arm would have to reach, and that reading is "
            "pseudo-replicated"
        ),
        "pooled_pseudo_replicated_reading": {
            "what_it_is": (
                "the three seeds pooled into one per-arm count, which treats molecules as exchangeable "
                "within an arm. This lock PROHIBITS that as process-level confidence "
                "(statistics.pseudo_replication_prohibition); it is reported because it is the most "
                "favourable honest reading of the design and it still does not rescue it"
            ),
            "smallest_arm_count_reaching_alpha_uncorrected": uncorrected,
            "smallest_arm_count_reaching_alpha_under_holm": corrected,
            "holm_family_size": n_arms,
            "minimum_detectable_rate_per_molecule_under_holm": (
                None if corrected is None else corrected / per_arm_budget
            ),
            "minimum_detectable_rate_ratio_under_holm": (
                None
                if corrected is None or baseline_rate <= 0
                else corrected / per_arm_budget / baseline_rate
            ),
            "minimum_detectable_rate_as_a_fraction_of_the_known_active_rate": (
                None
                if corrected is None or ceiling_rate_per_molecule <= 0
                else corrected / per_arm_budget / ceiling_rate_per_molecule
            ),
        },
        "instability_note": (
            "the baseline rate rests on the qualifying scaffolds actually observed, and its exact "
            "binomial interval spans more than two orders of magnitude. Every count above moves across "
            "that interval by more than the effect the design is asked to resolve"
        ),
        "evidence_class": evidence_class,
        "note": note,
    }


def clopper_pearson_lower(successes: int, trials: int, level: float = 0.95) -> float:
    """Exact binomial lower bound — what a rate of one-out-of-one is actually worth.

    A nomination rate observed as 1/1 is not 100%: its exact two-sided 95% lower
    bound is 0.025. Planning a five-campaign confirmation against the point
    estimate rather than against this bound is how a study runs out of nominees.
    """
    if trials <= 0 or successes <= 0:
        return 0.0
    alpha = 1.0 - level
    if successes >= trials:
        return (alpha / 2.0) ** (1.0 / trials)
    lo, hi = 0.0, float(successes) / trials
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # I_p(k, n-k+1) is the probability of at most k-1 successes' complement and
        # increases with p; solve it to alpha/2.
        if _betainc(float(successes), float(trials - successes + 1), mid) < alpha / 2.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# pilot aggregation
# ---------------------------------------------------------------------------
def summarise_campaign(record: Mapping[str, Any]) -> Record:
    """One row of the pilot table, read from a campaign record that already ran."""
    pool = list(record.get("candidate_pool") or [])
    scored = [c for c in pool if c.get("development_metric_value") is not None]
    consumed = (record.get("consumed") or {}).get("measured") or {}
    unknown = (record.get("consumed") or {}).get("unknown_telemetry_attempts") or {}
    nomination = record.get("nomination")
    return {
        "campaign_id": record.get("campaign_id"),
        "state": record.get("state"),
        "terminal_reason": record.get("terminal_reason"),
        "exit_code": record.get("exit_code"),
        "cycles_completed": record.get("cycles_completed"),
        "n_candidates": len(pool),
        "n_scored": len(scored),
        "outcomes": [c.get("outcome") for c in pool],
        "failure_categories": [c.get("failure_category") for c in pool if c.get("failure_category")],
        "development_values": [c.get("development_metric_value") for c in scored],
        "evaluated_partitions": sorted({str(c.get("evaluated_partition")) for c in scored}),
        "nominated": nomination is not None,
        "nominated_artifact_id": None if nomination is None else nomination.get("artifact_id"),
        "wall_seconds": consumed.get("wall_s"),
        "llm_output_tokens": consumed.get("llm_output_tokens"),
        "unknown_telemetry": dict(unknown),
        "cost_evidence_class": "measured" if consumed else "unavailable",
    }


def pilot_assessment(campaigns: Iterable[Mapping[str, Any]], **context: Any) -> Record:
    """Throughput, variance, failures and nomination rate over the pilot campaigns.

    Every rate here is reported with its denominator and an exact binomial lower
    bound, because "one campaign in one produced a nominee" and "a nomination
    rate of 100%" are different statements and only the first is true.
    """
    rows = [summarise_campaign(c) for c in campaigns]
    n = len(rows)
    nominated = sum(1 for r in rows if r["nominated"])
    cycles = sum(int(r["cycles_completed"] or 0) for r in rows)
    scored = sum(int(r["n_scored"]) for r in rows)
    tokens = [r["llm_output_tokens"] for r in rows if r["llm_output_tokens"] is not None]
    walls = [r["wall_seconds"] for r in rows if r["wall_seconds"] is not None]
    values = [v for r in rows for v in r["development_values"]]
    return {
        "contract_version": "1.0.0",
        "record_kind": "pilot_feasibility_assessment",
        "n_campaigns": n,
        "campaigns": rows,
        "throughput": {
            "cycles_completed": cycles,
            "cycles_producing_a_development_metric": scored,
            "scored_cycle_rate": None if cycles == 0 else scored / cycles,
            "scored_cycle_rate_lower_bound_95": clopper_pearson_lower(scored, cycles),
            "campaigns_producing_a_nominee": nominated,
            "nomination_rate": None if n == 0 else nominated / n,
            "nomination_rate_lower_bound_95": clopper_pearson_lower(nominated, n),
            "rate_note": (
                "rates are reported with their denominators. A rate estimated on a handful of "
                "campaigns has an exact lower bound far below its point estimate, and that bound "
                "is what a budget has to be planned against"
            ),
        },
        "cost": {
            "llm_output_tokens_measured": sum(tokens) if tokens else None,
            "llm_output_tokens_campaigns_measured": len(tokens),
            "llm_output_tokens_campaigns_unknown": n - len(tokens),
            "wall_seconds_measured": round(sum(walls), 3) if walls else None,
            "max_campaign_llm_output_tokens": max(tokens) if tokens else None,
            "telemetry_note": "an unmeasured quantity is null beside its count, never a zero",
        },
        "development_values": values,
        "failures": sorted(
            {o for r in rows for o in r["outcomes"] if o not in {"DEVELOPMENT_IMPROVEMENT", None}}
        ),
        **context,
    }


# ---------------------------------------------------------------------------
# the synthetic engineering fixture
# ---------------------------------------------------------------------------
def example_protocol() -> Record:
    """A **synthetic engineering fixture** that validates. Not a research record.

    Negative tests start here and change exactly one field, so a refusal is
    attributable to that field and not to an unrelated missing key. The target
    hash is deliberately not the ZAK manifest's, and ``fixture`` is true, so this
    record can never be mistaken for the frozen study lock.
    """
    digest = lambda ch: "sha256:" + ch * 64  # noqa: E731 - a fixture address, not a real one
    return {
        "contract_version": "1.0.0",
        "record_kind": "study_protocol",
        "fixture": True,
        "phase": "confirmation",
        "protocol_version": "fixture-1",
        "target_id": "FIXTURE_TARGET",
        "primary_track": "A",
        "primary_endpoint": "fixture_endpoint",
        "target_hash": digest("1"),
        "data_hash": digest("2"),
        "split_hash": digest("3"),
        "evaluator_hash": digest("4"),
        "baseline_hashes": {"FIX-B0-parent": digest("5"), "FIX-B1-simple": digest("6")},
        "primary_metric": "fixture_mae",
        "selection_rule": "the best development-partition value of the primary metric over the whole pool",
        "nomination_rule": "one nomination per campaign per registered track, on development evidence only",
        "margins": {
            "benchmark_accepted": {
                "rule": "the paired interval excludes no improvement and constraints hold",
                "coverage_non_regression": 0.02,
            },
            "practical_gain": {"regression_relative_mae_reduction": 0.05},
        },
        "budgets": {
            "per_arm": {
                "ARM_A": {"llm_output_tokens": 75000.0, "gpu_seconds": 17280.0, "wall_seconds": 86400.0},
                "ARM_B": {"llm_output_tokens": 75000.0, "gpu_seconds": 17280.0, "wall_seconds": 86400.0},
            },
            "symmetry_note": "identical caps across arms; confirmation cannot enlarge one arm",
        },
        "pilot_seeds": [41, 42, 43],
        "campaign_seeds": [101, 102, 103, 104, 105],
        "deployment_seeds": [11, 12, 13],
        "statistics": {
            "replicate_hierarchy": ["campaign", "deployment_seed", "molecule"],
            "family_resampling": "cluster bootstrap over structural families at the molecule level",
            "n_bootstrap_draws": 10000,
            "bootstrap_seed": 20260912,
            "interval_level": 0.95,
            "failed_campaign_treatment": "INTENTION_TO_RUN: contributes a paired difference of 0.0",
            "multiplicity_correction": "one primary endpoint; everything else exploratory",
            "power": {
                "min_power": 0.8,
                "alpha": 0.05,
                "locked_n_campaigns": 5,
                "effect_under_test": 0.03,
                "process_sd_point": 0.02,
                "process_sd_interval": [0.012, 0.056],
                "power_at_locked_n_point": 0.74,
                "power_at_locked_n_worst_case": 0.16,
                "n_for_min_power_worst_case": 23,
                "evidence_class": "synthetic_fixture",
            },
        },
        "max_final_nominations": 1,
        "resource_regime": "CONTROLLED",
        "feasibility": {
            "classification": "EXPLORATORY",
            "reasons": ["the fixture's worst-case power is below its own locked minimum"],
            "budget_authorised": False,
        },
        "ablation": {
            "arms": [
                {"arm_id": "FIX-ABL-1", "inference_label": "CONFIRMATORY", "description": "fixture arm"},
                {
                    "arm_id": "FIX-ABL-2",
                    "inference_label": "POST_HOC_EXPLANATORY",
                    "description": "fixture arm",
                },
            ],
            "budget_per_arm": {"llm_output_tokens": 75000.0},
            "exclusion_set_derivation": "union of admitted resources across arms, hashed before evaluation",
        },
        "evaluation_exposure": {
            "test_access_count": 0,
            "revealed_at": None,
            "boundary": "NONE_SAME_UID",
            "sealed": False,
            "public_exposure": "UNKNOWN",
        },
    }


__all__ = [
    "CONTRACT",
    "FEASIBILITY_CLASSIFICATIONS",
    "INFERENCE_LABELS",
    "PHASES",
    "REQUIRED_LOCK_FIELDS",
    "REQUIRED_POWER_FIELDS",
    "REQUIRED_STATISTICS_FIELDS",
    "RESOURCE_REGIMES",
    "assess_count_power",
    "assess_power",
    "canonical_json",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "example_protocol",
    "freeze_protocol",
    "load_lock",
    "minimum_detectable_effect",
    "n_for_power",
    "paired_power",
    "permutation_min_two_sided_p",
    "pilot_assessment",
    "poisson_pmf",
    "poisson_sf",
    "protocol_hash",
    "sd_interval",
    "seeds_for_attainable_alpha",
    "sign_test_min_two_sided_p",
    "smallest_count_reaching_alpha",
    "summarise_campaign",
    "validate_protocol",
]
