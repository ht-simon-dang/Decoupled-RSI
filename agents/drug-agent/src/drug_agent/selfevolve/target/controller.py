"""The campaign controller: propose, dispatch, read the evidence, decide, stop.

Step 11 of the per-target evolution program. Steps 1–10 built everything an
experiment needs — contracts, data, splits, an independent evaluator, a budget
ledger, an action dispatcher — and left one thing missing: something that
*drives* them without a human between the steps. This module is that driver.

One cycle, and the order is the contract:

```text
RUNNING -> plan validation -> budget reservation -> candidate execution
        -> independent development evaluation -> append history
        -> next hypothesis OR NOMINATED OR COMPLETED/NO_IMPROVEMENT
Any missing mandatory input -> BLOCKED with structured cause
Any controller/infrastructure crash -> FAILED, recoverable event state
```

Four rules shape the whole file, and each exists because its opposite is the
easy, wrong implementation.

**The stop rule is data, declared before the campaign starts, and the campaign
cannot edit it.** :class:`StopRule` is frozen, :meth:`TargetCampaignController.run`
reads ``max_cycles`` and never writes it, and a resumed campaign counts the
cycles already in its history against the same number. A campaign that keeps
proposing until something finally beats the parent is not a search, it is a
multiple-comparisons machine with the comparisons hidden, so *the loop has no
branch that grants a cycle because the last one did not improve*. A valid cycle
that finds nothing better is ``NO_IMPROVEMENT`` and is kept.

**The pool is the whole pool.** Every attempt — crashed, refused, unscored,
worse, better — is appended to a durable JSONL history and appears in
``candidate_pool``. Nomination selects from it; it does not construct it.

**A number the controller did not get from the independent evaluator is not a
number.** The controller reads ``envelope["independent_evaluation"]`` and
nothing else: a worker's own ``result.json`` never reaches
``development_metric_value``, and a measurement whose ``evaluated_partition`` is
not ``development`` cannot nominate, whatever its value.

**The frozen test surface is absent.** :data:`CONTROLLER_ACTIONS` is exactly
Step 10's registered set, a menu naming anything else is refused at
construction, and every :class:`~..contracts.Nomination` this module emits
carries ``test_access_count = 0`` — which the contract itself re-checks.

What this module deliberately does **not** do is call an improvement an
improvement. ``Comparison`` requires an uncertainty interval that excludes no
improvement before ``IMPROVED`` may be written, and one development point
estimate has no interval. A campaign that nominates therefore reports
``INCONCLUSIVE`` and hands the paired comparison to the confirmation step.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    ArtifactKind,
    Campaign,
    CampaignState,
    ContractError,
    ErrorCategory,
    ExperimentStatus,
    FailureCategory,
    Nomination,
    ScientificDecision,
    Track,
    validate_record,
)
from .worker import SUPPORTED_ACTIONS, dispatch_plan

Record = dict[str, Any]

CONTROLLER_CONTRACT_VERSION = "1.0.0"

#: Marks a record produced by an engineering fixture rather than by a
#: measurement. Anything carrying it is kept out of scientific reporting.
SYNTHETIC_EVIDENCE_CLASS = "synthetic_test"

#: What a field says when nothing measured it. The same token
#: :mod:`drug_agent.selfevolve.target.reporting` uses, spelled here rather than
#: imported, because the controller must not pull the reporting package into a
#: campaign process. It is never a synonym for "available" and never a zero.
NOT_MEASURED = "NOT_MEASURED"

#: The interventions this controller may cause. Exactly Step 10's registered
#: set — a controller that could name one more would be advertising a capability
#: no handler backs, and a controller that could name a *final-test* API would
#: put the held-out partition inside the loop. Both are refused at construction.
CONTROLLER_ACTIONS: tuple[str, ...] = tuple(SUPPORTED_ACTIONS)

#: Substrings that name the frozen final-test surface. An action menu containing
#: one of these is refused rather than filtered: silently dropping it would hide
#: a campaign configuration that believed it could read the held-out set.
FINAL_TEST_WORDS: tuple[str, ...] = ("test", "final", "holdout", "held_out", "heldout")

#: How a development metric is read. ``mae_pactivity`` is Track A's frozen
#: primary metric and lower is better; ``qualifying_scaffolds`` is Track B's and
#: higher is. A metric absent from this table has no direction, so no
#: improvement can be computed for it and nothing measured under it can nominate.
METRIC_ORIENTATION: dict[str, str] = {
    "mae_pactivity": "LOWER_IS_BETTER",
    "qualifying_scaffolds": "HIGHER_IS_BETTER",
}

#: The nomination rule, frozen here and copied verbatim into every
#: :class:`~..contracts.Nomination`. It names the partition, the comparator and
#: the pool, because a selection rule that only says "the best one" does not say
#: which evidence it was best on.
DEVELOPMENT_SELECTION_RULE = (
    "the candidate with the best independently measured DEVELOPMENT-partition value of the "
    "campaign's primary metric, over the complete candidate pool (every attempt, including "
    "failures and non-improvements), required to beat the declared incumbent by more than the "
    "pre-declared minimum improvement. No held-out result is consulted: test_access_count is 0."
)


#: How the campaign's declared budget (``configs/research/zak-c9r.yaml`` →
#: ``budget``) maps onto ledger resources, and the factor that converts the
#: declared unit into the ledger's. One table, used by every operator entry
#: point: ``scripts/research/run_experiment.py`` dispatched a single plan
#: against a private copy of it, and a second copy is how a campaign ends up
#: enforcing a different cap from the experiment inside it.
CAP_FACTORS: dict[str, tuple[str, float]] = {
    "wall_hours": ("wall_seconds", 3600.0),
    "gpu_hours": ("gpu_seconds", 3600.0),
    "llm_output_tokens": ("llm_output_tokens", 1.0),
    "max_concurrent_gpu_jobs": ("gpu_slots", 1.0),
    "max_experiment_proposals": ("experiment_proposals", 1.0),
}


def campaign_caps(budget: Mapping[str, Any]) -> dict[str, float]:
    """``{ledger resource: cap}`` from a campaign configuration's ``budget`` block."""
    caps: dict[str, float] = {}
    for declared, (resource, factor) in CAP_FACTORS.items():
        if declared in budget and budget[declared] is not None:
            caps[resource] = float(budget[declared]) * factor
    return caps


def declare_campaign_caps(ledger: Any, budget: Mapping[str, Any], *, note: str = "") -> dict[str, float]:
    """Declare every campaign cap on *ledger* before anything can be reserved.

    ``BudgetLedger.declare`` refuses an enlargement, so calling this on a
    resumed campaign re-asserts the same numbers and cannot widen them.
    """
    caps = campaign_caps(budget)
    for resource, limit in sorted(caps.items()):
        ledger.declare(resource, limit, note=note)
    return caps


#: Where the campaign's frozen protocol and its measured baselines live,
#: relative to the repository root.
CAMPAIGN_CONFIG = Path("configs") / "research" / "zak-c9r.yaml"

#: Every protocol a campaign in this study may be registered against, resolved BY CONTENT ADDRESS.
#:
#: A campaign record names the protocol it was frozen against by ``protocol_hash``, and that hash is
#: the sha256 of a protocol file's raw bytes. :func:`check_campaign_registration` tries each file
#: here and enforces the caps of the one whose bytes the record's hash names, so the hash is the
#: selector and there is nothing else to pass.
#:
#: WHY A SECOND FILE EXISTS AT ALL. The operator authorised ten times the resources for the PFM DPO
#: round, which a preference-optimisation fine-tune needs and a sampling arm does not. Raising the
#: caps inside ``zak-c9r.yaml`` would have moved that file's sha256 and de-registered all 139
#: existing records, so the operator directed that a variant be created instead. ``zak-c9r-10x.yaml``
#: is a byte-for-byte copy of the parent with four budget numbers changed; every scientific block is
#: identical, and the evaluator address is untouched, so results stay comparable across the two.
#:
#: WHY THE RECORD DOES NOT SIMPLY CARRY THE FILENAME. ``Campaign`` lives in ``contracts.py``, one of
#: the eighteen guarded files, so adding a field there would move the evaluator address every record
#: declares. Resolution by hash needs no new field.
#:
#: This CANNOT be used to widen a cap quietly: to obtain the larger caps a record must declare the
#: hash of the file that grants them, which is one greppable line in the record, and a record whose
#: hash matches no file here does not register at all.
#:
#: The FIRST entry stays the default for everything that reads the protocol for science rather than
#: for budget — the incumbent reference, the benchmark and the baselines all come from the parent, so
#: a budget variant can never move a number anybody is measured against.
CAMPAIGN_CONFIG_VARIANTS: tuple[Path, ...] = (
    CAMPAIGN_CONFIG,
    Path("configs") / "research" / "zak-c9r-10x.yaml",
)


#: Protocol addresses that a record may still declare although no file hashes to them any more, and
#: the successor whose caps apply instead.
#:
#: WHY THIS EXISTS. A record's ``protocol_hash`` is the content address of the protocol file it was
#: registered under, so re-freezing that file de-registers every record naming the old bytes: they
#: stop resolving, and a campaign that cannot resolve its protocol cannot be resumed. The 2026-09-30
#: re-freeze moved both files -- ``benchmark.qualification_hash``, ``benchmark.calibration_hash``,
#: ``benchmark.evaluator_hash`` and ``protocol.protocol_hash`` all had to follow the novelty exclusion
#: set becoming digests rather than plaintext SMILES -- and 154 campaign records name the old
#: addresses.
#:
#: The alternative was to rewrite those records' ``protocol_hash`` to the new value. That is worse: a
#: record states the protocol that governed it, and editing the statement to match a file it never
#: saw is a claim about a registration that never happened. So the records keep what they declared and
#: the supersession is written down here, where it is one greppable table rather than 154 silent
#: edits.
#:
#: WHY IT CANNOT WIDEN A CAP, MECHANICALLY AND NOT BY PROMISE. Each entry pins the content address of
#: the ``budget`` block the superseded file carried, taken from git before the re-freeze.
#: :func:`resolve_campaign_protocol` recomputes the successor's budget address and REFUSES if it
#: differs, so a supersession can only ever forward a record to caps identical to the ones it
#: registered under. Both blocks were verified identical when this table was written; the check is
#: there for the next re-freeze, which may not be.
CAMPAIGN_CONFIG_SUPERSESSIONS: dict[str, tuple[Path, str, str]] = {
    "sha256:7692799a834085d5b7590ad6c9f4e073d62be9f55188e794fbf3a88c27a7190d": (
        CAMPAIGN_CONFIG,
        "sha256:9e08a12596f0c747831a0d8ddceab480cd8e6d22ac18fc18950918935c48b576",
        "2026-09-30 novelty-digest re-freeze: the qualification record stopped serialising the "
        "accepted corpus as plaintext SMILES, which moved the qualification, calibration, evaluator "
        "and protocol-lock addresses this file declares. No budget number moved",
    ),
    "sha256:729f335f1d13ed88bc0276d7db0027fc6111ba82507fd8529774ece30a4d4550": (
        Path("configs") / "research" / "zak-c9r-10x.yaml",
        "sha256:d0f0789fbb5605165a9a6b192a25d6c4542f00b4e70b6ec5655ddf1969c778fd",
        "2026-09-30 novelty-digest re-freeze, applied to the 10x budget variant for the same reason "
        "and changing none of its four raised caps",
    ),
}


def resolve_campaign_protocol(
    declared: str, *, root: Path | str | None = None
) -> tuple[Path, dict[str, Any]] | None:
    """The protocol file whose bytes hash to *declared*, and its ``budget`` block.

    ``None`` when no registered protocol variant matches, which is a refusal and never a default:
    guessing a protocol for a record that names one nobody has is how a campaign ends up measured
    against bars it never declared.
    """
    repo = Path(root) if root is not None else _repo_root()
    for relative in CAMPAIGN_CONFIG_VARIANTS:
        path = repo / relative
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if "sha256:" + hashlib.sha256(raw).hexdigest() != str(declared):
            continue
        # NOT suppressed. A suppressed parse failure leaves budget == {}, campaign_caps({}) returns
        # {}, every `outer.get(resource)` in the cap loop is None, and the whole loop is a no-op --
        # so a record passes with NO caps applied at all. The refusal contract covers "no variant
        # matched"; it did not cover "matched, and the budget was unreadable", which is the failure
        # that actually widens a cap. Now that this function is the only source of caps, an
        # unreadable protocol is a refusal.
        import yaml

        try:
            parsed = yaml.safe_load(raw.decode("utf-8")) or {}
        except Exception as exc:  # any parse failure must refuse, not widen a cap
            raise ValueError(
                f"the protocol {relative} matched by content address but could not be parsed, so "
                f"its caps cannot be read and no cap would be enforced: {type(exc).__name__}: {exc}"
            ) from exc
        budget = dict((parsed.get("budget") if isinstance(parsed, Mapping) else None) or {})
        if not budget:
            raise ValueError(
                f"the protocol {relative} matched by content address but declares no budget block, "
                f"so no cap would be enforced"
            )
        return relative, budget

    superseded = CAMPAIGN_CONFIG_SUPERSESSIONS.get(str(declared))
    if superseded is not None:
        relative, pinned_budget_hash, _reason = superseded
        path = repo / relative
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        import yaml

        try:
            parsed = yaml.safe_load(raw.decode("utf-8")) or {}
        except Exception as exc:  # same rule as above: refuse, never widen a cap
            raise ValueError(
                f"the successor protocol {relative} could not be parsed, so the caps of the "
                f"superseded {declared} cannot be honoured: {type(exc).__name__}: {exc}"
            ) from exc
        budget = dict((parsed.get("budget") if isinstance(parsed, Mapping) else None) or {})
        if not budget:
            raise ValueError(
                f"the successor protocol {relative} declares no budget block, so forwarding a "
                f"record from {declared} to it would enforce no cap"
            )
        live_budget_hash = "sha256:" + hashlib.sha256(
            json.dumps(budget, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if live_budget_hash != pinned_budget_hash:
            # The one thing a supersession must never do. Forwarding to different caps would let a
            # re-freeze silently grant every historical record whatever the new file allows.
            raise ValueError(
                f"{declared} is recorded as superseded by {relative}, but that file's budget block "
                f"now hashes to {live_budget_hash} rather than the {pinned_budget_hash} the "
                f"superseded protocol carried. A supersession may forward a record only to caps "
                f"identical to the ones it registered under; re-freeze the supersession deliberately"
            )
        return relative, budget

    return None
BASELINE_RESULTS = Path("configs") / "research" / "zak-c9r" / "baseline-results.json"

#: What every number in this campaign MEANS: its direction, the value a perfect
#: agent would score, the value no skill at all scores, and how wide the interval
#: around it is. Registered before any candidate search, and read rather than
#: retyped for the same reason as :data:`BASELINE_RESULTS`.
MEASUREMENT_SEMANTICS = Path("configs") / "research" / "zak-c9r" / "measurement-semantics.json"

#: The frozen Track A parent this campaign is measured against.
INCUMBENT_BASELINE_ID = "A-B0-parent-rf"
INCUMBENT_ARTIFACT_ID = "art_zak_c9r_parent_rf"

#: Step 8's nine-arm generation sweep: every Track B arm this programme has
#: measured, each over the same fixed submission budget. It is the only
#: committed record that states a Track B count at the budget this campaign
#: series actually runs at, which is why :func:`frozen_track_b_incumbent_and_references`
#: reads the incumbent out of it rather than out of :data:`BASELINE_RESULTS`.
GENERATION_SWEEP = Path("configs") / "research" / "zak-c9r" / "generation-sweep.json"

#: The frozen Track B parent: REINVENT4's unconditional prior, which knows
#: nothing about ZAK and is still the best arm the campaign's one generative
#: action can reach.
TRACK_B_INCUMBENT_BASELINE_ID = "B-B6-reinvent4-prior"
TRACK_B_INCUMBENT_ARTIFACT_ID = "art_zak_c9r_parent_reinvent4_prior"

#: Its arm name in :data:`GENERATION_SWEEP`.
TRACK_B_INCUMBENT_ARM = "reinvent4"

#: Track B's primary metric — and, deliberately, the exact field name the sweep
#: records it under, so the name the campaign is scored on and the key its
#: comparator is read from cannot come apart.
TRACK_B_METRIC = "qualifying_scaffolds"


def frozen_incumbent_and_references(root: Path | str | None = None) -> tuple[Record, Record]:
    """The number to beat, and how weak the evidence around it is.

    Both are **read** out of the committed records — never retyped — because a
    hand-copied constant is exactly how "the incumbent" and "what the baselines
    measured" drift apart. Every entry point that starts a campaign
    (``scripts/research/run_campaign.py``, ``drugagent self-evolve
    target-campaign``) uses this one definition.

    The references half is what Step 7 and Step 8 measured about how little
    signal this target has: the trivial-median reference the learned arms are
    all worse than, the label-shuffle p-value, and the oracle positive control
    that makes the null interpretable. The agent is told all of it, because a
    controller that fed the agent only the leaderboard would have it chase a
    number that Step 7 already showed is mostly noise.

    Raises:
        FileNotFoundError: the committed baseline record is not on this host.
    """
    repo = Path(root) if root is not None else _repo_root()
    record = json.loads((repo / BASELINE_RESULTS).read_text(encoding="utf-8"))
    arms = {str(a.get("baseline_id")): a for a in record.get("arms") or []}
    parent = arms[INCUMBENT_BASELINE_ID]
    config: Record = {}
    with contextlib.suppress(Exception):
        import yaml

        config = dict(yaml.safe_load((repo / CAMPAIGN_CONFIG).read_text(encoding="utf-8")) or {})
    track_a = dict((config.get("baselines") or {}).get("track_a") or {})
    measured = dict((config.get("benchmark") or {}).get("measured") or {})
    incumbent: Record = {
        "artifact_id": INCUMBENT_ARTIFACT_ID,
        "baseline_id": INCUMBENT_BASELINE_ID,
        "metric": str(parent["primary_metric"]),
        "value": float(parent["primary_value"]),
        "coverage": parent.get("coverage"),
        "n_pool": parent.get("n_pool"),
        "pool_hash": parent.get("pool_hash"),
        "evaluated_partition": record.get("evaluated_partition"),
        "evaluator_hash": record.get("evaluator_hash"),
        "source": str(BASELINE_RESULTS),
    }
    references: Record = {
        "declared_baselines_mae_pactivity": {
            str(a.get("baseline_id")): a.get("primary_value")
            for a in record.get("arms") or []
            if a.get("track") == "A"
        },
        "trivial_median_predictor_mae": track_a.get("trivial_median_predictor_mae"),
        "all_learned_arms_worse_than_median": track_a.get("all_learned_arms_worse_than_median"),
        "b1_label_shuffle_permutation_p": track_a.get("b1_label_shuffle_permutation_p"),
        "track_a_probe": measured.get("track_a_probe"),
        "track_a_permutation_positive_control": measured.get("track_a_permutation_positive_control"),
        "n_development_compounds": parent.get("n_pool"),
        "note": (
            "Step 7 measured that this predictor family has essentially no signal on this target "
            "(label-shuffle p = 0.3134, against an oracle positive control at p = 0.00498), and "
            "every learned Track A baseline is worse than predicting the development median. "
            "NO_IMPROVEMENT is an expected outcome here and a complete one."
        ),
    }
    return incumbent, references


def frozen_track_b_incumbent_and_references(root: Path | str | None = None) -> tuple[Record, Record]:
    """Track B's number to beat, and the whole measured table it stands in.

    The same read-never-retype discipline as :func:`frozen_incumbent_and_references`,
    and on this track one further reason to insist on it: the repository holds
    **two** committed counts for the same parent generator, and only one of them
    is a comparator.

    ``baseline-results.json`` records ``B-B6-reinvent4-prior`` at
    ``primary_value = 1.0``. That was measured over a 24-molecule probe
    portfolio. Step 8's generation sweep re-measured the same prior over 150
    molecules and counted 4. A campaign that submits 150 molecules and compares
    itself to the 24-molecule figure would clear its bar on **submission volume
    alone** — six times the molecules at the same per-molecule rate — and report
    an improvement that is an artefact of the denominator. So the incumbent is
    read from the sweep, at the budget this series actually runs at, and
    ``n_submitted`` travels next to it so the volume match is visible in the
    record rather than assumed by the reader.

    ``evaluator_hash`` is the **live** kernel digest, not the one the sweep
    froze: it is the address of the evaluator that will score this campaign, and
    a candidate that changed the judgement would show up here as a mismatch
    against the sweep rather than silently inherit its authority.

    The references half is the whole nine-arm table, because the incumbent's
    count only means something beside the arms either side of it. The spread is
    what says this endpoint separates methods at all, and where in that spread
    the two arms ``change_sampling`` can actually reach sit is what says how
    much of the spread is available to this loop.

    Raises:
        FileNotFoundError: the committed sweep is not on this host.
        ValueError: the sweep carries no row for the incumbent arm. There is no
            default to fall back to — an absent comparator is not a comparator
            of zero, and a Track B campaign without one has nothing to beat.
    """
    from ..evaluation.target_benchmark import evaluator_hash

    repo = Path(root) if root is not None else _repo_root()
    record = json.loads((repo / GENERATION_SWEEP).read_text(encoding="utf-8"))
    declared = dict(record.get("declared_secondary_rate") or {})
    rows = [dict(row) for row in declared.get("rows") or []]
    parent = next((row for row in rows if str(row.get("arm")) == TRACK_B_INCUMBENT_ARM), None)
    if parent is None:
        raise ValueError(
            f"{GENERATION_SWEEP} carries no declared_secondary_rate row for arm "
            f"{TRACK_B_INCUMBENT_ARM!r}, so there is no measured Track B incumbent to beat. "
            f"Arms present: {sorted(str(row.get('arm')) for row in rows)}"
        )
    n_submitted = int(parent["n_submitted"])
    incumbent: Record = {
        "artifact_id": TRACK_B_INCUMBENT_ARTIFACT_ID,
        "baseline_id": TRACK_B_INCUMBENT_BASELINE_ID,
        "metric": TRACK_B_METRIC,
        "value": float(parent[TRACK_B_METRIC]),
        "n_submitted": n_submitted,
        # Stage D scored the sweep against the development-partition exclusion
        # boundary, exactly as this campaign is scored; the field is stated
        # because a measurement whose partition is not `development` cannot
        # nominate, whatever its value.
        "evaluated_partition": "development",
        "evaluator_hash": evaluator_hash(),
        "source": str(GENERATION_SWEEP),
    }
    references: Record = {
        "measured_arms_at_fixed_submission_budget": [
            {
                "arm": row.get("arm"),
                "conditioning": row.get("conditioning"),
                TRACK_B_METRIC: row.get(TRACK_B_METRIC),
                "n_submitted": row.get("n_submitted"),
            }
            for row in rows
        ],
        "n_submitted_per_arm": n_submitted,
        "known_active_rate_per_molecule": declared.get("known_active_rate_per_molecule"),
        "note": (
            f"Every experiment in this campaign series submits {n_submitted} molecules, the same "
            f"budget every arm in the table above was measured at, so a count produced here is "
            f"directly comparable to that table and to the incumbent. A count measured at any "
            f"other budget is not comparable to either. `change_sampling` reaches the backends "
            f"'pfm' and 'reinvent4' and nothing else, and those two are the BOTTOM of that "
            f"table: the arms at the top of it are NOT selectable from this loop, so headroom "
            f"here has to be earned by changing how sampling is done rather than by choosing a "
            f"known winner. A qualifying count of ZERO is the EXPECTED outcome at this bar and "
            f"is a complete result, not a broken arm: the seven gates are hard enough that REAL "
            f"human-measured ZAK actives clear them at 1 in 6 "
            f"(known_active_rate_per_molecule above)."
        ),
    }
    return incumbent, references


#: Which frozen incumbent a campaign gets, keyed by the metric
#: :func:`_campaign_metric` derives from the record's own ``track``.
#:
#: This lives here, beside both loaders, because it previously did not exist and
#: each caller chose for itself. ``scripts/research/run_campaign.py`` dispatched
#: on the track; ``drugagent self-evolve target-campaign`` called
#: :func:`frozen_incumbent_and_references` unconditionally — the Track A loader —
#: while every campaign record on disk is Track B. The CLI's own comment claimed
#: the two "cannot disagree about it", which was the invariant the code did not
#: hold. One table, imported by both, is what makes that comment true.
FROZEN_INCUMBENT_BY_METRIC: dict[str, Callable[[Path], tuple[Record, Record]]] = {
    "mae_pactivity": frozen_incumbent_and_references,
    "qualifying_scaffolds": frozen_track_b_incumbent_and_references,
}


def frozen_incumbent_for_campaign(
    campaign: Mapping[str, Any], *, root: Path | str | None = None
) -> tuple[str, Record, Record]:
    """The metric and the frozen incumbent this campaign is scored against.

    Returns ``(metric, incumbent, references)``. Raises :class:`KeyError` when the
    campaign's metric has no registered frozen incumbent, because handing a
    campaign the wrong track's number to beat is worse than refusing to start:
    every cycle afterwards would be compared against a quantity from a different
    experiment, and nothing downstream would notice.
    """
    metric = _campaign_metric(campaign)
    loader = FROZEN_INCUMBENT_BY_METRIC.get(metric)
    if loader is None:
        raise KeyError(
            f"campaign {campaign.get('campaign_id')!r} is scored on metric {metric!r} and no "
            f"frozen incumbent is registered for it; the registered metrics are "
            f"{sorted(FROZEN_INCUMBENT_BY_METRIC)}"
        )
    incumbent, references = loader(Path(root) if root is not None else _repo_root())
    return metric, incumbent, references


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def check_campaign_registration(campaign: Mapping[str, Any], *, root: Path | str | None = None) -> list[str]:
    """What is wrong with this campaign's registration, before anything is spent.

    A ``Campaign`` record is a registration, not a request: it names the
    protocol it was frozen against and the budget it was granted. Two things
    have to still be true when it runs, and both are checked by every entry
    point so the CLI cannot be the weaker one:

    1. ``protocol_hash`` still matches ``configs/research/zak-c9r.yaml``. A
       campaign registered against a different protocol is a different campaign;
       re-freeze deliberately rather than run under a changed one.
    2. Its budget does not exceed the campaign caps declared in that file. A
       campaign may **tighten** its budget — every campaign this step ran did —
       but "write a bigger number in the record" must not be a way to enlarge a
       cap, for the same reason ``BudgetLedger.declare`` refuses one.

    Returns:
        A list of human-readable problems; empty when the registration holds.
        Deliberately a list rather than an exception, so a caller can report
        every problem at once.
    """
    repo = Path(root) if root is not None else _repo_root()
    problems: list[str] = []
    declared = str(campaign.get("protocol_hash") or "")
    # Resolved by content address across CAMPAIGN_CONFIG_VARIANTS: the record's hash names the
    # protocol whose caps apply. A record matching none of them does not register, and the message
    # names every candidate and what each one currently hashes to, because "does not match" without
    # the measured addresses beside it is the least actionable refusal this function could give.
    resolved = resolve_campaign_protocol(declared, root=repo)
    if resolved is None:
        measured_now = []
        for relative in CAMPAIGN_CONFIG_VARIANTS:
            try:
                digest = "sha256:" + hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            except OSError as exc:
                digest = f"<unreadable: {exc}>"
            measured_now.append(f"{relative} -> {digest}")
        return [
            f"protocol_hash: the record was registered against {declared!r}, which is none of the "
            f"registered protocols ({'; '.join(measured_now)}). The campaign was frozen against a "
            f"different protocol; re-freeze deliberately rather than running under a changed one"
        ]
    config_relative, budget_block = resolved
    outer = campaign_caps(budget_block)
    for resource, amount in sorted(dict(campaign.get("budget") or {}).items()):
        cap = outer.get(str(resource))
        if cap is not None and float(amount) > cap + 1e-9:
            problems.append(
                f"budget.{resource}: the record asks for {float(amount)} and the campaign cap "
                f"declared in {config_relative} is {cap}. A campaign may tighten its budget, "
                f"never enlarge it"
            )
    return problems


#: The blocks inside one ``metrics`` entry that are themselves transcribed
#: measurements, each naming the record it came from. Walked by
#: :func:`verify_measurement_semantics`; an entry that carries no
#: ``source_contains`` (the incumbent reference, whose value is re-read from
#: ``baseline-results.json`` by :func:`frozen_incumbent_and_references`) is not
#: checked here, because there is no quoted substring to check.
_SEMANTICS_METRIC_BLOCKS: tuple[str, ...] = (
    "no_skill_reference",
    "incumbent_reference",
    "known_active_ceiling",
    "measured_separation_at_150_molecules",
)


def _sourced_entries(record: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Every labelled block of *record* that quotes a substring of a source file."""
    found: list[tuple[str, Mapping[str, Any]]] = []
    for metric, block in sorted(dict(record.get("metrics") or {}).items()):
        for name in _SEMANTICS_METRIC_BLOCKS:
            entry = dict(block or {}).get(name)
            if isinstance(entry, Mapping):
                found.append((f"metrics.{metric}.{name}", entry))
    for index, entry in enumerate(record.get("auxiliary_measurements") or []):
        if isinstance(entry, Mapping):
            label = str(entry.get("measurement_id") or index)
            found.append((f"auxiliary_measurements.{label}", entry))
    return [(label, e) for label, e in found if e.get("source") and e.get("source_contains")]


def verify_measurement_semantics(record: Mapping[str, Any], *, root: Path | str | None = None) -> list[str]:
    """What in this semantics record has drifted away from the record it quotes.

    The measurement semantics are a *transcription*: every number in them was
    measured somewhere else, by some other step, and copied here so the proposer
    can be shown it. A transcription with no back-pointer is how a campaign ends
    up citing a number that nothing measures any more — which is the failure this
    whole record exists to stop, pointed at itself.

    So every entry names its origin in ``source`` and quotes a verbatim
    ``source_contains`` out of it, and this re-reads each origin and checks the
    quote is still there. It cannot prove the transcribed *number* is right — a
    file can be edited around its own quoted line — but it does mean a record
    that was regenerated, re-measured or deleted cannot go on silently backing a
    number the agent is being shown as evidence.

    Returns:
        A list of human-readable problems; empty when every quote still holds.
        A list rather than an exception, for :func:`check_campaign_registration`'s
        reason: a caller re-freezing this record wants every drifted entry at
        once, not the first one.
    """
    repo = Path(root) if root is not None else _repo_root()
    problems: list[str] = []
    for label, entry in _sourced_entries(record):
        source = str(entry["source"])
        quoted = str(entry["source_contains"])
        try:
            text = (repo / source).read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{label}: the source {source} it was transcribed from is unreadable: {exc}")
            continue
        if quoted not in text:
            problems.append(f"{label}: the substring quoted from {source} is no longer present there")
    return problems


def measurement_semantics(root: Path | str | None = None) -> Record:
    """What every number this campaign shows an agent actually means.

    Read out of the committed record, never retyped, on the same discipline as
    :func:`frozen_incumbent_and_references` — and then *verified*, because a
    semantics record whose quotes no longer resolve is worse than no semantics
    record at all: it is a stale authority the proposer has no way to doubt.

    The record carries no per-compound value and no held-out quantity — counts,
    rates, correlations and intervals only — so it is admissible in the evidence
    channel that :func:`structured_feedback` hands to a model.

    Raises:
        FileNotFoundError: the committed record is not on this host.
        ValueError: one or more entries no longer match the source they quote.
            Every problem is listed, not just the first.
    """
    repo = Path(root) if root is not None else _repo_root()
    record: Record = json.loads((repo / MEASUREMENT_SEMANTICS).read_text(encoding="utf-8"))
    problems = verify_measurement_semantics(record, root=repo)
    if problems:
        raise ValueError(
            f"{MEASUREMENT_SEMANTICS} no longer matches the records it was transcribed from: "
            + "; ".join(problems)
        )
    return record


def _scalar(value: Any) -> float | None:
    """*value* as a finite float, or ``None`` if it is not a plain number.

    ``bool`` is excluded deliberately: ``True`` is an ``int`` in Python and a
    flag silently arithmetic-ing itself into a distance is exactly the kind of
    number this module exists to keep out of the agent's evidence.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric_semantics(metric: str, *, semantics: Mapping[str, Any], incumbent: Mapping[str, Any]) -> Record:
    """How to read *metric*: its direction, its perfect value, and the incumbent's distance.

    ``mae_pactivity = 0.49`` says nothing on its own. It is only meaningful next
    to three other numbers: 0.0 is perfect, 0.444117 is what predicting the
    development median scores, and the incumbent is therefore 0.046 *worse* than
    a constant. A proposer shown only the first number reads a campaign as
    winnable that a proposer shown all four reads as already answered.

    ``distance_to_no_skill_reference`` is signed so that POSITIVE always means
    the incumbent is better than the no-skill reference, whichever way the metric
    runs. It is ``None`` when either side is missing or when the no-skill
    reference is not a scalar — ``qualifying_scaffolds`` quotes a
    ``value_per_molecule`` rather than a ``value``, because a raw count is only
    comparable at a fixed submission budget, and no distance is the honest answer
    there rather than a number manufactured out of two different denominators.

    An unregistered metric returns ``registered = False`` and a plain reason
    rather than raising: the caller is a feedback record being assembled, and a
    metric nobody wrote semantics for should cost the agent that one block, not
    the whole evidence channel.
    """
    block = dict(semantics.get("metrics") or {}).get(metric)
    if not isinstance(block, Mapping):
        return {
            "metric": metric,
            "registered": False,
            "reason": (
                f"no measurement semantics are registered for {metric!r}, so nothing here says "
                f"which direction is better, what perfect would be, or how wide its interval is. "
                f"Registered metrics: {sorted(dict(semantics.get('metrics') or {}))}"
            ),
        }
    no_skill = _scalar(dict(block.get("no_skill_reference") or {}).get("value"))
    incumbent_value = _scalar(incumbent.get("value"))
    orientation = str(block.get("orientation") or "")
    distance: float | None = None
    if no_skill is not None and incumbent_value is not None:
        if orientation == "LOWER_IS_BETTER":
            distance = no_skill - incumbent_value
        elif orientation == "HIGHER_IS_BETTER":
            distance = incumbent_value - no_skill
    return {
        **dict(block),
        "metric": metric,
        "registered": True,
        "incumbent_value": incumbent.get("value"),
        "distance_to_no_skill_reference": distance,
        "distance_to_no_skill_reference_note": (
            "signed so POSITIVE means the incumbent is BETTER than the no-skill reference and "
            "NEGATIVE means it is worse, whichever direction the metric runs. null means the two "
            "are not comparable as scalars and no distance was manufactured"
        ),
    }


def poisson_resolution(expected_count: float) -> Record:
    """How far apart two COUNT results have to be before the difference is real.

    ``qualifying_scaffolds`` is a count, and a count endpoint has no variance of
    its own to estimate: its spread is fixed by its mean. At an expected 4
    qualifying scaffolds the Poisson SD is 2, so an arm scoring 5 against an arm
    scoring 4 has produced nothing — one SD is the floor of what this budget can
    resolve, and the campaign owns no replicate that would narrow it.

    This is stated rather than left implicit because the arithmetic runs the
    wrong way round for intuition: the SMALLER the expected count, the LARGER
    the difference has to be *as a fraction of it*, so the endpoints where a
    campaign has the least to work with are exactly the ones where a small
    improvement looks most convincing.

    A negative or non-finite input returns a record with null values and a
    stated reason rather than raising: the caller is assembling an evidence
    record, and a nonsense expectation should be visible in it, not fatal to it.
    """
    lam = _scalar(expected_count)
    note = (
        "a COUNT endpoint has no variance of its own to estimate -- its spread is fixed by its "
        "mean, at sd = sqrt(mean). A difference smaller than one sd is NOT distinguishable from "
        "Poisson noise at this budget, whatever it looks like, and no replicate here narrows it."
    )
    if lam is None or lam < 0.0:
        return {
            "expected_count": None,
            "poisson_sd": None,
            "smallest_difference_above_one_sd": None,
            "smallest_difference_above_two_sd": None,
            "note": note,
            "reason": (
                f"an expected count must be a finite, non-negative number and {expected_count!r} "
                f"is not, so no Poisson resolution can be stated for it"
            ),
        }
    sd = math.sqrt(lam)
    return {
        "expected_count": lam,
        "poisson_sd": sd,
        "smallest_difference_above_one_sd": sd,
        "smallest_difference_above_two_sd": 2.0 * sd,
        "note": note,
    }


class CycleOutcome(StrEnum):
    """What one cycle produced. None of these is a software crash."""

    #: The proposer's plan did not validate. Nothing launched, nothing charged
    #: to the experiment budget — but the reasoning call that produced it was.
    PLAN_REJECTED = "PLAN_REJECTED"
    #: The run launched and did not produce a usable candidate: a broken
    #: candidate, a timeout, a missing backend. A *result*, never retried.
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    #: A transient infrastructure fault. Retried under a fresh request key, and
    #: every attempt is charged.
    INFRASTRUCTURE_RETRY = "INFRASTRUCTURE_RETRY"
    #: The run succeeded and no independent development number came back — or
    #: one came back that was not measured on the development partition.
    #: Unknown, never zero.
    UNSCORED = "UNSCORED"
    #: A valid, complete cycle that found nothing better. A first-class result.
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    #: A valid cycle whose independently measured development value beats the
    #: declared incumbent by more than the declared minimum.
    DEVELOPMENT_IMPROVEMENT = "DEVELOPMENT_IMPROVEMENT"
    #: The reasoning backend could not serve. Nothing was dispatched, so this
    #: is not a candidate — and it does not spend a declared cycle either.
    PROPOSER_UNAVAILABLE = "PROPOSER_UNAVAILABLE"
    #: The controller itself faulted. Same rule: recorded, charged, not a cycle.
    CONTROLLER_CRASH = "CONTROLLER_CRASH"


class TerminalReason(StrEnum):
    """Why the loop stopped. Recorded separately from the campaign state."""

    NOMINATION_SELECTED = "NOMINATION_SELECTED"
    CYCLE_BUDGET_EXHAUSTED = "CYCLE_BUDGET_EXHAUSTED"
    RESOURCE_BUDGET_EXHAUSTED = "RESOURCE_BUDGET_EXHAUSTED"
    MALFORMED_PLAN_LIMIT = "MALFORMED_PLAN_LIMIT"
    MISSING_MANDATORY_INPUT = "MISSING_MANDATORY_INPUT"
    PROPOSER_UNAVAILABLE = "PROPOSER_UNAVAILABLE"
    CONTROLLER_CRASH = "CONTROLLER_CRASH"


#: Campaign state -> the run state whose exit code it shares. There is one exit
#: table in this repository (``drug_agent.cli.main._RUN_EXIT_CODES``) and this is
#: a mapping onto it, not a second copy. ``COMPLETED`` shares ``SUCCEEDED``'s
#: zero on purpose: a campaign that ran its declared cycles and found nothing
#: better is a successful execution carrying a scientific result.
_STATE_AS_RUN_STATE: dict[str, str] = {
    CampaignState.NOMINATED.value: "SUCCEEDED",
    CampaignState.COMPLETED.value: "SUCCEEDED",
    CampaignState.BLOCKED.value: "CREATED",  # nothing ran
    CampaignState.RUNNING.value: "RUNNING",  # neither finished nor failed
    CampaignState.FAILED.value: "FAILED",
}


def campaign_exit_code(state: str) -> int:
    """Process status for a campaign state.

    A scientific outcome is not an execution failure: ``COMPLETED`` with
    ``NO_IMPROVEMENT`` exits 0. What earns a non-zero status is the campaign not
    having produced the answer it was asked for.
    """
    from ...cli.main import run_exit_code

    return run_exit_code(_STATE_AS_RUN_STATE.get(str(state), "FAILED"))


# ---------------------------------------------------------------------------
# the declared stop rule
# ---------------------------------------------------------------------------
#: Hard ceiling on ``StopRule.max_proposer_retries``. A retry loop that an
#: operator can widen without limit is not a bounded loop.
MAX_PROPOSER_RETRIES_CEILING = 3

#: Seconds to wait before proposer retry *i* (the last entry repeats). Short on
#: purpose: the measured failure is an output-formatting artifact that a fresh
#: session fixes immediately, not a backend outage that needs waiting out.
PROPOSER_RETRY_BACKOFF_S: tuple[float, ...] = (5.0, 20.0)

#: Substrings that make a ``ProposerUnavailable`` TERMINAL rather than
#: transient. Retrying any of these is pointless at best: the subscription
#: refusal and the authentication refusal both come back identically on the
#: next call, and :class:`~..backends.ClaudeCLIBackend` latches
#: ``quota_exhausted`` so the retry would not even reach the provider. Matched
#: case-insensitively against the failure message.
TERMINAL_PROPOSER_FAILURE_MARKERS: tuple[str, ...] = (
    # subscription / billing
    "usage limit",
    "quota",
    "credit balance",
    "billing",
    "upgrade to",
    "rate limit",
    "rate_limit",
    "429",
    # authentication / authorisation
    "authentication",
    "unauthorized",
    "unauthorised",
    "not authenticated",
    "invalid api key",
    "please run /login",
    "oauth token",
    "403 forbidden",
    "401 ",
    # the backend is not installed, or this process already spent its own bound
    "is not on path",
    "binary vanished",
    "call budget exhausted",
    "already settled in the budget ledger",
)


def _is_terminal_proposer_failure(message: str) -> bool:
    """Is this reasoning-backend failure one a fresh invocation cannot fix?

    Only a *transient* failure is retried. The measured population this exists
    for is the transient kind — a well-formed JSON reply with an empty
    ``result`` and exit 1, produced when the CLI's own five-attempt
    ``StructuredOutput`` budget ran out against ``--json-schema`` — which a new
    session clears. A spent subscription or a missing credential is a different
    thing, and asking again is how a campaign turns one refusal into five.
    """
    low = (message or "").lower()
    return any(marker in low for marker in TERMINAL_PROPOSER_FAILURE_MARKERS)


def _proposer_failure_is_retryable(exc: ProposerUnavailable) -> bool:
    """May this failure be answered with a fresh invocation?

    Two conditions, and the second is the conservative one.

    A retry is justified by a specific piece of evidence: a reasoning backend
    *attempted* the call and reported how it went. That is what
    ``ProposerUnavailable.usage`` carries, and it is what makes "the session was
    poisoned, start a new one" a sensible response. A ``ProposerUnavailable``
    raised with **no backend report at all** is a proposer that declined before
    any backend was reached — there is no session to refresh and nothing to
    learn from asking again, so it keeps the original behaviour and stops the
    campaign. Absence of evidence is not licence to retry.

    The second condition is the message: a spent subscription, a missing
    credential or a missing binary comes back identically next time.
    """
    if not dict(exc.usage or {}).get("backend"):
        return False
    return not _is_terminal_proposer_failure(str(exc))


def _sleep(seconds: float) -> None:
    """Indirection so a test can make the retry backoff free."""
    if seconds > 0:
        time.sleep(seconds)


def _proposer_backoff_s(attempt: int) -> float:
    """Seconds before retry ``attempt`` (1-based); the last step repeats."""
    if not PROPOSER_RETRY_BACKOFF_S:
        return 0.0
    index = min(max(attempt, 1) - 1, len(PROPOSER_RETRY_BACKOFF_S) - 1)
    return float(PROPOSER_RETRY_BACKOFF_S[index])


@dataclass(frozen=True)
class StopRule:
    """What bounds the campaign, declared before it starts.

    Frozen on purpose. Every field here is a number a losing campaign would like
    to change, and the whole point of writing them down before the first cycle
    is that the campaign cannot.
    """

    #: How many cycles this campaign may ever run, across restarts. A cycle is
    #: spent whether it improved, failed or was refused; only an infrastructure
    #: retry of an already-counted cycle is not a new cycle.
    max_cycles: int
    #: Stop as soon as a cycle improves. ``False`` keeps searching to the cycle
    #: budget and nominates from the whole pool at the end.
    stop_on_first_improvement: bool = False
    #: How much better than the incumbent a candidate must be to count as an
    #: improvement, in the metric's own units, stated as improvement (positive
    #: is better) whatever the metric's direction. ``0.0`` means strictly better.
    min_improvement: float = 0.0
    #: Consecutive unvalidatable proposals before the campaign gives up. A
    #: proposer that cannot write a valid plan will not learn to by being asked
    #: again; spinning would spend the token budget on nothing.
    max_consecutive_malformed_plans: int = 2
    #: Retries for ``INFRASTRUCTURE_ERROR`` within one cycle. A
    #: ``CANDIDATE_ERROR`` is a result and is never retried.
    max_infrastructure_retries: int = 1
    #: Fresh proposer invocations to try after a ``ProposerUnavailable`` before
    #: the campaign is declared BLOCKED. This is an INFRASTRUCTURE bound, not a
    #: scientific one: it buys no extra cycle, no extra experiment and no extra
    #: chance to improve — a retried call that succeeds runs the SAME cycle the
    #: failed one was going to run. A 26-campaign run measured why it is needed:
    #: four campaigns died PROPOSER_UNAVAILABLE and three of the four were the
    #: tail of an output-formatting distribution in which 57% of calls already
    #: needed a second ``StructuredOutput`` attempt and a campaign that failed
    #: its first attempt with the identical error recovered on its second. The
    #: control arm runs a uniform-random proposer with no model in the loop and
    #: is structurally immune, so this failure can only ever remove TREATMENT
    #: campaigns — retrying removes a bias, it does not create one. ``0``
    #: restores the old behaviour: one failed call ends the campaign.
    max_proposer_retries: int = 1
    #: Output-token headroom required before a cycle starts, when the ledger
    #: declares that cap. The Step 10 measurement was 20,899 tokens for one
    #: worker session against a 100,000 campaign cap, so the token budget binds
    #: before the proposal budget and a cycle that cannot be paid for must not
    #: start. ``None`` disables the pre-check (the ledger still refuses).
    per_cycle_output_tokens: float | None = None

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("a campaign with no cycles is a stopped campaign; max_cycles must be >= 1")
        if self.min_improvement < 0:
            raise ValueError(
                "min_improvement is stated as improvement and cannot be negative: a negative "
                "threshold would accept a regression as a win"
            )
        if self.max_consecutive_malformed_plans < 1:
            raise ValueError("max_consecutive_malformed_plans must be >= 1")
        if self.max_infrastructure_retries < 0:
            raise ValueError("max_infrastructure_retries cannot be negative")
        if self.max_proposer_retries < 0:
            raise ValueError("max_proposer_retries cannot be negative")
        if self.max_proposer_retries > MAX_PROPOSER_RETRIES_CEILING:
            raise ValueError(
                f"max_proposer_retries is bounded at {MAX_PROPOSER_RETRIES_CEILING}: an "
                f"unbounded proposer retry loop would spend the campaign's token budget on "
                f"a backend that is not coming back"
            )

    def to_record(self) -> Record:
        return {
            "max_cycles": self.max_cycles,
            "stop_on_first_improvement": self.stop_on_first_improvement,
            "min_improvement": self.min_improvement,
            "max_consecutive_malformed_plans": self.max_consecutive_malformed_plans,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "max_proposer_retries": self.max_proposer_retries,
            "per_cycle_output_tokens": self.per_cycle_output_tokens,
            "note": (
                "declared before the campaign ran. The controller reads these and never writes "
                "them: a campaign cannot buy itself another cycle by failing to improve"
            ),
        }


# ---------------------------------------------------------------------------
# durable evidence memory
# ---------------------------------------------------------------------------
class ExperimentHistory:
    """Append-only JSONL of every attempt this campaign ever made.

    It is the campaign's memory and its resume point at once. Nothing is ever
    rewritten or removed: a crashed attempt, a refused plan and a
    non-improvement are all evidence, and a history that kept only the winners
    would make the nomination unreadable and the cycle budget unenforceable.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> Record:
        payload = dict(record)
        payload.setdefault("contract_version", CONTROLLER_CONTRACT_VERSION)
        payload.setdefault("recorded_at", _now())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return payload

    def records(self) -> list[Record]:
        if not self.path.is_file():
            return []
        out: list[Record] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line is a crashed writer, not a reason to
                # lose the rest of the campaign's memory.
                continue
            if isinstance(loaded, dict):
                out.append(loaded)
        return out

    def cycles_completed(self) -> int:
        """Cycles spent. An infrastructure retry of a counted cycle is not one."""
        return sum(1 for r in self.records() if r.get("counts_as_cycle", True))

    def consumed(self) -> Record:
        """Summed ``actual_cost`` across every attempt.

        ``None`` is unknown telemetry and stays distinguishable from a measured
        zero: a resource that was never observed reports ``null`` with its
        unobserved-attempt count, not a total that silently treats it as free.
        """
        totals: dict[str, float] = {}
        unknown: dict[str, int] = {}
        for record in self.records():
            for resource, amount in dict(record.get("actual_cost") or {}).items():
                if amount is None:
                    unknown[resource] = unknown.get(resource, 0) + 1
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    totals[resource] = round(totals.get(resource, 0.0) + float(amount), 9)
        return {
            "measured": totals,
            "unknown_telemetry_attempts": unknown,
            "note": "a null amount was never observed; it is not a zero and is not summed",
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# structured feedback — what the agent is told before it proposes
# ---------------------------------------------------------------------------
#: Where a rejected ``ExperimentPlan`` field's accepting values come from.
#:
#: Keyed by the *last dotted segment* of the field a :class:`ContractError`
#: names, because ``validate_plan`` reports ``ExperimentPlan.kind`` while the
#: proposal schema calls the same thing ``kind``.
_FIELD_VOCABULARY = {
    "kind": "kind_values",
    "primary_track": "primary_track_values",
    "action": "action_values",
    "requested_budget": "declarable_budget_resources",
}


def plan_contract(allowed_actions: Sequence[str] = CONTROLLER_ACTIONS) -> Record:
    """The closed vocabularies of :class:`~..contracts.ExperimentPlan`, verbatim.

    Every list here is **read** out of the contract enums or out of
    :data:`CAP_FACTORS`, never retyped, so a tenth ``ArtifactKind`` or a renamed
    ledger resource reaches the proposer without anyone remembering to update a
    string.

    It exists because the first three real campaigns lost three of their six
    cycles to vocabulary the agent had no way to read:

    * ``step11-pilot`` c1 — ``[INVALID_ENUM] ExperimentPlan.kind`` for an
      invented value. ``action_catalogue`` publishes a per-action
      ``artifact_kinds`` list, which is a different field and could not answer
      the question.
    * ``step11-pilot-b`` c2 — ``[BUDGET_UNDECLARED]`` for
      ``requested_budget = {"cycles": 1}``. ``cycles`` is a campaign stop-rule
      concept that the feedback record itself displays; nothing declares a cap
      for it and the ledger is right to refuse it.

    Neither was a scientific error. Both were this controller failing to state
    its own contract, and a proposer that has to guess an enum is being graded
    on something other than its science.
    """
    return {
        "plan_fields_are_supplied_by": (
            "the campaign supplies experiment_id, campaign_id, parent_id and development_metric; "
            "every field below is the proposal's own"
        ),
        "kind_values": [k.value for k in ArtifactKind],
        "primary_track_values": [t.value for t in Track],
        "action_values": sorted(str(a) for a in allowed_actions),
        "declarable_budget_resources": sorted({resource for resource, _f in CAP_FACTORS.values()}),
        "not_budget_resources": ["cycles"],
        "notes": {
            "kind": (
                "ExperimentPlan.kind is one of kind_values and nothing else -- and it must ALSO "
                "appear in the chosen action's 'artifact_kinds' list in action_catalogue, "
                "because validate_plan refuses a kind the action cannot produce. The two lists "
                "are different fields and a plan has to satisfy both"
            ),
            "requested_budget": (
                "requested_budget keys come from declarable_budget_resources, and only those for "
                "which remaining.resources shows a cap on this campaign. 'cycles' is how many "
                "experiment cycles are left, not something that can be spent"
            ),
            "expected_outputs": (
                "relative file paths this experiment writes into its own output directory, each "
                "checked for existence afterwards; a description of a file is not a file"
            ),
        },
    }


def binding_constraints(
    history: Sequence[Mapping[str, Any]],
    *,
    allowed_actions: Sequence[str] = CONTROLLER_ACTIONS,
) -> list[Record]:
    """Every plan rejection in *history*, as a named prohibition on the next plan.

    The difference this draws is the one Step 11 is named for. A rejection that
    appears only in ``prior_hypotheses`` is *narrative*: the proposer reads it
    alongside twenty other facts and may or may not act on it.
    ``step11-pilot-b`` is the measured case — cycle 1 failed, cycle 2 was told
    so, and cycle 2 then produced a plan-validity error of its own.

    A prohibition is different in kind: a stable ``(error_category, field)``
    pair, the values that field will accept, and how many times the campaign
    has already spent a cycle on it. Nothing here rejects anything — the
    admission gate is still ``validate_plan`` — and nothing here is a novelty
    check. A repeated *hypothesis* remains a legitimate replication; a repeated
    *contract error* is a cycle the campaign pays for and learns nothing from.
    """
    contract = plan_contract(allowed_actions)
    found: dict[tuple[str, str], Record] = {}
    for record in history:
        if str(record.get("outcome") or "") != CycleOutcome.PLAN_REJECTED.value:
            continue
        category = str(record.get("error_category") or "")
        if not category:
            continue
        field = str(record.get("error_field") or "")
        key = (category, field)
        entry = found.get(key)
        if entry is None:
            vocabulary = _FIELD_VOCABULARY.get(field.rsplit(".", 1)[-1])
            entry = {
                "error_category": category,
                "field": field,
                "first_seen_cycle": record.get("cycle"),
                "last_seen_cycle": record.get("cycle"),
                "occurrences": 0,
                "rejected_reason": str(record.get("reason") or "")[:800],
                "allowed_values": list(contract[vocabulary]) if vocabulary else None,
                "prohibition": (
                    f"a plan that repeats {category} on {field or 'this plan'} is refused again "
                    f"and costs another cycle. Do not resend that shape"
                    + (
                        f"; {field.rsplit('.', 1)[-1]} must be one of allowed_values below"
                        if vocabulary
                        else ""
                    )
                ),
            }
            found[key] = entry
        entry["occurrences"] = int(entry["occurrences"]) + 1
        entry["last_seen_cycle"] = record.get("cycle")
    return sorted(found.values(), key=lambda e: (str(e["error_category"]), str(e["field"])))


def structured_feedback(
    *,
    campaign: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    incumbent: Mapping[str, Any],
    metric: str,
    remaining: Mapping[str, Any],
    references: Mapping[str, Any] | None = None,
    semantics: Mapping[str, Any] | None = None,
    cycle: int,
    min_improvement: float | None = None,
    capability_notes: Sequence[str] = (),
    allowed_actions: Sequence[str] = CONTROLLER_ACTIONS,
    probe_capabilities: bool = False,
) -> Record:
    """Everything the proposer is given, and nothing it must not see.

    Counts, categories, hashes and the campaign's own numbers. No held-out
    label, no raw prompt, no path under the protected registry: Appendix A §6
    says to report counts and hashes rather than values, and this record is the
    one thing in the loop that is handed to a model.

    Negative findings are carried verbatim rather than summarised. A controller
    that told the agent only about its successes would have it re-propose every
    failure, and the failures are the cheapest information the campaign owns.

    *semantics* is the measurement-semantics record from
    :func:`measurement_semantics`, and it is optional for one reason: this
    function must stay usable with no repository on disk. Given ``None`` the
    record is byte-for-byte what it has always been. Given the record, three
    keys are ADDED and none is changed — what every number means, the auxiliary
    measurements that govern the hypotheses a proposer actually writes, and the
    rules for reading them. A campaign proposed a hypothesis on "AUC 0.74" out
    of a 13,731-character feedback record that contained no AUC at all; the
    quantity that governed it, a rank correlation of +0.078 against a perfect
    value of -1.0, was never in the record. These keys are that omission,
    closed.

    *probe_capabilities* is forwarded to :func:`action_catalogue` and carries
    that function's cost, not a semantic choice: ``True`` measures the host live
    (MEASURED at 5.76s warm, 29.1s cold, because the probe fine-tunes the real
    prior) and ``False`` reads the committed Step 8 probe record, labelled
    ``measured_live: false``. It defaults to ``False`` so that describing the
    feedback record is cheap; :meth:`TargetCampaignController._feedback` — the
    one path that actually hands this record to a model — passes ``True``, so a
    running campaign's agent is told exactly what it was told before.
    """
    prior: list[Record] = []
    failure_categories: dict[str, int] = {}
    negatives: list[Record] = []
    seen_hypotheses: dict[str, int] = {}
    data_additions: list[Record] = []
    for record in history:
        key = _hypothesis_key(str(record.get("hypothesis") or ""))
        seen_hypotheses[key] = seen_hypotheses.get(key, 0) + 1
        entry = {
            "experiment_id": record.get("experiment_id"),
            "cycle": record.get("cycle"),
            "hypothesis": record.get("hypothesis"),
            "action": record.get("action"),
            "backend": record.get("backend"),
            "changed_factors": list(record.get("changed_factors") or []),
            "outcome": record.get("outcome"),
            "status": record.get("status"),
            "failure_category": record.get("failure_category"),
            "reason": record.get("reason"),
            "development_metric_value": record.get("development_metric_value"),
            "improvement_vs_incumbent": record.get("improvement"),
            "evaluated_partition": record.get("evaluated_partition"),
            "actual_cost": record.get("actual_cost"),
            "replication_of": record.get("replication_of"),
        }
        prior.append(entry)
        category = record.get("failure_category")
        if category and category != FailureCategory.NONE.value:
            failure_categories[str(category)] = failure_categories.get(str(category), 0) + 1
        if record.get("outcome") in (
            CycleOutcome.NO_IMPROVEMENT.value,
            CycleOutcome.EXECUTION_FAILURE.value,
            CycleOutcome.PLAN_REJECTED.value,
            CycleOutcome.UNSCORED.value,
        ):
            negatives.append(
                {
                    "experiment_id": record.get("experiment_id"),
                    "hypothesis": record.get("hypothesis"),
                    "outcome": record.get("outcome"),
                    "reason": record.get("reason"),
                    "development_metric_value": record.get("development_metric_value"),
                    "note": "a negative finding, preserved verbatim; do not re-derive it by re-running it",
                }
            )
        for artifact in record.get("data_additions") or []:
            data_additions.append(dict(artifact))

    measured = [
        float(r["development_metric_value"]) for r in history if r.get("development_metric_value") is not None
    ]
    # Always non-empty: how weak the evidence is, is itself evidence, and a
    # feedback record that omitted it would let the agent read one point
    # estimate as a settled result. Caller-supplied references (the frozen
    # label-shuffle p, the trivial-predictor reference) override nothing here;
    # they are added to it.
    uncertainty: Record = {
        "n_independent_development_measurements": len(measured),
        "measured_range": [min(measured), max(measured)] if measured else None,
        "replications_per_hypothesis": dict(seen_hypotheses),
        "interval_available": False,
        "min_improvement_required": min_improvement,
        "note": (
            "every value above is ONE point estimate on ONE development pool. No repeat-run "
            "interval exists, so nothing here can support an IMPROVED claim: a nomination is a "
            "development-partition selection, not a demonstrated improvement."
        ),
        **dict(references or {}),
    }

    feedback: Record = {
        "contract_version": CONTROLLER_CONTRACT_VERSION,
        "cycle": cycle,
        "campaign_id": campaign.get("campaign_id"),
        "target_id": campaign.get("target_id"),
        "track": campaign.get("track"),
        "resource_regime": campaign.get("resource_regime"),
        "metric": metric,
        "orientation": METRIC_ORIENTATION.get(metric, "UNKNOWN"),
        "incumbent": dict(incumbent),
        "allowed_actions": list(allowed_actions),
        "action_catalogue": action_catalogue(allowed_actions, probe=bool(probe_capabilities)),
        "plan_contract": plan_contract(allowed_actions),
        "scoring_contract": scoring_contract(metric, allowed_actions),
        "binding_constraints": binding_constraints(history, allowed_actions=allowed_actions),
        "prior_hypotheses": prior,
        "hypothesis_repetitions": seen_hypotheses,
        "failure_categories": failure_categories,
        "negative_findings": negatives,
        "data_additions": data_additions,
        "remaining": dict(remaining),
        "uncertainty": uncertainty,
        "capability_notes": list(capability_notes),
        "replication_policy": (
            "A repeated hypothesis is a legitimate replication and is dispatched. Novelty is not "
            "an admission criterion here; state plainly when a proposal is a replication and why."
        ),
        "evidence_policy": (
            "Every number above was computed by the campaign's frozen evaluator on the DEVELOPMENT "
            "partition. The held-out partition is not readable from this loop and no action can "
            "reach it. A null metric is a measurement that was never made, never a zero."
        ),
    }
    if semantics is not None:
        feedback["metric_semantics"] = metric_semantics(metric, semantics=semantics, incumbent=incumbent)
        feedback["auxiliary_measurements"] = list(semantics.get("auxiliary_measurements") or [])
        feedback["reading_rules"] = list(semantics.get("reading_rules") or [])
    return feedback


#: What each registered intervention needs in ``inputs``, and what it writes.
#:
#: Transcribed by hand from the handler bodies in
#: :mod:`drug_agent.selfevolve.target.actions` — deliberately, and with a test
#: (`test_every_registered_action_tells_the_agent_what_its_inputs_must_look_like`)
#: that fails the moment a registered action has no entry here. ``ActionSpec``
#: would be the natural home, but ``worker.py`` is inside the evaluator's
#: integrity manifest and adding a field to it would move the frozen evaluator's
#: content address for a documentation string.
#:
#: It exists because the first real campaigns lost cycles to shapes the agent
#: could not have known: ``distill`` proposed with ``inputs.teacher`` as a
#: string (the handler does ``dict(...)`` on it and raised ``ValueError``), and
#: an ``expected_outputs`` entry written as a sentence instead of a filename.
#: An intervention nobody documents is an intervention nobody can propose.
ACTION_INPUT_CONTRACTS: dict[str, Record] = {
    "fit_predictor": {
        "scores": "YES",
        "inputs": {
            "backend": "'sklearn_rf' or 'chemprop'",
            "preparation": {
                "stratum_id": "the campaign's frozen stratum id, a string",
                "endpoint_label": "e.g. 'pKi'",
                "seed": "int",
                "n_estimators / max_depth": "numbers (sklearn_rf)",
                "epochs / batch_size / val_fraction": "numbers (chemprop)",
            },
        },
        "writes": ["predictions.json", "model.joblib (sklearn_rf) or chemprop/ (chemprop)"],
        "expected_outputs_example": ["predictions.json"],
        "scored": "yes - predictions.json is scored on the development pool by the frozen evaluator",
    },
    "develop_algorithm": {
        "scores": "CONDITIONAL",
        "inputs": {
            "backend": "'claude_code' (an experiment worker writes the module) or 'none'",
            "command": "ARGUMENT VECTOR, e.g. ['{python}', '-m', '<module>', '--output-dir', '{output}']",
            "module_to_write": "the single module path the worker may create",
            "preparation": "as for fit_predictor; passed through to your module",
        },
        "writes": ["whatever your module writes; write predictions.json to be scored"],
        "expected_outputs_example": ["predictions.json"],
        "scored": "yes, if your module writes predictions.json",
    },
    "develop_architecture": {
        "scores": "CONDITIONAL",
        "inputs": {
            "backend": "'claude_code' or 'none'",
            "command": "ARGUMENT VECTOR, as for develop_algorithm",
            "module_to_write": "the single module path the worker may create",
        },
        "writes": ["whatever your module writes; write predictions.json to be scored"],
        "expected_outputs_example": ["predictions.json"],
        "scored": "yes, if your module writes predictions.json",
    },
    "distill": {
        "scores": "NO",
        "inputs": {
            "backend": "'sklearn_gbr' (default), 'sklearn_rf' or 'sklearn_ridge'",
            "teacher": {
                "endpoint": "REQUIRED string naming the COMPUTATIONAL endpoint the student learns",
                "table": (
                    "REQUIRED path to a teacher table written by the teacher pass, e.g. "
                    "'.external/research/zak-c9r/derived/distill/<id>/teacher_table.json'. "
                    "This action LEARNS from teacher rows and never produces them"
                ),
            },
            "synthetic_weight": (
                "number, default 1.0. The per-row loss weight on GENERATED_STRUCTURE rows; "
                "anchor rows are always 1.0. 0.0 is the no-distillation control and removes the "
                "synthetic rows rather than zero-weighting them"
            ),
            "holdout_fraction / max_per_family / seed / n_estimators / max_depth": "numbers",
        },
        "writes": ["student.joblib", "student_card.json", "training_rows.json", "distillation.json"],
        "expected_outputs_example": ["student.joblib", "distillation.json"],
        "scored": (
            "NO - a student learns a named computational endpoint and writes no predictions.json, "
            "so this action cannot produce a development mae_pactivity by itself"
        ),
        "note": (
            "'teacher' is an OBJECT with both keys. A string here is a candidate error. "
            "TWO COMPUTATIONS, ON PURPOSE: the teacher table is built by a separate scoring pass "
            "(`python -m drug_agent.selfevolve.target.distill --stage select` then `--stage "
            "teacher`), and this action consumes it. A table that is absent, or whose rows do "
            "not carry teacher/checkpoint, molecule/pose/receptor, evaluator, experiment and "
            "data-partition identity, is refused with that command in the unblock. A teacher "
            "prediction labelled as experimental evidence is refused outright"
        ),
    },
    "evaluate_development": {
        "scores": "YES",
        "inputs": {
            "backend": "'target_benchmark'",
            "predictions": "REQUIRED path to an existing predictions.json produced by an earlier experiment",
            "preparation": "as for fit_predictor",
        },
        "writes": ["predictions.json", "endpoint.json"],
        "expected_outputs_example": ["predictions.json", "endpoint.json"],
        "scored": "yes - it re-scores an existing prediction file",
    },
    "acquire_resource": {
        "scores": "NO",
        "inputs": {
            "backend": "'staged_local'",
            "resource_regime": "'CONTROLLED' on this host",
            "resource": {
                "identifier / path / source / license / trust_label": "ALL REQUIRED",
                "uri": "must be ABSENT - this worker does not reach the network",
            },
        },
        "writes": ["acquisition.json", "acquired_<identifier>"],
        "expected_outputs_example": ["acquisition.json"],
        "scored": (
            "NO - staging data writes no predictions.json, so this action cannot produce a "
            "development metric by itself. The file must already be on disk"
        ),
    },
    "adapt_generator": {
        "scores": "NO",
        # ``training`` is MANDATORY and its absence is an outright FAILED
        # (actions.py: "the one action whose defaults are not allowed to stand in
        # for a proposal"). It was omitted from this contract while the handler
        # required it, so a proposer reading only the catalogue wrote plans that
        # could not run. The exclusion registry is deliberately absent: it is
        # fixed at PROTECTED_REGISTRY_RELATIVE and cannot be named by a plan.
        "inputs": {
            "backend": "'pfm' or 'reinvent4'",
            "training": {
                "partition": "string, default 'train' - NEVER the held-out partition, which is refused",
                "epochs": "int, default 10",
                "batch_size": "int, default 64",
                "device": "optional string",
                "input_model": "optional path to the prior to start from (alias: input_model_file)",
                "output_model": "optional path for the checkpoint (alias: checkpoint)",
            },
        },
        # A successful run writes four things, not one. Advertising only the
        # probe understated the action and left the checkpoint - the thing the
        # next experiment needs - undocumented.
        "writes": [
            "capability_probe.json",
            "training_smiles.smi",
            "adaptation.json",
            "generator/<backend>_tl.prior",
        ],
        "expected_outputs_example": ["capability_probe.json", "adaptation.json"],
        "hand_off": (
            "On success the result notes carry CHECKPOINT WRITTEN: <absolute path>. Pass that "
            "path VERBATIM as inputs.sampling.checkpoint of a later change_sampling experiment. "
            "Do not reconstruct it from the experiment id and do not pass a description of where "
            "it can be found - change_sampling refuses both. These weights change nothing until "
            "a change_sampling experiment names them"
        ),
        # This string used to end "and no backend on this host exposes a
        # training verb, so it answers UNAVAILABLE". That was a *hardcoded*
        # claim about the host sitting in the one document the agent reads
        # before it chooses, and it went on being served after the capability
        # it describes changed. What stays here is the part that is true by
        # construction — this handler writes a probe record, not a
        # predictions.json — and the host-dependent half now travels as a
        # measurement, in ``measured_capability``, produced by
        # :func:`measured_training_capability`.
        "scored": (
            "NO - it writes capability_probe.json and never a predictions.json, so it cannot "
            "move the development metric in the cycle that runs it. Whether the training verb "
            "it probes exists on this host is a measurement, not a constant: read "
            "measured_capability on this catalogue entry"
        ),
    },
    "change_sampling": {
        "scores": "TRACK_B_ONLY",
        # Four keys the handler reads were missing from this contract:
        # n_generated, checkpoint, selection and max_runtime_seconds. The worst
        # was ``checkpoint`` - it is the entire transfer-learning hand-off, and
        # six of eleven refused cycles on disk passed a DESCRIPTION of the path
        # instead of the path, because the contract never said the key existed.
        "inputs": {
            "backend": "'pfm' or 'reinvent4'",
            "sampling": {
                "n_molecules": "int - the SUBMISSION budget; the metric is an absolute count at this",
                "n_generated": (
                    "int, default n_molecules - how many to GENERATE before selecting. Must be "
                    ">= n_molecules. Generating more than you submit is how selection gets "
                    "something to choose between; it is bounded by the run's generation ceiling"
                ),
                "selection": (
                    "'generation_order' (default) | 'cheap_gate_prefilter' | 'surrogate_rank'. "
                    "Which n_molecules of the generated pool to submit. Docking-free gates only: "
                    "selecting on structure_score or geometry is refused, because those need a "
                    "pose and selecting on them is a lookup, not a prediction"
                ),
                "checkpoint": (
                    "optional ABSOLUTE path to a prior written by a previous adapt_generator "
                    "experiment - copy it verbatim from that result's CHECKPOINT WRITTEN note. "
                    "Omit it to sample the parent weights"
                ),
                "seed": "int, default 42",
                "batch_size": "int, default n_molecules",
                "max_runtime_seconds": "float, default 900",
                "pocket": "path (pfm)",
            },
        },
        "writes": ["molecules.json", "funnel.json"],
        "expected_outputs_example": ["molecules.json"],
        "scored": "Track B only - molecules.json is scored as qualifying_scaffolds, not as mae_pactivity",
    },
    "prepare_receptor": {
        "scores": "NO",
        "inputs": {
            "backend": "'pocket_cut'",
            "receptor": {"pdb": "path", "ligand_comp_id": "string", "chain": "string", "radius": "float"},
        },
        "writes": ["receptor_pocket.pdb", "preparation.json"],
        "expected_outputs_example": ["receptor_pocket.pdb"],
        "scored": "NO - a pocket is an input to a later experiment, not a scored artifact",
    },
}


def scoring_contract(metric: str, allowed_actions: Sequence[str] = CONTROLLER_ACTIONS) -> Record:
    """Which of the permitted actions can put a number on the board, and which cannot.

    Every real campaign so far spent its cycles on actions that cannot produce a
    development metric at all — ``acquire_resource`` twice, ``distill`` twice —
    and each time for a *sound* reason: the agent diagnosed data scarcity and
    tried to add data. The diagnosis was right. What it could not see is that
    this platform has no single action that both manufactures teacher rows and
    learns from them, so that plan needs two cycles and the campaign had two.

    ``action_catalogue`` already carried that fact in prose (``"scored": "NO -
    staging data writes no predictions.json"``). This states it as structure, at
    the top level, next to the number of cycles left. It narrows nothing: an
    action marked ``NO`` is still a legitimate proposal and is still dispatched.
    It is the difference between a campaign that hides its own success condition
    and one that states it.
    """
    buckets: dict[str, list[str]] = {"YES": [], "CONDITIONAL": [], "NO": [], "TRACK_B_ONLY": []}
    for name in sorted(str(a) for a in allowed_actions):
        verdict = str((ACTION_INPUT_CONTRACTS.get(name) or {}).get("scores") or "NO")
        buckets.setdefault(verdict, []).append(name)
    return {
        "development_metric": metric,
        "orientation": METRIC_ORIENTATION.get(metric, "UNKNOWN"),
        "computed_by": (
            "the parent checkout's frozen evaluator, on the DEVELOPMENT partition, from the "
            "predictions.json the experiment leaves behind, after the experiment process exits"
        ),
        "actions_that_produce_it": list(buckets["YES"]),
        "actions_that_produce_it_if_your_module_writes_predictions": list(buckets["CONDITIONAL"]),
        "actions_that_cannot_produce_it": sorted(buckets["NO"] + buckets["TRACK_B_ONLY"]),
        "note": (
            "an action in actions_that_cannot_produce_it is a legitimate proposal and will be "
            "dispatched; it just cannot move the metric in the cycle that runs it. A two-step "
            "plan needs two cycles, and remaining.cycles says how many there are"
        ),
    }


#: The actions whose catalogue entry carries a measured capability record.
#:
#: Exactly one, and deliberately. Every probe reachable from here costs
#: something real — ``molecule.generate.reinvent4``/``fit`` fine-tunes the prior
#: and was MEASURED at 29.1s cold here, ``property.predict.chemprop``/``fit``
#: really trains — so probing the whole menu would put a training run inside the
#: act of describing it, on every cycle. ``adapt_generator`` is the one action
#: whose menu entry was *lying*, so it is the one that gets measured, once per
#: process (see :data:`_CAPABILITY_CACHE`).
#:
#: An action absent from this tuple carries no ``measured_capability`` key at
#: all. That is the point: an absent measurement must never be readable as an
#: available one.
MEASURED_CAPABILITY_ACTIONS: tuple[str, ...] = ("adapt_generator",)

#: One live probe per action per process, not one per cycle.
#:
#: MEASURED, and the reason this cache exists rather than being an optimisation:
#: ``probe_capability("molecule.generate.reinvent4", "fit")`` took **29.1
#: seconds** cold and **5.5 seconds** warm on this host, and returned
#: ``AVAILABLE_EXECUTED`` — because that probe no longer inspects for a method
#: name, it fine-tunes the real prior and checks that the checkpoint differs
#: from the one it started from. Re-running that on every feedback packet would
#: spend a campaign's budget describing its own menu. A capability does not
#: change inside one campaign process, so measuring it once is not a weaker
#: answer than measuring it repeatedly — and the second render measured 0.000s.
_CAPABILITY_CACHE: dict[tuple[str, bool], Record] = {}


def measured_training_capability(action: str = "adapt_generator", *, probe: bool = True) -> Record:
    """Ask the host whether *action*'s declared capabilities are there.

    This exists because the menu used to *assert* the answer. The catalogue
    entry for ``adapt_generator`` carried the sentence "no backend on this host
    exposes a training verb, so it answers UNAVAILABLE" as a literal string, and
    that sentence was served to the proposer on every cycle regardless of what
    any probe had measured. A controller that tells the agent an action is dead
    is steering it away from that action, and it goes on steering after the
    action starts working — which is no longer hypothetical here: the
    ``molecule.generate.reinvent4``/``fit`` probe now reports
    ``AVAILABLE_EXECUTED``.

    The measurement itself is :func:`~.worker.action_support`, the registered
    API for this question, rather than a second aggregation rule written here
    that could disagree with the dispatcher about what "available" means. It
    reads the pairs from the action's own ``ActionSpec``, so a backend added or
    removed there needs no edit here.

    Args:
        action: the registered intervention to ask about.
        probe: ``True`` runs the probes live and pays their cost once per
            process; ``False`` reads the committed Step 8 probe record instead,
            which is cheap and is a measurement taken earlier on this host — and
            which is stale exactly when it matters, so it is not the default.

    Returns a record whose ``availability_on_this_host`` is one of the support
    tokens, or ``"NOT_MEASURED"`` when the machinery could not be reached at
    all. ``NOT_MEASURED`` is never a synonym for available.
    """
    key = (str(action), bool(probe))
    if key in _CAPABILITY_CACHE:
        return dict(_CAPABILITY_CACHE[key])

    measured: Record
    try:
        from .worker import action_support

        support = action_support(str(action), probe=bool(probe))
        measured = {
            "action": str(action),
            "availability_on_this_host": str(support.get("support") or NOT_MEASURED),
            "backends_available": list(support.get("backends_available") or []),
            "backends_unprobed": list(support.get("backends_unprobed") or []),
            "evidence_class": support.get("evidence_class"),
            "reason": support.get("reason"),
            "unblock": support.get("unblock"),
            "capabilities": [
                {
                    "capability_id": row.get("capability_id"),
                    "verb": row.get("verb"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "unblock": row.get("unblock"),
                    "probed_now": row.get("probed_now"),
                }
                for row in (support.get("capabilities") or [])
                if isinstance(row, Mapping)
            ],
        }
    except Exception as exc:
        # A record-keeping helper must never be able to stop a campaign. An
        # unreachable probe is an unmeasured capability, which is not an
        # available one and is not an unavailable one either.
        measured = {
            "action": str(action),
            "availability_on_this_host": NOT_MEASURED,
            "backends_available": [],
            "backends_unprobed": [],
            "evidence_class": "not measured",
            "reason": f"the capability probe could not be reached: {type(exc).__name__}: {exc}",
            "unblock": "read the reason above; a probe that cannot run is an infrastructure finding",
            "capabilities": [],
        }

    measured["measured_by"] = "drug_agent.selfevolve.target.worker.action_support"
    measured["measured_live"] = bool(probe)
    measured["note"] = (
        "measured on this host, once per campaign process and cached, not transcribed from a "
        "sentence in the menu. Whatever it says, this action is still a legitimate proposal: it "
        "is dispatched and, if a verb is missing, answered with a CAPABILITY_UNAVAILABLE result "
        "carrying its unblock. It is never refused at the door"
    )
    _CAPABILITY_CACHE[key] = measured
    return dict(measured)


def action_catalogue(
    allowed_actions: Sequence[str] = CONTROLLER_ACTIONS, *, probe: bool = False
) -> list[Record]:
    """What each permitted intervention is and what it needs, from the registry.

    Derived from Step 10's :data:`~.worker.ACTIONS` rather than retyped, so an
    action whose backends change cannot go on being described the old way to the
    one component that has to write plans against it.

    Entries named in :data:`MEASURED_CAPABILITY_ACTIONS` also carry a
    ``measured_capability`` record, rather than a sentence about the host baked
    into :data:`ACTION_INPUT_CONTRACTS`.

    ``probe`` decides *how* that record is obtained, and it defaults to the
    cheap side ON PURPOSE. ``probe=True`` measures the host live and pays the
    real cost once per process: ``molecule.generate.reinvent4``/``fit`` is not
    an inspection, it fine-tunes the prior, and it was MEASURED here at 5.76s
    warm and 29.1s cold *inside this function*. ``probe=False`` reads the
    committed Step 8 probe record instead — a measurement taken earlier on this
    host, labelled ``measured_live: false`` so it can never be misread as a
    live one.

    The default is ``False`` because the overwhelming majority of callers are
    describing the menu, not running a campaign: two test modules call
    ``action_catalogue()`` with no arguments purely to read ``inputs_contract``
    and ``artifact_kinds``, and :func:`~.ablation.history_suppression_diff`
    builds the feedback record twice to diff its fields while documenting that
    it "consumes no budget, contacts no model and reads no partition". None of
    them looks at ``measured_capability`` at all, and a GPU fine-tune inside a
    menu-description function is a cost none of them asked for.

    The campaign path is the caller that *does* need the live answer, and it
    asks for it explicitly: :meth:`TargetCampaignController._feedback` passes
    ``probe_capabilities=True`` through :func:`structured_feedback`. What a
    running campaign's agent is told is therefore unchanged by this default.
    """
    from .worker import ACTIONS

    out: list[Record] = []
    for name in allowed_actions:
        spec = ACTIONS.get(str(name))
        if spec is None:  # pragma: no cover - _check_menu refuses these first
            continue
        entry: Record = {
            "action": spec.action,
            "summary": spec.summary,
            "backends": sorted(spec.backends),
            "default_backend": spec.default_backend,
            "artifact_kinds": list(spec.artifact_kinds),
            "requires_capabilities": {
                backend: [list(pair) for pair in pairs] for backend, pairs in spec.backends.items()
            },
            "worker_authored": spec.worker_authored,
            **{
                k: v
                for k, v in dict(ACTION_INPUT_CONTRACTS.get(spec.action) or {}).items()
                if k != "inputs"
            },
            "inputs_contract": dict(ACTION_INPUT_CONTRACTS.get(spec.action) or {}).get("inputs"),
            "note": (
                "this action runs code an experiment worker writes, so inputs.command must "
                "declare the ARGUMENT VECTOR that runs it"
                if spec.worker_authored
                else "this action runs a fixed adapter this repository already ships"
            ),
        }
        if spec.action in MEASURED_CAPABILITY_ACTIONS:
            entry["measured_capability"] = measured_training_capability(
                spec.action, probe=bool(probe)
            )
        out.append(entry)
    return out


def _hypothesis_key(text: str) -> str:
    return " ".join(str(text).lower().split())


# ---------------------------------------------------------------------------
# the proposer
# ---------------------------------------------------------------------------
class Proposer(Protocol):
    """Whatever decides what to try next, given the campaign's evidence."""

    name: str

    def propose(self, feedback: Record) -> Record:  # pragma: no cover - protocol
        ...


#: What an ``expected_outputs`` entry may look like: a relative path of ordinary
#: filename segments. No spaces, no absolute path, no ``..``.
EXPECTED_OUTPUT_PATTERN = r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$"


def proposal_schema(allowed_actions: Sequence[str] = CONTROLLER_ACTIONS) -> dict[str, Any]:
    """The JSON the reasoning model is asked for, with the closed vocabularies closed.

    Deliberately *not* a whole ``ExperimentPlan``: identity fields (experiment
    id, campaign id, the metric) are the campaign's, not the agent's, and
    letting a model write them would let it rename an experiment into a
    different campaign's budget.

    ``action``, ``kind`` and ``primary_track`` are enumerated here rather than
    left as free strings. The first real campaign produced a correct diagnosis
    ("three structurally different learners all fail to beat the trivial median
    and the label-shuffle test is non-significant, so the bottleneck is data
    scarcity, not model family") and then lost the whole cycle because it wrote
    ``kind = "data_scarcity_probe"``. The vocabulary was in the prose and in
    ``action_catalogue``; the CLI enforces ``--json-schema``, so it belongs
    here. Refusing a well-reasoned proposal on a spelling is an interface
    defect, not a scientific result.

    The prose ``maxLength`` numbers below are an INTERFACE CAP ON PROSE LENGTH,
    NOT A SCIENTIFIC THRESHOLD. No gate, metric, budget or nomination reads the
    length of ``diagnosis``, ``hypothesis`` or a ``changed_factors`` entry;
    they exist only to keep one response inside the call's output-token cap. A
    26-campaign run measured what the old numbers cost: of 56 rejected
    ``StructuredOutput`` attempts, 40 were rejected for ``/diagnosis`` exceeding
    its 1200-character maximum (observed lengths 1206-2376) and 8 for a
    ``/changed_factors/0`` over 120, and the CLI's retry budget of five is a
    hard wall — five rejections lose the whole cycle. Raising the caps refuses
    fewer correct proposals and changes no measurement, exactly the argument
    this docstring already makes above about ``kind``.
    """
    return {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string", "maxLength": 4000},
            "hypothesis": {"type": "string", "maxLength": 4000},
            "action": {"type": "string", "enum": sorted(str(a) for a in allowed_actions)},
            "kind": {"type": "string", "enum": [k.value for k in ArtifactKind]},
            "primary_track": {"type": "string", "enum": [t.value for t in Track]},
            "changed_factors": {"type": "array", "items": {"type": "string", "maxLength": 400}},
            "inputs": {"type": "object"},
            # A relative path inside the experiment's own output directory, and
            # nothing else. The first real campaign dispatched an experiment whose
            # expected output was the sentence "acquisition.json provenance record
            # (source, licence, trust_label, sha256, measurement_relation) for the
            # staged CHEMBL3886 activities dump"; the handler wrote
            # acquisition.json, the sentence was never a file, and a real run was
            # recorded as an execution failure over a naming convention.
            "expected_outputs": {
                "type": "array",
                "items": {"type": "string", "pattern": EXPECTED_OUTPUT_PATTERN, "maxLength": 120},
            },
            "requested_budget": {"type": "object"},
            "stop_rule": {"type": "string", "maxLength": 300},
            "is_replication": {"type": "boolean"},
            "expected_evidence": {"type": "string", "maxLength": 400},
        },
        # Ordered short structural fields FIRST, long prose LAST. The required
        # set is identical — this is presentation order inside the schema the
        # CLI shows the model, not a change to what is required. A model that
        # runs out of room writes the tail of the object badly, and the measured
        # census says the tail is where the losses were: 45 of 56 rejected
        # attempts were missing 'hypothesis' and 32 were missing 'action', both
        # of which used to sit behind 1200 characters of 'diagnosis'. Cheap to
        # satisfy, and it enters no measurement.
        "required": [
            "action",
            "kind",
            "changed_factors",
            "inputs",
            "expected_outputs",
            "requested_budget",
            "stop_rule",
            "hypothesis",
            "diagnosis",
        ],
    }


#: The unconstrained-menu schema, for a caller that has no campaign to hand.
PROPOSAL_SCHEMA: dict[str, Any] = proposal_schema()

PROPOSER_SYSTEM = """\
You are the scientific controller of a bounded, autonomous drug-discovery
evolution campaign against one protein target. You are given the campaign's own
measured evidence: every prior hypothesis, what it produced, why it failed, what
it cost, and how much budget remains.

Diagnose the measured bottleneck, then propose ONE bounded next experiment from
the campaign's action menu. Rules that are not negotiable:

* Propose only an action in `allowed_actions`. Nothing else exists.
* `binding_constraints` are PROHIBITIONS, not history. Each one names an
  (error_category, field) pair a previous cycle was already refused for, and the
  values that field accepts. A proposal that repeats one is refused again and
  costs a whole cycle. Obey every one of them.
* `scoring_contract` says which actions can produce the development number at
  all. An action it lists as unable to produce one is still a legitimate
  proposal -- it simply cannot move the metric in the cycle that runs it, and a
  two-step plan needs two of the cycles `remaining.cycles` reports.
* `plan_contract` holds the closed vocabularies: `kind` is one of
  `plan_contract.kind_values` and nothing else (it is NOT the per-action
  `artifact_kinds` list), and `requested_budget` keys come only from
  `plan_contract.declarable_budget_resources`.
* The number that decides anything is computed by a frozen evaluator outside
  your experiment, on the DEVELOPMENT partition. Never propose to compute,
  report or optimise a score yourself, and never propose to touch the evaluator,
  the benchmark records, the baselines or any held-out data.
* A repeated hypothesis is a legitimate replication. If the best next move is to
  repeat an earlier experiment, say so and set is_replication.
* NO_IMPROVEMENT is a complete, publishable result. Do not propose an experiment
  whose only justification is that the campaign has not yet found a win.
* `requested_budget` uses ONLY the resource names listed under
  `remaining.resources`, and must fit inside those amounts. `remaining.cycles`
  is NOT a resource. A name nobody declared a cap for is refused, and an
  experiment that cannot be paid for is refused before it launches.
* `expected_outputs` are RELATIVE FILE PATHS the experiment writes into its own
  output directory -- e.g. ["predictions.json"], ["molecules.json"] -- never a
  description of a file. Every one of them is checked for existence afterwards,
  and a name that is not a file is recorded as a failed experiment.
* `inputs.command`, if present, is an ARGUMENT VECTOR (a list of strings), never
  a shell string.
* Every number in `metric_semantics` and `auxiliary_measurements` carries a
  `perfect_value`, a `no_skill_value` and an `interval`. A number whose interval
  contains its `no_skill_value` is INDISTINGUISHABLE FROM NO SKILL, whatever its
  point estimate looks like. Read `governs` and `does_not_govern` before citing a
  measurement: a quantity that governs discrimination between two classes does
  NOT govern regression within one, and citing it for the wrong question has
  already cost this campaign a cycle.
* IF A QUANTITY IS NOT IN THIS RECORD, YOU WERE NOT GIVEN IT AND IT IS NOT
  EVIDENCE. Do not cite a number you did not receive, however confident you are
  of it. You have no tools and read no files; a number that is not in the record
  above came from you, not from a measurement, and it is a fabrication even when
  it happens to be close to right.
* If `metric_semantics.falsifiable` is false, say so in `diagnosis` and do not
  propose an experiment whose only purpose is to move that metric. A perfect
  agent and a broken agent score the same on a metric that cannot be falsified.

BE BRIEF. `diagnosis` and `hypothesis` are at most four sentences each; every
other string is one line. The whole object must fit inside this call's output
token cap -- a response that overruns it is lost entirely, the campaign is
charged for it, and no experiment runs. Do not restate the evidence you were
given, do not write a plan narrative, and do not think out loud outside the JSON.

Reply with the JSON object only.
"""


#: What a failed reasoning call is allowed to leave behind in the campaign
#: record. An ALLOWLIST, deliberately: the backend's ``raw`` also carries the
#: child's ``stdout``, and a proposal's inputs can name molecules and assay
#: rows, so prompts and conversation are recorded as a content address and a
#: length elsewhere in this module and in ``worker.py`` and must not be copied
#: here. These keys are error metadata — process status, CLI subtype, turn
#: count, the provider's error status, permission refusals, the resolved model
#: identity and the child's stderr — and none of them is a proposal's text.
PROPOSER_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "returncode",
    "subtype",
    "num_turns",
    "duration_api_ms",
    "api_error_status",
    "permission_denials",
    "stderr",
    "model_requested",
    "model_resolved",
    "model_usage",
)

#: Longest ``stderr`` excerpt kept. An interface cap on how much diagnostic text
#: a history line carries; it enters no measurement.
PROPOSER_DIAGNOSTIC_STDERR_CHARS = 1000


def _proposer_diagnostics(raw: Any) -> Record | None:
    """The failure metadata from a backend result, with the conversation removed."""
    if not isinstance(raw, Mapping):
        return None
    out: Record = {}
    for key in PROPOSER_DIAGNOSTIC_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "stderr" and isinstance(value, str):
            value = value[:PROPOSER_DIAGNOSTIC_STDERR_CHARS]
        out[key] = value
    if not out:
        return None
    out["note"] = (
        "error metadata only. The prompt and the model's prose are deliberately not here: a "
        "proposal's inputs can name molecules and assay rows, and this module records prompts "
        "as a content address and a length"
    )
    return out


@dataclass
class ClaudeProposer:
    """The real reasoning call: Claude, through the installed CLI.

    A thin adapter over :class:`~..backends.ClaudeCLIBackend`, which already
    reserves this call's output-token cap on the campaign ledger *before* the
    subprocess starts and settles what the CLI reported afterwards. Nothing here
    re-implements that accounting.

    The model's answer is a *proposal*, not a plan: the controller supplies the
    campaign's identity fields and then puts the whole thing through
    ``validate_plan``. A proposal that names an unregistered action, a protected
    path or a shell string is refused there and recorded as a cycle — which is
    the behaviour under test, not an error path to be smoothed over.
    """

    backend: Any
    name: str = "claude_cli"
    model: str | None = None
    max_output_tokens: int = 8192
    timeout_s: float = 600.0
    system: str = PROPOSER_SYSTEM
    #: Narrows the schema's ``action`` enum to this campaign's menu. Taken from
    #: the feedback record when the caller leaves it unset, so the proposer and
    #: the controller cannot disagree about what may be proposed.
    allowed_actions: tuple[str, ...] | None = None

    def propose(self, feedback: Record) -> Record:
        menu = self.allowed_actions or tuple(feedback.get("allowed_actions") or CONTROLLER_ACTIONS)
        result = self.backend.complete(
            system=self.system,
            user=json.dumps(feedback, indent=2, sort_keys=True, default=str),
            role="deep_researcher",
            schema=proposal_schema(menu),
            model=self.model,
            timeout_s=self.timeout_s,
            max_output_tokens=self.max_output_tokens,
        )
        usage = {
            "backend": getattr(result, "backend", "") or self.name,
            "model": getattr(result, "model", "") or (self.model or ""),
            "output_tokens": getattr(result, "output_tokens", None),
            "input_tokens": getattr(result, "input_tokens", None),
            "cost_estimate_usd": getattr(result, "cost_estimate_usd", None),
            "session_id": getattr(result, "session_id", None),
            "elapsed_s": getattr(result, "elapsed_s", None),
            "ok": bool(getattr(result, "ok", False)),
            "error": getattr(result, "error", None),
            # Why the call failed, not what it was asked. A blocked campaign
            # used to record the string "exit 1" and nothing else, because the
            # backend's ``raw`` diagnostics were dropped here; diagnosing the
            # 26-campaign run needed CLI transcripts from outside the campaign
            # record, which nothing guarantees will survive.
            "diagnostics": _proposer_diagnostics(getattr(result, "raw", None)),
        }
        if not getattr(result, "ok", False):
            raise ProposerUnavailable(str(getattr(result, "error", "") or "the reasoning call failed"), usage)
        parsed = getattr(result, "parsed", None)
        if not isinstance(parsed, Mapping):
            raise MalformedProposal(
                "the reasoning call returned no JSON object matching the proposal schema", usage
            )
        proposal = dict(parsed)
        proposal["usage"] = usage
        proposal["evidence_class"] = "measured"
        return proposal


class ProposerUnavailable(RuntimeError):
    """The reasoning backend could not serve. Recoverable by waiting."""

    def __init__(self, message: str, usage: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.usage: Record = dict(usage or {})


class MalformedProposal(ValueError):
    """The reasoning backend answered with something that is not a proposal."""

    def __init__(self, message: str, usage: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.usage: Record = dict(usage or {})


# ---------------------------------------------------------------------------
# nomination
# ---------------------------------------------------------------------------
def _improvement(metric: str, value: float, incumbent: float) -> float | None:
    """Improvement over the incumbent, positive meaning better, or ``None``.

    The sign convention is ``Comparison``'s and is load-bearing: an MAE
    *reduction* of 0.09 is ``+0.09``. A metric with no declared direction gets
    ``None`` rather than a guess — a comparison whose direction was assumed is
    not a comparison.
    """
    orientation = METRIC_ORIENTATION.get(metric)
    if orientation == "LOWER_IS_BETTER":
        return incumbent - value
    if orientation == "HIGHER_IS_BETTER":
        return value - incumbent
    return None


def build_candidate_pool(
    history: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    incumbent: Mapping[str, Any],
    min_improvement: float,
) -> list[Record]:
    """Every attempt the campaign made, with its eligibility decided and stated.

    The complete pool, in cycle order — failures and non-improvements included.
    Eligibility is recorded *with its reason* on every entry, so a reader can
    see why each candidate was or was not nominable without re-deriving it.
    """
    incumbent_value = incumbent.get("value")
    nondispatch = {CycleOutcome.PROPOSER_UNAVAILABLE.value, CycleOutcome.CONTROLLER_CRASH.value}
    pool: list[Record] = []
    for record in history:
        if str(record.get("outcome") or "") in nondispatch:
            # An attempt, kept in the history, but never a candidate: there is
            # no experiment behind it to nominate.
            continue
        artifacts = list(record.get("artifacts") or [])
        value = record.get("development_metric_value")
        improvement = record.get("improvement")
        reason: str | None = None
        if str(record.get("status") or "") != ExperimentStatus.SUCCEEDED.value:
            reason = (
                f"the run did not succeed ({record.get('failure_category')}); a failed experiment "
                f"has a missing scientific value, not a bad one"
            )
        elif not artifacts:
            reason = "the run produced no persisted artifact, so there is nothing to nominate"
        elif value is None:
            reason = (
                "no independently measured development value came back; unknown is not a score "
                "and cannot be ranked"
            )
        elif str(record.get("evaluated_partition") or "") != "development":
            reason = (
                f"the measurement's partition is {record.get('evaluated_partition')!r}, not "
                f"'development'; nomination is a development-only decision"
            )
        elif incumbent_value is None:
            reason = "no incumbent value was declared, so no improvement can be established"
        elif improvement is None:
            reason = f"metric {metric!r} has no declared direction, so improvement is undefined"
        elif float(improvement) <= float(min_improvement):
            reason = (
                f"improvement {float(improvement):.6f} does not exceed the declared minimum "
                f"{float(min_improvement):.6f}"
            )
        pool.append(
            {
                "cycle": record.get("cycle"),
                "experiment_id": record.get("experiment_id"),
                "artifact_id": artifacts[0] if artifacts else None,
                "artifacts": artifacts,
                "hypothesis": record.get("hypothesis"),
                "action": record.get("action"),
                "outcome": record.get("outcome"),
                "status": record.get("status"),
                "failure_category": record.get("failure_category"),
                "development_metric": metric,
                "development_metric_value": value,
                "evaluated_partition": record.get("evaluated_partition"),
                "improvement": improvement,
                "actual_cost": record.get("actual_cost"),
                "replication_of": record.get("replication_of"),
                "nomination_eligible": reason is None,
                "ineligible_reason": reason,
            }
        )
    return pool


def select_nomination(
    campaign: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> Record:
    """The frozen development rule, applied to the complete pool.

    Returns the chosen entry and a validated :class:`~..contracts.Nomination`,
    or ``None`` with the reason nothing was nominable. The record is built
    through ``validate_record``, so its ``test_access_count = 0`` is checked by
    the contract rather than asserted here.
    """
    eligible = [c for c in pool if c.get("nomination_eligible")]
    if not eligible:
        return {
            "nomination": None,
            "selected": None,
            "rule": DEVELOPMENT_SELECTION_RULE,
            "n_pool": len(pool),
            "n_eligible": 0,
            "reason": (
                "no candidate in the pool carried an independently measured development value "
                "that beat the declared incumbent by the declared minimum"
            ),
        }
    best = max(eligible, key=lambda c: (float(c["improvement"]), -int(c.get("cycle") or 0)))
    nomination = validate_record(
        Nomination,
        {
            "campaign_id": str(campaign["campaign_id"]),
            "artifact_id": str(best["artifact_id"]),
            "protocol_hash": str(campaign["protocol_hash"]),
            "development_selection_rule": DEVELOPMENT_SELECTION_RULE,
            "nominated_at": _now(),
            "test_access_count": 0,
        },
    )
    return {
        "nomination": nomination,
        "selected": dict(best),
        "rule": DEVELOPMENT_SELECTION_RULE,
        "n_pool": len(pool),
        "n_eligible": len(eligible),
        "reason": None,
    }


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------
@dataclass
class _Blocked:
    code: str
    field: str
    detail: str
    unblock: str

    def to_record(self) -> Record:
        return {"code": self.code, "field": self.field, "detail": self.detail, "unblock": self.unblock}


class TargetCampaignController:
    """Runs a bounded target-evolution campaign to a terminal state.

    The one supported way to cause an experiment is Step 10's
    :func:`~.worker.dispatch_plan`, which admits the plan, reserves on the
    campaign ledger, isolates the candidate, verifies its outputs and computes
    the development number with the parent's frozen evaluator. ``dispatch`` is
    injectable only so a deterministic transition test can drive the state
    machine without launching subprocesses; production passes nothing and gets
    the real dispatcher.

    Raises:
        ValueError: the campaign record, the action menu or the stop rule is
            invalid. A configuration error is refused at construction rather
            than discovered halfway through a paid campaign.
    """

    def __init__(
        self,
        campaign: Mapping[str, Any],
        *,
        proposer: Proposer,
        ledger: Any,
        output_root: Path | str,
        allowed_actions: Sequence[str],
        stop_rule: StopRule,
        incumbent: Mapping[str, Any],
        dispatch: Callable[..., Record] | None = None,
        dispatch_kwargs: Mapping[str, Any] | None = None,
        references: Mapping[str, Any] | None = None,
        semantics: Mapping[str, Any] | None = None,
        capability_notes: Sequence[str] = (),
        root: Path | str | None = None,
        campaign_phase: str = "ADAPTATION",
    ) -> None:
        self.campaign = validate_record(Campaign, campaign)
        self.proposer = proposer
        self.ledger = ledger
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.allowed_actions = _check_menu(allowed_actions)
        self.stop_rule = stop_rule
        self.incumbent = dict(incumbent)
        self.dispatch = dispatch if dispatch is not None else dispatch_plan
        self.dispatch_kwargs = dict(dispatch_kwargs or {})
        self.references = dict(references or {})
        # Not `dict(semantics or {})`: absent and empty are different answers
        # here. `structured_feedback` adds the semantics keys only when it is
        # given a record, so None has to survive as None or a controller built
        # without one would hand the proposer three empty blocks that look like
        # measurements nobody made.
        self.semantics = dict(semantics) if semantics is not None else None
        self.capability_notes = list(capability_notes)
        self.root = Path(root) if root is not None else None
        self.campaign_phase = campaign_phase
        self.metric = str(self.incumbent.get("metric") or _campaign_metric(self.campaign))
        self.history = ExperimentHistory(self.output_root / "history.jsonl")
        self.checkpoint_path = self.output_root / "checkpoint.json"
        self.events: list[Record] = []

    # -- entry gate -------------------------------------------------------
    def missing_mandatory_inputs(self) -> _Blocked | None:
        """The prerequisites a cycle cannot run without, checked before spending.

        Each answer names the field and what would unblock it, because "blocked"
        without a cause is indistinguishable from "crashed".
        """
        if self.incumbent.get("value") is None:
            return _Blocked(
                code="MISSING_MANDATORY_INPUT",
                field="incumbent.value",
                detail=(
                    "no incumbent development value was declared, so no cycle could establish "
                    "improvement against anything. An absent comparator is not a comparator of zero"
                ),
                unblock=(
                    "pass incumbent={'artifact_id': …, 'value': <the parent's measured development "
                    "metric>}; for Track A it is the frozen parent RF's mae_pactivity"
                ),
            )
        if self.metric not in METRIC_ORIENTATION:
            return _Blocked(
                code="MISSING_MANDATORY_INPUT",
                field="development_metric",
                detail=(
                    f"no direction is declared for metric {self.metric!r}, so 'better' is undefined "
                    f"and no candidate could ever be nominated"
                ),
                unblock=f"declare {self.metric!r} in METRIC_ORIENTATION, or run a metric that is",
            )
        try:
            caps = dict(self.ledger.caps())
        except Exception as exc:  # a ledger that cannot be read is a missing input
            return _Blocked(
                code="MISSING_MANDATORY_INPUT",
                field="ledger",
                detail=f"the campaign ledger could not be read: {type(exc).__name__}: {exc}",
                unblock="point the campaign at a readable budget ledger",
            )
        missing = [r for r in ("experiment_proposals",) if r not in caps]
        if missing:
            return _Blocked(
                code="MISSING_MANDATORY_INPUT",
                field="ledger.caps",
                detail=(
                    f"the campaign ledger declares no cap for {missing}; an undeclared cap is not "
                    f"an unlimited cap, and an unbounded proposal loop is exactly what this step "
                    f"must not build"
                ),
                unblock=(
                    "declare the campaign's caps on the ledger before the loop starts, e.g. "
                    "ledger.declare('experiment_proposals', 6)"
                ),
            )
        return None

    # -- the loop ---------------------------------------------------------
    def run(self) -> Record:
        """Run to a terminal state and return the whole campaign record."""
        started = time.time()
        blocked = self.missing_mandatory_inputs()
        if blocked is not None:
            return self._finish(
                CampaignState.BLOCKED,
                TerminalReason.MISSING_MANDATORY_INPUT,
                started=started,
                blocked=blocked,
            )

        spent_at_start = self.history.cycles_completed()
        consecutive_malformed = _trailing_malformed(self.history.records())
        #: Consecutive reasoning-backend failures at this point in the loop. Not
        #: a cycle counter: it is reset the moment a call serves, and it buys no
        #: science budget of any kind.
        proposer_attempt = 0
        terminal = TerminalReason.CYCLE_BUDGET_EXHAUSTED
        state = CampaignState.COMPLETED
        blocked_cause: _Blocked | None = None
        recoverable = False

        try:
            while self.history.cycles_completed() < self.stop_rule.max_cycles:
                cycle = self.history.cycles_completed() + 1
                shortfall = self._resource_shortfall()
                if shortfall is not None:
                    self.events.append({"event": "resource_budget_exhausted", "at": _now(), **shortfall})
                    terminal = TerminalReason.RESOURCE_BUDGET_EXHAUSTED
                    break

                feedback = self._feedback(cycle)
                try:
                    proposal = self.proposer.propose(feedback)
                except ProposerUnavailable as exc:
                    # A failed reasoning call used to end the campaign outright.
                    # The measured population it ends is mostly transient — the
                    # CLI's own five-attempt schema-validation budget running
                    # out — and a *fresh* invocation clears it, because this
                    # backend never resumes a session: every ``propose`` is a
                    # new subprocess with a new context, so the retry is not the
                    # poisoned conversation asked a sixth time.
                    retryable = _proposer_failure_is_retryable(exc)
                    is_terminal = not retryable
                    will_retry = retryable and proposer_attempt < self.stop_rule.max_proposer_retries
                    self.events.append(
                        {
                            "event": "proposer_unavailable",
                            "at": _now(),
                            "cycle": cycle,
                            "proposer_attempt": proposer_attempt,
                            "reason": str(exc),
                            "classification": "TERMINAL" if is_terminal else "TRANSIENT",
                            "will_retry": will_retry,
                            "retries_remaining": max(
                                0, self.stop_rule.max_proposer_retries - proposer_attempt
                            )
                            if not is_terminal
                            else 0,
                        }
                    )
                    # Every attempt is written to durable memory before the
                    # decision to retry, so a retried cycle is auditable and a
                    # recovered campaign never hides the call it had to repeat.
                    self._record_nondispatch(
                        cycle,
                        outcome=CycleOutcome.PROPOSER_UNAVAILABLE,
                        reason=str(exc),
                        usage=exc.usage,
                        attempt=proposer_attempt,
                        will_retry=will_retry,
                        classification="TERMINAL" if is_terminal else "TRANSIENT",
                    )
                    if will_retry:
                        proposer_attempt += 1
                        backoff = _proposer_backoff_s(proposer_attempt)
                        self.events.append(
                            {
                                "event": "proposer_retry",
                                "at": _now(),
                                "cycle": cycle,
                                "proposer_attempt": proposer_attempt,
                                "backoff_s": backoff,
                                "note": (
                                    "a fresh reasoning call for the SAME cycle. It spends no "
                                    "additional cycle and no additional experiment budget; the "
                                    "output tokens it costs are charged to the ledger as usual"
                                ),
                            }
                        )
                        _sleep(backoff)
                        continue
                    blocked_cause = _Blocked(
                        code="PROPOSER_UNAVAILABLE",
                        field="proposer",
                        detail=(
                            f"{exc} (after {proposer_attempt + 1} attempt(s); "
                            f"{'terminal, not retried' if is_terminal else 'retry budget spent'})"
                        ),
                        unblock=(
                            "restore the reasoning backend and re-run the same command; the "
                            "history and the ledger are the resume point"
                        ),
                    )
                    state = CampaignState.BLOCKED
                    terminal = TerminalReason.PROPOSER_UNAVAILABLE
                    recoverable = True
                    break
                except MalformedProposal as exc:
                    self._record_plan_rejection(
                        cycle,
                        proposal={"evidence_class": "measured", "usage": exc.usage},
                        reason=str(exc),
                        category=ErrorCategory.INVALID_TYPE.value,
                        error_field="proposal",
                    )
                    consecutive_malformed += 1
                    if consecutive_malformed >= self.stop_rule.max_consecutive_malformed_plans:
                        terminal = TerminalReason.MALFORMED_PLAN_LIMIT
                        break
                    continue

                # The backend served. The retry budget is for *consecutive*
                # outages and is spent only while one is in progress.
                proposer_attempt = 0

                try:
                    plan = self._assemble_plan(cycle, proposal)
                except ContractError as exc:
                    self._record_plan_rejection(
                        cycle,
                        proposal=proposal,
                        reason=str(exc),
                        category=exc.category.value,
                        error_field=str(exc.field or ""),
                    )
                    consecutive_malformed += 1
                    if consecutive_malformed >= self.stop_rule.max_consecutive_malformed_plans:
                        terminal = TerminalReason.MALFORMED_PLAN_LIMIT
                        break
                    continue
                consecutive_malformed = 0

                outcome, exhausted = self._run_cycle(cycle, plan, proposal)
                if exhausted:
                    self.events.append(
                        {
                            "event": "resource_budget_exhausted",
                            "at": _now(),
                            "resource": "unknown",
                            "detail": "the dispatcher refused the reservation",
                        }
                    )
                    terminal = TerminalReason.RESOURCE_BUDGET_EXHAUSTED
                    break
                if (
                    outcome == CycleOutcome.DEVELOPMENT_IMPROVEMENT
                    and self.stop_rule.stop_on_first_improvement
                ):
                    terminal = TerminalReason.NOMINATION_SELECTED
                    break
        except Exception as exc:  # a controller crash is a recorded state, not a traceback
            import traceback

            blocked_cause = _Blocked(
                code="CONTROLLER_CRASH",
                field="controller",
                detail=f"{type(exc).__name__}: {exc}",
                unblock=(
                    "read the traceback in the checkpoint, repair the fault and re-run the same "
                    "command; the campaign resumes from its history and its ledger"
                ),
            )
            self.events.append(
                {
                    "event": "controller_crash",
                    "at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc()[-2000:],
                }
            )
            with contextlib.suppress(Exception):
                self._record_nondispatch(
                    self.history.cycles_completed() + 1,
                    outcome=CycleOutcome.CONTROLLER_CRASH,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            return self._finish(
                CampaignState.FAILED,
                TerminalReason.CONTROLLER_CRASH,
                started=started,
                blocked=blocked_cause,
                recoverable=True,
                cycles_already_spent=spent_at_start,
            )

        return self._finish(
            state,
            terminal,
            started=started,
            blocked=blocked_cause,
            recoverable=recoverable,
            cycles_already_spent=spent_at_start,
        )

    # -- one cycle --------------------------------------------------------
    def _run_cycle(self, cycle: int, plan: Record, proposal: Mapping[str, Any]) -> tuple[str, bool]:
        """Dispatch one admitted plan, retrying only genuine infrastructure faults."""
        attempt = 0
        while True:
            experiment_id = str(plan["experiment_id"])
            output_dir = self.output_root / "experiments" / experiment_id
            kwargs: Record = {
                "allowed_actions": list(self.allowed_actions),
                "output_dir": output_dir,
                "log_dir": output_dir / "logs",
                "ledger": self.ledger,
                "request_key": f"{self.campaign['campaign_id']}:{experiment_id}",
                "campaign_phase": self.campaign_phase,
                **self.dispatch_kwargs,
            }
            if self.root is not None:
                kwargs.setdefault("root", self.root)
            envelope = self.dispatch(plan, **kwargs)
            result = dict(envelope.get("result") or {})
            failure_category = str(result.get("failure_category") or "")
            retryable = (
                failure_category == FailureCategory.INFRASTRUCTURE_ERROR.value
                and attempt < self.stop_rule.max_infrastructure_retries
            )
            record = self._record_dispatch(
                cycle,
                plan=plan,
                proposal=proposal,
                envelope=envelope,
                counts_as_cycle=not retryable,
                attempt=attempt,
            )
            if failure_category == FailureCategory.BUDGET_EXHAUSTED.value:
                return str(record["outcome"]), True
            if not retryable:
                return str(record["outcome"]), False
            attempt += 1
            plan = dict(plan)
            plan["experiment_id"] = f"{experiment_id}r{attempt}"

    # -- record keeping ---------------------------------------------------
    def _record_dispatch(
        self,
        cycle: int,
        *,
        plan: Mapping[str, Any],
        proposal: Mapping[str, Any],
        envelope: Mapping[str, Any],
        counts_as_cycle: bool,
        attempt: int,
    ) -> Record:
        result = dict(envelope.get("result") or {})
        evaluation = dict(envelope.get("independent_evaluation") or {})
        status = str(result.get("status") or "")
        failure_category = str(result.get("failure_category") or "")
        partition = str(evaluation.get("evaluated_partition") or "")
        measured = evaluation.get("status") == "MEASURED" and evaluation.get("value") is not None

        value: float | None = None
        improvement: float | None = None
        outcome: str
        reason = envelope.get("reason")
        if status == ExperimentStatus.SUCCEEDED.value:
            if not measured:
                outcome = CycleOutcome.UNSCORED.value
                reason = reason or (
                    f"the independent evaluator returned {evaluation.get('status')!r} for "
                    f"{self.metric!r}: {evaluation.get('reason')}"
                )
            elif partition != "development":
                outcome = CycleOutcome.UNSCORED.value
                reason = reason or (
                    f"the measurement's partition is {partition!r}, not 'development'. It is "
                    f"recorded, and it cannot decide a nomination"
                )
            else:
                value = float(evaluation["value"])
                improvement = _improvement(self.metric, value, float(self.incumbent["value"]))
                better = improvement is not None and improvement > self.stop_rule.min_improvement
                outcome = (
                    CycleOutcome.DEVELOPMENT_IMPROVEMENT.value
                    if better
                    else CycleOutcome.NO_IMPROVEMENT.value
                )
        elif failure_category == FailureCategory.INFRASTRUCTURE_ERROR.value and not counts_as_cycle:
            outcome = CycleOutcome.INFRASTRUCTURE_RETRY.value
        else:
            outcome = CycleOutcome.EXECUTION_FAILURE.value

        hypothesis = str(plan.get("hypothesis") or "")
        counts = _generation_counts(envelope)
        return self.history.append(
            {
                "cycle": cycle,
                "attempt": attempt,
                "counts_as_cycle": counts_as_cycle,
                "experiment_id": str(plan.get("experiment_id") or result.get("experiment_id") or ""),
                "campaign_id": str(self.campaign["campaign_id"]),
                "hypothesis": hypothesis,
                "action": plan.get("action"),
                "backend": dict(plan.get("inputs") or {}).get("backend"),
                "changed_factors": list(plan.get("changed_factors") or []),
                "outcome": outcome,
                "status": status,
                "failure_category": failure_category,
                "reason": reason,
                "error_category": envelope.get("error_category"),
                "development_metric": self.metric,
                "development_metric_value": value,
                "improvement": improvement,
                "evaluated_partition": partition or None,
                "evaluator_hash": evaluation.get("evaluator_hash"),
                "n_pool": evaluation.get("n_pool"),
                # The generation denominator, beside the numerator it belongs
                # to. Without it a campaign record cannot answer "qualifying
                # scaffolds per molecule generated" at all: before this key,
                # history.jsonl carried zero occurrences of n_generated, n_raw
                # or n_submitted across every campaign on disk, so the funnel
                # was only ever reconstructible from the experiment directory,
                # which the campaign record does not carry.
                #
                # None, never 0. A campaign that generated nothing and a
                # campaign whose generator count was not recorded are different
                # facts, and a zero here would make the second read as the
                # first — and make any rate computed from it look infinite.
                "n_generated": counts["n_generated"],
                "n_submitted": counts["n_submitted"],
                "generation_counts_provenance": counts["provenance"],
                "coverage": evaluation.get("coverage"),
                "pool_hash": evaluation.get("pool_hash"),
                "artifacts": list(result.get("artifacts") or []),
                "actual_cost": dict(result.get("actual_cost") or {}),
                "replication_of": self._replication_of(hypothesis, cycle=cycle),
                "proposal": _proposal_record(proposal),
                "data_additions": _data_additions(envelope),
                "support": dict(envelope.get("support") or {}),
                "worker_claim_is_not_evidence": (
                    "the value above came from envelope['independent_evaluation'], computed by the "
                    "parent's frozen evaluator. The run's own result.json is testimony, never a score"
                ),
            }
        )

    def _record_nondispatch(
        self,
        cycle: int,
        *,
        outcome: CycleOutcome,
        reason: str,
        usage: Mapping[str, Any] | None = None,
        attempt: int = 0,
        will_retry: bool = False,
        classification: str | None = None,
    ) -> Record:
        """A terminal event that never reached a dispatch, kept in the record.

        It does **not** count as a cycle: a backend outage or a controller fault
        must not spend the campaign's declared science budget. But it is an
        attempt, it cost something, and a campaign whose durable memory did not
        mention it would let a resumed run believe nothing had happened.

        ``attempt`` numbers the reasoning call within one stalled cycle, so a
        campaign that recovered on a retry says so in its own history rather
        than looking like a campaign that never stumbled.
        """
        spent = dict(usage or {}).get("output_tokens")
        base_id = f"{self._experiment_id(cycle)}_nondispatch"
        return self.history.append(
            {
                "cycle": cycle,
                "attempt": attempt,
                "proposer_attempt": attempt,
                "will_retry": bool(will_retry),
                "failure_classification": classification,
                "counts_as_cycle": False,
                "experiment_id": base_id if attempt == 0 else f"{base_id}_r{attempt}",
                "campaign_id": str(self.campaign["campaign_id"]),
                "hypothesis": "",
                "action": None,
                "outcome": outcome.value,
                "status": None,
                "failure_category": None,
                "reason": reason,
                "development_metric": self.metric,
                "development_metric_value": None,
                "improvement": None,
                "evaluated_partition": None,
                "artifacts": [],
                "actual_cost": {
                    "attempts": 0.0,
                    "wall_s": 0.0,
                    "llm_output_tokens": float(spent) if isinstance(spent, (int, float)) else None,
                },
                "replication_of": None,
                "proposal": {"evidence_class": "measured", "usage": dict(usage or {})},
                "note": (
                    "nothing was dispatched, so this is not a candidate and does not appear in the "
                    "candidate pool. It is an attempt, and attempts are preserved"
                    + (
                        f". A fresh reasoning call followed this one (retry {attempt + 1}); it "
                        f"runs the SAME cycle and buys no science budget"
                        if will_retry
                        else ""
                    )
                ),
            }
        )

    def _record_plan_rejection(
        self,
        cycle: int,
        *,
        proposal: Mapping[str, Any],
        reason: str,
        category: str,
        error_field: str,
    ) -> Record:
        """A refused proposal is a spent cycle with no experiment behind it.

        It costs the reasoning call that produced it — which the backend already
        charged to the ledger — and no experiment budget, because nothing
        launched. Recorded rather than retried silently.
        """
        hypothesis = str(proposal.get("hypothesis") or "")
        usage = dict(proposal.get("usage") or {})
        tokens = usage.get("output_tokens")
        return self.history.append(
            {
                "cycle": cycle,
                "attempt": 0,
                "counts_as_cycle": True,
                "experiment_id": self._experiment_id(cycle),
                "campaign_id": str(self.campaign["campaign_id"]),
                "hypothesis": hypothesis,
                "action": proposal.get("action"),
                "backend": dict(proposal.get("inputs") or {}).get("backend"),
                "changed_factors": list(proposal.get("changed_factors") or []),
                "outcome": CycleOutcome.PLAN_REJECTED.value,
                "status": None,
                "failure_category": None,
                "reason": reason,
                "error_category": category,
                "error_field": error_field,
                "development_metric": self.metric,
                "development_metric_value": None,
                "improvement": None,
                "evaluated_partition": None,
                "artifacts": [],
                "actual_cost": {
                    "attempts": 0.0,
                    "wall_s": 0.0,
                    "llm_output_tokens": float(tokens) if isinstance(tokens, (int, float)) else None,
                },
                "replication_of": self._replication_of(hypothesis, cycle=cycle),
                "proposal": _proposal_record(proposal),
                "note": (
                    "the plan never launched, so no experiment budget was reserved; the reasoning "
                    "call that produced it was charged by the backend that made it"
                ),
            }
        )

    # -- helpers ----------------------------------------------------------
    def _experiment_id(self, cycle: int) -> str:
        return f"exp_{self.campaign['campaign_id']}_c{cycle:02d}"

    def _replication_of(self, hypothesis: str, *, cycle: int) -> str | None:
        """The earliest *earlier cycle* that stated the same hypothesis, if any.

        Recorded, never acted on: a repeated hypothesis is a legitimate
        replication and is dispatched like any other. The flag exists so the
        agent can see it is repeating itself and say why, and so a reader can
        tell a deliberate replication from an accidental one.

        A retry of the *same* cycle is not a replication of itself, so earlier
        attempts of this cycle are skipped.
        """
        key = _hypothesis_key(hypothesis)
        if not key:
            return None
        for record in self.history.records():
            if int(record.get("cycle") or 0) >= cycle:
                continue
            if _hypothesis_key(str(record.get("hypothesis") or "")) == key:
                return str(record.get("experiment_id") or "") or None
        return None

    def _feedback(self, cycle: int) -> Record:
        return structured_feedback(
            campaign=self.campaign,
            history=self.history.records(),
            incumbent=self.incumbent,
            metric=self.metric,
            remaining=self._remaining(cycle),
            references=self.references,
            semantics=self.semantics,
            cycle=cycle,
            min_improvement=self.stop_rule.min_improvement,
            capability_notes=self.capability_notes,
            allowed_actions=self.allowed_actions,
            # The campaign is the caller that hands this record to a model, so
            # it is the caller that pays for the live host measurement. Every
            # other caller (tests describing the menu, the ablation field-diff)
            # takes the committed Step 8 probe record by default and runs no
            # fine-tune. See action_catalogue's `probe`.
            probe_capabilities=True,
        )

    def _remaining(self, cycle: int) -> Record:
        """Cycles left, and — separately — what the ledger will still pay for.

        The two are separate keys because they used to be siblings: the first
        real campaign read ``remaining.cycles`` as a spendable resource and
        proposed ``requested_budget = {"cycles": 1}``, which nothing declares a
        cap for, so the plan was refused. ``resources`` is exactly the set of
        names a ``requested_budget`` may use.
        """
        resources: Record = {}
        for resource in sorted(self.ledger.caps()):
            with contextlib.suppress(Exception):
                resources[resource] = self.ledger.usage(resource)["available"]
        return {
            "cycles": self.stop_rule.max_cycles - (cycle - 1),
            "resources": resources,
            "note": (
                "'cycles' is how many experiment cycles this campaign may still run and is NOT a "
                "budget resource. Only the names under 'resources' may appear in requested_budget"
            ),
        }

    def _resource_shortfall(self) -> Record | None:
        """Is there enough left to pay for another cycle? Checked before reasoning.

        A cycle whose experiment could never be reserved must not spend a
        reasoning call first: Step 10 measured 20,899 output tokens for one
        worker session, so the wasted call is not cheap.
        """
        caps = dict(self.ledger.caps())
        wanted: list[tuple[str, float]] = [("experiment_proposals", 1.0)]
        if self.stop_rule.per_cycle_output_tokens is not None:
            wanted.append(("llm_output_tokens", float(self.stop_rule.per_cycle_output_tokens)))
        for resource, need in wanted:
            if resource not in caps:
                continue
            usage = self.ledger.usage(resource)
            available = usage.get("available")
            if available is None:
                continue
            if float(available) + 1e-9 < need:
                return {
                    "resource": resource,
                    "needed": need,
                    "available": float(available),
                    "limit": usage.get("limit"),
                    "detail": (
                        f"{resource}: {available} of {usage.get('limit')} remains and a cycle needs "
                        f"{need}. The campaign stops here rather than starting work it cannot pay for"
                    ),
                }
        return None

    def _assemble_plan(self, cycle: int, proposal: Mapping[str, Any]) -> Record:
        """Turn one proposal into an admitted ``ExperimentPlan``.

        Identity is the campaign's and science is the agent's: the experiment
        id, the campaign id and the development metric are supplied here so a
        proposal cannot rename itself into another campaign's budget or change
        what it will be scored on. Everything that decides *what is tried* comes
        from the proposal and is then put through Step 10's ``validate_plan``,
        which is the only admission gate.
        """
        from .api import validate_plan

        inputs = dict(proposal.get("inputs") or {})
        plan = {
            "experiment_id": self._experiment_id(cycle),
            "campaign_id": str(self.campaign["campaign_id"]),
            "hypothesis": str(proposal.get("hypothesis") or ""),
            "kind": str(proposal.get("kind") or ""),
            "primary_track": str(proposal.get("primary_track") or self.campaign["track"]),
            "parent_id": str(proposal.get("parent_id") or self.campaign["parent_artifact_id"]),
            "action": str(proposal.get("action") or ""),
            "inputs": inputs,
            "expected_outputs": list(proposal.get("expected_outputs") or []),
            "changed_factors": list(proposal.get("changed_factors") or []),
            "requested_budget": dict(proposal.get("requested_budget") or {}),
            "development_metric": self.metric,
            "stop_rule": str(proposal.get("stop_rule") or ""),
        }
        validation = validate_plan(plan, list(self.allowed_actions))
        self._require_declared_resources(validation["plan"])
        return dict(validation["plan"])

    def _require_declared_resources(self, plan: Mapping[str, Any]) -> None:
        """Refuse a plan that asks for a resource this campaign never declared.

        ``BudgetLedger.reserve`` raises ``BUDGET_UNDECLARED`` for an unknown
        resource, and ``dispatch_plan`` catches only ``BudgetExceeded`` — so
        before this check the first real campaign's second proposal, which asked
        for ``requested_budget = {"cycles": 1}``, took the whole controller down
        as a crash. It is a bad plan, not a broken controller, and it belongs in
        the history as one.

        The cap rule is untouched and nothing is declared on the plan's behalf:
        an undeclared cap is still not a cap. The check is here rather than in
        the dispatcher because the controller is what owns the ledger.
        """
        caps = set(self.ledger.caps())
        requested = dict(plan.get("requested_budget") or {})
        undeclared = sorted(str(r) for r in requested if str(r) not in caps)
        if not undeclared:
            return
        raise ContractError(
            ErrorCategory.BUDGET_UNDECLARED,
            f"requested_budget names {undeclared}, for which this campaign's ledger declares no "
            f"cap; it declares {sorted(caps)}. An undeclared cap is not an unlimited cap, so the "
            f"plan is refused rather than scheduled against nothing",
            contract="ExperimentPlan",
            field="requested_budget",
            missing=undeclared,
        )

    def _finish(
        self,
        state: CampaignState,
        terminal: TerminalReason,
        *,
        started: float,
        blocked: _Blocked | None = None,
        recoverable: bool = False,
        cycles_already_spent: int = 0,
    ) -> Record:
        history = self.history.records()
        pool = build_candidate_pool(
            history,
            metric=self.metric,
            incumbent=self.incumbent,
            min_improvement=self.stop_rule.min_improvement,
        )
        selection = select_nomination(self.campaign, pool, metric=self.metric)
        nomination = selection["nomination"]
        if state is CampaignState.COMPLETED and nomination is not None:
            state = CampaignState.NOMINATED
            if terminal is TerminalReason.CYCLE_BUDGET_EXHAUSTED:
                terminal = TerminalReason.NOMINATION_SELECTED
        decision = _scientific_decision(state, history, nomination)
        record: Record = {
            "contract_version": CONTROLLER_CONTRACT_VERSION,
            "campaign_id": str(self.campaign["campaign_id"]),
            "target_id": str(self.campaign["target_id"]),
            "track": str(self.campaign["track"]),
            "development_metric": self.metric,
            "orientation": METRIC_ORIENTATION.get(self.metric, "UNKNOWN"),
            "state": state.value,
            "terminal_reason": terminal.value,
            "scientific_decision": None if decision is None else decision.value,
            "recoverable": recoverable,
            "blocked_cause": blocked.to_record() if blocked is not None else None,
            "incumbent": dict(self.incumbent),
            "stop_rule": self.stop_rule.to_record(),
            "cycles_already_spent": cycles_already_spent,
            "cycles_completed": sum(1 for r in history if r.get("counts_as_cycle", True)),
            "attempts": len(history),
            "candidate_pool": pool,
            "selection": {k: v for k, v in selection.items() if k != "nomination"},
            "nomination": nomination,
            "consumed": self.history.consumed(),
            "ledger": _ledger_summary(self.ledger),
            "events": list(self.events),
            "history_path": str(self.history.path),
            "checkpoint_path": str(self.checkpoint_path),
            "output_root": str(self.output_root),
            "wall_s": round(time.time() - started, 3),
            "finished_at": _now(),
            "exit_code": campaign_exit_code(state.value),
            "claim_note": (
                "A nomination is a DEVELOPMENT-partition selection made before any held-out result "
                "exists. It is not a claim of improvement: Comparison requires an uncertainty "
                "interval excluding no improvement, and one point estimate has none."
            ),
        }
        self._write_checkpoint(record)
        summary = self.output_root / "campaign.json"
        summary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return record

    def _write_checkpoint(self, record: Mapping[str, Any]) -> Path:
        payload = {
            "contract_version": CONTROLLER_CONTRACT_VERSION,
            "campaign_id": record["campaign_id"],
            "state": record["state"],
            "terminal_reason": record["terminal_reason"],
            "recoverable": record["recoverable"],
            "cycles_completed": record["cycles_completed"],
            "max_cycles": self.stop_rule.max_cycles,
            "history_path": str(self.history.path),
            "output_root": str(self.output_root),
            "ledger_path": str(getattr(self.ledger, "path", "")),
            "blocked_cause": record["blocked_cause"],
            "events": record["events"],
            "written_at": _now(),
            "resume_command": (
                "re-run the same campaign command. The history and the ledger are the state: "
                "already-spent cycles are counted against the same max_cycles and an "
                "already-settled request key is refused rather than charged twice"
            ),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        # `checkpoint.json` is the resume point and is overwritten; the JSONL
        # beside it is the log and is not. The first real campaign stopped
        # BLOCKED and the resumed run overwrote the only record of why.
        with (self.output_root / "checkpoints.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return self.checkpoint_path


# ---------------------------------------------------------------------------
# module-private helpers
# ---------------------------------------------------------------------------
def _check_menu(allowed_actions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(allowed_actions, (str, bytes)) or not isinstance(allowed_actions, Iterable):
        raise ValueError("allowed_actions is a list of action names")
    menu = [str(a) for a in allowed_actions]
    if not menu:
        raise ValueError("an empty action menu is a stopped campaign, not an unrestricted one")
    unknown = sorted({a for a in menu if a not in CONTROLLER_ACTIONS})
    if unknown:
        leaky = sorted(a for a in unknown if any(word in a.lower() for word in FINAL_TEST_WORDS))
        detail = (
            f" {leaky} name the frozen final-test surface, which is absent from this loop by "
            f"construction: no action here reads the held-out partition."
            if leaky
            else ""
        )
        raise ValueError(
            f"the campaign menu names {unknown}, which this controller cannot dispatch; the "
            f"registered interventions are {list(CONTROLLER_ACTIONS)}.{detail}"
        )
    return tuple(menu)


def _campaign_metric(campaign: Mapping[str, Any]) -> str:
    track = str(campaign.get("track") or "")
    return "qualifying_scaffolds" if track.startswith("B") or "STRUCTURE_ONLY" in track else "mae_pactivity"


def _trailing_malformed(history: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for record in reversed(list(history)):
        if record.get("outcome") == CycleOutcome.PLAN_REJECTED.value:
            count += 1
            continue
        break
    return count


#: The one source below that is a file on disk rather than a record in the
#: dispatch envelope. Spelled as a constant so the provenance string a campaign
#: row carries says plainly that this number was read out of the experiment
#: directory, not handed over by the dispatcher.
_FUNNEL_FILE = "output_dir/funnel.json"

#: Where a generation denominator may be found in a dispatch envelope, in
#: preference order, as ``(dotted path, key)``.
#:
#: ``n_generated`` is the canonical key and is looked for first everywhere it
#: could appear. ``n_raw`` is the pre-existing spelling the generation handlers
#: already write (``actions.change_sampling`` and the worker-authored sampling
#: modules both emit it), and it is accepted as a fallback *only* because the
#: recorded ``generation_counts_provenance`` says which one was read — a
#: campaign record that silently blended two different definitions would be
#: worse than one with a null.
_GENERATED_SOURCES: tuple[tuple[str, str], ...] = (
    ("independent_evaluation", "n_generated"),
    ("independent_evaluation.portfolio", "n_generated"),
    ("independent_evaluation.portfolio.funnel", "n_generated"),
    ("result.evidence", "n_generated"),
    ("worker_claim.evidence", "n_generated"),
    (_FUNNEL_FILE, "n_generated"),
    ("independent_evaluation.portfolio.funnel", "n_raw"),
    ("result.evidence", "n_raw"),
    ("worker_claim.evidence", "n_raw"),
    (_FUNNEL_FILE, "n_raw"),
)

#: Where the *submitted* count may be found, same convention. The evaluator's
#: own portfolio funnel is consulted first: that number is computed by the
#: frozen evaluator from the molecules file, so it is a measurement, where the
#: worker's own evidence block is testimony.
_SUBMITTED_SOURCES: tuple[tuple[str, str], ...] = (
    ("independent_evaluation.portfolio.funnel", "n_submitted"),
    ("independent_evaluation.portfolio", "n_evaluated"),
    ("result.evidence", "n_submitted"),
    ("worker_claim.evidence", "n_submitted"),
    (_FUNNEL_FILE, "n_submitted"),
    ("worker_claim.evidence", "n_unique"),
)


def _dig(envelope: Mapping[str, Any], path: str, cache: dict[str, Any]) -> Mapping[str, Any] | None:
    """The mapping at *path*, or ``None``. ``path`` may name the funnel file.

    The funnel file is consulted last and only because it has to be: the
    handlers write ``n_raw`` (and, once the canonical key lands, ``n_generated``)
    into ``funnel.json``, and several dispatches on disk carry that number
    nowhere else — the worker-authored sampling modules put it in the file but
    not in their claimed ``evidence`` block. Reading only the envelope would
    have made this whole field a permanent ``null`` for exactly the campaigns
    that generated the most molecules, which is a field that silently does
    nothing.
    """
    if path == _FUNNEL_FILE:
        if _FUNNEL_FILE not in cache:
            cache[_FUNNEL_FILE] = _read_funnel_file(envelope)
        loaded = cache[_FUNNEL_FILE]
        return loaded if isinstance(loaded, Mapping) else None
    current: Any = envelope
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current if isinstance(current, Mapping) else None


def _read_funnel_file(envelope: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """``<output_dir>/funnel.json``, or ``None`` for any reason at all.

    Read-only, and every failure is an absence rather than an exception: a
    record-keeping helper must not be able to fail a dispatch that already ran.
    """
    output_dir = envelope.get("output_dir")
    if not output_dir:
        return None
    try:
        path = Path(str(output_dir)) / "funnel.json"
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return loaded if isinstance(loaded, Mapping) else None


def _count_from(envelope: Mapping[str, Any], sources: Sequence[tuple[str, str]]) -> tuple[int | None, str]:
    """The first integer one of *sources* actually carries, and where it came from.

    An absent key, a null, a bool and a non-integral number all return
    ``(None, ...)``. Nothing here invents a zero: a count nobody wrote is
    unknown, and unknown is not none-generated.
    """
    cache: dict[str, Any] = {}
    for path, key in sources:
        holder = _dig(envelope, path, cache)
        if holder is None or key not in holder:
            continue
        value = holder.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not value.is_integer():
            continue
        return int(value), f"{path}.{key}"
    return None, NOT_MEASURED


def _generation_counts(envelope: Mapping[str, Any]) -> Record:
    """The generation funnel's two denominators, if this dispatch recorded them.

    Written defensively on purpose: the canonical ``n_generated`` key is being
    added to ``funnel.json`` separately, and this must record ``null`` rather
    than fail or guess while it is absent. ``provenance`` names the exact record
    and key each number was read from, so a reader can tell an evaluator-
    computed count from the worker's own testimony without trusting this
    function's preference order.
    """
    generated, generated_from = _count_from(envelope, _GENERATED_SOURCES)
    submitted, submitted_from = _count_from(envelope, _SUBMITTED_SOURCES)
    return {
        "n_generated": generated,
        "n_submitted": submitted,
        "provenance": {
            "n_generated": generated_from,
            "n_submitted": submitted_from,
            "note": (
                "a path under independent_evaluation is the parent's frozen evaluator speaking. "
                f"One under worker_claim, result.evidence or {_FUNNEL_FILE} is the run's own "
                "testimony about how much it generated, which nothing here independently "
                f"verifies. {NOT_MEASURED} means no record this dispatch left behind carried the "
                "count, and it is not a zero"
            ),
        },
    }


def _proposal_record(proposal: Mapping[str, Any]) -> Record:
    """The proposal as evidence: what was asked for and what it cost.

    The model's own text is kept — it is the campaign's reasoning record — but
    the *prompt* it was given is not written here. Appendix A §6: counts and
    hashes, not values.
    """
    return {
        "evidence_class": str(proposal.get("evidence_class") or "measured"),
        "diagnosis": proposal.get("diagnosis"),
        "hypothesis": proposal.get("hypothesis"),
        "action": proposal.get("action"),
        "is_replication": bool(proposal.get("is_replication") or False),
        "expected_evidence": proposal.get("expected_evidence"),
        "usage": dict(proposal.get("usage") or {}),
    }


def _data_additions(envelope: Mapping[str, Any]) -> list[Record]:
    """Resources an experiment staged, so the next proposal knows they exist."""
    out: list[Record] = []
    for record in envelope.get("artifact_records") or []:
        if str(dict(record).get("kind") or "") == "DATA_OR_LIBRARY":
            out.append(
                {
                    "artifact_id": dict(record).get("artifact_id"),
                    "kind": "DATA_OR_LIBRARY",
                    "experiment_id": dict(record).get("experiment_id"),
                }
            )
    return out


def _ledger_summary(ledger: Any) -> Record:
    try:
        caps = dict(ledger.caps())
        return {
            "path": str(getattr(ledger, "path", "")),
            "usage": {resource: ledger.usage(resource) for resource in sorted(caps)},
        }
    except Exception as exc:  # a ledger that cannot be summarised is reported, not guessed
        return {"error": f"{type(exc).__name__}: {exc}"}


def _scientific_decision(
    state: CampaignState,
    history: Sequence[Mapping[str, Any]],
    nomination: Mapping[str, Any] | None,
) -> ScientificDecision | None:
    """The campaign's scientific outcome. Never ``IMPROVED``, and that is deliberate.

    ``Comparison`` admits ``IMPROVED`` only with an uncertainty interval whose
    lower bound excludes no improvement. A campaign nominates on a single
    development point estimate per candidate, which has no interval, so the
    strongest honest answer it can give is ``INCONCLUSIVE`` — the paired
    comparison with an interval is the confirmation step's work, not this one's.
    """
    if state in (CampaignState.BLOCKED, CampaignState.FAILED):
        return None
    if nomination is not None:
        return ScientificDecision.INCONCLUSIVE
    valid = [r for r in history if r.get("outcome") == CycleOutcome.NO_IMPROVEMENT.value]
    if valid:
        return ScientificDecision.NO_IMPROVEMENT
    return ScientificDecision.INSUFFICIENT_DATA


__all__ = [
    "ACTION_INPUT_CONTRACTS",
    "CAP_FACTORS",
    "CONTROLLER_ACTIONS",
    "CONTROLLER_CONTRACT_VERSION",
    "DEVELOPMENT_SELECTION_RULE",
    "FINAL_TEST_WORDS",
    "MEASURED_CAPABILITY_ACTIONS",
    "MEASUREMENT_SEMANTICS",
    "METRIC_ORIENTATION",
    "NOT_MEASURED",
    "EXPECTED_OUTPUT_PATTERN",
    "MAX_PROPOSER_RETRIES_CEILING",
    "PROPOSAL_SCHEMA",
    "PROPOSER_DIAGNOSTIC_KEYS",
    "PROPOSER_RETRY_BACKOFF_S",
    "PROPOSER_SYSTEM",
    "TERMINAL_PROPOSER_FAILURE_MARKERS",
    "SYNTHETIC_EVIDENCE_CLASS",
    "ClaudeProposer",
    "action_catalogue",
    "binding_constraints",
    "CycleOutcome",
    "ExperimentHistory",
    "MalformedProposal",
    "Proposer",
    "ProposerUnavailable",
    "StopRule",
    "TargetCampaignController",
    "TerminalReason",
    "build_candidate_pool",
    "campaign_caps",
    "campaign_exit_code",
    "check_campaign_registration",
    "proposal_schema",
    "declare_campaign_caps",
    "frozen_incumbent_and_references",
    "frozen_track_b_incumbent_and_references",
    "measured_training_capability",
    "measurement_semantics",
    "metric_semantics",
    "poisson_resolution",
    "select_nomination",
    "plan_contract",
    "scoring_contract",
    "structured_feedback",
    "verify_measurement_semantics",
]
