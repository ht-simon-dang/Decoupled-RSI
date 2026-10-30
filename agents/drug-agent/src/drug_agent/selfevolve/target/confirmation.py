"""Step 15 — the confirmation boundary, and the refusal that guards the held-out set.

A confirmation study is only a confirmation if the thing being scored was chosen
*before* the score existed. Every mechanism in this module exists to make that
ordering enforceable rather than promised:

* **The gate authenticates its own input.** Every mechanism below reads out of
  one document — the lock-closed check, the locked side of the preflight, the
  ledger's denominator *and* its one-nomination cap, all of the gate conditions,
  the intention-to-run denominator and the recorded cost. So the protocol is
  checked against the address this study is REGISTERED at
  (:data:`REGISTERED_PROTOCOL_HASH`) by :func:`authenticate_protocol` before any
  of it is trusted, and :func:`confirmation_readiness` re-derives that address
  itself as its first condition. Re-hashing a document and finding it agrees
  with the hash stored beside it proves only that the file agrees with itself: a
  forged lock re-stamped with this project's own ``protocol_hash()`` is
  self-consistent exactly as the real one is. **A self-consistent document is
  not an authentic one.**
* **A ledger entry is addressed, not counted.** Entries are chained content
  addresses (:func:`_entry_digest`), the release pins the chain head, and
  :meth:`~NominationBoundary.audit` reports an entry whose content moved, a
  chain that no longer links and a ledger carrying no addresses at all. A
  release that recorded only *how many* entries there were is passed unchanged
  by editing one of them in place, which is both how a second nomination hides
  in the first one's row and how a model hash gets re-pointed at whatever is on
  disk.
* **Addresses are measured, never accepted.**
  :func:`measure_model_address` derives a nominated artifact's address from its
  own canonical bytes, and :func:`reload_nominated_artifact` refuses a
  caller-supplied address that disagrees with it. An address a caller types is
  the caller's claim about bytes; comparing a record with a claim about itself
  is the circular self-comparison this module refuses everywhere.

* **Nominations are frozen before any test feedback is released.**
  :class:`NominationBoundary` is a small append-only ledger with three states.
  While it is ``OPEN`` a campaign may write exactly one terminal record. Only
  when *every* locked campaign seed has a terminal record does it reach
  ``NOMINATIONS_COMPLETE``, and only then may :meth:`~NominationBoundary.release`
  hand a final decision back to the worker. A nomination written after the
  release is ``TEST_LEAKAGE``, not a late entry.
* **A second final nomination for the same campaign/track is rejected.** The
  lock says ``max_final_nominations = 1`` and
  ``nomination_rule.max_per_campaign_per_registered_track = 1``. Re-nominating is
  how a campaign quietly takes the better of two tries, so it is refused by
  number rather than by convention.
* **Changing a model hash invalidates its nomination.** A nomination names a
  content address. If the bytes behind it move, the artifact that was nominated
  is not the artifact being scored, and :func:`reload_nominated_artifact` refuses
  with ``HASH_MISMATCH``.
* **A failed reload is a failed reload.** A weights-only predictor reloads fixed
  weights. Retraining produces a *different* artifact and is refused as a
  substitute (``UNSUPPORTED_CLAIM``) — replaying a fit is a different experiment
  and has to be labelled as one.
* **Per-artifact evidence and process evidence are different things.**
  :func:`artifact_evidence` (C2) resamples the held-out family groups behind one
  fitted artifact. :func:`process_evidence` (C3) resamples *campaigns*, because
  the lock's replication unit is the campaign and
  ``molecule.is_replication_unit`` is false. Reporting a per-molecule interval as
  process confidence is refused, not merely discouraged.
* **Failures stay in the denominator.** :func:`intention_to_run_table` builds the
  table from the *locked* campaign seeds, never from the campaigns that
  succeeded. A no-nominee campaign contributes ``delta_i = 0.0`` — a statement
  about the process delivering nothing — and a failed *evaluation* contributes a
  missing value with a recorded cost, which is a different fact.

**Five properties that hold as CLASSES, not as fixed instances.** Each of these
was once fixed in one function and left unfixed in its siblings, which is how the
same defect was found three times in three audits:

* **No caller-controlled ``**kwargs`` is merged over a record's own fields.**
  :func:`_refuse_protected` is applied at every site that takes them, and
  :data:`KWARGS_GUARDED` lists those sites so a test can enumerate them: a new
  sibling that merges kwargs without a guard fails the suite rather than waiting
  for the next audit.
* **A held-out value is refused STRUCTURALLY, not by name.** A nomination carries
  a number in exactly two places — ``development_value`` and a flat ``cost``
  mapping over :data:`COST_RESOURCE_FIELDS` — so a number anywhere else, under
  any name, at any depth, is refused without being read
  (:func:`_numeric_payload_paths`). :data:`HELD_OUT_VALUE_FIELDS` is the second
  layer, for the leaks that are not numbers, and nothing depends on it being
  complete. No refusal ever prints the value; they name the path.
* **A record that asserts the partition was NOT READ carries no unexplained
  magnitude.** The same structural rule as above, applied to the ASSEMBLED RECORD
  rather than to a nomination, and inverted:
  :func:`refuse_false_assurance` takes an ALLOWLIST of the paths a record of that
  kind may carry a number at — counts, protocol-declared costs at registered
  resource names, digests, seeds — and refuses a magnitude anywhere else, under
  any name, at any depth, without reading it
  (:func:`magnitude_paths_outside`, :data:`NOT_RUN_MAGNITUDE_PATHS`).
  :data:`FALSE_ASSURANCE_GUARDED` and :data:`FALSE_ASSURANCE_EXEMPT` list every
  emitter, and the test that reads them enumerates the emitters FROM THE SOURCE:
  the round that introduced this guard found three siblings that had none.
* **A safety-relevant state is DERIVED, never accepted.** The release that makes a
  held-out value readable comes from the ledger
  (:meth:`~NominationBoundary._adopt_release`), not from a word a caller typed or
  a ``released`` block in an unauthenticated file; and the protocol is
  canonicalised once (:func:`canonicalised`) so the document that is hashed is
  the object that is read.
* **The ledger is tamper evident at its HEAD as well as its genesis, released or
  not.** :meth:`~NominationBoundary.head_record` pins the ledger's length and its
  chain head into every persisted record, because a chain verified only forward
  from the genesis makes every valid PREFIX a valid ledger and leaves the last
  entry free.

The ceiling on all of this is stated where it applies and again in
``docs/research/handoffs/step-15.md`` §9.0: these are keyless digests over files
their editor holds. They stop the cheap partial edit — which is what an honest
mistake and a quietly-opened gate both look like — and they do not stop a file
rewritten consistently end to end, nor anything at all against code running
inside this process.

**And the part that matters most today.** Nothing here reads, scores or generates
against the held-out partition, and :func:`confirmation_readiness` is written so
that it cannot be talked into doing so. It returns a gate: a list of named
conditions, each with what the lock requires, what was observed and what would
unblock it. Execution is the exception that has to clear every condition; the
default, and the outcome under ``zak-c9r-2`` today, is ``NOT_RUN`` with its
reason, its cost and its unblock recorded in full by :func:`not_run_record`.

Under ``zak-c9r-2`` the registered primary is Track B generation quality, whose
``primary_endpoint.partition`` is the development-derived pocket and exclusion
set; it reads no held-out compound at all, and it is already complete. The only
consumer of the held-out partition is the demoted Track A regression secondary,
which the lock registers ``NOT_RUN``, whose estimator is
``delta_i = mae_pactivity(nominee_i) - mae_pactivity(A-B0-parent-rf)``, and whose
main research arm ``B5-self-evolution`` has never been executed. With zero
nominees that estimator is undefined, so running the confirmation today would
score frozen baselines plus a constant, spend ``test_access_count`` from 0 to 1
irreversibly, and close the lock — while measuring nothing the programme's
central question depends on. That is the argument the gate encodes.
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from .contracts import ContractError, ErrorCategory

Record = dict[str, Any]

CONTRACT = "ConfirmationBoundary"

#: The four content addresses ``confirm`` re-measures before it will start. A
#: mismatch here means the study on disk is not the study that was locked.
PREFLIGHT_HASH_FIELDS: tuple[str, ...] = ("target_hash", "data_hash", "split_hash", "evaluator_hash")

#: The protocol version and the content address it is REGISTERED at.
#:
#: Everything in this module reads out of the protocol: the lock-closed check,
#: the locked side of every preflight comparison, the ledger's denominator *and*
#: its one-nomination cap, every gate condition, the intention-to-run
#: denominator and the cost figures in the NOT_RUN record. So the protocol is
#: authenticated against this address before any of it is trusted.
#:
#: It is a module constant and not an argument, deliberately. An authentication
#: that takes the address it checks against from its caller authenticates
#: nothing: the caller simply supplies the address of whatever it brought. The
#: same address is carried by ``configs/research/zak-c9r.yaml``,
#: ``generation_sweep.PROTOCOL_HASH`` and the committed branch-status record, and
#: changing it here means registering a superseding protocol version.
REGISTERED_PROTOCOL_VERSION = "zak-c9r-2"
REGISTERED_PROTOCOL_HASH = "sha256:f960f2c8635ba3053dadbdc05a138d89b4ce6b8ef91456cf0fcaa82c7656366b"

#: The keys a ledger entry owns. :meth:`NominationBoundary.record` writes every
#: one of them itself and refuses a caller that supplies one through ``**extra``.
#: ``nominating_pid`` and ``sequence`` are the fresh-process evidence
#: :func:`reload_nominated_artifact` later tests against; a caller that can
#: overwrite them can make a same-process reload report itself as a fresh one.
PROTECTED_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "campaign_id",
        "track",
        "outcome",
        "artifact_id",
        "model_hash",
        "development_value",
        "cost",
        "reason",
        "nominating_pid",
        "sequence",
        "entry_digest",
        "previous_digest",
    }
)

#: Names under which a held-out measurement could reach a ledger entry.
#: :meth:`NominationBoundary.record` only ever runs *before* the release — it
#: refuses outright afterwards — so a held-out value in an entry it is writing is
#: a test result that exists before the boundary opened, whatever it is called.
#:
#: **This list is deliberately NOT the refusal.** A denylist of names is only ever
#: as complete as whoever last edited it, and the previous round proved it: the
#: scan read only TOP-LEVEL keys, so ``cost={"held_out_value": 0.31}`` walked
#: through both the ledger's refusal and the table's. The refusal that actually
#: holds is structural — :func:`_numeric_payload_paths`, which refuses a number
#: anywhere on a nomination outside the two places a number belongs, under ANY
#: name. This list is the second layer: it names the leaks that are not numbers
#: (a stringified score, a list of held-out labels) and it makes the message
#: readable. It is matched RECURSIVELY, at any depth, after normalisation.
HELD_OUT_VALUE_FIELDS: frozenset[str] = frozenset(
    {
        "held_out_value",
        "held_out_values",
        "held_out_score",
        "held_out_metric",
        "held_out_delta",
        "test_value",
        "test_score",
    }
)

#: The normalised stems :func:`_held_out_name_paths` refuses at any depth. Names
#: are normalised (lower-cased, non-alphanumerics dropped) before matching, so
#: ``heldOutValue``, ``held-out-value`` and ``HELD_OUT_VALUE`` are one name.
HELD_OUT_NAME_STEMS: tuple[str, ...] = (
    "heldout",
    "holdout",
    "testvalue",
    "testscore",
    "testmetric",
    "testdelta",
)

#: The resource names a nomination's ``cost`` mapping may use. An ALLOWLIST, and
#: that direction is the point: a denylist that falls behind lets a new name
#: through, and an allowlist that falls behind refuses a legitimate one loudly.
#: ``cost`` is the one free-form mapping on an entry that may carry numbers, so
#: it is the one place a held-out number could otherwise sit under a name nobody
#: thought to forbid.
COST_RESOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "llm_output_tokens",
        "llm_input_tokens",
        "gpu_seconds",
        "cpu_seconds",
        "wall_seconds",
        "rows",
        "oracle_calls",
        "docking_calls",
        "usd",
        "evidence_class",
        "note",
    }
)

#: The fields :meth:`NominationBoundary.release` writes itself. ``entries_digest``
#: and ``n_final_records`` are the whole of the post-release tamper evidence: a
#: caller that can type them can pin the chain at a value it chose and every
#: later edit audits clean.
PROTECTED_RELEASE_FIELDS: frozenset[str] = frozenset(
    {"released_at", "released_by", "n_final_records", "entries_digest", "head_digest", "released_state"}
)

#: The fields :func:`not_run_record` asserts for itself. These are the record's
#: claims about the held-out partition; a caller that can overwrite them can
#: publish a branch-status record asserting whatever it likes.
PROTECTED_STATUS_FIELDS: frozenset[str] = frozenset(
    {
        # Found by the enumerating guard test below, not by a reviewer: the
        # record writes `contract_version`, `protocol_version` and `track` for
        # itself and they were not in this set, so `not_run_record(..., track="B")`
        # published a branch-status record about a track the gate never ran for.
        "contract_version",
        "protocol_version",
        "track",
        "record_kind",
        "status",
        "implementation_status",
        "test_access_count",
        "held_out_partition",
        "blocking_conditions",
        "reasons",
        "conditions_for_a_legitimate_later_run",
        "cost_if_run",
        "why_not_failed",
        "what_was_delivered_instead",
        "nomination_ledger_assurance",
        "held_out_execution_path",
    }
)

#: The ledger design fields :meth:`NominationBoundary.from_protocol` reads out of
#: the authenticated lock. A caller that could pass them through ``**options``
#: would be setting the denominator and the cap it is supposed to be bound by.
#:
#: ``tracks`` is here because it was NOT, and that omission was the whole of the
#: defect: the one-nomination cap is keyed on ``(campaign_id, track)``, so
#: ``tracks`` is half the slot key, and ``from_protocol(protocol, tracks=[])``
#: was accepted. With no tracks there are no slots, :meth:`outstanding` goes
#: empty, :meth:`is_complete` returns ``True`` over ZERO entries,
#: ``NOMINATIONS_COMPLETE`` reports met and :meth:`release` succeeds with
#: ``n_final_records: 0``. Both previous guard tests were blind to it by
#: construction — one checked that every kwargs-merging function is registered,
#: the other that each refuses the fields ALREADY in its set — so the guard now
#: enumerates the reachable fields from the CODE (see :data:`KWARGS_SINK_EXEMPT`).
PROTECTED_LEDGER_DESIGN_FIELDS: frozenset[str] = frozenset(
    {"campaign_ids", "tracks", "max_per_campaign_per_track"}
)

#: Fields that reach a guarded function's callee through ``**kwargs`` and are
#: DELIBERATELY left settable, each with its reason. The guard test enumerates
#: what is reachable from the code itself — the keys of the record a function
#: returns, or the signature of the constructor its kwargs flow into — and fails
#: on any field that is in neither the protected set nor this one. That is what
#: stops the guard being blind by construction: it can no longer be satisfied by
#: a protected set that is missing the field nobody thought of.
KWARGS_SINK_EXEMPT: dict[str, frozenset[str]] = {
    "NominationBoundary.record": frozenset(),
    "NominationBoundary.release": frozenset(),
    # `path` is where the ledger persists. It is not part of the design the
    # ledger is bound by and it is not half of a slot key.
    "NominationBoundary.from_protocol": frozenset({"path"}),
    "not_run_record": frozenset(),
}

#: The tracks the confirmation ledger holds slots for when the lock names none.
#: A module constant rather than a parameter default: a value that can be
#: defaulted in silently is a value that can be emptied in silently, and this one
#: is half a slot key.
CONFIRMATION_LEDGER_TRACKS: tuple[str, ...] = ("A",)

#: Every field a ledger's ``head`` pin must carry. A pin is ALL of them or it is
#: not a pin. :meth:`NominationBoundary._head_violations` used to compare a field
#: only ``if declared.get(field) is not None``, and to report a missing pin only
#: for an EMPTY mapping — so ``"head": {"released": false}`` was a "present" pin
#: with nothing to compare, tail deletion was invisible again, the slot re-opened
#: and :meth:`record` accepted a replacement. Nulling a field now fails closed.
HEAD_PIN_FIELDS: tuple[str, ...] = (
    "n_entries",
    "entries_digest",
    "head_digest",
    "highest_sequence",
    "released",
)

#: The external anchor. Every previous round added another self-referential check
#: over the ledger, and the same file satisfied each one by copy-paste: the most
#: recent was a ``released`` block "derived" by comparing it against the
#: ``entries_digest`` and the entry count that :meth:`NominationBoundary.to_record`
#: PRINTS INTO THE SAME FILE. **No keyless digest over a file its editor holds can
#: authenticate that file**, and a document that agrees with itself is
#: self-consistent, not authentic — the sentence this module opens with applies to
#: the ledger verbatim.
#:
#: The repository history is an anchor the ledger's editor does not hold in the
#: same breath. The ledger head digest is recorded in a COMMITTED file that is not
#: the ledger, and the release is corroborated against THAT rather than against
#: fields the ledger prints about itself.
#:
#: **What this buys:** a cheap partial edit — one file, no commit, which is what
#: an honest mistake and a quietly-opened gate both look like — no longer
#: corroborates anything. **What it does not buy:** it does not stop someone who
#: edits the ledger AND writes AND commits a matching anchor. That is a
#: reviewable source-control change rather than a data edit, which is the whole
#: of the improvement. It is not authentication and it is not claimed as any.
ANCHOR_RECORD_KIND = "confirmation_nomination_anchor"

#: Where the committed anchor lives. A module constant for the same reason
#: :data:`REGISTERED_PROTOCOL_HASH` is: an anchor whose location comes from the
#: caller anchors nothing, because the caller supplies the path of whatever it
#: brought.
REGISTERED_ANCHOR_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "research" / "zak-c9r" / "nomination-anchor.json"
)

#: Every function in this module that merges caller-controlled ``**kwargs`` into
#: a record or a design, and the field set each one refuses.
#:
#: This exists so the guard can be enumerated rather than remembered.
#: ``test_every_kwargs_merging_function_refuses_its_protected_fields`` walks this
#: module for functions carrying a ``**kwargs`` parameter and fails if one is not
#: registered here — because the last round fixed ``record()`` and left
#: ``release()`` and ``not_run_record()`` merging last, and a guard that covers
#: only today's call sites rots exactly that way.
KWARGS_GUARDED: dict[str, frozenset[str]] = {
    "NominationBoundary.record": PROTECTED_ENTRY_FIELDS,
    "NominationBoundary.release": PROTECTED_RELEASE_FIELDS,
    "NominationBoundary.from_protocol": PROTECTED_LEDGER_DESIGN_FIELDS,
    "not_run_record": PROTECTED_STATUS_FIELDS,
}


class BoundaryState(StrEnum):
    """Where a campaign set is relative to the release boundary."""

    OPEN = "OPEN"  # nominations may still be written
    NOMINATIONS_COMPLETE = "NOMINATIONS_COMPLETE"  # every locked seed has a terminal record
    RELEASED = "RELEASED"  # held-out feedback has been handed back; nothing may be nominated


class CampaignOutcome(StrEnum):
    """Terminal states a locked campaign can reach, per the lock's failure treatment.

    Every one of them is terminal and every one of them stays in the denominator.
    They differ in what they contribute, which is the whole point of keeping them
    apart: ``NO_NOMINEE`` is a measured statement that the process delivered
    nothing, and ``EVALUATION_FAILED`` is a *missing* value with a recorded cost.
    """

    NOMINATED = "NOMINATED"
    NO_NOMINEE = "NO_NOMINEE"
    CANDIDATE_ERROR = "CANDIDATE_ERROR"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    EVALUATION_FAILED = "EVALUATION_FAILED"


#: Outcomes that contribute ``delta_i = 0.0``: the process ran and delivered no
#: improvement. ``EVALUATION_FAILED`` is deliberately absent — a failed
#: measurement has a missing value, not a zero one.
ZERO_DELTA_OUTCOMES: frozenset[str] = frozenset(
    {
        CampaignOutcome.NO_NOMINEE.value,
        CampaignOutcome.CANDIDATE_ERROR.value,
        CampaignOutcome.INFRASTRUCTURE_FAILURE.value,
    }
)

#: The locked replication unit for *process* evidence. The lock sets
#: ``molecule.is_replication_unit = false``; a per-molecule bootstrap reported as
#: process confidence is pseudo-replication and is refused by name.
PROCESS_REPLICATION_UNIT = "campaign"
FORBIDDEN_PROCESS_UNITS: frozenset[str] = frozenset({"molecule", "compound", "prediction", "row"})

#: The full paths a flat, registered ``cost`` mapping occupies on a nomination.
#: Written out rather than matched with ``startswith("cost.")``: a prefix admits
#: ``cost.anything``, and this programme has now been bitten four times by a
#: check that admits a name nobody enumerated. Every member of this set is a
#: REGISTERED resource, so a number here is a resource quantity by construction.
COST_MAGNITUDE_PATHS: frozenset[str] = frozenset(f"cost.{field}" for field in COST_RESOURCE_FIELDS)

#: Training modes a nominated predictor may declare. Only the first one may be
#: reloaded and scored; the second is a different experiment.
WEIGHTS_ONLY = "WEIGHTS_ONLY"
RETRAINED = "RETRAINED"


def _fail(
    category: ErrorCategory, message: str, *, field: str | None = None, missing: Sequence[str] = ()
) -> ContractError:
    return ContractError(category, message, contract=CONTRACT, field=field, missing=missing)


# ---------------------------------------------------------------------------
# the two refusals that every caller-facing entry point shares
# ---------------------------------------------------------------------------
def _refuse_protected(
    supplied: Mapping[str, Any], protected: frozenset[str], *, what: str, why: str
) -> None:
    """Refuse a caller that supplied a field the callee writes for itself.

    CLASS 1. ``**kwargs`` merged LAST over a record's own fields is the same
    defect wherever it appears, and it appeared in three places: ``record()``
    (fixed last round), ``release()`` — over ``entries_digest`` and
    ``n_final_records``, which are the entire post-release tamper evidence — and
    ``not_run_record()``, over ``status``, ``test_access_count`` and
    ``held_out_partition``, which are the record's claims about the partition.

    Merging kwargs FIRST is not a fix on its own: it silently drops what the
    caller passed, so a caller that believed it was recording provenance records
    nothing and is never told. So the clash is refused by name instead, and
    :data:`KWARGS_GUARDED` lists every function that has to do it.
    """
    clashes = sorted(set(supplied) & protected)
    if clashes:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"{clashes} may not be supplied by the caller: {what} writes them itself. {why} "
            f"Pass the caller's own provenance under different names",
            field=clashes[0],
            missing=clashes,
        )


def _normalised(name: str) -> str:
    """A field name reduced to its comparable stem: lower case, alphanumerics only."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _walk(
    value: Any, keys: tuple[str, ...] = (), path: str = ""
) -> Iterable[tuple[str, tuple[str, ...], Any]]:
    """Yield ``(path, keys, leaf)`` for every leaf under ``value``, at any depth.

    Containers are walked; a string is a leaf and is never walked as a sequence.
    ``keys`` is every mapping key on the way down, so a value buried under a
    held-out name — ``held_out_value.detail[0].raw`` — is still attributed to the
    name it was filed under rather than to the innermost key.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk(item, (*keys, str(key)), child)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            yield from _walk(item, keys, f"{path}[{index}]")
        return
    yield path, keys, value


def _mappings(value: Any, path: str = "") -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield ``(path, mapping)`` for every mapping under ``value``, at any depth.

    :func:`_walk` yields leaves and loses the shape; some refusals are about the
    shape — a row that says it NOMINATED and carries a delta, at whatever depth
    and under whatever key it was filed.
    """
    if isinstance(value, Mapping):
        yield path, value
        for key, item in value.items():
            yield from _mappings(item, f"{path}.{key}" if path else str(key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _mappings(item, f"{path}[{index}]")


def _looks_like_a_number(leaf: Any) -> bool:
    """Whether a leaf carries a magnitude: a real number, or a string that is one.

    A stringified score is still a score. ``bool`` is not a number here — ``True``
    carries no measurement — and neither is a string that does not parse.
    """
    if isinstance(leaf, bool):
        return False
    if isinstance(leaf, (int, float)):
        return True
    if isinstance(leaf, str):
        try:
            float(leaf.strip())
        except ValueError:
            return False
        return True
    return False


def _held_out_name_paths(value: Any) -> list[str]:
    """Paths filed under a name that means a held-out measurement. Never a value.

    CLASS 2, second layer. The first layer is :func:`_numeric_payload_paths`,
    which is structural and therefore complete for numbers. This one catches the
    leaks that are not numbers — a stringified score, a list of held-out labels —
    and gives the refusal a readable name. It is explicitly NOT relied on to be
    exhaustive, and nothing here depends on it being so.

    A leaf that is ``None`` is not a leak: ``held_out_value: null`` is a record
    saying the value has NOT been measured, which is exactly what the table
    reports as a missing delta.
    """
    stems = {_normalised(f) for f in HELD_OUT_VALUE_FIELDS}

    def names_a_measurement(key: str) -> bool:
        normal = _normalised(key)
        return normal in stems or any(stem in normal for stem in HELD_OUT_NAME_STEMS)

    return sorted(
        {
            path
            for path, keys, leaf in _walk(value)
            if leaf is not None and any(names_a_measurement(key) for key in keys)
        }
    )


def _held_out_measurement_paths(value: Any) -> list[str]:
    """Paths where a held-out MEASUREMENT sits, at any depth. Never a value.

    Deliberately narrower than :func:`_held_out_name_paths`, because this one runs
    over a whole assembled record rather than over a nomination, and the gate's
    own condition vocabulary legitimately uses the words: ``primary_reads_held_out``
    is a boolean statement about the design and ``held_out_endpoint_partition`` is
    the name of a partition. Neither is a measurement. A measurement has a
    MAGNITUDE, so that is what this looks for — a number, or a string that is one,
    filed under a name that means a held-out reading.
    """
    stems = {_normalised(f) for f in HELD_OUT_VALUE_FIELDS}

    def names_a_measurement(key: str) -> bool:
        normal = _normalised(key)
        return normal in stems or any(stem in normal for stem in HELD_OUT_NAME_STEMS)

    return sorted(
        {
            path
            for path, keys, leaf in _walk(value)
            if _looks_like_a_number(leaf) and any(names_a_measurement(key) for key in keys)
        }
    )


def _numeric_payload_paths(value: Any) -> list[str]:
    """Paths carrying a number, at any depth. Never the number itself.

    CLASS 2, first layer, and the one that actually holds. A held-out measurement
    is a NUMBER, and a denylist of the names a number might arrive under is only
    as complete as whoever last edited it. So the refusal is structural instead:
    on a nomination, a number is admissible in exactly two places —
    ``development_value``, which is a named argument, and a flat ``cost`` mapping
    over :data:`COST_RESOURCE_FIELDS`. A number anywhere else, under any name, at
    any depth, is an unexpected payload and is refused without being read.

    ``bool`` is not a number here: ``True`` carries no measurement.
    """
    return sorted(
        path
        for path, _keys, leaf in _walk(value)
        if isinstance(leaf, (int, float)) and not isinstance(leaf, bool)
    )


#: List indices, collapsed. ``reasons[2].observed.x`` and ``reasons[7].observed.x``
#: are one path in an allowlist: the row number is not part of what a field means.
_INDEX = re.compile(r"\[\d+\]")


def path_pattern(path: str) -> str:
    """A leaf's path with every list index collapsed to ``[]``."""
    return _INDEX.sub("[]", path)


def magnitude_paths_outside(value: Any, allowed: frozenset[str]) -> list[str]:
    """Every path carrying a magnitude that *allowed* does not name. Never a value.

    The inversion. :func:`_numeric_payload_paths` and
    :func:`_held_out_measurement_paths` both answer "is this number one of the
    ones I thought to forbid", and a name-shaped question has a name-shaped
    answer: it is only ever as complete as whoever last edited the list. This one
    asks the opposite question — "is this number one of the small set this record
    is permitted to carry" — so a magnitude under a name nobody anticipated, at
    any depth, is refused by default instead of admitted by default.

    A magnitude is a number or a string that parses as one; ``bool`` is not one
    and neither is ``None``.
    """
    return sorted(
        {
            path
            for path, _keys, leaf in _walk(value)
            if _looks_like_a_number(leaf) and path_pattern(path) not in allowed
        }
    )


def _refuse_cost_shape(cost: Mapping[str, Any] | None, *, campaign_id: str) -> None:
    """A nomination's ``cost`` is a FLAT mapping over registered resource names.

    ``cost`` is the one free-form mapping on an entry that may legitimately carry
    numbers, which makes it the one place a held-out number could sit under a
    name nobody thought to forbid. Two structural rules close that: no nesting,
    and no key outside :data:`COST_RESOURCE_FIELDS`. Neither rule reads a value.
    """
    if cost is None:
        return
    if not isinstance(cost, Mapping):
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            f"campaign {campaign_id!r} supplied a {type(cost).__name__} as its cost; cost is a "
            f"flat mapping of resource name to quantity",
            field="cost",
        )
    nested = sorted(
        key
        for key, item in cost.items()
        if isinstance(item, (Mapping, list, tuple, set, frozenset))
    )
    if nested:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"campaign {campaign_id!r} nests {nested} inside its cost. A nomination's cost is a "
            f"FLAT mapping of resource name to quantity: a nested structure is where a held-out "
            f"number travels under a key the denylist never named. The values are not reported here",
            field="cost",
            missing=nested,
        )
    unregistered = sorted(key for key in cost if str(key) not in COST_RESOURCE_FIELDS)
    if unregistered:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"campaign {campaign_id!r} records cost under {unregistered}, which are not "
            f"registered cost resources {sorted(COST_RESOURCE_FIELDS)}. cost is the one mapping "
            f"on a nomination that may carry numbers, so it is an ALLOWLIST: an allowlist that "
            f"falls behind refuses a legitimate resource loudly, and a denylist that falls "
            f"behind lets a held-out number through silently. The values are not reported here",
            field="cost",
            missing=unregistered,
        )


# ---------------------------------------------------------------------------
# authenticating the gate's own input
# ---------------------------------------------------------------------------
def _derived_protocol_hash(protocol: Mapping[str, Any]) -> str:
    """The project's own canonical content address for a protocol document."""
    from .protocol import protocol_hash

    return protocol_hash(protocol)


def canonicalised(protocol: Mapping[str, Any]) -> Record:
    """The protocol as a plain nested ``dict``, round-tripped through the bytes that get hashed.

    The residual this closes structurally. ``_derived_protocol_hash`` canonicalises
    a document to bytes and hashes them; every consumer then went back to the
    ORIGINAL object and read fields off it. A ``Mapping`` whose ``keys()`` and
    ``__getitem__``/``get()`` disagree — or one that simply answers differently the
    second time — is therefore hashed as one document and read as another, and a
    shallow ``dict(document)`` does not help because the nested values are still
    the caller's objects.

    So the document is canonicalised ONCE, here, and the plain dict that comes
    back out of the canonical bytes is what everything downstream consumes.
    Hashing and reading cannot then diverge, because there is only one object and
    it is the one the address was taken over.

    This is a structural fix, not a defence: reaching it requires running Python
    inside the confirming process, and an attacker who can do that already holds
    the artifacts. See the threat model in docs/research/handoffs/step-15.md.
    """
    from ...util import canonical_json

    # A protocol is DATA read off disk: plain objects, arrays and scalars. Anything
    # else is a live object, and `canonical_json` serialises one it does not
    # recognise as its `repr()` — which would canonicalise to a string while the
    # consumer went on calling `.get()` on the object. So the document is required
    # to be plain data before it is canonicalised at all, which is both the honest
    # check and what keeps the hashed form and the read form the same thing.
    def plain(value: Any, path: str = "protocol") -> None:
        if isinstance(value, Mapping):
            if type(value) is not dict:
                raise _fail(
                    ErrorCategory.INVALID_TYPE,
                    f"{path} is a {type(value).__name__}, not a plain record. A protocol is data "
                    f"read off disk; a live object can answer one way when it is hashed and "
                    f"another when it is read, so it is refused rather than canonicalised",
                    field="protocol",
                )
            for key, item in value.items():
                plain(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                plain(item, f"{path}[{index}]")
            return
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise _fail(
                ErrorCategory.INVALID_TYPE,
                f"{path} is a {type(value).__name__}, which is not a JSON scalar. A protocol is "
                f"data read off disk",
                field="protocol",
            )

    plain(protocol)
    loaded = json.loads(canonical_json(protocol))
    if not isinstance(loaded, dict):  # pragma: no cover - a checked Mapping loads as an object
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            "the protocol did not canonicalise to a record",
            field="protocol",
        )
    return loaded


def protocol_is_authentic(protocol: Mapping[str, Any]) -> bool:
    """Whether a protocol document hashes to the REGISTERED address. Never raises.

    Used by :func:`confirmation_readiness`, which reports conditions rather than
    raising, so that a forged protocol cannot open the gate even when the caller
    bypasses :func:`authenticate_protocol` by calling the gate directly.
    """
    try:
        return _derived_protocol_hash(protocol) == REGISTERED_PROTOCOL_HASH
    except (TypeError, ValueError):  # a document that will not canonicalise is not the lock
        return False


def authenticate_protocol(
    document: Mapping[str, Any],
    *,
    source: str | None = None,
    registered_hash: str = REGISTERED_PROTOCOL_HASH,
) -> tuple[Record, Record]:
    """Authenticate a protocol document against the REGISTERED content address.

    This is the check everything downstream depends on, and it is the one that
    was missing. :func:`~.protocol.load_lock` re-derives a lock's hash and
    compares it with the hash stored *in the same file*: that proves the file
    agrees with itself and nothing more. A lock with fields flipped and
    re-stamped with this project's own ``protocol_hash()`` is self-consistent
    exactly as the real one is, and a gate that believes it has been handed the
    locked design when it has been handed a forged one opens every condition at
    once — the lock-closed check, the locked side of the preflight, the ledger's
    denominator and its one-nomination cap, all seven gate conditions, the
    intention-to-run denominator and the recorded cost all read out of that one
    document.

    So the comparison here is against the address this study is **registered**
    at, which is :data:`REGISTERED_PROTOCOL_HASH` and not anything the caller
    brought with the document.

    Args:
        document: a frozen lock (``record_kind == "study_protocol_lock"``) or the
            protocol source record.
        source: the path it was read from, for the message and the record.
        registered_hash: the registered address. Present so a *test* can pin a
            different registration; every production path takes the default, and
            a caller that supplies one is authenticating against its own claim.

    Returns:
        ``(protocol, authentication_record)`` — the authenticated protocol and
        the statement of how it was authenticated.

    Raises:
        ValueError: ``INVALID_TYPE`` for a document that is not a record,
            ``MISSING_FIELD`` for a lock carrying no protocol, ``HASH_MISMATCH``
            when a lock disagrees with itself or when the derived address is not
            the registered one.
    """
    if not isinstance(document, Mapping):
        raise _fail(
            ErrorCategory.INVALID_TYPE,
            f"{type(document).__name__} is not a protocol document; there is nothing to authenticate",
            field="protocol",
        )
    kind = str(document.get("record_kind") or "")
    stored = ""
    if kind == "study_protocol_lock":
        carried = document.get("protocol")
        if not isinstance(carried, Mapping):
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                f"the lock at {source or '<in memory>'} carries no protocol record, so there is "
                f"nothing to authenticate against the registered address",
                field="protocol",
                missing=["protocol"],
            )
        protocol = dict(carried)
        stored = str(document.get("protocol_hash") or "")
    else:
        protocol = dict(document)

    # Canonicalised BEFORE it is hashed, so the object handed back is the one the
    # address was taken over rather than the one it was taken from, and so a
    # document that is not plain data is refused as such: see `canonicalised`.
    protocol = canonicalised(protocol)
    measured = _derived_protocol_hash(protocol)
    if stored and stored != measured:
        raise _fail(
            ErrorCategory.HASH_MISMATCH,
            f"the lock at {source or '<in memory>'} stores protocol_hash {stored!r} and the "
            f"protocol beside it hashes to {measured!r}. A lock whose stored address was edited "
            f"is not a lock",
            field="protocol_hash",
        )
    if measured != registered_hash:
        raise _fail(
            ErrorCategory.HASH_MISMATCH,
            f"this document hashes to {measured!r} and the REGISTERED protocol address is "
            f"{registered_hash!r}. A document that agrees with its own stored hash is "
            f"SELF-CONSISTENT, not authentic — a forged lock re-stamped with this project's own "
            f"protocol_hash() is self-consistent too, and the whole confirmation reads out of "
            f"this one file: the lock-closed check, the locked side of the preflight, the "
            f"ledger's denominator and its one-nomination cap, every gate condition, the "
            f"intention-to-run denominator and the recorded cost. Register a superseding "
            f"protocol version and confirm against that; do not edit the document",
            field="protocol_hash",
        )

    record: Record = {
        "contract_version": "1.0.0",
        "record_kind": "confirmation_protocol_authentication",
        "input_kind": kind or "study_protocol",
        "source": source,
        "registered_protocol_hash": registered_hash,
        "measured_protocol_hash": measured,
        "protocol_version": protocol.get("protocol_version"),
        "lock_self_consistent": bool(stored) and stored == measured,
        "authenticated": True,
        "rule": (
            "the protocol is authenticated against the REGISTERED content address before any of "
            "it is trusted. A self-consistent document is not an authentic one"
        ),
    }
    return protocol, record


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def preflight(
    protocol: Mapping[str, Any],
    measured: Mapping[str, Any],
    *,
    described_as_sealed: bool = False,
) -> Record:
    """Refuse to start a confirmation against a study that is not the locked one.

    The refusals, in this order, because the order is what makes a report
    readable: a hash that disagrees is reported as *that* hash, never as a
    downstream symptom.

    1. ``MISSING_FIELD`` when ``evaluator_hash`` was not supplied. The evaluation
       kernel is the one address this command can always re-derive from the
       bytes on disk, so there is no excuse for not measuring it and no reason to
       accept a caller's word for it.
    2. ``HASH_MISMATCH`` when a supplied address differs from the locked one —
       for any of :data:`PREFLIGHT_HASH_FIELDS`.
    3. ``HASH_MISMATCH`` when any locked baseline arm's content address differs
       from the measured one. A confirmation is a *comparison*; a moved
       comparator is a moved result.
    4. ``UNSUPPORTED_CLAIM`` when the caller describes the configuration as
       sealed and ``evaluation_exposure.sealed`` is false. On this host a
       candidate launched through ``execute_candidate`` runs as the same OS user
       and can read the protected registry, so no run here is sealed; saying it
       is would be the single most misleading sentence in the report.

    **What this function refuses to pretend.** ``target_hash``, ``data_hash`` and
    ``split_hash`` cannot be re-derived here: rebuilding the split would mean
    re-partitioning, which reads the held-out family assignment, and that is the
    one thing this whole module exists to prevent. So an address the caller did
    not supply is reported under ``not_independently_measured`` with the locked
    value beside it — **never** under ``hashes_agreed``. Comparing a record with
    itself and calling the result agreement is how a preflight becomes
    decoration, and the returned ``status`` says which of the two happened.

    Args:
        protocol: the validated protocol (``lock["protocol"]``, or the source).
        measured: addresses measured independently of the protocol. Must carry
            ``evaluator_hash``; may carry any of the others and an optional
            ``baseline_hashes`` mapping.
        described_as_sealed: whether the caller intends to *call* this run sealed.

    Returns:
        A JSON-ready record separating the addresses that were compared and
        agreed from those nothing independently measured.

    Raises:
        ValueError: a :class:`~.contracts.ContractError` as described above.
    """
    compared: Record = {}
    unmeasured: Record = {}
    for field in PREFLIGHT_HASH_FIELDS:
        locked = protocol.get(field)
        seen = measured.get(field)
        if seen is None:
            if field == "evaluator_hash":
                raise _fail(
                    ErrorCategory.MISSING_FIELD,
                    "nothing measured evaluator_hash for this checkout. The evaluation kernel "
                    "is re-derivable from the bytes on disk at any time, so an unmeasured "
                    "evaluator address is a check that was skipped, not a check that passed",
                    field=field,
                    missing=[field],
                )
            unmeasured[field] = locked
            continue
        if locked != seen:
            raise _fail(
                ErrorCategory.HASH_MISMATCH,
                f"the lock commits to {field} {locked!r} and this checkout produces {seen!r}. "
                f"The study on disk is not the study that was locked; re-freeze the records "
                f"deliberately and register the new version, do not edit the address",
                field=field,
            )
        compared[field] = locked

    locked_baselines = dict(protocol.get("baseline_hashes") or {})
    measured_baselines = measured.get("baseline_hashes")
    baseline_report: Record = {
        "checked": False,
        "arms": sorted(locked_baselines),
        "why_not_checked": (
            "nothing measured the locked comparator arms in this run, so the moved-comparator "
            "refusal could not fire. This is a check that did NOT run, not a check that passed"
        ),
    }
    if measured_baselines is not None:
        moved = sorted(
            arm for arm, address in locked_baselines.items() if dict(measured_baselines).get(arm) != address
        )
        if moved:
            raise _fail(
                ErrorCategory.HASH_MISMATCH,
                f"{len(moved)} locked baseline arm(s) no longer hash to their locked address: "
                f"{moved}. A confirmation is a comparison against these arms, so a moved "
                f"comparator is a moved result",
                field="baseline_hashes",
                missing=moved,
            )
        baseline_report = {"checked": True, "arms": sorted(locked_baselines), "moved": []}

    exposure = dict(protocol.get("evaluation_exposure") or {})
    sealed = bool(exposure.get("sealed", False))
    if described_as_sealed and not sealed:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            "this run is described as sealed and the lock's evaluation_exposure.sealed is "
            f"false: {exposure.get('sealed_note') or 'the lock does not claim a seal'}. "
            "A process control at the ingestion boundary is not an access-control boundary, "
            "and an unsealed run described as sealed is a false claim about the evidence",
            field="evaluation_exposure.sealed",
        )

    return {
        "contract_version": "1.0.0",
        "record_kind": "confirmation_preflight",
        "hashes_agreed": compared,
        "not_independently_measured": unmeasured,
        "not_independently_measured_note": (
            "these addresses were NOT re-derived here and are therefore NOT reported as "
            "agreeing. target_hash, data_hash and split_hash cannot be re-measured without "
            "rebuilding the split, which reads the held-out family assignment. The locked "
            "value is shown so a caller can supply an independently measured one"
        ),
        "baseline_hashes": baseline_report,
        # Named at the TOP level, not only as a nested `checked: false`. A safety
        # check that did not run has to be visible where a reader looks, and a
        # caller (`confirm` is one) has to be able to refuse on it.
        "unchecked_safety_checks": (
            [] if baseline_report["checked"] or not locked_baselines else ["baseline_hashes"]
        ),
        "sealed": sealed,
        "described_as_sealed": described_as_sealed,
        "status": "PREFLIGHT_OK" if not unmeasured else "PREFLIGHT_OK_WITH_UNMEASURED",
    }


# ---------------------------------------------------------------------------
# the release boundary
# ---------------------------------------------------------------------------
#: What the first entry's ``previous_digest`` links to. A literal, so an empty
#: ledger's head is still a value that can be recorded and compared.
LEDGER_GENESIS = "genesis"


def load_external_anchor(path: Path | str | None = None) -> Record | None:
    """The committed record of the ledger head, read from a file that is NOT the ledger.

    See :data:`ANCHOR_RECORD_KIND` for what this does and does not buy. A document
    that is not an anchor record for THIS protocol is not an anchor: the
    ``record_kind`` and the registered protocol address are both required, so a
    stray or mis-aimed file cannot anchor this study by accident.

    Returns ``None`` when there is no usable anchor, and every caller treats that
    as *unanchored* rather than as *anchored by default*.
    """
    anchor_path = Path(path) if path is not None else REGISTERED_ANCHOR_PATH
    try:
        document = json.loads(anchor_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, Mapping):
        return None
    if str(document.get("record_kind") or "") != ANCHOR_RECORD_KIND:
        return None
    if str(document.get("protocol_hash") or "") != REGISTERED_PROTOCOL_HASH:
        return None
    record = dict(document)
    record["anchor_path"] = str(anchor_path)
    return record


def anchor_commit_state(path: Path | str | None = None) -> str:
    """Whether the anchor file is COMMITTED, as the repository sees it.

    This is the one thing that makes the anchor external at all. An anchor file
    edited beside the ledger and left uncommitted is the same cheap single-pass
    edit the anchor exists to defeat, so it is reported as ``UNCOMMITTED_CHANGES``
    and corroborates nothing. ``UNKNOWN_NO_GIT`` fails closed for the same reason:
    an anchor whose commit state cannot be established is not an anchor.

    Returns one of ``COMMITTED``, ``UNCOMMITTED_CHANGES``, ``UNTRACKED``,
    ``ABSENT`` or ``UNKNOWN_NO_GIT``. It reads the repository, never the network,
    and it never reads the held-out partition.
    """
    anchor_path = Path(path) if path is not None else REGISTERED_ANCHOR_PATH
    if not anchor_path.is_file():
        return "ABSENT"
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", anchor_path.name],
            cwd=str(anchor_path.parent),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN_NO_GIT"
    if completed.returncode != 0:
        return "UNKNOWN_NO_GIT"
    line = completed.stdout.strip()
    if not line:
        return "COMMITTED"
    return "UNTRACKED" if line.startswith("??") else "UNCOMMITTED_CHANGES"


def _entry_digest(entry: Mapping[str, Any], previous: str) -> str:
    """The content address of one ledger entry, chained to the one before it.

    Over the entry's whole content — including anything a caller added — with the
    two chain fields removed, because they are the address rather than part of
    what it addresses. Chained, so that editing an entry in place and re-stamping
    only *that* entry's address breaks the link the next entry carries.
    """
    from ...util import canonical_json, hash_bytes

    body = {k: v for k, v in entry.items() if k not in ("entry_digest", "previous_digest")}
    return hash_bytes(canonical_json({"previous": previous, "entry": body}).encode("utf-8"))


class NominationBoundary:
    """The append-only ledger that keeps nominations ahead of test feedback.

    The ledger is created from the lock's own ``campaign_seeds``, so its
    denominator is the locked design and not the set of campaigns that happened
    to finish. It refuses, by :class:`~.contracts.ErrorCategory`:

    * a second terminal record for the same ``(campaign_id, track)``
      — ``INVALID_VALUE``, because ``max_final_nominations`` is 1;
    * any nomination once the boundary has been released — ``TEST_LEAKAGE``;
    * a release while any locked campaign is still open — ``UNSUPPORTED_CLAIM``;
    * a campaign id the lock never registered — ``INVALID_VALUE``;
    * a caller that supplies one of :data:`PROTECTED_ENTRY_FIELDS` for itself —
      ``INVALID_VALUE``, because an entry whose own process evidence was typed by
      its author records nothing;
    * a held-out value in a nomination — ``TEST_LEAKAGE``; see
      :data:`HELD_OUT_VALUE_FIELDS`.

    Every entry carries a chained content address, and :meth:`audit` re-derives
    the chain: a ledger is not a count of rows, because a row edited in place
    keeps the count.
    """

    def __init__(
        self,
        campaign_ids: Sequence[str],
        *,
        tracks: Sequence[str] = ("A",),
        max_per_campaign_per_track: int = 1,
        path: Path | str | None = None,
    ) -> None:
        if not campaign_ids:
            raise _fail(
                ErrorCategory.EMPTY_VALUE,
                "a boundary with no locked campaigns has nothing to hold back; the "
                "denominator is the locked design, so it cannot be empty",
                field="campaign_ids",
            )
        if not tracks:
            raise _fail(
                ErrorCategory.EMPTY_VALUE,
                "a boundary with no tracks holds no slots: the one-nomination cap is keyed on "
                "(campaign_id, track), so an empty track set empties the denominator, makes "
                "is_complete() true over zero entries and lets the boundary release with "
                "n_final_records 0. `tracks` is half the slot key and it fails closed",
                field="tracks",
            )
        if max_per_campaign_per_track < 1:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"max_per_campaign_per_track is {max_per_campaign_per_track}; a campaign that "
                f"may nominate nothing cannot produce a confirmation",
                field="max_per_campaign_per_track",
            )
        self.campaign_ids = [str(c) for c in campaign_ids]
        self.tracks = [str(t) for t in tracks]
        self.max_per_campaign_per_track = int(max_per_campaign_per_track)
        self.path = Path(path) if path is not None else None
        self._entries: list[Record] = []
        self._state = BoundaryState.OPEN
        self._released: Record | None = None
        #: What a ledger read off disk declared about its own design, when that
        #: is not where the design came from. `audit` compares the two.
        self._declared: Record | None = None
        #: What a ledger read off disk declared about its own HEAD: how many
        #: entries it carries and what they chain to. `None` means this boundary
        #: was built fresh in this process and there is nothing to compare; an
        #: EMPTY mapping means it was loaded from a record that carried no pin at
        #: all, which `audit` reports rather than passes. See `_head_violations`.
        self._declared_head: Record | None = None
        #: Why a `released` block read off disk was not corroborated, if it was
        #: not. A release claim is a safety-relevant state and is therefore
        #: DERIVED, never believed: see `_adopt_release`.
        self._release_uncorroborated: list[str] = []
        #: The COMMITTED external anchor, when this ledger came off disk. `None`
        #: for a boundary built in this process — it is not a file yet, so there
        #: is nothing for an external record to anchor. See `_anchor_violations`.
        self._anchor: Record | None = None

    # -- state ------------------------------------------------------------
    @property
    def state(self) -> BoundaryState:
        if self._state is BoundaryState.RELEASED:
            return BoundaryState.RELEASED
        return BoundaryState.NOMINATIONS_COMPLETE if self.is_complete() else BoundaryState.OPEN

    @property
    def release_is_corroborated(self) -> bool:
        """Whether a ``released`` block read off disk is borne out by the entries beside it.

        ``True`` for a boundary that released in this process, and for one whose
        loaded release pins a digest that addresses the entries actually present.
        See :meth:`_adopt_release`.
        """
        return not self._release_uncorroborated

    @property
    def uncorroborated_release_reasons(self) -> list[str]:
        """Why a loaded ``released`` block was not corroborated. Empty when it was."""
        return list(self._release_uncorroborated)

    @property
    def entries(self) -> list[Record]:
        """A copy. The ledger is append-only through :meth:`record`."""
        return [dict(e) for e in self._entries]

    def outstanding(self) -> list[str]:
        """Locked ``(campaign, track)`` pairs with no terminal record yet."""
        # `.get` rather than `[...]`: a ledger resumed from disk may be malformed,
        # and an outstanding-slot query should report the slot as outstanding
        # rather than raise a KeyError that reads like an infrastructure fault.
        done = {(e.get("campaign_id"), e.get("track")) for e in self._entries}
        return sorted(
            f"{campaign}/{track}"
            for campaign in self.campaign_ids
            for track in self.tracks
            if (campaign, track) not in done
        )

    def is_complete(self) -> bool:
        return not self.outstanding()

    # -- writing ----------------------------------------------------------
    def record(
        self,
        campaign_id: str,
        *,
        track: str = "A",
        outcome: str = CampaignOutcome.NOMINATED.value,
        artifact_id: str | None = None,
        model_hash: str | None = None,
        development_value: float | None = None,
        cost: Mapping[str, Any] | None = None,
        reason: str | None = None,
        **extra: Any,
    ) -> Record:
        """Write one campaign's single terminal record.

        ``outcome`` is one of :class:`CampaignOutcome`. ``NOMINATED`` requires an
        ``artifact_id`` and a ``model_hash`` — a nomination that names no content
        address cannot be shown to be the thing that was later scored. Every
        other outcome is terminal too and occupies the campaign's one slot, which
        is what keeps a failed campaign in the denominator instead of leaving the
        slot open for a luckier retry.

        Raises:
            ValueError: ``TEST_LEAKAGE`` after release or for a held-out value,
                ``INVALID_VALUE`` for an unregistered campaign, a second final
                nomination or a caller-supplied protected field,
                ``INVALID_ENUM`` for an unregistered outcome, ``MISSING_FIELD``
                for a nomination with no artifact.
        """
        if self._state is BoundaryState.RELEASED:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"campaign {campaign_id!r} tried to nominate after held-out feedback was "
                f"released at {self._released and self._released.get('released_at')!r}. A "
                f"nomination written once a test result exists is not a pre-test nomination",
                field="campaign_id",
            )
        # CLASS 4. Detecting truncation in `audit` is not enough on its own: the
        # executed consequence was that deleting the tail entry RE-OPENED that
        # campaign's slot, so this method went on to write a genuine second final
        # nomination into a ledger that already had one. A ledger whose head does
        # not match what it pinned is not the ledger this entry would be appended
        # to, so nothing is appended to it.
        disturbed = self._disturbance()
        if disturbed:
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                f"campaign {campaign_id!r} is writing into a ledger that no longer matches the "
                f"head it pinned: {[v['violation'] for v in disturbed]}. Deleting the tail entry "
                f"re-opens that campaign's slot, so an append to a disturbed ledger is how a "
                f"second final nomination becomes a first one. Audit the ledger and resolve the "
                f"violation deliberately; do not write over it",
                field="entries",
                missing=[str(v["violation"]) for v in disturbed],
            )
        if campaign_id not in self.campaign_ids:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"{campaign_id!r} is not one of the locked campaigns {self.campaign_ids}. "
                f"An unregistered campaign is a different experiment and must be registered "
                f"separately rather than joined to this denominator",
                field="campaign_id",
            )
        if track not in self.tracks:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"{track!r} is not one of the registered tracks {self.tracks}",
                field="track",
            )
        if outcome not in {member.value for member in CampaignOutcome}:
            raise _fail(
                ErrorCategory.INVALID_ENUM,
                f"{outcome!r} is not a registered terminal outcome; the vocabulary is "
                f"{sorted(m.value for m in CampaignOutcome)}",
                field="outcome",
            )

        existing = [
            e for e in self._entries if e.get("campaign_id") == campaign_id and e.get("track") == track
        ]
        if len(existing) >= self.max_per_campaign_per_track:
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"campaign {campaign_id!r} track {track!r} already has a final record "
                f"({existing[0].get('outcome')}, artifact {existing[0].get('artifact_id')!r}) and the "
                f"lock allows {self.max_per_campaign_per_track} per campaign per track. A second "
                f"final nomination is taking the better of two tries after the fact",
                field="campaign_id",
            )

        if outcome == CampaignOutcome.NOMINATED.value and not (artifact_id and model_hash):
            named = (("artifact_id", artifact_id), ("model_hash", model_hash))
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                "a nomination must name both the artifact and the content address of its "
                "model; a nomination with no address cannot be shown to be the artifact "
                "that was later reloaded and scored",
                field="model_hash",
                missing=[key for key, value in named if not value],
            )

        # CLASS 1. `**extra` used to be merged LAST, so a caller could overwrite
        # the very fields this ledger exists to record — nominating_pid and
        # sequence are the fresh-process evidence `reload_nominated_artifact`
        # tests against, and a caller that can set them can make a same-process
        # reload report itself as a fresh one.
        _refuse_protected(
            extra,
            PROTECTED_ENTRY_FIELDS,
            what="the ledger",
            why=(
                "nominating_pid and sequence are the process evidence a later reload is checked "
                "against, and an entry whose own evidence was typed by its author records "
                "nothing."
            ),
        )

        # CLASS 2. A held-out measurement cannot exist yet: `record` refuses
        # outright once the boundary is RELEASED, so every call to it happens
        # BEFORE any held-out feedback was handed back. A value here is a test
        # result that predates the release, whatever key it arrived under and
        # HOWEVER DEEPLY IT IS NESTED — the previous round scanned only top-level
        # keys and copied `cost` verbatim, so `cost={"held_out_value": 0.31}`
        # walked through this refusal and the table's and landed in both
        # confirmation.json and intention-to-run.json while the record went on
        # asserting the partition was NOT READ.
        _refuse_cost_shape(cost, campaign_id=campaign_id)
        payload: Record = {**extra, "cost": dict(cost) if cost is not None else None}
        named_paths = _held_out_name_paths(payload)
        if named_paths:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"campaign {campaign_id!r} is writing a held-out measurement into its nomination "
                f"at {named_paths} while the boundary is {self.state.value}. Nominations are frozen "
                f"BEFORE any held-out feedback is released, so a held-out value in a nomination "
                f"is a test result that existed before the boundary opened. Held-out values are "
                f"attached to the intention-to-run table after the release, never to a "
                f"nomination. The value itself is not reported here",
                field=named_paths[0],
                missing=named_paths,
            )
        # An ALLOWLIST of full paths, not a `cost.` prefix: a prefix admits
        # `cost.anything`, and the one mapping on an entry that may carry numbers
        # is the last place to accept a name nobody enumerated.
        stray = [
            path for path in _numeric_payload_paths(payload) if path not in COST_MAGNITUDE_PATHS
        ]
        if stray:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"campaign {campaign_id!r} attaches {len(stray)} unexpected numeric value(s) to "
                f"its nomination at {stray}. A held-out measurement is a NUMBER, and a denylist "
                f"of the names a number might arrive under is only ever as complete as whoever "
                f"last edited it. So a nomination carries a number in exactly two places — "
                f"development_value, which is a named argument, and a flat cost mapping over the "
                f"registered resources. A number anywhere else, under any name, at any depth, is "
                f"refused without being read. The values are not reported here",
                field=stray[0],
                missing=stray,
            )
        previous = self.head_digest()
        entry: Record = {
            **extra,
            "campaign_id": campaign_id,
            "track": track,
            "outcome": outcome,
            "artifact_id": artifact_id,
            "model_hash": model_hash,
            "development_value": development_value,
            "cost": dict(cost) if cost is not None else None,
            "reason": reason,
            "nominating_pid": os.getpid(),
            "sequence": len(self._entries),
        }
        entry["previous_digest"] = previous
        entry["entry_digest"] = _entry_digest(entry, previous)
        self._entries.append(entry)
        self._repin()
        self._persist()
        return dict(entry)

    def _repin(self) -> None:
        """Advance a loaded ledger's head pin over a legitimate append.

        The pin records the head as it stood when the ledger was last written by
        this module. Appending through :meth:`record` IS that, so the pin moves
        with it; every other way the entries can change is not, and leaves the
        pin where it was. A boundary built fresh in this process has no pin and
        acquires none here — it acquires one when it is persisted.
        """
        if self._declared_head is not None:
            self._declared_head = self.head_record()

    def head_digest(self) -> str:
        """The chain head as RECORDED: the last entry's stored address.

        What a new entry links to. Compare :meth:`content_digest`, which is the
        same chain re-derived from what the entries now say — the two agree only
        while nothing has been edited.
        """
        if not self._entries:
            return LEDGER_GENESIS
        return str(self._entries[-1].get("entry_digest") or LEDGER_GENESIS)

    def content_digest(self) -> str:
        """The chain head RE-DERIVED from every entry's current content.

        This is what the release pins and what an audit re-computes, because the
        recorded head is the last entry's stored address and is therefore blind
        to an edit made anywhere before it.
        """
        previous = LEDGER_GENESIS
        for entry in self._entries:
            previous = _entry_digest(entry, previous)
        return previous

    def head_record(self) -> Record:
        """The pin over the ledger's HEAD: how long it is and what it chains to.

        CLASS 4. :meth:`_chain_violations` verifies each link forward from
        :data:`LEDGER_GENESIS` and never compares the resulting head to anything,
        and every head and count check used to sit inside ``if self._released is
        not None``. So a valid PREFIX was always a valid ledger and the last entry
        was always free: editing the last entry and re-stamping only it audited
        clean and re-pointed a nominated model hash, and deleting the tail entry
        re-opened that campaign's slot for a genuine second final nomination.

        This is the other anchor. It is written into every persisted ledger — in
        the unreleased state as well as the released one — and :meth:`audit`
        compares it with the entries actually present. It is a keyless digest
        over a file its editor holds, so it does not stop a ledger rewritten and
        re-pinned end to end; what it stops is the CHEAP edit, which is the one
        an honest mistake and a quietly-opened gate both look like.
        """
        return {
            "n_entries": len(self._entries),
            "entries_digest": self.content_digest(),
            "head_digest": self.head_digest(),
            "highest_sequence": max((int(e.get("sequence") or 0) for e in self._entries), default=-1),
            "released": self._released is not None,
        }

    def _adopt_release(self, released: Mapping[str, Any]) -> None:
        """Take a ``released`` block off disk only if the entries CORROBORATE it.

        CLASS 3. A release is the safety-relevant state in this module: it is
        what lets :meth:`decisions_for_worker` hand entries back to the worker and
        what lets :func:`intention_to_run_table` read a held-out value. It used to
        be believed on sight, so a hand-written ``"released": {...}`` block in an
        unauthenticated file flipped that switch through the CLI.

        **That derivation was circular, and this is the repair.** The pinned
        ``entries_digest`` and ``n_final_records`` it compared against are printed
        into the SAME file by :meth:`to_record`, so corroboration was satisfied by
        copy-paste: a hand-typed ``released`` block built out of the ledger's own
        published fields returned ``release_is_corroborated`` true, audited clean,
        handed every final decision back through :meth:`decisions_for_worker` and
        recorded ``boundary_state: RELEASED``. No keyless digest over a file its
        editor holds can authenticate that file, and a document that agrees with
        itself is self-consistent rather than authentic — the sentence this module
        opens with applies to the ledger verbatim.

        So the release is corroborated against the COMMITTED EXTERNAL ANCHOR: a
        record of this ledger's head kept in a different file, in the repository
        history, which the ledger's editor does not control in the same breath.
        The file-internal checks are kept as a cheap first pass, and the anchor is
        what a release actually has to satisfy. See :data:`ANCHOR_RECORD_KIND` for
        what that buys and what it does not.

        The boundary still takes the RELEASED *state*, deliberately: that is the
        refusing direction, since :meth:`record` refuses while RELEASED. What the
        uncorroborated claim does not get is the release's PRIVILEGES —
        :meth:`decisions_for_worker` refuses and :meth:`audit` reports it.
        """
        self._state = BoundaryState.RELEASED
        self._released = dict(released)
        reasons: list[str] = []
        pinned = str(self._released.get("entries_digest") or "")
        if not pinned:
            reasons.append("it pins no entries_digest, so it addresses nothing")
        elif pinned != self.content_digest():
            reasons.append("its entries_digest does not address the entries beside it")
        counted = self._released.get("n_final_records")
        if counted is None:
            reasons.append("it records no n_final_records")
        elif int(counted) != len(self._entries):
            reasons.append(f"it records {counted} final records and {len(self._entries)} are present")
        reasons.extend(self._external_release_reasons())
        self._release_uncorroborated = reasons

    def _external_release_reasons(self) -> list[str]:
        """Why the COMMITTED anchor does not corroborate this release. Empty when it does.

        The anchor has to cover the release EXACTLY — the same entries, the same
        count, and a head that records the release — where :meth:`_anchor_violations`
        only requires it to certify a prefix. Appending is the ordinary operation
        and must not cost a commit per entry; opening the one-way door is the
        extraordinary one and does.
        """
        anchor = self._anchor
        if anchor is None:
            return [
                "no committed external anchor records a release over this ledger. The "
                "`entries_digest` and `n_final_records` a `released` block pins are printed into "
                "the SAME file by to_record(), so a block built out of them corroborates itself "
                "by copy-paste; the anchor is the one record the ledger's editor does not write"
            ]
        head = dict(anchor.get("head") or {})
        reasons: list[str] = []
        if not head.get("released"):
            reasons.append("the committed external anchor records no release over this ledger")
        if str(head.get("entries_digest") or "") != self.content_digest():
            reasons.append(
                "the committed external anchor does not address the entries beside the claim"
            )
        counted = head.get("n_entries")
        if counted is None or int(counted) != len(self._entries):
            reasons.append(
                f"the committed external anchor records {counted!r} entries and "
                f"{len(self._entries)} are present"
            )
        state = anchor_commit_state(anchor.get("anchor_path"))
        if state != "COMMITTED":
            reasons.append(
                f"the external anchor is {state}: an anchor edited beside the ledger and left "
                f"uncommitted is the same cheap single-pass edit it exists to defeat, and one "
                f"whose commit state cannot be established is not an anchor"
            )
        return reasons

    # -- the boundary itself ----------------------------------------------
    def release(self, *, released_at: str, released_by: str, **provenance: Any) -> Record:
        """Open the boundary, once, after every locked campaign is terminal.

        This is the moment the worker may be handed a final decision. It is a
        one-way door: :meth:`record` refuses afterwards.

        Raises:
            ValueError: ``UNSUPPORTED_CLAIM`` while any locked campaign is still
                open, ``INCONSISTENT_STATUS`` on a second release.
        """
        if self._state is BoundaryState.RELEASED:
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                "the boundary has already been released; a second release would mean a second "
                "pass over the held-out partition",
                field="state",
            )
        disturbed = self._disturbance()
        if disturbed:
            raise _fail(
                ErrorCategory.INCONSISTENT_STATUS,
                f"the ledger no longer matches the head it pinned "
                f"({[v['violation'] for v in disturbed]}), so a release over it would pin a "
                f"chain head over entries that are already not the ones that were written",
                field="entries",
                missing=[str(v["violation"]) for v in disturbed],
            )
        outstanding = self.outstanding()
        if outstanding:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"{len(outstanding)} locked campaign/track slot(s) have no final record yet: "
                f"{outstanding}. All planned campaigns must finish nominations before any "
                f"result can influence a controller or an experimental choice",
                field="campaign_ids",
                missing=outstanding,
            )
        # CLASS 1. `**provenance` used to be merged LAST, over `entries_digest`
        # and `n_final_records` — the two fields that are the ENTIRE post-release
        # tamper evidence. A caller that could type them could pin the chain at a
        # value it chose, and every later edit would audit clean against it.
        _refuse_protected(
            provenance,
            PROTECTED_RELEASE_FIELDS,
            what="the release",
            why=(
                "entries_digest and n_final_records are the whole of the post-release tamper "
                "evidence: a release pinned at a value its caller chose proves nothing about "
                "the entries it released."
            ),
        )
        self._state = BoundaryState.RELEASED
        self._released = {
            # Merged FIRST as well as refused above: the guard is the refusal a
            # caller is told about, and this ordering is what holds if a future
            # edit ever loosens it.
            **provenance,
            "released_at": released_at,
            "released_by": released_by,
            "n_final_records": len(self._entries),
            # The COUNT alone is not tamper evidence: editing an existing entry
            # in place keeps it. The chain head addresses the content of every
            # entry as it stood at the release.
            "entries_digest": self.content_digest(),
            # The RECORDED head as well as the re-derived one. The two differ
            # exactly when the last entry was edited and re-stamped, which is the
            # edit a chain anchored only at genesis cannot see.
            "head_digest": self.head_digest(),
        }
        self._release_uncorroborated = []
        self._repin()
        self._persist()
        return dict(self._released)

    def decisions_for_worker(self) -> list[Record]:
        """Final decisions, and only after the release.

        Raises:
            ValueError: ``TEST_LEAKAGE`` before the boundary opens.
        """
        if self._state is not BoundaryState.RELEASED:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"the worker asked for final decisions while the boundary is {self.state.value} "
                f"({len(self.outstanding())} slot(s) outstanding). Feeding a final decision to an "
                f"evolving process before every nomination is locked is test-guided reselection",
                field="state",
            )
        if self._release_uncorroborated:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"the worker asked for final decisions against a ledger whose release claim is "
                f"NOT corroborated by the entries beside it: "
                f"{'; '.join(self._release_uncorroborated)}. A `released` block is the switch "
                f"that hands entries back to an evolving process, so it is derived from the "
                f"entries rather than believed on sight: a claim typed into an unauthenticated "
                f"file opens nothing",
                field="released",
                missing=list(self._release_uncorroborated),
            )
        return self.entries

    # -- auditing a ledger that came off disk ------------------------------
    def audit(self) -> list[Record]:
        """Re-check the boundary's invariants against the entries as they stand.

        :meth:`record` enforces them as a ledger is written, but the file is what
        a later reader believes, and a file can be hand-edited. This replays the
        invariants over whatever was loaded and names every violation instead of
        raising on the first, so a report can list them all.
        """
        violations: list[Record] = []
        if self._declared is not None:
            expected = {
                "campaign_ids": list(self.campaign_ids),
                "tracks": list(self.tracks),
                "max_per_campaign_per_track": self.max_per_campaign_per_track,
            }
            differing = sorted(k for k, v in expected.items() if self._declared.get(k) != v)
            if differing:
                violations.append(
                    {
                        "violation": "LEDGER_DISAGREES_WITH_LOCK",
                        "campaign_id": None,
                        "detail": (
                            f"the ledger declares { ({k: self._declared.get(k) for k in differing}) } "
                            f"and the lock says { ({k: expected[k] for k in differing}) }. A ledger "
                            f"that sets its own denominator or its own nomination cap is not "
                            f"recording the locked design"
                        ),
                    }
                )
        seen: dict[tuple[str, str], int] = {}
        for index, entry in enumerate(self._entries):
            campaign = str(entry.get("campaign_id"))
            track = str(entry.get("track"))
            if campaign not in self.campaign_ids:
                violations.append(
                    {
                        "violation": "UNREGISTERED_CAMPAIGN",
                        "campaign_id": campaign,
                        "detail": f"{campaign!r} is not one of the locked campaigns {self.campaign_ids}",
                    }
                )
            if str(entry.get("outcome")) not in {member.value for member in CampaignOutcome}:
                violations.append(
                    {
                        "violation": "UNREGISTERED_OUTCOME",
                        "campaign_id": campaign,
                        "detail": f"{entry.get('outcome')!r} is not a registered terminal outcome",
                    }
                )
            key = (campaign, track)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > self.max_per_campaign_per_track:
                violations.append(
                    {
                        "violation": "SECOND_FINAL_NOMINATION",
                        "campaign_id": campaign,
                        "detail": (
                            f"{campaign}/{track} carries {seen[key]} final records and the lock "
                            f"allows {self.max_per_campaign_per_track}"
                        ),
                    }
                )
            if entry.get("outcome") == CampaignOutcome.NOMINATED.value and not entry.get("model_hash"):
                violations.append(
                    {
                        "violation": "NOMINATION_WITHOUT_AN_ADDRESS",
                        "campaign_id": campaign,
                        "detail": (
                            "a nomination that names no model address cannot be shown to be what was scored"
                        ),
                    }
                )
            # CLASS 4. `audit` validated campaign_id and never `track`, while its
            # own one-nomination cap is keyed on (campaign, track) — so an entry
            # filed under an unregistered track took a slot the cap never counted.
            if track not in self.tracks:
                violations.append(
                    {
                        "violation": "UNREGISTERED_TRACK",
                        "campaign_id": campaign,
                        "detail": (
                            f"{track!r} is not one of the registered tracks {self.tracks}, and the "
                            f"one-nomination cap is keyed on (campaign, track): an entry on an "
                            f"unregistered track occupies no slot the cap counts"
                        ),
                    }
                )
            # CLASS 4. `sequence` and `nominating_pid` are written by `record` and
            # are the fresh-process evidence a later reload is checked against,
            # and `audit` inspected neither: `nominating_pid: null` made
            # `reload_nominated_artifact` report fresh_process true for a reload
            # in the very process that wrote the nomination.
            if entry.get("sequence") != index:
                violations.append(
                    {
                        "violation": "ENTRY_SEQUENCE_CHANGED",
                        "campaign_id": campaign,
                        "detail": (
                            f"entry {index} records sequence {entry.get('sequence')!r}. The "
                            f"sequence is the entry's position as it was written, so a sequence "
                            f"that no longer matches the position means entries were reordered, "
                            f"removed or inserted"
                        ),
                    }
                )
            # A ledger off disk can carry a `cost` that `record` would have
            # refused, and the table copies `cost` verbatim into
            # intention-to-run.json. Reported here so the CLI refuses with
            # BOUNDARY_VIOLATION rather than raising out of the table.
            try:
                _refuse_cost_shape(entry.get("cost"), campaign_id=campaign)
            except ContractError as error:
                violations.append(
                    {
                        "violation": "ENTRY_COST_IS_NOT_A_FLAT_RESOURCE_MAPPING",
                        "campaign_id": campaign,
                        "detail": (
                            f"entry {index}'s cost is not a flat mapping over the registered "
                            f"resources: {error.missing or 'wrong type'}. cost is the one mapping "
                            f"on a nomination that may carry numbers, and it is copied verbatim "
                            f"into the intention-to-run table, so it is the other route a "
                            f"held-out number travels by. The values are not reported here"
                        ),
                    }
                )
            if entry.get("nominating_pid") is None:
                violations.append(
                    {
                        "violation": "NOMINATION_WITHOUT_PROCESS_EVIDENCE",
                        "campaign_id": campaign,
                        "detail": (
                            "the entry carries no nominating_pid. That field is what a later "
                            "reload is checked against, and a nomination with none makes a "
                            "reload in the nominating process report itself as a fresh one"
                        ),
                    }
                )
        violations.extend(self._head_violations())
        violations.extend(self._anchor_violations())
        violations.extend(self._chain_violations())
        if self._released is not None:
            recorded = self._released.get("n_final_records")
            if recorded is not None and int(recorded) != len(self._entries):
                violations.append(
                    {
                        "violation": "ENTRIES_CHANGED_AFTER_RELEASE",
                        "campaign_id": None,
                        "detail": (
                            f"the release recorded {recorded} final records and the ledger now "
                            f"carries {len(self._entries)}. Entries were written after the "
                            f"held-out feedback was released"
                        ),
                    }
                )
            released_digest = str(self._released.get("entries_digest") or "")
            if not released_digest:
                violations.append(
                    {
                        "violation": "RELEASE_WITHOUT_AN_ENTRIES_DIGEST",
                        "campaign_id": None,
                        "detail": (
                            "the release recorded no content address over the entries it "
                            "released, so the entries now on file cannot be shown to be the ones "
                            "that were released. A release that records only how many records "
                            "there were is passed unchanged by editing one of them in place"
                        ),
                    }
                )
            elif released_digest != self.content_digest():
                violations.append(
                    {
                        "violation": "ENTRIES_CHANGED_AFTER_RELEASE",
                        "campaign_id": None,
                        "detail": (
                            f"the release pinned the entries at {released_digest!r} and they now "
                            f"hash to {self.content_digest()!r} with the SAME record count. An entry "
                            f"was edited in place after the held-out feedback was released"
                        ),
                    }
                )
            if self.outstanding():
                violations.append(
                    {
                        "violation": "RELEASED_WHILE_INCOMPLETE",
                        "campaign_id": None,
                        "detail": f"released with {self.outstanding()} still outstanding",
                    }
                )
        if self._release_uncorroborated:
            violations.append(
                {
                    "violation": "UNCORROBORATED_RELEASE",
                    "campaign_id": None,
                    "detail": (
                        f"the ledger carries a `released` block that the entries beside it do "
                        f"not corroborate: {'; '.join(self._release_uncorroborated)}. A release "
                        f"is the switch that hands entries back to an evolving process and lets "
                        f"a held-out value be read into a delta, so it is DERIVED from the "
                        f"entries rather than accepted from the file that claims it"
                    ),
                }
            )
        return violations

    def _head_violations(self) -> list[Record]:
        """Compare the ledger's declared HEAD with the entries actually present.

        CLASS 4, the other end of the chain. :meth:`_chain_violations` walks
        forward from :data:`LEDGER_GENESIS` and stops, so a valid PREFIX of a
        valid ledger is itself a valid ledger and the last entry is unanchored.
        These are the checks that close that, and they run in the UNRELEASED
        state too — the previous round put every head and count check inside
        ``if self._released is not None``.

        What each one catches. ``LEDGER_TRUNCATED``: deleting the tail entry, which
        re-opens that campaign's slot so ``record`` writes a genuine second final
        nomination. ``LEDGER_HEAD_MOVED``: editing the last entry and re-stamping
        only it, which re-points a nominated model hash and breaks no link because
        there is no link after it. ``RELEASE_RECORD_REMOVED``: setting
        ``"released": null``, which erases the ``entries_digest`` pin and re-opens
        ``record`` after a release. ``LEDGER_HAS_NO_HEAD_PIN``: deleting the pin
        rather than forging it. ``LEDGER_HEAD_PIN_INCOMPLETE``: **nulling a field
        of it instead**, which is how this re-opened after the previous round —
        a missing pin was reported only for an EMPTY mapping and every comparison
        was skipped when the declared value was ``None``, so ``"head": {"released":
        false}`` was a "present" pin with nothing to compare. A pin is now all of
        :data:`HEAD_PIN_FIELDS` or it is not a pin.
        """
        if self._declared_head is None:  # built here, in this process; nothing to compare
            return []
        declared = self._declared_head
        if not declared:
            return [
                {
                    "violation": "LEDGER_HAS_NO_HEAD_PIN",
                    "campaign_id": None,
                    "detail": (
                        "the ledger carries no head pin, so its LENGTH is not tamper evident: a "
                        "valid prefix of a valid ledger is otherwise a valid ledger, and deleting "
                        "the tail entry re-opens that campaign's slot for a second final "
                        "nomination. Every ledger this module writes carries one"
                    ),
                }
            ]
        blank = [field for field in HEAD_PIN_FIELDS if declared.get(field) is None]
        if blank:
            return [
                {
                    "violation": "LEDGER_HEAD_PIN_INCOMPLETE",
                    "campaign_id": None,
                    "detail": (
                        f"the ledger's head pin declares no {blank}. A pin with a null or missing "
                        f"field is NOT a pin: every comparison against it is skipped, so nulling "
                        f"one field is exactly as good as deleting the whole pin and tail deletion "
                        f"becomes invisible again. The pin is all of {list(HEAD_PIN_FIELDS)} or it "
                        f"is nothing"
                    ),
                }
            ]
        actual = self.head_record()
        violations: list[Record] = []
        # `blank` above guarantees every pin field is present and not None.
        declared_n = int(declared["n_entries"])
        if declared_n != len(self._entries):
            violations.append(
                {
                    "violation": (
                        "LEDGER_TRUNCATED" if declared_n > len(self._entries) else "LEDGER_EXTENDED"
                    ),
                    "campaign_id": None,
                    "detail": (
                        f"the ledger pins {declared_n} entries and {len(self._entries)} are "
                        f"present. Deleting the tail entry re-opens that campaign's slot, so the "
                        f"ledger's length is pinned as well as its content"
                    ),
                }
            )
        for field, why in (
            (
                "entries_digest",
                "the chain re-derived from every entry's current content no longer matches what "
                "the ledger pinned over it",
            ),
            (
                "head_digest",
                "the LAST entry's recorded address is not the one the ledger pinned: the tail "
                "was edited and re-stamped, which breaks no link because there is no link after it",
            ),
            (
                "highest_sequence",
                "the highest sequence written no longer matches the pin, so entries were removed "
                "or inserted",
            ),
        ):
            if declared.get(field) != actual.get(field):
                violations.append(
                    {
                        "violation": "LEDGER_HEAD_MOVED",
                        "campaign_id": None,
                        "detail": (
                            f"the ledger pins {field} {declared.get(field)!r} and it is now "
                            f"{actual.get(field)!r}: {why}"
                        ),
                    }
                )
        if declared.get("released") and self._released is None:
            violations.append(
                {
                    "violation": "RELEASE_RECORD_REMOVED",
                    "campaign_id": None,
                    "detail": (
                        "the ledger pins a release and carries none. Setting `released` to null "
                        "erases the entries_digest the release pinned over the entries and "
                        "re-opens `record` after held-out feedback was already handed back"
                    ),
                }
            )
        # The other direction, which was never checked: the pin says the ledger was
        # not released and a `released` block is sitting beside it. That is a
        # release APPENDED to a ledger the pin was taken over before it, which is
        # precisely the hand-typed release block.
        if not declared.get("released") and self._released is not None:
            violations.append(
                {
                    "violation": "RELEASE_RECORD_ADDED",
                    "campaign_id": None,
                    "detail": (
                        "the ledger pins an UNRELEASED head and carries a `released` block. The "
                        "block was added after the pin was taken, which is what a release typed "
                        "into the file by hand looks like: a release this module wrote would have "
                        "advanced the pin with it"
                    ),
                }
            )
        return violations

    def _anchor_violations(self) -> list[Record]:
        """Compare a ledger that came off disk with the COMMITTED external anchor.

        Every round before this one added another check the ledger satisfied out
        of its own published fields, and the red team satisfied each one by
        copy-paste. This is the first check whose reference is not in the file
        being checked: the anchor is a separate, committed record, and the ledger's
        editor does not control it in the same breath.

        The anchor certifies a PREFIX, deliberately. Appending through
        :meth:`record` is the legitimate operation and must not require a commit
        per entry, so a ledger longer than the anchor is fine as long as its first
        ``n_entries`` entries chain to the anchored digest. What the anchor has to
        cover exactly is the RELEASE — the one-way door — and that is enforced in
        :meth:`_adopt_release` rather than here.

        See :data:`ANCHOR_RECORD_KIND` for what this does and does not buy.
        """
        if self._declared_head is None:  # built here; it is not a file yet
            return []
        anchor = self._anchor
        if anchor is None:
            return [
                {
                    "violation": "LEDGER_NOT_EXTERNALLY_ANCHORED",
                    "campaign_id": None,
                    "detail": (
                        f"no committed anchor record for this protocol was found at "
                        f"{REGISTERED_ANCHOR_PATH}. A ledger's own digests are printed into the "
                        f"ledger by the same writer, so they corroborate it only by copy-paste; "
                        f"the anchor is the one reference that is not in the file being checked"
                    ),
                }
            ]
        head = dict(anchor.get("head") or {})
        blank = [field for field in HEAD_PIN_FIELDS if head.get(field) is None]
        if blank:
            return [
                {
                    "violation": "ANCHOR_HEAD_PIN_INCOMPLETE",
                    "campaign_id": None,
                    "detail": (
                        f"the committed anchor declares no {blank}, so it anchors nothing. An "
                        f"anchor is all of {list(HEAD_PIN_FIELDS)} or it is not an anchor"
                    ),
                }
            ]
        anchored_n = int(head["n_entries"])
        if anchored_n > len(self._entries):
            return [
                {
                    "violation": "LEDGER_BEHIND_ANCHOR",
                    "campaign_id": None,
                    "detail": (
                        f"the committed anchor records {anchored_n} entries and the ledger carries "
                        f"{len(self._entries)}. Entries were removed below the point the repository "
                        f"has a record of, which no edit to the ledger alone can conceal"
                    ),
                }
            ]
        prefix = LEDGER_GENESIS
        for entry in self._entries[:anchored_n]:
            prefix = _entry_digest(entry, prefix)
        if prefix != str(head["entries_digest"]):
            return [
                {
                    "violation": "LEDGER_DISAGREES_WITH_ANCHOR",
                    "campaign_id": None,
                    "detail": (
                        f"the ledger's first {anchored_n} entries chain to {prefix!r} and the "
                        f"committed anchor records {head['entries_digest']!r}. A ledger rewritten "
                        f"and re-pinned end to end still audits clean against its own fields; it "
                        f"does not audit clean against a record it did not write"
                    ),
                }
            ]
        return []

    def _disturbance(self) -> list[Record]:
        """Everything that says this ledger is not the one that was written.

        :meth:`record` and :meth:`release` both refuse over a disturbed ledger
        rather than merely reporting it, because detection was not enough for
        truncation: the slot still LOOKS open, so appending has to be refused.
        """
        return self._head_violations() + self._anchor_violations()

    def _chain_violations(self) -> list[Record]:
        """Replay the entry chain and name every entry whose CONTENT has moved.

        The defect this exists for: a release that records only ``n_final_records``
        detects an *appended* entry and nothing else, so editing an existing entry
        in place — a second nomination that reuses the first one's row, or a model
        hash re-pointed at the artifact that is actually on disk — keeps the count
        and audits completely clean.

        What it does and does not buy. An entry edited in place no longer matches
        its own recorded address; an entry edited and re-stamped no longer links to
        the entry after it; a ledger whose entries carry no addresses at all is
        reported as not tamper evident, so stripping the field is not a way out.
        A whole ledger rewritten and re-chained end to end by someone holding the
        file is NOT detectable here — no keyless digest can — which is why the
        release pins the head and ``confirm`` records it.
        """
        violations: list[Record] = []
        previous = LEDGER_GENESIS
        unstamped: list[str] = []
        for index, entry in enumerate(self._entries):
            campaign = str(entry.get("campaign_id"))
            expected = _entry_digest(entry, previous)
            recorded = str(entry.get("entry_digest") or "")
            if not recorded:
                unstamped.append(f"{campaign}[{index}]")
                previous = expected
                continue
            linked = str(entry.get("previous_digest") or "")
            if linked != previous:
                violations.append(
                    {
                        "violation": "LEDGER_CHAIN_BROKEN",
                        "campaign_id": campaign,
                        "detail": (
                            f"entry {index} links to {linked!r} and the entry before it addresses "
                            f"to {previous!r}. An entry was edited and re-stamped, or the order "
                            f"was changed: the ledger is no longer the sequence it recorded"
                        ),
                    }
                )
            elif recorded != expected:
                violations.append(
                    {
                        "violation": "ENTRY_CONTENT_CHANGED",
                        "campaign_id": campaign,
                        "detail": (
                            f"entry {index} carries the address {recorded!r} and its content now "
                            f"hashes to {expected!r}. The row was edited in place after it was "
                            f"written, which keeps the record COUNT unchanged and is how a second "
                            f"nomination, or a model hash re-pointed at whatever is on disk, "
                            f"passes an audit that only counts"
                        ),
                    }
                )
            previous = recorded
        if unstamped:
            violations.append(
                {
                    "violation": "LEDGER_NOT_TAMPER_EVIDENT",
                    "campaign_id": None,
                    "detail": (
                        f"{len(unstamped)} of {len(self._entries)} entries carry no content "
                        f"address ({unstamped}), so nothing about them can be shown to be "
                        f"unedited. A ledger written by hand, or one whose addresses were "
                        f"stripped, is not evidence of what was nominated before any test "
                        f"feedback existed"
                    ),
                }
            )
        return violations

    # -- persistence -------------------------------------------------------
    def to_record(self) -> Record:
        return {
            "contract_version": "1.0.0",
            "record_kind": "confirmation_nomination_ledger",
            "state": self.state.value,
            "campaign_ids": list(self.campaign_ids),
            "tracks": list(self.tracks),
            "max_per_campaign_per_track": self.max_per_campaign_per_track,
            "outstanding": self.outstanding(),
            "entries_digest": self.content_digest(),
            # CLASS 4. The ledger is pinned at its HEAD as well as chained from
            # genesis, and the pin is written in the unreleased state too. See
            # `head_record` for what it does and does not buy.
            "head": self.head_record(),
            "entries": self.entries,
            "released": dict(self._released) if self._released else None,
        }

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_record(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, path: Path | str | None = None) -> NominationBoundary:
        """Reconstruct a ledger from disk, so an interrupted run resumes rather than duplicates."""
        boundary = cls(
            list(record.get("campaign_ids") or []),
            tracks=list(record.get("tracks") or ("A",)),
            max_per_campaign_per_track=int(record.get("max_per_campaign_per_track", 1)),
            path=path,
        )
        boundary._entries = [dict(e) for e in record.get("entries") or []]
        boundary._declared_head = dict(record.get("head") or {})
        # The anchor is loaded BEFORE the release is adopted, because the release
        # is corroborated against it rather than against the file it is written in.
        boundary._anchor = load_external_anchor()
        released = record.get("released")
        if released:
            boundary._adopt_release(released)
        return boundary

    @classmethod
    def resume_under_protocol(
        cls, protocol: Mapping[str, Any], record: Mapping[str, Any], *, path: Path | str | None = None
    ) -> NominationBoundary:
        """Resume a ledger, taking the DESIGN from the lock and only the entries from the file.

        :meth:`from_record` believes the file about its own campaign list and its
        own one-nomination cap, which is right for a round trip and wrong for a
        ledger that may have been edited: a file that declares
        ``max_per_campaign_per_track: 99`` would otherwise audit clean. Here the
        denominator and the cap come from the protocol, the entries come from the
        file, and any disagreement between the two is itself recorded as a
        violation by :meth:`audit`.
        """
        boundary = cls.from_protocol(protocol, path=path)
        entries = record.get("entries") or []
        if not isinstance(entries, list) or any(not isinstance(e, Mapping) for e in entries):
            raise _fail(
                ErrorCategory.INVALID_TYPE,
                "the nomination ledger's `entries` is not a list of records, so it cannot be "
                "read as a statement of what was nominated before any test feedback existed",
                field="entries",
            )
        boundary._entries = [dict(e) for e in entries]
        boundary._declared_head = dict(record.get("head") or {})
        boundary._anchor = load_external_anchor()
        released = record.get("released")
        if released:
            boundary._adopt_release(released)
        boundary._declared = {
            "campaign_ids": list(record.get("campaign_ids") or []),
            "tracks": list(record.get("tracks") or []),
            "max_per_campaign_per_track": record.get("max_per_campaign_per_track"),
        }
        return boundary

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any], **options: Any) -> NominationBoundary:
        """Build the ledger the lock describes: its seeds, its one-nomination cap.

        The cap is read out of the protocol, which is why the protocol has to have
        been authenticated against its registered address first: a forged document
        supplying ``max_per_campaign_per_registered_track: 3`` would otherwise set
        the cap it is supposed to be constrained by.
        :func:`authenticate_protocol` is that check;
        :func:`confirmation_readiness` refuses independently of it.

        The TRACK set comes out of the lock too, and it is not a parameter. It is
        half the ``(campaign_id, track)`` slot key the cap is keyed on, and it was
        reachable through ``**options`` until this round: ``from_protocol(protocol,
        tracks=[])`` was accepted, :meth:`outstanding` went empty,
        :meth:`is_complete` returned ``True`` over ZERO entries,
        ``NOMINATIONS_COMPLETE`` reported met and :meth:`release` succeeded with
        ``n_final_records: 0``.

        Raises:
            ValueError: ``INVALID_VALUE`` when the protocol's two statements of
                the cap disagree. A document that says 1 in one field and 3 in the
                other has no single cap to enforce.
        """
        seeds = list(protocol.get("campaign_seeds") or [])
        rule = dict(protocol.get("nomination_rule") or {})
        per_track = rule.get("max_per_campaign_per_registered_track")
        overall = protocol.get("max_final_nominations")
        if per_track is not None and overall is not None and int(per_track) != int(overall):
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"the protocol declares nomination_rule.max_per_campaign_per_registered_track "
                f"{per_track!r} and max_final_nominations {overall!r}. A design that states its "
                f"one-nomination cap twice and differently has no cap: the ledger would enforce "
                f"whichever field it happened to read first",
                field="nomination_rule.max_per_campaign_per_registered_track",
            )
        # CLASS 1. `**options` reaches the constructor, where `campaign_ids` is
        # the denominator and `max_per_campaign_per_track` is the cap this
        # classmethod exists to read OUT of the authenticated lock. A caller that
        # could pass either would be setting the design it is meant to be bound by.
        _refuse_protected(
            options,
            PROTECTED_LEDGER_DESIGN_FIELDS,
            what="the lock",
            why=(
                "the denominator and the one-nomination cap are read out of the authenticated "
                "protocol, which is the whole reason this constructor exists."
            ),
        )
        # The track whose registered endpoint actually reads the held-out
        # partition. Under zak-c9r-2 that is the demoted Track A secondary, so this
        # derives exactly ("A",) and the locked denominator is unchanged. It is
        # STATED rather than defaulted: a value that can be defaulted in silently
        # is a value that can be emptied in silently, and the constructor refuses
        # an empty track set either way.
        demoted = dict((protocol.get("primary_endpoint") or {}).get("superseded_track_a_endpoint") or {})
        held_out_track = str(demoted.get("track") or "").strip()
        return cls(
            [f"campaign-seed-{seed}" for seed in seeds],
            tracks=(held_out_track,) if held_out_track else CONFIRMATION_LEDGER_TRACKS,
            max_per_campaign_per_track=int(per_track or overall or 1),
            **options,
        )


def ledger_anchor_record(
    boundary: NominationBoundary, *, anchored_at: str, anchored_by: str
) -> Record:
    """The anchor document to write to :data:`REGISTERED_ANCHOR_PATH` and COMMIT.

    Deliberately not wired to a command-line flag. Minting an anchor has to cost a
    reviewed source-control change, because that cost is the entire difference
    between this and the self-referential digests every previous round added — a
    one-command mint beside the ledger would hand the forger both halves.

    The operational rule this implies, stated plainly: **appending is free and
    releasing is not.** A ledger may run ahead of its anchor while campaigns
    nominate, because the anchor certifies a prefix. Before the boundary is
    released, the anchor must be re-recorded over the full ledger and committed,
    or the release is uncorroborated from then on.
    """
    return {
        "contract_version": "1.0.0",
        "record_kind": ANCHOR_RECORD_KIND,
        "protocol_version": REGISTERED_PROTOCOL_VERSION,
        "protocol_hash": REGISTERED_PROTOCOL_HASH,
        "head": boundary.head_record(),
        "anchored_at": anchored_at,
        "anchored_by": anchored_by,
        "what_this_is": (
            "the nomination ledger's head digest, recorded in a committed file that is NOT the "
            "ledger. audit(), record() and release() compare the ledger against this rather than "
            "against fields the ledger prints about itself"
        ),
        "what_this_does_not_buy": (
            "it is not authentication. It raises the cost of a cheap partial edit - one file, no "
            "commit - to a cost that also requires writing and committing a matching anchor, "
            "which is a reviewable source-control change rather than a data edit. It does not "
            "stop someone who edits both"
        ),
    }


# ---------------------------------------------------------------------------
# artifact reload
# ---------------------------------------------------------------------------
def measure_model_address(artifact: Mapping[str, Any]) -> str:
    """Derive an artifact's content address from its OWN bytes.

    The defect this closes: ``reload_nominated_artifact`` took
    ``measured_model_hash`` from its caller and nothing anywhere in the programme
    derived a model address from bytes, so "the bytes on disk now hash to X" was
    a sentence the caller wrote. A hash a caller types is the caller's claim, and
    comparing a nomination against the caller's claim is the circular
    self-comparison this module refuses everywhere else.

    The address is the same one :func:`~.registry.resolve_artifact` reports as
    ``version_hash``: the canonical bytes of the *validated* artifact record,
    which names every component's content address. Move any component and this
    address moves with it.
    """
    from ...util import canonical_json, hash_bytes
    from .contracts import EvolutionArtifact, validate_record

    validated = validate_record(EvolutionArtifact, artifact)
    return hash_bytes(canonical_json(validated).encode("utf-8"))


def reload_nominated_artifact(
    nomination: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    target_hash: str,
    measured_model_hash: str | None = None,
    training_mode: str = WEIGHTS_ONLY,
    resolver: Any = None,
    store: Any = None,
    require_fresh_process: bool = True,
) -> Record:
    """Reload exactly the nominated artifact, in a fresh process, from its address.

    Three refusals, and each one corresponds to a way a confirmation quietly
    stops confirming the thing it nominated:

    * ``HASH_MISMATCH`` — the model's measured content address is not the one the
      nomination named. Changing a model hash **invalidates** the nomination; it
      does not update it. The address is **measured here**, by
      :func:`measure_model_address`, from the artifact record's own canonical
      bytes — it is not taken from the caller.
    * ``UNSUPPORTED_CLAIM`` — the caller supplied a ``measured_model_hash`` that
      disagrees with the measurement. A typed address is a claim about bytes, not
      a measurement of them, and a claim that disagrees with the bytes is refused
      rather than preferred to them.
    * ``UNSUPPORTED_CLAIM`` — ``training_mode`` is ``RETRAINED``. A weights-only
      predictor uses fixed weights. Refitting produces a different artifact, and
      calling that a replay of the nominated one is the substitution this check
      exists to stop. Training repeatability is a separate, clearly labelled
      experiment.
    * ``UNSUPPORTED_CLAIM`` — the reload is happening in the same process that
      wrote the nomination. A reload that never left the nominating process is
      not evidence that the artifact reloads.

    Args:
        nomination: the ledger entry, carrying ``model_hash`` and ``nominating_pid``.
        artifact: the ``EvolutionArtifact`` record to resolve.
        target_hash: the target manifest this resolution is for.
        measured_model_hash: optional cross-check. The address is measured from
            the artifact's bytes either way; supplying one that disagrees with
            the measurement is refused, and supplying one that agrees changes
            nothing. It may not stand in for the measurement.
        training_mode: :data:`WEIGHTS_ONLY` or :data:`RETRAINED`.
        resolver: defaults to :func:`..target.registry.resolve_artifact`.
        store: optional content-addressed store; without one the resolution is
            structural and says so.
        require_fresh_process: compare pids. Only a test harness turns this off.

    Returns:
        The resolution record, with the reload evidence attached.

    Raises:
        ValueError: as above, plus whatever the artifact contract raises.
    """
    if str(nomination.get("outcome") or CampaignOutcome.NOMINATED.value) != CampaignOutcome.NOMINATED.value:
        raise _fail(
            ErrorCategory.INCONSISTENT_STATUS,
            f"campaign {nomination.get('campaign_id')!r} terminated "
            f"{nomination.get('outcome')!r} and has no nominee to reload. It stays in the "
            f"denominator; it does not acquire an artifact",
            field="outcome",
        )
    declared = str(nomination.get("model_hash") or "")
    if not declared:
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            "the nomination names no model address, so nothing can be shown to be the "
            "artifact that was nominated",
            field="model_hash",
            missing=["model_hash"],
        )
    measured = measure_model_address(artifact)
    if measured_model_hash is not None and str(measured_model_hash) != measured:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            f"the caller supplied measured_model_hash {measured_model_hash!r} and the artifact's "
            f"own bytes hash to {measured!r}. An address typed by the caller is a CLAIM about the "
            f"bytes, not a measurement of them; this reload measures the address itself and will "
            f"not accept a value in its place",
            field="measured_model_hash",
        )
    if declared != measured:
        raise _fail(
            ErrorCategory.HASH_MISMATCH,
            f"campaign {nomination.get('campaign_id')!r} nominated model {declared!r} and the "
            f"bytes on disk now hash to {measured!r}. The nomination is INVALIDATED: "
            f"a moved model is a different artifact and was never nominated. It is not "
            f"re-pointed at the new address",
            field="model_hash",
        )
    if training_mode == RETRAINED:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            "the nominated artifact was RETRAINED rather than reloaded. A weights-only "
            "predictor uses FIXED weights; refitting produces a different artifact and cannot "
            "substitute for a failed reload. Register the refit as its own clearly labelled "
            "training-repeatability experiment instead",
            field="training_mode",
        )
    if training_mode != WEIGHTS_ONLY:
        raise _fail(
            ErrorCategory.INVALID_ENUM,
            f"{training_mode!r} is not a registered training mode; expected {WEIGHTS_ONLY!r} "
            f"or {RETRAINED!r}",
            field="training_mode",
        )
    reload_pid = os.getpid()
    nominating_pid = nomination.get("nominating_pid")
    if require_fresh_process and nominating_pid is not None and int(nominating_pid) == reload_pid:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            f"the reload is running in pid {reload_pid}, the same process that wrote the "
            f"nomination. A reload that never left the nominating process shows only that the "
            f"object is still in memory, not that the artifact reloads from its address",
            field="nominating_pid",
        )

    resolve = resolver
    if resolve is None:  # pragma: no cover - the real resolver, exercised by the CLI
        from .registry import resolve_artifact

        resolve = resolve_artifact

    resolution = dict(resolve(artifact, target_hash, store=store))
    reported = str(resolution.get("version_hash") or "")
    if reported and reported != measured:
        raise _fail(
            ErrorCategory.HASH_MISMATCH,
            f"the resolver reports version_hash {reported!r} for this artifact and its canonical "
            f"bytes hash to {measured!r}. Two derivations of one address disagree, so neither can "
            f"be reported as the artifact's address",
            field="version_hash",
        )
    resolution["reload_evidence"] = {
        "class": "C2_PER_ARTIFACT",
        "nominated_model_hash": declared,
        "measured_model_hash": measured,
        "model_address_measured_from": (
            "the artifact record's own canonical bytes, by confirmation.measure_model_address - "
            "NOT supplied by the caller"
        ),
        "model_address_cross_checked_against_resolver": bool(reported),
        "caller_supplied_model_hash": measured_model_hash,
        "training_mode": training_mode,
        "retraining_is_a_different_artifact": True,
        "nominating_pid": nominating_pid,
        "reload_pid": reload_pid,
        "fresh_process": nominating_pid is None or int(nominating_pid) != reload_pid,
        "references_verified": resolution.get("references_verified", False),
    }
    return resolution


# ---------------------------------------------------------------------------
# intention to run
# ---------------------------------------------------------------------------
def intention_to_run_table(
    protocol: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    comparator_value: float | None = None,
    boundary: NominationBoundary | None = None,
    boundary_state: BoundaryState | str | None = None,
) -> Record:
    """Build the per-campaign table from the *locked* seeds, never from the survivors.

    The denominator is ``protocol["campaign_seeds"]``. A locked campaign with no
    record at all is reported as ``ABSENT`` with a missing delta — it is still
    counted, and it is not silently dropped. ``delta_i = 0.0`` is a statement
    about the process delivering nothing; a failed *evaluation* has a missing
    value and a recorded cost, and the two are kept apart.

    The boundary state decides whether a held-out value may be read into a delta
    at all, which makes it the safety-relevant state here.

    **CLASS 3.** It used to be CALLER-TYPED and cross-checked against nothing, so
    ``boundary_state="RELEASED"`` on a genuinely OPEN ledger produced measured
    deltas and, downstream, a C3 interval out of held-out values. A safety-relevant
    state is DERIVED, never accepted: pass the ``boundary`` these records came
    from and the state is read off it. ``boundary_state`` survives only as a
    cross-check — supplying one that disagrees with the boundary is refused — and
    on its own it may never be ``RELEASED``, because a caller typing the word is
    exactly the claim this refuses.

    A held-out value in a record while the boundary is not ``RELEASED`` is a test
    result that exists before the boundary opened, and the table refuses to read
    it rather than reporting it as a measured delta. The scan is RECURSIVE: the
    previous round read only top-level keys, so a value nested inside ``cost``
    walked through it. Only the campaign and the path are named; no value is ever
    reported by the refusal.

    Raises:
        ValueError: ``INVALID_VALUE`` for a record naming an unlocked campaign —
            adding a campaign after the fact is how a denominator grows a winner;
            ``TEST_LEAKAGE`` for a held-out value before the release;
            ``INVALID_ENUM`` for an unregistered boundary state;
            ``UNSUPPORTED_CLAIM`` for a caller-typed ``RELEASED`` with no
            boundary to derive it from, or one that disagrees with the boundary.
    """
    typed: BoundaryState | None = None
    if boundary_state is not None:
        try:
            typed = BoundaryState(str(boundary_state))
        except ValueError:
            raise _fail(
                ErrorCategory.INVALID_ENUM,
                f"{boundary_state!r} is not a registered boundary state; the vocabulary is "
                f"{sorted(m.value for m in BoundaryState)}",
                field="boundary_state",
            ) from None
    if boundary is not None:
        state = boundary.state
        if typed is not None and typed is not state:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"the caller says the boundary is {typed.value} and the boundary itself is "
                f"{state.value}. The state that decides whether a held-out value may be read is "
                f"derived from the ledger, not accepted from its caller; a typed state that "
                f"disagrees with the ledger is the claim this refuses",
                field="boundary_state",
            )
        if state is BoundaryState.RELEASED and not boundary.release_is_corroborated:
            raise _fail(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"the ledger claims a release the entries beside it do not corroborate: "
                f"{'; '.join(boundary.uncorroborated_release_reasons)}. A release is what makes a "
                f"held-out value readable into a delta, so an uncorroborated claim does not "
                f"open it",
                field="released",
            )
    elif typed is BoundaryState.RELEASED:
        raise _fail(
            ErrorCategory.UNSUPPORTED_CLAIM,
            "boundary_state='RELEASED' was typed by the caller with no boundary to derive it "
            "from. RELEASED is the state that lets a held-out value be read into a measured "
            "delta, so it is derived from the ledger that actually released — pass `boundary=`. "
            "A caller that can type the word can turn an OPEN ledger's nominations into "
            "measured deltas and a C3 process interval out of held-out values",
            field="boundary_state",
        )
    else:
        state = typed if typed is not None else BoundaryState.OPEN
    seeds = list(protocol.get("campaign_seeds") or [])
    locked = [f"campaign-seed-{seed}" for seed in seeds]
    by_campaign = {str(r.get("campaign_id")): dict(r) for r in records}
    if state is not BoundaryState.RELEASED:
        # CLASS 2. Counts, campaign ids and PATHS only: a refusal that printed
        # the value would itself be the leak it is refusing. The scan walks the
        # whole record to any depth — the previous round tested only top-level
        # keys, so `cost={"held_out_value": 0.31}` was copied verbatim into both
        # confirmation.json and intention-to-run.json by a table that went on
        # reporting the boundary as unreleased.
        carrying = sorted(
            f"{campaign}:{path}"
            for campaign, supplied_record in by_campaign.items()
            for path in _held_out_name_paths(supplied_record)
        )
        if carrying:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"{len(carrying)} record(s) carry a held-out value while the boundary is "
                f"{state.value}: {carrying}. A held-out value that exists before the release is a "
                f"test result the nominations were not frozen ahead of, so it is refused rather "
                f"than read into a delta. The value itself is not reported here",
                field="held_out_value",
                missing=carrying,
            )
    # CLASS 2, the other half. The table copies `cost` VERBATIM into
    # intention-to-run.json, so the same structural rule the ledger enforces when
    # an entry is written is enforced again on whatever is handed to the table —
    # records reach here off disk, and a file can be hand-edited after the fact.
    for campaign, supplied_record in by_campaign.items():
        _refuse_cost_shape(supplied_record.get("cost"), campaign_id=campaign)
        # CLASS 2, the structural half, applied to records that reach this table
        # OFF DISK. The name scan above is the second layer and it is skipped
        # once the boundary is released; this one is not skipped, because a
        # held-out measurement is a NUMBER and the name it travels under is the
        # attacker's choice. A record carries a number in the places `record()`
        # writes one and nowhere else.
        allowed = {"development_value", "nominating_pid", "sequence"}
        if state is BoundaryState.RELEASED:
            # The one number a release legitimately adds, and only after it.
            allowed.add("held_out_value")
        # `not path.startswith("cost.")` used to stand where COST_MAGNITUDE_PATHS
        # stands now. A prefix exclusion admits `cost.<anything>`, so a verbatim
        # ledger cost channel rode into intention-to-run.json — and from there
        # into a record asserting the partition was NOT READ — under any key its
        # author chose. Only the registered resource paths are admitted now, and
        # the branch-status record's own allowlist admits none of them at all.
        stray = [
            path
            for path in _numeric_payload_paths(supplied_record)
            if path not in allowed and path not in COST_MAGNITUDE_PATHS
        ]
        if stray:
            raise _fail(
                ErrorCategory.TEST_LEAKAGE,
                f"campaign {campaign!r} hands this table {len(stray)} unexpected numeric "
                f"value(s) at {stray}. A record reaching the table came off a file that can be "
                f"hand-edited, and the table copies what it is given into intention-to-run.json, "
                f"so the same structural rule the ledger enforces when an entry is WRITTEN is "
                f"enforced again on whatever is READ. The values are not reported here",
                field=stray[0],
                missing=stray,
            )
    unlocked = sorted(set(by_campaign) - set(locked))
    if unlocked:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"{unlocked} are not locked campaigns {locked}; a campaign registered after the "
            f"design was locked is a separate experiment, not an extra row in this table",
            field="campaign_id",
            missing=unlocked,
        )

    rows: list[Record] = []
    for campaign in locked:
        record = by_campaign.get(campaign)
        if record is None:
            rows.append(
                {
                    "campaign_id": campaign,
                    "outcome": "ABSENT",
                    "delta": None,
                    "delta_is_missing": True,
                    "in_denominator": True,
                    "why": "the locked campaign produced no terminal record at all",
                    "cost": None,
                }
            )
            continue
        outcome = str(record.get("outcome"))
        delta: float | None
        why: str
        if outcome == CampaignOutcome.NOMINATED.value:
            value = record.get("held_out_value")
            if value is None:
                delta, why = None, "nominated, and its held-out value has not been measured"
            elif comparator_value is None:
                delta, why = None, "nominated and scored, but no comparator value was supplied"
            else:
                delta = float(value) - float(comparator_value)
                why = "measured paired difference against the required comparator"
        elif outcome == CampaignOutcome.EVALUATION_FAILED.value:
            delta = None
            why = (
                "the EVALUATION failed: the scientific value is MISSING and the cost is "
                "recorded. It is not a zero"
            )
        elif outcome in ZERO_DELTA_OUTCOMES:
            delta = 0.0
            why = (
                "the process delivered no improvement; delta = 0.0 is that statement, not a "
                "measured metric of 0"
            )
        else:  # pragma: no cover - the enum is closed
            delta, why = None, f"unregistered outcome {outcome!r}"
        rows.append(
            {
                "campaign_id": campaign,
                "outcome": outcome,
                "delta": delta,
                "delta_is_missing": delta is None,
                "in_denominator": True,
                "why": why,
                "cost": record.get("cost"),
            }
        )

    measured = [r["delta"] for r in rows if r["delta"] is not None]
    treatment = (protocol.get("statistics") or {}).get("failed_campaign_treatment")
    return {
        "contract_version": "1.0.0",
        "record_kind": "intention_to_run_table",
        "boundary_state": state.value,
        "treatment": (
            str(treatment.get("denominator")) if isinstance(treatment, Mapping) else "INTENTION_TO_RUN"
        ),
        "denominator": len(locked),
        "locked_campaigns": locked,
        "n_with_a_value": len(measured),
        "n_missing": len(rows) - len(measured),
        "n_nominated": sum(1 for r in rows if r["outcome"] == CampaignOutcome.NOMINATED.value),
        "rows": rows,
        "never": [
            "dropping a campaign from the denominator",
            "taking the best of several restarts",
            "treating a failed evaluation as a zero metric value rather than a missing one",
        ],
    }


# ---------------------------------------------------------------------------
# evidence: C2 per artifact, C3 pooled across the process
# ---------------------------------------------------------------------------
def _percentile_interval(draws: Sequence[float], level: float) -> tuple[float, float]:
    ordered = sorted(draws)
    if not ordered:  # pragma: no cover - guarded by the callers
        raise _fail(ErrorCategory.EMPTY_VALUE, "no bootstrap draws", field="draws")
    tail = (1.0 - level) / 2.0
    last = len(ordered) - 1
    lo = ordered[max(0, min(last, round(tail * last)))]
    hi = ordered[max(0, min(last, round((1.0 - tail) * last)))]
    return lo, hi


def artifact_evidence(
    artifact_id: str,
    family_values: Mapping[str, Sequence[float]],
    *,
    deployment_seed_values: Mapping[int, float] | None = None,
    level: float = 0.95,
    n_draws: int = 10000,
    seed: int = 20260912,
) -> Record:
    """**C2** — uncertainty for ONE fitted artifact, from its held-out family groups.

    For a fixed predictor the only randomness left at scoring time is which
    families ended up in the held-out partition, so the resample is over
    *families*, with every value inside a family carried together. For a
    generated portfolio the deployment seeds vary too, and they are recorded
    beside the family interval rather than pooled into it.

    This is per-artifact evidence. It is never the process interval, and the
    record says so in ``evidence_class`` so a reader cannot mistake one for the
    other.
    """
    if not family_values:
        raise _fail(
            ErrorCategory.EMPTY_VALUE,
            "C2 evidence needs the held-out family groups this artifact was scored over",
            field="family_values",
        )
    groups = [list(map(float, values)) for values in family_values.values()]
    flat = [v for group in groups for v in group]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_draws):
        resampled = [v for _ in groups for v in rng.choice(groups)]
        draws.append(statistics.fmean(resampled) if resampled else float("nan"))
    lo, hi = _percentile_interval(draws, level)
    return {
        "contract_version": "1.0.0",
        "record_kind": "artifact_evidence",
        "evidence_class": "C2_PER_ARTIFACT",
        "artifact_id": artifact_id,
        "replication_unit": "held_out_family_group",
        "n_families": len(groups),
        "n_values": len(flat),
        "point": statistics.fmean(flat),
        "interval_level": level,
        "interval": [lo, hi],
        "n_bootstrap_draws": n_draws,
        "bootstrap_seed": seed,
        "deployment_seed_values": dict(deployment_seed_values) if deployment_seed_values else None,
        "deployment_seed_spread": (
            max(deployment_seed_values.values()) - min(deployment_seed_values.values())
            if deployment_seed_values
            else None
        ),
        "is_process_evidence": False,
        "note": (
            "per-artifact uncertainty over held-out family groups. This is NOT the "
            "campaign-to-campaign process interval and may never be reported as one"
        ),
    }


def process_evidence(
    campaign_deltas: Mapping[str, float | None],
    *,
    denominator: int,
    replication_unit: str = PROCESS_REPLICATION_UNIT,
    level: float = 0.95,
    n_draws: int = 10000,
    seed: int = 20260912,
) -> Record:
    """**C3** — pooled process evidence, resampling CAMPAIGNS.

    The lock's replication unit is the campaign and ``molecule.is_replication_unit``
    is false, so a per-molecule resample reported as process confidence is
    pseudo-replication and is refused by name.

    Every locked campaign is in ``campaign_deltas``; a missing value stays a
    missing value. The record reports the denominator it was computed against and
    how many of those slots actually carried a number, because an interval over
    two survivors of five campaigns is not an interval over the process.

    Raises:
        ValueError: ``INVALID_VALUE`` for a forbidden replication unit or a
            denominator smaller than the campaigns supplied; ``EMPTY_VALUE`` when
            nothing carries a value.
    """
    if replication_unit in FORBIDDEN_PROCESS_UNITS:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"{replication_unit!r} is not the locked replication unit for process evidence. "
            f"The lock sets molecule.is_replication_unit = false; a per-{replication_unit} "
            f"bootstrap reported as process confidence is pseudo-replication",
            field="replication_unit",
        )
    if replication_unit != PROCESS_REPLICATION_UNIT:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"{replication_unit!r} is not {PROCESS_REPLICATION_UNIT!r}",
            field="replication_unit",
        )
    if denominator < len(campaign_deltas):
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"the denominator {denominator} is smaller than the {len(campaign_deltas)} campaigns "
            f"supplied; the denominator is the locked design and never shrinks to fit",
            field="denominator",
        )
    values = [float(v) for v in campaign_deltas.values() if v is not None]
    if not values:
        return {
            "contract_version": "1.0.0",
            "record_kind": "process_evidence",
            "evidence_class": "C3_PROCESS",
            "replication_unit": replication_unit,
            "denominator": denominator,
            "n_with_a_value": 0,
            "point": None,
            "interval": None,
            "status": "UNDEFINED_NO_VALUES",
            "why": (
                "no locked campaign carries a value, so the paired difference estimator is "
                "undefined. An undefined estimator is reported as undefined, not as zero"
            ),
        }
    rng = random.Random(seed)
    draws = [statistics.fmean([rng.choice(values) for _ in values]) for _ in range(n_draws)]
    lo, hi = _percentile_interval(draws, level)
    return {
        "contract_version": "1.0.0",
        "record_kind": "process_evidence",
        "evidence_class": "C3_PROCESS",
        "replication_unit": replication_unit,
        "denominator": denominator,
        "n_with_a_value": len(values),
        "point": statistics.fmean(values),
        "interval_level": level,
        "interval": [lo, hi],
        "n_bootstrap_draws": n_draws,
        "bootstrap_seed": seed,
        "status": "COMPUTED",
        "note": (
            "campaigns are the replication unit. Per-artifact C2 intervals are recorded "
            "separately and are never pooled into this one"
        ),
    }


def evidence_record(per_artifact: Sequence[Mapping[str, Any]], pooled: Mapping[str, Any]) -> Record:
    """Keep C2 and C3 in separate, separately labelled keys.

    Raises:
        ValueError: ``INVALID_VALUE`` if a C2 record is handed in as the pooled
            process evidence, or vice versa.
    """
    for entry in per_artifact:
        if entry.get("evidence_class") != "C2_PER_ARTIFACT":
            raise _fail(
                ErrorCategory.INVALID_VALUE,
                f"{entry.get('evidence_class')!r} is not per-artifact C2 evidence",
                field="per_artifact",
            )
    if pooled.get("evidence_class") != "C3_PROCESS":
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"{pooled.get('evidence_class')!r} is not pooled process C3 evidence; a "
            f"per-artifact interval reported as process confidence is pseudo-replication",
            field="pooled",
        )
    return {
        "contract_version": "1.0.0",
        "record_kind": "confirmation_evidence",
        "c2_per_artifact": [dict(e) for e in per_artifact],
        "c3_process": dict(pooled),
        "separation_rule": (
            "C2 is uncertainty for one fitted artifact over held-out family groups. C3 is "
            "uncertainty for the process over campaigns. They are recorded separately and "
            "neither may be reported as the other"
        ),
    }


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _safe_protocol_hash(protocol: Mapping[str, Any]) -> str | None:
    """The derived address, or ``None`` for a document that will not canonicalise."""
    try:
        return _derived_protocol_hash(protocol)
    except (TypeError, ValueError):
        return None


def _condition(identifier: str, *, requires: str, observed: Any, met: bool, unblock: str) -> Record:
    return {
        "id": identifier,
        "requires": requires,
        "observed": observed,
        "met": bool(met),
        "unblock": unblock,
    }


def confirmation_readiness(
    protocol: Mapping[str, Any],
    *,
    track: str = "A",
    boundary: NominationBoundary | None = None,
    authorisation: Mapping[str, Any] | None = None,
    operator_release_requested: bool = False,
) -> Record:
    """Decide whether the held-out partition may be read at all. Default: no.

    Every condition is derived from the lock or from the ledger, so a flag alone
    can never open the gate — ``operator_release_requested`` is *necessary* and
    nowhere near sufficient, and it is listed last for exactly that reason.

    The conditions:

    ``PROTOCOL_AUTHENTIC``
        the document this gate is reading must hash to the REGISTERED protocol
        address. It is first because every condition below it is derived from
        this document: a forged protocol that is merely self-consistent could
        otherwise satisfy all seven at once. The gate re-derives the address
        itself rather than trusting a caller to have authenticated it.
    ``LOCK_OPEN``
        ``test_access_count == 0`` and ``revealed_at is None``. Once either
        moves, this lock is closed and the only admissible record is a
        superseding version that cannot call itself the original confirmation.
    ``TRACK_CONSUMES_HELD_OUT``
        the branch being confirmed must actually have a held-out endpoint. Under
        ``zak-c9r-2`` the registered primary reads no held-out compound, so there
        is nothing to spend the partition on there.
    ``TRACK_NOT_DEMOTED_UNRUN``
        the branch must not be registered ``NOT_RUN``. Track A is a demoted
        secondary whose lock text says the branch stays ``NOT_RUN``.
    ``MAIN_ARM_EXECUTED``
        the branch's main research arm must have run. ``B5-self-evolution`` is
        ``NOT_RUN``; with zero nominees the estimator
        ``delta_i = mae(nominee_i) - mae(parent)`` has no ``nominee_i`` and is
        undefined.
    ``BUDGET_AUTHORISED``
        ``budgets.authorisation_state.additional_authorisation_required`` must be
        zero, or an operator authorisation record must cover it.
    ``NOMINATIONS_COMPLETE``
        every locked campaign has its one terminal record and the boundary is
        ready to release.
    ``OPERATOR_RELEASE_REQUESTED``
        the caller explicitly asked, in writing, to spend the partition.

    Returns:
        A record whose ``decision`` is ``READY`` only when every condition is
        met, and ``NOT_RUN`` otherwise, with the blocking conditions named.
    """
    # The gate is reachable without `authenticate_protocol` — that is deliberate,
    # so that a forged protocol cannot open it by bypassing the front door — which
    # means it has to do its own canonicalisation as well as its own hashing. One
    # object, hashed and read: see `canonicalised`.
    try:
        protocol = canonicalised(protocol)
    except (TypeError, ValueError):
        # A document that will not canonicalise is not the lock, and
        # PROTOCOL_AUTHENTIC below reports exactly that.
        protocol = dict(protocol)

    exposure = dict(protocol.get("evaluation_exposure") or {})
    tracks = [dict(t) for t in (protocol.get("registered_tracks") or [])]
    endpoint = dict(protocol.get("primary_endpoint") or {})

    access = exposure.get("test_access_count", 0)
    revealed = exposure.get("revealed_at")
    authentic = protocol_is_authentic(protocol)
    conditions: list[Record] = [
        _condition(
            "PROTOCOL_AUTHENTIC",
            requires=f"the protocol hashes to the registered address {REGISTERED_PROTOCOL_HASH}",
            observed={
                "registered_protocol_hash": REGISTERED_PROTOCOL_HASH,
                "measured_protocol_hash": (
                    _derived_protocol_hash(protocol) if authentic else _safe_protocol_hash(protocol)
                ),
                "protocol_version": protocol.get("protocol_version"),
                "authentic": authentic,
            },
            met=authentic,
            unblock=(
                "confirm against the registered protocol, or register a superseding version and "
                "confirm against that. A document that agrees with its own stored hash is "
                "self-consistent, not authentic: every condition below is read out of this "
                "document, so an unauthenticated one could satisfy all of them at once"
            ),
        ),
        _condition(
            "LOCK_OPEN",
            requires="evaluation_exposure.test_access_count == 0 and revealed_at is null",
            observed={"test_access_count": access, "revealed_at": revealed},
            met=(access == 0 and revealed is None),
            unblock=(
                "none. Once a held-out result is revealed this lock is closed: a later study "
                "needs a superseding version with its own exposure declaration, and it may "
                "never be presented as the original confirmation"
            ),
        )
    ]

    primary_partition = str(endpoint.get("partition") or "")
    demoted = dict(endpoint.get("superseded_track_a_endpoint") or {})
    held_out_endpoint = demoted if str(demoted.get("track") or "") == track else {}
    primary_is_held_out = primary_partition.strip().lower() == "test"
    conditions.append(
        _condition(
            "TRACK_CONSUMES_HELD_OUT",
            requires=f"track {track!r} has a registered endpoint whose partition is the held-out one",
            observed={
                "primary_endpoint.partition": primary_partition,
                "primary_reads_held_out": primary_is_held_out,
                "held_out_endpoint_found_for_track": bool(held_out_endpoint),
                "held_out_endpoint_partition": held_out_endpoint.get("partition"),
            },
            met=primary_is_held_out or bool(held_out_endpoint),
            unblock=(
                "register a protocol version whose primary endpoint actually reads the held-out "
                "partition. Under zak-c9r-2 the registered primary is Track B generation quality "
                "on a development-derived pocket and exclusion set, and it is already complete"
            ),
        )
    )

    registered = next((t for t in tracks if str(t.get("track")) == track), {})
    reason = str(registered.get("reason") or "")
    branch_not_run = "NOT_RUN" in reason.upper() or str(registered.get("status")) == "NOT_RUN"
    conditions.append(
        _condition(
            "TRACK_NOT_DEMOTED_UNRUN",
            requires=f"the registered record for track {track!r} does not declare the branch NOT_RUN",
            observed={"status": registered.get("status"), "declares_not_run": branch_not_run},
            met=bool(registered) and not branch_not_run,
            unblock=(
                "a superseding protocol version that promotes the branch and funds it. Demoting "
                "a branch and then running it anyway is the post-hoc retrofit the demotion exists "
                "to prevent"
            ),
        )
    )

    comparators = [dict(c) for c in (protocol.get("comparators") or [])]
    main_arm = next((c for c in comparators if str(c.get("role")) == "MAIN_RESEARCH_ARM"), {})
    main_status = str(main_arm.get("status") or "ABSENT")
    conditions.append(
        _condition(
            "MAIN_ARM_EXECUTED",
            requires="the MAIN_RESEARCH_ARM comparator has been executed, so nominees exist",
            observed={"id": main_arm.get("id"), "status": main_status},
            met=main_status == "EXECUTED",
            unblock=(
                "execute the main research arm. With zero nominees the registered estimator "
                "delta_i = mae_pactivity(nominee_i) - mae_pactivity(A-B0-parent-rf) has no "
                "nominee_i and is undefined; confirming today could only score frozen baselines "
                "plus a constant"
            ),
        )
    )

    state = dict((protocol.get("budgets") or {}).get("authorisation_state") or {})
    shortfall = float(state.get("additional_authorisation_required") or 0.0)
    # CLASS 3. This used to read `float((authorisation or {}).get(...))` straight
    # out of a file named on the command line — an unauthenticated document whose
    # only claim to authority was that someone passed its path. The number is a
    # safety-relevant state, so it is not read unless the document BINDS itself
    # to this study: the registered protocol address and the track it authorises.
    # A record for another protocol, another track, or one that names neither
    # covers nothing, and `covered` stays at 0.0.
    #
    # This is a binding check, NOT an authentication: no keyless check can
    # authenticate a file its author holds, and the record says so in
    # `authentication` rather than implying a strength it does not have. What it
    # buys is that a stray or mis-aimed authorisation cannot cover this gate by
    # accident, and that the committed record states what was actually checked.
    supplied = dict(authorisation or {})
    binds = {
        "record_kind": str(supplied.get("record_kind") or "") == "confirmation_budget_authorisation",
        "protocol_hash": str(supplied.get("protocol_hash") or "") == REGISTERED_PROTOCOL_HASH,
        "track": str(supplied.get("authorises_track") or "") == track,
        "authorised_by": bool(str(supplied.get("authorised_by") or "").strip()),
    }
    unbound = sorted(key for key, ok in binds.items() if not ok)
    covered = (
        0.0 if unbound else float(supplied.get("additional_output_tokens_authorised") or 0.0)
    )
    conditions.append(
        _condition(
            "BUDGET_AUTHORISED",
            requires="budgets.authorisation_state.additional_authorisation_required is covered",
            observed={
                "additional_authorisation_required": shortfall,
                # NOT the quantity. `covered` is a float read out of a file named
                # on the command line, and this observed block is copied verbatim
                # into the committed branch-status record — the one artifact that
                # says the held-out partition was NOT READ. A caller who can write
                # a binding authorisation file could therefore place an
                # attacker-chosen MAGNITUDE inside that record with no source edit
                # and no in-process execution, under a name no denylist forbids.
                # The gate needs the COMPARISON, not the number, so the record
                # carries the comparison and the number stays outside it.
                "shortfall_is_covered_by_an_operator_record": shortfall <= covered,
                "authorised_quantity_is_not_recorded_here": (
                    "deliberately. The quantity comes from an unauthenticated file the caller "
                    "names, and this block is copied into a record that asserts the held-out "
                    "partition was NOT READ. A record asserting that may carry no magnitude an "
                    "operator chose; it carries whether the shortfall is covered, and the "
                    "authorisation record itself is the place to read the quantity"
                ),
                "binding_resource": state.get("binding_resource"),
                "authorisation_supplied": bool(supplied),
                "authorisation_binds_to_this_study": not unbound,
                "authorisation_not_bound_by": unbound,
                "authentication": (
                    "UNAUTHENTICATED_OPERATOR_RECORD - bound to the registered protocol address "
                    "and this track, NOT cryptographically authenticated. A keyless check cannot "
                    "authenticate a file its author holds"
                ),
            },
            met=shortfall <= covered,
            unblock=(
                f"an explicit operator authorisation for {shortfall:,.0f} additional "
                f"{state.get('binding_resource') or 'llm_output_tokens'}, as a record whose "
                f"record_kind is 'confirmation_budget_authorisation', whose protocol_hash is "
                f"{REGISTERED_PROTOCOL_HASH}, whose authorises_track is {track!r} and which "
                f"names an authorised_by. A record that does not bind to this study covers "
                f"nothing. The design may not be shrunk to fit: dropping a campaign, widening a "
                f"margin or relaxing a gate is refused by validate_protocol or is a new "
                f"protocol version"
            ),
        )
    )

    outstanding = boundary.outstanding() if boundary is not None else None
    conditions.append(
        _condition(
            "NOMINATIONS_COMPLETE",
            requires="every locked campaign/track slot has its one final record before any release",
            observed={
                "boundary_present": boundary is not None,
                "state": boundary.state.value if boundary is not None else None,
                "outstanding": outstanding,
            },
            met=boundary is not None and boundary.is_complete(),
            unblock=(
                "finish every locked campaign's nomination. Nominations are frozen before any "
                "test feedback is released, and an incomplete set cannot be released"
            ),
        )
    )

    conditions.append(
        _condition(
            "OPERATOR_RELEASE_REQUESTED",
            requires="the caller explicitly asked to spend the held-out partition",
            observed=operator_release_requested,
            met=bool(operator_release_requested),
            unblock="pass the explicit release flag, once every condition above is already met",
        )
    )

    blocking = [c["id"] for c in conditions if not c["met"]]
    return {
        "contract_version": "1.0.0",
        "record_kind": "confirmation_readiness",
        "protocol_version": protocol.get("protocol_version"),
        "track": track,
        "decision": "READY" if not blocking else "NOT_RUN",
        "blocking_conditions": blocking,
        "conditions": conditions,
        "default_is_refusal": True,
        "rule": (
            "the held-out partition is read only when EVERY condition is met. The refusal is "
            "the default and the execution is the explicit, argued exception"
        ),
    }


#: The paths at which the confirmation branch-status record — the one committed
#: artifact that asserts the held-out partition was NOT READ — may carry a
#: magnitude. An ALLOWLIST of path patterns with list indices collapsed, and the
#: direction is the whole point: a denylist of held-out names is only as complete
#: as whoever last edited it, and this programme has been bitten by that four
#: times. Every member here is a COUNT, a protocol-declared COST, or the record's
#: own ``test_access_count``. Nothing here is a quantity an operator, a ledger or
#: a file on the command line supplies.
NOT_RUN_MAGNITUDE_PATHS: frozenset[str] = frozenset(
    {
        # the record's own claim about the partition
        "test_access_count",
        # the gate's conditions, whose observed values are read out of the
        # AUTHENTICATED protocol and nowhere else
        "conditions_for_a_legitimate_later_run[].observed.test_access_count",
        "conditions_for_a_legitimate_later_run[].observed.additional_authorisation_required",
        "reasons[].observed.test_access_count",
        "reasons[].observed.additional_authorisation_required",
        # what a later legitimate run would cost, every field read out of the
        # protocol's own budget block
        "cost_if_run.additional_authorisation_required",
        "cost_if_run.authorised_output_tokens_to_date",
        "cost_if_run.measured_output_tokens_spent",
        "cost_if_run.llm_output_tokens_confirmation",
        "cost_if_run.llm_output_tokens_step16_ablation",
        "cost_if_run.llm_output_tokens_total",
        # the intention-to-run denominator and its counts. `rows[].delta` is
        # here because 0.0 for a campaign that delivered nothing is a statement
        # about the PROCESS; a delta that is not 0.0 is refused below whatever
        # outcome it is filed under, and `rows[].cost` is deliberately absent —
        # a verbatim ledger cost is an operator-supplied magnitude and has no
        # business in a record asserting the partition was never read.
        "intention_to_run.denominator",
        "intention_to_run.n_with_a_value",
        "intention_to_run.n_missing",
        "intention_to_run.n_nominated",
        "intention_to_run.rows[].delta",
    }
)


#: Every function in this programme that emits a record asserting the held-out
#: partition was NOT READ or that ``test_access_count`` is 0, and the allowlist
#: each one is guarded by.
#:
#: This exists so the guard can be ENUMERATED FROM THE CODE rather than
#: remembered. ``test_every_emitter_of_a_not_read_assertion_is_guarded_or_exempt``
#: parses the target package and ``scripts/research`` for functions that write
#: such an assertion as a literal and fails on one that is in neither this map nor
#: :data:`FALSE_ASSURANCE_EXEMPT`. The previous round shipped a guard on exactly
#: one of them, and the enumeration is why that cannot happen again.
FALSE_ASSURANCE_GUARDED: dict[str, str] = {
    "confirmation.not_run_record": "NOT_RUN_MAGNITUDE_PATHS",
    "ablation.arm_not_run_record": "ARM_NOT_RUN_MAGNITUDE_PATHS",
    "reporting.branch_status_table": "BRANCH_STATUS_MAGNITUDE_PATHS",
}

#: Emitters the same enumeration finds that are deliberately NOT guarded by a
#: path allowlist, each with the reason and with what holds in its place. An
#: entry here is a decision with an argument, not a gap: a path allowlist over a
#: record whose job is to report measurements would have to name the whole study
#: schema, and an allowlist that large refuses nothing.
FALSE_ASSURANCE_EXEMPT: dict[str, str] = {
    "export_submitted_molecules.export": (
        "the manifest asserts of an export of GENERATED compounds, whose rows are molecules "
        "this programme invented rather than rows of the split. A path allowlist over it would "
        "name a schema that contains no partition field to begin with. What holds instead is "
        "executable and stronger than a registry: the exporter refuses any column whose name "
        "matches a comparison-compound substring, and it reads every CSV back off disk and "
        "runs `scan_for_compound_identities` over the bytes it actually wrote, deleting the "
        "file and exiting non-zero on a hit. The assertion is therefore checked, not declared."
    ),
    "controller.select_nomination": (
        "the assertion is made by the Nomination CONTRACT, not by a hand-built mapping. "
        "`validate_record` refuses an unknown key outright and the dataclass has six declared "
        "fields with no free-form depth, so there is no path at which an unregistered magnitude "
        "could sit. A closed schema is a stronger allowlist than a path list."
    ),
    "protocol.example_protocol": (
        "a fixture PROTOCOL document rather than a record. `evaluation_exposure.test_access_count` "
        "is the lock's declaration of what a study may do, which `validate_protocol` re-checks; it "
        "asserts nothing about what any run read."
    ),
    "reporting.build_package": (
        "the evidence package's job is to carry the study's development-stage measurements, so a "
        "path allowlist over it would have to name the whole schema and would refuse nothing. What "
        "holds instead: `verify_test_access_count` re-scans the entire tree BEFORE the package is "
        "written and the command refuses on a non-zero count; `label_safety` refuses a package "
        "carrying any per-compound identity; and every NOT_RUN record the package embeds was "
        "guarded at its own emitter."
    ),
    "pic50_independent_estimate.main": (
        "same class as `reporting.build_package`: a record of development-partition measurements "
        "whose whole content is magnitudes. It reads the TRAIN and development partitions only, "
        "its own preconditions are measured rather than asserted, and the held-out partition has "
        "no code path into it."
    ),
    "run_target_study.cmd_confirm": (
        "a stdout summary, not a written record. Every field in it is copied out of the "
        "branch-status record that `not_run_record` already refused to build over an unregistered "
        "magnitude, or out of the intention-to-run table that enforces the same rule on the way in."
    ),
    "run_target_study.cmd_ablate": (
        "the same: a stdout summary over counts the ablation report already computed. The records "
        "it writes — `ablation.json` and the pinned status — carry the per-arm NOT_RUN records, "
        "and those are guarded at `ablation.arm_not_run_record`."
    ),
}


def refuse_false_assurance(
    record: Mapping[str, Any],
    *,
    allowed_magnitude_paths: frozenset[str] = NOT_RUN_MAGNITUDE_PATHS,
    what: str = "this record",
) -> None:
    """The narrowest thing that must hold, and the one this programme exists for.

    The goal: **a record that asserts the held-out partition was NOT READ should
    not be able to carry evidence that it was.** That artifact — a committed
    statement of integrity that is false — is worse than no record at all, and it
    is the failure every other mechanism here is only indirectly about. So the
    refusal is placed on the record itself rather than on the inputs that reach
    it: it does not depend on the ledger being authentic, on any digest, on the
    anchor, or on the gate having refused.

    **THIS GUARD DOES NOT ACHIEVE THAT GOAL, AND THE CLAIM THAT IT DID IS
    WITHDRAWN.** Four successive hardening rounds each fixed the emitters they
    were told about and left siblings unguarded; an adversarial verifier broke the
    property after every one of them, the fourth time through a pure CLI route
    with no source edit and no in-process execution
    (``ablate --distill-ablation <file> --pin``, whose committed ``ablation_status``
    record asserts ``test_results_viewed: false`` while carrying a caller-supplied
    magnitude at depth under a benign name). ``ablation.ablation_report`` and
    ``run_target_study._pinned_ablation_record`` are two such unguarded emitters.

    The enumeration meant to prevent a fourth miss is itself vacuous: its AST scan
    matches only a *constant* ``test_access_count: 0`` / ``held_out_partition``
    literal, so it cannot see emitters that build the assertion dynamically —
    including ``ablation.arm_not_run_record``, the round that wrote it. Three
    further bypasses stand: stringification through ``canonical_json`` hides nested
    magnitudes; a mapping key containing ``.`` or ``[N]`` forges an allowlisted
    path; and ``_looks_like_a_number`` ignores any magnitude carrying a unit,
    prefix or separator.

    Read this guard as **raising the cost of a careless or accidental false
    assertion, not as preventing a deliberate one.** What actually holds is
    structural and is recorded in ``confirmation-status.json``: there is no code
    path anywhere in this programme that reads the held-out partition, and
    ``test_access_count`` is 0. See blocker B-PROPERTY-D.

    Four structural checks, none of which reads a value or reports one:

    1. **A held-out magnitude anywhere, at any depth, under a name that means a
       held-out reading.** This one is the readable message, not the refusal that
       holds; see 2.
    2. **A magnitude at any path the caller did not register.** The refusal that
       actually holds, and the one the previous three rounds did not have. A
       measurement is a NUMBER, and the name it travels under is the attacker's
       choice: ``{"audit": {"q": 0.31}}`` names nothing forbidden and is a
       held-out reading all the same. So the question is inverted. *allowed_
       magnitude_paths* names the small set of places a record of this kind is
       permitted to carry a number — counts, costs at registered resource paths,
       digests, seeds — and a magnitude anywhere else, under any name, at any
       depth, is refused unread. An allowlist that falls behind refuses a
       legitimate field loudly; a denylist that falls behind admits a held-out
       number silently.
    3. **Anything in the record that says a boundary was RELEASED.** The release
       is the moment held-out feedback is handed back; a record that carries one
       cannot also say nothing was read. This catches the case a value scan
       cannot: a hand-typed release over genuine entries, where the leak is the
       CLAIM rather than a number.
    4. **A row that carries a delta it cannot have.** A nominee's delta is
       ``mae(nominee) - mae(parent)`` on the held-out split; it cannot exist
       unless the split was read, whatever key it was filed under. A delta that
       is not exactly ``0.0`` is refused under EVERY outcome, because 0.0 is the
       only delta the table writes for a campaign that nominated nothing.

    Args:
        record: the assembled record, immediately before it is returned.
        allowed_magnitude_paths: the path patterns this record kind may carry a
            magnitude at, list indices collapsed to ``[]``.
        what: how the refusal names the record, for the message.

    Raises:
        ValueError: ``TEST_LEAKAGE``, naming the paths and never the values.
    """
    measurements = _held_out_measurement_paths(record)
    if measurements:
        raise _fail(
            ErrorCategory.TEST_LEAKAGE,
            f"{what} asserts the held-out partition was NOT READ and carries {len(measurements)} "
            f"held-out measurement(s) at {measurements}. A committed record that is false about "
            f"the partition is the artifact this programme exists to prevent, so the record is "
            f"refused rather than written. The values are not reported here",
            field="held_out_partition",
            missing=measurements,
        )
    unexplained = magnitude_paths_outside(record, allowed_magnitude_paths)
    if unexplained:
        raise _fail(
            ErrorCategory.TEST_LEAKAGE,
            f"{what} asserts the held-out partition was NOT READ and carries {len(unexplained)} "
            f"magnitude(s) at {unexplained}, which are not among the {len(allowed_magnitude_paths)} "
            f"paths a record of this kind may carry a number at. A held-out measurement is a "
            f"NUMBER and the name it arrives under is its author's choice, so the check is an "
            f"ALLOWLIST of paths rather than a denylist of names: counts, costs at registered "
            f"resource paths, digests and seeds, and nothing else. Register the path deliberately "
            f"or do not carry the number. The values are not reported here",
            field="held_out_partition",
            missing=unexplained,
        )
    released_at_paths = sorted(
        {path for path, _keys, leaf in _walk(record) if leaf == BoundaryState.RELEASED.value}
    )
    if released_at_paths:
        raise _fail(
            ErrorCategory.TEST_LEAKAGE,
            f"{what} asserts the held-out partition was NOT READ while {released_at_paths} "
            f"record a RELEASED boundary. The release is the moment held-out feedback is handed "
            f"back to an evolving process: a record carrying one may not also claim nothing was "
            f"read, however the release got there. A confirmation that DID run needs its own "
            f"record kind, not this one",
            field="held_out_partition",
            missing=released_at_paths,
        )
    scored = sorted(
        {
            f"{path}.delta" if path else "delta"
            for path, mapping in _mappings(record)
            if "outcome" in mapping
            and mapping.get("delta") is not None
            and (
                str(mapping.get("outcome") or "") == CampaignOutcome.NOMINATED.value
                or not (
                    _looks_like_a_number(mapping["delta"]) and float(mapping["delta"]) == 0.0
                )
            )
        }
    )
    if scored:
        raise _fail(
            ErrorCategory.TEST_LEAKAGE,
            f"{what} asserts the held-out partition was NOT READ and carries a delta that cannot "
            f"exist at {scored}. A nominee's delta is the paired difference against the parent on "
            f"the held-out split; the only delta this table writes without reading that split is "
            f"0.0, for a campaign that delivered nothing. The values are not reported here",
            field="held_out_partition",
            missing=scored,
        )


def not_run_record(protocol: Mapping[str, Any], readiness: Mapping[str, Any], **provenance: Any) -> Record:
    """The committed statement that the confirmation branch did not run, and why.

    It carries the reason, the cost that would have to be authorised, and the
    exact conditions under which the branch could later be run legitimately —
    which are the same conditions :func:`confirmation_readiness` enforces, not a
    prose restatement of them.
    """
    # CLASS 1. `**provenance` used to be merged LAST, after `status`,
    # `test_access_count`, `held_out_partition`, `blocking_conditions` and
    # `reasons` — so the branch-status record itself, the one committed artifact
    # that says the partition was never read, could be made to assert whatever
    # its caller passed. This is the record whose claims matter most, so it is
    # the one that may least be overwritten.
    _refuse_protected(
        provenance,
        PROTECTED_STATUS_FIELDS,
        what="the branch-status record",
        why=(
            "status, test_access_count and held_out_partition are this record's claims about "
            "the held-out partition; a record whose claims were typed by its caller asserts "
            "nothing."
        ),
    )
    state = dict((protocol.get("budgets") or {}).get("authorisation_state") or {})
    totals = dict((protocol.get("budgets") or {}).get("totals_required_for_the_locked_design") or {})
    exposure = dict(protocol.get("evaluation_exposure") or {})
    access = exposure.get("test_access_count", 0)
    if access != 0 or exposure.get("revealed_at") is not None:
        raise _fail(
            ErrorCategory.INCONSISTENT_STATUS,
            f"the protocol declares test_access_count {access!r} and revealed_at "
            f"{exposure.get('revealed_at')!r}, so this record may not assert that the held-out "
            f"partition was NOT READ. A lock under which a result was revealed needs a superseding "
            f"version that says so; it may never be presented as the original confirmation",
            field="test_access_count",
        )
    record: Record = {
        # Merged FIRST as well as refused above, for the same reason as `release`.
        **provenance,
        "contract_version": "1.0.0",
        "record_kind": "confirmation_branch_status",
        "protocol_version": protocol.get("protocol_version"),
        "track": readiness.get("track"),
        "status": "NOT_RUN",
        "implementation_status": "COMPLETE",
        "why_not_failed": (
            "NOT_RUN is a recorded scientific state, not a gap. The machinery is complete and "
            "tested; the branch was not executed because the conditions below are not met"
        ),
        "reasons": [c for c in (readiness.get("conditions") or []) if not c.get("met")],
        "blocking_conditions": list(readiness.get("blocking_conditions") or []),
        "cost_if_run": {
            "binding_resource": state.get("binding_resource"),
            "additional_authorisation_required": state.get("additional_authorisation_required"),
            "authorised_output_tokens_to_date": state.get("authorised_output_tokens_to_date"),
            "measured_output_tokens_spent": state.get("measured_output_tokens_spent"),
            "llm_output_tokens_confirmation": totals.get("llm_output_tokens_confirmation"),
            "llm_output_tokens_step16_ablation": totals.get("llm_output_tokens_step16_ablation"),
            "llm_output_tokens_total": totals.get("llm_output_tokens_total"),
            "irreversible_cost": (
                "test_access_count moves 0 -> 1 permanently and the lock closes. That cost is "
                "paid once, cannot be refunded, and is not recorded as a token figure anywhere"
            ),
        },
        "conditions_for_a_legitimate_later_run": readiness.get("conditions"),
        "test_access_count": exposure.get("test_access_count"),
        "held_out_partition": (
            "NOT READ. No held-out compound was read, scored, fitted on or generated against to "
            "produce this record, and no compound identity or label appears in it"
        ),
        "what_was_delivered_instead": (
            "the confirmation machinery: the preflight refusals, the nomination boundary with "
            "its two rejections, the fresh-process artifact reload, the separated C2/C3 evidence "
            "and the intention-to-run denominator — all exercised against fixtures, none against "
            "the held-out partition"
        ),
        # Stated in the committed record rather than only in a handoff, because a
        # gate that overstates its own strength is worth less than one that
        # describes it. Four rounds of red-teaming ended here: every keyless
        # digest the ledger takes over itself is satisfiable by whoever holds the
        # file, so the ledger is ADVISORY and the structural fact below is what
        # actually holds.
        "nomination_ledger_assurance": {
            "class": "ADVISORY, NOT TAMPER-EVIDENT",
            "what_it_guarantees": (
                "that a CHEAP PARTIAL edit is caught. The entries are a chained content address, "
                "the ledger is pinned at its head in the unreleased state too, a pin missing or "
                "nulling a field is not a pin, and the head is compared against a digest "
                "committed to the repository in a file that is NOT the ledger. record() and "
                "release() refuse over a disturbed ledger rather than merely reporting it"
            ),
            "what_it_does_NOT_guarantee": (
                "authenticity. No keyless digest over a file its editor holds can authenticate "
                "that file. A ledger rewritten end to end, re-chained, re-pinned AND re-anchored "
                "by a matching committed anchor audits clean, and nothing keyless could detect "
                "it. The external anchor raises the cost from one uncommitted file edit to a "
                "reviewable source-control change; it does not stop someone who edits both"
            ),
            "why_that_is_acceptable": (
                "because the thing the ledger gates leads nowhere. See held_out_execution_path"
            ),
        },
        "held_out_execution_path": (
            "NONE. There is no code path anywhere in this programme that reads the held-out "
            "partition. `confirm` writes READY_NOT_EXECUTED even when every gate condition is "
            "met, and the run that would spend test_access_count is not wired into it. Every "
            "exploit against the ledger therefore opens conditions on a gate that leads nowhere. "
            "This structural fact, not the ledger's tamper evidence, is what keeps "
            "test_access_count at 0"
        ),
    }
    # The narrowest invariant in the module, and the last thing that happens: this
    # record may not be written over evidence that the partition WAS read, however
    # that evidence reached it. See `refuse_false_assurance`.
    refuse_false_assurance(record, what="the branch-status record")
    return record
