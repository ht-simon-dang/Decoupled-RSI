"""The stable public surface the per-target program's acceptance checks call.

Functions appear here as the step that owns them implements them. Step 2 owns
:func:`endpoint_result`; Step 3 owns :func:`validate_artifact`; Step 5 owns
:func:`normalize_measurement`; Step 6 owns :func:`partition_families`; Step 7
owns :func:`enrichment_factor` and :func:`qualifying_scaffolds`; Step 9 owns
:func:`reserve_budget`; Step 10 owns :func:`validate_plan`; Step 12 owns
:func:`validate_teacher_row`; Step 13 owns :func:`resolve_artifact`; Step 14
owns :func:`validate_protocol`.

:func:`endpoint_result` is the whole of the status/value contract:

* a number is reported **only** when the measurement succeeded, so nothing
  downstream can average an absence into a mean as if it were a zero;
* a genuine measured zero is a success carrying ``0.0``, and stays
  distinguishable from an absence forever after;
* a value that is not finite is not a measurement at all — NaN compares false
  against every threshold, which is indistinguishable from a criterion the arm
  legitimately missed, so it is refused at the boundary instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .budget import reserve_budget as _reserve_budget
from .contracts import EvolutionArtifact, validate_record
from .data import normalize_measurement as _normalize_measurement
from .splits import partition_families as _partition_families

Record = dict[str, Any]

#: The endpoint states declared by the program specification (Appendix A §7.1).
#: ``value`` is null unless the state is ``SUCCEEDED``.
ENDPOINT_STATES = ("SUCCEEDED", "FAILED", "UNAVAILABLE", "INVALID_INPUT")


def endpoint_result(status: str, value: float | None) -> Record:
    """Return ``{"status": ..., "value": ...}``, or raise ``ValueError``.

    >>> endpoint_result("SUCCEEDED", 0.0)
    {'status': 'SUCCEEDED', 'value': 0.0}
    >>> endpoint_result("FAILED", None)
    {'status': 'FAILED', 'value': None}

    Raises:
        ValueError: if ``status`` is not one of :data:`ENDPOINT_STATES`, if the
            presence of a value disagrees with the status, or if the value is
            not a finite number.
    """
    if status not in ENDPOINT_STATES or (status == "SUCCEEDED") != (value is not None):
        raise ValueError(
            f"status/value mismatch: status={status!r} value={value!r}; "
            f"a value is required for SUCCEEDED and forbidden otherwise, "
            f"and status must be one of {list(ENDPOINT_STATES)}"
        )
    if value is not None:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"endpoint value {value!r} is not a number") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"endpoint value {value!r} is not finite and is not a measurement")
        return {"status": status, "value": numeric}
    return {"status": status, "value": None}


def validate_artifact(record: Mapping[str, Any]) -> Record:
    """Validate one evolution artifact and return it as JSON-ready data.

    The artifact contract is in :mod:`.contracts`; this is the stable name the
    program's acceptance checks call. Validation here is *structural*: every
    required field is present (a required field stays required when its value is
    empty or null), identifiers are non-blank, every hash is a canonical content
    address, ``entrypoint`` is an argument vector rather than a shell string, and
    the artifact's ``kind`` carries the component that kind is defined by —
    weights for ``WEIGHTS``, two or more structures for ``RECEPTOR_ENSEMBLE``,
    and so on. An ``ALGORITHM`` with no weights at all is valid.

    It deliberately does **not** open the artifact store. A schema test may use a
    synthetic digest; an artifact about to be scheduled must pass
    :func:`drug_agent.selfevolve.target.contracts.verify_artifact_references`,
    which is where real content-addressed files are checked.

    Nothing in this function looks at how the artifact performed. ``kind`` and
    scientific decision are separate records on purpose.

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying a stable ``category``, the offending ``field`` and, for
            missing or unknown keys, the full list.
    """
    return validate_record(EvolutionArtifact, record)


def normalize_measurement(value: float, unit: str, relation: str) -> Record:
    """Standardize one concentration measurement without losing the original.

    Step 5 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.data.normalize_measurement`; this is the
    stable name the program's acceptance checks call.

    Three things it refuses to do, because each one silently changes a dataset's
    meaning:

    * it never converts a unit it does not have an exact factor for — a
      mass-per-volume value has no molar equivalent without a molecular weight,
      so ``ug.mL-1`` is refused rather than guessed;
    * it never drops the inequality on a censored record. ``> 1000 nM`` stays a
      bound, and the bound *reverses direction* when the value becomes a
      negative log concentration: pActivity is then at most 6, not at least;
    * it never overwrites what the source said. ``source_value``,
      ``source_unit`` and ``source_relation`` survive verbatim next to the
      canonical form, including the micro symbol the source happened to use.

    >>> row = normalize_measurement(1000.0, "nM", ">")
    >>> row["value_M"], row["relation"], row["exact"]
    (1e-06, '>', False)

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``INVALID_ENUM`` for an unknown unit or relation,
            ``INVALID_VALUE`` for a nonpositive concentration, or
            ``NON_FINITE_VALUE`` for NaN and infinity.
    """
    return _normalize_measurement(value, unit, relation)


def partition_families(rows: list[Record], seed: int, **options: Any) -> Record:
    """Allocate whole structural families to train/development/test.

    Step 6 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.splits.partition_families`; this is the
    stable name the program's acceptance checks call.

    Three things it refuses to do, because each one is a leak wearing a
    reasonable-looking disguise:

    * it never divides a connected structural family. A family is a connected
      component over shared nonempty Bemis-Murcko scaffold, shared curated
      series, shared connectivity key and Morgan r2/2048 Tanimoto at least 0.80,
      and the component moves whole — so a compound's stereoisomers, its
      cross-source aliases and its measurements in other endpoints all land on
      one side;
    * it never rescues an infeasible split. A single component too large for
      60/20/20 is reported in ``infeasible_reasons``, and the similarity cutoff
      is a module constant rather than a parameter that can be quietly lowered;
    * it never depends on anything but its inputs. The same rows and seed give
      the same membership in any order, the returned record carries no timestamp
      and no measured value, and it compares equal to itself on a second call —
      which is what makes "never reshuffle after seeing candidate performance"
      checkable rather than a promise.

    >>> rows = [{"molecule_id": str(i), "family_id": str(i // 2)} for i in range(4)]
    >>> out = partition_families(rows, seed=41)
    >>> out["membership"]["0"] == out["membership"]["1"]
    True

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``EMPTY_IDENTIFIER`` for a blank molecule id,
            ``MISSING_FIELD`` when a molecule has neither a family nor a
            structure or when only some rows declare a family, and
            ``INVALID_VALUE`` for a contradictory grouping or for fractions that
            are not a positive distribution over the three partitions.
    """
    return _partition_families(rows, seed, **options)


def enrichment_factor(
    labels: list[int],
    scores: list[float],
    fraction: float = 0.1,
    *,
    molecule_ids: Sequence[str] | None = None,
) -> float:
    """Enrichment at the top *fraction* of a ranked pool.

    Step 7 owns this. The implementation is
    :func:`drug_agent.selfevolve.evaluation.target_benchmark.enrichment_factor`;
    this is the stable name the program's acceptance checks call. It is imported
    lazily because the evaluation kernel imports this module, and a module-level
    import here would close that cycle.

    ``EF = (actives in the top ceil(fraction * N) / selected count) / (actives / N)``.
    The selected count uses ``ceil``, so at ``N = 15`` and ``fraction = 0.1`` two
    compounds are selected rather than one.

    Three things it refuses to do, because each one turns a ranking into a
    number that cannot be checked:

    * it never lets row order decide the winner. Ties are broken by
      ``molecule_ids`` — a keyword-only, defaulted extension to the signature
      Appendix A §7.2 declares, so every existing positional call is unchanged —
      and a tie that straddles the top-k boundary with no ids to break it is
      *refused* rather than resolved by the order the rows arrived in;
    * it never accepts an absence as a score. A NaN or infinite score raises,
      because NaN compares false against every threshold and is therefore
      indistinguishable from a compound that was genuinely ranked last;
    * it never invents a denominator. A dataset with no active has an overall
      active fraction of zero, and enrichment against it is undefined rather
      than zero or infinite.

    >>> enrichment_factor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
    10.0

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``INVALID_VALUE`` for unequal lengths, an empty dataset, a
            non-binary label, an out-of-range fraction, a dataset with no
            active, a duplicate molecule id or an unbreakable tie;
            ``NON_FINITE_VALUE`` for a NaN or infinite score; and
            ``EMPTY_IDENTIFIER`` for a blank molecule id.
    """
    from ..evaluation.target_benchmark import enrichment_factor as _enrichment_factor

    return _enrichment_factor(labels, scores, fraction, molecule_ids=molecule_ids)


def qualifying_scaffolds(rows: list[Record]) -> int:
    """Unique scaffolds among the molecules that passed every required gate.

    Step 7 owns this. The implementation is
    :func:`drug_agent.selfevolve.evaluation.target_benchmark.qualifying_scaffolds`;
    this is the stable name the program's acceptance checks call, imported
    lazily for the same reason as :func:`enrichment_factor`.

    Each row needs ``molecule_id``, ``scaffold_id`` and a boolean ``qualifies``.
    Each molecule contributes at most once and each scaffold contributes at most
    once, which is Appendix A §3.3's counting rule rather than a convenience.

    Three things it refuses to do:

    * it never collapses acyclic molecules into one scaffold. An acyclic
      molecule has an *empty* Bemis-Murcko scaffold, and Step 6 had already
      established that empty scaffolds are not one family;
      :func:`drug_agent.selfevolve.evaluation.target_benchmark.scaffold_id`
      gives it ``acyclic:<molecule_id>`` and a blank ``scaffold_id`` here is
      refused with that convention named;
    * it never assumes a missing gate result. A row with no ``qualifies`` flag
      raises instead of defaulting either way, because an absent gate is not a
      failed gate;
    * it never silently reconciles two rows that disagree about one molecule.

    >>> qualifying_scaffolds([
    ...     {"molecule_id": "a", "scaffold_id": "s1", "qualifies": True},
    ...     {"molecule_id": "b", "scaffold_id": "s1", "qualifies": True},
    ...     {"molecule_id": "c", "scaffold_id": "s2", "qualifies": False},
    ... ])
    1

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``EMPTY_IDENTIFIER`` for a blank molecule id,
            ``MISSING_FIELD`` for an absent ``qualifies`` flag or a blank
            scaffold id, ``INVALID_TYPE`` for a non-boolean flag, and
            ``INVALID_VALUE`` when two rows disagree about one molecule.
    """
    from ..evaluation.target_benchmark import qualifying_scaffolds as _qualifying_scaffolds

    return _qualifying_scaffolds(rows)


def reserve_budget(ledger: Path, resource: str, amount: float, limit: float) -> Record:
    """Hold *amount* of *resource* against *limit*, before the spending starts.

    Step 9 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.budget.reserve_budget` over
    :class:`~drug_agent.selfevolve.target.budget.BudgetLedger`; this is the
    stable name the program's acceptance checks call.

    Three things it refuses to do, because each one is how a declared budget
    quietly stops being one:

    * it never checks and spends in two steps. The read that decides and the
      write that commits are one ``BEGIN IMMEDIATE`` transaction against the
      ledger file, so two *processes* racing for the last GPU slot produce one
      admission and one refusal — which is a different guarantee from two
      threads in one interpreter, and the one that matters when campaign arms
      run as separate subprocesses;
    * it never lets a cap grow on demand. A limit that disagrees with what was
      already declared for this resource is refused rather than adopted, so
      "pass a bigger number next time" is not an escape hatch, and Appendix A
      §4's rule that confirmation cannot enlarge one arm's allocation survives
      contact with the code;
    * it never treats an unknown cap as permission. ``limit=None`` is
      ``BUDGET_UNDECLARED``, not unlimited.

    A refused reservation reserves nothing: the transaction rolls back, so the
    ledger after a refusal is byte-for-byte what it was before.

    >>> import tempfile, pathlib
    >>> db = pathlib.Path(tempfile.mkdtemp()) / "budget.sqlite"
    >>> reserve_budget(db, "oracle_calls", 8, 10)["reserved"]
    8.0

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.budget.BudgetExceeded` —
            itself a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError` —
            carrying ``INVALID_VALUE`` and the resource as its ``field`` when
            the reservation would pass the cap; ``BUDGET_UNDECLARED`` for a null
            cap; ``INVALID_VALUE`` for a negative amount or an enlarged cap; and
            ``NON_FINITE_VALUE`` for NaN or infinity.
    """
    return _reserve_budget(ledger, resource, amount, limit)


def validate_plan(plan: Mapping[str, Any], allowed_actions: list[str]) -> Record:
    """Admit one structured experiment plan against the campaign's action menu.

    Step 10 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.worker.validate_plan`; this is the
    stable name the program's acceptance checks call, and it is a thin adapter
    rather than a second copy of the rules.

    Three things it refuses to do, because each one is how a reasoning loop
    stops being bounded:

    * it never lets the agent reach the thing that grades it. The protected set
      is *derived* from the evaluation kernel's own integrity manifest — plus
      the frozen qualification and calibration records, the committed baseline
      manifests, the campaign configuration and the 0600 held-out registry — so
      a file added to the manifest becomes unreachable without anyone
      remembering to retype it, and ``..`` is resolved before the comparison
      rather than matched as text. ``rewrite_final_test`` does not stop here
      because it is blacklisted; it stops because the nine interventions that
      exist are the only ones that exist;
    * it never accepts a command as a string. A model-authored shell fragment
      is ``ENTRYPOINT_NOT_ARGV``, the same category Step 3 froze for artifact
      entry points, because nothing in this dispatcher runs anything through a
      shell and a string would be somebody else's quoting rules;
    * it never confuses *unauthorised* with *unimplemented*. An action outside
      the registered nine is ``INVALID_ENUM``; a registered action this campaign
      did not authorise is ``ACTION_NOT_PERMITTED``; and a campaign menu naming
      an action with no handler behind it is refused on the menu, because an
      advertisement is not a capability.

    Availability is deliberately **not** an admission criterion. Appendix A's
    acceptance rule is that any tool category may be proposed; a missing backend
    is answered by
    :func:`drug_agent.selfevolve.target.worker.action_support` with the missing
    verb and its unblock, and by a ``CAPABILITY_UNAVAILABLE`` experiment result,
    not by a refusal at the door.

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``MISSING_FIELD``/``INVALID_TYPE``/``EMPTY_VALUE`` for a
            malformed plan, ``INVALID_ENUM`` for an unregistered action or
            backend, ``ACTION_NOT_PERMITTED`` for an unauthorised one,
            ``PROTECTED_PATH`` for a plan that reaches the referee,
            ``ENTRYPOINT_NOT_ARGV`` for a command string, ``INVALID_VALUE`` for
            an output outside the experiment's own directory, and
            ``BUDGET_UNDECLARED`` for a requested budget that names no cap.
    """
    from .worker import validate_plan as _validate_plan

    return _validate_plan(plan, allowed_actions)


def validate_teacher_row(row: Mapping[str, Any]) -> Record:
    """Admit one distillation training row, with its lineage and its label semantics.

    Step 12 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.distill.validate_teacher_row`; this is
    the stable name the program's acceptance checks call, and it is a thin
    adapter rather than a second copy of the rules.

    Three things it refuses to do, because each one is how a synthetic training
    set stops being distinguishable from measured evidence:

    * it never lets a teacher prediction wear an experimental label. A
      ``teacher_prediction`` row carrying ``OBSERVED``, ``CURATED``, ``MEASURED``
      or ``EXPERIMENTAL`` is ``UNSUPPORTED_CLAIM`` on ``evidence_type`` — and so
      is the mirror image, a ``measured_label`` row claiming ``PREDICTED``. The
      same category catches the two back doors: a teacher row naming a measured
      biological endpoint as its ``metric``, and a teacher row whose ``value``
      is a copy of the compound's ``measured_p_activity``. ``MEASURED`` is in
      the recognised vocabulary on purpose, so the rejection is about the claim
      rather than about an unknown word;
    * it never accepts an unnamed computation. Teacher, checkpoint, molecule,
      pose, receptor, evaluator, experiment and data partition are all required,
      every hash must be a resolvable content address, and ``pose_id`` may be
      null only when the teacher call did not succeed — because a failed backend
      call has no pose and is not a low score;
    * it never reads a missing key as a null value. ``measured_p_activity`` and
      ``uncertainty`` must be present even when they are null, so "nothing was
      measured" and "nobody said" stay different statements, and a row from the
      held-out partition is refused by name as ``TEST_LEAKAGE`` rather than
      silently admitted.

    The status/value pairing is delegated to :func:`endpoint_result` above, so a
    genuine ``0.0`` teacher score survives and a non-finite one is refused at the
    same boundary as every other measurement in this program.

    >>> from drug_agent.selfevolve.target.distill import example_teacher_row
    >>> validate_teacher_row(example_teacher_row())["evidence_type"]
    'PREDICTED'

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``MISSING_FIELD``, ``UNKNOWN_FIELD``, ``INVALID_ENUM``,
            ``EMPTY_IDENTIFIER``, ``INVALID_REFERENCE``, ``INVALID_VALUE``,
            ``NON_FINITE_VALUE``, ``INCONSISTENT_STATUS``, ``TEST_LEAKAGE`` or
            ``UNSUPPORTED_CLAIM``.
    """
    from .distill import validate_teacher_row as _validate_teacher_row

    return _validate_teacher_row(row)


def resolve_artifact(artifact: Mapping[str, Any], target_hash: str, **options: Any) -> Record:
    """Resolve one evolution artifact for one target, or raise ``ValueError``.

    Step 13 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.registry.resolve_artifact`; this is the
    stable name the program's acceptance checks call.

    What it is for: an artifact an earlier campaign produced has to be usable
    again — reloaded, materialized and called — from its content addresses
    alone, without the conversation, the worker checkout or the campaign
    directory that made it. The one thing that must never happen along the way
    is that an artifact evolved for one target is quietly used for another, so
    **target compatibility is checked first**, before the record's structure,
    before the store and before the entrypoint. A wrong-target artifact
    therefore always fails as ``TARGET_MISMATCH`` and never as an unrelated
    missing key — which matters, because a rejection for the wrong reason is
    indistinguishable from a rejection for no reason.

    After compatibility it checks, in order: the artifact contract itself
    (required fields, canonical content addresses, an argv entrypoint, and the
    component the ``kind`` is defined by); that the entrypoint does not name a
    file the independent evaluator depends on; and — when a content-addressed
    ``store`` is supplied — that every referenced component really exists in it.
    Passing ``materialize_into`` additionally writes each component out and
    re-hashes it, so "reloadable" is demonstrated rather than asserted.

    Resolution says nothing about performance. Whether a resolved artifact helps
    is a :class:`~drug_agent.selfevolve.target.contracts.Comparison`, and the
    two are deliberately not joined here.

    >>> resolve_artifact({"target_manifest_hash": "sha256:" + "a" * 64}, "sha256:" + "b" * 64)
    Traceback (most recent call last):
        ...
    drug_agent.selfevolve.target.registry.ResolutionError: [TARGET_MISMATCH] ...

    Args:
        artifact: an :class:`~drug_agent.selfevolve.target.contracts.EvolutionArtifact`
            record.
        target_hash: the target manifest hash being resolved *for*.
        **options: ``store`` and ``materialize_into``, see the implementation.

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``TARGET_MISMATCH`` for an incompatible target,
            ``MISSING_FIELD`` when the artifact names no target,
            ``ENTRYPOINT_NOT_ARGV`` for a command string, ``MISSING_COMPONENT``
            when the kind's defining component is absent,
            ``UNVERIFIABLE_REFERENCE`` for content the store does not hold,
            ``HASH_MISMATCH`` when materialized bytes disagree with the declared
            digest, and ``PROTECTED_PATH`` for an entrypoint that reaches the
            independent evaluator.
    """
    from .registry import resolve_artifact as _resolve_artifact

    return _resolve_artifact(artifact, target_hash, **options)


def validate_protocol(protocol: Mapping[str, Any]) -> Record:
    """Validate one study protocol lock, or raise ``ValueError``.

    Step 14 owns this. The implementation is
    :func:`drug_agent.selfevolve.target.protocol.validate_protocol`; this is the
    stable name the program's acceptance checks call, and it is a thin adapter
    rather than a second copy of the rules.

    A protocol lock is what makes a confirmation study pre-registered rather
    than narrated afterwards. Three things it refuses to do, because each one is
    how a frozen study stops being frozen:

    * **it never lets a label outrun its own arithmetic.** ``feasibility``
      may read ``CONFIRMATORY`` only when the protocol's own frozen power block
      reaches its own ``min_power`` *at the worst case of its variance
      interval*, and only when the budget it declares is authorised. A study
      that cannot detect the effect it is designed around is labelled
      ``EXPLORATORY`` with its reasons, never relaxed until it looks adequate;
    * **it never lets a nomination be made on the answer.** ``selection_rule``
      naming the held-out partition is ``TEST_LEAKAGE``, ``max_final_nominations``
      is exactly one per campaign per registered track, and a confirmation seed
      already spent on a pilot is ``INVALID_VALUE`` because a reused seed
      replays a pilot rather than running an independent campaign;
    * **it never lets a lock be re-frozen after exposure.** Once
      ``evaluation_exposure.test_access_count`` is non-zero or a reveal time is
      recorded, the only admissible record is one declaring what it
      ``supersedes`` with ``presented_as_original_confirmation`` false. There is
      no path that produces a fresh "original" confirmation after a test result
      has been seen.

    The one asymmetry worth stating: ``phase="pilot"`` relaxes the required
    fourteen fields, because a pilot is not a lock — but it still cannot
    classify itself as confirmatory.

    >>> from drug_agent.selfevolve.target.protocol import example_protocol
    >>> validate_protocol(example_protocol())["max_final_nominations"]
    1

    Args:
        protocol: a study protocol record. See
            :data:`drug_agent.selfevolve.target.protocol.REQUIRED_LOCK_FIELDS`.

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying ``MISSING_FIELD``, ``EMPTY_VALUE``, ``UNKNOWN_FIELD``,
            ``INVALID_TYPE``, ``INVALID_ENUM``, ``INVALID_REFERENCE``,
            ``INVALID_VALUE``, ``TEST_LEAKAGE`` or ``UNSUPPORTED_CLAIM``.
    """
    from .protocol import validate_protocol as _validate_protocol

    return _validate_protocol(protocol)
