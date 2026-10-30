"""Immutable evolution-artifact versions, and how the master consumes one.

Step 13 of the per-target evolution program. Everything before this step
*produced* artifacts; nothing could pick one up again. This module is the half
that makes an evolved tool reusable: a registered artifact can be resolved,
materialized and loaded from its content addresses alone, in a fresh process,
with no access to the conversation, the campaign directory or the worker
checkout that created it.

Four rules shape the design, and each one exists because its absence is a
specific way to get a wrong answer.

**Compatibility is checked before anything else.** An artifact names the target
manifest it was built for. :func:`resolve_artifact` compares that field with the
target the caller is asking about *first* — before schema completeness, before
the store, before the entrypoint — so a wrong-target resolution always fails for
being the wrong target. A check ordered the other way would reject a mismatched
artifact for a missing unrelated field and never say the thing that mattered.

**A kind is plumbing; a gain is evidence.** The registry stores ten artifact
kinds behind one contract. That is schema support and nothing more. Whether a
particular artifact helps is a :class:`~..target.contracts.Comparison`, it is
recorded in ``evidence`` at acceptance time, and this module never derives one
from the other. Registering a ``RECEPTOR_ENSEMBLE`` does not mean a receptor
ensemble has been shown to work.

**Benchmark acceptance is evidence registration, not production activation.**
The three states are distinct rows in distinct tables:
``development_candidate`` is where every registration starts,
``benchmark_accepted`` records a comparison, and ``master_active`` is a
*pointer* an operator moves under the deployment's existing adoption policy
(:mod:`drug_agent.selfevolve.promotion.gate` — ``auto_promote`` is off by
default and promotion needs a human). Accepting cannot activate, because
accepting writes a different table.

**A running job keeps the artifact it started with.** :func:`open_master_job`
resolves and *pins* at startup and writes the pin to the registry file. A later
activation, or a rollback, changes the pointer and leaves every pin alone, so a
job that is halfway through a portfolio cannot have its scorer swapped
underneath it.

The referee is not in scope for any of this. A registered receptor artifact may
guide generation or ranking; it may never become the structure the independent
evaluator scores against. That is enforced three ways: the roles that name the
evaluator are reserved and refuse registration; :func:`referee_receptor` reports
the frozen referee so a test can assert it did not move; and
:func:`check_structure_identity` puts every receptor component through Step 5's
``pocket_matches_target``, so a structure from another protein is refused for
this target rather than registered as an evolution of it.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn

from ...util import canonical_json, hash_bytes
from .contracts import (
    CONTRACT_VERSION,
    ArtifactKind,
    ContractError,
    ErrorCategory,
    EvolutionArtifact,
    validate_record,
)

Record = dict[str, Any]

__all__ = [
    "COMPONENT_FIELDS",
    "REGISTRY_ENV",
    "LOADERS",
    "RESERVED_ROLES",
    "ArtifactRegistry",
    "ArtifactState",
    "MasterJob",
    "PointerName",
    "ResolutionError",
    "check_structure_identity",
    "component_references",
    "default_store_for",
    "example_artifact",
    "master_tool_cards",
    "materialize_artifact",
    "master_target_hash",
    "open_master_job",
    "referee_receptor",
    "registry_path",
    "require_adoption_authorization",
    "resolve_artifact",
]


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
class ArtifactState(StrEnum):
    """The three states the work order requires to stay distinguishable.

    They are not a ladder the registry walks on its own. ``benchmark_accepted``
    is written by :meth:`ArtifactRegistry.accept` from an independent evaluation
    record; ``master_active`` is a pointer moved by
    :meth:`ArtifactRegistry.activate` under the deployment's adoption policy.
    Nothing promotes itself.
    """

    DEVELOPMENT_CANDIDATE = "development_candidate"
    BENCHMARK_ACCEPTED = "benchmark_accepted"
    MASTER_ACTIVE = "master_active"


class PointerName(StrEnum):
    """The two movable pointers per role.

    ``champion`` is a development-evidence selection: which candidate this role
    would nominate. ``master_active`` is what the user-serving system actually
    loads. Keeping them apart is what stops "our best development result" from
    meaning "what production runs".
    """

    CHAMPION = "champion"
    MASTER_ACTIVE = "master_active"


#: Roles that name the independent evaluator. Appendix A §3.3: an evolved
#: receptor "may guide candidate generation or ranking, but cannot be
#: substituted into the independent evaluator to make its own score easier."
#: Registration under one of these names is refused outright, so the refusal
#: happens before an artifact is anywhere near the benchmark.
RESERVED_ROLES: tuple[str, ...] = (
    "referee_receptor",
    "referee",
    "evaluator",
    "benchmark_evaluator",
    "evaluation_kernel",
    "activity_gate",
    "structure_threshold",
)

#: The artifact contract's component fields, mapped to the resolution key each
#: one is reported under. Algorithm code, neural weights and receptor
#: coordinates are different *fields* of the same contract — different
#: hash-pinned component references behind one artifact — rather than different
#: record types.
COMPONENT_FIELDS: dict[str, str] = {
    "code_hash": "code",
    "environment_hash": "environment",
    "weights_hashes": "weights",
    "structure_hashes": "structures",
    "training_data_hashes": "training_data",
}

_SINGLE_FIELDS = ("code_hash", "environment_hash")


class ResolutionError(ContractError):
    """An artifact could not be resolved for the requested target.

    A :class:`~..target.contracts.ContractError`, therefore a ``ValueError``,
    carrying the stable :class:`~..target.contracts.ErrorCategory` the caller
    branches on.
    """


def _fail(
    category: ErrorCategory,
    message: str,
    *,
    field: str | None = None,
    missing: Sequence[str] = (),
) -> NoReturn:
    raise ResolutionError(category, message, contract="EvolutionArtifact", field=field, missing=missing)


# ---------------------------------------------------------------------------
# The frozen referee
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def referee_receptor(root: Path | str | None = None) -> Record:
    """The independent evaluator's own receptor and docking parameters.

    Read out of the frozen ``calibration.json``, never re-derived here: this
    function exists so a test can assert that registering, accepting or
    activating a receptor artifact left the referee exactly where Step 7 froze
    it. ``status`` is ``UNAVAILABLE`` when the record is absent, which is a
    different statement from "unchanged".
    """
    base = Path(root) if root is not None else _repo_root()
    path = base / "configs" / "research" / "zak-c9r" / "calibration.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "UNAVAILABLE", "reason": f"{path} is unreadable ({exc!r})", "source": str(path)}
    scorer = dict(record.get("scorer") or {})
    return {
        "status": "FROZEN",
        "source": str(path),
        "engine": scorer.get("engine"),
        "metric": scorer.get("metric"),
        "receptor_sha256": scorer.get("receptor_sha256"),
        "box_center": scorer.get("box_center"),
        "box_size": scorer.get("box_size"),
        "seed": scorer.get("seed"),
        "exhaustiveness": scorer.get("exhaustiveness"),
        "n_poses": scorer.get("n_poses"),
        "evaluator_hash": record.get("evaluator_hash"),
        "note": (
            "A registered receptor artifact may guide candidate generation or ranking. "
            "It is never substituted into the independent evaluator: these are the "
            "parameters every arm is scored against, and they do not move because an "
            "artifact was registered."
        ),
    }


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def component_references(record: Mapping[str, Any]) -> Record:
    """``{component group: [content address, ...]}`` for a validated artifact.

    One artifact contract carries three physically different kinds of payload —
    source code, fitted weights, receptor coordinates — and this is where they
    stay apart while sharing one record.
    """
    out: Record = {}
    for field, key in COMPONENT_FIELDS.items():
        value = record.get(field)
        out[key] = [str(value)] if field in _SINGLE_FIELDS else [str(v) for v in (value or [])]
    return out


def _check_entrypoint(entrypoint: Sequence[str]) -> list[str]:
    """Refuse an entrypoint that reaches a file the referee depends on."""
    from .worker import is_protected_path

    argv = [str(part) for part in entrypoint]
    for part in argv:
        rule = is_protected_path(part)
        if rule is not None:
            _fail(
                ErrorCategory.PROTECTED_PATH,
                f"the entrypoint element {part!r} names {rule!r}, which the independent "
                f"evaluator depends on; an artifact may be loaded by the master, never "
                f"substituted into the judge",
                field="entrypoint",
            )
    return argv


def resolve_artifact(
    artifact: Mapping[str, Any],
    target_hash: str,
    *,
    store: Any = None,
    materialize_into: Path | str | None = None,
    target_manifest: Mapping[str, Any] | None = None,
) -> Record:
    """Resolve one artifact for one target, or raise ``ValueError``.

    The order of the checks is the contract, not an implementation detail:

    1. **target compatibility**, so a wrong-target artifact always fails for
       being the wrong target and never for an unrelated missing key;
    2. **structure**, via :func:`..target.contracts.validate_record` — required
       fields, canonical content addresses, argv entrypoint, and the component
       the ``kind`` is defined by;
    3. **the entrypoint**, which may not name a protected evaluator file;
    4. **the bytes**, when a ``store`` is supplied: every referenced content
       address must really be in it.

    Args:
        artifact: an :class:`~..target.contracts.EvolutionArtifact` record.
        target_hash: the target manifest the caller is resolving *for*.
        store: optional content-addressed store (``exists``/``get_bytes``).
            Without one the result is structural only and says so in
            ``references_verified``.
        materialize_into: optional directory; every referenced component is
            written there and its digest re-checked against the declared one.
        target_manifest: the built ``target_manifest.json`` payload. Supplied for
            a receptor artifact, every structure component is checked against the
            target's own residue map; without it the check is reported
            ``NOT_CHECKED`` and never treated as agreement.

    Returns:
        A JSON-ready resolution record: the validated artifact, its components
        grouped by kind, the argv, and — for receptor kinds — the explicit
        statement that the referee is not in scope.

    Raises:
        ResolutionError: a :class:`ValueError` carrying a stable
            ``ErrorCategory``. ``TARGET_MISMATCH`` for an incompatible target,
            ``MISSING_FIELD`` for an absent one, ``UNVERIFIABLE_REFERENCE`` for
            content the store does not hold, ``HASH_MISMATCH`` when materialized
            bytes disagree with the declared digest, ``PROTECTED_PATH`` for an
            entrypoint that reaches the evaluator, and whatever the artifact
            contract raises for a structural defect.
    """
    wanted = str(target_hash or "").strip()
    if not wanted:
        _fail(
            ErrorCategory.EMPTY_IDENTIFIER,
            "resolve_artifact needs the target manifest hash it is resolving for; "
            "an unnamed target cannot be compared with anything",
            field="target_hash",
        )
    if not isinstance(artifact, Mapping):
        _fail(ErrorCategory.INVALID_TYPE, f"{type(artifact).__name__} is not an artifact record")
    declared = str(artifact.get("target_manifest_hash") or "").strip()
    if not declared:
        _fail(
            ErrorCategory.MISSING_FIELD,
            "the artifact does not say which target manifest it was built for, so it "
            "cannot be shown to be compatible with any",
            field="target_manifest_hash",
            missing=["target_manifest_hash"],
        )
    if declared != wanted:
        _fail(
            ErrorCategory.TARGET_MISMATCH,
            f"the artifact is scoped to target manifest {declared!r} and this resolution "
            f"is for {wanted!r}; an artifact evolved for one target is not evidence for "
            f"another, and changing its scope silently is how a result moves targets",
            field="target_manifest_hash",
        )

    validated = validate_record(EvolutionArtifact, artifact)
    argv = _check_entrypoint(validated["entrypoint"])
    components = component_references(validated)

    verified = False
    missing: list[str] = []
    if store is not None:
        for group, hashes in components.items():
            for content_hash in hashes:
                if not store.exists(content_hash):
                    missing.append(f"{group}:{content_hash}")
        if missing:
            _fail(
                ErrorCategory.UNVERIFIABLE_REFERENCE,
                f"{len(missing)} referenced component(s) are not in the store: {missing}. "
                f"A hash with no bytes behind it is a claim, not a reloadable artifact",
                field="artifact_id",
                missing=missing,
            )
        verified = True

    version_hash = hash_bytes(canonical_json(validated).encode("utf-8"))
    resolution: Record = {
        "contract_version": CONTRACT_VERSION,
        "artifact_id": validated["artifact_id"],
        "version_hash": version_hash,
        "kind": validated["kind"],
        "parent_id": validated["parent_id"],
        "experiment_id": validated["experiment_id"],
        "target_manifest_hash": declared,
        "compatible_with": wanted,
        "entrypoint": argv,
        "components": components,
        "references_verified": verified,
        "artifact": validated,
        "materialized": None,
    }
    is_receptor = str(validated["kind"]) in (
        ArtifactKind.RECEPTOR_STRUCTURE.value,
        ArtifactKind.RECEPTOR_ENSEMBLE.value,
    )
    if is_receptor:
        resolution["referee"] = {
            "may_guide_generation_or_ranking": True,
            "may_replace_referee": False,
            "frozen_referee": referee_receptor(),
        }
    if materialize_into is not None:
        resolution["materialized"] = materialize_artifact(resolution, materialize_into, store=store)
    if is_receptor:
        resolution["structure_identity"] = check_structure_identity(
            resolution, target_manifest=target_manifest, store=store
        )
    return resolution


def _load_manifest(source: Mapping[str, Any] | Path | str | None) -> Record | None:
    """Accept the built target manifest as a mapping or as a path to one."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return dict(source)
    try:
        return dict(json.loads(Path(source).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ResolutionError(
            ErrorCategory.INVALID_VALUE,
            f"the target manifest {source} is declared but unreadable ({exc!r}); refusing to "
            f"resolve against an unverifiable target identity",
            contract="ArtifactRegistry",
            field="target_manifest",
        ) from exc


def check_structure_identity(
    resolution: Mapping[str, Any],
    *,
    target_manifest: Mapping[str, Any] | None,
    store: Any = None,
) -> Record:
    """Check every receptor component really is this target's structure.

    Appendix A §3.3: a receptor change "must preserve the task's identity and
    declared construct scope". The check is Step 5's
    :func:`..target.data.pocket_matches_target` — the one whose absence let the
    platform generate against a soluble epoxide hydrolase pocket and report
    ``ACCEPT`` for ZAK — reused rather than re-derived, so one rule decides this
    for the generator and for the registry.

    Without a ``target_manifest`` there is nothing to compare against, and an
    undeclared target is reported ``NOT_CHECKED`` rather than treated as
    agreement.

    Raises:
        ResolutionError: ``TARGET_MISMATCH`` on ``structure_hashes`` when a
            component does not belong to this target.
    """
    hashes = list(dict(resolution["components"]).get("structures") or [])
    if not hashes:
        return {"status": "NOT_APPLICABLE", "reason": "the artifact declares no structures"}
    if target_manifest is None:
        return {
            "status": "NOT_CHECKED",
            "reason": (
                "no target manifest was supplied, so structural identity was not compared. "
                "An unchecked structure is not a matching structure"
            ),
            "n_structures": len(hashes),
        }
    materialized = resolution.get("materialized")
    if not materialized:
        if store is None:
            return {
                "status": "NOT_CHECKED",
                "reason": "no store and no materialized components: there are no coordinates to read",
                "n_structures": len(hashes),
            }
        with tempfile.TemporaryDirectory(prefix="artifact_structures_") as tmp:
            return check_structure_identity(
                {**dict(resolution), "materialized": materialize_artifact(resolution, tmp, store=store)},
                target_manifest=target_manifest,
                store=store,
            )
    from .data import pocket_matches_target

    verdicts: list[Record] = []
    mismatched: list[str] = []
    paths = list(dict(materialized["files"]).get("structures") or [])
    for content_hash, path in zip(hashes, paths, strict=True):
        verdict = pocket_matches_target(path, target_manifest)
        verdicts.append(
            {
                "content_hash": content_hash,
                "match": bool(verdict["match"]),
                "reason": verdict["reason"],
                "identity": verdict["identity"],
                "n_residues_checked": verdict["n_residues_checked"],
            }
        )
        if not verdict["match"]:
            mismatched.append(content_hash)
    if mismatched:
        _fail(
            ErrorCategory.TARGET_MISMATCH,
            f"{len(mismatched)} of {len(hashes)} receptor component(s) do not belong to target "
            f"{target_manifest.get('alias')!r} "
            f"({target_manifest.get('uniprot_accession')}): {verdicts}. "
            f"A receptor artifact may guide generation for this target; a structure from another "
            f"protein is not a worse receptor for it, it is a different question",
            field="structure_hashes",
            missing=mismatched,
        )
    return {
        "status": "MATCH",
        "n_structures": len(hashes),
        "target_alias": target_manifest.get("alias"),
        "declared_construct": (target_manifest.get("structure") or {}).get("pdb_id"),
        "per_structure": verdicts,
    }


def materialize_artifact(
    resolution: Mapping[str, Any],
    destination: Path | str,
    *,
    store: Any,
) -> Record:
    """Write a resolved artifact's components out and re-check every digest.

    This is what "reloadable without the original conversation" means
    concretely: the bytes come from the content-addressed store, they are
    written under names derived from their own digests, and each one is hashed
    again after the write. A store that returned different bytes than the record
    declares is ``HASH_MISMATCH`` here rather than a silently different model.
    """
    if store is None:
        _fail(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            "materializing needs a content-addressed store; there is nowhere else for the bytes to come from",
            field="artifact_id",
        )
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    files: Record = {}
    for group, hashes in dict(resolution["components"]).items():
        paths: list[str] = []
        for content_hash in hashes:
            data = store.get_bytes(content_hash)
            measured = hash_bytes(data)
            if measured != content_hash:  # pragma: no cover - content addressed store
                _fail(
                    ErrorCategory.HASH_MISMATCH,
                    f"{group} component declared {content_hash} and the store returned bytes "
                    f"hashing to {measured}",
                    field=group,
                )
            out = root / group / content_hash.replace(":", "_")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            paths.append(str(out))
        files[group] = paths
    return {"root": str(root), "files": files, "n_files": sum(len(v) for v in files.values())}


# ---------------------------------------------------------------------------
# Loaders — a closed vocabulary, declared at registration
# ---------------------------------------------------------------------------
def _load_distilled_student(resolution: Mapping[str, Any]) -> Any:
    """Reload a Step 12 student from its single weights component."""
    materialized = resolution.get("materialized")
    if not materialized:
        _fail(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            "a student is loaded from bytes on disk; resolve it with materialize_into",
            field="weights_hashes",
        )
    paths = list(dict(materialized["files"]).get("weights") or [])
    if len(paths) != 1:
        _fail(
            ErrorCategory.MISSING_COMPONENT,
            f"the distilled_student loader needs exactly one weights component, got {len(paths)}",
            field="weights_hashes",
        )
    from .distill import load_student

    return load_student(Path(paths[0]))


def _load_structure_set(resolution: Mapping[str, Any]) -> list[Path]:
    """Return the materialized receptor coordinate files, in declared order."""
    materialized = resolution.get("materialized")
    if not materialized:
        _fail(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            "a structure set is loaded from bytes on disk; resolve it with materialize_into",
            field="structure_hashes",
        )
    return [Path(p) for p in dict(materialized["files"]).get("structures") or []]


def _load_json_document(resolution: Mapping[str, Any]) -> Any:
    """Return the parsed single weights/training-data component as JSON."""
    materialized = resolution.get("materialized")
    if not materialized:
        _fail(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            "a json document is loaded from bytes on disk; resolve it with materialize_into",
            field="weights_hashes",
        )
    files = dict(materialized["files"])
    paths = list(files.get("weights") or []) + list(files.get("training_data") or [])
    if len(paths) != 1:
        _fail(
            ErrorCategory.MISSING_COMPONENT,
            f"the json_document loader needs exactly one document component, got {len(paths)}",
            field="weights_hashes",
        )
    return json.loads(Path(paths[0]).read_text(encoding="utf-8"))


def _load_argv(resolution: Mapping[str, Any]) -> list[str]:
    """Return the entrypoint argv with ``{artifact}`` substituted.

    The generic loader: for an artifact the master runs as a subprocess rather
    than imports. Substitution is a closed vocabulary, never a template engine.
    """
    materialized = resolution.get("materialized") or {}
    root = str(materialized.get("root") or "")
    return [
        part.replace("{artifact}", root) if isinstance(part, str) else part
        for part in resolution["entrypoint"]
    ]


#: How the master turns a resolved artifact into something callable. A closed
#: registry on purpose: a loader named in a registration must be one of these,
#: so an artifact cannot nominate arbitrary import paths for the master to run.
LOADERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "distilled_student": _load_distilled_student,
    "structure_set": _load_structure_set,
    "json_document": _load_json_document,
    "argv": _load_argv,
}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id          TEXT PRIMARY KEY,
    version_hash         TEXT NOT NULL,
    role                 TEXT NOT NULL,
    kind                 TEXT NOT NULL,
    loader               TEXT NOT NULL,
    target_manifest_hash TEXT NOT NULL,
    parent_id            TEXT,
    experiment_id        TEXT NOT NULL,
    record_json          TEXT NOT NULL,
    registered_at        REAL NOT NULL,
    registered_by        TEXT NOT NULL,
    fixture              INTEGER NOT NULL DEFAULT 0,
    note                 TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS acceptances (
    artifact_id   TEXT PRIMARY KEY,
    evidence_json TEXT NOT NULL,
    accepted_at   REAL NOT NULL,
    accepted_by   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pointers (
    role          TEXT NOT NULL,
    pointer       TEXT NOT NULL,
    artifact_id   TEXT NOT NULL,
    previous_id   TEXT,
    authorization_json TEXT NOT NULL DEFAULT '{}',
    moved_at      REAL NOT NULL,
    PRIMARY KEY (role, pointer)
);
CREATE TABLE IF NOT EXISTS pins (
    job_id               TEXT NOT NULL,
    role                 TEXT NOT NULL,
    artifact_id          TEXT NOT NULL,
    version_hash         TEXT NOT NULL,
    target_manifest_hash TEXT NOT NULL,
    pinned_at            REAL NOT NULL,
    released_at          REAL,
    PRIMARY KEY (job_id, role)
);
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    at          REAL NOT NULL,
    event       TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT '',
    artifact_id TEXT NOT NULL DEFAULT '',
    detail_json TEXT NOT NULL DEFAULT '{}'
);
"""


class ArtifactRegistry:
    """Durable, immutable artifact versions plus the pointers the master reads.

    One SQLite file, opened from as many processes as you like: every mutation
    is a single ``BEGIN IMMEDIATE`` transaction, so a check and the write that
    depends on it cannot be split by another process. The file *is* the state —
    a restart re-reads the same registrations, the same acceptances, the same
    active pointer and the same outstanding job pins.

    Immutability is by content address. :meth:`register` hashes the canonical
    artifact record; re-registering the identical record is a no-op, and
    re-registering the same ``artifact_id`` with different content is
    ``HASH_MISMATCH``. A changed artifact is a new version with its own id and
    its ``parent_id`` pointing at what it came from, which is also what
    :meth:`rollback` walks.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        store: Any = None,
        target_manifest_hash: str = "",
        target_manifest: Mapping[str, Any] | Path | str | None = None,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.path = Path(path)
        self.store = store
        self.target_manifest = _load_manifest(target_manifest)
        if not target_manifest_hash and self.target_manifest is not None:
            target_manifest_hash = str(self.target_manifest.get("manifest_hash") or "")
        self.target_manifest_hash = str(target_manifest_hash or "")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                ("contract_version", CONTRACT_VERSION),
            )
            if self.target_manifest_hash:
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                    ("target_manifest_hash", self.target_manifest_hash),
                )
            row = conn.execute("SELECT value FROM meta WHERE key = 'target_manifest_hash'").fetchone()
            if row is not None:
                declared = str(row[0])
                if self.target_manifest_hash and declared != self.target_manifest_hash:
                    raise ResolutionError(
                        ErrorCategory.TARGET_MISMATCH,
                        f"this registry file was created for target manifest {declared!r} and "
                        f"was opened for {self.target_manifest_hash!r}; one registry file "
                        f"holds one target's artifacts",
                        contract="ArtifactRegistry",
                        field="target_manifest_hash",
                    )
                self.target_manifest_hash = declared

    # -- plumbing ---------------------------------------------------------
    # The same connection shape :mod:`.budget` established: autocommit plus an
    # explicit ``BEGIN IMMEDIATE`` for every write, so a check and the write that
    # depends on it are one transaction, and the connection is always closed.
    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            with contextlib.suppress(sqlite3.DatabaseError):
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _event(
        self,
        conn: sqlite3.Connection,
        event: str,
        *,
        role: str = "",
        artifact_id: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO events(at, event, role, artifact_id, detail_json) VALUES(?,?,?,?,?)",
            (time.time(), event, role, artifact_id, canonical_json(dict(detail or {}))),
        )

    @staticmethod
    def _role(value: Any) -> str:
        role = str(value or "").strip()
        if not role:
            raise ResolutionError(
                ErrorCategory.EMPTY_IDENTIFIER,
                "a registration needs a role: which part of the master this artifact backs. "
                "An unnamed role cannot be pointed at, activated or rolled back",
                contract="ArtifactRegistry",
                field="role",
            )
        if role in RESERVED_ROLES:
            raise ResolutionError(
                ErrorCategory.ACTION_NOT_PERMITTED,
                f"{role!r} names the independent evaluator. An evolved artifact may guide "
                f"generation or ranking and is never substituted into the judge "
                f"(Appendix A §3.3). Reserved roles: {list(RESERVED_ROLES)}",
                contract="ArtifactRegistry",
                field="role",
            )
        return role

    # -- registration -----------------------------------------------------
    def register(
        self,
        artifact: Mapping[str, Any],
        *,
        role: str,
        loader: str = "argv",
        registered_by: str = "",
        fixture: bool = False,
        note: str = "",
    ) -> Record:
        """Register one immutable artifact version as a ``development_candidate``.

        Every registration starts in the same state whatever its evidence looks
        like. Acceptance and activation are separate calls with separate
        authority, because a registry that could promote on the way in would
        make "registered" and "in production" the same word.

        Args:
            artifact: an :class:`~..target.contracts.EvolutionArtifact` record.
            role: which part of the master this artifact backs. Reserved
                evaluator roles are refused.
            loader: one of :data:`LOADERS`.
            registered_by: who or what registered it.
            fixture: mark the row as a labelled synthetic engineering fixture,
                so acceptance-state machinery can be exercised without any real
                artifact being described as accepted.
            note: free text kept with the row.

        Raises:
            ResolutionError: ``TARGET_MISMATCH`` for an artifact scoped to a
                different target, ``HASH_MISMATCH`` for a second, different
                record under an existing id, ``INVALID_ENUM`` for an unknown
                loader, ``ACTION_NOT_PERMITTED`` for a reserved role, and
                whatever the artifact contract raises for a structural defect.
        """
        role = self._role(role)
        if str(loader) not in LOADERS:
            raise ResolutionError(
                ErrorCategory.INVALID_ENUM,
                f"loader {loader!r} is not one of {sorted(LOADERS)}; the master loads an "
                f"artifact through a closed vocabulary, not through a path the artifact names",
                contract="ArtifactRegistry",
                field="loader",
            )
        target = self.target_manifest_hash or str(artifact.get("target_manifest_hash") or "")
        resolution = resolve_artifact(
            artifact, target, store=self.store, target_manifest=self.target_manifest
        )
        record = resolution["artifact"]
        artifact_id = str(record["artifact_id"])
        version_hash = str(resolution["version_hash"])
        now = time.time()
        already = False
        with self._write() as conn:
            existing = conn.execute(
                "SELECT version_hash, role, loader FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["version_hash"]) != version_hash:
                    raise ResolutionError(
                        ErrorCategory.HASH_MISMATCH,
                        f"{artifact_id!r} is already registered as {existing['version_hash']} and "
                        f"this record hashes to {version_hash}. A registered version is immutable: "
                        f"register the change as a new artifact whose parent_id is {artifact_id!r}",
                        contract="ArtifactRegistry",
                        field="artifact_id",
                    )
                # Registering the identical record again is a no-op, not an error:
                # a resumed campaign re-offering what it already registered must
                # not fail, and the content address is what proves it is the same.
                self._event(conn, "register_idempotent", role=role, artifact_id=artifact_id)
                already = True
            if not already:
                if not self.target_manifest_hash:
                    self.target_manifest_hash = target
                    conn.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                        ("target_manifest_hash", target),
                    )
                conn.execute(
                    "INSERT INTO artifacts(artifact_id, version_hash, role, kind, loader, "
                    "target_manifest_hash, parent_id, experiment_id, record_json, registered_at, "
                    "registered_by, fixture, note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        artifact_id,
                        version_hash,
                        role,
                        str(record["kind"]),
                        str(loader),
                        str(record["target_manifest_hash"]),
                        record["parent_id"],
                        str(record["experiment_id"]),
                        canonical_json(record),
                        now,
                        str(registered_by),
                        1 if fixture else 0,
                        str(note),
                    ),
                )
                self._event(
                    conn,
                    "register",
                    role=role,
                    artifact_id=artifact_id,
                    detail={
                        "version_hash": version_hash,
                        "kind": str(record["kind"]),
                        "loader": str(loader),
                        "fixture": bool(fixture),
                        "references_verified": bool(resolution["references_verified"]),
                    },
                )
        return self.entry(artifact_id)

    # -- reads ------------------------------------------------------------
    def entry(self, artifact_id: str) -> Record:
        """One registry entry: the immutable record plus its current state."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (str(artifact_id),)
            ).fetchone()
            if row is None:
                raise ResolutionError(
                    ErrorCategory.MISSING_FIELD,
                    f"no artifact {artifact_id!r} is registered here; an id with no registration "
                    f"is a claim, not an artifact",
                    contract="ArtifactRegistry",
                    field="artifact_id",
                    missing=[str(artifact_id)],
                )
            acceptance = conn.execute(
                "SELECT * FROM acceptances WHERE artifact_id = ?", (str(artifact_id),)
            ).fetchone()
            active = conn.execute(
                "SELECT artifact_id FROM pointers WHERE role = ? AND pointer = ?",
                (str(row["role"]), PointerName.MASTER_ACTIVE.value),
            ).fetchone()
        state = ArtifactState.DEVELOPMENT_CANDIDATE
        if acceptance is not None:
            state = ArtifactState.BENCHMARK_ACCEPTED
        if active is not None and str(active["artifact_id"]) == str(artifact_id):
            state = ArtifactState.MASTER_ACTIVE
        return {
            "contract_version": CONTRACT_VERSION,
            "artifact_id": str(row["artifact_id"]),
            "version_hash": str(row["version_hash"]),
            "role": str(row["role"]),
            "kind": str(row["kind"]),
            "loader": str(row["loader"]),
            "target_manifest_hash": str(row["target_manifest_hash"]),
            "parent_id": row["parent_id"],
            "experiment_id": str(row["experiment_id"]),
            "state": state.value,
            "benchmark_accepted": acceptance is not None,
            "acceptance": (
                {
                    "evidence": json.loads(str(acceptance["evidence_json"])),
                    "accepted_at": float(acceptance["accepted_at"]),
                    "accepted_by": str(acceptance["accepted_by"]),
                }
                if acceptance is not None
                else None
            ),
            "registered_at": float(row["registered_at"]),
            "registered_by": str(row["registered_by"]),
            "fixture": bool(row["fixture"]),
            "note": str(row["note"]),
            "artifact": json.loads(str(row["record_json"])),
        }

    def entries(self, *, role: str | None = None, state: str | None = None) -> list[Record]:
        """Every registration, newest last. Filtered by role and/or state."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT artifact_id FROM artifacts ORDER BY registered_at, artifact_id"
            ).fetchall()
        out = [self.entry(str(r["artifact_id"])) for r in rows]
        if role is not None:
            out = [e for e in out if e["role"] == str(role)]
        if state is not None:
            out = [e for e in out if e["state"] == str(state)]
        return out

    def pointers(self, *, role: str | None = None) -> list[Record]:
        """The candidate/champion and master-active pointers, per role."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pointers ORDER BY role, pointer").fetchall()
        out = [
            {
                "role": str(r["role"]),
                "pointer": str(r["pointer"]),
                "artifact_id": str(r["artifact_id"]),
                "previous_id": r["previous_id"],
                "authorization": json.loads(str(r["authorization_json"])),
                "moved_at": float(r["moved_at"]),
            }
            for r in rows
        ]
        return [p for p in out if role is None or p["role"] == str(role)]

    def events(self, limit: int = 200) -> list[Record]:
        """The append-only audit trail, oldest first."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY seq LIMIT ?", (int(limit),)).fetchall()
        return [
            {
                "seq": int(r["seq"]),
                "at": float(r["at"]),
                "event": str(r["event"]),
                "role": str(r["role"]),
                "artifact_id": str(r["artifact_id"]),
                "detail": json.loads(str(r["detail_json"])),
            }
            for r in rows
        ]

    # -- acceptance: evidence registration, not activation -----------------
    def accept(
        self,
        artifact_id: str,
        *,
        evidence: Mapping[str, Any],
        accepted_by: str,
    ) -> Record:
        """Record that an independent comparison accepted this artifact.

        This registers *evidence*. It does not change what the master loads, it
        does not touch a pointer, and it does not affect a running job. Appendix
        A §5's ``BENCHMARK_ACCEPTED`` is "a reproducible positive improvement on
        the registered metric, with its paired uncertainty interval excluding no
        improvement and all relevant constraints satisfied" — every one of those
        conditions is enforced by the
        :class:`~..target.contracts.Comparison` contract, which is why the
        evidence must be one rather than a free-form dictionary.

        Raises:
            ResolutionError: ``MISSING_FIELD`` for an unregistered artifact,
                ``INVALID_VALUE`` when the comparison is about a different
                candidate, ``UNSUPPORTED_CLAIM`` when the decision is not
                ``IMPROVED``; and whatever the ``Comparison`` contract raises,
                including ``UNSUPPORTED_CLAIM`` for an interval that does not
                exclude no improvement.
        """
        entry = self.entry(artifact_id)
        comparison = validate_record("Comparison", evidence)
        if str(comparison["candidate_id"]) != str(artifact_id):
            raise ResolutionError(
                ErrorCategory.INVALID_VALUE,
                f"the comparison is about candidate {comparison['candidate_id']!r} and this "
                f"acceptance is for {artifact_id!r}; evidence is not transferable between "
                f"artifacts",
                contract="Comparison",
                field="candidate_id",
            )
        if str(comparison["decision"]) != "IMPROVED":
            raise ResolutionError(
                ErrorCategory.UNSUPPORTED_CLAIM,
                f"benchmark acceptance needs an IMPROVED comparison; this one decided "
                f"{comparison['decision']!r}. NO_IMPROVEMENT, INCONCLUSIVE and "
                f"INSUFFICIENT_DATA are complete scientific outcomes and none of them "
                f"accepts an artifact",
                contract="Comparison",
                field="decision",
            )
        approver = str(accepted_by or "").strip()
        if not approver:
            raise ResolutionError(
                ErrorCategory.EMPTY_IDENTIFIER,
                "an acceptance records who accepted it",
                contract="ArtifactRegistry",
                field="accepted_by",
            )
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO acceptances(artifact_id, evidence_json, accepted_at, "
                "accepted_by) VALUES(?,?,?,?)",
                (str(artifact_id), canonical_json(comparison), time.time(), approver),
            )
            self._event(
                conn,
                "benchmark_accepted",
                role=entry["role"],
                artifact_id=str(artifact_id),
                detail={
                    "effect": comparison["effect"],
                    "interval": comparison["interval"],
                    "primary_metric": comparison["primary_metric"],
                    "claim_scope": comparison["claim_scope"],
                    "activation": "none — acceptance registers evidence and moves no pointer",
                },
            )
        return self.entry(artifact_id)

    # -- pointers ---------------------------------------------------------
    def set_champion(self, artifact_id: str, *, selected_by: str, rule: str) -> Record:
        """Point ``champion`` at a registered candidate for its role.

        A development-evidence selection. It is deliberately cheap — no operator
        approval, no effect on the master — because it says what the campaign
        would nominate, not what production runs.
        """
        entry = self.entry(artifact_id)
        return self._move_pointer(
            role=str(entry["role"]),
            pointer=PointerName.CHAMPION,
            artifact_id=str(artifact_id),
            authorization={"selected_by": str(selected_by), "development_selection_rule": str(rule)},
            event="champion_set",
        )

    def activate(
        self,
        artifact_id: str,
        *,
        authorization: Mapping[str, Any],
        config: Any = None,
    ) -> Record:
        """Make a registered artifact ``master_active``, under the existing policy.

        This changes how the user-serving system behaves, so it reuses the
        deployment's adoption authorization rather than inventing a second one:
        :class:`drug_agent.selfevolve.config.EvolveConfig` ships ``auto_promote``
        off, and ``drugagent self-evolve promote`` refuses without
        ``--i-approve``. Here the equivalent is
        ``authorization["operator_approved"] is True`` plus a named approver.
        With ``auto_promote`` on, an authorization that says so is accepted and
        the record says which of the two routes was used — it is never silent.

        Activation does not touch any :meth:`pin`, so a job already running
        keeps the artifact it started with.
        """
        entry = self.entry(artifact_id)
        decision = require_adoption_authorization(authorization, config=config)
        return self._move_pointer(
            role=str(entry["role"]),
            pointer=PointerName.MASTER_ACTIVE,
            artifact_id=str(artifact_id),
            authorization=decision,
            event="master_activated",
            detail={
                "state_before": entry["state"],
                "benchmark_accepted": entry["benchmark_accepted"],
                "fixture": entry["fixture"],
                "running_jobs_unaffected": (
                    "pins are resolved at job startup and are not rewritten by an activation"
                ),
            },
        )

    def rollback(
        self,
        role: str,
        *,
        pointer: str | PointerName = PointerName.MASTER_ACTIVE,
        authorization: Mapping[str, Any] | None = None,
        to: str | None = None,
        config: Any = None,
    ) -> Record:
        """Move a pointer back to the current artifact's parent.

        Rollback deliberately does **not** require the parent to be
        ``benchmark_accepted``: the point of a rollback is to reach a known
        previous state, and requiring fresh evidence to undo a bad activation
        would be a trap. It does require the parent to be registered here and
        target-compatible, and — for ``master_active`` — the same operator
        authorization an activation needs, because reverting production is also
        a production change.
        """
        name = PointerName(str(pointer))
        current = self.pointers(role=str(role))
        pointed = next((p for p in current if p["pointer"] == name.value), None)
        if pointed is None:
            raise ResolutionError(
                ErrorCategory.MISSING_FIELD,
                f"role {role!r} has no {name.value} pointer to roll back",
                contract="ArtifactRegistry",
                field="pointer",
            )
        entry = self.entry(str(pointed["artifact_id"]))
        destination = str(to) if to else entry["parent_id"]
        if not destination:
            raise ResolutionError(
                ErrorCategory.MISSING_FIELD,
                f"{entry['artifact_id']!r} declares no parent, so there is no earlier version "
                f"to roll back to. Name one explicitly with to=... if a different artifact is "
                f"the intended fallback",
                contract="ArtifactRegistry",
                field="parent_id",
            )
        parent = self.entry(destination)
        if str(parent["role"]) != str(role):
            raise ResolutionError(
                ErrorCategory.INVALID_VALUE,
                f"{destination!r} backs role {parent['role']!r}, not {role!r}; a rollback stays "
                f"inside one role",
                contract="ArtifactRegistry",
                field="role",
            )
        decision: Record
        if name is PointerName.MASTER_ACTIVE:
            decision = require_adoption_authorization(authorization or {}, config=config)
        else:
            decision = {"operator_approved": False, "route": "development_pointer"}
        decision = dict(decision)
        decision["rolled_back_from"] = entry["artifact_id"]
        return self._move_pointer(
            role=str(role),
            pointer=name,
            artifact_id=destination,
            authorization=decision,
            event="rollback",
            detail={"from": entry["artifact_id"], "to": destination},
        )

    def _move_pointer(
        self,
        *,
        role: str,
        pointer: PointerName,
        artifact_id: str,
        authorization: Mapping[str, Any],
        event: str,
        detail: Mapping[str, Any] | None = None,
    ) -> Record:
        entry = self.entry(artifact_id)
        if str(entry["role"]) != str(role):
            raise ResolutionError(
                ErrorCategory.INVALID_VALUE,
                f"{artifact_id!r} backs role {entry['role']!r}, not {role!r}",
                contract="ArtifactRegistry",
                field="role",
            )
        # The pointer the master reads must resolve, here, now: an activation
        # that cannot be loaded is an outage rather than an adoption.
        self.resolve(artifact_id)
        now = time.time()
        with self._write() as conn:
            previous = conn.execute(
                "SELECT artifact_id FROM pointers WHERE role = ? AND pointer = ?",
                (role, pointer.value),
            ).fetchone()
            previous_id = str(previous["artifact_id"]) if previous is not None else None
            conn.execute(
                "INSERT INTO pointers(role, pointer, artifact_id, previous_id, "
                "authorization_json, moved_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(role, pointer) DO UPDATE SET artifact_id = excluded.artifact_id, "
                "previous_id = excluded.previous_id, "
                "authorization_json = excluded.authorization_json, moved_at = excluded.moved_at",
                (role, pointer.value, artifact_id, previous_id, canonical_json(dict(authorization)), now),
            )
            self._event(
                conn,
                event,
                role=role,
                artifact_id=artifact_id,
                detail={
                    "pointer": pointer.value,
                    "previous_id": previous_id,
                    "authorization": dict(authorization),
                    **dict(detail or {}),
                },
            )
        return self.entry(artifact_id)

    # -- resolution and pinning -------------------------------------------
    def resolve(
        self,
        artifact_id: str,
        *,
        target_hash: str | None = None,
        materialize_into: Path | str | None = None,
    ) -> Record:
        """Resolve a registered artifact for this registry's target.

        The bytes are re-checked against the store every time, so an artifact
        whose weights were deleted after registration fails here rather than at
        the moment the master tries to score with it.
        """
        entry = self.entry(artifact_id)
        wanted = str(target_hash) if target_hash else self.target_manifest_hash
        resolution = resolve_artifact(
            entry["artifact"],
            wanted,
            store=self.store,
            materialize_into=materialize_into,
            target_manifest=self.target_manifest,
        )
        resolution["role"] = entry["role"]
        resolution["loader"] = entry["loader"]
        resolution["state"] = entry["state"]
        resolution["fixture"] = entry["fixture"]
        return resolution

    def load(self, artifact_id: str, *, into: Path | str) -> Any:
        """Materialize and load a registered artifact through its declared loader."""
        entry = self.entry(artifact_id)
        resolution = self.resolve(artifact_id, materialize_into=into)
        return LOADERS[str(entry["loader"])](resolution)

    def pin(self, *, job_id: str, role: str, target_hash: str | None = None) -> Record:
        """Resolve the role's active artifact once, and record the pin durably.

        Called at job startup. Everything the job does afterwards reads the pin,
        never the pointer, so an activation or rollback that happens while the
        job runs cannot change what it is using. The pin survives a restart
        because it is in the registry file, not in the job's memory.
        """
        job = str(job_id or "").strip()
        if not job:
            raise ResolutionError(
                ErrorCategory.EMPTY_IDENTIFIER,
                "a pin belongs to a named job; an anonymous pin cannot be looked up again",
                contract="ArtifactRegistry",
                field="job_id",
            )
        role_name = str(role)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM pins WHERE job_id = ? AND role = ? AND released_at IS NULL",
                (job, role_name),
            ).fetchone()
        if existing is not None:
            return self._pin_record(existing)
        pointed = next(
            (p for p in self.pointers(role=role_name) if p["pointer"] == PointerName.MASTER_ACTIVE.value),
            None,
        )
        if pointed is None:
            raise ResolutionError(
                ErrorCategory.MISSING_FIELD,
                f"role {role_name!r} has no master_active artifact, so there is nothing to pin. "
                f"A registered or benchmark-accepted artifact is not active: activation is a "
                f"separate, authorized step",
                contract="ArtifactRegistry",
                field="pointer",
            )
        resolution = self.resolve(str(pointed["artifact_id"]), target_hash=target_hash)
        now = time.time()
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pins(job_id, role, artifact_id, version_hash, "
                "target_manifest_hash, pinned_at, released_at) VALUES(?,?,?,?,?,?,NULL)",
                (
                    job,
                    role_name,
                    str(resolution["artifact_id"]),
                    str(resolution["version_hash"]),
                    str(resolution["target_manifest_hash"]),
                    now,
                ),
            )
            self._event(
                conn,
                "pinned",
                role=role_name,
                artifact_id=str(resolution["artifact_id"]),
                detail={"job_id": job, "version_hash": resolution["version_hash"]},
            )
            row = conn.execute(
                "SELECT * FROM pins WHERE job_id = ? AND role = ?", (job, role_name)
            ).fetchone()
        return self._pin_record(row)

    def pinned(self, *, job_id: str, role: str) -> Record | None:
        """What this job pinned at startup, whatever the pointer says now."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pins WHERE job_id = ? AND role = ? AND released_at IS NULL",
                (str(job_id), str(role)),
            ).fetchone()
        return self._pin_record(row) if row is not None else None

    def release(self, *, job_id: str, role: str) -> None:
        """Mark a job's pin finished. The row is kept; it is the audit trail."""
        with self._write() as conn:
            conn.execute(
                "UPDATE pins SET released_at = ? WHERE job_id = ? AND role = ? AND released_at IS NULL",
                (time.time(), str(job_id), str(role)),
            )
            self._event(conn, "released", role=str(role), detail={"job_id": str(job_id)})

    @staticmethod
    def _pin_record(row: sqlite3.Row) -> Record:
        return {
            "job_id": str(row["job_id"]),
            "role": str(row["role"]),
            "artifact_id": str(row["artifact_id"]),
            "version_hash": str(row["version_hash"]),
            "target_manifest_hash": str(row["target_manifest_hash"]),
            "pinned_at": float(row["pinned_at"]),
            "released_at": None if row["released_at"] is None else float(row["released_at"]),
        }

    # -- reporting --------------------------------------------------------
    def report(self) -> Record:
        """Everything an operator needs to see: states, pointers and open pins.

        ``kinds_registered`` is a count of schema support. It says nothing about
        whether any of those kinds has been shown to help, which is why the
        record carries that sentence rather than leaving it to be assumed.
        """
        entries = self.entries()
        by_state: dict[str, int] = {state.value: 0 for state in ArtifactState}
        kinds: dict[str, int] = {}
        for entry in entries:
            by_state[str(entry["state"])] = by_state.get(str(entry["state"]), 0) + 1
            kinds[str(entry["kind"])] = kinds.get(str(entry["kind"]), 0) + 1
        with self._connect() as conn:
            open_pins = [
                self._pin_record(r)
                for r in conn.execute(
                    "SELECT * FROM pins WHERE released_at IS NULL ORDER BY pinned_at"
                ).fetchall()
            ]
        return {
            "contract_version": CONTRACT_VERSION,
            "path": str(self.path),
            "target_manifest_hash": self.target_manifest_hash,
            "n_artifacts": len(entries),
            "n_fixture_artifacts": sum(1 for e in entries if e["fixture"]),
            "by_state": by_state,
            "kinds_registered": kinds,
            "kinds_note": (
                "a count of schema support, not of demonstrated gains. Whether a particular "
                "artifact helped is its Comparison record and nothing else."
            ),
            "pointers": self.pointers(),
            "open_pins": open_pins,
            "referee": referee_receptor(),
        }


# ---------------------------------------------------------------------------
# Adoption authorization — the deployment's existing policy, reused
# ---------------------------------------------------------------------------
def require_adoption_authorization(authorization: Mapping[str, Any], *, config: Any = None) -> Record:
    """Refuse an activation the deployment's adoption policy would refuse.

    There is already a policy in this repository and this does not replace it:
    :class:`drug_agent.selfevolve.config.EvolveConfig` ships ``auto_promote``
    false, ``drugagent self-evolve promote`` prints the report and refuses
    without ``--i-approve``, and the promoted overlay still has to be switched
    on by an environment variable. The registry's equivalent of ``--i-approve``
    is ``operator_approved``; the equivalent of the overlay file is the
    ``master_active`` pointer.

    Returns:
        The authorization as recorded, with ``route`` naming which of the two
        permitted routes was taken.

    Raises:
        ResolutionError: ``ACTION_NOT_PERMITTED`` when neither route applies,
            ``EMPTY_IDENTIFIER`` when nobody is named as the approver.
    """
    if config is None:
        from ..config import EvolveConfig

        config = EvolveConfig()
    auto = bool(getattr(config, "auto_promote", False))
    approved = bool(authorization.get("operator_approved"))
    approver = str(authorization.get("approved_by") or "").strip()
    if not approved and not auto:
        raise ResolutionError(
            ErrorCategory.ACTION_NOT_PERMITTED,
            "activating an artifact changes how the user-serving system behaves. "
            "EvolveConfig.auto_promote is off (the default), so this needs an explicit "
            "operator approval: pass authorization={'operator_approved': True, "
            "'approved_by': ...}. Benchmark acceptance registers evidence and does not "
            "activate anything.",
            contract="ArtifactRegistry",
            field="authorization",
        )
    if not approver:
        raise ResolutionError(
            ErrorCategory.EMPTY_IDENTIFIER,
            "an activation records who authorized it; an anonymous adoption cannot be audited",
            contract="ArtifactRegistry",
            field="approved_by",
        )
    return {
        "operator_approved": approved,
        "approved_by": approver,
        "note": str(authorization.get("note") or ""),
        "auto_promote_setting": auto,
        "route": "operator_approved" if approved else "auto_promote_enabled",
        "authorized_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Master consumption
# ---------------------------------------------------------------------------
#: Environment override for the registry file, mirroring the ``Settings`` field
#: name so ``DRUGAGENT_TARGET_ARTIFACT_REGISTRY`` works with or without a
#: ``Settings`` instance in hand.
REGISTRY_ENV = "DRUGAGENT_TARGET_ARTIFACT_REGISTRY"


def _settings_value(settings: Any, name: str) -> Any:
    return getattr(settings, name, None) if settings is not None else None


def default_store_for(path: Path | str) -> Any:
    """The content-addressed store that sits beside a registry file.

    One convention in one place: ``<registry>.sqlite`` is accompanied by
    ``<registry parent>/artifacts``. The CLI, the tool pool and
    :func:`open_master_job` all use it, so a deployment does not have to be told
    twice where the bytes are.
    """
    from ...storage.artifacts import LocalArtifactStore

    return LocalArtifactStore(Path(path).parent / "artifacts")


def registry_path(settings: Any = None) -> Path | None:
    """Where this deployment's artifact registry lives, or ``None``.

    An absent registry is not an error: a deployment that has adopted nothing
    has nothing to read, and that is reported as ``NOT_CONFIGURED`` rather than
    as a failure.
    """
    configured = _settings_value(settings, "target_artifact_registry")
    if configured:
        return Path(str(configured)).expanduser()
    env = os.environ.get(REGISTRY_ENV, "").strip()
    return Path(env).expanduser() if env else None


def master_target_hash(settings: Any = None) -> str:
    """The target manifest hash this deployment declares, or ``""``.

    Read from the deployment's own ``target_manifest`` setting — the same file
    the generator's pocket is checked against — so "which target is this master
    serving" has one answer, not two.
    """
    manifest = _settings_value(settings, "target_manifest")
    if not manifest:
        return ""
    try:
        record = json.loads(Path(str(manifest)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(record.get("manifest_hash") or "")


def open_master_job(
    job_id: str,
    *,
    roles: Sequence[str],
    settings: Any = None,
    registry: ArtifactRegistry | None = None,
    target_hash: str | None = None,
) -> MasterJob:
    """Start a master job and pin every evolved component it will use.

    Resolution happens **once**, here, at startup. Everything the job does
    afterwards goes through :meth:`MasterJob.pinned`, so a version accepted or
    activated while the job runs changes nothing for it. A role with no
    ``master_active`` artifact is recorded as unpinned rather than substituted:
    the master then uses whatever it used before, which is the correct behaviour
    for a deployment that has adopted nothing.
    """
    if registry is None:
        path = registry_path(settings)
        if path is None or not Path(path).exists():
            return MasterJob(
                job_id=str(job_id),
                registry=None,
                pins={},
                unpinned=list(roles),
                reason="NOT_CONFIGURED" if path is None else "REGISTRY_ABSENT",
            )
        # Constructed without the manifest so that a deployment/registry target
        # disagreement is reported by the explicit check below, which names both
        # hashes and says what loading them anyway would mean, rather than by the
        # constructor's more general one.
        registry = ArtifactRegistry(path, store=default_store_for(path))
        registry.target_manifest = _load_manifest(_settings_value(settings, "target_manifest"))
    wanted = str(target_hash) if target_hash else (master_target_hash(settings) or None)
    if wanted and registry.target_manifest_hash and wanted != registry.target_manifest_hash:
        raise ResolutionError(
            ErrorCategory.TARGET_MISMATCH,
            f"this deployment declares target manifest {wanted!r} and the registry at "
            f"{registry.path} holds artifacts for {registry.target_manifest_hash!r}. "
            f"Loading them here would apply another target's evolution to this one",
            contract="ArtifactRegistry",
            field="target_manifest_hash",
        )
    pins: dict[str, Record] = {}
    unpinned: list[str] = []
    for role in roles:
        try:
            pins[str(role)] = registry.pin(job_id=str(job_id), role=str(role), target_hash=wanted)
        except ResolutionError as exc:
            if exc.category is not ErrorCategory.MISSING_FIELD:
                raise
            unpinned.append(str(role))
    return MasterJob(job_id=str(job_id), registry=registry, pins=pins, unpinned=unpinned, reason="OK")


class MasterJob:
    """One master run's frozen view of the evolved artifacts it may use.

    Holds pins, not pointers. :meth:`load` reloads a pinned artifact from its
    content addresses, in this process, with no reference to the campaign that
    produced it — which is the acceptance criterion this step exists to satisfy.
    """

    def __init__(
        self,
        *,
        job_id: str,
        registry: ArtifactRegistry | None,
        pins: Mapping[str, Record],
        unpinned: Sequence[str],
        reason: str,
    ) -> None:
        self.job_id = str(job_id)
        self.registry = registry
        self.pins = dict(pins)
        self.unpinned = list(unpinned)
        self.reason = str(reason)

    def pinned(self, role: str) -> Record | None:
        """The version this job pinned for *role*, or ``None`` if it pinned none."""
        return self.pins.get(str(role))

    def resolve(self, role: str, *, materialize_into: Path | str | None = None) -> Record:
        """Resolve the pinned version — never whatever is active now."""
        pin = self.pins.get(str(role))
        if pin is None or self.registry is None:
            raise ResolutionError(
                ErrorCategory.MISSING_FIELD,
                f"job {self.job_id!r} pinned no artifact for role {role!r} ({self.reason})",
                contract="MasterJob",
                field="role",
            )
        resolution = self.registry.resolve(str(pin["artifact_id"]), materialize_into=materialize_into)
        if str(resolution["version_hash"]) != str(pin["version_hash"]):  # pragma: no cover
            raise ResolutionError(
                ErrorCategory.HASH_MISMATCH,
                f"the pinned version {pin['version_hash']} no longer matches the registered "
                f"record {resolution['version_hash']}; a registered version is immutable",
                contract="MasterJob",
                field="version_hash",
            )
        resolution["pinned_for_job"] = self.job_id
        return resolution

    def load(self, role: str, *, into: Path | str) -> Any:
        """Materialize and load the pinned artifact through its declared loader."""
        pin = self.pins.get(str(role))
        if pin is None or self.registry is None:
            raise ResolutionError(
                ErrorCategory.MISSING_FIELD,
                f"job {self.job_id!r} pinned no artifact for role {role!r} ({self.reason})",
                contract="MasterJob",
                field="role",
            )
        return self.registry.load(str(pin["artifact_id"]), into=into)

    def close(self) -> None:
        """Release every pin. The rows stay as the audit trail."""
        if self.registry is None:
            return
        for role in list(self.pins):
            self.registry.release(job_id=self.job_id, role=role)

    def manifest(self) -> Record:
        """What this job is running with, for the run record."""
        return {
            "contract_version": CONTRACT_VERSION,
            "job_id": self.job_id,
            "status": self.reason,
            "registry": str(self.registry.path) if self.registry is not None else None,
            "pins": dict(self.pins),
            "unpinned_roles": list(self.unpinned),
        }


def master_tool_cards(settings: Any = None) -> list[Record]:
    """Describe every ``master_active`` evolved artifact as a tool-pool entry.

    This is the seam :func:`drug_agent.tools.registry.tool_pool` calls, so an
    adopted artifact shows up in ``drugagent tools``, the capability matrix and
    the run manifest exactly like every other capability — including, and
    especially, when it is ``UNAVAILABLE``. Three things make a card
    unavailable, and each one names itself:

    * the registry is configured but absent;
    * the deployment's declared target manifest is not the one the artifact was
      evolved for, which is the wrong-target failure this module exists to make
      loud; or
    * a referenced component is no longer in the store.

    Returns an empty list when no registry is configured. A deployment that has
    adopted nothing is not a deployment with a broken tool.
    """
    path = registry_path(settings)
    if path is None or not Path(path).exists():
        return []
    wanted = master_target_hash(settings)
    try:
        # Deliberately opened without the deployment's manifest: a *listing* must
        # report a wrong-target artifact as an UNAVAILABLE card naming both
        # hashes, not raise. The comparison happens per artifact below.
        registry = ArtifactRegistry(path, store=default_store_for(path))
    except (ResolutionError, sqlite3.Error) as exc:  # pragma: no cover - unreadable file
        return [
            {
                "tool_id": "target.evolved.registry",
                "title": "Evolved target artifacts",
                "status": "UNAVAILABLE",
                "unavailable_reason": f"{path} is not a readable artifact registry ({exc!r})",
            }
        ]
    cards: list[Record] = []
    for pointer in registry.pointers():
        if pointer["pointer"] != PointerName.MASTER_ACTIVE.value:
            continue
        entry = registry.entry(str(pointer["artifact_id"]))
        status, reason = "AVAILABLE", None
        if wanted and str(entry["target_manifest_hash"]) != wanted:
            status = "UNAVAILABLE"
            reason = (
                f"this artifact is scoped to target manifest "
                f"{entry['target_manifest_hash']} and this deployment declares {wanted}"
            )
        else:
            try:
                registry.resolve(str(entry["artifact_id"]))
            except ResolutionError as exc:
                status, reason = "UNAVAILABLE", str(exc)
        cards.append(
            {
                "tool_id": f"target.evolved.{entry['role']}",
                "title": f"Evolved {entry['role']} ({entry['kind']})",
                "description": (
                    f"Artifact {entry['artifact_id']} from experiment {entry['experiment_id']}, "
                    f"loaded through the {entry['loader']} loader. Registered support for an "
                    f"artifact kind is not evidence that this artifact improves anything; that "
                    f"is its Comparison record."
                ),
                "status": status,
                "unavailable_reason": reason,
                "backend": "target-artifact-registry",
                "artifact_id": entry["artifact_id"],
                "version_hash": entry["version_hash"],
                "kind": entry["kind"],
                "role": entry["role"],
                "state": entry["state"],
                "benchmark_accepted": entry["benchmark_accepted"],
                "fixture": entry["fixture"],
                "target_manifest_hash": entry["target_manifest_hash"],
                "registry_path": str(path),
            }
        )
    return cards


# ---------------------------------------------------------------------------
# A complete valid artifact, for tests that need to change exactly one field
# ---------------------------------------------------------------------------
#: The bytes the fixture stores for each component group. Small, inert, and
#: obviously synthetic.
#: Each entry is a format-string template taking the component index, so two
#: members of one group are distinct content addresses while every JSON-shaped
#: component stays valid JSON and is therefore loadable.
_FIXTURE_BYTES: dict[str, str] = {
    "code": '{{"kind": "synthetic engineering fixture", "component": "code", "index": {index}}}\n',
    "environment": '{{"kind": "synthetic engineering fixture", "python": "3.12", "index": {index}}}\n',
    "weights": '{{"kind": "synthetic engineering fixture", "weights": [0.0], "index": {index}}}\n',
    "structures": "REMARK synthetic engineering fixture receptor {index}\nEND\n",
    "training_data": '{{"kind": "synthetic engineering fixture", "rows": [], "index": {index}}}\n',
}


def example_artifact(
    store: Any = None,
    *,
    kind: str = ArtifactKind.ALGORITHM.value,
    artifact_id: str = "art_fixture_001",
    target_manifest_hash: str = "sha256:" + "1" * 64,
    n_structures: int = 0,
    n_weights: int = 0,
    n_training_data: int = 0,
    **overrides: Any,
) -> Record:
    """A **synthetic engineering fixture**: one complete, valid artifact record.

    It exists so that every negative test can start from a record that really
    validates and change exactly one field. A rejection is only evidence about
    the field under test if the record would otherwise have been accepted, and
    the positive companion is asserted in the same test file.

    With a ``store``, the component bytes are actually written to it and the
    record carries their real content addresses, so the fixture can be resolved,
    materialized and reloaded rather than only validated.

    Nothing here is a research measurement. No number in this record means
    anything about ZAK, about binding, or about any artifact's performance.
    """
    counts = {
        "weights": max(int(n_weights), 1 if kind in ("WEIGHTS", "TRAINED_MODEL", "SCORER") else 0),
        "structures": max(
            int(n_structures),
            2 if kind == "RECEPTOR_ENSEMBLE" else (1 if kind == "RECEPTOR_STRUCTURE" else 0),
        ),
        "training_data": max(int(n_training_data), 1 if kind == "DATA_OR_LIBRARY" else 0),
    }

    def address(group: str, index: int = 0) -> str:
        data = _FIXTURE_BYTES[group].format(index=index).encode("utf-8")
        if store is not None:
            return str(
                store.put_bytes(
                    data, kind=f"fixture_{group}", produced_by="example_artifact", metadata={}
                ).content_hash
            )
        return hash_bytes(data)

    record: Record = {
        "artifact_id": artifact_id,
        "parent_id": None,
        "kind": str(kind),
        "target_manifest_hash": target_manifest_hash,
        "entrypoint": ["python", "-m", "drug_agent.selfevolve.target.registry", "--stage", "report"],
        "code_hash": address("code"),
        "weights_hashes": [address("weights", i) for i in range(counts["weights"])],
        "structure_hashes": [address("structures", i) for i in range(counts["structures"])],
        "environment_hash": address("environment"),
        "training_data_hashes": [address("training_data", i) for i in range(counts["training_data"])],
        "experiment_id": "exp_fixture_001",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Operator entry point
# ---------------------------------------------------------------------------
def _cli(argv: Sequence[str] | None = None) -> int:
    """``python -m drug_agent.selfevolve.target.registry --stage ...``.

    The supported way to look at, resolve and load a registered artifact without
    a Python session — and therefore without the conversation that produced it.
    ``activate`` deliberately refuses without ``--i-approve``, exactly as
    ``drugagent self-evolve promote`` does.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m drug_agent.selfevolve.target.registry",
        description=(
            "Immutable evolution-artifact versions and the pointers the master reads. "
            "register/accept/activate are three separate authorities on purpose: "
            "benchmark acceptance registers evidence and activates nothing."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("report", "list", "resolve", "load", "register", "accept", "activate", "rollback"),
    )
    parser.add_argument("--registry", default="", help="path to the registry file")
    parser.add_argument("--artifact", default="", help="artifact id, or a JSON record to register")
    parser.add_argument("--role", default="")
    parser.add_argument("--loader", default="argv", choices=sorted(LOADERS))
    parser.add_argument("--evidence", default="", help="path to a Comparison record (accept)")
    parser.add_argument("--into", default="", help="directory to materialize components into")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--fixture", action="store_true", help="mark as a synthetic test fixture")
    parser.add_argument(
        "--i-approve",
        action="store_true",
        help="the operator approval an activation needs. Without it, activate refuses.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = args.registry or os.environ.get(REGISTRY_ENV, "")
    if not path:
        print(f"no registry: pass --registry or set {REGISTRY_ENV}")
        return 2
    registry = ArtifactRegistry(path, store=default_store_for(path))

    def emit(payload: Any) -> int:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    if args.stage == "report":
        return emit(registry.report())
    if args.stage == "list":
        return emit(registry.entries(role=args.role or None))
    if args.stage == "resolve":
        return emit(registry.resolve(args.artifact, materialize_into=args.into or None))
    if args.stage == "load":
        if not args.into:
            print("--into is required for load: the components have to be written somewhere")
            return 2
        loaded = registry.load(args.artifact, into=args.into)
        return emit({"artifact_id": args.artifact, "loaded": repr(loaded)})
    if args.stage == "register":
        record = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
        return emit(
            registry.register(
                record,
                role=args.role,
                loader=args.loader,
                registered_by=args.approved_by or "cli",
                fixture=bool(args.fixture),
                note=args.note,
            )
        )
    if args.stage == "accept":
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        return emit(registry.accept(args.artifact, evidence=evidence, accepted_by=args.approved_by or "cli"))
    if args.stage == "activate":
        if not args.i_approve:
            print(
                "Not activated. Activating an artifact changes how the user-serving system "
                "behaves. Re-run with --i-approve once you have read the evidence:\n"
                f"  python -m drug_agent.selfevolve.target.registry --stage resolve "
                f"--registry {path} --artifact {args.artifact}"
            )
            return 0
        return emit(
            registry.activate(
                args.artifact,
                authorization={
                    "operator_approved": True,
                    "approved_by": args.approved_by or os.environ.get("USER", "operator"),
                    "note": args.note,
                },
            )
        )
    if args.stage == "rollback":
        if not args.i_approve:
            print("Not rolled back. Re-run with --i-approve: this changes the deployment.")
            return 0
        return emit(
            registry.rollback(
                args.role,
                authorization={
                    "operator_approved": True,
                    "approved_by": args.approved_by or os.environ.get("USER", "operator"),
                    "note": args.note,
                },
            )
        )
    return 2  # pragma: no cover - argparse restricts the choices


if __name__ == "__main__":  # pragma: no cover - operator entry point
    raise SystemExit(_cli())
