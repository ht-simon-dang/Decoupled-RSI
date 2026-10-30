"""Materialise a candidate and evaluate *its* code, in its own process.

Step 4 of the per-target evolution program. The defect this module exists to
close is the audit's gate-0 finding, re-confirmed at Step 1: the evaluation
kernel defaulted its working directory to the operator's checkout and no caller
ever passed anything else, so every gate-0 verdict described the operator's
working tree. A candidate carrying a syntax error passed the gate whose entire
purpose is to catch one.

Four things have to be true at once for an evaluation to be about the candidate,
and each of them is a separate failure mode that has actually happened here:

1. **The files must be the candidate's.** A patch is applied inside an isolated
   checkout (:mod:`..candidates.workspace`), never in the live tree, and the
   resulting tree is content-addressed so the record names what really ran.
2. **The interpreter must load the candidate's source.** This is the one that
   bites: the conda environment carries an *editable install* of the parent
   checkout, so a subprocess that merely ``cd``\\ s into a candidate still
   imports the parent. :func:`candidate_env` puts the candidate's source root on
   ``PYTHONPATH`` ahead of it, and :func:`probe_origin` then *verifies* in a
   fresh subprocess where the module actually came from. Verification is the
   load-bearing half: a mechanism that silently fails is how the defect got here.
3. **Weights are part of the candidate.** A weights-only candidate has source
   byte-identical to its parent, so source identity alone proves nothing.
   :func:`weight_digests` hashes the checkpoints the candidate actually carries.
4. **The parent must come back untouched.** Nothing here writes to the parent
   checkout; the acceptance tests hash it before and after.

Failure has categories, and they are not interchangeable. A patch that will not
apply is ``INVALID_CANDIDATE``. A candidate whose own code is broken is
``CANDIDATE_ERROR`` — a *result*, recorded and never retried. A missing
interpreter or a checkout that could not be created is ``INFRASTRUCTURE_ERROR``
— retried under the same logical experiment ID, with every attempt charged. A
command that outlives its budget is ``TIMEOUT``, and its whole process group is
terminated so nothing is left holding a GPU. Scientific non-improvement is none
of these: it is ``SUCCEEDED`` with an honest metric.

**Isolation is not a security boundary.** A git worktree is a directory, not a
sandbox: code running inside one can read anything the operator can read,
including held-out labels and evaluator internals. Appendix A §6 says so
explicitly. See :data:`ACCESS_BOUNDARY_NOTE` for what this module does and does
not provide, and what a later benchmark worker must add before calling a run
sealed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..candidates.workspace import CandidateWorkspace, WorkspaceError, workspace_for
from ..repository import RepositoryLayout, project_path_candidates
from ..config import EvolveConfig
from ..schemas import Candidate
from .contracts import (
    CONTRACT_VERSION,
    ContractError,
    ErrorCategory,
    ExperimentPlan,
    ExperimentResult,
    ExperimentStatus,
    FailureCategory,
    require_experiment_provenance,
    validate_record,
    verify_artifact_references,
)

Record = dict[str, Any]

#: What isolation here does and does not mean. Quoted into handoffs on purpose.
ACCESS_BOUNDARY_NOTE = (
    "A candidate checkout is a filesystem isolation boundary, not an access-control "
    "boundary: the child process runs as the same OS user and can read every path that "
    "user can read. It guarantees that the code and weights under evaluation are the "
    "candidate's and that the parent checkout is not modified. It does NOT guarantee that "
    "held-out labels or evaluator internals were unreachable. A run may be called sealed "
    "only when the benchmark worker additionally runs under a separate OS user, container "
    "or host whose filesystem permissions exclude the protected paths, and records which "
    "of those was used."
)

#: Environment variables a candidate process is allowed to inherit. Everything
#: else is dropped rather than filtered, so a new secret-shaped variable in the
#: operator's shell cannot reach a candidate by being unanticipated.
ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TZ",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_NUMERIC",
        # Toolchain / runtime the candidate genuinely needs.
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        # GPU placement. Without these a candidate silently lands on device 0.
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "NVIDIA_VISIBLE_DEVICES",
        # Determinism / thread caps, so a comparison is not decided by scheduling.
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)

#: A name matching this never reaches a candidate, allowlisted or not, and its
#: value is never written to a log, a record or a handoff.
CREDENTIAL_NAME = re.compile(
    r"(?i)(?:^|_)(?:key|keys|token|secret|secrets|password|passwd|pwd|credential|credentials|"
    r"auth|apikey|access|session|cookie|signature)(?:$|_)|(?:^|_)api[-_]?key"
)

#: Deliberately *not* inherited: the parent's import path and site configuration.
#: ``PYTHONPATH`` in particular is how the operator's checkout reaches a child.
ENV_NEVER_INHERIT: frozenset[str] = frozenset(
    {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONNOUSERSITE"}
)

_SENTINEL = "<<<DRUGAGENT_ORIGIN>>>"

_ORIGIN_PROBE = r"""
import importlib, json, os, sys

root = os.path.realpath(os.environ.get("DRUGAGENT_CANDIDATE_ROOT", ""))


def _inside(path):
    if not path or not root:
        return False
    return path == root or path.startswith(root + os.sep)


out = {
    "executable": sys.executable,
    "cwd": os.path.realpath(os.getcwd()),
    "sys_path": sys.path[:8],
    "candidate_root": root,
    "cwd_inside_candidate": _inside(os.path.realpath(os.getcwd())),
    "modules": {},
}
for name in json.loads(sys.argv[1]):
    entry = {"origin": None, "package_root": None, "inside_candidate": False, "error": None}
    try:
        mod = importlib.import_module(name)
        paths = list(getattr(mod, "__path__", []) or [])
        origin = getattr(mod, "__file__", None) or (paths[0] if paths else None)
        entry["origin"] = os.path.realpath(origin) if origin else None
        entry["package_root"] = os.path.realpath(paths[0]) if paths else None
        entry["inside_candidate"] = _inside(entry["origin"])
    except BaseException as exc:  # a broken candidate must be reported, not raised
        entry["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["modules"][name] = entry
sys.stdout.write("%s" % ("<<<DRUGAGENT_ORIGIN>>>" + json.dumps(out)))
"""


class CandidateMaterializationError(RuntimeError):
    """The candidate checkout could not be created — an infrastructure fault."""


class CandidatePatchError(ValueError):
    """The candidate's patch does not apply — a deterministic candidate fault."""

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail: dict[str, Any] = dict(detail or {})


class CandidateOriginError(RuntimeError):
    """Code was loaded from outside the candidate checkout.

    Raised rather than warned: an evaluation that imported the operator's tree
    is not a weaker measurement of the candidate, it is a measurement of
    something else.
    """


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterializedCandidate:
    """One candidate, on disk, with its identity measured rather than assumed."""

    experiment_id: str
    candidate_id: str
    root: Path
    source_root: Path
    interpreter: str
    mode: str
    parent_root: Path
    parent_commit: str | None
    candidate_revision: str
    patch_applied: bool
    patch_result: dict[str, Any] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    workspace: CandidateWorkspace | None = None

    def describe(self) -> Record:
        """JSON-ready identity of what is about to be measured."""
        return {
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "root": str(self.root),
            "source_root": str(self.source_root),
            "interpreter": self.interpreter,
            "mode": self.mode,
            "parent_root": str(self.parent_root),
            "parent_commit": self.parent_commit,
            "candidate_revision": self.candidate_revision,
            "patch_applied": self.patch_applied,
            "changed_files": list(self.changed_files),
            "notes": list(self.notes),
        }

    def cleanup(self, *, keep: bool = False, force: bool = True) -> Record:
        """Remove the checkout this worker created.

        ``force`` defaults to true here and nowhere else: the workspace was
        created by :func:`materialize` seconds ago, at a detached HEAD, under the
        workspace directory, and every byte of what makes it different from its
        parent is already recorded as ``candidate.patch`` and
        ``candidate_revision``. A patch candidate's worktree *always* carries
        modifications, so without this every evaluation would abandon a full
        checkout and a ``.git/worktrees`` registration.
        """
        if self.workspace is None:
            return {"removed": False, "reason": "no workspace to remove"}
        return self.workspace.cleanup(keep=keep, force=force)


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(cwd), check=False, timeout=120
    )


def _tree_revision(root: Path, mode: str) -> str:
    """A content address for the materialised tree.

    A worktree gets git's own tree object, which is exactly the identity a later
    reproduction needs. A copy has no git, so the digest is taken over the file
    contents directly — a different algorithm, and labelled as one.
    """
    if mode == "worktree":
        add = _git("add", "-A", cwd=root)
        write = _git("write-tree", cwd=root)
        if add.returncode == 0 and write.returncode == 0 and write.stdout.strip():
            return f"tree:{write.stdout.strip()}"
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        try:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(b"UNREADABLE")
    return f"sha256:{digest.hexdigest()}"


def _default_source_root(root: Path) -> Path:
    """``<root>/src`` when it is a source root, else the checkout root itself."""
    src = root / "src"
    return src if src.is_dir() else root


def materialize(
    candidate: Candidate,
    *,
    experiment_id: str,
    cfg: EvolveConfig | None = None,
    parent_root: Path | None = None,
    workspace_dir: Path | None = None,
    source_root: Path | None = None,
    interpreter: str | None = None,
) -> MaterializedCandidate:
    """Create the candidate's checkout, apply its patch, and measure its identity.

    Raises:
        CandidatePatchError: the patch does not apply. Deterministic, the
            candidate's own fault, and never worth a retry.
        CandidateMaterializationError: the checkout could not be created or the
            parent is not usable. Infrastructure, and retryable.
    """
    cfg = cfg or EvolveConfig()
    parent = Path(parent_root) if parent_root is not None else cfg.repo_root()
    if not parent.is_dir():
        raise CandidateMaterializationError(f"parent checkout {parent} does not exist")

    try:
        workspace = workspace_for(candidate, cfg, root=parent, workspace_dir=workspace_dir)
    except (OSError, WorkspaceError) as exc:
        raise CandidateMaterializationError(f"{type(exc).__name__}: {exc}") from exc
    if not workspace.path.is_dir():
        raise CandidateMaterializationError(
            f"workspace {workspace.path} was not created ({'; '.join(workspace.notes)})"
        )

    patch_result = workspace.apply_patch(candidate.patch)
    # The patch file is an artefact of applying it, not part of the candidate.
    stray = workspace.path / ".selfevolve-candidate.patch"
    if stray.is_file():
        with contextlib.suppress(OSError):  # best effort: it is not part of the candidate
            stray.unlink()
    if not patch_result.get("applied"):
        workspace.cleanup(force=True)
        raise CandidatePatchError(
            f"candidate {candidate.candidate_id} carries a patch that does not apply to "
            f"{workspace.parent_commit or parent}",
            detail=patch_result,
        )

    root = workspace.path.resolve()
    try:
        changed_files = workspace.changed_files()
        protected = [p for p in changed_files if is_protected_path(p, root=root)]
        if protected:
            raise ValueError(f"candidate patch changes protected paths: {protected}")
    except (ValueError, RuntimeError) as exc:
        workspace.cleanup(force=True)
        raise CandidatePatchError(str(exc), detail={"scope_or_protection_error": str(exc)}) from exc
    if source_root is None:
        candidate_source = _default_source_root(root)
    else:
        supplied = Path(source_root)
        if supplied.is_absolute():
            try:
                supplied = supplied.resolve().relative_to(parent.resolve())
            except ValueError:
                if supplied.resolve().is_relative_to(root):
                    supplied = supplied.resolve().relative_to(root)
                else:
                    workspace.cleanup(force=True)
                    raise CandidateMaterializationError("source_root must belong to the parent or candidate project")
        candidate_source = (root / supplied).resolve()
        if not candidate_source.is_relative_to(root):
            workspace.cleanup(force=True)
            raise CandidateMaterializationError("source_root escapes the candidate project")
    return MaterializedCandidate(
        experiment_id=experiment_id,
        candidate_id=candidate.candidate_id,
        root=root,
        source_root=candidate_source,
        interpreter=interpreter or sys.executable,
        mode=workspace.mode,
        parent_root=parent.resolve(),
        parent_commit=workspace.parent_commit,
        candidate_revision=_tree_revision(root, workspace.mode),
        patch_applied=True,
        patch_result=dict(patch_result),
        changed_files=changed_files,
        notes=list(workspace.notes),
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
def candidate_env(
    mc: MaterializedCandidate,
    *,
    extra: Mapping[str, str] | None = None,
    allow_network: bool = False,
) -> dict[str, str]:
    """The child's environment: an allowlist, never the operator's whole shell.

    ``PYTHONPATH`` is set to the candidate's source root and to nothing else, so
    the parent's editable install cannot win the import. Anything whose *name*
    looks like a credential is refused even if explicitly passed in.
    """
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in ENV_NEVER_INHERIT or name not in ENV_ALLOWLIST:
            continue
        if CREDENTIAL_NAME.search(name):
            continue
        env[name] = value
    for name, value in (extra or {}).items():
        if CREDENTIAL_NAME.search(name):
            raise ValueError(
                f"refusing to pass {name!r} into a candidate process: the name is "
                "credential-shaped, and a candidate never needs the operator's secrets"
            )
        env[str(name)] = str(value)
    if os.pathsep in str(mc.source_root):
        raise ValueError("candidate source path contains the PYTHONPATH separator; use a checkout path without ':'")
    env["PYTHONPATH"] = str(mc.source_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["DRUGAGENT_CANDIDATE_ROOT"] = str(mc.root)
    env["DRUGAGENT_CANDIDATE_REVISION"] = mc.candidate_revision
    env.setdefault("DRUGAGENT_ALLOW_NETWORK", "1" if allow_network else "0")
    return env


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    """An environment safe to write into a record: names kept, values summarised."""
    out: dict[str, str] = {}
    for name, value in sorted(env.items()):
        if CREDENTIAL_NAME.search(name):
            out[name] = "<redacted>"
        elif name in ("PYTHONPATH", "DRUGAGENT_CANDIDATE_ROOT", "DRUGAGENT_CANDIDATE_REVISION"):
            out[name] = value
        else:
            out[name] = f"<{len(value)} chars>"
    return out


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _kill_group(proc: subprocess.Popen[Any], *, grace_s: float = 5.0) -> dict[str, Any]:
    """Terminate the child's whole process group. Never leaves a GPU child behind.

    SIGTERM to the group first, so a job with a checkpoint handler gets to write
    one; SIGKILL to the group after the grace period, because a job that ignores
    SIGTERM while holding a GPU is exactly what must not survive.

    The direct child is reaped between probes. Without that the group still
    answers ``kill(pgid, 0)`` through the child's own zombie, and every
    cancellation would report an orphan that does not exist.
    """
    info: dict[str, Any] = {"process_group_killed": False, "orphans_remaining": None, "signals": []}
    if not hasattr(os, "killpg"):  # pragma: no cover - POSIX hosts only here
        proc.kill()
        return info
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:  # pragma: no cover - raced with its own exit
        proc.poll()
        info["process_group_killed"] = True
        info["orphans_remaining"] = False
        return info

    def _group_empty() -> bool:
        proc.poll()  # reap the direct child so its zombie does not count as an orphan
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        return False

    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
            info["signals"].append(sig.name)
            info["process_group_killed"] = True
        except ProcessLookupError:
            break
        deadline = time.time() + wait
        while time.time() < deadline:
            if _group_empty():
                info["orphans_remaining"] = False
                return info
            time.sleep(0.05)
    info["orphans_remaining"] = not _group_empty()
    return info


def run_in_candidate(
    mc: MaterializedCandidate,
    label: str,
    argv: Sequence[str],
    *,
    timeout_s: float = 1800.0,
    env_extra: Mapping[str, str] | None = None,
    allow_network: bool = False,
    log_dir: Path | None = None,
) -> Record:
    """Run one command in the candidate checkout and keep everything it produced.

    The command runs in its own session, so a timeout can terminate the whole
    process group rather than the direct child alone — a generator that forked a
    CUDA worker leaves nothing holding the device.

    Output goes to files, not pipes: a killed process still leaves its log, and
    a chatty job cannot deadlock on a full pipe buffer.
    """
    argv = [str(a) for a in argv]
    substituted = False
    if argv and argv[0] == sys.executable and mc.interpreter != sys.executable:
        argv[0] = mc.interpreter
        substituted = True
    env = candidate_env(mc, extra=env_extra, allow_network=allow_network)

    owned_dir: Path | None = None
    if log_dir is None:
        owned_dir = Path(tempfile.mkdtemp(prefix="candidate-logs-"))
        target = owned_dir
    else:
        target = Path(log_dir)
        target.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{mc.experiment_id}-{label}").strip("-")
    out_path = target / f"{slug}.stdout.log"
    err_path = target / f"{slug}.stderr.log"

    started = time.time()
    result: Record = {
        "label": label,
        "argv": argv,
        "cwd": str(mc.root),
        "interpreter": mc.interpreter,
        "interpreter_substituted": substituted,
        "candidate_revision": mc.candidate_revision,
        "timed_out": False,
        "cancelled": False,
        "process_group_killed": False,
        "orphans_remaining": False,
        "launch_error": None,
        "stdout_log": str(out_path),
        "stderr_log": str(err_path),
        "env": redact_env(env),
    }
    with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
        proc: subprocess.Popen[bytes] | None
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(mc.root),
                env=env,
                stdout=out_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            result["exit_status"] = 127
            result["launch_error"] = f"{type(exc).__name__}: {exc}"
            err_f.write(f"{type(exc).__name__}: {exc}\n".encode())
            proc = None
        if proc is not None:
            try:
                result["exit_status"] = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                result["exit_status"] = 124
                result.update(_kill_group(proc))
                # Already SIGKILLed; this only reaps it.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
            except BaseException:
                # Ctrl-C, SIGTERM to the operator, a cancelled job. The child is
                # in its own session, so it would otherwise survive this process
                # and keep whatever it holds — a GPU, most expensively. Kill the
                # group, then let the interrupt continue: cancellation is the
                # caller's to handle, orphaning is not.
                result["cancelled"] = True
                result["exit_status"] = 130
                result.update(_kill_group(proc))
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                raise
    out = out_path.read_text(encoding="utf-8", errors="replace")
    err = err_path.read_text(encoding="utf-8", errors="replace")
    if owned_dir is not None:
        shutil.rmtree(owned_dir, ignore_errors=True)
        result["stdout_log"] = None
        result["stderr_log"] = None

    result["ok"] = result["exit_status"] == 0
    result["started_at"] = started
    result["finished_at"] = time.time()
    result["elapsed_s"] = round(time.time() - started, 3)
    result["stdout_head"] = out[:2000]
    result["stdout_tail"] = out[-4000:] if len(out) > 2000 else ""
    result["stderr_tail"] = err[-4000:]
    return result


def probe_origin(
    mc: MaterializedCandidate,
    *,
    modules: Sequence[str] | None = None,
    timeout_s: float = 180.0,
    log_dir: Path | None = None,
) -> Record:
    """Ask a *fresh* subprocess where its code actually came from.

    This is the check the gate-0 defect needed and did not have. ``modules``
    defaults to ``drug_agent`` when the candidate carries it, and to nothing when
    the candidate is some other tree.
    """
    names = list(modules) if modules is not None else _default_modules(mc)
    result = run_in_candidate(
        mc,
        "origin-probe",
        (sys.executable, "-c", _ORIGIN_PROBE, json.dumps(names)),
        timeout_s=timeout_s,
        log_dir=log_dir,
    )
    payload: Record = {
        "modules": {},
        "executable": None,
        "cwd": None,
        "sys_path": [],
        "candidate_root": str(mc.root),
        "ok": False,
        "command": {k: result[k] for k in ("exit_status", "timed_out", "launch_error", "elapsed_s")},
        "stderr_tail": result["stderr_tail"],
    }
    raw = result["stdout_head"] + result["stdout_tail"]
    if _SENTINEL in raw:
        try:
            payload.update(json.loads(raw.split(_SENTINEL, 1)[1]))
            payload["ok"] = result["exit_status"] == 0
        except ValueError:
            payload["ok"] = False
    return payload


def _default_modules(mc: MaterializedCandidate) -> list[str]:
    return ["drug_agent"] if (mc.source_root / "drug_agent").is_dir() else []


def verify_origin(mc: MaterializedCandidate, probe: Mapping[str, Any]) -> str:
    """Return the observed import origin, or raise if it is not the candidate's.

    With no module to probe, the origin is the candidate's own source root — a
    path, still measured, still inside the candidate, and never the parent's.
    """
    if not probe.get("ok"):
        raise CandidateOriginError(
            f"the origin probe did not run: {probe.get('command')} {probe.get('stderr_tail', '')[-400:]}"
        )
    if not probe.get("cwd_inside_candidate"):
        raise CandidateOriginError(
            f"the child ran in {probe.get('cwd')!r}, which is outside the candidate checkout {mc.root}"
        )
    origins: list[str] = []
    for name, entry in sorted((probe.get("modules") or {}).items()):
        if entry.get("error"):
            raise CandidateOriginError(f"module {name!r} could not be imported: {entry['error']}")
        if not entry.get("inside_candidate"):
            raise CandidateOriginError(
                f"module {name!r} was imported from {entry.get('origin')!r}, which is outside "
                f"the candidate checkout {mc.root} — the parent's editable install, an overlay "
                "or a stale site-packages copy is masquerading as the candidate"
            )
        origins.append(str(entry["origin"]))
    return origins[0] if origins else str(mc.source_root)


def weight_digests(mc: MaterializedCandidate, paths: Iterable[str]) -> dict[str, str]:
    """``{relative path: sha256 or MISSING}`` for the candidate's own checkpoints.

    A weights-only candidate has source identical to its parent, so this is the
    only thing that distinguishes it. ``MISSING`` is recorded rather than skipped:
    a checkpoint the record claims and the checkout does not have is a finding.
    """
    out: dict[str, str] = {}
    for rel in paths:
        path = (mc.root / rel).resolve()
        try:
            path.relative_to(mc.root)
        except ValueError:
            out[str(rel)] = "OUTSIDE_CANDIDATE"
            continue
        try:
            out[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            out[str(rel)] = "MISSING"
    return out


#: :func:`weight_digests` under its other name — a config file and a checkpoint
#: are hashed the same way and the distinction is in what the caller asks for.
config_digests = weight_digests


# ---------------------------------------------------------------------------
# the whole job
# ---------------------------------------------------------------------------
def _checkpoint(log_dir: Path | None, payload: Mapping[str, Any]) -> Path | None:
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"checkpoint-{payload['experiment_id']}.json"
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


_RETRYABLE = (FailureCategory.INFRASTRUCTURE_ERROR,)


def execute_candidate(
    candidate: Candidate,
    *,
    experiment_id: str,
    commands: Sequence[tuple[str, Sequence[str]]]
    | Callable[[MaterializedCandidate], Sequence[tuple[str, Sequence[str]]]],
    cfg: EvolveConfig | None = None,
    parent_root: Path | None = None,
    workspace_dir: Path | None = None,
    source_root: Path | None = None,
    interpreter: str | None = None,
    modules: Sequence[str] | None = None,
    weight_paths: Sequence[str] = (),
    config_paths: Sequence[str] = (),
    artifacts: Sequence[str] = (),
    development_metrics: Mapping[str, float | None] | None = None,
    log_dir: Path | None = None,
    timeout_s: float = 1800.0,
    allow_network: bool = False,
    max_infrastructure_retries: int = 0,
    retry_delay_s: float = 0.0,
    keep_workspace: bool | None = None,
    env_extra: Mapping[str, str] | None = None,
    on_materialized: Callable[[MaterializedCandidate], Mapping[str, Any]] | None = None,
) -> Record:
    """Materialise, verify, run — and return a validated ``ExperimentResult``.

    One logical experiment. A transient infrastructure fault is retried under
    the *same* ``experiment_id`` and every attempt is charged to ``actual_cost``;
    a deterministic candidate fault is returned on the first attempt, because
    retrying it would only spend budget to reach the same answer.

    Cost is recorded on failure too. ``None`` in ``actual_cost`` means the
    telemetry is unavailable, never that the resource was free.

    ``commands`` may be a callable taking the materialised candidate, for a
    caller whose argument vector mentions the checkout that does not exist yet.
    ``on_materialized`` runs once the checkout exists and its import origin has
    been verified, *before* any command — that is where Step 10's experiment
    worker writes its code and where the candidate's identity is hashed. It
    returns a record; ``{"ok": False, "failure_category": ..., "reason": ...}``
    ends the attempt without running the commands.

    The return value is a validated ``ExperimentResult`` and nothing else. When
    ``log_dir`` is given, the full execution detail — candidate identity, origin
    probe, per-command capture, checkpoint and configuration digests — is written
    to ``<log_dir>/detail-<experiment_id>.json`` and the resumable checkpoint to
    ``<log_dir>/checkpoint-<experiment_id>.json``.
    """
    started = time.time()
    interventions: list[dict[str, Any]] = []
    attempts = 0
    last: Record | None = None
    budget = max(0, int(max_infrastructure_retries))

    while attempts <= budget:
        attempts += 1
        last = _attempt(
            candidate,
            experiment_id=experiment_id,
            attempt=attempts,
            commands=commands,
            cfg=cfg,
            parent_root=parent_root,
            workspace_dir=workspace_dir,
            source_root=source_root,
            interpreter=interpreter,
            modules=modules,
            weight_paths=weight_paths,
            config_paths=config_paths,
            log_dir=log_dir,
            timeout_s=timeout_s,
            allow_network=allow_network,
            keep_workspace=keep_workspace,
            started=started,
            is_final_attempt=attempts > budget,
            interventions=interventions,
            env_extra=env_extra,
            on_materialized=on_materialized,
        )
        if last["failure_category"] not in {c.value for c in _RETRYABLE} or attempts > budget:
            break
        interventions.append(
            {
                "event": "infrastructure_retry",
                "experiment_id": experiment_id,
                "attempt": attempts,
                "at": time.time(),
                "reason": last["detail"].get("reason", "infrastructure fault"),
                "note": "same logical experiment; the attempt is charged to actual_cost",
            }
        )
        if retry_delay_s:
            time.sleep(retry_delay_s)

    assert last is not None
    record = {
        "experiment_id": experiment_id,
        "status": last["status"],
        "exit_code": last["exit_code"],
        "candidate_revision": last["candidate_revision"],
        "import_origin": last["import_origin"],
        "artifacts": list(artifacts),
        "development_metrics": dict(development_metrics or {}),
        "actual_cost": {
            "attempts": float(attempts),
            "wall_s": round(time.time() - started, 3),
            "gpu_hours": None,
            "llm_output_tokens": None,
        },
        "failure_category": last["failure_category"],
        "intervention_log": interventions,
    }
    validated = validate_record(ExperimentResult, record)
    _checkpoint(
        log_dir,
        {
            **last["checkpoint"],
            "attempts": attempts,
            "actual_cost": record["actual_cost"],
            "status": record["status"],
            "failure_category": record["failure_category"],
        },
    )
    # The execution detail — materialisation, origin probe, per-command capture,
    # checkpoint digests — is evidence, but it is not a field of the contract,
    # and a record that carries unknown fields is not a valid ExperimentResult.
    # It goes beside the checkpoint, on disk, where it can be read later.
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / f"detail-{experiment_id}.json").write_text(
            json.dumps({"experiment_id": experiment_id, **last["detail"]}, indent=2, default=str),
            encoding="utf-8",
        )
    return validated


def _outcome(status: ExperimentStatus, category: FailureCategory) -> tuple[str, str, int]:
    from ...cli.main import run_exit_code

    return status.value, category.value, run_exit_code(status.value)


def _attempt(
    candidate: Candidate,
    *,
    experiment_id: str,
    attempt: int,
    commands: Sequence[tuple[str, Sequence[str]]]
    | Callable[[MaterializedCandidate], Sequence[tuple[str, Sequence[str]]]],
    cfg: EvolveConfig | None,
    parent_root: Path | None,
    workspace_dir: Path | None,
    source_root: Path | None,
    interpreter: str | None,
    modules: Sequence[str] | None,
    weight_paths: Sequence[str],
    config_paths: Sequence[str],
    log_dir: Path | None,
    timeout_s: float,
    allow_network: bool,
    keep_workspace: bool | None,
    started: float,
    is_final_attempt: bool,
    interventions: list[dict[str, Any]],
    env_extra: Mapping[str, str] | None = None,
    on_materialized: Callable[[MaterializedCandidate], Mapping[str, Any]] | None = None,
) -> Record:
    """One attempt. Returns the pieces :func:`execute_candidate` assembles."""
    base_checkpoint: dict[str, Any] = {
        "experiment_id": experiment_id,
        "candidate_id": candidate.candidate_id,
        "attempt": attempt,
        "started_at": started,
        "updated_at": time.time(),
        "phase": "materialize",
        "resumable": True,
        "candidate_revision": "",
        "workspace": None,
        "completed_commands": [],
        "note": "materialising the candidate checkout",
    }
    _checkpoint(log_dir, base_checkpoint)

    # --- phase 1: materialise ------------------------------------------
    try:
        mc = materialize(
            candidate,
            experiment_id=experiment_id,
            cfg=cfg,
            parent_root=parent_root,
            workspace_dir=workspace_dir,
            source_root=source_root,
            interpreter=interpreter,
        )
    except CandidatePatchError as exc:
        status, category, code = _outcome(ExperimentStatus.FAILED, FailureCategory.INVALID_CANDIDATE)
        return {
            "status": status,
            "failure_category": category,
            "exit_code": code,
            "candidate_revision": f"unmaterialized:{candidate.candidate_id}",
            "import_origin": "UNAVAILABLE: the candidate was never materialised",
            "detail": {"reason": str(exc), "patch": exc.detail},
            "checkpoint": {
                **base_checkpoint,
                "phase": "materialize",
                "resumable": False,
                "updated_at": time.time(),
                "note": "the patch does not apply; nothing to resume",
            },
        }
    except CandidateMaterializationError as exc:
        status, category, code = _outcome(ExperimentStatus.FAILED, FailureCategory.INFRASTRUCTURE_ERROR)
        return {
            "status": status,
            "failure_category": category,
            "exit_code": code,
            "candidate_revision": f"unmaterialized:{candidate.candidate_id}",
            "import_origin": "UNAVAILABLE: the candidate checkout could not be created",
            "detail": {"reason": str(exc)},
            "checkpoint": {
                **base_checkpoint,
                "phase": "materialize",
                "updated_at": time.time(),
                "note": "checkout creation failed; retryable",
            },
        }

    checkpoint = {
        **base_checkpoint,
        "phase": "verify_origin",
        "candidate_revision": mc.candidate_revision,
        "workspace": str(mc.root),
        "updated_at": time.time(),
        "note": "verifying that the child imports the candidate's code",
    }
    _checkpoint(log_dir, checkpoint)

    detail: dict[str, Any] = {
        "candidate": mc.describe(),
        "weights": weight_digests(mc, weight_paths),
        "configs": weight_digests(mc, config_paths),
        "commands": [],
        "access_boundary": ACCESS_BOUNDARY_NOTE,
    }

    def _finish(status_: ExperimentStatus, category_: FailureCategory, origin: str) -> Record:
        keep = keep_workspace
        if keep is None:
            # Kept only when it is worth resuming from: an interrupted or
            # infrastructure-faulted *final* attempt. An attempt that is about to
            # be retried leaves nothing behind, or a retried job would accumulate
            # one abandoned checkout per attempt.
            keep = is_final_attempt and category_ in (
                FailureCategory.TIMEOUT,
                FailureCategory.INFRASTRUCTURE_ERROR,
            )
        detail["cleanup"] = mc.cleanup(keep=keep)
        status, category, code = _outcome(status_, category_)
        return {
            "status": status,
            "failure_category": category,
            "exit_code": code,
            "candidate_revision": mc.candidate_revision,
            "import_origin": origin,
            "detail": detail,
            "checkpoint": {
                **checkpoint,
                "updated_at": time.time(),
                "resumable": bool(keep),
                "workspace": str(mc.root) if keep else None,
                "completed_commands": [c["label"] for c in detail["commands"] if c.get("ok")],
            },
        }

    # --- phase 2: where did the code come from? -------------------------
    probe = probe_origin(mc, modules=modules, log_dir=log_dir)
    detail["origin_probe"] = probe
    try:
        origin = verify_origin(mc, probe)
    except CandidateOriginError as exc:
        detail["reason"] = str(exc)
        launch = (probe.get("command") or {}).get("launch_error")
        category = (
            FailureCategory.INFRASTRUCTURE_ERROR
            if launch or not probe.get("ok")
            else FailureCategory.CANDIDATE_ERROR
        )
        checkpoint["phase"] = "verify_origin"
        return _finish(ExperimentStatus.FAILED, category, f"UNVERIFIED: {exc}")
    detail["import_origin"] = origin

    # --- phase 2b: whoever writes the experiment's code, if anyone -------
    if on_materialized is not None:
        checkpoint["phase"] = "worker"
        checkpoint["updated_at"] = time.time()
        _checkpoint(log_dir, checkpoint)
        worker = dict(on_materialized(mc))
        detail["worker"] = worker
        if not worker.get("ok", True):
            detail["reason"] = str(worker.get("reason") or "the experiment worker did not finish")
            category = FailureCategory(
                str(worker.get("failure_category") or FailureCategory.CANDIDATE_ERROR.value)
            )
            return _finish(ExperimentStatus.FAILED, category, origin)

    # --- phase 3: the candidate's own commands --------------------------
    checkpoint["phase"] = "commands"
    checkpoint["updated_at"] = time.time()
    _checkpoint(log_dir, checkpoint)
    resolved = list(commands(mc)) if callable(commands) else list(commands)
    detail["argv"] = [list(argv) for _label, argv in resolved]
    for label, argv in resolved:
        outcome = run_in_candidate(
            mc,
            label,
            argv,
            timeout_s=timeout_s,
            allow_network=allow_network,
            log_dir=log_dir,
            env_extra=env_extra,
        )
        detail["commands"].append(outcome)
        checkpoint["completed_commands"] = [c["label"] for c in detail["commands"] if c.get("ok")]
        checkpoint["updated_at"] = time.time()
        _checkpoint(log_dir, checkpoint)
        if outcome["timed_out"]:
            detail["reason"] = f"{label} exceeded its {timeout_s}s budget"
            return _finish(ExperimentStatus.FAILED, FailureCategory.TIMEOUT, origin)
        if outcome["launch_error"]:
            detail["reason"] = f"{label} could not be launched: {outcome['launch_error']}"
            return _finish(ExperimentStatus.FAILED, FailureCategory.INFRASTRUCTURE_ERROR, origin)
        if not outcome["ok"]:
            detail["reason"] = (
                f"{label} exited {outcome['exit_status']}: "
                + (outcome["stderr_tail"] or outcome["stdout_tail"] or outcome["stdout_head"])[-400:]
            )
            return _finish(ExperimentStatus.FAILED, FailureCategory.CANDIDATE_ERROR, origin)

    checkpoint["phase"] = "complete"
    return _finish(ExperimentStatus.SUCCEEDED, FailureCategory.NONE, origin)


# ===========================================================================
# Step 10: structured experiment plans, protected paths and the dispatcher
# ===========================================================================
#
# The defect this half of the module exists to close is the one the program's
# acceptance criterion names: a reasoning loop that can *propose* experiments
# but cannot *cause* them, or — worse — one that can cause anything at all,
# including an edit to the thing that grades it.
#
# Three rules make the difference, and each is enforced rather than documented:
#
# 1. **An advertised action executes or reports a precise capability failure.**
#    A callable that raises on entry is not support. :data:`ACTIONS` registers
#    nine interventions; :func:`action_support` answers for each one from Step
#    8's per-verb probes — live when asked to probe, from the committed probe
#    record otherwise — and an unavailable backend produces a structured
#    ``CAPABILITY_UNAVAILABLE`` result naming the missing verb and the unblock,
#    not a refusal at the door. Any category may be proposed.
# 2. **Commands are argument vectors.** A model-authored string never reaches a
#    shell: ``run_in_candidate`` takes argv, ``shell=True`` appears nowhere, and
#    :func:`validate_plan` refuses a string where a vector is required with
#    ``ENTRYPOINT_NOT_ARGV`` — the same category Step 3 froze for artifacts.
# 3. **The referee is unreachable.** :func:`protected_paths` is *derived* from
#    the evaluation kernel's own integrity manifest, so a file added to the
#    manifest becomes protected without anyone remembering to retype it, and a
#    plan naming one — as an input, as an output, or through ``..`` — is refused
#    with ``PROTECTED_PATH`` before anything launches.
#
# And one rule about evidence: **the worker's prose is not a measurement.** A
# run is accepted only when its process exited zero, every declared output
# exists and deserializes, every declared digest matches the bytes on disk, and
# every artifact reference resolves in the store. The development number is then
# computed by the *parent's* evaluator over the frozen development pool, in this
# process, from the predictions the run wrote — never read out of the worker's
# own result file.


@dataclass(frozen=True)
class ActionSpec:
    """One registered intervention: what it is, what it needs, what runs it."""

    action: str
    summary: str
    #: Artifact kinds this intervention can legitimately produce.
    artifact_kinds: tuple[str, ...]
    #: ``backend name -> the (capability_id, verb) pairs that backend needs``.
    #: An empty tuple means the intervention needs no external scientific
    #: adapter — not that its availability is unknown.
    backends: Mapping[str, tuple[tuple[str, str], ...]]
    default_backend: str
    #: Where the work actually happens, as ``module:callable``. Never blank:
    #: an action with no handler would be an advertisement.
    handler: str
    #: True when the intervention's body is code the experiment worker writes,
    #: rather than a fixed adapter this repository already ships.
    worker_authored: bool = False


_ACTIONS_ORDERED: tuple[ActionSpec, ...] = (
    ActionSpec(
        action="acquire_resource",
        summary=(
            "Stage an accessible dataset, model or structure through the ingestion "
            "boundary, preserving source, licence, trust label and measurement relation."
        ),
        artifact_kinds=("DATA_OR_LIBRARY", "WEIGHTS", "RECEPTOR_STRUCTURE", "COMPOSITE_PACKAGE"),
        backends={"staged_local": ()},
        default_backend="staged_local",
        handler="drug_agent.selfevolve.target.actions:acquire_resource",
    ),
    ActionSpec(
        action="fit_predictor",
        summary=(
            "Fit a Track A activity predictor on the admitted train partition and predict "
            "the development pool."
        ),
        artifact_kinds=("TRAINED_MODEL", "SCORER", "WEIGHTS", "COMPOSITE_PACKAGE"),
        backends={
            "sklearn_rf": (),
            "chemprop": (("property.predict.chemprop", "fit"), ("property.predict.chemprop", "predict")),
        },
        default_backend="sklearn_rf",
        handler="drug_agent.selfevolve.target.actions:fit_predictor",
    ),
    ActionSpec(
        action="adapt_generator",
        summary="Target-adapt a pretrained molecular generator's weights.",
        artifact_kinds=("WEIGHTS", "TRAINED_MODEL", "COMPOSITE_PACKAGE"),
        backends={
            "pfm": (("molecule.generate.pfm", "fit"),),
            "reinvent4": (("molecule.generate.reinvent4", "fit"),),
        },
        default_backend="reinvent4",
        handler="drug_agent.selfevolve.target.actions:adapt_generator",
    ),
    ActionSpec(
        action="distill",
        summary="Train a cheap student on a named computational teacher endpoint and report its agreement.",
        artifact_kinds=("SCORER", "TRAINED_MODEL", "WEIGHTS", "COMPOSITE_PACKAGE"),
        backends={"sklearn_gbr": (), "sklearn_rf": (), "sklearn_ridge": ()},
        default_backend="sklearn_gbr",
        handler="drug_agent.selfevolve.target.actions:distill",
    ),
    ActionSpec(
        action="change_sampling",
        summary="Vary the generator's sampling policy and record the full generation funnel.",
        artifact_kinds=("SAMPLING_POLICY", "ALGORITHM", "COMPOSITE_PACKAGE"),
        backends={
            "pfm": (("molecule.generate.pfm", "sample"),),
            "reinvent4": (("molecule.generate.reinvent4", "sample"),),
        },
        default_backend="reinvent4",
        handler="drug_agent.selfevolve.target.actions:change_sampling",
    ),
    ActionSpec(
        action="develop_algorithm",
        summary="Write and run new scientific code in the candidate checkout.",
        artifact_kinds=(
            "ALGORITHM",
            "SAMPLING_POLICY",
            "SCORER",
            "TRAINED_MODEL",
            "WEIGHTS",
            "COMPOSITE_PACKAGE",
        ),
        backends={"claude_code": (), "none": ()},
        default_backend="claude_code",
        handler="drug_agent.selfevolve.target.worker:run_declared_command",
        worker_authored=True,
    ),
    ActionSpec(
        action="develop_architecture",
        summary="Write and run a new model architecture in the candidate checkout.",
        artifact_kinds=(
            "MODEL_ARCHITECTURE",
            "ALGORITHM",
            "TRAINED_MODEL",
            "WEIGHTS",
            "COMPOSITE_PACKAGE",
        ),
        backends={"claude_code": (), "none": ()},
        default_backend="claude_code",
        handler="drug_agent.selfevolve.target.worker:run_declared_command",
        worker_authored=True,
    ),
    ActionSpec(
        action="prepare_receptor",
        summary="Cut and check a receptor pocket, refusing one that is not this target.",
        artifact_kinds=("RECEPTOR_STRUCTURE", "RECEPTOR_ENSEMBLE", "COMPOSITE_PACKAGE"),
        backends={"pocket_cut": ()},
        default_backend="pocket_cut",
        handler="drug_agent.selfevolve.target.actions:prepare_receptor",
    ),
    ActionSpec(
        action="evaluate_development",
        summary="Score an artifact's predictions on the development partition with the frozen evaluator.",
        artifact_kinds=("ALGORITHM", "SCORER", "TRAINED_MODEL", "COMPOSITE_PACKAGE"),
        backends={"target_benchmark": ()},
        default_backend="target_benchmark",
        handler="drug_agent.selfevolve.target.actions:evaluate_development",
    ),
)

#: The registered interventions, by name.
ACTIONS: dict[str, ActionSpec] = {spec.action: spec for spec in _ACTIONS_ORDERED}

#: Their names. An action outside this tuple is not an intervention this
#: repository can cause, whatever a plan or a campaign menu says.
SUPPORTED_ACTIONS: tuple[str, ...] = tuple(spec.action for spec in _ACTIONS_ORDERED)

#: Plan keys whose value must be an argument vector rather than a command
#: string. A string here is the single most dangerous thing a model-authored
#: plan could carry, so it is refused by name.
ARGV_KEYS: frozenset[str] = frozenset(
    {"command", "argv", "cmd", "entrypoint", "args", "verification_command"}
)

#: Directories and files that are protected in addition to the evaluation
#: kernel's own integrity manifest: the frozen benchmark records, the committed
#: baseline manifests and results, the campaign configuration, and the 0600
#: protected split registry. Appendix A §6: "Changes to the evaluator, benchmark
#: labels, lock, master objective, or baseline artifacts are rejected by the
#: controller."
EXTRA_PROTECTED_PATHS: tuple[str, ...] = (
    "src/drug_agent/selfevolve/evaluation/",
    "src/drug_agent/selfevolve/target/baselines.py",
    "src/drug_agent/selfevolve/target/baseline_arms.py",
    "src/drug_agent/selfevolve/target/budget.py",
    "configs/research/zak-c9r.yaml",
    "configs/research/zak-c9r/",
    ".external/research/zak-c9r/protected/",
    # The frozen study protocol lock. `configs/research/zak-c9r/` already covers
    # the committed protocol source; this is the runtime lock the confirmation
    # study is actually read from, and Appendix A §6 names the lock explicitly
    # among the records a candidate may not change.
    ".external/research/zak-c9r/protocol.lock.json",
    "docs/research/handoffs/",
)

#: Substituted into a plan's declared argument vector. Anything else is left
#: alone: substitution is a fixed, closed vocabulary, never a template engine.
ARGV_PLACEHOLDERS: tuple[str, ...] = ("{python}", "{output}", "{repo}", "{candidate}", "{plan}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def protected_paths(root: Path | str | None = None) -> list[str]:
    """Every repository-relative path a plan may not reach, in a stable order.

    Derived, not retyped: the evaluation kernel's own integrity manifest is the
    source of truth for which files decide whether a candidate succeeded, so a
    file added to that manifest becomes protected here without anyone
    remembering to update a second list. :data:`EXTRA_PROTECTED_PATHS` adds the
    records the manifest cannot cover because they are data rather than code —
    the frozen qualification and calibration, the baseline manifests and the
    0600 held-out registry.
    """
    del root  # the manifest is package-relative; the rule set is not per-checkout
    from ..evaluation.integrity import kernel_manifest

    # kernel_manifest() keys are relative to ``src/``; plans speak in
    # repository-relative paths, so the prefix is added once, here.
    rules = {f"src/{key}" for key in kernel_manifest()}
    rules.update(EXTRA_PROTECTED_PATHS)
    return sorted(rules)


def _normalise(raw: str) -> str:
    text = str(raw).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return os.path.normpath(text) if text else text


def is_protected_path(path: str, *, root: Path | str | None = None) -> str | None:
    """The protection rule *path* falls under, or ``None``.

    ``..`` is resolved before the comparison rather than matched as text, so
    ``work/../../src/drug_agent/selfevolve/target/api.py`` is recognised as the
    frozen acceptance surface it actually names. An absolute path is compared
    against the repository root as well as lexically, because a plan that writes
    to ``/home/.../repo/src/...`` is naming the same file by a different route.
    """
    text = str(path).strip()
    if not text:
        return None
    rules = protected_paths()
    repo = Path(root) if root is not None else _repo_root()

    candidates = project_path_candidates(text, repo)

    for value in candidates:
        if not value or value in (".", ".."):
            continue
        for rule in rules:
            if rule.endswith("/"):
                if value == rule.rstrip("/") or value.startswith(rule):
                    return rule
            elif value == rule:
                return rule
    return None


def _iter_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Every string leaf in a nested mapping/sequence, with its dotted path."""
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_strings(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_strings(item, f"{prefix}[{index}]")


def _plan_fail(
    category: ErrorCategory,
    message: str,
    *,
    field: str,
    missing: Sequence[str] = (),
) -> ContractError:
    return ContractError(category, message, contract="ExperimentPlan", field=field, missing=missing)


def validate_plan(
    plan: Mapping[str, Any],
    allowed_actions: Sequence[str],
    *,
    root: Path | str | None = None,
) -> Record:
    """Admit one structured experiment plan, or refuse it for a stated reason.

    The order of the checks is part of the contract, because each answer is a
    different fact about the plan:

    1. **schema** — every field Appendix A §7.1 requires, through the frozen
       :class:`~drug_agent.selfevolve.target.contracts.ExperimentPlan`;
    2. **the campaign menu** — a menu naming an intervention this repository
       cannot run is a configuration error (``INVALID_ENUM`` on
       ``allowed_actions``), not a licence to attempt it;
    3. **registration** — an action outside :data:`SUPPORTED_ACTIONS` is
       ``INVALID_ENUM``. This is where ``rewrite_final_test`` stops: editing the
       referee is not one of the interventions that exist;
    4. **authorisation** — a registered action the campaign did not authorise is
       ``ACTION_NOT_PERMITTED``, a deliberately different answer from (3);
    5. **capability shape** — the declared backend must be one this action has,
       and the produced ``kind`` must be one this action can produce;
    6. **argument vectors** — a command expressed as a string is
       ``ENTRYPOINT_NOT_ARGV``;
    7. **protected paths** — no input and no expected output may reach the
       evaluator, the frozen records, the baselines or the held-out registry,
       by any spelling.

    Nothing here consults a backend or spends anything: availability is
    :func:`action_support`'s answer and is deliberately *not* an admission
    criterion, because Appendix A's acceptance rule is that any tool category
    may be proposed and an unavailable one produces an actionable result.

    Returns:
        A JSON-ready record carrying the validated plan and what admission
        established about it.

    Raises:
        ValueError: a
            :class:`~drug_agent.selfevolve.target.contracts.ContractError`
            carrying the stable category above and the offending field.
    """
    validated = validate_record(ExperimentPlan, plan)

    if isinstance(allowed_actions, (str, bytes)) or not isinstance(allowed_actions, Sequence):
        raise _plan_fail(
            ErrorCategory.INVALID_TYPE,
            f"allowed_actions is a list of action names, got {type(allowed_actions).__name__}",
            field="allowed_actions",
        )
    menu = [str(a) for a in allowed_actions]
    unimplemented = sorted({a for a in menu if a not in ACTIONS})
    if unimplemented:
        raise _plan_fail(
            ErrorCategory.INVALID_ENUM,
            f"the campaign menu authorises {unimplemented}, which this dispatcher does not "
            f"implement; the registered interventions are {list(SUPPORTED_ACTIONS)}. A menu "
            f"entry with no handler behind it is an advertisement, not a capability",
            field="allowed_actions",
            missing=unimplemented,
        )

    action = str(validated["action"])
    if action not in ACTIONS:
        raise _plan_fail(
            ErrorCategory.INVALID_ENUM,
            f"{action!r} is not one of the registered interventions "
            f"{list(SUPPORTED_ACTIONS)}. Editing the evaluator, the benchmark records or the "
            f"held-out labels is not an intervention this repository can perform, by "
            f"construction rather than by policy",
            field="action",
        )
    if not menu:
        raise _plan_fail(
            ErrorCategory.ACTION_NOT_PERMITTED,
            "the campaign authorises no actions at all, so nothing can be dispatched; "
            "an empty menu is a stopped campaign, not an unrestricted one",
            field="allowed_actions",
        )
    if action not in menu:
        raise _plan_fail(
            ErrorCategory.ACTION_NOT_PERMITTED,
            f"{action!r} is a registered intervention but this campaign authorises only {sorted(menu)}",
            field="action",
        )

    spec = ACTIONS[action]
    inputs = dict(validated["inputs"])
    backend = str(inputs.get("backend") or spec.default_backend)
    if backend not in spec.backends:
        raise _plan_fail(
            ErrorCategory.INVALID_ENUM,
            f"{action!r} has no backend named {backend!r}; it offers {sorted(spec.backends)}",
            field="inputs",
        )
    kind = str(validated["kind"])
    if kind not in spec.artifact_kinds:
        raise _plan_fail(
            ErrorCategory.INVALID_ENUM,
            f"{action!r} produces {list(spec.artifact_kinds)}, not a {kind!r} artifact",
            field="kind",
        )

    for key, value in inputs.items():
        if key in ARGV_KEYS and isinstance(value, str):
            raise _plan_fail(
                ErrorCategory.ENTRYPOINT_NOT_ARGV,
                f"inputs[{key!r}] is an argument vector, not an interpolated shell command; "
                f"pass {value.split()!r} rather than {value!r}. Nothing in this dispatcher "
                f"runs a command through a shell, so a string here would be a word-split "
                f"waiting for someone else's quoting rules",
                field="inputs",
            )
        if key in ARGV_KEYS:
            if not isinstance(value, (list, tuple)) or not value:
                raise _plan_fail(
                    ErrorCategory.ENTRYPOINT_NOT_ARGV,
                    f"inputs[{key!r}] must be a non-empty argument vector, got {value!r}",
                    field="inputs",
                )
            for part in value:
                if not isinstance(part, str) or not part.strip():
                    raise _plan_fail(
                        ErrorCategory.ENTRYPOINT_NOT_ARGV,
                        f"inputs[{key!r}] holds {part!r}; every element is a non-blank string",
                        field="inputs",
                    )

    for where, text in _iter_strings(inputs):
        rule = is_protected_path(text, root=root)
        if rule is not None:
            raise _plan_fail(
                ErrorCategory.PROTECTED_PATH,
                f"inputs.{where} names {text!r}, which falls under the protected rule "
                f"{rule!r}. The evaluator, the frozen benchmark records, the baseline "
                f"artefacts and the held-out registry are not reachable from a plan",
                field="inputs",
                missing=[text],
            )

    outputs: list[str] = []
    for raw in validated["expected_outputs"]:
        text = str(raw).strip()
        rule = is_protected_path(text, root=root)
        if rule is not None:
            raise _plan_fail(
                ErrorCategory.PROTECTED_PATH,
                f"expected output {text!r} falls under the protected rule {rule!r}; an "
                f"experiment writes into its own output directory and nowhere else",
                field="expected_outputs",
                missing=[text],
            )
        if Path(text).is_absolute():
            raise _plan_fail(
                ErrorCategory.INVALID_VALUE,
                f"expected output {text!r} is absolute; outputs are named relative to the "
                f"experiment's own output directory so that two experiments cannot collide "
                f"and none can write outside its own",
                field="expected_outputs",
            )
        if _normalise(text).startswith(".."):
            raise _plan_fail(
                ErrorCategory.INVALID_VALUE,
                f"expected output {text!r} climbs out of the experiment's output directory",
                field="expected_outputs",
            )
        outputs.append(_normalise(text))

    return {
        "contract_version": CONTRACT_VERSION,
        "plan": validated,
        "action": action,
        "backend": backend,
        "kind": kind,
        "allowed_actions": sorted(menu),
        "artifact_kinds": list(spec.artifact_kinds),
        "required_capabilities": [list(pair) for pair in spec.backends[backend]],
        "handler": spec.handler,
        "worker_authored": spec.worker_authored,
        "expected_outputs": outputs,
        "protected_rules_checked": len(protected_paths()),
        "checks": [
            "schema",
            "campaign_menu",
            "action_registered",
            "action_permitted",
            "backend_and_kind",
            "argument_vectors",
            "protected_paths",
        ],
    }


# ---------------------------------------------------------------------------
# truthful availability
# ---------------------------------------------------------------------------
#: Step 8's per-verb probe record, committed. Used when a caller asks for
#: availability without paying to re-measure it. It is a *recorded measurement*
#: with its own evidence class, not an assumption, and ``probe=True`` re-runs
#: the real probe when the answer has to be current.
COMMITTED_CAPABILITY_RECORD = "configs/research/zak-c9r/baseline-results.json"


def _committed_probes(root: Path | str | None = None) -> dict[tuple[str, str], Record]:
    path = (Path(root) if root is not None else _repo_root()) / COMMITTED_CAPABILITY_RECORD
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[str, str], Record] = {}
    for row in payload.get("capability_probes") or []:
        if isinstance(row, Mapping) and row.get("capability_id") and row.get("verb"):
            out[(str(row["capability_id"]), str(row["verb"]))] = dict(row)
    return out


def action_support(
    action: str,
    *,
    backend: str | None = None,
    probe: bool = True,
    workdir: Path | str | None = None,
    root: Path | str | None = None,
) -> Record:
    """Whether this intervention can actually run here, and if not, why not.

    ``probe=True`` asks the adapters themselves, one verb at a time, through
    Step 8's :func:`~drug_agent.selfevolve.target.baselines.probe_capability` —
    a real, deliberately tiny computation or an explicit unavailability with a
    reason. ``probe=False`` reads the committed probe record instead, which is a
    measurement taken earlier on this host rather than a guess, and says so in
    ``evidence_class``.

    With no ``backend``, every backend the action offers is evaluated and the
    action is ``AVAILABLE`` when *some* backend is fully available. With a named
    backend, only that backend's verbs decide.

    An ``UNAVAILABLE`` answer always carries a ``reason`` naming the missing
    verb and an ``unblock`` saying what would make it available. That is the
    whole point: an unavailable capability is an actionable structured result,
    never a refusal to consider the proposal.

    There are four support tokens, not three, and the fourth is the distinction
    Step 15A measured and Step 15C completed. ``UNAVAILABLE`` is a backend that
    never ran; ``EXECUTED_NO_YIELD`` is one that ran, consumed a cost recorded
    beside it, and returned nothing this pipeline can use; ``UNKNOWN`` is a verb
    nobody has probed. All three refuse a dispatch — only ``AVAILABLE`` lets one
    start — but they are three different findings and reporting a working backend
    that spent real GPU seconds as an unprobed one would be a false record.
    """
    spec = ACTIONS.get(str(action))
    if spec is None:
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"{action!r} is not one of the registered interventions {list(SUPPORTED_ACTIONS)}",
            contract="ExperimentPlan",
            field="action",
        )
    names = [backend] if backend is not None else sorted(spec.backends)
    if backend is not None and backend not in spec.backends:
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"{spec.action!r} has no backend named {backend!r}; it offers {sorted(spec.backends)}",
            contract="ExperimentPlan",
            field="inputs",
        )

    committed = {} if probe else _committed_probes(root)
    rows: list[Record] = []
    seen: set[tuple[str, str]] = set()
    per_backend: dict[str, str] = {}
    for name in names:
        pairs = spec.backends[str(name)]
        if not pairs:
            per_backend[str(name)] = "AVAILABLE"
            continue
        statuses: list[str] = []
        for capability_id, verb in pairs:
            key = (capability_id, verb)
            if key not in seen:
                seen.add(key)
                if probe:
                    from .baselines import probe_capability

                    record = probe_capability(capability_id, verb, workdir=workdir)
                else:
                    record = committed.get(key) or {
                        "capability_id": capability_id,
                        "verb": verb,
                        "status": "NOT_TESTED",
                        "evidence_class": "not tested",
                        "evidence": "no committed probe record and no live probe was requested",
                        "reason": None,
                        "unblock": f"run probe_capability({capability_id!r}, {verb!r})",
                    }
                rows.append(
                    {
                        "capability_id": capability_id,
                        "verb": verb,
                        "status": str(record.get("status")),
                        "evidence_class": str(record.get("evidence_class") or ""),
                        "evidence": str(record.get("evidence") or ""),
                        "reason": record.get("reason"),
                        "unblock": record.get("unblock"),
                        "probed_now": bool(probe),
                    }
                )
            status = next(r["status"] for r in rows if (r["capability_id"], r["verb"]) == key)
            statuses.append(status)
        if any(s == "UNAVAILABLE" for s in statuses):
            per_backend[str(name)] = "UNAVAILABLE"
        elif any(s == "EXECUTED_NO_YIELD" for s in statuses):
            # Step 15A's fourth probe status, completed here by the step
            # authorised to re-freeze. A backend that RAN and returned nothing
            # usable is not a backend nobody asked about: it has a measured cost
            # on the record and a funnel saying where its output went. Collapsing
            # it into UNKNOWN would report a working backend that spent real GPU
            # seconds as an unprobed one, which is the opposite of what happened.
            # It is still unusable — `baselines.UNUSABLE_STATUSES` holds both —
            # and every gate here refuses it exactly as it refuses UNAVAILABLE.
            per_backend[str(name)] = "EXECUTED_NO_YIELD"
        elif all(s == "AVAILABLE_EXECUTED" for s in statuses):
            per_backend[str(name)] = "AVAILABLE"
        else:
            # A verb nobody has asked is not a verb that works. There is no
            # status here meaning "probably fine", so an unprobed backend is
            # UNKNOWN and the dispatcher refuses it with that word rather than
            # launching on an assumption.
            per_backend[str(name)] = "UNKNOWN"

    available = sorted(n for n, s in per_backend.items() if s == "AVAILABLE")
    unknown = sorted(n for n, s in per_backend.items() if s == "UNKNOWN")
    executed_no_yield = sorted(n for n, s in per_backend.items() if s == "EXECUTED_NO_YIELD")
    unavailable_rows = [r for r in rows if r["status"] == "UNAVAILABLE"]
    untested_rows = [r for r in rows if r["status"] == "NOT_TESTED"]
    no_yield_rows = [r for r in rows if r["status"] == "EXECUTED_NO_YIELD"]
    # An unprobed verb still outranks a probed-and-empty one: it might work.
    # A backend that already ran and yielded nothing is the more specific answer,
    # so it is only reported when nothing above it applies.
    if available:
        support = "AVAILABLE"
    elif unknown:
        support = "UNKNOWN"
    elif executed_no_yield:
        support = "EXECUTED_NO_YIELD"
    else:
        support = "UNAVAILABLE"
    if support == "AVAILABLE":
        reason = None
        unblock = None
    elif support == "EXECUTED_NO_YIELD":
        missing = ", ".join(f"{r['capability_id']}/{r['verb']}" for r in no_yield_rows)
        detail = "; ".join(str(r["reason"]) for r in no_yield_rows if r.get("reason"))
        reason = (
            f"{spec.action} ran on this host and produced nothing the pipeline can use: "
            f"{missing} returned EXECUTED_NO_YIELD. The cost it consumed is on the probe "
            f"record and is a measured outcome, not an absent backend. {detail}"
        )
        unblock = " | ".join(str(r["unblock"]) for r in no_yield_rows if r.get("unblock"))
    elif support == "UNKNOWN":
        missing = ", ".join(f"{r['capability_id']}/{r['verb']}" for r in untested_rows)
        reason = (
            f"no backend for {spec.action} has been probed on this host: {missing} is "
            f"NOT_TESTED. An unprobed verb is not an available one"
        )
        unblock = "dispatch with probe=True, or record a probe result for the verbs above"
    else:
        missing = ", ".join(f"{r['capability_id']}/{r['verb']}" for r in unavailable_rows)
        detail = "; ".join(str(r["reason"]) for r in unavailable_rows if r.get("reason"))
        reason = (
            f"{spec.action} needs {missing}, and no backend on this host provides "
            f"{'it' if len(unavailable_rows) == 1 else 'them'}. {detail}"
        )
        unblock = " | ".join(str(r["unblock"]) for r in unavailable_rows if r.get("unblock"))
    return {
        "contract_version": CONTRACT_VERSION,
        "action": spec.action,
        "backend": backend,
        "backends_considered": sorted(str(n) for n in names),
        "backends_available": available,
        "backends_unprobed": unknown,
        "backends_executed_without_yield": executed_no_yield,
        "support": support,
        "evidence_class": "executed" if probe else "inspected (committed Step 8 probe record)",
        "capabilities": rows,
        "reason": reason,
        "unblock": unblock,
        "handler": spec.handler,
    }


# ---------------------------------------------------------------------------
# argument vectors — never a shell fragment
# ---------------------------------------------------------------------------
def run_declared_command(
    validation: Mapping[str, Any],
    *,
    interpreter: str,
    output_dir: Path,
    plan_path: Path,
    repo_root: Path,
    candidate_root: Path,
) -> list[str]:
    """The argument vector a worker-authored experiment runs, after substitution.

    The plan declares ``inputs["command"]`` as a vector — :func:`validate_plan`
    has already refused a string — and exactly five placeholders are substituted:
    ``{python}``, ``{output}``, ``{repo}``, ``{candidate}`` and ``{plan}``. There
    is no template engine and no shell: an element that is not a placeholder is
    passed through byte for byte.

    ``argv[0]`` must be the candidate's own interpreter or an executable inside
    the candidate checkout. A worker cannot name ``/bin/sh`` and get a shell.

    Raises:
        ContractError: ``ENTRYPOINT_NOT_ARGV`` for a missing or unusable vector,
            ``PROTECTED_PATH`` when any element names a protected file.
    """
    inputs = dict(validation["plan"]["inputs"])
    raw = inputs.get("command")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise _plan_fail(
            ErrorCategory.ENTRYPOINT_NOT_ARGV,
            f"a {validation['action']!r} plan declares the command it runs in "
            f"inputs['command'] as an argument vector; got {raw!r}",
            field="inputs",
        )
    substitution = {
        "{python}": str(interpreter),
        "{output}": str(output_dir),
        "{repo}": str(repo_root),
        "{candidate}": str(candidate_root),
        "{plan}": str(plan_path),
    }
    argv = [substitution.get(str(part), str(part)) for part in raw]
    for part in argv:
        rule = is_protected_path(part)
        if rule is not None:
            raise _plan_fail(
                ErrorCategory.PROTECTED_PATH,
                f"the declared command names {part!r}, which falls under the protected rule {rule!r}",
                field="inputs",
                missing=[part],
            )
    head = Path(argv[0])
    allowed = str(head) == str(interpreter) or str(head.resolve()) == str(Path(interpreter).resolve())
    if not allowed:
        try:
            allowed = head.resolve().is_relative_to(Path(candidate_root).resolve())
        except (OSError, ValueError):  # pragma: no cover - unresolvable path
            allowed = False
    if not allowed:
        raise _plan_fail(
            ErrorCategory.ENTRYPOINT_NOT_ARGV,
            f"argv[0] is {argv[0]!r}; a worker-authored experiment runs the candidate's own "
            f"interpreter ({interpreter}) or an executable inside its checkout, so that a "
            f"plan cannot name an interpreter of its own choosing",
            field="inputs",
        )
    return argv


def _handler_argv(
    validation: Mapping[str, Any],
    *,
    interpreter: str,
    output_dir: Path,
    plan_path: Path,
    repo_root: Path,
    candidate_root: Path,
) -> list[str]:
    """The argument vector for one action: the worker's, or this repository's."""
    spec = ACTIONS[str(validation["action"])]
    if spec.worker_authored:
        return run_declared_command(
            validation,
            interpreter=interpreter,
            output_dir=output_dir,
            plan_path=plan_path,
            repo_root=repo_root,
            candidate_root=candidate_root,
        )
    return [
        str(interpreter),
        "-m",
        "drug_agent.selfevolve.target.actions",
        "--action",
        spec.action,
        "--backend",
        str(validation["backend"]),
        "--plan",
        str(plan_path),
        "--output-dir",
        str(output_dir),
        "--repo-root",
        str(repo_root),
    ]


# ---------------------------------------------------------------------------
# the Claude Code experiment worker
# ---------------------------------------------------------------------------
#: Tools the write-capable experiment role gets. It writes inside the candidate
#: checkout and nowhere else; ``cwd`` is the workspace, never the live tree.
CLAUDE_WORKER_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep", "Write", "Edit", "Bash")

#: What a worker is told, every time. Quoted into the handoff on purpose: the
#: prompt is hashed rather than logged, so the invariant part has to live in the
#: source where a reviewer can read it.
CLAUDE_WORKER_CHARTER = (
    "You are a bounded scientific experiment worker inside an isolated candidate checkout. "
    "Implement the hypothesis below as real, runnable code in THIS checkout only. "
    "You do not decide whether the experiment succeeded: an independent evaluator outside "
    "this checkout scores what you produce, from the files you write, and your own prose and "
    "self-reported numbers are not read as evidence. "
    "Never modify the evaluation kernel, the frozen benchmark records, the baseline "
    "manifests, the campaign configuration or any held-out data; those paths are listed "
    "below and a change to any of them is refused at admission. "
    "Write every requested output to the output directory named below, exactly under the "
    "file names requested. Run commands as ordinary argument vectors. "
    "Stay inside the declared budget and stop at the declared stop rule."
)


def claude_cli_interface(binary: str = "claude", *, timeout_s: float = 30.0) -> Record:
    """What the *installed* CLI offers. Probed, never assumed.

    The work order says to probe the interface rather than invent flags or model
    names, so this reads ``--version`` and ``--help`` from the binary on PATH and
    reports which of the flags this module intends to use actually exist. An
    absent flag is reported, not worked around.
    """
    path = shutil.which(binary)
    record: Record = {
        "binary": path,
        "available": bool(path),
        "version": None,
        "flags_present": [],
        "flags_required": ["--print", "--model", "--output-format", "--permission-mode"],
        "flags_missing": [],
        "reason": None if path else f"{binary!r} is not on PATH",
    }
    if not path:
        return record
    try:
        version = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=timeout_s, check=False
        )
        record["version"] = (version.stdout or version.stderr or "").strip()[:200]
        helper = subprocess.run(
            [path, "--help"], capture_output=True, text=True, timeout=timeout_s, check=False
        )
        text = (helper.stdout or "") + (helper.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["available"] = False
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record
    flags = sorted(set(re.findall(r"--[a-zA-Z][a-zA-Z0-9-]+", text)))
    record["flags_present"] = flags
    record["flags_missing"] = [f for f in record["flags_required"] if f not in flags]
    if record["flags_missing"]:
        record["available"] = False
        record["reason"] = (
            f"the installed CLI ({record['version']}) does not offer {record['flags_missing']}; "
            f"this dispatcher does not invent flags"
        )
    return record


def _worker_prompt(
    validation: Mapping[str, Any], *, output_dir: Path, candidate_root: Path
) -> tuple[str, str]:
    """The system and user prompts. Returned so the caller can hash them."""
    plan = validation["plan"]
    requested = "\n".join(f"  - {name}" for name in validation["expected_outputs"])
    budget = "\n".join(f"  - {k}: {v}" for k, v in sorted(plan["requested_budget"].items()))
    protected = "\n".join(f"  - {rule}" for rule in protected_paths())
    system = (
        f"{CLAUDE_WORKER_CHARTER}\n\n"
        f"Protected paths (a change to any of these is refused at admission):\n{protected}\n"
    )
    user = (
        f"Experiment: {plan['experiment_id']}\n"
        f"Campaign: {plan['campaign_id']}\n"
        f"Action: {plan['action']}\n"
        f"Primary track: {plan['primary_track']}\n"
        f"Artifact kind to produce: {plan['kind']}\n\n"
        f"Hypothesis:\n{plan['hypothesis']}\n\n"
        f"Changed factors (what this experiment varies, and nothing else):\n"
        + "\n".join(f"  - {f}" for f in plan["changed_factors"])
        + f"\n\nInputs:\n{json.dumps(plan['inputs'], indent=2, sort_keys=True)}\n\n"
        f"Checkout you may modify: {candidate_root}\n"
        f"Output directory (write the requested files here, by these exact names):\n{output_dir}\n"
        f"Requested outputs:\n{requested}\n\n"
        f"Budget:\n{budget}\n"
        f"Stop rule: {plan['stop_rule']}\n\n"
        f"Development criterion the independent evaluator will apply: "
        f"{plan['development_metric']}\n"
        f"Also write {output_dir}/result.json declaring what you produced:\n"
        '  {"status": "SUCCEEDED"|"FAILED", "notes": "...", '
        '"artifacts": [{"artifact_id": "...", "kind": "...", "entrypoint": ["python", "..."], '
        '"weights": ["<file in the output directory>"], "structures": [], "training_data": []}]}\n'
        "Declare only files you actually wrote. A declared digest that disagrees with the "
        "bytes on disk, a missing output, or a claimed success with a nonzero exit status is "
        "refused."
    )
    return system, user


def _prompt_digest(text: str) -> Record:
    """A prompt's identity without its contents.

    Appendix A §6 keeps protected data out of logs; a plan's inputs can name
    molecules and assay rows, so the prompt is recorded as a content address and
    a length, never as text.
    """
    return {
        "sha256": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "lines": text.count("\n") + 1,
    }


def run_claude_experiment(
    validation: Mapping[str, Any],
    *,
    candidate_root: Path,
    output_dir: Path,
    log_dir: Path | None = None,
    backend: Any = None,
    binary: str = "claude",
    model: str | None = None,
    max_output_tokens: int = 8192,
    timeout_s: float = 900.0,
    ledger: Any = None,
) -> Record:
    """Ask the installed Claude Code CLI to implement one experiment, in place.

    Reuses :class:`~drug_agent.selfevolve.backends.claude_cli_backend.ClaudeCLIBackend`
    — the repository's one supported, subscription-only path to the CLI, which
    already strips the API-billing variables from the child environment and
    settles its output-token reservation against Step 9's ledger. Nothing here
    opens a second route to a provider.

    What is recorded: the resolved model identity, the CLI version, both prompt
    content addresses, the number of turns, the measured token usage, and the
    implementation diff the worker left in the checkout. What is **not**
    recorded: the prompt text, because a plan's inputs can name molecules.

    Returns a record with ``ok``; on failure it also carries ``failure_category``
    and a ``reason``. The worker's own prose is carried as evidence of what it
    said, never as evidence of what happened.
    """
    started = time.time()
    interface = claude_cli_interface(binary)
    plan = validation["plan"]
    system, user = _worker_prompt(validation, output_dir=output_dir, candidate_root=candidate_root)
    record: Record = {
        "phase": "worker",
        "interface": interface,
        "prompt": {"system": _prompt_digest(system), "user": _prompt_digest(user)},
        "charter_sha256": "sha256:" + hashlib.sha256(CLAUDE_WORKER_CHARTER.encode()).hexdigest(),
        "requested_model": model,
        "resolved_model": None,
        "cwd": str(candidate_root),
        "allowed_tools": list(CLAUDE_WORKER_TOOLS),
        "ok": False,
        "failure_category": None,
        "reason": None,
        "usage": {"output_tokens": None, "input_tokens": None, "num_turns": None},
        "diff": None,
    }
    if not interface["available"]:
        record["failure_category"] = FailureCategory.INFRASTRUCTURE_ERROR.value
        record["reason"] = f"the Claude Code CLI is not usable here: {interface['reason']}"
        record["elapsed_s"] = round(time.time() - started, 3)
        return record

    if backend is None:
        from ..backends.claude_cli_backend import ClaudeCLIBackend

        backend = ClaudeCLIBackend(
            binary=binary,
            cwd=str(candidate_root),
            allowed_tools=CLAUDE_WORKER_TOOLS,
            timeout_s=timeout_s,
            ledger=ledger,
            default_max_output_tokens=max_output_tokens,
        )
    result = backend.complete(
        system=system,
        user=user,
        role=f"experiment-worker:{plan['action']}",
        model=model,
        timeout_s=timeout_s,
        allowed_tools=CLAUDE_WORKER_TOOLS,
        cwd=str(candidate_root),
        max_output_tokens=max_output_tokens,
    )
    raw = dict(getattr(result, "raw", None) or {})
    record["resolved_model"] = raw.get("model_resolved") or getattr(result, "model", None)
    # A session can touch more than one model. The one above is the one that
    # generated the most output; this is all of them, so a silent change of
    # model mid-study is visible rather than averaged away.
    record["models_used"] = raw.get("model_usage")
    record["session_id"] = getattr(result, "session_id", None)
    record["usage"] = {
        "output_tokens": getattr(result, "output_tokens", None),
        "input_tokens": getattr(result, "input_tokens", None),
        "num_turns": raw.get("num_turns"),
        "cost_estimate_usd": getattr(result, "cost_estimate_usd", None),
    }
    record["permission_denials"] = raw.get("permission_denials")
    text = str(getattr(result, "text", "") or "")
    report_path: Path | None = None
    if log_dir is not None and text:
        # The worker's own account of what it did — its plan, in its words. Kept
        # beside the run rather than in the record, because it is *testimony*:
        # a reviewer needs to be able to read it, and nothing downstream may
        # score it. Run directories live under `.external/` and are gitignored,
        # so this is not a route for a plan's inputs to reach a commit.
        report_path = Path(log_dir) / f"worker-report-{plan['experiment_id']}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
    record["worker_prose"] = {
        "sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "path": str(report_path) if report_path else None,
        "note": "recorded as evidence of what the worker said, never as evidence of what happened",
    }
    record["diff"] = _candidate_diff(
        candidate_root, log_dir=log_dir, experiment_id=str(plan["experiment_id"])
    )
    record["elapsed_s"] = round(time.time() - started, 3)
    touched = list(record["diff"]["touches_protected"])
    if touched:
        # Appendix A §6: "Changes to the evaluator, benchmark labels, lock,
        # master objective, or baseline artifacts are rejected by the
        # controller." The referee is already out of reach — the evaluation runs
        # in the dispatcher's process against the *parent's* modules, so a
        # worktree edit could not have moved it — but a candidate that tried is
        # a rejected candidate, not a successful one, and the attempt ends here
        # instead of going on to run whatever else it wrote.
        record["ok"] = False
        record["failure_category"] = FailureCategory.INVALID_CANDIDATE.value
        record["reason"] = (
            f"the experiment worker changed {len(touched)} protected path(s) in its checkout "
            f"({touched}); the evaluator, the frozen records, the baseline artefacts and the "
            f"held-out registry are not a candidate's to edit, and a candidate that edits them "
            f"is refused rather than scored"
        )
        return record
    if not getattr(result, "ok", False):
        record["failure_category"] = FailureCategory.INFRASTRUCTURE_ERROR.value
        record["reason"] = f"the Claude Code CLI call failed: {getattr(result, 'error', None)}"
        return record
    record["ok"] = True
    return record


def _candidate_diff(root: Path, *, log_dir: Path | None, experiment_id: str) -> Record:
    """Inspect all Git-root changes, mapping them back to the task project."""
    root = Path(root).resolve()
    layout = RepositoryLayout.discover(root)
    checkout = layout.git_root or root
    # Include newly authored files in the saved patch, while path inspection
    # separately includes ignored additions and both sides of renames/deletions.
    _git("add", "--all", "--intent-to-add", ".", cwd=checkout)
    proc = _git("--no-pager", "diff", "HEAD", "--no-color", "--binary", cwd=checkout)
    stat = _git("--no-pager", "diff", "HEAD", "--stat", "--no-color", cwd=checkout)
    changed = layout.changed_paths()
    others = _git("ls-files", "--others", "-z", cwd=checkout)
    untracked = [name for name in others.stdout.split("\0") if name]
    text = proc.stdout if proc.returncode == 0 else ""
    path: Path | None = None
    if log_dir is not None and text:
        path = Path(log_dir) / f"worker-diff-{experiment_id}.patch"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    protected: list[str] = []
    outside: list[str] = []
    for name in changed:
        try:
            relative = layout.project_path(name)
        except ValueError:
            outside.append(name)
            protected.append(name)
            continue
        if is_protected_path(relative, root=root):
            protected.append(name)
    return {
        "sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "bytes": len(text.encode()),
        "stat": (stat.stdout or "").strip()[-2000:],
        "untracked_files": sorted(untracked)[:50],
        "n_untracked": len(untracked),
        "patch_path": str(path) if path else None,
        "project_prefix": layout.project_prefix,
        "outside_project": sorted(outside),
        "touches_protected": sorted(protected),
    }


# ---------------------------------------------------------------------------
# what the run left behind — verified, never taken on trust
# ---------------------------------------------------------------------------
def verify_produced_outputs(output_dir: Path | str, expected_outputs: Sequence[str]) -> Record:
    """Every declared output exists, is non-empty, and deserializes.

    ``validate_artifact`` proves a *record* is well formed; this proves the
    *bytes* are there and readable. A ``.json`` output that does not parse is a
    failure with the parser's own message, because "the worker said it wrote
    predictions" and "there are predictions" are different facts.
    """
    root = Path(output_dir)
    rows: list[Record] = []
    missing: list[str] = []
    malformed: list[Record] = []
    for relative in expected_outputs:
        path = root / relative
        if not path.is_file():
            missing.append(str(relative))
            continue
        data = path.read_bytes()
        row: Record = {
            "path": str(relative),
            "bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "deserialized": None,
        }
        if not data:
            malformed.append({"path": str(relative), "reason": "the file is empty"})
        elif str(relative).endswith(".json"):
            try:
                json.loads(data.decode("utf-8"))
                row["deserialized"] = "json"
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                malformed.append(
                    {
                        "path": str(relative),
                        "reason": f"declared as JSON and does not deserialize: {type(exc).__name__}: {exc}",
                    }
                )
        rows.append(row)
    ok = not missing and not malformed
    reasons = [f"expected output {m!r} was never written" for m in missing]
    reasons += [f"expected output {m['path']!r} {m['reason']}" for m in malformed]
    return {
        "ok": ok,
        "outputs": rows,
        "missing": missing,
        "malformed": malformed,
        "reason": "; ".join(reasons) or None,
    }


def _environment_record(mc: MaterializedCandidate, env: Mapping[str, str]) -> Record:
    return {
        "kind": "candidate_environment",
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "platform": sys.platform,
        "interpreter": mc.interpreter,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "source_root": str(mc.source_root),
        "env": redact_env(env),
    }


def _tree_manifest(source_root: Path) -> Record:
    """``{relative path: sha256}`` over the candidate's own source.

    This is what ``code_hash`` addresses: a weights-only candidate has source
    byte-identical to its parent, so the manifest is what makes "which code was
    this artifact produced by" answerable at all.
    """
    root = Path(source_root)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:  # pragma: no cover - a file that vanished mid-walk
            out[path.relative_to(root).as_posix()] = "UNREADABLE"
    return {"kind": "candidate_source_manifest", "root": root.name, "n_files": len(out), "files": out}


def _target_manifest_hash(root: Path, override: str | None) -> str | None:
    if override:
        return str(override)
    path = root / ".external" / "research" / "zak-c9r" / "derived" / "target_manifest.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["manifest_hash"])
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None


def _assemble_artifacts(
    claims: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    output_dir: Path,
    store: Any,
    code_hash: str,
    environment_hash: str,
    target_manifest_hash: str,
) -> Record:
    """Turn what the run *declared* into artifact records the store can prove.

    Every declared file is read from the output directory and put into the
    content-addressed store, so ``weights_hashes`` addresses bytes that exist. A
    claim that also states a digest is checked against the measured one and a
    disagreement is ``HASH_MISMATCH``: a worker that reports a checkpoint's hash
    and writes different bytes has produced an unattributable artifact, not a
    slightly-off record.

    ``code_hash`` defaults to the candidate's own source manifest. A claim may
    override it — that is how a worker points at the specific module bundle its
    artifact needs — and the override is then verified to exist, which is what
    refuses a weights artifact with no code behind it.
    """
    records: list[Record] = []
    problems: list[Record] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            problems.append(
                {
                    "category": ErrorCategory.INVALID_TYPE.value,
                    "reason": f"artifact claim {index} is not a mapping",
                }
            )
            continue
        stored: dict[str, list[str]] = {
            "weights_hashes": [],
            "structure_hashes": [],
            "training_data_hashes": [],
        }
        declared_map = {
            "weights_hashes": claim.get("weights") or [],
            "structure_hashes": claim.get("structures") or [],
            "training_data_hashes": claim.get("training_data") or [],
        }
        failed = False
        for declared_field, names in declared_map.items():
            for name in names:
                path = output_dir / str(name)
                if not path.is_file():
                    problems.append(
                        {
                            "category": ErrorCategory.UNVERIFIABLE_REFERENCE.value,
                            "reason": (
                                f"artifact claim declares {declared_field[:-7]} file {name!r}, "
                                f"which was never written"
                            ),
                        }
                    )
                    failed = True
                    continue
                data = path.read_bytes()
                measured = "sha256:" + hashlib.sha256(data).hexdigest()
                ref = store.put_bytes(
                    data,
                    kind=str(claim.get("kind") or "artifact"),
                    produced_by=str(plan["experiment_id"]),
                    metadata={"declared_as": declared_field, "file": str(name)},
                )
                if str(ref.content_hash) != measured:  # pragma: no cover - content addressed
                    problems.append(
                        {
                            "category": ErrorCategory.HASH_MISMATCH.value,
                            "reason": (
                                f"the store addressed {name!r} as {ref.content_hash} and its "
                                f"bytes hash to {measured}"
                            ),
                        }
                    )
                    failed = True
                stored[declared_field].append(measured)
        declared_digests = claim.get("digests") or {}
        for name, declared in sorted(dict(declared_digests).items()):
            path = output_dir / str(name)
            measured_digest = (
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            )
            if measured_digest != str(declared):
                problems.append(
                    {
                        "category": ErrorCategory.HASH_MISMATCH.value,
                        "reason": (
                            f"the run declared {name!r} as {declared!r} and the bytes on disk hash "
                            f"to {measured_digest!r}; a declared digest that disagrees with what "
                            f"was written makes the artifact unattributable"
                        ),
                    }
                )
                failed = True
        for hash_field in ("weights_hashes", "structure_hashes", "training_data_hashes"):
            for extra in claim.get(hash_field) or []:
                if str(extra) not in stored[hash_field]:
                    stored[hash_field].append(str(extra))
        if failed:
            continue
        records.append(
            {
                "artifact_id": str(claim.get("artifact_id") or f"art_{plan['experiment_id']}_{index}"),
                "parent_id": claim.get("parent_id", plan.get("parent_id")),
                "kind": str(claim.get("kind") or plan["kind"]),
                "target_manifest_hash": target_manifest_hash,
                "entrypoint": list(claim.get("entrypoint") or []),
                "code_hash": str(claim.get("code_hash") or code_hash),
                "weights_hashes": stored["weights_hashes"],
                "structure_hashes": stored["structure_hashes"],
                "training_data_hashes": stored["training_data_hashes"],
                "environment_hash": str(claim.get("environment_hash") or environment_hash),
                "experiment_id": str(plan["experiment_id"]),
            }
        )
    return {"artifacts": records, "problems": problems}


def independent_development_evaluation(
    plan: Mapping[str, Any], *, output_dir: Path, root: Path | None = None
) -> Record:
    """Score what the run produced, here, with the parent's frozen evaluator.

    This is the half that makes a worker's output a measurement. It runs in the
    *dispatcher's* process, imports the parent checkout's evaluation kernel, and
    reads only the prediction file the run wrote. The run's own ``result.json``
    may claim any number it likes; nothing here looks at it.
    """
    metric = str(plan["development_metric"])
    out = Path(output_dir)
    base: Record = {
        "metric": metric,
        "status": "NOT_APPLICABLE",
        "value": None,
        "evaluated_partition": "development",
        "source": None,
        "evaluator_hash": None,
        "reason": None,
    }
    if metric == "mae_pactivity":
        from ..evaluation.target_benchmark import evaluator_hash, regression_endpoint
        from .baseline_arms import prepare_track_a
        from .baselines import pool_hash

        source = out / "predictions.json"
        base["source"] = "predictions.json"
        base["evaluator_hash"] = evaluator_hash()
        if not source.is_file():
            base["status"] = "UNAVAILABLE"
            base["reason"] = "the run wrote no predictions.json, so there is nothing to score"
            return base
        predictions = json.loads(source.read_text(encoding="utf-8"))
        prepared = prepare_track_a(dict(plan["inputs"].get("preparation") or {}))
        endpoint = regression_endpoint(
            pool=list(prepared["pool"]),
            labels=dict(prepared["labels"]),
            predictions=dict(predictions),
            failure_penalty=float(prepared["failure_penalty"]),
        )
        base.update(
            {
                "status": "MEASURED",
                "value": endpoint["mae"],
                "n_pool": endpoint["n_pool"],
                "n_scored": endpoint["n_scored"],
                "n_failed": endpoint["n_failed"],
                "coverage": endpoint["coverage"],
                "pool_hash": pool_hash(prepared["pool"]),
                "endpoint": endpoint,
                "boundary": prepared["boundary"],
            }
        )
        return base
    if metric == "qualifying_scaffolds":
        from ..evaluation.target_benchmark import evaluate_portfolio, evaluator_hash

        source = out / "molecules.json"
        base["source"] = "molecules.json"
        base["evaluator_hash"] = evaluator_hash()
        qualification_path = (
            (root or _repo_root()) / "configs" / "research" / "zak-c9r" / "qualification.json"
        )
        if not source.is_file() or not qualification_path.is_file():
            base["status"] = "UNAVAILABLE"
            base["reason"] = (
                "Track B qualification needs both the run's molecules.json and the frozen "
                "qualification record; one of them is absent"
            )
            return base
        molecules = json.loads(source.read_text(encoding="utf-8"))
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        portfolio = evaluate_portfolio(molecules, qualification=qualification)
        base.update(
            {
                "status": "MEASURED",
                "value": float(portfolio.get("qualifying_scaffolds", 0)),
                "portfolio": portfolio,
            }
        )
        return base
    base["reason"] = (
        f"no independent evaluator in this repository computes {metric!r}; the development "
        f"number is therefore unknown rather than zero"
    )
    return base


# ---------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------
def _undispatched(
    validation: Mapping[str, Any],
    *,
    failure_category: FailureCategory,
    reason: str,
    started: float,
    candidate_id: str,
    extra_cost: Mapping[str, float | None] | None = None,
) -> Record:
    """A valid ``ExperimentResult`` for a run that correctly never launched.

    A refused reservation and a missing backend are *results*, not crashes, and
    both carry a cost: the time spent deciding is real even when no science
    happened. The development metric is ``None`` — unknown, never zero.
    """
    from ...cli.main import run_exit_code

    plan = validation["plan"]
    record = {
        "experiment_id": str(plan["experiment_id"]),
        "status": ExperimentStatus.FAILED.value,
        "exit_code": run_exit_code(ExperimentStatus.FAILED.value),
        "candidate_revision": f"undispatched:{candidate_id}",
        "import_origin": f"UNAVAILABLE: {(reason or 'no reason recorded')[:200]}",
        "artifacts": [],
        "development_metrics": {str(plan["development_metric"]): None},
        "actual_cost": {
            "attempts": 0.0,
            "wall_s": round(time.time() - started, 3),
            "gpu_hours": None,
            "llm_output_tokens": None,
            **dict(extra_cost or {}),
        },
        "failure_category": failure_category.value,
        "intervention_log": [
            {
                "event": "not_dispatched",
                "experiment_id": str(plan["experiment_id"]),
                "failure_category": failure_category.value,
                "reason": reason,
                "at": time.time(),
            }
        ],
    }
    return validate_record(ExperimentResult, record)


def dispatch_plan(
    plan: Mapping[str, Any],
    *,
    allowed_actions: Sequence[str],
    output_dir: Path | str,
    log_dir: Path | str | None = None,
    ledger: Any = None,
    store: Any = None,
    store_root: Path | str | None = None,
    root: Path | str | None = None,
    cfg: EvolveConfig | None = None,
    candidate: Candidate | None = None,
    parent_root: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    source_root: Path | str | None = None,
    interpreter: str | None = None,
    timeout_s: float = 1800.0,
    probe: bool = True,
    allow_network: bool = False,
    campaign_phase: str = "ADAPTATION",
    request_key: str | None = None,
    target_manifest_hash: str | None = None,
    claude_backend: Any = None,
    claude_model: str | None = None,
    claude_timeout_s: float = 900.0,
    max_output_tokens: int = 8192,
    max_infrastructure_retries: int = 0,
    keep_workspace: bool | None = None,
) -> Record:
    """Cause one experiment — schema, budget, capability, isolation, evidence.

    The order is the contract, and every step before the launch is a step that
    can stop it:

    1. :func:`validate_plan` admits the plan, or refuses it and nothing runs;
    2. :func:`action_support` answers whether the backend exists. An
       unavailable one produces a ``CAPABILITY_UNAVAILABLE`` result naming the
       missing verb — the proposal was accepted, the capability is not there;
    3. Step 9's ledger reserves every declared resource **before** anything
       launches. A refusal is a ``BUDGET_EXHAUSTED`` result and the child is
       never started;
    4. Step 4's :func:`execute_candidate` materialises an isolated checkout,
       verifies that the child really imports the *candidate's* code, and runs
       the action's argument vector under a wall-time cap with process-group
       termination;
    5. every declared output is verified to exist and deserialize, every
       declared digest is checked against the bytes on disk, and every artifact
       reference is resolved in the content-addressed store;
    6. the development number is computed **here**, by the parent's frozen
       evaluator, from the files the run wrote. The run's own ``result.json``
       cannot raise a failure to a success, and its claimed scores are never
       read into ``development_metrics``.

    Returns:
        A record carrying the validated ``ExperimentResult``, the provenance,
        the capability report, the independent evaluation and the execution
        detail. Never a bare metric.
    """
    from .budget import BudgetExceeded

    started = time.time()
    validation = validate_plan(plan, allowed_actions, root=root)
    vplan = validation["plan"]
    spec = ACTIONS[str(validation["action"])]
    experiment_id = str(vplan["experiment_id"])

    repo = Path(root) if root is not None else _repo_root()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = Path(log_dir) if log_dir is not None else output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    plan_path = logs / f"plan-{experiment_id}.json"
    plan_path.write_text(json.dumps(vplan, indent=2, sort_keys=True), encoding="utf-8")

    # A re-run into the same directory must not inherit the previous run's
    # evidence. Without this, "the declared output exists and deserializes"
    # could be satisfied by bytes an earlier attempt left behind, and a run that
    # wrote nothing at all would be accepted on somebody else's work.
    stale: list[str] = []
    for relative in [*validation["expected_outputs"], "result.json"]:
        previous = output_dir / relative
        if previous.is_file():
            previous.unlink()
            stale.append(str(relative))

    if spec.worker_authored and not vplan["inputs"].get("command"):
        raise _plan_fail(
            ErrorCategory.ENTRYPOINT_NOT_ARGV,
            f"a {spec.action!r} plan runs code the worker wrote, so it must declare the "
            f"argument vector that runs it in inputs['command']",
            field="inputs",
        )

    candidate_id = candidate.candidate_id if candidate is not None else f"cand_{experiment_id}"
    envelope: Record = {
        "contract_version": CONTRACT_VERSION,
        "experiment_id": experiment_id,
        "action": spec.action,
        "backend": validation["backend"],
        "validation": validation,
        "output_dir": str(output_dir),
        "log_dir": str(logs),
        "plan_path": str(plan_path),
        "reservations": [],
        "worker": None,
        "verification": None,
        "provenance": None,
        "independent_evaluation": {
            "metric": str(vplan["development_metric"]),
            "status": "NOT_RUN",
            "value": None,
        },
        "detail": None,
        "reason": None,
        "error_category": None,
        "worker_claim": None,
        "cleared_stale_outputs": stale,
    }

    support = action_support(spec.action, backend=validation["backend"], probe=probe, root=repo)
    envelope["support"] = support
    if support["support"] != "AVAILABLE":
        unsupported = str(support["reason"])
        envelope["reason"] = unsupported
        envelope["result"] = _undispatched(
            validation,
            failure_category=FailureCategory.CAPABILITY_UNAVAILABLE,
            reason=unsupported,
            started=started,
            candidate_id=candidate_id,
        )
        _write_envelope(logs, experiment_id, envelope)
        return envelope

    # --- reserve before anything launches -------------------------------
    reservations: list[Record] = []
    if ledger is not None:
        requested = dict(vplan["requested_budget"])
        # ``max_experiment_proposals`` is a declared campaign cap that Step 9
        # left unenforced because it had no proposal loop to reserve against.
        # This is that loop: when the campaign has declared the cap, dispatching
        # a plan spends one proposal, so the seventh plan of a six-proposal
        # campaign is refused *before* it launches rather than counted afterwards.
        if "experiment_proposals" not in requested and "experiment_proposals" in ledger.caps():
            requested["experiment_proposals"] = 1.0
        try:
            for resource, amount in sorted(requested.items()):
                reservations.append(
                    dict(
                        ledger.reserve(
                            str(resource),
                            float(amount),
                            owner=f"experiment:{experiment_id}",
                            phase=campaign_phase,
                            request_key=f"{request_key}:{resource}" if request_key else None,
                            note=f"{spec.action} for {experiment_id}",
                        )
                    )
                )
        except BudgetExceeded as exc:
            for held in reservations:
                with contextlib.suppress(Exception):
                    ledger.release(held["reservation_id"], note="the experiment never launched")
            envelope["reservations"] = reservations
            envelope["reason"] = str(exc)
            envelope["error_category"] = ErrorCategory.INVALID_VALUE.value
            envelope["result"] = _undispatched(
                validation,
                failure_category=FailureCategory.BUDGET_EXHAUSTED,
                reason=str(exc),
                started=started,
                candidate_id=candidate_id,
            )
            _write_envelope(logs, experiment_id, envelope)
            return envelope
        envelope["reservations"] = reservations
        already = [r for r in reservations if r.get("already_completed")]
        if already:
            resumed = (
                f"request key {request_key!r} was already settled in the budget ledger; "
                f"a resumed request is neither charged nor executed twice"
            )
            envelope["reason"] = resumed
            envelope["result"] = _undispatched(
                validation,
                failure_category=FailureCategory.BUDGET_EXHAUSTED,
                reason=resumed,
                started=started,
                candidate_id=candidate_id,
            )
            _write_envelope(logs, experiment_id, envelope)
            return envelope

    if store is None:
        from ...storage.artifacts import LocalArtifactStore

        store = LocalArtifactStore(
            Path(store_root) if store_root is not None else output_dir.parent / "artifacts"
        )
    manifest_hash = _target_manifest_hash(repo, target_manifest_hash)

    identity: Record = {"code_hash": None, "environment_hash": None}
    worker_record: Record = {}

    def _on_materialized(mc: MaterializedCandidate) -> Record:
        env = candidate_env(mc, allow_network=allow_network)
        identity["code_hash"] = str(store.put_json(_tree_manifest(mc.source_root)).content_hash)
        identity["environment_hash"] = str(store.put_json(_environment_record(mc, env)).content_hash)
        if not (spec.worker_authored and validation["backend"] == "claude_code"):
            return {
                "ok": True,
                "phase": "identity",
                "note": "this action runs a fixed adapter; no experiment worker",
            }
        # The plan's own ``llm_output_tokens`` reservation, if it declared one,
        # was taken before anything launched and is the charge for this call.
        # Handing the ledger to the backend as well would reserve a second cap
        # and then charge the same generated tokens twice, so the backend is
        # given the ledger only when the plan left that resource undeclared.
        worker_ledger = None if "llm_output_tokens" in dict(vplan["requested_budget"]) else ledger
        record = run_claude_experiment(
            validation,
            candidate_root=mc.root,
            output_dir=output_dir,
            log_dir=logs,
            backend=claude_backend,
            model=claude_model,
            max_output_tokens=max_output_tokens,
            timeout_s=claude_timeout_s,
            ledger=worker_ledger,
        )
        worker_record.update(record)
        return record

    def _commands(mc: MaterializedCandidate) -> list[tuple[str, list[str]]]:
        argv = _handler_argv(
            validation,
            interpreter=mc.interpreter,
            output_dir=output_dir,
            plan_path=plan_path,
            repo_root=repo,
            candidate_root=mc.root,
        )
        return [(spec.action, argv)]

    base = execute_candidate(
        candidate or Candidate(candidate_id=candidate_id, patch="", files_changed=[]),
        experiment_id=experiment_id,
        commands=_commands,
        cfg=cfg,
        parent_root=Path(parent_root) if parent_root is not None else None,
        workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
        source_root=Path(source_root) if source_root is not None else None,
        interpreter=interpreter,
        log_dir=logs,
        timeout_s=timeout_s,
        allow_network=allow_network,
        on_materialized=_on_materialized,
        env_extra={
            "DRUGAGENT_EXPERIMENT_ID": experiment_id,
            "DRUGAGENT_OUTPUT_DIR": str(output_dir),
            "DRUGAGENT_REPO_ROOT": str(repo),
        },
        max_infrastructure_retries=max_infrastructure_retries,
        keep_workspace=keep_workspace,
    )
    detail: Record = {}
    detail_path = logs / f"detail-{experiment_id}.json"
    if detail_path.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
    envelope["detail"] = detail
    envelope["worker"] = worker_record or detail.get("worker")
    envelope["identity"] = dict(identity)

    status = str(base["status"])
    failure_category = str(base["failure_category"])
    reason: str | None = detail.get("reason")
    error_category: str | None = None

    # What the run said about itself. Read as testimony, never as a verdict.
    claim_path = output_dir / "result.json"
    worker_claim: Record | None = None
    if claim_path.is_file():
        try:
            loaded = json.loads(claim_path.read_text(encoding="utf-8"))
            worker_claim = dict(loaded) if isinstance(loaded, Mapping) else {"malformed": True}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            worker_claim = {"malformed": True, "error": f"{type(exc).__name__}: {exc}"}
    envelope["worker_claim"] = worker_claim
    claimed = str((worker_claim or {}).get("status") or "").upper()
    if status != ExperimentStatus.SUCCEEDED.value and claimed in ("SUCCEEDED", "OK", "SUCCESS"):
        envelope["claim_contradicted"] = {
            "claimed_status": claimed,
            "observed_status": status,
            "observed_exit_code": base["exit_code"],
            "note": (
                "the run reported success and its process did not; the recorded outcome is "
                "the observed one. A worker does not grade itself"
            ),
        }

    artifact_records: list[Record] = []
    if status == ExperimentStatus.SUCCEEDED.value:
        verification = verify_produced_outputs(output_dir, validation["expected_outputs"])
        envelope["verification"] = verification
        if not verification["ok"]:
            status = ExperimentStatus.FAILED.value
            failure_category = FailureCategory.CANDIDATE_ERROR.value
            reason = verification["reason"]
        elif worker_claim is not None and worker_claim.get("malformed"):
            status = ExperimentStatus.FAILED.value
            failure_category = FailureCategory.CANDIDATE_ERROR.value
            reason = f"result.json does not deserialize: {worker_claim.get('error')}"
        elif claimed in ("UNAVAILABLE", "CAPABILITY_UNAVAILABLE"):
            # The handler ran, asked its backend, and was told no. That is an
            # answer with an unblock in it, not a broken candidate, and the two
            # are not interchangeable: one is retried by a human adding a verb,
            # the other by fixing code.
            status = ExperimentStatus.FAILED.value
            failure_category = FailureCategory.CAPABILITY_UNAVAILABLE.value
            reason = str((worker_claim or {}).get("reason") or (worker_claim or {}).get("notes"))
        elif claimed and claimed not in ("SUCCEEDED", "OK", "SUCCESS"):
            status = ExperimentStatus.FAILED.value
            failure_category = FailureCategory.CANDIDATE_ERROR.value
            reason = f"the run reported status {claimed!r}: {(worker_claim or {}).get('notes')}"
        else:
            if manifest_hash is None:
                status = ExperimentStatus.FAILED.value
                failure_category = FailureCategory.DATA_UNAVAILABLE.value
                reason = (
                    "no target manifest hash is resolvable on this host, so a produced artifact "
                    "cannot be scoped to a target; an artifact with no target identity is refused"
                )
            else:
                assembled = _assemble_artifacts(
                    (worker_claim or {}).get("artifacts") or [],
                    plan=vplan,
                    output_dir=output_dir,
                    store=store,
                    code_hash=str(identity["code_hash"]),
                    environment_hash=str(identity["environment_hash"]),
                    target_manifest_hash=manifest_hash,
                )
                envelope["artifact_problems"] = assembled["problems"]
                if assembled["problems"]:
                    status = ExperimentStatus.FAILED.value
                    failure_category = FailureCategory.CANDIDATE_ERROR.value
                    error_category = str(assembled["problems"][0]["category"])
                    reason = str(assembled["problems"][0]["reason"])
                else:
                    artifact_records = assembled["artifacts"]
                    try:
                        for record in artifact_records:
                            verify_artifact_references(record, store)
                    except ContractError as exc:
                        artifact_records = []
                        status = ExperimentStatus.FAILED.value
                        failure_category = FailureCategory.CANDIDATE_ERROR.value
                        error_category = exc.category.value
                        reason = str(exc)

    if status == ExperimentStatus.SUCCEEDED.value:
        evaluation = independent_development_evaluation(vplan, output_dir=output_dir, root=repo)
        envelope["independent_evaluation"] = evaluation
    else:
        evaluation = envelope["independent_evaluation"]

    metric = str(vplan["development_metric"])
    metrics: dict[str, float | None] = {metric: None}
    if evaluation.get("status") == "MEASURED" and evaluation.get("value") is not None:
        metrics[metric] = float(evaluation["value"])
        if evaluation.get("coverage") is not None:
            metrics["coverage"] = float(evaluation["coverage"])

    from ...cli.main import run_exit_code

    interventions = list(base["intervention_log"])
    if status != str(base["status"]):
        interventions.append(
            {
                "event": "acceptance_refused",
                "experiment_id": experiment_id,
                "observed_status": base["status"],
                "recorded_status": status,
                "failure_category": failure_category,
                "error_category": error_category,
                "reason": reason,
                "at": time.time(),
            }
        )
    cost = dict(base["actual_cost"])
    tokens = ((worker_record or {}).get("usage") or {}).get("output_tokens")
    cost["llm_output_tokens"] = (
        float(tokens) if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) else None
    )
    final = {
        **base,
        "status": status,
        "failure_category": failure_category,
        "exit_code": run_exit_code(status),
        "artifacts": [r["artifact_id"] for r in artifact_records],
        "development_metrics": metrics,
        "actual_cost": cost,
        "intervention_log": interventions,
    }
    result = validate_record(ExperimentResult, final)
    envelope["result"] = result
    envelope["reason"] = reason
    envelope["error_category"] = error_category
    envelope["artifact_records"] = artifact_records
    envelope["provenance"] = require_experiment_provenance(
        result,
        plan=vplan,
        artifacts=artifact_records,
        store=store,
        run_environment_hash=identity["environment_hash"],
    )

    if ledger is not None and reservations:
        wall = float(cost.get("wall_s") or 0.0)
        for held in reservations:
            resource = str(held["resource"])
            consumed: float | None
            telemetry = "MEASURED"
            if resource in ("wall_seconds", "wall_s"):
                consumed = min(wall, float(held["reserved"]))
            elif resource == "experiment_proposals":
                # A dispatched plan is a spent proposal whether or not it
                # succeeded. Appendix A §4 charges failed trials.
                consumed = float(held["reserved"])
            elif resource == "llm_output_tokens":
                value = cost.get("llm_output_tokens")
                consumed = float(value) if value is not None else None
                telemetry = "MEASURED" if value is not None else "UNAVAILABLE"
            else:
                consumed = None
                telemetry = "UNAVAILABLE"
            with contextlib.suppress(Exception):
                ledger.settle(
                    held["reservation_id"],
                    consumed=consumed,
                    telemetry=telemetry,
                    status="SETTLED" if status == ExperimentStatus.SUCCEEDED.value else "FAILED",
                    note=f"{spec.action} {experiment_id}",
                )
            if consumed is not None and consumed > float(held["reserved"]) + 1e-9:
                # A reservation is a hold, not a hard stop for every resource: a
                # multi-turn agentic session generates across many responses and
                # ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` bounds one response, not the
                # session. The ledger charges what was really spent — an
                # overspend is never forgiven — and the overrun is recorded here
                # so the estimate can be corrected rather than quietly repeated.
                # The *campaign* cap is still the real bound: the next
                # reservation sees the whole committed total.
                overrun = {
                    "event": "reservation_overrun",
                    "experiment_id": experiment_id,
                    "resource": resource,
                    "reserved": float(held["reserved"]),
                    "consumed": float(consumed),
                    "note": (
                        "the reservation under-estimated what this work consumed; it was "
                        "charged in full and the campaign cap still binds"
                    ),
                }
                envelope.setdefault("budget_overruns", []).append(overrun)
                with contextlib.suppress(Exception):
                    ledger.intervene("reservation_overrun", overrun)
    _write_envelope(logs, experiment_id, envelope)
    return envelope


def _write_envelope(log_dir: Path, experiment_id: str, envelope: Mapping[str, Any]) -> Path:
    """The whole dispatch record, beside the checkpoint, for replay and review."""
    path = Path(log_dir) / f"dispatch-{experiment_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(envelope), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


__all__ = [
    "ACCESS_BOUNDARY_NOTE",
    "ACTIONS",
    "ARGV_KEYS",
    "ARGV_PLACEHOLDERS",
    "ActionSpec",
    "CLAUDE_WORKER_CHARTER",
    "CLAUDE_WORKER_TOOLS",
    "COMMITTED_CAPABILITY_RECORD",
    "CREDENTIAL_NAME",
    "CandidateMaterializationError",
    "CandidateOriginError",
    "CandidatePatchError",
    "ENV_ALLOWLIST",
    "ENV_NEVER_INHERIT",
    "EXTRA_PROTECTED_PATHS",
    "MaterializedCandidate",
    "SUPPORTED_ACTIONS",
    "action_support",
    "candidate_env",
    "claude_cli_interface",
    "config_digests",
    "dispatch_plan",
    "execute_candidate",
    "independent_development_evaluation",
    "is_protected_path",
    "materialize",
    "probe_origin",
    "protected_paths",
    "redact_env",
    "run_claude_experiment",
    "run_declared_command",
    "run_in_candidate",
    "validate_plan",
    "verify_origin",
    "verify_produced_outputs",
    "weight_digests",
]
