"""Isolated candidate projects inside flat repositories or monorepo worktrees."""
from __future__ import annotations

import posixpath
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import EvolveConfig
from ..repository import RepositoryLayout
from ..schemas import Candidate


class WorkspaceError(RuntimeError):
    pass


def _git(*args: str, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(cwd), check=False, timeout=120
    )
    if check and proc.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def repo_state(root: Path) -> dict[str, Any]:
    layout = RepositoryLayout.discover(root)
    head = _git("rev-parse", "HEAD", cwd=root)
    return {
        "is_git": layout.git_root is not None,
        "has_commits": layout.git_root is not None and head.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "git_root": str(layout.git_root) if layout.git_root is not None else None,
        "project_prefix": layout.project_prefix,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).stdout.strip(),
        "dirty_files": [
            line[3:] for line in _git("status", "--porcelain", cwd=root).stdout.splitlines() if line
        ],
    }


@dataclass
class CandidateWorkspace:
    """path is always the task project; checkout_root owns its Git worktree."""
    candidate_id: str
    path: Path
    mode: str
    parent_commit: str | None = None
    created_at: float = 0.0
    notes: list[str] = field(default_factory=list)
    _repo_root: Path | None = None
    checkout_root: Path | None = None
    project_prefix: str = ""
    parent_project_prefix: str = ""

    def apply_patch(self, patch: str) -> dict[str, Any]:
        if not patch.strip():
            return {"applied": True, "note": "no patch to apply", "exit_status": 0}
        patch_file = self.path / ".selfevolve-candidate.patch"
        checkout = self.checkout_root or self.path
        try:
            patch_file.write_text(patch, encoding="utf-8")
        except OSError as exc:
            return {"applied": False, "error": str(exc), "exit_status": 1}
        # Git parses filenames (including quoted names and binary patches).
        parsed = _git("apply", "--numstat", "-z", str(patch_file), cwd=checkout)
        if parsed.returncode:
            return {"applied": False, "exit_status": parsed.returncode, "stderr": parsed.stderr[-2000:]}
        paths = []
        for record in parsed.stdout.split("\0"):
            if record:
                columns = record.split("\t", 2)
                if len(columns) != 3 or not columns[2]:
                    return {"applied": False, "exit_status": 1, "error": "unrecognized patch path"}
                paths.append(columns[2])
        if not paths:
            return {"applied": False, "exit_status": 1, "error": "patch changes no files"}
        prefix = self.parent_project_prefix
        prefixed = [bool(prefix and p.startswith(prefix + "/")) for p in paths]
        if any(prefixed) and not all(prefixed):
            return {"applied": False, "exit_status": 1, "error": "mixed project-relative and Git-relative patch paths"}
        for raw in paths:
            value = raw[len(prefix) + 1:] if all(prefixed) else raw
            norm = posixpath.normpath(value.replace("\\", "/"))
            if (norm.startswith("/") or norm == ".." or norm.startswith("../")
                    or (prefix and not all(prefixed) and norm.startswith("agents/"))):
                return {"applied": False, "exit_status": 1, "error": f"patch path outside task project: {raw}"}
        args = ["apply", "--verbose"]
        if all(prefixed) and not self.project_prefix:
            # A filtered copy has its own flat Git root.
            args.append(f"-p{1 + len(Path(prefix).parts)}")
        elif not all(prefixed) and self.project_prefix:
            args.append(f"--directory={self.project_prefix}")
        proc = _git(*args, str(patch_file), cwd=checkout)
        return {
            "applied": proc.returncode == 0,
            "exit_status": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "patch_bytes": len(patch),
        }

    def diff(self) -> str:
        return _git("diff", "HEAD", "--", cwd=self.checkout_root or self.path).stdout

    def changed_files(self) -> list[str]:
        layout = RepositoryLayout.discover(self.path)
        result = []
        for name in layout.changed_paths():
            result.append(layout.project_path(name))
        return result

    def cleanup(self, *, keep: bool = False, force: bool = False) -> dict[str, Any]:
        checkout = self.checkout_root or self.path
        if keep:
            return {"removed": False, "reason": "kept by request", "path": str(checkout)}
        if self.mode == "worktree" and self._repo_root is not None:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            proc = _git(*args, str(checkout), cwd=self._repo_root)
            reason = proc.stderr.strip()[:400]
            return {
                "removed": proc.returncode == 0, "mode": "worktree", "forced": force,
                "reason": reason, "note": reason, "path": str(checkout),
            }
        try:
            shutil.rmtree(checkout)
            return {"removed": True, "mode": "copy"}
        except OSError as exc:
            return {"removed": False, "mode": "copy", "reason": str(exc)}


def workspace_for(
    candidate: Candidate,
    cfg: EvolveConfig | None = None,
    *,
    root: Path | None = None,
    workspace_dir: Path | None = None,
) -> CandidateWorkspace:
    """Keep task-project coordinates while creating a complete detached worktree."""
    cfg = cfg or EvolveConfig()
    root = (Path(root) if root is not None else cfg.repo_root()).resolve()
    base = (Path(workspace_dir) if workspace_dir is not None else cfg.resolved_workspace_dir()).resolve()
    base.mkdir(parents=True, exist_ok=True)
    name = f"{candidate.candidate_id}_{uuid.uuid4().hex[:6]}"
    state = repo_state(root)
    prefix = state["project_prefix"]
    if state["is_git"] and state["has_commits"]:
        checkout = base / name
        git_root = Path(state["git_root"])
        proc = _git("worktree", "add", "--detach", str(checkout), state["head"], cwd=git_root)
        if proc.returncode == 0:
            project = checkout / prefix if prefix else checkout
            if not project.is_dir():
                _git("worktree", "remove", "--force", str(checkout), cwd=git_root)
                raise WorkspaceError(f"committed project {prefix!r} is absent from the candidate checkout")
            return CandidateWorkspace(
                candidate_id=candidate.candidate_id, path=project, checkout_root=checkout,
                mode="worktree", parent_commit=state["head"], created_at=time.time(),
                notes=[f"detached worktree at {state['head'][:12]}; project prefix={prefix or '.'}"],
                _repo_root=git_root, project_prefix=prefix, parent_project_prefix=prefix,
            )
        note = f"worktree unavailable ({proc.stderr.strip()[:200]}); using a project copy"
    else:
        note = "no committed parent; using an isolated project copy"
    path = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=str(base)))
    try:
        _copy_tree(root, path)
        # A candidate copy must never discover or stage files in an enclosing
        # repository. Its private snapshot also makes worker diffs auditable.
        _git("init", "-q", cwd=path, check=True)
        _git("add", "--force", "--all", cwd=path, check=True)
        _git("-c", "user.name=Candidate snapshot", "-c", "user.email=candidate@localhost",
             "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
             "commit", "--allow-empty", "-qm", "Isolated candidate baseline", cwd=path, check=True)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return CandidateWorkspace(
        candidate_id=candidate.candidate_id, path=path, checkout_root=path, mode="copy",
        parent_commit=state.get("head"), created_at=time.time(), notes=[note],
        _repo_root=path, parent_project_prefix=prefix,
    )


_SKIP = {
    ".git", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__",
    ".external", ".drugagent", ".research", "node_modules", "share-ready",
    "self-evolve-master", "running-history", "runs", "logs", ".env", ".env.local",
}


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in _SKIP or item.name.endswith(".zip"):
            continue
        target = dst / item.name
        if item.is_dir():
            # Exclude the destination when callers choose a workspace inside src.
            if dst.is_relative_to(item.resolve()):
                continue
            shutil.copytree(item, target, ignore=shutil.ignore_patterns(*_SKIP, "*.pyc"), dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


__all__ = ["CandidateWorkspace", "WorkspaceError", "repo_state", "workspace_for"]
