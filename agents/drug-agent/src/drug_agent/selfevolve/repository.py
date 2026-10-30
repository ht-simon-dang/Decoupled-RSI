"""Project and Git checkout coordinates for flat and monorepo installations."""
from __future__ import annotations

import os
import posixpath
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositoryLayout:
    project_root: Path
    git_root: Path | None
    project_prefix: str

    @classmethod
    def discover(cls, root: Path | str) -> "RepositoryLayout":
        project = Path(root).resolve()
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=project,
                capture_output=True, text=True, check=False, timeout=30,
            )
        except OSError:
            return cls(project, None, "")
        if proc.returncode:
            return cls(project, None, "")
        git_root = Path(proc.stdout.strip()).resolve()
        try:
            prefix = project.relative_to(git_root).as_posix()
        except ValueError:
            return cls(project, None, "")
        return cls(project, git_root, "" if prefix == "." else prefix)

    def project_path(self, git_path: str) -> str:
        """Convert a Git-root path, refusing sibling projects and traversal."""
        value = posixpath.normpath(str(git_path).replace("\\", "/"))
        if value.startswith("/") or value == ".." or value.startswith("../"):
            raise ValueError(f"path escapes the project: {git_path}")
        if self.project_prefix:
            prefix = self.project_prefix + "/"
            if not value.startswith(prefix):
                raise ValueError(f"path is outside {self.project_prefix}: {git_path}")
            value = value[len(prefix):]
        return value

    def changed_paths(self) -> list[str]:
        """Git-root paths including deletions, staged changes and ignored additions."""
        if self.git_root is None:
            return []
        result: set[str] = set()
        for argv in (
            ["diff", "--name-only", "-z", "--no-renames", "HEAD", "--"],
            ["ls-files", "--others", "-z"],
        ):
            proc = subprocess.run(
                ["git", *argv], cwd=self.git_root, capture_output=True,
                text=True, check=False, timeout=30,
            )
            if proc.returncode:
                raise RuntimeError(f"cannot inspect candidate changes: {proc.stderr.strip()}")
            result.update(item for item in proc.stdout.split("\0") if item)
        return sorted(result)


def project_path_candidates(raw: str, root: Path | str) -> set[str]:
    """Normalize the accepted project-relative, Git-relative and absolute spellings.

    Protection checks intentionally also inspect a path with leading parent
    components removed, preserving the conservative handling of plan strings.
    """
    text = str(raw).strip().replace("\\", "/")
    if not text:
        return set()
    project = Path(root).resolve()
    layout = RepositoryLayout.discover(project)
    norm = posixpath.normpath(text)
    candidates = {norm}
    if layout.project_prefix:
        prefix = layout.project_prefix + "/"
        if norm.startswith(prefix):
            candidates.add(norm[len(prefix):])
    resolved = Path(os.path.normpath(text)) if Path(text).is_absolute() else project / norm
    try:
        candidates.add(resolved.resolve().relative_to(project).as_posix())
    except ValueError:
        pass
    for value in tuple(candidates):
        while value.startswith("../"):
            value = value[3:]
            candidates.add(value)
        if layout.project_prefix and value.startswith(layout.project_prefix + "/"):
            candidates.add(value[len(layout.project_prefix) + 1:])
    return candidates
