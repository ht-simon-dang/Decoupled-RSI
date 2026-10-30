"""Path/argument validation, output limiting, and external-binary sandboxing."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

_SHELL_METACHARACTERS = re.compile(r"[;&|`$><\n\r\\!*?(){}\[\]]")


class PathTraversalError(ValueError):
    """Raised when a path escapes its permitted root."""


class ArgumentValidationError(ValueError):
    """Raised when a tool argument fails validation (e.g. shell metacharacters)."""


class OutputTooLargeError(RuntimeError):
    """Raised when tool output exceeds the configured limit."""


def safe_join(root: Path | str, *parts: str) -> Path:
    """Join under ``root`` and guarantee the result stays within it.

    Rejects absolute components and ``..`` traversal. Returns a resolved path.
    """
    root_p = Path(root).resolve()
    candidate = root_p
    for part in parts:
        if part is None:
            raise ArgumentValidationError("path component is None")
        p = Path(part)
        if p.is_absolute():
            raise PathTraversalError(f"absolute path component not allowed: {part!r}")
        candidate = candidate / p
    resolved = candidate.resolve()
    if root_p != resolved and root_p not in resolved.parents:
        raise PathTraversalError(f"path {resolved} escapes root {root_p}")
    return resolved


def validate_tool_args(args: list[str]) -> list[str]:
    """Validate CLI arguments destined for an external binary.

    Ensures each argument is a plain token with no shell metacharacters. The
    caller must invoke via ``subprocess`` with ``shell=False``.
    """
    validated: list[str] = []
    for a in args:
        if not isinstance(a, str):
            raise ArgumentValidationError(f"non-string argument: {a!r}")
        if _SHELL_METACHARACTERS.search(a):
            raise ArgumentValidationError(f"shell metacharacter in argument: {a!r}")
        if "\x00" in a:
            raise ArgumentValidationError("null byte in argument")
        validated.append(a)
    return validated


def limit_output(data: str | bytes, max_bytes: int) -> str | bytes:
    """Raise if output exceeds ``max_bytes``; otherwise return it unchanged."""
    size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
    if size > max_bytes:
        raise OutputTooLargeError(f"tool output {size} bytes exceeds limit {max_bytes}")
    return data


def run_sandboxed(
    binary: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 60.0,
    max_output_bytes: int = 5_000_000,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external binary with validated args, no shell, timeout and limits.

    This is the single approved path for invoking native tools (Vina, OpenMM
    helpers, etc.). ``shell=False`` is enforced and arguments are validated.
    """
    validate_tool_args([binary, *args])
    proc = subprocess.run(
        [binary, *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=env,
        check=False,
    )
    limit_output(proc.stdout or "", max_output_bytes)
    return proc


def shlex_safe(text: str) -> str:
    return shlex.quote(text)
