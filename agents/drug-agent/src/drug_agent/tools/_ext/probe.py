"""Can *that* interpreter import *those* modules — without importing or spawning it.

This is the cheapest honest answer to "is the backend installed", and it runs far
more often than it looks: on every ``drugagent tools``, every ``drugagent doctor``,
every ``GET /api/pools``, every report render, and once per adapter construction —
and :func:`drug_agent.tools.registry.tool_pool` constructs every adapter on every
call, with no cache. With one external tool that was a rounding error. With thirty
it is the difference between a listing that returns in milliseconds and one that
takes seconds, so the two obvious implementations are both ruled out:

``import <mod>``
    Costs about a second for torch, permanently pollutes ``sys.modules``, and can
    abort the process outright when a CUDA extension is half-built — the exact
    state this function exists to detect.

``subprocess.run([python, "-c", "import ..."])``
    Costs a process launch per module per adapter per call, and inherits the same
    crash risk in the child.

So: :func:`importlib.util.find_spec` for the running interpreter, and a
``site-packages`` glob for any other one.

**The symlink trap.** An environment's ``bin/python`` is normally a symlink into
the base installation. :meth:`pathlib.Path.resolve` follows it, walks *out* of the
environment, probes the base environment's ``site-packages``, and reports every
conda environment as identical to its base. ``os.path.abspath`` is therefore
deliberate, not an oversight, and carries a ``# noqa: PTH100`` wherever ruff's
pathlib rules would otherwise "modernise" it into the bug.

**What the glob cannot see.** ``lib/python*/site-packages/<mod>/__init__.py`` finds
package directories only. It misses single-file modules (``six.py``), PEP-420
namespace packages, and bare C-extension modules
(``<mod>.cpython-312-x86_64-linux-gnu.so`` on this host). :func:`missing_modules`
checks all three shapes, because across thirty backends the single-file and
extension-module cases stop being hypothetical: ``lmdb`` and ``spyrmsd`` are
extension modules, and probing only for a directory would report them absent on a
machine where they are installed and working.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

#: Where a foreign interpreter's third-party packages live, relative to its prefix.
_SITE_PACKAGES_GLOBS: tuple[str, ...] = (
    "lib/python*/site-packages",  # POSIX
    "Lib/site-packages",  # Windows
)


def is_current_interpreter(python: str) -> bool:
    """Whether *python* is the interpreter this process is running under."""
    try:
        # abspath, never resolve(): see the module docstring's symlink trap.
        candidate = Path(os.path.abspath(python))  # noqa: PTH100 - resolve() leaves the env
        running = Path(os.path.abspath(sys.executable))  # noqa: PTH100 - same reason
    except (OSError, ValueError):
        return False
    return candidate == running


def _found_in_site_packages(prefix: Path, mod: str) -> bool:
    """Whether *mod* is installed under *prefix*, in any of the three shapes."""
    for sp_glob in _SITE_PACKAGES_GLOBS:
        for site in prefix.glob(sp_glob):
            if (site / mod / "__init__.py").is_file():
                return True  # ordinary package
            if (site / f"{mod}.py").is_file():
                return True  # single-file module
            if any(site.glob(f"{mod}.*.so")) or any(site.glob(f"{mod}.pyd")):
                return True  # compiled extension module
            if (site / mod).is_dir():
                return True  # PEP-420 namespace package
    return False


def missing_modules(python: str, modules: tuple[str, ...]) -> list[str]:
    """Which of *modules* ``python`` cannot import, in declaration order.

    Never imports and never spawns. For the running interpreter this is
    :func:`importlib.util.find_spec`; for any other one it is a filesystem probe
    of that interpreter's ``site-packages``.
    """
    if not modules:
        return []
    if is_current_interpreter(python):
        absent: list[str] = []
        for mod in modules:
            try:
                found = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                # A module whose parent package raises on import is not importable
                # here either, which is exactly what the caller needs to know.
                found = False
            if not found:
                absent.append(mod)
        return absent
    try:
        candidate = Path(os.path.abspath(python))  # noqa: PTH100 - resolve() leaves the env
    except (OSError, ValueError):
        return list(modules)
    prefix = candidate.parent.parent
    return [mod for mod in modules if not _found_in_site_packages(prefix, mod)]


def can_import(python: str, modules: tuple[str, ...]) -> bool:
    """Whether *python* can import every one of *modules*."""
    return not missing_modules(python, modules)


def conda_env_pythons(names: tuple[str, ...]) -> tuple[str, ...]:
    """Interpreters to try, in order, for the named conda environments.

    The process that launched the console is often *not* the environment a backend
    was installed into — the base environment's ``python``, first on PATH, is the
    usual culprit — and that one imports nothing. Looking under the active conda
    installation for the environments a spec names closes the loop without asking
    an operator to export an interpreter path in every shell.
    """
    candidates: list[str] = []
    roots: list[Path] = []

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        active = Path(prefix)
        candidates += [str(active / "bin" / "python"), str(active / "python.exe")]
        # <root>/envs/<name> for a named env; <root> itself for the base env.
        roots.append(active.parent.parent if active.parent.name == "envs" else active)
    conda_exe = os.environ.get("CONDA_EXE")  # <root>/bin/conda
    if conda_exe:
        roots.append(Path(conda_exe).parent.parent)

    for root in roots:
        for name in names:
            env = root / "envs" / name
            candidates += [str(env / "bin" / "python"), str(env / "python.exe")]

    seen: set[str] = set()
    unique: list[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return tuple(unique)
