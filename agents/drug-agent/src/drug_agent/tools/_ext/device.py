"""Can *that* interpreter's torch really launch a kernel on *this* machine's GPU?

:mod:`drug_agent.tools._ext.probe` answers "is the package installed". This module
answers the question that comes after it and that an installed package cannot
answer for itself: **will it run here**.

**Why the second question exists.** This host is an NVIDIA RTX PRO 6000 Blackwell,
compute capability 12.0 (``sm_120``). Two of the environments on it —
``drug-agent-flowmol3`` (torch 2.6.0+cu124) and ``drug-agent-omtra`` (torch
2.4.0+cu121) — import every module their adapter's ``runtime_modules`` names,
report ``torch.cuda.is_available() == True``, report the right device capability,
and then die at the first real kernel launch with::

    CUDA error: no kernel image is available for execution on the device

Their torch builds carry SASS only up to ``sm_90`` and ship no PTX to JIT from, so
there is nothing on the card for the device to execute. Each has a sibling
environment — ``drug-agent-flowmol3-gpu`` (2.7.1+cu128) and ``drug-agent-omtra-gpu``
(2.8.0+cu128) — whose torch was built with native ``sm_120`` SASS and which runs.

An import probe cannot tell these apart, and the name cannot either: picking the
obviously-named environment is how a capability reports AVAILABLE and then fails at
sample time, which is the failure class already on this programme's record. So the
evidence has to be an executed kernel. :func:`kernel_report` launches one — a real
``torch.randn(64, 64, device="cuda") @ itself``, synchronised so an asynchronous
fault is surfaced here rather than three layers into someone's sampler.

**What keeps it affordable.** ``tool_pool()`` constructs every adapter on every call
and there is no adapter cache, so a 1.3-second subprocess per interpreter per call
would be intolerable. Three things bound it:

* **nothing runs when there is no GPU.** :func:`host_gpu_fingerprint` asks
  ``nvidia-smi`` once per process (about 20 ms) and returns ``None`` on a machine
  with no device. Every question then answers ``UNKNOWN`` without spawning
  anything, so a CPU-only host and CI pay one ``nvidia-smi`` call in total and no
  capability changes its status;
* **nothing runs at import time.** The first probe happens when an adapter is
  constructed for a spec that declares ``needs_gpu``, never when this module is
  imported;
* **a definite answer is remembered.** Results are cached in this process and, for
  ``OK``/``FAIL``, on disk under ``$XDG_CACHE_HOME/drugagent``. The cache key is the
  interpreter path, the size and mtime of its ``torch/version.py``, and the host GPU
  fingerprint, so re-installing torch or moving the card to another machine
  invalidates it by construction. Inconclusive answers — a timeout, an interpreter
  that will not start — are deliberately **not** cached: a transient failure must
  not become a permanent verdict.

``UNKNOWN`` is a third answer and never a synonym for either of the other two. It
never blocks a capability and never wins a selection; it only declines to decide.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Statuses :func:`kernel_report` can return. ``UNKNOWN`` means the question was
#: not answered, which is not the same as answering "no".
KERNEL_OK = "OK"
KERNEL_FAIL = "FAIL"
KERNEL_UNKNOWN = "UNKNOWN"

#: Seconds one probe subprocess gets. Importing torch and initialising a CUDA
#: context measured 1.3 s on this host; the margin is for a cold page cache.
PROBE_TIMEOUT_SECONDS = 120.0

#: Bumped when the probe program changes meaning, so old entries are ignored
#: rather than trusted.
_CACHE_VERSION = "v1"

#: The whole probe, as a program. It prints one JSON object and nothing else that
#: matters: warnings from torch go to stderr and are read back as diagnosis.
_PROBE_SOURCE = """
import json, sys
out = {"torch": None, "arch_list": None, "capability": None}
try:
    import torch
    out["torch"] = torch.__version__
    try:
        out["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        pass
    if not torch.cuda.is_available():
        out["status"] = "FAIL"
        out["error"] = "torch.cuda.is_available() is False"
    else:
        out["capability"] = list(torch.cuda.get_device_capability(0))
        a = torch.randn(64, 64, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        float(b.sum().item())
        out["status"] = "OK"
except Exception as exc:
    out["status"] = "FAIL"
    out["error"] = f"{type(exc).__name__}: {exc}"
print("<<<DRUGAGENT_GPU_PROBE>>>" + json.dumps(out))
"""

_MARKER = "<<<DRUGAGENT_GPU_PROBE>>>"

#: Per-process memo, keyed the same way as the on-disk cache.
_MEMO: dict[str, dict[str, Any]] = {}

#: ``""`` until asked. Afterwards the fingerprint string, or ``None`` for "asked,
#: and this machine has no GPU" — which is a cached answer, not an unasked question.
_HOST_FINGERPRINT: str | None = ""


def host_gpu_fingerprint() -> str | None:
    """``"<compute_cap>|<name>|<driver>"`` for GPU 0, or ``None`` if there is none.

    Asked at most once per process. It is part of the cache key because the same
    interpreter is a different answer on a different card: a torch that runs on an
    A100 is exactly the one that fails here.
    """
    global _HOST_FINGERPRINT
    if _HOST_FINGERPRINT != "":
        return _HOST_FINGERPRINT
    _HOST_FINGERPRINT = None
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "--query-gpu=compute_cap,name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    first = next((ln.strip() for ln in proc.stdout.splitlines() if ln.strip()), "")
    if not first:
        return None
    _HOST_FINGERPRINT = "|".join(part.strip() for part in first.split(","))
    return _HOST_FINGERPRINT


def _torch_stamp(python: str) -> str:
    """A cheap identity for the torch installed under *python*.

    ``torch/version.py`` carries the wheel's version and CUDA tag and is rewritten
    by every install, so its size and mtime change exactly when the answer could.
    """
    try:
        # abspath, never resolve(): an env's bin/python symlinks into the base
        # installation, and resolving it would probe the base env's torch.
        prefix = Path(os.path.abspath(python)).parent.parent  # noqa: PTH100 - see above
    except (OSError, ValueError):
        return "?"
    for glob in ("lib/python*/site-packages/torch/version.py", "Lib/site-packages/torch/version.py"):
        for path in prefix.glob(glob):
            try:
                st = path.stat()
            except OSError:
                continue
            return f"{st.st_size}:{st.st_mtime_ns}"
    return "no-torch"


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "drugagent" / f"gpu-kernel-probe-{_CACHE_VERSION}.json"


def _cache_read() -> dict[str, Any]:
    path = _cache_path()
    try:
        parsed = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cache_write(key: str, report: dict[str, Any]) -> None:
    """Merge one entry into the on-disk cache, atomically and never fatally.

    A cache that cannot be written is a slower probe, not a broken one, so every
    filesystem failure here is swallowed.
    """
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = _cache_read()
        table[key] = report
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".gpu-probe-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(table, handle)
            Path(tmp).replace(path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
    except OSError:
        return


def kernel_report(python: str) -> dict[str, Any]:
    """Launch one real CUDA kernel in *python* and report what happened.

    Returns a record with ``status`` (``OK`` / ``FAIL`` / ``UNKNOWN``), the torch
    version and compiled ``arch_list`` it found, the device capability it saw, and
    on failure the error. ``UNKNOWN`` carries a ``reason`` and is never cached.
    """
    fingerprint = host_gpu_fingerprint()
    if fingerprint is None:
        return {
            "status": KERNEL_UNKNOWN,
            "reason": "no NVIDIA GPU is visible to nvidia-smi, so there is nothing to launch on",
            "interpreter": python,
        }
    key = f"{_CACHE_VERSION}|{os.path.abspath(python)}|{_torch_stamp(python)}|{fingerprint}"  # noqa: PTH100
    memo = _MEMO.get(key)
    if memo is not None:
        return dict(memo)
    cached = _cache_read().get(key)
    if isinstance(cached, dict) and cached.get("status") in {KERNEL_OK, KERNEL_FAIL}:
        _MEMO[key] = cached
        return dict(cached)

    report: dict[str, Any] = {"interpreter": python, "host_gpu": fingerprint}
    try:
        proc = subprocess.run(  # fixed argv; the program is a module constant
            [python, "-c", _PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
        )
    except subprocess.TimeoutExpired:
        report.update(
            status=KERNEL_UNKNOWN,
            reason=f"the probe did not finish inside {PROBE_TIMEOUT_SECONDS:.0f}s",
        )
        return report
    except OSError as exc:
        report.update(status=KERNEL_UNKNOWN, reason=f"could not launch {python}: {exc}")
        return report

    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(_MARKER)), None)
    if line is None:
        report.update(
            status=KERNEL_UNKNOWN,
            reason=(
                f"the probe wrote no result (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()[-400:]}"
            ),
        )
        return report
    try:
        payload = json.loads(line[len(_MARKER) :])
    except json.JSONDecodeError as exc:  # pragma: no cover - the program emits json
        report.update(status=KERNEL_UNKNOWN, reason=f"unparsable probe result: {exc}")
        return report
    report.update(payload)
    report.setdefault("status", KERNEL_UNKNOWN)
    if report["status"] in {KERNEL_OK, KERNEL_FAIL}:
        _MEMO[key] = report
        _cache_write(key, report)
    return report


def can_launch_kernel(python: str) -> bool | None:
    """``True``/``False`` when the question was answered, ``None`` when it was not."""
    status = kernel_report(python)["status"]
    if status == KERNEL_OK:
        return True
    if status == KERNEL_FAIL:
        return False
    return None


def kernel_failure_detail(report: dict[str, Any]) -> str:
    """One line an operator can act on, built only from what was measured."""
    arches = report.get("arch_list")
    built = ", ".join(str(a) for a in arches) if arches else "an unreported set of architectures"
    capability = report.get("capability")
    device = f"sm_{capability[0]}{capability[1]}" if isinstance(capability, list) and capability else "?"
    return (
        f"torch {report.get('torch') or '?'} in this interpreter is built for {built}, and this "
        f"machine's GPU is {device} ({report.get('host_gpu') or '?'}): a real kernel launch fails "
        f"with {report.get('error') or 'an unreported error'}"
    )


def current_interpreter() -> str:
    """The running interpreter, named once so callers do not import ``sys``."""
    return sys.executable


__all__ = [
    "KERNEL_FAIL",
    "KERNEL_OK",
    "KERNEL_UNKNOWN",
    "PROBE_TIMEOUT_SECONDS",
    "can_launch_kernel",
    "current_interpreter",
    "host_gpu_fingerprint",
    "kernel_failure_detail",
    "kernel_report",
]
