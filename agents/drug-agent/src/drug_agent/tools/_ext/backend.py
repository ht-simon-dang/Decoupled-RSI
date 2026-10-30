"""Find one external tool, judge whether it can really run, and launch it.

This generalises the mechanism :mod:`drug_agent.tools.pfm_tool` proved on one
model to the whole external catalogue. The shape is unchanged, because the shape
was right: a checkout under ``.external/``, an interpreter that may not be this
one, a driver script that runs *inside* the tool's environment, and JSON carried
over files rather than pipes.

**Why files, not pipes.** A driver prints progress, a CUDA fault prints a stack
trace, and a DataLoader worker prints warnings — all onto the same stdout a
payload would have to share. Writing ``request.json`` into a scratch directory and
reading ``result.json`` back keeps the structured channel clean and leaves
stdout/stderr free to be exactly what they are useful for: the last twenty lines
of a diagnostic when something fails.

**Why the result file is read before the exit code is judged.** PFM's driver
writes a structured failure into ``result.json`` *and* exits non-zero. The
original adapter checked the exit code first, so every driver-side failure threw
away the machine-readable reason in favour of a stderr tail. Here the result file
wins when it exists and parses: a driver that explains itself is believed.

Four things this adds over the PFM original, each because thirty tools make a
tolerable one-off into a systematic hazard:

``environment allowlist``
    ``env = dict(os.environ)`` hands the entire parent environment — API keys, DB
    URLs, cloud credentials — to third-party code cloned from GitHub. Acceptable
    once, for one audited model. :meth:`ExternalBackend.child_env` passes only the
    variables a subprocess legitimately needs, plus anything the spec names.

``process-group kill``
    ``subprocess.run(timeout=...)`` kills the direct child and nothing else, so a
    timed-out run that had forked CUDA workers leaves orphans holding GPU memory.
    The child gets its own session and the whole group is signalled.

``bounded output``
    ``capture_output=True`` buffers stdout and stderr in memory for the full
    timeout. A chatty driver with a progress bar can balloon RSS with output that
    is then truncated to twenty lines anyway, so the capture is spilled to files
    in the scratch directory and only the tail is read back.

``a timed subprocess``
    The ledger could not previously distinguish "the external model was slow" from
    "our adapter was slow". :meth:`run_json` returns the subprocess wall time so
    the adapter can meter the thing that actually took the time.

The exception type is a parameter, not a constant. ``tools/base.py`` prescribes
:class:`~drug_agent.tools.base.ToolUnavailable`, raised only through
``require_available()``; PFM's caller instead catches a domain ``GenerationError``
and degrades the run rather than killing the task. Both are live contracts in this
repository and both are pinned by tests, so this helper refuses to choose.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .device import KERNEL_FAIL, can_launch_kernel, kernel_failure_detail, kernel_report
from .probe import can_import, conda_env_pythons, missing_modules
from .spec import ExternalSpec

#: Environment variables a driver subprocess legitimately needs. Everything else
#: in the parent environment is withheld — see the module docstring.
_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "CONDA_PREFIX",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "TORCH_HOME",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONUNBUFFERED",
)

#: How many lines of a failed driver's stderr go into the raised error.
_ERROR_TAIL_LINES = 20

#: How many bytes of a driver's captured output are read back at all. The capture
#: is spilled to a file, so this bounds memory, not the driver.
_ERROR_TAIL_BYTES = 64_000


class ExternalToolError(RuntimeError):
    """The selected external backend ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`, which means the
    backend was never there. This one means it was there, it started, and it did
    not produce a usable answer — a crash, a timeout, an unparseable result.
    """


@dataclass
class DriverRun:
    """What one driver invocation produced, including how long the child really took."""

    payload: dict[str, Any]
    seconds: float
    returncode: int


class ExternalBackend:
    """Resolution, readiness and launch for one :class:`ExternalSpec`.

    Readiness is computed **once**, in ``__init__``, and cached — ``available()``
    and ``profile`` both read that cache, and ``tool_pool()`` builds a fresh
    adapter on every call. Construction therefore does filesystem work only: a
    handful of ``is_file`` calls and, for a foreign interpreter, one glob per
    module. It never imports the backend, never spawns a probe, and never touches
    the network. Call :meth:`refresh` in a long-lived process to notice an install
    that finished after startup.
    """

    def __init__(self, spec: ExternalSpec, settings: Any = None) -> None:
        self.spec = spec
        self._settings = settings
        self.root = self._resolve_root()
        self.python = self._resolve_python()
        self.device = self._resolve("device", f"DRUGAGENT_{spec.env_prefix}_DEVICE") or (
            "cuda:0" if spec.needs_gpu else "cpu"
        )
        self.driver = self._resolve_driver()
        self._missing = self._probe()

    # --- configuration -----------------------------------------------------
    def _resolve(self, suffix: str, env: str) -> str | None:
        """Where one configurable value comes from, in one place.

        Precedence, highest first:

        1. ``settings.external_tools["<prefix>"]["<suffix>"]`` — the profile YAML,
           which is how thirty tools are configured without ninety flat fields;
        2. a flat ``settings.<prefix>_<suffix>`` attribute — how ``pfm_*`` is
           configured, kept working because its spellings are documented;
        3. ``DRUGAGENT_<PREFIX>_<SUFFIX>`` in the environment — the escape hatch for
           an operator who cannot edit a profile;
        4. nothing, and the caller falls back to discovery.

        Adapters never read ``os.environ`` themselves. This is the only place
        precedence is decided, so it cannot differ between tools.
        """
        key = self.spec.env_prefix.lower()
        table = getattr(self._settings, "external_tools", None) if self._settings is not None else None
        if isinstance(table, dict):
            entry = table.get(key)
            if isinstance(entry, dict) and entry.get(suffix):
                return str(entry[suffix])
        attr = f"{key}_{suffix}"
        val = getattr(self._settings, attr, None) if self._settings is not None else None
        if val:
            return str(val)
        raw = os.environ.get(env)
        return raw or None

    def _repo_root(self) -> Path | None:
        """The checkout this package is installed from, if it is a source tree."""
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
                return parent
        return None

    def _resolve_root(self) -> Path | None:
        """Where the tool's checkout is, or ``None`` if there is not a real one."""
        explicit = self._resolve("root", self.spec.root_env_var)
        if explicit:
            cand = Path(explicit).expanduser()
            return cand.resolve() if cand.is_dir() else None
        if not self.spec.checkout_relpath:
            return None
        here = Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / self.spec.checkout_relpath
            if self._is_real_checkout(cand):
                return cand.resolve()
        return None

    def _is_real_checkout(self, cand: Path) -> bool:
        """A directory is only the tool if it holds the files the run path imports."""
        if not cand.is_dir():
            return False
        if not self.spec.sentinel_files:
            return True
        return all((cand / rel).is_file() for rel in self.spec.sentinel_files)

    def _resolve_python(self) -> str:
        """The interpreter the driver runs in.

        Selection asks two questions in this order, and the order is the whole
        point.

        **Which environment is this backend installed into?** ``selection_modules``
        — the one import that identifies it. Whether that install ran to
        *completion* is readiness' question, and keeping the two apart means a
        half-built environment is still chosen over one with nothing at all, then
        reported honestly module by module.

        **Of the environments that qualify, which can actually run on this
        machine's GPU?** Only asked for a spec that declares ``needs_gpu``, and
        only of candidates that already passed the first question. On this host
        ``drug-agent-flowmol3`` and ``drug-agent-omtra`` import every module their
        adapters name and then fail at the first kernel launch, because their torch
        builds stop at ``sm_90`` and the card is ``sm_120``; their ``-gpu`` siblings
        run. Both siblings pass an import probe, so an import probe would pick
        whichever came first in the list — a capability that reports AVAILABLE and
        dies at sample time. :func:`~drug_agent.tools._ext.device.can_launch_kernel`
        launches a real kernel instead, so the broken one loses on evidence rather
        than on name-ordering luck.

        An interpreter the device question could not answer (no GPU on the machine,
        a probe that would not start) ranks between the two: it never beats a
        measured pass and never loses to a measured failure. Nothing is dropped —
        if every candidate fails the device check the first is still selected, so
        :meth:`_probe` can report *why* rather than silently falling back to an
        interpreter that has none of the packages.
        """
        explicit = self._resolve("python", self.spec.python_env_var)
        if explicit:
            return explicit
        if not self.spec.selection_modules:
            return sys.executable
        candidates = [sys.executable]
        candidates += [c for c in conda_env_pythons(self.spec.conda_envs) if Path(c).is_file()]
        seen: set[str] = set()
        usable: list[str] = []
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if can_import(cand, self.spec.selection_modules):
                usable.append(cand)
        if not usable:
            return sys.executable
        if not self.spec.needs_gpu or len(usable) == 1:
            return usable[0]
        rank = {True: 0, None: 1, False: 2}
        # Stable: candidates the device check cannot separate keep declaration order.
        return sorted(usable, key=lambda cand: rank[can_launch_kernel(cand)])[0]

    def _resolve_driver(self) -> Path | None:
        """Locate this tool's driver under ``scripts/external/``."""
        if not self.spec.driver_name:
            return None
        here = Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / "scripts" / "external" / self.spec.driver_name
            if cand.is_file():
                return cand
        return None

    # --- readiness ---------------------------------------------------------
    def _probe(self) -> list[str]:
        """Every reason this backend cannot run here. Empty means it can.

        Each string names the broken thing *and* what fixes it, because these
        strings are the operator's only diagnostic — in the CLI table, in the
        capability profile, and in the exception raised at call time.
        """
        spec = self.spec
        problems: list[str] = list(spec.platform_blockers)

        if spec.binary and shutil.which(spec.binary) is None:
            problems.append(f"no {spec.binary!r} executable on PATH; {spec.install_hint or 'install it'}")

        if spec.checkout_relpath:
            if self.root is None:
                # Distinguish "you never configured this" from "what you configured
                # is not there". Telling an operator who set the variable that it is
                # unset sends them to re-clone a checkout they already have, when the
                # actual fault is one wrong path.
                configured = self._resolve("root", spec.root_env_var)
                if configured:
                    expected = ", ".join(spec.sentinel_files) or "its source tree"
                    problems.append(
                        f"{spec.root_env_var} points at {configured}, which is not a "
                        f"{spec.name} checkout (expected {expected} inside it)"
                    )
                else:
                    problems.append(
                        f"no {spec.name} checkout at {spec.checkout_relpath} and "
                        f"{spec.root_env_var} is unset; run `{spec.install_hint}`"
                    )
            else:
                for rel in spec.sentinel_files:
                    if not (self.root / rel).is_file():
                        problems.append(f"{self.root / rel} missing (not a {spec.name} checkout?)")
                for rel in spec.weight_relpaths:
                    if not self._weight_present(self.root / rel):
                        problems.append(
                            f"{spec.name} weights not found at {self.root / rel}; "
                            f"{spec.install_hint or 'fetch the checkpoint'}"
                        )

        if spec.driver_name and self.driver is None:
            problems.append(f"driver script scripts/external/{spec.driver_name} not found")

        if spec.runtime_modules:
            interp = self.python
            if not (interp and (Path(interp).is_file() or shutil.which(interp))):
                problems.append(f"{spec.name} interpreter {interp!r} is not executable")
            elif absent := missing_modules(interp, spec.runtime_modules):
                problems.append(
                    f"{spec.name} interpreter {interp} cannot import {', '.join(absent)} — run "
                    f"`{spec.install_hint}`, or set {spec.python_env_var} to an interpreter that has them"
                )

        # A GPU tool whose interpreter cannot launch a kernel here is UNAVAILABLE,
        # and saying so is the point: every module imports, the card is visible,
        # `torch.cuda.is_available()` is True, and the run dies at the first real
        # launch. Reported only on a *measured* failure — an unanswered question
        # (no GPU on the machine, a probe that would not start) changes nothing,
        # so a CPU-only host and CI see exactly the statuses they saw before.
        if spec.needs_gpu and spec.runtime_modules and not problems:
            report = kernel_report(self.python)
            if report.get("status") == KERNEL_FAIL:
                problems.append(
                    f"{spec.name} interpreter {self.python} cannot execute on this GPU: "
                    f"{kernel_failure_detail(report)}. Install a torch built for this device into "
                    f"that environment, or set {spec.python_env_var} to an interpreter that has one"
                )
        return problems

    @staticmethod
    def _weight_present(path: Path) -> bool:
        """Whether a declared weight artifact is really there.

        Some specs name a file and some name a directory — a benchmark dataset, an
        unpacked model bundle. ``exists()`` answers both and is wrong for the second:
        an empty directory left behind by an interrupted download satisfies it, the
        card reports AVAILABLE, and the tool fails at call time having claimed it
        could run. A directory therefore has to have something in it.
        """
        if path.is_dir():
            return any(path.iterdir())
        return path.is_file()

    def refresh(self) -> None:
        """Re-run discovery and readiness. For a server that outlived an install."""
        self.root = self._resolve_root()
        self.python = self._resolve_python()
        self.driver = self._resolve_driver()
        self._missing = self._probe()

    @property
    def missing(self) -> list[str]:
        """Cached readiness failures, in the order they were detected."""
        return list(self._missing)

    def ready(self) -> bool:
        """Whether the real backend can run on this machine."""
        return not self._missing

    def unavailable_reason(self, produces: str) -> str:
        """Why this capability is UNAVAILABLE — the whole reason, not a summary.

        *produces* completes the sentence "the platform ...", e.g. ``"generates
        nothing"``. The string has to stand on its own: it is what an operator
        sees in ``drugagent tools``, in the capability profile and in the error
        raised at call time, and there is no surrogate quietly covering for it.
        """
        detail = "; ".join(self._missing) if self._missing else "unknown"
        hint = f" Install it with `{self.spec.install_hint}`." if self.spec.install_hint else ""
        return (
            f"{self.spec.name} is not usable here, so this capability is UNAVAILABLE and "
            f"{produces} — there is no surrogate backend.{hint} Reason: {detail}."
        )

    def runtime_requirements(self) -> dict[str, Any]:
        """The card's machine-fixing checklist, filled from what was actually probed."""
        return {
            "gpu": self.spec.needs_gpu,
            "binary": self.spec.binary,
            "interpreter": self.python,
            "packages": list(self.spec.runtime_modules),
            "checkout": str(self.root) if self.root else None,
            "env_var": self.spec.root_env_var,
            "upstream": self.spec.upstream_url,
            "upstream_ref": self.spec.upstream_ref,
        }

    # --- launch ------------------------------------------------------------
    def child_env(self, *, seed: int | None = None, extra_pythonpath: tuple[str, ...] = ()) -> dict[str, str]:
        """The environment the driver runs in: an allowlist, not a copy.

        Third-party code cloned from GitHub does not need this process's API keys,
        database URLs or cloud credentials, and thirty such tools make "it is only
        one audited model" stop being true.
        """
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        parts = [str(self.root)] if self.root else []
        parts += [str(Path(self.root) / p) for p in extra_pythonpath if self.root]
        if parts:
            env["PYTHONPATH"] = os.pathsep.join(parts)
        if seed is not None:
            env.setdefault("PYTHONHASHSEED", str(seed))
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def run_json(
        self,
        request: dict[str, Any],
        *,
        timeout_s: float,
        error_cls: type[Exception] = ExternalToolError,
        extra_pythonpath: tuple[str, ...] = (),
        seed: int | None = None,
    ) -> DriverRun:
        """Run this tool's driver on *request* and return the parsed result.

        The driver receives one argument, the path of a ``request.json`` carrying
        *request* plus an ``out_dir`` the helper injects, and is expected to write
        ``<out_dir>/result.json``. Failures — a non-zero exit with no result, a
        timeout, an unparseable payload — are raised as *error_cls* with a message
        that names the budget or carries the last lines of stderr.
        """
        if self.driver is None:
            raise error_cls(f"{self.spec.name} driver script is not installed")
        workdir = Path(tempfile.mkdtemp(prefix=f"drugagent_{self.spec.env_prefix.lower()}_"))
        try:
            out_dir = workdir / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(request)
            payload["out_dir"] = str(out_dir)
            # `root` is injected for the same reason as `out_dir`: it is context the
            # helper knows and the driver cannot derive. A driver runs with the
            # checkout as its cwd, so most reach upstream files relatively — but one
            # that needs an absolute path to a config or a weight has no other way to
            # get it, and four of them read `request["root"]` expecting exactly this.
            # None of those four would ever have received it, which stayed invisible
            # only because they had no backend to run against.
            if self.root is not None:
                payload.setdefault("root", str(self.root))
            req_path = workdir / "request.json"
            req_path.write_text(json.dumps(payload, indent=2))
            result_path = out_dir / "result.json"

            stdout_path = workdir / "stdout.log"
            stderr_path = workdir / "stderr.log"
            started = time.perf_counter()
            # Spill the capture to files rather than buffering it in memory for the
            # whole timeout: a driver with a progress bar can produce megabytes that
            # would only ever be truncated to twenty lines.
            with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
                try:
                    proc = subprocess.Popen(
                        [self.python, str(self.driver), str(req_path)],
                        cwd=str(self.root) if self.root else None,
                        env=self.child_env(seed=seed, extra_pythonpath=extra_pythonpath),
                        stdout=out_f,
                        stderr=err_f,
                        # Its own process group, so a timeout can reap the CUDA
                        # workers and DataLoader children too, not just the driver.
                        start_new_session=True,
                    )
                except OSError as exc:
                    raise error_cls(f"could not launch {self.spec.name} driver: {exc}") from exc
                try:
                    returncode = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired as exc:
                    self._kill_group(proc)
                    raise error_cls(
                        f"{self.spec.name} exceeded its {timeout_s:.0f}s budget and was terminated. "
                        f"Lower the batch size, or raise DRUGAGENT_{self.spec.env_prefix}_TIMEOUT_SECONDS."
                    ) from exc
            seconds = time.perf_counter() - started

            # The result file wins over the exit code when it exists and parses: a
            # driver that wrote a structured failure has explained itself better
            # than a stderr tail ever will.
            if result_path.is_file():
                try:
                    parsed = json.loads(result_path.read_text())
                except (json.JSONDecodeError, OSError) as exc:
                    raise error_cls(
                        f"{self.spec.name} wrote an unreadable result.json (exit {returncode}): {exc}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise error_cls(
                        f"{self.spec.name} result.json is a {type(parsed).__name__}, expected an object"
                    )
                if parsed.get("ok") is False or (returncode != 0 and not parsed.get("ok")):
                    detail = parsed.get("error") or self._tail(stderr_path, stdout_path)
                    raise error_cls(f"{self.spec.name} failed (exit {returncode}): {detail}")
                return DriverRun(payload=parsed, seconds=seconds, returncode=returncode)

            raise error_cls(
                f"{self.spec.name} produced no result.json (exit {returncode}):\n"
                + self._tail(stderr_path, stdout_path)
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _kill_group(proc: subprocess.Popen[bytes]) -> None:
        """Signal the whole process group, then the child, ignoring races.

        The group is the point: a timed-out model that had forked CUDA workers or
        a DataLoader pool leaves them holding GPU memory if only the direct child
        is signalled. Every failure here is a race that has already resolved in the
        direction we wanted — the process is gone — so it is suppressed rather than
        reported.
        """
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                with contextlib.suppress(OSError):
                    proc.kill()
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _tail(*paths: Path) -> str:
        """The last lines of the first capture file that has anything in it."""
        for path in paths:
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                with path.open("rb") as fh:
                    fh.seek(max(0, path.stat().st_size - _ERROR_TAIL_BYTES))
                    text = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            if lines:
                return "\n".join(lines[-_ERROR_TAIL_LINES:])
        return "(the driver produced no output)"
