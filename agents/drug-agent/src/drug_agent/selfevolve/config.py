"""Configuration for the evolution loop. Safe, bounded defaults.

Two rules shaped every default here:

* **Nothing recursive runs unbounded.** ``max_cycles`` and ``max_candidates``
  are small, ``auto_promote`` is off, and the Claude budget is a hard count, not
  a suggestion. An evolver that can spend without limit is a billing incident
  waiting for a slow night.
* **Nothing reaches the network unless asked.** ``allow_internet`` is off. The
  research layer exists and is tested, but a self-modifying system that fetches
  and reads the internet by default is not a default anyone should inherit.

Everything is overridable by environment variable (``DRUGAGENT_EVOLVE_*``) or by
a YAML/JSON file, so a run is reproducible from a file rather than from a
remembered command line.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "drug_agent").is_dir():
            return parent
    return Path.cwd()


def _env(name: str, default: str) -> str:
    val = os.environ.get(f"DRUGAGENT_EVOLVE_{name}")
    return val if val else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[f"DRUGAGENT_EVOLVE_{name}"])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[f"DRUGAGENT_EVOLVE_{name}"])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"DRUGAGENT_EVOLVE_{name}")
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class EvolveConfig:
    # --- where things live ----------------------------------------------
    #: Journals of the user-serving system: the raw experience.
    journal_dir: str = field(default_factory=lambda: _env("JOURNAL_DIR", ""))
    #: Everything this subsystem writes: episodes, findings, candidates,
    #: evaluation records, archive, cycle logs.
    store_dir: str = field(default_factory=lambda: _env("STORE_DIR", ""))
    #: Where candidate workspaces are created (git worktrees or copies).
    workspace_dir: str = field(default_factory=lambda: _env("WORKSPACE_DIR", ""))

    # --- budget -----------------------------------------------------------
    max_cycles: int = field(default_factory=lambda: _env_int("MAX_CYCLES", 1))
    max_candidates: int = field(default_factory=lambda: _env_int("MAX_CANDIDATES", 3))
    #: Hard ceiling on Claude Code CLI invocations for one `run`/`step`.
    claude_call_budget: int = field(default_factory=lambda: _env_int("CLAUDE_BUDGET", 8))
    #: Hard ceiling on local-model calls made by the *controller* (not by the
    #: benchmark, which is the system under test and budgeted separately).
    ollama_call_budget: int = field(default_factory=lambda: _env_int("OLLAMA_BUDGET", 40))
    #: Wall-clock ceiling for one benchmark arm, seconds.
    benchmark_timeout_s: float = field(default_factory=lambda: _env_float("BENCH_TIMEOUT", 5400.0))

    # --- backends ---------------------------------------------------------
    controller_backend: str = field(default_factory=lambda: _env("CONTROLLER_BACKEND", "ollama"))
    controller_model: str = field(default_factory=lambda: _env("CONTROLLER_MODEL", ""))
    deep_backend: str = field(default_factory=lambda: _env("DEEP_BACKEND", "claude_cli"))
    deep_model: str = field(default_factory=lambda: _env("DEEP_MODEL", "opus"))
    critic_backend: str = field(default_factory=lambda: _env("CRITIC_BACKEND", "claude_cli"))
    critic_model: str = field(default_factory=lambda: _env("CRITIC_MODEL", "sonnet"))
    #: Fall back to the local model when the deep backend is unavailable or out
    #: of budget, instead of failing the cycle.
    escalation_fallback: bool = field(default_factory=lambda: _env_bool("ESCALATION_FALLBACK", True))

    # --- benchmark --------------------------------------------------------
    benchmark_id: str = field(default_factory=lambda: _env("BENCHMARK", "drugagent-master-v1"))
    #: Which split the loop is allowed to *study*. Held-out tasks are never
    #: shown to a candidate generator.
    discovery_split: str = "discovery"
    #: Repeats per task. The user-serving model is sampled, so >1 measures noise.
    repeats: int = field(default_factory=lambda: _env_int("REPEATS", 1))
    #: Cycle budget for one benchmark program. 1 keeps a full arm inside an hour.
    program_cycles: int = field(default_factory=lambda: _env_int("PROGRAM_CYCLES", 1))
    #: Tasks to run; empty means "every task in the selected splits".
    task_ids: list[str] = field(default_factory=list)

    # --- promotion --------------------------------------------------------
    #: Never true by default. A self-modifying system promoting itself into
    #: production without a human is the thing the brief asks us not to build.
    auto_promote: bool = field(default_factory=lambda: _env_bool("AUTO_PROMOTE", False))
    #: Minimum improvement in the primary metric to be worth considering.
    min_effect: float = field(default_factory=lambda: _env_float("MIN_EFFECT", 0.0))
    #: A candidate may not lose more than this on any guarded metric.
    max_regression: float = field(default_factory=lambda: _env_float("MAX_REGRESSION", 0.0001))
    #: Bootstrap resamples for the paired interval. Reported, never decisive at
    #: small n — the gate says so explicitly.
    bootstrap_resamples: int = field(default_factory=lambda: _env_int("BOOTSTRAP", 2000))

    # --- safety -----------------------------------------------------------
    allow_internet: bool = field(default_factory=lambda: _env_bool("ALLOW_INTERNET", False))
    #: Refuse to start if the working tree is dirty in files the loop may touch.
    require_clean_tree: bool = field(default_factory=lambda: _env_bool("REQUIRE_CLEAN_TREE", False))
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", False))

    # --- resolution -------------------------------------------------------
    def repo_root(self) -> Path:
        return _repo_root()

    def resolved_journal_dir(self) -> Path:
        if self.journal_dir:
            return Path(self.journal_dir).expanduser()
        from ..observability.journal import default_journal_root

        return default_journal_root()

    def resolved_store_dir(self) -> Path:
        if self.store_dir:
            return Path(self.store_dir).expanduser()
        return self.repo_root() / ".drugagent" / "evolve"

    def resolved_workspace_dir(self) -> Path:
        if self.workspace_dir:
            return Path(self.workspace_dir).expanduser()
        return self.repo_root() / ".drugagent" / "evolve" / "workspaces"

    def public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resolved"] = {
            "journal_dir": str(self.resolved_journal_dir()),
            "store_dir": str(self.resolved_store_dir()),
            "workspace_dir": str(self.resolved_workspace_dir()),
            "repo_root": str(self.repo_root()),
        }
        return d

    @classmethod
    def load(cls, path: str | Path | None = None) -> EvolveConfig:
        """Environment defaults, optionally overlaid by a JSON/YAML file."""
        cfg = cls()
        if not path:
            return cfg
        p = Path(path).expanduser()
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            return cfg
        data: Any
        if p.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            return cfg
        known = {f.name for f in fields(cls)}
        return cls(**{**asdict(cfg), **{k: v for k, v in data.items() if k in known}})


__all__ = ["EvolveConfig"]
