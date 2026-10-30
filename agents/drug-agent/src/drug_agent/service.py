"""High-level service helpers shared by the CLI and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .agents import build_registry
from .contracts.brief import ResearchBrief
from .orchestration.engine import Engine
from .settings.config import Settings, load_settings


def load_brief(path: str | Path) -> ResearchBrief:
    """Load and validate a brief from YAML/JSON. Raises ValueError on invalid."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("brief must be a mapping")
    return ResearchBrief.model_validate(raw)


def build_engine(
    settings: Settings | None = None, *, profile: str | None = None, deterministic: bool | None = None
) -> Engine:
    settings = settings or load_settings(profile=profile, deterministic=deterministic)
    settings = settings.resolved()
    registry = build_registry(settings)
    return Engine(settings, registry)


def run_brief(path: str | Path, *, profile: str = "local", deterministic: bool = True) -> dict[str, Any]:
    """Convenience: load a brief and execute it to completion (or first gate)."""
    settings = load_settings(profile=profile, deterministic=deterministic).resolved()
    engine = build_engine(settings)
    engine.history_source = "service"
    brief = load_brief(path)
    run = engine.create_run(brief, deterministic=deterministic, profile=profile)
    final = engine.run(run.id)
    return {"run_id": final.id, "state": str(final.state), "engine": engine}
