#!/usr/bin/env python
"""Print the most recently generated demo run's Markdown report."""

from __future__ import annotations

from pathlib import Path

from drug_agent.settings.config import load_settings


def main() -> int:
    runs_dir = Path(load_settings("local").resolved().runs_dir)
    if not runs_dir.exists():
        print("no runs found; run `make demo` first")
        return 1
    runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        report = run / "report.md"
        if report.exists():
            print(report.read_text())
            return 0
    print("no report found; run `make demo` first")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
