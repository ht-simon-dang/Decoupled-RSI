#!/usr/bin/env python3
"""Distribute canonical WebView assets into independently installable Agents."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "agents/drug-agent/src/drug_agent/api/static",
    ROOT / "agents/drug-agent/src/drug_agent/llm/web",
    ROOT / "agents/peptide-agent/src/peptide_agent/web",
    ROOT / "agents/mRNA-agent/src/mrna_agent/web",
    ROOT / "agents/Enzyme-Agent/src/enzyme_agent/web",
    ROOT / "agents/antibody-agent/src/antibody_agent/web",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for directory in TARGETS:
        for name in ("shared.css", "shared.js"):
            source = (ROOT / "web" / name).read_bytes()
            target = directory / name
            if not target.exists() or target.read_bytes() != source:
                if args.check:
                    stale.append(str(target.relative_to(ROOT)))
                else:
                    target.write_bytes(source)
    if stale:
        parser.exit(1, "Run python scripts/sync_web_theme.py:\n" + "\n".join(stale) + "\n")
    print("Shared WebView assets are synchronized.")


if __name__ == "__main__":
    main()
