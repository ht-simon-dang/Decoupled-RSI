"""Export JSON Schema for every contract to ``schemas/``.

Run: ``python -m drug_agent.cli.export_schemas [out_dir]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..contracts import CONTRACT_REGISTRY


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    index = {}
    for name, model in CONTRACT_REGISTRY.items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(model.json_schema(), indent=2, sort_keys=True))
        written.append(path)
        index[name] = f"{name}.schema.json"
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return written


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("schemas")
    written = export(out)
    print(f"exported {len(written)} schemas to {out}")


if __name__ == "__main__":
    main()
