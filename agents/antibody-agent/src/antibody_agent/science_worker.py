"""Private worker protocol: tool, JSON input path, JSON output path."""

import json
import sys
from pathlib import Path

from .config import atomic_json
from .contracts import ToolResult, Unavailable


def main():
    tool, source, destination = sys.argv[1:]
    request = json.loads(Path(source).read_text())
    try:
        if tool == "planning.qwen":
            from .llm import plan

            data = plan(request["arguments"]["instruction"])
        else:
            from .adapters import NATIVE

            if tool not in NATIVE:
                raise Unavailable("No native adapter; configure an external JSON bridge")
            data = NATIVE[tool](request["arguments"], request["runtime"], Path(source).parent)
        result = ToolResult(tool=tool, status="ok", data=data)
    except (ImportError, Unavailable) as exc:
        result = ToolResult(tool=tool, status="unavailable", error=str(exc))
    except Exception as exc:
        result = ToolResult(tool=tool, status="error", error=f"{type(exc).__name__}: {exc}")
    atomic_json(Path(destination), result.model_dump())


if __name__ == "__main__":
    main()
