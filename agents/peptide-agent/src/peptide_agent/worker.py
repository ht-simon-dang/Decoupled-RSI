"""Fixed generation worker; its process is terminated on wall-clock timeout."""
import sys

from .contracts import Generate, ToolResult, Unavailable
from .models import PepMLM


def main():
    try:
        request = Generate.model_validate_json(sys.stdin.read(20000))
        result = PepMLM().generate(request)
    except Unavailable as exc:
        result = ToolResult(tool="generation.pepmlm", status="unavailable", trust="UNAVAILABLE", warnings=[str(exc)])
    except Exception as exc:
        result = ToolResult(tool="generation.pepmlm", status="error", trust="UNAVAILABLE",
                            warnings=[f"{type(exc).__name__}: {exc}"])
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
