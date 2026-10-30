"""Fixed local scientific entry points, bounded subprocesses and a shared GPU lock."""
import fcntl
import json
import sys
import traceback
from pathlib import Path
from .config import atomic_json, data_root
from .contracts import Unavailable


def main():
    name, input_file, output_file = sys.argv[1:]
    args = json.loads(Path(input_file).read_text())
    work = Path(input_file).parent
    try:
        with (data_root()/".gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if name.startswith("generation."):
                from .generation import generate
                data = generate(name.split(".")[1], args, work)
            elif name == "structure.esmfold2":
                from .folding import fold
                data = fold(args, work)
            elif name == "planning.qwen":
                from .llm import plan
                data = plan(args["instruction"])
            elif name == 'evaluation.structure':
                from .tool_evaluation import structure_evaluation
                data = structure_evaluation(args)
            elif name == 'evaluation.thermompnn':
                from .stability import thermompnn
                data = thermompnn(args, work)
            else:
                raise ValueError("Unknown scientific worker operation")
        result = {"tool": name, "status": "ok", "data": data, "provenance": data.get("provenance", {})}
    except Exception as exc:
        traceback.print_exc()
        result = {"tool": name, "status": "unavailable" if isinstance(exc, Unavailable) else "error", "error": f"{type(exc).__name__}: {exc}"}
    atomic_json(output_file, result)


if __name__ == "__main__":
    main()
