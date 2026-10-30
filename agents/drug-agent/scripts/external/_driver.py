"""The driver side of the external-tool boundary.

Every script in this directory runs **inside somebody else's environment** — a
conda env pinned to torch 2.2 and numpy 1.26, a checkout whose ``requirements.txt``
wants Python 3.8. It therefore imports nothing from ``drug_agent`` and nothing
outside the standard library. That is the whole point of the boundary: the
platform can talk to a backend it could never import.

The contract, in both directions:

*In* — argv[1] is the path of a ``request.json`` object. It always carries
``out_dir``; everything else is the calling adapter's business.

*Out* — ``<out_dir>/result.json``, an object with ``ok``. On success ``ok`` is
true and the rest of the object is the payload the adapter parses. On failure
``ok`` is false and ``error`` says what went wrong, in a sentence an operator can
act on.

Writing the failure into the result file rather than only onto stderr is what lets
:meth:`drug_agent.tools._ext.backend.ExternalBackend.run_json` believe a driver
that explains itself: it reads the result before it judges the exit code. Exiting
non-zero as well is still correct — it is what a shell, a supervisor and a CI log
will notice.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def read_request(argv: list[str]) -> dict[str, Any]:
    """Parse the request object named by ``argv[1]``, or exit 2 explaining why."""
    if len(argv) < 2:
        sys.stderr.write(f"usage: {Path(argv[0]).name} <request.json>\n")
        raise SystemExit(2)
    path = Path(argv[1])
    try:
        request = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"cannot read request {path}: {exc}\n")
        raise SystemExit(2) from exc
    if not isinstance(request, dict) or "out_dir" not in request:
        sys.stderr.write(f"request {path} is not an object carrying out_dir\n")
        raise SystemExit(2)
    return request


def write_result(request: dict[str, Any], payload: dict[str, Any]) -> None:
    """Write a success payload and exit 0."""
    out_dir = Path(request["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {"ok": True}
    body.update(payload)
    (out_dir / "result.json").write_text(json.dumps(body, indent=2, default=str))
    raise SystemExit(0)


def fail(request: dict[str, Any], message: str, *, exc: BaseException | None = None) -> None:
    """Write a structured failure and exit 2.

    The traceback goes to stderr, where the adapter's tail will find it if the
    message alone is not enough; the message goes into the result file, where the
    adapter reads it in preference to any stderr tail.
    """
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    try:
        out_dir = Path(request["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({"ok": False, "error": message}, indent=2))
    except (OSError, KeyError, TypeError):
        sys.stderr.write(message + "\n")
    raise SystemExit(2)


def main(run: Any) -> None:
    """Run *run(request) -> dict* under the contract, turning any escape into a failure."""
    request = read_request(sys.argv)
    try:
        payload = run(request)
    except SystemExit:
        raise
    except BaseException as exc:
        fail(request, f"{type(exc).__name__}: {exc}", exc=exc)
    else:
        write_result(request, payload)
