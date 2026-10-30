"""Re-stamp the ``sha256`` front-matter field of every prompt file in this folder.

The loader in :mod:`drug_agent.llm.prompts` refuses a file whose body does not
match its recorded digest. That is the point: the recorded run history is keyed
on the exact prompt bytes, so an accidental reflow has to be loud. An *intended*
prompt change is re-stamped here, in its own visible step, so the digest change
shows up in the diff next to the text change that caused it.

    python system_prompts/stamp_checksums.py            # rewrite stale digests
    python system_prompts/stamp_checksums.py --check    # verify only; non-zero if stale

Kept dependency-free and importable from nowhere: it must run in a bare
interpreter, before the package is installed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def body_of(raw: str) -> tuple[str, str, str]:
    """Split a prompt file into (header, body, trailing) exactly as the loader does."""
    if not raw.startswith("---\n"):
        raise ValueError("prompt files must open with a '---' front-matter block")
    end = raw.find("\n---\n", 4)
    if end < 0:
        raise ValueError("front-matter block is not closed by a '---' line")
    header = raw[4:end]
    body = raw[end + 5 :]
    trailing = "\n" if body.endswith("\n") else ""
    return header, body[: len(body) - len(trailing)], trailing


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    stale: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if path.name in {"README.md", "README_zh.md"}:
            continue
        raw = path.read_text(encoding="utf-8")
        header, body, trailing = body_of(raw)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        lines = header.splitlines()
        rewritten = [f"sha256: {digest}" if ln.startswith("sha256:") else ln for ln in lines]
        if not any(ln.startswith("sha256:") for ln in lines):
            rewritten.append(f"sha256: {digest}")
        if rewritten == lines:
            continue
        stale.append(str(path.relative_to(ROOT)))
        if not check_only:
            path.write_text("\n".join(["---", *rewritten, "---", ""]) + body + trailing, encoding="utf-8")

    verb = "stale" if check_only else "re-stamped"
    print(f"{len(stale)} {verb}" + (": " + ", ".join(stale) if stale else ""))
    return 1 if (check_only and stale) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
