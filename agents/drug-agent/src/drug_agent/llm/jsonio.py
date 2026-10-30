"""Decoding JSON out of a local model's reply.

A hosted frontier model returns clean JSON. A locally-quantized one returns
JSON *somewhere inside* a reply that may also contain a reasoning monologue, a
restatement of the schema it was shown, prose either side of the fence, and —
most often on this stack — an object that simply stops mid-string because the
token budget ran out while the model was still thinking.

Everything here exists because one of those was observed against a real model,
and each is handled explicitly rather than by one broad ``try: json.loads``:

  * ``<think>``/``<thinking>`` blocks leak into the answer channel on some
    Ollama builds; their prose contains braces, so a naive brace scan finds the
    model's musings instead of its answer.
  * Models echo the example object from the prompt *before* the real one, so
    "take the first JSON block" picks the schema, not the answer. Candidates are
    therefore ranked, and a caller-supplied validator gets the deciding vote.
  * Trailing commas, ``//`` comments, single quotes, ``True/False/None`` and
    unquoted ``NaN`` all appear in practice and all are recoverable losslessly.
  * Truncation is *not* losslessly recoverable, so it is handled last, kept
    behind an explicit flag, and always reported — a silently auto-closed object
    is a fabricated one. The caller's real fix is to retry with a bigger budget,
    which is why :func:`decode` reports truncation as its own outcome.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

#: Reasoning wrappers that some builds emit into the answer channel.
_THINK_TAGS = ("think", "thinking", "reasoning", "reflection")

_THINK_RE = re.compile(
    r"<(" + "|".join(_THINK_TAGS) + r")\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_RE = re.compile(
    r"<(" + "|".join(_THINK_TAGS) + r")\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_LINE_COMMENT_RE = re.compile(r"(^|[\s,{\[])(//|#)[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", " ": " "})


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks from an answer-channel string.

    An *unclosed* opening tag is also dropped to the end of the string: that is
    what a reply truncated mid-thought looks like, and everything after it is
    monologue, never answer.
    """
    if not text:
        return ""
    cleaned = _THINK_RE.sub(" ", text)
    cleaned = _OPEN_THINK_RE.sub(" ", cleaned)
    return cleaned


def _balanced_spans(text: str) -> Iterator[str]:
    """Yield every top-level ``{...}`` span, quote- and escape-aware."""
    depth = 0
    in_str = False
    escape = False
    begin = -1
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}" and depth > 0:  # a stray closer outside any object is noise
            depth -= 1
            if depth == 0 and begin != -1:
                yield text[begin : i + 1]
                begin = -1


def _fenced_blocks(text: str) -> Iterator[str]:
    """Yield the body of every ``` fenced block, opening-language tag removed."""
    parts = text.split("```")
    # parts[1], parts[3], ... are inside fences.
    for body in parts[1::2]:
        stripped = body.lstrip()
        # Drop an opening info string such as "json" / "JSON5" on the first line.
        first_nl = stripped.find("\n")
        if first_nl != -1 and stripped[:first_nl].strip().isalnum():
            stripped = stripped[first_nl + 1 :]
        yield stripped.strip()


def iter_candidates(text: str) -> Iterator[str]:
    """Yield plausible JSON payloads, most-likely-to-be-the-answer first.

    Fenced blocks come first because the prompt asks for one; within each source
    the *last* occurrence is yielded first, since a model that restates the
    schema does so before it answers.
    """
    seen: set[str] = set()

    def _emit(chunk: str) -> Iterator[str]:
        chunk = chunk.strip()
        if not chunk or chunk in seen:
            return
        seen.add(chunk)
        yield chunk

    fenced = [b for b in _fenced_blocks(text) if b.lstrip().startswith("{")]
    for block in reversed(fenced):
        yield from _emit(block)
    spans = list(_balanced_spans(text))
    for span in reversed(spans):
        yield from _emit(span)
    # A reply that is bare JSON with no fence and no closing brace still deserves
    # a shot at the truncation repair path.
    brace = text.find("{")
    if brace != -1:
        yield from _emit(text[brace:])


def _relax(payload: str) -> str:
    """Apply the lossless syntax relaxations, in increasing order of nerve."""
    out = payload.translate(_SMART_QUOTES)
    out = _BLOCK_COMMENT_RE.sub(" ", out)
    out = _LINE_COMMENT_RE.sub(r"\1", out)
    out = _TRAILING_COMMA_RE.sub(r"\1", out)
    return out


def close_truncated(payload: str) -> str | None:
    """Close a payload that ends mid-object, or ``None`` if it cannot be.

    This *invents* structure, so it is only ever reached when the caller has
    explicitly opted in and is told the result was repaired. The last, partial
    key/value pair is discarded rather than guessed at.
    """
    depth_stack: list[str] = []
    in_str = False
    escape = False
    last_safe = -1  # index just after the last completed element at depth>=1
    for i, ch in enumerate(payload):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth_stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if depth_stack and depth_stack[-1] == ch:
                depth_stack.pop()
                last_safe = i + 1
            else:
                return None
        elif ch == "," and depth_stack:
            last_safe = i  # a comma at any depth ends a complete element
    if not depth_stack:
        return None  # nothing to close — this payload was not truncated
    if last_safe <= 0:
        return None
    head = payload[:last_safe].rstrip().rstrip(",")
    return head + "".join(reversed(depth_stack))


def loads_lenient(payload: str) -> tuple[Any, str]:
    """Parse ``payload``, returning ``(object, how)``. Raises ``ValueError``.

    ``how`` names the most invasive step that was needed, so callers can log
    "this model needed comment stripping" rather than only "it worked".
    """
    try:
        return json.loads(payload), "strict"
    except json.JSONDecodeError:
        pass
    relaxed = _relax(payload)
    if relaxed != payload:
        try:
            return json.loads(relaxed), "relaxed"
        except json.JSONDecodeError:
            pass
    # Python literals: True/False/None, single-quoted strings, tuples.
    try:
        obj = ast.literal_eval(relaxed)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError(f"not parseable as JSON or a Python literal: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"parsed to {type(obj).__name__}, not an object")
    return obj, "python-literal"


@dataclass
class Decoded:
    """Outcome of decoding one model reply."""

    obj: dict[str, Any] | None = None
    #: Which path produced ``obj``: strict / relaxed / python-literal / truncated.
    how: str = ""
    #: True when structure had to be invented to close a truncated payload.
    repaired: bool = False
    #: True when the reply looks cut off — the caller should retry with a
    #: bigger budget rather than ask the model to "fix" it.
    truncated: bool = False
    error: str | None = None
    #: One line per rejected candidate, for the journal.
    attempts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.obj is not None


def decode(
    text: str,
    *,
    validate: Callable[[dict[str, Any]], bool] | None = None,
    allow_truncated: bool = False,
) -> Decoded:
    """Extract one JSON object from a model reply.

    ``validate`` (typically "does this satisfy the target schema?") breaks ties
    between candidates. Without it the first candidate that parses wins.
    """
    result = Decoded()
    body = strip_reasoning(text)
    if not body.strip():
        result.error = "empty reply (only reasoning, no answer channel)"
        result.truncated = "<" in text  # an unclosed think tag ate the whole reply
        return result

    fallback: Decoded | None = None
    truncated_seen = False
    for candidate in iter_candidates(body):
        try:
            obj, how = loads_lenient(candidate)
        except ValueError as exc:
            if close_truncated(candidate) is not None:
                truncated_seen = True
            result.attempts.append(f"{str(exc)[:160]}")
            continue
        if not isinstance(obj, dict):
            result.attempts.append(f"candidate parsed to {type(obj).__name__}, not an object")
            continue
        if validate is None or validate(obj):
            result.obj, result.how = obj, how
            return result
        result.attempts.append("parsed but failed schema validation")
        if fallback is None:
            fallback = Decoded(obj=obj, how=how)

    if allow_truncated:
        for candidate in iter_candidates(body):
            closed = close_truncated(candidate)
            if closed is None:
                continue
            try:
                obj, _ = loads_lenient(closed)
            except ValueError:
                continue
            if isinstance(obj, dict) and (validate is None or validate(obj)):
                result.obj, result.how = obj, "truncated"
                result.repaired = True
                result.truncated = True
                return result

    if fallback is not None:
        # Nothing validated; hand back the best-parsed object so the caller can
        # show the model its own error instead of "no JSON found".
        fallback.error = "no candidate satisfied the schema"
        fallback.attempts = result.attempts
        return fallback
    result.truncated = truncated_seen
    result.error = "reply was cut off mid-object" if truncated_seen else "no JSON object found in the reply"
    return result
