"""The generation harness — the reliability layer around a local model.

One structured reasoning step goes through here, and this is where a 27B
quantized model becomes usable as an orchestrator. Responsibilities:

  * stream the model, forwarding the reasoning channel to the UI live;
  * ask for the answer in the strongest form the model supports — a
    schema-constrained decode when that is enabled, a fenced JSON block
    otherwise;
  * decode the reply (see :mod:`drug_agent.llm.jsonio`) and validate it;
  * respond to the *specific* failure rather than retrying blindly:

      truncation   the model ran out of budget mid-answer. Retrying the same
                   prompt cannot help, and "repair" is worse — it asks the model
                   to fix text it never finished. The budget is doubled and the
                   reasoning channel is dropped so the whole of it goes to the
                   answer.
      schema drift the reply parsed but the shape is wrong (a field called
                   ``key`` where the contract says ``property``). The model is
                   shown its own output, the exact validation error, and the
                   field names it must use.
      no JSON      the reply had no object in it at all. Same repair path, with
                   the output contract restated.

  * record every exchange — prompts, reasoning, raw reply, decode path, timing,
    token counts — to the request journal, so a failure is diagnosable after
    the fact rather than only while it is on screen.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..observability.journal import ChatExchange, RunJournal
from . import jsonio
from .config import Sampling
from .events import CONTENT, THINKING, RunSession
from .ollama import OllamaClient, OllamaError

T = TypeVar("T", bound=BaseModel)

#: Ceiling for the truncation retry, so a runaway model cannot spin forever.
_MAX_BUDGET = 24000
#: How many times a budget may be doubled before the reply is treated as hopeless.
_MAX_BUDGET_RETRIES = 2


def extract_json(text: str) -> str | None:
    """Backwards-compatible shim: the payload string, or ``None``.

    Prefer :func:`drug_agent.llm.jsonio.decode`, which reports *how* the payload
    was recovered and whether it was truncated — information this signature
    cannot carry.
    """
    import json as _json

    decoded = jsonio.decode(text)
    return _json.dumps(decoded.obj) if decoded.ok else None


class StructuredResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    thinking: str
    raw_content: str
    meta: dict
    #: How the object was recovered: strict / relaxed / python-literal / truncated.
    decode: str = "strict"
    #: True when structure had to be invented to close a truncated reply.
    repaired: bool = False
    #: Number of extra model calls this step needed.
    attempts: int = 1


def _validator(schema: type[T]):
    def _ok(obj: dict[str, Any]) -> bool:
        try:
            schema.model_validate(obj)
        except ValidationError:
            return False
        return True

    return _ok


def _required_fields(schema: type[BaseModel]) -> str:
    """A short reminder of the exact key names, for the repair prompt.

    Schema drift is almost always a *synonym*: the model writes ``key`` for
    ``property``, or ``operator`` for ``op``. Naming the fields fixes far more
    of those than restating the whole contract does.
    """
    names = list(schema.model_fields)
    return ", ".join(f"`{n}`" for n in names[:24])


def _use_schema_decoding(mode: str, phase: str) -> bool:
    """Whether to constrain decoding to the schema for this phase.

    ``auto`` enables it where it removes a *measured* failure: the phases whose
    output is a large nested object, where the model has been observed inventing
    field names. Planning and critique produce flatter objects, decode cleanly
    as fenced JSON, and are left unconstrained so the model writes freely.
    """
    normalized = (mode or "auto").strip().lower()
    if normalized in ("on", "1", "true", "always"):
        return True
    if normalized in ("off", "0", "false", "never"):
        return False
    return phase in ("transcribe", "refine")


async def _one_call(
    client: OllamaClient,
    model: str,
    *,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    think: bool,
    schema_format: dict[str, Any] | None,
    session: RunSession,
    phase: str,
    cycle: int,
) -> tuple[str, str, dict[str, Any]]:
    """Stream one completion; return (thinking, content, meta)."""
    thinking_buf: list[str] = []
    content_buf: list[str] = []
    meta: dict[str, Any] = {}
    async for chunk in client.chat_stream(
        model=model, messages=messages, think=think, options=options, format=schema_format
    ):
        if chunk.thinking:
            thinking_buf.append(chunk.thinking)
            await session.emit(THINKING, phase=phase, cycle=cycle, delta=chunk.thinking)
        if chunk.content:
            content_buf.append(chunk.content)
            await session.emit(CONTENT, phase=phase, cycle=cycle, delta=chunk.content)
        if chunk.done and chunk.meta:
            meta = {
                "eval_count": chunk.meta.get("eval_count"),
                "prompt_eval_count": chunk.meta.get("prompt_eval_count"),
                "done_reason": chunk.meta.get("done_reason"),
            }
    return "".join(thinking_buf), "".join(content_buf), meta


async def run_structured(
    client: OllamaClient,
    model: str,
    *,
    system: str,
    user: str,
    preset: Sampling,
    schema: type[T],
    session: RunSession,
    phase: str,
    cycle: int = 0,
    max_repairs: int = 2,
    think: bool = True,
    structured: str = "auto",
    journal: RunJournal | None = None,
) -> tuple[T, StructuredResult]:
    """Run one structured reasoning step. Streams thinking to ``session``.

    Returns the validated model instance plus the raw streaming result. Raises
    ``OllamaError`` if the model is unreachable, or ``ValueError`` if the output
    could not be coerced into ``schema`` even after repairs.
    """
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    messages = list(base_messages)
    schema_format = schema.model_json_schema() if _use_schema_decoding(structured, phase) else None
    validate = _validator(schema)

    budget = preset.num_predict
    use_think = think
    last_error = ""
    last_result: StructuredResult | None = None
    call_index = 0
    repairs_used = 0
    budget_retries = 0

    while True:
        # A repair carries the model's own bad output back to it; a budget retry
        # re-issues the original prompt. Distinguishing them matters here (a
        # repair wants a colder temperature, a budget retry does not) and in the
        # journal, where mislabelling the two would hide why a phase was slow.
        is_repair = len(messages) > len(base_messages)
        options = preset.with_budget(budget).options()
        if is_repair:
            options["temperature"] = 0.1  # repairs want obedience, not ideas

        started = time.perf_counter()
        thinking, raw, meta = await _one_call(
            client,
            model,
            messages=messages,
            options=options,
            think=use_think,
            schema_format=schema_format,
            session=session,
            phase=phase,
            cycle=cycle,
        )
        elapsed = time.perf_counter() - started
        hit_budget = meta.get("done_reason") == "length"

        # A reply that ran out of budget is incomplete, not malformed. Accepting
        # a closed-up version of it would silently drop whatever the model was
        # still writing, so that is only ever the very last resort.
        decoded = jsonio.decode(raw, validate=validate, allow_truncated=False)
        validated: T | None = None
        error: str | None = decoded.error
        if decoded.ok:
            try:
                validated = schema.model_validate(decoded.obj)
                error = None
            except ValidationError as exc:
                error = f"ValidationError: {exc}"

        mode = "schema" if schema_format else "fenced"
        if is_repair:
            mode += "+repair"
        elif budget_retries:
            mode += "+budget"
        if journal is not None:
            journal.record_chat(
                ChatExchange(
                    phase=phase,
                    cycle=cycle,
                    attempt=call_index,
                    model=model,
                    mode=mode,
                    messages=list(messages),
                    thinking=thinking,
                    content=raw,
                    parsed=decoded.obj,
                    ok=validated is not None,
                    decode=decoded.how,
                    error=error,
                    elapsed_s=elapsed,
                    options=options,
                    eval_count=meta.get("eval_count"),
                    prompt_eval_count=meta.get("prompt_eval_count"),
                    done_reason=meta.get("done_reason"),
                    repairs=list(decoded.attempts),
                )
            )

        if validated is not None:
            return validated, StructuredResult(
                thinking=thinking,
                raw_content=raw,
                meta=meta,
                decode=decoded.how,
                repaired=decoded.repaired,
                attempts=call_index + 1,
            )

        last_error = error or "unknown decoding failure"
        last_result = StructuredResult(
            thinking=thinking, raw_content=raw, meta=meta, decode=decoded.how, attempts=call_index + 1
        )
        call_index += 1

        # --- choose the response that fits the failure --------------------
        if (
            (hit_budget or decoded.truncated)
            and budget < _MAX_BUDGET
            and budget_retries < _MAX_BUDGET_RETRIES
        ):
            budget_retries += 1
            previous, budget = budget, min(_MAX_BUDGET, budget * 2)
            use_think = False  # give the whole budget to the answer
            messages = list(base_messages)
            await session.emit(
                "log",
                phase=phase,
                cycle=cycle,
                title="reply was cut off — retrying with a larger budget",
                detail=f"num_predict {previous} → {budget}, reasoning channel off",
            )
            continue

        if repairs_used < max_repairs:
            repairs_used += 1
            await session.emit(
                "log",
                phase=phase,
                cycle=cycle,
                title="self-repairing malformed output",
                detail=last_error[:300],
            )
            messages = [
                *base_messages,
                {"role": "assistant", "content": raw[:4000] or "(empty)"},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply could not be used. Error:\n"
                        f"{last_error[:1500]}\n\n"
                        f"The object's top-level keys must be exactly: {_required_fields(schema)}.\n"
                        "Use those names verbatim — do not rename or substitute them.\n"
                        "Return ONLY a single, valid JSON object inside a ```json code fence. "
                        "No commentary before or after. Fix the exact problem above."
                    ),
                },
            ]
            continue

        break

    # --- last resort: accept a closed-up truncated object, loudly ----------
    if last_result is not None:
        salvaged = jsonio.decode(last_result.raw_content, validate=validate, allow_truncated=True)
        if salvaged.ok:
            try:
                validated = schema.model_validate(salvaged.obj)
            except ValidationError:
                validated = None
            if validated is not None:
                await session.emit(
                    "log",
                    phase=phase,
                    cycle=cycle,
                    title="accepted a truncated reply after closing it",
                    detail="The model never finished this object; the incomplete tail was dropped.",
                )
                if journal is not None:
                    journal.record_chat(
                        ChatExchange(
                            phase=phase,
                            cycle=cycle,
                            attempt=call_index,
                            model=model,
                            mode="salvage",
                            messages=[],
                            content=last_result.raw_content,
                            parsed=salvaged.obj,
                            ok=True,
                            decode="truncated",
                            error="accepted after closing a truncated object",
                        )
                    )
                last_result.decode = "truncated"
                last_result.repaired = True
                return validated, last_result

    raise ValueError(
        f"structured generation failed after {call_index} model calls "
        f"({repairs_used} repairs, {budget_retries} budget retries): {last_error}"
    )


async def ensure_model_available(client: OllamaClient, model: str) -> None:
    """Fail fast (with a friendly message) if Ollama or the model is missing."""
    health = await client.health()
    if not health["ok"]:
        raise OllamaError(f"Ollama is not reachable: {health['error']}.  Start it with `ollama serve`.")
    names = health["models"]
    if model not in names and f"{model}:latest" not in names:
        raise OllamaError(
            f"model '{model}' is not installed in Ollama. Available: {', '.join(names) or '(none)'}. "
            f"Pull it with:  ollama pull {model}"
        )
