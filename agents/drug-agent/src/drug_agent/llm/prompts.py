"""Prompt engineering for the master agent.

Every reasoning step has a purpose-built system prompt. Common invariants are
factored into ``shared/master_rules.md`` and reused so the model behaves
consistently across planning, transcription, critique, and refinement. The
design principles:

  * **Role priming** — the model is a senior computational medicinal chemist and
    research-program lead, not a chatbot.
  * **Output contract** — exactly one fenced ```json block, shape given inline
    (we never use grammar-constrained decoding; it breaks this quantized model).
  * **Epistemic guardrail** — the LLM plans, transcribes, critiques, and decides;
    it must NEVER invent scientific numbers (potency, QED, SA, safety). Those
    come only from the tool-backed specialists.
  * **Vocabulary grounding** — transcription is constrained to the exact backend
    keys (objective names, dotted property keys, operators) so the output maps
    onto a ``ResearchBrief`` without guesswork.
  * **Adaptive refinement** — on failure the model must diagnose the cause and
    change specific generation levers, and explain each change (the ablation).
  * **Selection is ranking, not measurement** — the post-loop step that picks one
    candidate across all cycles compares tool-computed numbers and explains the
    trade-off; it is given no field in which to write a number of its own.

The prompt *text* no longer lives here. It lives in the ``system_prompts/``
library at the repository root — one Markdown file per phase and role, plus the
shared fragments every phase splices in — because a prompt is the thing this
platform is actually evolving, and a reviewer should be able to read and diff it
without reading Python. See ``system_prompts/README.md`` for the map. This module
keeps the *composition*: which fragments go where, and what runtime values fill
the slots.

Two properties the loader below exists to guarantee:

  * **Byte-exactness.** Each file carries a ``sha256`` of its body in front
    matter and is verified on first read. Journal entries record a prompt version
    id, and analyses compare runs across it; a stray reflow from an editor would
    silently break that comparability, so it is made loud instead.
  * **Cost.** Prompts are built per LLM call, so reads and the splice of the
    shared fragments are cached; a call after the first does at most one
    ``str.replace`` per runtime slot.

Every builder accepts two optional, default-empty extension points so the
self-evolution subsystem can propose prompt changes without forking this module:
``envelope_block`` (a tool-measured statement of what this deployment can reach,
see :mod:`drug_agent.tools.envelope_tool`) and ``addendum`` (free text a candidate
policy appends). With both empty every builder returns exactly the string it
returned before those parameters existed, which is what makes a candidate arm
and a baseline arm comparable — the diff really is the overlay.

``envelope_block`` is a *channel*, not a single fixed document. The master agent
composes what goes down it from up to three runtime-built sections — which tools
this deployment lacks (always, a fact), a run's stated hard ceilings, and the
measured envelope — and this module stays out of that decision: see
``agents/llm/master_agent.py`` for which section attaches to which phase and why.
"""

from __future__ import annotations

import hashlib
import re
from functools import cache, lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# The prompt library on disk
# --------------------------------------------------------------------------
#: Where ``system_prompts/`` is, in each of the two layouts this package ships in.
#: A wheel gets the folder force-included *inside* the package (see
#: ``[tool.hatch.build.targets.wheel.force-include]``); a source checkout or an
#: editable install keeps only the canonical copy at the repository root. The
#: packaged copy is tried first because in a wheel it is the only truth, and its
#: absence is what identifies a checkout.
_LIBRARY_CANDIDATES: tuple[Path, ...] = (
    Path(__file__).resolve().parents[1] / "system_prompts",  # installed wheel
    Path(__file__).resolve().parents[3] / "system_prompts",  # source checkout
)

#: Slot syntax. Doubled braces cannot occur in the prompt bodies (their JSON
#: shapes are ``indent=2``-formatted, so every brace is followed by a newline),
#: which is what makes a plain textual substitution safe here.
_SLOT = re.compile(r"\{\{([a-z_]+)\}\}")

#: Fragments spliced into more than one phase. Substituted once, then cached.
_SHARED = {
    "rules": "shared/master_rules.md",
    "objective_vocabulary": "shared/objective_vocabulary.md",
    "constraint_vocabulary": "shared/constraint_vocabulary.md",
}


@lru_cache(maxsize=1)
def _library_root() -> Path:
    for candidate in _LIBRARY_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "the system_prompts/ library is missing; looked in " + ", ".join(str(p) for p in _LIBRARY_CANDIDATES)
    )


@cache
def _block(name: str) -> str:
    """Read one prompt file, returning its body exactly as the model must see it.

    The file format is YAML front matter, then the body, then the single trailing
    newline every text file ends with — which is *not* part of the prompt and is
    removed here. A body that genuinely ends in a newline therefore ends the file
    with a blank line, and ``sha256`` covers the body after this normalization,
    so the header can be re-documented freely and the prompt still cannot drift.
    """
    path = _library_root() / name
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path}: prompt files must open with a '---' front-matter block")
    end = raw.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path}: front-matter block is not closed by a '---' line")
    meta = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in raw[4:end].splitlines())
    }
    body = raw[end + 5 :]
    if body.endswith("\n"):
        body = body[:-1]
    expected = meta.get("sha256", "")
    actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError(
            f"{path}: prompt body does not match its recorded sha256 "
            f"(recorded {expected or '<missing>'}, found {actual}). The recorded run history is "
            "keyed on these bytes; if the edit is intended, re-stamp with "
            "`python system_prompts/stamp_checksums.py` and say so in the commit."
        )
    return body


@cache
def _template(name: str) -> tuple[str, frozenset[str]]:
    """A phase template with its shared fragments already spliced in.

    Cached, so the per-call cost of a prompt whose only slots are shared (every
    phase but critique) is a dictionary lookup.
    """
    text = _block(name)
    for slot, fragment in _SHARED.items():
        token = "{{" + slot + "}}"
        if token in text:
            text = text.replace(token, _block(fragment))
    return text, frozenset(_SLOT.findall(text))


def _fill(name: str, **values: str) -> str:
    """Render a template, requiring the caller to supply exactly its open slots.

    Checking the slot set *before* substituting is deliberate: the values are
    untrusted runtime text (a user instruction, an evaluation report), so a
    leftover-brace scan of the result would fire on the data instead of the bug.
    """
    text, slots = _template(name)
    if slots != values.keys():
        missing = sorted(slots - values.keys())
        unknown = sorted(values.keys() - slots)
        raise KeyError(f"{name}: missing slots {missing}, unknown arguments {unknown}")
    for slot, value in values.items():
        text = text.replace("{{" + slot + "}}", value)
    return text


def _extend(system: str, envelope_block: str = "", addendum: str = "") -> str:
    """Append the optional policy sections. Empty inputs leave ``system`` alone."""
    parts = [system]
    if envelope_block:
        parts.append("\n\n" + envelope_block.strip())
    if addendum:
        parts.append("\n\n" + addendum.strip())
    return "".join(parts)


# --------------------------------------------------------------------------
# 1. Planning
# --------------------------------------------------------------------------
def plan_prompt(instruction: str, *, envelope_block: str = "", addendum: str = "") -> tuple[str, str]:
    system = _fill("master_agent/plan.system.md")
    user = _fill("master_agent/plan.user.md", instruction=instruction.strip())
    return _extend(system, envelope_block, addendum), user


# --------------------------------------------------------------------------
# 2. Transcription (NL -> ChemicalRequirements)
# --------------------------------------------------------------------------
def transcribe_prompt(
    instruction: str, plan_summary: str, *, envelope_block: str = "", addendum: str = ""
) -> tuple[str, str]:
    system = _fill("master_agent/transcribe.system.md")
    user = _fill(
        "master_agent/transcribe.user.md",
        instruction=instruction.strip(),
        plan_summary=plan_summary,
    )
    return _extend(system, envelope_block, addendum), user


# --------------------------------------------------------------------------
# 3. Critique (judge real results against the plan)
# --------------------------------------------------------------------------
def critique_prompt(
    plan_summary: str,
    requirements_summary: str,
    evaluation_json: str,
    cycle: int,
    max_cycles: int,
    *,
    envelope_block: str = "",
    addendum: str = "",
) -> tuple[str, str]:
    system = _fill("master_agent/critique.system.md", cycle=str(cycle), max_cycles=str(max_cycles))
    user = _fill(
        "master_agent/critique.user.md",
        plan_summary=plan_summary,
        requirements_summary=requirements_summary,
        evaluation_json=evaluation_json,
    )
    return _extend(system, envelope_block, addendum), user


# --------------------------------------------------------------------------
# 4. Refinement (adaptive prompt adjustment — the ablation core)
# --------------------------------------------------------------------------
def refine_prompt(
    requirements_json: str,
    critique_summary: str,
    evaluation_json: str,
    *,
    envelope_block: str = "",
    addendum: str = "",
) -> tuple[str, str]:
    system = _fill("master_agent/refine.system.md")
    user = _fill(
        "master_agent/refine.user.md",
        requirements_json=requirements_json,
        critique_summary=critique_summary,
        evaluation_json=evaluation_json,
    )
    return _extend(system, envelope_block, addendum), user


# --------------------------------------------------------------------------
# 5. Final cross-cycle selection (``dynamic`` ceiling mode only)
# --------------------------------------------------------------------------
def select_best_prompt(
    plan_summary: str,
    criteria_summary: str,
    candidates_json: str,
    *,
    envelope_block: str = "",
    addendum: str = "",
) -> tuple[str, str]:
    """The one step that runs after the loop, over every cycle's candidates.

    It keeps the same two extension points as the four loop phases even though
    no sealed policy field targets it yet: the master agent passes the measured
    envelope here in ``dynamic`` mode, so the model can tell "the best we got"
    apart from "everything this deployment can reach".
    """
    system = _fill("master_agent/select_best.system.md")
    user = _fill(
        "master_agent/select_best.user.md",
        plan_summary=plan_summary,
        criteria_summary=criteria_summary,
        candidates_json=candidates_json,
    )
    return _extend(system, envelope_block, addendum), user
