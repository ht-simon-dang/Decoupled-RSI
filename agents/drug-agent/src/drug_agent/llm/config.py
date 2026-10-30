"""Runtime configuration for the LLM plane.

Everything is overridable through environment variables so the same code runs
against a different Ollama host or model without edits. Defaults are tuned for
the recipe measured on this machine with ``qwen3.8:27b``:

  * **A separate reasoning channel.** ``think=True`` keeps the monologue out of
    the answer, which is what makes fenced-JSON parsing reliable at all.
  * **Generous token budgets.** Reasoning and answer share ``num_predict``.
    Measured: the transcription step spent ~1.9k tokens thinking before it
    started answering, and a 2400-token budget cut the JSON off mid-value. The
    presets below leave room for both, and the harness retries with a larger
    budget when the model still runs out.
  * **A mild repeat penalty.** Structured output is inherently repetitive —
    quotes, commas, recurring key names — so a high penalty actively fights
    valid JSON. 1.05 is enough to avoid degenerate loops without that cost.
  * **Schema-constrained decoding, when it earns its place.** Ollama can
    constrain generation to a JSON schema. See ``structured_outputs``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """The project root — the directory holding ``src/drug_agent``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "drug_agent").is_dir():
            return parent
    # Installed as a package rather than run from a checkout.
    return here.parents[3]


def _load_dotenv_once() -> None:
    """Fold the repo-root ``.env`` into the environment, without overriding it.

    ``.env.example`` tells operators to "Copy to .env and adjust", and the
    console reads configuration straight from ``os.environ``. Without this an
    operator who put ``LLM_AGENT_PFM_POCKET`` (or ``DRUGAGENT_PFM_ROOT``) in
    ``.env`` would be told PFM is unconfigured by a console looking somewhere
    other than where the documentation pointed them. Real environment variables
    still win.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    for base in (Path.cwd(), _repo_root()):
        env_path = base / ".env"
        if not env_path.is_file():
            continue
        try:
            from dotenv import dotenv_values

            items = dotenv_values(env_path)
        except Exception:  # a bad/absent parser must never block startup
            continue
        for key, value in items.items():
            if value is not None:
                os.environ.setdefault(key, value)
        return


_DOTENV_LOADED = False


def _env(name: str, default: str) -> str:
    _load_dotenv_once()
    val = os.environ.get(name)
    # An empty variable means "unset" here: `FOO=` in a .env file is how people
    # comment a setting out, and honouring it as "" would silently break lookups.
    return val if val else default


def _env_float(name: str, default: float) -> float:
    _load_dotenv_once()
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    _load_dotenv_once()
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    _load_dotenv_once()
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


#: The model this stack is tuned and tested against.
DEFAULT_MODEL = "qwen3.8:27b"


# --------------------------------------------------------------------------
# Ceiling modes — a RUN-LEVEL choice about what the planner is told it can reach
# --------------------------------------------------------------------------
# Measured problem: the planner sets success criteria blind. One recorded run set
# ``max_pIC50 >= 7.0`` against an observed deployment ceiling of 6.4582, so the
# criterion could never be met and the run burned all three cycles chasing it.
#
# There are two defensible answers and they are not the same experiment, so both
# are selectable per run rather than being a policy field: ``llm/policy.py`` is
# the seam the self-evolution arm writes through and is sealed, and a ceiling is
# an operator's statement about one launch, not a behaviour under study.
#
#   ``off``      the historical behaviour: no ceiling reaches the planner (the
#                measured envelope still appears if the sealed policy asks for
#                it). This stays the default so the recorded history replays.
#   ``hard``     the user states the ceilings at launch. They are spliced into
#                the planning and transcription prompts for that run only and
#                are binding: a criterion beyond one of them is refused.
#   ``dynamic``  the *measured* envelope is shown to the planner as advice, not
#                as a rule, and a final maximum-reasoning selection step picks
#                the best candidate across every cycle (see ``SELECT_BEST``).
CEILING_MODE_OFF = "off"
CEILING_MODE_HARD = "hard"
CEILING_MODE_DYNAMIC = "dynamic"
CEILING_MODES: tuple[str, ...] = (CEILING_MODE_OFF, CEILING_MODE_HARD, CEILING_MODE_DYNAMIC)


class CeilingError(ValueError):
    """A run-level ceiling the platform cannot honour, with the reason.

    Raised rather than absorbed. Every other ``_env_*`` helper here degrades to a
    default because an unreadable *preference* should not stop a program, but a
    ceiling is a correctness statement: silently dropping ``max_pIC50=6.5``
    would hand the planner back exactly the blindness the mode exists to remove.
    """


def parse_ceilings(values: Iterable[str]) -> dict[str, float]:
    """Parse ``METRIC=VALUE`` ceiling terms into a metric -> value mapping.

    Accepts the repeatable form (``--ceiling a=1 --ceiling b=2``) and the comma
    form (``--ceiling a=1,b=2``) interchangeably, because both are what people
    type. Metric names are validated against the keys the evaluation agent can
    actually compute: a ceiling on a metric nothing measures would be a rule the
    planner cannot check itself against.
    """
    from .schemas import METRIC_KEYS, normalize_metric

    out: dict[str, float] = {}
    for value in values:
        for term in str(value).split(","):
            term = term.strip()
            if not term:
                continue
            key, sep, raw = term.partition("=")
            if not sep:
                raise CeilingError(f"ceiling {term!r} is not METRIC=VALUE (for example max_pIC50=6.5)")
            # Fold the near-misses the models and operators both write ("qed" for
            # "qed_median") before rejecting, so a right ceiling spelled loosely
            # is honoured rather than refused.
            metric = normalize_metric(key.strip())
            if metric not in METRIC_KEYS:
                raise CeilingError(
                    f"unknown ceiling metric {key.strip()!r}; valid keys are: "
                    + ", ".join(sorted(METRIC_KEYS))
                )
            try:
                out[metric] = float(raw.strip())
            except ValueError:
                raise CeilingError(f"ceiling {term!r} has a non-numeric value {raw.strip()!r}") from None
    return out


def _env_ceilings(name: str) -> dict[str, float]:
    _load_dotenv_once()
    raw = os.environ.get(name)
    return parse_ceilings([raw]) if raw else {}


@dataclass(frozen=True)
class Sampling:
    """Sampling preset for one reasoning phase.

    ``num_predict`` bounds *reasoning plus answer*, which is why these numbers
    look large next to the size of the JSON they produce.
    """

    temperature: float
    top_p: float = 0.95
    top_k: int = 20
    repeat_penalty: float = 1.05
    num_predict: int = 6000

    def options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
        }

    def with_budget(self, num_predict: int) -> Sampling:
        return replace(self, num_predict=num_predict)


# Phase presets: planning is a little exploratory, transcription is precise,
# critique is analytical, refinement is moderately creative (it proposes new
# scaffolds / parameter moves) and emits the largest object, so it gets the
# largest budget.
PLANNING = Sampling(temperature=0.5, num_predict=6000)
TRANSCRIBE = Sampling(temperature=0.25, num_predict=6000)
CRITIQUE = Sampling(temperature=0.3, num_predict=5000)
REFINE = Sampling(temperature=0.45, num_predict=8000)

# The final cross-cycle selection step of ``dynamic`` ceiling mode: "maximum
# reasoning", made concrete. Two numbers differ from every loop phase above, for
# two different reasons.
#
#   num_predict=12000 — the largest budget in this module, 1.5x the previous
#   largest (REFINE). It is the only step whose input is *every* cycle at once:
#   max_cycles x shortlist_size candidate rows, each carrying five endpoint
#   numbers plus utility and Pareto rank, and the model has to hold all of them
#   in the reasoning channel while it compares. Reasoning and answer share this
#   budget (see ``Sampling``), and the answer itself is a handful of fields, so
#   effectively all of it is thinking room. Running out here does not degrade
#   the choice, it truncates the JSON — the failure the loop-phase budgets were
#   already raised twice to avoid, on inputs a third this size.
#
#   temperature=0.1 with top_p tightened to 0.85 — selection is a *ranking over
#   numbers the tools already computed* (ADR-0011: the master agent decides and
#   explains, it never invents or adjusts a score). There is nothing to explore:
#   the candidates exist, their scores exist, and the only work is choosing.
#   Sampling warmly here would make the same scoreboard yield a different winner
#   on a re-run, which is precisely what makes a selection unreviewable.
SELECT_BEST = Sampling(temperature=0.1, top_p=0.85, num_predict=12000)


@dataclass
class Config:
    # --- Ollama ---------------------------------------------------------
    ollama_host: str = field(
        default_factory=lambda: _env("LLM_AGENT_OLLAMA_HOST", _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    )
    model: str = field(default_factory=lambda: _env("LLM_AGENT_MODEL", DEFAULT_MODEL))
    request_timeout_s: float = field(default_factory=lambda: _env_float("LLM_AGENT_TIMEOUT", 900.0))
    keep_alive: str = field(default_factory=lambda: _env("LLM_AGENT_KEEP_ALIVE", "30m"))
    #: Stream the model's reasoning channel. Off makes runs faster and dumber.
    think: bool = field(default_factory=lambda: _env_bool("LLM_AGENT_THINK", True))
    #: Constrain decoding to the target JSON schema (Ollama ``format``).
    #: "auto" uses it only for the phases measured to benefit; "on"/"off" force it.
    structured_outputs: str = field(default_factory=lambda: _env("LLM_AGENT_STRUCTURED", "auto"))

    # --- Dynamic workflow ----------------------------------------------
    max_cycles: int = field(default_factory=lambda: _env_int("LLM_AGENT_MAX_CYCLES", 3))
    # "adaptive" = LLM diagnoses failure and rewrites the generation prompt.
    # "naive"    = re-run the identical requirements (the ablation baseline).
    refine_strategy: str = field(default_factory=lambda: _env("LLM_AGENT_REFINE_STRATEGY", "adaptive"))
    max_json_repairs: int = field(default_factory=lambda: _env_int("LLM_AGENT_MAX_REPAIRS", 2))

    # --- ceiling mode (run-level; see CEILING_MODES above) ----------------
    # These are launch parameters, not stored settings: the CLI puts them in the
    # process environment for the one run it is starting, and nothing writes them
    # to a config file, the profile, or the runtime policy. They are recorded in
    # the run journal, which is a record of what happened, not a place a later
    # run reads its defaults from.
    ceiling_mode: str = field(default_factory=lambda: _env("LLM_AGENT_CEILING_MODE", CEILING_MODE_OFF))
    ceilings: dict[str, float] = field(default_factory=lambda: _env_ceilings("LLM_AGENT_CEILINGS"))

    # --- drug_agent backend --------------------------------------------
    drug_agent_profile: str = field(default_factory=lambda: _env("DRUGAGENT_PROFILE", "local"))
    deterministic: bool = field(
        default_factory=lambda: _env("LLM_AGENT_DETERMINISTIC", "1") not in ("0", "false", "False")
    )

    # --- de-novo generation ---------------------------------------------
    # Candidate source for a run. "auto" — the default — means the MASTER AGENT
    # chooses it during transcription, from the generators installed here: which
    # arm suits a goal is a scientific judgement, and the console used to force it
    # on whoever clicked Launch. Setting this to a concrete value ("library",
    # "pfm", "library+pfm") pins it instead, which is what the ablation arms need.
    generator: str = field(default_factory=lambda: _env("LLM_AGENT_GENERATOR", "auto"))
    n_generated: int = field(default_factory=lambda: _env_int("LLM_AGENT_N_GENERATED", 40))
    # A PREPARED POCKET pdb for the target. The real PFM backend refuses without
    # one; the labelled surrogate does not need it.
    pfm_pocket_pdb: str = field(default_factory=lambda: _env("LLM_AGENT_PFM_POCKET", ""))

    # --- journal (per-request log system) --------------------------------
    journal_enabled: bool = field(default_factory=lambda: _env_bool("LLM_AGENT_JOURNAL", True))
    journal_dir: str = field(default_factory=lambda: _env("DRUGAGENT_JOURNAL_DIR", ""))

    # --- server ---------------------------------------------------------
    host: str = field(default_factory=lambda: _env("LLM_AGENT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("LLM_AGENT_PORT", 2003))

    def resolved_repo_root(self) -> Path:
        return _repo_root()

    def resolved_journal_dir(self) -> Path:
        from ..observability.journal import default_journal_root

        return Path(self.journal_dir).expanduser() if self.journal_dir else default_journal_root()

    def resolved_ceiling_mode(self) -> str:
        """The ceiling mode this run really uses, with the contradictions refused.

        Supplying a ceiling *is* asking for ``hard``, so a bare ``--ceiling
        max_pIC50=6.5`` does not silently do nothing. The two ways to be wrong
        are refused rather than guessed at: ``hard`` with nothing to enforce is
        an empty rule, and a user-stated ceiling alongside ``dynamic`` asks the
        planner to treat one number as binding and advisory at once.
        """
        mode = (self.ceiling_mode or CEILING_MODE_OFF).strip().lower()
        if mode not in CEILING_MODES:
            raise CeilingError(f"unknown ceiling mode {mode!r}; valid modes are: " + ", ".join(CEILING_MODES))
        if self.ceilings and mode == CEILING_MODE_OFF:
            return CEILING_MODE_HARD
        if self.ceilings and mode == CEILING_MODE_DYNAMIC:
            raise CeilingError(
                "ceiling mode 'dynamic' lets the master agent weigh the MEASURED envelope itself, "
                "so it cannot be combined with a user-supplied hard ceiling; drop --ceiling, or use "
                "--ceiling-mode hard"
            )
        if mode == CEILING_MODE_HARD and not self.ceilings:
            raise CeilingError(
                "ceiling mode 'hard' needs at least one --ceiling METRIC=VALUE to enforce "
                "(for example --ceiling max_pIC50=6.5)"
            )
        return mode

    def with_overrides(self, **kw: object) -> Config:
        return replace(self, **kw)  # type: ignore[arg-type]

    def public_dict(self) -> dict[str, object]:
        return {
            "ollama_host": self.ollama_host,
            "model": self.model,
            "think": self.think,
            "structured_outputs": self.structured_outputs,
            "max_cycles": self.max_cycles,
            "refine_strategy": self.refine_strategy,
            # Recorded so a journal entry says which ceiling the planner was
            # working under — a plan is not interpretable without it.
            "ceiling_mode": self.ceiling_mode,
            "ceilings": dict(self.ceilings),
            "drug_agent_profile": self.drug_agent_profile,
            "deterministic": self.deterministic,
            "generator": self.generator,
            "n_generated": self.n_generated,
            "pfm_pocket_pdb": self.pfm_pocket_pdb or None,
            "journal_enabled": self.journal_enabled,
            "journal_dir": str(self.resolved_journal_dir()),
            "pfm": self.generator_status(),
        }

    def generator_status(self) -> dict[str, object]:
        """Live PFM availability, so the console can say which backend will run.

        Imported lazily and defensively: a broken optional backend must degrade
        the badge, never break the health endpoint.
        """
        try:
            from ..tools.pfm_tool import PFMGeneratorAdapter

            adapter = PFMGeneratorAdapter(backend="auto")
            profile = adapter.profile
            # PFM being installed is necessary but not sufficient: it samples a
            # *pocket*, so without one a run would fail after the badge said "active".
            pocket = self.pfm_pocket_pdb or (
                str(adapter.paths.pocket_pdb) if adapter.paths.pocket_pdb else ""
            )
            # Whether the configured pocket is REALLY there, and whether a box
            # can be implied from it. The old code tested a string for
            # truthiness, so a path left over from a renamed checkout reported a
            # healthy pocket; the master agent's prompt then stated as fact that
            # "a pocket IS configured here" for a file that did not exist, and
            # the run died on the contract field that file was supposed to carry.
            from pathlib import Path as _Path

            exists = bool(pocket) and _Path(pocket).is_file()
            centre: list[float] | None = None
            reason: str | None = None
            if pocket and not exists:
                reason = "the configured pocket file does not exist on this machine"
            elif exists:
                from .bridge import _measured_centre

                measured = _measured_centre(pocket)
                centre = measured.get("center")
                if centre is None:
                    reason = measured.get("notes") or "no centre could be measured from the file"
            return {
                "backend": adapter.backend,
                "status": str(profile.status),
                "available": adapter.available(),
                "reason": profile.unavailable_reason,
                "root": str(adapter.paths.root) if adapter.paths.root else None,
                "pocket": pocket or None,
                "needs_pocket": bool(adapter.available() and not exists),
                "pocket_exists": exists,
                "pocket_center": centre,
                "pocket_reason": reason,
            }
        except Exception as exc:
            return {
                "backend": "unknown",
                "status": "UNAVAILABLE",
                "available": False,
                "reason": str(exc),
                "root": None,
                "pocket": None,
                "needs_pocket": False,
                "pocket_exists": False,
                "pocket_center": None,
                "pocket_reason": str(exc),
            }


def load_config() -> Config:
    return Config()
