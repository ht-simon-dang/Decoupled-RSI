"""Runtime settings.

Settings are layered: built-in defaults -> profile YAML (``configs/<profile>.yaml``)
-> environment variables (prefix ``DRUGAGENT_``). Secrets are only ever read
from the environment, never persisted to disk or manifests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..util import content_hash


def _default_root() -> Path:
    env = os.environ.get("DRUGAGENT_HOME")
    if env:
        return Path(env)
    return Path.cwd() / ".drugagent"


class Settings(BaseSettings):
    """Process configuration. Immutable per run once resolved."""

    model_config = SettingsConfigDict(
        env_prefix="DRUGAGENT_",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
        # `.env.example` opens with "Copy to .env and adjust", so honour that: without
        # this the documented way to configure the platform is inert, and an operator
        # who set DRUGAGENT_PFM_ROOT there is told PFM is not installed. A real
        # environment variable still wins over the file.
        env_file=".env",
        env_file_encoding="utf-8",
    )

    profile: str = "local"
    deterministic: bool = False
    seed: int = 1729

    # Paths
    home: Path = Field(default_factory=_default_root)
    runs_dir: Path | None = None
    artifact_dir: Path | None = None
    db_url: str | None = None

    # Concurrency / limits
    max_concurrency: int = 8
    #: Flat per-task time budget, and the floor under every scaled budget below.
    default_task_timeout_s: float = 120.0

    #: Extra attempts granted to a task that invokes an external capability, on
    #: top of its first one. Only *transient* faults consume them: the engine
    #: retries ``RetryableAgentError`` and unexpected exceptions, and never
    #: retries ``AgentError``, which is an agent saying "this is not worth trying
    #: again". Tasks with no ``capability_id`` get none, because a pure
    #: in-process step has no transient failure mode to retry through — retrying
    #: one would only spend budget to reach the same answer.
    #:
    #: 1 rather than 0 because every backend here is a subprocess or an HTTP
    #: call, and a single connection reset used to fail a whole run. 1 rather
    #: than 3 because a retry is charged, and a backend that needs three attempts
    #: is a finding, not a flake.
    transient_task_retries: int = 1

    # Per-candidate task budgets.
    #
    # A flat per-task budget is the wrong shape for the steps that walk the whole
    # candidate set: the same handler that costs ~10 s over the 84 library/seed/
    # analogue compounds of the sEH demo costs ~123 s once the brief also asks for
    # 40 PFM-generated structures, because de-novo molecules (cumulenes,
    # bridgeheads, spiro systems, 7-8-membered rings) are ~30x costlier to embed
    # and minimise than catalogue compounds. Sizing those steps per candidate is
    # what makes an overrun mean "something is wrong" rather than "this brief
    # enabled generation".
    #
    #: Seconds of budget granted per candidate in the run's store, for the tasks
    #: named below. 3.0 s is set from the worst arm measured rather than the
    #: average: MMFF94s embed+minimise over the PFM arm of the sEH brief averaged
    #: 2.83 s/molecule (max 41.4 s), against 0.12 s/molecule for the library arm.
    #: So this sizes a batch made *entirely* of de-novo structures, which is the
    #: worst case a brief can ask for; the observed mixed 124-candidate batch
    #: (123.3 s actual) gets 372 s, ~3x headroom.
    per_candidate_task_timeout_s: float = 3.0
    #: Tasks whose cost scales with the persisted candidate set. ``source_candidates``
    #: is deliberately absent: it is the step that *creates* the candidates, so the
    #: store is still empty when it starts and its cost tracks what the brief asked
    #: for, not what is on disk. Generation overruns are covered by the engine
    #: treating an overrun as an observation instead of a failure.
    per_candidate_budget_tasks: list[str] = Field(
        default_factory=lambda: [
            "standardize_candidates",
            "apply_hard_filters",
            "predict_activity_and_properties",
            "assess_structure",
            "assess_synthesis",
            "assess_safety",
            "assess_perturbation",
            "validate_all_results",
        ]
    )

    max_tool_output_bytes: int = 5_000_000

    # Network / security
    allow_network: bool = False
    egress_allowlist: list[str] = Field(default_factory=list)

    # LLM provider
    llm_provider: str = "fake"  # 'fake' | 'stub_openai' | ...
    llm_model: str = "deterministic-fake-1"

    # Backends
    workflow_backend: str = "embedded"  # 'embedded' | 'durable_stub'
    storage_backend: str = "local"  # 'local' | 's3_stub'

    # Observability
    #: Record every pipeline execution in the running history (``running-history/``).
    #: The LLM bridge turns this off for its nested runs, which the request
    #: journal that drove them already records.
    record_history: bool = True
    telemetry: str = "local"  # 'local' | 'otlp'
    otlp_endpoint: str | None = None

    # Scientific defaults
    fingerprint_bits: int = 2048
    fingerprint_radius: int = 2

    # De-novo generation (PFM). Its deps (torch + PyG + compiled extensions +
    # OpenBabel) install alongside this platform, so pfm_python normally stays
    # unset and sampling runs in a subprocess of the current interpreter; set it
    # only to target a different environment. Without pfm_root the generator
    # reports UNAVAILABLE and produces nothing: there is no surrogate backend.
    generator_backend: str = "auto"  # 'auto' | 'pfm' — 'mock' was removed
    pfm_root: Path | None = None  # a PFM checkout
    pfm_python: str | None = None  # defaults to sys.executable
    pfm_checkpoint: Path | None = None  # default <root>/model_params/pfm.pth
    pfm_training_yml: Path | None = None  # default <root>/model_params/training.yml
    pfm_x_predictor: Path | None = None  # override training.yml's absolute path
    pfm_h_predictor: Path | None = None  # override training.yml's absolute path
    pfm_device: str = "cuda:0"
    pfm_timeout_seconds: float = 1800.0
    # Default prepared-pocket PDB, used when a brief names no pocket file of its
    # own. Cut one with scripts/prepare_pocket.py.
    #
    # The docs, README and .env.example all spell this DRUGAGENT_PFM_POCKET, while the
    # field name would derive DRUGAGENT_PFM_POCKET_PDB. Accept both, so the documented
    # name works through a `.env` and not only as a live environment variable.
    pfm_pocket_pdb: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("DRUGAGENT_PFM_POCKET", "DRUGAGENT_PFM_POCKET_PDB"),
    )

    # The target this deployment is working on, as a built `target_manifest.json`
    # (see scripts/research/build_zak_target_data.py). When it is set, the pocket
    # above is checked against the manifest's residue map before PFM samples
    # anything, and a pocket from a different protein stops the run.
    #
    # This exists because `pfm_pocket_pdb` is a path with no owner: two recorded
    # ZAK runs generated, scored and ACCEPTed molecules against
    # `.external/pockets/seh_3i28_pocket10.pdb` — soluble epoxide hydrolase —
    # because nothing compared that file with the target in the brief. Left
    # unset, nothing is compared and the tool reports the check as not run.
    target_manifest: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("DRUGAGENT_TARGET_MANIFEST"),
    )

    # The evolution-artifact registry this deployment adopts from
    # (:class:`drug_agent.selfevolve.target.registry.ArtifactRegistry`). Left
    # unset, the master loads nothing evolved and says so: a deployment that has
    # adopted nothing is not a deployment with a broken tool.
    #
    # Pointing at it is not adoption either. The registry distinguishes
    # `development_candidate`, `benchmark_accepted` and `master_active`, and only
    # the third is loaded here -- moving that pointer still needs the same
    # explicit operator approval `drugagent self-evolve promote --i-approve`
    # needs, because it changes how the user-serving system behaves.
    #
    # An artifact in the registry is checked against `target_manifest` above
    # before it is used. An artifact evolved for a different target reports
    # UNAVAILABLE naming both hashes rather than being loaded, for the same
    # reason the pocket is checked: a computation against the wrong target is
    # not a worse answer, it is an answer to a different question.
    target_artifact_registry: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("DRUGAGENT_TARGET_ARTIFACT_REGISTRY", "DRUGAGENT_TARGET_REGISTRY"),
    )

    # Optional assessment arms.
    #
    # Every one of these names a tool that is installed, wired into its agent's
    # constructor, and declared in `agents/registry.AGENT_TOOLS` — and every one
    # of them was, until now, unreachable by any run: `build_registry` called the
    # constructors with the arms off and nothing above it could pass anything
    # else. A tool a deployment installs and no decision can switch on is not an
    # optional arm, it is dead weight, so these settings are the switch.
    #
    # They stay off by default for the reason they always were: a learned property
    # panel, a retrosynthetic search and a predicted complex each cost a run real
    # time, and a pipeline that did not ask for them must cost what it always
    # cost. What changed is that asking is now possible.
    #: Learned property panels the molecule agent runs: admetai | chemprop | unimol.
    property_backends: list[str] = Field(default_factory=list)
    #: Retrosynthetic search for the synthesis agent and the evaluator's route
    #: audit: 'none' | 'aizynthfinder'.
    route_backend: str = "none"
    #: How many candidates per batch get a real route search. A search is bounded
    #: at 120 s per molecule and the step runs on every candidate that passed the
    #: hard filters, so an uncapped arm is hours per cycle. Raise it on a host that
    #: can afford it; the agent reports what the cap left out either way.
    route_budget: int = 12
    #: Co-folded complex prediction for the structure agent: 'none' | 'boltz2' |
    #: 'protenix'.
    complex_backend: str = "none"
    #: Docking engine for the structure agent: 'auto' drives GNINA where it can
    #: really run, then AutoDock Vina, and 'none' switches docking off. Named here
    #: rather than hard-coded at the registration site so an operator can pin or
    #: disable it.
    dock_backend: str = "auto"
    #: How many candidates per batch get a real docking run. Docking became a real
    #: computation in repair R1 and costs ~10 s per flexible ligand measured here,
    #: so an uncapped arm does not fit the engine's own task budget at any batch
    #: size. Raise it on a host that can afford it; 0 switches the arm off while
    #: leaving the capability reachable. The agent reports what the cap left out
    #: either way.
    dock_budget: int = 8
    #: Path to the campaign budget ledger
    #: (:class:`drug_agent.selfevolve.target.budget.BudgetLedger`), or empty for
    #: none. ``dock_budget`` above is a *per-batch* cap held in one agent's
    #: memory; this is the campaign-wide, restart-surviving, cross-process
    #: account Appendix A §4's declared caps are enforced against. Setting it
    #: brings the existing per-batch cap into the ledger's view rather than
    #: creating a second, disagreeing one: the batch cap applies first and every
    #: dock it allows is reserved and settled against the campaign's cap.
    budget_ledger_path: str = ""
    #: The campaign cap on docking calls, declared on first use of the ledger.
    #: Zero leaves it undeclared, in which case an attached ledger refuses to
    #: schedule docking at all rather than assume an unlimited budget.
    budget_dock_calls: float = 0.0

    # External scientific tools (the catalogue in docs/): thirty upstream projects,
    # each with a checkout, sometimes its own interpreter, sometimes its own weights.
    #
    # One nested field rather than ninety flat ones. `pfm_*` above stays flat because
    # it predates this and its env-var spellings are documented and depended on; every
    # tool added since is configured here instead, keyed by the lowercased
    # `ExternalSpec.env_prefix`:
    #
    #     external_tools:
    #       gnina:   {binary: /opt/gnina/bin/gnina}
    #       boltz:   {root: /models/boltz, python: /envs/boltz/bin/python}
    #       diffdock: {root: /models/DiffDock, timeout_seconds: "3600"}
    #
    # Values are strings so a profile YAML, a JSON override and an environment
    # variable all mean the same thing. The escape hatch for an operator who cannot
    # edit a profile is still the environment: `DRUGAGENT_<PREFIX>_ROOT`,
    # `_PYTHON`, `_DEVICE`, `_TIMEOUT_SECONDS`, which win over nothing but are read
    # when this map says nothing. Resolution order lives in
    # `drug_agent.tools._ext.backend.ExternalBackend._resolve`.
    external_tools: dict[str, dict[str, str]] = Field(default_factory=dict)

    def resolved(self) -> Settings:
        """Fill in derived paths and enforce profile invariants."""
        home = Path(self.home)
        home.mkdir(parents=True, exist_ok=True)
        if self.runs_dir is None:
            self.runs_dir = home / "runs"
        if self.artifact_dir is None:
            self.artifact_dir = home / "artifacts"
        if self.db_url is None:
            self.db_url = f"sqlite:///{home / 'state.sqlite'}"
        # local profile is always offline & CPU
        if self.profile == "local":
            # allow_network may still be toggled explicitly, but default False.
            pass
        return self

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"home", "runs_dir", "artifact_dir", "db_url"})
        return content_hash(payload)

    def public_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        return d


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "local": {
        "allow_network": False,
        "workflow_backend": "embedded",
        "storage_backend": "local",
        "llm_provider": "fake",
        "telemetry": "local",
        "max_concurrency": 8,
    },
    "gpu": {
        "allow_network": False,
        "generator_backend": "auto",
        "workflow_backend": "embedded",
        "storage_backend": "local",
        "llm_provider": "fake",
        "telemetry": "local",
        "max_concurrency": 12,
    },
    "cluster": {
        "allow_network": True,
        "workflow_backend": "durable_stub",
        "storage_backend": "s3_stub",
        "llm_provider": "stub_openai",
        "telemetry": "otlp",
        "max_concurrency": 32,
    },
}


def load_settings(
    profile: str | None = None,
    *,
    deterministic: bool | None = None,
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Resolve settings from defaults + profile file + env + explicit overrides."""
    profile = profile or os.environ.get("DRUGAGENT_PROFILE", "local")
    base: dict[str, Any] = {"profile": profile}
    base.update(_PROFILE_DEFAULTS.get(profile, {}))

    # Optional YAML profile file
    config_dir = config_dir or Path("configs")
    pfile = config_dir / f"{profile}.yaml"
    if pfile.exists():
        loaded = yaml.safe_load(pfile.read_text()) or {}
        if isinstance(loaded, dict):
            base.update(loaded)

    # Values from the profile file are *defaults*, so let the environment override
    # them. Passing them as constructor kwargs would outrank every environment
    # variable in pydantic-settings, which silently ignored e.g.
    # DRUGAGENT_GENERATOR_BACKEND simply because configs/local.yaml mentions the key.
    base = {k: v for k, v in base.items() if not _configured_in_env(k)}

    # Explicit arguments are the caller's decision and outrank the environment.
    if deterministic is not None:
        base["deterministic"] = deterministic
    if overrides:
        base.update(overrides)

    settings = Settings(**base)
    return settings.resolved()


def _env_names(field_name: str) -> list[str]:
    """Every environment variable that can set ``field_name`` (aliases included)."""
    info = Settings.model_fields.get(field_name)
    alias = getattr(info, "validation_alias", None) if info is not None else None
    choices = getattr(alias, "choices", None)
    if choices:
        return [c for c in choices if isinstance(c, str)]
    if isinstance(alias, str):
        return [alias]
    return [f"DRUGAGENT_{field_name.upper()}"]


def _configured_in_env(field_name: str) -> bool:
    """True when the operator set this field in the environment or in ``.env``."""
    names = _env_names(field_name)
    if any(n in os.environ for n in names):
        return True
    try:
        from dotenv import dotenv_values

        dotenv = dotenv_values(".env") if Path(".env").is_file() else {}
    except Exception:  # a missing/broken parser must not change resolution
        return False
    return any(n in dotenv for n in names)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    global _SETTINGS
    _SETTINGS = settings
