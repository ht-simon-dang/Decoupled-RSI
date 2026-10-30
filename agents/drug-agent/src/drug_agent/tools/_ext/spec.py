"""One external scientific tool, described well enough to find, judge and run it.

An :class:`ExternalSpec` is *data about an upstream project*, not behaviour. It
answers the four questions the platform asks before it will claim a capability:

  1. **Where would it be?** — ``checkout_relpath`` under the repo, overridable by
     ``DRUGAGENT_<ENV_PREFIX>_ROOT``.
  2. **Is what is there really it?** — ``sentinel_files``, checked as files. A stale
     or empty ``.external/<Name>`` must never let the platform claim a model it
     cannot run, so the probe is "does this checkout contain the module the run
     path imports", not "does this directory exist".
  3. **Which interpreter runs it?** — ``selection_modules`` identifies the
     environment the backend was installed into; ``runtime_modules`` is everything
     the run path imports. The two are deliberately different: selection must be
     able to pick a *half-built* environment that readiness then rejects module by
     module, which is what turns a mid-run ``ModuleNotFoundError`` into an upfront
     UNAVAILABLE with a real reason.
  4. **What would fixing it cost?** — ``install_hint`` names the command, and
     ``upstream_url``/``upstream_ref`` pin exactly which commit the driver was
     written against.

``runtime_modules`` should list the modules a *broken* install actually loses,
not the headline dependency. PFM's precedent is the rule: it probes
``torch_scatter``/``torch_cluster``, which compile from source with no wheel on
any platform, because probing ``torch`` alone reported AVAILABLE and then crashed
at sample time.

Specs are frozen because the catalogue is read concurrently from the pool, the
CLI and the report, and because a spec that could be mutated at runtime would
make "which commit is this driver written against" unanswerable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExternalSpec:
    """Everything the platform needs to locate, validate and launch one upstream tool."""

    #: Human name as the upstream project spells it ("REINVENT4", "GNINA").
    name: str
    #: Uppercase token in ``DRUGAGENT_<ENV_PREFIX>_ROOT`` / ``_PYTHON`` / ``_DEVICE`` /
    #: ``_TIMEOUT_SECONDS``. Also the Makefile target stem, lowercased.
    env_prefix: str
    #: Where ``make <tool>-clone`` puts the checkout, relative to the repo root.
    checkout_relpath: str = ""
    #: Files that must exist *inside* the checkout for it to be the real thing.
    #: Checked as files, never as directories.
    sentinel_files: tuple[str, ...] = ()
    #: Driver script under ``scripts/external/``, run inside the tool's own
    #: interpreter. ``None`` for tools imported in-process.
    driver_name: str | None = None
    #: The narrow probe that identifies which environment the backend lives in.
    selection_modules: tuple[str, ...] = ()
    #: Everything the run path imports. A superset of ``selection_modules``.
    runtime_modules: tuple[str, ...] = ()
    #: Model weights that must exist, relative to the checkout root.
    weight_relpaths: tuple[str, ...] = ()
    #: Conda environments to consider, in order, when the running interpreter
    #: cannot import ``selection_modules``.
    conda_envs: tuple[str, ...] = ()
    #: Executable that must be on PATH, for binary-backed tools.
    binary: str | None = None
    #: Upstream repository and the exact commit the driver was written against.
    upstream_url: str = ""
    upstream_ref: str = ""
    #: Upstream licence, verbatim, plus its state on this machine.
    license: str = ""
    #: The command that would make this tool real here.
    install_hint: str = ""
    #: True when the tool needs a GPU to run at a usable speed.
    needs_gpu: bool = False
    #: Blockers known to be true of *this* platform regardless of installation —
    #: an upstream that publishes only x86-64 binaries, say. These are reported
    #: alongside discovery failures and are never silently dropped.
    platform_blockers: tuple[str, ...] = field(default=())

    @property
    def root_env_var(self) -> str:
        return f"DRUGAGENT_{self.env_prefix}_ROOT"

    @property
    def python_env_var(self) -> str:
        return f"DRUGAGENT_{self.env_prefix}_PYTHON"

    @property
    def settings_root_attr(self) -> str:
        return f"{self.env_prefix.lower()}_root"

    @property
    def settings_python_attr(self) -> str:
        return f"{self.env_prefix.lower()}_python"
