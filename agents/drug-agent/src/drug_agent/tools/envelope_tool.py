"""``platform.envelope`` — what this deployment can actually achieve.

Every scientific number the planner is given must come from a tool. This is the
tool that answers a question the platform could not previously answer at all:

    *Before we plan, what is the achievable range of each success metric on
    this machine, with these fixtures, these models and these backends?*

It exists because the request journal said it should. Across the five historical
requests, **every** plan set a potency target above what the deployment can
produce (targets of 7.0–8.0 against an observed ceiling of 6.4582), and every
plan that scored synthetic accessibility asked for ``sa_median <= 3.0–3.5``
against an observed 5.0–5.6 — from a *mocked* backend. The planner was not being
careless; it had no way to know. Nothing in the system published its own
operating range.

The measurement is a real run: two probe pipelines (one target-based, one
phenotype-based) with permissive requirements, through the same Engine, the same
RDKit, the same activity model and the same ranking plane a user request goes
through. The probe deliberately applies **no hard constraints**, so what comes
back is the envelope of the *candidate pool*, not of one brief's filter.

Cost is ~5 s and the result is cached per (fixture, settings) fingerprint, so a
planning step pays for it once.

Honesty rules baked in:

* the achievable column is an **observation**, not a cap. A probe runs the
  unseeded library, so a run whose seeds are themselves drug-like can exceed the
  probe on pass-rate-like metrics — the note on each range says which way it can
  move, and the prompt block tells the planner not to treat a number as a target
  merely because it is printed;
* a metric whose backend is ``MOCKED`` is reported with ``mocked=True`` and the
  reason, because a target set against a stand-in is not a scientific target;
* a metric with no data (no bioactivity fixture for the target) is reported as
  ``available=False``, never as zero;
* the numbers are ``CALCULATED``/``PREDICTED`` observations of this deployment,
  not claims about chemistry.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CapabilityStatus, CostClass, Determinism
from ..observability.toolmeter import measure

CAPABILITY_ID = "platform.envelope"

#: Bumped when the probe changes in a way that makes cached values wrong.
ENVELOPE_VERSION = 2


@dataclass
class MetricRange:
    """The observed range of one success metric on this deployment."""

    key: str
    available: bool = False
    #: Best value achievable in the direction the metric is scored.
    best: float | None = None
    median: float | None = None
    p90: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    n: int = 0
    backend: str = ""
    mocked: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityEnvelope:
    """Everything the planner needs to set a target it can actually hit."""

    version: int = ENVELOPE_VERSION
    measured_at: float = 0.0
    fingerprint: str = ""
    elapsed_s: float = 0.0
    #: Metric key -> observed range.
    metrics: dict[str, MetricRange] = field(default_factory=dict)
    #: tool_id -> status, so "this number came from a mock" is visible.
    tool_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    probe: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "measured_at": self.measured_at,
            "fingerprint": self.fingerprint,
            "elapsed_s": round(self.elapsed_s, 3),
            "metrics": {k: v.as_dict() for k, v in sorted(self.metrics.items())},
            "tool_status": dict(sorted(self.tool_status.items())),
            "warnings": list(self.warnings),
            "probe": self.probe,
        }

    # -- rendering -------------------------------------------------------
    def as_prompt_block(self) -> str:
        """The envelope as a compact block for a system prompt.

        Deliberately terse and unambiguous. It states what was measured, gives
        the achievable value per metric, and says which numbers came from a
        mocked backend — a planner that sets a hard target against a stand-in
        should at least be told it is doing so.
        """
        if not self.metrics:
            return ""
        lines = [
            "MEASURED CAPABILITY ENVELOPE OF THIS DEPLOYMENT",
            "(tool-computed on this machine by a probe run of the real pipeline —",
            " these are observations, not permissions, and not chemistry claims)",
            "",
            "| metric | best achievable | median | source |",
            "| --- | --- | --- | --- |",
        ]
        for key, rng in sorted(self.metrics.items()):
            if not rng.available:
                lines.append(f"| {key} | UNAVAILABLE | — | {rng.note or 'no data on this deployment'} |")
                continue
            src = rng.backend or "—"
            if rng.mocked:
                src += " (MOCKED — a deterministic stand-in, not a real backend)"
            lines.append(f"| {key} | {rng.best} | {rng.median} | {src} |")
        scope = str((self.probe or {}).get("candidate_source") or "library")
        lines += [
            "",
            f"Scope: these figures were measured with candidate_source={scope!r}. A run that",
            "samples a de-novo generator is not bound by them — it may do better or worse,",
            "and this envelope says nothing about that arm.",
            "",
            "How to read this. The achievable column is what this deployment has been",
            "OBSERVED to produce. It is not a cap on ambition and it is not a target:",
            "do not set a criterion AT a value merely because it appears here. Set the",
            "criterion the goal actually needs, and then check it against this column —",
            "a target beyond the achievable value has never been met on this deployment",
            "and cannot be reached by refinement, so it does not make the program",
            "stricter, it only guarantees the criterion reads as missed and the run",
            "burns its whole cycle budget. If the goal genuinely needs more than the",
            "envelope allows, set the criterion at what the goal needs, say in `risks`",
            "that the deployment cannot reach it, and explain why it still matters.",
        ]
        if self.warnings:
            lines += ["", "Envelope warnings: " + "; ".join(self.warnings)]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
_PROBE_TARGET: dict[str, Any] = {
    "kind": "target_based",
    "title": "capability probe (target)",
    "target_name": "soluble epoxide hydrolase (sEH)",
    "target_gene": "EPHX2",
    "primary_objective": "pIC50",
    "objectives": [
        {"name": "pIC50", "direction": "maximize", "weight": 1.0, "required": True},
        {"name": "qed", "direction": "maximize", "weight": 1.0},
        {"name": "sa_score", "direction": "minimize", "weight": 1.0},
        {"name": "hERG", "direction": "minimize", "weight": 1.0},
    ],
    "property_constraints": [],
    "seed_smiles": [],
    "novelty": "medium",
    "diversity_requirement": "medium",
    "generator": "library",
    "max_candidates": 400,
    "shortlist_size": 12,
}

_PROBE_PHENOTYPE: dict[str, Any] = {
    "kind": "phenotype_based",
    "title": "capability probe (phenotype)",
    "phenotype_description": "capability probe: generic phenotype-rescue program",
    "primary_objective": "restoration_score",
    "objectives": [
        {"name": "restoration_score", "direction": "maximize", "weight": 1.0, "required": True},
        {"name": "qed", "direction": "maximize", "weight": 1.0},
        {"name": "sa_score", "direction": "minimize", "weight": 1.0},
    ],
    "property_constraints": [],
    "seed_smiles": [],
    "novelty": "medium",
    "diversity_requirement": "medium",
    "generator": "library",
    "max_candidates": 400,
    "shortlist_size": 12,
}

#: The constraint set almost every historical plan reached for: Lipinski plus
#: "no structural alerts". Probed separately because ``hard_constraint_pass_rate``
#: measured with no constraints is always 1.0 and tells a planner nothing —
#: while ``== 1.0`` was the target set in 5 of 5 historical plans, and missed in 4.
_STANDARD_CONSTRAINTS: list[dict[str, Any]] = [
    {"property": "descriptors.mw", "op": "<=", "value": 500},
    {"property": "descriptors.logp", "op": "<=", "value": 5.0},
    {"property": "descriptors.hbd", "op": "<=", "value": 5},
    {"property": "descriptors.hba", "op": "<=", "value": 10},
    {"property": "n_structural_alerts", "op": "==", "value": 0},
]

#: Which direction each metric is "best" in. A metric absent here is reported
#: with best == maximum.
_MINIMIZE = frozenset({"sa_median", "herg_median"})

#: Endpoint -> the tool that produces it, for the mocked-backend warning.
_ENDPOINT_TOOL = {
    "pIC50": "model.activity",
    "qed": "chem.standardize",
    "sa_score": "synth.assess",
    "hERG": "safety.assess",
    "restoration_score": "perturb.assess",
}


def _endpoint_values(scorecards: list[dict[str, Any]], name: str) -> list[float]:
    out: list[float] = []
    for sc in scorecards:
        for e in sc.get("endpoints", []):
            if e.get("endpoint") == name and not e.get("missing") and e.get("raw_value") is not None:
                out.append(float(e["raw_value"]))
    return out


def _in_domain_fraction(scorecards: list[dict[str, Any]], name: str = "pIC50") -> tuple[int, int]:
    seen = hit = 0
    for sc in scorecards:
        for e in sc.get("endpoints", []):
            if e.get("endpoint") == name and e.get("applicability"):
                seen += 1
                hit += 1 if e["applicability"] == "IN_DOMAIN" else 0
    return hit, seen


def _range(
    key: str, values: list[float], *, backend: str, mocked: bool, top_k: int | None = None
) -> MetricRange:
    """Observed range of one metric, with ``best`` meaning *best achievable*.

    The distinction matters. ``max_pIC50`` is scored over the shortlist, so the
    best a shortlist can do is the pool maximum. ``sa_median`` is a *median* over
    the shortlist, so the best achievable value is the median of the ``top_k``
    most favourable molecules — not the single best one. Reporting the pool
    optimum for a median metric would hand the planner a target that is still
    unreachable, which is the failure this tool exists to remove.
    """
    if not values:
        return MetricRange(
            key=key,
            available=False,
            backend=backend,
            mocked=mocked,
            note="no values produced on this deployment",
        )
    minimize = key in _MINIMIZE
    ordered = sorted(values, reverse=not minimize)  # most favourable first
    n = len(ordered)
    note = ""
    if top_k:
        k = max(1, min(n, top_k))
        best = statistics.median(ordered[:k])
        note = f"best = median of the {k} most favourable candidates (a shortlist-median metric)"
    else:
        best = ordered[0]
    idx = max(0, min(n - 1, round(0.9 * (n - 1))))
    return MetricRange(
        key=key,
        available=True,
        best=round(best, 4),
        median=round(statistics.median(ordered), 4),
        p90=round(ordered[idx], 4),
        minimum=round(min(ordered), 4),
        maximum=round(max(ordered), 4),
        n=n,
        backend=backend,
        mocked=mocked,
        note=note,
    )


def _fingerprint(cfg: Any) -> str:
    """Identity of the thing being measured: fixtures + profile + generator.

    A cached envelope is only valid for the deployment that produced it, and
    "the fixtures changed" is the case that would otherwise go unnoticed.
    """
    parts: list[str] = [
        str(ENVELOPE_VERSION),
        str(getattr(cfg, "drug_agent_profile", "")),
        str(getattr(cfg, "generator", "")),
        str(getattr(cfg, "deterministic", "")),
    ]
    root = Path(getattr(cfg, "resolved_repo_root", lambda: Path.cwd())())
    for name in ("seh_bioactivity.csv", "compound_library.csv", "targets.json"):
        path = root / "fixtures" / "data" / name
        try:
            stat = path.stat()
            parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")
        except OSError:
            parts.append(f"{name}:missing")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cache_path(cfg: Any) -> Path:
    env = os.environ.get("DRUGAGENT_ENVELOPE_CACHE")
    if env:
        return Path(env).expanduser()
    root = Path(getattr(cfg, "resolved_repo_root", lambda: Path.cwd())())
    return root / ".drugagent" / "envelope.json"


def measure_envelope(cfg: Any, *, use_cache: bool = True, refresh: bool = False) -> CapabilityEnvelope:
    """Run the probe pipelines and report the achievable range of each metric.

    ``cfg`` is a :class:`drug_agent.llm.config.Config`. The probes go through
    ``run_pipeline`` unchanged, so whatever a user request would get, the probe
    gets: same Engine, same tools, same fixtures, same determinism setting.
    """
    from ..llm.bridge import run_pipeline
    from ..llm.schemas import ChemicalRequirements

    fingerprint = _fingerprint(cfg)
    cache = _cache_path(cfg)
    if use_cache and not refresh:
        cached = _read_cache(cache, fingerprint)
        if cached is not None:
            return cached

    started = time.time()
    warnings: list[str] = []
    tool_status: dict[str, str] = {}
    try:
        from .registry import tool_pool

        for card in tool_pool():
            tool_status[card.tool_id] = str(card.status)
    except Exception as exc:  # a broken optional backend must not break planning
        warnings.append(f"tool pool unavailable: {type(exc).__name__}")

    def _mocked(endpoint: str) -> tuple[str, bool]:
        tool = _ENDPOINT_TOOL.get(endpoint, "")
        status = tool_status.get(tool, "")
        return tool, status == "MOCKED"

    metrics: dict[str, MetricRange] = {}
    probe_info: dict[str, Any] = {}

    with measure(CAPABILITY_ID, "measure", backend="pipeline-probe") as m:
        # --- target-based probe -----------------------------------------
        try:
            run = run_pipeline(ChemicalRequirements.model_validate(_PROBE_TARGET), cfg)
        except Exception as exc:
            warnings.append(f"target probe failed: {type(exc).__name__}: {exc}")
            run = None
        if run is not None:
            scs = run.scorecards
            portfolio = run.portfolio or {}
            probe_info["target"] = {
                "run_id": run.run_id,
                "state": run.state,
                "n_candidates": len(run.candidates),
                "shortlist_size": len(portfolio.get("shortlist", [])),
            }
            shortlist_k = len(portfolio.get("shortlist", [])) or int(_PROBE_TARGET["shortlist_size"])
            for metric_key, endpoint in (
                ("max_pIC50", "pIC50"),
                ("median_pIC50", "pIC50"),
                ("qed_median", "qed"),
                ("sa_median", "sa_score"),
                ("herg_median", "hERG"),
            ):
                backend, mocked = _mocked(endpoint)
                metrics[metric_key] = _range(
                    metric_key,
                    _endpoint_values(scs, endpoint),
                    backend=backend,
                    mocked=mocked,
                    top_k=None if metric_key.startswith("max_") else shortlist_k,
                )
            hit, seen = _in_domain_fraction(scs)
            metrics["in_domain_fraction"] = MetricRange(
                key="in_domain_fraction",
                available=seen > 0,
                best=round(hit / seen, 4) if seen else None,
                median=round(hit / seen, 4) if seen else None,
                n=seen,
                backend="model.activity",
                note="" if seen else "no pIC50 predictions on this deployment",
            )
            div = portfolio.get("diversity")
            metrics["mean_pairwise_distance"] = MetricRange(
                key="mean_pairwise_distance",
                available=div is not None,
                best=round(float(div), 4) if div is not None else None,
                median=round(float(div), 4) if div is not None else None,
                n=len(portfolio.get("shortlist", [])),
                backend="chem.standardize",
            )
            clusters = portfolio.get("n_clusters")
            metrics["n_scaffold_clusters"] = MetricRange(
                key="n_scaffold_clusters",
                available=clusters is not None,
                best=float(clusters) if clusters is not None else None,
                median=float(clusters) if clusters is not None else None,
                n=len(scs),
                backend="chem.standardize",
            )
            metrics["shortlist_filled"] = MetricRange(
                key="shortlist_filled",
                available=True,
                best=1.0,
                median=1.0,
                n=len(portfolio.get("shortlist", [])),
                backend="ranking",
                note="the ranking plane fills the shortlist whenever candidates survive filters",
            )

        # --- constrained probe (what a normal brief's filters really pass) --
        try:
            spec = {
                **_PROBE_TARGET,
                "title": "capability probe (constrained)",
                "property_constraints": _STANDARD_CONSTRAINTS,
            }
            constrained_run = run_pipeline(ChemicalRequirements.model_validate(spec), cfg)
        except Exception as exc:
            warnings.append(f"constrained probe failed: {type(exc).__name__}: {exc}")
            constrained_run = None
        if constrained_run is not None:
            cscs = constrained_run.scorecards
            n_pass = sum(1 for sc in cscs if sc.get("hard_constraints_passed"))
            rate = round(n_pass / len(cscs), 4) if cscs else None
            probe_info["constrained"] = {
                "run_id": constrained_run.run_id,
                "n_candidates": len(constrained_run.candidates),
                "pass_rate": rate,
                "constraints": [f"{c['property']}{c['op']}{c['value']}" for c in _STANDARD_CONSTRAINTS],
            }
            metrics["hard_constraint_pass_rate"] = MetricRange(
                key="hard_constraint_pass_rate",
                available=bool(cscs),
                best=rate,
                median=rate,
                n=len(cscs),
                backend="probe (Lipinski + zero structural alerts, unseeded library)",
                note=(
                    "measured on the UNSEEDED library under a standard Lipinski + no-alerts "
                    "filter. A run whose seeds are themselves drug-like enumerates analogues "
                    "that pass more often, and has been observed above this figure, so treat it "
                    "as a floor for a well-seeded run rather than a ceiling. What it does "
                    "establish is that 1.0 is not reachable on this library."
                ),
            )

        # --- phenotype probe --------------------------------------------
        try:
            pheno = run_pipeline(ChemicalRequirements.model_validate(_PROBE_PHENOTYPE), cfg)
        except Exception as exc:
            warnings.append(f"phenotype probe failed: {type(exc).__name__}: {exc}")
            pheno = None
        if pheno is not None:
            probe_info["phenotype"] = {
                "run_id": pheno.run_id,
                "state": pheno.state,
                "n_candidates": len(pheno.candidates),
            }
            backend, mocked = _mocked("restoration_score")
            pheno_k = len((pheno.portfolio or {}).get("shortlist", [])) or int(
                _PROBE_PHENOTYPE["shortlist_size"]
            )
            metrics["restoration_median"] = _range(
                "restoration_median",
                _endpoint_values(pheno.scorecards, "restoration_score"),
                backend=backend,
                mocked=mocked,
                top_k=pheno_k,
            )
        m["n_metrics"] = float(len(metrics))
        m["n_available"] = float(sum(1 for r in metrics.values() if r.available))

    for key, rng in metrics.items():
        if rng.mocked:
            warnings.append(f"{key} comes from a MOCKED backend ({rng.backend})")
    if "max_pIC50" in metrics and not metrics["max_pIC50"].available:
        warnings.append("no bioactivity data for the probe target; pIC50 criteria are unscorable here")

    envelope = CapabilityEnvelope(
        measured_at=time.time(),
        fingerprint=fingerprint,
        elapsed_s=time.time() - started,
        metrics=metrics,
        tool_status=tool_status,
        warnings=sorted(set(warnings)),
        probe={
            **probe_info,
            # WHAT WAS MEASURED. Both probes hard-code `"generator": "library"`
            # (see `_PROBE_TARGET` / `_PROBE_PHENOTYPE`), so every number here is
            # what the LIBRARY arm reaches. That was unstated while runs were
            # library-only too. Now that the master agent chooses the source, a
            # run sampling a de-novo model can exceed these figures without the
            # envelope being wrong — it was never a claim about that arm — and a
            # reader with no scope recorded cannot tell the two apart.
            "candidate_source": _PROBE_TARGET.get("generator", "library"),
        },
    )
    if use_cache:
        _write_cache(cache, envelope)
    return envelope


def _read_cache(path: Path, fingerprint: str) -> CapabilityEnvelope | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("version") != ENVELOPE_VERSION or raw.get("fingerprint") != fingerprint:
        return None
    metrics = {k: MetricRange(**v) for k, v in (raw.get("metrics") or {}).items()}
    return CapabilityEnvelope(
        version=raw["version"],
        measured_at=raw.get("measured_at", 0.0),
        fingerprint=raw["fingerprint"],
        elapsed_s=raw.get("elapsed_s", 0.0),
        metrics=metrics,
        tool_status=raw.get("tool_status", {}),
        warnings=raw.get("warnings", []),
        probe=raw.get("probe", {}),
    )


def _write_cache(path: Path, envelope: CapabilityEnvelope) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(envelope.as_dict(), indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass  # a cache that cannot be written is not a reason to fail planning


def profile() -> CapabilityProfile:
    """The tool card for the pool. Always AVAILABLE: it needs only the platform."""
    return CapabilityProfile(
        capability_id=CAPABILITY_ID,
        title="Deployment capability envelope",
        description=(
            "Probes the real pipeline and reports the achievable range of every success "
            "metric on this machine, marking values that come from mocked backends. Exists "
            "so a plan can set targets this deployment can actually reach."
        ),
        backend="pipeline-probe",
        status=CapabilityStatus.AVAILABLE,
        determinism=Determinism.DETERMINISTIC,
        cost_class=CostClass.MODERATE,
        provides_mock=False,
        input_schema_ref=None,
        output_schema_ref=None,
        runtime_requirements={"pipeline": "drug_agent Engine", "fixtures": "fixtures/data"},
        known_failure_modes=[
            "no bioactivity fixture for the probe target (pIC50 range unavailable)",
            "a metric whose backend is MOCKED describes the stand-in, not the real method",
        ],
        validation_checks=[
            "every reported range names its backend",
            "mocked backends are flagged and never presented as measurements of chemistry",
            "a metric with no data is reported unavailable rather than zero",
        ],
        license="Apache-2.0",
    )


__all__ = [
    "CAPABILITY_ID",
    "ENVELOPE_VERSION",
    "CapabilityEnvelope",
    "MetricRange",
    "measure_envelope",
    "profile",
]
