"""Held-out conformance suites for tools the loop adds or changes.

This module exists because of one measurement. *Beyond Task Completion: A
Verification-vs.-Conformance Gap in Tool-Evolving Agents* (arXiv:2604.00392, KDD
2026 workshop) found that across 222 tools synthesised by agents and preserved
for reuse, **96.8% scored 0.00 on held-out per-tool correctness** — while passing
the in-session verifier that certified them. Passing the call that motivated a
tool certifies *that call*. It says nothing about the capability.

So: a candidate that adds or changes a tool must clear a suite written against
the **capability**, on inputs the tool was never shown, and the suite is authored
here — inside the integrity manifest — not by the candidate.

Each case is a plain function returning ``(passed, detail)``. No mocking of the
thing under test: a conformance case calls the real tool the way an agent would.

Suites are keyed by ``capability_id``. A candidate that changes a tool with no
suite is **rejected**, not waved through: "we have no way to check this" is a
reason to refuse, not a reason to accept.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable
from typing import Any

#: One case: a name, what it establishes, and the check itself.
ConformanceCase = tuple[str, str, Callable[[], tuple[bool, str]]]


# ---------------------------------------------------------------------------
# platform.envelope
# ---------------------------------------------------------------------------
def _envelope_reports_a_direction_aware_best() -> tuple[bool, str]:
    """A minimised metric's best value is its lowest, not its highest.

    Held out from the motivating case: the finding that produced this tool was
    about `max_pIC50`, a maximised metric. Getting the direction wrong there is
    invisible; getting it wrong on `sa_median` hands the planner a target in the
    wrong half of the range.
    """
    from ...tools.envelope_tool import _range

    up = _range("qed_median", [0.1, 0.5, 0.9], backend="t", mocked=False, top_k=1)
    down = _range("sa_median", [2.0, 5.0, 9.0], backend="t", mocked=False, top_k=1)
    ok = up.best == 0.9 and down.best == 2.0
    return ok, f"qed best={up.best} (want 0.9); sa best={down.best} (want 2.0)"


def _envelope_median_metrics_are_not_pool_optima() -> tuple[bool, str]:
    """A shortlist-median metric cannot reach the single best molecule's value."""
    from ...tools.envelope_tool import _range

    values = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    rng = _range("sa_median", values, backend="t", mocked=False, top_k=3)
    ok = rng.best == 3.0 and rng.minimum == 2.0
    return ok, f"best={rng.best} (want 3.0, the median of 2/3/4), minimum={rng.minimum}"


def _envelope_reports_absence_rather_than_zero() -> tuple[bool, str]:
    """A metric with no data must be unavailable. Zero is a measurement."""
    from ...tools.envelope_tool import _range

    rng = _range("max_pIC50", [], backend="t", mocked=False)
    ok = rng.available is False and rng.best is None
    return ok, f"available={rng.available}, best={rng.best}"


def _envelope_flags_a_mocked_backend() -> tuple[bool, str]:
    """A target set against a deterministic stand-in must be visibly so."""
    from ...tools.envelope_tool import CapabilityEnvelope, MetricRange

    env = CapabilityEnvelope(
        metrics={
            "sa_median": MetricRange(
                key="sa_median", available=True, best=5.1, median=6.6, backend="synth.assess", mocked=True
            )
        }
    )
    block = env.as_prompt_block()
    ok = "MOCKED" in block
    return ok, "the prompt block " + ("names the mock" if ok else "hides the mock")


def _envelope_runs_on_a_deployment_it_was_not_tuned_for() -> tuple[bool, str]:
    """The real probe, end to end, with the cache bypassed.

    The held-out part: it is asserted only that every *available* metric carries
    an achievable value and names a backend — not that any particular number
    comes back. A suite that pinned the numbers would fail the moment the
    fixtures changed, which is a different thing from the tool being wrong.
    """
    from ...llm.config import Config
    from ...tools.envelope_tool import measure_envelope

    with contextlib.redirect_stdout(io.StringIO()):
        env = measure_envelope(Config(), use_cache=False)
    if not env.metrics:
        return False, "the probe produced no metrics at all"
    for key, rng in env.metrics.items():
        if rng.available and rng.best is None:
            return False, f"{key} is available but carries no achievable value"
        if rng.available and not rng.backend:
            return False, f"{key} does not name the backend that produced it"
    return (
        True,
        f"{len(env.metrics)} metrics, {sum(1 for r in env.metrics.values() if r.available)} available",
    )


def _envelope_prompt_block_is_empty_when_there_is_nothing_to_say() -> tuple[bool, str]:
    """No metrics must render nothing, not a table of blanks.

    A planner with no envelope is the pre-existing behaviour, which is a strictly
    better failure than a prompt full of empty cells.
    """
    from ...tools.envelope_tool import CapabilityEnvelope

    ok = CapabilityEnvelope().as_prompt_block() == ""
    return ok, "empty envelope renders " + ("nothing" if ok else "a table")


SUITES: dict[str, list[ConformanceCase]] = {
    "platform.envelope": [
        (
            "direction_aware_best",
            "a minimised metric's best is its lowest value",
            _envelope_reports_a_direction_aware_best,
        ),
        (
            "median_is_not_the_optimum",
            "a shortlist-median metric reports the median of the best k",
            _envelope_median_metrics_are_not_pool_optima,
        ),
        (
            "absence_is_not_zero",
            "a metric with no data is unavailable",
            _envelope_reports_absence_rather_than_zero,
        ),
        ("mocks_are_visible", "a value from a mocked backend says so", _envelope_flags_a_mocked_backend),
        (
            "empty_renders_nothing",
            "an empty envelope produces no prompt block",
            _envelope_prompt_block_is_empty_when_there_is_nothing_to_say,
        ),
        (
            "real_probe",
            "the probe runs against this deployment and reports coherently",
            _envelope_runs_on_a_deployment_it_was_not_tuned_for,
        ),
    ],
}


def run_suite(capability_id: str) -> dict[str, Any]:
    """Run one capability's conformance suite. Never raises."""
    cases = SUITES.get(capability_id)
    if cases is None:
        return {
            "capability_id": capability_id,
            "has_suite": False,
            "passed": False,
            "reason": (
                "no held-out conformance suite exists for this capability. A tool that can only "
                "be checked on the example that motivated it has not been checked — see "
                "arXiv:2604.00392. Write a suite in evaluation/conformance.py first."
            ),
            "cases": [],
        }
    started = time.time()
    rows: list[dict[str, Any]] = []
    for name, establishes, fn in cases:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        rows.append({"case": name, "establishes": establishes, "passed": bool(ok), "detail": detail})
    return {
        "capability_id": capability_id,
        "has_suite": True,
        "passed": all(r["passed"] for r in rows),
        "n_cases": len(rows),
        "n_failed": sum(1 for r in rows if not r["passed"]),
        "elapsed_s": round(time.time() - started, 3),
        "cases": rows,
    }


#: Which adapter module implements which capability. Explicit because it is not
#: derivable — tool files are named for the library they bind (``rdkit_tool.py``
#: implements ``chem.standardize``, ``singlecell_tool.py`` implements
#: ``perturb.assess``). A new tool adds a line here as part of registering it,
#: and a *renamed* one must be updated here or its conformance suite silently
#: stops running — ``tests/test_pools.py`` asserts this table still resolves.
#:
#: Keys are paths **below** ``tools/``, not basenames. They were basenames while
#: every adapter sat in one flat directory; the external catalogue put them in
#: domain subpackages, where two packages may legitimately hold files of the same
#: name, and a basename key would then map one of them to the other's capability.
CAPABILITY_MODULES: dict[str, str] = {
    "rdkit_tool.py": "chem.standardize",
    "sklearn_tool.py": "model.activity",
    "pfm_tool.py": "molecule.generate.pfm",
    "vina_tool.py": "structure.dock",
    "mmff_tool.py": "structure.simulate",
    "safety_tool.py": "safety.assess",
    "safc_tool.py": "activity.safc",
    "synthesis_tool.py": "synth.assess",
    "singlecell_tool.py": "perturb.assess",
    "envelope_tool.py": "platform.envelope",
    # De-novo generation (ADR-0012).
    "generation/reinvent4_tool.py": "molecule.generate.reinvent4",
    "generation/synformer_tool.py": "molecule.generate.synformer",
    "generation/prexsyn_tool.py": "molecule.generate.prexsyn",
    "generation/reasyn_tool.py": "molecule.generate.reasyn",
    "generation/rxnflow_tool.py": "molecule.generate.rxnflow",
    "generation/flowr_root_tool.py": "molecule.generate.flowr_root",
    "generation/flowr_tool.py": "molecule.generate.flowr",
    "generation/drugflow_tool.py": "molecule.generate.drugflow",
    "generation/molcraft_tool.py": "molecule.generate.molcraft",
    "generation/moljo_tool.py": "molecule.generate.moljo",
    "generation/molpilot_tool.py": "molecule.generate.molpilot",
    "generation/pocketxmol_tool.py": "molecule.generate.pocketxmol",
    "generation/omtra_tool.py": "molecule.generate.omtra",
    "generation/shepherd_tool.py": "molecule.generate.shepherd",
    "generation/pilot_tool.py": "molecule.generate.pilot",
    "generation/flowmol3_tool.py": "molecule.generate.flowmol3",
    "generation/pocket2mol_tool.py": "molecule.generate.pocket2mol",
    "generation/targetdiff_tool.py": "molecule.generate.targetdiff",
    "generation/pocketflow_tool.py": "molecule.generate.pocketflow",
    # Structure.
    "structure/gnina_tool.py": "structure.dock.gnina",
    "structure/diffdock_tool.py": "structure.dock.diffdock",
    "structure/flowdock_tool.py": "structure.dock.flowdock",
    "structure/boltz2_tool.py": "structure.complex.boltz2",
    "structure/posebusters_tool.py": "structure.validate.posebusters",
    # Properties, retrosynthesis, free energy, benchmarking.
    "properties/admetai_tool.py": "property.admet.admetai",
    "properties/chemprop_tool.py": "property.predict.chemprop",
    "properties/unimol_tool.py": "property.represent.unimol",
    "retrosynthesis/aizynthfinder_tool.py": "synth.route.aizynthfinder",
    "freeenergy/openfe_tool.py": "binding.freeenergy.openfe",
    "benchmarking/posebench_tool.py": "benchmark.pose.posebench",
    # Target evidence: the layer that precedes any chemistry. Both read a
    # network-fetched snapshot from disk and never open a socket at run time.
    "target/opentargets_tool.py": "target.evidence.opentargets",
    "target/depmap_tool.py": "target.essentiality.depmap",
    # Binding-site detection. These are what let the platform find a pocket on an
    # arbitrary structure, rather than depending on one hand-prepared file.
    "pockets/fpocket_tool.py": "pocket.detect.fpocket",
    "pockets/p2rank_tool.py": "pocket.detect.p2rank",
    "pockets/vnegnn_tool.py": "pocket.detect.vnegnn",
    "structure/experimental_tool.py": "structure.experimental",
    "structure/protenix_tool.py": "structure.predict.protenix",
    "generation/synflownet_tool.py": "molecule.generate.synflownet",
}


def capabilities_touched(candidate: Any) -> list[str]:
    """Which tool capabilities a candidate adds or changes.

    Read from the paths it declares and from the patch text, so a candidate
    cannot avoid conformance by describing its change differently in prose.
    """
    from ...tools.registry import TOOL_OPERATIONS

    touched: set[str] = set()
    blobs = [str(p) for p in (candidate.files_changed or [])] + [str(candidate.patch or "")]
    haystack = " ".join(blobs)
    for module, capability_id in CAPABILITY_MODULES.items():
        if f"tools/{module}" in haystack:
            touched.add(capability_id)
    for capability_id in TOOL_OPERATIONS:
        if capability_id in haystack:
            touched.add(capability_id)
    # A brand-new adapter module has no entry yet; naming it is the point.
    #
    # Matching is on the path *below* ``tools/``, not on the basename, because the
    # external catalogue put adapters in domain subpackages and two packages may
    # legitimately hold a file with the same name.
    for blob in blobs:
        stripped = blob.strip()
        if "tools/" not in stripped or not stripped.endswith(".py"):
            continue
        rel = stripped.rsplit("tools/", 1)[-1]
        if rel in CAPABILITY_MODULES:
            continue
        if rel.startswith("_ext/") or rel.endswith("/__init__.py"):
            continue  # shared machinery and domain abstractions bind no capability
        if rel in ("registry.py", "base.py", "__init__.py", "llm_provider_tool.py"):
            continue
        touched.add(f"unregistered:{rel}")
    return sorted(touched)


def run_for_candidate(candidate: Any) -> dict[str, Any]:
    """Conformance for every capability a candidate touches.

    A candidate that touches no tool gets ``required=False`` and an empty result
    — honest, and not a pass it did not earn.
    """
    capabilities = capabilities_touched(candidate)
    if not capabilities:
        return {
            "required": False,
            "capabilities": [],
            "passed": True,
            "note": "this candidate changes no tool implementation",
        }
    suites = {cid: run_suite(cid) for cid in capabilities}
    return {
        "required": True,
        "capabilities": capabilities,
        "passed": all(s["passed"] for s in suites.values()),
        "suites": suites,
    }


__all__ = ["SUITES", "ConformanceCase", "capabilities_touched", "run_for_candidate", "run_suite"]
