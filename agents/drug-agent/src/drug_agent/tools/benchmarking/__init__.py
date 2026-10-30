"""Benchmarks: how good is a *method*, measured over targets it did not choose.

Every other domain in this package is asked about a molecule. A generator is asked
for candidates, a docking engine for a pose, a property model for an endpoint, and
the answer travels attached to that one molecule. Benchmarking asks a different
question and returns a different kind of object: given a method that has already
run over a fixed set of targets, how often was it right, and how often was it right
for reasons that survive inspection.

That difference earns a domain of its own, because the failure modes are not the
ones the scientific tools have. A property model fails by returning a wrong number
for a molecule. A benchmark fails by returning a *flattering* number for a method,
and it has three standard ways to do that:

**Silent attrition.** The scoring harness can only score targets the method
actually produced a file for. Drop the rest and a method that solved 40 of 428
targets, and crashed on the other 388, reports the same success rate as one that
solved 40 of 40. So :class:`MethodBenchmark` carries ``n_targets_expected`` — the
size of the target list the method was given — next to the number it was scored on,
and publishes ``coverage`` as a first-class metric. A rate with no coverage beside
it is not a result.

**One headline number.** "RMSD below 2 Å" counts a pose that is in the right place
and physically impossible as a success. That is why the upstream harnesses report
accuracy and physical validity as two numbers rather than one, and why every
outcome here keeps its per-check booleans: ``success_rate`` is the conjunction,
``rmsd_within_2a_rate`` is the looser number, and both are visible at once so
nobody has to guess which one a table meant.

**Borrowed provenance.** A benchmark result inherits the terms of everything it
consumed — the dataset, the receptor structures, and the predictions being scored.
Those are usually three different licences, and at least one of them is usually not
the benchmark's own. The adapter records what it consumed; the capability card's
``data_restrictions`` says what an operator has to check before a number leaves the
building.

:class:`BenchmarkToolAdapter` owns the part that must not differ between harnesses:
:meth:`summarise`, which turns per-target rows into rates computed from real counts.
A backend is not allowed to report its own success rate, for the same reason a
generator is not allowed to report its own validity.

**A missing check is not a passed check.** When a scoring table has no value for a
validity column — the pose failed to load, the check never ran — this counts the
target as not valid and flags it, rather than letting an absent value pass through
a truthiness test. That is deliberately stricter than a plain "all columns true"
over a table with holes in it, so ``n_targets_with_missing_checks`` is reported
alongside the rates: a divergence from an upstream-published number should be
visible, not silent.

**A benchmark number is not a capability.** Learning here that some docking method
scores well on PoseBusters Benchmark does not make that method runnable on this
machine, and nothing in this package may treat it as though it did. These adapters
report on methods; they do not become them.

**Refusal.** A harness with no backend, no dataset or no predictions to score
reports UNAVAILABLE and raises :class:`~drug_agent.tools.base.ToolUnavailable` from
``require_available()``, per :mod:`drug_agent.tools.base`. There is no partial
benchmark and no estimated rate: a success rate assembled from anything other than
scored targets would be the easiest number in this package to mistake for evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .._ext.adapter import ExternalToolAdapter


class BenchmarkError(RuntimeError):
    """The selected benchmark harness ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the harness was
    installed, the evaluation started, and it did not produce a usable scoring
    table — a missing dataset, a predictions directory with nothing in it, a
    scoring crash.
    """


@dataclass
class TargetOutcome:
    """One benchmark target, scored for one method.

    ``checks`` holds the harness' own validity columns verbatim, minus the accuracy
    column, so ``pb_valid`` is a conjunction over exactly the checks the harness
    defines for this dataset rather than a set this package invented.
    """

    target_id: str
    #: Symmetry-corrected RMSD to the crystal ligand, in Ångström. ``None`` when the
    #: harness scored the target but could not produce a distance.
    rmsd: float | None = None
    #: The harness' accuracy verdict for this target, kept separate from the
    #: validity checks because it is the one column that is about being *right*
    #: rather than being *possible*.
    rmsd_within_2a: bool = False
    #: Validity checks by their upstream column name, accuracy column excluded.
    checks: dict[str, bool] = field(default_factory=dict)
    #: Recoverable per-target problems: a check with no value, a missing file.
    qc_flags: list[str] = field(default_factory=list)

    @property
    def pb_valid(self) -> bool:
        """Whether every validity check the harness applied to this target held."""
        return bool(self.checks) and all(self.checks.values())

    @property
    def success(self) -> bool:
        """Accurate *and* physically valid — the conjunction, not either half."""
        return self.rmsd_within_2a and self.pb_valid


@dataclass
class MethodBenchmark:
    """What one method scored on one dataset, with the counts behind every rate."""

    method: str = ""
    dataset: str = ""
    #: How many targets the method was asked for, read from the harness' own input
    #: table. The denominator of ``coverage``; never the denominator of a rate.
    n_targets_expected: int = 0
    targets: list[TargetOutcome] = field(default_factory=list)
    engine: str = ""
    engine_version: str = ""
    #: The scoring table this summary was read from, kept so a number can be traced
    #: back to the file that produced it.
    results_path: str = ""
    #: What the harness was actually asked to do — method, dataset, repeat index,
    #: baseline flags — so two summaries can be compared without guessing.
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Terms attached to the data this evaluation consumed, carried onto the result
    #: rather than left only on the capability card.
    data_restrictions: str = ""

    @property
    def n_scored(self) -> int:
        return len(self.targets)

    @property
    def coverage(self) -> float | None:
        """Fraction of the requested targets that were actually scored."""
        if self.n_targets_expected <= 0:
            return None
        return self.n_scored / self.n_targets_expected

    @property
    def n_targets_with_missing_checks(self) -> int:
        """Targets where at least one validity column had no value."""
        return sum(1 for t in self.targets if any(f.endswith("_missing") for f in t.qc_flags))

    @property
    def rmsd_within_2a_rate(self) -> float | None:
        """Accuracy alone. Counts physically impossible poses as successes."""
        if not self.targets:
            return None
        return sum(1 for t in self.targets if t.rmsd_within_2a) / len(self.targets)

    @property
    def pb_valid_rate(self) -> float | None:
        """Physical validity alone. Says nothing about being in the right place."""
        if not self.targets:
            return None
        return sum(1 for t in self.targets if t.pb_valid) / len(self.targets)

    @property
    def success_rate(self) -> float | None:
        """Accurate and valid. The number a method should be judged on."""
        if not self.targets:
            return None
        return sum(1 for t in self.targets if t.success) / len(self.targets)

    @property
    def median_rmsd(self) -> float | None:
        values = [t.rmsd for t in self.targets if t.rmsd is not None]
        return median(values) if values else None

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        Coverage is metered beside the rates on purpose: a harness that quietly
        stops finding prediction files reports an unchanged success rate over a
        shrinking denominator, and the count is the only place that shows.
        """
        out: dict[str, float] = {
            "benchmark_n_scored": float(self.n_scored),
            "benchmark_n_expected": float(self.n_targets_expected),
            "benchmark_n_missing_checks": float(self.n_targets_with_missing_checks),
        }
        for key, value in (
            ("benchmark_coverage", self.coverage),
            ("benchmark_rmsd_within_2a_rate", self.rmsd_within_2a_rate),
            ("benchmark_pb_valid_rate", self.pb_valid_rate),
            ("benchmark_success_rate", self.success_rate),
            ("benchmark_median_rmsd", self.median_rmsd),
        ):
            if value is not None:
                out[key] = float(value)
        return out


class BenchmarkToolAdapter(ExternalToolAdapter):
    """Shared per-target summarisation for the benchmark harnesses."""

    produces_nothing = "benchmarks nothing"

    #: Suffix a qc flag carries when a validity column had no value for a target.
    missing_check_suffix = "_missing"

    def summarise(
        self,
        rows: list[dict[str, Any]],
        *,
        method: str,
        dataset: str,
        n_targets_expected: int,
        engine_version: str,
        results_path: str = "",
        config: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        data_restrictions: str = "",
    ) -> MethodBenchmark:
        """Turn a harness' per-target rows into a :class:`MethodBenchmark`.

        Each row carries ``target_id``, an optional ``rmsd``, the accuracy verdict
        ``rmsd_within_2a``, and ``checks`` — the harness' validity columns. A check
        with no value becomes ``False`` plus a ``<column>_missing`` flag, so a hole
        in the scoring table lowers the validity rate instead of passing through it.
        """
        outcomes: list[TargetOutcome] = []
        for index, row in enumerate(rows):
            checks: dict[str, bool] = {}
            flags: list[str] = list(row.get("qc_flags") or [])
            for name, value in (row.get("checks") or {}).items():
                if value is None:
                    checks[str(name)] = False
                    flags.append(f"{name}{self.missing_check_suffix}")
                else:
                    checks[str(name)] = bool(value)
            rmsd = row.get("rmsd")
            outcomes.append(
                TargetOutcome(
                    target_id=str(row.get("target_id") or f"target_{index}"),
                    rmsd=float(rmsd) if rmsd is not None else None,
                    rmsd_within_2a=bool(row.get("rmsd_within_2a")),
                    checks=checks,
                    qc_flags=flags,
                )
            )
        return MethodBenchmark(
            method=method,
            dataset=dataset,
            n_targets_expected=int(n_targets_expected),
            targets=outcomes,
            engine=self.engine,
            engine_version=engine_version,
            results_path=results_path,
            config=dict(config or {}),
            warnings=list(warnings or []),
            data_restrictions=data_restrictions,
        )


__all__ = [
    "BenchmarkError",
    "BenchmarkToolAdapter",
    "MethodBenchmark",
    "TargetOutcome",
]
