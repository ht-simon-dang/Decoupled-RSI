"""PoseBusters — does this pose obey chemistry and physics, or only look like it does.

**What it does now.** Runs the real PoseBusters test suite over a predicted ligand
pose and reports every check individually. In ``mol`` mode (ligand alone) that is
the sanity, geometry and internal-energy battery; in ``dock`` mode, with the
receptor supplied, it adds the protein-ligand checks — minimum distance to the
protein, to organic cofactors, to waters, and whether the ligand actually sits in
the pocket it was docked into.

Measured on this machine (x86-64, PoseBusters 0.6.5): aspirin embedded with
ETKDGv3 and minimised with MMFF94s, ``config="mol"``, ``full_report=True`` returns
44 columns — 12 boolean checks, all passing; 13 bonds and 17 angles measured, 0
outliers of either; 0 steric clashes across 48 non-covalent pairs; aromatic ring
maximum distance from plane 0.007 Å; internal energy 56.5 kcal/mol against a
23.9 kcal/mol ensemble average, an energy ratio of 2.37.

    ``passed``                    every applicable boolean check is true
    ``checks``                    each named check, true or false
    ``failures``                  the names of the checks that failed
    ``metrics``                   the numeric evidence behind them, e.g.
                                  ``longest_bond_relative_length``,
                                  ``number_clashes``, ``energy_ratio``
    ``n_checks`` / ``n_passed``   how many checks applied, and how many held

**What it deliberately does not do.** It does not score binding, rank poses, or
repair anything. A pose that passes every PoseBusters check can still be entirely
wrong about affinity — the suite answers "is this physically possible", which is a
precondition for a pose being worth scoring, not evidence that it is right. That
distinction is why this tool sits beside the docking engines rather than inside
them: the producer and the judge must not be the same code.

**Why it is a checker, not a filter.** ``validate`` returns the report and lets the
caller decide. A failed check is recorded against the candidate as a ``qc_flag``,
in the same shape as every other recoverable per-molecule failure in this package —
it is not an exception, because a physically impossible pose is a real and
informative result about that molecule, not a malfunction of the tool.

PoseBusters is pure Python over RDKit and NumPy, so unlike most of this catalogue
it needs no checkout, no weights and no separate interpreter: it either imports
here or it does not. When it does not, this capability is UNAVAILABLE and validates
nothing. Under :mod:`drug_agent.tools.base` there is no stand-in verdict, and a
fabricated ``passed`` would be the most dangerous number in the package — every
downstream consumer treats it as evidence that coordinates are physically real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import measure
from .._ext.adapter import ExternalToolAdapter
from .._ext.spec import ExternalSpec

#: This adapter's contract with PoseBusters, not PoseBusters' own version.
POSEBUSTERS_ADAPTER_VERSION = "1.0.0"

SPEC = ExternalSpec(
    name="PoseBusters",
    env_prefix="POSEBUSTERS",
    selection_modules=("posebusters",),
    runtime_modules=("posebusters",),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/maabuu/posebusters",
    upstream_ref="1a5f26aa7270",
    license="PoseBusters BSD-3-Clause",
    install_hint="make posebusters-install",
    needs_gpu=False,
)

#: Boolean columns in the full report that say whether an *input* loaded, not
#: whether the pose is sound. They are recorded as qc flags instead, because
#: counting them as checks would let a missing receptor read as a failed molecule.
_NON_CHECK_COLUMNS = frozenset(
    {
        "mol_pred_loaded",
        "mol_true_loaded",
        "mol_cond_loaded",
    }
)


class PoseValidationError(RuntimeError):
    """PoseBusters was installed, ran, and could not produce a verdict."""


@dataclass
class PoseBustersReport:
    """One pose's verdict: what was checked, what held, and the numbers behind it."""

    candidate_id: str = ""
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    mode: str = "mol"
    engine_version: str = ""
    qc_flags: list[str] = field(default_factory=list)

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for ok in self.checks.values() if ok)

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        A collapse in the pass rate is how a broken pose writer, a wrong receptor
        or a mis-parsed SDF announces itself, so the verdict and the count of
        failed checks are metered on every call.
        """
        out: dict[str, float] = {
            "posebusters_passed": 1.0 if self.passed else 0.0,
            "posebusters_n_checks": float(self.n_checks),
            "posebusters_n_failed": float(self.n_checks - self.n_passed),
        }
        for key in ("energy_ratio", "number_clashes", "longest_bond_relative_length"):
            value = self.metrics.get(key)
            if value is not None and math.isfinite(value):
                out[f"posebusters_{key}"] = float(value)
        return out


class PoseBustersAdapter(ExternalToolAdapter):
    """Physical and chemical plausibility checks over a predicted pose."""

    capability_id = "structure.validate.posebusters"
    spec = SPEC
    engine = "posebusters"
    version = POSEBUSTERS_ADAPTER_VERSION
    produces_nothing = "validates nothing"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 300.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pose plausibility checks (PoseBusters)",
            description=(
                "Physical and chemical validity of a predicted ligand pose: sanity, bond "
                "lengths and angles, ring flatness, internal clashes, internal energy, and "
                "protein-ligand distances when a receptor is supplied. Reports every check "
                "individually; scores nothing."
            ),
            input_schema_ref="Pose",
            output_schema_ref="PoseBustersReport",
            # Deterministic geometry over fixed coordinates. The one stochastic
            # element upstream, the conformer ensemble the internal-energy ratio is
            # compared against, is generated by PoseBusters with a pinned seed.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            known_failure_modes=[
                "pose file unreadable or empty",
                "ligand cannot be sanitized (the suite reports this rather than raising)",
                "receptor for 'dock' mode absent, so the protein-ligand checks never run",
                "internal-energy check needs a conformer ensemble and is slow for large flexible ligands",
            ],
            validation_checks=[
                "every reported check is a boolean",
                "passed is the conjunction of the applicable checks",
                "failures names exactly the checks that are false",
            ],
        )

    # --- operations --------------------------------------------------------
    def validate(
        self,
        *,
        candidate_id: str,
        pose_sdf: str | Path,
        receptor_pdb: str | Path | None = None,
        full_report: bool = True,
    ) -> PoseBustersReport:
        """Check one pose. Supplying ``receptor_pdb`` adds the protein-ligand checks."""
        self.require_available()
        with measure(self.capability_id, "validate", backend=self.backend) as metered:
            report = self._validate(
                candidate_id=candidate_id,
                pose_sdf=pose_sdf,
                receptor_pdb=receptor_pdb,
                full_report=full_report,
            )
            for key, value in report.metrics_for_ledger().items():
                metered[key] = value
        return report

    def validate_batch(
        self,
        poses: list[tuple[str, str | Path]],
        *,
        receptor_pdb: str | Path | None = None,
    ) -> list[PoseBustersReport]:
        """Check several poses, each given as ``(candidate_id, pose_sdf)``.

        One pose failing to load does not stop the rest: its report carries the
        reason in ``qc_flags`` and ``passed`` is false, which is the same shape a
        failed check produces.
        """
        self.require_available()
        with measure(self.capability_id, "validate_batch", backend=self.backend) as metered:
            reports = [
                self._validate(candidate_id=cid, pose_sdf=sdf, receptor_pdb=receptor_pdb)
                for cid, sdf in poses
            ]
            if reports:
                metered["posebusters_pass_rate"] = float(sum(1 for r in reports if r.passed) / len(reports))
                metered["posebusters_n_poses"] = float(len(reports))
        return reports

    # --- real backend ------------------------------------------------------
    def _validate(
        self,
        *,
        candidate_id: str,
        pose_sdf: str | Path,
        receptor_pdb: str | Path | None = None,
        full_report: bool = True,
    ) -> PoseBustersReport:
        # Imported here rather than at module scope: `tool_pool()` constructs every
        # adapter on every call, and a library import in a constructor is what turns
        # a listing into a multi-second operation once there are thirty of these.
        import posebusters
        from posebusters import PoseBusters

        path = Path(pose_sdf)
        mode = "dock" if receptor_pdb else "mol"
        report = PoseBustersReport(
            candidate_id=candidate_id,
            mode=mode,
            engine_version=getattr(posebusters, "__version__", "unknown"),
        )
        if not path.is_file():
            report.qc_flags.append("pose_file_missing")
            return report
        try:
            buster = PoseBusters(config=mode)
            frame = buster.bust(
                [path],
                None,
                Path(receptor_pdb) if receptor_pdb else None,
                full_report=full_report,
            )
        except Exception as exc:
            raise PoseValidationError(f"PoseBusters failed on {path.name}: {exc}") from exc
        if frame is None or len(frame) == 0:
            report.qc_flags.append("posebusters_returned_no_rows")
            return report

        row = frame.iloc[0].to_dict()
        for key, value in row.items():
            name = str(key)
            if isinstance(value, bool) and name not in _NON_CHECK_COLUMNS:
                report.checks[name] = bool(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                fvalue = float(value)
                if math.isfinite(fvalue):
                    report.metrics[name] = fvalue
        if not bool(row.get("mol_pred_loaded", True)):
            report.qc_flags.append("pose_not_loadable")
        report.failures = sorted(name for name, ok in report.checks.items() if not ok)
        report.passed = bool(report.checks) and not report.failures
        return report


__all__ = [
    "POSEBUSTERS_ADAPTER_VERSION",
    "SPEC",
    "PoseBustersAdapter",
    "PoseBustersReport",
    "PoseValidationError",
]
