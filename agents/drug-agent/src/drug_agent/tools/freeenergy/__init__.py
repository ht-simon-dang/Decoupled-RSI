"""Alchemical free energies: the binding difference between two ligands, computed rather than scored.

One upstream project lives here, and the package exists as much to keep it from
being misread as to run it.

**What a relative binding free energy is.** Two ligands A and B that share a common
core are joined by a non-physical path: a hybrid topology in which the atoms unique
to A are switched off while the atoms unique to B are switched on, across a ladder
of lambda windows sampled with replica exchange. Integrating the work along that
path with the ligand bound gives ΔG_complex(A→B); doing it again with the ligand in
water gives ΔG_solvent(A→B); and the thermodynamic cycle closes on the number a
medicinal chemist wants,

    ΔΔG_bind(A→B) = ΔG_complex(A→B) − ΔG_solvent(A→B),

the change in binding free energy on turning A into B. It is a physics-based
estimate carrying a statistical error bar, not a model score, and unlike a docking
score it is on the scale of an experiment: kcal/mol, where RT·ln 10 at 298.15 K —
1.364 kcal/mol — is one log unit of affinity.

**Why it is not a first-pass screen.** Every input has to be prepared before the
calculation means anything: a whole receptor, protonated, with the crystallographic
ligand and waters resolved and chain breaks capped; ligands posed in that
receptor's frame with consistent protonation and partial charges; and a common
core large enough that the atom map between neighbours covers most of each
molecule. The perturbation has to be small — a substituent, a ring atom, a linker
change — because the hybrid topology assumes the two end states share their atoms
and the estimate degrades as they stop doing so. Each edge is then a repeated
multi-nanosecond replica-exchange simulation in explicit solvent, run twice. This
package is for a handful of closely related, carefully prepared compounds around a
lead, after docking and property filters have done their work. It is never pointed
at raw generator output.

**What the base owns.** :class:`FreeEnergyToolAdapter` assembles the contract every
engine returns — a :class:`~drug_agent.contracts.structure.SimulationResult` per
edge with the real engine name on it — and holds the ΔΔG arithmetic above with the
two legs' uncertainties added in quadrature, so a second engine could not disagree
with the first about what a completed edge is. ``SimulationResult.engine`` defaults
to the literal string ``"mock"`` in the contract, so an adapter that forgets to set
it ships an object that claims to be fake; the base sets it from ``self.engine``,
and a test asserts it is never ``"mock"``.

**Refusal.** An engine with no backend reports UNAVAILABLE and computes no free
energies, per :mod:`drug_agent.tools.base`. Nothing cheaper stands in for it:
MMFF94s strain (``structure.simulate``) is a real and honestly-named calculation,
but it is not a binding free energy and this package never returns it as one.
"""

from __future__ import annotations

import math
from typing import Any

from ...contracts.enums import TrustLabel
from ...contracts.structure import SimulationResult
from .._ext.adapter import ExternalToolAdapter


class FreeEnergyError(RuntimeError):
    """The selected free-energy backend ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the engine was
    installed, the transformation started, and it did not return a usable
    estimate — a force-field parameterisation failure, a CUDA fault, a driver
    that exceeded its budget.
    """


class FreeEnergyToolAdapter(ExternalToolAdapter):
    """Shared result assembly and thermodynamic-cycle arithmetic for free-energy engines."""

    produces_nothing = "computes no free energies"
    #: The unit every observable named ``*_kcal_mol`` is reported in. Engines
    #: convert at the boundary; nothing downstream sees a kJ/mol.
    energy_unit = "kcal/mol"
    #: The protocol id written onto results when the operation supplies none.
    protocol_id: str = ""

    @staticmethod
    def finite(value: Any) -> float | None:
        """Coerce an upstream value to a finite float, or ``None``.

        MBAR reports a failed window as NaN and a Quantity's magnitude may be a
        numpy scalar; a NaN that reaches ``observables`` reads as a measurement
        that happens to be NaN rather than as an estimate that was not obtained.
        """
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def ddg_from_legs(
        dg_complex: float,
        dg_solvent: float,
        u_complex: float,
        u_solvent: float,
    ) -> tuple[float, float]:
        """Close the cycle: ``ΔΔG_bind = ΔG_complex − ΔG_solvent``.

        The two legs are independent simulations, so their uncertainties add in
        quadrature. This is the same arithmetic ``openfe gather`` performs
        (``openfecli/commands/gather.py::_get_ddgs``), kept here so every engine
        in this package closes the cycle the same way.
        """
        return dg_complex - dg_solvent, math.sqrt(u_complex * u_complex + u_solvent * u_solvent)

    def simulation_result(
        self,
        *,
        candidate_id: str,
        receptor_id: str | None,
        engine_version: str,
        n_steps: int,
        observables: dict[str, Any],
        converged: bool,
        protocol: str | None = None,
        qc_flags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_ref: str | None = None,
    ) -> SimulationResult:
        """One edge's outcome as the shared contract, with the real engine stamped on it.

        Observables that are not finite floats are dropped rather than written as
        NaN; their absence, together with ``converged=False`` and a ``qc_flag``, is
        how an edge that produced no estimate is reported.
        """
        clean: dict[str, float] = {}
        for key, value in observables.items():
            number = self.finite(value)
            if number is not None:
                clean[str(key)] = number
        return SimulationResult(
            candidate_id=candidate_id,
            receptor_id=receptor_id,
            protocol=protocol or self.protocol_id,
            engine=self.engine,
            engine_version=engine_version or "unknown",
            n_steps=int(n_steps),
            observables=clean,
            stability_score=None,
            converged=bool(converged),
            trust_label=TrustLabel.CALCULATED,
            qc_flags=list(qc_flags or []),
            artifact_ref=artifact_ref,
            metadata=dict(metadata or {}),
        )


__all__ = ["FreeEnergyError", "FreeEnergyToolAdapter"]
