"""Synthetic accessibility — RDKit's own SA score, and no invented route.

**Real, and now actually the thing it was named after.** ``sa_score`` is
:func:`sascorer.calculateScore`, the Ertl & Schuffenhauer synthetic-accessibility
score (*J. Cheminform.* 2009, **1**:8) that ships with RDKit under
``Contrib/SA_Score``. It is a fragment-contribution score: every atom environment
in the molecule is looked up in a table of ECFP4 fragment frequencies counted over
~1M PubChem compounds, and that average is corrected by explicit ring-complexity,
stereo, macrocycle and size terms. The scale is 1 (easy) to 10 (hard), which is
the direction and the range the ranking plane already normalizes ``sa_score`` on.

**Removed, and why it had to go.** The score reported here until now was a
home-made complexity proxy — ``1 + log10(n_atoms+1) + BertzCT/(8*n_atoms) +
0.5*n_stereo + 0.25*max(0, n_rings-2) + 0.8*(spiro+bridgehead) + 1.2*macrocycles``,
clamped into [1, 10] — carrying the backend string ``rdkit-sa``. It was not RDKit's
SA score, and it was checkably wrong in the direction that matters: it is a
*minimize* objective, so a molecule it over-penalizes is a molecule the shortlist
drops. Measured on this machine, proxy against the real score:

    aspirin 5.446 / **1.580**, caffeine 7.681 / **2.298**, imatinib 7.892 /
    **2.332**, ibuprofen 5.412 / **2.192**, the shipped sEH candidate
    ``O=C(O)c1ccc(-c2ccc(F)cc2F)cc1O`` 6.610 / **1.829**

The proxy called caffeine and imatinib "hard" and flagged them ``high_complexity``;
it also saturated at the 10.000 ceiling for taxol (**5.916**) and erythromycin
(**6.088**), so three different natural products came out indistinguishable. A
number that cannot separate taxol from erythromycin is not measuring synthetic
difficulty, and the whole time the real scorer was importable in this environment
and unused.

**The contrib import, and why it needs a ``sys.path`` hook.** ``sascorer`` is an
RDKit *contribution*, not a module of the ``rdkit`` package: it is installed as a
loose ``sascorer.py`` in ``RDConfig.RDContribDir/SA_Score`` with no package
``__init__.py`` above it, so ``import sascorer`` cannot resolve until that
directory is on ``sys.path``. Appending it is upstream's own documented usage, not
a workaround, and :func:`_sascorer` does it exactly once behind an
:func:`functools.lru_cache`.

**If it cannot load, this capability is UNAVAILABLE and scores nothing.** There is
no fallback, renamed or otherwise: the only fallback available was the proxy above,
and a wrong ``sa_score`` under any backend name is still a wrong number in a
ranking objective (ADR-0011). ``sascorer.py`` and its fragment table
``fpscores.pkl.gz`` ship inside the ``rdkit`` package this project already hard-
depends on, so their absence means a partial install rather than a missing
optional extra — and :func:`_sascorer` proves the path by scoring a molecule
rather than by resolving an import, because the fragment table is loaded lazily on
the first call and a truncated one would otherwise surface mid-run.

**Fabricated, and removed earlier.** ``_mock_route`` returned a ``SynthesisRoute``
whose every field was hash-derived: the step count was ``1 + int(sa)//2 +
hash(smiles) % 2``; each step's ``reaction_type`` was drawn from a fixed list of
six names by ``hash(smiles + i)``; ``reactants`` and ``product`` were left empty,
so a "step" was a label with no chemistry in it. For the same sEH candidate it
emitted "amide_coupling -> buchwald_hartwig -> reductive_amination ->
ester_hydrolysis -> reductive_amination" on a molecule that contains no amide, no
amine and no ester. A route is an object rather than a number, so it is gone
rather than mocked: ``routes`` is empty and ``best_route_id`` is ``None`` unless
:mod:`drug_agent.tools.retrosynthesis.aizynthfinder_tool` is wired in behind
``agent.synthesis``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from rdkit import Chem
from rdkit.Chem import RDConfig

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ..contracts.synthesis import SynthesisPlan
from ..observability.toolmeter import measure
from ._ext.adapter import NO_BACKEND
from .base import ToolAdapter

#: Attribution for the number, carried wherever the number is.
SA_METHOD = "Ertl & Schuffenhauer (2009) SA score, RDKit Contrib sascorer"

#: The molecule the readiness probe scores. Ethanol is the cheapest thing that
#: still exercises the whole path, fragment-table load included.
_PROBE_SMILES = "CCO"


def _contrib_dir() -> Path:
    """Where RDKit installs the SA_Score contribution on this interpreter."""
    return Path(RDConfig.RDContribDir) / "SA_Score"


@lru_cache(maxsize=1)
def _sascorer() -> ModuleType | None:
    """Import RDKit's ``sascorer`` and prove it can score, or return ``None``.

    Two things are deliberate here. The ``sys.path`` append is required because
    the SA_Score contribution is a loose module rather than a subpackage of
    ``rdkit`` (see this module's docstring). And the probe *calls*
    ``calculateScore`` instead of stopping at a successful import, because the
    import is not the step that fails quietly: ``calculateScore`` loads its
    ~2 MB fragment-contribution table (``fpscores.pkl.gz``) lazily on the first
    call, so a missing or truncated table would otherwise pass readiness and
    raise in the middle of a candidate batch. Paying for one ethanol here is what
    lets :meth:`SynthesisAdapter.available` be a fact rather than a hope.
    """
    contrib = str(_contrib_dir())
    if contrib not in sys.path:
        sys.path.append(contrib)
    try:
        import sascorer

        probe = Chem.MolFromSmiles(_PROBE_SMILES)
        if probe is None:  # pragma: no cover - RDKit that cannot parse CCO is broken
            return None
        float(sascorer.calculateScore(probe))
    except Exception:
        # Any failure — no module, no fragment table, an unpickleable table —
        # means the same thing to a caller: this tool has no score to give.
        return None
    return sascorer


def _unavailable_reason() -> str:
    """The exact missing artifact and the command that puts it back."""
    return (
        f"RDKit's SA_Score contribution did not load, so synth.assess scores nothing. "
        f"Expected {_contrib_dir() / 'sascorer.py'} with its fragment table "
        f"fpscores.pkl.gz beside it; both ship inside the rdkit package this project "
        f"hard-depends on, so an absence means a partial or non-standard RDKit install "
        f"rather than a missing optional extra. Fix with `make env`. There is no "
        f"fallback score: the proxy that used to stand in here was not the SA score and "
        f"mis-ranked drug-like molecules by 3-5 units (ADR-0011)."
    )


def _sa_label(score: float) -> str:
    """Coarse band for a reader, on the Ertl scale rather than the proxy's.

    The thresholds are unchanged, and they suit the real scale better than they
    ever suited the proxy: measured here, approved drugs land at 1.6-2.3, the sEH
    demo series at 1.8-3.5, and the two natural products that saturated the proxy
    at taxol 5.916 / erythromycin 6.088 — so "hard" now separates the molecules a
    chemist would call hard instead of catching caffeine.
    """
    if score < 3.5:
        return "easy"
    if score < 6.0:
        return "moderate"
    return "hard"


class SynthesisAdapter(ToolAdapter):
    capability_id = "synth.assess"

    def __init__(self, backend: str = "rdkit-sa") -> None:
        # An UNAVAILABLE card names no engine at all, or it claims something it
        # cannot do; `NO_BACKEND` is the constant the pool test pins that on.
        self.backend = backend if _sascorer() is not None else NO_BACKEND

    def available(self) -> bool:
        return _sascorer() is not None

    @property
    def profile(self) -> CapabilityProfile:
        ready = self.available()
        return CapabilityProfile(
            capability_id=self.capability_id,
            # 3.0.0, not 2.x: the number under `sa_score` changed backend, so a
            # scorecard from before this version is not differenceable against one
            # after it. The version is the machine-readable half of that warning.
            version="3.0.0",
            title="Synthetic accessibility",
            description=(
                "RDKit's Ertl-Schuffenhauer synthetic-accessibility score in [1, 10] "
                "(1 easy, 10 hard), from the fragment-contribution table shipped in "
                "RDKit's SA_Score contribution. Emits no retrosynthetic route: no "
                "retrosynthesis backend is configured."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="SynthesisPlan",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE if ready else CapabilityStatus.UNAVAILABLE,
            provides_mock=False,
            backend=self.backend,
            license="RDKit BSD-3-Clause; sascorer by Ertl & Schuffenhauer, J. Cheminform. 2009, 1:8",
            known_failure_modes=[
                "unparseable SMILES",
                "RDKit installed without its Contrib tree, or without fpscores.pkl.gz",
                "a fragment-contribution score is a literature-frequency estimate, not a "
                "route: an unprecedented but easily-made scaffold scores as hard",
            ],
            validation_checks=["sa in [1,10]", "aspirin scores below caffeine scores below taxol"],
            unavailable_reason=None if ready else _unavailable_reason(),
        )

    def assess(self, candidate_id: str, smiles: str) -> SynthesisPlan:
        self.require_available()
        with measure(self.capability_id, "assess", backend=self.backend) as metered:
            plan = self._assess(candidate_id, smiles)
            if plan.sa_score is not None:
                metered["sa_score"] = float(plan.sa_score)
        return plan

    def _assess(self, candidate_id: str, smiles: str) -> SynthesisPlan:
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return SynthesisPlan(
                candidate_id=candidate_id, sa_score=None, risk_factors=["unparseable_smiles"]
            )
        scorer = _sascorer()
        if scorer is None:  # pragma: no cover - require_available() already refused
            raise AssertionError("unreachable")
        sa = round(float(scorer.calculateScore(rdmol)), 3)
        risk = ["high_complexity"] if sa >= 6.0 else []
        # No route, and therefore no building-block availability: both were
        # hash-derived. `building_block_availability` stays None rather than 0.0,
        # which would read as "we checked and found none".
        return SynthesisPlan(
            candidate_id=candidate_id,
            sa_score=sa,
            sa_label=_sa_label(sa),
            routes=[],
            best_route_id=None,
            risk_factors=risk,
            building_block_availability=None,
            trust_label=TrustLabel.PREDICTED,
            limitation=f"{SA_METHOD}; no retrosynthetic route computed",
        )
