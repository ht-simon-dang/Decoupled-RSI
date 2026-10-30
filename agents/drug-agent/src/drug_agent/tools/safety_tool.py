"""Toxicity & safety: real substructure matching, and real pretrained tox endpoints.

**What was here, and why it was an ADR-0011 violation.** This module used to build
an 800-row synthetic feature matrix by sampling Gaussian noise around the
descriptor statistics of 32 embedded drug-like SMILES, *label* those rows from a
hand-written seven-weight linear rule plus a second Gaussian draw
(``(sigmoid(norm @ w + bias) + rng.normal(0, 0.15)) > 0.5``), fit a
``LogisticRegression`` under ``CalibratedClassifierCV`` to those invented labels,
and report the resulting probabilities as ``hERG``, ``AMES`` and
``hepatotoxicity``. The fit was real; the labels were the rule's own output, so
the model could only ever recover the rule, and the ``test_auc`` / ``brier`` /
``ece`` it published were a measurement of that recovery rather than of toxicity.
Its docstring said "synthetic, demonstration-only" — but the card said
``provides_mock=False`` and ``AVAILABLE``, the endpoints carried
``calibrated=True``, and the numbers reached ``herg_median``, the shortlist and
the reports as ordinary predictions. Nothing downstream could tell them from a
measurement, which is exactly the failure ADR-0011 names.

It was also checkably wrong, in both directions. Measured on this machine before
the change:

    aflatoxin B1  hERG 0.206, AMES 0.022, hepatotoxicity 0.120 -> ``no_signal``,
                  0 flags, "No safety signal detected across the panel" for a
                  Group-1 carcinogen and hepatotoxin
    dofetilide    hERG 0.145 -> not flagged, for a class-III antiarrhythmic whose
                  mechanism of action *is* hERG blockade

**What is here now.** Two real things and nothing else.

``structural_alerts`` is unchanged: deterministic RDKit substructure matching of
the six tox-relevant SMARTS in :data:`_TOX_ALERTS`. It extrapolates nothing, so it
is in-domain for any molecule RDKit can parse, and it is why this capability stays
``AVAILABLE`` on a host without ADMET-AI.

``hERG``, ``AMES`` and ``DILI`` come from
:class:`~drug_agent.tools.properties.admetai_tool.AdmetAiAdapter` — the released
ADMET-AI Chemprop ensembles, trained by their authors on the TDC datasets, with
published test-set metrics (hERG AUROC 0.839, AMES 0.882, DILI 0.881) that travel
on each endpoint's ``detail``. Nothing is fitted at call time. The same two
compounds, after:

    aflatoxin B1  AMES **0.850** (HIGH, flagged), DILI **0.949** (HIGH, flagged),
                  hERG 0.417 -> 2 flags from the panel, ``no_signal`` False
    dofetilide    hERG **0.919** (HIGH, flagged), DILI 0.717 (HIGH, flagged)
    ethanol       hERG 0.010, AMES 0.120, DILI 0.173 -> no panel flag, the benign
                  control still reads benign

**Why three endpoints and not ADMET-AI's nineteen.** ``safety.assess``'s contract
is these three plus alerts, and they are the ones the ranking plane and the LLM's
objective vocabulary name (``hERG`` is a demo objective; ``herg_median`` is a
sealed evaluator metric). The full nineteen-endpoint toxicity panel is one call
away under its own capability id, ``property.admet.admetai``, where the reader can
see which model produced what. One of those nineteen, ``LD50_Zhu``, is a
regression in log(1/(mol/kg)) rather than a probability, and pushing it through
this capability would break the ``[0, 1]`` invariant
:class:`~drug_agent.agents.specialists.safety_agent.SafetyAgent` re-checks
independently — another reason the selection is explicit rather than "everything
upstream offers".

**Renamed, deliberately.** ``hepatotoxicity`` is now ``DILI``, ADMET-AI's own id
for the Drug-Induced-Liver-Injury endpoint that actually produces the number.
``hERG`` and ``AMES`` keep their names because the model behind them answers the
same question the name asks. Naming an endpoint after something other than the
model that predicted it is the last of this tool's inherited fictions.

**Without ADMET-AI the three endpoints report nothing**, the capability stays
``AVAILABLE`` for the alerts it really computes, the backend string drops to
``rdkit-alerts`` so a reader can tell the two deployments apart, and every profile
carries a flagged ``tox_panel`` endpoint naming the missing install — because a
molecule nobody scored must never come out looking like a clean one. There is no
descriptor-derived stand-in: that is what was just removed.

Crucially, and as before: a "no detected signal" result is never a safety claim.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator, Sequence
from typing import Any

from rdkit import Chem

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import (
    ApplicabilityDomain,
    CapabilityStatus,
    CostClass,
    Determinism,
    Severity,
)
from ..contracts.predictions import SafetyEndpoint, SafetyProfile
from ..observability.toolmeter import measure
from .base import ToolAdapter
from .properties.admetai_tool import AdmetAiAdapter

# Tox-relevant structural alerts (supplement chem alerts).
_TOX_ALERTS = {
    "nitroaromatic": "[a][NX3+](=O)[O-]",
    "aromatic_amine": "[NX3;H2,H1][c]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "hydrazine": "[NX3][NX3]",
    "aromatic_nitroso": "[c][NX2]=O",
}

#: The toxicity endpoints this capability reports, under ADMET-AI's own ids. See
#: the module docstring for why it is these three and why the third is `DILI`.
PANEL_ENDPOINTS = ("hERG", "AMES", "DILI")

#: Backend names. Both are true statements about what ran; neither is `"none"`,
#: because the alerts half computes on any host and an AVAILABLE card that named
#: no engine would be as misleading as an UNAVAILABLE one that named a backend.
BACKEND_WITH_PANEL = "rdkit-alerts+admet_ai"
BACKEND_ALERTS_ONLY = "rdkit-alerts"

_SEVERITY_ORDER = (Severity.NONE, Severity.LOW, Severity.MODERATE, Severity.HIGH, Severity.CRITICAL)


@contextlib.contextmanager
def _panel_output_off_stdout() -> Iterator[None]:
    """Keep the ensembles' progress bar off stdout, without discarding it.

    Lightning renders a rich progress bar and its GPU banner to ``sys.stdout`` on
    every predict call. ``drugagent run --json`` writes its payload to that same
    stream, so an interleaved bar turns a machine-readable result into a parse
    error — it did, in four CLI tests, the moment this tool started calling
    ADMET-AI on the default path. Redirecting to stderr keeps an operator's view
    of what the model is doing while leaving stdout to whoever owns it;
    discarding the output would hide a real failure to fix a formatting problem.

    ``redirect_stdout`` swaps a global, so this wraps only the calls made from
    :meth:`SafetyAdapter.prefetch` and the single-molecule fallback — both of
    which run before or outside the candidate fan-out, never inside it.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


class SafetyAdapter(ToolAdapter):
    """Structural alerts, plus three pretrained toxicity endpoints when they exist."""

    capability_id = "safety.assess"

    def __init__(self, *, settings: Any = None) -> None:
        # Constructed unconditionally and cheaply: `AdmetAiAdapter.__init__` does
        # filesystem readiness work only and defers importing admet_ai (2.5 s)
        # until the first prediction, and `tool_pool()` builds every adapter on
        # every call.
        self.panel = AdmetAiAdapter(settings=settings)
        self.backend = BACKEND_WITH_PANEL if self.panel.available() else BACKEND_ALERTS_ONLY
        # Filled by `prefetch` and read by `assess`. Batch-scoped rather than
        # process-scoped: each prefetch replaces it, so a long-lived service does
        # not accumulate one entry per molecule it has ever seen.
        self._panel_cache: dict[str, SafetyProfile] = {}

    def available(self) -> bool:
        """True everywhere: substructure matching needs nothing but RDKit."""
        return True

    @property
    def profile(self) -> CapabilityProfile:
        panelled = self.panel.available()
        endpoints = ", ".join(PANEL_ENDPOINTS)
        return CapabilityProfile(
            capability_id=self.capability_id,
            # 2.0.0: the three endpoint numbers changed backend entirely, so a
            # SafetyProfile written by 1.x is not differenceable against one
            # written by this version.
            version="2.0.0",
            title="Toxicity & safety",
            description=(
                f"RDKit structural-alert matching, plus {endpoints} from the released "
                "ADMET-AI ensembles (TDC-trained, nothing fitted at call time)."
                if panelled
                else (
                    "RDKit structural-alert matching only. "
                    f"{endpoints} are not evaluated: ADMET-AI is not installed here."
                )
            ),
            input_schema_ref="Molecule",
            output_schema_ref="SafetyProfile",
            # DETERMINISTIC in the sense the enum offers: fixed weights, a fixed
            # featurizer, no sampling anywhere on the path. It is not the same as
            # bit-identical — see the last known failure mode.
            determinism=Determinism.DETERMINISTIC,
            # MODERATE rather than CHEAP once the panel is wired: it is a neural
            # ensemble, and a caller choosing between tools should be told.
            cost_class=CostClass.MODERATE if panelled else CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE,
            provides_mock=False,
            backend=self.backend,
            license=(
                "RDKit BSD-3-Clause; ADMET-AI MIT (Copyright (c) 2026 Kyle Swanson), "
                "weights trained on Therapeutics Data Commons datasets"
                if panelled
                else "RDKit BSD-3-Clause"
            ),
            # No synthetic-label restriction any more: nothing here is fitted to
            # invented labels. What remains is upstream's own training-set scope.
            data_restrictions=(
                "Endpoint scores are uncalibrated sigmoid outputs of models trained on "
                "public TDC datasets; not validated toxicology and not a probability of harm."
                if panelled
                else None
            ),
            known_failure_modes=[
                "unparseable SMILES",
                "molecule far outside ADMET-AI's TDC training distributions still scored "
                "confidently: no applicability check is performed",
                "structural alerts flag substructures, not toxicity",
                "endpoint scores are not bit-reproducible on a GPU: float32 reduction order "
                "moves them by ~1e-8 between runs of the identical batch, which is far below "
                "the precision anything reports but does make scorecards.json differ byte-wise",
            ],
            validation_checks=[
                "probabilities in [0,1]",
                "no_signal != safe",
                "an unscored molecule is never no_signal",
            ],
        )

    # --- batching ----------------------------------------------------------
    def prefetch(self, smiles: Sequence[str]) -> int:
        """Score a whole batch through the ensembles once, and cache it.

        ADMET-AI's cost is dominated by fixed per-call Lightning and CUDA setup
        rather than by batch size, so one call over *n* molecules is far cheaper
        than *n* calls over one — on this machine the difference between seconds
        and minutes over a candidate batch, against a 120 s task budget. This is
        not a separate metered tool operation: the work is metered where it
        happens, under ``property.admet.admetai.assess_safety_batch``.

        Returns how many molecules the panel produced a profile for, which is 0
        when ADMET-AI is absent.
        """
        self._panel_cache = {}
        if not self.panel.available():
            return 0
        unique = list(dict.fromkeys(s for s in smiles if s))
        if not unique:
            return 0
        # `candidate_ids` are placeholders: the cache is keyed by SMILES because
        # `assess` is called per candidate and only has the SMILES in hand.
        with _panel_output_off_stdout():
            profiles = self.panel.assess_safety_batch(unique, candidate_ids=unique)
        self._panel_cache = {p.candidate_id: p for p in profiles}
        return len(self._panel_cache)

    # --- assessment --------------------------------------------------------
    def assess(self, candidate_id: str, smiles: str) -> SafetyProfile:
        with measure(self.capability_id, "assess", backend=self.backend) as metered:
            profile = self._assess(candidate_id, smiles)
            metered["n_flags"] = float(profile.n_flags)
            metered["no_signal"] = 1.0 if profile.no_signal else 0.0
        return profile

    def _assess(self, candidate_id: str, smiles: str) -> SafetyProfile:
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return self._assemble(
                candidate_id,
                [
                    SafetyEndpoint(
                        endpoint="parse",
                        flag=True,
                        severity=Severity.HIGH,
                        detail="unparseable_smiles",
                        model_id="parser",
                    )
                ],
            )

        endpoints = [self._alerts_endpoint(rdmol)]
        endpoints.extend(self._panel_endpoints(candidate_id, smiles))
        return self._assemble(candidate_id, endpoints)

    def _alerts_endpoint(self, rdmol: Chem.Mol) -> SafetyEndpoint:
        """The half of this tool that was always real, unchanged."""
        alerts = self._tox_alerts(rdmol)
        return SafetyEndpoint(
            endpoint="structural_alerts",
            flag=bool(alerts),
            severity=Severity.MODERATE if alerts else Severity.NONE,
            # Substructure matching extrapolates nothing, so this really is
            # in-domain for any molecule RDKit can parse.
            applicability=ApplicabilityDomain.IN_DOMAIN,
            calibrated=False,
            model_id="tox_alerts",
            model_version="1.0.0",
            alerts=alerts,
            detail=f"{len(alerts)} tox structural alert(s)" if alerts else "no tox alerts",
            limitation=(
                "Substructures associated with reactivity or assay interference; not a toxicity prediction."
            ),
        )

    def _panel_endpoints(self, candidate_id: str, smiles: str) -> list[SafetyEndpoint]:
        """The three pretrained endpoints, or one endpoint saying they are absent."""
        if not self.panel.available():
            return [self._absent_panel_endpoint()]
        cached = self._panel_cache.get(smiles)
        if cached is None and self._panel_cache:
            # A prefetch covered this batch and this molecule is not in it, so
            # ADMET-AI screened it out or dropped it. Asking again would drop it
            # again — and would do so from inside the candidate fan-out, where the
            # stdout redirect above is not safe to take. Report the gap instead.
            return [self._absent_panel_endpoint(detail=f"{self.panel.capability_id} returned no row")]
        if cached is None:
            with _panel_output_off_stdout():
                cached = self.panel.assess_safety(candidate_id, smiles)
        wanted = {e.endpoint: e for e in cached.endpoints}
        selected = [wanted[name] for name in PANEL_ENDPOINTS if name in wanted]
        if not selected:
            # The panel ran and produced no toxicity row for this molecule. That
            # is not a pass, so it is reported as the absence it is.
            return [self._absent_panel_endpoint(detail=f"{self.panel.capability_id} returned no row")]
        return selected

    def _absent_panel_endpoint(self, detail: str | None = None) -> SafetyEndpoint:
        """A flagged placeholder for endpoints that were not evaluated.

        ``flag=True`` with ``severity=NONE`` is the convention
        :class:`~drug_agent.tools.properties.admetai_tool.AdmetAiAdapter` already
        uses for "nothing was assessed": the flag keeps ``no_signal`` False so an
        unscored molecule cannot read as a clean one, while the severity stays
        NONE because no model judged anything.
        """
        return SafetyEndpoint(
            endpoint="tox_panel",
            flag=True,
            severity=Severity.NONE,
            applicability=ApplicabilityDomain.UNKNOWN,
            calibrated=False,
            model_id=self.panel.capability_id,
            model_version=self.panel.version,
            detail=detail or (self.panel.unavailable_reason or "toxicity panel unavailable"),
            limitation=(
                f"{', '.join(PANEL_ENDPOINTS)} were not evaluated for this molecule. "
                "Not assessed is not safe."
            ),
        )

    def _assemble(self, candidate_id: str, endpoints: list[SafetyEndpoint]) -> SafetyProfile:
        max_sev = max((e.severity for e in endpoints), key=_SEVERITY_ORDER.index)
        n_flags = sum(1 for e in endpoints if e.flag)
        no_signal = n_flags == 0
        return SafetyProfile(
            candidate_id=candidate_id,
            endpoints=endpoints,
            max_severity=max_sev,
            n_flags=n_flags,
            no_signal=no_signal,
            summary=(
                f"No flag across {len(endpoints)} safety endpoint(s) (NOT a safety claim)."
                if no_signal
                else f"{n_flags} endpoint flag(s) of {len(endpoints)}; max severity {max_sev}."
            ),
        )

    def _tox_alerts(self, rdmol: Chem.Mol) -> list[str]:
        from .rdkit_tool import _compile_smarts

        hits = []
        for name, sm in _TOX_ALERTS.items():
            patt = _compile_smarts(sm)
            if patt is not None and rdmol.HasSubstructMatch(patt):
                hits.append(name)
        return hits
