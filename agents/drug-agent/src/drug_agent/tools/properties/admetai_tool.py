"""ADMET-AI — 41 pretrained TDC endpoints and 11 RDKit descriptors for one molecule.

**What it does now.** Loads the released ADMET-AI Chemprop ensembles into this
process and predicts every endpoint they were trained on. Nothing is fitted at call
time: the weights ship inside the ``admet_ai`` wheel — ten ``.pt`` checkpoints under
``admet_ai/resources/models``, two ensembles of five, one multitask head covering 31
classification endpoints and one covering 10 regression endpoints — so the numbers
are reproducible and their training sets are published. Alongside them RDKit
computes 11 physicochemical descriptors, and each of the resulting 52 values is
paired with its percentile against the 2,845 DrugBank-approved drugs bundled with
the package.

Re-measured on this machine on 2026-09-09 (x86-64, NVIDIA RTX PRO 6000 Blackwell
Workstation Edition, sm_120, admet_ai 2.0.1, chemprop 2.2.4, torch 2.14.0+cu130,
rdkit 2026.03.6), ``ADMETModel().predict(smiles=[aspirin, caffeine])`` returns a
DataFrame of shape ``(2, 104)`` — 52 values and their 52 DrugBank percentiles —
while Lightning logs ``GPU available: True (cuda), used: True``. Importing
``admet_ai`` costs 2.4-2.6 s, constructing ``ADMETModel()`` a further 0.1-0.2 s, and
the first two-molecule prediction 3.6-4.2 s over three runs. Inference is dominated
by fixed per-call setup rather than by batch size, which is why the batch operations
below exist and why callers should prefer them. Every predicted value below was
reproduced bit-for-bit against the aarch64 host's numbers: the architecture changed
the timings and nothing else.

Aspirin, ``CC(=O)Oc1ccccc1C(=O)O``, as this adapter splits it:

    ``descriptors``   molecular_weight 180.159 Da, logP 1.3101, tpsa 63.6 Å²,
                      hydrogen_bond_acceptors 3, hydrogen_bond_donors 1,
                      Lipinski 4.0 of 4, QED 0.5501, stereo_centers 0,
                      PAINS_alert 0, BRENK_alert 1, NIH_alert 0
    ``endpoints``     AMES 0.0801, hERG 0.0208, DILI 0.6746, ClinTox 0.1095,
                      Carcinogens_Lagunin 0.3984, SR-MMP 0.0072,
                      LD50_Zhu 2.069 log(1/(mol/kg))

**Two kinds of number, kept apart.** The 11 physicochemical values are deterministic
RDKit calculations — ``MolWt``, ``MolLogP``, ``CalcTPSA``, ``qed``, the Lipinski
count and the PAINS/BRENK/NIH substructure catalogues. The other 41 are neural
network outputs with published test-set error. Both arrive in one upstream
DataFrame, and merging them would let a QED of 0.55 and a DILI probability of 0.67
be read off the same row as though they carried the same warranty. So descriptors go
to :attr:`PropertyPanel.descriptors` and predictions to
:attr:`PropertyPanel.endpoints`, and the split is taken from ADMET-AI's own
``admet.csv`` ``category`` column rather than from a list maintained here.

**The safety operations.** :meth:`assess_safety` builds a
:class:`~drug_agent.contracts.predictions.SafetyProfile` from the 19 endpoints
``admet.csv`` files under ``Toxicity`` — hERG, ClinTox, AMES, DILI,
Carcinogens_Lagunin, LD50_Zhu, Skin_Reaction and the 12 Tox21 nuclear-receptor and
stress-response assays — plus one deterministic ``structural_alerts`` endpoint from
the RDKit filter catalogues. It exists so this can replace the synthetic-label
baseline in :mod:`drug_agent.tools.safety_tool`, whose three endpoints are fitted to
labels generated from a documented latent rule. The severity ladder and flag
threshold are deliberately copied from that module (≥0.7 HIGH, ≥0.4 MODERATE, ≥0.2
LOW, flag at ≥0.5) so that swapping the backend changes where the probabilities come
from without silently changing what a severity means.

**Regression endpoints are not thresholded.** ``LD50_Zhu`` is the one toxicity
endpoint that is a regression, in log(1/(mol/kg)); aspirin scores 2.069. Passing
that through a ladder built for probabilities would manufacture a severity out of a
unit mismatch, so regression endpoints are recorded with their value and their unit,
``flag=None``, and severity ``NONE`` — which means "not assessed here", stated in the
endpoint's ``limitation``, not "safe".

**Reliability travels with each number.** ADMET-AI publishes per-endpoint test-set
metrics in its own ``admet.csv``, and they vary by more than the endpoints do: hERG
reaches AUROC 0.839 and AUPRC 0.906, while SR-ATAD5 reaches AUPRC 0.312 and
NR-Aromatase 0.475. Each :class:`SafetyEndpoint` therefore carries the upstream
figure for its own endpoint in ``detail``, attributed to that file, together with the
molecule's DrugBank percentile when the reference set is loaded — aspirin's hERG
score sits at the 13.4th percentile of approved drugs.

**What it deliberately does not do.** It does not check applicability. Every endpoint
returns a confident number for a molecule nothing like its training set, and no
adapter here claims a domain it has not tested, so ``applicability`` is ``UNKNOWN``
on every predicted endpoint and the specialist agent that turns a panel into a
:class:`~drug_agent.contracts.predictions.Prediction` owns that judgement. It does
not calibrate: the classification outputs are raw sigmoid scores, ``calibrated`` is
false throughout, and a 0.67 DILI score is not a 67% chance of liver injury. It does
not fit anything to project data — that is Chemprop's job — and it constructs no
``Prediction``.

**Why chemprop is pinned below 2.3.** From 2.3.0 chemprop imports ``cuik_molmaker``,
a compiled accelerator built against an older RDKit, which fails here with
``ImportError: libRDKitAbbreviations-0c759a56.so.1: cannot open shared object file``
against rdkit 2026.03.5. 2.2.4 is the newest release that imports, and it satisfies
admet-ai's own ``chemprop>=2.2.2`` requirement, so the pin costs nothing. The
reasoning is recorded next to the pin in ``environment.yml``.

**One readiness gap, handled at call time.** The shared probe resolves module
locations with :func:`importlib.util.find_spec` and never executes them, which is
what keeps ``tool_pool()`` fast across thirty adapters. A dependency that is present
but unimportable — exactly the chemprop case above — therefore passes readiness.
:meth:`_load_model` catches that import failure and raises
:class:`~drug_agent.tools.properties.PropertyModelError` naming it, so the backend
that broke is identified rather than surfacing as a bare ``ImportError`` mid-run.

Without ``admet_ai`` this capability is UNAVAILABLE and predicts nothing. There is no
descriptor-only fallback: an adapter that answered a hERG query with a logP would be
returning a number of a different kind under the name of the one that was asked for.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import ApplicabilityDomain, CostClass, Determinism, Severity
from ...contracts.predictions import SafetyEndpoint, SafetyProfile
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import PropertyModelError, PropertyPanel, PropertyToolAdapter

#: This adapter's contract with ADMET-AI, not ADMET-AI's own version.
ADMETAI_ADAPTER_VERSION = "1.0.0"

SPEC = ExternalSpec(
    name="ADMET-AI",
    env_prefix="ADMETAI",
    selection_modules=("admet_ai",),
    # Everything `admet_ai.admet_model` imports on the prediction path, read off its
    # import block rather than reduced to the headline dependency. chemprop and
    # lightning are the two that actually go missing or go wrong here: chemprop
    # because of the 2.3.x RDKit ABI break documented above, lightning because it is
    # a separate distribution from torch and a torch-only environment still fails.
    runtime_modules=(
        "admet_ai",
        "chemprop",
        "lightning",
        "torch",
        "rdkit",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "tqdm",
    ),
    conda_envs=("drug-agent",),
    upstream_url="https://github.com/swansonk14/admet_ai",
    # No checkout: ADMET-AI is installed from PyPI and carries its own weights, so
    # the version of the wheel is the only meaningful pin.
    upstream_ref="v2.0.1 (PyPI wheel; no .external checkout)",
    license=(
        "ADMET-AI MIT (Copyright (c) 2026 Kyle Swanson). The bundled "
        "drugbank_approved.csv reference set carries DrugBank's own licence terms."
    ),
    install_hint="make env",
    # Upstream selects CUDA when it is present and CPU otherwise, so a GPU is a
    # speedup rather than a requirement. It used the GPU on this machine.
    needs_gpu=False,
)

#: Suffix upstream appends for the DrugBank comparison of every column.
_PERCENTILE_SUFFIX = "drugbank_approved_percentile"

#: ``category`` values in ADMET-AI's ``admet.csv`` that this adapter routes on.
_PHYSCHEM_CATEGORY = "Physicochemical"
_TOXICITY_CATEGORY = "Toxicity"

#: The RDKit filter catalogues ADMET-AI counts, folded into one safety endpoint.
_ALERT_DESCRIPTORS = ("PAINS_alert", "BRENK_alert", "NIH_alert")

#: Probability at or above which a classification endpoint raises a flag, and the
#: severity ladder above it. Both are taken from ``safety_tool.py`` unchanged, so
#: replacing that backend with this one does not also move the goalposts.
_FLAG_THRESHOLD = 0.5
_SEVERITY_LADDER: tuple[tuple[float, Severity], ...] = (
    (0.7, Severity.HIGH),
    (0.4, Severity.MODERATE),
    (0.2, Severity.LOW),
)
_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.NONE,
    Severity.LOW,
    Severity.MODERATE,
    Severity.HIGH,
    Severity.CRITICAL,
)

_PANEL_LIMITATION = (
    "Fixed pretrained panel. Endpoints are those ADMET-AI's authors trained on TDC "
    "datasets; no applicability-domain check is performed, so a molecule outside "
    "those training distributions still receives a confident number. Classification "
    "outputs are uncalibrated sigmoid scores."
)

_CLASSIFICATION_LIMITATION = (
    "Uncalibrated sigmoid score from a fixed pretrained model; not a probability of "
    "harm and not applicability-checked."
)


class AdmetAiAdapter(PropertyToolAdapter):
    """Pretrained ADMET endpoint prediction over the released ADMET-AI ensembles."""

    capability_id = "property.admet.admetai"
    spec = SPEC
    engine = "admet_ai"
    version = ADMETAI_ADAPTER_VERSION
    produces_nothing = "predicts nothing"
    modality = "pretrained ADMET panel (41 TDC endpoints, 11 RDKit descriptors)"

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        include_drugbank: bool = True,
        num_workers: int | None = None,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 600.0)
        #: Load the DrugBank reference set so endpoints can carry a percentile.
        self.include_drugbank = include_drugbank
        #: DataLoader workers. ``None`` keeps upstream's own choice (8 with a GPU).
        self.num_workers = num_workers
        # The ensembles are built on first use and cached here, never in __init__:
        # `tool_pool()` constructs every adapter on every call, and importing
        # admet_ai alone costs 2.52 s on this machine.
        self._model: Any = None
        self._info: dict[str, dict[str, Any]] | None = None

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Pretrained ADMET panel (ADMET-AI)",
            description=(
                "41 absorption, distribution, metabolism, excretion and toxicity endpoints "
                "from the released ADMET-AI Chemprop ensembles, plus 11 RDKit "
                "physicochemical descriptors and a DrugBank-approved percentile for each. "
                "Predictions and calculations are reported separately. Uses CUDA when "
                "present. Nothing is fitted at call time and no applicability check is made."
            ),
            input_schema_ref="SMILES",
            output_schema_ref="PropertyPanel",
            # Fixed weights and a deterministic featurizer: the same SMILES gives the
            # same row every time. No sampling anywhere on the prediction path.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            known_failure_modes=[
                "SMILES RDKit cannot parse (screened out before inference, flagged per molecule)",
                "chemprop present but unimportable against a newer RDKit, which the "
                "find_spec readiness probe cannot detect and which surfaces at load time",
                "molecule far outside the TDC training distributions still scored confidently",
                "uncalibrated classification scores read as probabilities of harm",
                "LD50_Zhu is a regression in log(1/(mol/kg)) and is never thresholded",
            ],
            validation_checks=[
                "calculated RDKit descriptors never appear among predicted endpoints",
                "endpoint names are ADMET-AI's own, verbatim",
                "classification scores lie in [0, 1]",
                "no_signal is not a safety claim",
            ],
        )

    # --- operations --------------------------------------------------------
    def predict_properties(
        self,
        smiles: Sequence[str],
        *,
        candidate_ids: Sequence[str] | None = None,
        include_percentiles: bool = False,
    ) -> list[PropertyPanel]:
        """Predict the full panel for each molecule, one panel per input in order.

        ``include_percentiles`` adds each value's DrugBank-approved percentile beside
        it, under upstream's own ``<name>_drugbank_approved_percentile`` key and in
        the same dict as the value it describes. It is off by default because a
        percentile is a comparison against a reference set rather than a property of
        the molecule, and doubling every panel invites the two to be averaged.
        """
        self.require_available()
        with measure(self.capability_id, "predict_properties", backend=self.backend) as metered:
            panels = self._panels(list(smiles), candidate_ids, include_percentiles)
            for key, value in self.coverage(panels).items():
                metered[key] = value
        return panels

    def assess_safety(self, candidate_id: str, smiles: str) -> SafetyProfile:
        """Build one molecule's safety profile from the real toxicity endpoints.

        Argument order and metric names match
        :meth:`drug_agent.tools.safety_tool.SafetyAdapter.assess`, so this is a
        drop-in replacement for the synthetic-label baseline.
        """
        self.require_available()
        with measure(self.capability_id, "assess_safety", backend=self.backend) as metered:
            profile = self._profiles([smiles], [candidate_id])[0]
            metered["n_flags"] = float(profile.n_flags)
            metered["no_signal"] = 1.0 if profile.no_signal else 0.0
        return profile

    def assess_safety_batch(
        self,
        smiles: Sequence[str],
        *,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[SafetyProfile]:
        """Safety profiles for a whole batch in one pass over the ensembles.

        Preferred over repeated :meth:`assess_safety` calls: the cost is dominated by
        per-call Lightning and CUDA setup, so one call over *n* molecules is far
        cheaper than *n* calls over one.
        """
        self.require_available()
        with measure(self.capability_id, "assess_safety_batch", backend=self.backend) as metered:
            profiles = self._profiles(list(smiles), candidate_ids)
            if profiles:
                metered["admet_n_molecules"] = float(len(profiles))
                metered["admet_flag_rate"] = float(sum(1 for p in profiles if p.n_flags) / len(profiles))
                metered["admet_n_flags_mean"] = float(sum(p.n_flags for p in profiles) / len(profiles))
        return profiles

    # --- real backend ------------------------------------------------------
    def _load_model(self) -> Any:
        """Build the ensembles once and keep them on the instance."""
        if self._model is not None:
            return self._model
        # Imported here, not at module scope: this pulls torch, chemprop and
        # lightning behind it, and `tool_pool()` builds every adapter on every call.
        try:
            from admet_ai import ADMETModel
            from admet_ai.constants import DEFAULT_DRUGBANK_PATH, DEFAULT_MODELS_DIR
        except ImportError as exc:
            raise PropertyModelError(
                f"ADMET-AI passed the readiness probe but failed to import: {exc}. The probe "
                "resolves module locations without executing them, so a dependency that is "
                "installed but unimportable — chemprop 2.3.x against rdkit 2026.03.5 is the "
                "known case — can only surface here. Reinstall with `make env`."
            ) from exc

        models_dir = Path(DEFAULT_MODELS_DIR)
        if not any(models_dir.glob("**/*.pt")):
            raise PropertyModelError(
                f"ADMET-AI ships its weights inside the wheel, but {models_dir} holds no .pt "
                "checkpoints, so there is nothing to predict with. Reinstall admet-ai."
            )

        kwargs: dict[str, Any] = {
            "include_physchem": True,
            "drugbank_path": DEFAULT_DRUGBANK_PATH if self.include_drugbank else None,
        }
        if self.num_workers is not None:
            kwargs["num_workers"] = self.num_workers
        try:
            self._model = ADMETModel(**kwargs)
        except Exception as exc:
            raise PropertyModelError(f"ADMET-AI ensembles failed to load: {exc}") from exc
        return self._model

    def _admet_info(self) -> dict[str, dict[str, Any]]:
        """ADMET-AI's own endpoint table, keyed by endpoint id.

        This is what decides which columns are calculated descriptors, which are
        toxicity endpoints, and whether an endpoint is a classification or a
        regression. Reading it from the package keeps those three judgements in step
        with the installed version instead of with a list maintained here.
        """
        if self._info is None:
            from admet_ai.admet_info import get_admet_info

            frame = get_admet_info()
            self._info = {str(row["id"]): dict(row) for _, row in frame.iterrows()}
        return self._info

    def _predict(self, smiles: list[str]) -> dict[str, dict[str, Any]]:
        """Run the ensembles once over *smiles*, returning ``{smiles: {column: value}}``.

        Callers pass a de-duplicated list of parseable SMILES: upstream indexes its
        result frame by SMILES, so a duplicate would collapse two rows into one, and
        it silently drops anything RDKit rejects.
        """
        if not smiles:
            return {}
        model = self._load_model()
        try:
            frame = model.predict(smiles=smiles)
        except Exception as exc:
            raise PropertyModelError(
                f"ADMET-AI inference failed on {len(smiles)} molecule(s): {exc}"
            ) from exc
        rows: dict[str, dict[str, Any]] = frame.to_dict(orient="index")
        return rows

    # --- panels ------------------------------------------------------------
    def _panels(
        self,
        smiles: list[str],
        candidate_ids: Sequence[str] | None,
        include_percentiles: bool,
    ) -> list[PropertyPanel]:
        rows, info = self._rows_for(smiles)
        panels: list[PropertyPanel] = []
        for idx, smi in enumerate(smiles):
            cid = self._candidate_id(candidate_ids, idx)
            row = rows.get(smi)
            if row is None:
                panels.append(self.empty_panel(cid, smi, self._missing_reason(smi)))
                continue
            panels.append(self._panel(cid, smi, row, info, include_percentiles))
        return panels

    def _panel(
        self,
        candidate_id: str,
        smiles: str,
        row: dict[str, Any],
        info: dict[str, dict[str, Any]],
        include_percentiles: bool,
    ) -> PropertyPanel:
        """Split one upstream row into calculated descriptors and predicted endpoints."""
        descriptors: dict[str, float] = {}
        endpoints: dict[str, float] = {}
        unclassified = 0
        for column, raw in row.items():
            name = str(column)
            base, is_percentile = self._split_percentile(name)
            if is_percentile and not include_percentiles:
                continue
            value = self.numeric(raw)
            if value is None:
                continue
            meta = info.get(base)
            if meta is None:
                unclassified += 1
                continue
            # A percentile inherits the provenance of the quantity it ranks: the
            # percentile of a calculated logP is still a statement about a
            # calculation, and the key keeps upstream's suffix so nothing is hidden.
            if meta.get("category") == _PHYSCHEM_CATEGORY:
                descriptors[name] = value
            else:
                endpoints[name] = value

        panel = PropertyPanel(
            candidate_id=candidate_id,
            smiles=smiles,
            endpoints=endpoints,
            descriptors=descriptors,
            model_id=self.capability_id,
            model_version=self.version,
            limitation=_PANEL_LIMITATION,
        )
        if unclassified:
            # A column admet.csv does not describe is dropped rather than guessed
            # into one bucket or the other, and the count says it happened.
            panel.qc_flags.append(f"unclassified_columns={unclassified}")
        return panel

    # --- safety ------------------------------------------------------------
    def _profiles(
        self,
        smiles: list[str],
        candidate_ids: Sequence[str] | None,
    ) -> list[SafetyProfile]:
        rows, info = self._rows_for(smiles)
        profiles: list[SafetyProfile] = []
        for idx, smi in enumerate(smiles):
            cid = self._candidate_id(candidate_ids, idx)
            row = rows.get(smi)
            if row is None:
                profiles.append(self._no_result_profile(cid, self._missing_reason(smi)))
                continue
            profiles.append(self._profile(cid, row, info))
        return profiles

    def _profile(
        self,
        candidate_id: str,
        row: dict[str, Any],
        info: dict[str, dict[str, Any]],
    ) -> SafetyProfile:
        endpoints: list[SafetyEndpoint] = [self._alerts_endpoint(row)]
        for name, meta in info.items():
            if meta.get("category") != _TOXICITY_CATEGORY:
                continue
            value = self.numeric(row.get(name))
            if value is None:
                continue
            endpoints.append(self._tox_endpoint(name, meta, value, row))

        max_severity = max((e.severity for e in endpoints), key=_SEVERITY_ORDER.index)
        n_flags = sum(1 for e in endpoints if e.flag)
        no_signal = n_flags == 0
        return SafetyProfile(
            candidate_id=candidate_id,
            endpoints=endpoints,
            max_severity=max_severity,
            n_flags=n_flags,
            no_signal=no_signal,
            summary=(
                f"No flag across {len(endpoints)} ADMET-AI safety endpoint(s) (NOT a safety claim)."
                if no_signal
                else f"{n_flags} endpoint flag(s) of {len(endpoints)}; max severity {max_severity}."
            ),
        )

    def _tox_endpoint(
        self,
        name: str,
        meta: dict[str, Any],
        value: float,
        row: dict[str, Any],
    ) -> SafetyEndpoint:
        """One toxicity endpoint, thresholded only if it is a classification."""
        label = str(meta.get("name") or name)
        task_type = str(meta.get("task_type") or "").strip()

        if task_type == "classification":
            severity = self._severity(value)
            flag: bool | None = value >= _FLAG_THRESHOLD
            limitation = _CLASSIFICATION_LIMITATION
        else:
            # A regression output has no probability scale, so there is no honest
            # threshold for it here. NONE means "not assessed", never "safe".
            units = str(meta.get("units") or "").strip()
            severity = Severity.NONE
            flag = None
            limitation = (
                f"{label} is a regression output in {units}, not a probability. It is "
                "reported unthresholded; severity NONE means this panel does not judge "
                "it, not that the molecule is safe."
            )

        return SafetyEndpoint(
            endpoint=name,
            score=value,
            flag=flag,
            severity=severity,
            # Never claimed: no applicability test is run against ADMET-AI's
            # training distributions, and the panel must not imply one was.
            applicability=ApplicabilityDomain.UNKNOWN,
            calibrated=False,
            model_id=f"admet_ai:{name}",
            model_version=self._backend_version(),
            detail=self._endpoint_detail(label, meta, row, name),
            limitation=limitation,
        )

    def _alerts_endpoint(self, row: dict[str, Any]) -> SafetyEndpoint:
        """PAINS, BRENK and NIH substructure counts as one deterministic endpoint."""
        counts: dict[str, int] = {}
        for key in _ALERT_DESCRIPTORS:
            value = self.numeric(row.get(key))
            if value is not None:
                counts[key] = int(value)
        fired = [f"{key}={n}" for key, n in counts.items() if n > 0]
        total = float(sum(counts.values()))
        return SafetyEndpoint(
            endpoint="structural_alerts",
            score=total,
            flag=bool(fired),
            severity=Severity.MODERATE if fired else Severity.NONE,
            # Substructure matching extrapolates nothing, so this one really is
            # in-domain for any molecule RDKit can parse.
            applicability=ApplicabilityDomain.IN_DOMAIN,
            calibrated=False,
            model_id="rdkit.FilterCatalog:PAINS+BRENK+NIH",
            model_version=self._rdkit_version(),
            alerts=fired,
            detail=(
                f"{int(total)} substructure alert(s) across the PAINS, BRENK and NIH "
                "catalogues, counted deterministically by RDKit."
                if fired
                else "No PAINS, BRENK or NIH substructure alerts."
            ),
            limitation=(
                "Structural alerts flag substructures associated with assay interference "
                "or liability; they are not toxicity predictions."
            ),
        )

    def _endpoint_detail(
        self,
        label: str,
        meta: dict[str, Any],
        row: dict[str, Any],
        name: str,
    ) -> str:
        """Endpoint name, upstream's published test metric, and the DrugBank percentile.

        The test metrics are ADMET-AI's own, from the ``admet.csv`` it ships, and are
        attributed as such. They belong next to the score because they are not
        uniform: across the 19 toxicity endpoints the published AUPRC ranges from
        0.312 to 0.906, and a reader who cannot see which one they are holding cannot
        weigh it.
        """
        parts = [label]
        auroc = self.numeric(meta.get("AUROC"))
        auprc = self.numeric(meta.get("AUPRC"))
        r2 = self.numeric(meta.get("R^2"))
        mae = self.numeric(meta.get("MAE"))
        if auroc is not None and auprc is not None:
            parts.append(f"TDC test AUROC {auroc:.3f}, AUPRC {auprc:.3f} (ADMET-AI admet.csv)")
        elif r2 is not None and mae is not None:
            parts.append(f"TDC test R^2 {r2:.3f}, MAE {mae:.3f} (ADMET-AI admet.csv)")
        percentile = self.numeric(row.get(f"{name}_{_PERCENTILE_SUFFIX}"))
        if percentile is not None:
            parts.append(f"{percentile:.1f}th percentile among DrugBank-approved drugs")
        return "; ".join(parts)

    def _no_result_profile(self, candidate_id: str, reason: str) -> SafetyProfile:
        """A profile for a molecule the panel could not score, which is not a pass.

        ``no_signal`` would read as "nothing found", so the failure is carried as a
        flagged endpoint instead: an unscored molecule must never look like a clean
        one.
        """
        return SafetyProfile(
            candidate_id=candidate_id,
            endpoints=[
                SafetyEndpoint(
                    endpoint="panel",
                    flag=True,
                    severity=Severity.NONE,
                    applicability=ApplicabilityDomain.UNKNOWN,
                    model_id=self.capability_id,
                    model_version=self.version,
                    detail=reason,
                    limitation="No endpoint was evaluated for this molecule.",
                )
            ],
            max_severity=Severity.NONE,
            n_flags=1,
            no_signal=False,
            summary=f"ADMET-AI produced no panel for this molecule ({reason}); nothing was assessed.",
        )

    # --- helpers -----------------------------------------------------------
    def _rows_for(self, smiles: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Screen, de-duplicate, run the ensembles once, and fetch the endpoint table."""
        keep, _ = self.screen(smiles)
        unique = list(dict.fromkeys(keep))
        rows = self._predict(unique)
        info = self._admet_info() if rows else {}
        return rows, info

    def _missing_reason(self, smiles: str) -> str:
        """Why one molecule has no row: rejected by screening, or dropped upstream."""
        keep, _ = self.screen([smiles])
        return "unparseable_smiles" if not keep else "admetai_returned_no_row"

    @staticmethod
    def _candidate_id(candidate_ids: Sequence[str] | None, idx: int) -> str:
        if candidate_ids is None or idx >= len(candidate_ids):
            return ""
        return candidate_ids[idx]

    @staticmethod
    def _split_percentile(column: str) -> tuple[str, bool]:
        """``('hERG', True)`` for a percentile column, ``('hERG', False)`` otherwise."""
        suffix = f"_{_PERCENTILE_SUFFIX}"
        if column.endswith(suffix):
            return column[: -len(suffix)], True
        return column, False

    @staticmethod
    def _severity(score: float) -> Severity:
        for threshold, severity in _SEVERITY_LADDER:
            if score >= threshold:
                return severity
        return Severity.NONE

    @staticmethod
    def _backend_version() -> str:
        import admet_ai

        return str(getattr(admet_ai, "__version__", "unknown"))

    @staticmethod
    def _rdkit_version() -> str:
        import rdkit

        return str(getattr(rdkit, "__version__", "unknown"))


__all__ = ["ADMETAI_ADAPTER_VERSION", "SPEC", "AdmetAiAdapter"]
