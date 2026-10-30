"""Molecule Generation & Virtual-Screening Agent.

Sources candidates, standardizes and de-duplicates chemical identity, and
predicts activity/property endpoints with uncertainty and applicability domain.
Activity comes from the sklearn baseline model; properties are RDKit-CALCULATED.
No numbers originate from an LLM.

Three sourcing strategies, selected by ``brief.candidate_source``:

``library`` / ``seed`` / ``analogue``
    The fixture compound snapshot, the brief's seeds, and deterministic
    single-point analogue enumeration around them.
``pfm``
    De-novo generation, which is executed by
    :class:`~drug_agent.agents.specialists.design_agent.DesignAgent` rather than
    here. The token names the *strategy*; which model runs is
    ``settings.generator_backend``, resolved by
    :func:`~drug_agent.agents.specialists.design_agent.select_generator` over
    every generator in the tool pool — PFM by default, so an unchanged deployment
    samples what the recorded history sampled. Generated structures enter the
    funnel as ordinary candidates tagged with their generator's own origin
    (``pfm``, ``pocket2mol``, ``reinvent4``, …) and are screened, filtered and
    ranked exactly like library compounds — the design agent proposes, the rest
    of the pipeline disposes. A generator that is not installed reports
    UNAVAILABLE and this arm produces nothing: no surrogate, and no quiet
    substitution of one model for another (ADR-0010, ADR-0011).

The split is deliberate. Sourcing, identity and screening are this agent's
domain and are the same work whatever proposed the molecule; proposing molecules
for a *site* is the design agent's, and it is the arm that owns the twenty-one
generative models. ``AGENT_TOOLS`` follows the code: the generator ids sit on
``agent.design``, which is what really calls them.

Three learned property backends — ADMET-AI, Chemprop and Uni-Mol — extend
``predict_activity_and_properties`` when ``property_backends`` asks for them.
They are opt-in because each is a heavyweight model whose panel changes what a
run costs and what it produces, and a run that did not ask for one must be the
run it was before. Asked for and absent, a backend is skipped with its reason
recorded — the same treatment docking gets in the structure agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest, AgentResult, ValidationIssue
from ...contracts.base import Quantity
from ...contracts.capability import CapabilityProfile
from ...contracts.chem import Candidate
from ...contracts.enums import (
    ApplicabilityDomain,
    CapabilityStatus,
    CostClass,
    Determinism,
    TrustLabel,
)
from ...contracts.predictions import Prediction
from ...data.normalize import dedup_by_inchikey
from ...tools.pfm_tool import GenerationBatch
from ...tools.properties.admetai_tool import AdmetAiAdapter
from ...tools.properties.chemprop_tool import ChempropAdapter
from ...tools.properties.unimol_tool import UniMolAdapter
from ...tools.rdkit_tool import STANDARDIZER_VERSION, Cheminformatics
from ...tools.sklearn_tool import BaselineActivityModel
from . import SpecialistAgent, unavailable_note
from .design_agent import DesignAgent, select_generator

if TYPE_CHECKING:
    from ...tools.properties import PropertyPanel
    from ..orchestration.context import RunContext

#: Re-exported so a caller that has always resolved a generator through this
#: module keeps working; the function itself now lives with the agent that owns
#: the generators.
__all__ = ["MoleculeAgent", "select_generator"]

#: The optional learned property backends, by the short name a caller passes.
#: Full tool ids are accepted too, so a config can name either.
PROPERTY_TOOL_IDS: dict[str, str] = {
    "admetai": AdmetAiAdapter.capability_id,
    "chemprop": ChempropAdapter.capability_id,
    "unimol": UniMolAdapter.capability_id,
}


def _panel_failure(tool: Any, exc: Exception) -> str:
    """One line for an optional panel that was asked for and produced nothing."""
    return f"{tool.capability_id} produced NOTHING: {type(exc).__name__}: {exc}"


def _selected_property_tools(names: Sequence[str]) -> frozenset[str]:
    """Normalize requested property backends to tool ids, refusing unknown ones."""
    wanted: set[str] = set()
    for name in names:
        key = (name or "").strip().lower()
        tool_id = PROPERTY_TOOL_IDS.get(key, key if key in set(PROPERTY_TOOL_IDS.values()) else "")
        if not tool_id:
            raise ValueError(
                f"unknown property backend {name!r}; this agent wires: {', '.join(sorted(PROPERTY_TOOL_IDS))}"
            )
        wanted.add(tool_id)
    return frozenset(wanted)


class MoleculeAgent(SpecialistAgent):
    agent_id = "agent.molecule"
    version = "1.0.0"
    actions = {
        "source_candidates": "Ingest library, enumerate seed analogues, generate de-novo.",
        "standardize_candidates": "Standardize, de-duplicate chemical identity.",
        "predict_activity_and_properties": "Baseline activity + property predictions.",
    }

    def __init__(
        self,
        *,
        settings: object | None = None,
        generator_backend: str = "auto",
        property_backends: Sequence[str] = (),
    ) -> None:
        self.chem = Cheminformatics()
        # The de-novo arm, delegated whole. This agent still *drives* generation
        # when the brief asks for it, but the model, its selection and its record
        # belong to the design agent, so there is one owner of the generator
        # family rather than two objects that must agree about it.
        self.design = DesignAgent(settings=settings, generator_backend=generator_backend)
        wired = _selected_property_tools(property_backends)
        self.admet = AdmetAiAdapter(settings=settings) if AdmetAiAdapter.capability_id in wired else None
        self.chemprop = ChempropAdapter(settings=settings) if ChempropAdapter.capability_id in wired else None
        self.unimol = UniMolAdapter(settings=settings) if UniMolAdapter.capability_id in wired else None

    @property
    def generator(self) -> Any:
        """The design agent's generator. One object, reachable from both agents.

        A property rather than a second attribute: two references that can drift
        apart is how a run reports one model in its batch record and samples
        another.
        """
        return self.design.generator

    @generator.setter
    def generator(self, adapter: Any) -> None:
        self.design.generator = adapter

    @property
    def generator_origin(self) -> str:
        """The ``origin`` tag this run's generator writes onto every molecule.

        Read off the adapter, which is what lets ``n_generated`` count the arm
        that actually ran instead of the one this agent used to be hard-wired to.
        """
        return self.design.generator_origin

    def describe_capability(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.agent_id,
            version=self.version,
            title="Molecule generation & virtual screening",
            description="Source, standardize, and score candidate molecules.",
            input_schema_ref="ResearchBrief",
            output_schema_ref="Candidate",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            status=CapabilityStatus.AVAILABLE,
            backend=f"rdkit+sklearn,generate={self.generator.backend}{self._property_backend_suffix()}",
            known_failure_modes=["no valid candidates", "empty training set"],
            validation_checks=["all candidates have InChIKey", "predictions in plausible range"],
        )

    def _property_backend_suffix(self) -> str:
        """``,properties=…`` naming the optional backends, or nothing when none are wired."""
        wired = [t.backend for t in (self.admet, self.chemprop, self.unimol) if t is not None]
        return f",properties={'+'.join(wired)}" if wired else ""

    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        if request.action == "source_candidates":
            return self._source(request, ctx)
        if request.action == "standardize_candidates":
            return self._standardize(request, ctx)
        if request.action == "predict_activity_and_properties":
            return self._predict(request, ctx)
        return self._result(
            request,
            status="error",
            validation_issues=[ValidationIssue(code="unknown_action", message=request.action)],
        )

    # --- source ------------------------------------------------------------
    def _source(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        from ...contracts.brief import ResearchBrief  # local import: avoids a cycle

        raw_brief = request.inputs["brief"]
        brief = ResearchBrief.model_validate(raw_brief)
        sources = brief.candidate_sources()
        max_c = brief.max_candidates
        seeds = list(brief.seed_smiles)
        # Library from persisted compound snapshot.
        lib = [r["smiles"] for r in ctx.store.list_objects("compound", run_id=ctx.run.id)]

        warnings: list[str] = []
        issues: list[ValidationIssue] = []
        for token in brief.unknown_candidate_sources():
            issues.append(
                ValidationIssue(
                    code="unknown_candidate_source",
                    message=f"candidate_source token {token!r} is not supported by this build",
                    severity="warning",
                )
            )

        raw: list[tuple[str, str, list[str]]] = []  # (smiles, origin, lineage)
        if "seed" in sources:
            for s in seeds:
                raw.append((s, "seed", []))

        # De-novo generation runs before the (usually much larger) library so the
        # max_candidates cap cannot starve the molecules the generator proposed.
        batch: GenerationBatch | None = None
        if "pfm" in sources:
            batch = self.design.generate_batch(brief, seeds + lib, ctx)
            warnings.extend(batch.warnings)
            for gm in batch.molecules:
                raw.append((gm.smiles, gm.origin, []))
            if batch.errors:
                # A generator failure is fatal only when generation was the *only*
                # way to get candidates. Otherwise the run continues on the other
                # strategies and the failure stays loudly visible — as a warning
                # issue, a run warning, and a persisted generation_batch record —
                # rather than silently yielding a portfolio with no generated arm.
                fatal = not (sources - {"pfm"})
                issues.append(
                    ValidationIssue(
                        code="generation_failed",
                        message="; ".join(batch.errors)[:500],
                        severity="error" if fatal else "warning",
                    )
                )

        if "analogue" in sources:
            for s in seeds:
                for a in self.chem.enumerate_analogues(s):
                    raw.append((a, "analogue", [s]))
        if "library" in sources:
            for s in lib:
                raw.append((s, "library", []))

        # Deterministic de-dup on raw SMILES, cap at max_candidates.
        seen: set[str] = set()
        stored = 0
        counts: dict[str, int] = {}
        for smiles, origin, lineage in raw:
            if smiles in seen:
                continue
            seen.add(smiles)
            if stored >= max_c:
                break
            ctx.store.put_object(
                "raw_candidate",
                {"id": f"raw_{stored:05d}", "smiles": smiles, "origin": origin, "lineage": lineage},
                run_id=ctx.run.id,
            )
            stored += 1
            counts[origin] = counts.get(origin, 0) + 1

        # Count the origin the generator that ran actually wrote, not the one this
        # agent used to hard-wire: a pocket2mol batch tags 'pocket2mol'.
        n_generated = counts.get(self.generator_origin, 0)
        metrics = {
            "n_sourced": float(stored),
            "n_seed": float(counts.get("seed", 0)),
            "n_analogue": float(counts.get("analogue", 0)),
            "n_library": float(counts.get("library", 0)),
            "n_generated": float(n_generated),
        }
        if batch is not None:
            metrics.update(batch.metrics())

        breakdown = ", ".join(f"{n} {origin}" for origin, n in sorted(counts.items())) or "none"
        return self._result(
            request,
            payload={"n_sourced": stored, "n_generated": n_generated, "origins": counts},
            metrics=metrics,
            warnings=warnings,
            validation_issues=issues,
            explanation=f"Sourced {stored} raw candidates ({breakdown}).",
        )

    # --- standardize -------------------------------------------------------
    def _standardize(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        raws = ctx.store.list_objects("raw_candidate", run_id=ctx.run.id)
        molecules = []
        origin_by_ik: dict[str, tuple[str, list[str]]] = {}
        invalid = 0
        for r in raws:
            mol = self.chem.standardize(r["smiles"], source=r["origin"], lineage=r["lineage"])
            if not mol.sanitization_ok or not mol.inchikey:
                invalid += 1
                continue
            molecules.append(mol)
            origin_by_ik.setdefault(mol.inchikey, (r["origin"], r["lineage"]))

        unique, dups = dedup_by_inchikey(molecules)
        n_cand = 0
        for mol in unique:
            ctx.store.put_object("molecule", mol, run_id=ctx.run.id)
            origin, lineage = origin_by_ik.get(mol.inchikey or "", ("library", []))
            cand = Candidate(
                run_id=ctx.run.id,
                molecule_id=mol.id,
                inchikey=mol.inchikey,
                canonical_smiles=mol.canonical_smiles,
                origin=origin,
                lineage=lineage,
                scaffold=self.chem.scaffold(mol.canonical_smiles or ""),
            )
            ctx.store.put_object("candidate", cand, run_id=ctx.run.id)
            n_cand += 1

        metrics = {
            "n_standardized": float(len(molecules)),
            "n_unique": float(len(unique)),
            "n_invalid": float(invalid),
            "n_duplicates": float(len(dups)),
            "validity": round(len(molecules) / max(len(raws), 1), 4),
            "uniqueness": round(len(unique) / max(len(molecules), 1), 4),
        }
        return self._result(
            request,
            payload={"n_candidates": n_cand},
            metrics=metrics,
            explanation=f"Standardized to {n_cand} unique candidates "
            f"({invalid} invalid, {len(dups)} duplicates removed).",
        )

    # --- predict -----------------------------------------------------------
    def _fit_model(self, ctx: RunContext) -> BaselineActivityModel:
        if "activity_model" in ctx.blackboard:
            return ctx.blackboard["activity_model"]
        assays = ctx.store.list_objects("assay", run_id=ctx.run.id)
        smiles, y = [], []
        for a in assays:
            smi = a.get("molecule_smiles")
            val = (a.get("measurement") or {}).get("value")
            if smi and val is not None:
                smiles.append(smi)
                y.append(float(val))
        model = BaselineActivityModel(endpoint="pIC50", seed=ctx.run.seed)
        if len(smiles) >= 5:
            model.fit(smiles, y)
        ctx.blackboard["activity_model"] = model
        return model

    def _predict(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        model = self._fit_model(ctx)
        candidates = self.candidates(ctx)
        n_pred = 0
        issues: list[ValidationIssue] = []

        def work(cand: dict) -> list[Prediction]:
            smi = cand["canonical_smiles"]
            preds: list[Prediction] = []
            # activity
            if model.fitted:
                out = model.predict_one(smi)
                preds.append(
                    Prediction(
                        candidate_id=cand["id"],
                        molecule_inchikey=cand.get("inchikey"),
                        endpoint="pIC50",
                        kind="activity",
                        quantity=Quantity(
                            value=out.value, unit="pIC50", std=out.std, trust_label=TrustLabel.PREDICTED
                        ),
                        applicability=out.applicability,
                        applicability_score=out.applicability_score,
                        model_id=model.model_version,
                        model_version=model.model_version,
                        trust_label=TrustLabel.PREDICTED,
                        limitation=None if out.std is not None else "no uncertainty estimate",
                    )
                )
            # properties (CALCULATED via RDKit)
            desc = self.chem.descriptors_from_smiles(smi)
            for ep in ("qed", "logp", "mw", "tpsa"):
                if ep in desc:
                    preds.append(
                        Prediction(
                            candidate_id=cand["id"],
                            molecule_inchikey=cand.get("inchikey"),
                            endpoint=ep,
                            kind="property",
                            quantity=Quantity(
                                value=desc[ep], unit=_UNIT.get(ep), trust_label=TrustLabel.CALCULATED
                            ),
                            applicability=ApplicabilityDomain.IN_DOMAIN,
                            model_id="rdkit",
                            model_version=STANDARDIZER_VERSION,
                            trust_label=TrustLabel.CALCULATED,
                            limitation="deterministic descriptor; no model uncertainty",
                        )
                    )
            return preds

        all_preds = ctx.map_candidates(work, candidates, label="predict")
        for preds in all_preds:
            for p in preds:
                ctx.store.put_object("prediction", p, run_id=ctx.run.id)
                n_pred += 1

        n_learned, warnings = self._learned_properties(candidates, ctx)
        n_pred += n_learned
        return self._result(
            request,
            payload={"n_predictions": n_pred},
            metrics={"n_predictions": float(n_pred)},
            warnings=warnings,
            validation_issues=issues,
            explanation=f"Produced {n_pred} activity/property predictions.",
        )

    # --- optional learned property backends --------------------------------
    def _learned_properties(self, candidates: list[dict], ctx: RunContext) -> tuple[int, list[str]]:
        """Run whichever of ADMET-AI, Chemprop and Uni-Mol this agent was wired with.

        Batched on purpose: for all three the cost is dominated by loading the
        model, so one call over N molecules is far cheaper than N calls over one —
        which is also why this does not go through ``map_candidates``.

        A backend that cannot compute on this host is skipped and says why, exactly
        as ``structure.dock`` is skipped: an absent optional tool costs the run that
        arm, never the run.
        """
        if not (self.admet or self.chemprop or self.unimol) or not candidates:
            return 0, []
        ids = [c["id"] for c in candidates]
        smiles = [c["canonical_smiles"] for c in candidates]
        inchikeys = {c["id"]: c.get("inchikey") for c in candidates}
        n_pred = 0
        warnings: list[str] = []

        # Every panel below is OPTIONAL — switched on by `property_backends` and
        # off by default — so each is wrapped: a panel that is asked for, reports
        # itself available and then raises must cost its own predictions and
        # nothing else. Measured on the sibling arm: one broken route engine
        # ended a cycle that had already built 90 candidates, with nothing scored.
        if self.admet is not None:
            if not self.admet.available():
                warnings.append(unavailable_note(self.admet))
            else:
                try:
                    for panel in self.admet.predict_properties(smiles, candidate_ids=ids):
                        n_pred += self._store_panel(panel, inchikeys, ctx)
                except Exception as exc:
                    warnings.append(_panel_failure(self.admet, exc))

        if self.chemprop is not None:
            if not self.chemprop.available():
                warnings.append(unavailable_note(self.chemprop))
            else:
                # Chemprop predicts only where this deployment has trained a
                # checkpoint. No checkpoint is not a failure and not a zero — it is
                # an endpoint this deployment cannot predict, said out loud.
                try:
                    trained = self.chemprop.trained_endpoints()
                    if not trained:
                        warnings.append(
                            f"{self.chemprop.capability_id} predicted NOTHING: no trained checkpoint "
                            f"for any endpoint on this deployment"
                        )
                    for endpoint in trained:
                        for panel in self.chemprop.predict(smiles, endpoint=endpoint, candidate_ids=ids):
                            n_pred += self._store_panel(panel, inchikeys, ctx)
                except Exception as exc:
                    warnings.append(_panel_failure(self.chemprop, exc))

        if self.unimol is not None:
            if not self.unimol.available():
                warnings.append(unavailable_note(self.unimol))
            else:
                # Embeddings, not predictions: they carry no endpoint, unit or
                # uncertainty, so they are persisted as their own object type and
                # deliberately not counted as predictions.
                try:
                    for rep in self.unimol.represent(candidate_ids=ids, smiles_list=smiles):
                        ctx.store.put_object(
                            "molecule_representation",
                            {"id": f"repr_{rep.candidate_id}", "dim": rep.dim, **asdict(rep)},
                            run_id=ctx.run.id,
                        )
                except Exception as exc:
                    warnings.append(_panel_failure(self.unimol, exc))
        return n_pred, warnings

    def _store_panel(self, panel: PropertyPanel, inchikeys: dict[str, str | None], ctx: RunContext) -> int:
        """Persist one backend's endpoint panel as predictions. Returns how many.

        ``panel.descriptors`` are dropped rather than stored: they are RDKit
        calculations this run already has under the same names, and two different
        numbers behind one endpoint name is how a scorecard stops meaning anything.
        """
        n = 0
        for endpoint, value in panel.endpoints.items():
            ctx.store.put_object(
                "prediction",
                Prediction(
                    candidate_id=panel.candidate_id,
                    molecule_inchikey=inchikeys.get(panel.candidate_id),
                    endpoint=endpoint,
                    kind="property",
                    quantity=Quantity(value=float(value), trust_label=TrustLabel.PREDICTED),
                    # These models publish no applicability domain of their own, and
                    # inventing one here would be a claim the backend never made.
                    applicability=ApplicabilityDomain.UNKNOWN,
                    model_id=panel.model_id,
                    model_version=panel.model_version,
                    trust_label=TrustLabel.PREDICTED,
                    limitation=panel.limitation or None,
                ),
                run_id=ctx.run.id,
            )
            n += 1
        return n

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if request.action == "standardize_candidates":
            cands = ctx.store.list_objects("candidate", run_id=ctx.run.id)
            missing = [c["id"] for c in cands if not c.get("inchikey")]
            if missing:
                issues.append(
                    ValidationIssue(
                        code="identity_missing", message=f"{len(missing)} candidates lack InChIKey"
                    )
                )
        return issues


_UNIT = {"qed": "unitless", "logp": "logP", "mw": "g/mol", "tpsa": "A^2"}
