"""Evaluation & feedback — the independent validator, with tools of its own.

Never the producer of the results it validates. That is the whole design, and it
is why this agent now holds tools: an evaluator with no way to compute anything
can only re-read fields that a producer already wrote, which catches a missing
value and nothing else. The four checks it used to make were all of that kind.

What it computes here, it computes *about* results rather than *for* ranking, and
nothing it writes is a :class:`~drug_agent.contracts.predictions.Prediction` — so
no number this agent produces can reach a scorecard, a Pareto front or a
shortlist. That containment is what lets it use a model without having a stake in
the verdict.

``chem.standardize``
    Identity, re-derived. The old check asked whether ``inchikey`` was non-empty;
    this one re-standardizes the stored SMILES and compares the InChIKey it gets.
    A candidate whose identity does not survive a second pass through the same
    standardizer is a candidate whose lineage, de-duplication and every join on
    ``inchikey`` downstream are wrong. Always on: it costs milliseconds.
``structure.validate.posebusters``
    Coordinates, judged. Poses that already carry a validation record are
    *audited* — a failed record becomes a run-visible issue, which nothing did
    before — and poses that carry none are validated here. A pose that no one
    checked and no one flagged is the one that reaches a report as a number.
``structure.dock.gnina``
    Docking scores, reproduced. Opt-in ``rescore_backend='gnina'`` re-scores the
    *stored coordinates* with ``score_pose`` — scoring only, no search — and
    flags any pose whose recorded score cannot be reproduced from the file it was
    written to. The producer searches; the evaluator only scores what it is given.
``synth.route.aizynthfinder``
    Makeability, checked independently. Opt-in ``route_backend='aizynthfinder'``
    searches routes for the best few candidates and raises an issue where there
    is none. The synthesis agent uses the same engine to *fill in*
    ``SynthesisPlan.routes``; this asks a different question — is what we are
    about to recommend actually makeable — and answers it about a bounded subset,
    because a tree search per candidate is minutes of work.
``property.admet.admetai``
    Safety, cross-checked. Opt-in ``admet_backend='admetai'`` runs an independent
    endpoint panel over the same subset and reports where it and the run's own
    safety model land on opposite sides of their shared 0.5 decision boundary.
    A disagreement is not a verdict about the molecule; it is a fact about the
    two models, and it is reported as one.

All four of the optional arms are off by default and every one of them, wired and
absent, is skipped with its reason recorded — a run that did not ask for them
costs exactly what it always cost, and one that did and cannot run them is told
so rather than being handed a quiet pass.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest, AgentResult, ValidationIssue
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism
from ...tools.properties.admetai_tool import AdmetAiAdapter
from ...tools.rdkit_tool import Cheminformatics
from ...tools.retrosynthesis.aizynthfinder_tool import AiZynthFinderAdapter
from ...tools.safc_tool import SafcAdapter
from ...tools.structure import StructureError
from ...tools.structure.gnina_tool import GninaAdapter
from ...tools.structure.posebusters_tool import PoseBustersAdapter
from . import PlatformAgent

if TYPE_CHECKING:
    from ...orchestration.context import RunContext

#: Names each optional arm accepts. Full tool ids work too, so a config can name
#: either. ``auto`` means "use it if this host can", which is how the docking and
#: route arms are switched on elsewhere in the platform.
RESCORE_BACKENDS = frozenset({"auto", "gnina", GninaAdapter.capability_id})
ROUTE_BACKENDS = frozenset({"auto", "aizynthfinder", AiZynthFinderAdapter.capability_id})
ADMET_BACKENDS = frozenset({"auto", "admetai", AdmetAiAdapter.capability_id})
SAFC_BACKENDS = frozenset({"auto", "safc", SafcAdapter.capability_id})
OFF = frozenset({"", "none"})

#: How many candidates the expensive audits look at. Bounded on purpose: a
#: retrosynthetic search is minutes per molecule, and an audit of everything is an
#: audit nobody can afford to run.
DEFAULT_AUDIT_SIZE = 8

#: How far a re-scored pose may sit from its recorded score before the record is
#: called unreproducible, in kcal/mol. Re-scoring the *same* coordinates with the
#: *same* engine should land within a rounding error of the original; a whole
#: kcal/mol is deliberately generous, so what this flags is a broken pose file or
#: a score that never came from these coordinates, not scoring noise.
RESCORE_TOLERANCE_KCAL = 1.0

#: The decision boundary two probability models must agree on to be said to agree
#: at all. Not a threshold this agent invents for a molecule — it is the point
#: where each model's own probability changes its mind.
PROBABILITY_BOUNDARY = 0.5


class EvaluationFeedbackAgent(PlatformAgent):
    """Independent evaluator — never the producer of the results it validates."""

    agent_id = "agent.evaluation"
    actions = {"validate_all_results": "Independent validation of run results."}

    def __init__(
        self,
        *,
        settings: Any = None,
        validate_poses: bool = True,
        rescore_backend: str = "none",
        route_backend: str = "none",
        admet_backend: str = "none",
        # Off by default like the other three, and for a sharper reason: one SAFC
        # score is one Vina run per receptor conformation, so a submission-sized
        # batch is thousands of docks and hours of wall clock. That is a decision
        # a run makes deliberately, never a default an evaluator inherits.
        safc_backend: str = "none",
        audit_size: int = DEFAULT_AUDIT_SIZE,
    ) -> None:
        self.chem = Cheminformatics()
        self.validator = PoseBustersAdapter(settings=settings) if validate_poses else None
        self.rescorer = self._wire("rescore", rescore_backend, RESCORE_BACKENDS, GninaAdapter, settings)
        self.router = self._wire("route", route_backend, ROUTE_BACKENDS, AiZynthFinderAdapter, settings)
        self.admet = self._wire("admet", admet_backend, ADMET_BACKENDS, AdmetAiAdapter, settings)
        self.safc = self._wire("safc", safc_backend, SAFC_BACKENDS, SafcAdapter, settings)
        self.audit_size = max(0, int(audit_size))

    @staticmethod
    def _wire(label: str, name: str, accepted: frozenset[str], cls: Any, settings: Any) -> Any:
        """One optional arm, refusing a name it does not drive rather than ignoring it."""
        wanted = (name or "none").strip().lower()
        if wanted in OFF:
            return None
        if wanted not in accepted:
            known = ", ".join(sorted(n for n in accepted if "." not in n))
            raise ValueError(f"unknown {label} backend {name!r}; this agent drives: none, {known}")
        return cls(settings=settings)

    def describe_capability(self) -> CapabilityProfile:
        wired = [
            t.backend
            for t in (self.validator, self.rescorer, self.router, self.admet, self.safc)
            if t is not None
        ]
        return CapabilityProfile(
            capability_id=self.agent_id,
            version="1.0.0",
            title="Evaluation & feedback",
            description="Independent schema/identity/pose/route/safety validation of a run's own output.",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE,
            backend=",".join(["builtin", f"identity={self.chem.profile.backend}", *wired]),
            validation_checks=[
                "identity present and re-derivable from the stored SMILES",
                "predictions self-describing",
                "no unsupported safety claims",
                "poses judged, and failed judgements surfaced",
                "recommended candidates are makeable (when a route engine is wired)",
            ],
        )

    # --- execution --------------------------------------------------------
    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        issues: list[ValidationIssue] = []
        warnings: list[str] = []
        metrics: dict[str, float] = {}
        candidates = ctx.store.list_objects("candidate", run_id=ctx.run.id)
        predictions = ctx.store.list_objects("prediction", run_id=ctx.run.id)

        # 1) identity, re-derived rather than merely present
        issues.extend(self._check_identity(candidates))

        # 2) every prediction is self-describing (endpoint/unit/model/uncertainty-or-limitation)
        for p in predictions:
            q = p.get("quantity") or {}
            has_unc = q.get("std") is not None or p.get("limitation")
            if not (p.get("endpoint") and p.get("model_version") and has_unc):
                issues.append(
                    ValidationIssue(
                        code="prediction_incomplete",
                        severity="warning",
                        message=f"prediction {p.get('id')} under-specified",
                    )
                )

        # 3) no unsupported safety claims: safety profiles must carry the disclaimer
        for s in ctx.store.list_objects("safety", run_id=ctx.run.id):
            if s.get("no_signal") and "not a claim of safety" not in (s.get("disclaimer", "")).lower():
                issues.append(
                    ValidationIssue(
                        code="safety_claim", severity="error", message="safety profile missing disclaimer"
                    )
                )

        # 4) the coordinates, and 5-6) the bounded audits
        self._audit_poses(request, ctx, issues, warnings, metrics)
        subset = self._audit_subset(candidates, predictions)
        self._audit_routes(ctx, subset, issues, warnings, metrics)
        self._audit_safety(ctx, subset, issues, warnings, metrics)

        n_err = sum(1 for i in issues if i.severity == "error")
        status = "error" if n_err else "ok"
        return self._result(
            request,
            status=status,
            validation_issues=issues,
            payload={
                "n_candidates": len(candidates),
                "n_predictions": len(predictions),
                "n_errors": n_err,
                "n_warnings": len(issues) - n_err,
            },
            metrics={
                "validation_errors": float(n_err),
                "validation_warnings": float(len(issues) - n_err),
                **metrics,
            },
            warnings=warnings,
            explanation=f"Independently validated {len(candidates)} candidates / "
            f"{len(predictions)} predictions: {n_err} errors, {len(issues) - n_err} warnings.",
        )

    # --- identity ---------------------------------------------------------
    def _check_identity(self, candidates: list[dict]) -> list[ValidationIssue]:
        """Identity present, and the same molecule when derived a second time.

        Cheap enough to be unconditional, and it is the one check whose failure
        invalidates everything joined on ``inchikey`` — lineage, de-duplication,
        the assay join, the report's own tables.

        An InChIKey has two halves and they fail differently. The first block is
        the **skeleton** hash: if that moves, the stored SMILES is a different
        molecule from the stored identity and everything joined on either is
        wrong, so it is an error. The second block carries stereochemistry, and it
        moves for a benign, systematic reason on this platform —
        :meth:`Cheminformatics._standardize` writes ``canonical_smiles`` with
        ``isomericSmiles=False`` while taking the InChIKey from the stereo-bearing
        molecule, so a candidate with a defined stereocentre stores a flat SMILES
        beside a stereo key, and the stereo survives only on
        ``Molecule.isomeric_smiles``. That is worth saying out loud on every run it
        happens in — anything that re-derives identity from ``canonical_smiles``
        gets a different key — but it is not a corrupted candidate, so it is a
        warning and it names where the stereochemistry did go.
        """
        issues: list[ValidationIssue] = []
        for c in candidates:
            stored = str(c.get("inchikey") or "")
            if not stored:
                issues.append(
                    ValidationIssue(
                        code="identity", severity="error", message=f"candidate {c['id']} lacks InChIKey"
                    )
                )
                continue
            smiles = c.get("canonical_smiles")
            if not smiles:
                continue
            rederived = self.chem.standardize(str(smiles)).inchikey or ""
            if not rederived or rederived == stored:
                continue
            if rederived.split("-")[0] != stored.split("-")[0]:
                issues.append(
                    ValidationIssue(
                        code="identity_mismatch",
                        severity="error",
                        message=(
                            f"candidate {c['id']} stores {stored} but its SMILES re-derives to "
                            f"{rederived}: a different skeleton, so one of the two is not this molecule"
                        ),
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        code="identity_stereo_dropped",
                        severity="warning",
                        message=(
                            f"candidate {c['id']} stores {stored} but its canonical_smiles re-derives "
                            f"to {rederived}: same skeleton, no stereochemistry — the stored SMILES is "
                            f"flat and the stereo is only on the molecule's isomeric_smiles"
                        ),
                    )
                )
        return issues

    # --- poses ------------------------------------------------------------
    def _receptor_pdb(self, request: AgentRequest, ctx: RunContext) -> Path | None:
        """The pocket the poses were docked into, from the run or the request."""
        detected = ctx.blackboard.get("pocket") if hasattr(ctx, "blackboard") else None
        raw = request.inputs.get("receptor_pdb") or getattr(detected, "pdb_path", None)
        if not raw:
            brief = request.inputs.get("brief") or {}
            pocket = ((brief.get("generation") or {}) if isinstance(brief, dict) else {}).get("pocket") or {}
            raw = pocket.get("pdb_path")
        if not raw:
            return None
        path = Path(str(raw)).expanduser().resolve()
        return path if path.is_file() else None

    def _audit_poses(
        self,
        request: AgentRequest,
        ctx: RunContext,
        issues: list[ValidationIssue],
        warnings: list[str],
        metrics: dict[str, float],
    ) -> None:
        """Judge the coordinates, and surface judgements nobody surfaced.

        A pose validation that failed used to live only on ``pose.qc_flags`` and in
        a ``pose_validation`` record. Neither is a validation *issue*, so a run
        could ship a physically impossible pose without its own evaluator ever
        saying a word.
        """
        poses = ctx.store.list_objects("pose", run_id=ctx.run.id)
        if not poses:
            return
        recorded = {
            str(r.get("candidate_id")): r
            for r in ctx.store.list_objects("pose_validation", run_id=ctx.run.id)
        }
        receptor = self._receptor_pdb(request, ctx)
        n_new = 0
        for pose in poses:
            cid = str(pose.get("candidate_id"))
            report = recorded.get(cid)
            if report is None and self.validator is not None and self.validator.available():
                report = self._validate_pose(cid, pose, receptor, ctx)
                n_new += int(report is not None)
            if report is not None and not report.get("passed", True):
                failures = ", ".join(report.get("failures") or []) or "unspecified checks"
                issues.append(
                    ValidationIssue(
                        code="pose_implausible",
                        severity="warning",
                        message=f"pose for {cid} failed physical validation: {failures}",
                    )
                )
        if self.validator is not None and not self.validator.available():
            warnings.append(
                f"{self.validator.capability_id} produced NOTHING: "
                f"{self.validator.profile.unavailable_reason or 'unavailable'}"
            )
        metrics["n_poses_audited"] = float(len(poses))
        if n_new:
            metrics["n_poses_validated_here"] = float(n_new)
        self._rescore_poses(poses, receptor, ctx, issues, warnings, metrics)

    def _validate_pose(
        self, candidate_id: str, pose: dict, receptor: Path | None, ctx: RunContext
    ) -> dict | None:
        """Run the physical checks on a pose nobody checked, and persist the report."""
        from ..specialists.structure_agent import sdf_record  # the one writer of this format

        block = pose.get("pose_ref")
        if not block:
            return None
        with TemporaryDirectory(prefix="drugagent_eval_pose_") as tmp:
            sdf = Path(tmp) / f"{candidate_id}.sdf"
            sdf.write_text(sdf_record(str(block)))
            report = self.validator.validate(  # type: ignore[union-attr]
                candidate_id=candidate_id, pose_sdf=sdf, receptor_pdb=receptor
            )
        record = {"id": f"posecheck_eval_{candidate_id}", **asdict(report)}
        ctx.store.put_object("pose_validation", record, run_id=ctx.run.id)
        return record

    def _rescore_poses(
        self,
        poses: list[dict],
        receptor: Path | None,
        ctx: RunContext,
        issues: list[ValidationIssue],
        warnings: list[str],
        metrics: dict[str, float],
    ) -> None:
        """Re-score stored coordinates and check the recorded score comes back."""
        if self.rescorer is None:
            return
        if not self.rescorer.available():
            warnings.append(
                f"{self.rescorer.capability_id} produced NOTHING: "
                f"{self.rescorer.profile.unavailable_reason or 'unavailable'}"
            )
            return
        if receptor is None:
            warnings.append(
                f"{self.rescorer.capability_id} produced NOTHING: no receptor file to score against "
                f"(pass 'receptor_pdb', or detect a pocket first)"
            )
            return
        deltas: list[float] = []
        for pose in poses:
            cid = str(pose.get("candidate_id"))
            block, recorded = pose.get("pose_ref"), pose.get("score")
            if not block:
                continue
            try:
                scored = self._score_one(cid, str(block), receptor)
            except StructureError as exc:
                issues.append(
                    ValidationIssue(
                        code="pose_unscorable",
                        severity="warning",
                        message=f"pose for {cid} could not be re-scored: {exc}",
                    )
                )
                continue
            delta = None if (recorded is None or scored.score is None) else abs(scored.score - recorded)
            ctx.store.put_object(
                "pose_rescore",
                {
                    "id": f"rescore_{cid}",
                    "candidate_id": cid,
                    "recorded_score": recorded,
                    "independent_score": scored.score,
                    "score_unit": scored.score_unit,
                    "engine": scored.engine,
                    "engine_version": scored.engine_version,
                    "delta": delta,
                },
                run_id=ctx.run.id,
            )
            if delta is None:
                continue
            deltas.append(delta)
            if delta > RESCORE_TOLERANCE_KCAL:
                issues.append(
                    ValidationIssue(
                        code="pose_score_unreproducible",
                        severity="warning",
                        message=(
                            f"pose for {cid} scores {scored.score:.2f} on re-scoring but was recorded as "
                            f"{recorded:.2f} {scored.score_unit}"
                        ),
                    )
                )
        if deltas:
            metrics["n_poses_rescored"] = float(len(deltas))
            metrics["rescore_mean_abs_delta"] = round(sum(deltas) / len(deltas), 4)

    def _score_one(self, candidate_id: str, block: str, receptor: Path) -> Any:
        from ..specialists.structure_agent import sdf_record

        with TemporaryDirectory(prefix="drugagent_eval_score_") as tmp:
            sdf = Path(tmp) / f"{candidate_id}.sdf"
            sdf.write_text(sdf_record(block))
            return self.rescorer.score_pose(  # type: ignore[union-attr]
                candidate_id=candidate_id, receptor_pdb=receptor, pose_sdf=sdf
            )

    # --- the bounded audits ------------------------------------------------
    def _audit_subset(self, candidates: list[dict], predictions: list[dict]) -> list[dict]:
        """The few candidates the expensive audits look at, chosen deterministically.

        The best predicted potency first, because the question these audits ask is
        about what the run is going to recommend. Ties and missing predictions fall
        back to candidate id, so the subset is the same on every rerun.
        """
        if not self.audit_size or not (self.router or self.admet):
            return []
        best: dict[str, float] = {}
        for p in predictions:
            if p.get("kind") != "activity":
                continue
            value = (p.get("quantity") or {}).get("value")
            cid = str(p.get("candidate_id"))
            if value is not None:
                best[cid] = max(best.get(cid, float("-inf")), float(value))
        ordered = sorted(candidates, key=lambda c: (-best.get(str(c["id"]), float("-inf")), str(c["id"])))
        return ordered[: self.audit_size]

    def _audit_routes(
        self,
        ctx: RunContext,
        subset: list[dict],
        issues: list[ValidationIssue],
        warnings: list[str],
        metrics: dict[str, float],
    ) -> None:
        """Is what this run is about to recommend actually makeable?"""
        if self.router is None:
            return
        if not self.router.available():
            warnings.append(
                f"{self.router.capability_id} produced NOTHING: "
                f"{self.router.profile.unavailable_reason or 'unavailable'}"
            )
            return
        solved = 0
        for candidate in subset:
            cid = str(candidate["id"])
            # The audit is optional, so its failure costs the audit. An evaluator
            # that raises would fail `validate_all_results` and take the whole
            # cycle's scoring with it — the arm exists to add confidence, and it
            # must not be able to subtract results.
            try:
                plan = self.router.plan(candidate_id=cid, smiles=str(candidate["canonical_smiles"]))
            except Exception as exc:
                warnings.append(
                    f"{self.router.capability_id} produced NOTHING for {cid}: {type(exc).__name__}: {exc}"
                )
                break
            feasible = [r for r in plan.routes if r.feasible]
            solved += bool(feasible)
            ctx.store.put_object(
                "route_audit",
                {
                    "id": f"routeaudit_{cid}",
                    "candidate_id": cid,
                    "engine": self.router.backend,
                    "n_routes": len(plan.routes),
                    "n_feasible": len(feasible),
                    "best_route_id": plan.best_route_id,
                    "building_block_availability": plan.building_block_availability,
                },
                run_id=ctx.run.id,
            )
            if not feasible:
                issues.append(
                    ValidationIssue(
                        code="no_feasible_route",
                        severity="warning",
                        message=f"{cid} is among the best-scoring candidates and has no route to stock",
                    )
                )
        if subset:
            metrics["n_routes_audited"] = float(len(subset))
            metrics["route_audit_solved_fraction"] = round(solved / len(subset), 4)

    def _audit_safety(
        self,
        ctx: RunContext,
        subset: list[dict],
        issues: list[ValidationIssue],
        warnings: list[str],
        metrics: dict[str, float],
    ) -> None:
        """A second opinion on the endpoints the run's own safety model reported."""
        if self.admet is None:
            return
        if not self.admet.available():
            warnings.append(
                f"{self.admet.capability_id} produced NOTHING: "
                f"{self.admet.profile.unavailable_reason or 'unavailable'}"
            )
            return
        if not subset:
            return
        ids = [str(c["id"]) for c in subset]
        try:
            profiles = self.admet.assess_safety_batch(
                [str(c["canonical_smiles"]) for c in subset], candidate_ids=ids
            )
        except Exception as exc:
            warnings.append(f"{self.admet.capability_id} produced NOTHING: {type(exc).__name__}: {exc}")
            return
        own = self._own_safety_scores(ctx, set(ids))
        deltas: list[float] = []
        for profile in profiles:
            cid = str(profile.candidate_id)
            comparisons: dict[str, dict[str, float]] = {}
            for endpoint in profile.endpoints:
                if endpoint.score is None:
                    continue
                key = endpoint.endpoint.strip().lower()
                mine = own.get((cid, key))
                if mine is None:
                    continue
                delta = abs(float(endpoint.score) - mine)
                deltas.append(delta)
                comparisons[key] = {"independent": float(endpoint.score), "run": mine, "delta": delta}
                if (float(endpoint.score) >= PROBABILITY_BOUNDARY) != (mine >= PROBABILITY_BOUNDARY):
                    issues.append(
                        ValidationIssue(
                            code="safety_disagreement",
                            severity="warning",
                            message=(
                                f"{cid} {endpoint.endpoint}: this run reports {mine:.2f} and "
                                f"{self.admet.backend} reports {endpoint.score:.2f} — the two models "
                                f"disagree about which side of {PROBABILITY_BOUNDARY} it is on"
                            ),
                        )
                    )
            ctx.store.put_object(
                "safety_audit",
                {
                    "id": f"safetyaudit_{cid}",
                    "candidate_id": cid,
                    "engine": self.admet.backend,
                    "n_endpoints": len(profile.endpoints),
                    "compared": comparisons,
                },
                run_id=ctx.run.id,
            )
        metrics["n_safety_audited"] = float(len(profiles))
        if deltas:
            metrics["safety_mean_abs_delta"] = round(sum(deltas) / len(deltas), 4)

    @staticmethod
    def _own_safety_scores(ctx: RunContext, ids: set[str]) -> dict[tuple[str, str], float]:
        """This run's own per-endpoint safety probabilities, keyed for comparison."""
        out: dict[tuple[str, str], float] = {}
        for profile in ctx.store.list_objects("safety", run_id=ctx.run.id):
            cid = str(profile.get("candidate_id"))
            if cid not in ids:
                continue
            for endpoint in profile.get("endpoints") or []:
                score = endpoint.get("score")
                name = str(endpoint.get("endpoint", "")).strip().lower()
                if score is not None and name:
                    out[(cid, name)] = float(score)
        return out
