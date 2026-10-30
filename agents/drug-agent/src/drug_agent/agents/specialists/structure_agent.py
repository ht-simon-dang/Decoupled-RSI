"""Molecular function & dynamics — coordinates: measured, docked, predicted, judged.

Six tools behind one agent, and they do not have the same availability. That
asymmetry is deliberate and visible: the agent does not average over it or call
itself DEGRADED, because a run either has poses or it does not.

``structure.simulate``
    A real MMFF94s conformer minimisation. Runs for every candidate, everywhere.
``structure.dock.gnina`` / ``structure.dock``
    Docking, and since repair R1 there are two real backends rather than one.
    ``dock_backend="auto"`` drives GNINA when this host can really run it and
    otherwise falls back to AutoDock Vina, which runs as a subprocess out of its
    own conda environment; a host with neither reports the same absence, for the
    same reason, that it reported when neither existed. Both need what docking
    has always needed and this pipeline never supplied: a *prepared pocket PDB*
    and a search box. Both come off the brief's
    :class:`~drug_agent.contracts.brief.PocketSpec` — the same pocket the
    generative models sample inside — so a brief that names one gets poses and a
    brief that does not gets a warning saying exactly that, rather than a
    descriptor-derived pose. The two scores are both kcal/mol and are still not
    interchangeable: they are different empirical functions over different ligand
    preparations, and nothing here averages or compares them.
``structure.validate.posebusters``
    The judge of the coordinates the docking arm produced. It runs only when
    there are poses, because there is nothing else it could be asked about.
``structure.complex.boltz2`` / ``structure.predict.protenix``
    A predicted complex, wired in by ``complex_backend`` — Boltz-2 with its
    affinity head, or Protenix (open AlphaFold3) with its confidence head and no
    affinity claim at all. Both are off by default: a predicted complex costs
    minutes per candidate and a run that did not ask for one must cost what it
    always cost.
``structure.experimental``
    The local RCSB coordinate cache, and the reason the two complex models are
    reachable at all. They need the target's protein *sequence*, and the note
    here used to read "nothing in a brief carries [one] today". A brief carries
    PDB ids, and a deposited file carries SEQRES, so the sequence is now read out
    of the measured structure the brief already names — offline, from the cache,
    with the entry's own provenance. A caller with a construct of its own still
    passes ``protein_sequence`` and wins.

Every one of the optional three is skipped-with-a-reason when it cannot compute
here. None of them is ever substituted for another: a missing arm costs the run
that arm, never the run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest
from ...contracts.base import Quantity
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import ApplicabilityDomain, CapabilityStatus, TrustLabel
from ...contracts.predictions import Prediction
from ...tools.mmff_tool import SimulationAdapter
from ...tools.structure import StructureError
from ...tools.structure.boltz2_tool import Boltz2Adapter
from ...tools.structure.experimental_tool import ExperimentalStructureAdapter
from ...tools.structure.gnina_tool import GninaAdapter
from ...tools.structure.posebusters_tool import PoseBustersAdapter
from ...tools.structure.protenix_tool import ProtenixAdapter
from ...tools.vina_tool import DockingAdapter
from . import Assessment, CandidateAssessmentAgent, unavailable_note

if TYPE_CHECKING:
    from ...contracts.brief import PocketSpec
    from ...contracts.structure import Pose
    from ...orchestration.context import RunContext

#: How many candidates in one batch get a real docking run.
#:
#: Docking was free while it produced nothing. Repair R1 made it real, and one
#: flexible drug-like ligand is **~10 s measured on this host** — 8.9 s for GNINA
#: and 10.5 s for Vina on the ZAK 6JUU site at ``--exhaustiveness 8``. This step
#: runs on every candidate that passed the hard filters, ninety of them in a
#: routine cycle, so an uncapped arm is fifteen minutes for one arm of one cycle
#: against an engine task budget of ``max(default_task_timeout_s,
#: per_candidate_task_timeout_s * n)`` — 120 s floor, 3 s per candidate. Uncapped
#: docking does not fit that budget at any batch size, so the arm would be killed
#: mid-batch and the run would lose it entirely.
#:
#: This is exactly the cap :data:`~drug_agent.agents.specialists.synthesis_agent.
#: DEFAULT_ROUTE_BUDGET` exists for, and it works the same way: candidates are
#: docked in batch order, and :meth:`StructureDynamicsAgent.batch_warnings` says
#: how many were left out. Batch order is not a ranking and this cap is not a
#: sample of the best candidates — docking happens before anything is ranked, so
#: there is no "best" to take yet. What the cap buys is an affordable, honest
#: answer about part of the batch instead of an unaffordable one about all of it.
#: Raise it on a host that can afford it; set it to 0 to switch the arm off while
#: keeping the capability reachable.
DEFAULT_DOCK_BUDGET = 8

#: Names ``complex_backend`` accepts, and which model each one binds. ``auto``
#: means Boltz-2, which is the one of the two that also predicts an affinity.
COMPLEX_BACKENDS: dict[str, type[Boltz2Adapter] | type[ProtenixAdapter]] = {
    "auto": Boltz2Adapter,
    "boltz2": Boltz2Adapter,
    Boltz2Adapter.capability_id: Boltz2Adapter,
    "protenix": ProtenixAdapter,
    ProtenixAdapter.capability_id: ProtenixAdapter,
}


def select_docking(backend: str, *, settings: Any = None) -> DockingAdapter | GninaAdapter:
    """Resolve ``dock_backend`` to the adapter this agent will ask for poses.

    ``auto`` prefers GNINA when this host can really run it, then AutoDock Vina,
    and otherwise leaves the vina-shaped :class:`DockingAdapter` in place to
    report its own absence. GNINA is preferred not because its empirical score is
    better — it is Vina's, by descent — but because it also returns a CNN pose
    score, which is an independent opinion about the pose the search produced.
    """
    wanted = (backend or "auto").strip().lower()
    if wanted in {"gnina", GninaAdapter.capability_id}:
        return GninaAdapter(settings=settings)
    if wanted in {"vina", DockingAdapter.capability_id}:
        return DockingAdapter(backend="vina", settings=settings)
    if wanted == "none":
        # The documented disable switch. It used to work by accident, because the
        # vina-shaped adapter had no backend; now that it has one, switching
        # docking off has to be something this agent actually asks for.
        return DockingAdapter(backend="none", settings=settings)
    if wanted != "auto":
        raise ValueError(f"unknown docking backend {backend!r}; this agent drives: auto, gnina, vina, none")
    gnina = GninaAdapter(settings=settings)
    return gnina if gnina.available() else DockingAdapter(backend="auto", settings=settings)


def sdf_record(block: str) -> str:
    """One pose's mol block as a well-formed SDF record.

    :class:`~drug_agent.contracts.structure.Pose` is a ``SchemaModel``, and those
    strip surrounding whitespace on assignment — which eats a mol block's *first*
    line, the empty molecule title. The three-line header then reads as two, the
    counts line lands where an atom line is expected, and every consumer that
    hands ``pose_ref`` to a chemistry library gets "could not load molecule"
    rather than a pose. Restoring the line here keeps the validator working; the
    contract is where a mol block should stop being whitespace-stripped.
    """
    title = "" if block.startswith("\n") else "\n"
    return f"{title}{block}\n$$$$\n"


@dataclass(slots=True)
class Receptor:
    """What a run knows about the protein its candidates are assessed against.

    Assembled once per batch from the request inputs and the brief, because every
    candidate is docked into the same site and re-deriving it per molecule would
    be a per-molecule chance to disagree with itself.
    """

    receptor_id: str | None = None
    pocket_pdb: Path | None = None
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None
    sequence: str | None = None
    #: Why docking cannot run on this batch. Empty exactly when it can.
    blocker: str = ""

    @property
    def can_dock(self) -> bool:
        return bool(self.pocket_pdb and self.center and self.size)


#: Returned by :meth:`StructureDynamicsAgent._reserve_dock` when the campaign
#: ledger refused the call. A distinct sentinel rather than ``None`` because
#: "no ledger is attached" and "the ledger said no" must not be the same answer.
_REFUSED = object()


class StructureDynamicsAgent(CandidateAssessmentAgent):
    agent_id = "agent.structure"
    actions = {"assess_structure": "Dock, validate poses, predict complexes, minimise conformers."}
    batch_label = "dock"
    count_key = "n_assessed"

    def __init__(
        self,
        dock_backend: str = "auto",
        sim_backend: str = "auto",
        *,
        settings: Any = None,
        validate_poses: bool = True,
        complex_backend: str = "none",
        dock_budget: int = DEFAULT_DOCK_BUDGET,
        budget_ledger: Any = None,
        budget_resource: str = "dock_calls",
        budget_owner: str = "agent.structure",
    ) -> None:
        self.dock = select_docking(dock_backend, settings=settings)
        self.dock_budget = max(0, int(dock_budget))
        #: How many candidates this batch was docked, and how many wanted docking
        #: and did not get it because the budget ran out. Both reported, never
        #: silently dropped: a portfolio where only eight of ninety molecules have
        #: a pose is a different object from one where all ninety do.
        self._docks_run = 0
        self._docks_skipped = 0
        #: Step 9's campaign ledger, when this agent runs inside a budgeted
        #: campaign. ``dock_budget`` is a *per-batch* cap that lives in this
        #: object's memory and resets with every new agent; the ledger is the
        #: campaign-wide account that survives restarts and is shared across
        #: processes. Attaching one brings an existing cap into the ledger's view
        #: instead of creating a second, disagreeing one: the per-batch limit
        #: still applies first, and each dock the batch limit allows is then
        #: reserved and settled against the campaign's declared cap.
        self.budget_ledger = budget_ledger
        self.budget_resource = budget_resource
        self.budget_owner = budget_owner
        #: Docks the campaign ledger refused, as distinct from docks the
        #: per-batch budget refused. A molecule with no pose because the campaign
        #: ran out of budget is a recorded fact, not a missing row.
        self._docks_refused_by_ledger = 0
        self.sim = SimulationAdapter(backend=sim_backend)
        # Pose validation is on by default because it costs nothing when there are
        # no poses, and an unchecked pose is how a physically impossible geometry
        # reaches a scorecard as a number.
        self.validator = PoseBustersAdapter(settings=settings) if validate_poses else None
        wanted = (complex_backend or "none").strip().lower()
        if wanted not in COMPLEX_BACKENDS and wanted not in {"none", ""}:
            raise ValueError(
                f"unknown complex backend {complex_backend!r}; this agent drives: none, boltz2, protenix"
            )
        model = COMPLEX_BACKENDS.get(wanted)
        self.complex_model = model(settings=settings) if model is not None else None
        #: Reasons the complex arm produced nothing this batch. Collected, never
        #: raised: it is optional, and an optional arm must not fail the run.
        self._complex_failures: list[str] = []
        # The measured-coordinate cache. Not an arm of its own: it is how a PDB id
        # in the brief becomes the protein sequence the complex models need, and
        # it costs one file parse per batch.
        self.structures = ExperimentalStructureAdapter(settings=settings)

    def describe_capability(self) -> CapabilityProfile:
        # The agent is AVAILABLE while at least one of its tools can really
        # compute, and its backend string names which. DEGRADED would be a
        # kinder word than the situation deserves when docking contributes
        # nothing: a run either has poses or it does not.
        p = self.sim.profile
        p.capability_id = self.agent_id
        p.title = "Molecular function & dynamics"
        p.status = (
            CapabilityStatus.AVAILABLE
            if (self.dock.available() or self.sim.available())
            else CapabilityStatus.UNAVAILABLE
        )
        p.backend = f"dock={self.dock.backend},sim={self.sim.backend}"
        if self.validator is not None:
            p.backend += f",validate={self.validator.backend}"
        if self.complex_model is not None:
            p.backend += f",complex={self.complex_model.backend}"
        p.unavailable_reason = None if self.dock.available() else self.dock.profile.unavailable_reason
        return p

    def prepare(self, request: AgentRequest, ctx: RunContext) -> Receptor:
        """Resolve the one receptor this batch is assessed against.

        The pocket is the brief's own — ``generation.pocket`` — so the site the
        generative models sample inside is the site the candidates are docked
        into. A caller driving the agent directly may override either half
        through the request inputs.
        """
        # Per batch, not per process.
        self._complex_failures = []
        self._docks_run = 0
        self._docks_skipped = 0
        inputs = request.inputs
        pocket = self._pocket(inputs)
        receptor = Receptor(
            receptor_id=inputs.get("receptor_id")
            or (pocket.receptor_id or pocket.pdb_id if pocket else None),
            sequence=inputs.get("protein_sequence") or self._cached_sequence(inputs, pocket),
        )
        raw_path = inputs.get("receptor_pdb") or (pocket.pdb_path if pocket else None)
        if not raw_path:
            receptor.blocker = "no prepared pocket PDB (brief.generation.pocket.pdb_path)"
            return receptor
        # Absolute, because the docking engines run in their own scratch directory:
        # a relative pocket path resolves against the wrong cwd and the engine says
        # only that it could not open the file.
        pdb = Path(str(raw_path)).expanduser().resolve()
        if not pdb.is_file():
            receptor.blocker = f"prepared pocket PDB {pdb} does not exist"
            return receptor
        receptor.pocket_pdb = pdb
        center = list(pocket.center or []) if pocket else []
        if pocket is None or len(center) < 3:
            # A box is not optional and there is no sane default: docking the
            # whole protein would be a different experiment reported as this one.
            receptor.blocker = "the pocket names no centre, so there is no search box"
            return receptor
        edge = 2.0 * float(pocket.radius)
        receptor.center = (float(center[0]), float(center[1]), float(center[2]))
        receptor.size = (edge, edge, edge)
        return receptor

    def _cached_sequence(self, inputs: dict[str, Any], pocket: PocketSpec | None) -> str | None:
        """The target's sequence, read out of the measured structure the brief names.

        The complex models take one polymer sequence, and a deposited entry may
        hold several chains — 3I28 is a homodimer. The longest polymer chain is
        used, which is the construct that was crystallised; a caller that means a
        specific assembly passes ``protein_sequence`` and this is never consulted.
        Absent cache, absent entry or absent SEQRES all yield ``None``, and the
        complex arm then says what it is missing instead of guessing at it.
        """
        if not self.structures.available():
            return None
        from ...contracts.brief import ResearchBrief  # local import: avoids a cycle

        raw = inputs.get("brief")
        brief = ResearchBrief.model_validate(raw) if raw else None
        ids = [i for i in (pocket.pdb_id if pocket else None,) if i]
        if brief is not None and brief.target is not None:
            ids.extend(brief.target.pdb_ids)
        for pdb_id in ids:
            try:
                chains = self.structures.resolve(str(pdb_id)).sequences
            except StructureError:
                continue
            if chains:
                return max(chains.values(), key=len)
        return None

    @staticmethod
    def _pocket(inputs: dict[str, Any]) -> PocketSpec | None:
        """The brief's binding site, if this run has one."""
        from ...contracts.brief import ResearchBrief  # local import: avoids a cycle

        raw = inputs.get("brief")
        if not raw:
            return None
        brief = ResearchBrief.model_validate(raw)
        return brief.generation.pocket if brief.generation else None

    def assess_candidate(self, candidate: dict, prep: Receptor, ctx: RunContext) -> Assessment:
        """Dock, judge the pose, predict the complex, and minimise — each if real.

        The four tools have different availability and different inputs, and every
        branch below is a real computation or nothing at all. Neither fabricates,
        and a run that produced no poses says so.
        """
        assessment = Assessment()
        poses = self._dock_candidate(candidate, prep, ctx, assessment)
        if poses:
            best = poses[0]
            assessment.objects.append(("pose", best))
            assessment.predictions.append(
                Prediction(
                    candidate_id=candidate["id"],
                    molecule_inchikey=candidate.get("inchikey"),
                    endpoint="dock_score",
                    kind="structure",
                    quantity=Quantity(
                        value=best.score, unit=best.score_unit, trust_label=TrustLabel.CALCULATED
                    ),
                    applicability=ApplicabilityDomain.BORDERLINE,
                    model_id=f"dock:{best.engine}",
                    model_version=best.engine_version,
                    trust_label=TrustLabel.CALCULATED,
                )
            )
            self._validate_pose(candidate, best, prep, assessment)
        self._predict_complex(candidate, prep, ctx, assessment)

        if self.sim.available():
            sim = self.sim.simulate(candidate["id"], candidate["canonical_smiles"], prep.receptor_id)
            assessment.objects.append(("simulation", sim))
            strain = sim.observables.get("strain_span_kcal_mol")
            if strain is not None:
                assessment.predictions.append(
                    Prediction(
                        candidate_id=candidate["id"],
                        molecule_inchikey=candidate.get("inchikey"),
                        endpoint="strain_span",
                        kind="structure",
                        quantity=Quantity(value=strain, unit="kcal/mol", trust_label=TrustLabel.CALCULATED),
                        applicability=ApplicabilityDomain.IN_DOMAIN,
                        model_id=f"sim:{sim.engine}",
                        model_version=sim.engine_version,
                        trust_label=TrustLabel.CALCULATED,
                        limitation="ligand-only conformational strain; not a binding estimate",
                    )
                )
        return assessment

    # --- docking -----------------------------------------------------------
    def _dock_candidate(
        self, candidate: dict, prep: Receptor, ctx: RunContext, assessment: Assessment
    ) -> list[Pose]:
        """Poses for one candidate, or none — and a record of why there were none.

        A ligand the engine cannot place is a fact about that molecule, not about
        the backend, so it is written down and the batch continues. A backend that
        is absent produced nothing for every candidate and says so once, in
        :meth:`batch_warnings`, rather than once per molecule.
        """
        pocket_pdb, center, size = prep.pocket_pdb, prep.center, prep.size
        if not (self.dock.available() and pocket_pdb and center and size):
            return []
        if self._docks_run >= self.dock_budget:
            self._docks_skipped += 1
            return []
        reservation = self._reserve_dock(candidate["id"])
        if reservation is _REFUSED:
            self._docks_refused_by_ledger += 1
            self._docks_skipped += 1
            return []
        self._docks_run += 1
        try:
            # Both backends take the same four facts — which molecule, which
            # receptor file, where the box is and how big it is — and both are
            # keyword-only about the box, because a docking run with a default box
            # would be a different experiment reported as this one.
            poses = self.dock.dock(
                candidate_id=candidate["id"],
                smiles=candidate["canonical_smiles"],
                receptor_pdb=pocket_pdb,
                receptor_id=prep.receptor_id,
                center=center,
                size=size,
                n_poses=1,
                seed=ctx.run.seed,
            )
        except StructureError as exc:
            # A failed dock is spend, not a refund: the engine ran. The
            # reservation is settled as consumed so the campaign's ledger
            # charges the failure Appendix A §4 says to charge.
            self._settle_dock(reservation, consumed=1.0, ok=False)
            assessment.objects.append(
                (
                    "dock_failure",
                    {
                        "id": f"dockfail_{candidate['id']}",
                        "candidate_id": candidate["id"],
                        "engine": self.dock.backend,
                        "reason": str(exc)[:500],
                    },
                )
            )
            return []
        self._settle_dock(reservation, consumed=1.0, ok=True)
        return poses

    # --- campaign budget ---------------------------------------------------
    def _reserve_dock(self, candidate_id: str) -> Any:
        """Hold one docking call in the campaign ledger, or refuse the dock.

        Returns ``None`` when no ledger is attached (the historical behaviour),
        the reservation record when one was granted, and :data:`_REFUSED` when
        the campaign cap is spent — in which case the molecule gets no pose and
        :meth:`batch_warnings` says so, rather than the batch quietly overrunning
        a declared budget.
        """
        if self.budget_ledger is None:
            return None
        from drug_agent.selfevolve.target.budget import BudgetExceeded

        try:
            return self.budget_ledger.reserve(
                self.budget_resource,
                1.0,
                owner=self.budget_owner,
                phase="EVALUATION",
                note=f"dock {candidate_id}",
            )
        except BudgetExceeded:
            return _REFUSED

    def _settle_dock(self, reservation: Any, *, consumed: float, ok: bool) -> None:
        if reservation is None or reservation is _REFUSED:
            return
        self.budget_ledger.settle(
            reservation["reservation_id"],
            consumed=consumed,
            telemetry="MEASURED",
            status="SETTLED" if ok else "FAILED",
        )

    def _validate_pose(self, candidate: dict, pose: Pose, prep: Receptor, assessment: Assessment) -> None:
        """Check the pose's physical plausibility, when a validator is wired and real.

        PoseBusters reads coordinates from a file and a :class:`Pose` carries them
        as a mol block, so the block goes to a scratch SDF for the length of the
        check. A failed check flags the pose it was run on by name: the
        coordinates stay, and what is known about them travels with them.

        The receptor handed over is the same *pocket* file the pose was docked
        into, which is what makes the clash checks meaningful and what makes the
        ``not_too_far_away_{waters,inorganic_cofactors,organic_cofactors}`` checks
        fail on a cut-out that contains none of those. That is why the failing
        checks are recorded by name and why :meth:`batch_metrics` reports the
        per-check rate beside the strict all-checks-passed one.
        """
        if self.validator is None or not self.validator.available() or not pose.pose_ref:
            return
        with TemporaryDirectory(prefix="drugagent_pose_") as tmp:
            sdf = Path(tmp) / f"{candidate['id']}.sdf"
            sdf.write_text(sdf_record(pose.pose_ref))
            report = self.validator.validate(
                candidate_id=candidate["id"], pose_sdf=sdf, receptor_pdb=prep.pocket_pdb
            )
        assessment.objects.append(
            ("pose_validation", {"id": f"posecheck_{candidate['id']}", **asdict(report)})
        )
        if not report.passed:
            pose.qc_flags = [*pose.qc_flags, "posebusters_failed", *report.failures]

    # --- predicted complex -------------------------------------------------
    def _predict_complex(
        self, candidate: dict, prep: Receptor, ctx: RunContext, assessment: Assessment
    ) -> None:
        """Predict the complex and its affinity, when a sequence and a model exist.

        The predicted coordinates are deliberately *not* stored as ``pose``
        objects: they were not docked, and counting them as poses would let a run
        that never docked anything report a docking arm.
        """
        if self.complex_model is None or not self.complex_model.available() or not prep.sequence:
            return
        if self._complex_failures:
            # Broken for the batch, not for this molecule. Retrying a model that
            # costs minutes per candidate, ninety times, to fail identically, is
            # the expensive way to learn nothing.
            return
        # An optional arm may cost its own output and nothing else. A complex
        # model that is switched on, reports itself available and then raises
        # would otherwise end the cycle with nothing scored — see the same guard
        # and the same measured failure in `synthesis_agent._add_routes`.
        try:
            if isinstance(self.complex_model, ProtenixAdapter):
                # Same question, different signature: Protenix samples a *set* of
                # seeds and reports no affinity at all, so nothing here may invent one.
                prediction: Any = self.complex_model.predict(
                    candidate_id=candidate["id"],
                    protein_sequence=prep.sequence,
                    ligand_smiles=candidate["canonical_smiles"],
                    receptor_id=prep.receptor_id,
                    seeds=(int(ctx.run.seed),),
                )
            else:
                prediction = self.complex_model.predict(
                    candidate_id=candidate["id"],
                    protein_sequence=prep.sequence,
                    ligand_smiles=candidate["canonical_smiles"],
                    receptor_id=prep.receptor_id,
                    seed=ctx.run.seed,
                )
        except Exception as exc:
            self._complex_failures.append(
                f"{self.complex_model.capability_id} produced NOTHING: {type(exc).__name__}: {exc}"
            )
            return
        affinity = getattr(prediction, "affinity", None)
        assessment.objects.append(
            (
                "complex",
                {
                    "id": f"complex_{candidate['id']}",
                    "candidate_id": candidate["id"],
                    "receptor_id": prep.receptor_id,
                    "engine": self.complex_model.backend,
                    "engine_version": prediction.engine_version,
                    "confidence": prediction.confidence,
                    "n_poses": len(prediction.poses),
                    "affinity": asdict(affinity) if affinity is not None else None,
                    "qc_flags": list(prediction.qc_flags),
                    "warnings": list(prediction.warnings),
                },
            )
        )
        if affinity is None or affinity.pred_value is None:
            return
        assessment.predictions.append(
            Prediction(
                candidate_id=candidate["id"],
                molecule_inchikey=candidate.get("inchikey"),
                # Upstream's own encoding, under a name that cannot be mistaken for
                # the measured potency the activity model predicts.
                endpoint="affinity_log_ic50_um",
                kind="structure",
                quantity=Quantity(
                    value=float(affinity.pred_value),
                    unit="log10(IC50/uM)",
                    trust_label=TrustLabel.PREDICTED,
                ),
                applicability=ApplicabilityDomain.UNKNOWN,
                model_id=f"complex:{affinity.engine or self.complex_model.backend}",
                model_version=affinity.engine_version or prediction.engine_version,
                trust_label=TrustLabel.PREDICTED,
                limitation="affinity head over a PREDICTED complex; comparable only between actives",
            )
        )

    # --- batch reporting ---------------------------------------------------
    def batch_metrics(self, prep: Receptor, assessments: list[Assessment]) -> dict[str, float]:
        counts = Counter(store_type for a in assessments for store_type, _ in a.objects)
        # n_poses is reported even when it is zero: whether a run produced poses at
        # all is the question this agent exists to answer.
        metrics = {"n_poses": float(counts["pose"])}
        if self._docks_run or self._docks_skipped:
            # The denominator a reader needs is what was DOCKED, not what was
            # assessed: with a budget those differ, and dividing by the batch size
            # would report a pose yield of 0.09 for an arm that placed every ligand
            # it was asked to place.
            metrics["n_docked"] = float(self._docks_run)
            metrics["n_dock_skipped_over_budget"] = float(self._docks_skipped)
        if counts["dock_failure"]:
            metrics["n_dock_failures"] = float(counts["dock_failure"])
        if counts["pose_validation"]:
            reports = [
                obj for a in assessments for store_type, obj in a.objects if store_type == "pose_validation"
            ]
            checks = sum(len(r["checks"]) for r in reports)
            held = sum(1 for r in reports for ok in r["checks"].values() if ok)
            metrics["n_pose_validations"] = float(len(reports))
            metrics["posebusters_pass_rate"] = round(sum(1 for r in reports if r["passed"]) / len(reports), 4)
            # The strict rate is all-or-nothing over 27 checks and goes to zero on
            # a pocket cut-out that has no waters or cofactors to be near. The
            # per-check rate is what moves when a pose is actually bad.
            metrics["posebusters_check_pass_rate"] = round(held / checks, 4) if checks else 0.0
        if counts["complex"]:
            metrics["n_complexes"] = float(counts["complex"])
        return metrics

    def batch_warnings(self, prep: Receptor) -> list[str]:
        warnings: list[str] = []
        if not self.dock.available():
            warnings.append(
                "docking produced NO poses: " + (self.dock.profile.unavailable_reason or "unavailable")
            )
        elif not prep.can_dock:
            warnings.append(f"docking produced NO poses: {prep.blocker}")
        elif self.dock_budget == 0:
            warnings.append(
                f"docking produced NO poses: the per-batch docking budget is 0, so "
                f"{self.dock.backend} was available and never called"
            )
        if self._docks_skipped:
            warnings.append(
                f"docking covered {self._docks_run} of {self._docks_run + self._docks_skipped} "
                f"candidates: the per-batch budget is {self.dock_budget}, and the remainder carry "
                f"no pose and no dock_score"
            )
        if self._docks_refused_by_ledger:
            warnings.append(
                f"the campaign budget ledger refused {self._docks_refused_by_ledger} docking "
                f"call(s) for {self.budget_resource!r}: those candidates carry no pose because "
                f"the declared cap was spent, which is a budget outcome and not a backend failure"
            )
        if self.validator is not None and not self.validator.available():
            warnings.append(unavailable_note(self.validator))
        if self.complex_model is not None:
            if not self.complex_model.available():
                warnings.append(unavailable_note(self.complex_model))
            elif not prep.sequence:
                warnings.append(
                    f"{self.complex_model.capability_id} produced NOTHING: no protein sequence for "
                    f"this target (pass 'protein_sequence' in the request inputs)"
                )
            elif self._complex_failures:
                n = len(self._complex_failures)
                warnings.append(
                    f"complex prediction produced NOTHING for {n} candidate(s): " + self._complex_failures[0]
                )
        if not self.sim.available():  # pragma: no cover - MMFF94s is always available
            warnings.append("conformer minimisation unavailable")
        return warnings

    def explain_batch(self, n: int, prep: Receptor) -> str:
        parts = []
        if not self.dock.available():
            parts.append(f"docking skipped ({self.dock.backend})")
        elif not prep.can_dock:
            parts.append(f"docking skipped ({prep.blocker})")
        else:
            parts.append(
                f"docked {self._docks_run} of {self._docks_run + self._docks_skipped} "
                f"with {self.dock.backend}"
                if self._docks_skipped
                else f"docked with {self.dock.backend}"
            )
            if self.validator is not None and self.validator.available():
                parts.append("poses checked with posebusters")
        if self.complex_model is not None and self.complex_model.available() and prep.sequence:
            parts.append(f"complexes predicted with {self.complex_model.backend}")
        parts.append(f"minimised with {self.sim.backend}" if self.sim.available() else "no minimisation")
        return f"Structure assessment over {n} candidates: " + ", ".join(parts) + "."
