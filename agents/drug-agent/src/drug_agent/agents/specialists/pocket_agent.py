"""Pockets — turn a structure into the one thing the generators refuse to run without.

This agent closes a hole that ran the length of the pipeline. Every
pocket-conditioned generator in the catalogue — PocketXMol, TargetDiff,
PocketFlow, Pocket2Mol, OMTRA, DrugFlow, MolCRAFT, PILOT — needs a *prepared
pocket PDB*: a few hundred atoms cut around a site, not a whole protein. So does
GNINA, which needs the box that pocket implies. Nothing in this platform could
produce one. A brief could carry ``generation.pocket.pdb_path`` if a human had
already cut a file by hand with ``scripts/prepare_pocket.py``, and if it did not,
every one of those tools answered ``generation.pocket.pdb_path=None`` and
produced nothing. The refusal was correct; the gap was that no code path could
ever satisfy it.

Three detectors sit behind one selection, exactly as the docking engines do:

``pocket.detect.p2rank``
    A random forest over solvent-accessible surface points, trained on ligandable
    sites. ``auto`` prefers it, because "which cavity would a ligand bind" is the
    question this pipeline is asking and P2Rank is the only one of the three that
    was trained on it.
``pocket.detect.fpocket``
    Voronoi alpha-sphere geometry. No model, no training set, available wherever
    the conda package is — the fallback when P2Rank is not installed, and the
    right choice when a cavity's *extent* matters more than its ligandability.
``pocket.detect.vnegnn``
    An E(3)-equivariant GNN. UNAVAILABLE on every machine today because upstream
    publishes no checkpoint; selecting it produces nothing and says so.

**The structure comes from the cache, not the network.** ``structure.experimental``
resolves a PDB id or a target name against the local RCSB coordinate cache, so a
brief that names ``pdb_ids: ["5AI0", "3I28"]`` reaches a real, measured file with
no egress at all. A run whose target has no cached structure gets a ``partial``
result naming the id and the fetch command, which is the honest outcome — the
alternative would be detecting pockets in a structure nobody measured.

What lands on the blackboard is a :class:`~drug_agent.contracts.brief.PocketSpec`
whose ``pdb_path`` is a file that exists, and
:meth:`~drug_agent.agents.specialists.design_agent.DesignAgent.pocket_for` reads
it before it reads the brief. That is the whole point: the site this run detected
is the site this run samples in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest, AgentResult, ValidationIssue
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism
from ...tools.pockets import DEFAULT_N_PREPARED, DEFAULT_POCKET_RADIUS, PocketDetection, PocketError
from ...tools.pockets.fpocket_tool import FpocketAdapter
from ...tools.pockets.p2rank_tool import P2RankAdapter
from ...tools.pockets.vnegnn_tool import VnEgnnAdapter
from ...tools.structure import StructureError
from ...tools.structure.experimental_tool import ExperimentalStructureAdapter
from . import SpecialistAgent, unavailable_note

if TYPE_CHECKING:
    from ...contracts.brief import ResearchBrief
    from ...orchestration.context import RunContext

#: The detectors, by the name ``detector`` accepts. Full tool ids work too.
DetectorAdapter = FpocketAdapter | P2RankAdapter | VnEgnnAdapter


def select_detector(name: str, *, settings: Any = None) -> DetectorAdapter:
    """Resolve ``detector`` to the adapter this agent will ask for a site.

    ``auto`` prefers P2Rank when this host can really run it and otherwise falls
    back to fpocket — the same shape ``select_docking`` uses, and for the same
    reason: a machine without the preferred backend must still detect pockets
    rather than silently doing nothing. An unknown name raises; two detectors do
    not find the same site, and a run that used a different one than it was
    configured for is not reproducible.
    """
    wanted = (name or "auto").strip().lower()
    if wanted in {"fpocket", FpocketAdapter.capability_id}:
        return FpocketAdapter(settings=settings)
    if wanted in {"p2rank", P2RankAdapter.capability_id}:
        return P2RankAdapter(settings=settings)
    if wanted in {"vnegnn", VnEgnnAdapter.capability_id}:
        return VnEgnnAdapter(settings=settings)
    if wanted != "auto":
        raise ValueError(
            f"unknown pocket detector {name!r}; this agent drives: auto, p2rank, fpocket, vnegnn"
        )
    p2rank = P2RankAdapter(settings=settings)
    return p2rank if p2rank.available() else FpocketAdapter(settings=settings)


class PocketAgent(SpecialistAgent):
    """Find the binding site, and write the file the rest of the pipeline needs."""

    agent_id = "agent.pocket"
    version = "1.0.0"
    actions = {"detect_pocket": "Detect the binding site in a structure and prepare its pocket PDB."}

    def __init__(self, *, settings: Any = None, detector: str = "auto") -> None:
        self.settings = settings
        self.detector = select_detector(detector, settings=settings)
        # The cache is how a PDB id becomes coordinates without a network call.
        # It is a separate capability with its own card, so its absence costs the
        # id-resolution path and nothing else: a caller that passes a file still
        # gets a pocket.
        self.structures = ExperimentalStructureAdapter(settings=settings)

    def describe_capability(self) -> CapabilityProfile:
        available = bool(self.detector.available())
        return CapabilityProfile(
            capability_id=self.agent_id,
            version=self.version,
            title="Binding-site detection",
            description="Detect the binding site in a receptor and prepare the pocket PDB it implies.",
            input_schema_ref="ResearchBrief",
            output_schema_ref="PocketSpec",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            status=CapabilityStatus.AVAILABLE if available else CapabilityStatus.UNAVAILABLE,
            backend=f"detect={self.detector.backend},structures={self.structures.backend}",
            unavailable_reason=None if available else self.detector.profile.unavailable_reason,
            known_failure_modes=[
                "no structure for this target in the local cache",
                "the detector found no cavity in the structure it was given",
            ],
            validation_checks=[
                "the prepared pocket file exists and carries atoms",
                "the spec's centre is the detector's own, not a default",
            ],
        )

    # --- the structure ----------------------------------------------------
    def _structure_path(self, request: AgentRequest, brief: ResearchBrief | None) -> tuple[Path | None, str]:
        """The receptor coordinates to search, and — when there are none — why.

        A caller's explicit path wins. Otherwise the brief's PDB ids are resolved
        against the local cache in the order the brief lists them, and finally the
        target name/gene is looked up. ``brief.generation.pocket.pdb_path`` is
        deliberately *not* consulted: that is a prepared pocket, the output of
        this agent's job, not an input to it.
        """
        raw = request.inputs.get("structure_pdb") or request.inputs.get("receptor_pdb")
        if raw:
            path = Path(str(raw)).expanduser().resolve()
            return (path, "") if path.is_file() else (None, f"structure file {path} does not exist")
        if brief is None or brief.target is None:
            return None, "the request names no structure and the brief has no target"
        if not self.structures.available():
            return None, str(self.structures.profile.unavailable_reason or "no local structure cache")
        for pdb_id in brief.target.pdb_ids:
            try:
                return Path(self.structures.resolve(pdb_id).path).resolve(), ""
            except StructureError:
                continue
        for name in (brief.target.gene_symbol, brief.target.uniprot, brief.target.name):
            if not name:
                continue
            try:
                return Path(self.structures.best_for_target(name).path).resolve(), ""
            except StructureError:
                continue
        listed = ", ".join(brief.target.pdb_ids) or brief.target.name
        return None, (
            f"no cached experimental structure for {listed}; fetch one with "
            f"`python scripts/fetch_experimental_structures.py "
            f"{(brief.target.pdb_ids or ['<PDB_ID>'])[0]}`"
        )

    # --- execution --------------------------------------------------------
    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        if request.action != "detect_pocket":
            return self._result(
                request,
                status="error",
                validation_issues=[ValidationIssue(code="unknown_action", message=request.action)],
            )
        brief = self._brief(request)
        if not self.detector.available():
            note = unavailable_note(self.detector)
            return self._result(request, status="partial", warnings=[note], explanation=note)

        structure, blocker = self._structure_path(request, brief)
        if structure is None:
            note = f"no pocket detected: {blocker}"
            return self._result(request, status="partial", warnings=[note], explanation=note)

        out_dir = Path(ctx.settings.runs_dir or ".") / ctx.run.id / "pockets"
        out_dir.mkdir(parents=True, exist_ok=True)
        radius = float(request.inputs.get("pocket_radius") or DEFAULT_POCKET_RADIUS)
        n_prepared = int(request.inputs.get("n_prepared") or DEFAULT_N_PREPARED)
        try:
            detection = self.detector.detect(structure, out_dir=out_dir, n_prepared=n_prepared, radius=radius)
        except PocketError as exc:
            note = f"no pocket detected: {self.detector.backend} failed on {structure.name}: {exc}"
            return self._result(request, status="partial", warnings=[note], explanation=note)

        return self._record(request, ctx, brief, structure, detection, radius)

    def _record(
        self,
        request: AgentRequest,
        ctx: RunContext,
        brief: ResearchBrief | None,
        structure: Path,
        detection: PocketDetection,
        radius: float,
    ) -> AgentResult:
        """Persist every site, and promote the best one to the run's PocketSpec."""
        receptor_id, pdb_id = self._identity(request, brief, structure)
        for pocket in detection.pockets:
            ctx.store.put_object(
                "pocket",
                {
                    "id": f"pocket_{detection.engine}_{pocket.rank:02d}",
                    "run_id": ctx.run.id,
                    "receptor_id": receptor_id,
                    "pdb_id": pdb_id,
                    "structure_path": str(structure),
                    "engine": pocket.engine,
                    "engine_version": pocket.engine_version,
                    "snapshot_id": detection.snapshot_id,
                    "rank": pocket.rank,
                    "label": pocket.label,
                    "score": pocket.score,
                    "score_kind": pocket.score_kind,
                    "probability": pocket.probability,
                    "druggability": pocket.druggability,
                    "volume_a3": pocket.volume_a3,
                    "center": list(pocket.center),
                    "radius_a": pocket.radius_a,
                    "n_atoms": pocket.n_atoms,
                    "n_residues": pocket.n_residues,
                    "residues": list(pocket.residues),
                    "pdb_path": pocket.pdb_path,
                    "properties": dict(pocket.properties),
                    "warnings": list(pocket.warnings),
                },
                run_id=ctx.run.id,
            )

        best = detection.top
        if best is None or not best.pdb_path:
            note = (
                f"no pocket detected: {detection.engine} returned "
                f"{len(detection.pockets)} site(s) and prepared no file"
            )
            return self._result(
                request,
                status="partial",
                metrics=detection.metrics(),
                warnings=[note],
                explanation=note,
            )

        spec = best.to_pocket_spec(receptor_id=receptor_id, pdb_id=pdb_id, radius=radius)
        # The blackboard is where the design agent looks first; the object store is
        # where a resumed process, the report and the bundle look. Both, because
        # the blackboard is explicitly not authoritative.
        ctx.blackboard["pocket"] = spec
        ctx.store.put_object(
            "pocket_spec",
            {"id": f"pocketspec_{ctx.run.id}", **spec.model_dump(mode="json")},
            run_id=ctx.run.id,
        )
        # Immutable copy, so the file the generators were pointed at can still be
        # produced after the run directory is gone.
        ref = ctx.artifacts.put_text(
            Path(best.pdb_path).read_text(),
            kind="pdb",
            media_type="chemical/x-pdb",
            produced_by=self.agent_id,
            run_id=ctx.run.id,
            metadata={"engine": best.engine, "rank": best.rank, "center": list(best.center)},
        )
        ctx.emit_event(
            "pocket.detected",
            engine=detection.engine,
            n_pockets=len(detection.pockets),
            pdb_path=best.pdb_path,
            artifact=getattr(ref, "content_hash", None),
        )
        centre = ", ".join(f"{c:.2f}" for c in best.center)
        return self._result(
            request,
            payload={
                "n_pockets": len(detection.pockets),
                "pocket": spec.model_dump(mode="json"),
                "engine": detection.engine,
            },
            metrics=detection.metrics(),
            warnings=list(detection.warnings),
            explanation=(
                f"{detection.engine} {detection.engine_version} found {len(detection.pockets)} site(s) in "
                f"{structure.name}; prepared the top one ({best.n_atoms} atoms, {best.n_residues} residues) "
                f"at [{centre}] as {best.pdb_path}."
            ),
        )

    @staticmethod
    def _identity(
        request: AgentRequest, brief: ResearchBrief | None, structure: Path
    ) -> tuple[str | None, str | None]:
        """What to call this receptor, from the brief when it says and the file otherwise."""
        receptor_id = request.inputs.get("receptor_id")
        pdb_id = request.inputs.get("pdb_id") or structure.stem.split("_")[0].upper()
        if not receptor_id and brief is not None and brief.target is not None:
            receptor_id = brief.target.chembl_target_id or brief.target.gene_symbol
        return (str(receptor_id) if receptor_id else None), (str(pdb_id) if pdb_id else None)

    @staticmethod
    def _brief(request: AgentRequest) -> ResearchBrief | None:
        from ...contracts.brief import ResearchBrief  # local import: avoids a cycle

        raw = request.inputs.get("brief")
        return ResearchBrief.model_validate(raw) if raw else None

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        """A spec whose file is not there is worse than no spec at all.

        The generators take ``pdb_path`` on trust and hand it to a subprocess, so
        a path that does not resolve surfaces three layers down as "could not open
        file". This is read back from the store rather than from the result, so a
        bug in :meth:`execute` cannot also silence the check on it.
        """
        issues: list[ValidationIssue] = []
        for spec in ctx.store.list_objects("pocket_spec", run_id=ctx.run.id):
            path = spec.get("pdb_path")
            if not path:
                issues.append(
                    ValidationIssue(
                        code="pocket_not_prepared",
                        message=f"{spec.get('id')} carries no pocket file; no generator can use it",
                    )
                )
                continue
            file = Path(str(path))
            if not file.is_file():
                issues.append(
                    ValidationIssue(code="pocket_file_missing", message=f"pocket file {file} does not exist")
                )
            elif "ATOM" not in file.read_text():
                issues.append(
                    ValidationIssue(
                        code="pocket_file_empty", message=f"pocket file {file} has no ATOM records"
                    )
                )
            if not spec.get("center"):
                issues.append(
                    ValidationIssue(
                        code="pocket_centre_missing",
                        message=f"{spec.get('id')} names no centre, so it implies no search box",
                        # A warning, not an error. `invoke_agent` turns any
                        # error-severity issue into an AgentError, which would kill
                        # the whole run at a task that did not exist before — and a
                        # centre-less pocket is still usable by TargetDiff,
                        # Pocket2Mol and PocketFlow. The models that do need one
                        # refuse for themselves, by name, at the point of use.
                        severity="warning",
                    )
                )
        return issues
