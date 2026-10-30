"""De-novo design — the agent that samples molecules that did not exist before.

Everything else in the specialist plane starts from a molecule someone already
has: the library, the seeds, the analogues around them. This one starts from a
*target site* and proposes structures, which is a different job with a different
failure mode, and it is the one arm of the pipeline that can produce nothing at
all and still be working correctly.

The generator family is selected, not bound. ``settings.generator_backend`` names
which of the twenty-one de-novo models a run samples from and
:func:`select_generator` resolves it over
:func:`~drug_agent.tools.registry.generator_adapters`, so a model added to the
pool is selectable without touching this file. An unknown name raises rather than
falling back: a run that quietly sampled a different model than the one it was
configured for is not reproducible, and a silent substitution is exactly what
ADR-0011 forbids. A model that is not installed here reports UNAVAILABLE, this
arm produces nothing, and the failure is written down — as batch ``errors``, a
persisted ``generation_batch`` record and a warning — rather than being answered
with molecules from some other model.

**Where the pocket comes from.** The pocket-conditioned generators — PocketXMol,
TargetDiff, PocketFlow, OMTRA and the rest — refuse without a prepared pocket
PDB, and until :class:`~drug_agent.agents.specialists.pocket_agent.PocketAgent`
existed the only way a run could have one was for a human to have cut it by hand
and written the path into the brief. :meth:`DesignAgent.pocket_for` therefore
looks at the blackboard *first*: a :class:`~drug_agent.contracts.brief.PocketSpec`
the pocket agent detected in this run outranks whatever the brief was written
with, because it was computed from this run's own structure. The brief's own
pocket is the fallback, and a bare ``pdb_id`` is provenance only — it names the
structure without preparing it, which is what makes the generator's refusal
honest instead of mysterious.

This agent owns the generators; :class:`~drug_agent.agents.specialists.molecule_agent.MoleculeAgent`
owns the funnel they feed. The de-novo arm of ``source_candidates`` is executed
here — the molecule agent delegates to :meth:`generate_batch` — so the tool ids
in ``AGENT_TOOLS['agent.design']`` are ids this class really calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest, AgentResult, ValidationIssue
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ...tools.base import ToolUnavailable
from ...tools.generation import GenerationBackendError
from ...tools.pfm_tool import (
    ORIGIN_PFM,
    GenerationBatch,
    GenerationError,
    PFMGeneratorAdapter,
)
from . import SpecialistAgent, unavailable_note

if TYPE_CHECKING:
    from ...contracts.brief import ResearchBrief
    from ...orchestration.context import RunContext
    from ...tools.generation import GenerativeToolAdapter

    #: What this agent samples from: the in-tree PFM adapter or any of the
    #: catalogue generators. They share ``generate()``, ``available()`` and a
    #: capability card, which is the whole interface this agent needs.
    GeneratorAdapter = PFMGeneratorAdapter | GenerativeToolAdapter


#: Names that select the in-tree PFM adapter. ``auto`` is the shipped default, so
#: a deployment that configures nothing keeps sampling PFM.
PFM_GENERATOR_NAMES = frozenset({"", "auto", "pfm", PFMGeneratorAdapter.capability_id})


def select_generator(name: str, *, settings: Any = None) -> GeneratorAdapter:
    """Resolve a generator name to the adapter that implements it.

    ``settings.generator_backend`` names *which* de-novo model a run samples from,
    not merely which backend PFM uses — PFM has exactly one. Selection happens
    over :func:`~drug_agent.tools.registry.generator_adapters` rather than a
    second table here, so a generator added to the pool is selectable without
    touching this file.

    An unknown name raises rather than falling back, which is the refusal
    :class:`PFMGeneratorAdapter` already makes for an unknown backend.
    """
    wanted = (name or "auto").strip().lower()
    if wanted in PFM_GENERATOR_NAMES:
        # The module-level name, deliberately: the suite substitutes a hermetic
        # stub for it, and PFM stays the default so recorded runs replay.
        return PFMGeneratorAdapter(backend="auto", settings=settings)
    # Imported here because it constructs every adapter in the pool: worth doing
    # once for a run that asked for a catalogue generator, wasteful for one that
    # did not.
    from ...tools.registry import GENERATOR_ID_PREFIX, generator_adapters

    adapters = generator_adapters(settings)
    adapter = adapters.get(wanted) or adapters.get(f"{GENERATOR_ID_PREFIX}{wanted}")
    if adapter is None:
        known = ", ".join(sorted(tool_id.removeprefix(GENERATOR_ID_PREFIX) for tool_id in adapters))
        raise ValueError(f"unknown generator {name!r}; this build ships: auto, {known}")
    generator: GeneratorAdapter = adapter
    return generator


class DesignAgent(SpecialistAgent):
    """Sample de-novo molecules for the run's target site."""

    agent_id = "agent.design"
    version = "1.0.0"
    actions = {
        "generate_candidates": "Sample de-novo molecules from the selected generator, in the run's pocket."
    }

    def __init__(self, *, settings: Any = None, generator_backend: str = "auto") -> None:
        self.settings = settings
        self.generator = select_generator(generator_backend, settings=settings)

    @property
    def generator_origin(self) -> str:
        """The ``origin`` tag this run's generator writes onto every molecule.

        The catalogue generators tag with their own engine name; PFM predates them
        and tags with :data:`ORIGIN_PFM`. Reading it off the adapter is what lets
        a batch be counted against the arm that actually ran.
        """
        return str(getattr(self.generator, "engine", "") or ORIGIN_PFM)

    def describe_capability(self) -> CapabilityProfile:
        available = bool(self.generator.available())
        return CapabilityProfile(
            capability_id=self.agent_id,
            version=self.version,
            title="De-novo molecular design",
            description="Sample new structures for a binding site with the selected generative model.",
            input_schema_ref="ResearchBrief",
            output_schema_ref="GenerationBatch",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.EXPENSIVE,
            status=CapabilityStatus.AVAILABLE if available else CapabilityStatus.UNAVAILABLE,
            backend=self.generator.backend,
            unavailable_reason=None if available else self.generator.profile.unavailable_reason,
            known_failure_modes=[
                "the selected model is not installed on this host (produces nothing)",
                "no prepared pocket, so a pocket-conditioned model refuses",
                "sampling produced no valid, unique molecule",
            ],
            validation_checks=[
                "every molecule carries the generator's own origin tag",
                "no molecule is emitted while the generator is unavailable",
            ],
        )

    # --- the pocket -------------------------------------------------------
    def pocket_for(self, brief: ResearchBrief, ctx: RunContext | None) -> dict[str, Any]:
        """The binding site this batch samples in, as the generators' own dict.

        Order matters and is the point of the pocket agent existing. A
        :class:`PocketSpec` on the blackboard was *detected in this run* from this
        run's structure, so it outranks the one the brief was written with. The
        brief's pocket comes next. A bare PDB id is provenance only: it names the
        structure and prepares nothing, so a pocket-conditioned model still
        refuses — which is the honest outcome and the one this pipeline had until
        pockets could be detected.
        """
        detected = getattr(ctx, "blackboard", {}).get("pocket") if ctx is not None else None
        if detected is not None:
            return dict(detected) if isinstance(detected, dict) else detected.model_dump(mode="json")
        spec = brief.generation.pocket if brief.generation else None
        if spec is not None:
            return spec.model_dump(mode="json")
        if brief.target and brief.target.pdb_ids:
            return {"pdb_id": brief.target.pdb_ids[0]}
        return {}

    # --- the batch --------------------------------------------------------
    def generate_batch(
        self, brief: ResearchBrief, context_smiles: list[str], ctx: RunContext
    ) -> GenerationBatch:
        """Sample one batch and persist an auditable record of it.

        A generator failure degrades the arm rather than killing the run: the
        batch's ``errors`` are returned to the caller and the whole configuration
        is persisted either way, so a run whose generated arm collapsed is still
        explainable after the fact.
        """
        spec = brief.generation
        pocket = self.pocket_for(brief, ctx)
        seed = (spec.seed if spec and spec.seed is not None else None) or ctx.run.seed

        try:
            batch = self.generator.generate(
                n_molecules=spec.n_molecules if spec else 50,
                pocket=pocket,
                context_smiles=context_smiles,
                seed=int(seed),
                batch_size=spec.batch_size if spec else 10,
                min_heavy_atoms=spec.min_heavy_atoms if spec else 10,
                max_heavy_atoms=spec.max_heavy_atoms if spec else 70,
                keep_invalid=spec.keep_invalid if spec else False,
                max_runtime_seconds=spec.max_runtime_seconds if spec else None,
            )
        except (GenerationError, GenerationBackendError, ToolUnavailable, ValueError, OSError) as exc:
            # Several ways one call can fail, one outcome. PFM raises its own
            # GenerationError; the catalogue generators raise ToolUnavailable when
            # the model is not installed here and GenerationBackendError when it is
            # and sampling failed. None of them may be answered with molecules from
            # some other model, so all of them degrade the arm identically.
            #
            # ValueError and OSError are here because this method's own docstring
            # promises "a generator failure degrades the arm rather than killing
            # the run", and for one class of failure that was false: a generator
            # refusing a missing contract input (PocketXMol without a pocket
            # centre) raised a bare ValueError that walked past this clause and
            # killed the run at `source_candidates`. The adapters now raise typed
            # errors, and this catch is the belt to that braces — twenty-one
            # adapters wrap twenty-one upstream codebases, and one of them reading
            # a bad path should cost the arm, not the program.
            batch = GenerationBatch(
                backend=self.generator.backend,
                model_id=self.generator.capability_id,
                n_requested=spec.n_molecules if spec else 0,
                errors=[str(exc)],
                warnings=[f"de-novo generation produced no candidates: {exc}"],
                config={"backend": self.generator.backend, "pocket": pocket},
            )

        ctx.store.put_object(
            "generation_batch",
            {
                "id": f"gen_{ctx.run.id}_{batch.backend}",
                "run_id": ctx.run.id,
                "method": str(spec.method) if spec else "pfm",
                # Which model ran. ``method`` is the brief's on/off selector and
                # only knows 'pfm'; this is the id of the adapter that sampled.
                "generator": self.generator.capability_id,
                "backend": batch.backend,
                "model_id": batch.model_id,
                "model_version": batch.model_version,
                "is_surrogate": False,  # no surrogate backend exists any more
                "origin_tag": self.generator_origin,
                "trust_label": str(TrustLabel.PROPOSED),
                "seed": int(seed),
                "pocket": pocket,
                "config": batch.config,
                "metrics": batch.metrics(),
                "warnings": batch.warnings,
                "errors": batch.errors,
                "n_kept": len(batch.molecules),
            },
            run_id=ctx.run.id,
        )
        ctx.emit_event(
            "generation.completed",
            backend=batch.backend,
            n_kept=len(batch.molecules),
            validity=batch.validity,
            uniqueness=batch.uniqueness,
        )
        return batch

    # --- as a step of its own ---------------------------------------------
    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        """Sample a batch and put it into the funnel as raw candidates.

        The de-novo arm of the default graph runs through the molecule agent,
        which delegates to :meth:`generate_batch`; this action is the same batch
        as a step of its own, for a caller that wants design separated from
        sourcing. The ids are prefixed ``raw_design_`` so the two arms cannot
        collide in the store, and ``standardize_candidates`` picks these up like
        any other raw candidate.
        """
        from ...contracts.brief import ResearchBrief  # local import: avoids a cycle

        if request.action != "generate_candidates":
            return self._result(
                request,
                status="error",
                validation_issues=[ValidationIssue(code="unknown_action", message=request.action)],
            )
        brief = ResearchBrief.model_validate(request.inputs["brief"])
        context = [r["smiles"] for r in ctx.store.list_objects("compound", run_id=ctx.run.id)]
        context.extend(brief.seed_smiles)

        batch = self.generate_batch(brief, context, ctx)
        seen: set[str] = set()
        stored = 0
        for molecule in batch.molecules:
            if molecule.smiles in seen:
                continue
            seen.add(molecule.smiles)
            ctx.store.put_object(
                "raw_candidate",
                {
                    "id": f"raw_design_{stored:05d}",
                    "smiles": molecule.smiles,
                    "origin": molecule.origin,
                    "lineage": [],
                },
                run_id=ctx.run.id,
            )
            stored += 1

        issues: list[ValidationIssue] = []
        if batch.errors:
            issues.append(
                ValidationIssue(
                    code="generation_failed", message="; ".join(batch.errors)[:500], severity="warning"
                )
            )
        warnings = list(batch.warnings)
        if not self.generator.available():
            warnings.append(unavailable_note(self.generator))
        return self._result(
            request,
            status="partial" if stored == 0 else "ok",
            payload={"n_generated": stored, "generator": self.generator.capability_id},
            metrics={"n_generated": float(stored), **batch.metrics()},
            warnings=warnings,
            validation_issues=issues,
            explanation=(
                f"Sampled {stored} de-novo molecules with {self.generator.capability_id} "
                f"({self.generator.backend})."
            ),
        )

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        """ADR-0011, enforced rather than merely performed.

        A generator that cannot run here must have produced *nothing*. This reads
        the store rather than the result, so a bug in :meth:`execute` cannot also
        silence the check on :meth:`execute`.
        """
        if self.generator.available():
            return []
        emitted = [
            r
            for r in ctx.store.list_objects("raw_candidate", run_id=ctx.run.id)
            if str(r.get("id", "")).startswith("raw_design_")
        ]
        if not emitted:
            return []
        return [
            ValidationIssue(
                code="unavailable_capability_emitted_output",
                severity="error",
                message=(
                    f"{self.generator.capability_id} is UNAVAILABLE here yet {len(emitted)} "
                    f"molecules were attributed to it"
                ),
            )
        ]
