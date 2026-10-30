"""Bridge from the LLM's ``ChemicalRequirements`` to the deterministic platform.

This is the ONLY place scientific work happens: RDKit validates the LLM-proposed
seeds, and the full multi-agent Engine (literature, hypotheses, generation,
screening, synthesis, safety, perturbation, ranking, report) runs per cycle. The
LLM never computes any of these numbers.

Candidate structures come from the library + seed analogues, from the **PFM**
de-novo generative model, or from both — see ``requirements.generator``. PFM
molecules are proposals: they are screened, filtered and ranked by exactly the
same pipeline as library compounds, and the master agent never sees a number the
model produced about them.

Everything here is synchronous/CPU-bound; the master agent calls it through
``asyncio.to_thread`` so the event stream stays live.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config
from .schemas import ChemicalRequirements

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..observability.journal import RunJournal


def _prepare_environment(cfg: Config) -> None:
    """Point the platform at an offline state home and the fixture directory.

    The scientific fixtures resolve relative to the repo root, so the working
    directory has to be there before the molecule agent reads them.
    """
    os.environ.setdefault("DRUGAGENT_PROFILE", cfg.drug_agent_profile)
    os.environ.setdefault("DRUGAGENT_ALLOW_NETWORK", "0")
    os.environ.setdefault("DRUGAGENT_HOME", str(Path.home() / ".llm-agent" / "drugagent"))
    with contextlib.suppress(OSError):
        os.chdir(cfg.resolved_repo_root())


@lru_cache(maxsize=1)
def _rdkit():
    from rdkit import (
        Chem,  # type: ignore
        RDLogger,  # type: ignore
    )

    RDLogger.DisableLog("rdApp.*")
    return Chem


def validate_seed_smiles(smiles: list[str], cfg: Config) -> tuple[list[str], list[str]]:
    """Return (valid_canonical, dropped). Invalid SMILES never reach the backend."""
    _prepare_environment(cfg)
    Chem = _rdkit()
    valid: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for smi in smiles:
        smi = (smi or "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            dropped.append(smi)
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            valid.append(canon)
    return valid, dropped


#: Every protein target this deployment holds bioactivity data for.
#:
#: The activity model is trained per run from the bioactivity connector's assays.
#: With no matching target id the connector returns zero assays, the model cannot
#: be trained, and NO pIC50 is produced for any candidate — which makes
#: ``max_pIC50`` / ``median_pIC50`` unscorable and makes a brief that marks
#: ``pIC50`` *required* fail outright at the ranking step.
#:
#: One table, two readers: :func:`_infer_chembl_target_id` matches against it and
#: :func:`bioactivity_coverage` states it to the master agent's planner, so what
#: the planner is told and what the pipeline can do cannot drift apart.
BIOACTIVITY_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "chembl_target_id": "TARGET_EPHX2",
        "label": "soluble epoxide hydrolase (sEH / EPHX2, UniProt P34913)",
        "tokens": ("ephx2", "seh", "p34913"),
        "phrases": ("epoxide hydrolase",),
    },
)


def bioactivity_coverage() -> list[str]:
    """Human-readable labels for the targets that have bioactivity data here."""
    return [str(t["label"]) for t in BIOACTIVITY_TARGETS]


def _infer_chembl_target_id(req: ChemicalRequirements) -> str | None:
    """Map the LLM's target description to a target id the connector recognizes.

    For targets with no fixture we leave it unset, and pIC50 is then honestly
    unavailable rather than silently wrong. See :data:`BIOACTIVITY_TARGETS`.
    """
    hay = " ".join(filter(None, [req.target_name, req.target_gene, req.target_uniprot])).lower()
    words = set(hay.replace("/", " ").replace("(", " ").replace(")", " ").split())
    for entry in BIOACTIVITY_TARGETS:
        if words & set(entry["tokens"]) or any(ph in hay for ph in entry["phrases"]):
            return str(entry["chembl_target_id"])
    return None


#: Above this many heavy atoms a file is a whole structure, not a pocket cut, and
#: its centroid is a point in the middle of the protein rather than a site. Sampling
#: there would silently produce molecules for the wrong box, which is strictly worse
#: than refusing: a run that crashes is debuggable, a run that quietly samples the
#: wrong cavity is not. `ResearchBrief.PocketSpec` documents the same expectation.
_MAX_POCKET_HEAVY_ATOMS = 1500


def _measured_centre(pdb_path: str) -> dict[str, Any]:
    """The heavy-atom centroid of a prepared pocket file, as PocketSpec fields.

    PocketXMol's de-novo tasks (``sbdd``, ``sbdd_simple``) refuse without
    ``PocketSpec.center``, and until now nothing in the LLM path ever set it: the
    brief carried ``pdb_path`` alone, so every pocket-conditioned backend except
    PFM failed on a contract field that was three lines of arithmetic away from
    the file already named. This is a MEASUREMENT of the given file, not a
    prediction — the note says so, so a report can never present it as a detected
    site. A detector's own centre still outranks it: the pocket agent writes one
    to the blackboard and ``DesignAgent.pocket_for`` reads that first.

    Degrades to no centre rather than raising. A missing centre is a refusal the
    generator makes clearly; a crash here would be one nobody asked for.
    """
    try:
        from ..tools.pockets._pdb import centroid, parse_atoms

        atoms = [a for a in parse_atoms(Path(pdb_path).read_text()) if not a.is_hydrogen]
    except Exception:
        return {}
    if not atoms:
        return {}
    if len(atoms) > _MAX_POCKET_HEAVY_ATOMS:
        return {
            "notes": (
                f"{len(atoms)} heavy atoms: this is a whole structure, not a pocket cut, so no "
                "centre was derived from it. Tasks that denoise inside a box will refuse."
            )
        }
    return {
        "center": [round(c, 3) for c in centroid([a.xyz for a in atoms])],
        "notes": (
            "centre measured as the heavy-atom centroid of the configured pocket file; "
            "it is not a detector prediction"
        ),
    }


def _generation_block(req: ChemicalRequirements, cfg: Config) -> tuple[str, dict[str, Any] | None]:
    """Map ``requirements.generator`` onto the brief's candidate_source + generation.

    ``generator="library"`` reproduces the original behaviour byte-for-byte (no
    generation block at all), so enabling PFM is always an explicit, visible act.
    The pocket comes from the operator's configuration, not from the LLM: a
    binding site is a deployment input, and inventing one would be exactly the
    kind of fabricated scientific input this system forbids.
    """
    generator = (req.generator or "library").lower()
    if "pfm" not in generator:
        return "fixture_library", None

    candidate_source = "pfm" if generator == "pfm" else "fixture_library+pfm"
    generation: dict[str, Any] = {
        "method": "pfm",
        "n_molecules": int(req.n_generated),
        "batch_size": min(20, max(1, int(req.n_generated))),
    }
    pocket: dict[str, Any] = {}
    if cfg.pfm_pocket_pdb:
        pocket["pdb_path"] = cfg.pfm_pocket_pdb
        pocket.update(_measured_centre(cfg.pfm_pocket_pdb))
    if req.target_gene or req.target_name:
        pocket["receptor_id"] = req.target_gene or req.target_name
    if pocket:
        generation["pocket"] = pocket
    # How the binding site is obtained. "detect" schedules the pocket agent,
    # which is the only producer of a centre a detector actually predicted;
    # "brief" uses the configured file as-is. See `GenerationSpec.pocket_source`.
    generation["pocket_source"] = req.resolved_pocket_source()
    generation["pocket_detector"] = req.pocket_detector
    return candidate_source, generation


def requirements_to_brief_dict(
    req: ChemicalRequirements, cfg: Config
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Map ChemicalRequirements -> a ResearchBrief dict. Also returns
    (valid_seeds, dropped_seeds) so the UI can report what RDKit rejected."""
    valid_seeds, dropped = validate_seed_smiles(req.seed_smiles, cfg)

    # Objectives: dedupe by name, ensure >= 1.
    objectives: list[dict[str, Any]] = []
    seen_obj: set[str] = set()
    for obj in req.objectives:
        name = (obj.name or "").strip()
        if not name or name in seen_obj:
            continue
        seen_obj.add(name)
        entry: dict[str, Any] = {"name": name, "direction": obj.direction, "weight": float(obj.weight)}
        if obj.target_value is not None:
            entry["target_value"] = float(obj.target_value)
        entry["required"] = bool(obj.required)
        objectives.append(entry)
    if not objectives:
        objectives = [{"name": "pIC50", "direction": "maximize", "weight": 1.0, "required": True}]

    # Hard constraints.
    constraints: list[dict[str, Any]] = []
    for i, pc in enumerate(req.property_constraints):
        prop = (pc.property or "").strip()
        if not prop:
            continue
        constraints.append(
            {
                "name": f"c_{i}_{prop.replace('.', '_')}",
                "property": prop,
                "op": pc.op,
                "value": pc.value,
                "rationale": pc.rationale or None,
            }
        )

    candidate_source, generation = _generation_block(req, cfg)
    brief: dict[str, Any] = {
        "kind": req.kind,
        "title": req.title or "LLM-directed discovery program",
        "objectives": objectives,
        "hard_constraints": constraints,
        "seed_smiles": valid_seeds,
        "candidate_source": candidate_source,
        "max_candidates": int(req.max_candidates),
        "shortlist_size": int(req.shortlist_size),
        "approval_gates": ["portfolio_review"],
        # The target dossier is a graph decision, so it rides the brief; the three
        # assessment arms are agent-construction decisions, so they ride the
        # settings overrides in `_execute`. Same origin, different destination.
        "assess_target_evidence": bool(req.target_evidence),
        "tags": [
            "llm-agent",
            f"novelty:{req.novelty}",
            f"generator:{req.generator}",
            # The ARM token above is "pfm" for every de-novo model, so a recorded
            # brief could never say which one sampled it. This is the engine.
            f"engine:{req.generator_backend}",
        ],
    }
    if generation is not None:
        brief["generation"] = generation
    if req.kind == "phenotype_based":
        brief["phenotype"] = {
            "description": req.phenotype_description or req.primary_objective or "phenotype rescue",
            "cell_context": req.cell_context,
        }
    # The target block is ADDITIVE, not the other half of an if/else. A phenotype
    # program that can still name a protein — "reverse this cell state, the
    # handle is GLP1R" — used to lose that name entirely, and with it the only
    # input the pocket agent resolves a structure from. Structure-based de-novo
    # design was therefore impossible for every phenotype brief, which is exactly
    # the shape the bioactivity block steers uncovered targets into.
    if req.kind != "phenotype_based" or req.target_gene or req.target_uniprot or req.pdb_ids:
        brief["target"] = {
            "name": req.target_name or req.title or "unspecified target",
            "gene_symbol": req.target_gene,
            "uniprot": req.target_uniprot,
            "chembl_target_id": _infer_chembl_target_id(req),
            # What the pocket agent resolves against the local RCSB cache.
            "pdb_ids": list(req.pdb_ids),
        }
    if req.kind != "phenotype_based":
        brief["dataset_snapshot"] = "seh_demo_v1"
    return brief, valid_seeds, dropped


@dataclass
class BackendRun:
    run_id: str
    state: str
    brief: dict[str, Any]
    portfolio: dict[str, Any] | None
    scorecards: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    safety: list[dict[str, Any]] = field(default_factory=list)
    generation: list[dict[str, Any]] = field(default_factory=list)
    #: What the pipeline's agents said produced nothing. Every optional arm the
    #: master agent can switch on reports its own failure this way, and until
    #: these crossed the bridge the model saw the symptom (an unmeasured
    #: criterion) with the cause (a broken route engine) nowhere in view.
    agent_warnings: list[dict[str, Any]] = field(default_factory=list)
    valid_seeds: list[str] = field(default_factory=list)
    dropped_seeds: list[str] = field(default_factory=list)
    error: str | None = None


def _execute(
    brief: Any,
    cfg: Config,
    home: str,
    generator_backend: str = "auto",
    *,
    property_backends: list[str] | None = None,
    route_backend: str = "none",
    complex_backend: str = "none",
) -> tuple[Any, Any]:
    """Create and drive one engine run to completion inside ``home``.

    ``generator_backend`` names WHICH de-novo model the design agent samples from
    when the brief's ``candidate_source`` includes the de-novo arm. It is threaded
    through the settings override the engine already reads
    (``settings.generator_backend`` — see ``agents/__init__.py``) rather than a
    second selection path, so the console can reach the same generator catalogue
    the CLI can. ``auto`` is the shipped default and resolves to PFM, which is
    why a run that names nothing behaves exactly as it did before.

    ``property_backends``, ``route_backend`` and ``complex_backend`` reach the
    specialist agents the same way and for the same reason: ``build_registry``
    constructs every agent out of ``settings``, so a per-run switch on an agent's
    optional arm has to be a setting. Each defaults to off, so a caller that names
    nothing gets the pipeline it always got.
    """
    from ..service import build_engine
    from ..settings.config import load_settings

    settings = load_settings(
        profile=cfg.drug_agent_profile,
        deterministic=cfg.deterministic,
        overrides={
            "home": home,
            "runs_dir": str(Path(home) / "runs"),
            "artifact_dir": str(Path(home) / "artifacts"),
            "db_url": f"sqlite:///{Path(home) / 'state.sqlite'}",
            "generator_backend": generator_backend or "auto",
            "property_backends": list(property_backends or []),
            "route_backend": route_backend or "none",
            "complex_backend": complex_backend or "none",
        },
    )
    engine = build_engine(settings)
    # This pipeline run is one cycle of a request the master agent is already
    # journaling; a second, standalone history entry would double-count it and
    # put the evolver's own pipeline runs into the corpus it analyzes.
    engine.record_history = False
    run = engine.create_run(brief, deterministic=cfg.deterministic, profile=cfg.drug_agent_profile)
    run = engine.run(run.id)

    # Auto-approve any human gate so the pipeline completes; the *master
    # agent's* ACCEPT is the real decision surfaced to the user.
    for _ in range(5):
        if str(run.state) != "AWAITING_APPROVAL":
            break
        approved_any = False
        for gate in brief.approval_gates or ["portfolio_review"]:
            try:
                engine.approve_gate(
                    run.id, gate, by="master_agent", reason="auto-approved by LLM master agent"
                )
                approved_any = True
            except Exception:
                continue
        run = engine.resume(run.id)
        if not approved_any:
            break
    return engine, run


def run_pipeline(
    req: ChemicalRequirements,
    cfg: Config,
    *,
    journal: RunJournal | None = None,
    cycle: int = 0,
) -> BackendRun:
    """Run the full drug_agent pipeline for one set of requirements (blocking).

    Each call runs in a FRESH, isolated state home (temp dir) so runs are
    independent and reproducible: the candidate library is loaded clean from
    fixtures every time, never polluted by prior runs. This is what makes the
    deterministic guarantee — and therefore the naive-retry ablation baseline —
    actually hold.

    When a ``journal`` is supplied, every instrumented tool call made anywhere
    inside this pipeline is metered and recorded under this cycle's scope, along
    with a one-line record of the run itself.
    """
    _prepare_environment(cfg)
    from ..contracts.brief import ResearchBrief
    from ..observability.toolmeter import ToolLedger, metering

    brief_dict, valid_seeds, dropped = requirements_to_brief_dict(req, cfg)
    brief = ResearchBrief.model_validate(brief_dict)

    ledger = ToolLedger(label=f"cycle-{cycle}")
    home = tempfile.mkdtemp(prefix="llmagent_run_")
    try:
        with metering(ledger):
            engine, run = _execute(
                brief,
                cfg,
                home,
                generator_backend=req.generator_backend,
                property_backends=list(req.property_backends),
                route_backend=req.route_backend,
                complex_backend=req.complex_backend,
            )
            store = engine.store
            portfolios = store.list_objects("portfolio", run_id=run.id)
            result = BackendRun(
                run_id=run.id,
                state=str(run.state),
                brief=brief_dict,
                portfolio=portfolios[0] if portfolios else None,
                scorecards=store.list_objects("scorecard", run_id=run.id),
                candidates=store.list_objects("candidate", run_id=run.id),
                safety=store.list_objects("safety", run_id=run.id),
                generation=store.list_objects("generation_batch", run_id=run.id),
                agent_warnings=store.list_objects("agent_warning", run_id=run.id),
                valid_seeds=valid_seeds,
                dropped_seeds=dropped,
                # WHY the run ended where it did. The engine writes the failing
                # task and its reason onto `run.error` in `_finalize`; dropping it
                # here left the master agent with a FAILED state and no cause, so
                # every refinement cycle was a guess. Measured: a run whose
                # ranking step refused ("no predictions exist for required
                # objective pIC50") spent all three cycles rewriting seed SMILES.
                error=getattr(run, "error", None),
            )
    finally:
        shutil.rmtree(home, ignore_errors=True)

    if journal is not None:
        journal.record_tools(ledger, scope=f"cycle-{cycle}", cycle=cycle)
        journal.record_pipeline(
            cycle=cycle,
            run_id=result.run_id,
            state=result.state,
            candidate_source=brief_dict.get("candidate_source"),
            n_candidates=len(result.candidates),
            shortlist_size=len((result.portfolio or {}).get("shortlist", [])),
            seeds_used=len(valid_seeds),
            seeds_dropped=len(dropped),
            tool_totals=ledger.totals(),
            error=result.error,
        )
    return result
