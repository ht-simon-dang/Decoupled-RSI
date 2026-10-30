"""The master agent — dynamic, self-correcting workflow.

Flow (all narrated as live events):

    PLAN ─▶ TRANSCRIBE ─▶ ┌───────────── cycle ─────────────┐
                          │  dispatch → CHEMICAL agent       │
                          │  dispatch → EVALUATION agent     │
                          │  CRITIQUE (LLM judges vs plan)   │
                          │  DECISION: ACCEPT / REFINE / …   │
                          │  if REFINE: adjust requirements  │◀─┐
                          └──────────────────────────────────┘  │
                                       └──── loop until ACCEPT ──┘ (or budget)

Ablation switch (``strategy``):
  * ``adaptive`` — on failure the LLM diagnoses the root cause and rewrites the
    generation requirements (new seeds / relaxed constraints / reweighted
    objectives) before the next cycle.
  * ``naive`` — re-run the identical requirements. Under the deterministic
    backend this reproduces identical results, which is exactly the point: blind
    retry cannot improve, adaptive re-prompting can.

A :class:`~drug_agent.llm.policy.RuntimePolicy` is resolved once per program and
recorded in the journal. Its defaults reproduce the behaviour this module had
before the policy existed, so an unconfigured deployment is unchanged; the
self-evolution subsystem uses it to run a candidate arm against a baseline arm
in the same process without either leaking into the other.

Ceiling mode (``off`` / ``hard`` / ``dynamic``, see :mod:`drug_agent.llm.config`)
is the other switch, and it is a *run* parameter rather than a policy field
because it is an operator's statement about one launch. ``off`` is the default
and reproduces the recorded history exactly. ``hard`` splices the operator's
stated ceilings into planning and transcription as binding limits. ``dynamic``
shows the planner the measured envelope as advice and adds one step after the
loop: a maximum-reasoning pass that picks the best candidate across *all* cycles.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ...llm import config as cfgmod
from ...llm import events as ev
from ...llm import prompts
from ...llm.config import (
    CEILING_MODE_DYNAMIC,
    CEILING_MODE_HARD,
    CEILING_MODE_OFF,
    CeilingError,
    Config,
)
from ...llm.harness import ensure_model_available, run_structured
from ...llm.ollama import OllamaClient, OllamaError
from ...llm.policy import RuntimePolicy, active_policy
from ...llm.repair import merge_adjustment, repair_criteria_for_goal
from ...llm.schemas import (
    METRIC_KEYS,
    BestCandidateSelection,
    ChemicalRequirements,
    Critique,
    RequirementsAdjustment,
    SuccessCriterion,
    TaskPlan,
    candidate_source_label,
)
from ...observability.journal import RunJournal
from . import ReasoningAgent
from .chemical_agent import ChemicalAgent
from .evaluation_agent import EvaluationAgent, EvaluationReport

_DEFAULT_CRITERIA = [
    SuccessCriterion(metric="max_pIC50", target=">= 6.5", rationale="a credible potent hit"),
    SuccessCriterion(
        metric="hard_constraint_pass_rate", target=">= 0.5", rationale="most candidates drug-like"
    ),
    SuccessCriterion(metric="shortlist_filled", target=">= 1.0", rationale="a full shortlist"),
    SuccessCriterion(metric="mean_pairwise_distance", target=">= 0.3", rationale="chemical diversity"),
]


def _envelope_block(policy: RuntimePolicy, cfg: Config, *, force: bool = False) -> str:
    """The measured capability envelope, as a prompt block — or "" if off/broken.

    Measuring costs a few seconds of real pipeline and is cached per deployment
    fingerprint. It must never be able to stop a program: a planner without the
    envelope is the pre-existing behaviour, and that is a strictly better
    failure than no plan at all.

    ``force`` is what ``dynamic`` ceiling mode passes. The two sealed policy
    flags gate the envelope as an *experiment* the self-evolution arm is running;
    a run launched in ``dynamic`` mode has asked for the same measurement as an
    operator decision, and the measurement should not depend on whether an
    unrelated overlay happens to be loaded.
    """
    if not force and not policy.capability_envelope and not policy.capability_envelope_transcribe:
        return ""
    try:
        from ...tools.envelope_tool import measure_envelope

        return measure_envelope(cfg).as_prompt_block()
    except Exception:
        return ""


def _reachable_tool_cards() -> list[Any]:
    """Capability cards for every tool some agent can actually reach.

    NOT cached: each caller caches its own finished block instead. One shared
    cache here would mean a caller clearing its own cache still saw a stale pool
    through this one, and the pool is exactly what a test swaps to describe a
    different deployment.

    Raises nothing — callers degrade to "" rather than failing, because a prompt
    without these facts is the pre-existing behaviour and a strictly better
    failure than no run at all.
    """
    from ...agents.registry import AGENT_TOOLS
    from ...settings.config import load_settings
    from ...tools.registry import tool_pool

    reachable = {tool_id for tools in AGENT_TOOLS.values() for tool_id in tools}
    # Resolved settings, for the reason `cli/main.py:tools` documents: a bare
    # `Settings()` never reads `configs/<profile>.yaml`, so it under-reports
    # exactly the catalogue tools an operator pointed at another interpreter.
    return [c for c in tool_pool(load_settings().resolved()) if c.tool_id in reachable]


@lru_cache(maxsize=1)
def installed_generators() -> tuple[str, ...]:
    """Bare names of the de-novo generators that can sample on this machine.

    Cached for the process (building it constructs every adapter in the pool) and
    cleared the same way the prompt blocks are: ``installed_generators.cache_clear()``.
    """
    from ...tools.registry import GENERATOR_ID_PREFIX

    try:
        cards = _reachable_tool_cards()
    except Exception:
        return ()
    return tuple(
        sorted(
            c.tool_id.removeprefix(GENERATOR_ID_PREFIX)
            for c in cards
            if c.tool_id.startswith(GENERATOR_ID_PREFIX) and str(c.status) == "AVAILABLE"
        )
    )


@lru_cache(maxsize=1)
def installed_generator_facts() -> tuple[dict[str, Any], ...]:
    """What each installed de-novo generator conditions on and REFUSES without.

    A sibling of :func:`installed_generators`, deliberately not a replacement:
    that one returns bare names and is consumed as such by the console, the CLI
    and the pre-flight check.

    Every fact is read off the adapter itself (``conditioning``,
    ``synthesis_aware``, ``required_inputs``), never hand-listed here, so a model
    added to the catalogue describes itself and a table in a prompt cannot drift
    out of date against the code that runs.
    """
    from ...settings.config import load_settings
    from ...tools.registry import GENERATOR_ID_PREFIX, generator_adapters

    try:
        adapters = generator_adapters(load_settings().resolved())
    except Exception:
        return ()
    out: list[dict[str, Any]] = []
    for tool_id, adapter in sorted(adapters.items()):
        try:
            if not adapter.available():
                continue
        except Exception:
            continue
        out.append(
            {
                "name": tool_id.removeprefix(GENERATOR_ID_PREFIX),
                "conditioning": str(getattr(adapter, "conditioning", "pocket")),
                "synthesis_aware": bool(getattr(adapter, "synthesis_aware", False)),
                "required_inputs": tuple(getattr(adapter, "required_inputs", ()) or ()),
            }
        )
    return tuple(out)


#: How a `required_inputs` token reads to the model. A token with no entry is
#: shown verbatim, so a generator added later degrades to its field name rather
#: than vanishing from the row.
_INPUT_LABELS = {
    "pocket_pdb": "a prepared pocket file",
    "pocket_center": "its numeric centre",
    "ref_ligand_sdf": "a reference-ligand SDF",
    "input_ligand": "an input ligand",
    "fragment_atom_indices": "fragment atom indices",
}


def _needs_phrase(required: tuple[str, ...]) -> str:
    """The `refuses without` cell: an AND-list, because every token is mandatory."""
    parts = [_INPUT_LABELS.get(t, t) for t in required]
    if not parts:
        return "nothing"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " AND " + parts[-1]


def _detector_menu() -> str:
    """The detectors that can really run here, as a prompt fragment.

    Offering `vnegnn` while the availability block above says it is NOT installed
    is the same contradiction this whole change is about, one level down: a menu
    is a promise, and every option on it has to be real.
    """
    installed = [
        card.tool_id.removeprefix("pocket.detect.")
        for card in _reachable_tool_cards_safe()
        if card.tool_id.startswith("pocket.detect.") and str(card.status) == "AVAILABLE"
    ]
    if not installed:
        return '"auto" (no detector is installed here, so detection will find nothing)'
    return '"auto" (recommended), or one of: ' + ", ".join(sorted(installed))


def _reachable_tool_cards_safe() -> list[Any]:
    try:
        return _reachable_tool_cards()
    except Exception:
        return []


def _generator_block(cfg: Config) -> str:
    """What the de-novo arm can do here, so the master agent can drive it.

    The console used to make this choice with three buttons — "Library",
    "Library + PFM", "PFM only" — which put a scientific question to whoever
    clicked Launch and named one model out of the eleven installed. The decision
    moved to the agent that reads the goal. This block is the fact base it
    decides against; nothing here is a recommendation.

    It states three things the previous version did not, each of which cost a
    whole run. WHAT EACH MODEL NEEDS: the old block printed eleven bare names and
    one universal claim that de-novo generation is structure-conditioned, false
    for four of them, and said nothing about inputs — so the agent chose
    PocketXMol, whose de-novo tasks refuse without a numeric pocket centre, and
    all three cycles died on it. HOW TO OBTAIN A POCKET: the old block said the
    pocket was "a deployment input you do not choose or invent", which was an
    instruction never to ask for the one thing its choice required. WHAT THE
    CONFIGURED POCKET ACTUALLY IS: it asserted "a pocket IS configured here" from
    a path string it never stat-ed, and named a protein (sEH) unrelated to the
    goal.
    """
    facts = installed_generator_facts()
    if not facts:
        return (
            "CANDIDATE SOURCE\n"
            "No de-novo generator can sample on this deployment, so every candidate comes\n"
            "from the compound library plus deterministic analogues of your seed_smiles.\n"
            'Set generator to "library". Asking for de-novo structures here produces none.'
        )
    status = cfg.generator_status()
    lines = [
        "CANDIDATE SOURCE AND DE-NOVO GENERATOR — YOUR DECISION",
        "(installed facts below, read from this machine; the choice is yours)",
        "",
        "`generator` says which arms supply candidate structures:",
        '  "library"         the compound-library snapshot + deterministic analogues of',
        "                    your seed_smiles. Always available, needs no structure.",
        "                    Best when the goal is to FIND or PRIORITISE compounds.",
        '  "library+denovo"  both arms. Best when the goal wants novel chemotypes but',
        "                    should still be anchored by known compounds.",
        '  "denovo"          newly generated structures only. Best when the goal asks',
        "                    for genuinely new scaffolds and known chemistry is not wanted.",
        "",
        "`generator_backend` names WHICH model samples the de-novo arm. THEY ARE NOT",
        "INTERCHANGEABLE — each refuses to run without its own inputs. Installed here:",
        "",
        "| model | conditions on | refuses without | route? |",
        "| --- | --- | --- | --- |",
    ]
    for f in facts:
        needs = _needs_phrase(f["required_inputs"])
        lines.append(
            f"| {f['name']} | {f['conditioning']} | {needs} | {'yes' if f['synthesis_aware'] else 'no'} |"
        )
    unconditional = [f["name"] for f in facts if f["conditioning"] == "none"]
    needs_centre = [f["name"] for f in facts if "pocket_center" in f["required_inputs"]]
    unreachable = [f["name"] for f in facts if "ref_ligand_sdf" in f["required_inputs"]]
    lines += [
        "",
        'Read the table before you choose. "conditions on = none" means the model',
        "samples without any structure and a pocket you supply is discarded: "
        + (", ".join(unconditional) or "(none here)")
        + ".",
        '"route? = yes" means the model generates in reaction space and can report a',
        "synthesis route with the molecule.",
    ]
    if needs_centre:
        lines += [
            "",
            f"{', '.join(needs_centre)} additionally needs the pocket's numeric CENTRE, not",
            "just a pocket file. Choosing it without one is a guaranteed failure, not a",
            "degraded result: the run stops before anything is scored.",
        ]
    if unreachable:
        lines += [
            "",
            f"{', '.join(unreachable)} needs a reference-ligand SDF, which this deployment",
            "cannot produce. It is installed but not usable; choose another model.",
        ]
    lines += [
        "",
        "HOW TO GET A POCKET — this is yours to ask for:",
        '  `pocket_source: "detect"` runs this deployment\'s pocket detector on the',
        "    structure your target resolves to and hands the generator the site it finds,",
        "    INCLUDING the numeric centre. This is the only way to obtain a centre that a",
        "    detector actually predicted.",
        '  `pocket_source: "brief"` uses the pocket the operator configured, as-is.',
        '  `pocket_source: "auto"` detects when your request names a structure, else brief.',
        "  `pocket_detector`: " + _detector_menu() + ".",
        '  `pdb_ids`: name the structure to detect in, e.g. ["3I28"]. It is resolved',
        "    against a LOCAL structure cache; no network call is made.",
        "  Detection needs something to detect IN, so a pocket-conditioned model wants",
        "    kind=target_based with a named protein, or a pdb_id.",
        "  You never write x/y/z coordinates yourself. The detector measures them.",
    ]
    # The configured pocket, described honestly: which protein it is, whether the
    # file is actually there, and whether a centre can be had from it. The old
    # block asserted all three from a path string.
    pocket = status.get("pocket")
    lines.append("")
    if not pocket:
        lines.append(
            'No pocket is configured on this deployment, so `pocket_source: "brief"` supplies nothing.'
        )
    else:
        exists = bool(status.get("pocket_exists"))
        lines.append(f"The operator's configured pocket: {pocket}")
        if not exists:
            lines.append(
                "  THIS FILE DOES NOT EXIST on this machine. Treat the configured pocket as "
                'absent: use pocket_source="detect", or choose a model that needs no pocket.'
            )
        else:
            centre = status.get("pocket_center")
            lines.append(
                "  A centre can be measured from it: "
                + (
                    str(centre)
                    if centre
                    else "no — the file is not a pocket cut, so a box cannot be implied from it."
                )
            )
            lines.append(
                "  It is one specific site in one specific protein. If your goal is not that "
                "protein, detecting a pocket in YOUR target is the correct move; sampling this "
                "cavity would produce molecules shaped for the wrong site."
            )
    lines += [
        "",
        "`n_generated` is how many structures the de-novo arm samples (typical 20-200).",
        'It is ignored when generator is "library".',
        "",
        "De-novo molecules are proposals. They are standardized, screened, filtered and",
        "ranked by exactly the same pipeline as library compounds, and no number about",
        "them comes from the generative model.",
    ]
    return "\n".join(lines)


#: What the console passes when it is NOT pinning a candidate source — i.e. when
#: the master agent is to choose one. Deliberately not a valid brief token, so it
#: can never reach the pipeline unresolved.
GENERATOR_AUTO = "auto"

#: The smallest de-novo sample count that is still a de-novo arm. Below this the
#: run would report a generative program it never really ran.
_MIN_DE_NOVO_SAMPLES = 20


def resolve_candidate_source(req: ChemicalRequirements, cfg: Config) -> list[str]:
    """Check the chosen candidate source against what can really run here.

    Returns the notes to surface; ``req`` is corrected in place. The de-novo arm
    is turned OFF rather than quietly pointed at a different model whenever the
    choice cannot be honoured, because substituting one generative model for
    another — or a library search for a de-novo one — would put a number in front
    of the user that did not come from the method they were told about.
    """
    notes: list[str] = []
    if "pfm" not in req.generator:  # the wire token for "the de-novo arm"
        return notes
    # `n_generated` used to be the operator's number and is now the model's, so a
    # value the schema clamped out of garbage can reach the pipeline for the first
    # time. Sampling one molecule is not a small run, it is a dud that will read
    # as "de-novo produced nothing"; a sampling BUDGET is a resource, not a
    # measurement, so raising it and saying so invents nothing.
    if req.n_generated < _MIN_DE_NOVO_SAMPLES:
        notes.append(
            f"De-novo generation was chosen with n_generated={req.n_generated}, which would "
            f"sample too few structures to be a de-novo arm at all. Raised to "
            f"{_MIN_DE_NOVO_SAMPLES}."
        )
        req.n_generated = _MIN_DE_NOVO_SAMPLES
    engines = installed_generators()
    backend = req.generator_backend or GENERATOR_AUTO
    if not engines:
        notes.append(
            "De-novo generation was chosen but no generator is installed on this "
            "deployment. The de-novo arm is OFF and the run continues on the library "
            "alone; nothing is substituted for it."
        )
    elif backend != GENERATOR_AUTO and backend not in engines:
        notes.append(
            f"De-novo generation was chosen with generator_backend={backend!r}, which is "
            f"not installed here (installed: {', '.join(engines)}). The de-novo arm is OFF "
            f"rather than sampled from a different model."
        )
    else:
        missing = _unsatisfiable_inputs(req, cfg, backend)
        if not missing:
            return notes
        notes.append(missing)
    req.generator = "library"
    req.generator_backend = GENERATOR_AUTO
    return notes


def _resolved_backend_name(backend: str) -> str:
    """``auto`` named as the model it actually resolves to, for messages."""
    if backend != GENERATOR_AUTO:
        return backend
    engines = installed_generators()
    return "pfm" if "pfm" in engines else (engines[0] if engines else backend)


def _unsatisfiable_inputs(req: ChemicalRequirements, cfg: Config, backend: str) -> str | None:
    """Why the chosen backend cannot run here, or ``None`` when it can.

    The old check asked one question — "is a pocket configured?" — of every
    backend, and asked it of PFM's adapter whatever model had been chosen. It was
    wrong in both directions. It turned the arm off for the four installed models
    that need no pocket at all, and it waved through PocketXMol, whose de-novo
    tasks also need a numeric centre, straight into the failure this whole change
    exists to stop.

    The question is now per-backend and comes from the adapter's own
    ``required_inputs``, so it cannot drift from what the model will actually
    refuse.
    """
    name = _resolved_backend_name(backend)
    facts = {f["name"]: f for f in installed_generator_facts()}
    fact = facts.get(name)
    if fact is None:  # not describable here; the installed check already ran
        return None
    required = set(fact["required_inputs"])
    if not required:
        return None  # unconditional or ligand-conditioned: nothing to satisfy

    detecting = req.resolved_pocket_source() == "detect"
    status = cfg.generator_status()
    have_pocket = bool(status.get("pocket_exists"))
    have_centre = status.get("pocket_center") is not None

    if "ref_ligand_sdf" in required:
        return (
            f"{name} needs a reference-ligand SDF alongside the pocket, and this deployment "
            f"has no way to produce one. The de-novo arm is OFF rather than sampled from a "
            f"different model. Pocket-conditioned models that need only a pocket here: "
            f"{', '.join(n for n, f in sorted(facts.items()) if f['required_inputs'] == ('pocket_pdb',))}."
        )
    if "pocket_pdb" in required and not (detecting or have_pocket):
        reason = status.get("pocket_reason") or "no pocket is configured on this deployment"
        return (
            f"{name} is pocket-conditioned and no pocket is available: {reason}. Either set "
            f'pocket_source="detect" and name a protein or a pdb_id so the pocket agent can '
            f"find one, or choose a model that needs no structure "
            f"({', '.join(n for n, f in sorted(facts.items()) if not f['required_inputs'])}). "
            f"The de-novo arm is OFF for this run."
        )
    if "pocket_center" in required and not (detecting or have_centre):
        return (
            f"{name} denoises inside an explicit box, so it needs the pocket's numeric centre "
            f"and not only a pocket file — and none is available here. Either set "
            f'pocket_source="detect" (the detector measures the centre), or choose a '
            f"pocket-conditioned model that needs no centre "
            f"({', '.join(n for n, f in sorted(facts.items()) if f['required_inputs'] == ('pocket_pdb',))}). "
            f"The de-novo arm is OFF for this run."
        )
    return None


@lru_cache(maxsize=1)
def _availability_block() -> str:
    """Which capabilities this deployment does NOT have, as a compact prompt block.

    The planner was previously told nothing about deployment capability, so it
    could ask for a metric no installed tool can produce and then spend the whole
    cycle budget failing to reach it. This states the gap in ids plus one clause
    each, because it goes into every planning call.

    **Why this is not gated by ``RuntimePolicy.capability_envelope``.** That flag
    turns on an experimental treatment the self-evolution arm is studying: giving
    the planner *measured numbers* — "this deployment has been observed to reach
    pIC50 6.4582" — which is a claim about what the pipeline achieves and is
    fair to A/B. "AutoDock Vina is not installed on this machine" is not a
    treatment and not a measurement; it is a fact about the deployment, of the
    same kind as the model name, and withholding it does not make a run a
    cleaner baseline, only a less informed one. So the two are independent on
    purpose: the envelope stays behind the sealed flag, this is always on.

    Sourced live from the tool pool intersected with the agent/tool wiring, never
    from a hand-kept list — a tool no agent can reach is not this run's
    limitation (that is exactly the split ``reporting/report.py`` makes), and a
    list maintained here would drift the moment a backend was installed.

    Cached for the process: what is installed cannot change mid-run, and building
    it constructs every adapter in the pool. Degrades to "" rather than raising —
    a planner without this block is the pre-existing behaviour.
    """
    from ...tools.registry import GENERATOR_ID_PREFIX

    try:
        cards = _reachable_tool_cards()
    except Exception:
        return ""
    missing = [c for c in cards if str(c.status) != "AVAILABLE"]
    if not missing:
        return ""

    lines = [
        "DEPLOYMENT CAPABILITIES THAT ARE NOT INSTALLED HERE",
        "(read from this machine's live tool pool — a fact about the deployment,",
        " not a measurement, not a target, and not something refinement can change)",
        "",
        "These capabilities produce NOTHING on this deployment. Nothing is substituted",
        "for them: there is no mock, no fallback and no estimate.",
    ]
    # The de-novo generators are one family — one operation, one contract, twenty
    # implementations — so naming each absent one on its own line would spend the
    # whole block on a single capability the planner selects over as a group.
    generators = [c for c in missing if c.tool_id.startswith(GENERATOR_ID_PREFIX)]
    for card in (c for c in missing if not c.tool_id.startswith(GENERATOR_ID_PREFIX)):
        lines.append(f"- {card.tool_id} — {card.title or card.description}".rstrip(" —"))
    if generators:
        family = sum(1 for c in cards if c.tool_id.startswith(GENERATOR_ID_PREFIX))
        names = ", ".join(sorted(c.tool_id.removeprefix(GENERATOR_ID_PREFIX) for c in generators))
        lines.append(f"- de-novo generators — {len(generators)} of {family} not installed: {names}")
    lines += [
        "",
        "A success criterion, objective or property constraint whose value can only come",
        "from one of these can NEVER be satisfied here. It does not make the program",
        "stricter, it makes it unscorable, and the run spends its whole cycle budget on",
        "it. Plan with what is installed; if the goal genuinely needs a missing",
        "capability, say so in `risks` instead of encoding it as a criterion.",
    ]
    return "\n".join(lines)


#: Capabilities the master agent can actually direct, grouped by the requirements
#: field that reaches each one.
#:
#: Naming a tool here is a promise that some field of `ChemicalRequirements`
#: reaches it. That promise used to hold for exactly four tools — the pocket
#: detectors and the structure cache — and the block was written as though those
#: were the whole directable surface. They were not. They were the only ones
#: anybody had wired: `build_registry` constructed the molecule, synthesis,
#: structure and evaluation agents with their optional arms pinned off, and no
#: run parameter could pass anything else, so five installed tools that
#: `AGENT_TOOLS` advertised as reachable were reachable by nothing.
#:
#: Each group is (heading, how-you-reach-it, what-it-costs, {tool_id: one line}).
#: A group whose tools are all absent is dropped, so this describes THIS machine.
_DIRECTABLE_GROUPS: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "BINDING SITE",
        "`pocket_source`, `pocket_detector` and `pdb_ids`",
        "seconds",
        {
            "pocket.detect.p2rank": "rank ligandable cavities in a structure (the `auto` detector)",
            "pocket.detect.fpocket": "find cavities by alpha-sphere geometry",
            "pocket.detect.vnegnn": "find cavities with an E(3)-equivariant GNN",
            "structure.experimental": ("resolve a PDB id or target name to a cached, measured structure"),
        },
    ),
    (
        "TARGET EVIDENCE",
        "`target_evidence: true`",
        "seconds; it reads local snapshots, never the network",
        {
            "target.evidence.opentargets": ("association, tractability, safety and genetics for one gene"),
            "target.essentiality.depmap": "how essential the gene is across cell lines",
        },
    ),
    (
        "LEARNED PROPERTY PANELS",
        '`property_backends`, e.g. ["admetai", "chemprop"]',
        "roughly a minute per cycle for a few hundred candidates",
        {
            "property.admet.admetai": "a broad ADMET panel — absorption, clearance, tox flags",
            "property.predict.chemprop": "a trained message-passing model, where a checkpoint exists",
            "property.represent.unimol": "3D molecular representations for downstream prediction",
        },
    ),
    (
        "SYNTHETIC ROUTES",
        '`route_backend: "aizynthfinder"`',
        "a minute or two per molecule, for the first 12 candidates of each batch",
        {
            "synth.route.aizynthfinder": (
                "a real retrosynthetic route to purchasable starting material, not just a heuristic score"
            )
        },
    ),
    (
        "PREDICTED COMPLEXES",
        '`complex_backend: "boltz2"` or `"protenix"`',
        "MINUTES PER CANDIDATE — the most expensive thing you can switch on. "
        "It needs kind=target_based: the structure step it belongs to does not run "
        "for a phenotype program, so switching it on there buys nothing",
        {
            "structure.complex.boltz2": "a co-folded complex with an affinity head",
            "structure.predict.protenix": "a co-folded complex with a confidence head, no affinity",
        },
    ),
)

#: Tools that run every cycle whether or not you ask. Listed because a planner
#: told only about switches infers that everything else is off — the same
#: inference that read an absence list and concluded "pocket tools unavailable"
#: while three pocket tools were installed.
_ALWAYS_ON = {
    "chem.standardize": "identity, standardization and fingerprints",
    "model.activity": "the baseline potency model, refit each cycle",
    "synth.assess": "an SA score for every candidate",
    "safety.assess": "structural alerts and safety flags",
    "structure.simulate": "conformer minimisation",
    "structure.validate.posebusters": "physical plausibility of any pose produced",
}


@lru_cache(maxsize=1)
def installed_arms() -> dict[str, list[str]]:
    """The OPTIONAL assessment arms this deployment could run, by their field.

    The console's counterpart to :func:`installed_generators`: it cannot report
    what a run declined to use unless it knows what was on offer. Keys are the
    `ChemicalRequirements` field that switches the arm on, values the installed
    tools behind it, so a caller renders the offer without restating the wiring.
    """
    try:
        installed = {card.tool_id for card in _reachable_tool_cards() if str(card.status) == "AVAILABLE"}
    except Exception:
        return {}
    fields = {
        "target_evidence": "TARGET EVIDENCE",
        "property_backends": "LEARNED PROPERTY PANELS",
        "route_backend": "SYNTHETIC ROUTES",
        "complex_backend": "PREDICTED COMPLEXES",
    }
    by_heading = {heading: tools for heading, _, _, tools in _DIRECTABLE_GROUPS}
    out: dict[str, list[str]] = {}
    for field, heading in fields.items():
        present = sorted(t for t in by_heading.get(heading, {}) if t in installed)
        if present:
            out[field] = present
    return out


@lru_cache(maxsize=1)
def _reachable_block() -> str:
    """Capabilities that ARE installed and that the master agent can direct.

    `_availability_block` lists only what is missing, which is exactly half the
    picture and the half that misleads. Measured: the planner read
    "pocket.detect.vnegnn — NOT INSTALLED HERE", concluded in its own plan that
    "pocket tools unavailable", and never asked for a pocket — while p2rank,
    fpocket and the structure cache were all installed and reachable. An absence
    list teaches a model that everything it was not told about is absent too.

    Only capabilities the master agent can actually reach through a requirements
    field are listed. Naming a tool it cannot direct would be a different kind of
    lie — and for most of this block, that lie is what the wiring used to make
    true: the tools were installed, declared reachable, and switched on by
    nothing.
    """
    try:
        installed = {
            card.tool_id: card for card in _reachable_tool_cards() if str(card.status) == "AVAILABLE"
        }
    except Exception:
        return ""
    if not installed:
        return ""
    lines = [
        "CAPABILITIES YOU CAN DIRECT ON THIS DEPLOYMENT",
        "(installed and reachable; the absence list above is not the whole picture)",
        "",
        "Each group names the requirements field that switches it on, and what it",
        "costs. Everything here defaults to OFF except where the group says otherwise:",
        "a run that asks for nothing runs the cheap spine and nothing else. Switch on",
        "what the goal actually needs — an unasked-for arm is a capability this",
        "deployment paid to install and your program declined to use.",
    ]
    any_group = False
    for heading, field, cost, tools in _DIRECTABLE_GROUPS:
        present = {tid: why for tid, why in tools.items() if tid in installed}
        if not present:
            continue
        any_group = True
        lines += ["", f"{heading} — set {field}", f"  cost: {cost}"]
        lines += [f"  - {tid} — {why}" for tid, why in sorted(present.items())]
    if not any_group:
        return ""

    always = {tid: why for tid, why in _ALWAYS_ON.items() if tid in installed}
    if always:
        lines += ["", "ALWAYS ON — these run every cycle and you do not switch them:"]
        lines += [f"  - {tid} — {why}" for tid, why in sorted(always.items())]

    lines += [
        "",
        "You never call any of these yourself and you never write a coordinate, a",
        "route or a predicted value. You set the field; the pipeline runs the tool and",
        "every number that reaches a scorecard comes from the tool, not from you.",
    ]
    structures = _cached_structures()
    if structures:
        lines += [
            "",
            "Structures already in the local cache (no network call needed): " + ", ".join(structures) + ".",
            "A target with no cached structure cannot have a pocket detected in it; say so",
            "in `risks` rather than choosing a pocket-conditioned generator anyway.",
        ]
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _cached_structures() -> tuple[str, ...]:
    """PDB accessions the local structure cache can serve, read from the cache.

    Read, never hand-listed: a literal would be a new falsehood the first time an
    operator fetched or removed one.
    """
    try:
        from ...tools.structure.experimental_tool import ExperimentalStructureAdapter

        adapter = ExperimentalStructureAdapter()
        return tuple(sorted({p.stem.upper().split("_")[0] for p in adapter.entry_paths}))
    except Exception:
        return ()


@lru_cache(maxsize=1)
def _bioactivity_block() -> str:
    """Which targets this deployment can predict potency FOR, as a prompt block.

    A capability being installed says nothing about whether it has data for *your*
    target. The activity model is trained per run from the bioactivity connector's
    assays; for a target with no assays it cannot be trained, no candidate gets a
    pIC50, and a brief that marked ``pIC50`` required does not merely score badly —
    the ranking step refuses to rank at all and the run FAILS with nothing scored.

    Measured, and the reason this block exists: a "design a weight-loss drug"
    request planned against MC4R, transcribed ``pIC50`` as a required objective,
    and every one of its three cycles failed at ``integrate_objectives`` having
    built 108, 98 and 98 candidates. The console showed "0 of 5 criteria" three
    times over with no cause given. Nothing was wrong with the chemistry: the
    deployment simply has no MC4R bioactivity data, and no one had told the
    planner that this was a thing it needed to know.
    """
    from ...llm.bridge import bioactivity_coverage

    covered = bioactivity_coverage()
    if not covered:
        return ""
    return "\n".join(
        [
            "BIOACTIVITY DATA ON THIS DEPLOYMENT",
            "(a fact about the data installed here, not a measurement and not a target)",
            "",
            "The activity model that produces pIC50 is trained per run from this",
            "deployment's assay data. That data covers exactly "
            + (f"{len(covered)} target:" if len(covered) == 1 else f"{len(covered)} targets:"),
        ]
        + [f"- {label}" for label in covered]
        + [
            "",
            "For ANY other protein target there are no assays, the model cannot be trained,",
            "and NO candidate receives a pIC50. Consequences you must plan around:",
            "- `max_pIC50` and `median_pIC50` will have no value and cannot be scored.",
            "- An objective named `pIC50` with `required: true` makes the run FAIL: the",
            "  ranking step refuses to rank while a required objective has no predictions,",
            "  so the cycle ends with nothing scored at all — not a low score, no score.",
            "",
            "If the goal names a target that is not listed above, you have three honest",
            "options: pursue it with `pIC50` present but `required: false` and let the",
            "property/portfolio criteria carry the program; restate the goal as a",
            "phenotype-rescue program and score `restoration_median`; or say in `risks`",
            "that potency cannot be assessed here. Do NOT set a required potency",
            "objective, or a pIC50 success criterion, for an uncovered target.",
            "",
            "Note what the phenotype option COSTS. A phenotype program that names no",
            "protein leaves nothing for the structure cache to resolve and nothing for a",
            "pocket detector to work in — and a pocket-conditioned generator chosen",
            "alongside it will still be dispatched, and will still fail for want of a",
            "pocket. If the program wants structure-based de-novo design, stay",
            "`target_based`, name the protein, and use option one. If you do choose a",
            "phenotype program, you may still name `target_gene` or `pdb_ids` as the",
            "structural handle, and you should, whenever a de-novo arm is in play.",
        ]
    )


def _ceiling_block(ceilings: Mapping[str, float]) -> str:
    """The operator's stated ceilings for THIS run, as a binding prompt block.

    Mode 1 of the two ceiling modes. Unlike the measured envelope this is not an
    observation the planner may argue with — the operator said it at launch, and
    the planner's job is to plan inside it. It exists for one run: it is built
    here from run parameters and written nowhere.
    """
    if not ceilings:
        return ""
    # Which way "ceiling" points depends on how the metric is scored, and the one
    # place that already knows is the envelope tool's own direction table. Reused
    # rather than restated so the two cannot disagree about `sa_median`.
    from ...tools.envelope_tool import _MINIMIZE as _LOWER_IS_BETTER

    lines = [
        "HARD CEILINGS FOR THIS RUN (stated by the operator when the run was launched)",
        "",
        "| metric | ceiling | what it permits |",
        "| --- | --- | --- |",
    ]
    for key in sorted(ceilings):
        value = ceilings[key]
        if key in _LOWER_IS_BETTER:
            lines.append(f"| {key} | {value} | do not ask for a value BELOW {value} |")
        else:
            lines.append(f"| {key} | {value} | do not ask for a value ABOVE {value} |")
    lines += [
        "",
        "These are BINDING for this run. This deployment cannot go past them, so a",
        "success criterion, objective `target_value` or property constraint beyond one of",
        "them cannot be met by any amount of refinement — it does not make the program",
        "more ambitious, it only guarantees the criterion reads as missed and the cycle",
        "budget is spent. Set each criterion at or inside its ceiling. If the goal",
        "genuinely needs more than a ceiling allows, keep the criterion inside the",
        "ceiling and say in `risks` that the deployment cannot reach what the goal wants.",
        "A metric with no ceiling listed here is unconstrained by this block.",
    ]
    return "\n".join(lines)


def _join_blocks(*blocks: str) -> str:
    """Stack the non-empty prompt sections into the single ``envelope_block`` slot.

    ``prompts._extend`` offers one channel for runtime-built system-prompt text;
    a run can have up to three sections to send down it (what is not installed,
    the run's hard ceilings, the measured envelope), and which ones apply is this
    module's decision, not the prompt library's.
    """
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def _partition_criteria(
    criteria: list[SuccessCriterion],
) -> tuple[list[SuccessCriterion], list[SuccessCriterion]]:
    """Split criteria into those the evaluation agent can compute, and the rest.

    A criterion naming a metric nothing measures can never be marked met, so a
    plan containing one makes ACCEPT unreachable and the run burns every cycle
    chasing a target it cannot score. Unscorable criteria are therefore excluded
    from the accounting — but reported, never silently dropped: the planner
    wanting to measure something the platform cannot is a real finding.
    """
    scorable = [c for c in criteria if c.metric in METRIC_KEYS]
    unscorable = [c for c in criteria if c.metric not in METRIC_KEYS]
    return scorable, unscorable


def _plan_summary(plan: TaskPlan) -> str:
    crits = "; ".join(f"{c.metric} {c.target}" for c in plan.success_criteria) or "(none)"
    return f"Goal: {plan.goal_restatement}\nStrategy: {plan.strategy}\nSuccess criteria: {crits}"


def _req_summary(req: ChemicalRequirements) -> str:
    objs = ", ".join(
        f"{o.name}({o.direction},w={o.weight}{',req' if o.required else ''})" for o in req.objectives
    )
    cons = ", ".join(f"{c.property}{c.op}{c.value}" for c in req.property_constraints) or "(none)"
    return (
        f"kind={req.kind}; target={req.target_name or req.phenotype_description}; "
        f"objectives=[{objs}]; constraints=[{cons}]; seeds={len(req.seed_smiles)}; "
        f"novelty={req.novelty}; generator={req.generator}"
        + (f"(backend={req.generator_backend}, n={req.n_generated})" if "pfm" in req.generator else "")
        + f"; max_candidates={req.max_candidates}; shortlist={req.shortlist_size}"
        + f"; arms=[{_arms_summary(req)}]"
    )


def _arms_summary(req: ChemicalRequirements) -> str:
    """The optional assessment arms this run switched on, as a short phrase.

    Named in the summary because "arms=[none]" is a decision the operator should
    be able to see and question — the arms spent a long time being off with no
    way to say so, and a silent default reads exactly like a silent failure.
    """
    arms: list[str] = []
    if req.property_backends:
        arms.append("properties:" + "+".join(req.property_backends))
    if req.route_backend != "none":
        arms.append("routes:" + req.route_backend)
    if req.complex_backend != "none":
        arms.append("complex:" + req.complex_backend)
    if req.target_evidence:
        arms.append("target-evidence")
    return ", ".join(arms) or "none"


def _req_public(req: ChemicalRequirements) -> dict[str, Any]:
    return req.model_dump()


def _n_criteria_measured(report: EvaluationReport) -> int:
    """How many success criteria this cycle actually produced a value for.

    ``met`` is tri-state in :class:`~...evaluation_agent.MetricResult`: True,
    False, or ``None`` for "no value was computed". ``n_criteria_met`` counts
    only True, and ``n_criteria_total`` counts every criterion whether or not it
    was scored, so neither of them notices a cycle that stopped measuring.
    """
    return sum(1 for c in report.criteria if c.met is not None)


def _cycle_score_key(report: EvaluationReport) -> tuple[int, int, float]:
    """Rank one cycle against another: coverage first, then met, then utility.

    The old key was ``(n_criteria_met, headline_utility)`` and it rewarded data
    loss. Measured: a refinement emptied the objectives, so the potency criterion
    stopped producing a value (``met=None``, uncounted rather than failed) and
    the now-vacuous constraint pass-rate flipped to met. Criteria met ROSE from
    2 to 3, utility from 0.6981 to 1.0, and the cycle that had lost its
    requirements won the run.

    Putting measurement coverage first makes the comparison well-founded again:
    "3 of 3 met" over three measured criteria is not a better result than "2 of 4"
    over four, it is a less informative one, and a cycle that measured strictly
    fewer criteria cannot be preferred no matter how its remaining numbers look.
    Within equal coverage the ordering is exactly the old one, so a run whose
    cycles all measure the same criteria — every healthy run — selects precisely
    as it did before. The criteria list is fixed for the whole program, so
    coverage can only fall when a cycle lost the data to score one.
    """
    return (_n_criteria_measured(report), report.n_criteria_met, report.headline_utility or 0.0)


def _metrics_point(cycle: int, report: EvaluationReport) -> dict[str, Any]:
    def _m(key: str) -> float | None:
        for m in report.metrics:
            if m.key == key:
                return m.value
        return None

    return {
        "cycle": cycle,
        "n_criteria_met": report.n_criteria_met,
        "n_criteria_measured": _n_criteria_measured(report),
        "n_criteria_total": report.n_criteria_total,
        "headline_utility": report.headline_utility,
        "max_pIC50": _m("max_pIC50"),
        "hard_constraint_pass_rate": _m("hard_constraint_pass_rate"),
        "diversity": report.diversity,
        "n_pareto": report.n_pareto,
        "n_clusters": report.n_clusters,
        "shortlist_size": report.shortlist_size,
    }


class MasterAgent(ReasoningAgent):
    """Plans the research program and decides what happens each cycle.

    The only component in the system that calls a language model. It transcribes
    the user's instruction into machine-usable chemical requirements, dispatches
    the chemical and evaluation agents, critiques what comes back against its own
    success criteria, and decides whether to accept, refine, or replan. It never
    computes a scientific number.
    """

    agent_id = "agent.llm.master"
    llm_backed = True
    actions = {
        "plan": "Turn the instruction into a research plan with measurable success criteria.",
        "transcribe": "Turn intent into a ChemicalRequirements object (the generation prompt).",
        "critique": "Judge tool-computed results against the plan's criteria.",
        "refine": "Diagnose a shortfall and rewrite the generation requirements.",
    }

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = OllamaClient(
            cfg.ollama_host, timeout_s=cfg.request_timeout_s, keep_alive=cfg.keep_alive
        )
        # The two agents this one dispatches to. Neither calls a model.
        self.chemical = ChemicalAgent()
        self.evaluation = EvaluationAgent()

    def open_journal(
        self,
        session: ev.RunSession,
        instruction: str,
        *,
        source: str = "console",
        **config: Any,
    ) -> RunJournal | None:
        """Start the durable record for this request and attach it to the stream.

        Returns ``None`` when journaling is switched off. The journal is wired in
        as the session's event sink, so every event reaches disk as it is emitted
        rather than only if some consumer happens to be listening.

        Token-level streaming deltas are counted rather than stored: they run to
        thousands per program and the harness already writes the complete
        reasoning and reply of every exchange to ``chat.jsonl``. Keeping them
        would make the event log ten times larger and say nothing new.
        """
        if not self.cfg.journal_enabled:
            return None
        # A launch parameter left unset arrives as None and means "not specified",
        # not "null". Dropping it lets the deployment default underneath stand,
        # rather than overwriting `n_generated: 40` with `null` in the manifest
        # until transcription fills the real value back in.
        stated = {k: v for k, v in config.items() if v is not None}
        journal = RunJournal(
            session.run_id,
            root=self.cfg.resolved_journal_dir(),
            instruction=instruction,
            source=source,
            config={**self.cfg.public_dict(), **stated},
        )
        session.on_event = journal.record_event_filtered(ev.STREAMING_DELTAS)
        return journal

    async def run(
        self,
        session: ev.RunSession,
        instruction: str,
        *,
        strategy: str | None = None,
        max_cycles: int | None = None,
        model: str | None = None,
        generator: str | None = None,
        n_generated: int | None = None,
        ceiling_mode: str | None = None,
        ceilings: Mapping[str, float] | None = None,
        source: str = "console",
        journal: RunJournal | None = None,
    ) -> None:
        cfg = self.cfg
        strategy = (strategy or cfg.refine_strategy).lower()
        max_cycles = max_cycles or cfg.max_cycles
        model = model or cfg.model
        # `auto` — the console's default — hands the choice of candidate source to
        # the master agent. An explicit value is an OPERATOR PIN (`--generator`),
        # kept because the ablation arms need to hold the source fixed.
        generator = (generator or cfg.generator or GENERATOR_AUTO).lower()
        # Left as None when the caller said nothing: the master agent's own
        # n_generated then stands. Only a pinned source carries a pinned count.
        # Explicit arguments win; otherwise the launch parameters the CLI put on
        # the config for this one run. `resolved_ceiling_mode` is what refuses a
        # contradictory pair, so it runs before any model is contacted.
        run_ceilings = dict(cfg.ceilings if ceilings is None else ceilings)
        try:
            run_mode = (
                cfg.resolved_ceiling_mode() if ceiling_mode is None else (ceiling_mode or "").strip().lower()
            )
        except CeilingError as exc:
            # A launch parameter, so it fails before a model is contacted and
            # before a journal exists — and it fails as a run event rather than a
            # traceback, because the console has no other way to show it.
            await session.emit(ev.RUN_FAILED, title="Invalid ceiling for this run", error=str(exc))
            await session.close(error=str(exc))
            return
        policy = active_policy()
        if journal is None:
            journal = self.open_journal(
                session,
                instruction,
                source=source,
                model=model,
                strategy=strategy,
                generator=generator,
                n_generated=n_generated,
                max_cycles=max_cycles,
                ceiling_mode=run_mode,
                ceilings=run_ceilings,
                policy=policy.public_dict(),
                prompt_version=policy.prompt_version(),
                behaviour_version=policy.behaviour_version(),
            )
        try:
            result = await self._run(
                session,
                instruction,
                strategy=strategy,
                max_cycles=max_cycles,
                model=model,
                generator=generator,
                n_generated=n_generated,
                ceiling_mode=run_mode,
                ceilings=run_ceilings,
                journal=journal,
                policy=policy,
            )
        except OllamaError as exc:
            await session.emit(ev.RUN_FAILED, title="LLM error", error=str(exc))
            await session.close(error=str(exc))
            if journal is not None:
                journal.finish("FAILED", error=str(exc))
        except Exception as exc:
            await session.emit(ev.RUN_FAILED, title=f"{type(exc).__name__}", error=str(exc))
            await session.close(error=str(exc))
            if journal is not None:
                journal.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        else:
            if journal is not None:
                # A request whose every cycle's pipeline died scored nothing, and
                # recording that as SUCCEEDED was how three identical crashes came
                # to sit in the history as a healthy run: the index said SUCCEEDED,
                # `error` was null, and the offline analyzer therefore never saw
                # the failure class at all. The master agent's own loop completing
                # is not the same as the program producing a result.
                cycle_errors = [
                    err for c in result.get("cycles", []) if (err := (c.get("chemical") or {}).get("error"))
                ]
                every_cycle_failed = bool(cycle_errors) and len(cycle_errors) == len(result.get("cycles", []))
                journal.finish(
                    "FAILED" if every_cycle_failed else "SUCCEEDED",
                    error=(
                        f"every cycle's pipeline failed; the first reason was: {cycle_errors[0]}"
                        if every_cycle_failed
                        else None
                    ),
                    outcome={
                        "best_cycle": result.get("best_cycle"),
                        "cycles_run": len(result.get("cycles", [])),
                        "scoreboard": result.get("scoreboard", []),
                        "verdict": (result.get("best", {}).get("critique", {}) or {}).get("overall_verdict"),
                        "n_cycles_failed": len(cycle_errors),
                    },
                )

    async def _run(
        self,
        session: ev.RunSession,
        instruction: str,
        *,
        strategy: str,
        max_cycles: int,
        model: str,
        generator: str = GENERATOR_AUTO,
        n_generated: int | None = None,
        ceiling_mode: str = CEILING_MODE_OFF,
        ceilings: Mapping[str, float] | None = None,
        journal: RunJournal | None = None,
        policy: RuntimePolicy | None = None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        policy = policy or active_policy()
        ceilings = ceilings or {}
        # `dynamic` asks for the measurement as an operator decision, so it takes
        # the envelope whether or not the sealed policy flag (an experiment of the
        # self-evolution arm) happens to be on. `hard` does not measure at all:
        # the operator already said what the limit is.
        envelope = _envelope_block(policy, cfg, force=ceiling_mode == CEILING_MODE_DYNAMIC)
        # Always on and independent of any policy flag — see `_availability_block`.
        # `_bioactivity_block` is the same kind of fact one level down: not which
        # capabilities exist, but which targets the installed data can speak to.
        availability = _join_blocks(_availability_block(), _reachable_block(), _bioactivity_block())
        ceiling_text = _ceiling_block(ceilings) if ceiling_mode == CEILING_MODE_HARD else ""
        # The catalogue goes to the phases that act on it: transcription chooses a
        # generator, refinement re-chooses one after a failure. The PLANNER is
        # deliberately not shown it — a plan written around a model rather than
        # around the goal is the failure mode there. It is built even on an
        # operator-pinned run, because the refiner still has to diagnose a tool
        # failure against the real catalogue.
        generator_text = _generator_block(cfg)
        transcribe_generator_text = "" if generator != GENERATOR_AUTO else generator_text
        await session.emit(
            ev.RUN_STARTED,
            title="Master agent engaged",
            instruction=instruction,
            model=model,
            strategy=strategy,
            max_cycles=max_cycles,
            generator=generator,
            n_generated=n_generated,
            ceiling_mode=ceiling_mode,
            ceilings=dict(ceilings),
            config=cfg.public_dict(),
            policy=policy.public_dict(),
        )
        if availability:
            await session.emit(
                ev.LOG,
                phase="plan",
                title="Deployment capability stated to the planner",
                detail=availability.splitlines()[0],
                block=availability,
            )
        if ceiling_text:
            await session.emit(
                ev.LOG,
                phase="plan",
                title=f"Hard ceilings in force for this run: {len(ceilings)}",
                detail="; ".join(f"{k}={v}" for k, v in sorted(ceilings.items())),
                ceilings=dict(ceilings),
                ceiling_mode=ceiling_mode,
            )
        await ensure_model_available(self.client, model)
        run_model = model

        # ---- PHASE 1: PLAN ------------------------------------------------
        await session.emit(ev.PHASE_STARTED, phase="plan", title="Planning the research program")
        # The planner gets, in order: what this deployment does not have (fact,
        # always); the operator's binding ceilings if this is a `hard` run; and
        # the measured envelope if `dynamic` asked for it or the sealed policy
        # flag turned the experiment on.
        sys_p, usr_p = prompts.plan_prompt(
            instruction,
            envelope_block=_join_blocks(
                availability,
                ceiling_text,
                envelope if (policy.capability_envelope or ceiling_mode == CEILING_MODE_DYNAMIC) else "",
            ),
            addendum=policy.plan_prompt_addendum,
        )
        plan, _ = await run_structured(
            self.client,
            run_model,
            system=sys_p,
            user=usr_p,
            preset=cfgmod.PLANNING,
            schema=TaskPlan,
            session=session,
            phase="plan",
            max_repairs=cfg.max_json_repairs,
            think=cfg.think,
            structured=cfg.structured_outputs,
            journal=journal,
        )
        criteria, unscorable = _partition_criteria(plan.success_criteria or list(_DEFAULT_CRITERIA))
        if not criteria:
            # The plan asked only for things nothing measures. Fall back rather
            # than run three cycles that can never be judged.
            criteria = list(_DEFAULT_CRITERIA)
        if unscorable:
            await session.emit(
                ev.LOG,
                phase="plan",
                title=f"{len(unscorable)} success criteria cannot be measured — excluded from scoring",
                detail="; ".join(f"{c.metric} ({c.target})" for c in unscorable),
                unscorable=[c.model_dump() for c in unscorable],
                measurable_metrics=sorted(METRIC_KEYS),
            )
        await session.emit(
            ev.PLAN_READY,
            phase="plan",
            title="Plan ready",
            plan=plan.model_dump(),
            criteria=[c.model_dump() for c in criteria],
            unscorable=[c.model_dump() for c in unscorable],
        )

        # ---- PHASE 2: TRANSCRIBE -----------------------------------------
        await session.emit(
            ev.PHASE_STARTED, phase="transcribe", title="Transcribing intent → chemical requirements"
        )
        # The transcriber also writes numeric limits — objective `target_value`s
        # and property constraints — so a stated hard ceiling binds it too. The
        # *measured* envelope stays behind its own sealed flag here, because
        # `dynamic` mode's envelope is advice to the planner about what to aim
        # for, not a second opinion about how to phrase a constraint.
        sys_t, usr_t = prompts.transcribe_prompt(
            instruction,
            _plan_summary(plan),
            envelope_block=_join_blocks(
                availability,
                transcribe_generator_text,
                ceiling_text,
                envelope if policy.capability_envelope_transcribe else "",
            ),
            addendum=policy.transcribe_prompt_addendum,
        )
        req, _ = await run_structured(
            self.client,
            run_model,
            system=sys_t,
            user=usr_t,
            preset=cfgmod.TRANSCRIBE,
            schema=ChemicalRequirements,
            session=session,
            phase="transcribe",
            max_repairs=cfg.max_json_repairs,
            think=cfg.think,
            structured=cfg.structured_outputs,
            journal=journal,
        )
        # ---- CANDIDATE SOURCE --------------------------------------------
        # The master agent chooses it. Whether a goal wants NEW chemistry or a
        # search of KNOWN chemistry is a scientific judgement about the goal, and
        # the console used to put it to whoever clicked Launch as three buttons —
        # "Library", "Library + PFM", "PFM only" — while naming one model out of
        # the eleven installed. An explicit `generator` is still honoured, because
        # the ablation arms have to hold the source fixed across runs.
        pinned = generator != GENERATOR_AUTO
        if pinned:
            req.generator = generator
            if n_generated is not None:
                req.n_generated = n_generated
        # Whatever chose it, the choice is checked against what can really run.
        for note in resolve_candidate_source(req, cfg):
            await session.emit(
                ev.LOG,
                phase="transcribe",
                title="Candidate source changed — the de-novo arm cannot run here",
                detail=note,
                generator=req.generator,
                generator_backend=req.generator_backend,
                installed_generators=list(installed_generators()),
            )
        await session.emit(
            ev.LOG,
            phase="transcribe",
            title=(
                f"Candidate source: {candidate_source_label(req.generator, req.generator_backend)}"
                + (" (pinned by the operator)" if pinned else " (chosen by the master agent)")
            ),
            detail=(
                f"generator={req.generator}; generator_backend={req.generator_backend}; "
                f"n_generated={req.n_generated}"
            ),
            generator=req.generator,
            generator_backend=req.generator_backend,
            n_generated=req.n_generated,
            pinned=pinned,
        )
        # What the agent asked the rest of the platform to do, stated in the open.
        # Every one of these arms was unreachable before it had a field, so a run
        # that leaves them all off should have to show that it chose to.
        await session.emit(
            ev.LOG,
            phase="transcribe",
            title=f"Assessment arms switched on: {_arms_summary(req)}",
            detail=(
                f"property_backends={list(req.property_backends) or '[]'}; "
                f"route_backend={req.route_backend}; complex_backend={req.complex_backend}; "
                f"target_evidence={req.target_evidence}"
            ),
            property_backends=list(req.property_backends),
            route_backend=req.route_backend,
            complex_backend=req.complex_backend,
            target_evidence=req.target_evidence,
        )
        if journal is not None:
            # The journal index has one `generator` column and it is read as "what
            # this run actually sampled from"; leaving the launch sentinel there
            # would make every console run read `auto` forever.
            journal.config["generator"] = req.generator
            journal.config["generator_backend"] = req.generator_backend
            journal.config["n_generated"] = req.n_generated
            journal.config["property_backends"] = list(req.property_backends)
            journal.config["route_backend"] = req.route_backend
            journal.config["complex_backend"] = req.complex_backend
            journal.config["target_evidence"] = req.target_evidence

        # The goal kind is only known now, and it decides whether an efficacy
        # metric is computable at all. Reported, never silently dropped.
        if policy.goal_aware_criteria:
            criteria, goal_blocked = repair_criteria_for_goal(criteria, req.kind)
            if goal_blocked:
                unscorable = unscorable + goal_blocked
                await session.emit(
                    ev.LOG,
                    phase="transcribe",
                    title=(
                        f"{len(goal_blocked)} success criteria cannot apply to a "
                        f"{req.kind} program — excluded from scoring"
                    ),
                    detail="; ".join(f"{c.metric} ({c.target})" for c in goal_blocked),
                    goal_blocked=[c.model_dump() for c in goal_blocked],
                    kind=req.kind,
                    policy="goal_aware_criteria",
                )
        await session.emit(
            ev.REQUIREMENTS_READY,
            phase="transcribe",
            title="Chemical requirements (generation prompt)",
            requirements=_req_public(req),
            summary=_req_summary(req),
        )

        # ---- PHASE 3: DYNAMIC CYCLES -------------------------------------
        cycles: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for cycle in range(1, max_cycles + 1):
            await session.emit(
                ev.CYCLE_STARTED,
                phase="cycle",
                cycle=cycle,
                title=f"Cycle {cycle} / {max_cycles}",
                strategy=strategy,
                requirements=_req_public(req),
            )

            # Dispatch → chemical agent
            await session.emit(
                ev.DISPATCH, phase="cycle", cycle=cycle, title="→ CHEMICAL agent", agent="chemical"
            )
            backend_run, chem = await self.chemical.run(req, cfg, journal=journal, cycle=cycle)
            await session.emit(
                ev.CHEMICAL_RESULT,
                phase="cycle",
                cycle=cycle,
                title="Chemical agent result",
                result=chem.model_dump(),
                summary=chem.summary_line(),
            )
            # A pipeline that stopped early is the single most important fact
            # about the cycle, and it used to reach nobody: the state said FAILED,
            # every metric was null, and the reason died inside the engine. The
            # console showed a cycle that "scored 0 of 5" with no cause, and the
            # critic invented one. Surface it as its own event.
            if chem.error:
                await session.emit(
                    ev.LOG,
                    phase="cycle",
                    cycle=cycle,
                    title=f"Pipeline {chem.state} — nothing could be scored this cycle",
                    detail=chem.error,
                    pipeline_state=chem.state,
                    pipeline_error=chem.error,
                    n_pool_candidates=chem.n_candidates,
                    generator=req.generator,
                    generator_backend=req.generator_backend,
                    pocket_source=req.resolved_pocket_source(),
                )
                if journal is not None:
                    # Written HERE, not by `index_knowledge`, which sits downstream
                    # of the portfolio and therefore never runs on the runs whose
                    # failure is most worth remembering. A crashed cycle left no
                    # durable trace of what was attempted, so the offline analysis
                    # saw a healthy-looking run with null metrics.
                    journal.record_pipeline(
                        cycle=cycle,
                        run_id=chem.run_id,
                        state=chem.state,
                        error=chem.error,
                        generator=req.generator,
                        generator_backend=req.generator_backend,
                        pocket_source=req.resolved_pocket_source(),
                        n_candidates=chem.n_candidates,
                        note="recorded at the point of failure",
                    )

            # Dispatch → evaluation agent (independent judge)
            await session.emit(
                ev.DISPATCH, phase="cycle", cycle=cycle, title="→ EVALUATION agent", agent="evaluation"
            )
            report = self.evaluation.evaluate(backend_run, criteria, req.shortlist_size)
            await session.emit(
                ev.EVALUATION_RESULT,
                phase="cycle",
                cycle=cycle,
                title="Evaluation agent report",
                report=report.model_dump(),
            )
            point = _metrics_point(cycle, report)
            await session.emit(ev.METRICS, phase="cycle", cycle=cycle, title="Scoreboard", point=point)

            # Track the best cycle: criteria MEASURED, then met, then utility.
            score_key = _cycle_score_key(report)
            record = {
                "cycle": cycle,
                "requirements": _req_public(req),
                "chemical": chem.model_dump(),
                "evaluation": report.model_dump(),
                "metrics": point,
                "score_key": list(score_key),
            }
            if best is None or score_key > tuple(best["score_key"]):
                best = record
            elif score_key[1:] > tuple(best["score_key"][1:]):
                # It would have won on met-and-utility alone, and lost only on
                # coverage. That is the exact shape of the failure this ordering
                # exists to stop, so it is reported rather than silently applied.
                await session.emit(
                    ev.LOG,
                    phase="cycle",
                    cycle=cycle,
                    title=f"Cycle {cycle} not preferred: it measured fewer success criteria",
                    detail=(
                        f"cycle {cycle} scored {score_key[1]}/{score_key[0]} measured criteria "
                        f"(utility {score_key[2]}) but cycle {best['cycle']} measured "
                        f"{best['score_key'][0]}; unmeasured criteria are missing evidence, not "
                        "passes, so a cycle with less of it cannot rank above one with more"
                    ),
                    candidate_cycle=cycle,
                    incumbent_cycle=best["cycle"],
                    candidate_score_key=list(score_key),
                    incumbent_score_key=list(best["score_key"]),
                )

            # ---- CRITIQUE (LLM judges) -----------------------------------
            await session.emit(
                ev.PHASE_STARTED, phase="critique", cycle=cycle, title="Critiquing results vs plan"
            )
            sys_c, usr_c = prompts.critique_prompt(
                _plan_summary(plan),
                _req_summary(req),
                report.compact_json(),
                cycle,
                max_cycles,
                envelope_block=envelope if policy.capability_envelope else "",
                addendum=policy.critique_prompt_addendum,
            )
            critique, _ = await run_structured(
                self.client,
                run_model,
                system=sys_c,
                user=usr_c,
                preset=cfgmod.CRITIQUE,
                schema=Critique,
                session=session,
                phase="critique",
                cycle=cycle,
                max_repairs=cfg.max_json_repairs,
                think=cfg.think,
                structured=cfg.structured_outputs,
                journal=journal,
            )
            record["critique"] = critique.model_dump()
            await session.emit(
                ev.CRITIQUE_READY,
                phase="critique",
                cycle=cycle,
                title="Critique",
                critique=critique.model_dump(),
            )

            # Decision — the LLM's verdict, with deterministic safety nets.
            verdict = critique.overall_verdict
            if report.all_criteria_met:
                verdict = "ACCEPT"
            budget_left = cycle < max_cycles
            final = verdict == "ACCEPT" or not budget_left
            await session.emit(
                ev.DECISION,
                phase="cycle",
                cycle=cycle,
                title=f"Decision: {verdict}",
                verdict=verdict,
                is_final=final,
                reason=critique.summary or critique.diagnosis,
            )
            cycles.append(record)
            if final:
                break

            # ---- REFINE (the ablation branch) ----------------------------
            if strategy == "naive":
                await session.emit(
                    ev.REQUIREMENTS_ADJUSTED,
                    phase="refine",
                    cycle=cycle,
                    title="NAIVE RETRY — requirements unchanged (ablation baseline)",
                    strategy="naive",
                    changes=[],
                    diff=[],
                    requirements=_req_public(req),
                )
                continue

            await session.emit(
                ev.PHASE_STARTED,
                phase="refine",
                cycle=cycle,
                title="Adapting the generation prompt (diagnosing failure)",
            )
            # The refiner is the phase most likely to be looking at a TOOL
            # failure, and it was the one phase shown no tools: it diagnosed a
            # missing pocket centre correctly and then proposed generators that
            # do not exist here, because nothing had ever told it which do. It
            # gets the same deployment facts the transcriber gets.
            sys_r, usr_r = prompts.refine_prompt(
                req.model_dump_json(indent=2),
                f"verdict={critique.overall_verdict}; diagnosis={critique.diagnosis}",
                report.compact_json(),
                envelope_block=_join_blocks(
                    availability,
                    generator_text,
                    envelope if policy.capability_envelope else "",
                ),
                addendum=policy.refine_prompt_addendum,
            )
            adjustment, _ = await run_structured(
                self.client,
                run_model,
                system=sys_r,
                user=usr_r,
                preset=cfgmod.REFINE,
                schema=RequirementsAdjustment,
                session=session,
                phase="refine",
                cycle=cycle,
                max_repairs=cfg.max_json_repairs,
                think=cfg.think,
                structured=cfg.structured_outputs,
                journal=journal,
            )
            new_req = adjustment.adjusted_requirements
            # Refinement may only move *generation levers* — never redefine the
            # program identity. Preserve kind/target/phenotype from the original
            # transcription (a mis-flipped kind is what silently breaks scoring).
            new_req.kind = req.kind
            new_req.title = req.title
            new_req.target_name = req.target_name
            new_req.target_gene = req.target_gene
            new_req.target_uniprot = req.target_uniprot
            new_req.phenotype_description = req.phenotype_description
            new_req.cell_context = req.cell_context
            new_req.primary_objective = req.primary_objective
            # ---- THE CANDIDATE-SOURCE PIN --------------------------------
            # Held while the cycles are comparable, released when they are not.
            #
            # An unconditional pin was measured doing real harm: a run chose
            # PocketXMol, the pipeline refused for want of a pocket centre, the
            # refiner diagnosed that correctly and proposed a different backend —
            # and these two lines threw the fix away, every cycle, so three cycles
            # re-ran byte-identical requirements and failed identically. The
            # comparability the pin protects does not exist in that situation:
            # a cycle that scored nothing has nothing to compare.
            #
            # So: pin after a cycle that produced results, release after a cycle
            # that failed outright. Either way say which, in the open.
            source_change = (
                new_req.generator != req.generator or new_req.generator_backend != req.generator_backend
            )
            pin_released = bool(chem.error) and source_change
            if not pin_released:
                new_req.generator = req.generator
                new_req.generator_backend = req.generator_backend
            if source_change:
                await session.emit(
                    ev.LOG,
                    phase="refine",
                    cycle=cycle,
                    title=(
                        "Refinement moved the candidate source after a failed cycle"
                        if pin_released
                        else "Refinement guard: candidate source held for the program"
                    ),
                    detail=(
                        f"{req.generator}/{req.generator_backend} -> "
                        f"{new_req.generator}/{new_req.generator_backend}"
                        + (
                            "; the previous cycle scored nothing, so there is no comparability "
                            "to protect and the diagnosis is allowed to act"
                            if pin_released
                            else "; the previous cycle produced results, so the source is held "
                            "and the change is refused to keep the cycles comparable"
                        )
                    ),
                    released=pin_released,
                    old_generator=req.generator,
                    old_generator_backend=req.generator_backend,
                    new_generator=new_req.generator,
                    new_generator_backend=new_req.generator_backend,
                )
            merge_notes: list[str] = []
            if policy.refine_carry_forward:
                new_req, merge_notes = merge_adjustment(
                    req,
                    new_req,
                    adjustment.changes,
                    max_drop_fraction=policy.refine_max_drop_fraction,
                )
                for note in merge_notes:
                    await session.emit(
                        ev.LOG,
                        phase="refine",
                        cycle=cycle,
                        title="Refinement guard",
                        detail=note,
                        policy="refine_carry_forward",
                    )
            # ---- CORRECTNESS GUARD (not a policy) ------------------------
            # A refinement that empties a previously non-empty objectives or
            # property_constraints list is a DECODE FAILURE, not an adjustment.
            # Measured: one refinement returned both lists empty, the journal
            # recorded n_decode_failures=2 on that step, and the raw
            # adjusted_requirements JSON leaked into strategy_note — the model
            # never produced the object at all. Everything downstream then
            # collapsed: an objective-less brief scores nothing, the potency
            # criterion stopped being measured, and the degraded cycle went on to
            # win the run.
            #
            # This is unconditional on purpose. `RuntimePolicy.refine_carry_forward`
            # (sealed, off by default) is a *behavioural* treatment the
            # self-evolution arm is studying — which fields a legitimate
            # adjustment may drop, and by how much. Whether the platform accepts
            # an object the decoder failed to produce is not a treatment; it is
            # correctness, and no arm should be running without it. The guard sits
            # after the merge so `merge_adjustment` keeps its own semantics: if
            # carry-forward is on and restored the lists, nothing fires here.
            emptied = [
                name
                for name, before, after in (
                    ("objectives", req.objectives, new_req.objectives),
                    ("property_constraints", req.property_constraints, new_req.property_constraints),
                )
                if before and not after
            ]
            if emptied:
                # The whole object is suspect, not just the empty lists: it is the
                # output of a step that failed to decode. Keep the requirements
                # that were already working and let the next cycle re-run them.
                await session.emit(
                    ev.LOG,
                    phase="refine",
                    cycle=cycle,
                    title="Refinement REJECTED — it emptied " + " and ".join(emptied),
                    detail=(
                        "a refinement that deletes every "
                        + " and every ".join(emptied)
                        + " is a decode failure, not an adjustment; the previous requirements are kept "
                        "and the next cycle re-runs them"
                    ),
                    emptied=emptied,
                    rejected_changes=[c.model_dump() for c in adjustment.changes],
                    guard="refine_never_empties_requirements",
                )
                new_req = req
            diff = _diff_requirements(req, new_req)
            await session.emit(
                ev.REQUIREMENTS_ADJUSTED,
                phase="refine",
                cycle=cycle,
                title=(
                    "Adaptive re-prompt — requirements UNCHANGED (refinement rejected)"
                    if emptied
                    else "Adaptive re-prompt — requirements adjusted"
                ),
                strategy="adaptive",
                changes=[] if emptied else [c.model_dump() for c in adjustment.changes],
                strategy_note=adjustment.strategy_note,
                diff=diff,
                requirements=_req_public(new_req),
                rejected=emptied,
            )
            record["adjustment"] = {
                "changes": [] if emptied else [c.model_dump() for c in adjustment.changes],
                "diff": diff,
                "guard_notes": merge_notes,
                "rejected": emptied,
                "rejected_changes": [c.model_dump() for c in adjustment.changes] if emptied else [],
            }
            req = new_req

        # ---- FINAL STEP (dynamic ceiling mode): pick one candidate ---------
        assert best is not None
        selection: dict[str, Any] | None = None
        if ceiling_mode == CEILING_MODE_DYNAMIC:
            selection = await self._select_best_candidate(
                session,
                cycles,
                plan=plan,
                best=best,
                model=run_model,
                envelope=envelope,
                journal=journal,
            )

        # ---- FINALIZE -----------------------------------------------------
        result = {
            "instruction": instruction,
            "policy": policy.public_dict(),
            "strategy": strategy,
            # The RESOLVED source, not the launch sentinel: a caller reading this
            # wants to know what the run sampled from, and "auto" is not an answer.
            "generator": req.generator,
            "generator_backend": req.generator_backend,
            "generator_chosen_by": "operator" if pinned else "master_agent",
            "model": run_model,
            "ceiling_mode": ceiling_mode,
            "ceilings": dict(ceilings),
            "plan": plan.model_dump(),
            "criteria": [c.model_dump() for c in criteria],
            "unscorable_criteria": [c.model_dump() for c in unscorable],
            "best_cycle": best["cycle"],
            "best": best,
            "cycles": cycles,
            "scoreboard": [c["metrics"] for c in cycles],
            "portfolio": best["evaluation"],
            "final_selection": selection,
        }
        await session.emit(
            ev.RUN_SUCCEEDED,
            title=f"Program complete — best cycle {best['cycle']} "
            f"({best['evaluation']['n_criteria_met']}/{best['evaluation']['n_criteria_total']} criteria met)",
            result=result,
        )
        await session.close(result=result)
        return result

    async def _select_best_candidate(
        self,
        session: ev.RunSession,
        cycles: list[dict[str, Any]],
        *,
        plan: TaskPlan,
        best: dict[str, Any],
        model: str,
        envelope: str,
        journal: RunJournal | None,
    ) -> dict[str, Any] | None:
        """Pick the single best candidate across EVERY cycle, at maximum reasoning.

        The loop's own selector picks a winning *cycle*, and the shortlist of that
        cycle is what the user sees — which means a better molecule found in cycle
        1 is thrown away because cycle 3 scored better overall. This step exists to
        stop that: it is given every candidate from every cycle with the
        per-endpoint numbers the tool plane computed, and it chooses one.

        ADR-0011 is the whole shape of it. The model ranks and explains; it never
        produces a number. So the scores travel one way — into the prompt — and
        what comes back is an id and a reason. The id is then looked up in the
        table it was offered, and every score attached to the answer is re-read
        from the evaluation report, never from the model's reply. A name that was
        not in the table is a hallucination and the deterministic winner stands.

        Returns ``None`` only if there was nothing to choose between.
        """
        # One row per candidate per cycle, with its tool-computed endpoints. This
        # is deliberately every cycle's shortlist rather than the winning cycle's:
        # the point of the step is to see across cycles.
        rows: list[dict[str, Any]] = []
        for record in cycles:
            for cand in record["evaluation"].get("top_candidates", []):
                rows.append({"cycle": record["cycle"], **cand})
        if not rows:
            await session.emit(
                ev.LOG,
                phase="select_best",
                title="No candidates to select from — skipping the final selection",
                detail="no cycle produced a shortlist",
            )
            return None

        by_id = {str(row["candidate_id"]): row for row in rows if row.get("candidate_id")}
        # The deterministic fallback: the top of the winning cycle's shortlist —
        # exactly what the run would have presented without this step at all.
        best_top = best["evaluation"].get("top_candidates") or []
        fallback = {"cycle": best["cycle"], **best_top[0]} if best_top else rows[0]

        criteria_summary = (
            "\n".join(
                f"cycle {record['cycle']}: "
                + "; ".join(
                    f"{c['key']} target={c['target']} observed={c['value']} "
                    f"{'MET' if c['met'] else ('NOT MEASURED' if c['met'] is None else 'MISSED')}"
                    for c in record["evaluation"].get("criteria", [])
                )
                for record in cycles
            )
            or "(no criteria were scored)"
        )

        await session.emit(
            ev.PHASE_STARTED,
            phase="select_best",
            title=f"Selecting the best candidate across {len(cycles)} cycles ({len(rows)} candidates)",
        )
        sys_s, usr_s = prompts.select_best_prompt(
            _plan_summary(plan),
            criteria_summary,
            json.dumps(rows, indent=2, default=str),
            envelope_block=envelope,
        )
        chosen, _ = await run_structured(
            self.client,
            model,
            system=sys_s,
            user=usr_s,
            preset=cfgmod.SELECT_BEST,
            schema=BestCandidateSelection,
            session=session,
            phase="select_best",
            max_repairs=self.cfg.max_json_repairs,
            think=self.cfg.think,
            structured=self.cfg.structured_outputs,
            journal=journal,
        )
        named = chosen.candidate_id.strip()
        row = by_id.get(named)
        resolved = row is not None
        if row is None:
            await session.emit(
                ev.LOG,
                phase="select_best",
                title="Selection named a candidate that was not offered — keeping the ranked winner",
                detail=(
                    f"{named!r} is not one of the {len(by_id)} candidate ids in the table; "
                    f"falling back to {fallback.get('candidate_id')!r} from cycle {fallback.get('cycle')}"
                ),
                claimed_id=named,
                offered_ids=sorted(by_id),
            )
            row = fallback
        selection = {
            # Every number here is copied out of the evaluation report's row for
            # the chosen id. Nothing numeric comes from the model's reply.
            "candidate": dict(row),
            "candidate_id": row.get("candidate_id"),
            "cycle": row.get("cycle"),
            "reason": chosen.reason,
            "runner_up_id": chosen.runner_up_id if chosen.runner_up_id in by_id else "",
            "tradeoff": chosen.tradeoff,
            "confidence": chosen.confidence,
            "model_named_id": named,
            #: False means the model named an id nobody offered and the
            #: deterministic winner stood in — visible, never papered over.
            "resolved": resolved,
            "n_candidates_considered": len(rows),
            "n_cycles_considered": len(cycles),
            "preset": {
                "num_predict": cfgmod.SELECT_BEST.num_predict,
                "temperature": cfgmod.SELECT_BEST.temperature,
            },
        }
        await session.emit(
            ev.LOG,
            phase="select_best",
            title=f"Best candidate across all cycles: {row.get('candidate_id')} (cycle {row.get('cycle')})",
            detail=chosen.reason,
            selection=selection,
        )
        return selection


def _diff_requirements(old: ChemicalRequirements, new: ChemicalRequirements) -> list[dict[str, str]]:
    """Field-level diff for the UI (old → new), independent of the LLM's own list."""
    diff: list[dict[str, str]] = []

    def _add(field: str, o: Any, n: Any) -> None:
        if o != n:
            diff.append({"field": field, "old": str(o), "new": str(n)})

    # The candidate source is diffable now that a failed cycle may move it. While
    # the pin was unconditional these two could not change, so their absence cost
    # nothing; the moment the pin can release, a refinement that swapped the
    # backend would otherwise be journalled as `diff: []` — which is exactly the
    # shape the offline analyzer skips as "no adjustment happened".
    _add("generator", old.generator, new.generator)
    _add("generator_backend", old.generator_backend, new.generator_backend)
    _add("pocket_source", old.pocket_source, new.pocket_source)
    _add("pdb_ids", ",".join(old.pdb_ids) or "(none)", ",".join(new.pdb_ids) or "(none)")
    # The assessment arms are NOT pinned — they change what is measured, not what
    # is generated — so a refinement may move them every cycle and the journal has
    # to show it.
    _add(
        "property_backends",
        "+".join(old.property_backends) or "(none)",
        "+".join(new.property_backends) or "(none)",
    )
    _add("route_backend", old.route_backend, new.route_backend)
    _add("complex_backend", old.complex_backend, new.complex_backend)
    _add("target_evidence", old.target_evidence, new.target_evidence)
    _add("max_candidates", old.max_candidates, new.max_candidates)
    _add("n_generated", old.n_generated, new.n_generated)
    _add("shortlist_size", old.shortlist_size, new.shortlist_size)
    _add("novelty", old.novelty, new.novelty)
    _add("diversity_requirement", old.diversity_requirement, new.diversity_requirement)

    old_seeds, new_seeds = set(old.seed_smiles), set(new.seed_smiles)
    if old_seeds != new_seeds:
        added = sorted(new_seeds - old_seeds)
        removed = sorted(old_seeds - new_seeds)
        _add(
            "seed_smiles",
            f"{len(old_seeds)} seeds",
            f"{len(new_seeds)} seeds (+{len(added)}/-{len(removed)})",
        )

    old_obj = {o.name: o for o in old.objectives}
    new_obj = {o.name: o for o in new.objectives}
    for name in sorted(set(old_obj) | set(new_obj)):
        before, after = old_obj.get(name), new_obj.get(name)
        if before is None and after is not None:
            _add(f"objective:{name}", "—", f"added ({after.direction},w={after.weight})")
        elif after is None and before is not None:
            _add(f"objective:{name}", f"{before.direction},w={before.weight}", "removed")
        elif (
            before is not None
            and after is not None
            and (
                before.direction,
                before.weight,
                before.target_value,
                before.required,
            )
            != (after.direction, after.weight, after.target_value, after.required)
        ):
            _add(
                f"objective:{name}",
                f"{before.direction},w={before.weight},tv={before.target_value},req={before.required}",
                f"{after.direction},w={after.weight},tv={after.target_value},req={after.required}",
            )

    old_con = {c.property: c for c in old.property_constraints}
    new_con = {c.property: c for c in new.property_constraints}
    for prop in sorted(set(old_con) | set(new_con)):
        was, now = old_con.get(prop), new_con.get(prop)
        if was is None and now is not None:
            _add(f"constraint:{prop}", "—", f"{now.op}{now.value}")
        elif now is None and was is not None:
            _add(f"constraint:{prop}", f"{was.op}{was.value}", "removed")
        elif was is not None and now is not None and (was.op, was.value) != (now.op, now.value):
            _add(f"constraint:{prop}", f"{was.op}{was.value}", f"{now.op}{now.value}")
    return diff
