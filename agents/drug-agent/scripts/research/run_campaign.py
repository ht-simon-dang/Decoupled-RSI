#!/usr/bin/env python
"""Run one bounded, autonomous target-evolution campaign to a terminal state.

Step 11 of the per-target evolution program, and the operator-facing entry point
to the loop. Step 10's ``run_experiment.py`` dispatches **one** plan a human
wrote; this runs the whole cycle — Claude diagnoses the campaign's measured
bottleneck, proposes a bounded intervention, the dispatcher runs it in an
isolated checkout, the *parent's frozen evaluator* scores what it produced on the
development partition, and the result goes back into the next proposal — until
the declared stop rule says stop.

Everything about the bound is declared before the run and recorded:

* the **campaign record** (``--campaign``) declares the budget, and this script
  refuses one that exceeds the caps in ``configs/research/zak-c9r.yaml``. The
  ledger's own ``declare`` then refuses any later enlargement;
* ``--max-cycles`` is the cycle bound, counted across restarts from the durable
  history. There is no branch anywhere that grants a cycle because the last one
  did not improve;
* ``--per-cycle-output-tokens`` is the token headroom a cycle must have *before*
  it spends a reasoning call. Step 10 measured 20,899 output tokens for one
  worker session against a 100,000 campaign cap, so the token budget binds
  before the proposal budget and a cycle that cannot be paid for must not start;
* ``--max-generated`` is the ceiling on molecules an experiment may **generate**,
  written into ``<output>/run_constraints.json`` and enforced by the handler that
  generates them. It narrows one way, exactly like ``--allow``. It is not the
  submission budget — that stays at the registered 150, and a ceiling below it is
  refused rather than applied — and it is not a cost control, because the measured
  cost tracks molecules *scored* (13.5235 s each) rather than molecules generated:
  one campaign generated 2,000 candidates, cheap RDKit filters cut them to 1,213
  and only the submission was docked. It is a control on selection pressure.

**It bills the operator.** Every cycle makes at least one real Claude call.
Re-running the same campaign directory resumes it: already-spent cycles count
against the same ``--max-cycles``, and an already-settled request key is refused
rather than charged and executed twice.

Two things are read out of the committed records before a cycle can start, and
either one missing is a refusal rather than a run:

* the **frozen incumbent** — the number this campaign has to beat — and it is
  chosen by the campaign's own track. Track A is measured against the parent
  RandomForest's development MAE; Track B against the REINVENT4 prior's
  qualifying-scaffold count at the fixed 150-molecule submission budget;
* the **measurement semantics**, which say what every number the proposer is
  shown actually means. A campaign whose semantics record is absent or has
  drifted away from the records it quotes does not run on the old evidence
  channel: it stops, and says which entries drifted.

The exit status is the campaign's own. A campaign that ran its declared cycles
and found nothing better exits **0** carrying ``NO_IMPROVEMENT``: a scientific
outcome is not an execution failure.

Usage::

    PYTHONPATH=src conda run --no-capture-output -n drug-agent python \\
        scripts/research/run_campaign.py \\
        --campaign configs/research/zak-c9r/campaigns/step11-pilot.json \\
        --output .external/research/zak-c9r/derived/campaigns/step11-pilot \\
        --max-cycles 2 --per-cycle-output-tokens 40000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # pragma: no cover - operator convenience
    sys.path.insert(0, str(REPO / "src"))

from drug_agent.selfevolve.backends import ClaudeCLIBackend  # noqa: E402
from drug_agent.selfevolve.backends.claude_cli_backend import EFFORT_LEVELS  # noqa: E402
from drug_agent.selfevolve.target.actions import (  # noqa: E402
    # Imported rather than retyped: this script writes the file and the handlers
    # read it, and two spellings of the same filename is a constraint that is
    # declared here and silently never found there.
    RUN_CONSTRAINTS_FILENAME,
)
from drug_agent.selfevolve.target.budget import BudgetLedger  # noqa: E402
from drug_agent.selfevolve.target.controller import (  # noqa: E402
    CONTROLLER_ACTIONS,
    # The metric->loader pairing, imported rather than restated. The comment
    # below foresaw this bug and guarded only half of it: the track RULE was
    # imported, but the pairing of metric to frozen incumbent was still spelled
    # out here, and `drugagent self-evolve target-campaign` spelled out a
    # different one. It measured every Track B campaign against Track A's
    # incumbent until both were made to read this table.
    FROZEN_INCUMBENT_BY_METRIC,
    MAX_PROPOSER_RETRIES_CEILING,
    ClaudeProposer,
    StopRule,
    TargetCampaignController,
    # Private on purpose and imported anyway: it is the ONE place the track rule
    # lives. A second copy of "which track is scored on qualifying_scaffolds"
    # here is how an operator entry point ends up measuring a Track B campaign
    # against Track A's incumbent.
    _campaign_metric,
    check_campaign_registration,
    measurement_semantics,
)
from drug_agent.selfevolve.target.random_proposer import (  # noqa: E402
    UniformRandomProposer,
)

#: What the proposer is told about this host that it cannot measure for itself,
#: on EITHER track. Facts with evidence behind them, not encouragement: every
#: one of these was probed one verb at a time in Steps 8 and 10 and is recorded
#: in ``configs/research/zak-c9r/baseline-results.json``.
#: The adapt_generator note is deliberately written against the PROBE rather
#: than against a fixed answer. It used to state flatly that no generator on
#: this host exposes a 'fit' verb, which was Step 8's measurement and is no
#: longer necessarily true: a training verb has since been implemented for
#: molecule.generate.reinvent4, and the committed probe record is what decides
#: which of the two paragraphs below a given host is in. A note that hard-coded
#: either answer would be a false statement to the proposer on some host, and a
#: proposer that is told an available action is unavailable does not propose it.
_HOST_CAPABILITY_NOTES: tuple[str, ...] = (
    "adapt_generator's availability is decided by the committed capability probe for "
    "molecule.generate.pfm/'fit' and molecule.generate.reinvent4/'fit' -- see "
    "configs/research/zak-c9r/baseline-results.json, and 'requires_capabilities' in the action "
    "catalogue you were given. Where the verb is absent the action returns "
    "CAPABILITY_UNAVAILABLE naming it, and the proposal is spent learning what the probe already "
    "said. Where it is present, adapt_generator fine-tunes the generator prior on the ADMITTED "
    "train partition (the held-out partition is offered to the exclusion boundary and refused, "
    "and the counts are reported), and writes a checkpoint. It is scored on nothing by itself.",
    "adapt_generator inputs: backend, and OPTIONALLY training, an object taking partition "
    "(default 'train'), epochs, batch_size, device, input_model (a checkpoint to continue from; "
    "omit it to start from the parent prior) and output_model (the checkpoint's name RELATIVE to "
    "the experiment's output directory; default 'generator/<backend>_tl.prior'). It writes "
    "capability_probe.json, training_smiles.smi, adaptation.json and the checkpoint -- declare "
    "whichever of those your expected_outputs names. The exclusion registry is NOT an input: it "
    "is fixed, and the admitted/quarantined/held-out counts are reported on every run.",
    "adapt_generator and change_sampling are a TWO-CYCLE pair. The checkpoint adapt_generator "
    "writes changes nothing until a later change_sampling names it in inputs.sampling.checkpoint; "
    "without that the sampler loads the parent prior and the arm re-measures the incumbent. The "
    "adapt_generator result's notes end with 'CHECKPOINT WRITTEN: <absolute path>'; that exact "
    "string is what inputs.sampling.checkpoint takes in the second cycle.",
    "property.predict.chemprop fit/predict/export work; resume does not.",
    "structure.dock.vina and structure.dock.gnina dock for real. perturb.assess has no backend.",
)

#: The plan *shape* a worker-written module has to be asked for. Track-neutral:
#: it is about how the dispatcher launches the module, not about what the module
#: is scored on.
_WORKER_MODULE_PLAN_NOTE: str = (
    "A develop_algorithm / develop_architecture plan MUST declare inputs.command as an argument "
    "vector, e.g. ['{python}', '-m', 'drug_agent.selfevolve.target.experiments.<module>', "
    "'--output-dir', '{output}'], and inputs.module_to_write naming the single module to write."
)

#: Track A: predict a pActivity for each development molecule. Unchanged from
#: the single tuple this file carried before Track B existed — every campaign
#: run so far was shown exactly these notes, in this order.
TRACK_A_CAPABILITY_NOTES: tuple[str, ...] = (
    *_HOST_CAPABILITY_NOTES,
    "For a Track A predictor, the data interface is: "
    "from drug_agent.selfevolve.target.baseline_arms import prepare_track_a; "
    "prepared = prepare_track_a({'stratum_id': 'CHEMBL3886|Ki|F|BAO_0000019', "
    "'endpoint_label': 'pKi', 'seed': 42}). prepared['train'] maps inchikey -> row with "
    "row['molecule']['standard_smiles'] and row['p_value']: FIT ON THIS AND NOTHING ELSE. "
    "prepared['pool'] is the ordered list of development molecule ids to predict and "
    "prepared['evaluation'][key]['molecule']['standard_smiles'] is each SMILES. "
    "prepared['labels'] is the development answer key: DO NOT READ IT, do not fit on it and do "
    "not select a hyperparameter with it.",
    "A worker writes {molecule_id: endpoint_result(...)} to predictions.json and STOPS. "
    "from drug_agent.selfevolve.target.api import endpoint_result. Do not compute a score: an "
    "evaluator outside the checkout does that, and a self-reported number is not evidence.",
    _WORKER_MODULE_PLAN_NOTE,
)

#: Track B: submit molecules and be counted on distinct qualifying scaffolds.
#: The Track A data interface above is absent here on purpose — ``prepare_track_a``
#: and ``predictions.json`` are the wrong artefacts on this track, and a note
#: that names them would send a proposer to write a file nothing reads.
#:
#: The counts quoted below are Step 8's Stage D sweep, transcribed from
#: ``configs/research/zak-c9r/generation-sweep.json`` →
#: ``declared_secondary_rate.rows``; the campaign's own comparator is *read* out
#: of that same block by ``frozen_track_b_incumbent_and_references`` rather than
#: taken from this prose.
TRACK_B_CAPABILITY_NOTES: tuple[str, ...] = (
    *_HOST_CAPABILITY_NOTES,
    "change_sampling accepts inputs.backend 'pfm' or 'reinvent4' ONLY. Measured at Stage D over "
    "150 molecules those two produced 0 and 4 qualifying scaffolds respectively. The arms at the "
    "top of that nine-arm table -- synflownet (65) and pocketxmol (47) -- are NOT reachable from "
    "this action: no input to it selects them, so the headroom between 4 and 65 has to be earned "
    "by changing HOW the two reachable backends sample, not by naming a better one.",
    "change_sampling inputs: backend (above) and sampling, side by side under inputs. "
    "inputs.sampling takes n_molecules, seed, batch_size and max_runtime_seconds, plus pocket "
    "(a path to the prepared ZAK pocket PDB) for the pfm backend.",
    "inputs.sampling.n_generated is OPTIONAL and defaults to n_molecules. It is how many "
    "molecules the sampler is asked to produce, as distinct from n_molecules, which is how many "
    "are submitted. Setting it above n_molecules oversamples, and the submission is then the "
    "first n_molecules DISTINCT molecules IN THE ORDER THE SAMPLER EMITTED THEM, keeping the "
    "first occurrence of each duplicate -- generation order, not a ranking. Nothing chooses among "
    "them on docking score, on any property, or on any other quality signal. What oversampling "
    "buys is therefore a submission FILLED to the budget with distinct molecules, where an "
    "un-oversampled run submits however many distinct molecules 150 draws happened to contain; it "
    "does NOT buy a better-chosen 150, and the molecules are the same draws from the same "
    "generator either way. The rows in molecules.json are then written in canonical-SMILES "
    "ascending order, so the file you read back is NOT in generation order and the gen_NNNN ids "
    "are positions in that sorted listing rather than positions in the sampler's output -- "
    "generation order fixes WHICH molecules are submitted, not the order they are recorded in. "
    "funnel.json records the rule under selection_rule and the row order under "
    "submission_emission_order. "
    "It may never be below n_molecules. A run "
    "may declare a ceiling on it, and a ceiling that would cut into the submission budget is "
    "refused rather than applied.",
    "inputs.sampling.checkpoint is OPTIONAL and names a fine-tuned generator checkpoint for the "
    "reinvent4 backend. THE VALUE TO PASS IS THE ABSOLUTE PATH THE adapt_generator EXPERIMENT "
    "REPORTED -- it is quoted in that experiment's own result notes, after 'CHECKPOINT WRITTEN:', "
    "and recorded at evidence.adaptation.checkpoint and in its adaptation.json. Copy that string "
    "verbatim. It is NOT a path relative to anything: do not join it to the experiment id, do not "
    "relativize it, and do not write a description of where the path can be found instead of the "
    "path. A relative value is tried against the experiment's own output directory, the "
    "campaign's experiments root, the campaign root and the repository root, and if none of those "
    "holds the file the run is REFUSED rather than sampled -- a refusal names every base it "
    "tried. Omit the key to sample the parent prior. The sha256 of whatever is loaded is "
    "recorded in funnel.json either way, so an arm that meant to sample adapted weights and "
    "sampled the prior is visible afterwards instead of reading as the incumbent.",
    "inputs.sampling.n_molecules MUST be 150 for every experiment in this series. The incumbent "
    "is the same REINVENT4 prior measured at 150, and every arm in the nine-arm table was "
    "measured at 150. The metric is an absolute COUNT, not a rate, so a run at any other budget "
    "produces a number comparable to neither -- a larger budget would clear the bar on "
    "submission volume alone.",
    "change_sampling docks and property-profiles every molecule it samples and writes the scored "
    "rows to molecules.json. It does NOT grade itself: the frozen evaluator, outside the "
    "checkout, applies the seven gates to those rows and counts DISTINCT BEMIS-MURCKO SCAFFOLDS. "
    "Submitting duplicates of one scaffold raises nothing.",
    "develop_algorithm / develop_architecture are CONDITIONAL on this track, not unavailable: "
    "the evaluator scores whatever molecules.json the experiment directory holds, so a "
    "worker-written module that emits molecules.json in the same scored-row shape "
    "change_sampling writes is read and counted exactly the same way. The action catalogue "
    "describes those two actions in Track A terms; on this track the scored artefact they have "
    "to produce is molecules.json.",
    "A qualifying count of ZERO is a measured fixed-budget yield, not a broken arm. Six of the "
    "nine measured arms scored 0 or 1 of 150, and REAL human-measured ZAK actives clear the same "
    "seven gates at only 1 in 6. Report a zero as the result it is; re-running the same arm for "
    "a better draw spends a cycle on Poisson noise.",
    _WORKER_MODULE_PLAN_NOTE,
)

#: What a track's entry point returns: the frozen incumbent, and the references
#: around it that say how weak the evidence is.
_FrozenIncumbent = Callable[[Path], tuple[dict[str, Any], dict[str, Any]]]


def declare_run_constraints(
    output: Path, *, campaign_id: str, max_generated: int | None, allowed_actions: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Write this run's constraints into the campaign output root.

    ``(record, refusal)``: exactly one is ``None``.

    The constraints reach an action through the **campaign output root**,
    because nothing else does. ``worker.dispatch_plan`` has a closed keyword
    signature and ``worker._handler_argv`` builds a fixed argument vector, so a
    new dispatch keyword would have to change files inside the kernel integrity
    manifest; ``worker.candidate_env`` hands a candidate process an allowlist of
    variable *names*, so an unlisted environment variable is dropped rather than
    inherited. What every handler does already get is its output directory, and
    the controller puts that at ``<output root>/experiments/<experiment_id>``.

    **A ceiling only ever tightens.** This mirrors ``--allow``, which can remove
    an intervention from the menu and can never add one. Re-running a campaign
    directory with a larger ``--max-generated`` — or with none at all, once one
    has been declared — is refused, so a resumed campaign cannot quietly buy back
    the selection pressure the first run was capped at. A *smaller* value is
    accepted and the previous one is kept in ``max_generated_history``, so the
    tightening is legible rather than overwritten.

    That check is necessary and it is **not** what makes the ceiling airtight.
    This function writes and inspects exactly one path — the campaign root's —
    and a constraints file dropped into an experiment directory under it is
    invisible here, whatever it declares. The guarantee is enforced where the
    ceiling is read instead:
    :func:`~drug_agent.selfevolve.target.actions.run_constraints` takes the
    **minimum** over every constraints file on its bounded walk, so no file
    nested inside the run can raise what this one declares, and it refuses the
    run outright when one of them declares a looser ceiling.
    """
    path = output / RUN_CONSTRAINTS_FILENAME
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            return None, (
                f"{path} exists and does not parse ({exc}). A constraints file that cannot be "
                f"read is not an unconstrained run; repair or remove it deliberately"
            )
        if not isinstance(loaded, dict):
            return None, f"{path} does not hold a JSON object, so this run's ceiling is unreadable"
        previous = loaded

    prior_cap = previous.get("max_generated")
    history = [int(v) for v in (previous.get("max_generated_history") or []) if v is not None]
    if prior_cap is not None:
        if max_generated is None:
            return None, (
                f"{path} already declares max_generated={prior_cap} and this run passed no "
                f"--max-generated. Resuming without the ceiling would widen it, and a campaign "
                f"that can widen its own control is not a control. Pass --max-generated "
                f"{prior_cap} or lower"
            )
        if int(max_generated) > int(prior_cap):
            return None, (
                f"{path} declares max_generated={prior_cap} and this run asks for "
                f"{max_generated}. The ceiling may only tighten"
            )
        if int(max_generated) != int(prior_cap):
            history = [*history, int(prior_cap)]

    record = {
        "contract_version": "1.0.0",
        "record_kind": "campaign_run_constraints",
        "campaign_id": str(campaign_id),
        "max_generated": None if max_generated is None else int(max_generated),
        "max_generated_history": history,
        "allowed_actions": list(allowed_actions),
        "declared_by": "scripts/research/run_campaign.py",
        "declared_at": datetime.now(UTC).isoformat(),
        "note": (
            "max_generated is a ceiling on molecules GENERATED. It is a scientific control on "
            "selection pressure, not a cost control: scoring cost tracks molecules SCORED. It is "
            "a different quantity from the fixed submission budget the metric is an absolute "
            "count at, and a ceiling below that budget is refused by the handler rather than "
            "applied. It binds change_sampling only: develop_algorithm and develop_architecture "
            "run worker-authored code whose pool size no validator can see, and those arms are "
            "restricted by the action menu instead"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record, None


#: Which frozen incumbent, and which host notes, a campaign gets. Keyed by the
#: metric ``_campaign_metric`` derives from the campaign record's own ``track``
#: rather than by the track string itself, so this table cannot come to disagree
#: with the controller about which track a campaign is on.
#:
#: The LOADER half is not written out here — it is read from
#: ``controller.FROZEN_INCUMBENT_BY_METRIC``, the one table the CLI reads too.
#: Spelling the pairs out in both places is what let ``drugagent self-evolve
#: target-campaign`` hand every Track B campaign the Track A incumbent. Only the
#: host notes, which are this script's own, stay local.
_CAPABILITY_NOTES_BY_METRIC: dict[str, tuple[str, ...]] = {
    "mae_pactivity": TRACK_A_CAPABILITY_NOTES,
    "qualifying_scaffolds": TRACK_B_CAPABILITY_NOTES,
}
TRACK_SETUP: dict[str, tuple[_FrozenIncumbent, tuple[str, ...]]] = {
    metric: (loader, _CAPABILITY_NOTES_BY_METRIC[metric])
    for metric, loader in FROZEN_INCUMBENT_BY_METRIC.items()
    if metric in _CAPABILITY_NOTES_BY_METRIC
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign", required=True, type=Path, help="the Campaign record to run")
    parser.add_argument("--output", required=True, type=Path, help="campaign output root (resume point)")
    parser.add_argument("--ledger", default=None, help="budget ledger path (default: <output>/budget.sqlite)")
    parser.add_argument("--max-cycles", type=int, required=True, help="the declared cycle bound")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--per-cycle-output-tokens", type=float, default=None)
    parser.add_argument("--stop-on-first-improvement", action="store_true")
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    # Fresh proposer invocations after a transient ProposerUnavailable, bounded by the
    # controller's own ceiling. This buys no extra cycle, no extra experiment and no extra
    # chance to improve -- a retried call that succeeds runs the SAME cycle the failed one
    # was going to run -- so it is an infrastructure bound rather than a scientific one.
    # It matters because the random-menu control has no model in the loop and is
    # structurally immune to this failure, so a proposer outage can only ever remove
    # TREATMENT campaigns. Retrying removes that bias; it does not create one.
    parser.add_argument(
        "--max-proposer-retries",
        type=int,
        default=None,
        help=f"fresh proposer calls after a transient failure, 0..{MAX_PROPOSER_RETRIES_CEILING} "
        f"(default: the controller's own)",
    )
    parser.add_argument("--model", default=None, help="proposer model; the resolved id is recorded")
    # Reasoning effort is a per-call property of the CLI and it changes both what a
    # cycle costs and what it produces, so it is declared here and recorded beside
    # the model rather than left to whatever the CLI defaults to that week. An
    # unknown level is refused when the backend is constructed, before a cycle is
    # reserved, because a level the CLI rejects would otherwise be discovered after
    # the ledger had already been charged for the cycle carrying it.
    parser.add_argument(
        "--effort",
        default=None,
        choices=EFFORT_LEVELS,
        help="reasoning effort for the PROPOSER (default: the CLI's own)",
    )
    parser.add_argument(
        "--worker-effort",
        default=None,
        choices=EFFORT_LEVELS,
        help="reasoning effort for the experiment WORKER (default: --effort)",
    )
    parser.add_argument(
        "--proposer",
        choices=("claude", "random"),
        default="claude",
        help=(
            "'claude' is the reasoning controller. 'random' runs the registered "
            "ABL-2-shared-menu-random MECHANISM instead: uniform random selection over the "
            "same nine actions, fixed per-action defaults, no model anywhere in the loop and "
            "zero reasoning tokens. It is the control that separates choosing well from "
            "merely having a good menu, and it is seeded from the campaign record's own seed"
        ),
    )
    # 4096 was the first default and it TRUNCATED a real proposal: the CLI
    # returned "response exceeded the 4096 output token maximum", the answer was
    # lost, and the ledger was still charged 16,384 measured tokens for it.
    parser.add_argument("--proposer-max-output-tokens", type=int, default=16000)
    parser.add_argument("--proposer-timeout-s", type=float, default=600.0)
    parser.add_argument("--worker-max-output-tokens", type=int, default=16000)
    parser.add_argument("--worker-model", default=None)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--claude-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--allow",
        nargs="*",
        default=None,
        help=f"narrow the action menu (default: all of {list(CONTROLLER_ACTIONS)})",
    )
    parser.add_argument(
        "--max-generated",
        type=int,
        default=None,
        help="ENFORCED ceiling on the number of molecules an experiment may GENERATE. Like "
        "--allow it narrows one way and never widens: re-running a campaign directory with a "
        "larger value, or with none once one has been declared, is refused. This is NOT the "
        "submission budget -- that stays at the registered 150 and a ceiling below it is refused "
        "rather than applied -- and it is not a cost control, because scoring cost tracks "
        "molecules SCORED, not molecules generated. It is a control on selection pressure. It "
        "binds change_sampling only: develop_algorithm and develop_architecture run "
        "worker-authored code whose pool size is a module constant no validator can see, so "
        "those arms are restricted through --allow instead",
    )
    # The prior round's conclusions have to reach the proposer or every campaign
    # rediscovers them at a cycle apiece. `capability_notes` is the existing channel
    # -- it lands verbatim in the structured feedback -- and it had no operator route
    # into it. This is that route. The digest is recorded as the first note so a run
    # can always say WHICH text it was taught, which a path alone cannot.
    # The repository's own default is to record a prompt's content address and never its text,
    # which is right for records that get published. An operator auditing a four-day unattended
    # round needs the other thing: what the proposer was actually shown and what it actually
    # said. Opt-in, and off unless a directory is named.
    parser.add_argument(
        "--transcript-dir",
        default=None,
        type=Path,
        help="persist the FULL text of every model call here -- system prompt, user prompt, "
        "response, structured output and measured usage -- one JSON per call, in call order. "
        "The proposer's calls land in <dir>/proposer and the worker's in <dir>/worker",
    )
    parser.add_argument(
        "--lessons-file",
        default=None,
        type=Path,
        help="a UTF-8 text file whose blank-line-separated paragraphs are appended to the "
        "proposer's capability notes, preceded by a note naming the file and its sha256. The "
        "SAME file must be passed to treatment and control alike: notes that differ between arms "
        "are a difference between arms",
    )
    parser.add_argument("--no-probe", action="store_true", help="use the committed capability record")
    # Appendix A §4 lets the pilot change caps **before protocol freeze**, and
    # nothing may change one silently. This note is written next to every cap
    # this run declares, so a later reader of ledger.json sees who authorised
    # the number and why, rather than only the number.
    parser.add_argument(
        "--budget-note",
        default=None,
        help="justification recorded in the ledger beside every declared cap (e.g. an "
        "operator-authorised pilot increment and its reason)",
    )
    args = parser.parse_args(argv)

    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # --- the registration has to still hold ---------------------------------
    problems = check_campaign_registration(campaign, root=REPO)
    if problems:
        for problem in problems:
            print(f"REFUSED: {problem}", file=sys.stderr)
        return 2

    # --- and this run's one-way constraints, declared before anything runs ---
    if args.max_generated is not None and args.max_generated <= 0:
        print(
            f"REFUSED: --max-generated {args.max_generated} is not a number of molecules; a "
            f"ceiling of zero or less would forbid the generation it is meant to bound",
            file=sys.stderr,
        )
        return 2
    allowed_actions = list(args.allow) if args.allow else list(CONTROLLER_ACTIONS)
    constraints, refusal = declare_run_constraints(
        output,
        campaign_id=str(campaign["campaign_id"]),
        max_generated=args.max_generated,
        allowed_actions=allowed_actions,
    )
    if refusal is not None:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    assert constraints is not None
    if constraints["max_generated"] is None:
        print(f"generation ceiling: none declared (constraints {output / RUN_CONSTRAINTS_FILENAME})")
    else:
        print(
            f"generation ceiling: {constraints['max_generated']} molecules GENERATED per "
            f"experiment (the submission budget is unchanged); "
            f"constraints {output / RUN_CONSTRAINTS_FILENAME}"
        )

    # --- and so does the evidence channel -----------------------------------
    # A campaign that cannot read its measurement semantics must not fall back to
    # the channel that had none: that channel is what let a proposer cite an
    # "AUC 0.74" that appeared nowhere in its feedback record.
    try:
        semantics = measurement_semantics(REPO)
    except (OSError, ValueError) as exc:
        print(
            "REFUSED: the campaign's measurement semantics could not be read, so the proposer "
            "would be shown numbers with nothing saying what they mean:",
            file=sys.stderr,
        )
        for problem in str(exc).split("; "):
            print(f"  {problem}", file=sys.stderr)
        return 2

    # The campaign record's own track decides what it is measured against and
    # what it is told about this host. Both come out of one lookup, so a Track B
    # campaign cannot end up holding Track A's incumbent or Track A's notes.
    metric = _campaign_metric(campaign)
    setup = TRACK_SETUP.get(metric)
    if setup is None:
        print(
            f"REFUSED: campaign {campaign.get('campaign_id')!r} is scored on metric {metric!r} "
            f"and no frozen incumbent is registered for it; the registered metrics are "
            f"{sorted(TRACK_SETUP)}",
            file=sys.stderr,
        )
        return 2
    frozen, capability_notes = setup
    incumbent, references = frozen(REPO)
    print(
        f"incumbent: {incumbent['artifact_id']} {incumbent['metric']}={incumbent['value']}"
        f" (read from {incumbent['source']})"
    )

    ledger = BudgetLedger(
        Path(args.ledger) if args.ledger else output / "budget.sqlite",
        campaign_id=str(campaign["campaign_id"]),
    )
    for resource, limit in sorted(dict(campaign["budget"]).items()):
        note = f"declared by {args.campaign.name}"
        if args.budget_note:
            note = f"{note}; {args.budget_note}"
        ledger.declare(str(resource), float(limit), note=note)

    # --- the worker's reasoning effort, delivered WITHOUT touching the worker ----
    # `target/worker.py` is inside the frozen evaluation kernel's guarded set (see
    # `evaluation/integrity.py::_extra_guarded_files`: the worker decides which
    # checkout, interpreter and import path an evaluation runs under, so a candidate
    # able to edit it could point its own evaluation back at the parent). Adding an
    # `effort` parameter to it was tried and MOVED the frozen evaluator address from
    # sha256:fb281364... to sha256:e9fd2ca0..., which would have invalidated the
    # `evaluator_hash` every one of the 101 campaign records declares and the
    # accepted-corpus builder's frozen-evaluator gate. So the effort is delivered by
    # handing the worker a backend that already carries it: `run_claude_experiment`
    # builds one only when it is given none, and `complete()` resolves an unset
    # per-call effort from the backend's own default. Nothing guarded changes.
    #
    # The construction mirrors the one in `run_claude_experiment` exactly apart from
    # the effort -- same ledger, same default phase and owner, same token cap, same
    # timeout -- because a backend wired differently from the one it replaces would
    # account differently. `cwd` is deliberately NOT set here: the worker passes the
    # candidate checkout per call, and a per-call cwd wins over the backend's own.
    worker_effort = args.worker_effort or args.effort
    worker_backend_kwargs: dict[str, Any] = {}
    # Supplied when EITHER an effort or a transcript directory is asked for. Keying it on effort
    # alone would have left every worker call unrecorded on a run that asked only for
    # transcripts, which is the half of the conversation an operator most wants.
    if worker_effort or args.transcript_dir:
        worker_backend_kwargs["claude_backend"] = ClaudeCLIBackend(
            default_model=args.worker_model or args.model or "sonnet",
            default_effort=worker_effort,
            timeout_s=args.claude_timeout_s,
            ledger=ledger,
            default_max_output_tokens=args.worker_max_output_tokens,
            transcript_dir=(args.transcript_dir / "worker") if args.transcript_dir else None,
        )
        print(f"worker backend: effort {worker_effort} (supplied; the guarded worker is unchanged)")

    # The random control opens no reasoning backend at all. Constructing one and
    # then not calling it would bill the ledger nothing but would still refuse the
    # campaign on a host where the CLI is unavailable -- and the whole point of
    # this arm is that it needs no model. Its manifest says so: "consumes zero
    # output tokens by construction".
    proposer: Any
    if args.proposer == "random":
        proposer = UniformRandomProposer(seed=int(campaign["seed"]))
        print(
            f"proposer: uniform random over the registered menu, seed {campaign['seed']}"
            f" (the ABL-2 mechanism; no model in the loop, zero reasoning tokens)"
        )
    else:
        backend = ClaudeCLIBackend(
            default_model=args.model or "sonnet",
            default_effort=args.effort,
            transcript_dir=(args.transcript_dir / "proposer") if args.transcript_dir else None,
            timeout_s=args.proposer_timeout_s,
            cwd=REPO,
            allowed_tools=(),  # the proposer reasons; it does not touch the checkout
            ledger=ledger,
            budget_phase="ADAPTATION",
            budget_owner="campaign_proposer",
            default_max_output_tokens=args.proposer_max_output_tokens,
        )
        usable, why = backend.available()
        if not usable:
            print(f"BLOCKED: the reasoning backend is unavailable: {why}", file=sys.stderr)
            return 4
        proposer = ClaudeProposer(
            backend=backend,
            model=args.model,
            max_output_tokens=args.proposer_max_output_tokens,
            timeout_s=args.proposer_timeout_s,
        )

    # A ceiling the proposer is not told about is one it spends cycles walking
    # into. It is appended rather than folded into TRACK_*_CAPABILITY_NOTES
    # because it is this RUN's constraint, not this host's capability, and the
    # notes above have to stay identical across treatment and control.
    notes = list(capability_notes)
    if args.lessons_file is not None:
        # A lessons file that silently taught nothing is worse than none at all: the run
        # would record that it was taught and the proposer would have seen no lesson.
        try:
            lessons_text = args.lessons_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"REFUSED: --lessons-file {args.lessons_file} could not be read: {exc}", file=sys.stderr)
            return 2
        paragraphs = [p.strip() for p in lessons_text.split("\n\n") if p.strip()]
        if not paragraphs:
            print(
                f"REFUSED: --lessons-file {args.lessons_file} holds no non-empty paragraph, so it "
                f"would teach nothing while recording that it had",
                file=sys.stderr,
            )
            return 2
        digest = hashlib.sha256(lessons_text.encode("utf-8")).hexdigest()
        notes.append(
            f"The {len(paragraphs)} notes that follow are the prior round's established "
            f"conclusions, read from {args.lessons_file.name} (sha256:{digest}). They are findings "
            f"already measured on this target, not hypotheses to re-test. Design against them"
        )
        notes.extend(paragraphs)
        print(f"lessons: {len(paragraphs)} notes from {args.lessons_file} (sha256:{digest[:16]}...)")
    if constraints["max_generated"] is not None:
        notes.append(
            f"THIS RUN IS CAPPED: an experiment may GENERATE at most "
            f"{constraints['max_generated']} molecules. An inputs.sampling.n_generated above the "
            f"ceiling is tightened to it rather than refused, and funnel.json records what was "
            f"asked for, the ceiling, and what was actually emitted -- so proposing a larger pool "
            f"buys nothing. The submission budget is unchanged and unaffected"
        )

    controller = TargetCampaignController(
        campaign,
        proposer=proposer,
        ledger=ledger,
        output_root=output,
        allowed_actions=allowed_actions,
        stop_rule=StopRule(
            max_cycles=args.max_cycles,
            stop_on_first_improvement=args.stop_on_first_improvement,
            min_improvement=args.min_improvement,
            max_infrastructure_retries=args.max_infrastructure_retries,
            per_cycle_output_tokens=args.per_cycle_output_tokens,
            **(
                {"max_proposer_retries": args.max_proposer_retries}
                if args.max_proposer_retries is not None
                else {}
            ),
        ),
        incumbent=incumbent,
        references=references,
        semantics=semantics,
        capability_notes=notes,
        dispatch_kwargs={
            **worker_backend_kwargs,
            "probe": not args.no_probe,
            "timeout_s": args.timeout_s,
            "claude_timeout_s": args.claude_timeout_s,
            "claude_model": args.worker_model or args.model,
            "max_output_tokens": args.worker_max_output_tokens,
        },
    )
    record = controller.run()

    print(f"campaign {record['campaign_id']}: {record['state']} ({record['terminal_reason']})")
    print(f"  scientific decision: {record['scientific_decision']}")
    print(
        f"  cycles: {record['cycles_completed']}/{record['stop_rule']['max_cycles']}"
        f" ({record['attempts']} attempts recorded)"
    )
    for candidate in record["candidate_pool"]:
        print(
            f"  - {candidate['experiment_id']}: {candidate['outcome']}"
            f" {record['development_metric']}={candidate['development_metric_value']}"
            f" eligible={candidate['nomination_eligible']}"
        )
    if record["nomination"]:
        print(f"  nominated: {record['nomination']['artifact_id']} (test_access_count=0)")
    else:
        print(f"  no nomination: {record['selection']['reason']}")
    if record["blocked_cause"]:
        print(f"  cause: {record['blocked_cause']['code']} — {record['blocked_cause']['detail']}")
        print(f"  unblock: {record['blocked_cause']['unblock']}")
    ledger.write_report(output / "ledger.json")
    print(f"  history: {record['history_path']}")
    print(f"  campaign record: {output / 'campaign.json'}")
    print(f"  ledger report: {output / 'ledger.json'}")
    return int(record["exit_code"])


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
