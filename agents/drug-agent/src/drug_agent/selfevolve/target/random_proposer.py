"""ABL-2's proposer: uniform random choice over the same registered menu.

The control arm ``ABL-2-shared-menu-random`` is frozen in
``configs/research/zak-c9r/protocol.json`` and in its own study manifest
``configs/research/zak-c9r/ablations/ABL-2-shared-menu-random.json`` — seeds
211–215, ``inference_label: CONFIRMATORY`` — and it moves exactly three pointers
against the full self-evolution loop::

    /controller/proposer                  MODEL_PROPOSER -> NONE
    /controller/selection                 MODEL_PROPOSAL -> UNIFORM_RANDOM_OVER_THE_REGISTERED_MENU
    /controller/per_action_configuration  MODEL_CHOSEN   -> FIXED_DEFAULT

Nothing else moves. The menu, the caps, the plan contract, and the
``prior_hypotheses`` and ``binding_constraints`` the feedback record still
carries are all untouched, because an arm that changed a second thing would
answer a second question and attribute the difference to neither.

It is the single most important control in the program: without it a positive
self-evolution result cannot distinguish *the agent's reasoning works* from *the
menu contains something that works*.

**Why this control degrades honestly, and why that is not a defect to fix.**

Several of the nine registered actions cannot succeed from a cold start, and the
fixed default configuration for each of them is a straightforward attempt that
fails for its real reason:

* ``distill`` consumes a teacher table that a separate scoring pass produces; it
  never makes one, so the first cycle to draw it is refused with the exact
  command that would build the table;
* ``evaluate_development`` re-scores a ``predictions.json`` an earlier
  experiment wrote, and a cold campaign has none;
* ``adapt_generator`` runs a capability probe and returns whatever that probe
  measures. It answered ``CAPABILITY_UNAVAILABLE`` on every campaign recorded so
  far, because neither generator adapter bound a training verb; it is written
  here as an outcome that was *measured* rather than as a property of the host,
  because the verb is being worked on and this paragraph would otherwise go on
  describing a capability that had changed. Whether it is refused or does real
  work, the draw is not re-rolled and the default configuration does not move;
* ``develop_algorithm`` and ``develop_architecture`` run a module that, in this
  arm, nobody writes. The arm's own manifest says *there is no model in this
  loop*, so their default declares backend ``none``: summoning a code-writing
  worker would put a model back inside the control and turn it into the
  treatment it exists to be compared against.

**A random chooser spending cycles on actions that cannot score is the
measurement, not a bug in it.** That is precisely the quantity CMP-6 compares
against ``B5-self-evolution``. So this module must never:

* special-case, skip, retry or re-draw an action that cannot succeed;
* narrow the menu to the actions that happen to work;
* read ``prior_hypotheses``, ``negative_findings``, ``failure_categories``,
  ``binding_constraints`` or ``scoring_contract`` to steer the draw.

Every one of those would make this arm a *weaker agent* rather than the absence
of one, and the comparison would stop isolating controller choice. An adaptive
control is not a control. The menu and the declared budget caps are the only
things read here, and the second only to keep a plan payable.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .controller import CONTROLLER_ACTIONS, MalformedProposal

Record = dict[str, Any]

#: The registered arm this proposer *is*. Written into every diagnosis so a
#: campaign record cannot be mistaken for a reasoning run after the fact.
ABL2_ARM_ID = "ABL-2-shared-menu-random"

#: ``ClaudeProposer`` stamps ``"measured"`` on what a model returned. A draw is
#: not a model output and must never be read as one, so it gets its own class
#: rather than a shared label with the treatment arm.
RANDOM_CONTROL_EVIDENCE_CLASS = "random_control"

#: This proposer's name, and the ``backend`` its zero-token usage record names.
PROPOSER_NAME = "uniform_random_menu"

#: What every ABL-2 proposal changes relative to its parent, verbatim. One
#: entry, and it is the arm's declared factor rather than a per-action
#: description: nothing about the experiment was chosen, only which experiment.
CHANGED_FACTOR = "controller_choice: action drawn uniformly at random, not chosen"


class _CampaignSeed:
    """Where the campaign's own seed belongs inside a default configuration.

    A sentinel rather than a format string, so a default that somehow reached a
    plan without substitution would be a ``TypeError`` at JSON encoding instead
    of an experiment quietly running on the literal text ``"{seed}"``.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "CAMPAIGN_SEED"


CAMPAIGN_SEED = _CampaignSeed()

#: The fixed budget slice every ABL-2 proposal asks for.
#:
#: Read off the arm's own frozen manifest — ``wall_seconds`` 86,400,
#: ``gpu_seconds`` 43,200, ``max_concurrent_gpu_jobs`` 1 and
#: ``experiment_proposals`` 6 — as one experiment's even share of the six the
#: arm is allowed. ``llm_output_tokens`` is 0.0 because there is no model in
#: this loop; the arm's symmetric 75,000-token allocation exists so the
#: comparison is matched on *opportunity*, and spending any of it here would
#: make the arm's declared spend a fiction.
#:
#: Fixed, not computed. Sizing a request from what the campaign has left is the
#: first step of adapting to the campaign's state, and this arm does not adapt.
ABL2_CYCLE_BUDGET: Mapping[str, float] = MappingProxyType(
    {
        "experiment_proposals": 1.0,
        "wall_seconds": 14400.0,
        "gpu_seconds": 7200.0,
        "gpu_slots": 1.0,
        "llm_output_tokens": 0.0,
    }
)

#: The protocol's frozen primary stratum and its endpoint label, as the Track A
#: baselines declare them. Constants rather than a lookup into the feedback
#: record: reading the campaign back would be indistinguishable, in code, from
#: reading its evidence.
_PRIMARY_STRATUM_ID = "CHEMBL3886|Ki|F|BAO_0000019"
_ENDPOINT_LABEL = "pKi"

#: The teacher table ``distill``'s default names. It does not exist at the start
#: of a campaign and this arm never runs the two-stage teacher pass that would
#: create it, so the draw fails with the handler's own unblock. That failure is
#: the control working; see the module docstring before "fixing" it.
_ABL2_TEACHER_TABLE = f".external/research/zak-c9r/derived/distill/{ABL2_ARM_ID}/teacher_table.json"

#: The "fixed per-action default configuration" the arm's registration names:
#: ``kind``, ``inputs``, ``expected_outputs`` and ``stop_rule`` for each of the
#: nine registered interventions.
#:
#: Every ``kind`` is in ``plan_contract()["kind_values"]`` *and* in that action's
#: ``artifact_kinds``, because ``validate_plan`` refuses a kind the action
#: cannot produce and a control whose plans are all rejected measures nothing.
#: Transcribed against ``ACTION_INPUT_CONTRACTS``, with a test that fails the
#: moment a registered action has no entry here — the same arrangement, and for
#: the same reason, as ``ACTION_INPUT_CONTRACTS`` itself.
ABL2_DEFAULT_CONFIGURATIONS: Mapping[str, Record] = MappingProxyType(
    {
        "acquire_resource": {
            "kind": "DATA_OR_LIBRARY",
            "inputs": {
                "backend": "staged_local",
                "resource_regime": "CONTROLLED",
                "resource": {
                    "identifier": "zak_c9r_chembl_activities",
                    "path": ".external/research/zak-c9r/chembl/activities.json",
                    "source": "ChEMBL_37 web-service snapshot for CHEMBL3886, staged locally",
                    "license": "CC BY-SA 3.0 (ChEMBL, EMBL-EBI)",
                    "trust_label": "OBSERVED",
                },
            },
            "expected_outputs": ["acquisition.json"],
            "stop_rule": (
                "one staging pass over the declared local file; no network is reached and none is attempted"
            ),
        },
        "fit_predictor": {
            "kind": "TRAINED_MODEL",
            # The parent package's shipped RandomForest settings, which is what
            # "fixed default" has to mean for this action: a tuned one would be
            # a second changed pointer the arm never declared.
            "inputs": {
                "backend": "sklearn_rf",
                "preparation": {
                    "stratum_id": _PRIMARY_STRATUM_ID,
                    "endpoint_label": _ENDPOINT_LABEL,
                    "seed": CAMPAIGN_SEED,
                    "n_estimators": 200,
                    "max_depth": 12,
                },
            },
            "expected_outputs": ["predictions.json"],
            "stop_rule": (
                "one fit on the admitted train partition and one prediction pass over the development pool"
            ),
        },
        "adapt_generator": {
            # The action set and this fixed default are UNCHANGED, and must
            # stay unchanged: the control's whole claim is that it offers the
            # treatment arm's menu with the choosing removed, so an entry edited
            # to reflect a capability would make this a different menu and stop
            # it being a control.
            #
            # Only the prose moved. It used to assert "answers
            # CAPABILITY_UNAVAILABLE on this host: no generator backend exposes
            # a training verb" — a hardcoded fact about the host, in a comment,
            # which would have gone on being read as true after the training
            # verb landed. What the action does is run a capability probe and
            # report it; whichever way the probe comes back, the draw is not
            # re-rolled and nothing here is special-cased.
            "kind": "WEIGHTS",
            "inputs": {"backend": "reinvent4"},
            "expected_outputs": ["capability_probe.json"],
            "stop_rule": (
                "one capability probe; the measured answer is reported as it comes back, and an "
                "unavailable verb is reported as unavailable, never retried"
            ),
        },
        "distill": {
            "kind": "SCORER",
            "inputs": {
                "backend": "sklearn_gbr",
                "teacher": {
                    "endpoint": "vina_affinity",
                    "table": _ABL2_TEACHER_TABLE,
                },
                "synthetic_weight": 1.0,
                "holdout_fraction": 0.2,
                "seed": CAMPAIGN_SEED,
            },
            "expected_outputs": ["student.joblib", "distillation.json"],
            "stop_rule": (
                "one student fit on the named teacher table; an absent table is a failure, "
                "not a licence to make one"
            ),
        },
        "change_sampling": {
            # 150 molecules is fixed. The Track B incumbent is volume-matched at
            # 150 and any other submission budget makes the scaffold count
            # incomparable with the arm it is being compared against.
            "kind": "SAMPLING_POLICY",
            "inputs": {
                "backend": "reinvent4",
                "sampling": {
                    "n_molecules": 150,
                    "seed": CAMPAIGN_SEED,
                    "batch_size": 50,
                    "max_runtime_seconds": 1800,
                },
            },
            "expected_outputs": ["molecules.json"],
            "stop_rule": (
                "one sampling pass at the fixed 150-molecule submission budget; stop at 1800 seconds"
            ),
        },
        "develop_algorithm": {
            # backend "none", not "claude_code": this arm's manifest records
            # that there is no model in this loop. Nobody writes the module the
            # command runs, so the run fails with a missing module — the honest
            # cost of drawing a worker-authored action in an arm that has no
            # worker.
            "kind": "ALGORITHM",
            "inputs": {
                "backend": "none",
                "command": [
                    "{python}",
                    "-m",
                    "drug_agent.selfevolve.target.experiments.abl2_default_algorithm",
                    "--output-dir",
                    "{output}",
                ],
                "module_to_write": "src/drug_agent/selfevolve/target/experiments/abl2_default_algorithm.py",
                "preparation": {
                    "stratum_id": _PRIMARY_STRATUM_ID,
                    "endpoint_label": _ENDPOINT_LABEL,
                    "seed": CAMPAIGN_SEED,
                },
            },
            "expected_outputs": ["predictions.json"],
            "stop_rule": (
                "one run of the declared command; no experiment worker is launched, so an "
                "unwritten module is a failure"
            ),
        },
        "develop_architecture": {
            "kind": "MODEL_ARCHITECTURE",
            "inputs": {
                "backend": "none",
                "command": [
                    "{python}",
                    "-m",
                    "drug_agent.selfevolve.target.experiments.abl2_default_architecture",
                    "--output-dir",
                    "{output}",
                ],
                "module_to_write": (
                    "src/drug_agent/selfevolve/target/experiments/abl2_default_architecture.py"
                ),
                "preparation": {
                    "stratum_id": _PRIMARY_STRATUM_ID,
                    "endpoint_label": _ENDPOINT_LABEL,
                    "seed": CAMPAIGN_SEED,
                },
            },
            "expected_outputs": ["predictions.json"],
            "stop_rule": (
                "one run of the declared command; no experiment worker is launched, so an "
                "unwritten module is a failure"
            ),
        },
        "prepare_receptor": {
            "kind": "RECEPTOR_STRUCTURE",
            "inputs": {
                "backend": "pocket_cut",
                "receptor": {
                    "pdb": ".external/research/zak-c9r/rcsb/6JUU.pdb",
                    "ligand_comp_id": "C9R",
                    "chain": "A",
                    "radius": 10.0,
                },
            },
            "expected_outputs": ["receptor_pocket.pdb"],
            "stop_rule": (
                "one pocket cut, checked against the target manifest; a pocket below the "
                "identity threshold is a failure"
            ),
        },
        "evaluate_development": {
            # There is no earlier experiment to point at from a cold start, and
            # this arm may not go looking for one: reading the history for a
            # prediction file is reading the campaign's evidence. The default
            # names the file the contract's own example names, and the handler
            # reports that it is not on disk.
            "kind": "SCORER",
            "inputs": {
                "backend": "target_benchmark",
                "predictions": "predictions.json",
                "preparation": {
                    "stratum_id": _PRIMARY_STRATUM_ID,
                    "endpoint_label": _ENDPOINT_LABEL,
                    "seed": CAMPAIGN_SEED,
                },
            },
            "expected_outputs": ["predictions.json", "endpoint.json"],
            "stop_rule": (
                "one re-scoring pass over the named prediction file; an absent file is a "
                "failure, never an empty result"
            ),
        },
    }
)


def default_configuration(action: str, *, seed: int) -> Record:
    """*action*'s fixed default configuration, with the campaign seed filled in.

    Rebuilt on every call rather than handed out by reference, because the table
    is module state shared by every seed in the study and a caller that mutated
    what it got back would silently change what later campaigns propose.

    Raises:
        MalformedProposal: the campaign menu names an action this control
            declares no default for. The arm's registered factor is
            ``FIXED_DEFAULT``; improvising a configuration here would be an
            undeclared change to the arm, so the cycle is recorded as a refused
            proposal instead.
    """
    entry = ABL2_DEFAULT_CONFIGURATIONS.get(str(action))
    if entry is None:
        raise MalformedProposal(
            f"{action!r} is on the campaign menu but {ABL2_ARM_ID} declares no fixed default "
            f"configuration for it; it declares {sorted(ABL2_DEFAULT_CONFIGURATIONS)}. This arm "
            f"is registered as FIXED_DEFAULT, so inventing one would be an undeclared change",
            zero_token_usage(),
        )
    return _with_seed(dict(entry), seed)


def zero_token_usage() -> Record:
    """The usage record of a proposal no model made.

    Zeroes, not nulls. ``None`` in this program means *telemetry was not
    available*, and this arm's proposer consumption is not unknown — it is zero,
    by construction, because nothing was called. ``ok`` is ``True`` for the same
    reason: a draw cannot fail, so reporting a failure would be as wrong as
    reporting a cost.
    """
    return {
        "backend": PROPOSER_NAME,
        "model": "",
        "output_tokens": 0,
        "input_tokens": 0,
        "cost_estimate_usd": 0.0,
        "ok": True,
        "error": None,
    }


@dataclass
class UniformRandomProposer:
    """ABL-2's proposer. It draws an action; it does not reason about one.

    Satisfies the same :class:`~.controller.Proposer` protocol as
    :class:`~.controller.ClaudeProposer` and returns the same
    ``proposal_schema`` shape, so the controller's plan assembly, admission gate
    and history are identical in both arms — which is the whole point of a
    shared-menu control. The difference is confined to *how the action is
    chosen*, and it is visible in the record: ``evidence_class`` is
    ``"random_control"`` and the diagnosis says no diagnosis was performed.

    ``diagnosis`` and ``hypothesis`` state plainly that nothing was diagnosed. A
    control that narrated a reason it did not have would be a lie in the
    campaign history, and it would be an especially expensive one — the
    narration would read exactly like the treatment arm's.
    """

    #: The campaign's own seed, from the arm's registered list (211–215). The
    #: same seed and the same cycle give the same action, in this process and in
    #: any later one, so the arm can be re-run and audited.
    seed: int
    name: str = PROPOSER_NAME
    #: Narrows the draw to this menu instead of the feedback record's, taken
    #: from the feedback when unset so the proposer and the controller cannot
    #: disagree about what may be proposed. It exists for that agreement only:
    #: setting it to the actions that happen to work would silently delete the
    #: measurement this arm was registered to make.
    allowed_actions: tuple[str, ...] | None = None

    def propose(self, feedback: Record) -> Record:
        menu = self._menu(feedback)
        cycle = int(feedback.get("cycle") or 0)
        action = menu[self._rng(cycle).randrange(len(menu))]
        configuration = default_configuration(action, seed=self.seed)
        return {
            "diagnosis": _diagnosis(action, cycle=cycle, menu=menu, seed=self.seed),
            "hypothesis": _hypothesis(action),
            "action": action,
            "kind": str(configuration["kind"]),
            "changed_factors": [CHANGED_FACTOR],
            "inputs": dict(configuration["inputs"]),
            "expected_outputs": list(configuration["expected_outputs"]),
            "requested_budget": self._budget(feedback),
            "stop_rule": str(configuration["stop_rule"]),
            "usage": zero_token_usage(),
            "evidence_class": RANDOM_CONTROL_EVIDENCE_CLASS,
        }

    # -- the draw ---------------------------------------------------------
    def _menu(self, feedback: Record) -> tuple[str, ...]:
        """The campaign's own menu, deduplicated and in a stable order.

        Sorted rather than taken in whatever order the feedback listed them,
        because "the same seed and the same cycle give the same action" has to
        survive a caller that assembled the menu differently. Deduplicated
        because uniform means uniform over the *distinct* registered actions: a
        menu naming one twice would weight it twice, and nothing in the arm's
        registration says an action may be drawn more often than another.
        """
        declared = self.allowed_actions or tuple(feedback.get("allowed_actions") or CONTROLLER_ACTIONS)
        menu = tuple(sorted({str(a) for a in declared}))
        if not menu:
            raise MalformedProposal(
                "the campaign authorises no actions, so there is nothing to draw from; an empty "
                "menu is a stopped campaign, not an unrestricted one",
                zero_token_usage(),
            )
        return menu

    def _rng(self, cycle: int) -> random.Random:
        """This cycle's generator, reproducible from ``(seed, cycle)`` alone.

        Seeded from a SHA-256 digest rather than from ``hash()`` or from the
        global ``random`` module. ``hash()`` is salted per interpreter and the
        global generator is shared with everything else that ran in the process,
        so either would let a resumed campaign — or a re-run of a recorded one —
        draw a different action from the same registered seed. An arm that
        cannot be replayed is not a control anyone can check.
        """
        material = f"{PROPOSER_NAME}:{int(self.seed)}:{int(cycle)}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))

    # -- what it may pay with ---------------------------------------------
    def _budget(self, feedback: Record) -> Record:
        """A fixed slice of whatever this campaign declared a cap for.

        The declared caps are the only thing besides the menu that this control
        reads, and they are bookkeeping rather than evidence: a plan naming a
        resource nobody capped is refused before it launches, and a refused plan
        measures nothing. Amounts are :data:`ABL2_CYCLE_BUDGET` clamped to what
        remains — clamped, never enlarged, and never sized from what is left,
        because a request computed from the campaign's state is adaptation.

        A campaign that declared no caps at all is handed the table unclamped,
        so the ledger refuses the plan for the real reason rather than this
        proposer quietly dropping the request and asking for nothing.
        """
        available = dict((feedback.get("remaining") or {}).get("resources") or {})
        if not available:
            return dict(ABL2_CYCLE_BUDGET)
        budget: Record = {}
        for resource, amount in ABL2_CYCLE_BUDGET.items():
            if resource not in available:
                continue
            cap = available[resource]
            wanted = float(amount) if cap is None else min(float(amount), float(cap))
            budget[resource] = max(0.0, wanted)
        return budget or dict(ABL2_CYCLE_BUDGET)


def _with_seed(value: Any, seed: int) -> Any:
    """Copy *value*, replacing every :data:`CAMPAIGN_SEED` with *seed*."""
    if value is CAMPAIGN_SEED:
        return int(seed)
    if isinstance(value, Mapping):
        return {str(key): _with_seed(item, seed) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_with_seed(item, seed) for item in value]
    return value


def _diagnosis(action: str, *, cycle: int, menu: Sequence[str], seed: int) -> str:
    """What this arm knows about the campaign, which is nothing."""
    return (
        f"No diagnosis was performed. This is the registered {ABL2_ARM_ID} control arm: there is "
        f"no model in this loop, nothing read the campaign's evidence, and no bottleneck was "
        f"identified. {action!r} was drawn uniformly at random from the {len(menu)} actions this "
        f"campaign authorises, by a generator seeded from campaign seed {seed} and cycle {cycle}; "
        f"every other action on the menu was exactly as likely."
    )


def _hypothesis(action: str) -> str:
    """Why a control that states no hypothesis says so instead of inventing one."""
    return (
        f"None. A uniform random draw states no hypothesis, and writing one here would put a "
        f"reason into the campaign history that nothing in this arm ever had. What this cycle "
        f"measures is what the shared menu yields when the action is chosen at random rather than "
        f"chosen well: whatever {action!r} returns under its fixed default configuration — a "
        f"development number, CAPABILITY_UNAVAILABLE, or a failure because a prerequisite no "
        f"earlier cycle produced is missing — is this arm's result and is recorded as it stands."
    )


__all__ = [
    "ABL2_ARM_ID",
    "ABL2_CYCLE_BUDGET",
    "ABL2_DEFAULT_CONFIGURATIONS",
    "CAMPAIGN_SEED",
    "CHANGED_FACTOR",
    "PROPOSER_NAME",
    "RANDOM_CONTROL_EVIDENCE_CLASS",
    "UniformRandomProposer",
    "default_configuration",
    "zero_token_usage",
]
