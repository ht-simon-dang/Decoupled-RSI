"""The benchmark: tasks, and the gold rubric no arm can see.

Design rules, each of which exists because of something in the historical record:

1. **The rubric is not the agent's own success criteria.** Five of five requests
   set criteria the deployment could not satisfy, and one refinement reached
   ``hard_constraint_pass_rate == 1.0`` by deleting every constraint. A system
   graded on goals it also sets will improve that number without improving
   anything. So promotion is scored against the fixed criteria below, which the
   candidate generator never sees.

2. **The rubric is achievable but not free.** Every threshold was calibrated by
   running a competent hand-written requirement set through the real pipeline
   (``python -m drug_agent.selfevolve.cli calibrate``) and checking that a good
   transcription clears it and a careless one does not. The calibration output
   is stored beside the benchmark so the numbers are auditable, not asserted.

3. **Calibration, recorded.** ``drugagent self-evolve calibrate`` runs five
   hand-written requirement sets through the real pipeline and prints what each
   scores. Measured 2026-08-27: a competent target transcription scores 8/8 and
   a competent phenotype one 7/7; a careless target transcription (over-tight
   property window, single objective) scores 5/8; one that loses the gene symbol
   scores 6/8; a phenotype goal mis-transcribed as target-based scores 6/8. The
   0.80 pass threshold therefore separates "did the job" from "went through the
   motions", which is exactly the discrimination the decision metric needs.

4. **The rubric prefers the independently recomputed tier.** Where RDKit can
   answer from the shortlist SMILES alone, the criterion is scored on the
   kernel's own computation rather than on the platform's scorecards.

5. **Splits are honest about what they are.** ``historical`` tasks are the real
   instructions from the request journal — the ones the analysis studied.
   ``holdout`` tasks are constructed and were never in the journal, so they test
   whether an improvement generalises past the requests that motivated it.
   Synthetic tasks are labelled synthetic everywhere they are reported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

BENCHMARK_ID = "drugagent-master-v1"
#: Bumped whenever a task or a gold threshold changes. An evaluation record
#: carries it, so records from different rubric versions never get compared.
BENCHMARK_VERSION = "1.0.0"


@dataclass(frozen=True)
class GoldCriterion:
    """One kernel-owned success condition.

    ``metric`` is a key produced by
    :func:`~drug_agent.selfevolve.evaluation.metrics.kernel_metrics`. Keys
    prefixed ``independent_`` are the kernel's own RDKit recomputation.
    """

    metric: str
    op: str
    value: float
    why: str

    def score(self, metrics: dict[str, Any]) -> tuple[bool | None, Any]:
        observed = metrics.get(self.metric)
        if observed is None:
            return None, None
        try:
            v = float(observed)
        except (TypeError, ValueError):
            return None, observed
        ok = {
            "<": v < self.value,
            "<=": v <= self.value,
            ">": v > self.value,
            ">=": v >= self.value,
            "==": abs(v - self.value) < 1e-9,
        }.get(self.op)
        return (bool(ok) if ok is not None else None), round(v, 4)


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    split: str  # historical | holdout
    origin: str  # journal | synthetic
    instruction: str
    #: What a correct transcription should produce. Used only for reporting a
    #: transcription error, never as a gold criterion — the rubric grades the
    #: portfolio, not the intermediate object.
    expected_kind: str = "target_based"
    generator: str = "library"
    gold: tuple[GoldCriterion, ...] = ()
    source_episode: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gold"] = [asdict(g) for g in self.gold]
        return d


# ---------------------------------------------------------------------------
# The rubric
# ---------------------------------------------------------------------------
def _portfolio_gold() -> tuple[GoldCriterion, ...]:
    """Conditions every good outcome must meet, whatever the goal was.

    None of these can be satisfied by weakening the agent's own criteria: they
    are properties of the molecules that came out.
    """
    return (
        GoldCriterion(
            "shortlist_filled",
            ">=",
            1.0,
            "the run produced the shortlist it was asked for; a short shortlist is a failed search",
        ),
        GoldCriterion(
            "independent_constraint_pass_rate",
            ">=",
            0.75,
            "at least three quarters of the shortlist really satisfies the brief's own hard "
            "constraints, recomputed here from SMILES with RDKit",
        ),
        GoldCriterion(
            "independent_qed_median",
            ">=",
            0.55,
            "the shortlist is drug-like on an independent QED computation",
        ),
        GoldCriterion(
            "independent_pains_free_fraction",
            ">=",
            0.75,
            "most of the shortlist trips no PAINS filter (RDKit's catalogue, not the platform's)",
        ),
        GoldCriterion(
            "independent_mean_pairwise_distance",
            ">=",
            0.30,
            "the shortlist is not one scaffold wearing hats",
        ),
        GoldCriterion(
            "n_scaffold_clusters",
            ">=",
            5.0,
            "the pool the shortlist was drawn from spanned several scaffolds",
        ),
    )


def _target_gold() -> tuple[GoldCriterion, ...]:
    """Efficacy conditions for a protein-target program.

    ``6.0`` sits below the measured ceiling of 6.4582 and above what a careless
    requirement set reaches; ``in_domain_fraction >= 0.5`` is the platform's own
    honesty check — a potency the activity model cannot speak to is not a result.
    """
    return (
        GoldCriterion("max_pIC50", ">=", 6.0, "a credible potent hit inside what this deployment can reach"),
        GoldCriterion(
            "in_domain_fraction",
            ">=",
            0.5,
            "at least half the shortlist is inside the activity model's applicability domain",
        ),
    )


def _phenotype_gold() -> tuple[GoldCriterion, ...]:
    return (
        GoldCriterion(
            "restoration_median",
            ">=",
            0.015,
            "the shortlist genuinely moves the signature toward the healthy state — the pool "
            "median is -0.005, so a positive shortlist median means the ranking actually "
            "selected restorative compounds rather than ranking on something else",
        ),
    )


_HISTORICAL: list[dict[str, Any]] = [
    {
        "task_id": "seh_basic",
        "instruction": "Find potent, drug-like sEH (EPHX2) inhibitors with no structural alerts.",
        "expected_kind": "target_based",
        "source_episode": "mrun_29e7222e10dc",
        "notes": "The plainest form of the platform's core request.",
    },
    {
        "task_id": "seh_potent_synthesizable",
        "instruction": (
            "Find selective inhibitors of soluble epoxide hydrolase (sEH / EPHX2). They should be "
            "potent (predicted pIC50 >= 7), drug-like (Lipinski-compliant, no structural alerts), "
            "and synthetically accessible."
        ),
        "expected_kind": "target_based",
        "source_episode": "mrun_333fbbc07a32",
        "notes": (
            "The user asks for pIC50 >= 7, which this deployment cannot reach. The right behaviour "
            "is to produce the best portfolio available and say so — not to chase it for two cycles."
        ),
    },
    {
        "task_id": "endothelium_phenotype",
        "instruction": (
            "Prioritise small molecules that reverse a dysfunctional vascular-endothelium phenotype "
            "toward the healthy state, favouring high restoration score, good drug-likeness, and "
            "synthetic accessibility."
        ),
        "expected_kind": "phenotype_based",
        "source_episode": "mrun_4ff72b8fd33f",
        "notes": "The only phenotype request in the history, and the one that scored worst.",
    },
]

_HOLDOUT: list[dict[str, Any]] = [
    {
        "task_id": "seh_qed_led",
        "instruction": (
            "Shortlist sEH (EPHX2) inhibitors where drug-likeness leads: I care most about clean, "
            "well-behaved, structurally diverse molecules, with potency as a secondary consideration."
        ),
        "expected_kind": "target_based",
        "notes": "Synthetic. Inverts the usual objective ordering.",
    },
    {
        "task_id": "seh_easy_to_make",
        "instruction": (
            "I need sEH (EPHX2) inhibitor candidates my chemists can actually make this quarter. "
            "Favour straightforward, accessible chemistry over maximum potency, and keep them "
            "drug-like."
        ),
        "expected_kind": "target_based",
        "notes": "Synthetic. Puts the mocked synthesis backend in the lead objective.",
    },
    {
        "task_id": "endothelial_barrier_rescue",
        "instruction": (
            "Find compounds that restore endothelial barrier function in a damaged vascular "
            "monolayer. Rank by how strongly they reverse the disease signature, and keep the "
            "shortlist chemically diverse and free of obvious liabilities."
        ),
        "expected_kind": "phenotype_based",
        "notes": "Synthetic phenotype task; never appeared in the studied journals.",
    },
]


def _build() -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for spec in _HISTORICAL:
        kind = spec["expected_kind"]
        gold = _portfolio_gold() + (_phenotype_gold() if kind == "phenotype_based" else _target_gold())
        tasks.append(
            BenchmarkTask(
                task_id=spec["task_id"],
                split="historical",
                origin="journal",
                instruction=spec["instruction"],
                expected_kind=kind,
                gold=gold,
                source_episode=spec.get("source_episode"),
                notes=spec.get("notes", ""),
            )
        )
    for spec in _HOLDOUT:
        kind = spec["expected_kind"]
        gold = _portfolio_gold() + (_phenotype_gold() if kind == "phenotype_based" else _target_gold())
        tasks.append(
            BenchmarkTask(
                task_id=spec["task_id"],
                split="holdout",
                origin="synthetic",
                instruction=spec["instruction"],
                expected_kind=kind,
                gold=gold,
                notes=spec.get("notes", ""),
            )
        )
    return tasks


_TASKS: tuple[BenchmarkTask, ...] = tuple(_build())


def load_benchmark(
    *, splits: tuple[str, ...] = ("historical", "holdout"), task_ids: list[str] | None = None
) -> list[BenchmarkTask]:
    tasks = [t for t in _TASKS if t.split in splits]
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
    return tasks


def benchmark_fingerprint() -> str:
    """Digest over every task and threshold. Recorded on every evaluation."""
    payload = json.dumps([t.as_dict() for t in _TASKS], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe() -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "version": BENCHMARK_VERSION,
        "fingerprint": benchmark_fingerprint(),
        "n_tasks": len(_TASKS),
        "splits": {
            split: [t.task_id for t in _TASKS if t.split == split]
            for split in sorted({t.split for t in _TASKS})
        },
        "tasks": [t.as_dict() for t in _TASKS],
    }


#: A run is "good" when it clears this fraction of its gold criteria. Used only
#: to judge whether the agent's own ACCEPT/REFINE verdict was right.
GOLD_PASS_THRESHOLD = 0.80

__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_VERSION",
    "GOLD_PASS_THRESHOLD",
    "BenchmarkTask",
    "GoldCriterion",
    "benchmark_fingerprint",
    "describe",
    "load_benchmark",
]
