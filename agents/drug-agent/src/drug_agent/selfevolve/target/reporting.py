"""Step 17 — the research evidence package, and the entry point that reproduces it.

This module assembles everything a researcher who has never read the agent
conversation needs in order to (a) recompute every reported number and (b) find
every scientific artifact the programme produced. It is the last implementation
step, so its job is not to produce a new result but to make the existing ones
checkable — including the ones that are absent, blocked or negative.

Five rules shape it, and each exists because its absence is a specific way to
publish something untrue.

**Nothing is filtered to winners.** Every table is built from a *denominator*
that is declared first and then filled. A campaign that crashed, an arm that
never ran, a factor with no controlled comparison and a metric nobody measured
are **rows**, not omissions. :data:`MISSING` is written into a cell whose value
was never observed and :data:`MISSING_TOKEN` is what reaches the CSV, so a
reader cannot mistake an unmeasured quantity for a zero. There is no
``dropna``, no inner join and no plotting filter anywhere in this module;
:func:`assert_no_row_was_dropped` is called on every table before it is written
and raises if a row went missing between the denominator and the output.

**Cost of acquiring an artifact is not cost of using it.** :func:`cost_table`
separates ``COLD`` (the one-off spend that produced an artifact: generation,
docking, fitting, the reasoning tokens a campaign burned) from ``WARM`` (what it
costs to run the artifact afterwards). A distilled student is cheap only after
somebody paid for the teacher, and the table says which number is which.

**Computational scores and historical experimental endpoints are labelled
apart.** :data:`ENDPOINT_EVIDENCE` maps every endpoint this programme reports to
one of :data:`COMPUTATIONAL` or :data:`EXPERIMENTAL`, and every table column and
figure axis carries the label. A Vina affinity and a human-measured Ki are not
the same kind of number and no figure in this package lets them share an
unlabelled axis.

**Protected labels stay with their custodian.** Records that carry per-compound
identities — the development-pool absolute-error tables, the split's label
classes — are exported as *addresses* (path, sha256, row count) rather than
copied. :func:`scan_for_compound_identities` refuses to write a package that
contains an InChIKey or a compound accession, and the reproduction guide
describes custodian-controlled evaluation instead of shipping the labels. The
held-out partition is not read by anything here: ``test_access_count`` is
re-verified through the project's own scanner before the package is written.

**Every claim carries its own status.** :func:`scientific_claims` is a table of
the statements this package is allowed to make, each with ``SUPPORTED``,
``REFUSED`` or ``NOT_ESTABLISHED``, the record it is measured from, and the
narrower statement that replaces a refused one. The refusals are the point: the
top-ranked arm never sees the target, the gate it cleared is a docking threshold
this programme measured as carrying no potency signal, the study is
``EXPLORATORY`` with an empty rejection region, and the confirmation branch is
``NOT_RUN``.

The two entry points are :func:`build_package` (``run_target_study.py report``)
and :func:`reproduce_metrics` / :func:`reproduce_scientific`
(``run_target_study.py reproduce``). Metrics mode recomputes the tables and the
statistics from recorded raw results in a fresh process and reruns no costly
scientific tool; scientific mode reruns the subset declared in
:data:`SCIENTIFIC_SUBSET` and reports every member it cannot run as ``NOT_RUN``
with its reason, its measured cost and what would unblock it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .confirmation import refuse_false_assurance
from .contracts import ContractError, ErrorCategory

Record = dict[str, Any]

CONTRACT = "ResearchEvidencePackage"

REPO = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO / "configs" / "research" / "zak-c9r"
DERIVED_DIR = REPO / ".external" / "research" / "zak-c9r" / "derived"

#: The sentinel for a quantity that was never observed. It is ``None`` in a
#: record and :data:`MISSING_TOKEN` in a CSV, and it is never 0.
MISSING: None = None

#: What a cell says when nothing measured it and the answer is not a number. A
#: WORD, because the alternative this package shipped was the word ``AVAILABLE``
#: applied by default to eight interventions nothing probed.
NOT_MEASURED = "NOT_MEASURED"

#: What a missing cell looks like in an exported table. Deliberately a word and
#: not an empty string: an empty cell in a spreadsheet reads as zero to a
#: surprising number of people, and this one has to survive being pasted.
MISSING_TOKEN = "MISSING"

#: The study this package reports, checked rather than trusted.
REGISTERED_PROTOCOL_VERSION = "zak-c9r-2"
REGISTERED_PROTOCOL_HASH = "sha256:f960f2c8635ba3053dadbdc05a138d89b4ce6b8ef91456cf0fcaa82c7656366b"
REGISTERED_EVALUATOR_HASH = "sha256:695cc35698188f64e73d27a77f5c5dd51b7b960d95cb49f39a505a5ecaab0058"


class EvidenceType(StrEnum):
    """How a number was produced. Never inferred from the column name."""

    #: A program computed it. Docking scores, ADMET-AI endpoints, SA, QED, and
    #: every fitted predictor's output.
    COMPUTATIONAL = "COMPUTATIONAL_SCORE"
    #: A human being measured it in a laboratory, before this programme existed,
    #: and it reached here through a public database.
    EXPERIMENTAL = "HISTORICAL_EXPERIMENTAL_ENDPOINT"
    #: Seconds, tokens, calls. Not a scientific endpoint at all.
    RESOURCE = "RESOURCE_ACCOUNTING"


COMPUTATIONAL = EvidenceType.COMPUTATIONAL
EXPERIMENTAL = EvidenceType.EXPERIMENTAL

#: Every endpoint this programme reports, and what kind of evidence it is. A
#: figure axis or a table column built from an endpoint absent from this map is
#: refused by :func:`evidence_type`, because an unlabelled axis is how a docking
#: score ends up being read as a potency measurement.
ENDPOINT_EVIDENCE: dict[str, EvidenceType] = {
    # computational scores
    "vina_affinity": COMPUTATIONAL,
    "minimizedAffinity": COMPUTATIONAL,
    "CNNscore": COMPUTATIONAL,
    "CNNaffinity": COMPUTATIONAL,
    "herg": COMPUTATIONAL,
    "sa_score": COMPUTATIONAL,
    "qed": COMPUTATIONAL,
    "mw": COMPUTATIONAL,
    "logp": COMPUTATIONAL,
    "qualifying_scaffolds": COMPUTATIONAL,
    "scaffolds_per_molecule_submitted": COMPUTATIONAL,
    "intramolecular_validity": COMPUTATIONAL,
    "pic50_chemprop": COMPUTATIONAL,
    "pic50_affine_docking_map": COMPUTATIONAL,
    "pic50_parent_rf": COMPUTATIONAL,
    "mae_pactivity": COMPUTATIONAL,
    "mae_vs_teacher": COMPUTATIONAL,
    "training_proximity": COMPUTATIONAL,
    # historical experimental endpoints — measured by people, not by this study
    "pactivity_label": EXPERIMENTAL,
    "pKi": EXPERIMENTAL,
    "Ki": EXPERIMENTAL,
    "Kd": EXPERIMENTAL,
    "IC50": EXPERIMENTAL,
    "ED50": EXPERIMENTAL,
    "known_active_pass_rate": EXPERIMENTAL,
}

#: What a standard's verdict says when the arms' medians differ and nothing tested
#: whether the difference is one. The four standards below used to carry the
#: hard-coded string ``"SEPARATED"`` — no test, no interval, no n, and none of the
#: six preconditions this same package applied to the pIC50 standard before
#: failing it. That asymmetry inflated the flagship claim: one standard was held
#: to six measured preconditions and four were held to a literal. The literal is
#: gone and the observed spread is what is reported.
SPREAD_ONLY = "SPREAD_OBSERVED_NO_SEPARATION_TEST"

#: Why a descriptive spread is all four of these standards can carry here, stated
#: in every row so the verdict cannot be quoted without it.
_WHY_NO_SEPARATION_TEST = (
    "the nine arms' medians differ and this table reports that spread. Nothing tested whether the "
    "difference is one: no interval and no test was computed on any of these medians, and they are "
    "pooled over MOLECULES (150 per arm across 3 seeds) while the lock's registered replication "
    "unit is the generation SEED, at which the exact test's rejection region is empty. Of the six "
    "preconditions this package applied to the pIC50 standard before failing it, four cannot even "
    "be asked of a direct computational score - there is no fitted estimator, so there is no "
    "'better than a train-mean constant', no model error to compare a spread against, no training "
    "label support and no independent second model - and the two that could be asked, stability "
    "across seeds and spread against the instrument's error, have no measurement here because "
    "per-seed medians and an error bar were never computed. SPREAD, NOT SEPARATION"
)

#: The five standards this step assembled. ``pic50`` is in the table with
#: ``FAILED`` beside it rather than dropped, because a failed standard is the most
#: informative row in the benchmark.
#:
#: **These were not declared in advance.** No registered record in this programme
#: declares a set of benchmark standards: this tuple was assembled in step 17's
#: own module, after every result was seen. It is a post-hoc selection of
#: endpoints the sweep happened to have collected, and the ``denominator`` on
#: table-11 says so rather than implying a prior declaration.
BENCHMARK_STANDARDS: tuple[Record, ...] = (
    {
        "standard": "herg",
        "metric": "herg",
        "direction": "LOWER_IS_BETTER",
        "verdict": SPREAD_ONLY,
        "reported_per_arm": True,
        "why": _WHY_NO_SEPARATION_TEST,
    },
    {
        "standard": "sa",
        "metric": "sa_score",
        "direction": "LOWER_IS_BETTER",
        "verdict": SPREAD_ONLY,
        "reported_per_arm": True,
        "why": _WHY_NO_SEPARATION_TEST,
    },
    {
        "standard": "qed",
        "metric": "qed",
        "direction": "HIGHER_IS_BETTER",
        "verdict": SPREAD_ONLY,
        "reported_per_arm": True,
        "why": _WHY_NO_SEPARATION_TEST,
    },
    {
        "standard": "docking",
        "metric": "vina_affinity",
        "direction": "LOWER_IS_BETTER",
        "verdict": SPREAD_ONLY,
        "reported_per_arm": True,
        "why": _WHY_NO_SEPARATION_TEST,
    },
    {
        "reported_per_arm": False,
        "standard": "pic50",
        "metric": "pic50_chemprop",
        "direction": "HIGHER_IS_BETTER",
        "verdict": "FAILED",
        "why": (
            "three independent estimators were fitted and none ranks the nine methods at a "
            "resolution its own error permits: the TRAIN-fitted affine docking map is a monotone "
            "image of the Vina column, Chemprop is measurably worse than a train-mean constant "
            "(+0.02436 MAE, Wilcoxon p = 0.013) and its apparent 1.07 log-unit spread is the "
            "applicability-domain ordering (Spearman +0.95 against training proximity), and the "
            "parent RandomForest ties the constant. No pIC50 gate was added"
        ),
    },
)

#: The subset of scientific work ``reproduce --mode scientific`` is declared to
#: rerun, and the members it cannot. Declared as data so the command reports the
#: same list whether or not it can run any of it, and so a later worker with
#: docking approval can see exactly what changes.
SCIENTIFIC_SUBSET: tuple[Record, ...] = (
    {
        "id": "evaluator-kernel-digest",
        "what": "re-measure the frozen evaluator's kernel digest from the guarded files",
        "tool": "drug_agent.selfevolve.evaluation.integrity.kernel_digest",
        "runnable_here": True,
        "cost": "< 1 s, CPU",
    },
    {
        "id": "qualification-rule-replay",
        "what": (
            "re-execute the frozen seven-gate qualification rule over the stored per-molecule "
            "gate verdicts and recount qualifying scaffolds through the evaluator's own function"
        ),
        "tool": "drug_agent.selfevolve.evaluation.target_benchmark.qualifying_scaffolds",
        "runnable_here": True,
        "cost": "< 5 s, CPU, reads stored rows only",
    },
    {
        "id": "generation-sweep",
        "what": "re-sample 1,350 molecules from nine generators at seeds 301/302/303",
        "tool": "run_target_study.py sweep",
        "runnable_here": False,
        "reason": "REGENERATION_REQUIRED",
        "cost": "3,867.4 GPU-occupancy seconds of generation, measured",
        "unblock": "operator approval to regenerate, plus the nine generator environments",
    },
    {
        "id": "docking-and-endpoint-pass",
        "what": "re-dock and re-score 1,350 generated molecules against the frozen receptor",
        "tool": "run_target_study.py sweep (scoring phase)",
        "runnable_here": False,
        "reason": "REDOCKING_REQUIRED",
        "cost": "20,044.8 docking wall seconds plus 43.6 endpoint wall seconds, measured",
        "unblock": "operator approval to re-dock",
    },
    {
        "id": "posebusters-pass",
        "what": "re-run PoseBusters over 1,296 stored poses",
        "tool": "PoseBustersAdapter",
        "runnable_here": False,
        "reason": "ADAPTER_DEFECT_OPEN",
        "cost": "15,562.6 wall seconds, measured",
        "unblock": (
            "blocker B-15D-1 — the adapter counts every boolean full_report column as a check and "
            "one has inverted polarity, so its pass/fail verdict is wrong in dock mode for every "
            "caller. The decomposed per-check numbers in this package were recomputed around it"
        ),
    },
    {
        "id": "track-a-campaigns",
        "what": "run the five locked confirmation campaigns of B5-self-evolution",
        "tool": "run_target_study.py confirm",
        "runnable_here": False,
        "reason": "BRANCH_NOT_RUN",
        "cost": "681,319 LLM output tokens beyond an authorisation already 6,319 overdrawn",
        "unblock": "an explicit operator budget authorisation, and a held-out execution path that "
        "does not exist in this programme",
    },
    {
        "id": "confirmation-on-the-held-out-partition",
        "what": "evaluate a nominee against the held-out partition",
        "tool": "run_target_study.py confirm",
        "runnable_here": False,
        "reason": "NO_NOMINEE_AND_NO_EXECUTION_PATH",
        "cost": "test_access_count moves 0 to 1, once, irreversibly",
        "unblock": (
            "a nominee from an executed B5-self-evolution arm, an authorised budget, and a "
            "separately reviewed execution path added in front of the existing gate"
        ),
    },
)

#: Third-party code and data this study depends on, with how it is obtained and
#: under what terms. Restricted downloads are named as restricted; an adapter
#: that could not run for want of a licensed checkpoint is a row here and an
#: ``UNAVAILABLE`` row in the capability table, never a silent absence.
THIRD_PARTY: tuple[Record, ...] = (
    {
        "name": "ChEMBL",
        "kind": "DATA",
        "what_for": "the target's historical experimental activity records",
        "access": "public REST snapshot, captured once and content-addressed",
        "licence": "CC BY-SA 3.0 (ChEMBL data)",
        "restricted": False,
    },
    {
        "name": "BindingDB",
        "kind": "DATA",
        "what_for": "second activity source, deduplicated against ChEMBL",
        "access": "public download",
        "licence": "CC BY 3.0 US",
        "restricted": False,
    },
    {
        "name": "RCSB PDB 6JUU",
        "kind": "STRUCTURE",
        "what_for": "the receptor the C9R pocket is cut from",
        "access": "public download",
        "licence": "CC0 1.0",
        "restricted": False,
    },
    {
        "name": "UniProt Q9NYL2",
        "kind": "SEQUENCE",
        "what_for": "target identity and the sequence check",
        "access": "public download",
        "licence": "CC BY 4.0",
        "restricted": False,
    },
    {
        "name": "AutoDock Vina",
        "kind": "TOOL",
        "what_for": "the structure score every Track B gate is built on",
        "access": "source build, pinned",
        "licence": "Apache-2.0",
        "restricted": False,
    },
    {
        "name": "RDKit",
        "kind": "TOOL",
        "what_for": "standardisation, descriptors, scaffolds, SA score",
        "access": "conda-forge",
        "licence": "BSD-3-Clause",
        "restricted": False,
    },
    {
        "name": "ADMET-AI",
        "kind": "MODEL",
        "what_for": "the hERG endpoint used by the safety gate",
        "access": "public weights",
        "licence": "MIT (code); see upstream for weights",
        "restricted": False,
    },
    {
        "name": "PoseBusters",
        "kind": "TOOL",
        "what_for": "generated-pose validity checks",
        "access": "PyPI",
        "licence": "BSD-3-Clause",
        "restricted": False,
        "note": "blocker B-15D-1 is open against this platform's adapter, not against PoseBusters",
    },
    {
        "name": "the nine generator checkouts",
        "kind": "CODE",
        "what_for": "the benchmark arms",
        "access": "git clones pinned in scripts/external/CHECKOUTS.tsv; code only",
        "licence": "per upstream project; each clone carries its own",
        "restricted": False,
    },
    {
        "name": "generative checkpoints for the generator arms",
        "kind": "WEIGHTS",
        "what_for": "the sampling weights the nine arms load",
        "access": "RESTRICTED — multi-GB files behind Zenodo/Google Drive links, several requiring "
        "acceptance of an upstream licence before download. Not fetched by any script in this "
        "repository; each adapter's unavailable_reason names the exact file and where upstream "
        "says to get it",
        "licence": "per upstream project, varying; acceptance required for several",
        "restricted": True,
    },
    {
        "name": "Boltz-2",
        "kind": "MODEL",
        "what_for": "not used — excluded by standing decision",
        "access": "checkout present; NOT RUN in this study",
        "licence": "per upstream project",
        "restricted": False,
        "note": "its contamination audit is INCONCLUSIVE with the PubChem ingestion route open, so "
        "every 'no predictor separates these methods' statement is PROVEN for what could be run "
        "and UNTESTED for Boltz-2",
    },
)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _fail(category: ErrorCategory, message: str, *, field_name: str | None = None) -> ContractError:
    return ContractError(category, message, contract=CONTRACT, field=field_name)


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """The content address of a file, or ``ABSENT`` when it is not there.

    Never raises for a missing file: a package that refuses to describe a
    missing artifact cannot report that the artifact is missing.
    """
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return "ABSENT"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


#: The records the package is assembled from. ``key -> (relative path, where)``
#: where ``where`` is ``"config"`` for a committed record, ``"derived"`` for a
#: runtime record under the gitignored store, ``"docs"`` for a card.
RECORD_SOURCES: dict[str, tuple[str, str]] = {
    "protocol": ("protocol.json", "config"),
    "qualification": ("qualification.json", "config"),
    "calibration": ("calibration.json", "config"),
    "baseline_results": ("baseline-results.json", "config"),
    "generation_sweep": ("generation-sweep.json", "config"),
    "generator_availability": ("generator-availability.json", "config"),
    "pic50_estimate": ("pic50-independent-estimate.json", "config"),
    "ablation_status": ("ablation-status.json", "config"),
    "confirmation_status": ("confirmation-status.json", "config"),
    "comparator_lineage": ("comparator-lineage.json", "config"),
    "nomination_anchor": ("nomination-anchor.json", "config"),
    "protocol_lock": ("../protocol.lock.json", "derived"),
    "data_card": ("data_card.json", "derived"),
    "split_manifest": ("split_manifest.json", "derived"),
    "target_manifest": ("target_manifest.json", "derived"),
    "artifact_index": ("artifact_index.json", "derived"),
    "sweep_report": ("sweep/run-01/report.json", "derived"),
    "sweep_ledger": ("sweep/run-01/ledger.json", "derived"),
    "distill_ablation": ("distill/distill-step12-a/ablation/ablation.json", "derived"),
    "confirmation_record": ("confirm/step15/confirmation.json", "derived"),
    "environment": ("baseline-environment.json", "docs"),
}


@dataclass(frozen=True)
class StudyBundle:
    """Every record the package is built from, with where each one came from.

    A record that is absent is ``None`` here and an ``ABSENT`` row in
    :attr:`sources`. Nothing in this module treats an absent record as an empty
    one: a table built from a missing source reports ``SOURCE_ABSENT`` for its
    rows rather than producing zero rows, because zero rows and no source look
    identical downstream and mean opposite things.
    """

    study: Path
    configs: Path
    docs: Path
    repo: Path
    records: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Record] = field(default_factory=dict)
    campaigns: list[Record] = field(default_factory=list)
    campaign_ledgers: dict[str, Record] = field(default_factory=dict)
    sweep_cells: dict[str, Record] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        return self.records.get(key)

    def require(self, key: str) -> Record:
        value = self.records.get(key)
        if not isinstance(value, Mapping):
            raise _fail(
                ErrorCategory.MISSING_FIELD,
                f"the package needs record {key!r} and it is absent at "
                f"{self.sources.get(key, {}).get('path', '<unknown>')}",
                field_name=key,
            )
        return dict(value)

    def present(self, key: str) -> bool:
        return self.records.get(key) is not None


def load_study(
    study: Path | str,
    *,
    configs: Path | str | None = None,
    docs: Path | str | None = None,
    repo: Path | str | None = None,
) -> StudyBundle:
    """Read every record the package needs, recording what is missing.

    *study* is the derived-record root (``.external/research/zak-c9r/derived``
    on this host). The committed records come from *configs* and the cards from
    *docs*; both default to this checkout, because a package built from one
    checkout's code and another's protocol is not a package of either.
    """
    study_dir = Path(study)
    config_dir = Path(configs) if configs is not None else CONFIG_DIR
    docs_dir = Path(docs) if docs is not None else REPO / "docs" / "research"
    repo_dir = Path(repo) if repo is not None else REPO

    roots = {"config": config_dir, "derived": study_dir, "docs": docs_dir}
    records: dict[str, Any] = {}
    sources: dict[str, Record] = {}
    for key, (relative, where) in RECORD_SOURCES.items():
        path = (roots[where] / relative).resolve()
        payload = _read_json(path)
        records[key] = payload
        sources[key] = {
            "key": key,
            "path": _display_path(path, repo_dir),
            "where": where,
            "present": payload is not None,
            "sha256": sha256_file(path),
        }

    campaigns: list[Record] = []
    ledgers: dict[str, Record] = {}
    campaign_root = study_dir / "campaigns"
    if campaign_root.is_dir():
        for directory in sorted(p for p in campaign_root.iterdir() if p.is_dir()):
            record = _read_json(directory / "campaign.json")
            if not isinstance(record, Mapping):
                continue
            entry = dict(record)
            entry["_directory"] = _display_path(directory, repo_dir)
            entry["_record_sha256"] = sha256_file(directory / "campaign.json")
            campaigns.append(entry)
            ledger = _read_json(directory / "ledger.json")
            if isinstance(ledger, Mapping):
                ledgers[str(entry.get("campaign_id"))] = dict(ledger)

    cells: dict[str, Record] = {}
    cell_root = study_dir / "sweep" / "run-01" / "cells"
    if cell_root.is_dir():
        for directory in sorted(p for p in cell_root.iterdir() if p.is_dir()):
            arm_record = _read_json(directory / "arm.json")
            if not isinstance(arm_record, Mapping):
                continue
            cells[directory.name] = {
                "arm": dict(arm_record),
                "directory": _display_path(directory, repo_dir),
                "seed_dirs": sorted(p.name for p in directory.glob("seed-*") if p.is_dir()),
            }

    return StudyBundle(
        study=study_dir,
        configs=config_dir,
        docs=docs_dir,
        repo=repo_dir,
        records=records,
        sources=sources,
        campaigns=campaigns,
        campaign_ledgers=ledgers,
        sweep_cells=cells,
    )


def _display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def load_campaigns(directory: Path | str) -> list[Record]:
    """Every ``campaign.json`` under *directory*, terminal or not.

    Used by the reporter and by the synthetic-ledger test, which is the point:
    the ledger the test builds goes through the same loader as the real one, so
    a loader that quietly skipped a failed campaign would fail that test.
    """
    root = Path(directory)
    out: list[Record] = []
    if not root.is_dir():
        return out
    for candidate in sorted(root.rglob("campaign.json")):
        record = _read_json(candidate)
        if isinstance(record, Mapping):
            entry = dict(record)
            entry["_directory"] = str(candidate.parent)
            out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #


def evidence_type(metric: str) -> EvidenceType:
    """The evidence class of *metric*, refusing an endpoint nobody declared."""
    try:
        return ENDPOINT_EVIDENCE[metric]
    except KeyError:
        raise _fail(
            ErrorCategory.INVALID_ENUM,
            f"endpoint {metric!r} has no declared evidence type. A column whose evidence class "
            f"is unknown may not be reported: that is how a computed score is read as a "
            f"laboratory measurement. Add it to ENDPOINT_EVIDENCE with its class",
            field_name="metric",
        ) from None


def make_table(
    table_id: str,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator: str,
    notes: Sequence[str] = (),
    evidence: Mapping[str, str] | None = None,
) -> Record:
    """One table, with its denominator stated beside it.

    ``denominator`` is prose naming *what the rows are drawn from* — "all four
    campaigns that have a terminal record", "all nine admitted generator arms" —
    so a reader can tell a complete table from a filtered one without counting.
    """
    return {
        "id": table_id,
        "title": title,
        "columns": list(columns),
        "rows": [dict(row) for row in rows],
        "n_rows": len(rows),
        "denominator": denominator,
        "notes": list(notes),
        "evidence_by_column": dict(evidence or {}),
    }


def assert_no_row_was_dropped(table: Mapping[str, Any], expected: int) -> None:
    """Refuse a table whose row count disagrees with its own denominator.

    The failure this guards against is not a crash; it is a table that looks
    fine and is missing the arm that failed. Called before every write.
    """
    actual = int(table.get("n_rows") or 0)
    if actual != expected:
        raise _fail(
            ErrorCategory.INVALID_VALUE,
            f"table {table.get('id')!r} declares a denominator of {expected} rows and carries "
            f"{actual}. A row that vanished between the denominator and the output is the "
            f"failure this check exists for; filtering to the rows with results is not allowed",
            field_name="n_rows",
        )


def _cell(value: Any) -> Any:
    return MISSING_TOKEN if value is None else value


def write_table_csv(table: Mapping[str, Any], path: Path) -> Record:
    """Write one table as CSV, with ``MISSING`` for every unobserved cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(table["columns"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in table["rows"]:
            writer.writerow([_cell(row.get(column)) for column in columns])
    return {
        "table_id": table["id"],
        "path": path.name,
        "n_rows": len(table["rows"]),
        "sha256": sha256_file(path),
    }


def data_composition_table(bundle: StudyBundle) -> Record:
    """Rows, molecules, families and partitions, with the held-out side withheld."""
    card = bundle.get("data_card")
    split = bundle.get("split_manifest")
    rows: list[Record] = []
    if not isinstance(card, Mapping) or not isinstance(split, Mapping):
        rows.append(
            {
                "scope": "ALL",
                "n_rows": MISSING,
                "n_molecules": MISSING,
                "n_families": MISSING,
                "status": "SOURCE_ABSENT",
            }
        )
        return make_table(
            "table-01-data-composition",
            "Data composition",
            ["scope", "n_rows", "n_molecules", "n_families", "n_documents", "status", "note"],
            rows,
            denominator="the one curated dataset; the source record is absent on this host",
        )

    rows.append(
        {
            "scope": "curated (all sources, after standardisation)",
            "n_rows": card.get("n_rows"),
            "n_molecules": card.get("n_independent_molecules"),
            "n_families": card.get("n_families"),
            "n_documents": MISSING,
            "status": "MEASURED",
            "note": f"{card.get('n_strata')} assay strata; {card.get('n_excluded')} rows excluded (table-02)",
        }
    )
    counts = dict(split.get("counts") or {})
    composition = dict(split.get("composition") or {})
    for partition in ("train", "development", "test"):
        side = dict(counts.get(partition) or {})
        comp = dict(composition.get(partition) or {})
        weights = dict(side.get("weights") or {})
        rows.append(
            {
                "scope": f"partition: {partition}",
                "n_rows": comp.get("n_rows"),
                "n_molecules": side.get("n_molecules"),
                "n_families": side.get("n_families"),
                "n_documents": comp.get("n_documents"),
                "status": "WITHHELD_UNTIL_RELEASE" if partition == "test" else "MEASURED",
                "note": (
                    f"{int(weights.get('n_exact_primary') or 0)} exact-valued rows in the primary "
                    f"stratum"
                    + (
                        "; label composition is not printed while the boundary is closed, and no "
                        "compound of this partition was read by this package"
                        if partition == "test"
                        else ""
                    )
                ),
            }
        )
    return make_table(
        "table-01-data-composition",
        "Data composition and partitions",
        ["scope", "n_rows", "n_molecules", "n_families", "n_documents", "status", "note"],
        rows,
        denominator="the curated dataset plus all three registered partitions, none omitted",
        notes=[
            "Every activity value here is a HISTORICAL_EXPERIMENTAL_ENDPOINT: it was measured by "
            "people before this programme existed and reached this study through public databases.",
            "The held-out partition's sizes are structural and come from the registered split "
            "manifest. Its labels were not read: test_access_count is 0.",
        ],
        evidence={"n_rows": EXPERIMENTAL.value, "n_molecules": EXPERIMENTAL.value},
    )


def exclusion_table(bundle: StudyBundle) -> Record:
    """Every exclusion reason with its count — the complete 535, not a sample."""
    card = bundle.get("data_card")
    rows: list[Record] = []
    if not isinstance(card, Mapping):
        return make_table(
            "table-02-exclusions",
            "Exclusions",
            ["reason", "n_rows", "share_of_excluded", "status"],
            [{"reason": "ALL", "n_rows": MISSING, "share_of_excluded": MISSING, "status": "SOURCE_ABSENT"}],
            denominator="every exclusion reason; the source record is absent on this host",
        )
    summary = dict(card.get("exclusion_summary") or {})
    total = int(card.get("n_excluded") or 0)
    for reason in sorted(summary):
        count = int(summary[reason])
        rows.append(
            {
                "reason": reason,
                "n_rows": count,
                "share_of_excluded": round(count / total, 6) if total else MISSING,
                "status": "MEASURED",
            }
        )
    rows.append(
        {
            "reason": "TOTAL",
            "n_rows": total,
            "share_of_excluded": 1.0 if total else MISSING,
            "status": "MEASURED",
        }
    )
    table = make_table(
        "table-02-exclusions",
        "Exclusions from the curated dataset",
        ["reason", "n_rows", "share_of_excluded", "status"],
        rows,
        denominator=f"every distinct exclusion reason recorded by the data card, plus a total row "
        f"({total} excluded of {card.get('n_rows')} curated)",
        notes=[
            "No row was excluded after any result was seen. The exclusion rules are the "
            "standardiser's, applied at ingestion.",
        ],
    )
    assert_no_row_was_dropped(table, len(summary) + 1)
    return table


def baseline_capability_table(bundle: StudyBundle) -> Record:
    """Every baseline arm and every probed capability verb, available or not."""
    results = bundle.get("baseline_results")
    rows: list[Record] = []
    n_expected = 0
    if isinstance(results, Mapping):
        arms = list(results.get("arms") or [])
        n_expected += len(arms)
        for arm in arms:
            rows.append(
                {
                    "id": arm.get("baseline_id"),
                    "kind": "BASELINE_ARM",
                    "track": arm.get("track"),
                    "status": arm.get("status"),
                    "metric": arm.get("primary_metric"),
                    "value": arm.get("primary_value"),
                    "evidence_type": (
                        evidence_type(str(arm.get("primary_metric"))).value
                        if arm.get("primary_metric") in ENDPOINT_EVIDENCE
                        else MISSING
                    ),
                    "n_scored": arm.get("n_scored"),
                    "reason_or_unblock": arm.get("reason") or arm.get("unblock"),
                }
            )
        probes = list(results.get("capability_probes") or [])
        n_expected += len(probes)
        for probe in probes:
            rows.append(
                {
                    "id": f"{probe.get('capability_id')}::{probe.get('verb', 'fit')}",
                    "kind": "CAPABILITY_PROBE",
                    "track": MISSING,
                    "status": probe.get("status") or "UNAVAILABLE",
                    "metric": MISSING,
                    "value": MISSING,
                    "evidence_type": MISSING,
                    "n_scored": MISSING,
                    "reason_or_unblock": probe.get("reason") or probe.get("unblock"),
                }
            )
    availability = bundle.get("generator_availability")
    if isinstance(availability, Mapping):
        summary = list(availability.get("summary") or [])
        n_expected += len(summary)
        for entry in summary:
            rows.append(
                {
                    "id": entry.get("capability_id") or entry.get("arm"),
                    "kind": "GENERATOR",
                    "track": "B",
                    "status": entry.get("status"),
                    "metric": MISSING,
                    "value": MISSING,
                    "evidence_type": MISSING,
                    "n_scored": entry.get("n_unique") if "n_unique" in entry else MISSING,
                    "reason_or_unblock": entry.get("reason") or entry.get("unblock"),
                }
            )
    if not rows:
        rows.append({"id": "ALL", "kind": "SOURCE_ABSENT", "status": "SOURCE_ABSENT"})
        n_expected = 1
    table = make_table(
        "table-03-baseline-capabilities",
        "Baseline arms and measured capabilities",
        [
            "id",
            "kind",
            "track",
            "status",
            "metric",
            "value",
            "evidence_type",
            "n_scored",
            "reason_or_unblock",
        ],
        rows,
        denominator="every registered baseline arm, every probed capability verb and every "
        "surveyed generator — including the ones that are UNAVAILABLE or EXECUTED_NO_YIELD",
        notes=[
            "A capability is probed one verb at a time: a working sampler does not establish a "
            "working trainer, and the two are separate rows.",
            "EXECUTED_NO_YIELD is a measured yield of zero from a working backend, and it is a "
            "different finding from UNAVAILABLE. The status tokens keep them apart.",
        ],
    )
    assert_no_row_was_dropped(table, n_expected)
    return table


def artifact_type_table(bundle: StudyBundle) -> Record:
    """The ten registrable artifact kinds, and how many this programme produced.

    Nine of the ten are zero. That is the table: the registry supports ten kinds
    and this study produced two, and a table showing only the two would read as
    a claim about coverage that nothing here supports.
    """
    kinds = (
        "ALGORITHM",
        "MODEL_ARCHITECTURE",
        "WEIGHTS",
        "TRAINED_MODEL",
        "SCORER",
        "SAMPLING_POLICY",
        "RECEPTOR_STRUCTURE",
        "RECEPTOR_ENSEMBLE",
        "DATA_OR_LIBRARY",
        "COMPOSITE_PACKAGE",
    )
    produced: dict[str, list[str]] = {kind: [] for kind in kinds}
    for campaign in bundle.campaigns:
        for candidate in campaign.get("candidate_pool") or []:
            artifact_id = candidate.get("artifact_id")
            if not artifact_id:
                continue
            kind = str(candidate.get("artifact_kind") or "ALGORITHM")
            produced.setdefault(kind, []).append(str(artifact_id))
    distill = bundle.get("distill_ablation")
    if isinstance(distill, Mapping):
        experiment = str(distill.get("experiment_id") or "distill")
        for arm in distill.get("arms") or []:
            identifier = arm.get("artifact_id") or arm.get("arm_id")
            if not identifier:
                # The ablation record names its arms by synthetic weight rather
                # than by an artifact id. Deriving one keeps the fitted students
                # in the table; an arm with no card on disk is still not counted.
                weight = arm.get("synthetic_weight")
                if arm.get("card_path") and weight is not None:
                    identifier = f"{experiment}:student_w{weight}"
            if identifier:
                produced.setdefault("SCORER", []).append(str(identifier))
    rows = [
        {
            "artifact_kind": kind,
            "n_registered": len(produced.get(kind) or []),
            "artifact_ids": "; ".join(sorted(set(produced.get(kind) or []))) or MISSING,
            "benchmark_accepted": 0,
            "note": "registry support is plumbing; a registered kind is not evidence that the kind helped",
        }
        for kind in kinds
    ]
    table = make_table(
        "table-04-artifact-types",
        "Artifact kinds the registry supports, and what this study produced",
        ["artifact_kind", "n_registered", "artifact_ids", "benchmark_accepted", "note"],
        rows,
        denominator="all ten registrable artifact kinds, including the ones with zero artifacts",
        notes=[
            "benchmark_accepted is 0 for every kind: acceptance needs a comparison on the "
            "registered endpoint and the confirmation branch is NOT_RUN.",
        ],
    )
    assert_no_row_was_dropped(table, len(kinds))
    return table


#: The comparator each Track A arm is measured against, and the one arm this
#: study registered as a "conventional adaptation" control.
STANDARD_ADAPTATION_ARM = "A-B2-standard-adaptation-rf"
PARENT_ARM = "A-B0-parent-rf"


def comparator_identity_check(bundle: StudyBundle) -> Record:
    """Whether a registered comparator is the SAME MODEL as the baseline it controls for.

    The finding this exists to surface, and which the package previously rendered
    as an ordinary inconclusive point difference:
    ``A-B2-standard-adaptation-rf`` — the "conventional adaptation" control — has
    a per-compound absolute-error vector that is **bit-identical** to
    ``A-B0-parent-rf``'s across all 34 evaluated compounds. Its declared 3x3 grid
    search, selected by family-grouped 5-fold cross-validation on the TRAIN
    partition, chose the parent's own shipped configuration, so the fitted model
    IS the parent. A comparator that is bit-identical to its own baseline is not a
    comparator, and a reader has to be told that rather than shown a delta of
    0.000 with "INCONCLUSIVE" beside it.

    The identity is checked three ways, strongest last, and every check reports
    only counts, addresses and booleans. The per-compound error tables are keyed
    by compound identity: they are compared by CONTENT ADDRESS and neither their
    keys nor their values leave this function.
    """
    results = bundle.get("baseline_results")
    arms: dict[str, Mapping[str, Any]] = {}
    if isinstance(results, Mapping):
        arms = {
            str(a.get("baseline_id")): a
            for a in (results.get("arms") or [])
            if isinstance(a, Mapping)
        }
    parent = arms.get(PARENT_ARM)
    control = arms.get(STANDARD_ADAPTATION_ARM)
    if parent is None or control is None:
        return {
            "status": "SOURCE_ABSENT",
            "arm": STANDARD_ADAPTATION_ARM,
            "baseline": PARENT_ARM,
            "note": "baseline-results.json does not carry both arms; no identity claim is made",
        }

    same_value = (
        isinstance(parent.get("primary_value"), (int, float))
        and parent.get("primary_value") == control.get("primary_value")
        and parent.get("primary_metric") == control.get("primary_metric")
    )
    parent_model = dict(parent.get("model") or {})
    control_model = dict(control.get("model") or {})
    selected = dict(control_model.get("selected_config") or {})
    same_configuration = bool(selected) and all(
        parent_model.get(key) == value for key, value in selected.items()
    )
    same_seed = parent_model.get("seed") == control_model.get("seed")

    # The strongest check, and the only one that is about the PREDICTIONS rather
    # than about an aggregate: the per-compound error vectors, compared by
    # content address. Absent when the custodian's store is not on this host.
    digests: dict[str, str] = {}
    n_errors: dict[str, int] = {}
    for name, arm in ((PARENT_ARM, parent), (STANDARD_ADAPTATION_ARM, control)):
        path = arm.get("result_path")
        if not path:
            continue
        payload = _read_json(bundle.repo / str(path))
        if not isinstance(payload, Mapping):
            continue
        errors = dict((payload.get("endpoint") or {}).get("absolute_errors") or {})
        if not errors:
            continue
        digests[name] = sha256_bytes(canonical_json(errors).encode("utf-8"))
        n_errors[name] = len(errors)
    vectors_identical: bool | None = None
    if len(digests) == 2:
        vectors_identical = digests[PARENT_ARM] == digests[STANDARD_ADAPTATION_ARM]

    is_the_baseline = bool(same_value and same_configuration) or vectors_identical is True
    return {
        "status": "CHECKED",
        "arm": STANDARD_ADAPTATION_ARM,
        "baseline": PARENT_ARM,
        "role_it_was_registered_as": "STANDARD_ADAPTATION control",
        "same_primary_value": bool(same_value),
        "selected_configuration_is_the_parents": same_configuration,
        "selected_configuration": selected,
        "same_seed": bool(same_seed),
        "per_compound_error_vector_identical": vectors_identical,
        "per_compound_error_vector_addresses": digests or MISSING,
        "n_evaluated_compounds": n_errors.get(PARENT_ARM, MISSING),
        "is_the_baseline_it_controls_for": is_the_baseline,
        "finding": (
            f"{STANDARD_ADAPTATION_ARM} IS {PARENT_ARM}. Its declared 3x3 grid, selected by "
            f"family-grouped 5-fold cross-validation on the TRAIN partition, chose the parent's "
            f"own shipped configuration, and its per-compound absolute-error vector is "
            f"BIT-IDENTICAL to the parent's across every evaluated compound. A comparator that is "
            f"bit-identical to its own baseline is not a comparator: this study has NO "
            f"conventional-adaptation control, and the 0.000 difference between them is an "
            f"identity rather than a result."
        )
        if is_the_baseline
        else (
            f"{STANDARD_ADAPTATION_ARM} differs from {PARENT_ARM} on at least one of the three "
            f"checks, so it is a distinct fitted model."
        ),
        "what_it_cannot_support": (
            "any statement of the form 'self-evolution is no better than conventional adaptation'. "
            "There was no conventional adaptation to be no better than"
        ),
        "privacy": "counts, content addresses and booleans only; no compound identity and no "
        "per-compound value leaves this check",
    }


def comparison_table(bundle: StudyBundle) -> Record:
    """Parent, standard-adaptation and external comparators, with their identity.

    The comparator a number is measured against is part of the number. Every row
    names its comparator explicitly and says which partition it was measured on.
    """
    results = bundle.get("baseline_results")
    identity = comparator_identity_check(bundle)
    rows: list[Record] = []
    expected = 0
    if isinstance(results, Mapping):
        arms = [a for a in (results.get("arms") or []) if isinstance(a, Mapping)]
        parent = next((a for a in arms if a.get("baseline_id") == PARENT_ARM), None)
        parent_value = parent.get("primary_value") if isinstance(parent, Mapping) else None
        parent_metric = parent.get("primary_metric") if isinstance(parent, Mapping) else None
        expected += len(arms)
        for arm in arms:
            value = arm.get("primary_value")
            metric = arm.get("primary_metric")
            role = {
                PARENT_ARM: "PARENT",
                STANDARD_ADAPTATION_ARM: "STANDARD_ADAPTATION",
            }.get(str(arm.get("baseline_id")), "EXTERNAL_COMPARATOR")
            # The one arm that is not a comparator at all. Checked rather than
            # assumed: see `comparator_identity_check`.
            is_the_parent = (
                str(arm.get("baseline_id")) == STANDARD_ADAPTATION_ARM
                and identity.get("is_the_baseline_it_controls_for") is True
            )
            if is_the_parent:
                role = "NOT A COMPARATOR — IS THE PARENT"
            # A delta is only defined between two measurements of the SAME
            # endpoint. Subtracting a Track A mean absolute error from a Track B
            # scaffold count produces a number, and that number means nothing;
            # the row says so instead of printing it.
            delta: float | None = None
            if role == "PARENT":
                verdict = "IS THE COMPARATOR — every delta in this table is measured against it"
            elif metric is None or value is None:
                verdict = (
                    f"NO RESULT — this arm is {arm.get('status')}; "
                    f"{arm.get('reason') or arm.get('unblock') or 'no reason recorded'}"
                )
            elif metric != parent_metric:
                verdict = (
                    f"NOT COMPARABLE TO THE PARENT — this arm's endpoint is {metric!r} and the "
                    f"parent's is {parent_metric!r}. A cross-endpoint difference is not a delta"
                )
            elif is_the_parent:
                delta = (
                    float(value) - float(parent_value)
                    if isinstance(value, (int, float)) and isinstance(parent_value, (int, float))
                    else None
                )
                vectors = identity.get("per_compound_error_vector_identical")
                n_compounds = identity.get("n_evaluated_compounds")
                verdict = (
                    "NOT A COMPARATOR — this arm IS the parent. Its declared 3x3 grid, selected "
                    "by family-grouped 5-fold CV on the TRAIN partition, chose the parent's own "
                    "shipped configuration"
                    + (
                        f", and its per-compound absolute-error vector is BIT-IDENTICAL to the "
                        f"parent's across all {n_compounds} evaluated compounds"
                        if vectors is True
                        else " (the per-compound error tables are not on this host, so only the "
                        "aggregate and the configuration were compared)"
                    )
                    + ". The 0.0 difference is an identity, not a result: this study has NO "
                    "conventional-adaptation control"
                )
            elif isinstance(value, (int, float)) and isinstance(parent_value, (int, float)):
                delta = float(value) - float(parent_value)
                verdict = "INCONCLUSIVE — a point difference with no interval"
            else:
                verdict = "NO RESULT — the value is not a number"
            rows.append(
                {
                    "id": arm.get("baseline_id"),
                    "role": role,
                    "comparator": PARENT_ARM if role != "PARENT" else "(is the comparator)",
                    "partition": arm.get("evaluated_partition") or results.get("evaluated_partition"),
                    "metric": metric,
                    "evidence_type": (
                        evidence_type(str(metric)).value if metric in ENDPOINT_EVIDENCE else MISSING
                    ),
                    "value": arm.get("primary_value"),
                    "delta_vs_parent": delta,
                    "interval": MISSING,
                    "status": arm.get("status"),
                    "verdict": verdict,
                }
            )
    trivial = _trivial_constant_mae(bundle)
    rows.append(
        {
            "id": "trivial-median-constant",
            "role": "TRIVIAL_CONTROL",
            "comparator": "A-B0-parent-rf",
            "partition": "development",
            "metric": "mae_pactivity",
            "evidence_type": COMPUTATIONAL.value,
            "value": trivial,
            "delta_vs_parent": MISSING,
            "interval": MISSING,
            "status": "MEASURED" if trivial is not None else "SOURCE_ABSENT",
            "verdict": "REFERENCE — every Track A arm in this study is worse than this",
        }
    )
    expected += 1
    lineage = bundle.get("comparator_lineage")
    if isinstance(lineage, Mapping):
        entries = [e for e in (lineage.get("entries") or []) if isinstance(e, Mapping)]
        expected += len(entries)
        for entry in entries:
            superseding = dict(entry.get("superseding") or {})
            superseded = dict(entry.get("superseded") or {})
            identifier = entry.get("baseline_id") or entry.get("role")
            comparator = superseded.get("baseline_ids") or superseded.get("baseline_id")
            rows.append(
                {
                    "id": f"Track B comparator bar: {identifier}",
                    "role": "TRACK_B_COMPARATOR_LINEAGE",
                    "comparator": (
                        "; ".join(comparator) if isinstance(comparator, list) else comparator or MISSING
                    ),
                    "partition": "development",
                    "metric": "qualifying_scaffolds",
                    "evidence_type": COMPUTATIONAL.value,
                    "value": superseding.get("rate_per_molecule", MISSING),
                    "delta_vs_parent": MISSING,
                    "interval": superseding.get("interval_95_exact_binomial", MISSING),
                    "status": entry.get("status") or entry.get("shape") or "REGISTERED",
                    "verdict": (
                        str(entry.get("shape") or "CARRIED_FORWARD")
                        + " — the superseded bar was RETIRED, not reproduced"
                        + (
                            "; and it is POOLED_ACROSS_CONDITIONING, with every qualifying "
                            "scaffold coming from the arm that never sees the target"
                            if superseding.get("label") == "POOLED_ACROSS_CONDITIONING"
                            else ""
                        )
                    ),
                }
            )
    table = make_table(
        "table-05-parent-and-external-comparisons",
        "Comparisons against the parent, standard adaptation and external comparators",
        [
            "id",
            "role",
            "comparator",
            "partition",
            "metric",
            "evidence_type",
            "value",
            "delta_vs_parent",
            "interval",
            "status",
            "verdict",
        ],
        rows,
        denominator="every registered baseline arm including the UNAVAILABLE one, the trivial "
        "median-constant control, and every registered Track B comparator lineage entry",
        notes=[
            "Every value here is development-partition. No comparison in this study was made on "
            "the held-out partition.",
            "interval is MISSING for every baseline row: no baseline arm was run with "
            "replicates, so a point difference has no uncertainty estimate and is INCONCLUSIVE "
            "whatever its sign.",
            "delta_vs_parent is MISSING wherever the arm's endpoint is not the parent's. A Track "
            "B qualifying-scaffold count minus a Track A mean absolute error is arithmetic, not a "
            "comparison, and this table refuses to print it.",
            "The trivial median-constant control is the row to read the Track A arms against: "
            "every one of them is worse than predicting one number for everything.",
            str(identity.get("finding") or identity.get("note") or "")
            + (
                " This study therefore has NO conventional-adaptation comparator, and no "
                "statement of the form 'X is no better than conventional adaptation' can be made "
                "from these records."
                if identity.get("is_the_baseline_it_controls_for") is True
                else ""
            ),
        ],
    )
    assert_no_row_was_dropped(table, expected)
    return table


def _trivial_constant_mae(bundle: StudyBundle) -> float | None:
    calibration = bundle.get("calibration")
    if not isinstance(calibration, Mapping):
        return None
    track_a = dict(calibration.get("track_a") or {})
    reference = dict(track_a.get("failure_penalty_reference") or {})
    value = reference.get("development_median_pactivity_absolute_deviation")
    return float(value) if isinstance(value, (int, float)) else None


#: Every terminal campaign state the contract declares, so a state with no
#: campaigns is a zero row rather than an absent one.
CAMPAIGN_STATES: tuple[str, ...] = ("RUNNING", "BLOCKED", "NOMINATED", "COMPLETED", "FAILED")

#: Every scientific decision the contract declares, plus the absent case.
CAMPAIGN_DECISIONS: tuple[str, ...] = (
    "IMPROVED",
    "NO_IMPROVEMENT",
    "INCONCLUSIVE",
    "INSUFFICIENT_DATA",
    "NONE_RECORDED",
)


def campaign_outcome_table(campaigns: Sequence[Mapping[str, Any]]) -> Record:
    """One row per campaign, terminal state and all, plus a rate row per state.

    ``NO_IMPROVEMENT``, ``BLOCKED`` and ``FAILED`` campaigns are rows here for
    the same reason the locked protocol refuses to drop them from its
    denominator: the process delivering nothing is a scientific outcome with a
    value, and a table of the campaigns that nominated something measures the
    selection, not the method.
    """
    rows: list[Record] = []
    for campaign in campaigns:
        consumed = dict(campaign.get("consumed") or {})
        measured = dict(consumed.get("measured") or {})
        unknown = dict(consumed.get("unknown_telemetry_attempts") or {})
        nomination = campaign.get("nomination")
        blocked = campaign.get("blocked_cause")
        rows.append(
            {
                "campaign_id": campaign.get("campaign_id"),
                "state": campaign.get("state"),
                "terminal_reason": campaign.get("terminal_reason"),
                "scientific_decision": campaign.get("scientific_decision") or "NONE_RECORDED",
                "cycles_completed": campaign.get("cycles_completed"),
                "n_candidates": len(campaign.get("candidate_pool") or []),
                "nominated": bool(nomination),
                "llm_output_tokens": measured.get("llm_output_tokens"),
                "llm_output_tokens_unknown_attempts": unknown.get("llm_output_tokens", 0),
                "development_delta_vs_incumbent": campaign_development_delta(campaign),
                "wall_seconds": campaign.get("wall_s"),
                "exit_code": campaign.get("exit_code"),
                "blocked_cause": (blocked or {}).get("code") if isinstance(blocked, Mapping) else MISSING,
            }
        )
    n_campaigns = len(rows)
    by_state = {state: sum(1 for r in rows if r["state"] == state) for state in CAMPAIGN_STATES}
    by_state["OTHER"] = n_campaigns - sum(by_state.values())
    by_decision = {
        decision: sum(1 for r in rows if r["scientific_decision"] == decision)
        for decision in CAMPAIGN_DECISIONS
    }
    table = make_table(
        "table-06-campaign-outcomes",
        "Campaign outcomes — every campaign with a terminal record",
        [
            "campaign_id",
            "state",
            "terminal_reason",
            "scientific_decision",
            "cycles_completed",
            "n_candidates",
            "nominated",
            "llm_output_tokens",
            "llm_output_tokens_unknown_attempts",
            "development_delta_vs_incumbent",
            "wall_seconds",
            "exit_code",
            "blocked_cause",
        ],
        rows,
        denominator=f"all {n_campaigns} campaigns with a terminal record, none dropped: "
        + ", ".join(f"{state}={count}" for state, count in by_state.items() if count),
        notes=[
            "Success rate is reported as a fraction of ALL campaigns. A campaign that crashed or "
            "nominated nothing stays in the denominator.",
            "llm_output_tokens is MISSING, not 0, where the telemetry attempt returned nothing: "
            "an unknown amount was never observed and is not summed.",
            "development_delta_vs_incumbent is MISSING for a campaign that selected nothing. The "
            "locked estimator assigns delta_i = 0.0 to a no-nominee campaign as a statement about "
            "the PROCESS delivering nothing; that is not a measured metric value and this table "
            "does not print it as one.",
        ],
    )
    table["summary"] = {
        "n_campaigns": n_campaigns,
        "by_state": by_state,
        "by_scientific_decision": by_decision,
        "n_nominated": sum(1 for r in rows if r["nominated"]),
        "nomination_rate": (
            round(sum(1 for r in rows if r["nominated"]) / n_campaigns, 6) if n_campaigns else MISSING
        ),
        "n_failed": by_state.get("FAILED", 0),
        "failure_rate": round(by_state.get("FAILED", 0) / n_campaigns, 6) if n_campaigns else MISSING,
        "n_with_unknown_token_telemetry": sum(1 for r in rows if r["llm_output_tokens_unknown_attempts"]),
    }
    assert_no_row_was_dropped(table, n_campaigns)
    return table


def cost_table(bundle: StudyBundle) -> Record:
    """Cold (acquire/train) against warm (deploy), resource by resource."""
    rows: list[Record] = []
    sweep = bundle.get("generation_sweep")
    if isinstance(sweep, Mapping):
        measured = dict((sweep.get("cost") or {}).get("measured") or {})
        rows.append(
            {
                "item": "nine-arm generation sweep — sampling",
                "phase": "COLD",
                "what_it_buys": "1,350 unique molecules, 9 arms x 3 seeds x 50",
                "gpu_occupancy_seconds": measured.get("generation_gpu_seconds"),
                "wall_seconds": measured.get("generation_wall_seconds"),
                "llm_output_tokens": 0,
                "telemetry": measured.get("generation_gpu_telemetry"),
            }
        )
        rows.append(
            {
                "item": "nine-arm generation sweep — docking",
                "phase": "COLD",
                "what_it_buys": "one AutoDock Vina score per unique molecule",
                "gpu_occupancy_seconds": measured.get("docking_gpu_occupancy_seconds"),
                "wall_seconds": measured.get("scoring_docking_wall_seconds"),
                "llm_output_tokens": 0,
                "telemetry": measured.get("docking_gpu_occupancy_telemetry"),
            }
        )
        rows.append(
            {
                "item": "nine-arm generation sweep — endpoint panel",
                "phase": "COLD",
                "what_it_buys": "hERG and SA for every scored molecule",
                "gpu_occupancy_seconds": measured.get("scoring_gpu_seconds"),
                "wall_seconds": measured.get("scoring_endpoint_wall_seconds"),
                "llm_output_tokens": 0,
                "telemetry": measured.get("scoring_gpu_telemetry"),
            }
        )
        rows.append(
            {
                "item": "nine-arm generation sweep — PoseBusters",
                "phase": "COLD",
                "what_it_buys": "per-pose validity checks",
                "gpu_occupancy_seconds": 0,
                "wall_seconds": measured.get("scoring_posebusters_wall_seconds"),
                "llm_output_tokens": 0,
                "telemetry": "MEASURED",
            }
        )
    distill = bundle.get("distill_ablation")
    if isinstance(distill, Mapping):
        teacher = dict(distill.get("teacher_cost") or {})
        rows.append(
            {
                "item": "distillation teacher pass",
                "phase": "COLD",
                "what_it_buys": f"{teacher.get('n_teacher_calls')} teacher docking labels the "
                f"student is fitted on",
                "gpu_occupancy_seconds": MISSING,
                "wall_seconds": teacher.get("teacher_wall_seconds"),
                "llm_output_tokens": 0,
                "telemetry": "MEASURED",
            }
        )
        rows.append(
            {
                "item": "distilled student — later use",
                "phase": "WARM",
                "what_it_buys": "one imitation score per molecule, no docking",
                "gpu_occupancy_seconds": 0,
                "wall_seconds": MISSING,
                "llm_output_tokens": 0,
                "telemetry": "NOT_MEASURED",
            }
        )
    for campaign in bundle.campaigns:
        measured_cost = dict((campaign.get("consumed") or {}).get("measured") or {})
        rows.append(
            {
                "item": f"campaign {campaign.get('campaign_id')}",
                "phase": "COLD",
                "what_it_buys": (
                    "one nominated artifact"
                    if campaign.get("nomination")
                    else f"no artifact — {campaign.get('terminal_reason')}"
                ),
                "gpu_occupancy_seconds": MISSING,
                "wall_seconds": measured_cost.get("wall_s"),
                "llm_output_tokens": measured_cost.get("llm_output_tokens"),
                "telemetry": "MEASURED",
            }
        )
    rows.append(
        {
            "item": "parent RandomForest — later use",
            "phase": "WARM",
            "what_it_buys": "one pActivity prediction per molecule",
            "gpu_occupancy_seconds": 0,
            "wall_seconds": 2.265,
            "llm_output_tokens": 0,
            "telemetry": "MEASURED",
        }
    )
    rows.append(
        {
            "item": "dock-augmented RandomForest — later use",
            "phase": "WARM",
            "what_it_buys": "one pActivity prediction per molecule, plus one docking call each",
            "gpu_occupancy_seconds": MISSING,
            "wall_seconds": MISSING,
            "llm_output_tokens": 0,
            "telemetry": "NOT_MEASURED",
            "note": "its deployment cost is a docking call per molecule and was never separately "
            "measured; the 845.2 s in its own record is the COLD fit over 136 compounds",
        }
    )
    table = make_table(
        "table-07-costs-cold-and-warm",
        "Cost of acquiring an artifact against cost of using it",
        [
            "item",
            "phase",
            "what_it_buys",
            "gpu_occupancy_seconds",
            "wall_seconds",
            "llm_output_tokens",
            "telemetry",
            "note",
        ],
        rows,
        denominator="every phase of the sweep, every campaign including the failed ones, and "
        "every artifact whose later use has a cost — measured or explicitly not measured",
        notes=[
            "gpu_seconds throughout this study is OCCUPANCY: seconds during which the job held a "
            "CUDA context, sampled from nvidia-smi. It is not GPU utilisation and it is not GPU "
            "work. AutoDock Vina does no GPU computation, so the docking phase's occupancy is an "
            "idle ADMET-AI context resident in the same process.",
            "COLD is what it cost to produce the artifact. WARM is what it costs to run it "
            "afterwards. A distilled student is cheap only after somebody paid for the teacher.",
            "NOT_MEASURED is not zero. Two WARM rows say so rather than carrying a made-up number.",
            "A campaign's wall_seconds here is the LEDGER-CHARGED measured wall. Table-06 reports "
            "the same campaign's end-to-end wall, which is larger, because setup and teardown are "
            "not charged against a reserved resource. The two numbers answer different questions "
            "and neither is a correction of the other.",
        ],
        evidence={"gpu_occupancy_seconds": EvidenceType.RESOURCE.value},
    )
    assert_no_row_was_dropped(table, len(rows))
    return table


def uncertainty_table(bundle: StudyBundle) -> Record:
    """Every reported statistic, its interval, and the ones that have none."""
    rows: list[Record] = []
    sweep = bundle.get("generation_sweep")
    lock = bundle.get("protocol_lock")
    power: Mapping[str, Any] = {}
    if isinstance(lock, Mapping):
        protocol = dict(lock.get("protocol") or {})
        power = dict((protocol.get("statistics") or {}).get("power_generation_quality") or {})
    attain = dict(power.get("attainability_at_the_locked_replication_unit") or {})

    rows.append(
        {
            "quantity": "primary endpoint, qualifying_scaffolds per arm",
            "estimate": "per-arm counts, table-10",
            "interval": MISSING,
            "test": "Poisson upper tail against the measured baseline rate, Holm-Bonferroni across "
            "nine arms at family-wise alpha 0.05",
            "n_replicates": 3,
            "replication_unit": "generation seed",
            "status": "REGISTERED_BEFORE_ANY_ARM_RAN",
            "what_it_cannot_support": "a confirmed ranking. At n=3 seeds the exact paired sign "
            "test's smallest attainable two-sided p is "
            f"{attain.get('paired_sign_test_min_two_sided_p', 0.25)}, so the rejection region is "
            "EMPTY and power is 0.0 at every effect size",
        }
    )
    rows.append(
        {
            "quantity": "seeds needed before alpha=0.05 is attainable (paired sign test)",
            "estimate": attain.get("n_seeds_for_alpha_paired_sign_test"),
            "interval": MISSING,
            "test": "exact test discrete support, SINGLE-COMPARISON",
            "n_replicates": MISSING,
            "replication_unit": "generation seed",
            "status": "UNCORRECTED_FOR_MULTIPLICITY — blocker B-15E-1",
            "what_it_cannot_support": "provisioning a nine-arm family. Under the Holm correction "
            "the same lock registers, 2^(1-n) must reach alpha/9 = 0.005555..., which needs 9 "
            "seeds, not 6. Provisioning at the lock's figure buys two-thirds of the replicates "
            "the registered correction requires",
        }
    )
    rows.append(
        {
            "quantity": "seeds per arm needed before alpha=0.05 is attainable (two-sample permutation)",
            "estimate": attain.get("n_seeds_for_alpha_two_sample_permutation"),
            "interval": MISSING,
            "test": "exact test discrete support, SINGLE-COMPARISON",
            "n_replicates": MISSING,
            "replication_unit": "generation seed",
            "status": "UNCORRECTED_FOR_MULTIPLICITY — blocker B-15E-1",
            "what_it_cannot_support": "the same. 2/C(2n,n) reaches 0.005555... at n=6 per arm, "
            "not at the lock's 4",
        }
    )
    if isinstance(sweep, Mapping):
        primary = dict(sweep.get("primary") or {})
        evidence = dict(power.get("evidence") or {})
        observed = evidence.get("qualifying_scaffolds_observed")
        molecules = evidence.get("molecules_observed")
        widening = evidence.get("gate_widening_factor")
        rows.append(
            {
                # NOT "MEASURED", and NOT "exact binomial". The figure is
                # (1/48) x 0.5 and the interval is the Clopper-Pearson interval
                # for 1 of 48 with BOTH endpoints multiplied by 0.5. Scaling an
                # exact interval is not an exact interval for the scaled rate, and
                # nothing observed 48 replicates of the quantity this row reports:
                # the 48 molecules were counted under the NARROWER five-gate set.
                # This repo already caught and corrected exactly this mislabel once.
                "quantity": "baseline qualifying rate per molecule (derived, gate-widened)",
                "estimate": evidence.get("baseline_rate_per_molecule"),
                "interval": power.get("baseline_rate_interval_per_molecule"),
                "test": (
                    f"NONE on this quantity. The point estimate is "
                    f"({observed}/{molecules}) x {widening}, and the interval is the "
                    f"Clopper-Pearson exact-binomial interval for {observed} of {molecules} with "
                    f"BOTH endpoints multiplied by {widening}. A scaled exact interval is not an "
                    f"exact interval for the scaled rate and carries no stated coverage"
                ),
                "n_replicates": MISSING,
                "replication_unit": (
                    f"none for this quantity. {molecules} molecules were observed under the "
                    f"NARROWER five-gate set, not under the rate this row reports"
                ),
                "status": "DERIVED_FROM_A_MEASURED_COUNT",
                "what_it_cannot_support": "any precise expected count, and not a coverage claim "
                "either. The interval spans more than two orders of magnitude, every derived "
                "count moves across it by more than the effect the design is asked to resolve, "
                f"and the whole figure rests on {observed} observed qualifying scaffold(s) scaled "
                f"by a factor of {widening} derived from a second count of six",
            }
        )
        rows.append(
            {
                "quantity": "expected qualifying scaffolds per arm at the locked budget",
                "estimate": primary.get("baseline_lambda_per_arm"),
                "interval": power.get("expected_count_per_arm_interval"),
                "test": "derived from the baseline rate interval",
                "n_replicates": 9,
                "replication_unit": "arm",
                "status": "DERIVED",
                "what_it_cannot_support": "a claim that an arm's count is surprising, when the "
                "interval on the expectation is [0.04, 8.30]",
            }
        )
    distill = bundle.get("distill_ablation")
    if isinstance(distill, Mapping):
        for entry in distill.get("comparison_against_no_distillation") or []:
            rows.append(
                {
                    "quantity": f"distillation delta MAE vs teacher at synthetic_weight="
                    f"{entry.get('synthetic_weight')} (single fit)",
                    "estimate": entry.get("delta_mae_vs_control"),
                    "interval": entry.get("paired_interval"),
                    "test": "none on this record",
                    "n_replicates": 1,
                    "replication_unit": "fit",
                    "status": "INCONCLUSIVE",
                    "what_it_cannot_support": "a direction. One fit per weight on one holdout has "
                    "no uncertainty estimate, whatever its sign",
                }
            )
    ablation = bundle.get("ablation_status")
    if isinstance(ablation, Mapping):
        for entry in ablation.get("executed") or []:
            rows.append(
                {
                    "quantity": f"{entry.get('comparison_id')} ({entry.get('factor')})",
                    "estimate": entry.get("delta"),
                    "interval": entry.get("interval"),
                    "test": entry.get("test") or "see the ablation record",
                    "n_replicates": entry.get("n_replicates"),
                    "replication_unit": entry.get("replication_unit") or "see the ablation record",
                    "status": "DEVELOPMENT_STAGE",
                    "what_it_cannot_support": entry.get("what_it_cannot_separate")
                    or "a confirmatory attribution; no comparison in this study is confirmatory",
                }
            )
    table = make_table(
        "table-08-uncertainty",
        "Uncertainty, and the statistics that have none",
        [
            "quantity",
            "estimate",
            "interval",
            "test",
            "n_replicates",
            "replication_unit",
            "status",
            "what_it_cannot_support",
        ],
        rows,
        denominator="every statistic this package reports, whether or not it has an interval",
        notes=[
            "A row with interval MISSING has no uncertainty estimate at all. It is not a row with "
            "a narrow interval.",
            "The study is EXPLORATORY. Nothing in this table is a confirmed ranking.",
        ],
    )
    assert_no_row_was_dropped(table, len(rows))
    return table


#: The nine interventions the controller can propose. Named here so a menu item
#: nobody used is a zero row rather than an absent one.
INTERVENTIONS: tuple[str, ...] = (
    "acquire_resource",
    "fit_predictor",
    "adapt_generator",
    "distill",
    "change_sampling",
    "develop_algorithm",
    "develop_architecture",
    "prepare_receptor",
    "evaluate_development",
)


def intervention_table(campaigns: Sequence[Mapping[str, Any]], bundle: StudyBundle | None = None) -> Record:
    """The nine-item menu, with how often each was proposed and what came of it."""
    proposed: dict[str, int] = dict.fromkeys(INTERVENTIONS, 0)
    succeeded: dict[str, int] = dict.fromkeys(INTERVENTIONS, 0)
    failed: dict[str, int] = dict.fromkeys(INTERVENTIONS, 0)
    unresolved: dict[str, int] = dict.fromkeys(INTERVENTIONS, 0)
    for campaign in campaigns:
        for candidate in campaign.get("candidate_pool") or []:
            action = str(candidate.get("action") or "")
            if action not in proposed:
                proposed[action] = 0
                succeeded[action] = 0
                failed[action] = 0
                unresolved[action] = 0
            proposed[action] += 1
            status = candidate.get("status")
            if status == "SUCCEEDED":
                succeeded[action] += 1
            elif status == "FAILED":
                failed[action] += 1
            else:
                unresolved[action] += 1
    # Local, not module-level: `baselines` is not otherwise a dependency of the
    # reporting package and a report build should not drag it in unless it is
    # actually reading a probe. `UNUSABLE_STATUSES` rather than the literal
    # "UNAVAILABLE" because EXECUTED_NO_YIELD is also a refusal.
    from .baselines import UNUSABLE_STATUSES

    availability: dict[str, str] = {}
    if bundle is not None:
        results = bundle.get("baseline_results")
        if isinstance(results, Mapping):
            # The VERB matters, and this loop used to ignore it. `adapt_generator`
            # is the *training* action, so only a `fit` probe can speak for it —
            # but the filter was `capability_id.startswith("molecule.generate")`,
            # which also matched the two `sample` probes. The committed baseline
            # record holds four of them — pfm/fit, pfm/sample, reinvent4/fit,
            # reinvent4/sample — and both `sample` rows read AVAILABLE_EXECUTED,
            # so last-one-wins left this row reporting a verb the action does not
            # use. A working sampler is not a working trainer, and that wrong
            # answer is what the unconditional overwrite below was papering over.
            # Filtering on the verb removes the wrong answer instead of hiding
            # it behind a constant.
            training = [
                probe
                for probe in results.get("capability_probes") or []
                if str(probe.get("capability_id") or "").startswith("molecule.generate")
                and str(probe.get("verb") or "") == "fit"
            ]
            if training:
                statuses = {str(probe.get("status") or NOT_MEASURED) for probe in training}
                usable = sorted(statuses - set(UNUSABLE_STATUSES) - {NOT_MEASURED})
                if usable:
                    availability["adapt_generator"] = usable[0]
                elif statuses <= set(UNUSABLE_STATUSES):
                    availability["adapt_generator"] = sorted(statuses)[0]
    rows = [
        {
            "intervention": name,
            "n_proposed": proposed.get(name, 0),
            "n_succeeded": succeeded.get(name, 0),
            "n_failed": failed.get(name, 0),
            "n_without_a_terminal_status": unresolved.get(name, 0),
            # NOT_MEASURED, not "AVAILABLE". The default used to be the word
            # AVAILABLE for eight of the nine interventions, none of which was
            # probed — and it contradicted this package's own open-blockers
            # table, where B-ACQUISITION records that there is no working
            # ingestion path on this host at all. An unmeasured default that
            # reads as a positive result is the worst kind of MISSING.
            "availability_on_this_host": availability.get(name, NOT_MEASURED),
        }
        for name in sorted(proposed)
    ]
    rows_by_name = {row["intervention"]: row for row in rows}
    # The `adapt_generator` overwrite that stood here is GONE. It read
    #
    #     rows_by_name["adapt_generator"]["availability_on_this_host"] = "UNAVAILABLE"
    #
    # unconditionally, with the comment "measured: the molecule.generate
    # capability probe reported it" — and it ran whether or not any probe had
    # been read, whether or not the bundle carried one, and whatever the probe
    # actually said. The row was therefore a constant wearing a measurement's
    # label, and it would have gone on printing UNAVAILABLE after a training
    # verb landed. What decides the row now is the verb-filtered probe above,
    # and a bundle carrying no `fit` probe leaves the row NOT_MEASURED, which is
    # the honest answer and is not readable as AVAILABLE.
    if "acquire_resource" in rows_by_name:
        # recorded: open blocker B-ACQUISITION, "no working ingestion path on this host"
        rows_by_name["acquire_resource"]["availability_on_this_host"] = "UNAVAILABLE"
    table = make_table(
        "table-09-interventions",
        "The intervention menu, and what the loop actually did with it",
        [
            "intervention",
            "n_proposed",
            "n_succeeded",
            "n_failed",
            "n_without_a_terminal_status",
            "availability_on_this_host",
        ],
        rows,
        denominator=f"all {len(INTERVENTIONS)} registered interventions, including the ones never proposed",
        notes=[
            "availability_on_this_host is NOT_MEASURED unless something measured it, and this "
            "column used to default every row to AVAILABLE. adapt_generator now reports whatever "
            "the molecule.generate 'fit' probes in this bundle's baseline record say, aggregated "
            "across backends; it is NOT_MEASURED when the bundle carries no such probe. It is no "
            "longer pinned to UNAVAILABLE by the table itself, which is what it used to be: the "
            "row was overwritten with the literal string on every build, so it would have gone "
            "on saying UNAVAILABLE after a training verb appeared. acquire_resource is "
            "UNAVAILABLE because open blocker B-ACQUISITION records that there is no working "
            "ingestion path on this host. Nothing probed the other seven.",
            "A proposal for an unavailable backend is accepted, dispatched and answered with a "
            "CAPABILITY_UNAVAILABLE failure carrying its unblock. It is never refused at the door "
            "and never a stack trace, so an UNAVAILABLE row can still carry proposals.",
            "n_without_a_terminal_status counts candidates a campaign never reached, typically "
            "because it crashed or ran out of budget first. They are not failures and not "
            "successes.",
        ],
    )
    assert_no_row_was_dropped(table, len(rows))
    return table


def benchmark_primary_table(bundle: StudyBundle) -> Record:
    """The registered primary endpoint, per arm, with Holm and the caveats."""
    sweep = bundle.get("generation_sweep")
    rows: list[Record] = []
    if not isinstance(sweep, Mapping):
        return make_table(
            "table-10-benchmark-primary",
            "Registered primary endpoint",
            ["arm", "status"],
            [{"arm": "ALL", "status": "SOURCE_ABSENT"}],
            denominator="all nine admitted arms; the source record is absent on this host",
        )
    primary = dict(sweep.get("primary") or {})
    strata = dict(primary.get("strata") or {})
    holm = {row["name"]: row for row in (primary.get("holm") or {}).get("rows") or []}
    for conditioning in sorted(strata):
        for arm in strata[conditioning]:
            name = str(arm.get("arm"))
            holm_row = holm.get(name, {})
            submitted = arm.get("n_submitted") or 0
            scaffolds = arm.get("qualifying_scaffolds")
            rows.append(
                {
                    "arm": name,
                    "conditioning": conditioning,
                    "sees_the_target": conditioning != "none",
                    "n_submitted": arm.get("n_submitted"),
                    "n_evaluated": arm.get("n_evaluated"),
                    "qualifying_scaffolds": scaffolds,
                    "per_seed": "/".join(str(v) for v in arm.get("per_seed_qualifying_scaffolds") or []),
                    "rate": (
                        round(float(scaffolds) / float(submitted), 6)
                        if isinstance(scaffolds, (int, float)) and submitted
                        else MISSING
                    ),
                    "p_value": holm_row.get("p_value"),
                    "holm_threshold": holm_row.get("holm_threshold"),
                    "rejects_null_under_holm": holm_row.get("rejected"),
                    "meets_absolute_threshold": arm.get("meets_absolute_threshold"),
                    "evidence_type": COMPUTATIONAL.value,
                }
            )
    rows.sort(key=lambda r: (-(r["qualifying_scaffolds"] or 0), str(r["arm"])))
    table = make_table(
        "table-10-benchmark-primary",
        "Registered primary endpoint — qualifying scaffolds per arm",
        [
            "arm",
            "conditioning",
            "sees_the_target",
            "n_submitted",
            "n_evaluated",
            "qualifying_scaffolds",
            "per_seed",
            "rate",
            "p_value",
            "holm_threshold",
            "rejects_null_under_holm",
            "meets_absolute_threshold",
            "evidence_type",
        ],
        rows,
        denominator="all nine admitted generator arms, including the five that qualified zero scaffolds",
        notes=[
            "Every column here is a COMPUTATIONAL_SCORE. None of it is a measurement of activity.",
            f"The known-active reference rate is {primary.get('known_active_pass_rate')}: real, "
            "human-measured actives against this target clear this same seven-gate set at that "
            "rate, which is a HISTORICAL_EXPERIMENTAL_ENDPOINT and is the number the arm rates "
            "should be read against.",
            "The top-ranked arm is conditioning=none — it never sees the target. An unconditioned "
            "generator topping a structure-based qualification characterises the gate, not "
            "target-directed design.",
            "The dominant gate is a docking threshold, and docking's rank correlation against "
            "measured potency on this target is +0.078 (development, p=0.663) where -1.0 would "
            "be perfect. 'Docks better' is not 'more potent' here.",
        ],
    )
    assert_no_row_was_dropped(table, 9)
    return table


def benchmark_standards_table(bundle: StudyBundle) -> Record:
    """The five standards this step assembled: four descriptive spreads and one failure.

    No registered record declares a set of benchmark standards. The five rows are
    a post-hoc selection made in this module after every result was seen, and the
    denominator says so: a table that reads as reporting against a prior
    declaration is a stronger claim than the one the evidence supports.
    """
    sweep = bundle.get("generation_sweep")
    rows: list[Record] = []
    distributions: Mapping[str, Any] = {}
    if isinstance(sweep, Mapping):
        distributions = dict(sweep.get("unconstrained_distributions") or {})
    pic50 = bundle.get("pic50_estimate")
    for standard in BENCHMARK_STANDARDS:
        metric = str(standard["metric"])
        spread: Any = MISSING
        n_methods: Any = MISSING
        if standard["standard"] != "pic50" and distributions:
            medians = _medians_for(distributions, metric)
            if medians:
                spread = round(max(medians.values()) - min(medians.values()), 6)
                n_methods = len(medians)
        elif isinstance(pic50, Mapping):
            per_method = dict(pic50.get("per_method") or {})
            by_method = dict((per_method.get("chemprop_pic50") or {}).get("by_method") or {})
            n_methods = len(by_method) or MISSING
            verdict = dict(pic50.get("verdict") or {})
            # Reported so the number cannot be quoted without its refutation in
            # the same row: the spread is the applicability-domain ordering, and
            # the model that produced it loses to a constant.
            spread = verdict.get("spread_across_nine_method_medians")
        # The verdict a spread-only standard carries is DERIVED from whether a
        # spread was actually observed. A row whose source is absent says so
        # instead of carrying a verdict about data this host does not have.
        verdict_for_row = standard["verdict"]
        if standard.get("reported_per_arm") and spread is MISSING:
            verdict_for_row = "SOURCE_ABSENT"
        rows.append(
            {
                "standard": standard["standard"],
                "metric": metric,
                "evidence_type": evidence_type(metric).value,
                "direction": standard["direction"],
                "verdict": verdict_for_row,
                "n_methods_ranked": n_methods,
                "median_spread_across_methods": spread,
                "why": standard.get("why", ""),
            }
        )
    table = make_table(
        "table-11-benchmark-standards",
        "The five standards this step assembled, and what each one supports",
        [
            "standard",
            "metric",
            "evidence_type",
            "direction",
            "verdict",
            "n_methods_ranked",
            "median_spread_across_methods",
            "why",
        ],
        rows,
        denominator=(
            "all five standards STEP 17 ASSEMBLED, including the one that FAILED. No registered "
            "record in this programme declares a set of benchmark standards: this list was built "
            "in reporting.BENCHMARK_STANDARDS after every result was seen, out of endpoints the "
            "sweep happened to have collected. It is a post-hoc selection, not a pre-declaration"
        ),
        notes=[
            "The failed standard is the substantive negative result of this programme and is "
            "reported here rather than dropped.",
            f"The other four carry the verdict {SPREAD_ONLY}, which is a DESCRIPTION of the "
            "observed spread and not a test. They used to carry the hard-coded string SEPARATED "
            "with no test, no interval and no n behind it, while the pIC50 standard was held to "
            "six measured preconditions before being failed. The asymmetry inflated the claim.",
            "Boltz-2 was NOT tested: its contamination audit is INCONCLUSIVE with the PubChem "
            "ingestion route open. 'No available predictor separates these methods' is PROVEN for "
            "everything that could be run here and UNTESTED for Boltz-2.",
        ],
    )
    assert_no_row_was_dropped(table, len(BENCHMARK_STANDARDS))
    return table


def _medians_for(distributions: Mapping[str, Any], metric: str) -> dict[str, float]:
    """Per-arm medians of one endpoint, from the sweep's distribution block."""
    out: dict[str, float] = {}
    for arm, payload in distributions.items():
        if not isinstance(payload, Mapping):
            continue
        endpoints = payload.get("endpoints")
        entry = (endpoints if isinstance(endpoints, Mapping) else payload).get(metric)
        if isinstance(entry, Mapping) and isinstance(entry.get("median"), (int, float)):
            out[str(arm)] = float(entry["median"])
    return out


def _axis_winner_note(
    medians: Mapping[str, Mapping[str, float]], standards: Sequence[Mapping[str, Any]]
) -> str:
    """Which arm tops each axis, derived from the medians this table prints.

    The note this replaces said "No arm wins on more than one axis... there is no
    single best generator in this data", and it was FALSE against the package's
    own table-13 on the day it was written. It was a sentence somebody wrote once
    and nothing recomputed. This recomputes it, so a note that disagrees with the
    table it sits under cannot be shipped again.
    """
    winners: dict[str, str] = {}
    for standard in standards:
        metric = str(standard["metric"])
        column = dict(medians.get(metric) or {})
        if not column:
            continue
        higher = str(standard.get("direction")) == "HIGHER_IS_BETTER"
        winners[metric] = max(column, key=lambda a: column[a]) if higher else min(
            column, key=lambda a: column[a]
        )
    if not winners:
        return "no median column is populated, so no axis has a top arm here."
    counts: dict[str, list[str]] = {}
    for metric, arm in winners.items():
        counts.setdefault(arm, []).append(metric)
    multi = sorted((arm, sorted(axes)) for arm, axes in counts.items() if len(axes) > 1)
    tops = "; ".join(f"{metric}: {arm}" for metric, arm in sorted(winners.items()))
    if not multi:
        return (
            f"No arm tops more than one axis ({tops}). The standards disagree, and which "
            f"generator is 'best' depends on which standard is asked. Topping an axis is not "
            f"winning it: no test was applied to any of these medians."
        )
    named = "; ".join(f"{arm} tops {len(axes)} of {len(winners)} ({', '.join(axes)})" for arm, axes in multi)
    return (
        f"{named}. Per-axis top arm - {tops}. The standards still disagree and no arm tops all "
        f"four, so there is no single best generator here; but 'no arm wins on more than one "
        f"axis' is false against this table and is not what it says. Topping an axis is not "
        f"winning it: no test was applied to any of these medians."
    )


def benchmark_medians_table(bundle: StudyBundle) -> Record:
    """Per-arm medians of the four standards with a column, and nothing from the failed one.

    The failed standard has no column here on purpose. Section 3 of the
    operator-facing summary tells a reader to ignore the pIC50 column it prints;
    a column a reader has to be told to ignore is a column that gets quoted, so
    this package carries the pIC50 standard only as a row in ``table-11`` with
    ``FAILED`` beside it.

    The per-axis winners in the notes are DERIVED from the medians in the table
    rather than written down beside it. The note used to read "no arm wins on more
    than one axis", which was false against this table's own numbers on the day it
    was written: ``flowmol3`` wins two of the four.
    """
    sweep = bundle.get("generation_sweep")
    if not isinstance(sweep, Mapping):
        return make_table(
            "table-13-benchmark-standard-medians",
            "Per-arm medians of the working standards",
            ["arm", "status"],
            [{"arm": "ALL", "status": "SOURCE_ABSENT"}],
            denominator="all nine admitted arms; the source record is absent on this host",
        )
    distributions = dict(sweep.get("unconstrained_distributions") or {})
    pose_quality = dict(sweep.get("pose_quality") or {})
    working = [s for s in BENCHMARK_STANDARDS if s.get("reported_per_arm")]
    medians = {str(s["metric"]): _medians_for(distributions, str(s["metric"])) for s in working}
    strata = dict((sweep.get("primary") or {}).get("strata") or {})
    conditioning = {str(arm.get("arm")): name for name, entries in strata.items() for arm in entries}
    rows: list[Record] = []
    for arm in sorted(distributions):
        validity, n_poses = _intramolecular_validity(pose_quality.get(arm))
        row: Record = {
            "arm": arm,
            "conditioning": conditioning.get(arm, MISSING),
            "n_scored": (
                dict(distributions[arm]).get("n_rows") if isinstance(distributions[arm], Mapping) else MISSING
            ),
        }
        for standard in working:
            metric = str(standard["metric"])
            row[f"{metric}_median"] = medians[metric].get(arm)
        row["intramolecular_validity_pct"] = round(validity, 2) if validity is not None else MISSING
        row["n_poses"] = n_poses or MISSING
        rows.append(row)
    columns = (
        ["arm", "conditioning", "n_scored"]
        + [f"{s['metric']}_median" for s in working]
        + ["intramolecular_validity_pct", "n_poses"]
    )
    table = make_table(
        "table-13-benchmark-standard-medians",
        "Per-arm medians of the four standards with a per-arm column",
        columns,
        rows,
        denominator="all nine admitted arms, including the five that qualified zero scaffolds",
        notes=[
            "Every column is a COMPUTATIONAL_SCORE. Directions: herg lower is better, sa_score "
            "lower is better, qed higher is better, vina_affinity lower is better.",
            _axis_winner_note(medians, working),
            "The pIC50 standard has no column here. It FAILED — see table-11 — and a column a "
            "reader has to be told to ignore is a column that gets quoted.",
            "intramolecular_validity_pct is the DECOMPOSED figure. The adapter's own pass/fail "
            "verdict is zero by construction on this receptor and is invalid under B-15D-1.",
        ],
        evidence={f"{s['metric']}_median": COMPUTATIONAL.value for s in working},
    )
    assert_no_row_was_dropped(table, len(distributions))
    return table


#: Where table-12 may carry a magnitude. The table is assembled from records read
#: OFF DISK — `confirmation-status.json` and `ablation-status.json` — and it is
#: published, so the same inversion the emitters of those records apply is applied
#: again to what is read: a count of rows, and each row's declared
#: ``test_access_count``. `cost_if_run` is carried as canonical JSON text rather
#: than as a mapping, so a cost figure inside it is not a magnitude in this table.
BRANCH_STATUS_MAGNITUDE_PATHS: frozenset[str] = frozenset(
    {"n_rows", "rows[].test_access_count"}
)


def branch_status_table(bundle: StudyBundle) -> Record:
    """Every declared-but-unexecuted branch, with its reason and its unblock.

    The rows assert ``test_access_count`` 0 for records this process did not
    write, so the table is put through the same structural refusal its sources
    are: see :func:`~.confirmation.refuse_false_assurance`.
    """
    rows: list[Record] = []
    confirmation = bundle.get("confirmation_status")
    if isinstance(confirmation, Mapping):
        rows.append(
            {
                "branch": "confirmation (Track A, held-out)",
                "status": confirmation.get("status"),
                "test_access_count": confirmation.get("test_access_count"),
                "blocking_conditions": "; ".join(
                    str(c.get("condition") if isinstance(c, Mapping) else c)
                    for c in confirmation.get("blocking_conditions") or []
                ),
                "cost_if_run": canonical_json(confirmation.get("cost_if_run"))[:300],
            }
        )
    ablation = bundle.get("ablation_status")
    if isinstance(ablation, Mapping):
        for arm in ablation.get("not_run") or []:
            conditions = arm.get("conditions_for_a_legitimate_later_run") or []
            rows.append(
                {
                    "branch": f"ablation arm {arm.get('arm_id')}",
                    "status": "NOT_RUN",
                    "test_access_count": 0,
                    "blocking_conditions": "; ".join(
                        str(c.get("condition") if isinstance(c, Mapping) else c) for c in conditions
                    ),
                    "cost_if_run": canonical_json(arm.get("declared_cost") or arm.get("cost"))[:300],
                }
            )
    if not rows:
        rows.append({"branch": "ALL", "status": "SOURCE_ABSENT"})
    table = make_table(
        "table-12-branches-not-run",
        "Declared branches that did not execute",
        ["branch", "status", "test_access_count", "blocking_conditions", "cost_if_run"],
        rows,
        denominator="every branch the protocol declares and this study did not execute",
        notes=[
            "NOT_RUN is a recorded state with a reason, a cost and an unblock. It is not a gap "
            "and it is not a failure.",
        ],
    )
    assert_no_row_was_dropped(table, len(rows))
    refuse_false_assurance(
        table,
        allowed_magnitude_paths=BRANCH_STATUS_MAGNITUDE_PATHS,
        what="table-12, which asserts test_access_count 0 for every branch it lists,",
    )
    return table


ALL_TABLE_BUILDERS: tuple[str, ...] = (
    "table-01-data-composition",
    "table-02-exclusions",
    "table-03-baseline-capabilities",
    "table-04-artifact-types",
    "table-05-parent-and-external-comparisons",
    "table-06-campaign-outcomes",
    "table-07-costs-cold-and-warm",
    "table-08-uncertainty",
    "table-09-interventions",
    "table-10-benchmark-primary",
    "table-11-benchmark-standards",
    "table-12-branches-not-run",
    "table-13-benchmark-standard-medians",
)


def build_tables(bundle: StudyBundle) -> list[Record]:
    """Every declared table, in order, with nothing filtered to winners."""
    return [
        data_composition_table(bundle),
        exclusion_table(bundle),
        baseline_capability_table(bundle),
        artifact_type_table(bundle),
        comparison_table(bundle),
        campaign_outcome_table(bundle.campaigns),
        cost_table(bundle),
        uncertainty_table(bundle),
        intervention_table(bundle.campaigns, bundle),
        benchmark_primary_table(bundle),
        benchmark_standards_table(bundle),
        branch_status_table(bundle),
        benchmark_medians_table(bundle),
    ]


# --------------------------------------------------------------------------- #
# independent recomputation
# --------------------------------------------------------------------------- #


def recompute_qualifying_scaffold_yield(study: Path | str) -> Record:
    """Recount qualifying scaffolds per arm from the raw per-molecule records.

    Deliberately *not* a call to the evaluator's own ``qualifying_scaffolds``:
    this is set arithmetic written here, from the stored per-molecule
    ``qualifies`` verdicts and scaffold ids, so that agreement with the
    registered table is evidence about the table rather than evidence that one
    function agrees with itself. The evaluator's own function is re-executed
    separately by ``reproduce --mode scientific``.
    """
    root = Path(study) / "sweep" / "run-01" / "cells"
    per_arm: dict[str, Record] = {}
    if not root.is_dir():
        return {
            "status": "SOURCE_ABSENT",
            "path": str(root),
            "per_arm": {},
            "note": "the raw per-molecule records are not reachable from this bundle; the yield "
            "table cannot be independently recomputed here and is NOT reported as agreeing",
        }
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = directory.name
        scaffolds: set[str] = set()
        molecules: dict[str, tuple[str, bool]] = {}
        per_seed: dict[str, int] = {}
        n_rows = 0
        for seed_dir in sorted(directory.glob("seed-*")):
            payload = _read_json(seed_dir / "qualification.json")
            if not isinstance(payload, Mapping):
                continue
            seed_scaffolds: set[str] = set()
            for row in payload.get("rows") or []:
                n_rows += 1
                molecule_id = str(row.get("molecule_id") or "")
                scaffold = str(row.get("scaffold_id") or "")
                qualifies = bool(row.get("qualifies"))
                if not molecule_id or not scaffold:
                    continue
                molecules[molecule_id] = (scaffold, qualifies)
                if qualifies:
                    seed_scaffolds.add(scaffold)
            per_seed[seed_dir.name] = len(seed_scaffolds)
        scaffolds = {scaffold for scaffold, flag in molecules.values() if flag}
        per_arm[arm] = {
            "qualifying_scaffolds": len(scaffolds),
            "n_molecules_seen": len(molecules),
            "n_rows_read": n_rows,
            "per_seed_qualifying_scaffolds": [per_seed[k] for k in sorted(per_seed)],
        }
    return {"status": "RECOMPUTED", "path": str(root), "per_arm": per_arm}


def recompute_track_a_mae(study: Path | str, *, run: str = "run-13") -> Record:
    """Recompute each Track A arm's MAE as the mean of its recorded errors.

    Reads the per-arm ``result.json`` under the custodian's store, takes the
    mean of ``endpoint.absolute_errors`` and reports **the aggregate only**. The
    keys of that mapping are compound identities and are never carried out of
    this function: the return value is counts and statistics.
    """
    root = Path(study) / "baselines" / run
    if not root.is_dir():
        return {
            "status": "SOURCE_ABSENT",
            "path": str(root),
            "per_arm": {},
            "note": "the per-arm raw error tables live in the custodian's store and are not "
            "reachable here; the MAE table is NOT reported as agreeing",
        }
    per_arm: dict[str, Record] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        payload = _read_json(directory / "result.json")
        if not isinstance(payload, Mapping):
            continue
        endpoint = dict(payload.get("endpoint") or {})
        errors = endpoint.get("absolute_errors")
        if not isinstance(errors, Mapping) or not errors:
            per_arm[directory.name] = {
                "mae_recomputed": MISSING,
                "n_errors": 0,
                "status": "NO_ERROR_TABLE",
            }
            continue
        values = [float(v) for v in errors.values() if isinstance(v, (int, float))]
        per_arm[directory.name] = {
            "mae_recomputed": sum(values) / len(values) if values else MISSING,
            "n_errors": len(values),
            "recorded_primary_value": payload.get("primary_value"),
            "recorded_coverage": payload.get("coverage"),
            "status": "RECOMPUTED",
        }
    return {
        "status": "RECOMPUTED",
        "path": str(root),
        "per_arm": per_arm,
        "note": "aggregate statistics only; no compound identity or label leaves this function",
    }


def recompute_sweep_cost_total(study: Path | str) -> Record:
    """Re-add the sweep's total GPU occupancy from the per-cell raw cost records.

    The registered total is generation plus the endpoint panel; the docking
    phase's occupancy is reported separately and deliberately not added, because
    Vina does no GPU work and the resident context is an idle ADMET-AI model.
    This function reproduces that arithmetic from the 27 generation cells and
    the 9 scoring records rather than reading the total back.
    """
    root = Path(study) / "sweep" / "run-01" / "cells"
    if not root.is_dir():
        return {"status": "SOURCE_ABSENT", "path": str(root)}
    generation = 0.0
    endpoint = 0.0
    docking_occupancy = 0.0
    docking_wall = 0.0
    posebusters_wall = 0.0
    generation_wall = 0.0
    n_cells = 0
    n_scoring = 0
    unmeasured: list[str] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(directory.glob("seed-*")):
            payload = _read_json(seed_dir / "generation.json")
            if not isinstance(payload, Mapping):
                continue
            n_cells += 1
            cost = dict(payload.get("cost") or {})
            value = cost.get("gpu_seconds")
            if isinstance(value, (int, float)):
                generation += float(value)
            else:
                unmeasured.append(f"{directory.name}/{seed_dir.name}:gpu_seconds")
            wall = cost.get("wall_seconds")
            if isinstance(wall, (int, float)):
                generation_wall += float(wall)
        scoring = _read_json(directory / "scoring.json")
        if isinstance(scoring, Mapping):
            n_scoring += 1
            cost = dict(scoring.get("cost") or {})
            for key, sink in (
                ("endpoint_gpu_seconds", "endpoint"),
                ("docking_gpu_seconds", "docking_occupancy"),
                ("docking_wall_seconds", "docking_wall"),
                ("posebusters_wall_seconds", "posebusters_wall"),
            ):
                value = cost.get(key)
                if not isinstance(value, (int, float)):
                    unmeasured.append(f"{directory.name}:{key}")
                    continue
                if sink == "endpoint":
                    endpoint += float(value)
                elif sink == "docking_occupancy":
                    docking_occupancy += float(value)
                elif sink == "docking_wall":
                    docking_wall += float(value)
                else:
                    posebusters_wall += float(value)
    return {
        "status": "RECOMPUTED",
        "path": str(root),
        "n_generation_cells": n_cells,
        "n_scoring_records": n_scoring,
        "generation_gpu_seconds": round(generation, 4),
        "generation_wall_seconds": round(generation_wall, 4),
        "endpoint_gpu_seconds": round(endpoint, 4),
        "total_gpu_seconds": round(generation + endpoint, 4),
        "docking_gpu_occupancy_seconds": round(docking_occupancy, 4),
        "scoring_docking_wall_seconds": round(docking_wall, 4),
        "scoring_posebusters_wall_seconds": round(posebusters_wall, 4),
        "unmeasured_entries": unmeasured,
        "note": "total_gpu_seconds is generation plus the endpoint panel. The docking phase's "
        "occupancy is reported apart and NOT added, because AutoDock Vina does no GPU work and "
        "the resident context is an idle ADMET-AI model in the same process",
    }


def _close(left: Any, right: Any, *, tolerance: float = 0.05) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def crosscheck_yield_table(bundle: StudyBundle, recomputed: Mapping[str, Any]) -> Record:
    """Compare the independently recounted yields with the registered table."""
    sweep = bundle.get("generation_sweep")
    if not isinstance(sweep, Mapping) or recomputed.get("status") != "RECOMPUTED":
        return {
            "status": "NOT_CHECKED",
            "reason": "SOURCE_ABSENT",
            "reproduces": MISSING,
            "disagreements": [],
        }
    registered: dict[str, int] = {}
    strata = dict((sweep.get("primary") or {}).get("strata") or {})
    for entries in strata.values():
        for arm in entries:
            registered[str(arm.get("arm"))] = int(arm.get("qualifying_scaffolds") or 0)
    per_arm = dict(recomputed.get("per_arm") or {})
    disagreements: list[Record] = []
    for arm in sorted(set(registered) | set(per_arm)):
        left = registered.get(arm)
        right = (per_arm.get(arm) or {}).get("qualifying_scaffolds")
        if left != right:
            disagreements.append({"arm": arm, "registered": left, "recomputed": right})
    return {
        "status": "CHECKED",
        "n_arms": len(set(registered) | set(per_arm)),
        "reproduces": not disagreements,
        "disagreements": disagreements,
        "registered": registered,
        "recomputed": {arm: (per_arm.get(arm) or {}).get("qualifying_scaffolds") for arm in per_arm},
    }


def crosscheck_cost_total(bundle: StudyBundle, recomputed: Mapping[str, Any]) -> Record:
    """Compare the re-added cost total with the registered one."""
    sweep = bundle.get("generation_sweep")
    if not isinstance(sweep, Mapping) or recomputed.get("status") != "RECOMPUTED":
        return {"status": "NOT_CHECKED", "reason": "SOURCE_ABSENT", "reproduces": MISSING}
    measured = dict((sweep.get("cost") or {}).get("measured") or {})
    checks: list[Record] = []
    for key, recomputed_key in (
        ("generation_gpu_seconds", "generation_gpu_seconds"),
        ("scoring_gpu_seconds", "endpoint_gpu_seconds"),
        ("total_gpu_seconds", "total_gpu_seconds"),
        ("docking_gpu_occupancy_seconds", "docking_gpu_occupancy_seconds"),
        ("scoring_docking_wall_seconds", "scoring_docking_wall_seconds"),
        ("scoring_posebusters_wall_seconds", "scoring_posebusters_wall_seconds"),
    ):
        left = measured.get(key)
        right = recomputed.get(recomputed_key)
        checks.append(
            {
                "quantity": key,
                "registered": left,
                "recomputed": right,
                "agrees": _close(left, right, tolerance=0.11),
            }
        )
    return {
        "status": "CHECKED",
        "checks": checks,
        "reproduces": all(check["agrees"] for check in checks),
        "tolerance_seconds": 0.11,
        "tolerance_reason": "the registered figures are rounded to one decimal per phase and "
        "summed, so a re-addition of the unrounded per-cell records differs in the last digit",
    }


def crosscheck_mae_table(bundle: StudyBundle, recomputed: Mapping[str, Any]) -> Record:
    """Compare the recomputed Track A MAEs with the registered baseline table."""
    results = bundle.get("baseline_results")
    if not isinstance(results, Mapping) or recomputed.get("status") != "RECOMPUTED":
        return {"status": "NOT_CHECKED", "reason": "SOURCE_ABSENT", "reproduces": MISSING}
    registered = {
        str(arm.get("baseline_id")): arm.get("primary_value")
        for arm in results.get("arms") or []
        if isinstance(arm, Mapping) and arm.get("primary_metric") == "mae_pactivity"
    }
    per_arm = dict(recomputed.get("per_arm") or {})
    checks: list[Record] = []
    for arm in sorted(set(registered) | set(per_arm)):
        left = registered.get(arm)
        right = (per_arm.get(arm) or {}).get("mae_recomputed")
        checks.append(
            {
                "arm": arm,
                "registered": left,
                "recomputed": right,
                "n_errors": (per_arm.get(arm) or {}).get("n_errors"),
                # MISSING, not False: an arm with no recorded error table did not
                # disagree, it was never checkable. Writing False here would put a
                # failed comparison in the record where there was no comparison.
                "agrees": _close(left, right, tolerance=1e-9) if right is not None else MISSING,
                "status": (per_arm.get(arm) or {}).get("status", "NOT_IN_RAW_STORE"),
            }
        )
    checkable = [c for c in checks if c["recomputed"] is not None]
    return {
        "status": "CHECKED",
        "checks": checks,
        "n_checkable": len(checkable),
        "reproduces": bool(checkable) and all(c["agrees"] for c in checkable),
        "note": "arms whose raw error table is absent are reported as NOT_IN_RAW_STORE, not as agreeing",
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


_FIGURE_CAPTION_SUFFIX = (
    "Computational scores are drawn as circles and labelled [computational]; historical "
    "experimental endpoints are drawn as squares and labelled [experimental]."
)


def figure_performance_versus_budget(bundle: StudyBundle, path: Path) -> Record:
    """Qualifying scaffolds against the GPU occupancy each arm actually spent."""
    sweep = bundle.get("generation_sweep")
    if not isinstance(sweep, Mapping):
        return {"figure": path.name, "status": "SOURCE_ABSENT"}
    plt = _pyplot()
    points: list[tuple[str, float, int, str]] = []
    for cell in bundle.sweep_cells.values():
        arm = dict(cell["arm"])
        cost = dict(arm.get("generation_cost") or {})
        occupancy = cost.get("gpu_seconds")
        points.append(
            (
                str(arm.get("arm")),
                float(occupancy) if isinstance(occupancy, (int, float)) else float("nan"),
                int(arm.get("qualifying_scaffolds") or 0),
                str(arm.get("conditioning")),
            )
        )
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    for name, occupancy, scaffolds, conditioning in sorted(points):
        marker = "o" if conditioning != "none" else "^"
        axes.scatter(occupancy, scaffolds, marker=marker, s=70, zorder=3)
        axes.annotate(name, (occupancy, scaffolds), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes.set_xlabel("generation GPU OCCUPANCY seconds (resource accounting, not GPU work)", fontsize=9)
    axes.set_ylabel("qualifying scaffolds of 150\n[computational]", fontsize=9)
    axes.set_title("Performance against budget — nine arms, n=3 seeds each", fontsize=11)
    axes.grid(True, alpha=0.3)
    axes.margins(x=0.16, y=0.12)
    caption = (
        "Every arm is plotted, including the five that qualified zero scaffolds. Triangles never "
        "see the target (conditioning=none); circles are pocket- or ligand-conditioned. Spearman "
        "between occupancy and yield over these nine arms is -0.166, the opposite sign to 'yield "
        "is bought with compute'."
    )
    figure.text(0.03, 0.03, _wrap(caption, 104), fontsize=7, va="bottom")
    figure.subplots_adjust(bottom=0.34, left=0.13, right=0.97, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "figure": path.name,
        "status": "WRITTEN",
        "n_points": len(points),
        "sha256": sha256_file(path),
        "caption": caption,
    }


def campaign_development_delta(campaign: Mapping[str, Any]) -> float | None:
    """The campaign's selected candidate minus its own declared incumbent.

    ``None`` when the campaign selected nothing, which is a MISSING delta and
    never a delta of zero. The locked protocol assigns ``delta_i = 0.0`` to a
    no-nominee campaign *in its own estimator*, and that is a statement about
    the process delivering nothing; it is not a measured metric value and this
    function refuses to return it as one.
    """
    incumbent = dict(campaign.get("incumbent") or {})
    selected = dict((campaign.get("selection") or {}).get("selected") or {})
    value = selected.get("development_metric_value")
    base = incumbent.get("value")
    if isinstance(value, (int, float)) and isinstance(base, (int, float)):
        return float(value) - float(base)
    return None


def figure_paired_campaign_gains(bundle: StudyBundle, path: Path) -> Record:
    """Per-campaign outcome against the incumbent, failures included as rows."""
    plt = _pyplot()
    labels: list[str] = []
    values: list[float] = []
    colours: list[str] = []
    annotations: list[str] = []
    n_with_delta = 0
    for campaign in bundle.campaigns:
        labels.append(str(campaign.get("campaign_id", "")).replace("camp_zak_c9r_", ""))
        delta = campaign_development_delta(campaign)
        if delta is None:
            values.append(0.0)
            colours.append("#999999")
            annotations.append(f"{campaign.get('state')}\nno measured delta")
        else:
            n_with_delta += 1
            values.append(delta)
            colours.append("#2b6cb0" if delta < 0 else "#c05621")
            annotations.append(f"{campaign.get('state')}\nΔ={delta:+.5f}")
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    positions = range(len(labels))
    axes.bar(list(positions), values, color=colours)
    for index, text in enumerate(annotations):
        axes.annotate(
            text,
            (index, values[index]),
            ha="center",
            va="bottom" if values[index] >= 0 else "top",
            fontsize=7,
        )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes.set_ylabel(
        "Δ mae_pactivity vs the campaign's own\nincumbent [computational] (lower is better)",
        fontsize=8,
    )
    axes.set_title("Paired per-campaign gains — all campaigns, none dropped", fontsize=11)
    caption = (
        "Grey bars are campaigns with NO measured delta: a crash, or a terminal state with no "
        "eligible candidate. They are drawn at zero height and labelled, NOT omitted — the locked "
        f"protocol's denominator is every campaign. {n_with_delta} of {len(labels)} campaigns "
        "produced a measured delta, and it is INCONCLUSIVE: it has no interval, and both it and "
        "the reverted parent are worse than the trivial median constant (0.444117) on the same "
        "34-compound development pool. A zero-height grey bar is a MISSING delta, not a delta of "
        "zero."
    )
    figure.text(0.03, 0.02, _wrap(caption, 104), fontsize=7, va="bottom")
    figure.subplots_adjust(bottom=0.42, left=0.15, right=0.97, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "figure": path.name,
        "status": "WRITTEN",
        "n_campaigns": len(labels),
        "n_without_a_measured_delta": sum(1 for c in colours if c == "#999999"),
        "sha256": sha256_file(path),
        "caption": caption,
    }


def figure_artifact_genealogy(bundle: StudyBundle, path: Path) -> Record:
    """The artifact dependency graph, drawn from the manifest that validates it.

    Three node states, drawn apart, because collapsing them is how a package
    reports a broken reference as a missing one: **resolved** (filled), **no
    declared path** (open grey — an entity this study depends on that has no
    single file in this checkout, such as a public snapshot), and **dangling**
    (open red — a declared path that does not resolve).
    """
    plt = _pyplot()
    manifest = artifact_dependency_manifest(bundle)
    nodes = list(manifest["nodes"])
    edges = [tuple(edge) for edge in manifest["edges"]]

    positions: dict[str, tuple[float, float]] = {}
    per_layer: dict[int, int] = {}
    for node in nodes:
        layer = int(node.get("layer", 0))
        slot = per_layer.get(layer, 0)
        per_layer[layer] = slot + 1
        positions[node["id"]] = (float(layer) * 1.0, -float(slot) * 1.15)

    figure, axes = plt.subplots(figsize=(11.0, 6.4))
    for source, target in edges:
        if source in positions and target in positions:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            axes.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "#b0b0b0", "lw": 0.8},
            )
    for node in nodes:
        x, y = positions[node["id"]]
        if node["path"] and node["present"]:
            face, edge, suffix = "#2b6cb0", "#2b6cb0", ""
        elif node["path"]:
            face, edge, suffix = "none", "#c53030", "  DANGLING"
        else:
            face, edge, suffix = "none", "#909090", "  no single file"
        axes.scatter(
            x,
            y,
            s=90,
            marker="s" if node.get("evidence_type") == EXPERIMENTAL.value else "o",
            facecolors=face,
            edgecolors=edge,
            zorder=3,
        )
        # Elide the MIDDLE, never the tail: these ids differ only in their last
        # few characters, and a head-truncated label makes four campaigns look
        # like one.
        label = str(node["id"])
        if len(label) > 26:
            label = f"{label[:10]}…{label[-14:]}"
        axes.annotate(
            f"{label}\n[{node.get('kind')}]{suffix}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -9),
            fontsize=6,
            ha="center",
            va="top",
        )
    max_layer = max(int(node.get("layer", 0)) for node in nodes) if nodes else 0
    min_y = min(y for _, y in positions.values()) if positions else -1.0
    axes.set_xlim(-0.65, max_layer + 0.65)
    axes.set_ylim(min_y - 0.7, 0.6)
    axes.axis("off")
    axes.set_title("Artifact genealogy — every node the manifest declares", fontsize=11)
    caption = (
        "Filled nodes resolve to bytes on this host. Open RED nodes declare a path that does not "
        f"resolve — dangling references found: {manifest['n_dangling']}. Open GREY nodes have no "
        "single file in this checkout (a public snapshot, or an artifact recorded only inside a "
        "campaign); they are declared rather than omitted and they cannot dangle. Squares are "
        "historical experimental inputs [experimental]; circles are artifacts this programme "
        "computed [computational]."
    )
    figure.text(0.02, 0.03, _wrap(caption, 140), fontsize=7, va="bottom")
    figure.subplots_adjust(bottom=0.20, left=0.02, right=0.98, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "figure": path.name,
        "status": "WRITTEN",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_dangling": manifest["n_dangling"],
        "n_without_a_declared_path": sum(1 for node in nodes if not node["path"]),
        "sha256": sha256_file(path),
        "caption": caption,
    }


def _intramolecular_validity(payload: Any) -> tuple[float | None, int]:
    """Percent intramolecularly valid and the denominator, or ``(None, 0)``.

    Reads the DECOMPOSED block, never the adapter's own pass/fail verdict: that
    verdict is zero by construction on this receptor and is invalid under
    blocker B-15D-1. A record without the decomposition returns ``None``, which
    the figure draws as MISSING rather than as 0 %.
    """
    if not isinstance(payload, Mapping):
        return None, 0
    block = payload.get("intramolecular_validity")
    if not isinstance(block, Mapping):
        return None, 0
    passing = block.get("n_passing_every_check")
    measured = block.get("n_measured")
    if isinstance(passing, (int, float)) and isinstance(measured, (int, float)) and measured:
        return 100.0 * float(passing) / float(measured), int(measured)
    return None, 0


def figure_quality_versus_cost(bundle: StudyBundle, path: Path) -> Record:
    """Generated-structure quality against what the arm cost to run."""
    plt = _pyplot()
    sweep = bundle.get("generation_sweep")
    if not isinstance(sweep, Mapping):
        return {"figure": path.name, "status": "SOURCE_ABSENT"}
    pose_quality = dict(sweep.get("pose_quality") or {})
    points: list[tuple[str, float, float, int]] = []
    for name, cell in sorted(bundle.sweep_cells.items()):
        arm = dict(cell["arm"])
        cost = dict(arm.get("scoring_cost") or {})
        seconds = cost.get("seconds_per_molecule")
        validity, n_poses = _intramolecular_validity(pose_quality.get(name) or arm.get("pose_quality"))
        points.append(
            (
                name,
                float(seconds) if isinstance(seconds, (int, float)) else float("nan"),
                validity if validity is not None else float("nan"),
                n_poses,
            )
        )
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    for name, seconds, validity, n_poses in points:
        if validity != validity:  # NaN — an unmeasured quantity, drawn on the axis and labelled
            axes.scatter(seconds, 0.0, marker="x", color="#c53030", s=60, zorder=3)
            axes.annotate(
                f"{name}\nvalidity MISSING",
                (seconds, 0.0),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )
            continue
        axes.scatter(seconds, validity, marker="o", s=70, zorder=3)
        axes.annotate(
            f"{name} (n={n_poses})",
            (seconds, validity),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    axes.set_xlabel("scoring seconds per molecule (resource accounting)", fontsize=9)
    axes.set_ylabel("intramolecular pose validity, %\n[computational]", fontsize=9)
    axes.set_title("Quality against cost — decomposed pose validity, not the adapter's verdict", fontsize=11)
    axes.grid(True, alpha=0.3)
    axes.margins(x=0.18, y=0.14)
    caption = (
        "The adapter's own pass/fail verdict said 0 of 150 for all nine arms and is INVALID: "
        "blocker B-15D-1, it counts every boolean column as a check and one has inverted "
        "polarity, and three columns are vacuously false because the prepared receptor has no "
        "cofactors and no waters. The correct decomposed figure is 93.2 % intramolecular validity "
        "over 1,296 poses. The 54 molecules with no pose are the 54 docking failures and are "
        "recorded as absent measurements, never as zeros."
    )
    figure.text(0.03, 0.02, _wrap(caption, 104), fontsize=7, va="bottom")
    figure.subplots_adjust(bottom=0.38, left=0.13, right=0.97, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "figure": path.name,
        "status": "WRITTEN",
        "n_points": len(points),
        "sha256": sha256_file(path),
        "caption": caption,
    }


def figure_ablation_effects(bundle: StudyBundle, path: Path) -> Record:
    """Every registered ablation factor, executed or not."""
    plt = _pyplot()
    ablation = bundle.get("ablation_status")
    if not isinstance(ablation, Mapping):
        return {"figure": path.name, "status": "SOURCE_ABSENT"}
    factors = list((ablation.get("attribution") or {}).get("factors") or [])
    executed_by_id = {str(entry.get("comparison_id")): entry for entry in (ablation.get("executed") or [])}
    labels = [str(f.get("factor")) for f in factors]
    executed = [bool(f.get("separated_by_a_controlled_comparison")) for f in factors]
    captions: list[str] = []
    for factor in factors:
        names = [
            str(evidence.get("comparison_id"))
            for evidence in factor.get("evidence") or []
            if str(evidence.get("comparison_id")) in executed_by_id
        ]
        if names:
            deltas = []
            for name in names:
                delta = executed_by_id[name].get("delta")
                deltas.append(
                    f"{name} (Δ={delta:+.5f}, no interval)" if isinstance(delta, (int, float)) else name
                )
            captions.append("executed, DEVELOPMENT_STAGE: " + "; ".join(deltas))
        else:
            captions.append("NO CONTROLLED COMPARISON — every arm for this factor is NOT_RUN")
    figure, axes = plt.subplots(figsize=(8.6, 4.4))
    positions = list(range(len(labels)))
    axes.barh(
        positions,
        [1.0] * len(labels),
        color=["#2b6cb0" if flag else "#e2e8f0" for flag in executed],
        edgecolor="#4a5568",
    )
    for index, flag in enumerate(executed):
        axes.annotate(
            captions[index],
            (0.015, index),
            va="center",
            fontsize=7,
            color="white" if flag else "#1a202c",
        )
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=9)
    axes.set_xticks([])
    axes.set_xlim(0, 1)
    axes.set_title("Ablation effects — six declared factors", fontsize=11)
    caption = (
        "Not one attribution in this study is confirmatory. The three executed comparisons are "
        "development-partition only; the four predeclared ablation arms are NOT_RUN because the "
        "arm each one controls for, B5-self-evolution, has never been executed. The record's own "
        "sentence: no unique mechanism may be claimed for any observed difference."
    )
    figure.text(0.03, 0.02, _wrap(caption, 104), fontsize=7, va="bottom")
    figure.subplots_adjust(bottom=0.32, left=0.24, right=0.97, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "figure": path.name,
        "status": "WRITTEN",
        "n_factors": len(labels),
        "n_executed": sum(executed),
        "sha256": sha256_file(path),
        "caption": caption,
    }


def _wrap(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


FIGURE_BUILDERS: tuple[tuple[str, str], ...] = (
    ("figure-01-performance-versus-budget.png", "figure_performance_versus_budget"),
    ("figure-02-paired-campaign-gains.png", "figure_paired_campaign_gains"),
    ("figure-03-artifact-genealogy.png", "figure_artifact_genealogy"),
    ("figure-04-quality-versus-cost.png", "figure_quality_versus_cost"),
    ("figure-05-ablation-effects.png", "figure_ablation_effects"),
)


def build_figures(bundle: StudyBundle, directory: Path) -> list[Record]:
    directory.mkdir(parents=True, exist_ok=True)
    builders = {
        "figure_performance_versus_budget": figure_performance_versus_budget,
        "figure_paired_campaign_gains": figure_paired_campaign_gains,
        "figure_artifact_genealogy": figure_artifact_genealogy,
        "figure_quality_versus_cost": figure_quality_versus_cost,
        "figure_ablation_effects": figure_ablation_effects,
    }
    out: list[Record] = []
    for filename, builder in FIGURE_BUILDERS:
        out.append(builders[builder](bundle, directory / filename))
    return out


# --------------------------------------------------------------------------- #
# the artifact dependency manifest, and reference validation
# --------------------------------------------------------------------------- #


def artifact_dependency_manifest(bundle: StudyBundle) -> Record:
    """Every artifact this study depends on, what it depends on, and whether it resolves.

    A node that does not resolve is reported as ``DANGLING`` with its declared
    address. The package is still written — a package that refuses to describe a
    broken reference cannot report one — and the count is carried into the
    report and the figure.
    """
    nodes: list[Record] = []
    edges: list[tuple[str, str]] = []

    def add(
        node_id: str,
        kind: str,
        layer: int,
        path: str | None,
        *,
        evidence: EvidenceType = COMPUTATIONAL,
        address: str | None = None,
    ) -> None:
        resolved = False
        measured = "ABSENT"
        if path:
            candidate = (bundle.repo / path).resolve()
            resolved = candidate.exists()
            measured = sha256_file(candidate) if resolved else "ABSENT"
        nodes.append(
            {
                "id": node_id,
                "kind": kind,
                "layer": layer,
                "path": path,
                "present": resolved,
                "declared_address": address,
                "measured_address": measured,
                "evidence_type": evidence.value,
                "address_agrees": (address is None) or (address == measured),
            }
        )

    target_manifest = bundle.sources.get("target_manifest", {})
    data_card = bundle.sources.get("data_card", {})
    split = bundle.sources.get("split_manifest", {})
    add("chembl+bindingdb snapshot", "DATA_SOURCE", 0, None, evidence=EXPERIMENTAL)
    add("target_manifest", "TARGET", 1, target_manifest.get("path"), evidence=EXPERIMENTAL)
    add("data_card", "DATASET", 1, data_card.get("path"), evidence=EXPERIMENTAL)
    add("split_manifest", "SPLIT", 2, split.get("path"))
    add("evaluator", "EVALUATOR", 2, "src/drug_agent/selfevolve/evaluation")
    add("protocol lock zak-c9r-2", "PROTOCOL", 3, bundle.sources.get("protocol_lock", {}).get("path"))
    add("qualification rule", "RULE", 3, bundle.sources.get("qualification", {}).get("path"))
    add("calibration record", "RULE", 3, bundle.sources.get("calibration", {}).get("path"))
    add("receptor pocket 6JUU C9R", "RECEPTOR_STRUCTURE", 3, None)
    add("baseline table", "RESULT", 4, bundle.sources.get("baseline_results", {}).get("path"))
    add("generation sweep record", "RESULT", 4, bundle.sources.get("generation_sweep", {}).get("path"))
    add("pIC50 independent estimate", "RESULT", 4, bundle.sources.get("pic50_estimate", {}).get("path"))
    add("ablation status", "RESULT", 4, bundle.sources.get("ablation_status", {}).get("path"))
    add(
        "confirmation status (NOT_RUN)",
        "RESULT",
        4,
        bundle.sources.get("confirmation_status", {}).get("path"),
    )

    for source, target in (
        ("chembl+bindingdb snapshot", "data_card"),
        ("target_manifest", "data_card"),
        ("data_card", "split_manifest"),
        ("split_manifest", "protocol lock zak-c9r-2"),
        ("evaluator", "protocol lock zak-c9r-2"),
        ("evaluator", "qualification rule"),
        ("evaluator", "calibration record"),
        ("receptor pocket 6JUU C9R", "qualification rule"),
        ("qualification rule", "generation sweep record"),
        ("protocol lock zak-c9r-2", "generation sweep record"),
        ("protocol lock zak-c9r-2", "baseline table"),
        ("protocol lock zak-c9r-2", "ablation status"),
        ("protocol lock zak-c9r-2", "confirmation status (NOT_RUN)"),
        ("generation sweep record", "pIC50 independent estimate"),
        ("baseline table", "ablation status"),
    ):
        edges.append((source, target))

    for campaign in bundle.campaigns:
        campaign_id = str(campaign.get("campaign_id"))
        add(campaign_id, "CAMPAIGN", 5, campaign.get("_directory"))
        edges.append(("protocol lock zak-c9r-2", campaign_id))
        nomination = campaign.get("nomination")
        if isinstance(nomination, Mapping) and nomination.get("artifact_id"):
            artifact_id = str(nomination["artifact_id"])
            add(artifact_id, "ALGORITHM", 6, None)
            edges.append((campaign_id, artifact_id))

    dangling = [node for node in nodes if node["path"] and not node["present"]]
    mismatched = [node for node in nodes if not node["address_agrees"]]
    return {
        "nodes": nodes,
        "edges": [list(edge) for edge in edges],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_dangling": len(dangling),
        "dangling": [
            {"id": node["id"], "declared_path": node["path"], "kind": node["kind"]} for node in dangling
        ],
        "n_address_mismatches": len(mismatched),
        "note": "a node with path null is an entity this study depends on that has no single file "
        "in this checkout — a public snapshot, or an artifact recorded only inside a campaign. It "
        "is declared rather than silently omitted, and it is NOT counted as dangling, because a "
        "reference with no declared path cannot dangle",
    }


def validate_references(bundle: StudyBundle, package: Mapping[str, Any]) -> Record:
    """Check every required reference the package emits actually resolves.

    Three classes of reference are checked: the record sources the tables were
    built from, the files the package itself wrote, and the artifact dependency
    manifest's nodes. A dangling reference is reported with its declared address;
    it never becomes an absent row.
    """
    dangling: list[Record] = []
    checked = 0
    for key, source in bundle.sources.items():
        checked += 1
        if not source.get("present"):
            dangling.append(
                {
                    "class": "RECORD_SOURCE",
                    "id": key,
                    "declared_path": source.get("path"),
                    "why": "the record the package reads this table from is not on this host",
                }
            )
    output = Path(package["output_directory"])
    for group in ("tables", "figures", "raw"):
        for entry in package.get(group) or []:
            name = entry.get("path") or entry.get("figure")
            if not name:
                continue
            checked += 1
            if not (output / group / name).exists():
                dangling.append(
                    {
                        "class": f"PACKAGE_{group.upper()}",
                        "id": name,
                        "declared_path": f"{group}/{name}",
                        "why": "the package declares this file and did not write it",
                    }
                )
    manifest = package.get("artifact_dependency_manifest") or {}
    for node in manifest.get("dangling") or []:
        checked += 1
        dangling.append(
            {
                "class": "ARTIFACT_NODE",
                "id": node.get("id"),
                "declared_path": node.get("declared_path"),
                "why": "the artifact manifest declares a path that does not resolve",
            }
        )
    return {
        "n_references_checked": checked,
        "n_dangling": len(dangling),
        "dangling": dangling,
        "all_required_references_resolve": not dangling,
    }


# --------------------------------------------------------------------------- #
# identities, hashes, commands, environment
# --------------------------------------------------------------------------- #


_INCHIKEY = re.compile(r"\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b")
_COMPOUND_ACCESSION = re.compile(r"\bCHEMBL\d+\b")


def scan_for_compound_identities(text: str, *, allow: Iterable[str] = ()) -> list[str]:
    """Every per-compound identity in *text* that is not on the allow list.

    The allow list exists for the target's own accession and for document
    accessions, which are not compounds. Everything else is a compound identity
    and must not appear in a package that is allowed to publish counts and
    statistics only.
    """
    permitted = set(allow)
    found = list(_INCHIKEY.findall(text))
    found += [match for match in _COMPOUND_ACCESSION.findall(text) if match not in permitted]
    return sorted(set(found))


def git_state(repo: Path) -> Record:
    """The commit the package was built at, and whether the tree was clean."""

    def run(*args: str, strip: bool = True) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],  # fixed argv, no shell
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() if strip else completed.stdout

    # NOT stripped: a porcelain line is two status columns, a space, then the
    # path, and an unstaged entry's first column is a space. Stripping the
    # output eats it and the FIRST path comes back with its leading character
    # missing — ".gitignore" as "gitignore", which is a different file.
    status = run("status", "--porcelain", strip=False)
    lines = [line for line in (status or "").splitlines() if line.strip()]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "tree_clean": not lines if status is not None else MISSING,
        "dirty_paths": [line[3:] for line in lines],
        "describe": run("describe", "--always", "--dirty"),
    }


def prompt_identities(repo: Path) -> list[Record]:
    """Every system prompt, its checksum stamp and whether the stamp still holds."""
    out: list[Record] = []
    root = repo / "system_prompts"
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(errors="replace")
        stamped = None
        for line in text.splitlines():
            match = re.search(r"checksum[^0-9a-f]*([0-9a-f]{8,64})", line, flags=re.IGNORECASE)
            if match:
                stamped = match.group(1)
                break
        out.append(
            {
                "path": _display_path(path, repo),
                "sha256": sha256_file(path),
                "declared_checksum": stamped,
                "bytes": path.stat().st_size,
            }
        )
    return out


def model_identities(bundle: StudyBundle) -> Record:
    """Which models produced which number, named rather than implied."""
    environment = bundle.get("environment")
    models: Record = {}
    if isinstance(environment, Mapping):
        models = dict(environment.get("models") or {})
    reasoning = MISSING
    for campaign in bundle.campaigns:
        llm = dict(campaign.get("llm") or {})
        if llm.get("model"):
            reasoning = llm.get("model")
            break
    return {
        "reasoning_model_for_campaigns": reasoning,
        "reasoning_model_note": (
            "the campaign records carry the provider and the charged output tokens; where the "
            "model string is absent from a record it is MISSING here rather than assumed"
        ),
        "platform_defaults": models,
        "scientific_models": {
            "activity_predictor_parent": "sklearn RandomForest, model_version baseline-rf-1.0.0, "
            "n_estimators 200, max_depth 12, seed 42, Morgan r2/2048",
            "activity_predictor_external": "Chemprop D-MPNN, fitted on the 102 TRAIN rows only",
            "docking": "AutoDock Vina f458505-mod, exhaustiveness 8, seed 42, frozen box",
            "admet_panel": "ADMET-AI (hERG)",
            "generators": "nine upstream checkouts pinned in scripts/external/CHECKOUTS.tsv, each "
            "with its own weights; see table-03 for per-arm availability",
        },
    }


def _measured_evaluator_hash() -> str | None:
    """Re-measure the evaluator's kernel digest, normalised to ``sha256:<hex>``.

    ``integrity.kernel_digest()`` returns bare hex and every record in this
    study stores the prefixed form. Comparing the two without normalising is how
    a re-measurement that agrees gets reported as a mismatch.
    """
    try:
        from ..evaluation.integrity import kernel_digest

        digest = str(kernel_digest())
    except Exception:  # pragma: no cover - the digest is unavailable off the supported lane
        return None
    return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def registered_addresses(bundle: StudyBundle) -> Record:
    """The content addresses the study is registered at, measured where possible."""
    lock = bundle.get("protocol_lock")
    protocol: Record = {}
    if isinstance(lock, Mapping):
        protocol = dict(lock.get("protocol") or {})
    measured_evaluator = _measured_evaluator_hash()
    return {
        "protocol_version": protocol.get("protocol_version") or REGISTERED_PROTOCOL_VERSION,
        "protocol_hash": {
            "declared": protocol.get("protocol_hash")
            or (lock or {}).get("protocol_hash")
            or REGISTERED_PROTOCOL_HASH,
            "registered_constant": REGISTERED_PROTOCOL_HASH,
            "agrees": (
                ((lock or {}).get("protocol_hash") or REGISTERED_PROTOCOL_HASH) == REGISTERED_PROTOCOL_HASH
            ),
        },
        "target_hash": {
            "declared": protocol.get("target_hash"),
            "independently_measured": MISSING,
            "why_not": "re-deriving it rebuilds the split, which reads the held-out family "
            "assignment. Reported as unmeasured rather than as agreeing — Step 15 residual R6",
        },
        "data_hash": {
            "declared": protocol.get("data_hash"),
            "independently_measured": MISSING,
            "why_not": "same as target_hash — Step 15 residual R6",
        },
        "split_hash": {
            "declared": protocol.get("split_hash"),
            "independently_measured": MISSING,
            "why_not": "same as target_hash — Step 15 residual R6",
        },
        "evaluator_hash": {
            "declared": protocol.get("evaluator_hash") or REGISTERED_EVALUATOR_HASH,
            "independently_measured": measured_evaluator,
            "agrees": measured_evaluator == (protocol.get("evaluator_hash") or REGISTERED_EVALUATOR_HASH),
        },
        "qualification_hash": (bundle.get("qualification") or {}).get("qualification_hash"),
        "calibration_hash": (bundle.get("calibration") or {}).get("calibration_hash"),
        "record_addresses": {key: source["sha256"] for key, source in bundle.sources.items()},
    }


def seed_lists(bundle: StudyBundle) -> Record:
    lock = bundle.get("protocol_lock")
    protocol: Record = {}
    if isinstance(lock, Mapping):
        protocol = dict(lock.get("protocol") or {})
    return {
        "campaign_seeds": protocol.get("campaign_seeds"),
        "deployment_seeds": protocol.get("deployment_seeds"),
        "pilot_seeds": protocol.get("pilot_seeds"),
        "split_seed": (bundle.get("split_manifest") or {}).get("seed"),
        "bootstrap_seed": ((protocol.get("statistics") or {}) or {}).get("bootstrap_seed"),
        "model_fit_seed": 42,
        "docking_seed": 42,
        "note": "no seed was re-rolled and no seed was moved. The deployment seed is the "
        "replication unit for the generation-quality primary; the molecule is not",
    }


def environment_description(bundle: StudyBundle) -> Record:
    environment = bundle.get("environment")
    described: Record = {}
    if isinstance(environment, Mapping):
        described = {
            "host": environment.get("host"),
            "gpu": environment.get("gpu"),
            "interpreters": environment.get("interpreters"),
            "supported_import_lane": environment.get("supported_import_lane"),
            "captured_at": environment.get("generated_at"),
        }
    described["observed_at_package_time"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    described["import_lane_note"] = (
        "the editable install is declined on this host (blocker B4). Every command in this "
        "package runs with PYTHONPATH=src, and a run without it imports a different checkout"
    )
    return described


def exact_commands() -> list[Record]:
    """Every command that produced a number in this package, in order."""
    return [
        {
            "step": "0 — environment",
            "command": "export PYTHONPATH=src",
            "why": "the editable install is declined on this host; without this the import "
            "resolves to a different checkout",
        },
        {
            "step": "1 — baselines",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py baseline "
                "--manifest configs/research/zak-c9r/baselines --output <DERIVED>/baselines/run-13"
            ),
            "why": "the registered baseline table, including the UNAVAILABLE arm",
        },
        {
            "step": "2 — generator survey",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py generators "
                "--output <DERIVED>/generators/run-05 "
                "--pin configs/research/zak-c9r/generator-availability.json"
            ),
            "why": "per-generator availability and throughput, measured not assumed",
        },
        {
            "step": "3 — the nine-arm sweep",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py sweep "
                "--output <DERIVED>/sweep/run-01"
            ),
            "why": "1,350 molecules, 9 arms x 3 seeds x 50 unique. COSTLY: 3,893.2 GPU-occupancy "
            "seconds and 1,350 docking calls",
        },
        {
            "step": "4 — the sweep report",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py sweep-report "
                "--sweep <DERIVED>/sweep/run-01 --pin configs/research/zak-c9r/generation-sweep.json"
            ),
            "why": "the registered primary table and its Holm arithmetic",
        },
        {
            "step": "5 — ablations",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py ablate "
                "--protocol configs/research/zak-c9r/protocol.json "
                "--manifests configs/research/zak-c9r/ablations "
                "--output <DERIVED>/ablations/step16 "
                "--pin configs/research/zak-c9r/ablation-status.json"
            ),
            "why": "the registered comparisons, the configuration diffs and the four NOT_RUN arms",
        },
        {
            "step": "6 — the confirmation branch",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py confirm "
                "--protocol configs/research/zak-c9r/protocol.json "
                "--output <DERIVED>/confirm/step15 "
                "--pin configs/research/zak-c9r/confirmation-status.json"
            ),
            "why": "exits 5, NOT_RUN. It reads no partition and spends nothing. There is no "
            "held-out execution path anywhere in this programme",
        },
        {
            "step": "7 — this package",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py report "
                "--study <DERIVED> --output <PACKAGE>"
            ),
            "why": "the tables, the figures, the manifests and this command list",
        },
        {
            "step": "8 — reproduce it",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py reproduce "
                "--bundle <PACKAGE> --mode metrics"
            ),
            "why": "recomputes every table and statistic from the recorded raw results in a fresh "
            "process. Reruns no costly scientific tool",
        },
        {
            "step": "9 — rerun the declared scientific subset",
            "command": (
                "conda run -n drug-agent python scripts/research/run_target_study.py reproduce "
                "--bundle <PACKAGE> --mode scientific"
            ),
            "why": "re-measures the evaluator digest and re-executes the frozen qualification rule "
            "through the evaluator's own function. Every other member of the declared subset is "
            "reported NOT_RUN with its reason, its measured cost and its unblock",
        },
        {
            "step": "gates",
            "command": "conda run -n drug-agent mypy src && conda run -n drug-agent ruff check "
            "src scripts tests && make test",
            "why": "the three gates every step in this programme is held to",
        },
    ]


# --------------------------------------------------------------------------- #
# the claims, and the blockers
# --------------------------------------------------------------------------- #


def pooled_intramolecular_validity(bundle: StudyBundle) -> Record:
    """Pooled generated-pose validity, added up here rather than read back.

    The decomposed per-arm blocks are summed; molecules with no pose are counted
    apart and never as failures. The adapter's own pass/fail verdict is not read
    at all: it is zero by construction on this receptor (blocker B-15D-1).
    """
    sweep = bundle.get("generation_sweep")
    passing = 0
    measured = 0
    without_a_pose = 0
    if isinstance(sweep, Mapping):
        for payload in dict(sweep.get("pose_quality") or {}).values():
            if not isinstance(payload, Mapping):
                continue
            block = payload.get("intramolecular_validity")
            if isinstance(block, Mapping):
                passing += int(block.get("n_passing_every_check") or 0)
                measured += int(block.get("n_measured") or 0)
            without_a_pose += int(payload.get("n_no_pose") or 0)
    return {
        "n_passing": passing,
        "n_measured": measured,
        "n_without_a_pose": without_a_pose,
        "pass_rate_pct": round(100.0 * passing / measured, 1) if measured else MISSING,
    }


def scientific_claims(bundle: StudyBundle) -> list[Record]:
    """What this package is allowed to say, and what it refuses to say."""
    sweep = bundle.get("generation_sweep")
    primary: Record = dict((sweep or {}).get("primary") or {})
    holm_rejected = int(primary.get("n_arms_rejecting_the_null_under_holm") or 0)
    pooled_validity = pooled_intramolecular_validity(bundle)
    identity = comparator_identity_check(bundle)
    return [
        {
            "claim": "Nine published generators produce chemically distinct output on this target, "
            "separated on four of five standards",
            "status": "REFUSED AS WRITTEN",
            "replaced_by": "the nine arms' medians DIFFER on four endpoints and this package "
            "reports that spread. Nothing tested it: no interval and no test was computed on any "
            f"of those medians, the verdict is {SPREAD_ONLY}, and four of the six preconditions "
            "the pIC50 standard was failed on cannot be asked of a direct computational score at "
            "all. The four standards were assembled in this module after every result was seen, "
            "so they are not a pre-declared set either",
            "evidence": "configs/research/zak-c9r/generation-sweep.json -> unconstrained_"
            "distributions; table-11; table-13",
            "scope": "1,350 molecules, one target, one pocket, one seed triple",
        },
        {
            "claim": "This study carries a conventional-adaptation control to compare against",
            "status": "REFUSED",
            "replaced_by": str(identity.get("finding") or identity.get("note") or "")
            + " There is no conventional-adaptation comparator in this study to compare anything "
            "against, and no arm of the self-evolution track ran either.",
            "evidence": "configs/research/zak-c9r/baseline-results.json; table-05; blocker "
            "B-NO-ADAPTATION-CONTROL",
        },
        {
            "claim": "The pIC50 standard separates the nine methods",
            "status": "REFUSED",
            "replaced_by": "The pIC50 standard FAILED. Three independent estimators were fitted "
            "and none ranks the nine methods at a resolution its own error permits. Chemprop is "
            "measurably worse than a train-mean constant; its apparent spread is the "
            "applicability-domain ordering (Spearman +0.95 against training proximity); the "
            "affine docking map is a monotone image of the Vina column; the parent RandomForest "
            "ties the constant. No pIC50 gate was added, because gating on this would gate on "
            "familiarity and refuse the arms that explore furthest from the training set",
            "evidence": "configs/research/zak-c9r/pic50-independent-estimate.json; table-11",
        },
        {
            "claim": f"{holm_rejected} arms reject the registered null under Holm, so they are "
            "better generators",
            "status": "REFUSED",
            "replaced_by": "Two arms reject the null under Holm and this is NOT a confirmed "
            "ranking. The study is EXPLORATORY: at the locked replication unit (the generation "
            "seed, n=3) the exact paired sign test's smallest attainable two-sided p is 0.25, so "
            "the rejection region is EMPTY and power is 0.0 at every effect size. Three further "
            "facts travel with the table: the top-ranked arm (synflownet) is conditioning=none "
            "and never sees the target; the dominant gate is a docking threshold whose rank "
            "correlation against measured potency on this target is +0.078 (p=0.663) where -1.0 "
            "is perfect; and real human-measured ZAK actives clear the same seven-gate set at 1 "
            "in 6, so the two rejecting arms clear it at 2.6x and 1.9x the rate of the actives it "
            "was derived from",
            "evidence": "generation-sweep.json -> primary.holm, primary.known_active_pass_rate; "
            "protocol lock statistics.power_generation_quality; table-08, table-10",
        },
        {
            "claim": "Self-evolution improves on the parent",
            "status": "NOT_ESTABLISHED",
            "replaced_by": "The confirmation branch is NOT_RUN and B5-self-evolution has never "
            "been executed, so the registered estimator has zero nominees and is undefined. "
            "test_access_count is 0: no held-out compound was read, scored, fitted on or "
            "generated against by anything in this programme",
            "evidence": "configs/research/zak-c9r/confirmation-status.json; table-12",
        },
        {
            "claim": "Any molecule in this study binds ZAK, inhibits it, or is safe",
            "status": "REFUSED",
            "replaced_by": "A computational qualification is not activity. No molecule here has "
            "been synthesised, assayed or shown to do anything. Every endpoint in the benchmark "
            "except the historical activity labels is a COMPUTATIONAL_SCORE",
            "evidence": "table-10 notes; qualification.json limits_of_a_passing_molecule",
        },
        {
            "claim": "A unique mechanism explains any observed difference",
            "status": "REFUSED",
            "replaced_by": "Three of six declared factors have an executed controlled comparison "
            "(training, distillation, compute) and all three are DEVELOPMENT_STAGE. Data "
            "acquisition, search and memory have none. The ablation record's own sentence: no "
            "unique mechanism may be claimed for any observed difference",
            "evidence": "configs/research/zak-c9r/ablation-status.json -> attribution; figure-05",
        },
        {
            "claim": (
                f"{pooled_validity['pass_rate_pct']} % of generated 3D structures are intramolecularly valid"
                if pooled_validity["n_measured"]
                else "generated 3D structural validity"
            ),
            "status": "SUPPORTED" if pooled_validity["n_measured"] else "NOT_ESTABLISHED",
            "evidence": "generation-sweep.json -> pose_quality[arm].intramolecular_validity, "
            "pooled here over all nine arms; figure-04, table-13",
            "scope": f"{pooled_validity['n_passing']} of {pooled_validity['n_measured']} poses. "
            f"The {pooled_validity['n_without_a_pose']} molecules without a pose are the docking "
            "failures and are absent measurements, never zeros. The adapter's own 0-of-150 "
            "verdict is INVALID under blocker B-15D-1 and must not be quoted",
        },
        {
            "claim": "This study says anything about Boltz-2",
            "status": "REFUSED",
            "replaced_by": "Boltz-2 was not run. Its contamination audit is INCONCLUSIVE with the "
            "PubChem ingestion route open, so every negative potency result here is PROVEN for "
            "what could be run and UNTESTED for Boltz-2",
            "evidence": "docs/research/boltz2-contamination-audit.md",
        },
        {
            "claim": "These results transfer to another target, or to another pocket",
            "status": "REFUSED",
            "replaced_by": "One target, one pocket, one exclusion union. Nothing here is evidence "
            "of cross-target transfer, architecture novelty, or superiority to tools that were "
            "not run",
            "evidence": "protocol lock statistics.single_target_scope",
        },
    ]


def open_blockers() -> list[Record]:
    """Every open blocker a reader of this package has to know about."""
    return [
        {
            "id": "B-15D-1",
            "severity": "PLATFORM_WIDE",
            "summary": "PoseBustersAdapter._validate treats every boolean full_report column as a "
            "pass/fail check and one of them (most_extreme_clash_protein) has inverted polarity, "
            "so `passed` is wrong in dock mode for every caller — not only this sweep. It reaches "
            "structure_agent.py posebusters_pass_rate, its posebusters_failed qc_flag, and the "
            "metered pass rate in posebusters_tool.py",
            "consequence": "ANY dock-mode PoseBusters pass rate this platform has ever reported "
            "should be treated as invalid until the adapter is repaired. This package reports the "
            "decomposed per-check numbers instead",
            "status": "OPEN — not fixed here; out of this step's scope",
        },
        {
            "id": "B-15D-2",
            "severity": "SCIENTIFIC",
            "summary": "the frozen seven-gate set admits generated molecules at 2.6x the rate it "
            "admits the real known actives it was derived from: five of the six known ZAK actives "
            "fail it",
            "consequence": "the bar characterises itself. An operator decision is needed about "
            "what the gate set claims to predict BEFORE any threshold is discussed",
            "status": "OPEN — recorded as a finding; no threshold moved",
        },
        {
            "id": "B-15D-4",
            "severity": "PROCESS",
            "summary": "two step owners committed to the same working tree concurrently, so the "
            "history reads 15C -> 15E -> 15D",
            "consequence": "nothing collided — the paths were disjoint and both commits additive "
            "— but concurrent owners need worktrees or explicit serialisation",
            "status": "OPEN — process defect, not a data defect",
        },
        {
            "id": "B-15E-1",
            "severity": "BLOCKER",
            "summary": "the frozen lock's seed-attainability figures are uncorrected for "
            "multiplicity. It declares 6 seeds (paired sign test) and 4 seeds per arm (two-sample "
            "permutation) at alpha=0.05, while the SAME lock registers Holm across nine arms, "
            "whose smallest adjusted threshold is alpha/9 = 0.005555... Under that correction the "
            "answers are 9 seeds and 6 seeds per arm",
            "consequence": "anyone provisioning a follow-up sweep off the lock's number "
            "under-provisions by 50 % and buys two-thirds of the replicates the registered "
            "correction requires. The rejection region stays empty and the operator pays for the "
            "seeds anyway",
            "status": "OPEN — recorded, deliberately not fixed: fixing it means editing "
            "protocol.json and registering a zak-c9r-3 after the counts are known",
            "corrected_figures": {
                "paired_sign_test_seeds": 9,
                "two_sample_permutation_seeds_per_arm": 6,
                "lock_declares": {"paired_sign_test_seeds": 6, "two_sample_permutation_seeds_per_arm": 4},
                "arithmetic": "2^(1-n) <= 0.005555... first holds at n=9; 2/C(2n,n) <= 0.005555... "
                "first holds at n=6 per arm",
            },
        },
        {
            "id": "B-GUARD",
            "severity": "INTEGRITY",
            "summary": "target/confirmation.py and target/ablation.py are outside the evaluator's "
            "guarded set, and so is target/reporting.py, added by this step",
            "consequence": "a candidate could edit the confirmation boundary, the attribution "
            "logic or this reporter without moving the kernel digest",
            "status": "OPEN — deliberately: guarding them moves evaluator_hash away from the "
            "address zak-c9r-2 commits to, which would force a post-hoc protocol version. Unblock: "
            "fold all three into _extra_guarded_files() at the next legitimate re-freeze",
        },
        {
            "id": "B-NO-ADAPTATION-CONTROL",
            "severity": "SCIENTIFIC",
            "summary": "A-B2-standard-adaptation-rf, the registered 'conventional adaptation' "
            "control, IS A-B0-parent-rf. Its declared 3x3 grid, selected by family-grouped 5-fold "
            "cross-validation on the TRAIN partition, chose the parent's own shipped "
            "configuration, and its per-compound absolute-error vector is bit-identical to the "
            "parent's across all 34 evaluated compounds",
            "consequence": "this study has NO conventional-adaptation comparator. The 0.000 "
            "difference between them is an identity, not a measurement, and no statement of the "
            "form 'X is no better than conventional adaptation' can be made from these records. "
            "A real control needs a grid that excludes the parent's configuration, or a different "
            "adaptation recipe entirely",
            "status": "OPEN — recorded as a finding; nothing was refitted and no grid was widened",
        },
        {
            "id": "B-NOMINEES",
            "severity": "BLOCKER",
            "summary": "B5-self-evolution has never been executed, so the confirmation estimator "
            "has zero nominees and every confirmatory ablation is a control for something that "
            "does not exist",
            "consequence": "money alone does not fix this; the arm has to run",
            "status": "OPEN",
        },
        {
            "id": "B-TOKENS",
            "severity": "BLOCKER",
            "summary": "681,319 LLM output tokens for the confirmation branch and 231,319 for the "
            "ablation arms, against an authorisation already 6,319 overdrawn",
            "consequence": "NOT_RUN, not failed",
            "status": "OPEN",
        },
        {
            "id": "B-ACQUISITION",
            "severity": "BLOCKER",
            "summary": "no working ingestion path on this host, and the ACQUISITION_ENABLED "
            "exclusion union cannot be derived in advance",
            "consequence": "ABL-4 is blocked twice over and was labelled POST_HOC_EXPLANATORY in advance",
            "status": "OPEN",
        },
        {
            "id": "B-DOCKCACHE",
            "severity": "BLOCKER",
            "summary": "ABL-3 is deterministic and free in tokens but needs 136 train/development "
            "docking scores that are cached nowhere in this tree",
            "consequence": "the arm stays NOT_RUN under a session constraint, not under a "
            "property of the lock",
            "status": "OPEN",
        },
        {
            "id": "B-VARIANCE",
            "severity": "SCIENTIFIC",
            "summary": "no campaign-to-campaign variance estimate exists in which each campaign "
            "originates its own proposal",
            "consequence": "the by-campaign estimator has no measured spread",
            "status": "OPEN",
        },
        {
            "id": "B4",
            "severity": "OPERATIONAL",
            "summary": "the editable install is declined on this host; PYTHONPATH=src is the supported lane",
            "consequence": "a command run without it imports a different checkout and the numbers "
            "will not match",
            "status": "OPEN",
        },
    ]


def confirmation_boundary_residuals() -> list[Record]:
    """Step 15's residual list, carried into the package unchanged.

    These are the things the confirmation boundary does *not* guarantee. They
    belong in the evidence package because a reader deciding whether to trust
    the ``NOT_RUN`` record needs to know what that record can and cannot prove.
    """
    return [
        {
            "id": "R1'",
            "residual": "a nomination ledger rewritten end to end AND re-anchored by a matching "
            "committed anchor audits clean",
            "precondition": "write access to the ledger plus a committed source-control change",
            "status": "NARROWED, NOT CLOSED — closing it needs a key or an append-only store "
            "outside the repository",
        },
        {
            "id": "R2",
            "residual": "a forged protocol re-registered by editing REGISTERED_PROTOCOL_HASH or "
            "REGISTERED_ANCHOR_PATH in confirmation.py",
            "precondition": "write access to src/",
            "status": "OPEN — both are module constants precisely so this is a visible source "
            "change under review. B-GUARD is still owed",
        },
        {
            "id": "R3",
            "residual": "monkeypatching inside the confirming process",
            "precondition": "in-process code execution",
            "status": "OPEN, explicitly out of scope",
        },
        {
            "id": "R4",
            "residual": "a live object whose views diverge",
            "precondition": "in-process code execution",
            "status": "OPEN — structurally fixed for the protocol by canonicalisation, not claimed "
            "closed in general",
        },
        {
            "id": "R5",
            "residual": "a budget authorisation record is bound, not authenticated",
            "precondition": "write access to a file the operator names",
            "status": "OPEN — the record says so in its own authentication field",
        },
        {
            "id": "R6",
            "residual": "target_hash, data_hash and split_hash are not independently measurable",
            "precondition": "—",
            "status": "OPEN — re-deriving them rebuilds the split, which reads the held-out family "
            "assignment. This package reports them as declared-and-unmeasured, never as agreeing",
        },
        {
            "id": "R7",
            "residual": "confirmation.py is outside the evaluator's guarded set",
            "precondition": "a candidate editing the file",
            "status": "OPEN — B-GUARD, now covering ablation.py and reporting.py too",
        },
        {
            "id": "R8",
            "residual": "a number at a registered cost resource path can reach a ledger row's cost "
            "field in a re-chained entry",
            "precondition": "re-chaining the ledger by hand",
            "status": "OPEN and not closable by inspection; nothing reads cost as a score",
        },
        {
            "id": "R9",
            "residual": "anchor_commit_state trusts the local repository",
            "precondition": "rewriting local git history",
            "status": "OPEN — the anchor's strength is the reviewability of a commit, not the "
            "immutability of a local .git",
        },
        {
            "id": "R10",
            "residual": "REGISTERED_ANCHOR_PATH is derived from __file__'s fourth parent, which is "
            "the repository root only on the PYTHONPATH=src lane",
            "precondition": "installing the package instead",
            "status": "OPEN, and it FAILS CLOSED: off that lane every ledger read off disk is "
            "LEDGER_NOT_EXTERNALLY_ANCHORED and no release is corroborated",
        },
    ]


def corrections_to_the_operator_report() -> list[Record]:
    """Where the operator-facing summary and this package disagree, and who is right.

    The package is required to be consistent with ``report-drug-agent-ver1_0.txt``
    and to correct it where the package finds it wrong. This is that list. It is
    short by design: one substantive correction and one clarification.
    """
    return [
        {
            "location": "report-drug-agent-ver1_0.txt section 5, STATISTICAL STATUS",
            "as_written": "Six seeds would be needed before alpha=0.05 is attainable.",
            "correction": "Six is the SINGLE-COMPARISON figure. The same lock registers Holm "
            "across nine arms, whose smallest adjusted threshold is alpha/9 = 0.005555..., and "
            "2^(1-n) first reaches it at n=9. NINE seeds are needed for the paired sign test, and "
            "SIX per arm for the two-sample permutation test, not the lock's 6 and 4.",
            "why_it_matters": "a follow-up sweep provisioned at six seeds buys two-thirds of the "
            "replicates the registered correction requires and still has an empty rejection "
            "region",
            "blocker": "B-15E-1",
            "status": "CORRECTED IN THE PACKAGE; the operator report is amended in place at this "
            "step and the frozen lock is deliberately NOT edited",
        },
        {
            "location": "report-drug-agent-ver1_0.txt section 1, headline table",
            "as_written": "the pIC50* column is printed first in the five-standard table",
            "correction": "the column is invalid and section 3 says so. This package does not "
            "print a pIC50 ranking at all: table-11 carries the standard with verdict FAILED and "
            "no per-method values, so the number cannot be quoted out of context.",
            "why_it_matters": "a table column that a reader has to be told to ignore is a column "
            "that gets quoted",
            "blocker": None,
            "status": "DIVERGENCE BY DESIGN, recorded here rather than silently resolved",
        },
        {
            "location": "report-drug-agent-ver1_0.txt section 2, THE FOUR STANDARDS THAT WORKED",
            "as_written": "No method wins on more than one axis.",
            "correction": "flowmol3 tops TWO of the four axes — QED 0.779 and SA 2.18 — and is "
            "second on hERG behind prexsyn. prexsyn tops hERG and synflownet tops docking. The "
            "same section's own KEY TRADE-OFF paragraph already said so. No arm tops all four, "
            "and topping an axis is not winning it: no test was applied to any of these medians.",
            "why_it_matters": "the sentence was false against the table printed directly beneath "
            "it, and the same sentence was carried into the package as a table-13 note. The note "
            "is now DERIVED from the medians in the table rather than written beside them",
            "blocker": None,
            "status": "CORRECTED IN THE PACKAGE and amended in the operator report at this step",
        },
        {
            "location": "report-drug-agent-ver1_0.txt section 1 and section 2 headings",
            "as_written": "FOUR OF FIVE STANDARDS SUCCEEDED / THE FOUR STANDARDS THAT WORKED",
            "correction": "four of five standards produced a per-arm spread. None of them was "
            "TESTED. The package's verdict for all four is now "
            f"{SPREAD_ONLY}: no interval, no test, medians pooled over molecules while the "
            "registered replication unit is the seed, and four of the six preconditions the pIC50 "
            "standard was failed on cannot even be asked of a direct computational score.",
            "why_it_matters": "one standard was held to six measured preconditions and four were "
            "held to a hard-coded string. That asymmetry inflated the flagship claim",
            "blocker": None,
            "status": "CORRECTED IN THE PACKAGE and amended in the operator report at this step",
        },
        {
            "location": "report-drug-agent-ver1_0.txt section 4, Track A comparators",
            "as_written": "A-B2-standard-adaptation-rf is reported as a comparator with a point "
            "difference of 0.000 against the parent.",
            "correction": "A-B2 IS the parent. Its grid search selected the parent's own shipped "
            "configuration, and its per-compound absolute-error vector is BIT-IDENTICAL to "
            "A-B0-parent-rf's across all 34 evaluated compounds. The 'conventional adaptation' "
            "control is not a control; the study has no standard-adaptation comparator at all.",
            "why_it_matters": "a comparator that is bit-identical to its own baseline cannot "
            "bound anything. Rendering it as an ordinary inconclusive point difference invites a "
            "reader to treat 'no difference from conventional adaptation' as a measured finding",
            "blocker": "B-NO-ADAPTATION-CONTROL",
            "status": "CORRECTED IN THE PACKAGE; the arm is reported as NOT A COMPARATOR",
        },
    ]


# --------------------------------------------------------------------------- #
# the package
# --------------------------------------------------------------------------- #


def _raw_metric_exports(bundle: StudyBundle, directory: Path) -> list[Record]:
    """Identity-free raw metric files, and addresses for the ones that are not.

    Records carrying per-compound identities are exported as *addresses* — path,
    content address, row count — and stay with their custodian. The work order
    permits exactly this: a reproduction guide may specify custodian-controlled
    evaluation rather than publicly copying labels.
    """
    directory.mkdir(parents=True, exist_ok=True)
    out: list[Record] = []

    # 1. per-arm, per-seed qualification counts, scaffold ids replaced by their
    #    own content address so the unique-count is preserved and the structure
    #    is not published.
    rows: list[Record] = []
    for name, cell in sorted(bundle.sweep_cells.items()):
        arm_dir = bundle.study / "sweep" / "run-01" / "cells" / name
        for seed_dir in sorted(arm_dir.glob("seed-*")):
            payload = _read_json(seed_dir / "qualification.json")
            if not isinstance(payload, Mapping):
                continue
            for row in payload.get("rows") or []:
                gates = dict(row.get("gates") or {})
                rows.append(
                    {
                        "arm": name,
                        "seed": seed_dir.name.replace("seed-", ""),
                        "molecule_id": row.get("molecule_id"),
                        "scaffold_key": hashlib.sha256(
                            str(row.get("scaffold_id") or "").encode()
                        ).hexdigest()[:16],
                        "qualifies": bool(row.get("qualifies")),
                        "failed_gates": "|".join(row.get("failed_gates") or []),
                        "vina_affinity": (gates.get("structure_score") or {}).get("value"),
                        "herg": (gates.get("safety") or {}).get("value"),
                        "sa_score": (gates.get("synthetic_accessibility") or {}).get("value"),
                    }
                )
        del cell
    if rows:
        path = directory / "raw-qualification-rows.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _cell(v) for k, v in row.items()})
        out.append(
            {
                "path": path.name,
                "kind": "RAW_METRIC",
                "n_rows": len(rows),
                "sha256": sha256_file(path),
                "note": "one row per generated molecule. scaffold_key is the sha256 prefix of the "
                "scaffold id: the unique-count is exactly preserved and no chemical structure is "
                "published. Generated molecules are this study's own output and carry no label",
            }
        )

    # 2. per-cell cost records, the input to the independently re-added total.
    costs: list[Record] = []
    for name in sorted(bundle.sweep_cells):
        arm_dir = bundle.study / "sweep" / "run-01" / "cells" / name
        for seed_dir in sorted(arm_dir.glob("seed-*")):
            payload = _read_json(seed_dir / "generation.json")
            if isinstance(payload, Mapping):
                cost = dict(payload.get("cost") or {})
                costs.append(
                    {
                        "arm": name,
                        "seed": seed_dir.name.replace("seed-", ""),
                        "phase": "GENERATION",
                        "gpu_occupancy_seconds": cost.get("gpu_seconds"),
                        "wall_seconds": cost.get("wall_seconds"),
                        "telemetry": cost.get("gpu_telemetry"),
                    }
                )
        scoring = _read_json(arm_dir / "scoring.json")
        if isinstance(scoring, Mapping):
            cost = dict(scoring.get("cost") or {})
            costs.append(
                {
                    "arm": name,
                    "seed": "all",
                    "phase": "SCORING",
                    "gpu_occupancy_seconds": cost.get("endpoint_gpu_seconds"),
                    "wall_seconds": cost.get("wall_seconds"),
                    "telemetry": cost.get("endpoint_gpu_telemetry"),
                }
            )
    if costs:
        path = directory / "raw-cell-costs.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(costs[0].keys()))
            writer.writeheader()
            for row in costs:
                writer.writerow({k: _cell(v) for k, v in row.items()})
        out.append(
            {
                "path": path.name,
                "kind": "RAW_COST",
                "n_rows": len(costs),
                "sha256": sha256_file(path),
                "note": "the per-cell figures the package's cost total is re-added from",
            }
        )

    # 3. custodian-held records: addresses only.
    custodian: list[Record] = []
    for run_dir in (
        sorted((bundle.study / "baselines").glob("run-*")) if (bundle.study / "baselines").is_dir() else []
    ):
        for arm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            result = arm_dir / "result.json"
            if not result.exists():
                continue
            payload = _read_json(result) or {}
            errors = dict((payload.get("endpoint") or {}).get("absolute_errors") or {})
            custodian.append(
                {
                    "path": _display_path(result, bundle.repo),
                    "sha256": sha256_file(result),
                    "n_error_rows": len(errors),
                    "why_not_copied": "keyed by compound identity on the development partition; "
                    "the package reports the aggregate MAE and leaves the keys with the custodian",
                }
            )
    if custodian:
        path = directory / "custodian-held-records.json"
        path.write_text(json.dumps(custodian, indent=2) + "\n")
        out.append(
            {
                "path": path.name,
                "kind": "CUSTODIAN_ADDRESSES",
                "n_rows": len(custodian),
                "sha256": sha256_file(path),
                "note": "addresses only. A custodian with the store can recompute every aggregate "
                "in this package; nobody without it receives a compound identity",
            }
        )
    return out


#: What the ``independent_recomputation`` block actually is. It runs
#: ``reporting.recompute_*`` — the SAME functions, over the SAME files, that
#: produced the block recorded in ``manifest.json`` — in a second process. An
#: agreeing result shows the pipeline is deterministic and its inputs are
#: reachable. It is not a reimplementation and it is not a second opinion, and the
#: key it is filed under says "independent", so the block says what it is.
RECOMPUTATION_CLASS = "DETERMINISM_CHECK_NOT_AN_INDEPENDENT_REIMPLEMENTATION"

RECOMPUTATION_WHAT_IT_IS = (
    "the same functions (reporting.recompute_qualifying_scaffold_yield, recompute_track_a_mae, "
    "recompute_sweep_cost_total) over the same files, run again in a second process. Agreement "
    "shows the arithmetic is deterministic and the raw records are reachable and unchanged"
)

RECOMPUTATION_WHAT_IT_IS_NOT = (
    "a second opinion. Nothing here re-derives a number by a different method, in different code, "
    "or from a different source: `reproduce --mode metrics` reports a block that is BYTE-IDENTICAL "
    "to the one already in manifest.json, because it is the same computation. A defect in "
    "recompute_* reproduces perfectly and is invisible to this check. The field name "
    "'independent_recomputation' overstates it and is kept only so existing readers still find the "
    "block"
)


def _recomputation_block(yields: Record, mae: Record, costs: Record, crosschecks: Record) -> Record:
    """The recomputation block, labelled for what it is."""
    return {
        "class": RECOMPUTATION_CLASS,
        "is_independent": False,
        "what_it_is": RECOMPUTATION_WHAT_IT_IS,
        "what_it_is_NOT": RECOMPUTATION_WHAT_IT_IS_NOT,
        "qualifying_scaffold_yield": yields,
        "track_a_mae": mae,
        "sweep_cost_total": costs,
        "crosschecks": crosschecks,
    }


def build_package(
    study: Path | str,
    output: Path | str,
    *,
    configs: Path | str | None = None,
    docs: Path | str | None = None,
    repo: Path | str | None = None,
) -> Record:
    """Write the complete evidence package and return its own record."""
    bundle = load_study(study, configs=configs, docs=docs, repo=repo)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    tables = build_tables(bundle)
    table_records = [write_table_csv(table, out / "tables" / f"{table['id']}.csv") for table in tables]
    figure_records = build_figures(bundle, out / "figures")
    raw_records = _raw_metric_exports(bundle, out / "raw")

    yields = recompute_qualifying_scaffold_yield(bundle.study)
    mae = recompute_track_a_mae(bundle.study)
    costs = recompute_sweep_cost_total(bundle.study)
    crosschecks = {
        "qualifying_scaffold_yield_table": crosscheck_yield_table(bundle, yields),
        "track_a_mae_table": crosscheck_mae_table(bundle, mae),
        "sweep_cost_total": crosscheck_cost_total(bundle, costs),
    }

    manifest = artifact_dependency_manifest(bundle)
    try:
        from .generation_sweep import verify_test_access_count

        access = verify_test_access_count(bundle.repo)
    except Exception:  # pragma: no cover - the scanner is unavailable off the lane
        access = {"n_records": None, "all_zero": None, "values": [], "note": "SCANNER_UNAVAILABLE"}

    package: Record = {
        "record_kind": "research_evidence_package",
        "contract_version": "1.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_by": "scripts/research/run_target_study.py report",
        "output_directory": str(out),
        "study_directory": str(bundle.study),
        "protocol_version": REGISTERED_PROTOCOL_VERSION,
        "classification": "EXPLORATORY",
        "classification_reason": (
            "at the locked replication unit the exact test's discrete support does not reach "
            "alpha, so the rejection region is empty and power is 0.0 at every effect size. No "
            "analysis of these records can make this confirmatory; only more seeds can"
        ),
        "held_out_partition": {
            "test_access_count": 0,
            "records_declaring_it": access.get("n_records"),
            "all_zero": access.get("all_zero"),
            "statement": "no held-out compound was read, scored, fitted on or generated against "
            "by anything in this programme, and this package reads none either",
        },
        "sources": bundle.sources,
        "tables": table_records,
        "figures": figure_records,
        "raw": raw_records,
        "independent_recomputation": _recomputation_block(yields, mae, costs, crosschecks),
        "comparator_identity": comparator_identity_check(bundle),
        "addresses": registered_addresses(bundle),
        "git": git_state(bundle.repo),
        "commands": exact_commands(),
        "prompts": prompt_identities(bundle.repo),
        "models": model_identities(bundle),
        "seeds": seed_lists(bundle),
        "environment": environment_description(bundle),
        "artifact_dependency_manifest": manifest,
        "third_party": list(THIRD_PARTY),
        "scientific_subset": list(SCIENTIFIC_SUBSET),
        "claims": scientific_claims(bundle),
        "open_blockers": open_blockers(),
        "confirmation_boundary_residuals": confirmation_boundary_residuals(),
        "corrections_to_the_operator_report": corrections_to_the_operator_report(),
    }
    package["reference_validation"] = validate_references(bundle, package)

    (out / "manifest.json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    tables_by_id = {table["id"]: table for table in tables}
    (out / "tables.json").write_text(json.dumps(tables_by_id, indent=2, sort_keys=True) + "\n")
    report_text = render_report(package, tables)
    (out / "research-report.md").write_text(report_text)

    leaked = scan_for_compound_identities(
        report_text + json.dumps(package), allow=_permitted_accessions(bundle)
    )
    package["label_safety"] = {
        "compound_identities_in_the_package": leaked,
        "clean": not leaked,
        "allowed": sorted(_permitted_accessions(bundle)),
        "rule": "counts and statistics only; a per-compound label never appears",
    }
    (out / "manifest.json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    package["manifest_sha256"] = sha256_file(out / "manifest.json")
    return package


def compact_record(package: Mapping[str, Any]) -> Record:
    """The committed summary of a package whose full form is gitignored.

    The package itself lives under the derived store, which is not in source
    control. This is what a later reader finds *in* the repository: the
    addresses, the claims with their statuses, the blockers, the recomputation
    verdicts and the commands that rebuild the rest. It carries no table bodies
    and no raw rows, so it stays small and it cannot drift into being a second
    copy of the evidence.
    """
    return {
        "record_kind": "evidence_package_index",
        "contract_version": "1.0.0",
        "generated_at": package["generated_at"],
        "generated_by": package["generated_by"],
        "protocol_version": package["protocol_version"],
        "classification": package["classification"],
        "classification_reason": package["classification_reason"],
        "held_out_partition": package["held_out_partition"],
        "package_location": package["output_directory"],
        "package_location_note": (
            "the full package — research-report.md, the CSV tables, the figures, the raw metric "
            "exports and manifest.json — is written under the gitignored derived store. Rebuild it "
            "with the `report` command in `commands`; it is deterministic apart from its timestamp"
        ),
        "manifest_sha256": package.get("manifest_sha256"),
        "addresses": package["addresses"],
        "seeds": package["seeds"],
        "tables": [
            {"id": entry["table_id"], "n_rows": entry["n_rows"], "sha256": entry["sha256"]}
            for entry in package["tables"]
        ],
        "figures": [
            {"figure": entry.get("figure"), "status": entry.get("status"), "sha256": entry.get("sha256")}
            for entry in package["figures"]
        ],
        "independent_recomputation": {
            "class": RECOMPUTATION_CLASS,
            "is_independent": False,
            "what_it_is_NOT": RECOMPUTATION_WHAT_IT_IS_NOT,
            "crosschecks": {
                name: {
                    "reproduces": check.get("reproduces"),
                    "status": check.get("status"),
                    "disagreements": check.get("disagreements", []),
                }
                for name, check in package["independent_recomputation"]["crosschecks"].items()
            },
        },
        "comparator_identity": package["comparator_identity"],
        "reference_validation": package["reference_validation"],
        "label_safety": package["label_safety"],
        "artifact_dependency_manifest": {
            "n_nodes": package["artifact_dependency_manifest"]["n_nodes"],
            "n_edges": package["artifact_dependency_manifest"]["n_edges"],
            "n_dangling": package["artifact_dependency_manifest"]["n_dangling"],
            "dangling": package["artifact_dependency_manifest"]["dangling"],
            "nodes": [
                {
                    "id": node["id"],
                    "kind": node["kind"],
                    "path": node["path"],
                    "present": node["present"],
                    "measured_address": node["measured_address"],
                }
                for node in package["artifact_dependency_manifest"]["nodes"]
            ],
        },
        "claims": package["claims"],
        "open_blockers": package["open_blockers"],
        "confirmation_boundary_residuals": package["confirmation_boundary_residuals"],
        "corrections_to_the_operator_report": package["corrections_to_the_operator_report"],
        "third_party": package["third_party"],
        "scientific_subset": package["scientific_subset"],
        "commands": package["commands"],
        "models": package["models"],
        "git": package["git"],
    }


def _permitted_accessions(bundle: StudyBundle) -> set[str]:
    """The accessions that are not compounds: the target, and document ids."""
    permitted: set[str] = set()
    manifest = bundle.get("target_manifest")
    if isinstance(manifest, Mapping):
        target = dict(manifest.get("chembl_target") or {})
        for key in ("target_chembl_id", "id", "chembl_id"):
            value = target.get(key)
            if isinstance(value, str):
                permitted.add(value)
    permitted.add("CHEMBL3886")
    return permitted


def render_report(package: Mapping[str, Any], tables: Sequence[Mapping[str, Any]]) -> str:
    """The human-readable report, written from the same record the CLI returns."""
    lines: list[str] = []
    add = lines.append
    add("# Research evidence package — ZAK/MAP3K20 pocket C9R, protocol `zak-c9r-2`")
    add("")
    add(f"**Generated:** {package['generated_at']} by `{package['generated_by']}`  ")
    add(f"**Classification:** **{package['classification']}**  ")
    add(f"**Commit:** `{(package.get('git') or {}).get('commit')}`  ")
    add(f"**Protocol:** `{package['addresses']['protocol_hash']['declared']}`  ")
    add(
        f"**Evaluator:** `{package['addresses']['evaluator_hash']['declared']}` "
        f"(independently re-measured: `{package['addresses']['evaluator_hash']['independently_measured']}`)  "
    )
    add(f"**`test_access_count`: {package['held_out_partition']['test_access_count']}** — ")
    add(package["held_out_partition"]["statement"] + ".")
    add("")
    add("---")
    add("")
    add("## 0. Read this before any table")
    add("")
    reason = str(package["classification_reason"])
    add(f"**The study is {package['classification']}.** {reason[:1].upper() + reason[1:]}.")
    add("")
    add(
        "**The confirmation branch is `NOT_RUN`.** `B5-self-evolution` has never been executed, so "
        "the registered estimator has zero nominees and is undefined. There is no code path "
        "anywhere in this programme that spends `test_access_count`, and none was added here."
    )
    add("")
    add(
        "**The substantive result is the nine-generator benchmark.** Four standards — hERG, "
        "synthetic accessibility, QED and docking — show a SPREAD across the nine arms' medians, "
        "and nothing tested whether that spread is a separation: see table-11, verdict "
        f"`{SPREAD_ONLY}`. The fifth, pIC50, **FAILED** six measured preconditions on three "
        "independent estimators, and that failure is reported as a result rather than as a "
        "missing column. The five standards were assembled in this module after every result was "
        "seen; no registered record declares a set of standards."
    )
    add("")
    add(
        "**The top-ranked arm never sees the target.** `synflownet` is `conditioning=none`. The "
        "gate it cleared is dominated by a docking threshold, and this programme measured "
        "docking's rank correlation against real potency on this target at **+0.078** "
        "(development, p=0.663) where **-1.0** would be perfect. Real human-measured ZAK actives "
        "clear the same seven-gate set at **1 in 6**; the two Holm-rejecting arms clear it at "
        "**2.6x** and **1.9x** that rate. A generated arm clearing a bar derived from known "
        "actives faster than the actives themselves characterises the bar."
    )
    add("")
    identity = package.get("comparator_identity") or {}
    if identity.get("is_the_baseline_it_controls_for") is True:
        add(
            "**The conventional-adaptation control IS the parent.** "
            f"`{identity['arm']}` was registered as the standard-adaptation comparator for "
            f"`{identity['baseline']}`. Its declared 3x3 grid, selected by family-grouped 5-fold "
            "cross-validation on the TRAIN partition, chose the parent's own shipped "
            f"configuration `{identity.get('selected_configuration')}`, and its per-compound "
            "absolute-error vector is **bit-identical** to the parent's across all "
            f"{identity.get('n_evaluated_compounds')} evaluated compounds "
            f"(`{(identity.get('per_compound_error_vector_addresses') or {}).get(identity['arm'])}`). "
            "A comparator that is bit-identical to its own baseline is not a comparator: this "
            "study has **no** conventional-adaptation control, the 0.000 difference between them "
            "is an identity rather than a result, and no statement of the form *X is no better "
            "than conventional adaptation* can be made from these records. See blocker "
            "`B-NO-ADAPTATION-CONTROL` and table-05."
        )
        add("")
    add("## 1. Claims, and the ones this package refuses")
    add("")
    add("| claim | status | what replaces it |")
    add("| --- | :-: | --- |")
    for claim in package["claims"]:
        replacement = claim.get("replaced_by") or claim.get("scope") or ""
        add(f"| {claim['claim']} | **{claim['status']}** | {replacement} |")
    add("")
    add("## 2. Tables")
    add("")
    for table in tables:
        add(f"### `{table['id']}` — {table['title']}")
        add("")
        add(f"*Denominator:* {table['denominator']}. *Rows:* {table['n_rows']}.")
        add("")
        for note in table["notes"]:
            add(f"> {note}")
            add("")
        add(f"Full table: `tables/{table['id']}.csv`.")
        add("")
    add("## 3. Figures")
    add("")
    for figure in package["figures"]:
        add(f"### `{figure.get('figure')}` — {figure.get('status')}")
        add("")
        if figure.get("caption"):
            add(figure["caption"])
            add("")
    add("## 4. Recomputation — a determinism check, not a second opinion")
    add("")
    add(
        f"**Class: `{package['independent_recomputation']['class']}`.** It is "
        + str(package["independent_recomputation"]["what_it_is"])
        + ". **It is not** "
        + str(package["independent_recomputation"]["what_it_is_NOT"])
        + "."
    )
    add("")
    checks = package["independent_recomputation"]["crosschecks"]
    add("| what was recomputed | from | reproduces |")
    add("| --- | --- | :-: |")
    add(
        f"| qualifying-scaffold yield table, all nine arms | raw per-molecule gate verdicts | "
        f"**{checks['qualifying_scaffold_yield_table'].get('reproduces')}** |"
    )
    add(
        f"| Track A MAE table | raw per-arm absolute-error tables (custodian store) | "
        f"**{checks['track_a_mae_table'].get('reproduces')}** |"
    )
    add(
        f"| sweep GPU/wall cost totals | raw per-cell cost records | "
        f"**{checks['sweep_cost_total'].get('reproduces')}** |"
    )
    add("")
    add(
        "The yield recount is set arithmetic written in `target/reporting.py`, not a call to the "
        "evaluator's own `qualifying_scaffolds`. Agreement is therefore evidence about the "
        "registered table rather than evidence that one function agrees with itself; the "
        "evaluator's own function is re-executed separately by `reproduce --mode scientific`."
    )
    add("")
    add("## 5. Reproduction")
    add("")
    add("```bash")
    for command in package["commands"]:
        add(f"# {command['step']}: {command['why']}")
        add(command["command"])
    add("```")
    add("")
    add(f"Seeds: `{json.dumps(package['seeds'])}`")
    add("")
    add("## 6. Open blockers")
    add("")
    add("| id | severity | summary | status |")
    add("| --- | --- | --- | --- |")
    for blocker in package["open_blockers"]:
        add(f"| **{blocker['id']}** | {blocker['severity']} | {blocker['summary']} | {blocker['status']} |")
    add("")
    add("## 7. What the confirmation boundary does NOT guarantee")
    add("")
    add("| id | residual | status |")
    add("| --- | --- | --- |")
    for residual in package["confirmation_boundary_residuals"]:
        add(f"| {residual['id']} | {residual['residual']} | {residual['status']} |")
    add("")
    add("## 8. Corrections to the operator-facing summary")
    add("")
    for correction in package["corrections_to_the_operator_report"]:
        add(f"**{correction['location']}**")
        add("")
        add(f"- *As written:* {correction['as_written']}")
        add(f"- *Correction:* {correction['correction']}")
        add(f"- *Why it matters:* {correction['why_it_matters']}")
        add(f"- *Status:* {correction['status']}")
        add("")
    add("## 9. Dependencies, licences and restricted downloads")
    add("")
    add("| name | kind | access | licence | restricted |")
    add("| --- | --- | --- | --- | :-: |")
    for entry in package["third_party"]:
        add(
            f"| {entry['name']} | {entry['kind']} | {entry['access']} | {entry['licence']} | "
            f"{'**YES**' if entry['restricted'] else 'no'} |"
        )
    add("")
    add("## 10. Reference validation")
    add("")
    validation = package["reference_validation"]
    add(f"{validation['n_references_checked']} references checked; **{validation['n_dangling']} dangling**.")
    add("")
    for entry in validation["dangling"]:
        add(f"- `{entry['class']}` **{entry['id']}** → `{entry['declared_path']}`: {entry['why']}")
    add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# reproduction
# --------------------------------------------------------------------------- #


def reproduce_metrics(bundle_dir: Path | str, *, study: Path | str | None = None) -> Record:
    """Recompute every table and statistic from recorded raw results.

    Runs no generator, no docking, no fitting and no LLM. Everything here is
    arithmetic over records that already exist, performed in whatever process
    the caller started — which is the point: the CLI runs it in a fresh one.
    """
    directory = Path(bundle_dir)
    manifest = _read_json(directory / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            f"{directory} carries no manifest.json; it is not a package this command can "
            f"reproduce. Build one with `report --study DIR --output DIR` first",
            field_name="manifest.json",
        )
    study_dir = Path(study) if study is not None else Path(str(manifest.get("study_directory")))
    rebuilt = load_study(
        study_dir,
        configs=None,
        docs=None,
        repo=REPO,
    )
    tables = build_tables(rebuilt)
    recorded = _read_json(directory / "tables.json")
    recorded_tables = dict(recorded) if isinstance(recorded, Mapping) else {}

    comparisons: list[Record] = []
    for table in tables:
        before = recorded_tables.get(table["id"])
        if not isinstance(before, Mapping):
            comparisons.append(
                {
                    "table": table["id"],
                    "status": "NOT_IN_PACKAGE",
                    "agrees": False,
                    "n_rows_now": table["n_rows"],
                }
            )
            continue
        same_rows = canonical_json(before.get("rows")) == canonical_json(table["rows"])
        comparisons.append(
            {
                "table": table["id"],
                "status": "COMPARED",
                "agrees": same_rows,
                "n_rows_recorded": before.get("n_rows"),
                "n_rows_now": table["n_rows"],
                "denominator": table["denominator"],
            }
        )

    yields = recompute_qualifying_scaffold_yield(rebuilt.study)
    mae = recompute_track_a_mae(rebuilt.study)
    costs = recompute_sweep_cost_total(rebuilt.study)
    crosschecks = {
        "qualifying_scaffold_yield_table": crosscheck_yield_table(rebuilt, yields),
        "track_a_mae_table": crosscheck_mae_table(rebuilt, mae),
        "sweep_cost_total": crosscheck_cost_total(rebuilt, costs),
    }
    missing_sources = [key for key, source in rebuilt.sources.items() if not source.get("present")]
    shipped = verify_shipped_files(directory, manifest)
    return {
        "record_kind": "reproduction_metrics",
        "contract_version": "1.0.0",
        "mode": "metrics",
        "reran_any_costly_scientific_tool": False,
        "what_it_did_not_run": [entry["id"] for entry in SCIENTIFIC_SUBSET if not entry.get("runnable_here")],
        "bundle": str(directory),
        "study_directory": str(rebuilt.study),
        "process": {"pid": os.getpid(), "python": sys.version.split()[0]},
        "tables": comparisons,
        "n_tables": len(comparisons),
        "n_tables_agreeing": sum(1 for c in comparisons if c["agrees"]),
        "all_tables_reproduce": all(c["agrees"] for c in comparisons),
        "independent_recomputation": _recomputation_block(yields, mae, costs, crosschecks),
        "all_crosschecks_reproduce": all(check.get("reproduces") is True for check in crosschecks.values()),
        "shipped_files": shipped,
        "all_declared_files_verify": shipped["all_declared_files_verify"],
        "unreachable_sources": missing_sources,
        "what_this_mode_checked": [
            "every table in tables.json was rebuilt from the study records and compared row for row",
            "the qualifying-scaffold yield, the Track A mean absolute errors and the sweep cost "
            "total were recomputed from the raw records — by the SAME functions that produced the "
            "manifest's block, so that is a determinism check and not a second opinion",
            "every table CSV, figure and raw export the manifest declares was re-hashed and "
            "compared with its declared sha256",
        ],
        "what_this_mode_did_NOT_check": [
            "manifest.json's own content address: it is the document making the claims",
            "any file in the bundle the manifest does not declare",
            "anything scientific. No costly tool was rerun; see `reproduce --mode scientific`",
        ],
        "note": "a source this process could not reach is reported here and its table is NOT "
        "reported as agreeing. Metrics mode recomputes; it never imputes",
    }


def verify_shipped_files(bundle_dir: Path | str, manifest: Mapping[str, Any]) -> Record:
    """Re-hash every file the package DECLARES and compare it with the declared address.

    ``reproduce --mode metrics`` used to read ``tables.json`` and nothing else. It
    therefore exited 0 over a bundle whose published CSVs had been falsified and
    whose declared figure and raw export had been deleted: the CSVs are written
    from ``tables.json`` at build time and never read back, the figures and raw
    exports are never read at all, and not one declared ``sha256`` was checked.
    A reproduction that reports success over a bundle it did not open is worse
    than no reproduction, because it launders the falsification.

    Every table, figure and raw export the manifest declares is re-hashed here.
    A file that is absent is ``ABSENT``; one whose bytes moved is
    ``DIGEST_MISMATCH``; and either makes ``all_declared_files_verify`` false, on
    which the command exits non-zero.
    """
    directory = Path(bundle_dir)
    checks: list[Record] = []

    def check(kind: str, identifier: Any, relative: Any, declared: Any, root: Path) -> None:
        if not relative or not declared:
            checks.append(
                {
                    "kind": kind,
                    "id": identifier,
                    "path": relative,
                    "status": "NOT_DECLARED_WITH_AN_ADDRESS",
                    "verifies": False,
                    "why": "the manifest entry carries no path or no sha256, so nothing about "
                    "this file can be checked",
                }
            )
            return
        path = root / str(relative)
        if not path.is_file():
            checks.append(
                {
                    "kind": kind,
                    "id": identifier,
                    "path": str(relative),
                    "status": "ABSENT",
                    "verifies": False,
                    "declared_sha256": declared,
                    "why": "the package declares this file and the bundle does not carry it",
                }
            )
            return
        measured = sha256_file(path)
        agrees = measured == declared
        checks.append(
            {
                "kind": kind,
                "id": identifier,
                "path": str(relative),
                "status": "VERIFIED" if agrees else "DIGEST_MISMATCH",
                "verifies": agrees,
                "declared_sha256": declared,
                "measured_sha256": measured,
            }
        )

    tables_dir = directory / "tables"
    figures_dir = directory / "figures"
    raw_dir = directory / "raw"
    for entry in manifest.get("tables") or []:
        if isinstance(entry, Mapping):
            check(
                "TABLE", entry.get("table_id"), entry.get("path"), entry.get("sha256"), tables_dir
            )
    for entry in manifest.get("figures") or []:
        if isinstance(entry, Mapping) and entry.get("status") == "WRITTEN":
            check(
                "FIGURE", entry.get("figure"), entry.get("figure"), entry.get("sha256"), figures_dir
            )
    for entry in manifest.get("raw") or []:
        if isinstance(entry, Mapping):
            check("RAW", entry.get("path"), entry.get("path"), entry.get("sha256"), raw_dir)

    failed = [c for c in checks if not c["verifies"]]
    return {
        "n_declared": len(checks),
        "n_verified": len(checks) - len(failed),
        "n_failed": len(failed),
        "all_declared_files_verify": not failed,
        "failures": failed,
        "checks": checks,
        "what_was_checked": (
            "every table CSV, every figure recorded as WRITTEN and every raw export the manifest "
            "declares, re-hashed from the bundle on disk and compared with the sha256 the manifest "
            "recorded for it"
        ),
        "what_was_NOT_checked": (
            "manifest.json's own address (it is the document making the claims), the custodian-held "
            "records outside the bundle, and anything the manifest does not declare. A file added "
            "to the bundle that the manifest never mentioned is not detected here"
        ),
    }


def reproduce_scientific(
    bundle_dir: Path | str,
    *,
    study: Path | str | None = None,
    allow_regeneration: bool = False,
) -> Record:
    """Rerun the declared scientific subset, and report every member it cannot.

    Two members run here and they are real re-executions, not re-reads: the
    evaluator's kernel digest is re-measured from the guarded files, and the
    frozen qualification rule is re-executed through the evaluator's own
    ``qualifying_scaffolds`` over the stored per-molecule verdicts. Everything
    else in :data:`SCIENTIFIC_SUBSET` is ``NOT_RUN`` with its reason, its
    measured cost and its unblock — including with ``allow_regeneration``, which
    is necessary and nowhere near sufficient.
    """
    directory = Path(bundle_dir)
    manifest = _read_json(directory / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise _fail(
            ErrorCategory.MISSING_FIELD,
            f"{directory} carries no manifest.json",
            field_name="manifest.json",
        )
    study_dir = Path(study) if study is not None else Path(str(manifest.get("study_directory")))
    bundle = load_study(study_dir, repo=REPO)

    results: list[Record] = []
    for entry in SCIENTIFIC_SUBSET:
        if not entry.get("runnable_here"):
            results.append(
                {
                    **entry,
                    "status": "NOT_RUN",
                    "allow_regeneration_supplied": allow_regeneration,
                    "still_not_run_because": entry.get("reason"),
                    "note": "allow_regeneration is necessary and not sufficient: every member "
                    "below is blocked on at least one further condition that a flag cannot open",
                }
            )
            continue
        if entry["id"] == "evaluator-kernel-digest":
            measured = _measured_evaluator_hash()
            if measured is None:  # pragma: no cover - off the supported lane
                results.append({**entry, "status": "FAILED", "error": "kernel digest unavailable"})
                continue
            declared = REGISTERED_EVALUATOR_HASH
            results.append(
                {
                    **entry,
                    "status": "RERUN",
                    "measured": measured,
                    "declared": declared,
                    "agrees": measured == declared,
                }
            )
        elif entry["id"] == "qualification-rule-replay":
            results.append({**entry, **_replay_qualification_rule(bundle)})
    return {
        "record_kind": "reproduction_scientific",
        "contract_version": "1.0.0",
        "mode": "scientific",
        "bundle": str(directory),
        "study_directory": str(bundle.study),
        "allow_regeneration": allow_regeneration,
        "declared_subset_size": len(SCIENTIFIC_SUBSET),
        "n_rerun": sum(1 for r in results if r.get("status") == "RERUN"),
        "n_not_run": sum(1 for r in results if r.get("status") == "NOT_RUN"),
        "results": results,
        "all_rerun_members_agree": all(
            r.get("agrees") is not False for r in results if r.get("status") == "RERUN"
        ),
        "note": "nothing here regenerated, re-docked or read a partition. test_access_count is "
        "unchanged at 0",
    }


def _replay_qualification_rule(bundle: StudyBundle) -> Record:
    """Re-execute the evaluator's own scaffold count over the stored verdicts."""
    try:
        from ..evaluation.target_benchmark import qualifying_scaffolds
    except Exception as error:  # pragma: no cover - off the supported lane
        return {"status": "FAILED", "error": str(error)}
    root = bundle.study / "sweep" / "run-01" / "cells"
    if not root.is_dir():
        return {"status": "NOT_RUN", "still_not_run_because": "RAW_ROWS_ABSENT"}
    sweep = bundle.get("generation_sweep")
    registered: dict[str, int] = {}
    if isinstance(sweep, Mapping):
        for entries in dict((sweep.get("primary") or {}).get("strata") or {}).values():
            for arm in entries:
                registered[str(arm.get("arm"))] = int(arm.get("qualifying_scaffolds") or 0)
    per_arm: dict[str, int] = {}
    disagreements: list[Record] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        rows: list[Record] = []
        for seed_dir in sorted(directory.glob("seed-*")):
            payload = _read_json(seed_dir / "qualification.json")
            if isinstance(payload, Mapping):
                rows.extend(dict(row) for row in payload.get("rows") or [])
        if not rows:
            continue
        count = qualifying_scaffolds(rows)
        per_arm[directory.name] = count
        if directory.name in registered and registered[directory.name] != count:
            disagreements.append(
                {
                    "arm": directory.name,
                    "registered": registered[directory.name],
                    "re_executed": count,
                }
            )
    return {
        "status": "RERUN",
        "tool": "drug_agent.selfevolve.evaluation.target_benchmark.qualifying_scaffolds",
        "per_arm": per_arm,
        "n_arms": len(per_arm),
        "agrees": not disagreements,
        "disagreements": disagreements,
    }
