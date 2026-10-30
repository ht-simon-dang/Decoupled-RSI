"""The nine-arm generation sweep — Stage D of the per-target evolution program.

Appendix A §3.3 asks for one thing and forbids another. It asks that every
comparator arm receive *identical* caps, a common exclusion set and a common
post-processing path, and that the generation funnel be reported stage by stage
so a reader can see where the molecules went. It forbids reporting a backend
that ran out of budget as a fixed-budget qualifying yield.

This module runs that sweep against the frozen ``zak-c9r-2`` protocol. Nothing
here decides anything: the caps, the seeds, the gates, the heavy-atom window and
the novelty union were all frozen by steps 15A-15C, before any molecule existed.
What this module adds is the execution, the accounting and the arithmetic.

Three properties are worth stating because they are what make the numbers
readable rather than merely present.

**One sampling path.** Every arm is driven through the uniform
:meth:`~drug_agent.tools.generation.GenerativeToolAdapter.generate` interface,
with the same heavy-atom window, the same de-duplication and the same funnel
counters the base adapter owns. Nine backends cannot disagree about what
"valid" means, because none of them is asked.

**A zero is a measurement.** Step 15B measured that only **1 of the 6** real
known-active development controls clears the full seven-gate set, and step 15C
computed that 59.4 % of the 27 arm-seed cells are expected to be exactly zero.
A qualifying yield of zero is therefore the *expected* outcome at this bar. It
is not a broken arm, not an unavailable capability, not an affinity of zero and
not a reason to move anything. A **backend failure** is a different fact — a
missing scientific outcome with a consumed cost — and it carries a different
token, the way step 15A's gate review separated ``UNAVAILABLE`` from
``EXECUTED_NO_YIELD``.

**Every cost is measured or explicitly unavailable.** GPU seconds come from
:func:`~drug_agent.selfevolve.target.budget.gpu_accounting`, which after commit
``cdc04346`` follows a detached driver by process ancestry. A telemetry state of
``UNAVAILABLE`` is reported as itself and never collapsed to ``0.0``; a measured
``0.0`` means every poll saw no CUDA context belonging to this job, which is a
different fact and is also reported as itself.

Resumability is per arm per seed. Each cell writes its own ``generation.json``
and its own ``scoring.json``, so a failure in the eighth arm does not discard the
first seven and the dispatcher can read partial results without re-spending.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import resource as _resource
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from drug_agent.selfevolve.target.baselines import (
    PROBE_MAX_HEAVY_ATOMS,
    PROBE_MIN_HEAVY_ATOMS,
    generation_funnel,
    require_uncontaminated,
)
from drug_agent.selfevolve.target.budget import gpu_accounting

Record = dict[str, Any]

CONTRACT_VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO / "configs" / "research" / "zak-c9r"
SNAPSHOT = REPO / ".external" / "research" / "zak-c9r"
DEFAULT_LOCK = SNAPSHOT / "protocol.lock.json"

#: The qualification rule Stage D is scored through, frozen by step 15B,
#: re-frozen against the ``zak-c9r-2`` evaluator by step 15C, and re-frozen again
#: on 2026-09-25 when the novelty exclusion set was widened. Checked rather than
#: assumed: :func:`load_qualification` recomputes the hash of what it read and
#: refuses a record whose content no longer matches this address. That refusal is
#: what caught this change rather than letting it run silently.
#:
#: SUPERSEDED, kept so a past number can be read against the rule that produced it:
#:   step 15B/15C  sha256:1deed61ef505deb4bba6e41f039fa3a4768dcdd7ef88a75f2e2a93f10ae14cc7
#:                 novelty enforced 30 excluded molecules
#:
#: The widening added every molecule already recorded on this host, accepted and
#: rejected, taking the enforced set from 30 to 7,322. It closed a measured hole:
#: a campaign scored a perfect 150 of 150 by submitting 150 molecules taken
#: straight out of the accepted corpus, and none of those 1,002 molecules was in
#: the 30. Re-scored under this rule the same submission yields 0, with all 150
#: failing novelty. Widening can only make qualification harder and applies
#: identically to every arm, which is why the protocol permits it and says the
#: union never narrows.
QUALIFICATION_HASH = "sha256:365cc0cc936ed1fa4cabbd8cdd30c459ae21baf7f9838d25d3546494ad0240f2"

#: ``zak-c9r-2``. Read from the lock and compared, never re-derived here.
PROTOCOL_VERSION = "zak-c9r-2"
PROTOCOL_HASH = "sha256:f960f2c8635ba3053dadbdc05a138d89b4ce6b8ef91456cf0fcaa82c7656366b"

#: The measured teacher docking rate the ``zak-c9r-2`` lock records, over 230
#: rows and 1,668.013 seconds. Used for the pre-launch estimate only; what the
#: sweep reports afterwards is what it measured.
MEASURED_DOCK_SECONDS_PER_MOLECULE = 7.252230434782609

#: Step 15B's measured pass rate of the REAL known-active development controls
#: through this same seven-gate set. Quoted beside every arm's count, because it
#: is what stops a zero being read as a broken arm.
KNOWN_ACTIVE_PASS = (1, 6)

#: The per-molecule baseline qualification rate step 15C derived, and the
#: per-arm Poisson mean that follows from it at the locked budget. The Holm
#: family-wise procedure below tests each arm's observed count against this.
BASELINE_RATE_PER_MOLECULE = 0.010416666666666666
FAMILY_WISE_ALPHA = 0.05

#: The comparator bar step 15E registered in its place: **4 qualifying scaffolds
#: in 300 molecules**, measured on the same two generator roles the retired bar
#: was built from (``B-B0-pfm-frozen`` -> the sweep's ``pfm`` arm, 0 of 150;
#: ``B-B6-reinvent4-prior`` -> its ``reinvent4`` arm, 4 of 150), directly under
#: the seven gates rather than inferred through a 0.5 widening factor.
#: Registered at ``configs/research/zak-c9r/baseline-results.json``
#: ``track_b_de_novo_baseline.de_novo_comparator_rate.superseding``.
SUPERSEDING_COMPARATOR_RATE_PER_MOLECULE = 0.013333333333333334

#: The nine arms ``zak-c9r-2`` registered as the Holm family, before any arm ran.
COMPARATOR_FAMILY_SIZE = 9

#: The two named comparator bars, and which one a caller gets when it says
#: nothing.
#:
#: The distinction is not cosmetic. ``stage-d-registered-null`` is the rate that
#: was registered **before any arm ran** and is what the delivered Stage D result
#: was measured against; it must stay reachable, byte-exact, or that result stops
#: being reproducible. But it is **retired as the bar a FUTURE arm is measured
#: against**, and leaving it as the executable default would mean the new bar was
#: enforced only by prose while the retired bar was still what the code applied.
#:
#: Note the direction. The supersession **tightens** the requirement: at the
#: locked budget the per-arm Poisson mean moves from 1.5625 to 2.0, and the
#: smallest count surviving Holm across the nine arms moves from **6** qualifying
#: scaffolds in 150 to **7**. Making it live is a narrowing, never a widening.
STAGE_D_REGISTERED_NULL = "stage-d-registered-null"
REGISTERED_COMPARATOR = "step-15E-superseding"

COMPARATOR_BARS: dict[str, dict[str, Any]] = {
    STAGE_D_REGISTERED_NULL: {
        "name": STAGE_D_REGISTERED_NULL,
        "rate_per_molecule": BASELINE_RATE_PER_MOLECULE,
        "status": "RETIRED_AS_THE_FORWARD_BAR",
        "measured_under_n_gates": 5,
        "qualifying_scaffolds_observed": 1,
        "molecules_observed": 48,
        "n_seeds": 1,
        "registered_by": "step-15C",
        "source": (
            "configs/research/zak-c9r/protocol.json statistics.power_generation_quality.evidence - "
            "1 qualifying scaffold in 48 molecules under FIVE gates, carried through a "
            "gate_widening_factor of 0.5 to estimate the seven-gate rate"
        ),
        "use_it_for": (
            "reproducing the delivered Stage D result, which was measured against this rate before "
            "any arm ran. NEVER for scoring a new arm: step 15E superseded it"
        ),
    },
    REGISTERED_COMPARATOR: {
        "name": REGISTERED_COMPARATOR,
        "rate_per_molecule": SUPERSEDING_COMPARATOR_RATE_PER_MOLECULE,
        "status": "REGISTERED",
        "measured_under_n_gates": 7,
        "qualifying_scaffolds_observed": 4,
        "molecules_observed": 300,
        "n_seeds": 3,
        "registered_by": "step-15E-comparators",
        "source": (
            "configs/research/zak-c9r/baseline-results.json "
            "track_b_de_novo_baseline.de_novo_comparator_rate.superseding - the same two generator "
            "roles the retired bar was built from, measured directly under the seven gates"
        ),
        "label": "POOLED_ACROSS_CONDITIONING",
        "pooling_warning": (
            "this bar pools two conditioning classes: reinvent4 (conditioning `none`, 4 of 150) and "
            "pfm (conditioning `pocket`, 0 of 150). ALL FOUR qualifying scaffolds come from the "
            "unconditioned arm; the pocket-conditioned half contributes zero. The bar is set "
            "entirely by an arm that never sees the target"
        ),
        "use_it_for": "every de novo arm measured after step 15E",
    },
}

#: The declared absolute margin, frozen at step 14 and unchanged by 15C.
ABSOLUTE_THRESHOLD_SCAFFOLDS = 3

#: Every endpoint whose unconstrained distribution Appendix A §3.3 requires.
#: ``vina_affinity``, ``herg`` and ``sa_score`` are the three that also gate;
#: the rest gate nothing and are reported because a count with mode zero cannot
#: show where the arms differ. ``qed`` in particular is deliberately NOT a gate
#: — step 15B's §4.2 names it as the endpoint the SA-minimisation trap runs
#: through — and appears here only as an observation.
DISTRIBUTION_ENDPOINTS: tuple[str, ...] = (
    "vina_affinity",
    "herg",
    "sa_score",
    "mw",
    "logp",
    "n_structural_alerts",
    "qed",
    "n_heavy_atoms",
    "posebusters_intramolecular_n_failed",
    "posebusters_n_failed",
)

#: What the two PoseBusters distributions above mean, because one of them is a
#: property of the receptor file rather than of the molecules.
DISTRIBUTION_NOTES: dict[str, str] = {
    "posebusters_intramolecular_n_failed": (
        "failed checks in PoseBusters' `mol` battery - the molecule judged on its own. This is "
        "the generated-3D-quality number; zero is a clean molecule"
    ),
    "posebusters_n_failed": (
        "failed columns across the whole dock-mode column set, INCLUDING four that are not "
        "PoseBusters tests and cannot be True on this receptor for any molecule. A mode of 4 here "
        "is the vacuity floor, not four defects. Read posebusters_intramolecular_n_failed instead"
    ),
}

#: Morgan parameters, taken from the committed baseline manifests so the
#: nearest-neighbour numbers here are on the same footing as Step 8's.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

EXPLORATORY = "EXPLORATORY"

#: What this programme's ``gpu_seconds`` has always meant, stated here because
#: Stage D is the first step whose largest GPU figure is not GPU *work*.
#:
#: ``_GpuSampler`` polls ``nvidia-smi --query-compute-apps`` every 0.2 s and adds
#: one interval whenever any process in this job holds a CUDA context. It measures
#: **occupancy**, not utilisation, and Stage A's figures are occupancy too — OMTRA's
#: probe reported 24.6 GPU-seconds over 29.0 seconds of wall.
#:
#: For Stage D's docking pass that distinction is load-bearing and must not be
#: glossed. **AutoDock Vina does no GPU computation.** The occupancy the docking
#: pass reports is the ADMET-AI panel's model, resident in this same process from
#: an earlier endpoint call, holding roughly 700 MiB while Vina runs on the CPU.
#: The number is a real measurement of a real hold on the device; it is not
#: evidence that docking used the GPU, and no report may present it as such.
GPU_OCCUPANCY_SEMANTICS = (
    "gpu_seconds is OCCUPANCY: the seconds during which this job held a CUDA context, sampled "
    "from nvidia-smi compute processes at 0.2 s. It is not GPU utilisation and it is not GPU "
    "work. AutoDock Vina does no GPU computation, so the docking pass's occupancy is the "
    "ADMET-AI panel's resident model in the same process, not docking using the device"
)

#: Why the docking phase's occupancy is measured, reported, and **not charged**
#: against the campaign's GPU cap. This is a judgement, it is stated as one, and a
#: reviewer should check it rather than take it.
#:
#: The argument is the frozen record's own: ``zak-c9r-2``'s
#: ``budgets.evaluation_budget.track_b_per_arm_per_seed`` models Track B evaluation
#: as ``estimated_docking_wall_seconds`` and declares **no GPU term at all**,
#: because docking on this platform is AutoDock Vina and Vina is a CPU program.
#: Charging fifteen hours of CPU docking against a GPU cap — on the strength of an
#: idle 700 MiB model the same process happens to be holding — would refuse a study
#: for consuming a resource it is not consuming, and would contradict the budget
#: model the lock froze.
#:
#: What is *not* done: the number is not discarded, not rounded to zero and not
#: left out. It is measured, reported beside the charged total, and named here. A
#: long-lived idle CUDA context is a real hold on a shared device and a successor
#: may reasonably decide it should be charged; if so, the arithmetic is on the
#: record and the decision does not need re-measuring.
DOCKING_OCCUPANCY_NOTE = (
    "MEASURED and REPORTED, deliberately NOT CHARGED against the campaign gpu_seconds cap. "
    "zak-c9r-2's evaluation budget models Track B scoring as docking WALL seconds with no GPU "
    "term, because Vina is a CPU program. This occupancy is an idle ADMET-AI context resident in "
    "the same process, not docking using the device, and charging it would refuse the study for "
    "a resource it is not consuming. The number is on the record so a successor can decide "
    "otherwise without re-measuring it"
)


# ---------------------------------------------------------------------------
# PoseBusters, decomposed — what the 27 counted columns actually are
# ---------------------------------------------------------------------------
#: The generated-3D-quality battery: PoseBusters' ``mol`` configuration, which is
#: the ligand judged on its own. Verified against the installed engine rather
#: than copied from a paper — ``mol.yml`` and ``dock.yml`` declare these modules
#: with byte-identical parameters, so an intramolecular verdict measured in
#: ``dock`` mode is the same verdict ``mol`` mode gives.
#:
#: This is the number Appendix A §3.3 asks for. §3.3 also forbids inventing a
#: native-pose RMSD for a generated molecule, which is exactly why ``redock`` is
#: the wrong configuration here: a de novo molecule has no known native pose, so
#: ``mol`` is the only mode whose every check is answerable.
POSEBUSTERS_INTRAMOLECULAR_CHECKS: tuple[str, ...] = (
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "no_radicals_before_sanitization",
    "passes_valence_checks",
    "passes_kekulization",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
)

#: Which entity in the receptor each protein-context column needs in order to
#: mean anything. A column whose entity is absent from the receptor is
#: **vacuous**: it is not measuring the molecule, it is measuring the receptor
#: file. Which entities are present is not assumed here — it is counted, by
#: PoseBusters' own atom typing, in :func:`receptor_entity_census`.
POSEBUSTERS_CHECK_ENTITY: dict[str, str] = {
    "protein-ligand_maximum_distance": "protein",
    "minimum_distance_to_protein": "protein",
    "volume_overlap_with_protein": "protein",
    "most_extreme_clash_protein": "protein",
    "minimum_distance_to_organic_cofactors": "organic_cofactors",
    "volume_overlap_with_organic_cofactors": "organic_cofactors",
    "not_too_far_away_organic_cofactors": "organic_cofactors",
    "minimum_distance_to_inorganic_cofactors": "inorganic_cofactors",
    "volume_overlap_with_inorganic_cofactors": "inorganic_cofactors",
    "not_too_far_away_inorganic_cofactors": "inorganic_cofactors",
    "minimum_distance_to_waters": "waters",
    "volume_overlap_with_waters": "waters",
    "not_too_far_away_waters": "waters",
}

#: Boolean columns ``full_report=True`` emits that **PoseBusters does not select
#: as tests** — ``dock.yml`` does not name them in any module's
#: ``chosen_binary_test_output``. They reach this repository's report only
#: because :class:`~drug_agent.tools.structure.posebusters_tool.PoseBustersAdapter`
#: treats every boolean column as a check. Each is recorded with why it is not
#: one, because "it failed" is the opposite of the truth for the first of them.
POSEBUSTERS_NON_TEST_COLUMNS: dict[str, str] = {
    "most_extreme_clash_protein": (
        "INVERTED POLARITY and not a PoseBusters test. It is the `clash` flag of the single "
        "closest ligand-protein contact, carried into the report as a detail row, and True means "
        "there IS a clash. Counting it as a check where True is a pass inverts it: a pose with no "
        "clash at its closest contact is recorded as having FAILED. Measured here: it is exactly "
        "the complement of minimum_distance_to_protein on all 1,296 poses"
    ),
    "not_too_far_away_organic_cofactors": (
        "not a PoseBusters test in dock mode: the cofactor and water modules select only "
        "`no_clashes`, never `not_too_far_away`. The column is a raw module output that "
        "full_report exposes, and it is computed as `min(distance) <= 5.0` over an EMPTY set of "
        "candidate atoms, which is NaN <= 5.0, which is False"
    ),
    "not_too_far_away_inorganic_cofactors": (
        "not a PoseBusters test in dock mode; same raw-output and empty-set arithmetic as "
        "not_too_far_away_organic_cofactors"
    ),
    "not_too_far_away_waters": (
        "not a PoseBusters test in dock mode; same raw-output and empty-set arithmetic as "
        "not_too_far_away_organic_cofactors"
    ),
}

#: Why the ``mol`` configuration is the right invocation for this sweep, carried
#: onto the record so a reader does not have to reconstruct the argument.
POSEBUSTERS_INVOCATION_NOTE = (
    "HEADLINE MODE IS `mol`: the ligand judged on its own, recomputed from the pose_molblock "
    "already stored on every scoring row. Appendix A section 3.3 forbids inventing a native-pose "
    "RMSD for a generated molecule, so `redock` is unanswerable here - these molecules have no "
    "known native pose. `dock` is reported too, separately, because it is what the sweep ran; but "
    "its cofactor and water columns are vacuous against a prepared receptor that contains none of "
    "those entities, so they say nothing about any generator"
)

#: How the intramolecular battery was invoked, and why either invocation gives
#: the same verdict. Carried on the record so the reader does not have to take it.
POSEBUSTERS_INTRAMOLECULAR_MODE_NOTE = (
    "The fourteen intramolecular checks are declared by mol.yml and dock.yml with IDENTICAL "
    "functions and IDENTICAL parameters, so the verdict does not depend on which configuration "
    "invoked them. `mode` records which one did. `dock` is the cheaper invocation and the one the "
    "sweep itself used: mol.yml declares energy_ratio twice and functools.cache does not cache a "
    "call that raised, so a ligand whose conformer ensemble fails to embed pays for it three times "
    "under mol-then-dock against once under dock alone. `redock` is NOT used and no native-pose "
    "RMSD is invented: Appendix A section 3.3 forbids it for a molecule with no known native pose"
)

#: The sentence that must travel with any "passes every check" figure computed
#: over the mixed dock-mode column set against this receptor.
POSEBUSTERS_MIXED_TOTAL_NOTE = (
    "DO NOT READ AS A GENERATOR PROPERTY. This counts every boolean column the adapter collected "
    "in dock mode, including four that are not PoseBusters tests: three cofactor/water columns "
    "that are vacuously False because the prepared receptor contains no cofactors and no waters, "
    "and one inverted-polarity detail column whose False means the pose does NOT clash. Those "
    "four cannot all be True on this receptor for any molecule from any generator, so this figure "
    "is zero by construction and is reported only beside the decomposition"
)


# ---------------------------------------------------------------------------
# Frozen inputs, read and verified rather than retyped
# ---------------------------------------------------------------------------
def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_qualification(path: Path | None = None) -> Record:
    """The frozen seven-gate rule, with its address re-derived from its content.

    A rule that is merely *read* can have been edited; a rule whose hash is
    recomputed cannot. The recomputation uses the same function that wrote the
    address, so a single changed threshold moves it and this raises.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import qualification_hash

    record = _json(path or (CONFIG_DIR / "qualification.json"))
    declared = str(record.get("qualification_hash") or "")
    bare = {k: v for k, v in record.items() if k != "qualification_hash"}
    measured = qualification_hash(bare)
    if declared != measured:
        raise SystemExit(
            f"the qualification record's declared hash {declared} is not the hash of its "
            f"content {measured}; the frozen rule has been edited and the sweep refuses to score "
            f"through it"
        )
    if declared != QUALIFICATION_HASH:
        raise SystemExit(
            f"the qualification rule is {declared}, not the {QUALIFICATION_HASH} this stage is "
            f"registered against"
        )
    return record


def load_lock(path: Path | None = None) -> Record:
    """The runtime protocol lock, checked for version and hash."""
    record = _json(path or DEFAULT_LOCK)
    protocol = dict(record.get("protocol") or {})
    version = str(protocol.get("protocol_version") or "")
    digest = str(record.get("protocol_hash") or "")
    if version != PROTOCOL_VERSION or digest != PROTOCOL_HASH:
        raise SystemExit(
            f"the lock at {path or DEFAULT_LOCK} is {version} / {digest}, not "
            f"{PROTOCOL_VERSION} / {PROTOCOL_HASH}"
        )
    return record


def sweep_arms(qualification: Mapping[str, Any]) -> list[Record]:
    """The nine admitted arms with their conditioning class, read from the rule.

    Read rather than hand-listed, for the reason step 15B gave when it wired the
    same table to the measured survey: three of the nine arms never see the
    target, and a table that loses that column is not a measurement. Ordering is
    by class then id, so the primary table is stratified by construction and a
    pooled reading has to be asked for by name.
    """
    strat = dict(qualification.get("conditioning_stratification") or {})
    classes = dict(strat.get("classes") or {})
    rows: list[Record] = []
    synthesis_aware = set(strat.get("synthesis_aware") or ())
    for conditioning in ("pocket", "ligand", "none"):
        for capability_id in sorted(classes.get(conditioning) or ()):
            rows.append(
                {
                    "capability_id": capability_id,
                    "arm": capability_id.rsplit(".", 1)[-1],
                    "conditioning": conditioning,
                    "synthesis_aware": capability_id in synthesis_aware,
                }
            )
    expected = int(strat.get("n_admitted_arms") or 0)
    if expected and len(rows) != expected:
        raise SystemExit(f"the rule declares {expected} admitted arms and names {len(rows)}")
    return rows


def stage_d_caps(qualification: Mapping[str, Any]) -> Record:
    """The operator's Stage D caps, read from the frozen rule."""
    caps = dict((qualification.get("caps") or {}).get("stage_d") or {})
    if not caps:
        raise SystemExit("the qualification rule carries no caps.stage_d block")
    return caps


def sweep_seeds(lock: Mapping[str, Any]) -> list[int]:
    """The deployment seeds, read from the lock."""
    seeds = [int(s) for s in (lock.get("protocol") or {}).get("deployment_seeds") or ()]
    if not seeds:
        raise SystemExit("the lock declares no deployment_seeds")
    return seeds


def verify_test_access_count(root: Path | None = None) -> Record:
    """Every record in the tree that declares ``test_access_count``, and its value.

    Verified rather than asserted, and verified *before* anything is spent. The
    lock is legitimate as a superseding version only while this is zero; the
    moment it is not, the lock is closed and any change afterwards needs a
    further superseding version. This function reports counts and values only —
    it never reads a partition and never prints a held-out label.
    """
    base = root or REPO
    seen: list[Record] = []
    for path in sorted(
        [
            *(base / "configs").rglob("*.json"),
            *(base / "configs").rglob("*.yaml"),
            *((base / ".external" / "research").rglob("*.json") if (base / ".external").is_dir() else []),
        ]
    ):
        with contextlib.suppress(Exception):
            text = path.read_text()
            if "test_access_count" not in text:
                continue
            for value in _scan_access_counts(path, text):
                seen.append(value)
    values = [r["value"] for r in seen if r["value"] is not None]
    return {
        "records_declaring_it": seen,
        "n_records": len(seen),
        "values": sorted(set(values)),
        "all_zero": all(v == 0 for v in values) and bool(values),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "counts and values only; no partition was read and no held-out label is printed",
    }


def _scan_access_counts(path: Path, text: str) -> list[Record]:
    """Read the declared value out of one record, JSON or YAML."""
    out: list[Record] = []
    payload: Any = None
    with contextlib.suppress(Exception):
        payload = json.loads(text) if path.suffix == ".json" else __import__("yaml").safe_load(text)
    if payload is None:
        return out

    def walk(node: Any, pointer: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "test_access_count" and isinstance(value, (int, float)):
                    out.append(
                        {
                            "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                            "pointer": f"{pointer}/{key}",
                            "value": int(value),
                        }
                    )
                else:
                    walk(value, f"{pointer}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{pointer}[{index}]")

    walk(payload, "")
    return out


# ---------------------------------------------------------------------------
# The estimate, and the reservation it justifies
# ---------------------------------------------------------------------------
def _availability_record() -> Record:
    return _json(CONFIG_DIR / "generator-availability.json")


def estimate_sweep(
    *,
    n_arms: int,
    n_seeds: int,
    unique_cap: int,
    endpoint_seconds_per_molecule: float | None,
    endpoint_gpu_seconds_per_molecule: float | None,
    endpoint_rate_evidence: str,
    posebusters_seconds_per_molecule: float | None = None,
    docking_seconds_per_molecule: float | None = None,
    docking_gpu_seconds_per_molecule: float | None = None,
    docking_rate_evidence: str | None = None,
) -> Record:
    """What the sweep is expected to cost, before a single molecule is drawn.

    Generation comes from step 15A's executed throughput — a cold-start
    extrapolation that pays one model load for a handful of attempts, so it
    **overstates** the per-attempt cost at scale and errs in the direction a
    budget can afford to err in. Docking comes from the measured teacher rate the
    ``zak-c9r-2`` lock records. Endpoint scoring has no frozen rate, so the
    caller measures one and passes it in rather than letting this function invent
    a number.

    ``None`` anywhere is an unmeasured term and is carried as ``None``. A total
    that silently drops an unmeasured term is a smaller number presented as a
    fact, so the totals below say how many terms they could not add.
    """
    availability = _availability_record()
    arithmetic = dict(availability.get("operator_cap_arithmetic") or {})
    generation_wall = float(arithmetic.get("total_wall_seconds_all_admitted_all_seeds") or 0.0)
    generation_gpu = float(arithmetic.get("total_gpu_seconds_all_admitted_all_seeds") or 0.0)

    n_molecules = n_arms * n_seeds * unique_cap
    dock_rate = (
        float(docking_seconds_per_molecule)
        if docking_seconds_per_molecule
        else MEASURED_DOCK_SECONDS_PER_MOLECULE
    )
    dock_wall = n_molecules * dock_rate
    endpoint_wall = (
        None if endpoint_seconds_per_molecule is None else n_molecules * endpoint_seconds_per_molecule
    )
    endpoint_gpu = (
        None if endpoint_gpu_seconds_per_molecule is None else n_molecules * endpoint_gpu_seconds_per_molecule
    )
    posebusters_wall = (
        None if posebusters_seconds_per_molecule is None else n_molecules * posebusters_seconds_per_molecule
    )
    # Occupancy, not work (see GPU_OCCUPANCY_SEMANTICS). Measured and reported,
    # and deliberately NOT added to the gpu_seconds total the ledger holds
    # against the campaign's GPU cap -- see `docking_gpu_occupancy_note` below
    # for the argument and for why it is a judgement a reviewer should check.
    docking_gpu = (
        None if docking_gpu_seconds_per_molecule is None else n_molecules * docking_gpu_seconds_per_molecule
    )

    scoring_terms = [dock_wall, endpoint_wall, posebusters_wall]
    scoring_wall = sum(t for t in scoring_terms if t is not None)
    scoring_unmeasured = sum(1 for t in scoring_terms if t is None)

    gpu_terms = [generation_gpu, endpoint_gpu]
    gpu_total = sum(t for t in gpu_terms if t is not None)
    gpu_unmeasured = sum(1 for t in gpu_terms if t is None)

    return {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "stage_d_sweep_estimate",
        "n_arms": n_arms,
        "n_seeds": n_seeds,
        "unique_molecules_per_cell": unique_cap,
        "n_molecules_total": n_molecules,
        "generation": {
            "wall_seconds": round(generation_wall, 1),
            "gpu_seconds": round(generation_gpu, 1),
            "source": (
                "configs/research/zak-c9r/generator-availability.json -> "
                "operator_cap_arithmetic, step 15A's executed throughput at the operator's "
                "50-unique x 3-seed cap"
            ),
            "evidence_class": "extrapolated_from_measured",
            "direction_of_error": (
                "OVERSTATES. Every probe paid one model load for 2-4 attempts, so the per-attempt "
                "cost it reports is an upper bound and the sweep should be cheaper"
            ),
            "arms_excluded_from_the_gpu_total": list(
                arithmetic.get("arms_excluded_from_the_gpu_total") or ()
            ),
        },
        "scoring": {
            "docking_wall_seconds": round(dock_wall, 1),
            "docking_rate_seconds_per_molecule": dock_rate,
            "docking_rate_source": (
                docking_rate_evidence or "zak-c9r-2 lock, measured teacher rate: 1,668.013 s over 230 rows"
            ),
            "locked_teacher_docking_rate_seconds_per_molecule": MEASURED_DOCK_SECONDS_PER_MOLECULE,
            "endpoint_wall_seconds": None if endpoint_wall is None else round(endpoint_wall, 1),
            "endpoint_gpu_seconds": None if endpoint_gpu is None else round(endpoint_gpu, 1),
            "endpoint_rate_seconds_per_molecule": endpoint_seconds_per_molecule,
            "endpoint_gpu_seconds_per_molecule": endpoint_gpu_seconds_per_molecule,
            "endpoint_rate_evidence": endpoint_rate_evidence,
            "docking_gpu_occupancy_seconds": (None if docking_gpu is None else round(docking_gpu, 1)),
            "docking_gpu_occupancy_seconds_per_molecule": docking_gpu_seconds_per_molecule,
            "docking_gpu_semantics": GPU_OCCUPANCY_SEMANTICS,
            "docking_gpu_occupancy_note": DOCKING_OCCUPANCY_NOTE,
            "posebusters_wall_seconds": (None if posebusters_wall is None else round(posebusters_wall, 1)),
            "posebusters_rate_seconds_per_molecule": posebusters_seconds_per_molecule,
            "wall_seconds": round(scoring_wall, 1),
            "unmeasured_terms": scoring_unmeasured,
            "note": (
                "docking is CPU here (AutoDock Vina); the endpoint panel is ADMET-AI on the GPU. "
                "A term nobody measured is None and is counted in unmeasured_terms rather than "
                "added as zero"
            ),
        },
        "gpu_semantics": GPU_OCCUPANCY_SEMANTICS,
        "totals": {
            "wall_seconds": round(generation_wall + scoring_wall, 1),
            "wall_hours": round((generation_wall + scoring_wall) / 3600.0, 2),
            "gpu_seconds": round(gpu_total, 1),
            "gpu_seconds_charged_note": (
                "generation plus the endpoint panel -- the two phases that compute on the device. "
                "The docking phase's occupancy is reported beside it and is not charged here; "
                "see scoring.docking_gpu_occupancy_note"
            ),
            "docking_gpu_occupancy_seconds_not_charged": (
                None if docking_gpu is None else round(docking_gpu, 1)
            ),
            "gpu_unmeasured_terms": gpu_unmeasured,
            "wall_unmeasured_terms": scoring_unmeasured,
        },
    }


def measured_docking_rate(output: Path) -> Record | None:
    """The docking rate this sweep has actually measured, if it has measured one.

    The ``zak-c9r-2`` lock records **7.2522 s/molecule**, measured over 230 rows
    of *teacher* chemistry. Generated molecules are not that chemistry, and Vina's
    cost rises with rotatable-bond count, so a rate measured on one is not a rate
    for the other. When a completed arm exists in this run directory, its own rate
    is the better estimator and this returns it — with the denominator it was
    divided by, so nobody has to take it on trust.

    Returns ``None`` when nothing has been scored yet, which is the honest answer
    for the first launch: the lock's rate is then the only evidence there is.
    """
    seconds = 0.0
    gpu_seconds = 0.0
    pose_seconds = 0.0
    gpu_measured = True
    molecules = 0
    arms: list[str] = []
    for path in sorted((output / "cells").glob("*/scoring.json")):
        record = _load_checkpoint(path)
        if record is None:
            continue
        cost = dict(record.get("cost") or {})
        n = int(cost.get("n_molecules") or 0)
        if n <= 0 or record.get("scoring_failed"):
            continue
        seconds += float(cost.get("docking_wall_seconds") or 0.0)
        pose_seconds += float(cost.get("posebusters_wall_seconds") or 0.0)
        if cost.get("docking_gpu_telemetry") == "MEASURED":
            gpu_seconds += float(cost.get("docking_gpu_seconds") or 0.0)
        else:
            gpu_measured = False
        molecules += n
        arms.append(path.parent.name)
    if molecules <= 0:
        return None
    return {
        "seconds_per_molecule": round(seconds / molecules, 4),
        "docking_wall_seconds": round(seconds, 1),
        "gpu_seconds": round(gpu_seconds, 1) if gpu_measured else None,
        "gpu_seconds_per_molecule": round(gpu_seconds / molecules, 4) if gpu_measured else None,
        "gpu_telemetry": "MEASURED" if gpu_measured else "UNAVAILABLE",
        "gpu_semantics": GPU_OCCUPANCY_SEMANTICS,
        "posebusters_seconds_per_molecule": round(pose_seconds / molecules, 4),
        "n_molecules": molecules,
        "arms": arms,
        "evidence_class": "measured",
        "why_it_replaces_the_locked_rate": (
            "the lock's 7.2522 s/molecule was measured over 230 rows of teacher chemistry. "
            "Vina's cost rises with rotatable-bond count and generated molecules are not that "
            "chemistry, so a rate measured on this run's own generated molecules is the better "
            "estimator for the rest of this run"
        ),
    }


def measured_endpoint_rate(output: Path) -> Record | None:
    """The endpoint panel's rate on this run's own molecules, if it has one.

    The pre-launch probe times a dozen molecules and therefore amortises the
    panel's fixed Lightning setup over almost nothing, which overstates the
    per-molecule cost by more than an order of magnitude — ADMET-AI scored one
    arm's 150 molecules in less time than the twelve-molecule probe took. Once an
    arm has been scored, its rate is the better estimator.

    Using it also spares the run a second, unnecessary CUDA context: the probe's
    only purpose is the estimate, and a resumed sweep that already has a measured
    rate does not need to load the panel before it starts docking.
    """
    seconds = 0.0
    gpu_seconds = 0.0
    gpu_measured = True
    molecules = 0
    arms: list[str] = []
    for path in sorted((output / "cells").glob("*/scoring.json")):
        record = _load_checkpoint(path)
        if record is None:
            continue
        cost = dict(record.get("cost") or {})
        n = int(cost.get("n_molecules") or 0)
        if n <= 0 or record.get("scoring_failed"):
            continue
        seconds += float(cost.get("endpoint_wall_seconds") or 0.0)
        if cost.get("endpoint_gpu_telemetry") == "MEASURED":
            gpu_seconds += float(cost.get("endpoint_gpu_seconds") or 0.0)
        else:
            gpu_measured = False
        molecules += n
        arms.append(path.parent.name)
    if molecules <= 0:
        return None
    return {
        "n_probe_molecules": molecules,
        "n_scored": molecules,
        "wall_seconds": round(seconds, 3),
        "seconds_per_molecule": round(seconds / molecules, 4),
        "gpu_seconds": round(gpu_seconds, 3) if gpu_measured else None,
        "gpu_seconds_per_molecule": round(gpu_seconds / molecules, 4) if gpu_measured else None,
        "gpu_telemetry": "MEASURED" if gpu_measured else "UNAVAILABLE",
        "evidence_class": "measured",
        "source": f"this run's scored arms: {', '.join(arms)}",
        "direction_of_error": (
            "none known. Unlike the twelve-molecule probe it replaces, this rate amortises the "
            "panel's fixed setup over the batch size the sweep actually uses"
        ),
    }


def declare_caps(ledger: Any, *, wall_limit: float, gpu_limit: float, dock_calls: float) -> Record:
    """Declare the campaign caps this sweep spends against.

    The numbers come from ``configs/research/zak-c9r.yaml``'s budget block, which
    is the same authorisation every other step in this programme spends against.
    :meth:`BudgetLedger.declare` refuses an enlargement, so a re-declaration can
    only tighten.
    """
    ledger.declare("wall_seconds", wall_limit, unit="seconds", note="zak-c9r.yaml budget.wall_hours")
    ledger.declare("gpu_seconds", gpu_limit, unit="seconds", note="zak-c9r.yaml budget.gpu_hours")
    ledger.declare("dock_calls", dock_calls, unit="calls", note="one docking call per unique molecule")
    return {r: dict(v) for r, v in ledger.caps().items()}


def refuse_or_reserve(
    ledger: Any,
    *,
    estimate: Record,
    owner: str,
    phase: str = "EVALUATION",
    request_key: str | None = None,
) -> Record:
    """Hold the estimate, or refuse the launch by number.

    A launch that does not fit the remaining authorisation is refused **before**
    anything is started, the way step 14's pilot path refuses a campaign it
    cannot pay for. Started-and-abandoned is the outcome this exists to prevent:
    it spends the budget and produces no scientific outcome.

    Two properties make the refusal safe to act on.

    *A refusal holds nothing.* If the third resource does not fit, the two that
    already did are released before this returns, so a refused sweep leaves the
    budget exactly as it found it. Refusing while quietly holding two thirds of
    the estimate would make the *next* attempt fail for a reason nobody could
    see.

    *A resumed sweep is not charged twice.* With a ``request_key`` the ledger
    returns the original reservation rather than taking a second, so a restart
    reconciles against what it already holds instead of asking for it again —
    which is also why the affordability question is asked by *attempting* the
    reservation rather than by arithmetic on the remaining balance: the
    arithmetic cannot see that this caller already owns some of it.
    """
    from drug_agent.selfevolve.target.budget import BudgetExceeded

    wanted = {
        "wall_seconds": float(estimate["totals"]["wall_seconds"]),
        "gpu_seconds": float(estimate["totals"]["gpu_seconds"]),
        "dock_calls": float(estimate["n_molecules_total"]),
    }
    reservations: Record = {}
    top_ups: Record = {}
    try:
        for resource, amount in wanted.items():
            key = None if request_key is None else f"{request_key}:{resource}"
            record = ledger.reserve(
                resource,
                amount,
                owner=owner,
                phase=phase,
                request_key=key,
                note="stage D nine-arm generation sweep, estimated before the spend",
            )
            reservations[resource] = record
            # A resumed reservation holds what the *first* estimate asked for. If
            # the estimate has since been corrected upwards - because this run
            # measured a rate the lock did not have - the difference is reserved
            # now, before it is spent, rather than discovered as an overrun at
            # settlement. Re-using the original key would have returned the old,
            # too-small reservation and said nothing.
            # A reservation that has been released or settled holds nothing, whatever
            # its `reserved` column still says. Reading the column alone would make a
            # resumed run believe it was covered by a hold that no longer exists.
            held = float(record.get("reserved") or 0.0) if record.get("state") == "RESERVED" else 0.0
            index = 0
            while held < amount - 1e-9 and key is not None:
                index += 1
                extra = ledger.reserve(
                    resource,
                    amount - held,
                    owner=owner,
                    phase=phase,
                    request_key=f"{key}:topup-{index}",
                    note=(
                        "stage D top-up: the estimate was corrected upwards from a rate this run "
                        "measured, and the difference is reserved before it is spent"
                    ),
                )
                top_ups.setdefault(resource, []).append(dict(extra))
                held += float(extra.get("reserved") or 0.0) if extra.get("state") == "RESERVED" else 0.0
                if index > 8:  # a guard, not a policy: something else is wrong
                    break
    except BudgetExceeded as exc:
        fresh = [r for r in reservations.values() if not r.get("resumed")]
        fresh.extend(r for records in top_ups.values() for r in records if not r.get("resumed"))
        for record in fresh:
            with contextlib.suppress(Exception):
                ledger.release(
                    record["reservation_id"],
                    note="the sweep was refused; a refusal holds nothing",
                )
        shortfalls = _shortfalls(ledger, wanted)
        lines = "; ".join(
            f"{s['resource']}: the sweep needs {s['needed']:,.1f} and {s['remaining']:,.1f} remain "
            f"of the {float(s['limit']):,.1f} authorised ({float(s['already_committed']):,.1f} "
            f"already committed)"
            for s in shortfalls
        ) or str(exc)
        return {
            "status": "REFUSED_BUDGET",
            "evidence_class": "measured",
            "shortfalls": shortfalls,
            "ledger_message": str(exc),
            "top_ups": {},
            "reason": (
                f"{lines}. The sweep is not started rather than started and abandoned: a launch "
                f"that outruns its authorisation spends the budget and produces no scientific "
                f"outcome"
            ),
            "reservations": {},
        }
    return {
        "status": "RESERVED",
        "evidence_class": "measured",
        "reason": None,
        "shortfalls": [],
        "reservations": {k: dict(v) for k, v in reservations.items()},
        "top_ups": top_ups,
        "total_held": {
            resource: _held(reservations[resource]) + sum(_held(r) for r in top_ups.get(resource, []))
            for resource in reservations
        },
    }


def _held(record: Mapping[str, Any]) -> float:
    """What a reservation actually holds, which is nothing once it is not RESERVED."""
    return float(record.get("reserved") or 0.0) if record.get("state") == "RESERVED" else 0.0


def _shortfalls(ledger: Any, wanted: Mapping[str, float]) -> list[Record]:
    """Every resource the estimate outruns, not only the first one that did.

    Reported after the release, so the remaining balance quoted is the balance a
    reader would find if they looked — not one this refused call was holding a
    slice of.
    """
    out: list[Record] = []
    for resource, amount in wanted.items():
        usage = ledger.usage(resource)
        available = usage.get("available")
        if available is not None and float(amount) > float(available) + 1e-9:
            out.append(
                {
                    "resource": resource,
                    "needed": float(amount),
                    "remaining": float(available),
                    "limit": usage.get("limit"),
                    "already_committed": usage.get("committed"),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Generation — one arm, one seed
# ---------------------------------------------------------------------------
#: A generator that is asked for more samples than one call can supply is asked
#: again, and a second call with the *same* seed would redraw the same stream on
#: a seed-deterministic backend. The continuation seed is a declared function of
#: the deployment seed and the call index, so the stream is reproducible and the
#: replication unit is unchanged: the deployment seed is still what indexes the
#: replicate, and 301/302/303 still name three independent cells.
#:
#: This is NOT a re-roll. A re-roll would redraw a cell whose yield disappointed;
#: this only continues a cell that has not yet reached the common cap, and it
#: stops the moment the cap is reached or the attempt ceiling is exhausted.
def continuation_seed(seed: int, call_index: int) -> int:
    """The seed for call *call_index* of a cell declared at *seed*."""
    if call_index == 0:
        return int(seed)
    return int(seed) * 1000 + int(call_index)


def _generator_adapter(capability_id: str) -> Any:
    """The adapter for one capability id, from the live registry."""
    from drug_agent.settings.config import Settings
    from drug_agent.tools.registry import generator_adapters

    adapters = generator_adapters(Settings())
    adapter = adapters.get(capability_id)
    if adapter is None:
        raise SystemExit(f"{capability_id} is not a registered generator on this host")
    return adapter


def _merge_funnels(
    funnels: Sequence[Mapping[str, Any]],
    *,
    n_returned: int,
    n_duplicate_across_calls: int,
    n_beyond_cap: int,
) -> Record:
    """Add up a cell's per-call funnels so the arithmetic closes over the cell.

    Four losses happen between a raw sample and a molecule the expensive
    pipeline receives, and they are four different facts:

    * ``n_invalid`` — RDKit could not parse it.
    * ``n_duplicate`` — it repeated a structure the *same call* already drew.
    * ``n_screened_out_or_beyond_request`` — it fell outside the frozen 10-70
      heavy-atom window, **or** it was one more distinct molecule than that call
      asked for. The base adapter reports these through one counter and this
      record does not pretend otherwise: the name says both and the arithmetic
      closes on the pair rather than on a guess about which it was.
    * ``n_duplicate_across_calls`` — a continuation call redrew a structure an
      earlier call in this same cell had already returned.
    * ``n_beyond_cap`` — the cell had already reached the common cap of unique
      molecules when this one arrived.

    Calling the last two "screened out" would have said the heavy-atom window
    rejected molecules it never saw.
    """
    total: Record = {
        "n_requested": sum(int(f.get("n_requested") or 0) for f in funnels),
        "n_raw": sum(int(f.get("n_raw") or 0) for f in funnels),
        "n_valid": sum(int(f.get("n_valid") or 0) for f in funnels),
        "n_invalid": sum(int(f.get("n_invalid") or 0) for f in funnels),
        "n_distinct": sum(int(f.get("n_distinct") or 0) for f in funnels),
        "n_duplicate": sum(int(f.get("n_duplicate") or 0) for f in funnels),
        "n_screened_out_or_beyond_request": sum(int(f.get("n_screened_out") or 0) for f in funnels),
        "n_returned_by_calls": sum(int(f.get("n_returned") or 0) for f in funnels),
    }
    total["n_duplicate_across_calls"] = int(n_duplicate_across_calls)
    total["n_beyond_cap"] = int(n_beyond_cap)
    total["n_returned"] = int(n_returned)
    total["min_heavy_atoms"] = PROBE_MIN_HEAVY_ATOMS
    total["max_heavy_atoms"] = PROBE_MAX_HEAVY_ATOMS
    total["n_calls"] = len(funnels)
    total["arithmetic"] = (
        "n_raw = n_invalid + n_valid; "
        "n_valid = n_duplicate + n_distinct; "
        "n_distinct = n_screened_out_or_beyond_request + n_returned_by_calls; "
        "n_returned_by_calls = n_duplicate_across_calls + n_beyond_cap + n_returned"
    )
    total["arithmetic_closes"] = bool(
        total["n_raw"] == total["n_invalid"] + total["n_valid"]
        and total["n_valid"] == total["n_duplicate"] + total["n_distinct"]
        and total["n_distinct"] == total["n_screened_out_or_beyond_request"] + total["n_returned_by_calls"]
        and total["n_returned_by_calls"]
        == total["n_duplicate_across_calls"] + total["n_beyond_cap"] + total["n_returned"]
    )
    return total


def generate_cell(
    capability_id: str,
    seed: int,
    *,
    unique_cap: int,
    raw_cap: int,
    batch_size: int = 16,
    call_timeout_seconds: float = 2400.0,
    cell_wall_seconds: float = 5400.0,
    yield_hint: float | None = None,
    workdir: Path | None = None,
) -> Record:
    """Sample one arm-seed cell to the common cap, or say why it could not.

    The loop asks for as many raw attempts as the measured yield says it needs,
    stops the moment the cap is reached, and never exceeds the frozen
    ``raw_cap``. Every call's funnel is kept, so the cell's arithmetic closes
    stage by stage.

    Four terminal outcomes, and they are four different facts:

    ``COMPLETE``
        the cell reached the common cap.
    ``SHORTFALL``
        the arm exhausted the attempt ceiling or its wall allowance without
        reaching the cap. Appendix A §3.3 forbids reporting this as a
        fixed-budget qualifying yield; it is a shortfall with a consumed cost.
    ``EXECUTED_NO_YIELD``
        the backend ran and returned nothing the pipeline can use.
    ``BACKEND_FAILURE``
        the backend raised. A missing scientific outcome with a consumed cost,
        never a zero.
    """
    from drug_agent.selfevolve.target.baselines import _generation_inputs, _loaded_weights

    require_uncontaminated(f"{capability_id} Stage D cell seed={seed}")
    started = time.time()
    try:
        adapter = _generator_adapter(capability_id)
    except Exception as exc:  # an unconstructable adapter is a failure, not a zero
        return {
            "contract_version": CONTRACT_VERSION,
            "record_kind": "stage_d_generation_cell",
            "capability_id": capability_id,
            "seed": int(seed),
            "status": "BACKEND_FAILURE",
            "outcome_class": "ADAPTER_CONSTRUCTION_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "molecules": [],
            "n_unique_returned": 0,
            "funnel": _merge_funnels([], n_returned=0, n_duplicate_across_calls=0, n_beyond_cap=0),
            "calls": [],
            "cost": _cell_cost(started, _resource.getrusage(_resource.RUSAGE_CHILDREN), None),
        }
    interpreter = getattr(getattr(adapter, "ext", None), "python", None)
    before = _resource.getrusage(_resource.RUSAGE_CHILDREN)

    base: Record = {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "stage_d_generation_cell",
        "capability_id": capability_id,
        "seed": int(seed),
        "unique_cap": int(unique_cap),
        "raw_attempt_cap": int(raw_cap),
        "heavy_atom_window": [PROBE_MIN_HEAVY_ATOMS, PROBE_MAX_HEAVY_ATOMS],
        "conditioning": getattr(adapter, "conditioning", None),
        "synthesis_aware": bool(getattr(adapter, "synthesis_aware", False)),
        "interpreter": interpreter,
        "device": getattr(getattr(adapter, "ext", None), "device", None),
        "checkout": str(getattr(getattr(adapter, "ext", None), "root", "") or "") or None,
        "upstream_ref": getattr(getattr(adapter, "spec", None), "upstream_ref", None),
        "continuation_seed_rule": (
            "call 0 uses the declared deployment seed; call k>0 uses seed*1000+k. The deployment "
            "seed remains the replication unit; a continuation only completes a cell that has not "
            "reached the common cap and is never a re-draw of a finished one"
        ),
    }

    try:
        is_available = bool(adapter.available())
    except Exception as exc:  # an availability check that raises is a failure, not a zero
        return {
            **base,
            "status": "BACKEND_FAILURE",
            "outcome_class": "AVAILABILITY_CHECK_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "molecules": [],
            "n_unique_returned": 0,
            "funnel": _merge_funnels([], n_returned=0, n_duplicate_across_calls=0, n_beyond_cap=0),
            "calls": [],
            "cost": _cell_cost(started, before, None),
        }
    if not is_available:
        return {
            **base,
            "status": "BACKEND_FAILURE",
            "outcome_class": "UNAVAILABLE",
            "reason": getattr(getattr(adapter, "profile", None), "unavailable_reason", None)
            or "the adapter's capability card reports UNAVAILABLE",
            "molecules": [],
            "funnel": _merge_funnels([], n_returned=0, n_duplicate_across_calls=0, n_beyond_cap=0),
            "calls": [],
            "cost": _cell_cost(started, before, None),
        }

    try:
        payload, missing, provenance = _generation_inputs(adapter)
    except Exception as exc:  # the inputs could not even be assembled
        return {
            **base,
            "status": "BACKEND_FAILURE",
            "outcome_class": "INPUT_ASSEMBLY_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "molecules": [],
            "n_unique_returned": 0,
            "funnel": _merge_funnels([], n_returned=0, n_duplicate_across_calls=0, n_beyond_cap=0),
            "calls": [],
            "cost": _cell_cost(started, before, None),
        }
    if missing:
        return {
            **base,
            **provenance,
            "status": "BACKEND_FAILURE",
            "outcome_class": "MISSING_INPUT",
            "reason": f"this host has no {'; no '.join(missing)}",
            "molecules": [],
            "n_unique_returned": 0,
            "funnel": _merge_funnels([], n_returned=0, n_duplicate_across_calls=0, n_beyond_cap=0),
            "calls": [],
            "cost": _cell_cost(started, before, None),
        }

    base.update(provenance)
    with contextlib.suppress(Exception):
        base["weights"] = _loaded_weights(adapter)

    collected: dict[str, Record] = {}
    funnels: list[Record] = []
    calls: list[Record] = []
    n_duplicate_across_calls = 0
    n_beyond_cap = 0
    attempts = 0
    call_index = 0
    status = "COMPLETE"
    reason: str | None = None
    barren_calls = 0
    gpu_seconds_total = 0.0
    gpu_measured = True
    gpu_samples = 0

    while len(collected) < unique_cap and attempts < raw_cap:
        elapsed = time.time() - started
        if elapsed > cell_wall_seconds:
            status = "SHORTFALL"
            reason = (
                f"the cell's wall allowance of {cell_wall_seconds:.0f}s was exhausted after "
                f"{elapsed:.0f}s with {len(collected)} of {unique_cap} unique molecules and "
                f"{attempts} of {raw_cap} raw attempts consumed"
            )
            break
        want = unique_cap - len(collected)
        rate = yield_hint if (yield_hint and yield_hint > 0.0) else 1.0
        # One extra attempt per wanted molecule, so a call that under-delivers
        # by a little does not force a whole second model load.
        request = min(raw_cap - attempts, max(1, math.ceil(want / rate) + max(2, want // 5)))
        sub_seed = continuation_seed(seed, call_index)
        call_started = time.time()
        gpu: Record = {}
        try:
            # The yielded record is filled in place when the block exits, so
            # binding it here keeps the measurement reachable from the except
            # branch: a backend that failed still consumed GPU seconds.
            with gpu_accounting(descendant_of=os.getpid()) as sampled:
                gpu = sampled
                batch = adapter.generate(
                    n_molecules=int(request),
                    pocket=payload or None,
                    seed=int(sub_seed),
                    batch_size=int(min(batch_size, request)),
                    min_heavy_atoms=PROBE_MIN_HEAVY_ATOMS,
                    max_heavy_atoms=PROBE_MAX_HEAVY_ATOMS,
                    max_runtime_seconds=float(call_timeout_seconds),
                )
        except Exception as exc:  # a backend failure is a recorded outcome, not a crash
            gpu_seconds_total, gpu_measured, gpu_samples = _fold_gpu(
                gpu, gpu_seconds_total, gpu_measured, gpu_samples
            )
            calls.append(
                {
                    "call_index": call_index,
                    "seed": sub_seed,
                    "requested": int(request),
                    "wall_seconds": round(time.time() - call_started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "gpu_seconds": gpu.get("gpu_seconds"),
                    "gpu_telemetry": gpu.get("telemetry", "UNAVAILABLE"),
                }
            )
            status = "BACKEND_FAILURE"
            reason = (
                f"{type(exc).__name__}: {exc}. The backend started in {interpreter} and failed "
                f"after {time.time() - call_started:.1f}s on call {call_index}. This is a consumed "
                f"cost with no scientific outcome, not a yield of zero"
            )
            break

        gpu_seconds_total, gpu_measured, gpu_samples = _fold_gpu(
            gpu, gpu_seconds_total, gpu_measured, gpu_samples
        )
        returned = list(batch.molecules)
        funnel = generation_funnel(batch, len(returned))
        funnels.append(funnel)
        attempts += int(batch.n_raw)
        new = 0
        for molecule in returned:
            canonical = str(getattr(molecule, "smiles", "") or "")
            if not canonical:
                continue
            if canonical in collected:
                n_duplicate_across_calls += 1
                continue
            if len(collected) >= unique_cap:
                # Counted, not dropped: a molecule that arrived after the common
                # cap was full is a real sample with a real cost, and calling it
                # "screened out" would blame the heavy-atom window for it.
                n_beyond_cap += 1
                continue
            index = len(collected)
            collected[canonical] = {
                "molecule_id": f"{base['capability_id'].rsplit('.', 1)[-1]}_s{seed}_{index:04d}",
                "smiles": canonical,
                "n_heavy_atoms": getattr(molecule, "n_heavy_atoms", None),
                "sdf_block": getattr(molecule, "sdf_block", None),
                "qc_flags": list(getattr(molecule, "qc_flags", ()) or ()),
                "call_index": call_index,
                "call_seed": sub_seed,
                "sample_index": getattr(molecule, "sample_index", None),
            }
            new += 1
        calls.append(
            {
                "call_index": call_index,
                "seed": sub_seed,
                "requested": int(request),
                "n_raw": int(batch.n_raw),
                "n_valid": int(batch.n_valid),
                "n_unique": int(batch.n_unique),
                "n_returned": len(returned),
                "n_new": new,
                "wall_seconds": round(time.time() - call_started, 3),
                "gpu_seconds": gpu.get("gpu_seconds"),
                "gpu_telemetry": gpu.get("telemetry", "UNAVAILABLE"),
                "warnings": list(batch.warnings)[:3],
                "errors": list(batch.errors)[:3],
            }
        )
        call_index += 1
        if int(batch.n_raw) == 0:
            status = "BACKEND_FAILURE"
            reason = (
                "the backend returned a well-formed batch containing zero raw samples, so no "
                "further call can make progress. A consumed cost with no scientific outcome"
            )
            break
        if new == 0:
            barren_calls += 1
            if barren_calls >= 2:
                status = "SHORTFALL"
                reason = (
                    f"two consecutive continuation calls added no new structure, so the sampler "
                    f"is redrawing what it has already produced. Stopped at {len(collected)} of "
                    f"{unique_cap} unique molecules and {attempts} of {raw_cap} raw attempts "
                    f"rather than burning the remaining ceiling on duplicates"
                )
                break
        else:
            barren_calls = 0

    if status == "COMPLETE" and len(collected) < unique_cap:
        status = "SHORTFALL"
        reason = (
            f"the frozen ceiling of {raw_cap} raw attempts per arm per seed was exhausted with "
            f"{len(collected)} of {unique_cap} unique molecules. Appendix A section 3.3: a "
            f"shortfall against the common cap is a backend outcome with a consumed cost, never a "
            f"fixed-budget qualifying yield"
        )
    if status == "COMPLETE" and not collected:
        status = "EXECUTED_NO_YIELD"
    if status in {"SHORTFALL", "COMPLETE"} and not collected:
        status = "EXECUTED_NO_YIELD"
        reason = reason or (
            "the backend executed and every structure it produced fell outside the frozen "
            "10-70 heavy-atom window or repeated one already counted. A measured yield of zero "
            "from a working backend, not an absent backend and not an affinity of zero"
        )

    molecules = list(collected.values())
    return {
        **base,
        "status": status,
        "outcome_class": status,
        "reason": reason,
        "molecules": molecules,
        "n_unique_returned": len(molecules),
        "funnel": _merge_funnels(
            funnels,
            n_returned=len(molecules),
            n_duplicate_across_calls=n_duplicate_across_calls,
            n_beyond_cap=n_beyond_cap,
        ),
        "calls": calls,
        "cost": _cell_cost(
            started,
            before,
            {
                "gpu_seconds": round(gpu_seconds_total, 3) if gpu_measured else None,
                "gpu_telemetry": "MEASURED" if gpu_measured else "UNAVAILABLE",
                "gpu_samples": gpu_samples,
            },
        ),
    }


def _fold_gpu(gpu: Mapping[str, Any], total: float, measured: bool, samples: int) -> tuple[float, bool, int]:
    """Accumulate one call's GPU measurement without inventing one.

    A single call whose telemetry was unavailable makes the *cell's* total
    unknown: a sum that silently drops an unmeasured term is a smaller number
    presented as a fact. The already-accumulated seconds are kept so the record
    can still say what was seen.
    """
    if gpu.get("telemetry") == "MEASURED":
        return total + float(gpu.get("gpu_seconds") or 0.0), measured, samples + int(gpu.get("samples") or 0)
    return total, False, samples


def _cell_cost(started: float, before: Any, gpu: Mapping[str, Any] | None) -> Record:
    after = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    return {
        "wall_seconds": round(time.time() - started, 3),
        "cpu_user_seconds": round(after.ru_utime - before.ru_utime, 3),
        "cpu_system_seconds": round(after.ru_stime - before.ru_stime, 3),
        "max_rss_kb": int(after.ru_maxrss),
        "gpu_seconds": None if gpu is None else gpu.get("gpu_seconds"),
        "gpu_telemetry": "UNAVAILABLE" if gpu is None else gpu.get("gpu_telemetry", "UNAVAILABLE"),
        "gpu_samples": None if gpu is None else gpu.get("gpu_samples"),
    }


# ---------------------------------------------------------------------------
# Scoring — the frozen seven-gate pipeline, unchanged
# ---------------------------------------------------------------------------
def _heavy_atoms(smiles: str) -> int | None:
    """Heavy-atom count, recomputed from the submitted SMILES.

    Recomputed rather than carried across from the generation record, so the
    distribution is a property of the molecule that was scored rather than of
    the molecule the sampler said it drew.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else int(mol.GetNumHeavyAtoms())


def _qed(smiles: str) -> float | None:
    from rdkit import Chem
    from rdkit.Chem import QED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    with contextlib.suppress(Exception):
        return float(QED.qed(mol))
    return None


def _pose_sdf(molecule_id: str, block: str, poses_dir: Path) -> Path | None:
    """Write one docking pose as an SDF RDKit and PoseBusters can both read.

    The engines in this repository hand back a molblock, and not every one of
    them hands back a *well-formed* one: a V2000 block needs four header lines
    before its counts line and at least one path here produces three. Rather than
    patch the string, the block is parsed without sanitisation — which preserves
    the coordinates exactly, and is the point, because the pose is what is being
    judged — and written back through RDKit's own writer.

    Returns ``None`` when RDKit cannot read the block at all. That is an absent
    pose-quality measurement for one molecule, not a reason to stop scoring the
    other 1,349.
    """
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if mol is None and not block.lstrip().startswith("\n"):
        # A block missing its name line: give it one and try again. Stated
        # rather than silently prepended, because a header repair that changed
        # an atom would be a changed pose.
        mol = Chem.MolFromMolBlock("\n" + block, sanitize=False, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    mol.SetProp("_Name", molecule_id)
    path = poses_dir / f"{molecule_id}.sdf"
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()
    return path


def _posebusters_reports(rows: Sequence[Mapping[str, Any]], receptor_pdb: str, workdir: Path) -> Record:
    """PoseBusters over every docked pose, reported and gating nothing.

    Step 15B is explicit that the frozen ``geometry_limits`` are a *validity*
    screen at 0.8 A, deliberately far looser than PoseBusters' own criterion, and
    that they may not be reported as a pose-quality benchmark. PoseBusters is
    therefore run here as the generated-3D-quality evaluation Appendix A §3.3
    asks for — beside the docking score and outside the gate set.

    Because it gates nothing, it may not break anything either. One pose the
    validator cannot load is one missing exploratory measurement, recorded as
    such; it is not allowed to abort a scoring pass that is producing the primary.
    """
    from drug_agent.tools.structure.posebusters_tool import PoseBustersAdapter

    adapter = PoseBustersAdapter()
    if not adapter.available():
        return {
            "status": "UNAVAILABLE",
            "reason": getattr(adapter.profile, "unavailable_reason", None) or "unspecified",
            "reports": {},
        }
    poses_dir = workdir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    reports: Record = {}
    n_unloadable = 0
    n_failed_validation = 0
    engine_version = ""
    for row in rows:
        molecule_id = str(row["molecule_id"])
        block = str(row.get("pose_molblock") or "")
        if not block.strip():
            reports[molecule_id] = {"status": "NO_POSE", "reason": "the engine returned no pose block"}
            n_unloadable += 1
            continue
        path = _pose_sdf(molecule_id, block, poses_dir)
        if path is None:
            reports[molecule_id] = {
                "status": "UNREADABLE_POSE",
                "reason": "RDKit could not rebuild a conformer from the engine's molblock",
            }
            n_unloadable += 1
            continue
        try:
            report = adapter.validate(candidate_id=molecule_id, pose_sdf=path, receptor_pdb=receptor_pdb)
        except Exception as exc:  # an exploratory metric never stops the primary
            reports[molecule_id] = {
                "status": "VALIDATION_FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            n_failed_validation += 1
            continue
        engine_version = engine_version or report.engine_version
        reports[molecule_id] = {
            "status": "MEASURED",
            "passed": bool(report.passed),
            "n_checks": int(report.n_checks),
            "n_passed": int(report.n_passed),
            "n_failed": int(report.n_checks - report.n_passed),
            "failures": list(report.failures)[:10],
            "metrics": {k: float(v) for k, v in report.metrics.items() if _finite(v)},
        }

    measured = [r for r in reports.values() if r.get("status") == "MEASURED"]
    return {
        "status": "MEASURED" if measured else "NO_MEASUREMENT",
        "reason": None,
        "label": EXPLORATORY,
        "gates_nothing": (
            "PoseBusters is the generated-3D-quality evaluation Appendix A section 3.3 asks for. "
            "It is NOT one of the seven gates and no molecule qualifies or fails on it"
        ),
        "engine_version": engine_version,
        "n_poses": len(rows),
        "n_measured": len(measured),
        "n_unloadable": n_unloadable,
        "n_validation_failed": n_failed_validation,
        "n_passing_all_checks": sum(1 for r in measured if r.get("passed")),
        "reports": reports,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def receptor_entity_census(receptor_pdb: str | Path) -> Record:
    """How many atoms of each entity type this receptor actually contains.

    Counted with **PoseBusters' own atom typing**, not with a grep for HETATM, so
    the answer is the one the checks themselves would reach. A prepared docking
    receptor normally has its cofactors and waters stripped; when it does, every
    cofactor and water column in ``dock`` mode is answering a question about an
    empty set, and the honest word for the answer is *vacuous* rather than
    *failed*.
    """
    from posebusters.tools.protein import get_atom_type_mask
    from rdkit import Chem

    path = Path(receptor_pdb)
    if not path.is_file():
        return {"status": "UNAVAILABLE", "reason": f"no such receptor file: {path}", "entities": {}}
    mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False, proximityBonding=False)
    if mol is None:
        return {"status": "UNREADABLE", "reason": "RDKit could not read the receptor", "entities": {}}
    every = {"hydrogens", "protein", "organic_cofactors", "inorganic_cofactors", "waters"}
    entities = {
        entity: int(sum(get_atom_type_mask(mol, (every - {entity}) | {"hydrogens"})))
        for entity in ("protein", "organic_cofactors", "inorganic_cofactors", "waters")
    }
    return {
        "status": "MEASURED",
        "receptor": str(path),
        "sha256": sha256_file(path),
        "n_atoms": int(mol.GetNumAtoms()),
        "entities": entities,
        "absent_entities": sorted(name for name, n in entities.items() if n == 0),
        "how": "posebusters.tools.protein.get_atom_type_mask, the same typing the checks use",
    }


def sha256_file(path: str | Path) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def classify_pose_columns(census: Mapping[str, Any]) -> Record:
    """Label every boolean column this repository counts as a PoseBusters check.

    Four labels, and the classification is *derived* rather than asserted:

    ``INTRAMOLECULAR``
        the ligand judged on its own — the generated-3D-quality battery.
    ``PROTEIN_CONTEXT``
        needs the receptor, and the entity it needs is present in this one.
    ``VACUOUS_ENTITY_ABSENT``
        needs an entity this receptor does not contain. Its value is a property
        of the receptor file and says nothing about any generator. Whether a
        given column lands here is read off :func:`receptor_entity_census`, so a
        receptor that *does* carry waters would classify differently.
    ``NOT_A_POSEBUSTERS_TEST``
        a ``full_report`` diagnostic column that ``dock.yml`` never selects as a
        test, including one whose polarity is inverted.
    """
    entities = dict(census.get("entities") or {})
    out: Record = {}
    for name in POSEBUSTERS_INTRAMOLECULAR_CHECKS:
        out[name] = {
            "kind": "INTRAMOLECULAR",
            "entity": None,
            "selected_by_posebusters": True,
            "polarity": "NORMAL",
            "note": None,
        }
    for name, entity in POSEBUSTERS_CHECK_ENTITY.items():
        non_test = POSEBUSTERS_NON_TEST_COLUMNS.get(name)
        present = entities.get(entity)
        if non_test is not None:
            kind = "NOT_A_POSEBUSTERS_TEST"
        elif present == 0:
            kind = "VACUOUS_ENTITY_ABSENT"
        else:
            kind = "PROTEIN_CONTEXT"
        out[name] = {
            "kind": kind,
            "entity": entity,
            "entity_atoms_in_this_receptor": present,
            "selected_by_posebusters": non_test is None,
            "polarity": ("INVERTED" if name == "most_extreme_clash_protein" else "NORMAL"),
            "note": non_test,
        }
    return out


def _check_tally(
    per_pose: Sequence[Mapping[str, bool]],
    classification: Mapping[str, Any],
) -> Record:
    """Per-column applicable/passed/failed counts across a set of poses."""
    tally: Record = {}
    for checks in per_pose:
        for name, ok in checks.items():
            row = tally.setdefault(
                name,
                {
                    "n_applicable": 0,
                    "n_passed": 0,
                    "n_failed": 0,
                    **dict(classification.get(name) or {"kind": "UNCLASSIFIED"}),
                },
            )
            row["n_applicable"] += 1
            row["n_passed" if ok else "n_failed"] += 1
    for row in tally.values():
        n = int(row["n_applicable"])
        row["pass_rate"] = round(row["n_passed"] / n, 6) if n else None
    return dict(sorted(tally.items()))


def _validate_one_pose(task: tuple[str, str, str | None, bool]) -> Record:
    """One pose through PoseBusters, in ``dock`` and optionally also ``mol``.

    Module level and plain-dict in/out so it survives pickling into a worker
    process.

    **Why ``mol`` is optional, and what it costs.** ``mol.yml`` declares the
    ``energy_ratio`` module **twice** — the first copy selects no binary output,
    so it contributes no check — and PoseBusters caches the conformer ensemble
    with :func:`functools.cache`, keyed on the ligand's InChI. ``functools.cache``
    does **not** cache a call that raised, and the ensemble builder raises on
    exactly the ligands it cannot embed: large, strained, flexible ones. Those
    ligands therefore pay for the ensemble **three** times in a ``mol``-then-
    ``dock`` pass (twice for ``mol``'s duplicated module, once more for ``dock``)
    against the **one** attempt the sweep's ``dock``-only pass paid — and they are
    precisely the ligands that dominate the cost. Measured here: it turned a
    45-minute arm into a >2-hour one.

    Running ``dock`` alone costs exactly what the sweep costed and loses nothing,
    because ``mol.yml`` and ``dock.yml`` declare the fourteen intramolecular
    modules with identical functions and identical parameters. ``mol`` remains
    available, and the two are cross-checked against each other in §5 of the
    step-15D handoff on three arms and 437 poses, with exact agreement.
    """
    from drug_agent.tools.structure.posebusters_tool import PoseBustersAdapter

    molecule_id, sdf_path, receptor, run_mol = task
    adapter = PoseBustersAdapter()
    try:
        dock_report = (
            adapter.validate(candidate_id=molecule_id, pose_sdf=sdf_path, receptor_pdb=receptor)
            if receptor
            else None
        )
        mol_report = (
            adapter.validate(candidate_id=molecule_id, pose_sdf=sdf_path)
            if (run_mol or dock_report is None)
            else None
        )
    except Exception as exc:  # an exploratory metric never stops the primary
        return {
            "molecule_id": molecule_id,
            "status": "VALIDATION_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    source = mol_report if mol_report is not None else dock_report
    assert source is not None
    intramolecular = {
        name: ok for name, ok in source.checks.items() if name in POSEBUSTERS_INTRAMOLECULAR_CHECKS
    }
    return {
        "molecule_id": molecule_id,
        "status": "MEASURED",
        "engine_version": source.engine_version,
        "mol_checks": intramolecular,
        "intramolecular_mode": "mol" if mol_report is not None else "dock",
        "mol_metrics": {k: float(v) for k, v in source.metrics.items() if _finite(v)},
        "dock_checks": (None if dock_report is None else dict(dock_report.checks)),
    }


def recompute_pose_quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    receptor_pdb: str | Path | None,
    workdir: Path,
    stored: Mapping[str, Any] | None = None,
    verbose: bool = False,
    jobs: int = 1,
    mol_mode: bool = True,
) -> Record:
    """Re-measure pose quality from the stored ``pose_molblock``, decomposed.

    **Nothing is generated and nothing is re-docked.** Every coordinate this
    reads was written by the sweep; PoseBusters is pure CPU over RDKit, so the
    whole recomputation costs zero GPU-seconds. That is the point: the defect
    this repairs is in how a measurement was *summarised*, not in the poses.

    Two numbers come out, and they are reported apart because they answer
    different questions:

    **Intramolecular validity** — the headline. PoseBusters' ``mol``
    configuration: is this a physically possible molecule? For a de novo
    molecule with no known native pose that is the whole of what Appendix A
    §3.3's generated-3D-quality evaluation can ask, and it is answerable without
    a receptor at all.

    **Protein-context checks** — reported separately, per column, each labelled
    with whether the entity it needs exists in this receptor. Against a prepared
    docking receptor with its cofactors and waters stripped, most of them are
    vacuous. A vacuous check is not a failed check and a report that adds the two
    together is not measuring a generator.

    ``jobs`` spreads the poses across worker processes. It changes no verdict —
    PoseBusters' internal-energy ensemble is seeded — and exists because a
    handful of large strained ligands take minutes each on the conformer
    ensemble while the rest take under a second.
    """
    from drug_agent.tools.structure.posebusters_tool import PoseBustersAdapter

    adapter = PoseBustersAdapter()
    if not adapter.available():
        return {
            "status": "UNAVAILABLE",
            "reason": getattr(adapter.profile, "unavailable_reason", None) or "unspecified",
            "reports": {},
        }
    poses_dir = Path(workdir) / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    census = (
        receptor_entity_census(receptor_pdb) if receptor_pdb else {"status": "NOT_SUPPLIED", "entities": {}}
    )
    classification = classify_pose_columns(census)

    reports: Record = {}
    tasks: list[tuple[str, str, str | None, bool]] = []
    n_no_pose = n_unreadable = 0
    for row in rows:
        molecule_id = str(row["molecule_id"])
        block = str(row.get("pose_molblock") or "")
        if not block.strip():
            reports[molecule_id] = {"status": "NO_POSE", "reason": "the engine returned no pose block"}
            n_no_pose += 1
            continue
        path = _pose_sdf(molecule_id, block, poses_dir)
        if path is None:
            reports[molecule_id] = {
                "status": "UNREADABLE_POSE",
                "reason": "RDKit could not rebuild a conformer from the engine's molblock",
            }
            n_unreadable += 1
            continue
        tasks.append((molecule_id, str(path), str(receptor_pdb) if receptor_pdb else None, bool(mol_mode)))

    results = _run_pose_tasks(tasks, jobs=jobs, verbose=verbose)

    intramolecular: list[Mapping[str, bool]] = []
    context: list[Mapping[str, bool]] = []
    n_failed = 0
    engine_version = ""
    intramolecular_modes: set[str] = set()
    stored_reports = dict((stored or {}).get("reports") or {})
    reproduced = mismatched = compared = 0

    for result in results:
        molecule_id = str(result["molecule_id"])
        if result.get("status") != "MEASURED":
            reports[molecule_id] = {
                "status": "VALIDATION_FAILED",
                "reason": result.get("reason"),
            }
            n_failed += 1
            continue
        engine_version = engine_version or str(result.get("engine_version") or "")
        intramolecular_modes.add(str(result.get("intramolecular_mode") or "mol"))
        mol_checks = dict(result["mol_checks"])
        intramolecular.append(mol_checks)
        mol_failures = sorted(name for name, ok in mol_checks.items() if not ok)
        record: Record = {
            "status": "MEASURED",
            "intramolecular": {
                "mode": str(result.get("intramolecular_mode") or "mol"),
                "passed": not mol_failures,
                "n_checks": len(mol_checks),
                "n_failed": len(mol_failures),
                "failures": mol_failures,
            },
            "metrics": dict(result.get("mol_metrics") or {}),
        }
        dock_checks = result.get("dock_checks")
        if dock_checks is not None:
            dock_checks = dict(dock_checks)
            context.append(dock_checks)
            dock_failures = sorted(name for name, ok in dock_checks.items() if not ok)
            genuine = sorted(
                name
                for name in dock_failures
                if (classification.get(name) or {}).get("kind") in ("INTRAMOLECULAR", "PROTEIN_CONTEXT")
            )
            record["protein_context"] = {
                "mode": "dock",
                "n_columns": len(dock_checks),
                "failures_all_columns": dock_failures,
                "failures_excluding_vacuous_and_non_tests": genuine,
            }
            # Kept under its old name so the legacy `posebusters_n_failed`
            # distribution still resolves and stays comparable across runs. Its
            # caveat travels with it in DISTRIBUTION_NOTES.
            record["n_failed"] = len(dock_failures)
            record["n_checks"] = len(dock_checks)
            record["passed"] = not dock_failures
            record["failures"] = dock_failures
            previous = stored_reports.get(molecule_id) or {}
            if previous.get("status") == "MEASURED":
                compared += 1
                if sorted(previous.get("failures") or ()) == dock_failures:
                    reproduced += 1
                else:
                    mismatched += 1
        reports[molecule_id] = record

    measured = [r for r in reports.values() if r.get("status") == "MEASURED"]
    intra_tally = _check_tally(intramolecular, classification)
    context_tally = _check_tally(context, classification)
    n_intra_clean = sum(1 for c in intramolecular if all(c.values()))
    mixed_clean = sum(1 for c in context if all(c.values()))

    def by_kind(tally: Mapping[str, Any], kind: str) -> Record:
        return {name: row for name, row in tally.items() if row.get("kind") == kind}

    return {
        "status": "MEASURED" if measured else "NO_MEASUREMENT",
        "reason": None,
        "label": EXPLORATORY,
        "gates_nothing": (
            "PoseBusters is the generated-3D-quality evaluation Appendix A section 3.3 asks for. "
            "It is NOT one of the seven gates and no molecule qualifies or fails on it"
        ),
        "engine_version": engine_version,
        "invocation": POSEBUSTERS_INVOCATION_NOTE,
        "recomputed_from": (
            "the pose_molblock stored on each scoring row. No molecule was regenerated, no pose "
            "was re-docked, and PoseBusters is CPU-only: this recomputation cost 0.0 GPU-seconds"
        ),
        "n_poses": len(rows),
        "n_measured": len(measured),
        "n_no_pose": n_no_pose,
        "n_unreadable": n_unreadable,
        "n_validation_failed": n_failed,
        "intramolecular_validity": {
            "headline": True,
            "mode": ("+".join(sorted(intramolecular_modes)) if intramolecular_modes else "mol"),
            "mode_note": POSEBUSTERS_INTRAMOLECULAR_MODE_NOTE,
            "what_it_answers": "is this a physically possible molecule, judged on its own",
            "n_measured": len(intramolecular),
            "n_passing_every_check": n_intra_clean,
            "pass_rate": (round(n_intra_clean / len(intramolecular), 6) if intramolecular else None),
            "checks": intra_tally,
        },
        "protein_context": {
            "headline": False,
            "mode": "dock",
            "what_it_answers": "does this pose sit sensibly in THIS receptor file",
            "n_measured": len(context),
            "receptor_census": census,
            "vacuity": (
                "columns labelled VACUOUS_ENTITY_ABSENT need an entity this receptor does not "
                "contain. Their value is a property of the receptor file and says NOTHING about "
                "any generator. They are reported, labelled, and excluded from every headline"
            ),
            "applicable_checks": by_kind(context_tally, "PROTEIN_CONTEXT"),
            "vacuous_checks": by_kind(context_tally, "VACUOUS_ENTITY_ABSENT"),
            "columns_that_are_not_posebusters_tests": by_kind(context_tally, "NOT_A_POSEBUSTERS_TEST"),
            "unclassified_columns": by_kind(context_tally, "UNCLASSIFIED"),
        },
        "all_counted_columns_mixed": {
            "n_passing_every_counted_column": mixed_clean,
            "n_columns_counted": max((len(c) for c in context), default=0),
            "why": POSEBUSTERS_MIXED_TOTAL_NOTE,
        },
        "reproduces_the_recorded_run": {
            "compared": compared,
            "identical": reproduced,
            "different": mismatched,
            "note": (
                "the dock-mode recomputation is compared pose by pose against the verdicts the "
                "sweep itself recorded. PoseBusters' internal-energy ensemble is seeded, so an "
                "identical count is the expected result and a difference would mean the stored "
                "pose and the scored pose were not the same coordinates"
            ),
        },
        "reports": reports,
    }


def _run_pose_tasks(
    tasks: Sequence[tuple[str, str, str | None, bool]], *, jobs: int, verbose: bool
) -> list[Record]:
    """Validate every pose, on one process or several.

    The worker pool is ``spawn``-started so each child reads the single-threaded
    BLAS environment set below rather than inheriting a parent that already
    built a thread pool. Twelve RDKit workers each opening twelve BLAS threads
    is slower than one.
    """
    if not tasks:
        return []
    if jobs <= 1:
        out: list[Record] = []
        for index, task in enumerate(tasks):
            out.append(_validate_one_pose(task))
            if verbose and (index + 1) % 25 == 0:
                print(f"    pose quality {index + 1}/{len(tasks)}", flush=True)
        return out

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    env = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1")
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        with ProcessPoolExecutor(
            max_workers=int(jobs), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = []
            for index, result in enumerate(pool.map(_validate_one_pose, list(tasks))):
                results.append(result)
                if verbose and (index + 1) % 25 == 0:
                    print(f"    pose quality {index + 1}/{len(tasks)}", flush=True)
            return results
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def score_molecules(
    molecules: Sequence[tuple[str, str]],
    *,
    scorer: Mapping[str, Any],
    workdir: Path,
    posebusters: bool = True,
) -> Record:
    """Dock, profile and describe every submitted molecule, once.

    Reuses the Track B pipeline's own components — ``_dock_portfolio``,
    ``_generation_quality_endpoints`` and ``_descriptors`` — rather than writing
    a second scoring path, so the rows this produces are the rows
    ``evaluate_portfolio`` already knows how to read.
    """
    import importlib.util

    from drug_agent.selfevolve.target.baseline_arms import (
        _descriptors,
        _dock_portfolio,
        _generation_quality_endpoints,
        _repo_root,
    )

    workdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    docking_dir = (workdir / "docking").resolve()
    docking_dir.mkdir(parents=True, exist_ok=True)

    dock_started = time.time()
    with gpu_accounting(descendant_of=os.getpid()) as dock_gpu:
        rows = _dock_portfolio(list(molecules), scorer=scorer, workdir=docking_dir)
    dock_wall = time.time() - dock_started

    endpoint_started = time.time()
    with gpu_accounting(descendant_of=os.getpid()) as endpoint_gpu:
        endpoints = _generation_quality_endpoints(list(molecules))
    endpoint_wall = time.time() - endpoint_started

    for row in rows:
        row["endpoints"] = endpoints.get(str(row["molecule_id"]), {})
        if not row.get("properties"):
            row["properties"] = _descriptors(str(row["smiles"]))
        row["qed"] = _qed(str(row["smiles"]))
        row["n_heavy_atoms"] = _heavy_atoms(str(row["smiles"]))

    # ``_repo_root()``, not the module-level ``REPO``. Inside a candidate
    # checkout those are different directories: a candidate is a git worktree,
    # ``.external/`` is gitignored and therefore absent from it, and the
    # dispatcher passes the parent's root as ``DRUGAGENT_REPO_ROOT`` for exactly
    # this reason. ``_dock_portfolio`` above already honours it -- so the docking
    # half of this function resolved the receptor from the parent while this half
    # looked for it inside the candidate and raised SystemExit on the missing
    # 6JUU.pdb, after the docking pass had already run to completion.
    spec = importlib.util.spec_from_file_location(
        "zak_redock_sweep", _repo_root() / "scripts" / "research" / "dock_zak_reference_ligand.py"
    )
    receptor = ""
    receptor_error: str | None = None
    if spec is not None and spec.loader is not None:
        # This preparation sits OUTSIDE ``_posebusters_reports`` and so was
        # outside its "may not break anything" guarantee, which is what let a
        # pose-quality input abort a scoring pass that was producing the primary.
        # PoseBusters gates nothing; neither may the receptor it needs.
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            receptor = str(module.prepare_inputs(docking_dir)["receptor_pdb"])
        except (OSError, SystemExit, ValueError, KeyError) as exc:
            receptor_error = f"{type(exc).__name__}: {exc}"
    else:
        receptor_error = "the receptor preparation module could not be loaded"

    pose_started = time.time()
    if not posebusters:
        pose_quality: Record = {"status": "NOT_REQUESTED", "reason": None, "reports": {}}
    elif receptor_error is not None:
        pose_quality = {
            "status": "UNAVAILABLE",
            "reason": f"the protein context could not be prepared ({receptor_error})",
            "reports": {},
        }
    else:
        pose_quality = _posebusters_reports(rows, receptor, workdir)
    pose_wall = time.time() - pose_started
    for row in rows:
        report = (pose_quality.get("reports") or {}).get(str(row["molecule_id"]))
        row["posebusters"] = report

    return {
        "rows": rows,
        "pose_quality": pose_quality,
        "cost": {
            "wall_seconds": round(time.time() - started, 3),
            "docking_wall_seconds": round(dock_wall, 3),
            "docking_gpu_seconds": dock_gpu.get("gpu_seconds")
            if dock_gpu.get("telemetry") == "MEASURED"
            else None,
            "docking_gpu_telemetry": dock_gpu.get("telemetry", "UNAVAILABLE"),
            "endpoint_wall_seconds": round(endpoint_wall, 3),
            "endpoint_gpu_seconds": endpoint_gpu.get("gpu_seconds")
            if endpoint_gpu.get("telemetry") == "MEASURED"
            else None,
            "endpoint_gpu_telemetry": endpoint_gpu.get("telemetry", "UNAVAILABLE"),
            "posebusters_wall_seconds": round(pose_wall, 3),
            "n_molecules": len(molecules),
            "seconds_per_molecule": (
                round((time.time() - started) / len(molecules), 4) if molecules else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# The primary table, and the correction zak-c9r-2 registered in advance
# ---------------------------------------------------------------------------
def holm(pvalues: Mapping[str, float], alpha: float = FAMILY_WISE_ALPHA) -> Record:
    """Holm-Bonferroni over a family of tests, at family-wise ``alpha``.

    Holm sorts the p-values ascending and compares the *i*-th (0-based) against
    ``alpha / (m - i)``, stopping at the first failure: everything from there on
    is retained, whatever its own p-value. That step-down is the whole reason
    Holm is uniformly more powerful than Bonferroni while controlling the same
    family-wise error rate, and it is also why a rejection cannot be read off one
    arm's p-value alone.

    ``zak-c9r-2`` registered this correction across the nine admitted arms
    **before any arm ran** (``statistics.multiplicity_correction``), which is the
    only way registering it means anything.
    """
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    rows: list[Record] = []
    still_rejecting = True
    running = 0.0
    for index, (name, p) in enumerate(items):
        bar = alpha / (m - index)
        # The adjusted p-value is the running maximum of (m-i)*p, capped at 1.
        running = min(1.0, max(running, (m - index) * float(p)))
        rejected = bool(still_rejecting and float(p) <= bar)
        if not rejected:
            still_rejecting = False
        rows.append(
            {
                "name": name,
                "p_value": float(p),
                "rank": index + 1,
                "holm_threshold": bar,
                "holm_adjusted_p": running,
                "rejected": rejected,
            }
        )
    return {
        "procedure": "holm-bonferroni, step-down",
        "family_wise_alpha": alpha,
        "family_size": m,
        "family": sorted(pvalues),
        "rows": rows,
        "n_rejected": sum(1 for r in rows if r["rejected"]),
        "registered": ("zak-c9r-2 statistics.multiplicity_correction, registered before any arm was run"),
    }


def arm_p_value(count: int, *, baseline_lambda: float) -> float:
    """P(a baseline arm produces at least ``count`` qualifying scaffolds).

    The null is the measured baseline qualification rate at the locked per-arm
    budget, which is the only null this study has evidence for. ``count`` is a
    count of scaffolds, so the upper tail of a Poisson with that mean is the
    exact one-sided p-value; the orientation is ``HIGHER_IS_BETTER``, so only the
    upper tail is a rejection.
    """
    from drug_agent.selfevolve.target.protocol import poisson_sf

    return float(poisson_sf(int(count), float(baseline_lambda)))


def comparator_bar(name: str = REGISTERED_COMPARATOR) -> Record:
    """The named comparator bar, refusing a name nobody registered.

    The default is the bar step 15E **registered**, not the one it retired. A
    caller that says nothing therefore gets the superseding rate; the retired
    Stage D null is still reachable, but only by naming it.
    """
    try:
        bar = COMPARATOR_BARS[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown comparator bar {name!r}; registered: {sorted(COMPARATOR_BARS)}"
        ) from None
    return dict(bar)


def comparator_lambda(bar: Mapping[str, Any], caps: Mapping[str, Any] | None = None) -> float:
    """The per-arm Poisson mean a bar's per-molecule rate implies at ``caps``."""
    caps = dict(caps or {})
    return (
        float(bar["rate_per_molecule"])
        * float(caps.get("unique_molecules_per_arm_per_seed", 50))
        * float(caps.get("seeds", 3))
    )


def holm_equivalent_scaffolds(
    baseline_lambda: float,
    *,
    alpha: float = FAMILY_WISE_ALPHA,
    family_size: int = COMPARATOR_FAMILY_SIZE,
) -> int:
    """The smallest per-arm count whose upper tail survives Holm across the family.

    Holm's smallest adjusted threshold is ``alpha / family_size`` — the one the
    most significant hypothesis must clear — so this is the count an arm needs
    before any rejection is possible at all. Derived, never typed: at the retired
    bar it returns 6 and at the superseding bar 7, and a future change to either
    rate moves it without anyone having to remember to.
    """
    target = float(alpha) / float(family_size)
    return next(k for k in range(0, 1000) if arm_p_value(k, baseline_lambda=baseline_lambda) <= target)


def _summary(values: Sequence[float]) -> Record:
    """Five-number summary plus mean and sd, or explicit nulls for an empty set."""
    clean = [float(v) for v in values if v is not None and _finite(v)]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    ordered = sorted(clean)

    def q(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 6),
        "sd": round(statistics.stdev(ordered), 6) if len(ordered) > 1 else 0.0,
        "min": round(ordered[0], 6),
        "p25": round(q(0.25), 6),
        "median": round(q(0.5), 6),
        "p75": round(q(0.75), 6),
        "max": round(ordered[-1], 6),
    }


def _endpoint_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    """Pull every reported endpoint off the scored rows, unconstrained by any gate."""
    out: dict[str, list[float | None]] = {name: [] for name in DISTRIBUTION_ENDPOINTS}
    for row in rows:
        score = row.get("structure_score") or {}
        ok = score.get("status") == "SUCCEEDED" and score.get("value") is not None
        out["vina_affinity"].append(float(score["value"]) if ok else None)
        endpoints = row.get("endpoints") or {}
        for metric in ("herg", "sa_score"):
            record = endpoints.get(metric) or {}
            out[metric].append(
                float(record["value"])
                if record.get("status") == "SUCCEEDED" and record.get("value") is not None
                else None
            )
        properties = row.get("properties") or {}
        descriptors = properties.get("descriptors") or {}
        out["mw"].append(_maybe_float(descriptors.get("mw")))
        out["logp"].append(_maybe_float(descriptors.get("logp")))
        out["n_structural_alerts"].append(_maybe_float(properties.get("n_structural_alerts")))
        out["qed"].append(_maybe_float(row.get("qed")))
        out["n_heavy_atoms"].append(_maybe_float(row.get("n_heavy_atoms")))
        pose = row.get("posebusters") or {}
        out["posebusters_n_failed"].append(_maybe_float(pose.get("n_failed")))
        intra = pose.get("intramolecular") or {}
        out["posebusters_intramolecular_n_failed"].append(_maybe_float(intra.get("n_failed")))
    return out


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def distributions(rows: Sequence[Mapping[str, Any]]) -> Record:
    """Every endpoint's unconstrained distribution. EXPLORATORY, and labelled.

    Step 15C measured that the registered primary — a count whose mode is zero —
    cannot separate nine arms at three seeds at any effect size, and said in the
    lock that these distributions are where any separation will actually be
    visible. It also said, in the same paragraph, that they are **not** the
    primary and may not be promoted to it after the counts are seen: a
    per-molecule comparison between arms is the pseudo-replication the lock
    prohibits as process-level confidence.
    """
    values = _endpoint_values(rows)
    return {
        "label": EXPLORATORY,
        "why_exploratory": (
            "not the registered primary, and a per-molecule comparison between arms is the "
            "per-molecule pseudo-replication zak-c9r-2 prohibits as process-level confidence. "
            "Reported because Appendix A section 3.3 requires it and because a count with mode "
            "zero cannot show where the arms differ"
        ),
        "n_rows": len(rows),
        "endpoints": {
            name: {
                **_summary([v for v in vals if v is not None]),
                "n_missing": sum(1 for v in vals if v is None),
                "orientation": _orientation(name),
                **({"note": DISTRIBUTION_NOTES[name]} if name in DISTRIBUTION_NOTES else {}),
            }
            for name, vals in values.items()
        },
    }


def _orientation(metric: str) -> str:
    from drug_agent.selfevolve.evaluation.target_benchmark import score_orientation

    with contextlib.suppress(Exception):
        return str(score_orientation(metric))
    return "UNREGISTERED_REPORTED_ONLY"


def novelty_and_similarity(
    rows: Sequence[Mapping[str, Any]],
    exclusion_smiles: Sequence[str],
    *,
    n_reference_withheld: int = 0,
) -> Record:
    """Scaffold novelty and nearest-neighbour similarity, reported apart.

    Appendix A §3.3 is explicit that **scaffold novelty is not required for an
    improved analog-design method**, so neither number here gates anything and
    neither belongs in the headline. They are reported because §3.3 also requires
    them to be reported, separately.

    The reference set is the frozen common exclusion union — the same molecules
    the novelty gate uses — so "new" here means "not one of the molecules this
    study has already seen", not "unknown to chemistry".

    Both numbers need the reference molecules' **structures**: a Tanimoto needs a
    fingerprint and a scaffold comparison needs a scaffold, and neither can be
    computed from a digest. When the exclusion set is carried as digests — which
    is how it is carried now, so that reading the gates does not also hand over
    every molecule this programme has accepted — *exclusion_smiles* arrives empty
    and both numbers are reported as unavailable, with the withheld count and the
    reason. They are not reconstructed from some other copy of the corpus: doing
    that here would put the structures back into the campaign's own output, which
    is the leak this carries no plaintext in order to close.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from drug_agent.selfevolve.target.baseline_arms import _fingerprint_generator

    RDLogger.DisableLog("rdApp.*")
    generator = _fingerprint_generator(MORGAN_RADIUS, MORGAN_BITS)

    reference_fps = []
    reference_scaffolds: set[str] = set()
    for smiles in exclusion_smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        reference_fps.append(generator.GetFingerprint(mol))
        with contextlib.suppress(Exception):
            reference_scaffolds.add(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
    reference_scaffolds.discard("")

    per_molecule: list[Record] = []
    scaffolds: set[str] = set()
    new_scaffolds: set[str] = set()
    for row in rows:
        smiles = str(row.get("smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scaffold = ""
        with contextlib.suppress(Exception):
            scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        similarity = None
        if reference_fps:
            similarity = max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), reference_fps))
        key = scaffold or f"acyclic:{row.get('molecule_id')}"
        scaffolds.add(key)
        is_new = bool(scaffold) and scaffold not in reference_scaffolds
        if is_new or not scaffold:
            new_scaffolds.add(key)
        per_molecule.append(
            {
                "molecule_id": row.get("molecule_id"),
                "scaffold": key,
                "scaffold_is_new": bool(is_new or not scaffold),
                "nearest_neighbour_similarity": None if similarity is None else round(similarity, 6),
                "qualifies": bool(row.get("qualifies")),
            }
        )

    similarities = [
        r["nearest_neighbour_similarity"]
        for r in per_molecule
        if r["nearest_neighbour_similarity"] is not None
    ]
    qualifying = [r for r in per_molecule if r["qualifies"]]
    return {
        "label": EXPLORATORY,
        "reported_separately_from_the_headline": True,
        "why_it_does_not_gate": (
            "Appendix A section 3.3: scaffold novelty is not required for an improved "
            "analog-design method, so it informs and does not gate"
        ),
        "reference_set": "the frozen common exclusion union used by the novelty gate",
        "n_reference_molecules": len(reference_fps),
        "n_reference_scaffolds": len(reference_scaffolds),
        **(
            {
                "reference_structures_withheld": int(n_reference_withheld),
                "unavailable": (
                    "the novelty gate's exclusion set is carried as digests, not as plaintext "
                    "SMILES, so no fingerprint and no scaffold can be built for the "
                    f"{int(n_reference_withheld)} reference molecules. scaffold_is_new and "
                    "nearest_neighbour_similarity are therefore NOT MEASURED here rather than "
                    "measured against an empty reference, and every fraction below that divides "
                    "by the reference set is unavailable for the same reason"
                ),
            }
            if n_reference_withheld and not reference_fps
            else {}
        ),
        "n_molecules": len(per_molecule),
        "n_distinct_scaffolds": len(scaffolds),
        "n_new_scaffolds": len(new_scaffolds),
        "fraction_new_scaffolds": (round(len(new_scaffolds) / len(scaffolds), 6) if scaffolds else None),
        "nearest_neighbour_similarity": _summary(similarities),
        "qualifying_molecules": {
            "n": len(qualifying),
            "n_new_scaffolds": sum(1 for r in qualifying if r["scaffold_is_new"]),
            "nearest_neighbour_similarity": _summary(
                [
                    r["nearest_neighbour_similarity"]
                    for r in qualifying
                    if r["nearest_neighbour_similarity"] is not None
                ]
            ),
        },
        "per_molecule": per_molecule,
    }


# ---------------------------------------------------------------------------
# Orchestration, resumable per arm per seed
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return path


def _cell_dir(output: Path, arm: str, seed: int) -> Path:
    return output / "cells" / arm / f"seed-{seed}"


def _load_checkpoint(path: Path) -> Record | None:
    if not path.is_file():
        return None
    with contextlib.suppress(json.JSONDecodeError, OSError):
        record = json.loads(path.read_text())
        if isinstance(record, dict) and record.get("contract_version"):
            return record
    return None


def measure_endpoint_rate(smiles: Sequence[str]) -> Record:
    """Time the endpoint panel on a handful of molecules, to size the estimate.

    The estimate has to be made before the spend, and this programme has no
    frozen rate for the ADMET-AI + SA + alerts pass — step 15B ran it but did not
    meter it, and recorded its GPU seconds as ``UNAVAILABLE`` rather than as
    zero. So a rate is measured here rather than assumed.

    The measurement **overstates** the per-molecule cost at sweep scale, and by
    a known mechanism: ADMET-AI's cost is dominated by per-call Lightning setup
    rather than by batch size, so a small batch pays the whole fixed cost over
    very few molecules. That is the safe direction for a budget and it is stated
    rather than corrected for.
    """
    from drug_agent.selfevolve.target.baseline_arms import _generation_quality_endpoints

    pairs = [(f"rateprobe_{i:03d}", s) for i, s in enumerate(smiles)]
    started = time.time()
    with gpu_accounting(descendant_of=os.getpid()) as gpu:
        endpoints = _generation_quality_endpoints(pairs)
    wall = time.time() - started
    n = max(1, len(pairs))
    measured = gpu.get("telemetry") == "MEASURED"
    n_scored = sum(
        1
        for record in endpoints.values()
        if (record.get("herg") or {}).get("status") == "SUCCEEDED"
        or (record.get("sa_score") or {}).get("status") == "SUCCEEDED"
    )
    return {
        "n_probe_molecules": len(pairs),
        "n_scored": n_scored,
        "wall_seconds": round(wall, 3),
        "seconds_per_molecule": round(wall / n, 4),
        "gpu_seconds": round(float(gpu.get("gpu_seconds") or 0.0), 3) if measured else None,
        "gpu_seconds_per_molecule": (
            round(float(gpu.get("gpu_seconds") or 0.0) / n, 4) if measured else None
        ),
        "gpu_telemetry": gpu.get("telemetry", "UNAVAILABLE"),
        "evidence_class": "measured",
        "direction_of_error": (
            "OVERSTATES at sweep scale: ADMET-AI's cost is dominated by per-call Lightning setup "
            "rather than by batch size, and this probe amortises that setup over very few "
            "molecules"
        ),
    }


def run_sweep(
    *,
    output: Path,
    arms: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    caps: Mapping[str, Any],
    qualification: Mapping[str, Any],
    scorer: Mapping[str, Any],
    ledger: Any = None,
    posebusters: bool = True,
    resume: bool = True,
    yield_hints: Mapping[str, float] | None = None,
    call_timeout_seconds: float = 2400.0,
    cell_wall_seconds: float = 5400.0,
) -> Record:
    """Run every arm at every seed, generating then scoring, checkpointing each cell.

    Generation for one arm's three seeds runs first, so a single model load
    serves a seed; the arm's molecules are then scored in one pass, because the
    endpoint panel's cost is dominated by setup. Each cell's generation and
    scoring are written as they complete, so a failure in the eighth arm does not
    discard the first seven.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluate_portfolio, evaluator_hash
    from drug_agent.selfevolve.target.baseline_arms import _common_exclusion_set, prepare_track_a

    output.mkdir(parents=True, exist_ok=True)
    unique_cap = int(caps["unique_molecules_per_arm_per_seed"])
    raw_cap = int(caps["raw_generation_attempts_per_arm_per_seed"])

    prepared = prepare_track_a(dict(scorer["preparation"]))
    common = _common_exclusion_set(qualification, prepared)
    derived = common["qualification"]
    _atomic_write(
        output / "common-exclusion.json",
        {k: v for k, v in common.items() if k != "qualification"},
    )

    arm_records: list[Record] = []
    started = time.time()
    for entry in arms:
        capability_id = str(entry["capability_id"])
        arm = str(entry["arm"])
        cells: list[Record] = []
        for seed in seeds:
            cell_dir = _cell_dir(output, arm, int(seed))
            checkpoint = cell_dir / "generation.json"
            record = _load_checkpoint(checkpoint) if resume else None
            if record is None:
                print(
                    f"[{time.strftime('%H:%M:%S')}] generate {arm} seed={seed} "
                    f"cap={unique_cap} ceiling={raw_cap}",
                    flush=True,
                )
                record = generate_cell(
                    capability_id,
                    int(seed),
                    unique_cap=unique_cap,
                    raw_cap=raw_cap,
                    yield_hint=(yield_hints or {}).get(capability_id),
                    call_timeout_seconds=call_timeout_seconds,
                    cell_wall_seconds=cell_wall_seconds,
                    workdir=cell_dir,
                )
                _atomic_write(checkpoint, record)
                _charge(ledger, record, owner=f"GENERATE/{arm}/{seed}", phase="ADAPTATION")
            else:
                record = {**record, "provenance": "RESUMED_FROM_CHECKPOINT"}
            print(
                f"[{time.strftime('%H:%M:%S')}]   {arm} seed={seed} -> {record['status']} "
                f"{record.get('n_unique_returned', 0)}/{unique_cap} unique, "
                f"{record['funnel'].get('n_raw', 0)} raw, "
                f"gpu={record['cost'].get('gpu_seconds')} ({record['cost'].get('gpu_telemetry')})",
                flush=True,
            )
            cells.append(record)

        arm_records.append(
            _score_arm(
                entry=entry,
                cells=cells,
                output=output,
                derived=derived,
                scorer=scorer["scorer"],
                ledger=ledger,
                posebusters=posebusters,
                resume=resume,
                evaluate_portfolio=evaluate_portfolio,
            )
        )

    return {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "stage_d_generation_sweep",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": PROTOCOL_HASH,
        "qualification_hash": str(qualification.get("qualification_hash")),
        "derived_qualification_hash": str(derived.get("qualification_hash")),
        "evaluator_hash": evaluator_hash(),
        "common_exclusion": {k: v for k, v in common.items() if k != "qualification"},
        "caps": dict(caps),
        "seeds": [int(s) for s in seeds],
        "arms": arm_records,
        "wall_seconds": round(time.time() - started, 1),
    }


def _charge(ledger: Any, record: Mapping[str, Any], *, owner: str, phase: str) -> None:
    """Record one cell's measured consumption against the ledger.

    ``record_cost`` rather than a reservation, because the sweep's reservation
    was taken whole before anything ran; these rows are the measured actuals
    beside it. A GPU figure whose telemetry was unavailable is recorded as
    ``None`` with ``telemetry="UNAVAILABLE"``, never as zero.
    """
    if ledger is None:
        return
    cost = dict(record.get("cost") or {})
    with contextlib.suppress(Exception):
        ledger.record_cost(
            "wall_seconds",
            float(cost.get("wall_seconds") or 0.0),
            phase=phase,
            owner=owner,
            kind="OBSERVATION",
            note=str(record.get("status") or ""),
        )
        measured = cost.get("gpu_telemetry") == "MEASURED"
        ledger.record_cost(
            "gpu_seconds",
            float(cost.get("gpu_seconds") or 0.0) if measured else None,
            telemetry="MEASURED" if measured else "UNAVAILABLE",
            phase=phase,
            owner=owner,
            kind="OBSERVATION",
            note=str(record.get("status") or ""),
        )


def _score_arm(
    *,
    entry: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    output: Path,
    derived: Mapping[str, Any],
    scorer: Mapping[str, Any],
    ledger: Any,
    posebusters: bool,
    resume: bool,
    evaluate_portfolio: Any,
) -> Record:
    """Score one arm's cells and evaluate them through the frozen rule."""
    arm = str(entry["arm"])
    arm_dir = output / "cells" / arm
    scored_path = arm_dir / "scoring.json"
    scored = _load_checkpoint(scored_path) if resume else None

    submitted: list[tuple[str, str]] = []
    owner_of: dict[str, int] = {}
    for cell in cells:
        for molecule in cell.get("molecules") or ():
            submitted.append((str(molecule["molecule_id"]), str(molecule["smiles"])))
            owner_of[str(molecule["molecule_id"])] = int(cell["seed"])

    if scored is None:
        if submitted:
            print(
                f"[{time.strftime('%H:%M:%S')}] score {arm}: {len(submitted)} molecules "
                f"through the seven-gate pipeline",
                flush=True,
            )
            try:
                scored = score_molecules(
                    submitted, scorer=scorer, workdir=arm_dir / "scoring", posebusters=posebusters
                )
            except Exception as exc:  # one arm's scoring never discards eight others' work
                # A scoring pass that failed is a missing scientific outcome with a
                # consumed cost, and it is recorded as one. It is emphatically NOT a
                # qualifying yield of zero: nothing was judged, so nothing failed a gate.
                print(
                    f"[{time.strftime('%H:%M:%S')}]   {arm}: SCORING_FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
                scored = {
                    "rows": [],
                    "scoring_failed": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "n_molecules_not_scored": len(submitted),
                    "pose_quality": {
                        "status": "NOT_REACHED",
                        "reason": "the scoring pass raised before pose validation",
                        "reports": {},
                    },
                    "cost": _empty_scoring_cost(),
                }
        else:
            scored = {
                "rows": [],
                "pose_quality": {"status": "NO_MOLECULES", "reason": None, "reports": {}},
                "cost": _empty_scoring_cost(),
            }
        scored["contract_version"] = CONTRACT_VERSION
        _atomic_write(scored_path, scored)
        if ledger is not None:
            cost = dict(scored["cost"])
            _charge(
                ledger,
                {
                    "cost": {
                        "wall_seconds": cost.get("wall_seconds"),
                        "gpu_seconds": cost.get("endpoint_gpu_seconds"),
                        "gpu_telemetry": cost.get("endpoint_gpu_telemetry"),
                    },
                    "status": "SCORED",
                },
                owner=f"SCORE/{arm}",
                phase="EVALUATION",
            )
            with contextlib.suppress(Exception):
                ledger.record_cost(
                    "dock_calls",
                    float(cost.get("n_molecules") or 0),
                    phase="EVALUATION",
                    owner=f"SCORE/{arm}",
                    kind="OBSERVATION",
                )
    else:
        scored = {**scored, "provenance": "RESUMED_FROM_CHECKPOINT"}

    rows = list(scored.get("rows") or ())
    by_id = {str(r["molecule_id"]): r for r in rows}

    cell_results: list[Record] = []
    for cell in cells:
        cell_rows = [
            by_id[str(m["molecule_id"])]
            for m in (cell.get("molecules") or ())
            if str(m["molecule_id"]) in by_id
        ]
        evaluated = (
            evaluate_portfolio(cell_rows, qualification=derived) if cell_rows else _empty_evaluation(derived)
        )
        cell_results.append(
            {
                "seed": int(cell["seed"]),
                "status": cell["status"],
                "reason": cell.get("reason"),
                "n_unique_returned": int(cell.get("n_unique_returned") or 0),
                "funnel": dict(cell["funnel"]),
                "generation_cost": dict(cell["cost"]),
                "qualifying_scaffolds": int(evaluated["qualifying_scaffolds"]),
                "n_qualifying_compounds": int(evaluated["n_qualifying_compounds"]),
                "n_evaluated": int(evaluated["n_evaluated"]),
                "qualification_funnel": dict(evaluated["funnel"]),
            }
        )
        _atomic_write(
            _cell_dir(output, arm, int(cell["seed"])) / "qualification.json",
            {"contract_version": CONTRACT_VERSION, **evaluated},
        )

    # The arm-level table applies the counting rule across the three cells: each
    # evaluated molecule at most once. A structure redrawn at a second seed is a
    # real observation about the sampler and is counted as a duplicate rather
    # than as a second molecule.
    seen: set[str] = set()
    arm_rows: list[Mapping[str, Any]] = []
    cross_seed_duplicates = 0
    for row in rows:
        canonical = str(row.get("smiles") or "")
        if canonical in seen:
            cross_seed_duplicates += 1
            continue
        seen.add(canonical)
        arm_rows.append(row)
    arm_evaluation = (
        evaluate_portfolio(arm_rows, qualification=derived) if arm_rows else _empty_evaluation(derived)
    )
    for row in rows:
        row["qualifies"] = False
    qualifying_ids = {str(r["molecule_id"]) for r in (arm_evaluation.get("rows") or ()) if r.get("qualifies")}
    for row in rows:
        if str(row["molecule_id"]) in qualifying_ids:
            row["qualifies"] = True

    n_submitted = sum(int(c.get("n_unique_returned") or 0) for c in cells)
    scaffolds = int(arm_evaluation["qualifying_scaffolds"])
    record = {
        "capability_id": str(entry["capability_id"]),
        "arm": arm,
        "conditioning": str(entry["conditioning"]),
        "synthesis_aware": bool(entry["synthesis_aware"]),
        "cells": cell_results,
        "n_submitted": n_submitted,
        "n_evaluated": int(arm_evaluation["n_evaluated"]),
        "n_cross_seed_duplicate_molecules": cross_seed_duplicates,
        "qualifying_scaffolds": scaffolds,
        "n_qualifying_compounds": int(arm_evaluation["n_qualifying_compounds"]),
        "scaffolds_per_molecule_submitted": (round(scaffolds / n_submitted, 6) if n_submitted else None),
        "qualification_funnel": dict(arm_evaluation["funnel"]),
        "applied_endpoint_gates": list(arm_evaluation.get("applied_endpoint_gates") or ()),
        "unapplied_endpoint_gates": list(arm_evaluation.get("unapplied_endpoint_gates") or ()),
        "distributions": distributions(rows),
        "novelty": novelty_and_similarity(
            rows,
            list((derived.get("novelty") or {}).get("exclusion_smiles") or ()),
            n_reference_withheld=len(
                ((derived.get("novelty") or {}).get("exclusion_digests") or {}).get("digests") or ()
            ),
        ),
        "pose_quality": {k: v for k, v in (scored.get("pose_quality") or {}).items() if k != "reports"},
        "generation_cost": _sum_costs([dict(c["cost"]) for c in cells]),
        "scoring_cost": dict(scored.get("cost") or {}),
        "scoring_failed": bool(scored.get("scoring_failed")),
        "scoring_failure_reason": scored.get("reason") if scored.get("scoring_failed") else None,
        "n_molecules_not_scored": int(scored.get("n_molecules_not_scored") or 0),
        "backend_outcomes": [
            {
                "seed": int(c["seed"]),
                "status": str(c["status"]),
                "reason": c.get("reason"),
                "consumed_wall_seconds": (c.get("cost") or {}).get("wall_seconds"),
                "consumed_gpu_seconds": (c.get("cost") or {}).get("gpu_seconds"),
                "gpu_telemetry": (c.get("cost") or {}).get("gpu_telemetry"),
            }
            for c in cells
            if str(c["status"]) != "COMPLETE"
        ],
    }
    _atomic_write(arm_dir / "arm.json", {"contract_version": CONTRACT_VERSION, **record})
    print(
        f"[{time.strftime('%H:%M:%S')}]   {arm}: {scaffolds} qualifying scaffolds, "
        f"{record['n_qualifying_compounds']} qualifying compounds, "
        f"{n_submitted} molecules submitted",
        flush=True,
    )
    return record


def _empty_scoring_cost() -> Record:
    """The cost record of a scoring pass that produced no row.

    Every GPU field is ``None`` with ``UNAVAILABLE`` telemetry rather than
    ``0.0``: nobody looked, and a zero here would read as free.
    """
    return {
        "wall_seconds": 0.0,
        "docking_wall_seconds": 0.0,
        "docking_gpu_seconds": None,
        "docking_gpu_telemetry": "UNAVAILABLE",
        "endpoint_wall_seconds": 0.0,
        "endpoint_gpu_seconds": None,
        "endpoint_gpu_telemetry": "UNAVAILABLE",
        "posebusters_wall_seconds": 0.0,
        "n_molecules": 0,
        "seconds_per_molecule": None,
    }


def _empty_evaluation(qualification: Mapping[str, Any]) -> Record:
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluate_portfolio

    return evaluate_portfolio([], qualification=qualification)


def _sum_costs(costs: Sequence[Mapping[str, Any]]) -> Record:
    """Add up per-cell costs, refusing to add a term nobody measured."""
    measured = all(c.get("gpu_telemetry") == "MEASURED" for c in costs) and bool(costs)
    return {
        "wall_seconds": round(sum(float(c.get("wall_seconds") or 0.0) for c in costs), 3),
        "cpu_user_seconds": round(sum(float(c.get("cpu_user_seconds") or 0.0) for c in costs), 3),
        "gpu_seconds": (
            round(sum(float(c.get("gpu_seconds") or 0.0) for c in costs), 3) if measured else None
        ),
        "gpu_telemetry": "MEASURED" if measured else "UNAVAILABLE",
        "n_cells": len(costs),
        "n_cells_with_unmeasured_gpu": sum(1 for c in costs if c.get("gpu_telemetry") != "MEASURED"),
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def primary_table(
    arms: Sequence[Mapping[str, Any]],
    *,
    baseline_lambda: float,
    bar: Mapping[str, Any] | None = None,
) -> Record:
    """``qualifying_scaffolds`` stratified by conditioning class, Holm-corrected.

    The stratification is not cosmetic and the record says why: three of the nine
    arms never see the target at all, so a table pooled across classes silently
    compares an unconditioned prior against a pocket-conditioned model and calls
    the difference a method effect. A pooled reading is available below, labelled
    ``POOLED_ACROSS_CONDITIONING``, which is the only form ``zak-c9r-2`` permits
    it in.

    Holm runs over the **nine arms**, which is the family ``zak-c9r-2``
    registered, not over the three strata. Correcting within a stratum would be a
    different family from the one registered before the arms ran.
    """
    pvalues = {
        str(a["arm"]): arm_p_value(int(a["qualifying_scaffolds"]), baseline_lambda=baseline_lambda)
        for a in arms
    }
    correction = holm(pvalues)
    by_arm = {str(r["name"]): r for r in correction["rows"]}

    strata: dict[str, list[Record]] = {}
    for a in arms:
        row = {
            "arm": str(a["arm"]),
            "capability_id": str(a["capability_id"]),
            "synthesis_aware": bool(a["synthesis_aware"]),
            "qualifying_scaffolds": int(a["qualifying_scaffolds"]),
            "n_qualifying_compounds": int(a["n_qualifying_compounds"]),
            "n_submitted": int(a["n_submitted"]),
            "n_evaluated": int(a["n_evaluated"]),
            "per_seed_qualifying_scaffolds": [int(c["qualifying_scaffolds"]) for c in (a.get("cells") or ())],
            "meets_absolute_threshold": int(a["qualifying_scaffolds"]) >= ABSOLUTE_THRESHOLD_SCAFFOLDS,
            **{k: v for k, v in by_arm[str(a["arm"])].items() if k != "name"},
        }
        strata.setdefault(str(a["conditioning"]), []).append(row)

    return {
        "metric": "qualifying_scaffolds",
        "status": "PRIMARY",
        "stratified_by": "conditioning_class",
        "why_stratified": (
            "three of the nine admitted arms never see the target, so a pooled table compares an "
            "unconditioned prior against a pocket-conditioned model and calls the difference a "
            "method effect"
        ),
        "absolute_threshold_scaffolds": ABSOLUTE_THRESHOLD_SCAFFOLDS,
        "absolute_threshold_source": "margins.practical_gain.generation, frozen at step 14",
        "null_hypothesis": (
            "the arm qualifies scaffolds at the measured baseline rate; the test is the upper tail "
            "of a Poisson with the per-arm mean that rate implies at the locked budget"
        ),
        "baseline_lambda_per_arm": baseline_lambda,
        "comparator_bar": (dict(bar) if bar is not None else None),
        "comparator_bar_note": (
            "the bar in force for this table. A `name` of null with status EXPLICIT_BASELINE_LAMBDA "
            "means the caller passed its own per-arm mean and no registered bar is claimed - the "
            "same bar is then used for the declared secondary, so the two can never disagree. The "
            "retired Stage D null is not the default: see generation_sweep.COMPARATOR_BARS"
        ),
        "strata": {name: sorted(rows, key=lambda r: r["arm"]) for name, rows in sorted(strata.items())},
        "holm": correction,
        "n_arms_rejecting_the_null_under_holm": correction["n_rejected"],
        "known_active_pass_rate": f"{KNOWN_ACTIVE_PASS[0]} of {KNOWN_ACTIVE_PASS[1]}",
        "known_active_pass_rate_note": (
            "the REAL known-active development controls - compounds a human being measured as "
            "active against this target - clear this same seven-gate set at 1 in 6. A qualifying "
            "yield of zero is therefore the EXPECTED outcome at this bar: not a surprise, not a "
            "broken arm, not an unavailable capability and not an affinity of zero"
        ),
    }


def rate_secondary(
    arms: Sequence[Mapping[str, Any]],
    *,
    bar: Mapping[str, Any] | None = None,
    caps: Mapping[str, Any] | None = None,
    baseline_lambda: float | None = None,
) -> Record:
    """Scaffolds per molecule submitted — the declared scale-invariant secondary.

    The operator declared this secondary because the absolute threshold of 3 was
    derived at step 14 against a **24-molecule** probe and this sweep submits
    **150** per arm. The rate is what makes the two commensurable. Neither
    registered threshold moves: the primary is still the count and the margin is
    still 3.

    The Holm-equivalent count is **derived from the bar in force**, not typed.
    At the retired Stage D null it is 6 qualifying scaffolds in 150; at the bar
    step 15E registered it is 7. A secondary that quoted 6 beside a primary
    scored at the superseding bar would be reporting two different bars in one
    table.

    ``baseline_lambda`` is the per-arm mean the PRIMARY was scored at. It is
    passed whenever the caller holds its own mean, so that this table cannot be
    derived from a different bar than the primary it sits beside — which is
    exactly what happened when the two were resolved independently.
    """
    resolved = dict(bar) if bar is not None else comparator_bar()
    caps = dict(caps or {})
    per_arm = int(
        float(caps.get("unique_molecules_per_arm_per_seed", 50)) * float(caps.get("seeds", 3))
    )
    lam = float(baseline_lambda) if baseline_lambda is not None else comparator_lambda(resolved, caps)
    rate = resolved.get("rate_per_molecule")
    baseline_rate = float(rate) if rate is not None else (lam / per_arm if per_arm else 0.0)
    if baseline_rate <= 0.0:
        raise ValueError(
            "a comparator rate of zero makes every observed count significant; a null with no "
            "rate is not a bar and this table refuses to quote one"
        )
    needed = holm_equivalent_scaffolds(lam)
    holm_rate = round(needed / per_arm, 6)
    known_active = round(KNOWN_ACTIVE_PASS[0] / KNOWN_ACTIVE_PASS[1], 6)
    rows = [
        {
            "arm": str(a["arm"]),
            "conditioning": str(a["conditioning"]),
            "qualifying_scaffolds": int(a["qualifying_scaffolds"]),
            "n_submitted": int(a["n_submitted"]),
            "scaffolds_per_molecule_submitted": a.get("scaffolds_per_molecule_submitted"),
            "compounds_per_molecule_submitted": (
                round(int(a["n_qualifying_compounds"]) / int(a["n_submitted"]), 6)
                if int(a["n_submitted"])
                else None
            ),
        }
        for a in arms
    ]
    return {
        "metric": "scaffolds_per_molecule_submitted",
        "status": "DECLARED_SECONDARY",
        "why": (
            "the absolute threshold of 3 was derived against a 24-molecule probe and this sweep "
            "submits 150 per arm; the rate is what makes the two commensurable. Operator decision. "
            "Neither registered threshold moves"
        ),
        "baseline_rate_per_molecule": baseline_rate,
        "baseline_lambda_per_arm": lam,
        "comparator_bar": dict(resolved),
        "known_active_rate_per_molecule": known_active,
        "holm_equivalent_scaffolds": needed,
        "holm_equivalent_rate_per_molecule": holm_rate,
        "holm_equivalent_rate_note": (
            f"{needed} qualifying scaffolds in {per_arm} molecules is what an arm needs to survive "
            f"Holm across the nine arms, which is a per-molecule rate of {holm_rate * 100:.1f}% - "
            f"{holm_rate / baseline_rate:.2f}x the measured baseline and "
            f"{holm_rate / known_active * 100:.0f}% of the rate at which real known ZAK actives "
            "clear the same bar"
        ),
        "rows": sorted(rows, key=lambda r: (r["conditioning"], r["arm"])),
    }


def pooled_secondary(arms: Sequence[Mapping[str, Any]]) -> Record:
    """The table pooled across conditioning classes, labelled as ``zak-c9r-2`` requires."""
    return {
        "label": "POOLED_ACROSS_CONDITIONING",
        "status": "SECONDARY",
        "warning": (
            "this pools an unconditioned prior with a pocket-conditioned model. zak-c9r-2 permits "
            "it only as a secondary and only under this label"
        ),
        "rows": sorted(
            (
                {
                    "arm": str(a["arm"]),
                    "conditioning": str(a["conditioning"]),
                    "qualifying_scaffolds": int(a["qualifying_scaffolds"]),
                    "n_qualifying_compounds": int(a["n_qualifying_compounds"]),
                    "n_submitted": int(a["n_submitted"]),
                }
                for a in arms
            ),
            key=lambda r: (-r["qualifying_scaffolds"], r["arm"]),
        ),
    }


def funnel_table(arms: Sequence[Mapping[str, Any]]) -> list[Record]:
    """One row per arm per seed, with the generation funnel and its arithmetic."""
    rows: list[Record] = []
    for arm in arms:
        for cell in arm.get("cells") or ():
            funnel = dict(cell["funnel"])
            cost = dict(cell["generation_cost"])
            rows.append(
                {
                    "arm": str(arm["arm"]),
                    "conditioning": str(arm["conditioning"]),
                    "seed": int(cell["seed"]),
                    "status": str(cell["status"]),
                    "reason": cell.get("reason"),
                    "n_raw_attempts": int(funnel.get("n_raw") or 0),
                    "n_invalid": int(funnel.get("n_invalid") or 0),
                    "n_valid": int(funnel.get("n_valid") or 0),
                    "n_duplicate": int(funnel.get("n_duplicate") or 0),
                    "n_distinct": int(funnel.get("n_distinct") or 0),
                    "n_screened_out_or_beyond_request": int(
                        funnel.get("n_screened_out_or_beyond_request") or 0
                    ),
                    "n_duplicate_across_calls": int(funnel.get("n_duplicate_across_calls") or 0),
                    "n_beyond_cap": int(funnel.get("n_beyond_cap") or 0),
                    "n_unique_returned": int(funnel.get("n_returned") or 0),
                    "n_calls": int(funnel.get("n_calls") or 0),
                    "arithmetic_closes": bool(funnel.get("arithmetic_closes")),
                    "screened_out_reason": (
                        f"outside the frozen {PROBE_MIN_HEAVY_ATOMS}-{PROBE_MAX_HEAVY_ATOMS} "
                        f"heavy-atom window, or one more distinct molecule than that call asked "
                        f"for; the base adapter reports both through one counter"
                    ),
                    "arithmetic": funnel.get("arithmetic"),
                    "wall_seconds": cost.get("wall_seconds"),
                    "gpu_seconds": cost.get("gpu_seconds"),
                    "gpu_telemetry": cost.get("gpu_telemetry"),
                    "gpu_semantics": GPU_OCCUPANCY_SEMANTICS,
                    "qualifying_scaffolds": int(cell["qualifying_scaffolds"]),
                    "n_qualifying_compounds": int(cell["n_qualifying_compounds"]),
                }
            )
    return rows


def backend_outcomes(arms: Sequence[Mapping[str, Any]]) -> Record:
    """What the backends did, with the two kinds of "did not produce" kept apart.

    An empty list here could mean two opposite things — no backend failed, or
    nobody looked — and that ambiguity is exactly what this programme keeps
    refusing. So the field says what it records, says how many it recorded, and
    carries beside it the two arms that never entered the sweep, each with the
    token step 15A's gate review assigned it. The findings are read from the
    frozen qualification record rather than restated here, so they cannot drift.

    ``during_this_sweep``
        cells whose generation status was not ``COMPLETE``. A cell that completed
        and qualified nothing is **not** here: a qualifying yield of zero is a
        measured fixed-budget result, not a backend failure.
    ``not_admitted_before_this_sweep``
        ``pocket2mol`` **EXECUTED_NO_YIELD** — it ran, it cost GPU-seconds, and
        every structure it returned was below the frozen heavy-atom floor — and
        ``molpilot`` **UNAVAILABLE** — it raised before drawing a sample. One is a
        measured zero from a working backend; the other is an absent
        measurement. They are different facts and they keep different tokens.
    """
    during = [{"arm": str(a["arm"]), **dict(row)} for a in arms for row in (a.get("backend_outcomes") or ())]
    not_admitted = dict(
        (_json(CONFIG_DIR / "qualification.json").get("conditioning_stratification") or {}).get(
            "not_admitted"
        )
        or {}
    )
    return {
        "field_semantics": (
            "during_this_sweep records ONLY backends that failed while this sweep ran - a cell "
            "whose generation status was not COMPLETE. A completed cell that qualified nothing is "
            "a measured fixed-budget result and is deliberately NOT recorded here"
        ),
        "during_this_sweep": during,
        "n_during_this_sweep": len(during),
        "every_cell_completed": not during,
        "n_cells": sum(len(list(a.get("cells") or ())) for a in arms),
        "not_admitted_before_this_sweep": [
            {
                "capability_id": capability_id,
                "status": str(finding).split(" ", 1)[0].strip(" —-"),
                "finding": str(finding),
                "source": (
                    "configs/research/zak-c9r/qualification.json, conditioning_stratification.not_admitted"
                ),
            }
            for capability_id, finding in sorted(not_admitted.items())
        ],
        "why_the_two_are_kept_apart": (
            "EXECUTED_NO_YIELD is a measured zero from a backend that ran and consumed cost; "
            "UNAVAILABLE is an absent measurement. Collapsing either into the other would let a "
            "working backend read as missing, or a missing one read as a result"
        ),
    }


def build_report(
    sweep: Mapping[str, Any],
    *,
    estimate: Mapping[str, Any] | None = None,
    reservation: Mapping[str, Any] | None = None,
    ledger_report: Mapping[str, Any] | None = None,
    comparator: str = REGISTERED_COMPARATOR,
    baseline_lambda: float | None = None,
) -> Record:
    """Everything the operator asked for, in the shape they asked for it.

    ``comparator`` names the registered bar the arms are scored against and
    defaults to the bar step 15E **registered** — not the one it retired. A
    caller that wants the retired Stage D null must ask for it by name
    (``comparator=STAGE_D_REGISTERED_NULL``), which is what
    ``run_target_study.py sweep-report`` does, because that command exists to
    rebuild a result that was measured against it.

    ``baseline_lambda`` still overrides everything, for callers that hold their
    own mean; when it is used the report claims **no registration**, and — this
    is the part the gate review found missing — it overrides the bar for the
    WHOLE report rather than for the primary alone. It used to reach only
    ``primary_table``, while ``rate_secondary`` went on resolving the registered
    bar independently, so an explicit lambda produced a primary scored at one bar
    beside a declared secondary quoting another: the exact defect the
    registration round claimed to have made impossible. There is now one
    ``bar_in_force`` and one ``lam``, both tables are built from them, and the
    two are reconciled below before the report is returned.
    """
    arms = list(sweep.get("arms") or ())
    caps = dict(sweep.get("caps") or {})
    bar = comparator_bar(comparator)
    named = baseline_lambda is None
    if not named and float(baseline_lambda or 0.0) <= 0.0:
        raise ValueError(
            f"baseline_lambda {baseline_lambda!r} is not a positive per-arm mean. A null of zero "
            f"or less makes every observed count significant, which is a widening of every "
            f"threshold in this report at once"
        )
    lam = float(baseline_lambda) if baseline_lambda is not None else comparator_lambda(bar, caps)
    per_arm = float(caps.get("unique_molecules_per_arm_per_seed", 50)) * float(caps.get("seeds", 3))
    #: The one bar the whole report is scored against. For an explicit lambda it
    #: is an UNREGISTERED bar that says so, carrying the per-molecule rate that
    #: lambda implies at these caps, so the declared secondary is derived from
    #: the same null as the primary instead of from a bar nobody asked for.
    bar_in_force: Record = (
        dict(bar)
        if named
        else {
            "name": None,
            "status": "EXPLICIT_BASELINE_LAMBDA",
            "rate_per_molecule": (lam / per_arm if per_arm else None),
            "measured_under_n_gates": None,
            "registered_by": None,
            "source": "the caller's own per-arm mean, passed as baseline_lambda",
            "use_it_for": (
                "nothing registered. No comparator bar is claimed for this report; the registered "
                f"bar is {REGISTERED_COMPARATOR!r} at {SUPERSEDING_COMPARATOR_RATE_PER_MOLECULE} "
                "per molecule. See generation_sweep.COMPARATOR_BARS"
            ),
        }
    )
    measured = measured_totals(arms)
    report: Record = {
        "contract_version": CONTRACT_VERSION,
        "record_kind": "stage_d_generation_sweep_report",
        "classification": EXPLORATORY,
        "classification_reason": (
            "zak-c9r-2 is registered EXPLORATORY. At the locked replication unit - the generation "
            "seed, n = 3 - an exact paired sign test's smallest attainable two-sided p is 0.25, so "
            "the rejection region is EMPTY and the power is 0.0 at every effect size. Nothing "
            "below may be presented as a confirmatory separation between the arms"
        ),
        "protocol_version": sweep.get("protocol_version"),
        "protocol_hash": sweep.get("protocol_hash"),
        "qualification_hash": sweep.get("qualification_hash"),
        "derived_qualification_hash": sweep.get("derived_qualification_hash"),
        "evaluator_hash": sweep.get("evaluator_hash"),
        "caps": caps,
        "seeds": list(sweep.get("seeds") or ()),
        # The SAME object the primary and the declared secondary were built
        # from, so the header cannot name one bar while a table below uses
        # another.
        "comparator_bar": {
            **bar_in_force,
            "baseline_lambda_per_arm": lam,
            "applies_to": ["primary", "declared_secondary_rate"],
            "how_to_choose": (
                "build_report(..., comparator=...) or `--comparator` on the sweep CLI. The "
                "default is the bar step 15E registered; the retired Stage D null must be named"
            ),
            **(
                {}
                if named
                else {
                    "note": (
                        "the caller passed baseline_lambda directly, so no registered comparator "
                        "bar is claimed here - and it is applied to the declared secondary as well "
                        "as to the primary, so the report states ONE bar. The registered bar is "
                        f"{REGISTERED_COMPARATOR!r} at "
                        f"{SUPERSEDING_COMPARATOR_RATE_PER_MOLECULE} per molecule; see "
                        "generation_sweep.COMPARATOR_BARS"
                    )
                }
            ),
        },
        "primary": primary_table(arms, baseline_lambda=lam, bar=bar_in_force),
        "declared_secondary_rate": rate_secondary(
            arms, bar=bar_in_force, caps=caps, baseline_lambda=lam
        ),
        "secondary_pooled": pooled_secondary(arms),
        "raw_qualifying_compounds": [
            {
                "arm": str(a["arm"]),
                "conditioning": str(a["conditioning"]),
                "n_qualifying_compounds": int(a["n_qualifying_compounds"]),
                "qualifying_scaffolds": int(a["qualifying_scaffolds"]),
                "per_seed": [int(c["n_qualifying_compounds"]) for c in (a.get("cells") or ())],
            }
            for a in arms
        ],
        "generation_funnel": funnel_table(arms),
        "unconstrained_distributions": {str(a["arm"]): a["distributions"] for a in arms},
        "novelty_and_nearest_neighbour": {
            str(a["arm"]): {k: v for k, v in a["novelty"].items() if k != "per_molecule"} for a in arms
        },
        "pose_quality": {str(a["arm"]): a["pose_quality"] for a in arms},
        "backend_outcomes": backend_outcomes(arms),
        "cost": {
            "estimate": dict(estimate) if estimate else None,
            "measured": measured,
            "reservation": (
                {**dict(reservation), "gpu_semantics": GPU_OCCUPANCY_SEMANTICS} if reservation else None
            ),
            "ledger": (
                {**dict(ledger_report), "gpu_semantics": GPU_OCCUPANCY_SEMANTICS} if ledger_report else None
            ),
            "gpu_semantics": GPU_OCCUPANCY_SEMANTICS,
            "docking_gpu_occupancy_note": DOCKING_OCCUPANCY_NOTE,
        },
        "limits": _limits(),
    }
    # One bar, reconciled rather than assumed. The primary's mean and the mean
    # the declared secondary derived its Holm-equivalent count from must be the
    # same number, or the report is quoting two bars in one table and is refused
    # instead of published.
    primary_lambda = float(report["primary"]["baseline_lambda_per_arm"])
    secondary_lambda = float(report["declared_secondary_rate"]["baseline_lambda_per_arm"])
    if primary_lambda != secondary_lambda:  # pragma: no cover - the constructor above prevents it
        raise ValueError(
            f"the primary is scored at lambda {primary_lambda} and the declared secondary quotes "
            f"lambda {secondary_lambda}. A report whose primary and secondary are measured "
            f"against different bars states two bars at once and is not published"
        )
    return report


def measured_totals(arms: Sequence[Mapping[str, Any]]) -> Record:
    gen = [dict(a["generation_cost"]) for a in arms]
    score = [dict(a["scoring_cost"]) for a in arms]
    gen_measured = all(g.get("gpu_telemetry") == "MEASURED" for g in gen) and bool(gen)
    score_measured = all(s.get("endpoint_gpu_telemetry") == "MEASURED" for s in score) and bool(score)
    gen_gpu = sum(float(g.get("gpu_seconds") or 0.0) for g in gen)
    score_gpu = sum(float(s.get("endpoint_gpu_seconds") or 0.0) for s in score)
    docking_occupancy = sum(float(s.get("docking_gpu_seconds") or 0.0) for s in score)
    docking_occupancy_measured = all(s.get("docking_gpu_telemetry") == "MEASURED" for s in score) and bool(
        score
    )
    return {
        "generation_wall_seconds": round(sum(float(g.get("wall_seconds") or 0.0) for g in gen), 1),
        "generation_gpu_seconds": round(gen_gpu, 1) if gen_measured else None,
        "generation_gpu_telemetry": "MEASURED" if gen_measured else "UNAVAILABLE",
        "generation_cells_with_unmeasured_gpu": sum(
            int(g.get("n_cells_with_unmeasured_gpu") or 0) for g in gen
        ),
        "scoring_wall_seconds": round(sum(float(s.get("wall_seconds") or 0.0) for s in score), 1),
        "scoring_docking_wall_seconds": round(
            sum(float(s.get("docking_wall_seconds") or 0.0) for s in score), 1
        ),
        "scoring_endpoint_wall_seconds": round(
            sum(float(s.get("endpoint_wall_seconds") or 0.0) for s in score), 1
        ),
        "scoring_posebusters_wall_seconds": round(
            sum(float(s.get("posebusters_wall_seconds") or 0.0) for s in score), 1
        ),
        "scoring_gpu_seconds": round(score_gpu, 1) if score_measured else None,
        "scoring_gpu_telemetry": "MEASURED" if score_measured else "UNAVAILABLE",
        "docking_gpu_occupancy_seconds": (
            round(docking_occupancy, 1) if docking_occupancy_measured else None
        ),
        "docking_gpu_occupancy_telemetry": ("MEASURED" if docking_occupancy_measured else "UNAVAILABLE"),
        "docking_gpu_occupancy_note": DOCKING_OCCUPANCY_NOTE,
        "n_molecules_scored": sum(int(s.get("n_molecules") or 0) for s in score),
        "total_wall_seconds": round(
            sum(float(g.get("wall_seconds") or 0.0) for g in gen)
            + sum(float(s.get("wall_seconds") or 0.0) for s in score),
            1,
        ),
        "total_gpu_seconds": (round(gen_gpu + score_gpu, 1) if (gen_measured and score_measured) else None),
        "total_gpu_telemetry": ("MEASURED" if (gen_measured and score_measured) else "PARTIALLY_UNAVAILABLE"),
        "total_gpu_seconds_note": (
            "generation plus the endpoint panel. The docking phase's occupancy is reported above "
            "and is not added here; see docking_gpu_occupancy_note"
        ),
        "gpu_semantics": GPU_OCCUPANCY_SEMANTICS,
    }


def _limits() -> list[str]:
    """The limits the qualification record already states, carried forward.

    A later report may not contradict these, and the first of them is the one
    plan v1.1 requires verbatim.
    """
    qualification = _json(CONFIG_DIR / "qualification.json")
    stated = list(qualification.get("limits_of_a_passing_molecule") or ())
    return [
        *stated,
        (
            "The correlation sign convention: vina_affinity is LOWER_IS_BETTER and any pActivity "
            "estimate is HIGHER_IS_BETTER, so a PERFECT docking-based potency predictor has "
            "Spearman -1.0 and near zero is the bad case. Vina's measured development rank signal "
            "here is +0.078 - near zero and on the wrong side of it"
        ),
        (
            "This sweep is EXPLORATORY. Three generation seeds cannot support a confirmatory claim "
            "at any effect size, because an exact paired sign test over three replicates cannot "
            "produce a two-sided p below 0.25"
        ),
        (
            "A backend failure or a shortfall against the common cap is a missing scientific "
            "outcome with a consumed cost. It is never a fixed-budget qualifying yield of zero"
        ),
    ]


__all__ = [
    "ABSOLUTE_THRESHOLD_SCAFFOLDS",
    "BASELINE_RATE_PER_MOLECULE",
    "COMPARATOR_BARS",
    "COMPARATOR_FAMILY_SIZE",
    "REGISTERED_COMPARATOR",
    "STAGE_D_REGISTERED_NULL",
    "SUPERSEDING_COMPARATOR_RATE_PER_MOLECULE",
    "comparator_bar",
    "comparator_lambda",
    "holm_equivalent_scaffolds",
    "DISTRIBUTION_ENDPOINTS",
    "FAMILY_WISE_ALPHA",
    "DOCKING_OCCUPANCY_NOTE",
    "GPU_OCCUPANCY_SEMANTICS",
    "KNOWN_ACTIVE_PASS",
    "MEASURED_DOCK_SECONDS_PER_MOLECULE",
    "PROTOCOL_HASH",
    "PROTOCOL_VERSION",
    "QUALIFICATION_HASH",
    "DISTRIBUTION_NOTES",
    "POSEBUSTERS_CHECK_ENTITY",
    "POSEBUSTERS_INTRAMOLECULAR_CHECKS",
    "POSEBUSTERS_INTRAMOLECULAR_MODE_NOTE",
    "POSEBUSTERS_INVOCATION_NOTE",
    "POSEBUSTERS_MIXED_TOTAL_NOTE",
    "POSEBUSTERS_NON_TEST_COLUMNS",
    "arm_p_value",
    "backend_outcomes",
    "build_report",
    "classify_pose_columns",
    "continuation_seed",
    "declare_caps",
    "distributions",
    "estimate_sweep",
    "funnel_table",
    "generate_cell",
    "holm",
    "load_lock",
    "measured_docking_rate",
    "measured_endpoint_rate",
    "measured_totals",
    "load_qualification",
    "measure_endpoint_rate",
    "novelty_and_similarity",
    "pooled_secondary",
    "primary_table",
    "rate_secondary",
    "receptor_entity_census",
    "recompute_pose_quality",
    "refuse_or_reserve",
    "run_sweep",
    "score_molecules",
    "sha256_file",
    "stage_d_caps",
    "sweep_arms",
    "sweep_seeds",
    "verify_test_access_count",
]
