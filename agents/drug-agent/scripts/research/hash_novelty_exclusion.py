#!/usr/bin/env python
"""Carry the novelty exclusion set as digests, so reading the gates is not also reading the corpus.

THE LEAK THIS CLOSES, STRUCTURALLY
----------------------------------
Every campaign has to read ``configs/research/zak-c9r/qualification.json``. That is where the bars
it is judged by live -- ``vina <= -8.18``, ``sa_score <= 4.69``, ``descriptors.qed > 0.25`` -- and a
campaign that has not read them cannot aim at them. The same record carried
``novelty.exclusion_smiles``: **7,322 plaintext SMILES**, which is the entire corpus of molecules
this host has already recorded, accepted ones included. So the unavoidable read handed over every
accepted structure, for free, as a side effect of learning the rules.

The operator's standing constraint is that a campaign may see the summarised prior experience and
the NEGATIVES, and must not see the accepted molecules; the negatives exist as preference-learning
signal and nothing else. A record that ships the accepted structures to every reader does not meet
that constraint no matter what the prompt says.

WHAT REPLACES IT, AND WHY THE RULE IS UNCHANGED
-----------------------------------------------
``evaluate_portfolio`` enforced novelty with one membership test::

    novel = standardized_canonical_smiles not in exclusion_smiles

It now also accepts the same set carried as digests of those same standardized strings::

    novel = ... and sha256(standardized_canonical_smiles) not in exclusion_digests

The two are the same test. Measured on this record: all 7,322 SMILES are already standardized (the
round trip through ``_standardized`` changes none of them) and they produce 7,322 distinct digests,
so the mapping is exactly one-to-one and no molecule leaves or joins the set. **Any submission
re-evaluated under the digest record gets the identical count it got under the plaintext record**,
and ``--verify`` proves that on the recorded submissions rather than asserting it.

WHAT THIS DOES NOT CLOSE, SAID PLAINLY
--------------------------------------
1. ``.external/research/zak-c9r/derived/campaigns/*/experiments/*/molecules.json`` -- about 120
   files -- still hold every recorded molecule in plaintext, with its gate results beside it. That is
   where the 7,322 were unioned from in the first place. A campaign runs as the same user and can read
   them, and it does NOT have to go looking: an audit found ``.external`` symlinked into 17 candidate
   workspaces, so from a campaign's own working directory the path is
   ``./.external/research/zak-c9r/derived/campaigns/*/experiments/*/molecules.json`` -- one relative
   listing away, under a directory name the round records already teach campaigns to use. Those links
   are a hand-applied operator workaround for ``baselines._repo_root`` ignoring
   ``DRUGAGENT_REPO_ROOT``; linking only the tool directories a run actually needs
   (``.external/PFM``, ``.external/pockets``) rather than ``.external/research`` would keep the pocket
   working and close the path. What this change achieves is narrower than "the corpus is now hard to
   reach": reading the GATES no longer also reads the corpus. Reaching it is a separate act that an
   audit can see, which is not the same as it being unreachable, and no in-process mechanism can make
   it unreachable while every campaign runs as the same user.
1b. Those same ``molecules.json`` files mix ACCEPTED with REJECTED rows, and the rejected ones are the
   negatives a campaign is supposed to see for preference learning. So the remaining fix is separating
   the two polarities -- or digesting the accepted side -- not deleting the files.
2. The digests are UNSALTED. Anyone may therefore confirm that a claimed plaintext set is the one
   encoded here, which is what makes the record auditable -- and equally, anyone may hash a public
   compound library and recover whichever members happen to be in the set. The corpus is almost
   entirely de-novo generator output rather than catalogue compounds, so that recovers little; a
   salt would raise the cost, and would have to be distributed to every worktree that evaluates
   anything, and an arm that would mount the attack would equally read the salt file. The trade is
   recorded here rather than implied by the absence of a salt.
3. The 581 train-partition compounds are still absent from the enforced set. That gap is the
   parent record's and is carried forward untouched; this change is not the place to widen a rule.

Usage::

    PYTHONPATH=src conda run --no-capture-output -n drug-agent python \\
        scripts/research/hash_novelty_exclusion.py [--dry-run] [--verify]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # pragma: no cover - operator convenience
    sys.path.insert(0, str(REPO / "src"))

QUALIFICATION = REPO / "configs" / "research" / "zak-c9r" / "qualification.json"
CALIBRATION = REPO / "configs" / "research" / "zak-c9r" / "calibration.json"

#: The plaintext parent is archived OUTSIDE the repository, and this script does not write the path
#: into the record. Naming it there would restore exactly the read this change removes: a campaign
#: reading the gates would read where the structures went.
ARCHIVE = Path.home() / ".drug-agent-private" / "zak-c9r"

PARENT_HASH = "sha256:bcc522445d38d9bada5a783e34ff1c70474bf61a544e1c403f836f6ee40bb8de"

OPERATOR_WORDS = (
    "每個campaign應該只能看到過往的總結經驗,和負樣本,不應該直接看到過往收錄的合格分子。"
    "負樣本僅僅用來做DPO學習"
)


def _refuse(message: str) -> int:
    print(f"REFUSED: {message}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-evaluate every recorded submission under both records and refuse on any disagreement",
    )
    args = parser.parse_args()

    from drug_agent.selfevolve.evaluation.target_benchmark import (
        NOVELTY_DIGEST_ALGORITHM,
        calibration_hash,
        evaluator_hash,
        novelty_digests,
        qualification_hash,
    )

    if not QUALIFICATION.is_file():
        return _refuse(f"{QUALIFICATION} does not exist")
    q: dict[str, Any] = json.loads(QUALIFICATION.read_text(encoding="utf-8"))

    declared = str(q.get("qualification_hash") or "")
    if declared != PARENT_HASH:
        return _refuse(
            f"the committed qualification is {declared}, not the {PARENT_HASH} this script was "
            "written against. Re-read the record before converting it"
        )
    recomputed = qualification_hash(q)
    if recomputed != PARENT_HASH:
        return _refuse(
            f"the committed record does not hash to its own declared address: recomputed "
            f"{recomputed}, declared {PARENT_HASH}. It was edited without re-freezing"
        )

    novelty = dict(q.get("novelty") or {})
    plaintext = list(novelty.get("exclusion_smiles") or [])
    if "exclusion_digests" in novelty:
        return _refuse("this record already carries novelty.exclusion_digests; nothing to convert")
    if not plaintext:
        return _refuse("novelty.exclusion_smiles is empty; refusing to convert an empty set")
    if int(novelty.get("n_excluded") or 0) != len(plaintext):
        return _refuse(
            f"novelty declares n_excluded={novelty.get('n_excluded')} but carries "
            f"{len(plaintext)} SMILES; fix the record before converting it"
        )

    digests = novelty_digests(plaintext)
    if len(digests) != len(set(plaintext)):
        return _refuse(
            f"{len(set(plaintext))} distinct SMILES produced {len(digests)} distinct digests. The "
            "mapping must be one-to-one or the digest set is not the same set; refusing"
        )

    live_evaluator = evaluator_hash()
    parent_evaluator = str(q.get("evaluator_hash") or "")

    novelty["exclusion_digests"] = {
        "algorithm": NOVELTY_DIGEST_ALGORITHM,
        "n": len(digests),
        "digests": digests,
        "salted": False,
        "why_unsalted": (
            "an unsalted digest set is verifiable: anyone holding a claimed plaintext set can "
            "recompute these and confirm the record encodes exactly that set. A salt would raise "
            "the cost of hashing a public compound library against this set, would have to be "
            "distributed to every worktree that evaluates anything, and would not stop an arm that "
            "can already read the salt file. The residual exposure is stated rather than implied"
        ),
    }
    novelty["exclusion_smiles"] = []
    novelty["plaintext_withheld"] = (
        f"The {len(digests)} standardized SMILES this set excludes are NOT serialized here. Every "
        "campaign must read this record to learn the bars it is judged by, and a plaintext set made "
        "that unavoidable read a free copy of every molecule this host has already accepted. The "
        "gate is unchanged: evaluate_portfolio hashes each submitted molecule's standardized "
        "canonical SMILES and tests membership in exclusion_digests, which is the same membership "
        "test on the same set. The plaintext is archived outside this repository and its path is "
        "deliberately not named here."
    )
    q["novelty"] = novelty

    q["evaluator_hash"] = live_evaluator
    q["supersedes"] = {
        "parent_qualification_hash": PARENT_HASH,
        "reason": (
            "the novelty exclusion set stopped being serialized as plaintext SMILES, because every "
            "campaign must read this record for its own gates and the plaintext turned that read "
            "into a free copy of the accepted corpus"
        ),
        "operator_instruction": OPERATOR_WORDS,
        "instruction_date": "2026-09-30",
        "what_moved": {
            "novelty.exclusion_smiles": {"from": f"{len(plaintext)} plaintext SMILES", "to": "[]"},
            "novelty.exclusion_digests": {
                "from": None,
                "to": f"{len(digests)} {NOVELTY_DIGEST_ALGORITHM} digests of those same SMILES",
            },
            "evaluator_hash": {"from": parent_evaluator, "to": live_evaluator},
        },
        "what_did_not_move": [
            "the set itself: 1-to-1, all SMILES already standardized, no molecule left or joined",
            "n_excluded, and therefore the number of molecules the gate refuses",
            "every threshold: vina <= -8.18, sa_score <= 4.69, qed > 0.25, herg <= 0.9305925",
            "the geometry checks, identity, the counting rule, the required gate names",
            "the 581-train-partition gap, carried forward as the parent's open defect",
        ],
        "is_comparable": (
            "UNLIKE the 2026-09-25 re-freeze, this one changes no bar. The novelty gate applies the "
            "identical membership test to the identical set, so a count measured under this address "
            "equals the count the same submission would get under the parent. Verified by "
            "re-evaluating the recorded submissions under both records, not asserted."
        ),
        "evaluator_moved_and_why": (
            "evaluate_portfolio had to learn to test membership by digest. A digest set the "
            "evaluator could not compute over would make every molecule look novel, which is "
            "indistinguishable from a correctly empty exclusion set, so the evaluator refuses an "
            "algorithm it does not implement rather than passing everything. That is a change to a "
            "guarded file and it moves the evaluator address; records measured under the parent "
            "address keep declaring the parent address, because that is what measured them."
        ),
        "prior_supersedes": q.get("supersedes"),
    }
    q.pop("qualification_hash", None)
    q["qualification_hash"] = qualification_hash(q)

    print(f"parent qualification   {PARENT_HASH}")
    print(f"new qualification      {q['qualification_hash']}")
    print(f"  novelty              {len(plaintext)} plaintext SMILES -> {len(digests)} digests, 1:1")
    print(f"  n_excluded           {novelty['n_excluded']}, unchanged")
    print(f"  evaluator            {live_evaluator}")
    print("  thresholds           unchanged (no bar moved)")

    # calibration.json declares the same evaluator address, and
    # test_the_frozen_records_still_match_the_evaluator_that_produced_them checks BOTH records
    # against the live kernel. Its suggested remedy -- re-running calibrate_zak_benchmark.py -- must
    # NOT be used here: that script also writes qualification.json, from the 30 reference controls,
    # so it would silently revert both the operator's PFM acceptance standards and this change. The
    # two leaves the precedent says a re-freeze moves are moved directly instead.
    cal: dict[str, Any] = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    cal_parent_hash = str(cal.get("calibration_hash") or "")
    if calibration_hash(cal) != cal_parent_hash:
        return _refuse(
            f"calibration.json does not hash to its own declared address: recomputed "
            f"{calibration_hash(cal)}, declared {cal_parent_hash}"
        )
    before_leaves = _leaves(cal)
    cal["evaluator_hash"] = live_evaluator
    cal.pop("calibration_hash", None)
    cal["calibration_hash"] = calibration_hash(cal)
    moved = sorted(k for k, v in _leaves(cal).items() if before_leaves.get(k) != v)
    if moved != ["calibration_hash", "evaluator_hash"]:
        return _refuse(
            f"re-stamping calibration.json moved {moved}; exactly two leaves may move, the "
            "evaluator address and the record's own hash"
        )
    print(f"parent calibration     {cal_parent_hash}")
    print(f"new calibration        {cal['calibration_hash']}")
    print(f"  leaves moved         {len(moved)} of {len(before_leaves)}: {', '.join(moved)}")

    if args.verify:
        rc = _verify(q, plaintext)
        if rc:
            return rc

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.chmod(0o700)
    archive = ARCHIVE / f"qualification.plaintext-{PARENT_HASH.split(':')[1][:8]}.json"
    shutil.copy2(QUALIFICATION, archive)
    archive.chmod(0o600)
    QUALIFICATION.write_text(json.dumps(q, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CALIBRATION.write_text(json.dumps(cal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\narchived plaintext -> {archive}  (mode 600, outside the repository)")
    print(f"wrote              -> {QUALIFICATION}")
    print(f"wrote              -> {CALIBRATION}")
    print("\nTHE PINS THAT ASSERT THE OLD ADDRESSES MUST MOVE, or every run refuses:")
    print("  src/drug_agent/selfevolve/target/generation_sweep.py   QUALIFICATION_HASH")
    print("  scripts/research/evolution_comparison.py               EXPECTED_QUALIFICATION_HASH")
    print("  scripts/research/evolution_comparison.py               EXPECTED_EVALUATOR_HASH")
    print("  src/drug_agent/selfevolve/target/reporting.py          REGISTERED_EVALUATOR_HASH")
    print("  configs/research/zak-c9r.yaml       qualification_hash, calibration_hash, evaluator_hash")
    return 0


def _leaves(record: Any, path: str = "") -> dict[str, Any]:
    """Every scalar in *record*, keyed by its path, so a re-freeze can be shown to move only two."""
    out: dict[str, Any] = {}
    if isinstance(record, dict):
        for key, value in record.items():
            out.update(_leaves(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(record, list):
        for index, value in enumerate(record):
            out.update(_leaves(value, f"{path}[{index}]"))
    else:
        out[path] = record
    return out


def _verify(new_record: dict[str, Any], plaintext: list[str]) -> int:
    """Re-evaluate every recorded submission under both records and refuse on disagreement."""
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluate_portfolio

    parent = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    roots = sorted((REPO / ".external" / "research" / "zak-c9r" / "derived" / "campaigns").glob("*"))
    checked = 0
    disagreed: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("molecules.json")):
            try:
                molecules = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(molecules, list) or not molecules:
                continue
            try:
                before = evaluate_portfolio(molecules, qualification=parent)
                after = evaluate_portfolio(molecules, qualification=new_record)
            except Exception as exc:  # a submission neither record can score is not evidence
                print(f"  skipped {path.relative_to(REPO)}: {type(exc).__name__}: {exc}")
                continue
            checked += 1
            keys = ("qualifying_scaffolds", "n_qualifying_compounds")
            if any(before.get(k) != after.get(k) for k in keys) or (
                before["funnel"]["n_novelty_failed"] != after["funnel"]["n_novelty_failed"]
            ):
                disagreed.append(
                    f"{path.relative_to(REPO)}: "
                    f"{[before.get(k) for k in keys]}/{before['funnel']['n_novelty_failed']} vs "
                    f"{[after.get(k) for k in keys]}/{after['funnel']['n_novelty_failed']}"
                )
    print(f"\nverify: re-evaluated {checked} recorded submissions under both records")
    if disagreed:
        for line in disagreed[:20]:
            print(f"  DISAGREED  {line}", file=sys.stderr)
        return _refuse(
            f"{len(disagreed)} of {checked} submissions score differently under the digest record. "
            "The conversion is not enforcement-equivalent; nothing was written"
        )
    if not checked:
        return _refuse("verify found no recorded submission to check; refusing to claim equivalence")
    print(f"verify: all {checked} agree on qualifying_scaffolds, n_qualifying_compounds and n_novelty_failed")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
