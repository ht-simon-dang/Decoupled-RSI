"""Refuse a qualification record this checkout cannot actually enforce.

WHY THIS EXISTS, AND IT IS NOT HYPOTHETICAL
-------------------------------------------
``evaluate_portfolio`` lives in the guarded evaluator and is pinned by content address; a campaign's
``qualification.json`` is read off disk by path. The two therefore version **independently**, and an
audit of the 2026-09-30 novelty re-freeze executed the consequence: a candidate workspace is created
as ``git worktree add --detach <HEAD>``, so its ``src/`` is frozen at the commit that was HEAD when
the workspace was made, while ``worker.py`` exports ``DRUGAGENT_REPO_ROOT`` pointing at the PARENT
checkout — so ``baseline_arms._repo_root()`` reads the parent's **current** record. Pair a
pre-re-freeze evaluator with a post-re-freeze record and::

    excluded = set(qualification["novelty"]["exclusion_smiles"] or [])   # -> empty
    novel = canonical not in excluded                                     # -> always True

The novelty gate becomes a silent no-op. Measured on a recorded 150-molecule submission: 0 novelty
failures and 72 qualifying scaffolds under the stale pairing, against 33 and 60 under the matched
one. Nothing raised. That is the worst failure a novelty gate can have, because a silently empty
exclusion set is indistinguishable from a correctly empty one — and re-submitting already-accepted
molecules is a guaranteed perfect score, which an arm has already once collected.

The guarded evaluator now fails closed on every shape it can SEE. It cannot see a mismatch between
itself and the record, because it has no way to know what the record was written for. That check
belongs to the caller, and this module is it.

WHAT IT REFUSES, AND WHY EACH ONE
---------------------------------
* a record whose ``qualification_hash`` is not the address this checkout was built against — the
  stale-pairing case above, caught at the only place it is visible;
* a novelty block that declares **no** ``n_excluded`` — the evaluator's own consistency check is
  conditional on that field being present, so a record without it buys silence rather than safety.
  A record that genuinely enforces nothing says so with ``n_excluded: 0``;
* ``exclusion_digests.digests`` given as a **string** rather than a list. A string is a perfectly
  good iterable of one-character garbage, so ``{str(d) for d in digests}`` yields 64 entries that
  match nothing. With ``n`` absent as well, nothing in the evaluator notices;
* a digest that is not 64 lowercase hex characters. ``novelty_digest`` returns ``hexdigest()``, which
  is lowercase, and membership is tested on that exact string: an uppercase set matches nothing while
  the declared count still agrees, so no count-based check can catch it.

This module is deliberately OUTSIDE the kernel integrity manifest. Putting these checks in
``evaluate_portfolio`` would be better placement and would move the evaluator address that 139
campaign records declare, for guarantees whose reachable surface is the non-guarded loaders anyway —
:func:`worker.independent_development_evaluation` runs in the dispatcher's own process against the
parent's ``src``, so the guarded path always sees a matched pair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "NoveltyGuardError",
    "require_enforceable_novelty",
    "require_qualification_address",
]

_HEX = frozenset("0123456789abcdef")


class NoveltyGuardError(RuntimeError):
    """The qualification record and this checkout's evaluator do not match, or the record is unenforceable."""


def require_qualification_address(qualification: Mapping[str, Any], expected: str, *, what: str) -> None:
    """Raise unless *qualification* is the record this checkout was built against.

    A count measured against a record the code does not understand is not a weaker measurement, it is
    a different one wearing the same name.
    """
    declared = str(qualification.get("qualification_hash") or "")
    if declared != str(expected):
        raise NoveltyGuardError(
            f"{what}: the qualification record on disk is {declared or 'UNSTAMPED'}, but this "
            f"checkout was built against {expected}. Refusing to score. This is the stale-pairing "
            f"case: a candidate workspace pins src/ at the commit it was created from while "
            f"DRUGAGENT_REPO_ROOT points at the parent's current records, and an evaluator that "
            f"predates a novelty re-freeze reads the newer record's exclusion set as EMPTY — every "
            f"molecule then passes novelty silently. Recreate the workspace at the current commit, "
            f"or evaluate against the record this checkout matches"
        )


def require_enforceable_novelty(qualification: Mapping[str, Any], *, what: str) -> None:
    """Raise unless the record's novelty block is one the evaluator can actually apply."""
    novelty = qualification.get("novelty")
    if not isinstance(novelty, Mapping):
        raise NoveltyGuardError(
            f"{what}: the qualification record carries no novelty block, so the novelty gate would "
            f"admit every molecule without saying so"
        )

    plaintext = novelty.get("exclusion_smiles")
    if plaintext is not None and (isinstance(plaintext, (str, bytes)) or not isinstance(plaintext, Sequence)):
        raise NoveltyGuardError(
            f"{what}: novelty.exclusion_smiles must be a list, not {type(plaintext).__name__}. A "
            f"string is an iterable of single characters, which matches nothing"
        )
    n_plaintext = len(plaintext or ())

    n_digests = 0
    digest_rule = novelty.get("exclusion_digests")
    if digest_rule is not None:
        if not isinstance(digest_rule, Mapping):
            raise NoveltyGuardError(
                f"{what}: novelty.exclusion_digests must be a record carrying 'algorithm' and "
                f"'digests', not {type(digest_rule).__name__}"
            )
        digests = digest_rule.get("digests")
        if isinstance(digests, (str, bytes)) or not isinstance(digests, Sequence):
            raise NoveltyGuardError(
                f"{what}: novelty.exclusion_digests.digests must be a list, not "
                f"{type(digests).__name__}. A string iterates into one-character entries that match "
                f"nothing, and with 'n' absent the evaluator cannot tell"
            )
        malformed = [
            d
            for d in digests
            if not isinstance(d, str) or len(d) != 64 or not _HEX.issuperset(d)
        ]
        if malformed:
            raise NoveltyGuardError(
                f"{what}: {len(malformed)} of {len(digests)} exclusion digests are not 64 lowercase "
                f"hex characters, e.g. {malformed[:3]}. Membership is tested on hexdigest() output, "
                f"which is lowercase, so any other spelling matches nothing while the declared count "
                f"still agrees"
            )
        n_digests = len(set(digests))
        declared_n = digest_rule.get("n")
        if declared_n is not None and int(declared_n) != n_digests:
            raise NoveltyGuardError(
                f"{what}: novelty.exclusion_digests declares n={int(declared_n)} but carries "
                f"{n_digests} distinct digests"
            )

    declared_total = novelty.get("n_excluded")
    if declared_total is None:
        raise NoveltyGuardError(
            f"{what}: novelty declares no n_excluded. The evaluator's consistency check is "
            f"conditional on that field, so a record without it buys silence rather than safety. A "
            f"record that really excludes nothing states n_excluded: 0"
        )
    if int(declared_total) != n_plaintext + n_digests:
        raise NoveltyGuardError(
            f"{what}: novelty declares n_excluded={int(declared_total)} but carries {n_plaintext} "
            f"plaintext SMILES plus {n_digests} digests. A declared exclusion set wider than the "
            f"enforced one is how a 611-molecule union came to be enforced as 30"
        )
