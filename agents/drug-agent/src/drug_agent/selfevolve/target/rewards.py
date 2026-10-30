"""The preference target for a PFM fine-tune, with connectivity as a precondition.

WHY THERE IS A v2, MEASURED ON THIS HOST
----------------------------------------
``gold_reward_v1`` -- ``R = 0.50*SA_norm + 0.25*vina_n + 0.15*qed + 0.10*herg_n`` -- was
registered as the round's default DPO target. A DPO fine-tune of PFM's flow-matching weights
against it improved every axis it paid for and destroyed the model: SMILES containing ``.``
went from 16/150 = 11% on the frozen parent to 127/150 = 85% on the trained checkpoint, and
qualifying scaffolds fell 3 -> 1 at each of three paired seeds. Of the 128 unscorable
molecules, 126 failed as "meeko could not prepare the ligand" in about zero seconds, 127 were
multi-fragment, and of the 22 that could be docked NONE were. The count collapsed on
VALIDITY, not on chemistry: the survivors' medians improved on everything rewarded --
sa_score 6.27 -> 5.40, vina -7.74 -> -9.83, herg 0.83 -> 0.65 -- while QED, the smallest
weight at 0.15, went the wrong way, 0.33 -> 0.18, straight through the gate's own
``qed > 0.25``.

Fragmenting buys only a small SA gain -- multi-fragment sa_score median 5.378 against
single-fragment 5.448, 0.07 units -- so the dominant cause is loss of the connectivity prior
under the DPO gradient, with a weak reward incentive on top. This module does not claim
reward hacking. What it does claim is measurable and sufficient: NOTHING in the loop
penalised fragmentation. meeko rejects a multi-fragment ligand before any gate sees it, so
fragments never appeared as negatives, and v1 had no connectivity term, so they were never
scored down either. A fragment's sa_score is computed on the whole dotted string, which is
why an unusable molecule could look fine.

THE OPPOSITE FAILURE IS ALSO MEASURED AND MUST NOT BE TRADED FOR
----------------------------------------------------------------
An unlikelihood-trained checkpoint kept connectivity -- 2/150 = 1% multi-fragment -- and
pushed median sa_score to 6.71, WORSE than the frozen parent's 6.27. It scored zero. A
reward that adds a validity term and lets chemistry rot is not a fix, and neither is a
preference set built only out of valid-versus-invalid pairs. :func:`build_preference_pairs`
enforces that structurally rather than trusting it.

WHAT v2 CHANGES, AND WHY EACH CHANGE
------------------------------------
1. ADMISSIBILITY IS A PRECONDITION, NOT A TERM. An inadmissible molecule scores exactly
   ``0.0``; an admissible one scores at least :data:`ADMISSIBLE_FLOOR`. The separation is
   structural, not numeric, so no chemistry value can buy a fragment onto the chosen side.
   The pair builder additionally takes the chosen side only from the admissible pool, so the
   guarantee does not rest on comparing floats. Fragments remain fully usable as REJECTED
   examples -- they are exactly the negative signal that was missing.

2. EVERY CHEMISTRY TERM SATURATES A SHORT WAY INSIDE ITS FROZEN BAR. v1's vina term kept
   paying all the way to -12 kcal/mol, 3.8 units past the gate at -8.18, and that headroom is
   what financed the trade against QED. A term that has stopped paying cannot fund a trade
   against a term that has not. Anchors are named after the bars they come from, below.

3. QED IS RAISED 0.15 -> 0.20 AND GIVEN A RAMP THROUGH ITS OWN GATE. QED was the only
   rewarded axis that got worse, it is a strict property limit (``qed > 0.25``), and at 0.15
   it was the smallest term and lost every trade. Under v1, moving qed 0.18 -> 0.30 was worth
   0.018 of reward; under v2 it is worth 0.060.

4. SA STILL CARRIES THE MOST WEIGHT, BECAUSE IT IS STILL THE BOTTLENECK. Of 294 distinct PFM
   samples the SA gate refuses 270; the frozen parent fails it on about 93% of submissions.
   The operator's instruction stands and 0.50 is unchanged from v1.

5. THE OPTIMUM IS NOT A TRIVIAL CHEMOTYPE. An SA-heavy reward's cheapest win is a small,
   simple molecule, so admissibility carries a heavy-atom floor of :data:`MIN_HEAVY_ATOMS`,
   the same floor this programme already used to record ``pocket2mol`` as EXECUTED_NO_YIELD on
   four structures of 6/6/7/9 heavy atoms. Collapse onto one chemotype is answered at the set
   level, where it actually lives: :func:`score_batch` discounts the second and later members
   of a scaffold and :func:`build_preference_pairs` caps how many pairs one scaffold may
   supply. QED's raised weight covers the other end -- it penalises size and greasiness -- so
   no separate mw term is needed and none is added, because mw is a gate and the reward does
   not duplicate gates.

THE BOUNDARY, STATED RATHER THAN LEFT TO THE READER
---------------------------------------------------
* CONNECTED means exactly one connected component in the molecule RDKit parses from the
  string the record carries. Nothing is desalted, no largest fragment is selected, no charges
  are neutralised first.
* A SINGLE HEAVY FRAGMENT PLUS A SALT OR SOLVENT IS INADMISSIBLE, reported as
  ``multi_fragment_salt_like`` rather than merged into ``multi_fragment`` so a reader can tell
  salts from shrapnel. Four reasons, and the second is the decisive one: meeko refuses it in
  about zero seconds so it can never be docked and can never qualify; the qualification scores
  the string that was submitted, so scoring a desalted version scores a molecule the gate will
  never see; sa_score on the dotted string is not the sa_score of the parent molecule, which
  is the specific way an unusable sample looked fine; and desalting would launder a generation
  failure into a pass, when an 85% fragmentation rate is precisely the failure the reward has
  to be able to see. PFM samples atoms into a pocket -- it does not choose a salt form -- so a
  counterion here is a failed generation and not a formulation.
* A CHARGED SINGLE FRAGMENT IS ADMISSIBLE. Real ligands are charged at physiological pH,
  meeko and vina prepare a single charged fragment without complaint, and no frozen gate
  filters on charge. Net charge is recorded for reporting and never acted on. Radical
  electrons are recorded the same way: reported, not gated, because nothing here has measured
  what they cost.
* A MOLECULE WITH NO CARBON is inadmissible. sascorer and the ADMET-AI panel are not defined
  on an inorganic cluster, so a number computed for one is not a measurement.

CHEAP BEFORE DOCKING, EXACT AFTER
---------------------------------
``basis="exact"`` needs all four measurements. ``basis="cheap"`` scores whatever is present,
renormalised over the weights of the terms it had, and reports ``lower``/``upper`` bounds
computed by giving every missing term 0 and then 1 on the FULL weight set -- so a caller can
discard a molecule whose ``upper`` cannot reach a bar before spending a 300-CPU-second dock.
A cheap value and an exact value are NOT on the same scale and must never be compared. Every
:class:`Reward` carries its ``basis`` and the exact tuple of terms it used, and
:func:`require_comparable` refuses a mixed set rather than averaging one into the other.

WHAT THIS MODULE DOES NOT DO
----------------------------
It measures nothing except connectivity, size and scaffold, all of which are pure RDKit on a
string. sa_score, vina_affinity, qed and herg are READ from the record, because gold means
measured by the frozen scorer and a reward that quietly computed its own proxy would not be
gold. It also never calls the frozen evaluator: the scaffold convention below is a
re-implementation of :func:`drug_agent.selfevolve.evaluation.target_benchmark.scaffold_id`'s
rule, held to it by a test, so that the library a campaign imports cannot drag the evaluator's
address into its own dependencies.

Usage::

    from drug_agent.selfevolve.target.rewards import build_preference_pairs, score_batch

    rewards = score_batch(molecules)              # list[Reward], crowding applied
    pairs = build_preference_pairs(molecules)     # PairSet, chosen side always admissible
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ADMISSIBLE_FLOOR",
    "CROWDING_STRENGTH",
    "MIN_HEAVY_ATOMS",
    "Admissibility",
    "Pair",
    "PairSet",
    "Ramp",
    "Reward",
    "RewardInputError",
    "RewardSpec",
    "admissibility",
    "build_preference_pairs",
    "gold_reward_v1",
    "gold_reward_v2",
    "is_admissible",
    "read_measurements",
    "require_comparable",
    "reward_manifest",
    "reward_spec",
    "scaffold_key",
    "score_batch",
]


class RewardInputError(ValueError):
    """A record cannot be scored, or two scores cannot honestly be compared."""


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
#: An admissible molecule is worth at least this much no matter how bad its chemistry, and an
#: inadmissible one is worth exactly zero. The gap is what makes validity a precondition
#: rather than the largest term in a weighted sum -- a weighted sum is exactly what let 85%
#: fragmentation coexist with an improvement on every paid axis.
ADMISSIBLE_FLOOR = 0.05

#: The most a crowded scaffold's later members may be discounted. Bounded at a quarter so the
#: crowding term can break a tie toward scaffold spread but can never invert a real chemistry
#: difference -- the reward is still a chemistry reward.
CROWDING_STRENGTH = 0.25

#: The programme's own validity floor, not a new bar: ``baselines.PROBE_MIN_HEAVY_ATOMS`` and
#: the generation façade's ``min_heavy_atoms`` default are both 10, and it is the floor under
#: which ``pocket2mol``'s 6/6/7/9-heavy-atom output was recorded as EXECUTED_NO_YIELD. It sits
#: in admissibility because SA rewards simplicity and the simplest molecule is a tiny one.
MIN_HEAVY_ATOMS = 10


@dataclass(frozen=True)
class Ramp:
    """Linear credit in ``[0, 1]``: none at ``zero``, full at ``full``, clipped outside.

    ``zero > full`` gives a decreasing ramp, which is what LOWER_IS_BETTER measurements need.
    Saturation is the point: past ``full`` the term stops paying, so it cannot fund a trade
    against a term that has not yet been satisfied.
    """

    zero: float
    full: float

    def __post_init__(self) -> None:
        if self.zero == self.full:
            raise ValueError("a ramp needs two distinct anchors")

    def __call__(self, value: float) -> float:
        t = (float(value) - self.zero) / (self.full - self.zero)
        return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


#: Every anchor is named after the bar it comes from, so a reader can check it against
#: ``configs/research/zak-c9r/qualification.json`` instead of taking it on trust.
RAMPS_V2: Mapping[str, Ramp] = {
    # 8.00: unsynthesizable under any bar this programme has enforced, so v1's residual credit
    # there paid for moving one hopeless molecule to another. 3.2385: the PARENT
    # qualification's SA bar, the strictest this programme has used -- full credit at the
    # hardest bar, nothing beyond it. The operative bar, 4.69, sits inside the ramp, so SA
    # keeps a gradient across the whole range PFM actually occupies (median 5.40 to 6.71).
    "sa": Ramp(zero=8.00, full=3.2385),
    # -6.00 is v1's own zero anchor, kept so the low end stays legible against v1 records.
    # -9.00 is 0.82 kcal/mol inside the -8.18 gate -- clear the bar with margin, then stop --
    # and it is chosen so that the specific trade v1 financed is no longer profitable: under
    # v1, buying vina -8.18 -> -12.0 with qed 0.33 -> 0.18 was worth +0.137, and under v2 it is
    # worth -0.020, so the reward now refuses it. Paying past a modest margin is paying for
    # noise in any case: on this target the docking score's rank correlation with measured
    # potency is +0.078 where -1.0 would be perfect.
    "vina": Ramp(zero=-6.00, full=-9.00),
    # The gate is a strict qed > 0.25 and the v1 fine-tune ended at 0.18, so the ramp must be
    # live BELOW the gate or there is no gradient back up to it. 0.50 is an ordinarily
    # drug-like QED and the point past which more is not worth another axis.
    "qed": Ramp(zero=0.10, full=0.50),
    # Zero credit exactly at the frozen hERG bar, full credit at 0.50, which is where the
    # development controls' own distribution starts. A molecule that FAILS this gate still
    # scores its other terms -- it is a gate failure, not an invalid molecule -- because a
    # reject side where every failure collapses to the same number teaches nothing.
    "herg": Ramp(zero=0.9305925, full=0.50),
}

WEIGHTS_V2: Mapping[str, float] = {"sa": 0.50, "vina": 0.20, "qed": 0.20, "herg": 0.10}
WEIGHTS_V1: Mapping[str, float] = {"sa": 0.50, "vina": 0.25, "qed": 0.15, "herg": 0.10}

#: ``term -> the measurement it reads``. Kept explicit because the record shapes differ.
TERM_MEASUREMENT: Mapping[str, str] = {
    "sa": "sa_score",
    "vina": "vina_affinity",
    "qed": "qed",
    "herg": "herg",
}


@dataclass(frozen=True)
class RewardSpec:
    """One registered reward version, with its parent recorded."""

    version: str
    parent: str | None
    status: str
    weights: Mapping[str, float]
    ramps: Mapping[str, Ramp] | None
    floor: float
    crowding_strength: float
    min_heavy_atoms: int | None
    note: str

    def as_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "status": self.status,
            "weights": dict(self.weights),
            "ramps": (
                None
                if self.ramps is None
                else {k: {"zero": r.zero, "full": r.full} for k, r in self.ramps.items()}
            ),
            "floor": self.floor,
            "crowding_strength": self.crowding_strength,
            "min_heavy_atoms": self.min_heavy_atoms,
            "note": self.note,
        }

    @property
    def definition_hash(self) -> str:
        """Content address of the definition, so a record can name what it trained against."""
        payload = json.dumps(self.as_record(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


REWARD_VERSIONS: Mapping[str, RewardSpec] = {
    "gold_reward_v1": RewardSpec(
        version="gold_reward_v1",
        parent=None,
        status="SUPERSEDED",
        weights=WEIGHTS_V1,
        ramps=None,
        floor=0.0,
        crowding_strength=0.0,
        min_heavy_atoms=None,
        note=(
            "The round's registered default. Kept computable so numbers measured under it stay "
            "readable. It has no admissibility test, which is the defect: a DPO fine-tune "
            "against it reached 85% multi-fragment output and 1 qualifying scaffold from 3."
        ),
    ),
    "gold_reward_v2": RewardSpec(
        version="gold_reward_v2",
        parent="gold_reward_v1",
        status="REGISTERED",
        weights=WEIGHTS_V2,
        ramps=RAMPS_V2,
        floor=ADMISSIBLE_FLOOR,
        crowding_strength=CROWDING_STRENGTH,
        min_heavy_atoms=MIN_HEAVY_ATOMS,
        note=(
            "Admissibility is a precondition worth ADMISSIBLE_FLOOR, every chemistry term "
            "saturates inside its frozen bar so a satisfied term cannot fund a trade against "
            "an unsatisfied one, and QED rises 0.15 -> 0.20 because it was the one rewarded "
            "axis v1 made worse. SA keeps 0.50 on the operator's instruction."
        ),
    ),
}

DEFAULT_REWARD_VERSION = "gold_reward_v2"


def reward_spec(version: str = DEFAULT_REWARD_VERSION) -> RewardSpec:
    """The registered spec for *version*, or raise naming what is registered."""
    try:
        return REWARD_VERSIONS[version]
    except KeyError:
        raise RewardInputError(
            f"{version!r} is not a registered reward; registered: {sorted(REWARD_VERSIONS)}"
        ) from None


def reward_manifest(version: str = DEFAULT_REWARD_VERSION) -> dict[str, Any]:
    """The spec plus its content address, for a campaign's own ledger."""
    spec = reward_spec(version)
    return {**spec.as_record(), "definition_hash": spec.definition_hash}


# ---------------------------------------------------------------------------
# Admissibility: the check v1 did not have
# ---------------------------------------------------------------------------
#: Counterions and solvents, recognised only so that a multi-fragment sample can be REPORTED
#: as salt-like. None of them makes a molecule admissible; see the module docstring.
_SALT_LIKE = (
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[F-]",
    "[Na+]",
    "[K+]",
    "[Li+]",
    "[Ca+2]",
    "[Mg+2]",
    "[NH4+]",
    "Cl",
    "Br",
    "I",
    "O",
    "CO",
    "CCO",
    "CC#N",
    "CS(C)=O",
    "C1CCOC1",
    "CC(=O)O",
    "OC(=O)C(F)(F)F",
    "OS(=O)(=O)O",
    "OC(=O)C(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",
)


def _rdkit() -> tuple[Any, Any]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    return Chem, MurckoScaffold


_SALT_CACHE: dict[str, frozenset[str]] = {}


def _salt_like_canonical() -> frozenset[str]:
    if "set" not in _SALT_CACHE:
        Chem, _ = _rdkit()
        out = set()
        for smiles in _SALT_LIKE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                out.add(Chem.MolToSmiles(mol))
        _SALT_CACHE["set"] = frozenset(out)
    return _SALT_CACHE["set"]


@dataclass(frozen=True)
class Admissibility:
    """Whether a molecule may be a CHOSEN example, and why not when it may not.

    ``reason`` is the empty string when ``ok``. Everything else here is recorded for
    reporting: ``net_charge`` and ``n_radical_electrons`` are never acted on.
    """

    ok: bool
    reason: str
    n_fragments: int
    n_heavy_atoms: int
    net_charge: int
    n_radical_electrons: int
    canonical_smiles: str

    def as_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "n_fragments": self.n_fragments,
            "n_heavy_atoms": self.n_heavy_atoms,
            "net_charge": self.net_charge,
            "n_radical_electrons": self.n_radical_electrons,
            "canonical_smiles": self.canonical_smiles,
        }


def admissibility(smiles: str) -> Admissibility:
    """Parse *smiles* exactly as given and decide whether it may be a chosen example.

    The string is never desalted, stripped to its largest fragment or neutralised first: the
    qualification scores what was submitted, so anything else would score a molecule the gate
    never sees.
    """
    Chem, _ = _rdkit()
    text = str(smiles or "")
    mol = Chem.MolFromSmiles(text)
    # An EMPTY string is not None: RDKit hands back a 0-atom Mol, so the unparseable branch below is
    # skipped, GetMolFrags returns (), and `max(frags, ...)` raises ValueError from inside a helper
    # that is supposed to classify rather than throw. One record with an absent `smiles` then kills a
    # whole batch with an opaque "max() iterable argument is empty". A 0-atom molecule is refused as
    # unparseable, which is what it is.
    if mol is None or mol.GetNumAtoms() == 0:
        return Admissibility(False, "unparseable", 0, 0, 0, 0, "")

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    n_frags = len(frags)
    heavy = mol.GetNumHeavyAtoms()
    charge = Chem.GetFormalCharge(mol)
    radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    canonical = Chem.MolToSmiles(mol)

    if n_frags != 1:
        largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        others = [f for f in frags if f is not largest]
        salt_like = _salt_like_canonical()
        rest = {Chem.MolToSmiles(f) for f in others}
        reason = "multi_fragment_salt_like" if rest and rest <= salt_like else "multi_fragment"
        return Admissibility(False, reason, n_frags, largest.GetNumHeavyAtoms(), charge, radicals, canonical)

    if heavy < MIN_HEAVY_ATOMS:
        return Admissibility(False, "below_heavy_atom_floor", 1, heavy, charge, radicals, canonical)
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return Admissibility(False, "no_carbon", 1, heavy, charge, radicals, canonical)
    return Admissibility(True, "", 1, heavy, charge, radicals, canonical)


def is_admissible(smiles: str) -> bool:
    """:func:`admissibility` reduced to its verdict."""
    return admissibility(smiles).ok


def scaffold_key(smiles: str, molecule_id: str = "") -> str:
    """The scaffold a molecule is crowded against.

    The frozen evaluator's rule, re-implemented rather than imported: a cyclic molecule counts
    under its canonical Bemis-Murcko scaffold and an acyclic one under
    ``acyclic:<molecule_id>``, so two acyclic molecules are two scaffolds. A test holds this
    to the evaluator's own output; the library does not import it, because a reward that
    depended on the evaluator's address would move with it.
    """
    Chem, MurckoScaffold = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return f"acyclic:{molecule_id or smiles}"
    core = MurckoScaffold.GetScaffoldForMol(mol)
    canonical = Chem.MolToSmiles(core) if core is not None else ""
    return canonical or f"acyclic:{molecule_id or Chem.MolToSmiles(mol)}"


# ---------------------------------------------------------------------------
# Reading the measurements out of whatever shape the record has
# ---------------------------------------------------------------------------
#: ``measurement -> the keys it may appear under``. ``structure_score`` is where the evaluator
#: and the baseline arms put vina_affinity; ``safety`` is where a profile puts hERG.
_MEASUREMENT_KEYS: Mapping[str, tuple[str, ...]] = {
    "sa_score": ("sa_score", "synthetic_accessibility"),
    "vina_affinity": ("vina_affinity", "structure_score"),
    "qed": ("qed",),
    "herg": ("herg", "safety"),
}


def _scalar(obj: Any) -> float | None:
    """A float, or ``None`` for an absent or failed measurement."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        if "status" in obj or "value" in obj:
            if str(obj.get("status", "SUCCEEDED")) != "SUCCEEDED":
                return None
            return _scalar(obj.get("value"))
        return None
    if isinstance(obj, bool):
        raise RewardInputError(f"{obj!r} is a flag, not a measurement")
    if isinstance(obj, int | float):
        val = float(obj)
        if not math.isfinite(val):
            raise RewardInputError(f"{obj!r} is not finite and is not a measurement")
        return val
    raise RewardInputError(f"{obj!r} is not a measurement")


def read_measurements(record: Mapping[str, Any]) -> dict[str, float | None]:
    """``{sa_score, vina_affinity, qed, herg}`` from a record, or ``None`` where absent.

    Two places carrying two different values for one measurement is refused rather than
    resolved by key order: the caller knows which one is the frozen scorer's and this module
    does not.
    """
    if not isinstance(record, Mapping):
        raise RewardInputError(f"a molecule record must be a mapping, got {type(record).__name__}")
    # "endpoints" IS WHERE sa_score AND herg ACTUALLY LIVE, and leaving it out was not a cosmetic
    # omission. `_generation_quality_endpoints` and the frozen evaluator both write
    # record["endpoints"]["sa_score"] = {"status": ..., "value": ...}; nothing writes them at the top
    # level except this module's own test helper. So without this key a real submission reads as
    # {'sa_score': None, 'vina_affinity': None, 'qed': 0.07, 'herg': None} -- and the failure is
    # SILENT in the one direction that matters: every row is missing the same terms, so
    # `require_comparable` sees one consistent term set and passes, and `build_preference_pairs`
    # returns pairs ranked on QED alone with SA -- weight 0.50, the measured bottleneck -- absent.
    # Reproduced on committed data: 19 chemistry pairs, all QED-only, while every row carried
    # endpoints.sa_score.value the whole time.
    nested: list[Mapping[str, Any]] = [record]
    for outer in ("endpoints", "properties", "descriptors"):
        inner = record.get(outer)
        if isinstance(inner, Mapping):
            nested.append(inner)
            deeper = inner.get("descriptors")
            if isinstance(deeper, Mapping):
                nested.append(deeper)

    out: dict[str, float | None] = {}
    for name, keys in _MEASUREMENT_KEYS.items():
        seen: list[float] = []
        for holder in nested:
            for key in keys:
                if key in holder:
                    value = _scalar(holder[key])
                    if value is not None:
                        seen.append(value)
        distinct = [v for i, v in enumerate(seen) if not any(math.isclose(v, w) for w in seen[:i])]
        if len(distinct) > 1:
            raise RewardInputError(
                f"{name} appears more than once with different values {distinct}; "
                "the record must carry one measurement per endpoint"
            )
        out[name] = distinct[0] if distinct else None
    return out


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reward:
    """One molecule's reward, with everything a report has to quote beside it."""

    value: float
    basis: str
    version: str
    parent: str | None
    admissible: bool
    reason: str
    terms: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)
    lower: float = 0.0
    upper: float = 0.0
    crowding: float = 1.0
    scaffold: str = ""
    smiles: str = ""
    molecule_id: str = ""

    @property
    def term_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.terms))

    def as_record(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "basis": self.basis,
            "version": self.version,
            "parent": self.parent,
            "admissible": self.admissible,
            "reason": self.reason,
            "terms": dict(self.terms),
            "weights": dict(self.weights),
            "lower": self.lower,
            "upper": self.upper,
            "crowding": self.crowding,
            "scaffold": self.scaffold,
            "molecule_id": self.molecule_id,
        }


def require_comparable(rewards: Sequence[Reward]) -> None:
    """Raise unless every reward was computed the same way.

    A cheap value renormalised over three terms and an exact value over four are different
    scales. Averaging them, ranking them together or pairing across them is the mistake this
    refuses, and it refuses rather than warns because the round already lost a comparison to
    two scorers disagreeing quietly.
    """
    scored = [r for r in rewards if r.admissible]
    if len(scored) < 2:
        return
    first = scored[0]
    for other in scored[1:]:
        if (other.version, other.basis, other.term_names) != (
            first.version,
            first.basis,
            first.term_names,
        ):
            raise RewardInputError(
                "these rewards are not comparable: "
                f"{first.version}/{first.basis}/{first.term_names} against "
                f"{other.version}/{other.basis}/{other.term_names}"
            )


def _credits(measured: Mapping[str, float | None], ramps: Mapping[str, Ramp]) -> dict[str, float]:
    out: dict[str, float] = {}
    for term, name in TERM_MEASUREMENT.items():
        value = measured.get(name)
        if value is not None:
            out[term] = ramps[term](value)
    return out


def gold_reward_v2(
    record: Mapping[str, Any] | str,
    *,
    basis: str = "auto",
    crowding: float = 1.0,
) -> Reward:
    """The registered v2 reward for one molecule.

    ``basis="auto"`` is exact when vina_affinity is present and cheap otherwise;
    ``basis="exact"`` raises if it is absent, because a silent fallback is how a cheap number
    ends up quoted as an exact one. An inadmissible molecule is ``0.0`` with no terms scored:
    worthless as a chosen example, fully usable as a rejected one.
    """
    if basis not in {"auto", "exact", "cheap"}:
        raise RewardInputError(f"basis must be auto, exact or cheap, not {basis!r}")
    spec = reward_spec("gold_reward_v2")
    rec: Mapping[str, Any] = {"smiles": record} if isinstance(record, str) else record
    smiles = str(rec.get("smiles", ""))
    molecule_id = str(rec.get("molecule_id", "") or "")
    verdict = admissibility(smiles)

    if not verdict.ok:
        return Reward(
            value=0.0,
            basis="admissibility",
            version=spec.version,
            parent=spec.parent,
            admissible=False,
            reason=verdict.reason,
            lower=0.0,
            upper=0.0,
            crowding=1.0,
            scaffold=scaffold_key(smiles, molecule_id),
            smiles=smiles,
            molecule_id=molecule_id,
        )

    measured = read_measurements(rec)
    has_vina = measured.get("vina_affinity") is not None
    resolved = "exact" if basis == "exact" or (basis == "auto" and has_vina) else "cheap"
    if resolved == "exact" and not has_vina:
        raise RewardInputError(
            "basis='exact' needs vina_affinity; this record has none. Score it cheap, or dock "
            "it first -- do not compare a cheap value against an exact one."
        )

    credits = _credits(measured, RAMPS_V2)
    if not credits:
        raise RewardInputError(
            f"{smiles!r} is admissible but carries no measurement at all, so it has no reward; "
            "an unmeasured molecule is not a bad molecule and must not be scored as one"
        )
    if resolved == "exact" and len(credits) != len(WEIGHTS_V2):
        missing = sorted(set(WEIGHTS_V2) - set(credits))
        raise RewardInputError(f"basis='exact' needs every term; missing {missing}")

    mass = sum(WEIGHTS_V2[t] for t in credits)
    weights = {t: WEIGHTS_V2[t] / mass for t in credits}
    chem = sum(weights[t] * credits[t] for t in credits)
    clamped = 0.0 if crowding < 0.0 else 1.0 if crowding > 1.0 else float(crowding)

    def _lift(x: float) -> float:
        return ADMISSIBLE_FLOOR + (1.0 - ADMISSIBLE_FLOOR) * x

    # The bounds use the FULL weight set, so they bracket the exact value a dock would give.
    # `upper` is what lets a caller drop a molecule before paying for the pose.
    present = sum(WEIGHTS_V2[t] * credits[t] for t in credits)
    absent = sum(WEIGHTS_V2[t] for t in WEIGHTS_V2 if t not in credits)
    return Reward(
        value=_lift(chem * clamped),
        basis=resolved,
        version=spec.version,
        parent=spec.parent,
        admissible=True,
        reason="",
        terms=credits,
        weights=weights,
        lower=_lift(present * clamped),
        upper=_lift((present + absent) * clamped),
        crowding=clamped,
        scaffold=scaffold_key(smiles, molecule_id),
        smiles=smiles,
        molecule_id=molecule_id,
    )


def gold_reward_v1(record: Mapping[str, Any] | str) -> Reward:
    """The superseded v1 reward, kept computable so v1 records stay readable.

    It needs all four measurements, has no cheap path and no admissibility test, and reports
    ``admissible=True`` for every parseable molecule because that is what it did. Never
    compare a v1 value with a v2 value: :func:`require_comparable` refuses it for you.
    """
    spec = reward_spec("gold_reward_v1")
    rec: Mapping[str, Any] = {"smiles": record} if isinstance(record, str) else record
    smiles = str(rec.get("smiles", ""))
    measured = read_measurements(rec)
    missing = [n for n in ("sa_score", "vina_affinity", "qed", "herg") if measured.get(n) is None]
    if missing:
        raise RewardInputError(f"gold_reward_v1 needs every measurement; missing {missing}")

    sa = float(measured["sa_score"])  # type: ignore[arg-type]
    vina = float(measured["vina_affinity"])  # type: ignore[arg-type]
    qed = float(measured["qed"])  # type: ignore[arg-type]
    herg = float(measured["herg"])  # type: ignore[arg-type]
    terms = {
        "sa": (10.0 - sa) / 9.0,
        "vina": min(max((-vina - 6.0) / 6.0, 0.0), 1.0),
        "qed": qed,
        "herg": 1.0 - herg,
    }
    value = sum(WEIGHTS_V1[t] * terms[t] for t in WEIGHTS_V1)
    return Reward(
        value=value,
        basis="exact",
        version=spec.version,
        parent=spec.parent,
        admissible=True,
        reason="v1 had no admissibility test; that is the defect v2 exists to fix",
        terms=terms,
        weights=dict(WEIGHTS_V1),
        lower=value,
        upper=value,
        scaffold=scaffold_key(smiles, str(rec.get("molecule_id", "") or "")),
        smiles=smiles,
        molecule_id=str(rec.get("molecule_id", "") or ""),
    )


def score_batch(
    records: Iterable[Mapping[str, Any] | str],
    *,
    basis: str = "auto",
    crowding: bool = True,
) -> list[Reward]:
    """Score a whole batch, discounting the second and later member of each scaffold.

    Crowding is a property of a SET, so it is applied here and never by
    :func:`gold_reward_v2` on a single molecule. The best member of a scaffold keeps its full
    value and only its duplicates are discounted, by ``1 - CROWDING_STRENGTH*(1 - 1/rank)``:
    collapsing onto one chemotype therefore cannot raise the set's mean reward, while the one
    good representative of that chemotype is left intact.

    Input order is preserved, and the ranking is broken by canonical SMILES so a batch scores
    identically on a re-run.
    """
    scored = [gold_reward_v2(r, basis=basis) for r in records]
    # REFUSE A MIXED BASIS HERE, BEFORE RANKING. A cheap value renormalises over the terms that are
    # present and an exact one does not, so the two are different scales -- this module's own
    # docstring says they must never be compared. Ranking across the mix silently demotes a molecule
    # for lacking a docking score rather than for its chemistry: measured on two same-scaffold
    # molecules differing only in whether vina was present, the cheap one sorted second and took a
    # 0.875 crowding discount it had not earned. `build_preference_pairs` calls require_comparable
    # afterwards, but score_batch is advertised as a top-level API in its own right, so waiting for a
    # caller to check is waiting for the one caller who forgets.
    require_comparable(scored)
    if not crowding:
        return scored

    ranks: dict[int, int] = {}
    seen: dict[str, int] = {}
    for index, reward in sorted(
        ((i, r) for i, r in enumerate(scored) if r.admissible),
        key=lambda pair: (-pair[1].value, pair[1].smiles, pair[0]),
    ):
        seen[reward.scaffold] = seen.get(reward.scaffold, 0) + 1
        ranks[index] = seen[reward.scaffold]

    out: list[Reward] = []
    for index, reward in enumerate(scored):
        rank = ranks.get(index)
        if rank is None:
            out.append(reward)
            continue
        out.append(_crowded(reward, 1.0 - CROWDING_STRENGTH * (1.0 - 1.0 / rank)))
    return out


def _crowded(reward: Reward, factor: float) -> Reward:
    """Re-apply a crowding factor to an already-computed reward, floor intact.

    The discount multiplies the CHEMISTRY and never the floor, so an admissible molecule stays
    worth at least :data:`ADMISSIBLE_FLOOR` however crowded its scaffold is. Crowding must not
    be able to push a valid molecule down onto a fragment's score.
    """

    def _redo(value: float) -> float:
        chem = (value - ADMISSIBLE_FLOOR) / (1.0 - ADMISSIBLE_FLOOR)
        return ADMISSIBLE_FLOOR + (1.0 - ADMISSIBLE_FLOOR) * chem * factor

    return Reward(
        value=_redo(reward.value),
        basis=reward.basis,
        version=reward.version,
        parent=reward.parent,
        admissible=reward.admissible,
        reason=reward.reason,
        terms=reward.terms,
        weights=reward.weights,
        lower=_redo(reward.lower),
        upper=_redo(reward.upper),
        crowding=factor,
        scaffold=reward.scaffold,
        smiles=reward.smiles,
        molecule_id=reward.molecule_id,
    )


# ---------------------------------------------------------------------------
# Preference pairs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pair:
    """One ``(chosen, rejected)`` example. ``kind`` says which signal it carries."""

    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    margin: float
    kind: str
    chosen_id: str
    rejected_id: str
    chosen_scaffold: str
    rejected_admissible: bool
    rejected_reason: str
    basis: str
    terms: tuple[str, ...]
    reward_version: str

    def as_record(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_reward": self.chosen_reward,
            "rejected_reward": self.rejected_reward,
            "margin": self.margin,
            "kind": self.kind,
            "chosen_id": self.chosen_id,
            "rejected_id": self.rejected_id,
            "chosen_scaffold": self.chosen_scaffold,
            "rejected_admissible": self.rejected_admissible,
            "rejected_reason": self.rejected_reason,
            "basis": self.basis,
            "terms": list(self.terms),
            "reward_version": self.reward_version,
        }


@dataclass(frozen=True)
class PairSet:
    """A preference dataset, with the numbers that say whether it is one.

    ``independent_units`` is the count of DISTINCT molecules in the set, not the number of
    rows: a molecule that appears in three pairs contributes one molecule's worth of
    information, and quoting rows as ``n`` overstates the data. ``n_chemistry_pairs`` and
    ``n_validity_pairs`` are separate because a set made only of the second teaches validity
    only, and the measured consequence of that is the unlikelihood arm -- 1% fragments, median
    sa_score 6.71 against the frozen parent's 6.27, and a score of zero.
    """

    pairs: tuple[Pair, ...]
    n_scored: int
    n_admissible: int
    n_inadmissible: int
    n_chemistry_pairs: int
    n_validity_pairs: int
    independent_units: int
    fragment_rate: float
    inadmissible_reasons: Mapping[str, int]
    dropped: Mapping[str, int]
    reward_version: str
    basis: str
    manifest: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "n_pairs": len(self.pairs),
            "n_scored": self.n_scored,
            "n_admissible": self.n_admissible,
            "n_inadmissible": self.n_inadmissible,
            "n_chemistry_pairs": self.n_chemistry_pairs,
            "n_validity_pairs": self.n_validity_pairs,
            "independent_units": self.independent_units,
            "fragment_rate": self.fragment_rate,
            "inadmissible_reasons": dict(self.inadmissible_reasons),
            "dropped": dict(self.dropped),
            "reward_version": self.reward_version,
            "basis": self.basis,
            "reward_manifest": dict(self.manifest),
        }


def build_preference_pairs(
    records: Sequence[Mapping[str, Any] | str],
    *,
    basis: str = "auto",
    crowding: bool = True,
    min_margin: float = 0.05,
    max_pairs_per_scaffold: int = 4,
    max_uses_per_chosen: int = 2,
    min_chemistry_pair_fraction: float = 0.25,
    max_pairs: int | None = None,
) -> PairSet:
    """Build ``(chosen, rejected)`` pairs from a list of molecules.

    THE ONE GUARANTEE THIS FUNCTION MAKES STRUCTURALLY: the chosen side is drawn from the
    admissible pool only, so a multi-fragment, unparseable, sub-floor or carbon-free molecule
    can never be a chosen example. It is not enforced by comparing rewards -- floats tie, and
    a tie is how a fragment would get through -- but by never putting an inadmissible molecule
    in the pool the chosen side is taken from.

    Two kinds of pair, and the mix is controlled rather than left to the data:

    ``chemistry``  both sides admissible, the better reward chosen, margin at least
                   *min_margin*. These are built FIRST and are never sacrificed, because they
                   are the signal the unlikelihood arm lost.
    ``validity``   an admissible chosen against an inadmissible rejected. These are the
                   negative signal that was missing entirely -- meeko rejects a fragment
                   before any gate sees it, so no fragment has ever reached a preference set
                   as a negative. They are exempt from *min_margin*, since the reward gap is
                   structural, and capped so they cannot exceed
                   ``(1 - f)/f`` times the chemistry pairs.

    Refuses outright when no chemistry pair can be made, naming the counts, rather than
    returning a validity-only set that looks like a dataset.
    """
    if not 0.0 < min_chemistry_pair_fraction <= 1.0:
        raise RewardInputError("min_chemistry_pair_fraction must be in (0, 1]")
    if max_pairs_per_scaffold < 1 or max_uses_per_chosen < 1:
        raise RewardInputError("max_pairs_per_scaffold and max_uses_per_chosen must be >= 1")

    scored = score_batch(records, basis=basis, crowding=crowding)
    if not scored:
        raise RewardInputError("no molecules to pair")
    require_comparable(scored)

    admissible = sorted((r for r in scored if r.admissible), key=lambda r: (-r.value, r.smiles))
    inadmissible = [r for r in scored if not r.admissible]
    reasons: dict[str, int] = {}
    for reward in inadmissible:
        reasons[reward.reason] = reasons.get(reward.reason, 0) + 1
    fragments = sum(v for k, v in reasons.items() if k.startswith("multi_fragment"))

    dropped: dict[str, int] = {}

    def _drop(reason: str, n: int = 1) -> None:
        if n > 0:
            dropped[reason] = dropped.get(reason, 0) + n

    resolved_basis = admissible[0].basis if admissible else "admissibility"
    terms = admissible[0].term_names if admissible else ()
    scaffold_used: dict[str, int] = {}
    chosen_uses: dict[str, int] = {}

    def _may_choose(reward: Reward) -> str:
        if chosen_uses.get(reward.smiles, 0) >= max_uses_per_chosen:
            return "chosen_reuse_cap"
        if scaffold_used.get(reward.scaffold, 0) >= max_pairs_per_scaffold:
            return "scaffold_quota"
        return ""

    def _take(reward: Reward) -> None:
        chosen_uses[reward.smiles] = chosen_uses.get(reward.smiles, 0) + 1
        scaffold_used[reward.scaffold] = scaffold_used.get(reward.scaffold, 0) + 1

    def _pair(chosen: Reward, rejected: Reward, kind: str) -> Pair:
        _take(chosen)
        return Pair(
            chosen=chosen.smiles,
            rejected=rejected.smiles,
            chosen_reward=chosen.value,
            rejected_reward=rejected.value,
            margin=chosen.value - rejected.value,
            kind=kind,
            chosen_id=chosen.molecule_id,
            rejected_id=rejected.molecule_id,
            chosen_scaffold=chosen.scaffold,
            rejected_admissible=rejected.admissible,
            rejected_reason=rejected.reason,
            basis=resolved_basis,
            terms=terms,
            reward_version=DEFAULT_REWARD_VERSION,
        )

    # Chemistry pairs, best against worst and inward. Margins shrink monotonically as the walk
    # closes, so the first failure ends it and the remainder is counted rather than retried.
    chemistry: list[Pair] = []
    low, high = 0, len(admissible) - 1
    while low < high:
        chosen, rejected = admissible[low], admissible[high]
        if chosen.value - rejected.value < min_margin:
            _drop("chemistry_margin_below_min", (high - low + 1) // 2)
            break
        refusal = _may_choose(chosen)
        if refusal:
            _drop(refusal)
            low += 1
            continue
        chemistry.append(_pair(chosen, rejected, "chemistry"))
        low += 1
        high -= 1

    if not chemistry:
        raise RewardInputError(
            "no chemistry pair could be built: "
            f"{len(admissible)} admissible of {len(scored)} molecules, "
            f"{fragments} multi-fragment, drops {dropped or '{}'}. A validity-only preference "
            "set trains connectivity and lets chemistry rot, which is the measured "
            "unlikelihood failure -- median sa_score 6.71 against the frozen parent's 6.27 "
            "and a score of zero. Sample more molecules or lower min_margin deliberately."
        )

    # Validity pairs, capped so they cannot swamp the chemistry signal.
    ratio = (1.0 - min_chemistry_pair_fraction) / min_chemistry_pair_fraction
    allowance = int(len(chemistry) * ratio)
    validity: list[Pair] = []
    cursor = 0
    for rejected in inadmissible:
        if len(validity) >= allowance:
            _drop("validity_over_balance_cap")
            continue
        placed = False
        for _ in range(len(admissible)):
            chosen = admissible[cursor % len(admissible)]
            cursor += 1
            if not _may_choose(chosen):
                validity.append(_pair(chosen, rejected, "validity"))
                placed = True
                break
        if not placed:
            _drop("no_eligible_chosen")

    # Truncation takes validity pairs first: the balance cap already bounds them from above, so
    # removing them can only raise the chemistry fraction, never lower it.
    if max_pairs is not None:
        if max_pairs < 1:
            raise RewardInputError("max_pairs must be >= 1")
        room = max(0, max_pairs - len(chemistry))
        if len(validity) > room:
            _drop("over_max_pairs", len(validity) - room)
            validity = validity[:room]
        if len(chemistry) > max_pairs:
            _drop("over_max_pairs", len(chemistry) - max_pairs)
            chemistry = chemistry[:max_pairs]

    pairs = tuple(chemistry + validity)
    units = {p.chosen for p in pairs} | {p.rejected for p in pairs}
    return PairSet(
        pairs=pairs,
        n_scored=len(scored),
        n_admissible=len(admissible),
        n_inadmissible=len(inadmissible),
        n_chemistry_pairs=len(chemistry),
        n_validity_pairs=len(validity),
        independent_units=len(units),
        fragment_rate=fragments / len(scored),
        inadmissible_reasons=reasons,
        dropped=dropped,
        reward_version=DEFAULT_REWARD_VERSION,
        basis=resolved_basis,
        manifest=reward_manifest(DEFAULT_REWARD_VERSION),
    )
