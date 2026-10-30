"""The training-SMILES file a REINVENT4 transfer-learning run is allowed to read.

One function, :func:`build_training_smiles`, stands between the frozen ZAK/C9R
split and ``reinvent/runmodes/TL``. It exists because three separate things have
to be true of that file before a single gradient step is worth taking, and each
of them fails silently if it is left to the caller.

**It is the exclusion boundary, offered rather than asserted.** Every compound of
the requested partition is offered to Step 6's protected registry through the
same :func:`~drug_agent.selfevolve.target.splits.screen_records` every other arm
uses, and what the boundary refused is counted. ``exclusion_registry`` has no
default, deliberately: a screening step whose registry argument may be omitted is
a screening step that can be switched off by forgetting it, and that is the one
mistake this module is here to make impossible.

**The boundary is measured against the thing it is supposed to stop.** A
partition offer that contains no protected compound cannot tell you whether the
boundary works — it would pass just as happily against an empty registry. So the
held-out compounds are offered to the same registry as a positive control, in
the shape ``scripts/research/calibrate_zak_benchmark.py`` already uses, and the
control's counts come back in the returned record. Every one of them must be
quarantined and none admitted.

This uses the held-out partition's **identity only** — the molecule ids,
connectivity keys and family ids that ``split_membership.json`` records — exactly
as :mod:`drug_agent.selfevolve.target.baseline_arms` already does when it offers
every compound to the boundary. No measured value, label or activity of a
held-out compound is read here, by this module or by anything it calls, so the
project's ``test_access_count`` is unchanged by running it.

Where ``baseline_arms.py`` *reports* a non-zero leak count, this module
**raises**. A training file is not a report: by the time anyone reads the number,
the weights have already seen the compound.

**It is pre-filtered against the prior's vocabulary, because upstream does not
filter — it aborts.** ``reinvent/utils/config_parse.py:validate_tokens`` raises
``ValueError`` on the *first* out-of-vocabulary token it meets, and nothing in
``read_smiles_csv_file``, ``run_transfer_learning`` or ``Reinvent.main`` catches
it. One non-conforming compound out of several hundred therefore kills the whole
run before training starts, having consumed the model load and nothing else. So
every candidate is tokenised here with upstream's own
``SMILES_TOKENS_REGEX`` against the token set stored in the prior, and what was
dropped is counted rather than quietly skipped.

**And against the standardiser, because upstream does not abort there — it goes
quiet.** The vocabulary is the loud failure. The standardiser in front of it is
the silent one, and it is the more dangerous of the two.
``run_transfer_learning`` prepends ``RDKitStandardizer(None, isomeric=False)`` to
the read pipeline for the ``Reinvent`` generator; its ``apply_filter`` returns
``None`` for anything its default chain refuses, and ``read_smiles_csv_file``
answers that with ``if not smiles: continue`` (``config_parse.py:140-141``). No
exception, no non-zero exit, nothing in the result: 775 rows offered become 700
trained, and 700 is reported as if it were the answer to a different question.
Measured directly on a synthetic fixture: a 51-row file containing one
out-of-vocabulary compound trains on 50, exits 0, and says nothing.

So the same chain is transcribed here, from
``reinvent/chemistry/standardization/filter_registry.py``, and every compound is
run through it before it is admitted — largest fragment, hydrogen removal, salt
stripping, charge neutralisation, cleanup, the ``[C-]#[N+]``/``[OH+]``/``[SH]``
pattern refusal, and ``2 < heavy atoms < 70`` over C/N/O/F/S/Cl/Br. What the
chain refuses is counted under its own name in the returned record, so a
training set that shrank says so before any weight moves.

The filter is checked against the SMILES REINVENT4 will actually validate, not
the one the caller handed over. ``validate_tokens`` runs *after* the standardiser
and ``conversions.randomize_smiles``, on the randomised string. A molecule is
therefore admitted here only if it survives the standardiser and its standardised
form **and** a batch of randomised rewritings are all inside the vocabulary; and
what is written to the file is the *standardised* form, so that the rows the file
offers and the molecules REINVENT4 trains on are the same objects rather than two
counts that happen to agree. That is deliberately stricter than upstream's own
test: over-dropping costs a training molecule, while under-dropping costs either
the entire run or, worse, a silent fraction of it.

**And it is de-duplicated on the standardised form, because upstream reads the
file with ``remove_duplicates=True``.** ``run_transfer_learning`` passes that at
line 80, so two rows that arrive at the same string are one training molecule,
and a file offering both is a file whose row count is not what the model trained
on. The collapse is neither hypothetical nor rare, because stereochemistry is
stripped before anything is compared: an enantiomeric pair is one molecule, and
so are an amine and its protonated form, an acid and its sodium salt, and a
cis/trans alkene pair. Each of the four was verified directly, and a real ZAK
training partition of several hundred compounds contains them.

Worse, whether the collapse fires is a draw nobody can call in advance.
``remove_duplicates`` compares the string *after*
``conversions.randomize_smiles`` has rewritten it, so two copies of one molecule
collide only when their two random rewritings happen to agree — measured over
20,000 trials at 0.0144 for the enantiomeric pair below, 0.0548 for the amine
pair, 0.0267 for the acid pair and 0.0530 for the alkene pair, and at exactly
1.0 for a molecule symmetric enough to have one rewriting. Within a single fresh
process the draws are deterministic, so one file always collapses and another
never does, and which one you have cannot be read off the file. Over a partition
with a few dozen duplicated pairs the chance that at least one collides is not
small — and the run that pays for it has already spent the GPU, because the
shrinkage guard in ``Reinvent4Adapter.fit`` refuses *after* training and
``actions.py`` turns that refusal into ``adapt_generator FAILED``.

So the collapse is performed here, on the same canonical form upstream collapses
to — the standardised SMILES this module writes — before the file exists. After
that the file offers one row per distinct structure, ``remove_duplicates`` has
nothing left to remove whatever it randomises to, and the training-set size is a
fact rather than a draw. What was merged is counted under
``n_duplicates_collapsed`` and attributed group by group, because a partition of
775 compounds that is really 760 molecules is a finding about the data and not a
detail of the file writer.

**It is written in the format upstream actually reads.**
``read_smiles_csv_file``'s defaults are ``delimiter="\\t"`` and ``header=False``,
and ``run_transfer_learning`` calls it with ``columns=0`` for the unconditional
``Reinvent`` generator. A header line is consequently parsed as a molecule, and a
comma-separated file yields one column of garbage that then fails token
validation. This module writes one tab-delimited, header-less column, and that is
not a style choice.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def _repo_root() -> Path:
    """The checkout whose frozen split and protected registry apply.

    Identical in behaviour to :func:`baseline_arms._repo_root`: a candidate
    worktree gets the parent's snapshot through ``DRUGAGENT_REPO_ROOT``, and the
    override is honoured only when it really holds the snapshot, so a stale shell
    variable cannot silently redirect an ordinary run.
    """
    default = Path(__file__).resolve().parents[4]
    override = os.environ.get("DRUGAGENT_REPO_ROOT", "").strip()
    if override:
        candidate = Path(override).resolve()
        if candidate != default and (candidate / ".external" / "research" / "zak-c9r").is_dir():
            return candidate
    return default


REPO = _repo_root()
SNAPSHOT = REPO / ".external" / "research" / "zak-c9r"
DERIVED = SNAPSHOT / "derived"
PROTECTED = SNAPSHOT / "protected"

#: The partition name that must never become training data, whatever a caller
#: asks for. Named rather than inferred so the refusal is greppable.
HELD_OUT_PARTITION = "test"

#: How many randomised rewritings of each molecule are token-checked before it is
#: admitted. ``read_smiles_csv_file`` validates one randomisation per molecule per
#: run; checking a batch here makes the pre-filter stricter than the gate it is
#: protecting, which is the safe direction.
VOCAB_RANDOMISATIONS = 16

#: Fixed so two builds of the same partition drop the same molecules. It seeds
#: only this module's *check*; REINVENT4 randomises with its own RNG at read time.
VOCAB_RANDOM_SEED = 20260917

# ---------------------------------------------------------------------------
# Upstream's tokeniser, transcribed
# ---------------------------------------------------------------------------
# Copied verbatim from .external/REINVENT4/reinvent/datapipeline/filters/regex.py
# so that the pre-filter splits SMILES into exactly the tokens
# ``config_parse.validate_tokens`` will compare against the vocabulary. It is
# transcribed rather than imported because ``reinvent`` is not importable from
# this process — it lives on the external driver's PYTHONPATH, not this one's —
# and a *near* tokeniser would admit the compound that aborts the run.
_BRACKETS = r"\[[^]]+]"
_ALIPHATIC = r"Br?|Cl?|N|O|S|P|F|I"
_AROMATIC = r"b|c|n|o|s|p"
_BONDS = r"-|=|#|\$|:|\\|/"
_BRANCH = r"\(|\)"
_LABELS = r"%\d{2}|\d"
_MISC = r"\.|\*"
SMILES_TOKENS_REGEX = re.compile(
    rf"({_BRACKETS}|{_ALIPHATIC}|{_AROMATIC}|{_BONDS}|{_BRANCH}|{_LABELS}|{_MISC})"
)


def _json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the Step 5/6/7 build scripts first")
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The prior's vocabulary
# ---------------------------------------------------------------------------
class _TolerantUnpickler(pickle.Unpickler):
    """Read a torch pickle's object graph without the classes it references.

    The prior is a ``torch.save`` archive whose ``network`` value is a
    ``reinvent.models`` object, so an ordinary ``torch.load`` needs REINVENT4 on
    ``sys.path`` — which this process does not have and should not acquire just
    to read 131 strings. The vocabulary itself is plain ``dict``/``str``/``int``,
    so every foreign class is replaced by an inert stand-in and every tensor
    storage by ``None``. Nothing from the file is executed or called.
    """

    class _Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __setstate__(self, state: Any) -> None:
            pass

    _SAFE_MODULES = frozenset({"builtins", "collections", "copyreg", "uuid", "datetime"})

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] in self._SAFE_MODULES:
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError):
                return self._Stub
        return self._Stub

    def persistent_load(self, pid: Any) -> Any:
        return None


def prior_vocabulary_tokens(path: str | Path) -> set[str]:
    """The token set the prior at *path* accepts.

    Accepts either a REINVENT4 checkpoint (``.prior``/``.chkpt``, a ``torch.save``
    zip whose ``vocabulary`` maps ``"tokens"`` to ``token -> index``) or a JSON
    file carrying the same tokens, so a caller can pin a vocabulary without
    shipping a 23 MB pickle.

    The tokens are the same object ``reinvent/utils/helpers.py:
    get_tokens_from_vocabulary`` returns for a ``Reinvent`` prior — verified equal
    on ``priors/reinvent.prior`` — so a pre-filter against them is a pre-filter
    against the gate.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"vocabulary source {path} does not exist")

    if path.suffix.lower() == ".json":
        return _tokens_from_object(json.loads(path.read_text()), path)

    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith("data.pkl")]
            if not names:
                raise ValueError(f"{path} is a zip but carries no data.pkl")
            raw = archive.read(sorted(names)[0])
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path} is not a torch checkpoint archive and not a .json vocabulary") from exc

    return _tokens_from_object(_TolerantUnpickler(io.BytesIO(raw)).load(), path)


def _tokens_from_object(obj: Any, path: Path) -> set[str]:
    """Dig the token set out of a checkpoint dict or a JSON vocabulary."""
    node: Any = obj
    if isinstance(node, Mapping) and "vocabulary" in node:
        node = node["vocabulary"]
    if isinstance(node, Mapping) and "tokens" in node:
        node = node["tokens"]
    # A Mapping iterates its keys and a sequence its items; both spellings of a
    # vocabulary therefore reduce to the same set of token strings.
    if isinstance(node, (Mapping, list, tuple, set)):
        tokens = {str(k) for k in node}
    else:
        raise ValueError(f"{path} does not carry a readable vocabulary; found {type(node).__name__}")
    if not tokens:
        raise ValueError(f"{path} carries an empty vocabulary")
    return tokens


# ---------------------------------------------------------------------------
# Upstream's standardiser, transcribed
# ---------------------------------------------------------------------------
# Copied from .external/REINVENT4/reinvent/chemistry/standardization/filter_registry.py
# (FilterRegistry.standardize and the filters it calls, lines 59-199) and
# rdkit_standardizer.py, which is what run_transfer_learning installs as the
# first read action for the ``Reinvent`` generator. Transcribed for the same
# reason the tokeniser above is: ``reinvent`` is not importable from this
# process. A *near* standardiser is worse than none, because it would admit the
# compound that then vanishes from the training set without a word, so every
# constant below is upstream's own value and every stage is in upstream's order.

#: ``filter_registry.DEFAULT_ELEMENTS`` — C, N, O, F, S, Cl, Br, the elements the
#: classical Reinvent prior was trained on, by atomic number.
DEFAULT_ELEMENTS: frozenset[int] = frozenset({6, 7, 8, 9, 16, 17, 35})

#: ``filter_registry._valid_size``'s defaults. Strict on both sides: the test is
#: ``min < heavy < max``, so 2 heavy atoms is refused and so is 70.
MIN_HEAVY_ATOMS = 2
MAX_HEAVY_ATOMS = 70

#: ``filter_registry._unwanted_patterns``. A molecule matching any of these is
#: dropped outright, whatever else is true of it.
UNWANTED_PATTERNS: tuple[str, ...] = ("[C-]#[N+]", "[OH+]", "[SH]")

#: ``filter_registry.NEUTRALIZE_PATTERNS``, in order. Each is applied repeatedly
#: until it no longer matches, exactly as ``_neutralise_charges`` does.
NEUTRALIZE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("[n+;H]", "n"),  # Imidazoles
    ("[N+;!H0]", "N"),  # Amines
    ("[$([O-]);!$([O-][#7])]", "O"),  # Carboxylic acids and alcohols
    ("[S-;X1]", "S"),  # Thiols
    ("[$([N-;X2]S(=O)=O)]", "N"),  # Sulfonamides
    ("[$([N-;X2][C,N]=C)]", "N"),  # Enamines
    ("[n-]", "[nH]"),  # Tetrazoles
    ("[$([S-]=O)]", "S"),  # Sulfoxides
    ("[$([N-]C=O)]", "N"),  # Amides
)

#: ``RDKitStandardizer`` is constructed by ``run_transfer_learning`` as
#: ``RDKitStandardizer(None, isomeric=parameters.isomeric_smiles)``, and
#: ``isomeric_smiles`` defaults to False upstream and in
#: ``scripts/external/reinvent4_tl.py``. Stereochemistry is therefore dropped
#: from the SMILES the model reads, and it is dropped here too so the two agree.
STANDARDIZE_ISOMERIC = False


def _neutralise_charges(mol: Any) -> Any:
    """``filter_registry._neutralise_charges``, transcribed."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles

    reactions = [(MolFromSmarts(a), MolFromSmiles(b, False)) for a, b in NEUTRALIZE_PATTERNS]
    for reactant, product in reactions:
        while mol.HasSubstructMatch(reactant):
            mol = AllChem.ReplaceSubstructs(mol, reactant, product)[0]
    return mol


def standardize_mol(mol: Any) -> Any | None:
    """``FilterRegistry.standardize`` with its defaults, stage for stage.

    Returns the standardised molecule, or ``None`` where upstream's chain returns
    ``None`` — which is where ``apply_filter`` returns ``None`` and where
    ``read_smiles_csv_file`` then drops the row without a word.
    """
    from rdkit.Chem import SaltRemover, rdmolops
    from rdkit.Chem.rdmolfiles import MolFromSmarts
    from rdkit.Chem.rdmolops import RemoveHs

    if mol is None:
        return None

    # _get_largest_fragment
    largest = None
    for frag in rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        if frag is None:
            continue
        if largest is None or largest.GetNumHeavyAtoms() < frag.GetNumHeavyAtoms():
            largest = frag
    mol = largest
    if mol is None:
        return None

    # _remove_hydrogens, _remove_salts, _neutralise_charges
    mol = RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol is None:
        return None
    mol = SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)
    if mol is None:
        return None
    mol = _neutralise_charges(mol)
    if mol is None:
        return None

    # _general_cleanup
    rdmolops.Cleanup(mol)
    rdmolops.SanitizeMol(mol)
    mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol is None:
        return None

    # _unwanted_patterns
    for pattern in UNWANTED_PATTERNS:
        if mol.HasSubstructMatch(MolFromSmarts(pattern)):
            return None

    # _valid_size: the heavy-atom window and the element list, in that order
    if not MIN_HEAVY_ATOMS < mol.GetNumHeavyAtoms() < MAX_HEAVY_ATOMS:
        return None
    if not all(atom.GetAtomicNum() in DEFAULT_ELEMENTS for atom in mol.GetAtoms()):
        return None
    return mol


def standardize_smiles(smiles: str) -> str | None:
    """The SMILES upstream's standardiser would hand the tokeniser, or ``None``.

    ``None`` means ``read_smiles_csv_file`` would drop this row silently, which
    is the whole reason this function exists. An exception anywhere in the chain
    is also ``None``: upstream would let it propagate out of ``apply_filter`` and
    abort the run, so a compound that makes the chain raise is one this file must
    not contain either.
    """
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = standardize_mol(mol)
        if mol is None:
            return None
        standardized = Chem.MolToSmiles(mol, isomericSmiles=STANDARDIZE_ISOMERIC)
    except Exception:
        return None
    return standardized or None


def _in_vocabulary(smiles: str, tokens: set[str]) -> bool:
    """Whether every token upstream would extract from *smiles* is allowed.

    ``find_invalid_tokens`` drops tokens containing ``*`` before comparing, for
    Lib/LinkInvent attachment points; that is reproduced so the two agree.
    """
    found = {t for t in SMILES_TOKENS_REGEX.findall(smiles) if "*" not in t}
    return found.issubset(tokens)


#: The reasons a compound can fail to reach the training file. Each names the
#: stage of upstream's read pipeline that would have removed it, so a shrunken
#: training set can be attributed rather than merely noticed.
FILTER_REASONS: tuple[str, ...] = ("unparseable", "standardizer", "vocabulary")


def classify_smiles(smiles: str, tokens: set[str]) -> tuple[str | None, str | None]:
    """``(smiles to write, reason it was refused)`` — exactly one of the two is set.

    The stages are upstream's, in upstream's order, because that is the order in
    which a compound is lost:

    ``unparseable``
        RDKit cannot read the recorded SMILES at all. Upstream's
        ``conversions.smile_to_mol`` returns ``None`` and the row disappears.
    ``standardizer``
        :func:`standardize_smiles` refuses it — the heavy-atom window, a
        disallowed element, an unwanted pattern, or a chain stage that raised.
        This is the silent class: exit 0, no warning the caller sees, a smaller
        training set than anybody asked for.
    ``vocabulary``
        it survives standardisation but the standardised form, or one of
        :data:`VOCAB_RANDOMISATIONS` randomised rewritings of it, carries a token
        the prior does not know. This is the loud class: ``validate_tokens``
        would raise and kill the whole run on the first one.

    The randomisations are checked because ``read_smiles_csv_file`` validates
    tokens *after* ``conversions.randomize_smiles`` has rewritten the molecule,
    and a rewriting can introduce a ring-closure label the canonical form never
    used. What comes back is the *standardised* SMILES, not the caller's: it is
    the string upstream will tokenise, so it is the string worth writing.
    """
    from rdkit import Chem

    if Chem.MolFromSmiles(smiles) is None:
        return None, "unparseable"
    standardized = standardize_smiles(smiles)
    if standardized is None:
        return None, "standardizer"
    mol = Chem.MolFromSmiles(standardized)
    if mol is None:  # pragma: no cover - standardize_smiles round-trips its own output
        return None, "standardizer"
    if not _in_vocabulary(standardized, tokens):
        return None, "vocabulary"
    variants = Chem.MolToRandomSmilesVect(
        mol, VOCAB_RANDOMISATIONS, randomSeed=VOCAB_RANDOM_SEED, isomericSmiles=STANDARDIZE_ISOMERIC
    )
    if any(not _in_vocabulary(variant, tokens) for variant in variants):
        return None, "vocabulary"
    return standardized, None


def admissible_smiles(smiles: str, tokens: set[str]) -> str | None:
    """The SMILES to write, or ``None`` if upstream's read pipeline would lose it.

    The one-answer form of :func:`classify_smiles`, kept because callers that only
    need "would this molecule survive" — the REINVENT4 training probe among them —
    should not have to unpack a reason they will discard.
    """
    return classify_smiles(smiles, tokens)[0]


# ---------------------------------------------------------------------------
# The offer records
# ---------------------------------------------------------------------------
def _offer(inchikey: str, rows: Sequence[Mapping[str, Any]], family_of: Mapping[str, Any]) -> Record:
    """One acquisition record, shaped exactly as ``calibrate_zak_benchmark`` shapes it.

    Every identifier the registry can match on is present — molecule id,
    connectivity key, structural family and the full alias set — because matching
    on a subset is how a protected compound arrives under a foreign database's id
    and is admitted.
    """
    return {
        "molecule_id": str(rows[0]["molecule_id"]),
        "connectivity_key": rows[0].get("connectivity_key"),
        "family_id": family_of.get(inchikey),
        "aliases": sorted({str(r["molecule_id"]) for r in rows} | {inchikey}),
        "inchikey": inchikey,
    }


def build_training_smiles(
    *,
    partition: str,
    exclusion_registry: Path,
    out_path: Path,
    vocabulary_from: Path,
) -> Record:
    """Write the SMILES file a REINVENT4 TL run may train on, and say what it cost.

    Args:
        partition: the split partition to draw from, e.g. ``"train"``. The
            held-out partition is refused outright.
        exclusion_registry: path to Step 6's protected exclusion registry. No
            default, on purpose — see the module docstring.
        out_path: where the tab-delimited, header-less SMILES file is written.
        vocabulary_from: the prior whose vocabulary the compounds must satisfy,
            or a ``.json`` file carrying the same token set.

    Returns:
        ``path``, ``n_offered``, ``n_admitted``, ``n_quarantined``,
        ``n_held_out_offered``, ``n_held_out_admitted``, ``n_vocab_filtered``,
        ``sha256``, and the reconciliation added for the shrinkage finding:
        ``n_unparseable``, ``n_standardizer_filtered``, ``n_filtered``,
        ``n_duplicates_collapsed``, ``duplicate_collapse_groups``,
        ``n_written``, ``n_filtered_by_reason`` and ``reconciliation``.

        ``n_offered``/``n_admitted``/``n_quarantined`` describe the partition
        offer. ``n_held_out_offered``/``n_held_out_admitted`` are the positive
        control: the held-out compounds offered to the same registry, which must
        come back wholly quarantined.

        The rest account for every admitted compound that did *not* reach the
        file, by the stage of upstream's read pipeline that would have removed
        it — see :func:`classify_smiles`. ``n_unparseable`` is RDKit's refusal,
        ``n_standardizer_filtered`` is the silent class that motivated all of
        this, and ``n_vocab_filtered`` is the loud one that would have aborted
        the run. ``n_filtered`` is their sum.

        ``n_duplicates_collapsed`` is a different kind of number and is kept
        under its own name for that reason: those compounds were not refused,
        they were *merged*. Each is a compound record whose standardised
        structure another admitted record had already produced — the module
        docstring's four classes — and writing both would hand REINVENT4 a file
        whose row count exceeds the number of molecules in it, because
        ``read_smiles_csv_file`` is called with ``remove_duplicates=True``.
        ``duplicate_collapse_groups`` names each merge: the standardised SMILES,
        the compound whose row was kept, and the compounds absorbed into it, so
        the merge is attributable rather than merely counted. The identity

            ``n_written == n_admitted - n_filtered - n_duplicates_collapsed``

        holds exactly, is asserted before the file is written, and is spelled out
        in the ``reconciliation`` string so a reader does not have to do the
        subtraction to see that it balances. ``Reinvent4Adapter.fit`` then
        completes the chain by refusing to return unless REINVENT4's own
        ``n_training_smiles`` equals the row count of this file; between the two
        checks, a training set cannot shrink without somebody being told.

        After the merge, ``remove_duplicates`` can collapse nothing: two rows of
        this file are two different standardised structures, and a randomised
        rewriting of one molecule is never a valid SMILES of another. The
        shrinkage guard is not relaxed anywhere — it is the check that found
        this, and what changed is the writer that disagreed with it.

    Raises:
        ValueError: the partition is the held-out one, or is not in the split.
        RuntimeError: a held-out compound was admitted anywhere, the positive
            control did not refuse every held-out compound, the filter counts do
            not reconcile against the admitted count, or nothing survived to be
            written.
    """
    from .splits import screen_records

    partition = str(partition)
    if partition == HELD_OUT_PARTITION:
        raise ValueError(
            f"{partition!r} is the held-out partition; it is not training data for any "
            "arm, and a transfer-learning run that read it would contaminate every "
            "downstream comparison"
        )

    registry_path = Path(exclusion_registry)
    registry = _json(registry_path)
    rows = _json(DERIVED / "measurements.json")
    membership = _json(PROTECTED / "split_membership.json")["membership"]
    family_of = _json(PROTECTED / "split_membership.json")["family_of"]

    known = {str(v) for v in membership.values()}
    if partition not in known:
        raise ValueError(f"{partition!r} is not a partition of the frozen split; it names {sorted(known)}")

    by_compound: dict[str, list[Record]] = {}
    for row in rows:
        by_compound.setdefault(str(row["inchikey"]), []).append(dict(row))

    held_out_keys = {k for k, v in membership.items() if str(v) == HELD_OUT_PARTITION}

    offered = [
        _offer(key, rs, family_of)
        for key, rs in sorted(by_compound.items())
        if str(membership.get(key)) == partition
    ]
    screen = screen_records(offered, registry)
    admitted_keys = {str(r["inchikey"]) for r in screen["admitted"]}

    # The positive control calibrate_zak_benchmark.py runs: the same boundary,
    # offered the compounds it exists to refuse. A boundary that has never been
    # shown a protected compound has not been measured.
    control_offered = [
        _offer(key, rs, family_of) for key, rs in sorted(by_compound.items()) if key in held_out_keys
    ]
    control = (
        screen_records(control_offered, registry) if control_offered else {"admitted": [], "quarantined": []}
    )
    control_admitted = {str(r["inchikey"]) for r in control["admitted"]}

    leaked = sorted((admitted_keys & held_out_keys) | control_admitted)
    n_held_out_admitted = len(leaked)
    if n_held_out_admitted:
        raise RuntimeError(
            f"{n_held_out_admitted} held-out compound(s) were admitted by the registry at "
            f"{registry_path}, so this training file would contaminate the held-out "
            f"partition. Offending keys: {leaked[:5]}" + (" ..." if len(leaked) > 5 else "")
        )
    if control_offered and len(control["quarantined"]) != len(control_offered):
        raise RuntimeError(
            f"the exclusion boundary quarantined {len(control['quarantined'])} of "
            f"{len(control_offered)} held-out compounds offered as a positive control; it "
            "must refuse every one of them, and a boundary that refuses only some is not a "
            "boundary"
        )

    tokens = prior_vocabulary_tokens(vocabulary_from)

    written: list[str] = []
    filtered: dict[str, int] = dict.fromkeys(FILTER_REASONS, 0)
    # The de-duplication ``read_smiles_csv_file`` performs at read time, performed
    # here instead, on the same canonical form it collapses to: the standardised
    # SMILES, which is what this module writes and what upstream's standardiser
    # reproduces from it. Two compounds sharing that string are one training
    # molecule however either of them is later randomised; two compounds not
    # sharing it can never become one, because a randomised rewriting is always
    # a rewriting of its own molecule. So after this loop the file cannot shrink
    # at read time, rather than merely being unlikely to.
    kept_by_smiles: dict[str, str] = {}
    collapsed: dict[str, list[str]] = {}
    for key in sorted(admitted_keys):
        raw = str(by_compound[key][0]["molecule"]["standard_smiles"])
        standardized, reason = classify_smiles(raw, tokens)
        if standardized is None:
            filtered[str(reason)] += 1
            continue
        if standardized in kept_by_smiles:
            collapsed.setdefault(standardized, []).append(key)
            continue
        kept_by_smiles[standardized] = key
        written.append(standardized)

    n_filtered = sum(filtered.values())
    n_collapsed = sum(len(keys) for keys in collapsed.values())
    collapse_groups = [
        {
            "standardized_smiles": smiles,
            "kept": kept_by_smiles[smiles],
            "absorbed": sorted(keys),
        }
        for smiles, keys in sorted(collapsed.items())
    ]
    # By construction, and asserted anyway. Every admitted compound is either
    # written, counted under exactly one refusal reason, or merged into a row
    # already written; an edit that broke that would otherwise reintroduce the
    # silent shrinkage this whole function guards against, one stage earlier and
    # just as quietly.
    if len(written) + n_filtered + n_collapsed != len(admitted_keys):
        raise RuntimeError(
            f"the training-set counts do not reconcile: {len(admitted_keys)} admitted, "
            f"{len(written)} written, {n_filtered} filtered ({filtered}), {n_collapsed} "
            f"merged into an earlier row. Every admitted compound must be written, counted "
            f"under exactly one reason, or attributed to the row it collapsed onto, or the "
            f"record this returns cannot account for what the model trained on"
        )

    if not written:
        raise RuntimeError(
            f"no compound of partition {partition!r} survived the exclusion boundary, "
            f"upstream's standardiser and the vocabulary of {Path(vocabulary_from).name}; "
            f"there is nothing to train on ({filtered})"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Tab-delimited, no header, SMILES in column 0 — read_smiles_csv_file's own
    # defaults. A header here would be trained on as if it were a molecule.
    out.write_text("".join(f"{smiles}\n" for smiles in written))

    return {
        "path": str(out.resolve()),
        "n_offered": len(offered),
        "n_admitted": len(admitted_keys),
        "n_quarantined": len(screen["quarantined"]),
        "n_held_out_offered": len(control_offered),
        "n_held_out_admitted": n_held_out_admitted,
        "n_unparseable": filtered["unparseable"],
        "n_standardizer_filtered": filtered["standardizer"],
        "n_vocab_filtered": filtered["vocabulary"],
        "n_filtered": n_filtered,
        "n_filtered_by_reason": dict(filtered),
        "n_duplicates_collapsed": n_collapsed,
        "duplicate_collapse_groups": collapse_groups,
        "n_written": len(written),
        "reconciliation": (
            f"n_admitted({len(admitted_keys)}) - unparseable({filtered['unparseable']}) "
            f"- standardizer({filtered['standardizer']}) - vocabulary({filtered['vocabulary']}) "
            f"- duplicates_collapsed({n_collapsed}) = n_written({len(written)})"
        ),
        "sha256": sha256_file(out),
    }


__all__ = [
    "DEFAULT_ELEMENTS",
    "FILTER_REASONS",
    "HELD_OUT_PARTITION",
    "MAX_HEAVY_ATOMS",
    "MIN_HEAVY_ATOMS",
    "NEUTRALIZE_PATTERNS",
    "SMILES_TOKENS_REGEX",
    "STANDARDIZE_ISOMERIC",
    "UNWANTED_PATTERNS",
    "VOCAB_RANDOMISATIONS",
    "admissible_smiles",
    "build_training_smiles",
    "classify_smiles",
    "prior_vocabulary_tokens",
    "standardize_mol",
    "standardize_smiles",
]
