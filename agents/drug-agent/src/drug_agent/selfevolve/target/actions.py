"""The bodies of the interventions the dispatcher can actually cause.

Step 10 of the per-target evolution program. :mod:`.worker` decides *whether*
an experiment may run — schema, campaign authorisation, backend availability,
protected paths, budget — and then launches this module as an argument vector
inside the candidate's own checkout. Everything here is the work itself.

Two rules shape the whole file.

**An advertised action executes or reports a precise capability failure.** There
is no handler that raises on entry to stand in for one that was never written.
``adapt_generator`` is the worked example: it *asks* — one verb at a time,
through Step 8's probes — before it trains anything, and a host whose generator
exposes no training verb gets a structured ``UNAVAILABLE`` naming the missing
verb, the reason and what would unblock it. That is a result, not a crash, and
the proposal was accepted before it was answered. When the verb is there the
same handler runs the real fine-tuning loop and emits a WEIGHTS artifact; the
probe stayed a precondition rather than becoming a comment.

**A generated molecule is counted once, under one name.** Every generation path
here writes ``n_generated`` into ``funnel.json``. Across the first sixteen
campaigns the same quantity was recorded as ``n_raw``, ``n_raw_attempts``,
``n_candidates``, ``n_scored`` or not at all, and an analysis divided one
campaign's count by a generated denominator and another's by a scored one and
compared the two. ``n_raw`` is kept beside it, unchanged, so the older records
stay readable; ``n_generated`` is the key anything new reads.

**A handler never grades itself.** Every Track A handler writes
``predictions.json`` and stops. The development number is computed afterwards,
in the dispatcher's process, by the parent checkout's frozen evaluator, over the
frozen development pool. A handler that reported its own MAE would be asking to
be believed; this way there is nothing to believe.

Nothing here reads the held-out partition. The fit pool, the evaluation pool and
the permitted library all come from
:func:`drug_agent.selfevolve.target.baseline_arms.prepare_track_a`, which offers
every compound to Step 6's protected registry first and reports the leak count
as a measurement on every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from drug_agent.selfevolve.target import submission_policy
from drug_agent.selfevolve.repository import RepositoryLayout

Record = dict[str, Any]

#: What a handler may say about itself. ``SUCCEEDED`` is a claim the dispatcher
#: then has to verify against files on disk; ``UNAVAILABLE`` is the honest
#: answer when a backend this host does not have was asked for.
HANDLER_STATUSES: tuple[str, ...] = ("SUCCEEDED", "FAILED", "UNAVAILABLE")

#: Concentration of the one thing every handler needs and none of them may
#: choose for itself: where the campaign's data lives. Passed as an argument
#: vector element by the dispatcher, never discovered from ``__file__``, because
#: the candidate checkout is a git worktree and the campaign's snapshots are not
#: in git.
SNAPSHOT_RELATIVE = Path(".external") / "research" / "zak-c9r"

#: The exclusion boundary every training set is offered to, **fixed here and
#: deliberately not read out of ``inputs``**.
#:
#: Step 6's registry is what turns "we did not train on the held-out partition"
#: from an assertion into a measurement. A plan that could name its own registry
#: could name an empty one, and the boundary report would then be a count of
#: nothing with the same shape as a count of everything. The partition, the
#: epochs and the batch size are the proposer's to choose; which registry
#: guards the split is not.
PROTECTED_REGISTRY_RELATIVE = SNAPSHOT_RELATIVE / "protected" / "protected_exclusion_registry.json"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def _cost(started: float, **extra: Any) -> Record:
    import resource

    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_seconds": round(time.time() - started, 3),
        "cpu_user_seconds": round(me.ru_utime + kids.ru_utime, 3),
        "cpu_system_seconds": round(me.ru_stime + kids.ru_stime, 3),
        "max_rss_kb": int(max(me.ru_maxrss, kids.ru_maxrss)),
        "gpu_seconds": None,
        "gpu_telemetry": "UNAVAILABLE",
        **extra,
    }


#: Per action, the ``inputs`` keys whose contract shape is a JSON **object**.
#:
#: Declared in one place because the proposer is a language model and a model
#: that writes ``"teacher": "structure.dock.vina"`` where the contract wants
#: ``{"endpoint": ..., "rows": ...}`` is making a proposal, not an attack. Step
#: 11's third real campaign lost both its cycles to exactly that: the handler
#: coerced with ``dict(...)``, a bare string raised ``ValueError: dictionary
#: update sequence element #0 has length 1; 2 is required``, and the structured
#: refusal the handler already contained was never reached.
#:
#: ``tests/research/test_experiment_plans.py`` parametrises over this table and
#: separately greps the source for ``dict(inputs.get("...")``, so an eighth
#: coerced key cannot be added without being declared here.
MAPPING_INPUTS: dict[str, tuple[str, ...]] = {
    "fit_predictor": ("preparation",),
    "evaluate_development": ("preparation",),
    "adapt_generator": ("training",),
    "change_sampling": ("sampling",),
    "distill": ("teacher",),
    "acquire_resource": ("resource",),
    "prepare_receptor": ("receptor",),
}


#: What to do when a ``distill`` plan names a teacher table that is not there.
#: Step 11's campaign C lost both its cycles to this and the agent had no way to
#: know, because nothing told it that teacher rows are produced by a *different*
#: computation from the one that learns from them.
_TEACHER_PASS_UNBLOCK = (
    "produce the teacher table first, as its own artifact: "
    "`python -m drug_agent.selfevolve.target.distill --stage select --experiment-id <id>` then "
    "`--stage teacher --experiment-id <id> --ledger <ledger.sqlite> --dock-call-cap <n>`, then "
    "pass inputs.teacher = {'endpoint': <the endpoint name>, 'table': "
    "'.external/research/zak-c9r/derived/distill/<id>/teacher_table.json'}. "
    "Scoring and learning are two computations on purpose: a student whose labels were made "
    "inside its own training call cannot say which computation produced them."
)


def _object_input(
    inputs: Mapping[str, Any],
    key: str,
    *,
    action: str,
    backend: str,
    started: float,
) -> tuple[Record, Record | None]:
    """``(the mapping, a refusal)``: the refusal is ``None`` when the shape is right.

    Validated rather than normalised. Guessing that ``"teacher": "vina"`` means
    ``{"endpoint": "vina"}`` would be a silent rename of the endpoint a
    distilled student claims to have learned, and Appendix A §6 requires that
    endpoint to be *named*, never inferred. A wrong shape is reported back to
    the proposer with the field and the type it actually sent, which is
    something it can act on in the next cycle.
    """
    raw = inputs.get(key)
    if raw is None or isinstance(raw, Mapping):
        return dict(raw or {}), None
    refusal = _result(
        action,
        backend,
        "FAILED",
        notes=f"inputs[{key!r}] was not an object",
        reason=(
            f"inputs[{key!r}] must be a JSON object, and this plan sent a "
            f"{type(raw).__name__}. The handler cannot read the named fields out of a "
            f"scalar or a list, and inventing them would put values into the artifact "
            f"record that nobody proposed"
        ),
        unblock=(
            f"resend inputs[{key!r}] as an object whose keys are the ones named in this "
            f"action's inputs_contract"
        ),
        cost=_cost(started),
    )
    return {}, refusal


def _resolve(repo_root: Path, value: str) -> Path:
    """A plan-supplied path, resolved against the repository root when relative."""
    path = Path(str(value))
    return path if path.is_absolute() else (repo_root / path)


#: The bases a *relative* ``inputs.sampling.checkpoint`` is tried against, in
#: order, and the name each one is reported under.
#:
#: The defect this order answers: ``adapt_generator`` records its checkpoint two
#: ways — ``evidence.adaptation.checkpoint`` (absolute) and
#: ``checkpoint_relative_to_output_dir`` (``generator/<backend>_tl.prior``) — and
#: an agent that also knows the writing experiment's id reasonably joins the two
#: into ``<experiment_id>/generator/<backend>_tl.prior``. That third form is
#: relative to the campaign's *experiments* directory, which nothing documented
#: as a base, so it was resolved against the repository root, hit nothing, and
#: the run was refused. Each of these four is a directory this repository really
#: does address, so trying them in a declared order — and recording which one
#: answered — removes a whole class of failure the agent could not see.
#:
#: Most-specific first, deliberately: a name that resolves inside the
#: experiment's own output directory means *that* experiment's file and must not
#: be shadowed by a same-named file one level up.
#:
#: This is a resolution order, not a fall-back. Nothing here reaches for the
#: parent prior: when no base answers, the caller still refuses.
_CHECKPOINT_BASE_NAMES: tuple[str, str, str, str] = (
    "the experiment's own output directory",
    "the campaign's experiments root",
    "the campaign root",
    "the repository root",
)


def _checkpoint_bases(output_dir: Path, repo_root: Path) -> tuple[tuple[str, Path], ...]:
    """The four bases paired with their names, derived from *output_dir*.

    ``controller`` lays an experiment out as
    ``<campaign root>/experiments/<experiment_id>``, so the two directories above
    the output directory are the experiments root and the campaign root. They are
    derived rather than passed because a handler is given only its own
    ``output_dir`` and the repository root.
    """
    out = Path(output_dir).absolute()
    bases = (out, out.parent, out.parent.parent, Path(repo_root).absolute())
    return tuple(zip(_CHECKPOINT_BASE_NAMES, bases, strict=True))


def _resolve_checkpoint(
    value: str, *, output_dir: Path, repo_root: Path
) -> tuple[Path | None, str, list[tuple[str, Path]]]:
    """Resolve a declared checkpoint against each base in turn, and say how.

    Returns ``(path, base_name, attempts)``. *path* is ``None`` when no base
    held the file — never a substitute and never the parent prior. *attempts*
    names every base tried with the absolute path it formed, so the refusal can
    print the search it actually performed instead of leaving the next reader to
    reconstruct it.
    """
    declared = Path(str(value))
    if declared.is_absolute():
        attempts = [("the absolute path as given", declared)]
        return (declared if declared.is_file() else None), "the absolute path as given", attempts
    attempts = []
    for name, base in _checkpoint_bases(output_dir, repo_root):
        candidate = base / declared
        attempts.append((name, candidate))
        if candidate.is_file():
            return candidate, name, attempts
    return None, "", attempts


def _hexdigest(value: Any) -> str:
    """The hex half of a digest, however it was spelled.

    This repository writes ``"sha256:<hex>"`` and
    :meth:`Reinvent4Adapter._sha256` returns the bare hex, so two digests of the
    same bytes compare unequal as strings. Comparing the hex is the difference
    between detecting a checkpoint fall-back and reporting one that did not
    happen.
    """
    return str(value).split(":")[-1].strip().lower()


def _sha256_path(path: Path) -> str:
    """``sha256:`` over a file's bytes, read in chunks.

    :func:`sha256_bytes` is fine for a JSON record. A generator checkpoint is
    not one: the released REINVENT4 prior is 23,528,840 bytes and a fine-tuned
    one is the same size, so the hash of a checkpoint is streamed.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _adapter_sha256(adapter: Any, path: Path) -> str:
    """A checkpoint's digest, through the adapter's own cache when it has one.

    ``Reinvent4Adapter._sha256`` memoises on ``(path, size, mtime_ns)``, so
    routing through it hashes a 23 MB prior once per process instead of once per
    sampling call, and re-hashes a checkpoint a training run rewrote in place.
    """
    hasher = getattr(adapter, "_sha256", None)
    if callable(hasher):
        try:
            return f"sha256:{_hexdigest(hasher(Path(path)))}"
        except Exception:  # an adapter with a differently-shaped helper
            pass
    return _sha256_path(Path(path))


# ---------------------------------------------------------------------------
# Run-scoped constraints
# ---------------------------------------------------------------------------
#: The file ``scripts/research/run_campaign.py`` writes into a campaign's output
#: root, carrying the constraints every action dispatched under that run must
#: obey.
#:
#: A file, rather than an argument or an environment variable, because neither
#: of those reaches a handler. :func:`~.worker.dispatch_plan` has a closed
#: keyword signature and :func:`~.worker._handler_argv` builds a fixed argument
#: vector, so a new dispatch keyword would have to change the kernel;
#: :func:`~.worker.candidate_env` passes a candidate an **allowlist** of
#: variable names, so an unlisted ``DRUGAGENT_*`` is dropped rather than
#: inherited. The one channel that already exists is the directory the
#: dispatcher hands every handler: ``controller._run_cycle`` sets
#: ``output_dir = <campaign output root>/experiments/<experiment_id>``, so the
#: campaign root is two levels up from where a handler is already writing.
RUN_CONSTRAINTS_FILENAME = "run_constraints.json"

#: How far above ``output_dir`` to look. Exactly two levels, which is exactly the
#: campaign root: ``<campaign output root>/experiments/<experiment_id>`` is where
#: the controller puts a handler, so ``parents[1]`` is the campaign root and
#: ``parents[2]`` is already **outside the run**.
#:
#: It was 4 — slack for an operator nesting the output directory more deeply —
#: and that slack is a scoping hole, not slack. ``run_campaign_series.py`` writes
#: ``<series_out>/<campaign>/experiments/<experiment_id>``, so ``parents[2]`` is
#: the *shared series directory*: one ``run_constraints.json`` dropped there, by
#: an operator or by another campaign, silently bound every campaign in the
#: series, and ``declare_run_constraints`` never saw it because that function
#: only inspects the campaign's own root — so an inherited ceiling bypassed the
#: one-way narrowing check entirely.
_RUN_CONSTRAINTS_MAX_DEPTH = 2


def run_constraints(output_dir: Path, *, campaign_id: str | None = None) -> Record:
    """The run-scoped constraints for this dispatch.

    Five answers, and they are deliberately five:

    * ``{}`` — no constraints file in this experiment directory or the campaign
      root above it. That is what every campaign run before this file existed
      had, and every path below is byte-identical in that case. *Absent is
      unconstrained, not zero.*
    * a mapping — the constraints, plus ``source``, the file the binding ceiling
      came from, and ``sources``, every file the walk found.
    * ``{"unreadable": True, ...}`` — a constraints file exists and did not
      parse. This is **not** treated as absent: silently ignoring a ceiling
      because the file holding it was corrupt is the exact failure this
      mechanism exists to prevent, so the caller refuses the run instead.
    * ``{"foreign": True, ...}`` — a constraints file exists and **cannot be
      shown to belong to this dispatch**. See *ownership* below.
    * ``{"widening_conflict": True, ...}`` — two constraints files were found and
      one declares a looser ceiling than the other. See *the ceiling* below.

    **The ceiling is the minimum over every file found, never the nearest one.**
    The walk starts at the experiment directory, so the experiment directory is
    the first candidate, and returning the first file found made that directory a
    widening path: a campaign root declaring ``max_generated=300`` was overridden
    by an experiment-dir file declaring ``max_generated=100000`` under the same
    campaign name, and the effective cap became 100,000 sourced from the
    experiment-dir file. ``declare_run_constraints`` writes only the campaign
    root and never inspects the experiment directories under it, so its one-way
    narrowing guarantee could not see that file and did not cover it. Taking the
    minimum makes the guarantee hold here instead of there: a ceiling may tighten
    from any source and can never exceed the campaign root's, whatever is
    dropped below it. A file that declares *no* ceiling contributes nothing to
    the minimum rather than widening it to unconstrained.

    A file that declares a **looser** ceiling than another is refused, not
    quietly clamped to the tighter one. The minimum would make such a run safe to
    execute, so the refusal is not about the number — it is that the file has
    exactly two readings and both need a human. Either somebody meant to raise
    this run's ceiling, which is the one operation this control exists to forbid
    and which must not succeed quietly at 300 while a file in the tree says
    100,000; or the file is stale, left over from another configuration, in which
    case the ceiling it names is evidence about nothing and should be removed.
    Clamping silently would normalise a widening file living in the run
    directory. ``max_generated`` in the returned record is still the minimum, so
    a caller that reports the refusal reports the safe number alongside it.

    **Ownership is proved, not assumed.** A file is obeyed only when it declares
    a campaign id *and* the dispatch declares one *and* the two match. The
    comparison used to be skipped entirely when the dispatch supplied no
    ``campaign_id`` — and ``change_sampling`` coerced a missing plan field to
    ``None`` — so a dispatch carrying no campaign id turned the foreign check
    off and obeyed a campaign-root file naming ``SOMEBODY_ELSE`` at cap 400. A
    missing campaign id on the dispatch is not a weaker version of a match; it is
    precisely the case where the file cannot be shown to belong to this run, so
    it is the case that must refuse. The production controller always sets
    ``campaign_id``, which made that hole latent rather than live, and latent is
    not a reason to leave a check that disables itself on a missing field.
    """
    here = Path(output_dir).resolve()
    dispatched = str(campaign_id or "").strip()
    # Nearest-first, exactly as the walk finds them. The ORDER is retained for
    # reporting only -- nothing below picks a file by its position.
    found: list[tuple[Path, Mapping[str, Any]]] = []
    for candidate in [here, *list(here.parents)[:_RUN_CONSTRAINTS_MAX_DEPTH]]:
        path = candidate / RUN_CONSTRAINTS_FILENAME
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"source": str(path), "unreadable": True, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(loaded, Mapping):
            return {
                "source": str(path),
                "unreadable": True,
                "error": f"the constraints record is a {type(loaded).__name__}, not an object",
            }
        declared_raw = loaded.get("campaign_id")
        declared = str(declared_raw or "").strip()
        if not dispatched:
            why = (
                f"the constraints at {path} declare campaign_id "
                f"{(repr(declared) if declared else '(absent)')} and this dispatch carries no "
                f"campaign_id at all, so there is nothing to match the file against and no way "
                f"to show it was declared for this run"
            )
        elif not declared:
            why = (
                f"the constraints at {path} declare no campaign_id and this dispatch belongs to "
                f"campaign {dispatched!r}"
            )
        elif declared != dispatched:
            why = (
                f"the constraints at {path} declare campaign_id {declared!r} and this dispatch "
                f"belongs to campaign {dispatched!r}"
            )
        else:
            found.append((path, loaded))
            continue
        return {
            "source": str(path),
            "foreign": True,
            "declared_campaign_id": None if declared_raw is None else str(declared_raw),
            "dispatched_campaign_id": dispatched or None,
            "error": why,
        }
    if not found:
        return {}

    # --- the ceiling: the MINIMUM over every file found ---------------------
    ceilings: list[tuple[Path, int]] = []
    for path, loaded in found:
        raw = loaded.get("max_generated")
        if raw is None:  # this file declares no ceiling; it narrows nothing
            continue
        try:
            ceilings.append((path, int(raw)))
        except (TypeError, ValueError):
            # Not a usable count. Hand the offending value straight through so
            # :func:`generation_cap` produces the refusal, rather than growing a
            # second definition of "a ceiling that can be enforced" here -- and
            # hand it through even when some other file's ceiling is smaller,
            # because a ceiling nobody can read is not a constraint that a
            # readable one elsewhere excuses.
            return {
                **dict(found[-1][1]),
                "max_generated": raw,
                "source": str(path),
                "sources": [str(p) for p, _ in found],
            }

    # The OUTERMOST file is the base record: it is the campaign root's, the one
    # ``declare_run_constraints`` writes and guards. Its non-ceiling fields are
    # not overridable from a directory nested inside the run.
    record: Record = {**dict(found[-1][1]), "sources": [str(p) for p, _ in found]}
    if not ceilings:
        record["max_generated"] = None
        record["source"] = str(found[-1][0])
        return record
    # ``min`` returns the FIRST minimal element, and ``reversed`` walks
    # outermost-first, so two files agreeing on the tightest ceiling are
    # attributed to the campaign root rather than to the file nested under it.
    outward_in = list(reversed(ceilings))
    binding_path, binding = min(outward_in, key=lambda item: item[1])
    record["max_generated"] = binding
    record["source"] = str(binding_path)
    record["declared_ceilings"] = [{"source": str(p), "max_generated": v} for p, v in ceilings]
    # WHICH direction is the widening one is decided by position, not by the
    # minimum. Walking outward-in from the campaign root, the ceiling may only
    # fall: a file nested inside the run declaring a TIGHTER ceiling than the
    # root is the one-way narrowing this control is for, and is not a conflict.
    # A file declaring a ceiling LOOSER than one declared closer to the campaign
    # root is the widening, whether or not it ends up binding.
    widening: list[tuple[Path, int, Path, int]] = []
    tightest_so_far: tuple[Path, int] | None = None
    for path, value in outward_in:
        if tightest_so_far is not None and value > tightest_so_far[1]:
            widening.append((path, value, tightest_so_far[0], tightest_so_far[1]))
        if tightest_so_far is None or value < tightest_so_far[1]:
            tightest_so_far = (path, value)
    if widening:
        record["widening_conflict"] = True
        record["error"] = (
            "the run constraints found on the walk disagree, and they disagree in the widening "
            "direction: "
            + "; ".join(
                f"{p} declares max_generated={v}, above the max_generated={outer_v} declared "
                f"closer to the campaign root at {outer_p}"
                for p, v, outer_p, outer_v in widening
            )
            + f". The effective ceiling is {binding} and can never be more, but a file declaring "
            f"a looser ceiling than one above it either means somebody is raising this run's "
            f"ceiling -- the operation this control forbids -- or is stale and names a ceiling "
            f"that is evidence about nothing"
        )
    return record


def generation_cap(constraints: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """``(the ceiling on molecules generated, a problem)``.

    ``(None, None)`` means no ceiling was declared, which is not a ceiling of
    zero and not a ceiling of infinity — it is the unconstrained run the first
    sixteen campaigns were. A declared value that is not a usable count is a
    *problem*, never a quietly dropped constraint.
    """
    raw = constraints.get("max_generated")
    if raw is None:
        return None, None
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        return None, (
            f"max_generated is {raw!r}, a {type(raw).__name__}, and a ceiling that cannot be "
            f"read as a whole number of molecules cannot be enforced"
        )
    if cap <= 0:
        return None, (
            f"max_generated is {cap}; a ceiling of zero or less would forbid the generation it "
            f"is supposed to bound, which is a configuration error rather than a control"
        )
    return cap, None


def _result(
    action: str,
    backend: str,
    status: str,
    *,
    notes: str,
    artifacts: list[Record] | None = None,
    evidence: Mapping[str, Any] | None = None,
    reason: str | None = None,
    unblock: str | None = None,
    cost: Mapping[str, Any] | None = None,
) -> Record:
    if status not in HANDLER_STATUSES:  # pragma: no cover - programming error
        raise ValueError(f"{status!r} is not one of {HANDLER_STATUSES}")
    return {
        "contract_version": "1.0.0",
        "record_kind": "experiment_handler_result",
        "action": action,
        "backend": backend,
        "status": status,
        "notes": notes,
        "reason": reason,
        "unblock": unblock,
        "artifacts": list(artifacts or []),
        "evidence": dict(evidence or {}),
        "cost": dict(cost or {}),
        "scored_by": (
            "nothing here. The development number is computed by the parent checkout's "
            "frozen evaluator, from the files above, after this process exits."
        ),
    }


# ---------------------------------------------------------------------------
# Track A — fit a predictor, write predictions, score nothing
# ---------------------------------------------------------------------------
def _track_a_pool(preparation: Mapping[str, Any]) -> Record:
    from .baseline_arms import prepare_track_a

    return prepare_track_a(dict(preparation))


def fit_predictor(plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path) -> Record:
    """Fit a Track A activity predictor and predict the development pool.

    The fit pool is the admitted *train* partition of the frozen primary
    stratum and nothing else. Both backends write the identical
    ``predictions.json`` shape — ``{molecule_id: endpoint_result}`` — so the
    evaluator downstream cannot tell them apart, which is the point: two arms
    that are scored by the same code over the same pool are comparable, and two
    that each report their own number are not.
    """
    from drug_agent.selfevolve.target.api import endpoint_result

    started = time.time()
    inputs = dict(plan["inputs"])
    preparation, _refusal = _object_input(
        inputs, "preparation", action="fit_predictor", backend=backend, started=started
    )
    if _refusal is not None:
        return _refusal
    prepared = _track_a_pool(preparation)
    seed = int(preparation.get("seed", 42))
    endpoint_label = str(preparation.get("endpoint_label", "pKi"))

    train_keys = sorted(prepared["train"])
    train_smiles = [str(prepared["train"][k]["molecule"]["standard_smiles"]) for k in train_keys]
    train_y = [float(prepared["train"][k]["p_value"]) for k in train_keys]
    pool = list(prepared["pool"])
    pool_smiles = [str(prepared["evaluation"][k]["molecule"]["standard_smiles"]) for k in pool]

    predictions: dict[str, Any] = {}
    model_record: Record
    weights: list[str] = []

    if backend == "sklearn_rf":
        import joblib

        from drug_agent.tools.sklearn_tool import BaselineActivityModel

        max_depth = preparation.get("max_depth", 12)
        model = BaselineActivityModel(
            endpoint=endpoint_label,
            n_estimators=int(preparation.get("n_estimators", 200)),
            max_depth=None if max_depth is None else int(max_depth),
            seed=seed,
        )
        model.fit(train_smiles, train_y)
        for key, smiles in zip(pool, pool_smiles, strict=True):
            try:
                out = model.predict_one(smiles)
            except Exception as exc:  # a refusal is a failure with a reason, never a number
                print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                predictions[key] = endpoint_result("FAILED", None)
                continue
            predictions[key] = endpoint_result("SUCCEEDED", float(out.value))
        checkpoint = output_dir / "model.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # The fitted estimator and its state, not the wrapper: the wrapper holds
        # an RDKit ``Uncharger``, which Boost.Python refuses to pickle. The
        # payload below is everything needed to rebuild the wrapper and predict,
        # so the checkpoint is a loadable model rather than a file that happens
        # to exist. :func:`load_fitted_predictor` is the other half, and the
        # real end-to-end acceptance test loads the checkpoint back and predicts
        # with it rather than assuming the round trip works.
        joblib.dump(
            {
                "format": "drug_agent.selfevolve.target.actions/baseline-rf/1",
                "capability_id": BaselineActivityModel.capability_id,
                "model_version": model.model_version,
                "endpoint": model.endpoint,
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "seed": model.seed,
                "estimator": model._model,
                "train_fingerprints": model._train_fps,
                "y_mean": model._y_mean,
                "y_std": model._y_std,
            },
            checkpoint,
        )
        weights = ["model.joblib"]
        model_record = {
            "kind": "RandomForest on Morgan fingerprints",
            "backend": "sklearn_rf",
            "capability_id": BaselineActivityModel.capability_id,
            "model_version": model.model_version,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "seed": model.seed,
        }
    elif backend == "chemprop":
        from drug_agent.tools.properties.chemprop_tool import ChempropAdapter

        checkpoint_dir = output_dir / "chemprop"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter = ChempropAdapter(checkpoint_dir=checkpoint_dir)
        adapter.require_available()
        trained = adapter.train(
            endpoint=endpoint_label,
            smiles=train_smiles,
            y=train_y,
            epochs=int(preparation.get("epochs", 30)),
            val_fraction=float(preparation.get("val_fraction", 0.2)),
            batch_size=int(preparation.get("batch_size", 32)),
            seed=seed,
        )
        panels = adapter.predict(pool_smiles, endpoint=endpoint_label, candidate_ids=pool)
        for key, panel in zip(pool, panels, strict=True):
            values = [v for v in panel.endpoints.values() if isinstance(v, (int, float))]
            predictions[key] = (
                endpoint_result("SUCCEEDED", float(values[0]))
                if panel.ok and values
                else endpoint_result("FAILED", None)
            )
        path = Path(trained.path)
        if path.is_file():
            weights = [str(path.relative_to(output_dir))]
        model_record = {
            "kind": "Chemprop D-MPNN",
            "backend": "chemprop",
            "capability_id": adapter.capability_id,
            "model_version": adapter.version,
            "checkpoint": str(path),
            "n_train": trained.n_train,
            "n_val": trained.n_val,
            "epochs": int(preparation.get("epochs", 30)),
            "seed": seed,
        }
    else:  # pragma: no cover - validate_plan refuses an unregistered backend first
        raise SystemExit(f"fit_predictor has no backend named {backend!r}")

    _write(output_dir / "predictions.json", predictions)
    n_scored = sum(1 for v in predictions.values() if v["value"] is not None)
    return _result(
        "fit_predictor",
        backend,
        "SUCCEEDED",
        notes=(
            f"fitted on {len(train_keys)} admitted train compounds and predicted "
            f"{len(pool)} development compounds; {n_scored} scored"
        ),
        artifacts=[
            {
                "artifact_id": f"art_{plan['experiment_id']}_predictor",
                "kind": str(plan["kind"]),
                "parent_id": plan.get("parent_id"),
                "entrypoint": [
                    "python",
                    "-m",
                    "drug_agent.selfevolve.target.actions",
                    "--action",
                    "fit_predictor",
                    "--backend",
                    backend,
                ],
                "weights": weights,
            }
        ],
        evidence={
            "model": model_record,
            "n_train": len(train_keys),
            "n_pool": len(pool),
            "n_scored": n_scored,
            "stratum_id": prepared["stratum_id"],
            "split_hash": prepared["split_hash"],
            "boundary": prepared["boundary"],
            "evaluated_partition": "development",
        },
        cost=_cost(started),
    )


def load_fitted_predictor(path: Path | str) -> Any:
    """Rebuild the predictor a ``fit_predictor`` run persisted.

    The other half of the checkpoint written above. It exists so the artifact is
    a *callable interface* over persisted bytes rather than bytes alone — which
    is Appendix A §1's definition of an evolution candidate, and the property
    ``verify_artifact_references`` checks structurally and this function makes
    true in fact.
    """
    import joblib

    from drug_agent.tools.sklearn_tool import BaselineActivityModel

    payload = joblib.load(Path(path))
    model = BaselineActivityModel(
        endpoint=str(payload["endpoint"]),
        n_estimators=int(payload["n_estimators"]),
        max_depth=payload["max_depth"],
        seed=int(payload["seed"]),
    )
    model._model = payload["estimator"]
    model._train_fps = payload["train_fingerprints"]
    model._y_mean = float(payload["y_mean"])
    model._y_std = float(payload["y_std"])
    model._fitted = True
    return model


def evaluate_development(
    plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path
) -> Record:
    """Score an existing prediction file against the frozen development pool.

    The one action whose whole purpose is the evaluator. It copies the
    predictions it was given into the experiment's output directory so the
    dispatcher's own independent pass scores exactly the bytes this handler saw,
    and reports the endpoint it computed as *evidence* — the number that counts
    is still the one the dispatcher computes afterwards.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import (
        evaluator_hash,
        regression_endpoint,
    )

    started = time.time()
    inputs = dict(plan["inputs"])
    prepared_inputs, _refusal = _object_input(
        inputs, "preparation", action="evaluate_development", backend=backend, started=started
    )
    if _refusal is not None:
        return _refusal
    source = inputs.get("predictions")
    if not source:
        return _result(
            "evaluate_development",
            backend,
            "FAILED",
            notes="no predictions were named",
            reason="inputs['predictions'] must name the prediction file to score",
            unblock="run a fit_predictor experiment first and pass its predictions.json",
            cost=_cost(started),
        )
    path = _resolve(repo_root, str(source))
    if not path.is_file():
        return _result(
            "evaluate_development",
            backend,
            "FAILED",
            notes=f"{path} does not exist",
            reason=f"the named prediction file {str(source)!r} is not on disk",
            unblock="check the path; an absent file is not an empty prediction set",
            cost=_cost(started),
        )
    predictions = json.loads(path.read_text(encoding="utf-8"))
    prepared = _track_a_pool(prepared_inputs)
    endpoint = regression_endpoint(
        pool=list(prepared["pool"]),
        labels=dict(prepared["labels"]),
        predictions=dict(predictions),
        failure_penalty=float(prepared["failure_penalty"]),
    )
    _write(output_dir / "predictions.json", predictions)
    _write(output_dir / "endpoint.json", {**endpoint, "evaluator_hash": evaluator_hash()})
    return _result(
        "evaluate_development",
        backend,
        "SUCCEEDED",
        notes=f"scored {endpoint['n_scored']}/{endpoint['n_pool']} development compounds",
        artifacts=[],
        evidence={
            "evaluator_hash": evaluator_hash(),
            "n_pool": endpoint["n_pool"],
            "coverage": endpoint["coverage"],
            "source": str(path),
            "source_sha256": sha256_bytes(path.read_bytes()),
        },
        cost=_cost(started),
    )


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
#: The generator capability each ``adapt_generator`` backend trains, and the
#: verb Step 8 probes for it. One place, so the probe and the adapter cannot
#: come to disagree about which backend was asked.
_GENERATOR_CAPABILITY: dict[str, str] = {
    "pfm": "molecule.generate.pfm",
    "reinvent4": "molecule.generate.reinvent4",
}

#: The keyword arguments a training verb has to accept before this handler will
#: call it. Checked against the real signature rather than assumed: a ``fit``
#: that silently ignored ``input_model_file`` would fine-tune from a different
#: prior than the record says it did, and a ``fit`` that ignored
#: ``output_model_file`` would leave the checkpoint somewhere nobody addressed.
_FIT_REQUIRED_PARAMETERS: tuple[str, ...] = (
    "smiles_file",
    "output_model_file",
    "num_epochs",
    "batch_size",
    "input_model_file",
    "device",
)

#: What :func:`~.tl_data.build_training_smiles` has to report before a fitted
#: checkpoint is allowed out of this handler. The two held-out counts are the
#: positive control: a boundary that was never offered a protected compound has
#: not been measured, and one that admitted a protected compound has failed.
_TRAINING_SET_KEYS: tuple[str, ...] = (
    "path",
    "n_offered",
    "n_admitted",
    "n_quarantined",
    "n_held_out_offered",
    "n_held_out_admitted",
    "n_vocab_filtered",
    "sha256",
)


def _outside_source_control(output_dir: Path, repo_root: Path) -> str | None:
    """``None`` when a checkpoint may be written here, else why it may not.

    A fine-tuned REINVENT4 prior is ~23 MB of weights. Writing it into the
    tracked tree would put a binary into the repository whose content address 36
    campaign records pin, so the only permitted homes are outside the checkout
    entirely or inside ``.external/``, which ``.gitignore`` excludes and which is
    where the dispatcher already puts every campaign's derived output.
    """
    output = Path(output_dir).resolve()
    layout = RepositoryLayout.discover(repo_root)
    try:
        relative = output.relative_to(layout.project_root)
    except ValueError:
        relative = None
    if relative is not None and relative.parts and relative.parts[0] == ".external":
        return None
    checkout = layout.git_root or layout.project_root
    if not output.is_relative_to(checkout):
        return None
    return (
        f"{output_dir} is inside the Git checkout at {checkout} and not under the task "
        "project's '.external/' directory; checkpoint output cannot enter another "
        "agent project or a source-controlled location"
    )


def adapt_generator(plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path) -> Record:
    """Target-adapt a pretrained generator's weights, and prove what it learned from.

    Appendix A §4 asks for local training support to be *verified* rather than
    assumed, and the probe is kept as the **precondition** rather than replaced
    by this body: a host whose generator exposes no training verb still gets the
    structured ``UNAVAILABLE`` naming the verb and what would unblock it, which
    is a result a controller can act on, instead of an ``AttributeError``.

    When the verb is there, the run is three steps and each one is recorded:

    1. :func:`~.tl_data.build_training_smiles` builds the fine-tuning set by
       offering every candidate compound to Step 6's protected registry — the
       one fixed in :data:`PROTECTED_REGISTRY_RELATIVE`, never one the plan
       names — and reports how many it admitted, quarantined and dropped for
       being outside the prior's vocabulary. Its two held-out counts are the
       positive control, and this handler re-checks both rather than trusting
       that the builder raised: a boundary never *offered* a protected compound
       has not been measured, and one that admitted a protected compound has
       failed.
    2. the adapter's own ``fit`` writes a checkpoint, under ``.external/`` and
       therefore outside source control, and reports the sha256 of both the
       bytes it wrote and the prior it started from.
    3. the checkpoint is re-hashed here from the file on disk. A ``fit`` that
       reports one digest and writes different bytes has produced an
       unattributable artifact, which is a failure and not a rounding error.

    **``inputs.training`` is required, and defaults do not stand in for it.**
    Every other handler here runs on defaults when a plan names nothing, and
    ``fit_predictor`` genuinely should: it is seconds of RandomForest and it
    rewrites nothing. A fine-tuning run is the opposite on both counts — it is
    the longest action in the menu, it is the one that touches the exclusion
    boundary, and it rewrites the weights every later sampling call loads. A run
    that expensive and that consequential is not started because a plan said
    nothing, so an absent or empty ``training`` block is answered with the field
    list rather than with a default adaptation nobody proposed.

    **This handler does not sample and does not score.** The checkpoint it emits
    changes nothing on its own; it becomes a measurement only when a later
    ``change_sampling`` experiment is handed the path through
    ``inputs.sampling.checkpoint`` and records the digest it actually loaded.
    """
    started = time.time()
    inputs = dict(plan["inputs"])
    training, _refusal = _object_input(
        inputs, "training", action="adapt_generator", backend=backend, started=started
    )
    if _refusal is not None:
        return _refusal
    if not training:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes="no training block was named",
            reason=(
                "inputs['training'] is absent or empty. This is the one action whose defaults are "
                "not allowed to stand in for a proposal: it is the longest run in the menu, it is "
                "the run that offers compounds to the exclusion boundary, and it rewrites the "
                "weights every later sampling call loads. An adaptation nobody parameterised is "
                "not an experiment anyone proposed"
            ),
            unblock=(
                "send inputs.training as an object. Every field is optional once the block is "
                "there: partition (default 'train'), epochs (10), batch_size (64), device, "
                "input_model (a checkpoint to continue from; omit for the parent prior) and "
                "output_model (the checkpoint's name relative to the output directory, default "
                "'generator/<backend>_tl.prior'). inputs.training = {'partition': 'train'} is "
                "enough. The exclusion registry is NOT an input and cannot be named"
            ),
            cost=_cost(started),
        )

    from .baselines import probe_capability

    capability_id = _GENERATOR_CAPABILITY.get(backend, f"molecule.generate.{backend}")
    probes = [probe_capability(capability_id, "fit")]
    _write(output_dir / "capability_probe.json", probes)
    unavailable = [p for p in probes if p["status"] != "AVAILABLE_EXECUTED"]
    if unavailable:
        return _result(
            "adapt_generator",
            backend,
            "UNAVAILABLE",
            notes=f"{capability_id} does not expose a training verb on this host",
            reason="; ".join(f"{p['capability_id']}/{p['verb']}: {p['reason']}" for p in unavailable),
            unblock=" | ".join(str(p["unblock"]) for p in unavailable if p.get("unblock")),
            evidence={"probes": probes, "verbs_missing": [[p["capability_id"], "fit"] for p in unavailable]},
            cost=_cost(started),
        )

    # --- the adapter, and whether its training verb is the one contracted ----
    if backend == "reinvent4":
        from drug_agent.tools.generation.reinvent4_tool import Reinvent4Adapter

        adapter: Any = Reinvent4Adapter()
    elif backend == "pfm":
        from drug_agent.tools.pfm_tool import PFMGeneratorAdapter

        adapter = PFMGeneratorAdapter()
    else:  # pragma: no cover - validate_plan refuses an unregistered backend first
        raise SystemExit(f"adapt_generator has no backend named {backend!r}")

    if not adapter.available():
        try:
            unavailable_reason = str(getattr(adapter.profile, "unavailable_reason", None) or "")
        except Exception as exc:  # a profile that cannot be built is itself the reason
            unavailable_reason = f"{type(exc).__name__}: {exc}"
        return _result(
            "adapt_generator",
            backend,
            "UNAVAILABLE",
            notes=f"the {backend} adapter reports it is unavailable",
            reason=unavailable_reason or "unspecified",
            unblock=f"install the {backend} backend and its weights, then re-dispatch this plan",
            evidence={"probes": probes},
            cost=_cost(started),
        )

    fit = getattr(adapter, "fit", None)
    if not callable(fit):
        return _result(
            "adapt_generator",
            backend,
            "UNAVAILABLE",
            notes=f"the {backend} adapter exposes no fit(...) method",
            reason=(
                f"the capability probe reported {capability_id}/fit as executable but "
                f"{type(adapter).__name__} has no callable 'fit'. The probe and the adapter "
                f"disagree, and training against the disagreement would report an adaptation "
                f"that never happened"
            ),
            unblock=(
                "implement Reinvent4Adapter.fit(*, smiles_file, output_model_file, num_epochs, "
                "batch_size, input_model_file=None, device=None) -> dict, or withdraw the probe"
            ),
            evidence={"probes": probes},
            cost=_cost(started),
        )

    import inspect

    parameters = inspect.signature(fit).parameters
    missing_parameters = [p for p in _FIT_REQUIRED_PARAMETERS if p not in parameters]
    if missing_parameters:
        return _result(
            "adapt_generator",
            backend,
            "UNAVAILABLE",
            notes=f"{type(adapter).__name__}.fit does not accept {missing_parameters}",
            reason=(
                f"the training verb on this host is missing the keyword(s) {missing_parameters}. "
                f"Calling it anyway would train from a prior, for a number of epochs, or onto a "
                f"path that this record could not honestly name"
            ),
            unblock=(
                "fit(*, smiles_file, output_model_file, num_epochs, batch_size, "
                "input_model_file=None, device=None) -> dict is the contracted signature"
            ),
            evidence={"probes": probes, "fit_parameters": sorted(parameters)},
            cost=_cost(started),
        )

    # --- where the weights may be written -----------------------------------
    source_controlled = _outside_source_control(output_dir, repo_root)
    if source_controlled is not None:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes="this experiment's output directory is inside source control",
            reason=(
                f"{source_controlled}. A ~23 MB checkpoint committed into the tracked tree would "
                f"move the content address that 36 campaign records pin"
            ),
            unblock=(
                "dispatch with --output under .external/research/zak-c9r/derived/campaigns/, "
                "or anywhere outside the checkout"
            ),
            cost=_cost(started),
        )

    # --- the prior this run starts from, named before anything trains -------
    input_model_file: Path | None = None
    declared_input_model = training.get("input_model") or training.get("input_model_file")
    if declared_input_model:
        input_model_file = _resolve(repo_root, str(declared_input_model))
        if not input_model_file.is_file():
            return _result(
                "adapt_generator",
                backend,
                "FAILED",
                notes=f"{input_model_file} does not exist",
                reason=(
                    f"inputs['training']['input_model'] names {str(declared_input_model)!r}, which "
                    f"is not on disk. Falling back to the parent prior would report an adaptation "
                    f"of one checkpoint as an adaptation of another"
                ),
                unblock="pass the checkpoint an earlier adapt_generator experiment wrote, or omit "
                "the key to start from the parent prior",
                cost=_cost(started),
            )
    vocabulary_from = input_model_file
    if vocabulary_from is None:
        try:
            vocabulary_from = Path(adapter._prior_path())
        except Exception as exc:  # the parent prior is what fit(None) will load
            return _result(
                "adapt_generator",
                backend,
                "UNAVAILABLE",
                notes="the parent prior could not be resolved",
                reason=(
                    f"{type(exc).__name__}: {exc}. The training set has to be pre-filtered against "
                    f"the vocabulary of the prior the fit will start from, and that prior is not "
                    f"locatable here"
                ),
                unblock="make reinvent4-install (priors: https://doi.org/10.5281/zenodo.15641296)",
                cost=_cost(started),
            )

    # --- the training set, and the boundary measurement that guards it ------
    registry_path = repo_root / PROTECTED_REGISTRY_RELATIVE
    partition = str(training.get("partition") or "train")
    smiles_path = output_dir / "training_smiles.smi"
    try:
        from .tl_data import build_training_smiles

        built = dict(
            build_training_smiles(
                partition=partition,
                exclusion_registry=registry_path,
                out_path=smiles_path,
                vocabulary_from=vocabulary_from,
            )
        )
    except Exception as exc:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes="the fine-tuning set could not be built",
            reason=(
                f"build_training_smiles(partition={partition!r}) raised {type(exc).__name__}: "
                f"{exc}. No checkpoint was written. If this is the boundary refusing, it is the "
                f"mechanism working: a training set that reaches a protected compound is not a "
                f"training set to be repaired, it is a leak to be reported"
            ),
            unblock=(
                "read the reason above; a held-out admission is an integrity finding and must be "
                "reported, never worked around"
            ),
            evidence={"partition": partition, "exclusion_registry": str(registry_path)},
            cost=_cost(started),
        )

    absent = [key for key in _TRAINING_SET_KEYS if key not in built]
    if absent:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes=f"the training-set record is missing {absent}",
            reason=(
                f"build_training_smiles returned a record without {absent}. The admitted, "
                f"quarantined, vocabulary-filtered and held-out counts are the evidence that the "
                f"boundary ran; a fit on a set that cannot report them is unattributable"
            ),
            unblock="return every key in the build_training_smiles contract",
            evidence={"training_set": built},
            cost=_cost(started),
        )
    # Re-checked here rather than trusted: build_training_smiles raises on an
    # admitted held-out compound, and this handler is the second reader of the
    # same two numbers precisely because a single check is a single point of
    # silence.
    if int(built["n_held_out_admitted"]) != 0:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes=f"{built['n_held_out_admitted']} held-out compounds were admitted to the fit set",
            reason=(
                "the exclusion boundary admitted a protected compound. No checkpoint is emitted "
                "and nothing downstream may use this run"
            ),
            unblock="report this as an integrity finding against the split and the registry",
            evidence={"training_set": built},
            cost=_cost(started),
        )
    if int(built["n_held_out_offered"]) <= 0:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes="the exclusion boundary was never offered a held-out compound",
            reason=(
                "n_held_out_offered is 0, so the positive control did not run. A boundary that "
                "was never asked to refuse anything has not been measured, and 'nothing was "
                "admitted' from a boundary nothing was offered is not evidence"
            ),
            unblock=(
                "offer the held-out compounds to the same registry, as "
                "scripts/research/calibrate_zak_benchmark.py does, and report the counts"
            ),
            evidence={"training_set": built},
            cost=_cost(started),
        )
    if int(built["n_admitted"]) <= 0:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes="the fine-tuning set is empty",
            reason=(
                f"{built['n_offered']} compounds were offered, {built['n_quarantined']} "
                f"quarantined and {built['n_vocab_filtered']} dropped as out-of-vocabulary, "
                f"leaving none to train on. A fit over zero molecules would write a copy of the "
                f"prior under a new name"
            ),
            unblock=f"check that partition {partition!r} has admitted compounds in this snapshot",
            evidence={"training_set": built},
            cost=_cost(started),
        )

    # --- the fit ------------------------------------------------------------
    # The checkpoint's name is the plan's to choose, inside this experiment's own
    # output directory and nowhere else. A plan declares ``expected_outputs`` and
    # the dispatcher verifies them, so a handler that always wrote one fixed
    # filename would turn every plan that declared a different one into a
    # CANDIDATE_ERROR. It stays *relative*, and stays inside ``output_dir``,
    # because ``worker._store_artifacts`` resolves a declared weights file as
    # ``output_dir / name`` and because a handler that writes where a plan points
    # is a handler that can be pointed at the tracked tree.
    checkpoint_relative = Path("generator") / f"{backend}_tl.prior"
    declared_output = training.get("output_model") or training.get("checkpoint")
    if declared_output:
        candidate_relative = Path(str(declared_output))
        if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
            return _result(
                "adapt_generator",
                backend,
                "FAILED",
                notes=f"inputs['training']['output_model'] {str(declared_output)!r} leaves the experiment",
                reason=(
                    "the checkpoint's name is a path relative to this experiment's output "
                    "directory. An absolute path or one containing '..' would write weights "
                    "somewhere the dispatcher does not address and cannot verify"
                ),
                unblock="name a relative path such as 'weights.pt' or 'generator/tuned.prior'",
                cost=_cost(started),
            )
        checkpoint_relative = candidate_relative
    checkpoint_path = output_dir / checkpoint_relative
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 64))
    device = training.get("device")
    try:
        fitted = dict(
            fit(
                smiles_file=Path(built["path"]),
                output_model_file=checkpoint_path,
                num_epochs=epochs,
                batch_size=batch_size,
                input_model_file=input_model_file,
                device=None if device is None else str(device),
            )
        )
    except Exception as exc:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes=f"the training run raised {type(exc).__name__}",
            reason=f"{type(adapter).__name__}.fit raised {type(exc).__name__}: {exc}",
            unblock="read the traceback in this experiment's stderr log",
            evidence={"training_set": built, "epochs": epochs, "batch_size": batch_size},
            cost=_cost(started),
        )

    if not checkpoint_path.is_file():
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes=f"fit returned without writing {checkpoint_path}",
            reason=(
                "the training verb reported success and the checkpoint it was told to write is "
                "not on disk. A WEIGHTS artifact with no weights behind it is an advertisement"
            ),
            unblock="write the checkpoint to output_model_file, or report the failure",
            evidence={"training_set": built, "fit": fitted},
            cost=_cost(started),
        )
    measured_sha256 = _sha256_path(checkpoint_path)
    declared_sha256 = str(fitted.get("sha256") or "")
    if declared_sha256 and declared_sha256.split(":")[-1] != measured_sha256.split(":")[-1]:
        return _result(
            "adapt_generator",
            backend,
            "FAILED",
            notes="the checkpoint's declared digest disagrees with the bytes on disk",
            reason=(
                f"fit reported sha256 {declared_sha256!r} and {checkpoint_path.name} hashes to "
                f"{measured_sha256!r}. A checkpoint whose digest does not address its own bytes "
                f"cannot be cited by the sampling run that loads it"
            ),
            unblock="hash the file that was actually written",
            evidence={"training_set": built, "fit": fitted, "measured_sha256": measured_sha256},
            cost=_cost(started),
        )

    # The fine-tuning set is declared as this artifact's training data only when
    # it really sits under the output directory, because that is the only place
    # ``worker._store_artifacts`` resolves a declared file from. A builder that
    # wrote it elsewhere leaves the path in adaptation.json and claims nothing.
    training_data: list[str] = []
    built_path = Path(str(built["path"]))
    try:
        training_data = [str(built_path.resolve().relative_to(Path(output_dir).resolve()))]
    except (OSError, ValueError):
        training_data = []

    adaptation = {
        "backend": backend,
        "capability_id": capability_id,
        "partition": partition,
        "exclusion_registry": str(registry_path),
        "training_set": built,
        "input_model_file": str(input_model_file) if input_model_file else None,
        "input_model_is_parent_prior": input_model_file is None,
        "input_model_sha256": fitted.get("input_model_sha256"),
        "vocabulary_from": str(vocabulary_from),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": measured_sha256,
        "checkpoint_relative_to_output_dir": str(checkpoint_relative),
        "epochs": fitted.get("epochs", epochs),
        "batch_size": batch_size,
        "device": device,
        "n_training_smiles": fitted.get("n_training_smiles"),
        "wall_seconds": fitted.get("wall_seconds"),
        "gpu_seconds": fitted.get("gpu_seconds"),
        "note": (
            "this checkpoint changes nothing until a change_sampling experiment is given it as "
            "inputs.sampling.checkpoint. Without that, sampling loads the parent prior and the "
            "arm re-measures the incumbent with no error anywhere"
        ),
    }
    _write(output_dir / "adaptation.json", adaptation)
    return _result(
        "adapt_generator",
        backend,
        "SUCCEEDED",
        notes=(
            f"fine-tuned {capability_id} for {adaptation['epochs']} epochs on "
            f"{built['n_written']} molecules, written from {built['n_admitted']} admitted "
            f"{partition} compounds ({built['n_quarantined']} quarantined, "
            f"{built['n_vocab_filtered']} out-of-vocabulary, "
            f"{built['n_duplicates_collapsed']} collapsed onto another compound by the "
            f"standardiser); {built['n_held_out_offered']} held-out compounds were offered "
            f"to the boundary and {built['n_held_out_admitted']} were admitted. "
            # The root cause of the zak-c9r series' failed transfer-learning
            # cycles was that this string -- the text the agent reads -- did not
            # contain the path at all. It lived only in
            # evidence.adaptation.checkpoint, so the agent reconstructed a path
            # or, in six of the eleven refused cycles recorded under
            # derived/campaigns/, passed a DESCRIPTION of where the path could be
            # found ("<checkpoint path recorded in ..._c01/adaptation.json>").
            # change_sampling correctly refused every one. No resolution order
            # fixes that; only telling the agent the value does. So the value is
            # spelled out here, absolute, with what to do with it.
            f"CHECKPOINT WRITTEN: {checkpoint_path} -- pass this ABSOLUTE path "
            f"verbatim as inputs.sampling.checkpoint of a later change_sampling "
            f"experiment (not a path relative to this or any other directory, and "
            f"do not reconstruct it from the experiment id). Its sha256 is "
            f"{measured_sha256}. These weights change nothing until a "
            f"change_sampling experiment names them"
        ),
        artifacts=[
            {
                "artifact_id": f"art_{plan['experiment_id']}_generator",
                "kind": str(plan["kind"]),
                "parent_id": plan.get("parent_id"),
                "entrypoint": [
                    "python",
                    "-m",
                    "drug_agent.selfevolve.target.actions",
                    "--action",
                    "adapt_generator",
                    "--backend",
                    backend,
                ],
                "weights": [str(checkpoint_relative)],
                "training_data": training_data,
                "digests": {str(checkpoint_relative): measured_sha256},
            }
        ],
        evidence={
            "probes": probes,
            "adaptation": adaptation,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": measured_sha256,
            "input_model_sha256": fitted.get("input_model_sha256"),
            "boundary": {
                "exclusion_registry": str(registry_path),
                "n_offered": built["n_offered"],
                "n_admitted": built["n_admitted"],
                "n_quarantined": built["n_quarantined"],
                "n_vocab_filtered": built["n_vocab_filtered"],
                "n_duplicates_collapsed": built["n_duplicates_collapsed"],
                "n_written": built["n_written"],
                "n_held_out_offered": built["n_held_out_offered"],
                "n_held_out_admitted": built["n_held_out_admitted"],
            },
        },
        cost=_cost(started, gpu_seconds=fitted.get("gpu_seconds")),
    )


def change_sampling(plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path) -> Record:
    """Sample under an altered policy, score what came out, record the whole funnel.

    Raw attempts, valid molecules and unique molecules are three numbers, kept
    apart: a policy that triples the raw count and produces the same handful of
    unique valid molecules has not improved anything, and a single reported
    "generated N" would hide that.

    The molecules are scored before this process exits because the frozen
    evaluator does not score. ``evaluate_portfolio`` *reads* endpoint records and
    applies the seven gates to them; it docks nothing and profiles nothing. A
    portfolio of bare ``{molecule_id, smiles}`` is therefore not one it judges
    leniently, it is one it refuses outright — which is why no Track B campaign
    has ever been scorable through this action.

    The scoring pass is
    :func:`~drug_agent.selfevolve.target.generation_sweep.score_molecules`, the
    one the nine-arm sweep already runs, *called* rather than reimplemented: a
    second scoring path is how two definitions of "qualifying" end up in one
    repository. A molecule it could not score keeps its failed endpoint record
    and is written out as it is — ``evaluate_portfolio`` counts that molecule in
    ``n_scoring_failed``, and any substituted value would turn an absence into a
    number that can pass a gate.

    Scoring here is still not self-grading. This handler produces measurements;
    the gates, the thresholds and the qualifying count are applied afterwards, by
    the parent checkout's frozen evaluator, over these rows.

    **Generated and submitted are two numbers.** ``n_molecules`` is the
    submission budget the metric is defined at — 150 for this series, and the
    only number the evaluator's absolute count is comparable across.
    ``n_generated`` is how many the sampler was asked to produce, defaulting to
    ``n_molecules`` so that a plan which names neither behaves exactly as every
    campaign before this one did. Generating more than is submitted is selection
    pressure, and :func:`generation_cap` is the run-scoped ceiling on it: it may
    only *tighten* what a plan asked for, it is checked again against what the
    backend actually emitted, and a ceiling that would cut into the submission
    budget is refused rather than silently applied, because a run at some other
    submission budget is comparable to nothing.

    **The checkpoint is named and hashed.** ``inputs.sampling.checkpoint`` points
    sampling at a fine-tuned prior. It is the half of transfer learning that is
    easy to leave out and impossible to notice afterwards: a ``fit`` that works
    while generation still loads the parent prior produces an arm whose numbers
    are indistinguishable from the incumbent's, with no error, no warning and no
    failed probe. So the checkpoint is resolved and hashed *before* sampling,
    the adapter is asked whether it can honour an override *before* the GPU is
    spent, and the digest goes into the result.
    """
    started = time.time()
    inputs = dict(plan["inputs"])
    sampling, _refusal = _object_input(
        inputs, "sampling", action="change_sampling", backend=backend, started=started
    )
    if _refusal is not None:
        return _refusal
    n_molecules = int(sampling.get("n_molecules", 8))
    seed = int(sampling.get("seed", 42))
    batch_size = int(sampling.get("batch_size", n_molecules))

    # --- the run-scoped ceiling on molecules GENERATED ----------------------
    # Scoped to *this* campaign, by name. The plan carries the campaign it was
    # dispatched under, so a constraints file that names a different one -- or
    # that cannot be matched at all, because this dispatch names no campaign --
    # is a file this run cannot be shown to own, and is refused rather than
    # obeyed. The plan's field is passed THROUGH, not coerced: an "or None"
    # fallback here is what used to hand ``run_constraints`` the one argument
    # that switched its ownership check off.
    constraints = run_constraints(output_dir, campaign_id=plan.get("campaign_id"))
    if constraints.get("foreign"):
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=(
                f"the run constraints at {constraints.get('source')} cannot be shown to belong "
                f"to this campaign"
            ),
            reason=(
                f"{constraints.get('error')}. A constraints file that is present and cannot be "
                f"matched to the dispatched campaign cannot be obeyed (its ceiling was declared "
                f"for a run nobody can identify as this one) and must not be ignored (ignoring "
                f"it would run this experiment unconstrained on the strength of a name that did "
                f"not match, or of a name nobody supplied)"
            ),
            unblock=(
                f"remove {constraints.get('source')} if it is stale, or re-dispatch this "
                f"experiment with a campaign_id matching the one its campaign root declares "
                f"(this dispatch carried {constraints.get('dispatched_campaign_id')!r})"
            ),
            evidence={
                "constraints_source": constraints.get("source"),
                "declared_campaign_id": constraints.get("declared_campaign_id"),
                "dispatched_campaign_id": constraints.get("dispatched_campaign_id"),
            },
            cost=_cost(started),
        )
    if constraints.get("widening_conflict"):
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=(
                f"two run constraints files disagree about the generation ceiling, and the "
                f"disagreement widens it (effective ceiling {constraints.get('max_generated')})"
            ),
            reason=(
                f"{constraints.get('error')}. The ceiling is the minimum over every constraints "
                f"file on the walk, so this run could only ever have sampled under the tightest "
                f"of them; it is refused because a file declaring a looser ceiling inside a run "
                f"directory is either an attempt to widen a control that may only tighten, or a "
                f"stale file -- and running quietly under the tight number would leave either "
                f"one sitting in the tree unremarked"
            ),
            unblock=(
                "remove the constraints file that declares the looser ceiling, or re-declare it "
                "at or below the campaign root's, then re-dispatch"
            ),
            evidence={
                "constraints_sources": constraints.get("sources"),
                "declared_ceilings": constraints.get("declared_ceilings"),
                "effective_max_generated": constraints.get("max_generated"),
                "effective_max_generated_source": constraints.get("source"),
            },
            cost=_cost(started),
        )
    if constraints.get("unreadable"):
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"the run constraints at {constraints.get('source')} could not be read",
            reason=(
                f"{constraints.get('error')}. A constraints file that exists and does not parse "
                f"is not an unconstrained run: continuing would sample under a ceiling nobody "
                f"could read, and record it as though none had been declared"
            ),
            unblock=f"repair or remove {constraints.get('source')}, then re-dispatch",
            cost=_cost(started),
        )
    cap, cap_problem = generation_cap(constraints)
    if cap_problem is not None:
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes="the declared generation ceiling is not a usable count",
            reason=f"{cap_problem} (from {constraints.get('source')})",
            unblock="pass --max-generated as a positive integer, or omit it entirely",
            cost=_cost(started),
        )

    n_generated_requested = int(sampling.get("n_generated", n_molecules))
    if n_generated_requested < n_molecules:
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"n_generated={n_generated_requested} is below n_molecules={n_molecules}",
            reason=(
                "a plan cannot submit more molecules than it generates. n_molecules is the "
                "submission budget the metric is defined at and n_generated is the pool it is "
                "selected from, so the pool may equal the budget but never fall below it"
            ),
            unblock=f"raise inputs.sampling.n_generated to at least {n_molecules}, or drop the key",
            cost=_cost(started),
        )
    n_generate = n_generated_requested if cap is None else min(n_generated_requested, cap)
    if n_generate < n_molecules:
        # The cap binds below the submission budget. Sampling fewer would change
        # the denominator the registered metric is an absolute count at, and the
        # resulting number would be comparable to neither the incumbent nor the
        # nine-arm table.
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"the generation ceiling {cap} is below the submission budget {n_molecules}",
            reason=(
                f"this run declares max_generated={cap} and the plan submits {n_molecules} "
                f"molecules. Honouring the ceiling would sample fewer molecules than the "
                f"submission budget, and the registered metric is an absolute COUNT at a fixed "
                f"budget: a run at any other budget produces a number comparable to nothing"
            ),
            unblock=(
                f"raise --max-generated to at least the submission budget ({n_molecules}), or "
                f"restrict this arm's action menu instead of capping it"
            ),
            evidence={
                "max_generated": cap,
                "max_generated_source": constraints.get("source"),
                "n_molecules": n_molecules,
            },
            cost=_cost(started),
        )
    # ``batch_size`` is deliberately passed through untouched. REINVENT4 ignores
    # it (run_sampling sets the sampler's batch size to num_smiles and chunks at
    # SAMPLE_BATCH_SIZE) and PFM chunks with it; clamping it here would change
    # what an already-registered plan sends for no measurement anyone asked for.

    # Read from this checkout, and read it *before* sampling. The scorer's
    # operating point is frozen, a host that cannot read it cannot score anything
    # the sampler produces, and discovering that after the GPU has been spent
    # costs a whole run and reports nothing about the policy under test.
    calibration_path = repo_root / "configs" / "research" / "zak-c9r" / "calibration.json"
    try:
        scorer = dict(json.loads(calibration_path.read_text(encoding="utf-8"))["scorer"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result(
            "change_sampling",
            backend,
            "UNAVAILABLE",
            notes=f"no frozen scorer record at {calibration_path}",
            reason=(
                f"{calibration_path} did not yield a 'scorer' block "
                f"({type(exc).__name__}: {exc}). Sampling without it would produce molecules "
                f"that nobody can score against the registered operating point, and an "
                f"unscored portfolio is one the frozen evaluator refuses"
            ),
            unblock=(
                f"dispatch into a checkout that carries {calibration_path.name}, or rebuild it "
                f"with scripts/research/calibrate_zak_benchmark.py"
            ),
            cost=_cost(started),
        )

    declared_checkpoint = sampling.get("checkpoint")
    checkpoint_record: Record = {
        "requested": None,
        "requested_sha256": None,
        "reported_by_adapter": None,
        "reported_sha256": None,
        "incumbent_prior": None,
        "incumbent_prior_sha256": None,
        "declared": None if declared_checkpoint is None else str(declared_checkpoint),
        "resolved_via": None,
        "resolution_attempts": [],
        "verified": "NOT_REQUESTED",
        "note": (
            "a transfer-learning arm whose sampling call quietly loads the parent prior reports "
            "numbers indistinguishable from the incumbent's, with no error anywhere. These "
            "digests are what makes that detectable after the fact"
        ),
    }

    if backend == "reinvent4":
        from drug_agent.tools.generation.reinvent4_tool import Reinvent4Adapter

        adapter: Any = Reinvent4Adapter()
        if not adapter.available():
            return _result(
                "change_sampling",
                backend,
                "UNAVAILABLE",
                notes="the REINVENT4 adapter reports it is unavailable",
                reason=str(adapter.profile.unavailable_reason or "unspecified"),
                unblock="make reinvent4-install",
                cost=_cost(started),
            )

        # The prior this adapter loads when nothing overrides it, named and
        # hashed whether or not an override was asked for: it is the value an
        # override has to *differ* from, and without it "the checkpoint loaded"
        # is a digest with nothing to compare against.
        try:
            prior_path = Path(adapter._prior_path())
            if prior_path.is_file():
                checkpoint_record["incumbent_prior"] = str(prior_path)
                checkpoint_record["incumbent_prior_sha256"] = _adapter_sha256(adapter, prior_path)
        except Exception as exc:  # not fatal: the sampling call answers for itself
            checkpoint_record["incumbent_prior"] = f"UNRESOLVED ({type(exc).__name__}: {exc})"

        generate_kwargs: Record = {}
        if declared_checkpoint:
            # A relative value is tried against each documented base in turn and
            # the first that holds the file wins; the base that answered is
            # recorded so a later reader can see which spelling the agent used.
            # When none answers the run is still REFUSED — the refusal is the
            # guard that stopped this programme reporting weight-level evolution
            # it had not performed, and no branch here reaches for the prior.
            resolved, resolved_via, attempts = _resolve_checkpoint(
                str(declared_checkpoint), output_dir=output_dir, repo_root=repo_root
            )
            checkpoint_record["resolution_attempts"] = [
                {"base": name, "path": str(path)} for name, path in attempts
            ]
            if resolved is None:
                tried = "; ".join(f"{name} -> {path}" for name, path in attempts)
                return _result(
                    "change_sampling",
                    backend,
                    "FAILED",
                    notes=(
                        f"the named checkpoint {str(declared_checkpoint)!r} is not on disk under "
                        f"any of the {len(attempts)} bases tried"
                    ),
                    reason=(
                        f"inputs['sampling']['checkpoint'] names {str(declared_checkpoint)!r}, "
                        f"which does not exist. Sampling anyway would load the parent prior and "
                        f"report it as the adapted generator. Every base tried, in order, with "
                        f"the absolute path it formed: {tried}"
                    ),
                    unblock=(
                        "pass the ABSOLUTE checkpoint path the adapt_generator experiment "
                        "reported -- it is quoted in that experiment's own notes and recorded at "
                        "evidence.adaptation.checkpoint and in its adaptation.json -- rather "
                        "than a path relative to anything, or omit the key to sample the parent "
                        "prior deliberately"
                    ),
                    evidence={"checkpoint": dict(checkpoint_record)},
                    cost=_cost(started),
                )
            checkpoint_path = resolved
            checkpoint_record["resolved_via"] = resolved_via
            import inspect

            if "checkpoint" not in inspect.signature(adapter.generate).parameters:
                # THE silent failure this block exists to prevent. Sampling would
                # succeed, the fine-tuned weights would be ignored, and the arm
                # would re-measure the incumbent with nothing anywhere saying so.
                return _result(
                    "change_sampling",
                    backend,
                    "UNAVAILABLE",
                    notes="this checkout's generate() cannot load a checkpoint override",
                    reason=(
                        f"{type(adapter).__name__}.generate has no 'checkpoint' parameter, so the "
                        f"fine-tuned weights at {checkpoint_path} would be ignored and the parent "
                        f"prior sampled instead. That run would succeed and report numbers "
                        f"indistinguishable from the incumbent's, which is why it is refused "
                        f"rather than attempted"
                    ),
                    unblock=(
                        "add 'checkpoint: str | Path | None = None' to "
                        "drug_agent.tools.generation.generate(), thread it into _sample(), and "
                        "let Reinvent4Adapter._prior_path()/model_file honour the override"
                    ),
                    evidence={"checkpoint": str(checkpoint_path)},
                    cost=_cost(started),
                )
            generate_kwargs["checkpoint"] = checkpoint_path
            checkpoint_record["requested"] = str(checkpoint_path)
            checkpoint_record["requested_sha256"] = _adapter_sha256(adapter, checkpoint_path)
            checkpoint_record["verified"] = "UNCONFIRMED"

        batch = adapter.generate(
            n_molecules=n_generate,
            seed=seed,
            batch_size=batch_size,
            max_runtime_seconds=float(sampling.get("max_runtime_seconds", 900)),
            **generate_kwargs,
        )
    else:
        from drug_agent.tools.pfm_tool import PFMGeneratorAdapter

        if declared_checkpoint:
            return _result(
                "change_sampling",
                backend,
                "FAILED",
                notes="the pfm backend has no checkpoint override",
                reason=(
                    f"inputs['sampling']['checkpoint'] names {str(declared_checkpoint)!r} and the "
                    f"PFM adapter exposes no way to load one. Accepting the key and ignoring it "
                    f"would report a run against adapted weights that sampled the shipped model"
                ),
                unblock="drop inputs.sampling.checkpoint, or run this arm on the reinvent4 backend",
                cost=_cost(started),
            )
        adapter = PFMGeneratorAdapter()
        pocket = _resolve(
            repo_root,
            str(sampling.get("pocket") or (SNAPSHOT_RELATIVE / "derived" / "zak_6juu_C9R_pocket10.pdb")),
        )
        if not pocket.is_file():
            return _result(
                "change_sampling",
                backend,
                "UNAVAILABLE",
                notes=f"the declared pocket {pocket} is not on disk",
                reason="a structure-conditioned sampler with no pocket has no target",
                unblock="build the ZAK pocket with scripts/research/build_zak_target_data.py",
                cost=_cost(started),
            )
        # ``pocket=``, and a payload, not ``pocket_pdb=`` and a blob of PDB text.
        # PFMGeneratorAdapter.generate is keyword-only and declares no
        # ``pocket_pdb`` parameter and no ``**kwargs``, so the old call raised
        # TypeError before the sampler was ever reached: this backend of
        # change_sampling could never execute. The payload is the shape
        # ``baselines._generation_inputs`` assembles for a pocket-conditioned
        # adapter -- the pocket by PATH (the adapter reads and digests the file
        # itself), plus the entry and chain the pocket was carved from. No
        # ``center`` key: PFM's ``required_inputs`` is ``('pocket_pdb',)``, and
        # inventing a box centre it never asked for would be a second change.
        batch = adapter.generate(
            n_molecules=n_generate,
            seed=seed,
            batch_size=batch_size,
            pocket={
                "pdb_path": str(pocket),
                "pdb_id": "6JUU",
                "receptor_id": "6JUU:A",
            },
        )

    # --- what the sampler says it actually loaded ---------------------------
    # ``sample_provenance`` is the adapter's own statement of which weights it
    # read, and it is the only field trusted here. ``config['checkpoint']`` is
    # an echo of the *argument*, so comparing the request against that would
    # confirm nothing: it agrees with itself whether or not the override took.
    provenance = dict(dict(batch.config or {}).get("sample_provenance") or {})
    checkpoint_record["adapter_provenance"] = provenance or None
    reported = provenance.get("checkpoint")
    if reported:
        checkpoint_record["reported_by_adapter"] = str(reported)
        reported_digest = provenance.get("checkpoint_sha256")
        if reported_digest:
            checkpoint_record["reported_sha256"] = f"sha256:{_hexdigest(reported_digest)}"
        elif Path(str(reported)).is_file():
            checkpoint_record["reported_sha256"] = _sha256_path(Path(str(reported)))
    if checkpoint_record["requested_sha256"] and checkpoint_record["reported_sha256"]:
        checkpoint_record["verified"] = (
            "MATCHED"
            if _hexdigest(checkpoint_record["reported_sha256"])
            == _hexdigest(checkpoint_record["requested_sha256"])
            else "MISMATCH"
        )
        # An override that resolves to the parent prior is not an adapted
        # generator, however well the digests agree with each other — by path,
        # which is what the adapter reports, or by bytes, which catches a copy of
        # the prior sitting at a path that looks fine-tuned.
        prior_digest = checkpoint_record["incumbent_prior_sha256"]
        if provenance.get("is_parent_prior") or (
            prior_digest and _hexdigest(prior_digest) == _hexdigest(checkpoint_record["reported_sha256"])
        ):
            checkpoint_record["verified"] = "RESOLVED_TO_PARENT_PRIOR"
    elif not declared_checkpoint and checkpoint_record["incumbent_prior_sha256"]:
        checkpoint_record["verified"] = "INCUMBENT_PRIOR"

    smiles = [str(getattr(m, "smiles", "") or "") for m in batch.molecules]
    valid = [s for s in smiles if s]
    # SELECTION ORDER: the order the sampler emitted them, de-duplicated with the
    # first occurrence kept. This used to be `sorted(set(valid))`, which was
    # harmless for exactly as long as every unique molecule was submitted. The
    # n_generated split turned that sort into a *selection function*: with a pool
    # larger than the budget, `sorted(...)[:n_molecules]` submits the ASCII-
    # lexicographically first molecules. Measured on a real 196-molecule
    # REINVENT4 pool (first characters C 154 / O 25 / N 12 / [ 3 / F 2), the
    # first ten were all 'C'-initial and every N-, O-, F- and bracket-initial
    # molecule was excluded. A SMILES prefix tracks scaffold identity and
    # scaffold identity is what the primary endpoint counts, so that truncation
    # would have confounded an oversampling arm's qualifying_scaffolds with an
    # alphabetical artefact and made it incomparable to the incumbent's 150.
    # Generation order is the neutral rule: it adds no selection pressure the
    # plan did not ask for, and a quality-based pick (docking-score-greedy, say)
    # would be a new scientific intervention rather than a bug fix.
    unique = list(dict.fromkeys(valid))
    # THE SELECTION. Without oversampling this slice is a no-op -- the base
    # adapter already truncates `batch.molecules` to the number requested -- so a
    # plan that names no n_generated submits exactly the molecules every campaign
    # before this one submitted. With oversampling it is the difference between
    # submitting 150 and handing the evaluator a pool of 1,900 to count scaffolds
    # over.
    #
    # THE SELECTION RULE. `generation_order` is the default and reproduces every
    # campaign before this one exactly. `cheap_gate_prefilter` removes candidates
    # already knowable to fail a gate that needs no pose -- properties, synthetic
    # accessibility, hERG -- and then takes the first n_molecules of the survivors
    # in the SAME generation order. It never reads a docking output: selecting on
    # structure_score would be submitting the molecules an expensive measurement
    # had already passed, which is a lookup and not a prediction.
    # Measured across three seeds on pocketxmol: 54/58/59 -> 107/113/113. Measured
    # on reinvent4, whose bottleneck is the docking gate at 93% failure: 5 -> 4.
    # It is conditional on the generator, not a general gain.
    #
    # `surrogate_rank` is a REGISTERED rule that this action does not implement: it
    # needs a fitted surrogate, and this action has nowhere to get one. It is
    # refused below by name rather than left to fall through, because falling
    # through would submit in generation order under a funnel naming a rule that
    # never ran -- the silent default this block exists to prevent.
    implemented = (
        submission_policy.SELECTION_GENERATION_ORDER,
        submission_policy.SELECTION_CHEAP_PREFILTER,
    )
    selection = str(sampling.get("selection", submission_policy.SELECTION_GENERATION_ORDER))
    if selection in submission_policy.SELECTION_RULES and selection not in implemented:
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"selection rule {selection!r} is registered but not implemented here",
            reason=(
                f"inputs.sampling.selection is {selection!r}, which is a registered rule in "
                f"submission_policy but one this action cannot run: it needs a fitted surrogate "
                f"and this action has none. Submitting in generation order instead would record a "
                f"funnel naming a rule that never ran"
            ),
            unblock=f"set inputs.sampling.selection to one of {list(implemented)}, or omit it",
            cost=_cost(started),
        )
    if selection not in submission_policy.SELECTION_RULES:
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"unknown selection rule {selection!r}",
            reason=(
                f"inputs.sampling.selection is {selection!r}; the registered rules are "
                f"{list(submission_policy.SELECTION_RULES)}. A rule this action does not implement "
                f"is refused rather than silently falling back to the default, because the rule "
                f"decides which molecules spend the submission budget"
            ),
            unblock=(
                f"set inputs.sampling.selection to one of "
                f"{list(submission_policy.SELECTION_RULES)}, or omit it"
            ),
            cost=_cost(started),
        )

    cheap = None
    if selection == submission_policy.SELECTION_CHEAP_PREFILTER:
        try:
            qualification = json.loads(
                (repo_root / "configs" / "research" / "zak-c9r" / "qualification.json").read_text(
                    encoding="utf-8"
                )
            )
            cheap = submission_policy.cheap_gate_table(
                [(f"cand_{i:05d}", smiles) for i, smiles in enumerate(unique)],
                qualification=qualification,
            )
        except (OSError, ValueError) as exc:
            return _result(
                "change_sampling",
                backend,
                "FAILED",
                notes=f"the cheap-gate pre-filter could not run: {type(exc).__name__}",
                reason=(
                    f"inputs.sampling.selection asked for {selection!r} and the pre-filter raised "
                    f"{type(exc).__name__}: {exc}. Submitting in generation order instead would "
                    f"silently answer a different plan than the one dispatched"
                ),
                unblock="omit inputs.sampling.selection to submit in generation order deliberately",
                cost=_cost(started),
            )
        survivors = {r["smiles"] for r in cheap["rows"] if r["passes_cheap_gates"]}
        unique = [s for s in unique if s in survivors]

    selected = unique[:n_molecules]
    # EMISSION ORDER, and nothing else: the membership of the submission is
    # already fixed above. Sorting the chosen molecules keeps a non-oversampling
    # run byte-identical to the sixteen campaigns already on the record -- same
    # rows, same order, same gen_NNNN ids -- so this fix changes *which*
    # molecules an oversampling arm submits and changes nothing at all about an
    # arm that does not oversample.
    submitted = sorted(selected)
    molecules = [{"molecule_id": f"gen_{i:04d}", "smiles": s} for i, s in enumerate(submitted)]
    funnel = {
        # ONE canonical name for the generation denominator. Sixteen campaigns
        # recorded this quantity as n_raw, n_raw_attempts, n_candidates, n_scored
        # or nothing at all, and a later analysis divided one campaign's count by
        # a generated denominator and another's by a scored one. n_raw is kept
        # below, unchanged, so the older records still read; n_generated is the
        # key anything new reads.
        "n_generated": batch.n_raw,
        "n_generated_requested": n_generated_requested,
        "n_generated_after_cap": n_generate,
        "generation_cap": cap,
        "generation_cap_source": constraints.get("source"),
        # Every constraints file the walk found, not just the one the binding
        # ceiling came from: the ceiling is the MINIMUM over all of them, so
        # "which files were in scope" is part of reading the number.
        "generation_cap_sources": constraints.get("sources"),
        "generation_cap_binding": bool(cap is not None and cap < n_generated_requested),
        "n_raw": batch.n_raw,
        "n_valid": len(valid),
        "n_unique": len(unique),
        "n_invalid": batch.n_raw - len(valid),
        "n_submission_budget": n_molecules,
        "n_submitted": len(submitted),
        # WHICH molecules the budget was spent on, by name, because with
        # oversampling that is a scientific choice and not a formatting detail.
        "selection_rule": "generation_order_first_occurrence",
        "selection_pressure": "none",
        "submission_emission_order": "canonical_smiles_ascending",
        "selection_rule_note": (
            "the submission is the first n_submission_budget DISTINCT molecules in the order the "
            "sampler emitted them, keeping the first occurrence of each duplicate; the rows are "
            "then written in canonical-SMILES order, which fixes the gen_NNNN ids but not the "
            "membership. Generation order adds no selection pressure: it is not a quality ranking "
            "and it is not the ASCII ordering an earlier revision truncated on, which would have "
            "selected on SMILES prefix and therefore on scaffold identity -- the quantity the "
            "primary endpoint counts"
        ),
        "backend": batch.backend,
        "model_id": batch.model_id,
        "model_version": batch.model_version,
        "checkpoint": checkpoint_record,
        "errors": list(batch.errors),
        "sampling": sampling,
        "note": (
            "generated, valid, unique and submitted are four numbers and stay four numbers; a "
            "policy that raises the generated count without raising the unique count has changed "
            "nothing, and the submitted count is the fixed budget the metric is defined at"
        ),
    }
    counts = (
        f"{batch.n_raw} generated / {len(valid)} valid / {len(unique)} unique / "
        f"{len(submitted)} submitted molecules"
    )
    evidence = {
        "n_generated": batch.n_raw,
        "n_raw": batch.n_raw,
        "n_valid": len(valid),
        "n_unique": len(unique),
        "n_submitted": len(submitted),
        "generation_cap": cap,
        "checkpoint": checkpoint_record,
        "backend": batch.backend,
        "model_id": batch.model_id,
    }

    # --- the ceiling, enforced against what was actually emitted ------------
    # Asking for at most `cap` is a request. This is the enforcement: a backend
    # that emitted more than the ceiling has produced a run under a selection
    # pressure nobody declared, and the funnel is written so the overrun is on
    # the record rather than inferred from a missing file.
    if cap is not None and batch.n_raw > cap:
        _write(output_dir / "funnel.json", funnel)
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"{batch.n_raw} molecules were generated under a declared ceiling of {cap}",
            reason=(
                f"this run declares max_generated={cap} (from {constraints.get('source')}), the "
                f"sampler was asked for {n_generate} and it emitted {batch.n_raw}. The ceiling is "
                f"a control on selection pressure, so a run that exceeded it is not a run at a "
                f"slightly larger budget, it is a different experiment"
            ),
            unblock=(
                "check why the backend over-produced against its requested count; the sampled "
                "molecules are not usable under this arm's declared ceiling"
            ),
            evidence=evidence,
            cost=_cost(started),
        )

    # A declared checkpoint that was never confirmed is a FAILURE, not a pass.
    # The adapter states which weights it read in ``sample_provenance``; if it
    # said nothing, or named a path that is not on disk and carried no digest,
    # ``verified`` stays UNCONFIRMED and this run has no evidence at all about
    # which generator produced these molecules. Letting that SUCCEED is precisely
    # the silent fall-back to the parent prior the whole checkpoint mechanism
    # exists to make detectable: the arm would report incumbent-shaped numbers
    # and nothing anywhere would say why. Unconfirmed is refused on the same
    # footing as a mismatch.
    if declared_checkpoint and checkpoint_record["verified"] == "UNCONFIRMED":
        _write(output_dir / "funnel.json", funnel)
        said = f"sample_provenance {provenance!r}" if provenance else "no sample_provenance at all"
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes="the sampler never confirmed which checkpoint it loaded",
            reason=(
                f"the plan named {checkpoint_record['requested']} "
                f"({checkpoint_record['requested_sha256']}) and the adapter reported {said}, "
                f"so there is no digest to compare against the requested one. An unconfirmed "
                f"load is not a verified one: a generate() that silently fell back to the parent "
                f"prior looks exactly like this, and the arm would report the incumbent's numbers "
                f"as an adapted generator's"
            ),
            unblock=(
                "have the adapter report config['sample_provenance'] with 'checkpoint' and "
                "'checkpoint_sha256' for the weights it actually loaded, or drop "
                "inputs.sampling.checkpoint and report this arm as the incumbent it is"
            ),
            evidence=evidence,
            cost=_cost(started),
        )

    if checkpoint_record["verified"] in {"MISMATCH", "RESOLVED_TO_PARENT_PRIOR"}:
        _write(output_dir / "funnel.json", funnel)
        parent = checkpoint_record["verified"] == "RESOLVED_TO_PARENT_PRIOR"
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=(
                "the checkpoint the plan named resolves to the parent prior"
                if parent
                else "the sampler loaded a different checkpoint from the one the plan named"
            ),
            reason=(
                f"the plan named {checkpoint_record['requested']} "
                f"({checkpoint_record['requested_sha256']}) and the adapter reports it loaded "
                f"{checkpoint_record['reported_by_adapter']} "
                f"({checkpoint_record['reported_sha256']}). "
                + (
                    "That is the incumbent generator, so these molecules measure the incumbent "
                    "under an arm that claims to be adapted"
                    if parent
                    else "These molecules did not come from the generator this experiment claims "
                    "to be measuring"
                )
            ),
            unblock=(
                "point inputs.sampling.checkpoint at the checkpoint an adapt_generator experiment "
                "wrote, or drop the key and report this arm as the incumbent it is"
            ),
            evidence=evidence,
            cost=_cost(started),
        )

    if not unique:
        # No scoring pass is attempted and none is described: there is nothing to
        # score, and a scoring block over zero molecules would be a record of a
        # computation that never ran.
        _write(output_dir / "molecules.json", molecules)
        _write(output_dir / "funnel.json", funnel)
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=counts,
            reason="; ".join(batch.errors) or "no molecule survived validity filtering",
            unblock="inspect the sampler's errors above",
            evidence=evidence,
            cost=_cost(started),
        )

    try:
        from drug_agent.selfevolve.target.generation_sweep import score_molecules

        scored = score_molecules(
            [(str(m["molecule_id"]), str(m["smiles"])) for m in molecules],
            scorer=scorer,
            workdir=output_dir / "scoring",
        )
    except Exception as exc:
        # A scoring pass that raised is a missing measurement with a consumed
        # cost, not a portfolio of zero qualifiers — and emphatically not a
        # reason to write the unscored rows anyway. Those rows would reach the
        # evaluator as one refused molecule after another, far from the cause.
        _write(
            output_dir / "funnel.json",
            {
                **funnel,
                "scoring": {
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "cost": None,
                    "pose_quality_status": "NOT_REACHED",
                    "n_molecules_not_scored": len(molecules),
                },
            },
        )
        return _result(
            "change_sampling",
            backend,
            "FAILED",
            notes=f"{counts}; the scoring pass raised and none of them was scored",
            reason=(
                f"score_molecules raised {type(exc).__name__}: {exc}. The {len(molecules)} "
                f"sampled molecules are unscored, so no molecules.json was written: the frozen "
                f"evaluator reads precomputed endpoint records and cannot judge rows that "
                f"carry none"
            ),
            unblock=(
                "confirm the docking and endpoint backends this host actually has "
                "(`python -m drug_agent.selfevolve.target.baselines --probe`), then re-dispatch "
                "the same plan; the sampled molecules are not recoverable from this run"
            ),
            evidence=evidence,
            cost=_cost(started),
        )

    rows = list(scored["rows"])
    scoring_cost = dict(scored.get("cost") or {})
    # The pass measures docking and the endpoint panel separately, and the two
    # terms only add up when both were measured: UNAVAILABLE means nobody looked,
    # and adding it as a zero would bill the campaign less than the pass cost.
    gpu_measured = (
        scoring_cost.get("docking_gpu_telemetry") == "MEASURED"
        and scoring_cost.get("endpoint_gpu_telemetry") == "MEASURED"
    )
    gpu_seconds = (
        round(
            float(scoring_cost.get("docking_gpu_seconds") or 0.0)
            + float(scoring_cost.get("endpoint_gpu_seconds") or 0.0),
            3,
        )
        if gpu_measured
        else None
    )
    molecules_sha256 = _write(output_dir / "molecules.json", rows)
    _write(
        output_dir / "funnel.json",
        {
            **funnel,
            "scoring": {
                "status": "SUCCEEDED",
                "reason": None,
                "cost": scoring_cost,
                "pose_quality_status": (scored.get("pose_quality") or {}).get("status"),
                "scorer": {
                    "engine": scorer.get("engine"),
                    "metric": scorer.get("metric"),
                    "receptor_sha256": scorer.get("receptor_sha256"),
                    "source": str(calibration_path),
                },
            },
        },
    )
    return _result(
        "change_sampling",
        backend,
        "SUCCEEDED",
        notes=f"{counts}, scored through the registered operating point",
        artifacts=[
            {
                "artifact_id": f"art_{plan['experiment_id']}_policy",
                "kind": str(plan["kind"]),
                "parent_id": plan.get("parent_id"),
                "entrypoint": [
                    "python",
                    "-m",
                    "drug_agent.selfevolve.target.actions",
                    "--action",
                    "change_sampling",
                    "--backend",
                    backend,
                ],
                "training_data": ["molecules.json"],
            }
        ],
        evidence={**evidence, "molecules_sha256": molecules_sha256},
        cost=_cost(
            started, gpu_seconds=gpu_seconds, gpu_telemetry="MEASURED" if gpu_measured else "UNAVAILABLE"
        ),
    )


# ---------------------------------------------------------------------------
# Distillation, acquisition and receptor preparation
# ---------------------------------------------------------------------------
def distill(plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path) -> Record:
    """Train a cheap student on a named computational teacher endpoint.

    Appendix A §6: a scorer distillation learns a *named computational
    endpoint*, never silently a measured affinity. Step 12 moved the rules into
    :mod:`drug_agent.selfevolve.target.distill` and this handler is a thin
    dispatcher over them, so the action and the module cannot drift into two
    different definitions of what a teacher row is.

    **This handler does not score anything.** It consumes a teacher table that a
    separate scoring pass produced — Step 11's campaign C found that the
    platform had no way to make one, and the answer taken was to make the
    teacher pass a first-class artifact rather than to let this action quietly
    produce its own labels. A plan naming a table that does not exist is
    answered with the exact command that builds one.
    """
    started = time.time()
    inputs = dict(plan["inputs"])
    teacher, _refusal = _object_input(inputs, "teacher", action="distill", backend=backend, started=started)
    if _refusal is not None:
        return _refusal
    endpoint_name = str(teacher.get("endpoint") or "")
    rows_path = teacher.get("table") or teacher.get("rows")
    if not endpoint_name or not rows_path:
        return _result(
            "distill",
            backend,
            "FAILED",
            notes="the teacher was not fully named",
            reason=(
                "inputs['teacher'] needs both 'endpoint' (the computational endpoint the "
                "student learns) and 'table' (the teacher table a scoring pass produced). A "
                "student whose endpoint is unnamed cannot be reported honestly"
            ),
            unblock=_TEACHER_PASS_UNBLOCK,
            cost=_cost(started),
        )
    path = _resolve(repo_root, str(rows_path))
    if not path.is_file():
        return _result(
            "distill",
            backend,
            "FAILED",
            notes=f"{path} does not exist",
            reason=(
                f"the teacher table {str(rows_path)!r} is not on disk. This action learns from a "
                f"teacher table and never produces one, so that a student's labels can always be "
                f"traced to the computation that made them"
            ),
            unblock=_TEACHER_PASS_UNBLOCK,
            cost=_cost(started),
        )

    from drug_agent.selfevolve.target.contracts import ContractError
    from drug_agent.selfevolve.target.distill import (
        BACKEND_LEARNERS,
        DEFAULT_MAX_PER_FAMILY,
        build_student_training_set,
        evaluate_student,
        load_student,
        load_teacher_table,
        train_student,
    )

    learner = BACKEND_LEARNERS.get(backend)
    if learner is None:
        return _result(
            "distill",
            backend,
            "UNAVAILABLE",
            notes=f"no learner is registered for backend {backend!r}",
            reason=f"the registered distillation backends are {sorted(BACKEND_LEARNERS)}",
            unblock="name one of the backends above",
            cost=_cost(started),
        )
    try:
        table = load_teacher_table(path)
    except ContractError as exc:
        return _result(
            "distill",
            backend,
            "FAILED",
            notes="the teacher table does not satisfy the teacher-row contract",
            reason=(
                f"[{exc.category.value}] {exc.field or 'row'}: {exc}. Every teacher row must "
                f"name its teacher/checkpoint, molecule/pose/receptor, evaluator, experiment "
                f"and data partition, and a teacher prediction may never carry an experimental "
                f"evidence label"
            ),
            unblock=_TEACHER_PASS_UNBLOCK,
            cost=_cost(started),
        )
    except (ValueError, KeyError, TypeError) as exc:
        return _result(
            "distill",
            backend,
            "FAILED",
            notes=f"the teacher table could not be read: {type(exc).__name__}",
            reason=str(exc),
            unblock=_TEACHER_PASS_UNBLOCK,
            cost=_cost(started),
        )

    seed = int(inputs.get("seed", 42))
    synthetic_weight = float(inputs.get("synthetic_weight", 1.0))
    holdout_fraction = float(inputs.get("holdout_fraction", 0.25))
    max_per_family = int(inputs.get("max_per_family", DEFAULT_MAX_PER_FAMILY))
    registry_path = repo_root / SNAPSHOT_RELATIVE / "protected" / "protected_exclusion_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else None

    from drug_agent.selfevolve.target.distill import choose_holdout_families

    rows = list(table["rows"])
    holdout_families = choose_holdout_families(rows, fraction=holdout_fraction, seed=seed)
    holdout_rows = [
        r for r in rows if str(r["family_id"]) in set(holdout_families) and r["status"] == "SUCCEEDED"
    ]
    try:
        training_set = build_student_training_set(
            rows,
            synthetic_weight=synthetic_weight,
            exclusion_registry=registry,
            max_per_family=max_per_family,
            holdout_families=holdout_families,
        )
        trained = train_student(
            training_set,
            base_config={
                "learner": learner,
                "seed": seed,
                "n_estimators": int(inputs.get("n_estimators", 300)),
                "max_depth": inputs.get("max_depth", 3),
            },
            output_dir=output_dir,
            teacher_endpoint=endpoint_name,
        )
    except ContractError as exc:
        return _result(
            "distill",
            backend,
            "FAILED",
            notes="the training set could not be assembled",
            reason=f"[{exc.category.value}] {exc.field or ''}: {exc}",
            unblock=(
                "supply a teacher table with a real-data anchor and enough scored rows; a failed "
                "backend call is a missing outcome, never a low score"
            ),
            cost=_cost(started),
        )

    student = load_student(Path(trained["student_path"]))
    evaluation = evaluate_student(
        student, holdout_rows, teacher_seconds=dict(table.get("seconds_by_molecule") or {})
    )
    _write(
        output_dir / "distillation.json",
        {
            "teacher_endpoint": endpoint_name,
            "teacher_table": str(path),
            "teacher_table_sha256": sha256_bytes(path.read_bytes()),
            "teacher_id": table["teacher"]["teacher_id"],
            "teacher_checkpoint": table["teacher"]["teacher_checkpoint"],
            "receptor_hash": table["teacher"]["receptor_hash"],
            "evaluator_hash": table.get("evaluator_hash"),
            "learner": learner,
            "synthetic_weight": synthetic_weight,
            "control_arm": training_set["control_arm"],
            "n_teacher_rows": table["n_rows"],
            "n_train_rows": trained["n_train_rows"],
            "n_anchor_rows": training_set["anchor_rows"],
            "n_synthetic_rows": training_set["synthetic_rows"],
            "boundary": training_set["boundary"],
            "dedup": training_set["dedup"],
            "diversity": training_set["diversity"],
            "holdout_families": holdout_families,
            "evaluation": evaluation,
            "student_card": trained["card"],
            "seed": seed,
            "measured_label_used": False,
            "note": (
                "the student learns the named computational endpoint above and nothing else. "
                "The agreement reported is against held-out TEACHER scores on families that were "
                "removed from every training set; no measured assay label took part in this fit "
                "or in this number, and no claim about biology follows from it"
            ),
        },
    )
    mae = evaluation.get("mae_vs_teacher")
    return _result(
        "distill",
        backend,
        "SUCCEEDED",
        notes=(
            f"student fitted on {trained['n_train_rows']} teacher rows at synthetic weight "
            f"{synthetic_weight:g}; fresh holdout MAE vs teacher "
            f"{'unavailable' if mae is None else format(mae, '.4f')}"
        ),
        artifacts=[
            {
                "artifact_id": f"art_{plan['experiment_id']}_student",
                "kind": str(plan["kind"]),
                "parent_id": plan.get("parent_id"),
                "entrypoint": [
                    "python",
                    "-m",
                    "drug_agent.selfevolve.target.actions",
                    "--action",
                    "distill",
                    "--backend",
                    backend,
                ],
                "weights": ["student.joblib"],
                "training_data": ["distillation.json"],
            }
        ],
        evidence={
            "teacher_endpoint": endpoint_name,
            "teacher_id": table["teacher"]["teacher_id"],
            "synthetic_weight": synthetic_weight,
            "student_mae_vs_teacher": mae,
            "n_train": trained["n_train_rows"],
            "n_holdout": evaluation.get("n_scored"),
            "collapsed": evaluation.get("collapse", {}).get("collapsed"),
            "max_family_share": training_set["diversity"]["max_family_share"],
        },
        cost=_cost(started),
    )


def acquire_resource(plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path) -> Record:
    """Stage one pre-declared local resource through the ingestion boundary.

    Appendix A §6: ``CONTROLLED`` workers use pre-staged resources and no direct
    network retrieval. A plan naming an ``http(s)``/``ftp`` URI is therefore
    refused *here*, with the regime named, rather than fetched — the refusal is
    the boundary, and an ``ACQUISITION_ENABLED`` campaign submits the request to
    a separate ingestion process instead of widening this one.

    Provenance is mandatory, not decorative: source, licence and trust label are
    required because a staged file with none of them cannot be reported in a
    data card and cannot be excluded from a later comparison.
    """
    started = time.time()
    inputs = dict(plan["inputs"])
    resource, _refusal = _object_input(
        inputs, "resource", action="acquire_resource", backend=backend, started=started
    )
    if _refusal is not None:
        return _refusal
    regime = str(inputs.get("resource_regime") or "CONTROLLED").upper()
    uri = str(resource.get("uri") or "")
    if uri:
        return _result(
            "acquire_resource",
            backend,
            "UNAVAILABLE",
            notes=f"a network acquisition was proposed under the {regime} regime",
            reason=(
                f"{uri!r} is a remote retrieval and this worker does not reach the network. "
                f"Appendix A §6 gives CONTROLLED workers pre-staged resources only; an "
                f"ACQUISITION_ENABLED campaign submits the request to the separate ingestion "
                f"process, which deduplicates and excludes test families before anything is used"
            ),
            unblock=(
                "stage the resource locally through scripts/research/fetch_zak_sources.py (or the "
                "campaign's ingestion process) and re-propose it by path"
            ),
            cost=_cost(started),
        )
    missing = [k for k in ("identifier", "path", "source", "license", "trust_label") if not resource.get(k)]
    if missing:
        return _result(
            "acquire_resource",
            backend,
            "FAILED",
            notes=f"the resource declaration is missing {missing}",
            reason=(
                "an acquisition preserves source, licence and trust label; a staged file "
                "without them cannot appear in a data card or be excluded from a comparison"
            ),
            unblock=f"declare {missing} in inputs['resource']",
            cost=_cost(started),
        )
    path = _resolve(repo_root, str(resource["path"]))
    if not path.is_file():
        return _result(
            "acquire_resource",
            backend,
            "FAILED",
            notes=f"{path} does not exist",
            reason=f"the declared resource path {resource['path']!r} is not on disk",
            unblock="stage the file first; an absent file is not an empty resource",
            cost=_cost(started),
        )
    data = path.read_bytes()
    staged = output_dir / f"acquired_{Path(str(resource['identifier'])).name}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(data)
    record = {
        "identifier": str(resource["identifier"]),
        "source": str(resource["source"]),
        "license": str(resource["license"]),
        "trust_label": str(resource["trust_label"]),
        "resource_regime": regime,
        "source_path": str(path),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "staged_as": staged.name,
        "network_used": False,
        "publication": resource.get("publication"),
        "assay_metadata": resource.get("assay_metadata"),
        "measurement_relation": resource.get("measurement_relation"),
        "note": (
            "staged from a local, pre-declared path. No network call was made by this "
            "process; DRUGAGENT_ALLOW_NETWORK is "
            f"{os.environ.get('DRUGAGENT_ALLOW_NETWORK', 'unset')!r} in this child"
        ),
    }
    _write(output_dir / "acquisition.json", record)
    return _result(
        "acquire_resource",
        backend,
        "SUCCEEDED",
        notes=f"staged {len(data)} bytes from {path} with full provenance",
        artifacts=[
            {
                "artifact_id": f"art_{plan['experiment_id']}_resource",
                "kind": str(plan["kind"]),
                "parent_id": plan.get("parent_id"),
                "entrypoint": [
                    "python",
                    "-m",
                    "drug_agent.selfevolve.target.actions",
                    "--action",
                    "acquire_resource",
                    "--backend",
                    backend,
                ],
                "training_data": [staged.name, "acquisition.json"],
            }
        ],
        evidence=record,
        cost=_cost(started),
    )


def prepare_receptor(plan: Mapping[str, Any], *, backend: str, output_dir: Path, repo_root: Path) -> Record:
    """Cut a receptor pocket and refuse one that is not this target.

    The defect this refusal exists for is in the repository's own history: two
    runs generated, scored and ACCEPTed molecules "for ZAK" against a soluble
    epoxide hydrolase pocket. Every residue of the cut pocket is looked up in the
    target manifest's author-numbered residue map, and a pocket below the
    declared identity threshold is a failure with the measured identity in it,
    not a warning.
    """
    from drug_agent.selfevolve.target.data import pocket_matches_target
    from drug_agent.tools.pockets._pdb import cut_pocket, parse_atoms, write_pocket_pdb

    started = time.time()
    inputs = dict(plan["inputs"])
    receptor, _refusal = _object_input(
        inputs, "receptor", action="prepare_receptor", backend=backend, started=started
    )
    if _refusal is not None:
        return _refusal
    pdb_path = _resolve(repo_root, str(receptor.get("pdb") or (SNAPSHOT_RELATIVE / "rcsb" / "6JUU.pdb")))
    if not pdb_path.is_file():
        return _result(
            "prepare_receptor",
            backend,
            "FAILED",
            notes=f"{pdb_path} does not exist",
            reason="the declared structure is not on disk",
            unblock="fetch the deposition with scripts/research/fetch_zak_sources.py --allow-network",
            cost=_cost(started),
        )
    manifest_path = repo_root / SNAPSHOT_RELATIVE / "derived" / "target_manifest.json"
    if not manifest_path.is_file():
        return _result(
            "prepare_receptor",
            backend,
            "FAILED",
            notes="no target manifest is available to check the pocket against",
            reason=(
                "a pocket with no identity check is exactly the defect this action exists for; "
                "an absent check is never a pass"
            ),
            unblock="build the target manifest with scripts/research/build_zak_target_data.py",
            cost=_cost(started),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comp_id = str(receptor.get("ligand_comp_id") or manifest.get("reference_ligand", {}).get("comp_id") or "")
    chain = str(receptor.get("chain") or manifest.get("structure", {}).get("chain_id") or "A")
    radius = float(receptor.get("radius", 10.0))

    text = pdb_path.read_text(encoding="utf-8", errors="replace")
    site_points = [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in text.splitlines()
        if line.startswith("HETATM") and line[17:20].strip() == comp_id and line[21].strip() in (chain, "")
    ]
    if not site_points:
        return _result(
            "prepare_receptor",
            backend,
            "FAILED",
            notes=f"no {comp_id!r} atoms on chain {chain!r} in {pdb_path.name}",
            reason="the reference ligand defines the site; without it there is no pocket to cut",
            unblock="name the ligand component id actually present in the deposition",
            cost=_cost(started),
        )
    cut = cut_pocket(parse_atoms(text), site_points=site_points, radius=radius)
    pocket_path = output_dir / "receptor_pocket.pdb"
    pocket_path.parent.mkdir(parents=True, exist_ok=True)
    pocket_path.write_text(
        write_pocket_pdb(
            cut,
            provenance={
                "source": pdb_path.name,
                "chain": chain,
                "ligand": comp_id,
                "radius": radius,
                "experiment": str(plan["experiment_id"]),
            },
        ),
        encoding="utf-8",
    )
    verdict = pocket_matches_target(pocket_path, manifest)
    _write(
        output_dir / "preparation.json",
        {"cut": {"n_residues": cut.n_residues, "n_atoms": cut.n_atoms}, "identity": verdict},
    )
    if not verdict["match"]:
        return _result(
            "prepare_receptor",
            backend,
            "FAILED",
            notes=f"the cut pocket is not this target: identity {verdict['identity']}",
            reason=(
                f"{verdict['reason']}: {verdict['n_matched']}/{verdict['n_residues_checked']} residues "
                f"agree with the manifest's map, below the declared threshold {verdict['threshold']}"
            ),
            unblock="cut the pocket from the target's own deposition and chain",
            evidence={"identity": verdict},
            cost=_cost(started),
        )
    return _result(
        "prepare_receptor",
        backend,
        "SUCCEEDED",
        notes=(
            f"cut {cut.n_residues} residues / {cut.n_atoms} atoms at {radius} A around {comp_id}; "
            f"identity {verdict['identity']} against the target manifest"
        ),
        artifacts=[
            {
                "artifact_id": f"art_{plan['experiment_id']}_receptor",
                "kind": str(plan["kind"]),
                "parent_id": plan.get("parent_id"),
                "entrypoint": [
                    "python",
                    "-m",
                    "drug_agent.selfevolve.target.actions",
                    "--action",
                    "prepare_receptor",
                    "--backend",
                    backend,
                ],
                "structures": ["receptor_pocket.pdb"],
            }
        ],
        evidence={"identity": verdict, "n_residues": cut.n_residues, "n_atoms": cut.n_atoms},
        cost=_cost(started),
    )


HANDLERS = {
    "acquire_resource": acquire_resource,
    "fit_predictor": fit_predictor,
    "adapt_generator": adapt_generator,
    "distill": distill,
    "change_sampling": change_sampling,
    "prepare_receptor": prepare_receptor,
    "evaluate_development": evaluate_development,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=sorted(HANDLERS))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    # The campaign's snapshots and frozen records live in the *parent* checkout:
    # a candidate is a git worktree and `.external/` is gitignored, so a handler
    # that resolved them from its own `__file__` would find nothing. The
    # dispatcher already exports this; setting it from the argument vector too
    # means a handler run by hand behaves the same way.
    os.environ.setdefault("DRUGAGENT_REPO_ROOT", str(args.repo_root.resolve()))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = HANDLERS[args.action](
            plan, backend=args.backend, output_dir=output_dir, repo_root=args.repo_root.resolve()
        )
    except SystemExit:
        raise
    except Exception as exc:  # a crash is a recorded failure, not a silent one
        import traceback

        _write(
            output_dir / "result.json",
            _result(
                args.action,
                args.backend,
                "FAILED",
                notes=f"the handler raised {type(exc).__name__}",
                reason=f"{type(exc).__name__}: {exc}",
                unblock="read the traceback in this experiment's stderr log",
            ),
        )
        traceback.print_exc()
        return 1
    _write(output_dir / "result.json", result)
    return 0 if result["status"] == "SUCCEEDED" else 0 if result["status"] == "UNAVAILABLE" else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
