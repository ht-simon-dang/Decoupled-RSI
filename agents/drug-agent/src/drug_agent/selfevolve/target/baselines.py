"""Manifest-driven reference baselines and per-verb scientific capability probes.

Step 8 of the per-target evolution program. Three things live here, and they are
deliberately separate:

**The baseline manifest contract.** A baseline is not a function call that
somebody remembered to make with the right arguments; it is a committed record
naming an argument-vector entry point, the model and data content addresses it
consumes, its preparation settings, its *declared* tuning budget, the shape of
the output it promises and the capability verbs it needs. Anything a later
comparison has to hold fixed is in that record, so re-running an arm is reading a
file rather than reconstructing an argument list from a handoff.

**Per-verb capability probes.** ``sample``, ``fit``, ``resume``, ``predict`` and
``export`` are asked separately and answered separately. A working sampler does
not establish a working trainer, and the honest statuses are
``AVAILABLE_EXECUTED`` (a real computation was observed here), ``UNAVAILABLE``
(the backend did not run), ``EXECUTED_NO_YIELD`` (the backend *did* run and
returned nothing the pipeline can use) and ``NOT_TESTED``. There is still no
status meaning "probably fine" — ``EXECUTED_NO_YIELD`` is the opposite of a
hedge, and Appendix A §3.3 is why it exists. §3.3 draws two distinct obligations:
*"log backend failures as missing scientific outcomes with consumed cost"*, and
*"fixed-budget qualifying yield may be zero after failures, but this is not an
affinity score of zero"*. Those are different facts. MolPilot is the first —
its driver raised before drawing a sample. Pocket2Mol is the second — it ran for
152 s, consumed 135.8 GPU-seconds and returned four chemically valid structures
that are all below the declared heavy-atom floor. Giving both the token
``UNAVAILABLE`` would let a later step read a working backend that spent real GPU
time as an absent one, which is the same class of error as a silent zero.

**The contamination check.** ``tests/conftest.py`` installs an *autouse,
session-wide* fixture that monkeypatches
:class:`~drug_agent.tools.pfm_tool.PFMGeneratorAdapter` to
``tests/fakes.py::StubGenerator`` at two binding sites, so no test in this
repository has ever exercised real PFM sampling. Any lane that claims a real
backend must prove the substitution is not in force.
:func:`generator_substitution_check` is that proof, and it is written to fail
loudly: it reports ``contaminated=True`` *inside* the suite, where the fixture is
live, and ``False`` only in a process the fixture never reached.

Nothing in this module consults an LLM. :func:`run_baseline_manifest` records
whether the arm subprocess imported one, so "no LLM was involved" is an observed
property of the run rather than a promise in a docstring.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...tools._ext.device import kernel_report as device_kernel_report
from .contracts import CONTRACT_VERSION, ContractError, ErrorCategory

Record = dict[str, Any]

_CONTRACT = "BaselineManifest"

#: Where an arm's output directory is substituted into the manifest's argv. The
#: placeholder is mandatory: an entry point with nowhere to write cannot be
#: re-run by the study driver, and discovering that at execution time rather
#: than at validation time wastes the run.
OUTPUT_PLACEHOLDER = "{output}"

#: Appendix A §4's arm identifiers. B3/B4/B5 are controller arms owned by later
#: steps and are not reference baselines, so they are not registered here.
BASELINE_ARMS: tuple[str, ...] = ("B0", "B1", "B2", "B6")

#: Appendix A §3.1's registered tracks.
BASELINE_TRACKS: tuple[str, ...] = ("A", "B", "C")

#: The verbs a scientific adapter is probed for, one at a time.
CAPABILITY_VERBS: tuple[str, ...] = ("sample", "fit", "resume", "predict", "export")

#: Probe outcomes. ``AVAILABLE_EXECUTED`` is reserved for a real computation that
#: was observed on this host **and** produced usable output; nothing reaches it by
#: import success alone. ``EXECUTED_NO_YIELD`` is its near neighbour and is not a
#: softer version of it: the backend ran, it consumed a real cost, and what came
#: back does not pass the pipeline's own screening. See the module docstring.
CAPABILITY_STATUSES: tuple[str, ...] = (
    "AVAILABLE_EXECUTED",
    "EXECUTED_NO_YIELD",
    "UNAVAILABLE",
    "NOT_TESTED",
)

#: The statuses that mean "this verb cannot be relied on here". Both carry a
#: reason and an unblock; they differ only in whether the backend ran at all, and
#: every gate in this repository must refuse both. Kept as one tuple so a caller
#: cannot accidentally admit ``EXECUTED_NO_YIELD`` by testing for the string
#: ``"UNAVAILABLE"``.
#:
#: **Every gate now consults this tuple, including the dispatcher's.** Step 15A
#: had to leave ``worker._capability_support`` testing the literal
#: ``"UNAVAILABLE"``, because ``worker.py`` is inside
#: :func:`drug_agent.selfevolve.evaluation.integrity._extra_guarded_files` and
#: editing it moves the evaluator's content address, which Step 15A was not
#: authorised to do — measured: the one-line change failed
#: ``test_the_committed_zak_protocol_locks_against_the_evaluator_this_checkout_measures``
#: and two sibling frozen-record tests. Step 15C, which registers ``zak-c9r-2``
#: and re-freezes deliberately, completed it: ``_capability_support`` now maps a
#: probed-and-empty backend to its own ``EXECUTED_NO_YIELD`` support token
#: instead of to ``UNKNOWN``, and reports it in ``backends_executed_without_yield``.
#: The Step 15A boundary is kept anyway rather than retired — no capability
#: reachable from ``worker.ACTIONS`` emits ``EXECUTED_NO_YIELD``, and
#: ``test_no_action_backend_can_emit_the_new_status_token`` still fails if that
#: stops being true — because a second, independent guard on the same wiring
#: costs nothing and the day it fires is the day somebody should look.
UNUSABLE_STATUSES: tuple[str, ...] = ("UNAVAILABLE", "EXECUTED_NO_YIELD")

#: Every key a baseline manifest must carry.
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "contract_version",
    "baseline_id",
    "arm",
    "track",
    "description",
    "entrypoint",
    "model_hashes",
    "data_hashes",
    "preparation",
    "tuning_budget",
    "output_schema",
    "required_capabilities",
    "deviations",
)

#: Optional keys a manifest may carry. Anything outside the union of this and
#: :data:`MANIFEST_REQUIRED_FIELDS` is refused, so a typo in a field name cannot
#: become a silently ignored setting.
MANIFEST_OPTIONAL_FIELDS: tuple[str, ...] = (
    "notes",
    "limits",
    "upstream",
    "seed",
)

#: Declared output types the schema may name.
OUTPUT_TYPES: tuple[str, ...] = (
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
    "nullable_number",
)

#: Default local process limits for one bounded baseline probe. Step 9 owns the
#: integrated budget service; until it exists these are enforced per process and
#: recorded in a provisional ledger.
DEFAULT_WALL_SECONDS = 3600.0
DEFAULT_CPU_SECONDS = 7200
DEFAULT_ADDRESS_SPACE_BYTES = 0  # 0 disables the address-space cap (CUDA maps a lot)

#: The two binding sites ``tests/conftest.py`` patches. Named here rather than
#: rediscovered, so a site that stops being patched upstream shows up as a
#: changed report instead of a silently narrower check.
GENERATOR_BINDING_SITES: tuple[str, ...] = (
    "drug_agent.agents.specialists.design_agent",
    "drug_agent.tools.pfm_tool",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _fail(
    category: ErrorCategory,
    message: str,
    *,
    field: str | None = None,
    missing: Sequence[str] | None = None,
) -> None:
    raise ContractError(category, message, contract=_CONTRACT, field=field, missing=list(missing or []))


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def pool_hash(molecule_ids: Iterable[str]) -> str:
    """A content address for an evaluation pool, independent of row order."""
    return sha256_text("\n".join(sorted(str(m) for m in molecule_ids)))


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------
def baseline_manifest_paths(directory: str | Path) -> list[Path]:
    """Every manifest under *directory*, sorted. ``index.json`` is not one."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.json") if p.name != "index.json")


def load_manifest(path: str | Path) -> Record:
    text = Path(path).read_text()
    record = json.loads(text)
    if not isinstance(record, dict):
        _fail(ErrorCategory.INVALID_TYPE, f"{path} is not a JSON object", field="manifest")
    return record


def validate_baseline_manifest(record: Mapping[str, Any]) -> Record:
    """Refuse anything a later comparison could not reproduce from this file."""
    if not isinstance(record, Mapping):
        _fail(ErrorCategory.INVALID_TYPE, "a manifest must be a mapping", field="manifest")

    missing = [f for f in MANIFEST_REQUIRED_FIELDS if f not in record]
    if missing:
        _fail(
            ErrorCategory.MISSING_FIELD,
            f"the baseline manifest is missing {missing}; every field is needed to re-run "
            f"the arm from this file alone",
            field=missing[0],
            missing=missing,
        )
    allowed = set(MANIFEST_REQUIRED_FIELDS) | set(MANIFEST_OPTIONAL_FIELDS)
    unknown = sorted(set(record) - allowed)
    if unknown:
        _fail(
            ErrorCategory.UNKNOWN_FIELD,
            f"the baseline manifest declares {unknown}, which no contract knows; a key "
            f"nothing reads is a setting that silently does not apply",
            field=unknown[0],
        )

    if str(record["contract_version"]) != CONTRACT_VERSION:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"contract_version {record['contract_version']!r} is not {CONTRACT_VERSION!r}",
            field="contract_version",
        )
    baseline_id = record["baseline_id"]
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        _fail(
            ErrorCategory.EMPTY_IDENTIFIER,
            "baseline_id is blank; a baseline nothing can be addressed by cannot be cited in a comparison",
            field="baseline_id",
        )
    if str(record["arm"]) not in BASELINE_ARMS:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"arm {record['arm']!r} is not one of {list(BASELINE_ARMS)}",
            field="arm",
        )
    if str(record["track"]) not in BASELINE_TRACKS:
        _fail(
            ErrorCategory.INVALID_ENUM,
            f"track {record['track']!r} is not one of {list(BASELINE_TRACKS)}",
            field="track",
        )
    for field in ("description",):
        if not isinstance(record[field], str) or not record[field].strip():
            _fail(ErrorCategory.EMPTY_VALUE, f"{field} is blank", field=field)

    _validate_entrypoint(record["entrypoint"])
    _validate_hashes(record["model_hashes"], field="model_hashes")
    _validate_hashes(record["data_hashes"], field="data_hashes")

    if not isinstance(record["preparation"], Mapping) or not record["preparation"]:
        _fail(
            ErrorCategory.EMPTY_VALUE,
            "preparation is empty; identical data/evaluator preparation across arms is the "
            "point of recording it",
            field="preparation",
        )
    _validate_tuning_budget(record["tuning_budget"])
    _validate_output_schema(record["output_schema"])
    _validate_required_capabilities(record["required_capabilities"])

    if not isinstance(record["deviations"], list):
        _fail(
            ErrorCategory.INVALID_TYPE,
            "deviations must be a list; an empty list is the claim that there are none, and "
            "an absent key is no claim at all",
            field="deviations",
        )
    return dict(record)


def _validate_entrypoint(value: Any) -> None:
    if isinstance(value, str) or not isinstance(value, Sequence):
        _fail(
            ErrorCategory.ENTRYPOINT_NOT_ARGV,
            f"entrypoint {value!r} is not an argument vector; a shell string is parsed by a "
            f"shell, which is a different program with different quoting",
            field="entrypoint",
        )
    argv = list(value)
    if not argv:
        _fail(ErrorCategory.ENTRYPOINT_NOT_ARGV, "entrypoint is empty", field="entrypoint")
    for item in argv:
        if not isinstance(item, str) or not item.strip():
            _fail(
                ErrorCategory.ENTRYPOINT_NOT_ARGV,
                f"entrypoint element {item!r} is not a non-blank string",
                field="entrypoint",
            )
    if OUTPUT_PLACEHOLDER not in argv:
        _fail(
            ErrorCategory.ENTRYPOINT_NOT_ARGV,
            f"entrypoint does not contain {OUTPUT_PLACEHOLDER!r}, so the study driver has no "
            f"way to tell the arm where to write; an arm that chooses its own output path "
            f"cannot be re-run into a fresh directory",
            field="entrypoint",
        )


def _validate_hashes(value: Any, *, field: str) -> None:
    if not isinstance(value, Mapping):
        _fail(
            ErrorCategory.INVALID_TYPE,
            f"{field} must be a mapping of name to content address",
            field=field,
        )
    for name, address in value.items():
        if not isinstance(address, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", address):
            _fail(
                ErrorCategory.INVALID_REFERENCE,
                f"{field}[{name}] = {address!r} is not a sha256 content address; a path alone "
                f"does not pin what was loaded",
                field=field,
            )


def _validate_tuning_budget(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        _fail(
            ErrorCategory.BUDGET_UNDECLARED,
            "tuning_budget is empty; Appendix A §4 requires baseline tuning to have an "
            "explicit budget declared before tuning, so that 'we tuned until it looked "
            "respectable' is not available",
            field="tuning_budget",
        )
    numeric = {k: v for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not numeric:
        _fail(
            ErrorCategory.BUDGET_UNDECLARED,
            f"tuning_budget {dict(value)!r} declares no enforceable numeric cap",
            field="tuning_budget",
        )
    for key, amount in numeric.items():
        if amount < 0:
            _fail(
                ErrorCategory.INVALID_VALUE,
                f"tuning_budget[{key}] = {amount!r} is negative",
                field="tuning_budget",
            )


def _validate_output_schema(value: Any) -> None:
    if not isinstance(value, Mapping) or "required" not in value:
        _fail(
            ErrorCategory.MISSING_FIELD,
            "output_schema must declare a 'required' mapping of field name to type",
            field="output_schema",
            missing=["required"],
        )
    required = value["required"]
    if not isinstance(required, Mapping) or not required:
        _fail(ErrorCategory.EMPTY_VALUE, "output_schema.required is empty", field="output_schema")
    for name, kind in required.items():
        if kind not in OUTPUT_TYPES:
            _fail(
                ErrorCategory.INVALID_ENUM,
                f"output_schema.required[{name}] = {kind!r} is not one of {list(OUTPUT_TYPES)}",
                field="output_schema",
            )


def _validate_required_capabilities(value: Any) -> None:
    if not isinstance(value, list):
        _fail(
            ErrorCategory.INVALID_TYPE,
            "required_capabilities must be a list",
            field="required_capabilities",
        )
    for entry in value:
        if not isinstance(entry, Mapping):
            _fail(
                ErrorCategory.INVALID_TYPE,
                f"required_capabilities entry {entry!r} is not a "
                f"{{capability_id, verb}} pair; a bare capability id does not say which verb "
                f"the arm needs, and a working sampler is not a working trainer",
                field="required_capabilities",
            )
        if not str(entry.get("capability_id") or "").strip():
            _fail(
                ErrorCategory.EMPTY_IDENTIFIER,
                "required_capabilities entry has a blank capability_id",
                field="required_capabilities",
            )
        if str(entry.get("verb")) not in CAPABILITY_VERBS:
            _fail(
                ErrorCategory.INVALID_ENUM,
                f"required_capabilities verb {entry.get('verb')!r} is not one of {list(CAPABILITY_VERBS)}",
                field="required_capabilities",
            )


def validate_output(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> Record:
    """Check an arm's result against the shape its manifest promised."""
    if not isinstance(payload, Mapping):
        _fail(ErrorCategory.INVALID_TYPE, "the arm result is not a JSON object", field="result")
    required = dict(schema["required"])
    missing = [name for name in required if name not in payload]
    if missing:
        _fail(
            ErrorCategory.MISSING_FIELD,
            f"the arm result is missing {missing}, which its own manifest promised",
            field=missing[0],
            missing=missing,
        )
    checkers: dict[str, Any] = {
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "nullable_number": lambda v: v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)),
    }
    for name, kind in required.items():
        if not checkers[kind](payload[name]):
            _fail(
                ErrorCategory.INVALID_TYPE,
                f"the arm result field {name!r} is {type(payload[name]).__name__}, not {kind}",
                field=name,
            )
    return dict(payload)


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------
def generator_substitution_check() -> Record:
    """Is this process's generator the production class, or the suite's stub?

    Answers by *identity*, not by name: ``tests/conftest.py`` replaces the bound
    attribute with a ``lambda`` returning ``StubGenerator``, which still answers
    to ``PFMGeneratorAdapter`` at the call site. Comparing the bound object with
    the class the defining module exports catches that; comparing strings does
    not.
    """
    import importlib

    from drug_agent.tools import pfm_tool

    real = pfm_tool.PFMGeneratorAdapter
    # The defining module's own attribute may itself be patched, so recover the
    # class the class object knows itself to be.
    defining = f"{real.__module__}.{real.__qualname__}" if isinstance(real, type) else None

    sites: list[Record] = []
    for module_name in GENERATOR_BINDING_SITES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - a missing site is a real finding
            sites.append(
                {
                    "module": module_name,
                    "substituted": True,
                    "bound": None,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        bound = getattr(module, "PFMGeneratorAdapter", None)
        if not isinstance(bound, type):
            sites.append(
                {
                    "module": module_name,
                    "substituted": True,
                    "bound": f"{type(bound).__name__}:{bound!r}",
                    "detail": (
                        "the bound name is not a class, so a factory has been installed over "
                        "the production adapter"
                    ),
                }
            )
            continue
        substituted = bound.__qualname__ != "PFMGeneratorAdapter"
        sites.append(
            {
                "module": module_name,
                "substituted": bool(substituted),
                "bound": f"{bound.__module__}.{bound.__qualname__}",
                "detail": (
                    f"the bound class is {bound.__module__}.{bound.__qualname__}"
                    if substituted
                    else "the bound class is the production adapter"
                ),
            }
        )

    fakes_imported = "fakes" in sys.modules
    pytest_running = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
    stub_subclass = False
    stub_module = sys.modules.get("fakes")
    if stub_module is not None and isinstance(real, type):
        stub = getattr(stub_module, "StubGenerator", None)
        stub_subclass = isinstance(stub, type) and issubclass(stub, real)

    contaminated = any(site["substituted"] for site in sites) or fakes_imported
    reason = ""
    if contaminated:
        offenders = [s["module"] for s in sites if s["substituted"]]
        reason = (
            "the production generator is substituted in this process"
            + (f" at {offenders}" if offenders else "")
            + ("; tests/fakes.py is imported" if fakes_imported else "")
            + ". A molecule produced here is a fixture's output and must never be recorded "
            "as a real-backend result."
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "contaminated": bool(contaminated),
        "sites": sites,
        "defining_class": defining,
        "fakes_module_imported": bool(fakes_imported),
        "stub_subclasses_production_adapter": bool(stub_subclass),
        "pytest_in_process": bool(pytest_running),
        "reason": reason,
    }


def portfolio_status(*, n_evaluated: int, n_scoring_failed: int) -> str:
    """Did a Track B arm produce a measurement, or did its scorer simply not run?

    A qualifying-scaffold count of zero is a legitimate fixed-budget yield when
    the molecules were actually scored and did not clear the gates. It is NOT a
    legitimate result when *every* molecule failed to score: that is a broken
    scorer wearing a research number's clothes, and this step hit exactly that —
    a relative receptor path made Open Babel convert nothing, and all 24
    molecules came back as scoring failures in 0.05 s each while the arm still
    reported SUCCEEDED. The distinction is made here so both arms of it are
    testable without docking anything.
    """
    if n_evaluated <= 0:
        return "INSUFFICIENT_COVERAGE"
    if n_scoring_failed >= n_evaluated:
        return "INSUFFICIENT_COVERAGE"
    return "SUCCEEDED"


def require_uncontaminated(what: str) -> Record:
    """Raise unless this process's generator is the production class."""
    report = generator_substitution_check()
    if report["contaminated"]:
        raise ContaminatedLaneError(f"{what}: {report['reason']}")
    return report


class ContaminatedLaneError(RuntimeError):
    """A real-backend lane found the suite's generator substitution in force."""


# ---------------------------------------------------------------------------
# Capability probes — one verb at a time
# ---------------------------------------------------------------------------
def _probe_record(
    capability_id: str,
    verb: str,
    status: str,
    *,
    evidence_class: str,
    evidence: str,
    seconds: float,
    reason: str | None = None,
    unblock: str | None = None,
    source_ids: Mapping[str, Any] | None = None,
) -> Record:
    if status not in CAPABILITY_STATUSES:  # pragma: no cover - programming error
        raise ValueError(f"{status!r} is not one of {CAPABILITY_STATUSES}")
    return {
        "contract_version": CONTRACT_VERSION,
        "capability_id": capability_id,
        "verb": verb,
        "status": status,
        "evidence_class": evidence_class,
        "evidence": evidence,
        "seconds": round(float(seconds), 3),
        "reason": reason,
        "unblock": unblock,
        "source_ids": dict(source_ids or {}),
    }


def probe_capability(capability_id: str, verb: str, *, workdir: str | Path | None = None) -> Record:
    """Ask one backend for one verb and report only what was observed.

    Every probe is a real, deliberately tiny computation or an explicit
    unavailability. Nothing is inferred from another verb's answer: that is the
    whole reason the function takes a verb.
    """
    if verb not in CAPABILITY_VERBS:
        raise ValueError(f"{verb!r} is not one of {CAPABILITY_VERBS}")
    probe = _PROBES.get((capability_id, verb))
    if probe is None:
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="no probe is registered for this capability/verb pair",
            seconds=0.0,
            reason=(
                f"{capability_id} declares no {verb} probe here, so this host has no evidence "
                f"either way and must not assume one"
            ),
            unblock=f"register a {verb} probe for {capability_id} in target/baselines.py",
        )
    started = time.time()
    try:
        return probe(capability_id, verb, Path(workdir) if workdir else None)
    except Exception as exc:  # a probe that crashes is an unavailability with a reason
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence=f"the probe raised {type(exc).__name__}",
            seconds=time.time() - started,
            reason=f"{type(exc).__name__}: {exc}",
            unblock="read the reason above; a probe failure is an infrastructure finding",
        )


def _zak_pocket() -> Path:
    return _repo_root() / ".external" / "research" / "zak-c9r" / "derived" / "zak_6juu_C9R_pocket10.pdb"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


#: The screening window every probe applies, stated rather than inherited.
#: These are :meth:`GenerativeToolAdapter.generate`'s own defaults, and the sweep
#: will run under them, so a probe that silently took a different pair would
#: measure a yield the sweep could not reproduce. Naming them here also puts them
#: in the record, which is what makes a zero yield readable: Pocket2Mol's samples
#: at this pocket are chemically valid and 6-9 heavy atoms, so the floor is the
#: reason it returns nothing, and a reader can see that without re-running it.
PROBE_MIN_HEAVY_ATOMS = 10
PROBE_MAX_HEAVY_ATOMS = 70

#: One seed, declared, for every generator probe. These numbers are throughput,
#: not quality, so a seed sweep would buy nothing; fixing it makes the cost
#: comparable between two runs of this survey.
PROBE_SEED = 42


def generation_funnel(batch: Any, n_returned: int) -> Record:
    """The whole funnel, with every stage's loss named rather than implied.

    Appendix A §3.3 requires invalid proposals and duplicates to be counted, and
    the arithmetic only closes if each stage is reported: ``n_raw`` samples, of
    which ``n_invalid`` did not parse, leaving ``n_valid``; of those
    ``n_duplicate`` repeated a structure already seen, leaving ``n_distinct``; of
    those ``n_screened_out`` fell outside the heavy-atom window, leaving
    ``n_returned``.

    ``n_returned`` is deliberately the numerator every throughput rate is built
    on, because it is what the expensive pipeline would actually receive. Reading
    ``n_distinct`` as the yield would have made Pocket2Mol look like a working arm
    producing four molecules, when what it produced was four molecules the
    pipeline rejects.
    """
    distinct = int(getattr(batch, "n_unique", 0) or 0)
    return {
        "n_requested": batch.n_requested,
        "n_raw": batch.n_raw,
        "n_valid": batch.n_valid,
        "n_invalid": max(0, int(batch.n_raw) - int(batch.n_valid)),
        "n_distinct": distinct,
        "n_duplicate": max(0, int(batch.n_valid) - distinct),
        "n_screened_out": max(0, distinct - int(n_returned)),
        "n_returned": int(n_returned),
        "validity": batch.validity,
        "uniqueness": batch.uniqueness,
        "min_heavy_atoms": PROBE_MIN_HEAVY_ATOMS,
        "max_heavy_atoms": PROBE_MAX_HEAVY_ATOMS,
        "warnings": list(batch.warnings)[:5],
        "errors": list(batch.errors)[:5],
    }


def _probe_pfm_sample(capability_id: str, verb: str, workdir: Path | None) -> Record:
    from drug_agent.tools.pfm_tool import PFMGeneratorAdapter, resolve_pfm_paths

    require_uncontaminated("PFM sample probe")
    adapter = PFMGeneratorAdapter()
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="PFMPaths.missing() is non-empty",
            seconds=0.0,
            reason="; ".join(adapter._missing),
            unblock="make pfm-install, then set DRUGAGENT_PFM_ROOT/DRUGAGENT_PFM_CHECKPOINT",
        )
    pocket = _zak_pocket()
    if not pocket.is_file():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the ZAK pocket file is absent",
            seconds=0.0,
            reason=f"{pocket} is missing",
            unblock="run scripts/research/build_zak_target_data.py",
        )
    started = time.time()
    batch = adapter.generate(
        n_molecules=2,
        pocket={"pdb_path": str(pocket)},
        seed=PROBE_SEED,
        batch_size=2,
        # Stated, not inherited: these are ``generate()``'s own defaults, and
        # writing them here is what lets the funnel record the window it applied.
        min_heavy_atoms=PROBE_MIN_HEAVY_ATOMS,
        max_heavy_atoms=PROBE_MAX_HEAVY_ATOMS,
        max_runtime_seconds=900,
    )
    seconds = time.time() - started
    smiles = [getattr(m, "smiles", "") for m in batch.molecules]
    if not any(smiles):
        # The backend ran. That is a different fact from "the backend is absent",
        # and Appendix A §3.3 requires both to be logged as themselves.
        # NOT `EXECUTED_NO_YIELD`, and the reason is a boundary rather than a
        # judgement. `worker._capability_support` maps a status it does not
        # recognise to UNKNOWN ("nobody asked"), which would be worse here than
        # the coarser token — and `worker.py` is inside
        # `evaluation/integrity.py::_extra_guarded_files`, so teaching it the new
        # token moves the evaluator's content address and breaks three frozen
        # records. Step 15A may not do that. PFM and REINVENT4 are the only
        # capabilities `worker.ACTIONS` reaches, so they keep the coarse token and
        # carry the distinction in `evidence_class="executed"` and the funnel
        # below; `test_no_action_backend_can_emit_the_new_status_token` pins that
        # boundary so it cannot drift without somebody noticing.
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="executed",
            evidence=f"the sampler ran and returned {batch.n_raw} raw / 0 usable molecules",
            seconds=seconds,
            reason="; ".join(batch.errors) or "no molecule survived validity filtering",
            unblock="inspect the checkpoint/pocket pairing",
            source_ids={"funnel": generation_funnel(batch, 0)},
        )
    checkpoint = batch.config.get("checkpoint")
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=(
            f"sampled {batch.n_raw} raw / {batch.n_valid} valid molecules against the ZAK "
            f"6JUU C9R pocket; first SMILES {smiles[0]!r}"
        ),
        seconds=seconds,
        source_ids={
            "backend": batch.backend,
            "model_id": batch.model_id,
            "model_version": batch.model_version,
            "checkpoint": checkpoint,
            "checkpoint_sha256": (
                sha256_file(checkpoint) if checkpoint and Path(checkpoint).is_file() else None
            ),
            "pocket": str(pocket),
            "pocket_sha256": sha256_file(pocket),
            # The same funnel the nine generator probes report, so one throughput
            # table can carry all eleven arms with the same denominators.
            "interpreter": resolve_pfm_paths().python,
            "gpu_kernel": device_kernel_report(resolve_pfm_paths().python).get("status"),
            "gpu_kernel_torch": device_kernel_report(resolve_pfm_paths().python).get("torch"),
            "conditioning": adapter.conditioning,
            "required_inputs": list(adapter.required_inputs),
            "synthesis_aware": bool(adapter.synthesis_aware),
            "funnel": generation_funnel(batch, len(smiles)),
        },
    )


def _probe_pfm_fit(capability_id: str, verb: str, workdir: Path | None) -> Record:
    """Can PFM be *trained* here? Asked apart from whether it can sample."""
    from drug_agent.tools.pfm_tool import PFMGeneratorAdapter, resolve_pfm_paths

    adapter = PFMGeneratorAdapter()
    verbs = [name for name in ("fit", "train", "finetune", "transfer_learn") if hasattr(adapter, name)]
    paths = resolve_pfm_paths()
    root = Path(paths.root) if paths.root else None
    upstream = []
    if root is not None and root.is_dir():
        upstream = sorted(
            str(p.relative_to(root))
            for pattern in ("train*.py", "scripts/train*.py", "main*.py")
            for p in root.glob(pattern)
        )
    return _probe_record(
        capability_id,
        verb,
        "UNAVAILABLE" if not verbs else "NOT_TESTED",
        evidence_class="inspected",
        evidence=(
            f"the adapter exposes {verbs or 'no training verb'}; the upstream checkout offers "
            f"{upstream or 'no discovered training entry point'}"
        ),
        seconds=0.0,
        reason=(
            "PFMGeneratorAdapter is a sampling adapter: it has no fit/train/finetune method, so "
            "target fine-tuning of the frozen parent generator is not reachable through a "
            "supported entry point on this host. A working sampler is not a working trainer."
        )
        if not verbs
        else None,
        unblock=(
            "add a training adapter over the upstream training entry point, with its own "
            "environment, data loader and checkpoint policy, then re-probe this verb"
        )
        if not verbs
        else None,
        source_ids={"pfm_root": str(root) if root else None, "checkpoint": str(paths.checkpoint or "")},
    )


def _probe_reinvent_sample(capability_id: str, verb: str, workdir: Path | None) -> Record:
    from drug_agent.tools.generation.reinvent4_tool import (
        REINVENT4_PRIOR_RELPATH,
        Reinvent4Adapter,
    )

    adapter = Reinvent4Adapter()
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock="make reinvent4-install (priors: https://doi.org/10.5281/zenodo.15641296)",
        )
    started = time.time()
    batch = adapter.generate(
        n_molecules=4,
        seed=PROBE_SEED,
        batch_size=4,
        min_heavy_atoms=PROBE_MIN_HEAVY_ATOMS,
        max_heavy_atoms=PROBE_MAX_HEAVY_ATOMS,
        max_runtime_seconds=900,
    )
    seconds = time.time() - started
    smiles = [getattr(m, "smiles", "") for m in batch.molecules]
    if not any(smiles):
        # `UNAVAILABLE` rather than `EXECUTED_NO_YIELD`, for the same boundary
        # reason as the PFM probe above: REINVENT4 is reachable from
        # `worker.ACTIONS`, and `worker.py` is guarded by the evaluator's
        # integrity manifest.
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="executed",
            evidence=f"the sampler ran and returned {batch.n_raw} raw / 0 usable molecules",
            seconds=seconds,
            reason="; ".join(batch.errors) or "no molecule survived validity filtering",
            unblock="inspect the prior file type",
            source_ids={"funnel": generation_funnel(batch, 0)},
        )
    prior = Path(adapter.ext.root or "") / REINVENT4_PRIOR_RELPATH
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=(
            f"sampled {batch.n_raw} raw / {batch.n_valid} valid molecules from the "
            f"unconditional Reinvent prior; first SMILES {smiles[0]!r}"
        ),
        seconds=seconds,
        source_ids={
            "backend": batch.backend,
            "model_id": batch.model_id,
            "model_version": batch.model_version,
            "prior": str(prior),
            "prior_sha256": sha256_file(prior) if prior.is_file() else None,
            "upstream_ref": batch.config.get("upstream_ref"),
            "interpreter": adapter.ext.python,
            "device": adapter.ext.device,
            "gpu_kernel": device_kernel_report(adapter.ext.python).get("status"),
            "gpu_kernel_torch": device_kernel_report(adapter.ext.python).get("torch"),
            "checkout": str(adapter.ext.root or ""),
            "conditioning": adapter.conditioning,
            "required_inputs": list(adapter.required_inputs),
            "synthesis_aware": bool(adapter.synthesis_aware),
            "funnel": generation_funnel(batch, len(smiles)),
        },
    )


#: Epochs and batch for the transfer-learning probe. One epoch over twelve
#: molecules is the smallest thing that is still *training*: the optimiser steps,
#: the weights move, and the checkpoint that lands is provably not the prior. It
#: is not a scientific result and is never reported as one — a real adaptation
#: reads its SMILES from
#: :func:`drug_agent.selfevolve.target.tl_data.build_training_smiles`.
_PROBE_TL_EPOCHS = 1
_PROBE_TL_BATCH_SIZE = 4


def _probe_reinvent_fit(capability_id: str, verb: str, workdir: Path | None) -> Record:
    """REINVENT4 transfer learning — the verb Appendix A §4 says to verify.

    This probe *runs the trainer*. It used to be reflection only — it looked for
    a method name, found none, and reported UNAVAILABLE with a reason — and a
    reflection probe is exactly as convincing about a trainer as a spelled method
    name is, which is not at all. Now it fine-tunes the real prior on a twelve
    molecule synthetic set and checks the thing that actually matters: that the
    checkpoint which lands is not byte-identical to the prior it started from.

    The synthetic set is :data:`_PROBE_SMILES`, the same ethanol-and-benzene
    fixture the Chemprop probe fits on, written tab-delimited and header-less
    because that is what ``read_smiles_csv_file`` defaults to reading. Every
    molecule in it is checked against the prior's own vocabulary first: upstream's
    ``validate_tokens`` raises on the first out-of-vocabulary token and nothing
    catches it, so an unchecked fixture would report a broken trainer when what
    was broken was the fixture.
    """
    import shutil
    import tempfile

    from drug_agent.selfevolve.target import tl_data
    from drug_agent.tools.generation.reinvent4_tool import (
        REINVENT4_PRIOR_RELPATH,
        Reinvent4Adapter,
    )

    adapter = Reinvent4Adapter()
    root = Path(adapter.ext.root) if adapter.ext.root else None
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock="make reinvent4-install (priors: https://doi.org/10.5281/zenodo.15641296)",
            source_ids={"checkout": str(root) if root else None},
        )

    prior = Path(root or "") / REINVENT4_PRIOR_RELPATH
    if workdir is not None:
        scratch = Path(workdir) / "reinvent4_tl"
        scratch.mkdir(parents=True, exist_ok=True)
        retained = True
    else:
        scratch = Path(tempfile.mkdtemp(prefix="drugagent_reinvent4_tl_probe_"))
        retained = False

    try:
        tokens = tl_data.prior_vocabulary_tokens(prior)
        admitted = [s for s in _PROBE_SMILES if tl_data.admissible_smiles(s, tokens) is not None]
        if len(admitted) != len(_PROBE_SMILES):
            return _probe_record(
                capability_id,
                verb,
                "UNAVAILABLE",
                evidence_class="executed",
                evidence=(
                    f"{len(admitted)} of {len(_PROBE_SMILES)} probe molecules are inside the "
                    f"prior's {len(tokens)}-token vocabulary"
                ),
                seconds=0.0,
                reason=(
                    "the probe fixture itself is out of vocabulary, so a transfer-learning run "
                    "would abort inside validate_tokens before training and say nothing about "
                    "whether the trainer works"
                ),
                unblock="fix _PROBE_SMILES, not the trainer",
                source_ids={"prior": str(prior), "n_vocabulary_tokens": len(tokens)},
            )

        smiles_file = scratch / "probe_train.smi"
        # Tab-delimited, no header: read_smiles_csv_file's own defaults. A header
        # line here would be ingested as a molecule.
        smiles_file.write_text("".join(f"{s}\n" for s in _PROBE_SMILES))
        checkpoint_path = scratch / "probe_tl.chkpt"

        started = time.time()
        try:
            result = adapter.fit(
                smiles_file=smiles_file,
                output_model_file=checkpoint_path,
                num_epochs=_PROBE_TL_EPOCHS,
                batch_size=_PROBE_TL_BATCH_SIZE,
                device=adapter.ext.device,
            )
        except Exception as exc:  # the backend ran and failed; say which one and why
            seconds = time.time() - started
            return _probe_record(
                capability_id,
                verb,
                "UNAVAILABLE",
                evidence_class="executed",
                evidence=(
                    f"transfer learning was launched over {len(_PROBE_SMILES)} synthetic "
                    f"molecules and raised after {seconds:.1f}s"
                ),
                seconds=seconds,
                reason=f"{type(exc).__name__}: {exc}",
                unblock=(
                    "read the driver's own REINVENT4 log named in the error; the run mode is "
                    "reinvent/runmodes/TL and the driver is scripts/external/reinvent4_tl.py"
                ),
                source_ids={
                    "prior": str(prior),
                    "interpreter": adapter.ext.python,
                    "device": adapter.ext.device,
                    "checkout": str(root) if root else None,
                },
            )
        seconds = time.time() - started
        checkpoint = Path(result["checkpoint"])
        checkpoint_bytes = checkpoint.stat().st_size if checkpoint.is_file() else 0

        # The WEIGHTS digest, never the file's. ``reinvent/models/meta_data.py:64``
        # appends ``time.time()`` to the checkpoint's metadata on every save, so a
        # checkpoint file's sha256 is different on every run and addresses bytes
        # nobody — at any seed, on any machine — can produce a second time.
        # Measured 2026-09-17: four runs of this trainer gave four file digests and
        # one single weights digest, and a 1e-6 perturbation of one tensor changes
        # that weights digest while re-saving identical tensors under fresh metadata
        # does not. So the weights digest is a content digest and the file digest is
        # a timestamp; a probe row is evidence only if what it cites can be checked
        # again, and only one of the two can be.
        #
        # This is also the only digest that can carry the sentence it is used in.
        # "so the weights moved" is a claim about tensors, and the file digest
        # differs from the prior's whether or not one gradient step landed.
        weights_sha = result.get("weights_sha256")
        prior_weights_sha = result.get("input_model_weights_sha256")
        if weights_sha and prior_weights_sha:
            moved = (
                f"weights sha256 {str(weights_sha)[:16]}… — not the prior's "
                f"{str(prior_weights_sha)[:16]}…, so the weights moved"
            )
        else:
            # The driver could not read the tensor dict out of one of the two
            # checkpoints. Say so; do not substitute the file digest, which would
            # read as the same claim while meaning something weaker.
            moved = (
                "no weights digest was available from the driver, so this row does not "
                "claim the weights moved — only that the trainer ran"
            )

        return _probe_record(
            capability_id,
            verb,
            "AVAILABLE_EXECUTED",
            evidence_class="executed",
            evidence=(
                f"transfer-learned the Reinvent prior for {result['epochs']} epoch on "
                f"{result['n_training_smiles']} SYNTHETIC molecules in {seconds:.1f}s and wrote "
                f"{checkpoint.name} ({checkpoint_bytes} bytes), {moved}"
            ),
            seconds=seconds,
            source_ids={
                "checkpoint": str(checkpoint),
                "checkpoint_weights_sha256": weights_sha,
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_retained": retained,
                "input_model": str(prior),
                "input_model_sha256": result["input_model_sha256"],
                "input_model_weights_sha256": prior_weights_sha,
                "digest_note": (
                    "checkpoint_weights_sha256 covers the tensors alone and reproduces; the "
                    "checkpoint FILE's sha256 is deliberately not recorded here because "
                    "meta_data.py:64 stamps time.time() into every save, so it addresses bytes "
                    "that cannot be regenerated. input_model_sha256 is a file digest and is "
                    "kept: the prior is a fixed artifact on disk that nothing rewrites"
                ),
                "n_training_smiles": result["n_training_smiles"],
                "epochs": result["epochs"],
                "batch_size": _PROBE_TL_BATCH_SIZE,
                "wall_seconds": result["wall_seconds"],
                "gpu_seconds": result["gpu_seconds"],
                "interpreter": adapter.ext.python,
                "device": adapter.ext.device,
                "gpu_kernel": device_kernel_report(adapter.ext.python).get("status"),
                "checkout": str(root) if root else None,
                "upstream_ref": adapter.spec.upstream_ref,
                "driver": "scripts/external/reinvent4_tl.py",
                "fixture": (
                    "SYNTHETIC probe set; proves the trainer runs, never a scientific result. "
                    "A real adaptation draws its SMILES from tl_data.build_training_smiles, "
                    "which screens them through the Step 6 exclusion boundary first."
                ),
            },
        )
    finally:
        if not retained:
            shutil.rmtree(scratch, ignore_errors=True)


def _chemprop_adapter(workdir: Path | None) -> Any:
    from drug_agent.tools.properties.chemprop_tool import ChempropAdapter

    if workdir is not None:
        return ChempropAdapter(checkpoint_dir=workdir / "chemprop")
    return ChempropAdapter()


#: A small, cheap, unremarkable ligand for the docking readiness probe. Chosen to
#: be fast rather than interesting: the probe answers "does the scorer run here",
#: and nothing about this molecule is a ZAK claim.
PROBE_LIGAND_SMILES = "NC(=O)c1ccccc1"

#: A deliberately tiny, chemically trivial fit set for the capability probes. It
#: is synthetic and labelled synthetic: it proves the trainer runs, and it is
#: never a scientific result. The real Chemprop baseline is fitted on the
#: permitted ZAK partition by its own manifest.
_PROBE_SMILES: tuple[str, ...] = (
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCCc1ccccc1",
    "CC(=O)O",
    "CCC(=O)O",
    "CCCC(=O)O",
    "CCCCC(=O)O",
)
_PROBE_LABELS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0)


def _probe_chemprop_fit(capability_id: str, verb: str, workdir: Path | None) -> Record:
    adapter = _chemprop_adapter(workdir)
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock="install chemprop into the drug-agent environment",
        )
    started = time.time()
    checkpoint = adapter.train(
        endpoint="step08_probe",
        smiles=list(_PROBE_SMILES),
        y=list(_PROBE_LABELS),
        epochs=5,
        val_fraction=0.25,
        batch_size=4,
        seed=42,
    )
    seconds = time.time() - started
    path = Path(checkpoint.path)
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=(
            f"fitted a 5-epoch D-MPNN on a 12-molecule SYNTHETIC probe set and wrote "
            f"{path.name} ({path.stat().st_size} bytes)"
        ),
        seconds=seconds,
        source_ids={
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "n_train": checkpoint.n_train,
            "fixture": "SYNTHETIC probe set; proves the trainer runs, never a scientific result",
        },
    )


def _probe_chemprop_predict(capability_id: str, verb: str, workdir: Path | None) -> Record:
    adapter = _chemprop_adapter(workdir)
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock="install chemprop into the drug-agent environment",
        )
    if adapter.checkpoint_for("step08_probe") is None:
        fit = _probe_chemprop_fit(capability_id, "fit", workdir)
        if fit["status"] != "AVAILABLE_EXECUTED":
            return _probe_record(
                capability_id,
                verb,
                "UNAVAILABLE",
                evidence_class="unavailable",
                evidence="no checkpoint exists and the fit probe did not produce one",
                seconds=0.0,
                reason=fit["reason"],
                unblock=fit["unblock"],
            )
    started = time.time()
    panels = adapter.predict(["CCO", "c1ccccc1"], endpoint="step08_probe")
    seconds = time.time() - started
    values = [p.endpoints for p in panels]
    if not any(values):
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="executed",
            evidence="predict ran and produced no endpoint value",
            seconds=seconds,
            reason="every panel came back empty",
            unblock="inspect the checkpoint's output columns",
        )
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=f"predicted {len(panels)} molecules from a reloaded checkpoint: {values}",
        seconds=seconds,
        source_ids={
            "model_id": panels[0].model_id,
            "model_version": panels[0].model_version,
            "checkpoint": str(adapter.checkpoint_for("step08_probe")),
        },
    )


def _probe_chemprop_resume(capability_id: str, verb: str, workdir: Path | None) -> Record:
    """Reload a written checkpoint and continue fitting from it.

    ``resume`` is asked separately because "the trainer runs" and "an
    interrupted fit can be picked up" are different claims, and a long baseline
    that cannot be resumed has to be re-run from zero after every interruption.
    """
    adapter = _chemprop_adapter(workdir)
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock="install chemprop into the drug-agent environment",
        )
    verbs = [name for name in ("resume", "continue_training", "fit_more") if hasattr(adapter, name)]
    existing = adapter.checkpoint_for("step08_probe")
    if verbs:
        return _probe_record(
            capability_id,
            verb,
            "NOT_TESTED",
            evidence_class="inspected",
            evidence=f"the adapter exposes {verbs}; this probe did not exercise it",
            seconds=0.0,
        )
    started = time.time()
    from chemprop import models as chemprop_models  # type: ignore[import-untyped]

    if existing is None:
        fit = _probe_chemprop_fit(capability_id, "fit", workdir)
        if fit["status"] != "AVAILABLE_EXECUTED":
            return _probe_record(
                capability_id,
                verb,
                "UNAVAILABLE",
                evidence_class="unavailable",
                evidence="no checkpoint to reload",
                seconds=0.0,
                reason=fit["reason"],
                unblock=fit["unblock"],
            )
        existing = adapter.checkpoint_for("step08_probe")
    model = chemprop_models.load_model(Path(str(existing)), multicomponent=False)
    seconds = time.time() - started
    return _probe_record(
        capability_id,
        verb,
        "UNAVAILABLE",
        evidence_class="executed",
        evidence=(
            f"the written checkpoint reloads into a live {type(model).__name__}, so warm "
            f"restart is mechanically possible, but the adapter exposes no resume verb: "
            f"train() always starts a fresh trainer and overwrites the endpoint's checkpoint"
        ),
        seconds=seconds,
        reason=(
            "ChempropAdapter has no resume/continue_training entry point. An interrupted "
            "baseline fit must be re-run from the start, and that cost is charged, not hidden."
        ),
        unblock=(
            "add a resume verb that loads the checkpoint into the Lightning trainer with "
            "ckpt_path=, or accept re-running short fits from scratch and record the cost"
        ),
        source_ids={"checkpoint": str(existing)},
    )


def _probe_chemprop_export(capability_id: str, verb: str, workdir: Path | None) -> Record:
    adapter = _chemprop_adapter(workdir)
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock="install chemprop into the drug-agent environment",
        )
    existing = adapter.checkpoint_for("step08_probe")
    if existing is None:
        fit = _probe_chemprop_fit(capability_id, "fit", workdir)
        if fit["status"] != "AVAILABLE_EXECUTED":
            return _probe_record(
                capability_id,
                verb,
                "UNAVAILABLE",
                evidence_class="unavailable",
                evidence="nothing was written, so there is nothing to export",
                seconds=0.0,
                reason=fit["reason"],
                unblock=fit["unblock"],
            )
        existing = adapter.checkpoint_for("step08_probe")
    path = Path(str(existing))
    sidecar = path.with_suffix(".json")
    started = time.time()
    payload = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    seconds = time.time() - started
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=(
            f"the fit wrote a portable checkpoint at {path.name} ({path.stat().st_size} bytes) "
            f"with a {'present' if payload else 'missing'} provenance sidecar"
        ),
        seconds=seconds,
        source_ids={
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "sidecar": str(sidecar) if sidecar.is_file() else None,
            "sidecar_keys": sorted(payload) or None,
        },
    )


def _probe_vina_predict(capability_id: str, verb: str, workdir: Path | None) -> Record:
    """Dock one small ligand against the real ZAK receptor, for real.

    This is the verb the whole of Track B rests on, so it is executed rather
    than inspected: a resolvable executable is not a working scorer, and the
    failure this probe exists to catch — the conda environment's ``bin`` not
    reaching the subprocess, so ``vina`` and ``obabel`` are invisible — produces
    an adapter that looks available and scores nothing.
    """
    import importlib.util
    import tempfile

    from drug_agent.tools.vina_tool import DockingAdapter

    adapter = DockingAdapter()
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock=(
                "put the drug-agent conda environment's bin on PATH: running that "
                "environment's python directly hides the vina and obabel executables and "
                "docking silently reports UNAVAILABLE"
            ),
        )
    spec = importlib.util.spec_from_file_location(
        "zak_redock", _repo_root() / "scripts" / "research" / "dock_zak_reference_ligand.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise RuntimeError("could not load the receptor/box preparation module")
    redock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(redock)

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="zak_dock_probe_") as tmp:
        inputs = redock.prepare_inputs(Path(tmp))
        poses = adapter.dock(
            candidate_id="step08_dock_probe",
            smiles=PROBE_LIGAND_SMILES,
            receptor_pdb=inputs["receptor_pdb"],
            receptor_id=inputs["receptor_id"],
            center=inputs["center"],
            size=inputs["size"],
            n_poses=3,
            seed=42,
            exhaustiveness=8,
        )
    seconds = time.time() - started
    scores = [float(p.score) for p in poses if p.score is not None]
    if not scores:
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="executed",
            evidence="the engine ran and returned no scored pose",
            seconds=seconds,
            reason="no pose carried a score",
            unblock="inspect the receptor preparation and the search box",
        )
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=(
            f"docked a small probe ligand into the real 6JUU box and got {len(poses)} scored "
            f"poses, best {min(scores)} kcal/mol"
        ),
        seconds=seconds,
        source_ids={
            "engine_version": poses[0].engine_version,
            "receptor_id": inputs["receptor_id"],
            "box_center": list(inputs["center"]),
            "box_size": list(inputs["size"]),
            "trust_label": str(poses[0].trust_label),
            "probe_ligand": PROBE_LIGAND_SMILES,
            "note": (
                "a pipeline readiness check. It is not an affinity claim and not a "
                "calibration: the frozen threshold comes from Step 7's docked controls."
            ),
        },
    )


# ---------------------------------------------------------------------------
# The nine generators that had no probe at all
# ---------------------------------------------------------------------------
# Step 8 registered probes for two generators. The other nineteen answered
# "no probe is registered for this capability/verb pair", which is an honest
# answer and a useless one: it is indistinguishable from "this backend is
# broken", so a plan that needed to know which generators can sample here had to
# guess. These nine are the ones with a real checkout, real weights and a real
# interpreter on this host, so for these nine the question is answerable by
# running them.
#
# Three rules hold for every one of them, and they are the reason this is a
# table rather than nine hand-written functions:
#
# *`AVAILABLE_EXECUTED` still means observed molecules.* Step 8's rule, unchanged.
#   Every one of these backends imports cleanly in its own environment; two of
#   them imported cleanly and could not launch a CUDA kernel until this step
#   repaired their interpreter lookup. Import success has never been evidence
#   here and is not now.
# * *the pocket is passed explicitly, never inherited.* `.env` still sets
#   `DRUGAGENT_PFM_POCKET` to a soluble epoxide hydrolase site from PDB 3I28 —
#   the wrong protein, and the original defect this programme was opened on. A
#   probe that let a backend fall back to a configured default would reproduce it
#   silently, so the ZAK pocket goes in as an argument and its digest is recorded
#   on the record that comes out.
# * *what the arm needs is read from the arm.* `required_inputs` is derived by
#   each adapter from its own state, so building the payload from it means a
#   generator that starts needing a reference ligand cannot be handed one that
#   does not exist without the probe noticing.
#
# They are deliberately tiny: a handful of molecules is enough to prove a
# backend executes, and anything larger would be a sweep, which Stage B has not
# authorised and which cannot be scored before the scoring rule is frozen.


def _zak_reference_ligand() -> Path:
    """The deposited C9R ligand as an SDF, written by a committed script.

    Not derived here. MolPilot needs a reference ligand's *geometry*, and a
    PDB ``HETATM`` block has no bond orders, so building one correctly needs the
    component definition's SMILES as a template — which is
    ``scripts/research/build_zak_reference_ligand.py``'s job, not a probe's.
    """
    return _repo_root() / ".external" / "research" / "zak-c9r" / "derived" / "zak_6juu_C9R_ligand.sdf"


#: The C9R centroid, needed by any task that denoises inside a box. It is not a
#: free parameter: it is the deposited ligand's own centre, and it is the number
#: already frozen as ``scorer.box_center`` in ``configs/research/zak-c9r/calibration.json``.
#: ``build_zak_reference_ligand.py`` recomputes it from the entry and reproduces
#: all three coordinates, so the two records agree by measurement rather than by
#: being copied from one another.
ZAK_POCKET_CENTER: tuple[float, float, float] = (10.708, 3.405, 10.455)


#: One generator arm: where its adapter lives, and how small its probe is.
#: ``n_molecules`` stays single-digit by design — see the note above. ``seconds``
#: is a ceiling handed to ``generate(max_runtime_seconds=...)``, which the base
#: adapter passes down to the driver subprocess, so a backend that hangs is a
#: bounded failure with a reason rather than a wedged probe.
GENERATOR_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "molecule.generate.flowmol3",
        "module": "flowmol3_tool",
        "cls": "FlowMol3Adapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.molpilot",
        "module": "molpilot_tool",
        "cls": "MolPilotAdapter",
        "n": 2,
        "batch": 2,
        "seconds": 2700.0,
    },
    {
        "id": "molecule.generate.omtra",
        "module": "omtra_tool",
        "cls": "OmtraAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.pocket2mol",
        "module": "pocket2mol_tool",
        "cls": "Pocket2MolAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.pocketflow",
        "module": "pocketflow_tool",
        "cls": "PocketFlowAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.pocketxmol",
        "module": "pocketxmol_tool",
        "cls": "PocketXMolAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.prexsyn",
        "module": "prexsyn_tool",
        "cls": "PrexSynAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.synflownet",
        "module": "synflownet_tool",
        "cls": "SynFlowNetAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
    {
        "id": "molecule.generate.targetdiff",
        "module": "targetdiff_tool",
        "cls": "TargetDiffAdapter",
        "n": 4,
        "batch": 4,
        "seconds": 1800.0,
    },
)


def _generator_adapter(entry: Mapping[str, Any]) -> Any:
    module = importlib.import_module(f"drug_agent.tools.generation.{entry['module']}")
    return getattr(module, str(entry["cls"]))()


def _generation_inputs(adapter: Any) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """The pocket payload this adapter declares it needs, and what is missing.

    Returns ``(payload, missing, provenance)``. ``missing`` is non-empty when a
    declared input does not exist on this host, and that is a complete answer:
    an arm that cannot be fed is UNAVAILABLE naming the input, not an arm that is
    fed something else.
    """
    payload: dict[str, Any] = {}
    missing: list[str] = []
    provenance: dict[str, Any] = {"conditioning": getattr(adapter, "conditioning", None)}
    required = tuple(getattr(adapter, "required_inputs", ()) or ())
    provenance["required_inputs"] = list(required)
    provenance["synthesis_aware"] = bool(getattr(adapter, "synthesis_aware", False))

    if "pocket_pdb" in required or provenance["conditioning"] == "pocket":
        pocket = _zak_pocket()
        if pocket.is_file():
            payload["pdb_path"] = str(pocket)
            payload["pdb_id"] = "6JUU"
            payload["receptor_id"] = "6JUU:A"
            provenance["pocket"] = str(pocket)
            provenance["pocket_sha256"] = sha256_file(pocket)
        else:
            missing.append(f"pocket_pdb ({pocket} is absent; run scripts/research/build_zak_splits.py)")
    if "pocket_center" in required:
        payload["center"] = list(ZAK_POCKET_CENTER)
        provenance["pocket_center"] = list(ZAK_POCKET_CENTER)
    if "ref_ligand_sdf" in required:
        ligand = _zak_reference_ligand()
        if ligand.is_file():
            payload["ref_ligand_sdf"] = str(ligand)
            provenance["ref_ligand_sdf"] = str(ligand)
            provenance["ref_ligand_sha256"] = sha256_file(ligand)
        else:
            missing.append(
                f"ref_ligand_sdf ({ligand} is absent; run "
                f"PYTHONPATH=$PWD/src python scripts/research/build_zak_reference_ligand.py)"
            )
    for other in required:
        if other not in {"pocket_pdb", "pocket_center", "ref_ligand_sdf"}:
            missing.append(f"{other} (this probe has no source for it on this host)")
    return payload, missing, provenance


def _loaded_weights(adapter: Any) -> dict[str, Any]:
    """Digest every artifact the adapter's own spec says it loads.

    Read off ``weight_relpaths`` rather than hand-listed, so a checkpoint that is
    swapped underneath a run changes the record without anybody remembering to
    update it. A directory-shaped artifact (an unpacked bundle, a building-block
    index) is reported by name and size instead of hashed: hashing a multi-GB
    tree on every probe would cost more than the probe.
    """
    spec = getattr(adapter, "spec", None)
    root = getattr(getattr(adapter, "ext", None), "root", None)
    out: dict[str, Any] = {}
    if spec is None or root is None:
        return out
    for rel in getattr(spec, "weight_relpaths", ()) or ():
        path = Path(root) / rel
        if path.is_file():
            out[rel] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        elif path.is_dir():
            entries = sorted(p.name for p in path.iterdir())
            out[rel] = {"path": str(path), "sha256": None, "n_entries": len(entries)}
        else:
            out[rel] = {"path": str(path), "sha256": None, "present": False}
    return out


def _probe_generator_sample(capability_id: str, verb: str, workdir: Path | None) -> Record:
    """Sample a handful of molecules through one generator's own ``generate()``.

    The base :class:`~drug_agent.tools.generation.GenerativeToolAdapter` owns
    parsing, de-duplication, the heavy-atom window and the funnel counters, so
    this probe adds no second sampling path and cannot disagree with the pipeline
    about what "valid" means.
    """
    entry = _GENERATOR_PROBE_BY_ID[capability_id]
    require_uncontaminated(f"{capability_id} sample probe")
    adapter = _generator_adapter(entry)
    interpreter = getattr(getattr(adapter, "ext", None), "python", None)
    device = getattr(getattr(adapter, "ext", None), "device", None)
    if not adapter.available():
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="unavailable",
            evidence="the adapter's capability card reports UNAVAILABLE",
            seconds=0.0,
            reason=adapter.profile.unavailable_reason or "unspecified",
            unblock=adapter.spec.install_hint or f"install {adapter.spec.name} and re-run this probe",
            source_ids={"interpreter": interpreter, "device": device},
        )

    payload, missing, provenance = _generation_inputs(adapter)
    if missing:
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="inspected",
            evidence=f"{adapter.spec.name} declares required_inputs {provenance['required_inputs']}",
            seconds=0.0,
            reason=f"this host has no {'; no '.join(missing)}",
            unblock="produce the input named in the reason, then re-run this probe",
            source_ids={"interpreter": interpreter, "device": device, **provenance},
        )

    kernel = device_kernel_report(interpreter) if interpreter else {"status": "UNKNOWN"}
    given = {
        "interpreter": interpreter,
        "device": device,
        "gpu_kernel": kernel.get("status"),
        "gpu_kernel_torch": kernel.get("torch"),
        "gpu_kernel_arch_list": kernel.get("arch_list"),
        "checkout": str(getattr(getattr(adapter, "ext", None), "root", "") or ""),
        "upstream_ref": getattr(adapter.spec, "upstream_ref", None),
        "weights": _loaded_weights(adapter),
        **provenance,
    }
    started = time.time()
    try:
        batch = adapter.generate(
            n_molecules=int(entry["n"]),
            pocket=payload or None,
            seed=PROBE_SEED,
            batch_size=int(entry["batch"]),
            min_heavy_atoms=PROBE_MIN_HEAVY_ATOMS,
            max_heavy_atoms=PROBE_MAX_HEAVY_ATOMS,
            max_runtime_seconds=float(entry["seconds"]),
        )
    except Exception as exc:
        # Caught here rather than in `probe_capability`'s generic handler, which
        # records the exception and nothing else. A backend that ran and failed
        # consumed a real cost against real inputs, and the record has to say which
        # inputs — the interpreter, the checkpoint digests and the pocket digest —
        # or the next worker cannot tell a broken model from a mis-fed one.
        return _probe_record(
            capability_id,
            verb,
            "UNAVAILABLE",
            evidence_class="executed",
            evidence=(
                f"{adapter.spec.name} started in {interpreter} and failed after "
                f"{time.time() - started:.1f}s without returning a batch"
            ),
            seconds=time.time() - started,
            reason=f"{type(exc).__name__}: {exc}",
            unblock=(
                f"reproduce it directly in {interpreter} against the checkout at "
                f"{given['checkout'] or '(unresolved)'}; the reason above carries the driver's own "
                f"traceback. This is a consumed cost with no scientific outcome, not a yield of zero"
            ),
            source_ids=given,
        )
    seconds = time.time() - started
    smiles = [getattr(m, "smiles", "") for m in batch.molecules]
    funnel = generation_funnel(batch, len(smiles))
    source_ids: dict[str, Any] = {
        **given,
        "backend": batch.backend,
        "model_id": batch.model_id,
        "model_version": batch.model_version,
        "funnel": funnel,
    }
    if not any(smiles):
        return _probe_record(
            capability_id,
            verb,
            # NOT `UNAVAILABLE`. This backend ran, held a CUDA context and spent
            # real seconds; what it produced does not pass the pipeline's own
            # screening. A later step reading the token must be able to tell that
            # from a backend that never started, or it will record a working model
            # with a consumed cost as an absent one.
            "EXECUTED_NO_YIELD",
            evidence_class="executed",
            evidence=(
                f"the sampler ran for {seconds:.1f}s and returned {batch.n_raw} raw / "
                f"{batch.n_valid} parseable / {funnel['n_distinct']} distinct / 0 usable molecules"
            ),
            seconds=seconds,
            reason=(
                "; ".join(batch.errors)
                or (
                    f"the backend executed and produced {funnel['n_distinct']} distinct valid "
                    f"structures, and none of them is inside the platform's declared "
                    f"{PROBE_MIN_HEAVY_ATOMS}-{PROBE_MAX_HEAVY_ATOMS} heavy-atom window "
                    f"({funnel['n_invalid']} unparseable, {funnel['n_duplicate']} duplicate, "
                    f"{funnel['n_screened_out']} screened out). This is a measured yield of zero "
                    f"from a working backend, not an absent backend."
                )
            ),
            unblock=(
                "a backend that runs and yields nothing is a consumed cost with no scientific "
                "outcome. Read the funnel in source_ids. Widening the heavy-atom window would "
                "make this arm report available and would change what every other arm's yield "
                "means, so it is a protocol decision for the operator, not a probe setting"
            ),
            source_ids=source_ids,
        )
    return _probe_record(
        capability_id,
        verb,
        "AVAILABLE_EXECUTED",
        evidence_class="executed",
        evidence=(
            f"sampled {batch.n_raw} raw / {batch.n_valid} valid / {len(smiles)} unique molecules "
            f"in {seconds:.1f}s; first SMILES {smiles[0]!r}"
        ),
        seconds=seconds,
        source_ids=source_ids,
    )


_GENERATOR_PROBE_BY_ID: dict[str, Mapping[str, Any]] = {str(e["id"]): e for e in GENERATOR_PROBES}


_PROBES: dict[tuple[str, str], Any] = {
    ("molecule.generate.pfm", "sample"): _probe_pfm_sample,
    ("molecule.generate.pfm", "fit"): _probe_pfm_fit,
    ("molecule.generate.reinvent4", "sample"): _probe_reinvent_sample,
    ("molecule.generate.reinvent4", "fit"): _probe_reinvent_fit,
    ("property.predict.chemprop", "fit"): _probe_chemprop_fit,
    ("property.predict.chemprop", "predict"): _probe_chemprop_predict,
    ("property.predict.chemprop", "resume"): _probe_chemprop_resume,
    ("property.predict.chemprop", "export"): _probe_chemprop_export,
    ("structure.dock.vina", "predict"): _probe_vina_predict,
    **{(str(e["id"]), "sample"): _probe_generator_sample for e in GENERATOR_PROBES},
}


def probe_all(pairs: Sequence[tuple[str, str]], *, workdir: str | Path | None = None) -> list[Record]:
    return [probe_capability(cap, verb, workdir=workdir) for cap, verb in pairs]


#: Every pair the committed manifests can require, probed by the study driver
#: before any arm runs.
PROBE_MATRIX: tuple[tuple[str, str], ...] = tuple(sorted(_PROBES))


# ---------------------------------------------------------------------------
# Bounded manifest execution
# ---------------------------------------------------------------------------
def _rlimits(cpu_seconds: int, address_space_bytes: int) -> Any:
    def _apply() -> None:  # pragma: no cover - runs in the forked child
        os.setsid()
        if cpu_seconds > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if address_space_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))

    return _apply


def run_baseline_manifest(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    wall_seconds: float = DEFAULT_WALL_SECONDS,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    address_space_bytes: int = DEFAULT_ADDRESS_SPACE_BYTES,
    env_extra: Mapping[str, str] | None = None,
    check_capabilities: bool = True,
    ledger: Any = None,
    budget_resource: str = "wall_seconds",
    budget_phase: str = "BASELINE_TUNING",
    budget_owner: str | None = None,
    request_key: str | None = None,
    gpu_sampling: bool = True,
) -> Record:
    """Run one validated manifest as a bounded subprocess and read its result.

    Bounded means four things at once: a wall-clock timeout, a ``RLIMIT_CPU``
    ceiling, its own process session so the timeout terminates the whole group
    rather than the direct child — the mechanism Step 4's ``worker._kill_group``
    established, reused here rather than reimplemented — and, when *ledger* is
    supplied, an atomic reservation taken **before** anything is launched.

    Step 9 owns the ledger and this function sits in front of it rather than
    beside it. With one, an arm whose wall budget does not fit the campaign's
    remaining cap never reaches ``Popen``, and what the arm actually used is
    charged back on every exit path — success, failure, timeout, and an
    exception escaping the run alike. Without one the behaviour is exactly what
    Step 8 committed, so every existing caller is unchanged.

    ``gpu_sampling`` fills the GPU column Step 8 could only record as unknown:
    ``nvidia-smi`` compute processes are polled read-only for the arm's own
    process group. Where that is not possible the column stays
    ``UNAVAILABLE``, never zero. It changes no machine setting.
    """
    if ledger is None:
        return _run_manifest(
            manifest,
            output_dir,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            address_space_bytes=address_space_bytes,
            env_extra=env_extra,
            check_capabilities=check_capabilities,
            gpu_sampling=gpu_sampling,
        )

    record = validate_baseline_manifest(manifest)
    reservation = ledger.reserve(
        budget_resource,
        wall_seconds,
        owner=budget_owner or str(record["baseline_id"]),
        phase=budget_phase,
        request_key=request_key,
        note=f"baseline arm {record['baseline_id']}",
    )
    if reservation["already_completed"]:
        # A resumed campaign must not charge or execute the same arm twice.
        return {
            "contract_version": CONTRACT_VERSION,
            "baseline_id": record["baseline_id"],
            "arm": record["arm"],
            "track": record["track"],
            "description": record["description"],
            "output_dir": str(Path(output_dir)),
            "status": "SKIPPED_ALREADY_COMPLETED",
            "exit_status": None,
            "primary_metric": None,
            "primary_value": None,
            "reason": (
                f"request_key {request_key!r} was already settled in the ledger; re-running "
                f"it would charge and execute the same work twice"
            ),
            "budget": _budget_record(reservation),
            "cost": {},
            "finished_at": time.time(),
        }

    started = time.time()
    try:
        result = _run_manifest(
            manifest,
            output_dir,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            address_space_bytes=address_space_bytes,
            env_extra=env_extra,
            check_capabilities=check_capabilities,
            gpu_sampling=gpu_sampling,
        )
    except BaseException:
        # The attempt still burned wall time. An unrecorded cost would make the
        # campaign's ledger lie about what the failure was worth.
        ledger.settle(
            reservation["reservation_id"],
            consumed=round(time.time() - started, 3),
            telemetry="MEASURED",
            status="FAILED",
            note="the arm raised out of run_baseline_manifest",
        )
        raise

    cost = dict(result.get("cost") or {})
    elapsed = float(cost.get("wall_seconds") or round(time.time() - started, 3))
    if bool(cost.get("timed_out")):
        outcome = "CANCELLED"
    elif str(result.get("status")) == "SUCCEEDED":
        outcome = "SETTLED"
    else:
        outcome = "FAILED"
    settled = ledger.settle(
        reservation["reservation_id"],
        consumed=elapsed,
        telemetry="MEASURED",
        status=outcome,
        extra_costs={
            "cpu_user_seconds": cost.get("cpu_user_seconds"),
            "cpu_system_seconds": cost.get("cpu_system_seconds"),
            "gpu_seconds": cost.get("gpu_seconds"),
        },
    )
    result["budget"] = _budget_record(settled)
    return result


def _budget_record(reservation: Mapping[str, Any]) -> Record:
    return {
        "reservation_id": reservation["reservation_id"],
        "resource": reservation["resource"],
        "phase": reservation["phase"],
        "requested": reservation["requested"],
        "reserved": reservation["reserved"],
        "consumed": reservation["consumed"],
        "released": reservation["released"],
        "state": reservation["state"],
        "outcome": reservation["outcome"],
        "telemetry": reservation["telemetry"],
        "evidence_class": reservation["evidence_class"],
    }


def _run_manifest(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    wall_seconds: float,
    cpu_seconds: int,
    address_space_bytes: int,
    env_extra: Mapping[str, str] | None,
    check_capabilities: bool,
    gpu_sampling: bool = True,
) -> Record:
    """The bounded execution itself. Budget accounting is its caller's job."""
    from .budget import _GpuSampler
    from .worker import _kill_group

    record = validate_baseline_manifest(manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    base: Record = {
        "contract_version": CONTRACT_VERSION,
        "baseline_id": record["baseline_id"],
        "arm": record["arm"],
        "track": record["track"],
        "description": record["description"],
        "manifest_hash": sha256_text(json.dumps(record, sort_keys=True)),
        "output_dir": str(out),
        "started_at": started,
    }

    probes: list[Record] = []
    if check_capabilities:
        for entry in record["required_capabilities"]:
            probes.append(probe_capability(str(entry["capability_id"]), str(entry["verb"]), workdir=out))
        blocked = [p for p in probes if p["status"] != "AVAILABLE_EXECUTED"]
        if blocked:
            return {
                **base,
                "status": "UNAVAILABLE",
                "exit_status": None,
                "primary_metric": None,
                "primary_value": None,
                "capability_probes": probes,
                "reason": "; ".join(
                    f"{p['capability_id']}/{p['verb']} is {p['status']}: {p['reason'] or p['evidence']}"
                    for p in blocked
                ),
                "unblock": "; ".join(p["unblock"] for p in blocked if p.get("unblock")),
                "cost": _zero_cost(time.time() - started),
                "finished_at": time.time(),
            }

    argv = [str(out) if token == OUTPUT_PLACEHOLDER else token for token in record["entrypoint"]]
    argv = [sys.executable if token == "{python}" else token for token in argv]
    repo = _repo_root()
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(repo / "src"))
    env["DRUGAGENT_BASELINE_ID"] = str(record["baseline_id"])
    env.update({k: str(v) for k, v in (env_extra or {}).items()})

    stdout_path = out / "stdout.log"
    stderr_path = out / "stderr.log"
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    timed_out = False
    kill_report: Record = {}
    sampler: Any = None
    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(repo),
                env=env,
                stdout=out_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                preexec_fn=_rlimits(cpu_seconds, address_space_bytes),
            )
        except OSError as exc:
            return {
                **base,
                "status": "FAILED",
                "exit_status": 127,
                "primary_metric": None,
                "primary_value": None,
                "capability_probes": probes,
                "reason": f"{type(exc).__name__}: {exc}",
                "cost": _zero_cost(time.time() - started),
                "finished_at": time.time(),
            }
        if gpu_sampling:
            # Read-only: it polls which compute processes exist, and changes no
            # clock, power limit or persistence setting on the operator's card.
            with contextlib.suppress(ProcessLookupError, OSError):
                sampler = _GpuSampler(os.getpgid(proc.pid))
                sampler.start()
        try:
            exit_status = proc.wait(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_status = 124
            kill_report = _kill_group(proc)
            proc.poll()
    gpu = sampler.stop() if sampler is not None else _no_gpu_sample()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.time() - started

    cost = {
        "wall_seconds": round(elapsed, 3),
        "cpu_user_seconds": round(after.ru_utime - before.ru_utime, 3),
        "cpu_system_seconds": round(after.ru_stime - before.ru_stime, 3),
        "max_rss_kb": int(after.ru_maxrss),
        "gpu_seconds": gpu["gpu_seconds"],
        "gpu_telemetry": gpu["telemetry"],
        "max_gpu_memory_mib": gpu["max_gpu_memory_mib"],
        "gpu_samples": gpu["samples"],
        "gpu_note": gpu["reason"],
        "retries": 0,
        "timed_out": timed_out,
    }

    result_path = out / "result.json"
    payload: Record | None = None
    parse_error: str | None = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text())
            validate_output(payload, record["output_schema"])
        except (ValueError, OSError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            payload = None

    if exit_status != 0 or payload is None:
        return {
            **base,
            "status": "FAILED",
            "exit_status": exit_status,
            "primary_metric": None,
            "primary_value": None,
            "capability_probes": probes,
            "reason": parse_error
            or (
                f"the arm exceeded its {wall_seconds}s wall budget"
                if timed_out
                else f"the arm exited {exit_status}; see {stderr_path}"
            ),
            "stderr_tail": stderr_path.read_text(errors="replace")[-4000:],
            "kill_report": kill_report,
            "cost": cost,
            "finished_at": time.time(),
        }

    return {
        **base,
        "status": str(payload.get("status") or "SUCCEEDED"),
        "exit_status": exit_status,
        "capability_probes": probes,
        "result_path": str(result_path),
        "result": payload,
        "cost": {**cost, **{k: v for k, v in (payload.get("cost") or {}).items() if k not in cost}},
        "finished_at": time.time(),
    }


def _zero_cost(elapsed: float) -> Record:
    """The cost of an arm that never launched: real wall time, nothing else run."""
    gpu = _no_gpu_sample()
    return {
        "wall_seconds": round(elapsed, 3),
        "cpu_user_seconds": 0.0,
        "cpu_system_seconds": 0.0,
        "max_rss_kb": 0,
        "gpu_seconds": gpu["gpu_seconds"],
        "gpu_telemetry": gpu["telemetry"],
        "max_gpu_memory_mib": gpu["max_gpu_memory_mib"],
        "gpu_samples": gpu["samples"],
        "gpu_note": gpu["reason"],
        "retries": 0,
        "timed_out": False,
    }


def _no_gpu_sample() -> Record:
    """Nothing was sampled, so nothing is claimed. Never a zero standing in."""
    return {
        "telemetry": "UNAVAILABLE",
        "gpu_seconds": None,
        "max_gpu_memory_mib": None,
        "samples": 0,
        "reason": "no GPU sample was taken for this arm",
    }


__all__ = [
    "BASELINE_ARMS",
    "BASELINE_TRACKS",
    "CAPABILITY_STATUSES",
    "UNUSABLE_STATUSES",
    "CAPABILITY_VERBS",
    "ContaminatedLaneError",
    "DEFAULT_CPU_SECONDS",
    "DEFAULT_WALL_SECONDS",
    "GENERATOR_BINDING_SITES",
    "MANIFEST_OPTIONAL_FIELDS",
    "MANIFEST_REQUIRED_FIELDS",
    "OUTPUT_PLACEHOLDER",
    "OUTPUT_TYPES",
    "PROBE_MATRIX",
    "baseline_manifest_paths",
    "generator_substitution_check",
    "load_manifest",
    "pool_hash",
    "portfolio_status",
    "probe_all",
    "probe_capability",
    "require_uncontaminated",
    "run_baseline_manifest",
    "sha256_file",
    "sha256_text",
    "validate_baseline_manifest",
    "validate_output",
]
