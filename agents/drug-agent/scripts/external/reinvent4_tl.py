"""Transfer-learn a REINVENT4 prior onto a SMILES file — the TL half of the bridge.

This is the sibling of :mod:`reinvent4_sample`. It runs inside REINVENT4's own
environment, so it imports nothing from ``drug_agent`` and nothing outside the
standard library except REINVENT4 itself, and it is modelled line-for-line on the
sampling driver: build a JSON config, fill the ``argparse.Namespace`` that
``reinvent/utils/cli.py`` would have produced, and call ``reinvent.Reinvent:main``
directly. No second process, no reimplementation of the run mode.

Everything below was read out of the pinned checkout (``ee0d56f4a074``, upstream
version string ``4.8.24``). Nothing here is guessed.

``the entry point``
    ``reinvent/Reinvent.py:main``. It reads ``run_type`` off the config and
    dispatches with ``getattr(runmodes, f"run_{run_type}")``, so ``run_type =
    "transfer_learning"`` selects
    ``reinvent/runmodes/TL/run_transfer_learning.py:run_transfer_learning`` —
    exported from ``reinvent/runmodes/__init__.py`` under exactly that name.

``the config``
    Top level is ``reinvent/validation.py:ReinventConfig`` and the ``parameters``
    block is ``reinvent/runmodes/TL/validation.py:SectionParameters``. Both
    inherit ``GlobalConfig``, which is ``ConfigDict(extra="forbid")``, so a key
    this driver invents is a startup error rather than a silent no-op. Five
    parameters have no default and are therefore required: ``num_epochs``,
    ``batch_size``, ``input_model_file``, ``output_model_file`` and
    ``smiles_file``.

``the scheduler block``
    Omitted, which means ``TLConfig.scheduler`` defaults to ``{}`` and
    ``run_transfer_learning`` builds
    ``reinvent/runmodes/TL/configurations.py:StepLRConfiguration()`` — Adam at
    ``lr=1e-4`` with ``StepLR(step=10, gamma=0.95)`` and a floor of ``min=1e-7``
    below which ``Learning._terminate`` stops the run. A caller that wants
    different optimisation has to say so; this driver does not invent one.

``the SMILES file``
    ``run_transfer_learning`` hands the file to
    ``reinvent/utils/config_parse.py:read_smiles_csv_file`` with ``columns=0`` for
    the unconditional ``Reinvent`` generator, and that function's defaults are
    ``delimiter="\\t"`` and ``header=False``. A header line is therefore ingested
    as if it were a molecule and a comma-separated file yields one garbage column.
    The file must be TAB-delimited with no header and SMILES in column 0;
    :mod:`drug_agent.selfevolve.target.tl_data` is what writes it.

``the vocabulary is a hard gate, not a filter``
    ``read_smiles_csv_file`` calls ``config_parse.validate_tokens`` on every row,
    and that function **raises ``ValueError`` on the first out-of-vocabulary
    token** (``config_parse.py:201-205``). Nothing upstream catches it. One
    non-conforming compound therefore aborts the whole run before a single
    gradient step, which is why the training file is pre-filtered against the
    prior's own vocabulary before it ever reaches here. This driver does not
    soften that: if it happens, the run fails loudly with REINVENT4's own message.

``which file actually lands``
    ``reinvent/runmodes/TL/learning.py:_save_model`` writes
    ``f"{output_model_file}{suffix}"`` where the suffix is ``f".{epoch}.chkpt"``
    when it is called with an epoch (``learning.py:280-281``). It is called with
    one at every ``save_every_n_epochs`` boundary *and* at the final epoch, and
    **without** one at ``learning.py:217``, at the end of every epoch. So the
    plain, unsuffixed ``output_model_file`` is rewritten each epoch and holds the
    final weights; the suffixed siblings are intermediate snapshots. This driver
    returns the unsuffixed path as ``checkpoint`` and lists the snapshots
    separately, so nobody has to guess which one is the result.

``how many molecules actually trained``
    Read out of REINVENT4's own log line ``Read N input SMILES from ...``, which
    ``run_transfer_learning`` emits with ``len(smilies)`` *after* standardisation,
    randomisation and duplicate removal. That is the only number that is the count
    REINVENT4 actually trained on; the row count of the input file is reported
    beside it as ``n_smiles_file_rows`` and is not a substitute. If the line is
    absent the driver fails rather than reporting the file's row count under the
    name of a measurement it did not take.

``where a shrinking training set goes``
    ``read_smiles_csv_file`` applies the standardiser as an *action* and then does
    ``if not smiles: continue`` (``config_parse.py:140-141``). A compound the
    ``RDKitStandardizer`` drops therefore disappears from the training set with no
    exception, no non-zero exit and nothing in the returned payload — 775 rows
    offered can become 700 trained and the caller is told 700 with no way to know
    75 went missing. Upstream does say so, but only in its own log, at WARNING:
    ``rdkit_standardizer.apply_filter`` emits ``"<filter>" filter: <smiles> is
    invalid`` for each one. This driver counts the distinct SMILES named in those
    lines and reports it as ``n_standardizer_rejected``, so the adapter can
    reconcile ``n_smiles_file_rows`` against ``n_training_smiles`` and *say which
    stage* ate the difference instead of guessing. It is a count of distinct
    SMILES rather than of log lines because an unparseable row produces the same
    warning twice (once inside ``apply_filter``'s loop and once after it).

``the seed, and what it does not reach``
    ``Reinvent.py:120`` resolves it as ``args.seed or getattr(val_config, "seed",
    None)`` and only then calls ``reinvent.utils.helpers.set_seed``, which seeds
    ``random``, ``numpy``, ``torch`` and ``torch.cuda`` and sets
    ``cudnn.deterministic``. Two consequences this driver enforces rather than
    inherits. First, ``or`` is falsy-tested, so **seed 0 is silently no seed at
    all** — it is refused here by name instead of producing an unreproducible run
    under a config that claims to be seeded. Second, ``set_seed`` assigns
    ``os.environ["PYTHONHASHSEED"]`` inside an interpreter that has already
    started, where it does nothing; the value that matters is the one the launcher
    puts in this process's environment before exec, which
    ``ExternalBackend.child_env`` does.

    **Measured, 2026-09-17: for the Reinvent generator this seed changes nothing
    about the weights that land.** Both stochastic inputs of a TL epoch are out of
    ``set_seed``'s reach. The training ``DataLoader`` is built at
    ``runmodes/TL/learning.py:151-158`` with ``generator=torch.Generator(device=
    self.device)`` — a *fresh* generator, whose initial seed is the fixed
    ``67280421310721`` and which ``torch.manual_seed`` does not touch — so
    ``shuffle_each_epoch`` shuffles identically in every run. And
    ``chemistry/conversions.py:randomize_smiles`` reaches ``MolToSmiles(...,
    doRandom=True)``, which draws from RDKit's own default-seeded generator, not
    from ``random`` or ``numpy``: the same molecule yields the same sequence of
    rewritings in a fresh process under any ``set_seed`` argument, which was
    checked directly. Four one-epoch fits of one fixture — two at seed 42, two at
    seed 7 — produced one single weights digest between them.

    **So the seed buys reproducibility and not seed-sensitivity, and a caller who
    varies it gets the same adaptation twice rather than two samples of it.** Two
    campaigns of one TL arm are consequently not independent draws of the
    adaptation: they share one checkpoint's tensors bit for bit and differ only
    downstream of it. Upstream is deliberately not patched to change that —
    ``learning.py:151`` and ``conversions.py:181`` are a vendored checkout's code,
    and manufacturing a variance by editing them would put a number on the record
    that the program never measured.

    ``ExternalBackend.child_env`` is also not the independent second mechanism it
    reads as: ``PYTHONHASHSEED`` is on its inherited allowlist and the seed is
    applied with ``setdefault``, so a parent process that already exports
    ``PYTHONHASHSEED`` keeps its own value and this run's seed never arrives. The
    parents in this program do not set it, so nothing here is wrong today; it is
    written down so it is not counted as a belt beside the braces.

    The seed is passed regardless, and that is not ceremony: it is the only
    correct thing to send, it is what makes ``num_refs``-driven ``random.sample``
    in ``learning.py:_prepare_similarity`` reproducible, and it is what will carry
    the run the day any of the above changes upstream. What it must not be called
    is the reason a checkpoint is reproducible. That reason is the weights digest
    below; the *file* is unrepeatable for a cause no seed can reach.

``GPU seconds``
    Sampled read-only from ``nvidia-smi --query-compute-apps``, at the same 0.2 s
    interval and with the same "seconds in which this job held a CUDA context"
    definition as ``drug_agent/selfevolve/target/budget.py:_GpuSampler``, so the
    two numbers are comparable. It changes no clock, power limit or persistence
    setting. Where ``nvidia-smi`` is absent or fails, ``gpu_seconds`` is ``null``
    and ``gpu_telemetry`` says ``UNAVAILABLE`` — never a zero standing in for an
    unanswered question.

    Read it as what it measures: **seconds during which this process held a CUDA
    context**, not seconds of GPU arithmetic. A run with ``device = "cpu"`` still
    reports a second or so, because ``Reinvent.main`` calls
    ``torch.cuda.is_available()`` and ``torch.cuda.device_count()`` for its
    hardware report and that initialises a context on a machine that has a card.
    The number is comparable with every other GPU-second in this repository
    because it is the same definition, and it is not a utilisation figure.

Request JSON:
    input_model_file      absolute path to the prior/checkpoint to start from
    output_model_file     absolute path the fine-tuned checkpoint is written to
    smiles_file           absolute path to the TAB-delimited, header-less file
    num_epochs            epochs to train
    batch_size            SMILES per optimiser step
    device                torch device string, e.g. "cpu" or "cuda:0"
    seed                  RNG seed, passed to ``reinvent.utils.set_seed``. Optional
                          here only so an unseeded caller fails loudly rather than
                          silently: 0 is refused, because upstream's ``args.seed or
                          ...`` would treat it as unset
    save_every_n_epochs   optional; default num_epochs, so only the final epoch
                          writes a suffixed snapshot instead of every epoch
                          writing a ~23 MB one
    sample_batch_size     optional, default 100 (SectionParameters' own default
                          and its minimum) — the report step samples this many
                          molecules at each reporting epoch
    standardize_smiles    optional, default true (upstream default)
    randomize_smiles      optional, default true (upstream default)
    isomeric_smiles       optional, default false (upstream default)
    validation_smiles_file  optional; when given, upstream reports a validation NLL
    log_level             optional, default "info" — "info" is required for the
                          ``Read N input SMILES`` line this driver parses
    out_dir               where to write result.json

``why the file's sha256 is not the checkpoint's identity``
    ``reinvent/models/meta_data.py:update_model_data`` appends ``time.time()`` to
    ``metadata["updates"]`` on **every** save (line 64) and then computes
    ``metadata["hash_id"]`` over a pickle that contains it. A wall-clock reading
    is therefore baked into every checkpoint REINVENT4 writes, so two runs that
    produce bit-identical weights still produce different files. ``sha256`` of one
    of these files can never be regenerated by anybody, at any seed — measured,
    not assumed. ``checkpoint_weights_sha256`` is reported beside it and covers the
    tensors alone; that one is reproducible, and it is the digest to quote when
    the question is which generator a batch came from.

Response JSON (``out_dir/result.json``):
    checkpoint            absolute path of the unsuffixed final checkpoint
    checkpoint_sha256     sha256 of its bytes — unrepeatable, see above
    checkpoint_weights_sha256  sha256 of its tensors alone — the repeatable one
    checkpoint_bytes      its size
    input_model_sha256    sha256 of the prior it started from
    input_model_weights_sha256  and of that prior's tensors alone
    input_model_type      the generator name read out of the input pickle
    n_training_smiles     REINVENT4's own count of what it trained on
    n_smiles_file_rows    non-blank rows in the input file, as a cross-check
    n_standardizer_rejected  distinct SMILES upstream's standardiser refused, read
                          out of its own WARNING lines; the attribution for any
                          gap between the two counts above
    seed                  the seed that was set, echoed back
    epochs / batch_size   what was asked for
    wall_seconds          wall time of the reinvent main() call
    gpu_seconds           measured GPU seconds, or null
    gpu_telemetry         "MEASURED" or "UNAVAILABLE"
    snapshots[]           the suffixed per-epoch checkpoints that were written
    reinvent_version      ``reinvent.version.__version__``
    device                the device string the config asked for
    log_path              REINVENT4's own log for this run
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The sys.path line above must run first; this directory is not a package.
import _driver

#: Same poll interval as ``budget.py:GPU_POLL_SECONDS``, so a GPU-second measured
#: here means the same thing as a GPU-second measured around a baseline arm.
GPU_POLL_SECONDS = 0.2

#: ``run_transfer_learning`` logs this immediately after ``read_smiles_csv_file``
#: returns, with ``len(smilies)``. It is the count REINVENT4 actually trained on.
_READ_SMILES_RE = re.compile(r"Read (\d+) input SMILES from ")

#: ``reinvent/chemistry/standardization/rdkit_standardizer.py:apply_filter`` emits
#: exactly this, at WARNING, for every SMILES its filter chain drops — the only
#: trace a silently shrunken training set leaves anywhere.
_STANDARDIZER_REJECT_RE = re.compile(r'"[^"]+" filter: (.*?) is invalid')

#: ``numpy.random.seed`` refuses anything outside this, and
#: ``reinvent.utils.helpers.set_seed`` calls it unguarded, so a seed outside the
#: range is a traceback several hundred megabytes of model load too late.
_SEED_MAX = 2**32 - 1


def resolve_seed(raw: object) -> int:
    """The seed to hand ``reinvent.Reinvent:main``, or a refusal saying why.

    Upstream resolves ``seed = args.seed or getattr(val_config, "seed", None)``
    and seeds nothing when that is falsy, so ``0`` is not "seed zero" — it is an
    unseeded run wearing a seeded run's config. It is refused here rather than
    translated, because a caller who asked for 0 and silently got an unreproducible
    checkpoint is the exact failure this parameter exists to close.
    """
    if raw is None:
        raise ValueError(
            "no seed was given. Transfer learning without one writes a checkpoint whose "
            "bytes can never be regenerated, so the adapter always passes one"
        )
    try:
        seed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer, got {raw!r}") from exc
    if seed == 0:
        raise ValueError(
            "seed 0 is refused: reinvent/Reinvent.py:120 resolves the seed as "
            "'args.seed or ...', so a falsy 0 seeds nothing at all and the run would be "
            "unreproducible while reporting seed 0"
        )
    if not 0 < seed <= _SEED_MAX:
        raise ValueError(f"seed must be in 1..{_SEED_MAX} (numpy.random.seed's range), got {seed}")
    return seed


def count_standardizer_rejections(log_text: str) -> int:
    """Distinct SMILES upstream's standardiser dropped, from its own WARNING lines.

    Distinct SMILES rather than lines: ``apply_filter`` warns once per filter
    config while the molecule is already ``None`` *and* once more after the chain,
    so a row RDKit could not parse produces the identical message twice.
    """
    return len({m.group(1) for m in _STANDARDIZER_REJECT_RE.finditer(log_text)})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GpuSampler:
    """Poll ``nvidia-smi`` for compute processes belonging to this driver.

    Read-only: it asks which compute processes exist and changes no machine
    setting. "Belonging to" is this process or any descendant of it, because the
    CUDA context lives in this very process while ``reinvent_main`` runs.
    """

    def __init__(self, pid: int, interval: float = GPU_POLL_SECONDS) -> None:
        self.pid = pid
        self.interval = interval
        self.samples = 0
        self.matched = 0
        self.peak_mib: int | None = None
        self.state = "MEASURED"
        self.reason = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._binary = shutil.which("nvidia-smi")
        if self._binary is None:
            self.state = "UNAVAILABLE"
            self.reason = "nvidia-smi is not on PATH"

    def start(self) -> None:
        if self.state != "MEASURED":
            return
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self) -> None:
        if self._binary is None:
            return
        try:
            proc = subprocess.run(
                [
                    self._binary,
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.state = "UNAVAILABLE"
            self.reason = f"{type(exc).__name__}: {exc}"
            return
        if proc.returncode != 0:
            self.state = "UNAVAILABLE"
            self.reason = f"nvidia-smi exited {proc.returncode}"
            return
        self.samples += 1
        used_here = 0
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid, mib = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if pid == self.pid or _is_descendant(pid, self.pid):
                used_here += mib
        if used_here:
            self.matched += 1
            self.peak_mib = max(self.peak_mib or 0, used_here)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.state != "MEASURED":
            return {
                "gpu_telemetry": "UNAVAILABLE",
                "gpu_seconds": None,
                "max_gpu_memory_mib": None,
                "gpu_samples": self.samples,
                "gpu_note": self.reason,
            }
        return {
            "gpu_telemetry": "MEASURED",
            "gpu_seconds": round(self.matched * self.interval, 3),
            "max_gpu_memory_mib": self.peak_mib if self.peak_mib is not None else 0,
            "gpu_samples": self.samples,
            "gpu_note": (
                "sampled nvidia-smi compute processes for this driver's pid and its "
                "descendants; 0.0 means the job held no CUDA context in any sample"
            ),
        }


def _is_descendant(pid: int, ancestor: int, max_depth: int = 32) -> bool:
    """Whether *pid* is below *ancestor*, read from ``/proc/<pid>/stat``."""
    current = pid
    for _ in range(max_depth):
        if current == ancestor:
            return True
        if current <= 1:
            return False
        try:
            stat = Path(f"/proc/{current}/stat").read_text()
        except OSError:
            return False
        # The comm field is parenthesised and may itself contain spaces, so the
        # parent pid is the second field *after* the closing parenthesis.
        try:
            tail = stat[stat.rindex(")") + 1 :].split()
            current = int(tail[1])
        except (ValueError, IndexError):
            return False
    return False


def build_config(request: dict) -> dict:
    """The ReinventConfig body for one transfer-learning run.

    Keys are exactly those ``reinvent/validation.py:ReinventConfig`` and
    ``reinvent/runmodes/TL/validation.py:SectionParameters`` accept; both are
    ``extra="forbid"``, so an invented key is a startup error, not a silent
    no-op. ``scheduler`` is deliberately absent — see the module docstring.
    """
    num_epochs = int(request["num_epochs"])
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
    batch_size = int(request["batch_size"])
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # Default to one snapshot at the end rather than upstream's one per epoch:
    # the unsuffixed output_model_file is rewritten every epoch regardless
    # (learning.py:217), so per-epoch snapshots only add ~23 MB each.
    save_every = int(request.get("save_every_n_epochs") or num_epochs)
    sample_batch_size = int(request.get("sample_batch_size") or 100)

    parameters = {
        "input_model_file": str(request["input_model_file"]),
        "output_model_file": str(request["output_model_file"]),
        "smiles_file": str(request["smiles_file"]),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "save_every_n_epochs": max(save_every, 1),
        "sample_batch_size": max(sample_batch_size, 100),
        "standardize_smiles": bool(request.get("standardize_smiles", True)),
        "randomize_smiles": bool(request.get("randomize_smiles", True)),
        "isomeric_smiles": bool(request.get("isomeric_smiles", False)),
    }
    if request.get("validation_smiles_file"):
        parameters["validation_smiles_file"] = str(request["validation_smiles_file"])

    return {
        "run_type": "transfer_learning",
        "device": str(request.get("device") or "cpu"),
        "parameters": parameters,
    }


def count_rows(path: Path) -> int:
    """Non-blank, non-comment rows in the SMILES file.

    ``read_smiles_csv_file`` skips a row whose joined fields strip to nothing and
    a row starting with ``#``, so this is the number of rows it will consider —
    a cross-check on the count REINVENT4 reports, never a replacement for it.
    """
    rows = 0
    with path.open() as fh:
        for line in fh:
            stripped = line.replace("\t", "").strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows += 1
    return rows


def _network_state(save_dict: object) -> dict | None:
    """The tensor dict, found exactly the way ``meta_data._get_network`` finds it.

    Transcribed from ``reinvent/models/meta_data.py:_get_network`` so a digest
    taken here covers the same tensors upstream considers to be the model.
    """
    if not isinstance(save_dict, dict):
        return None
    if "decorator" in save_dict:  # Libinvent
        state = save_dict["decorator"].get("state")
    elif "network_state" in save_dict:  # Linkinvent, Mol2Mol
        state = save_dict["network_state"]
    else:  # Reinvent
        state = save_dict.get("network")
    return state if isinstance(state, dict) else None


def weights_sha256(path: Path) -> str | None:
    """A digest of the model's tensors, and of nothing that changes on its own.

    **Why the file's own sha256 cannot be this.** Every save goes through
    ``reinvent/models/meta_data.py:update_model_data``, which does
    ``metadata["updates"].append(time.time())`` at line 64 and then computes
    ``metadata["hash_id"]`` over a pickle that includes it. A wall-clock reading
    is therefore written into the checkpoint on every save, so two runs that
    produce *bit-identical weights* still produce different files, different
    ``sha256`` and different ``hash_id``. No seed changes that, and no sha256 of
    one of these files can ever be regenerated.

    This digest covers the tensors and their names, shapes and dtypes, in sorted
    key order, and excludes the metadata block entirely. It is the digest that
    can be quoted as evidence and produced again tomorrow, and it is the one that
    answers the question a checkpoint's identity is usually asked for: will this
    generator sample the same way.

    ``None`` when the file carries no recognisable tensor dict — reported as
    ``None`` rather than as a digest of nothing.
    """
    import torch

    state = _network_state(torch.load(str(path), map_location="cpu", weights_only=False))
    if not state:
        return None
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode("utf-8") + b"\0")
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("utf-8") + b"\0")
            digest.update(tensor.numpy().tobytes() + b"\0")
        else:  # not a tensor: hash its repr rather than skip it silently
            digest.update(repr(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def read_input_model_type(path: Path) -> str:
    """The generator name stored in the input checkpoint.

    ``create_adapter`` reads ``model_type`` out of the torch pickle and picks the
    run mode's model class from it, so a LibInvent or Mol2Mol file would train a
    different generator under the same call. Reporting it is how a caller can
    tell afterwards which generator it actually adapted.
    """
    import torch

    save_dict = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(save_dict, dict) and save_dict.get("model_type"):
        return str(save_dict["model_type"])
    return ""


def run(request: dict) -> dict:
    out_dir = Path(request["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "transfer_learning.json"
    log_path = out_dir / "reinvent.log"

    smiles_file = Path(request["smiles_file"])
    if not smiles_file.is_file():
        raise RuntimeError(f"training SMILES file {smiles_file} does not exist")
    input_model_file = Path(request["input_model_file"])
    if not input_model_file.is_file():
        raise RuntimeError(f"input model file {input_model_file} does not exist")
    output_model_file = Path(request["output_model_file"])
    output_model_file.parent.mkdir(parents=True, exist_ok=True)

    seed = resolve_seed(request.get("seed"))

    config = build_config(request)
    config_path.write_text(json.dumps(config, indent=2))

    from reinvent import version
    from reinvent.Reinvent import main as reinvent_main

    input_model_sha256 = sha256_file(input_model_file)
    input_model_weights_sha256 = weights_sha256(input_model_file)
    input_model_type = read_input_model_type(input_model_file)
    n_rows = count_rows(smiles_file)

    # The Namespace reinvent/utils/cli.py:parse_command_line() would have built.
    # `device` stays None so the config file's value is used, which is the path
    # reinvent/runmodes/utils/helpers.py:set_torch_device takes for a config-only
    # device; passing both would only duplicate it.
    args = Namespace(
        config_filename=config_path,
        config_format="json",
        device=None,
        log_filename=str(log_path),
        log_level=str(request.get("log_level") or "info"),
        seed=seed,
        dotenv_filename=None,
        enable_rdkit_log_levels=None,
    )

    sampler = GpuSampler(os.getpid())
    sampler.start()
    started = time.perf_counter()
    try:
        reinvent_main(args)
    finally:
        wall_seconds = time.perf_counter() - started
        gpu = sampler.stop()

    if not output_model_file.is_file():
        raise RuntimeError(
            f"REINVENT4 finished without writing {output_model_file}; see {log_path} for its own log"
        )

    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    match = _READ_SMILES_RE.search(log_text)
    if match is None:
        raise RuntimeError(
            "REINVENT4 did not log how many SMILES it read, so the number it trained on "
            f"cannot be reported honestly (log_level={args.log_level!r}, log {log_path}). "
            "The file's own row count is not a substitute for it."
        )
    n_training_smiles = int(match.group(1))
    n_standardizer_rejected = count_standardizer_rejections(log_text)

    snapshots = sorted(str(p) for p in output_model_file.parent.glob(f"{output_model_file.name}.*.chkpt"))

    return {
        "checkpoint": str(output_model_file.resolve()),
        "checkpoint_sha256": sha256_file(output_model_file),
        "checkpoint_weights_sha256": weights_sha256(output_model_file),
        "checkpoint_bytes": output_model_file.stat().st_size,
        "input_model_file": str(input_model_file.resolve()),
        "input_model_sha256": input_model_sha256,
        "input_model_weights_sha256": input_model_weights_sha256,
        "input_model_type": input_model_type,
        "n_training_smiles": n_training_smiles,
        "n_smiles_file_rows": n_rows,
        "n_standardizer_rejected": n_standardizer_rejected,
        "seed": seed,
        "epochs": int(config["parameters"]["num_epochs"]),
        "batch_size": int(config["parameters"]["batch_size"]),
        "wall_seconds": round(wall_seconds, 3),
        "snapshots": snapshots,
        "reinvent_version": version.__version__,
        "device": config["device"],
        "log_path": str(log_path),
        **gpu,
    }


if __name__ == "__main__":
    _driver.main(run)
