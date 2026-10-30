"""On-demand protein previews, cached by exact translated sequence and protocol.

This is a visualization sidecar. It never writes evaluations or changes delivery.
The existing Enzyme weights and GPU lock are shared; output stays under mRNA quota.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .contracts import identity, translate
from .storage import data_root, quota

MODEL = {"repo": "biohub/ESMFold2-Fast", "revision": "45fe8656f5b3ef493c17fcf9abe9a2968902e712", "license": "MIT"}
PROTOCOL = {**MODEL, "seed": 42, "loops": 1, "steps": 50, "version": 1}
MAX_RESIDUES = 600


def enzyme_data_root():
    return Path(os.environ.get("ENZYME_DATA_DIR", str(Path.home() / ".cache/decoupled-rsi/enzyme"))).expanduser()


def model_path():
    default = enzyme_data_root() / "models/esmfold2" / MODEL["revision"]
    return Path(os.environ.get("MRNA_ESMFOLD_DIR", str(default))).expanduser()


def model_ready():
    """Check the shared installation manifest before claiming the pinned model."""
    path = model_path()
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        files = manifest["files"]
        return (all(manifest.get(k) == MODEL[k] for k in ("repo", "revision"))
                and "config.json" in files
                and any(name.endswith(".safetensors") for name in files)
                and all(Path(name).name == name and (path / name).is_file()
                        and (path / name).stat().st_size == info["size"] for name, info in files.items()))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def protein_for(candidate):
    return translate(candidate["regions"]["cds"])


def validate_fold(data, sequence):
    """Require a complete, finite Cα trace with exact residue identity."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    structure = PDBParser(QUIET=True).get_structure("preview", io.StringIO(data["pdb"]))
    chains = list(structure[0].get_chains())
    if len(chains) != 1:
        raise ValueError("Expected exactly one predicted protein chain")
    residues = list(chains[0].get_residues())
    observed = "".join(seq1(r.resname) for r in residues)
    if data.get("sequence") != sequence or observed != sequence:
        raise ValueError("Prediction sequence does not match the candidate CDS translation")
    points = []
    for residue in residues:
        if "CA" not in residue:
            raise ValueError("Incomplete Cα backbone in prediction")
        point = [float(v) for v in residue["CA"].coord]
        if not all(math.isfinite(v) for v in point):
            raise ValueError("Non-finite structure coordinates")
        points.append(point)
    confidence = data["plddt_fraction"]
    if len(confidence) != len(sequence) or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in confidence):
        raise ValueError("Invalid per-residue confidence")
    if not math.isfinite(data["mean_plddt_fraction"]) or not math.isfinite(data["ptm"]):
        raise ValueError("Invalid structure confidence summary")
    return points


@contextmanager
def file_lock(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


class StructurePreview:
    def __init__(self):
        self.root = data_root() / "structure-previews"
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="protein-preview")
        self.lock = threading.RLock()
        self.pending = set()
        self.process = None
        self.closing = False

    def key(self, sequence):
        return identity({"sequence": sequence, "protocol": PROTOCOL})

    def _save(self, key, result):
        content = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
        with quota(len(content) + 65536):
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / (key + ".json")
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(content)
            temporary.replace(path)

    def status(self, sequence):
        key = self.key(sequence)
        base = {"kind": "encoded_protein", "key": key, "sequence": sequence,
                "residues": len(sequence), "provenance": PROTOCOL,
                "scope": "Sequence-only prediction of the encoded protein, not RNA. Same protein shares a cached fold; translation efficiency is not assessed."}
        if not 16 <= len(sequence) <= MAX_RESIDUES:
            return {**base, "status": "unavailable", "reason": f"Interactive preview supports 16–{MAX_RESIDUES} amino acids."}
        path = self.root / (key + ".json")
        if path.is_file():
            try:
                result = json.loads(path.read_text())
                if result.get("status") == "ready":
                    if result.get("provenance") != PROTOCOL:
                        raise ValueError("Cached model protocol mismatch")
                    result["points"] = validate_fold(result, sequence)
                return {**base, **result}
            except Exception as exc:
                return {**base, "status": "failed", "reason": "Cached structure failed validation: " + str(exc)}
        with self.lock:
            if key in self.pending:
                return {**base, "status": "running"}
        # A second server can inspect the same cache without launching another fold.
        if (self.root / (key + ".lock")).exists():
            try:
                with file_lock(self.root / (key + ".lock"), blocking=False):
                    pass
            except BlockingIOError:
                return {**base, "status": "running"}
        if not model_ready():
            return {**base, "status": "unavailable", "reason": "Pinned ESMFold2 manifest or weights are missing/incomplete; configure MRNA_ESMFOLD_DIR to the validated Enzyme installation."}
        return {**base, "status": "not_generated"}

    def submit(self, sequence):
        with self.lock:
            result = self.status(sequence)
            if result["status"] in {"ready", "running", "unavailable"}:
                return result
            if self.closing:
                raise ValueError("Preview worker is stopping; retry after server restart")
            with quota(65536):
                self.root.mkdir(parents=True, exist_ok=True)
                (self.root / (result["key"] + ".json")).unlink(missing_ok=True)
            self.pending.add(result["key"])
            self.pool.submit(self.execute, sequence)
            return {**result, "status": "running", "reason": None}

    def execute(self, sequence):
        key = self.key(sequence)
        try:
            with file_lock(self.root / (key + ".lock"), blocking=False):
                if self.closing:
                    return
                try:
                    data = self._predict(sequence)
                    validate_fold(data, sequence)
                    self._save(key, {**data, "status": "ready", "provenance": PROTOCOL,
                                     "created": time.time(), "template_conditioned": False})
                except Exception as exc:
                    if not self.closing:
                        self._save(key, {"status": "failed", "reason": str(exc)[:2000]})
        except BlockingIOError:
            pass
        finally:
            with self.lock:
                self.pending.discard(key)

    def _predict(self, sequence):
        if not model_ready():
            raise RuntimeError("Pinned ESMFold2 installation failed verification")
        # Respect Enzyme Agent's accelerator lock while reusing its model files.
        gpu_lock = Path(os.environ.get("MRNA_STRUCTURE_GPU_LOCK", str(enzyme_data_root() / ".gpu.lock"))).expanduser()
        # Nonblocking attempts let shutdown interrupt a wait for another agent.
        deadline = time.monotonic() + 600
        gpu_lock.parent.mkdir(parents=True, exist_ok=True)
        with gpu_lock.open("a") as lock:
            while True:
                if self.closing or time.monotonic() > deadline:
                    raise RuntimeError("Preview interrupted or accelerator wait timed out")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.2)
            with self.lock:
                if self.closing:
                    raise RuntimeError("Preview interrupted")
                proc = subprocess.Popen([sys.executable, "-m", "mrna_agent.structure_worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, start_new_session=True,
                    env=dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4"))
                self.process = proc
            try:
                output, error = proc.communicate(json.dumps({**PROTOCOL, "sequence": sequence,
                    "model_path": str(model_path())}), timeout=600)
                if proc.returncode:
                    raise RuntimeError("ESMFold2 prediction failed: " + error[-1500:])
                return json.loads(output)
            except subprocess.TimeoutExpired as exc:
                self._stop(proc)
                raise RuntimeError("Structure preview exceeded its 10 minute budget") from exc
            finally:
                with self.lock:
                    self.process = None

    @staticmethod
    def _stop(proc):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def close(self):
        with self.lock:
            self.closing = True
            if self.process:
                self._stop(self.process)
        self.pool.shutdown(wait=True, cancel_futures=True)
