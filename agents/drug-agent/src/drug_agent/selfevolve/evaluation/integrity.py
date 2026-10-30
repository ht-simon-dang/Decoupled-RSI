"""Tamper detection for the evaluation kernel.

A hash manifest over every file that decides whether a candidate succeeded: this
package, the runtime-policy module the arms are bound through, and the gold
rubric. The kernel computes it before an evaluation and again after, and refuses
to emit a record if they differ.

This is not a security boundary against a determined attacker with filesystem
access — nothing running in-process can be. It is a boundary against the failure
that actually happens: a candidate, or an implementation agent acting for one,
editing the thing that grades it because that is the shortest path to the number
going up. Making that visible turns a silent success into a loud quarantine.

The complement is process-level: candidates are applied as overlays or as
patches in a workspace *outside* this tree, and the loop never asks a model to
report whether tests passed — the kernel runs them and keeps the exit status.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..repository import project_path_candidates


class KernelIntegrityError(RuntimeError):
    """The evaluation kernel changed while it was supposed to be frozen."""


def _kernel_dir() -> Path:
    return Path(__file__).resolve().parent


def _extra_guarded_files() -> list[Path]:
    """Files outside this package that an arm's result also depends on."""
    pkg = _kernel_dir().parent.parent  # drug_agent/
    return [
        pkg / "llm" / "policy.py",  # how an arm is bound
        pkg / "selfevolve" / "schemas.py",  # the shape of the evidence
        # The status/value contract the harness records every endpoint through.
        # An arm that could edit this could turn a failed re-execution back into
        # a zero, which is exactly the tampering the manifest exists to catch.
        pkg / "selfevolve" / "target" / "api.py",
        # api.py is a thin surface over these record definitions: the required
        # fields, the kind vocabulary, the IMPROVED-needs-an-interval rule and
        # the nomination leakage check all live here. Guarding the surface while
        # leaving the rules editable would guard nothing.
        pkg / "selfevolve" / "target" / "contracts.py",
        # `api.validate_protocol` is a thin adapter over this module, and this
        # module holds the frozen study lock's rules: the required lock fields,
        # the canonical serialisation the lock's content address is taken over,
        # the seed policy, the one-nomination cap, the symmetric-budget rule and
        # the gate that stops a study calling itself confirmatory when its own
        # power arithmetic does not reach its own minimum. Appendix A §6 puts the
        # lock beside the evaluator among the things a candidate may not change,
        # so guarding the surface while leaving these editable would guard
        # nothing.
        pkg / "selfevolve" / "target" / "protocol.py",
        # The benchmark partition itself. `api.partition_families` is a thin
        # adapter over this module, so guarding the surface while leaving the
        # family edges, the similarity cutoff, the allocation and the exclusion
        # registry editable would guard nothing. A candidate able to edit it
        # could move a held-out family into training and still produce a
        # split_hash, which is the one tampering a frozen split exists to stop.
        pkg / "selfevolve" / "target" / "splits.py",
        # The candidate worker decides which checkout, which interpreter and
        # which import path an evaluation runs under, and verifies that the code
        # really came from the candidate. Gate 0 calls into it, so a candidate
        # able to edit it could point its own evaluation back at the parent —
        # the exact masquerade the module exists to prevent.
        pkg / "selfevolve" / "target" / "worker.py",
        # The isolated checkout itself: patch application and cleanup.
        pkg / "selfevolve" / "candidates" / "workspace.py",
        # Git/project path translation is part of the admission boundary.
        pkg / "selfevolve" / "repository.py",
    ]


def guarded_files() -> list[Path]:
    files = sorted(p for p in _kernel_dir().glob("*.py") if p.is_file())
    files += [p for p in _extra_guarded_files() if p.is_file()]
    return files


def kernel_manifest() -> dict[str, str]:
    """``{relative path: sha256}`` for every guarded file."""
    root = _kernel_dir().parents[2]  # src/
    out: dict[str, str] = {}
    for path in guarded_files():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = "UNREADABLE"
        try:
            key = str(path.relative_to(root))
        except ValueError:
            key = str(path)
        out[key] = digest
    return dict(sorted(out.items()))


def kernel_digest(manifest: dict[str, str] | None = None) -> str:
    """One digest over the whole manifest."""
    man = manifest if manifest is not None else kernel_manifest()
    return hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()


def verify_kernel(expected_digest: str) -> None:
    """Raise if the kernel is not byte-identical to ``expected_digest``."""
    actual = kernel_digest()
    if actual != expected_digest:
        raise KernelIntegrityError(
            "evaluation kernel changed during a measurement: "
            f"expected {expected_digest[:16]}…, found {actual[:16]}…"
        )


def diff_manifest(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    """What changed, for the quarantine record."""
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    return {
        "changed": changed,
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
    }


def candidate_touches_kernel(files_changed: list[str], *, root: Path | str | None = None) -> list[str]:
    """Which of a candidate's changed paths fall inside the guarded set.

    Checked *before* a candidate is evaluated, so a change to the judge is
    refused at admission rather than discovered afterwards.
    """
    guarded = set(kernel_manifest())
    project = Path(root) if root is not None else _kernel_dir().parents[3]
    hits: list[str] = []
    for raw in files_changed:
        variants = project_path_candidates(raw, project)
        for norm in variants:
            if any(norm == g or norm.endswith("/" + g) or g.endswith("/" + norm) for g in guarded):
                hits.append(raw)
                break
            if "/selfevolve/evaluation/" in norm or norm.endswith("llm/policy.py"):
                hits.append(raw)
                break
    return sorted(set(hits))



__all__ = [
    "KernelIntegrityError",
    "candidate_touches_kernel",
    "diff_manifest",
    "guarded_files",
    "kernel_digest",
    "kernel_manifest",
    "verify_kernel",
]
