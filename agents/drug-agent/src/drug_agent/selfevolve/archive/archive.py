"""The candidate archive: lineage, novelty rejection, and behaviour cells.

Three ideas from three papers, each cut down to what five episodes and a
six-task benchmark can actually support.

**Lineage and parent sampling (Darwin Gödel Machine).** Every candidate records
its parent and generation, and :meth:`Archive.sample_parent` draws from the whole
archive rather than always from the current best. The point is stepping stones:
a candidate that lost overall may still be the ancestor of one that wins.

**Novelty rejection (ShinkaEvolve).** A benchmark arm here costs about half an
hour of GPU time, so the expensive thing is not generating candidates, it is
scoring them. :meth:`Archive.is_novel` refuses a candidate whose descriptor —
the surfaces it touches, canonicalised, with prose collapsed — already exists.
That is what stops the loop re-proposing the same overlay in different words.

**Behaviour cells (MAP-Elites).** A candidate is retained if it is the best in
its ``(class, primary surface)`` cell even when it loses globally, so a planner
fix and a refiner fix never evict one another. With four operators the grid is
tiny; the value is that "keep the best of each kind" is a rule, not a habit.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from ..config import EvolveConfig
from ..schemas import Candidate, CandidateStatus, EvaluationRecord


class Archive:
    """Append-only store of candidates plus their evaluation records."""

    def __init__(self, cfg: EvolveConfig | None = None) -> None:
        self.cfg = cfg or EvolveConfig()
        self.dir = self.cfg.resolved_store_dir() / "archive"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

    # -- writing ---------------------------------------------------------
    def add(self, candidate: Candidate) -> Candidate:
        if not candidate.created_at:
            candidate.created_at = time.time()
        self._write(candidate)
        self._append_index(candidate, event="added")
        return candidate

    def update(self, candidate: Candidate) -> Candidate:
        self._write(candidate)
        self._append_index(candidate, event="updated")
        return candidate

    def set_status(self, candidate_id: str, status: CandidateStatus, reason: str = "") -> Candidate | None:
        cand = self.get(candidate_id)
        if cand is None:
            return None
        cand.status = status
        if reason:
            cand.notes = (cand.notes + "\n" if cand.notes else "") + reason
        return self.update(cand)

    def _write(self, candidate: Candidate) -> None:
        (self.dir / f"{candidate.candidate_id}.json").write_text(
            candidate.model_dump_json(indent=2), encoding="utf-8"
        )

    def _append_index(self, candidate: Candidate, *, event: str) -> None:
        row = {
            "ts": time.time(),
            "event": event,
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "generation": candidate.generation,
            "class": str(candidate.candidate_class),
            "cell": candidate.cell(),
            "novelty": candidate.novelty_descriptor(),
            "status": str(candidate.status),
            "hypothesis": candidate.hypothesis[:300],
            "author": candidate.author,
        }
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- reading ---------------------------------------------------------
    def get(self, candidate_id: str) -> Candidate | None:
        path = self.dir / f"{candidate_id}.json"
        if not path.is_file():
            return None
        try:
            return Candidate.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def all(self) -> list[Candidate]:
        out: list[Candidate] = []
        for path in sorted(self.dir.glob("cand_*.json")):
            try:
                out.append(Candidate.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda c: (c.generation, c.created_at))

    def resolve(self, prefix: str) -> Candidate | None:
        """Look up by full id or unambiguous prefix — CLI ergonomics."""
        exact = self.get(prefix)
        if exact is not None:
            return exact
        hits = [c for c in self.all() if c.candidate_id.startswith(prefix)]
        return hits[0] if len(hits) == 1 else None

    # -- novelty ---------------------------------------------------------
    def is_novel(self, candidate: Candidate) -> tuple[bool, str]:
        """``(novel, reason)``. Reason names the twin when it is not."""
        descriptor = candidate.novelty_descriptor()
        for other in self.all():
            if other.candidate_id == candidate.candidate_id:
                continue
            if other.novelty_descriptor() == descriptor:
                return False, (
                    f"identical to {other.candidate_id} (status {other.status}) — same surfaces "
                    "and same values, so evaluating it again would buy nothing"
                )
        return True, ""

    # -- lineage and cells ----------------------------------------------
    def lineage(self, candidate_id: str) -> list[Candidate]:
        chain: list[Candidate] = []
        seen: set[str] = set()
        cur = self.get(candidate_id)
        while cur is not None and cur.candidate_id not in seen:
            seen.add(cur.candidate_id)
            chain.append(cur)
            cur = self.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def children(self, candidate_id: str) -> list[Candidate]:
        return [c for c in self.all() if c.parent_id == candidate_id]

    def cells(self, scores: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Best candidate per behaviour cell.

        ``scores`` maps candidate id → primary metric. Without it the cells are
        reported with their occupants but no elite, because picking an elite
        without a score would mean picking by recency and calling it quality.
        """
        cells: dict[str, dict[str, Any]] = {}
        for cand in self.all():
            slot = cells.setdefault(cand.cell(), {"members": [], "elite": None, "elite_score": None})
            slot["members"].append(cand.candidate_id)
            if scores is None:
                continue
            score = scores.get(cand.candidate_id)
            if score is None:
                continue
            if slot["elite_score"] is None or score > slot["elite_score"]:
                slot["elite"] = cand.candidate_id
                slot["elite_score"] = score
        return dict(sorted(cells.items()))

    def sample_parent(
        self, *, rng: random.Random | None = None, scores: dict[str, float] | None = None
    ) -> Candidate | None:
        """Draw a parent, favouring cell elites but never only the global best.

        Weighting is deliberately mild — with a handful of candidates a sharp
        selection pressure just means "always the winner", which is the greedy
        hill-climbing the archive exists to avoid.
        """
        pool = [c for c in self.all() if c.status != CandidateStatus.QUARANTINED]
        if not pool:
            return None
        rng = rng or random.Random(0)
        elites = {slot["elite"] for slot in self.cells(scores).values() if slot.get("elite")}
        weights = [3.0 if c.candidate_id in elites else 1.0 for c in pool]
        return rng.choices(pool, weights=weights, k=1)[0]

    # -- records ---------------------------------------------------------
    def link_record(self, candidate_id: str, record: EvaluationRecord) -> None:
        """Note which evaluation records belong to which candidate."""
        path = self.dir / f"{candidate_id}.records.jsonl"
        row = {
            "record_id": record.record_id,
            "arm": record.arm,
            "mode": record.mode,
            "benchmark": f"{record.benchmark_id}@{record.benchmark_version}",
            "kernel_verified": record.kernel_verified,
            "ok": record.ok,
            "aggregates": record.aggregates,
            "ts": record.finished_at,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def records_for(self, candidate_id: str) -> list[dict[str, Any]]:
        path = self.dir / f"{candidate_id}.records.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    # -- reporting -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        cands = self.all()
        by_status: dict[str, int] = {}
        for c in cands:
            by_status[str(c.status)] = by_status.get(str(c.status), 0) + 1
        return {
            "dir": str(self.dir),
            "n_candidates": len(cands),
            "by_status": dict(sorted(by_status.items())),
            "cells": {k: v["members"] for k, v in self.cells().items()},
            "generations": sorted({c.generation for c in cands}),
        }


__all__ = ["Archive"]
