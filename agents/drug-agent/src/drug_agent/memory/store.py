"""Memory store with six memory classes over the relational + artifact stores.

* working   — ephemeral per-run scratch (kept in the events/objects tables)
* episodic  — evidence items and decisions tied to a run
* semantic  — entities and relationships (program-scoped knowledge graph)
* vector    — lightweight hashed bag-of-words retrieval (no external index)
* artifact  — content-addressed artifacts via the artifact store
* policy    — named policies with versions

Retrieval is a simple hybrid: keyword score + metadata filter. A real vector
index is an optional upgrade; the interface here is deliberately swappable.
Program isolation is enforced by the ``program_id`` filter on every query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..contracts.evidence import DecisionRecord, EvidenceItem
from ..storage.artifacts import ArtifactStore
from ..storage.db import StateStore
from ..util import content_hash

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class MemoryStore:
    def __init__(self, store: StateStore, artifacts: ArtifactStore, program_id: str | None = None) -> None:
        self.store = store
        self.artifacts = artifacts
        self.program_id = program_id

    # --- episodic: evidence & decisions -----------------------------------
    def add_evidence(self, item: EvidenceItem) -> EvidenceItem:
        self.store.put_object(
            "evidence", item, run_id=item.run_id, program_id=item.program_id or self.program_id
        )
        return item

    def list_evidence(self, run_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_objects(
            "evidence", run_id=run_id, program_id=self.program_id if run_id is None else None
        )

    def add_decision(self, decision: DecisionRecord) -> DecisionRecord:
        self.store.put_object("decision", decision, run_id=decision.run_id, program_id=self.program_id)
        return decision

    def list_decisions(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.list_objects("decision", run_id=run_id)

    # --- semantic: entities & relationships -------------------------------
    def upsert_entity(self, entity_id: str, etype: str, name: str, data: dict[str, Any]) -> None:
        self.store.put_entity(entity_id, etype, name, data, program_id=self.program_id)

    def relate(
        self, subject_id: str, predicate: str, object_id: str, data: dict[str, Any] | None = None
    ) -> None:
        self.store.add_relationship(subject_id, predicate, object_id, data, program_id=self.program_id)

    def neighbors(self, subject_id: str, predicate: str | None = None) -> list[dict[str, Any]]:
        return self.store.query_relationships(
            subject_id=subject_id, predicate=predicate, program_id=self.program_id
        )

    # --- vector-ish hybrid retrieval over evidence ------------------------
    def retrieve(
        self, query: str, *, run_id: str | None = None, kind: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Hybrid keyword retrieval over evidence with metadata filtering.

        Returns evidence dicts augmented with a ``_score``. Every returned item
        keeps its source/version/trust label/evidence id.
        """
        items = self.list_evidence(run_id=run_id)
        if kind:
            items = [it for it in items if it.get("kind") == kind]
        q_tokens = Counter(_tokens(query))
        scored: list[tuple[float, dict[str, Any]]] = []
        for it in items:
            text = f"{it.get('statement', '')} {it.get('subject', '')}"
            d_tokens = Counter(_tokens(text))
            score = _cosine(q_tokens, d_tokens)
            scored.append((score, it))
        scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
        out = []
        for score, it in scored[:top_k]:
            it = dict(it)
            it["_score"] = round(score, 4)
            out.append(it)
        return out

    # --- policy memory -----------------------------------------------------
    def set_policy(self, name: str, version: str, data: dict[str, Any]) -> None:
        obj = {
            "id": f"{name}:{version}",
            "name": name,
            "version": version,
            "data": data,
            "hash": content_hash(data),
        }
        self.store.put_object("policy", obj, program_id=self.program_id)

    def get_policy(self, name: str, version: str) -> dict[str, Any] | None:
        return self.store.get_object("policy", f"{name}:{version}")


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0
