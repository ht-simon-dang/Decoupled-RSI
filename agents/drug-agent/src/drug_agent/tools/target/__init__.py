"""Target evidence: what is known about a target before any molecule exists.

Every other tool in this package answers a question about a *compound* — will it
bind, will it be made, will it be safe. The two adapters here answer the question
that comes before those: **should this protein be drugged at all, and what is the
evidence?** That is the input a de-novo target agent needs, and this repository
did not have it — what stood in its place was
:class:`~drug_agent.data.connectors.OpenTargetsLikeConnector`, which reads
``fixtures/data/targets.json``: three hand-written associations whose ``source``
field literally says ``synthetic_demo`` and whose scores (hypertension 0.62,
inflammatory pain 0.55, diabetic neuropathy 0.41) were never computed by anything.
Under :mod:`drug_agent.tools.base` that is not evidence, and nothing here extends
it, imitates it, or falls back to it.

**The shape both adapters take: fetch once, read forever.**

The platform runs with ``allow_network=False`` and
:class:`~drug_agent.security.egress.EgressPolicy` refuses egress by default, so a
tool that phones an API at call time cannot work here — and should not: a score
that changes silently between two runs of the same brief is not reproducible
evidence. So each capability is split in two.

*Setup*, which may use the network, runs once, and is a script an operator
invokes: ``scripts/fetch_opentargets_snapshot.py`` and
``scripts/fetch_depmap_subset.py``. Each performs the real upstream request,
verifies what came back where upstream publishes a digest, and writes a snapshot
under ``.external/`` carrying the source URL, the exact query or file id, the
upstream version or release, a UTC fetch timestamp, and a content hash and
snapshot id from :class:`~drug_agent.data.snapshot.DataSnapshot`.

*Runtime*, which is these adapters, imports no HTTP client, opens no socket and
consults no fallback. It reads the snapshot, or it raises
:class:`~drug_agent.tools.base.ToolUnavailable` naming the missing file, the
identifier that is missing, and the command that would fetch it.

**A cached snapshot is real data; a fixture is not.** The difference is not where
the bytes sit — both are local JSON — it is whether the numbers were computed by
the source that owns them and whether a reader can check. Every value these
adapters return came off Open Targets' or DepMap's own pipeline, and every one is
returned with a :class:`~drug_agent.contracts.base.Provenance` naming the release
it came from. Nothing here interpolates a score for a target that was not fetched,
substitutes a related gene, or ages a snapshot into a guess.

**Two capabilities, deliberately not one.** Association evidence and essentiality
answer different questions and must never be traded for one another. Open Targets
says *this target is implicated in this disease, by this kind of evidence*. DepMap
says *knocking this gene out kills these cancer cell lines*. A target can score
well on the first and flat on the second — EPHX2 does exactly that — and reporting
either in place of the other would be a category error, not an approximation. So
they are separate adapters with separate ids, separate snapshots and separate
provenance, and neither one's absence is covered by the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...contracts.base import Provenance
from ...contracts.enums import TrustLabel
from ...contracts.evidence import EvidenceItem
from .._ext.adapter import ExternalToolAdapter
from ..base import ToolUnavailable


class TargetEvidenceError(RuntimeError):
    """The snapshot was there and could not be read as what it claims to be.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`, which means the
    snapshot was never fetched. This one means a file exists at the cache path and
    is truncated, is not JSON, or is missing the columns its data card promises —
    a corrupt download rather than an absent one.
    """


class TargetEvidenceAdapter(ExternalToolAdapter):
    """Shared snapshot handling for the target-evidence backends.

    What the two adapters have in common is the *discipline*, not the data: locate
    a cache directory, read a data card out of it, refuse by name when the record
    asked for is not in it, and stamp every value returned with the provenance of
    the release it came from. Getting any of those subtly different between the two
    is how a number ends up in a report with no way back to its source.
    """

    produces_nothing = "reports no target evidence"

    #: Name of the file every cache carries its data card in.
    index_name = "index.json"

    # --- cache -------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        """Where this tool's snapshot lives.

        Never ``None`` on an available adapter: ``ExternalBackend`` only resolves a
        root when the sentinel files are present, and their absence is what makes
        the capability UNAVAILABLE in the first place.
        """
        root = self.ext.root
        if root is None:  # pragma: no cover - guarded by require_available() first
            raise ToolUnavailable(self.capability_id, self.unavailable_reason or "no snapshot cache")
        return root

    def read_json(self, path: Path) -> dict[str, Any]:
        """Parse one JSON file of the cache, or say which file is broken."""
        try:
            loaded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TargetEvidenceError(f"{path} is present but unreadable: {exc}") from exc
        if not isinstance(loaded, dict):
            raise TargetEvidenceError(f"{path} is not a JSON object")
        return loaded

    def index(self) -> dict[str, Any]:
        """The cache's data card: provenance, release, and what is in it."""
        return self.read_json(self.cache_dir / self.index_name)

    def refuse(self, reason: str) -> Any:
        """Refuse *this request* — the cache is fine, what it was asked for is not.

        Deliberately :class:`ToolUnavailable` rather than a domain error, and
        deliberately a *different* string from the card's ``unavailable_reason``:
        the capability is available and the identifier is not cached, so the
        message has to name the identifier and the command that would add it. A
        caller that swallowed this and carried on would be inventing the answer,
        which is the one thing this package forbids.
        """
        raise ToolUnavailable(self.capability_id, reason)

    # --- provenance --------------------------------------------------------
    def provenance(
        self,
        card: dict[str, Any],
        *,
        source: str,
        source_version: str | None,
        trust_label: TrustLabel,
        notes: str | None = None,
    ) -> Provenance:
        """Stamp a value with the snapshot it was read out of.

        ``snapshot_id`` and ``content_hash`` come from the data card the fetch
        script wrote, so a number in a report can be traced to the exact bytes that
        were downloaded and, through the recorded query or file digest, back to
        upstream.
        """
        return Provenance(
            source=source,
            source_version=source_version,
            method=self.capability_id,
            method_version=self.version,
            snapshot_id=str(card.get("snapshot_id") or "") or None,
            content_hash=str(card.get("content_hash") or "") or None,
            trust_label=trust_label,
            notes=notes,
        )

    def item(
        self,
        *,
        kind: str,
        statement: str,
        subject: str,
        value: dict[str, Any],
        provenance: Provenance,
        trust_label: TrustLabel,
        quality_score: float | None = None,
    ) -> EvidenceItem:
        """One evidence-plane record, which is the contract both adapters emit.

        :class:`~drug_agent.contracts.evidence.EvidenceItem` already exists for
        exactly this — a source-attributed assertion with a trust label and a
        structured payload — so neither adapter invents a parallel result model.
        ``quality_score`` is left ``None`` unless upstream published a bounded
        confidence that genuinely means "how good is this evidence"; it is never a
        rescaled copy of the value.
        """
        return EvidenceItem(
            kind=kind,
            statement=statement,
            subject=subject,
            value=value,
            provenance=provenance,
            trust_label=trust_label,
            quality_score=quality_score,
        )


__all__ = ["TargetEvidenceAdapter", "TargetEvidenceError"]
