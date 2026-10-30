"""Open Targets Platform — target-disease association, tractability and safety.

Open Targets is the evidence layer a de-novo target programme starts from. For one
gene it answers three separate questions, and this adapter keeps them separate
because they fail independently:

    *Is it implicated?*   ``associatedDiseases`` — one overall association score per
                          disease in [0, 1], plus the per-datatype breakdown
                          (``genetic_association``, ``literature``, ``animal_model``,
                          ``rna_expression``, ``clinical``, ``affected_pathway``,
                          ``known_drug``, ``somatic_mutation``) that says *which kind*
                          of evidence produced it. A 0.33 built from literature is
                          not the same claim as a 0.33 built from genetics.
    *Is it drug-like?*    ``tractability`` — a boolean bucket table per modality
                          (SM, AB, PR, OC), from "Approved Drug" down to
                          "Druggable Family".
    *What goes wrong?*    ``safetyLiabilities`` — curated adverse events with their
                          direction, dosing, biosample and datasource.

**What is real here, and how to check it.** Every number comes from a snapshot
``scripts/fetch_opentargets_snapshot.py`` fetched from
``https://api.platform.opentargets.org/api/v4/graphql``. The snapshot carries the
three GraphQL documents verbatim, their variables, the API build and data release
the server reported at fetch time, a UTC timestamp, and a content hash and
snapshot id. Re-running the recorded query against the recorded release reproduces
the file. Measured for this repository's own target, EPHX2 / ENSG00000120915, on
API 26.6.3 / data release 26.06: 620 associated diseases, led by neurodegenerative
disease (MONDO_0005559) at 0.3379 from ``affected_pathway`` 0.5523 and
``literature`` 0.0699, then Alzheimer disease (MONDO_0004975) at 0.3325 from
``literature`` 0.8589 and ``genetic_association`` 0.5040; five small-molecule
tractability buckets true including "Advanced Clinical" and "Structure with
Ligand"; and zero safety liabilities.

**Offline at runtime, by construction.** This module imports no HTTP client. There
is no code path from an operation to a socket, so ``allow_network`` cannot change
what it does. A target that was never fetched is refused by
:class:`~drug_agent.tools.base.ToolUnavailable` naming the Ensembl id and the fetch
command — never served from a neighbouring gene, an older release, or
``fixtures/data/targets.json``, whose three hand-written associations are labelled
``synthetic_demo`` at source and are not evidence.

**Gentropy: fetched, not run — and the reasoning matters more than the verdict.**
Gentropy is Open Targets' PySpark toolkit for the *genetics* half of the platform:
it ingests GWAS summary statistics, runs fine-mapping (SuSiE / PICS) into credible
sets, and applies the Locus-to-Gene model that assigns each credible set to a gene.
It is tempting to add it as a third backend. It does not belong here, for three
reasons that are about correctness rather than convenience.

  1. *Its inputs cannot be offline.* L2G needs harmonised GWAS summary statistics
     (the Open Targets GWAS release is measured in terabytes), an LD reference
     panel, and a trained L2G model. Nothing in that list can sit beside a 736 KB
     snapshot in ``.external/``, and a runtime that fetched them would be the live
     call this platform forbids.
  2. *Running it would produce a different number with the same name.* L2G scores
     are model outputs conditioned on the release's feature matrix and the trained
     classifier. A locally re-run L2G score is *our* score, not Open Targets', and
     putting it in a field a reader will compare against the portal is precisely
     the substitution ADR-0011 exists to stop.
  3. *Its output is already in the snapshot.* ``Target.credibleSets`` serves the
     fine-mapped credible sets themselves, with ``l2GPredictions.score`` — the
     Gentropy pipeline's own output for the release. So this adapter reports the
     genetics layer through :meth:`OpenTargetsEvidenceAdapter.genetics` and labels
     it for what it is: an L2G score is ``PREDICTED`` provenance, an association
     score is ``CURATED``.

     What that layer says about EPHX2 is itself a finding, and it is one a
     reflexive "add Gentropy" would have obscured: all 326 credible sets naming
     EPHX2 in release 26.06 are molecular QTL (218 eQTL, 67 tuQTL, 29 sQTL, 10
     scQTL, 2 pQTL), there is **no** GWAS credible set among them, and **no** L2G
     prediction assigns a locus to EPHX2. The ``genetic_association`` 0.5040
     supporting Alzheimer disease therefore comes from other datasources, not from
     an L2G assignment — which is a materially different statement from "GWAS
     points at this gene", and the tool is required to be able to say so.

**What this tool deliberately does not do.** It does not rank targets, combine the
association score with anything, or fold the score into
``EvidenceItem.quality_score``. The association score is a harmonic-sum over
weighted datasource evidence — a measure of *how much evidence there is*, not of
how good any single piece is — and putting it in a field named "quality" invites a
downstream ranker to read it as confidence. It stays in ``value`` under its own
name, with its datatype breakdown beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.evidence import EvidenceItem
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import TargetEvidenceAdapter, TargetEvidenceError

#: This adapter's contract with the snapshot format, not Open Targets' own version.
OPENTARGETS_ADAPTER_VERSION = "1.0.0"

#: The command that populates the cache. Named in every refusal.
FETCH_COMMAND = "make opentargets-fetch TARGETS=<ENSG id>"

#: Modalities the Platform reports tractability buckets for.
MODALITY_NAMES: dict[str, str] = {
    "SM": "small molecule",
    "AB": "antibody",
    "PR": "PROTAC / protein degrader",
    "OC": "other clinical modality",
}

SPEC = ExternalSpec(
    name="Open Targets Platform snapshot",
    env_prefix="OPENTARGETS",
    # The "checkout" is a data cache rather than a source tree: this capability has
    # no upstream code to clone, only upstream *answers* to persist.
    checkout_relpath=".external/opentargets",
    # index.json is the cache's data card and queries.graphql is the exact text the
    # snapshots were produced with. A directory with neither is not a snapshot
    # cache, whatever else it contains.
    sentinel_files=("index.json", "queries.graphql"),
    # In-process, standard library only: the snapshot is JSON.
    driver_name=None,
    selection_modules=(),
    runtime_modules=(),
    # `weight_relpaths` names a *directory* here, as PoseBench's spec does for its
    # benchmark dataset, and `_weight_present` requires a non-empty one. That is
    # what makes an interrupted fetch — index.json written, no snapshot under it —
    # report UNAVAILABLE, instead of AVAILABLE with every target refused.
    weight_relpaths=("targets",),
    conda_envs=(),
    binary=None,
    upstream_url="https://platform.opentargets.org",
    upstream_ref="GraphQL API v4 (built 26.6.3, data release 26.06)",
    license=(
        "Open Targets Platform data are CC0 1.0; individual datasources keep their own terms "
        "(https://platform-docs.opentargets.org/licence). The snapshot is a verbatim copy of "
        "the API's answer to a recorded query."
    ),
    install_hint=FETCH_COMMAND,
    needs_gpu=False,
)


class OpenTargetsEvidenceAdapter(TargetEvidenceAdapter):
    """Target-disease association, tractability and safety, from a fetched snapshot."""

    capability_id = "target.evidence.opentargets"
    spec = SPEC
    engine = "opentargets-snapshot"
    version = OPENTARGETS_ADAPTER_VERSION
    produces_nothing = "reports no target evidence"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # Reading a 736 KB JSON file is a sub-second operation; the generous
        # external-tool default would only hide a hung filesystem.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 60.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Target-disease evidence (Open Targets Platform snapshot)",
            description=(
                "Association scores with their per-datatype breakdown, tractability buckets, "
                "curated safety liabilities and the fine-mapped credible sets and "
                "locus-to-gene scores of the genetics layer, read from a snapshot fetched "
                "from the Open Targets GraphQL API. Offline at runtime; a target with no "
                "snapshot is refused by name."
            ),
            input_schema_ref="str",
            output_schema_ref="EvidenceItem",
            # A snapshot is a file: the same file yields the same items, forever.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            data_restrictions=(
                "A release snapshot, not a live query. Values are Open Targets data release "
                "26.06 and do not move until the snapshot is re-fetched. Association scores "
                "are re-computed by Open Targets each release and are not comparable across "
                "releases."
            ),
            known_failure_modes=[
                "no snapshot for the requested Ensembl id (refused by name, never substituted)",
                "a symbol that the cache cannot resolve because only ids were fetched",
                "a snapshot file truncated by an interrupted fetch (raises TargetEvidenceError)",
                "an association score compared across releases, where it is not comparable",
                "a target with no GWAS credible set, where genetics evidence is QTL-only",
                "a datatype absent from datatypeScores, which means no evidence of that kind",
            ],
            validation_checks=[
                "every item carries the snapshot id and content hash it was read from",
                "association scores are Open Targets' own floats, never rescaled or combined",
                "L2G scores are labelled PREDICTED; association and tractability CURATED",
                "quality_score stays None: the association score is evidence mass, not quality",
                "a target absent from the cache raises ToolUnavailable naming the fetch command",
            ],
        )

    # --- cache lookup ------------------------------------------------------
    def cached_targets(self) -> list[dict[str, Any]]:
        """One row per snapshot on disk: id, symbol, release and what it holds."""
        self.require_available()
        index = self.index()
        targets = index.get("targets") or {}
        return [
            {
                "ensembl_id": ensembl_id,
                "approved_symbol": card.get("approved_symbol"),
                "data_version": (card.get("metadata") or {}).get("data_version"),
                "n_associated_diseases": card.get("n_associated_diseases"),
                "n_associated_diseases_cached": card.get("n_associated_diseases_cached"),
                "n_credible_sets": card.get("n_credible_sets"),
                "snapshot_id": card.get("snapshot_id"),
                "fetched_at": card.get("fetched_at"),
            }
            for ensembl_id, card in sorted((targets or {}).items())
        ]

    def data_card(self, target: str) -> dict[str, Any]:
        """The provenance of one target's snapshot: query, release, hash, timestamp."""
        self.require_available()
        return dict(self._snapshot(target)["data_card"])

    def _resolve(self, target: str) -> str:
        """An Ensembl id for *target*, which may be an id or an approved symbol.

        Resolution is against the cache's own index — the symbol map the fetch
        script wrote from the API's answer — never against a heuristic. An
        unresolvable symbol is refused rather than guessed at, because the Platform
        search itself returns near-misses (``EPHX2`` also matches
        ``ENSG00000289323``, "novel transcript, sense intronic to EPHX2").
        """
        index = self.index()
        targets = index.get("targets") or {}
        if target in targets:
            return target
        symbols = index.get("symbols") or {}
        resolved = symbols.get(target)
        if isinstance(resolved, str):
            return resolved
        known = ", ".join(sorted(symbols)) or "none"
        self.refuse(
            f"no Open Targets snapshot for {target!r} in {self.cache_dir}. Cached symbols: {known}. "
            f"Fetch it with `{FETCH_COMMAND}` (network is required for that step only), or "
            f"`python scripts/fetch_opentargets_snapshot.py --allow-network --symbol {target}`."
        )
        raise AssertionError("unreachable")  # pragma: no cover

    def _snapshot(self, target: str) -> dict[str, Any]:
        """The whole snapshot document for *target*, refusing by name when absent."""
        ensembl_id = self._resolve(target)
        path = Path(self.cache_dir) / "targets" / f"{ensembl_id}.json"
        if not path.is_file():
            self.refuse(
                f"{path} is missing: the index lists {ensembl_id} but its snapshot file is not "
                f"there. Re-fetch with `python scripts/fetch_opentargets_snapshot.py "
                f"--allow-network {ensembl_id}`."
            )
        document = self.read_json(path)
        if "payload" not in document or "data_card" not in document:
            raise TargetEvidenceError(f"{path} is not an Open Targets snapshot (no payload/data_card)")
        return document

    # --- operations --------------------------------------------------------
    def evidence(self, target: str, *, limit: int = 25) -> list[EvidenceItem]:
        """Every layer of the snapshot, in one list: association, tractability, safety, genetics.

        The single call a target agent makes. Ordering is association evidence
        first (strongest claim about disease relevance), then tractability, then
        safety, then genetics; *limit* caps the association and credible-set rows,
        which are the only two layers that can run to hundreds.
        """
        self.require_available()
        with measure(self.capability_id, "evidence", backend=self.backend) as metered:
            items = (
                self.associations(target, limit=limit)
                + self.tractability(target)
                + self.safety(target)
                + self.genetics(target, limit=limit)
            )
            metered["n_items"] = float(len(items))
        return items

    def associations(self, target: str, *, limit: int = 25, min_score: float = 0.0) -> list[EvidenceItem]:
        """Target-disease associations, strongest first, as Open Targets scored them.

        ``value`` carries the overall score, the per-datatype breakdown, the
        disease's therapeutic areas, this row's rank, and the total the release
        holds — so a caller reading 25 rows can see it is reading 25 of 620.
        """
        self.require_available()
        with measure(self.capability_id, "associations", backend=self.backend) as metered:
            document = self._snapshot(target)
            card, payload = document["data_card"], document["payload"]
            symbol, ensembl_id = self._identity(card)
            release = self._release(card)
            block = payload.get("associated_diseases") or {}
            rows = list(block.get("rows") or [])
            provenance = self.provenance(
                card,
                source="opentargets",
                source_version=release,
                trust_label=TrustLabel.CURATED,
                notes=(
                    "Open Targets overall association score: a harmonic-sum aggregation over "
                    "weighted datasource evidence, recomputed each release."
                ),
            )
            items: list[EvidenceItem] = []
            for rank, row in enumerate(rows):
                score = self._score(row.get("score"))
                if score is None or score < min_score:
                    continue
                disease = row.get("disease") or {}
                datatypes = {
                    str(d.get("id")): self._score(d.get("score")) for d in (row.get("datatypeScores") or [])
                }
                breakdown = ", ".join(f"{k} {v:.4f}" for k, v in sorted(datatypes.items()) if v is not None)
                items.append(
                    self.item(
                        kind="target_disease",
                        statement=(
                            f"{symbol} ({ensembl_id}) is associated with {disease.get('name')} "
                            f"({disease.get('id')}) at Open Targets overall score {score:.4f} "
                            f"in data release {release} [{breakdown or 'no datatype breakdown'}]."
                        ),
                        subject=ensembl_id,
                        value={
                            "disease_id": disease.get("id"),
                            "disease_name": disease.get("name"),
                            "therapeutic_areas": [
                                {"id": a.get("id"), "name": a.get("name")}
                                for a in (disease.get("therapeuticAreas") or [])
                            ],
                            "overall_score": score,
                            "datatype_scores": datatypes,
                            "rank": rank,
                            "n_associated_diseases_in_release": block.get("count"),
                            "n_associated_diseases_cached": block.get("n_rows"),
                            "data_release": release,
                        },
                        provenance=provenance,
                        trust_label=TrustLabel.CURATED,
                    )
                )
                if len(items) >= limit:
                    break
            metered["n_associations"] = float(len(items))
        return items

    def tractability(self, target: str) -> list[EvidenceItem]:
        """One item per modality: which of the Platform's buckets are true, and which are not.

        The whole boolean table is carried, not only the true rows, because "we
        checked and it is false" and "we did not check" are different statements
        and only the table distinguishes them.
        """
        self.require_available()
        with measure(self.capability_id, "tractability", backend=self.backend) as metered:
            document = self._snapshot(target)
            card, payload = document["data_card"], document["payload"]
            symbol, ensembl_id = self._identity(card)
            release = self._release(card)
            provenance = self.provenance(
                card,
                source="opentargets",
                source_version=release,
                trust_label=TrustLabel.CURATED,
                notes="Tractability buckets are curated assertions, not model predictions.",
            )
            buckets: dict[str, dict[str, bool]] = {}
            for row in payload.get("target", {}).get("tractability") or []:
                modality = str(row.get("modality") or "?")
                buckets.setdefault(modality, {})[str(row.get("label"))] = bool(row.get("value"))
            items: list[EvidenceItem] = []
            for modality, table in sorted(buckets.items()):
                true_labels = sorted(label for label, value in table.items() if value)
                items.append(
                    self.item(
                        kind="target_tractability",
                        statement=(
                            f"{symbol} ({ensembl_id}) {MODALITY_NAMES.get(modality, modality)} "
                            f"tractability in Open Targets release {release}: "
                            f"{len(true_labels)} of {len(table)} buckets true"
                            + (f" — {', '.join(true_labels)}." if true_labels else ".")
                        ),
                        subject=ensembl_id,
                        value={
                            "modality": modality,
                            "modality_name": MODALITY_NAMES.get(modality, modality),
                            "buckets": table,
                            "buckets_true": true_labels,
                            "n_true": len(true_labels),
                            "n_buckets": len(table),
                            "data_release": release,
                        },
                        provenance=provenance,
                        trust_label=TrustLabel.CURATED,
                    )
                )
            metered["n_modalities"] = float(len(items))
        return items

    def safety(self, target: str) -> list[EvidenceItem]:
        """Curated safety liabilities, or one item stating the release records none.

        An empty list would be ambiguous between "no liabilities" and "not looked
        at". A release that carries zero rows for a target is a checkable fact, so
        it is reported as one, with ``value['n_liabilities'] == 0``.
        """
        self.require_available()
        with measure(self.capability_id, "safety", backend=self.backend) as metered:
            document = self._snapshot(target)
            card, payload = document["data_card"], document["payload"]
            symbol, ensembl_id = self._identity(card)
            release = self._release(card)
            provenance = self.provenance(
                card,
                source="opentargets",
                source_version=release,
                trust_label=TrustLabel.CURATED,
                notes="Curated adverse events from the Platform's safety datasources.",
            )
            rows = list(payload.get("target", {}).get("safetyLiabilities") or [])
            if not rows:
                metered["n_liabilities"] = 0.0
                return [
                    self.item(
                        kind="target_safety",
                        statement=(
                            f"Open Targets data release {release} records no safety liabilities "
                            f"for {symbol} ({ensembl_id})."
                        ),
                        subject=ensembl_id,
                        value={"n_liabilities": 0, "event": None, "data_release": release},
                        provenance=provenance,
                        trust_label=TrustLabel.CURATED,
                    )
                ]
            items = [
                self.item(
                    kind="target_safety",
                    statement=(
                        f"{symbol} ({ensembl_id}) safety liability from {row.get('datasource')}: "
                        f"{row.get('event') or 'unnamed event'}"
                        f"{' (' + str(row.get('eventId')) + ')' if row.get('eventId') else ''}."
                    ),
                    subject=ensembl_id,
                    value={
                        "n_liabilities": len(rows),
                        "event": row.get("event"),
                        "event_id": row.get("eventId") or None,
                        "datasource": row.get("datasource"),
                        "literature": row.get("literature"),
                        "url": row.get("url"),
                        "effects": row.get("effects") or [],
                        "biosamples": row.get("biosamples") or [],
                        "data_release": release,
                    },
                    provenance=provenance,
                    trust_label=TrustLabel.CURATED,
                )
                for row in rows
            ]
            metered["n_liabilities"] = float(len(items))
        return items

    def genetics(self, target: str, *, limit: int = 25) -> list[EvidenceItem]:
        """The genetics layer: fine-mapped credible sets and locus-to-gene scores.

        This is Gentropy's output as the Platform serves it — see the module
        docstring for why it is read rather than re-run. The first item is always a
        summary of the whole credible-set collection by study type, because "326
        credible sets, none of them GWAS" is the finding for some targets and it
        would be invisible in a list of individual loci. Loci carrying an L2G
        prediction for this target follow, highest score first, labelled
        ``PREDICTED``.
        """
        self.require_available()
        with measure(self.capability_id, "genetics", backend=self.backend) as metered:
            document = self._snapshot(target)
            card, payload = document["data_card"], document["payload"]
            symbol, ensembl_id = self._identity(card)
            release = self._release(card)
            block = payload.get("credible_sets") or {}
            rows = list(block.get("rows") or [])

            by_type: dict[str, int] = {}
            for row in rows:
                key = str(row.get("studyType") or "unknown")
                by_type[key] = by_type.get(key, 0) + 1

            scored: list[tuple[float, dict[str, Any]]] = []
            for row in rows:
                for prediction in (row.get("l2GPredictions") or {}).get("rows") or []:
                    if (prediction.get("target") or {}).get("id") != ensembl_id:
                        continue
                    score = self._score(prediction.get("score"))
                    if score is not None:
                        scored.append((score, row))
            scored.sort(key=lambda pair: -pair[0])

            summary_provenance = self.provenance(
                card,
                source="opentargets",
                source_version=release,
                trust_label=TrustLabel.CURATED,
                notes=(
                    "Credible sets are produced upstream by Gentropy's fine-mapping "
                    "(SuSiE/PICS) over the release's GWAS and molecular-QTL studies."
                ),
            )
            breakdown = ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())) or "none"
            items = [
                self.item(
                    kind="target_genetics_summary",
                    statement=(
                        f"{symbol} ({ensembl_id}) has {block.get('count', len(rows))} fine-mapped "
                        f"credible sets in Open Targets release {release} ({breakdown}); "
                        f"{len(scored)} carry a locus-to-gene prediction naming this target."
                    ),
                    subject=ensembl_id,
                    value={
                        "n_credible_sets_in_release": block.get("count"),
                        "n_credible_sets_cached": block.get("n_rows", len(rows)),
                        "credible_sets_truncated": bool(block.get("truncated")),
                        "by_study_type": by_type,
                        "n_gwas_credible_sets": by_type.get("gwas", 0),
                        "n_l2g_predictions_for_target": len(scored),
                        "max_l2g_score": scored[0][0] if scored else None,
                        "data_release": release,
                    },
                    provenance=summary_provenance,
                    trust_label=TrustLabel.CURATED,
                )
            ]

            l2g_provenance = self.provenance(
                card,
                source="opentargets",
                source_version=release,
                trust_label=TrustLabel.PREDICTED,
                notes=(
                    "Locus-to-gene score: the output of Open Targets' trained L2G classifier "
                    "for this release, read from the Platform rather than recomputed."
                ),
            )
            for score, row in scored[:limit]:
                study = row.get("study") or {}
                variant = row.get("variant") or {}
                items.append(
                    self.item(
                        kind="target_locus_to_gene",
                        statement=(
                            f"Locus {row.get('studyLocusId')} ({study.get('studyType')} study "
                            f"{study.get('id')}, lead variant {variant.get('id')}) is assigned to "
                            f"{symbol} with locus-to-gene score {score:.4f} in release {release}."
                        ),
                        subject=ensembl_id,
                        value={
                            "l2g_score": score,
                            "study_locus_id": row.get("studyLocusId"),
                            "study_id": study.get("id"),
                            "study_type": study.get("studyType"),
                            "trait_from_source": study.get("traitFromSource"),
                            "diseases": study.get("diseases") or [],
                            "variant_id": variant.get("id"),
                            "rs_ids": variant.get("rsIds") or [],
                            "chromosome": row.get("chromosome"),
                            "position": row.get("position"),
                            "beta": row.get("beta"),
                            "standard_error": row.get("standardError"),
                            "p_value_mantissa": row.get("pValueMantissa"),
                            "p_value_exponent": row.get("pValueExponent"),
                            "finemapping_method": row.get("finemappingMethod"),
                            "data_release": release,
                        },
                        provenance=l2g_provenance,
                        trust_label=TrustLabel.PREDICTED,
                    )
                )
            metered["n_credible_sets"] = float(len(rows))
            metered["n_l2g_predictions"] = float(len(scored))
        return items

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _identity(card: dict[str, Any]) -> tuple[str, str]:
        """``(approved symbol, ensembl id)`` as the snapshot recorded them."""
        ensembl_id = str(card.get("ensembl_id") or "")
        return str(card.get("approved_symbol") or ensembl_id), ensembl_id

    @staticmethod
    def _release(card: dict[str, Any]) -> str:
        """The data release the snapshot was taken from, e.g. ``'26.06'``."""
        metadata = card.get("metadata") or {}
        return str(metadata.get("data_version") or "unknown")

    @staticmethod
    def _score(raw: Any) -> float | None:
        """A score from the snapshot as a float, or ``None`` if it is not one.

        Upstream nulls stay ``None``. Coercing a missing score to 0.0 would assert
        "no association", which is a claim the release did not make.
        """
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


__all__ = ["FETCH_COMMAND", "OPENTARGETS_ADAPTER_VERSION", "SPEC", "OpenTargetsEvidenceAdapter"]
