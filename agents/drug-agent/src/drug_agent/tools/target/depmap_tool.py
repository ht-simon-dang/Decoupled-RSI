"""DepMap — CRISPR gene effect, and where a dependency is selective.

DepMap answers a question association evidence cannot: *if you removed this gene,
would the cell die, and in which cells?* The Broad's Achilles project knocks out
every gene in every cell line of a large panel with pooled CRISPR screens, and
Chronos turns the guide-level readouts into a per-(gene, cell line) **gene effect**
score on a scale fixed by controls: **0 is no effect, and -1 is roughly the median
of common essential genes**. A gene effect at or below -0.5 is DepMap's own
threshold for calling a line *dependent*.

**This is not a substitute for association evidence, and it is not scored as one.**
``target.evidence.opentargets`` says a target is implicated in a disease.
``target.essentiality.depmap`` says a gene is required for the growth of cancer
cell lines. Those overlap only for oncology targets, and for everything else a
flat DepMap profile is the *expected* result rather than a negative finding — a
well-tolerated enzyme inhibitor target should look non-essential. So the two
capabilities are separate, neither covers for the other's absence, and this module
never converts a gene effect into a "target quality" number.

**Measured for this repository's own target.** DepMap Public 24Q4, all 1,178 cell
lines: EPHX2 has median gene effect **-0.0040** (mean -0.0045, range -0.3552 to
+0.6023), **zero** lines at or below the -0.5 dependency threshold, median
dependency probability **0.0230** (max 0.3018), and it is **not** in
``CRISPRInferredCommonEssentials``. Against the same release's controls — mean
gene effect over DepMap's 1,242 common-essential control genes is **-1.0758** and
over its 726 nonessential controls **-0.0116** — EPHX2 sits on the nonessential
anchor. The honest reading is that sEH is not a cancer-cell-autonomous dependency,
and that essentiality is not the argument for drugging it. The controls in the
same cache prove the pipeline can see a dependency when there is one, in both
regimes the tool must be able to tell apart: RPL3 is pan-essential (median
**-2.5259**, all 1,178 lines below the threshold), while KRAS is *selective* —
median **-0.3732** overall but **-1.7580** across 47 pancreas lines, 43 of them
dependent. A flat EPHX2 profile is therefore a measured result, not a blind spot.

**Why a subset, and exactly which one.** ``CRISPRGeneEffect.csv`` is 428,678,699
bytes and ``CRISPRGeneDependency.csv`` 421,115,594 — 17,916 genes across 1,178
lines each. ``scripts/fetch_depmap_subset.py`` streams both, verifies each against
the MD5 figshare publishes, keeps the columns of a declared 18-gene panel and
discards the rest without writing it. What is cached is upstream's own numbers for
those 18 genes over **all** 1,178 lines; what is not cached is every other gene,
and this adapter refuses those by name with the command that would add them. The
narrowing is written into ``index.json`` as ``genes_cached``,
``n_genes_in_release`` and a ``coverage`` sentence, so no reader can mistake the
cache for the release.

Three small files are cached whole because they are small and they are what makes
a single gene-effect number interpretable: ``Model.csv`` (cell-line annotation,
including ``OncotreeLineage``, which is how selectivity is asked at all),
``CRISPRInferredCommonEssentials.csv`` (DepMap's own pan-essential call), and the
two Achilles control lists. ``control_reference.csv`` is the one derived table:
per cell line, the mean gene effect over DepMap's own common-essential and
nonessential control sets, computed by the fetch script over **every** such column
of the full release while the stream was already passing through. It is labelled
``computed_by`` in the data card, its formula is recorded, and it is never
presented as a DepMap-published quantity.

**Offline at runtime.** This module imports no HTTP client. A gene outside
``genes_cached`` raises :class:`~drug_agent.tools.base.ToolUnavailable` naming the
gene and the fetch command; it is never answered from a paralogue, a lineage
average, or a value carried over from another release. DepMap re-fits Chronos each
quarter, so scores are not comparable across releases and the release is on every
item.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.evidence import EvidenceItem
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import TargetEvidenceAdapter, TargetEvidenceError

#: This adapter's contract with the cache format, not DepMap's own version.
DEPMAP_ADAPTER_VERSION = "1.0.0"

#: The command that populates the cache. Named in every refusal.
FETCH_COMMAND = "make depmap-fetch GENES=<symbol>"

#: DepMap's own threshold for calling a cell line dependent on a gene, on the
#: Chronos scale where 0 is no effect and -1 is the common-essential median.
DEPENDENT_THRESHOLD = -0.5

#: Gene-effect below this is a strong dependency — the common-essential anchor.
STRONG_DEPENDENCY_THRESHOLD = -1.0

#: Minimum lines in a lineage before its median is reported as a selectivity claim.
#: A lineage with two cell lines has a median; it does not have evidence.
MIN_LINES_PER_LINEAGE = 5

#: Cached tables the fetch script writes only when it was asked to. Their absence
#: is a *choice an operator made* (``--skip-dependency-probability`` saves a second
#: 421 MB stream), not a corrupt cache, so it is reported as "not cached" rather
#: than raised — and never as a zero, which would assert "no dependency". Every
#: other table is a sentinel of the spec, so a missing one really is corruption.
OPTIONAL_TABLES = frozenset({"gene_dependency.csv"})

SPEC = ExternalSpec(
    name="DepMap Public 24Q4 gene-effect subset",
    env_prefix="DEPMAP",
    # A data cache, not a source tree: DepMap publishes release files, not code
    # this platform runs.
    checkout_relpath=".external/depmap",
    sentinel_files=(
        "index.json",
        "gene_effect.csv",
        "Model.csv",
        "CRISPRInferredCommonEssentials.csv",
        "control_reference.csv",
    ),
    # In-process, standard library only: the cache is CSV and JSON.
    driver_name=None,
    selection_modules=(),
    runtime_modules=(),
    weight_relpaths=(),
    conda_envs=(),
    binary=None,
    upstream_url="https://depmap.org/portal/ (figshare+ article 27993248)",
    upstream_ref="DepMap Public 24Q4, doi 10.25452/figshare.plus.27993248.v1",
    license=(
        "DepMap Public 24Q4 is released CC BY 4.0 by the Broad Institute. The cache holds a "
        "column subset of CRISPRGeneEffect.csv and CRISPRGeneDependency.csv, MD5-verified "
        "against figshare's published digests."
    ),
    install_hint=FETCH_COMMAND,
    needs_gpu=False,
)


class DepMapEssentialityAdapter(TargetEvidenceAdapter):
    """CRISPR gene effect and dependency probability, from a pinned release subset."""

    capability_id = "target.essentiality.depmap"
    spec = SPEC
    engine = "depmap-24q4"
    version = DEPMAP_ADAPTER_VERSION
    produces_nothing = "reports no essentiality evidence"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # The cache is a few hundred kilobytes of CSV; a minute is already generous.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 60.0)
        self._tables: dict[str, Any] = {}

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="CRISPR gene essentiality and selectivity (DepMap Public 24Q4)",
            description=(
                "Chronos gene-effect scores and dependency probabilities for a cached gene "
                "panel across all 1,178 cell lines of DepMap Public 24Q4, with DepMap's own "
                "common-essential call, per-lineage selectivity, and the release's own "
                "essential and nonessential control anchors. Offline; a gene outside the "
                "cached panel is refused by name."
            ),
            input_schema_ref="str",
            output_schema_ref="EvidenceItem",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            data_restrictions=(
                "Cancer cell lines only — a gene effect says nothing about a normal tissue or "
                "a whole organism. Chronos is re-fit each quarterly release, so scores are not "
                "comparable across releases. The cache holds 17 of 17,916 genes; every other "
                "gene of the release is absent and is refused rather than approximated."
            ),
            known_failure_modes=[
                "a gene outside the cached panel (refused by name, never inferred)",
                "a lineage with too few cell lines to support a selectivity claim (reported, not ranked)",
                "reading a flat profile as a negative finding for a non-oncology target",
                "comparing a gene effect against a different DepMap release",
                "an interrupted fetch leaving a truncated CSV (raises TargetEvidenceError)",
                "cell lines in gene_effect.csv but absent from Model.csv (counted as unknown lineage)",
            ],
            validation_checks=[
                "every item carries the DepMap release and the cache's snapshot id",
                "gene-effect values are the release's own floats, never rescaled",
                "the -0.5 dependency threshold is DepMap's, and is stated on every summary",
                "the control anchors are computed over the release's own control gene lists",
                "an uncached gene raises ToolUnavailable naming the fetch command",
            ],
        )

    # --- cache -------------------------------------------------------------
    def cached_genes(self) -> list[str]:
        """The HGNC symbols this cache holds, and only those."""
        self.require_available()
        return sorted(self._symbols())

    def data_card(self) -> dict[str, Any]:
        """The release, the file digests, the gene coverage and the fetch timestamp."""
        self.require_available()
        return dict(self.index())

    def _symbols(self) -> dict[str, str]:
        """``{'EPHX2': 'EPHX2 (2053)'}`` — symbol to the column header it maps to."""
        cached = self._tables.get("symbols")
        if cached is None:
            header = self._matrix("gene_effect.csv")[0]
            cached = {self._symbol(column): column for column in header[1:]}
            self._tables["symbols"] = cached
        return dict(cached)

    def _matrix(self, name: str) -> tuple[list[str], list[list[str]]]:
        """``(header, rows)`` of one cached CSV, read once per adapter instance."""
        cached = self._tables.get(name)
        if cached is None:
            path = Path(self.cache_dir) / name
            if name in OPTIONAL_TABLES and not path.is_file():
                # Absent on purpose: report nothing rather than raise or invent.
                self._tables[name] = ([], [])
                return [], []
            try:
                with path.open(newline="") as handle:
                    rows = list(csv.reader(handle))
            except OSError as exc:
                raise TargetEvidenceError(f"{path} is present but unreadable: {exc}") from exc
            if not rows:
                raise TargetEvidenceError(f"{path} is empty; re-run `{FETCH_COMMAND}`")
            cached = (rows[0], rows[1:])
            self._tables[name] = cached
        return cached

    def _models(self) -> dict[str, dict[str, str]]:
        """``Model.csv`` keyed by ModelID — cell line name, lineage, primary disease."""
        cached = self._tables.get("models")
        if cached is None:
            path = Path(self.cache_dir) / "Model.csv"
            try:
                with path.open(newline="") as handle:
                    cached = {row["ModelID"]: row for row in csv.DictReader(handle) if row.get("ModelID")}
            except (OSError, KeyError) as exc:
                raise TargetEvidenceError(f"{path} is not a DepMap Model.csv: {exc}") from exc
            self._tables["models"] = cached
        return cached

    def _common_essentials(self) -> set[str]:
        """DepMap's own pan-essential call for this release, as symbols."""
        cached = self._tables.get("common_essentials")
        if cached is None:
            path = Path(self.cache_dir) / "CRISPRInferredCommonEssentials.csv"
            try:
                lines = path.read_text().splitlines()
            except OSError as exc:
                raise TargetEvidenceError(f"{path} is present but unreadable: {exc}") from exc
            cached = {self._symbol(line.strip()) for line in lines[1:] if line.strip()}
            self._tables["common_essentials"] = cached
        return set(cached)

    def _column(self, name: str, gene: str) -> dict[str, float]:
        """``{ModelID: value}`` for one gene of one cached matrix, missing values dropped."""
        header, rows = self._matrix(name)
        column = self._symbols().get(gene.upper())
        if column is None or column not in header:
            return {}
        index = header.index(column)
        out: dict[str, float] = {}
        for row in rows:
            if len(row) <= index or row[index] == "":
                continue
            try:
                out[row[0]] = float(row[index])
            except ValueError:  # pragma: no cover - upstream writes floats or nothing
                continue
        return out

    def _require_gene(self, gene: str) -> str:
        """The cached symbol for *gene*, or a refusal naming it and the fetch command."""
        symbol = gene.upper()
        symbols = self._symbols()
        if symbol in symbols:
            return symbol
        card = self.index()
        self.refuse(
            f"{symbol} is not in the DepMap cache at {self.cache_dir}: it holds "
            f"{len(symbols)} of {card.get('n_genes_in_release')} genes of "
            f"{card.get('release')} ({', '.join(sorted(symbols))}). Add it with "
            f"`python scripts/fetch_depmap_subset.py --allow-network --gene {symbol}` "
            f"(that step streams the 428 MB release file once) — this tool will not "
            f"infer an uncached gene from a paralogue or a lineage average."
        )
        raise AssertionError("unreachable")  # pragma: no cover

    # --- operations --------------------------------------------------------
    def essentiality(self, gene: str) -> EvidenceItem:
        """Is this gene a dependency, and how does it sit against the release's controls?

        One item, carrying the whole distribution summary rather than a verdict: the
        median, mean and range of the gene effect over every cell line screened, how
        many lines clear DepMap's -0.5 dependency threshold, DepMap's own
        common-essential call, the dependency probability when it is cached, and the
        release's essential and nonessential control anchors so the number can be
        read at all.
        """
        self.require_available()
        with measure(self.capability_id, "essentiality", backend=self.backend) as metered:
            symbol = self._require_gene(gene)
            card = self.index()
            release = str(card.get("release") or "unknown")
            effect = self._column("gene_effect.csv", symbol)
            if not effect:
                raise TargetEvidenceError(
                    f"gene_effect.csv holds no values for {symbol}; re-run `{FETCH_COMMAND}`"
                )
            values = sorted(effect.values())
            dependent = [v for v in values if v <= DEPENDENT_THRESHOLD]
            strong = [v for v in values if v <= STRONG_DEPENDENCY_THRESHOLD]
            probability = self._column("gene_dependency.csv", symbol)
            anchors = self._control_anchors()
            common_essential = symbol in self._common_essentials()
            median_effect = statistics.median(values)

            value: dict[str, Any] = {
                "gene": symbol,
                "release": release,
                "n_cell_lines": len(values),
                "median_gene_effect": round(median_effect, 6),
                "mean_gene_effect": round(statistics.fmean(values), 6),
                "min_gene_effect": round(values[0], 6),
                "max_gene_effect": round(values[-1], 6),
                "n_dependent_lines": len(dependent),
                "n_strongly_dependent_lines": len(strong),
                "fraction_dependent_lines": round(len(dependent) / len(values), 6),
                "dependent_threshold": DEPENDENT_THRESHOLD,
                "common_essential": common_essential,
                "common_essential_source": "CRISPRInferredCommonEssentials.csv",
                "control_anchors": anchors,
            }
            value["dependency_probability_cached"] = bool(probability)
            if probability:
                probabilities = sorted(probability.values())
                value["median_dependency_probability"] = round(statistics.median(probabilities), 6)
                value["max_dependency_probability"] = round(probabilities[-1], 6)
                value["n_lines_dependency_probability"] = len(probabilities)
            else:
                # Say the number is absent, and how to get it. Omitting the key
                # silently would read downstream as "no dependency probability
                # exists", which is a claim about DepMap rather than about this cache.
                value["dependency_probability_note"] = (
                    "CRISPRGeneDependency.csv was not cached (fetched with "
                    "--skip-dependency-probability); re-run scripts/fetch_depmap_subset.py "
                    "without that flag to add it."
                )

            statement = (
                f"{symbol} in {release}: median CRISPR gene effect {median_effect:.4f} over "
                f"{len(values)} cell lines (range {values[0]:.4f} to {values[-1]:.4f}); "
                f"{len(dependent)} line(s) at or below DepMap's {DEPENDENT_THRESHOLD} dependency "
                f"threshold; DepMap "
                f"{'lists' if common_essential else 'does not list'} it as a common essential. "
                f"Release controls anchor at {anchors['common_essential_control_mean']} "
                f"(common essential) and {anchors['nonessential_control_mean']} (nonessential)."
            )
            metered["median_gene_effect"] = float(median_effect)
            metered["n_dependent_lines"] = float(len(dependent))
        return self.item(
            kind="target_essentiality",
            statement=statement,
            subject=symbol,
            value=value,
            provenance=self.provenance(
                card,
                source="depmap",
                source_version=release,
                trust_label=TrustLabel.OBSERVED,
                notes=(
                    "Chronos gene-effect estimated from pooled CRISPR screen readouts — an "
                    "assay-derived quantity, on a scale where 0 is no effect and -1 is the "
                    "common-essential median. Not comparable across DepMap releases."
                ),
            ),
            trust_label=TrustLabel.OBSERVED,
        )

    def gene_effect(self, gene: str, *, limit: int = 25) -> list[EvidenceItem]:
        """The most dependent cell lines for *gene*, one item each, strongest first.

        This is the per-line detail behind :meth:`essentiality`. Cell lines are
        annotated from ``Model.csv``; a line present in the matrix but absent from
        the annotation table is returned with an unknown lineage rather than
        dropped, because dropping it would quietly change the denominator.
        """
        self.require_available()
        with measure(self.capability_id, "gene_effect", backend=self.backend) as metered:
            symbol = self._require_gene(gene)
            card = self.index()
            release = str(card.get("release") or "unknown")
            effect = self._column("gene_effect.csv", symbol)
            probability = self._column("gene_dependency.csv", symbol)
            models = self._models()
            provenance = self.provenance(
                card,
                source="depmap",
                source_version=release,
                trust_label=TrustLabel.OBSERVED,
                notes="One cell line's Chronos gene effect for this gene in this release.",
            )
            ranked = sorted(effect.items(), key=lambda pair: pair[1])
            items: list[EvidenceItem] = []
            for model_id, score in ranked[:limit]:
                model = models.get(model_id, {})
                lineage = model.get("OncotreeLineage") or "unknown"
                verdict = (
                    f" — at or below the {DEPENDENT_THRESHOLD} dependency threshold"
                    if score <= DEPENDENT_THRESHOLD
                    else ""
                )
                items.append(
                    self.item(
                        kind="target_essentiality_cell_line",
                        statement=(
                            f"{symbol} gene effect {score:.4f} in {model.get('CellLineName') or model_id} "
                            f"({model_id}, {lineage}) in {release}{verdict}."
                        ),
                        subject=symbol,
                        value={
                            "gene": symbol,
                            "model_id": model_id,
                            "cell_line": model.get("CellLineName"),
                            "lineage": lineage,
                            "primary_disease": model.get("OncotreePrimaryDisease"),
                            "gene_effect": round(score, 6),
                            "dependency_probability": (
                                round(probability[model_id], 6) if model_id in probability else None
                            ),
                            "dependent": score <= DEPENDENT_THRESHOLD,
                            "release": release,
                        },
                        provenance=provenance,
                        trust_label=TrustLabel.OBSERVED,
                    )
                )
            metered["n_cell_lines"] = float(len(items))
        return items

    def selectivity(self, gene: str) -> list[EvidenceItem]:
        """Per-lineage medians: where, if anywhere, the dependency is selective.

        One item per Oncotree lineage with at least :data:`MIN_LINES_PER_LINEAGE`
        screened lines. Smaller lineages are counted in the summary but not turned
        into a claim — a two-line median is arithmetic, not evidence — and the
        excluded lineages are named in every item so the omission is visible.
        """
        self.require_available()
        with measure(self.capability_id, "selectivity", backend=self.backend) as metered:
            symbol = self._require_gene(gene)
            card = self.index()
            release = str(card.get("release") or "unknown")
            effect = self._column("gene_effect.csv", symbol)
            models = self._models()

            grouped: dict[str, list[float]] = {}
            for model_id, score in effect.items():
                lineage = (models.get(model_id, {}).get("OncotreeLineage") or "unknown") or "unknown"
                grouped.setdefault(lineage, []).append(score)
            overall = statistics.median(sorted(effect.values()))
            small = sorted(k for k, v in grouped.items() if len(v) < MIN_LINES_PER_LINEAGE)

            provenance = self.provenance(
                card,
                source="depmap",
                source_version=release,
                trust_label=TrustLabel.OBSERVED,
                notes=(
                    "Per-lineage median of DepMap's own gene-effect values; the grouping is "
                    "Model.csv's OncotreeLineage and the median is the only thing computed here."
                ),
            )
            items: list[EvidenceItem] = []
            for lineage, scores in sorted(grouped.items(), key=lambda kv: statistics.median(kv[1])):
                if len(scores) < MIN_LINES_PER_LINEAGE:
                    continue
                median = statistics.median(scores)
                dependent = [s for s in scores if s <= DEPENDENT_THRESHOLD]
                items.append(
                    self.item(
                        kind="target_essentiality_lineage",
                        statement=(
                            f"{symbol} median gene effect {median:.4f} across {len(scores)} "
                            f"{lineage} cell lines in {release} "
                            f"({len(dependent)} at or below {DEPENDENT_THRESHOLD}); "
                            f"all-lineage median {overall:.4f}."
                        ),
                        subject=symbol,
                        value={
                            "gene": symbol,
                            "lineage": lineage,
                            "n_cell_lines": len(scores),
                            "median_gene_effect": round(median, 6),
                            "min_gene_effect": round(min(scores), 6),
                            "n_dependent_lines": len(dependent),
                            "delta_vs_all_lineages": round(median - overall, 6),
                            "all_lineage_median": round(overall, 6),
                            "min_cell_lines_for_a_claim": MIN_LINES_PER_LINEAGE,
                            "lineages_below_minimum": small,
                            "release": release,
                        },
                        provenance=provenance,
                        trust_label=TrustLabel.OBSERVED,
                    )
                )
            metered["n_lineages"] = float(len(items))
        return items

    # --- helpers -----------------------------------------------------------
    def _control_anchors(self) -> dict[str, Any]:
        """The release's own essential and nonessential anchors, from the cache.

        These are the two numbers a gene effect has to be read against, and they are
        derived rather than published: ``scripts/fetch_depmap_subset.py`` averaged
        DepMap's own control gene sets per cell line over the full release, and this
        averages those per-line means. The data card records the formula and marks
        the table ``computed_by`` that script; it is never presented as a DepMap
        number.
        """
        cached = self._tables.get("anchors")
        if cached is None:
            header, rows = self._matrix("control_reference.csv")
            anchors: dict[str, Any] = {}
            for key, column in (
                ("common_essential_control_mean", "common_essential_control_mean_gene_effect"),
                ("nonessential_control_mean", "nonessential_control_mean_gene_effect"),
            ):
                if column not in header:
                    anchors[key] = None
                    continue
                index = header.index(column)
                values = [float(r[index]) for r in rows if len(r) > index and r[index] != ""]
                anchors[key] = round(statistics.fmean(values), 6) if values else None
            anchors["computed_by"] = "scripts/fetch_depmap_subset.py + this adapter (mean of means)"
            anchors["control_gene_sets"] = (
                (self.index().get("files") or {})
                .get("CRISPRGeneEffect.csv", {})
                .get("control_reference", {})
                .get("sets")
            )
            cached = anchors
            self._tables["anchors"] = cached
        return dict(cached)

    @staticmethod
    def _symbol(header: str) -> str:
        """``'EPHX2 (2053)'`` -> ``'EPHX2'``; a header with no Entrez id is its own symbol."""
        cut = header.find(" (")
        return header[:cut] if cut > 0 and header.endswith(")") else header


__all__ = [
    "DEPENDENT_THRESHOLD",
    "DEPMAP_ADAPTER_VERSION",
    "FETCH_COMMAND",
    "SPEC",
    "DepMapEssentialityAdapter",
]
