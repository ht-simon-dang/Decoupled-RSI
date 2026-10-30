"""Fetch a pinned DepMap release and persist the gene subset this platform reads.

This is the **setup** half of ``target.essentiality.depmap``. It is the only piece
of that capability allowed to touch the network, and it runs once. The adapter
(:mod:`drug_agent.tools.target.depmap_tool`) never opens a socket: it reads what
this script wrote, and reports UNAVAILABLE naming this command when it is absent.

**Why a subset, stated precisely.** DepMap Public 24Q4 publishes
``CRISPRGeneEffect.csv`` at 428,678,699 bytes — 1,178 cell lines x 17,916 genes of
Chronos gene-effect scores — and ``CRISPRGeneDependency.csv`` at 421,115,594 bytes.
Vendoring either into a repository is not reasonable, and re-downloading them per
run is exactly the live call this platform forbids. So this script *streams* both
files, keeps only the columns of a declared gene panel, and discards the rest
without ever writing it to disk. What it keeps is upstream's own numbers, byte for
byte; what it drops is recorded, so the cache can never be mistaken for the whole
release:

    ``index.json`` names ``genes_cached`` (the exact resolved column headers),
    ``n_genes_in_release``, ``n_models_in_release`` and
    ``genes_requested_not_found``. The adapter refuses any gene outside
    ``genes_cached`` with the command that would add it. Nothing is narrowed
    silently and nothing is interpolated across the gap.

**Integrity is checked, not assumed.** figshare publishes an MD5 per file. Every
stream is hashed as it is consumed and the digest is compared with the published
one; a mismatch aborts before anything is written. That is what makes "we read the
real release" a verified statement rather than a claim about a URL.

**One derived table, labelled as derived.** ``control_reference.csv`` holds, per
cell line, the mean gene effect over DepMap's *own* common-essential control set
(``AchillesCommonEssentialControls.csv``, 1,247 genes, of which 1,242 are columns of
this release's matrix) and over its nonessential control set
(``AchillesNonessentialControls.csv``, 781 genes, 726 of them columns). The data
card records how many columns each set actually matched. Those two means are
computed here, over every one of those columns in the full release, while the
stream is already passing through — so they are a real calculation over the whole
release rather than over the cached panel. They are the calibration a single
gene-effect number needs (0 is "no effect", -1 is roughly the common-essential
median), and ``index.json`` records the formula and marks them ``computed_by``
this script rather than reported by DepMap.

Usage::

    python scripts/fetch_depmap_subset.py --allow-network
    python scripts/fetch_depmap_subset.py --allow-network --gene EPHX2 --gene KRAS

Network egress goes through :class:`drug_agent.security.egress.EgressPolicy` with
an allowlist pinned to the figshare hosts, so a redirect to somewhere else is
refused rather than followed silently. Without ``--allow-network`` the script
inherits the deny-by-default policy and stops before the first request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drug_agent.data.snapshot import DataSnapshot
from drug_agent.security.egress import EgressPolicy
from drug_agent.util import utcnow_iso

#: The release this script is pinned to. DepMap re-publishes quarterly and the
#: gene-effect scale is re-fit each time, so a cache that did not name its release
#: would be a number without a source.
RELEASE = "DepMap Public 24Q4"
FIGSHARE_ARTICLE = 27993248
FIGSHARE_DOI = "10.25452/figshare.plus.27993248.v1"
FIGSHARE_ARTICLE_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
RELEASE_PUBLISHED = "2024-12-10T16:07:56Z"
RELEASE_LICENSE = "CC BY 4.0 (Broad Institute DepMap portal, figshare+)"

#: Hosts this script is allowed to reach. ``ndownloader`` redirects to figshare's
#: object store, which urllib follows internally; both are named so the allowlist
#: describes the whole path rather than only its first hop.
ALLOWED_HOSTS = (
    "api.figshare.com",
    "ndownloader.figshare.com",
    "figshare.com",
    "s3-eu-west-1.amazonaws.com",
)

#: Where the cache lives, relative to the repository root.
DEFAULT_CACHE_RELPATH = ".external/depmap"


@dataclass(frozen=True)
class ReleaseFile:
    """One file of the pinned release, with the size and digest figshare publishes."""

    name: str
    file_id: int
    size_bytes: int
    md5: str
    #: ``"verbatim"`` files are small enough to keep whole; ``"subset"`` files are
    #: streamed and reduced to the gene panel without ever landing on disk.
    mode: str

    @property
    def url(self) -> str:
        return f"https://ndownloader.figshare.com/files/{self.file_id}"


#: The files this cache is built from. Sizes and MD5s are figshare's own, read from
#: the article metadata on 2026-09-08; they are asserted here so a silent
#: re-publication under the same file id cannot pass unnoticed.
RELEASE_FILES: tuple[ReleaseFile, ...] = (
    ReleaseFile("CRISPRGeneEffect.csv", 51064667, 428_678_699, "6edf7ade09b9b34199210b559d4745d3", "subset"),
    ReleaseFile(
        "CRISPRGeneDependency.csv", 51064631, 421_115_594, "0d3bdadf0c59264e39f7fbadf232ccdb", "subset"
    ),
    ReleaseFile("Model.csv", 51065297, 645_696, "675210d17675f3517b0ce39a3c274f16", "verbatim"),
    ReleaseFile(
        "CRISPRInferredCommonEssentials.csv", 51064916, 20_795, "545edef0d377f252db6e5e675201bd77", "verbatim"
    ),
    ReleaseFile(
        "AchillesCommonEssentialControls.csv",
        51063560,
        17_015,
        "9cc07e7ebc282984d8107341dfde8f26",
        "verbatim",
    ),
    ReleaseFile(
        "AchillesNonessentialControls.csv", 51063566, 11_490, "ca6a757db1cae49b913423ce73b818cf", "verbatim"
    ),
)

#: The default gene panel, and why each block is in it. Every one of these is a
#: real column of the release; the script fails loudly on any that is not.
#:
#: * the programme's target and its paralogues — the question the tool exists for;
#: * the CYP epoxygenases that make the epoxy-fatty-acid substrates sEH hydrolyses,
#:   so "is the axis essential" can be asked, not only "is the enzyme";
#: * six of DepMap's own common-essential controls, as the upper reference — a
#:   gene effect near -1 is what a genuine pan-essential looks like in this release;
#: * the first four of DepMap's own nonessential controls that appear as columns,
#:   in alphabetical order, as the lower reference. The selection rule is stated
#:   rather than curated so it is reproducible;
#: * KRAS, as a *selective*-dependency control. The panel would otherwise contain
#:   only pan-essential and flat genes, and the tool's selectivity claim would have
#:   nothing to demonstrate itself against. KRAS is the canonical case: flat-ish
#:   overall, profoundly dependent in pancreas.
#:
#: ``POLR2A`` is deliberately absent: it is not a column of 24Q4's
#: ``CRISPRGeneEffect.csv``, and naming a gene the release does not carry would
#: make the panel a wish rather than a description.
DEFAULT_GENES: tuple[str, ...] = (
    # target and paralogues
    "EPHX2",
    "EPHX1",
    "EPHX3",
    "EPHX4",
    # epoxy-fatty-acid axis
    "CYP2C8",
    "CYP2C9",
    "CYP2J2",
    # pan-essential positive controls (AchillesCommonEssentialControls)
    "RPL3",
    "PSMA1",
    "KIF11",
    "RAN",
    "SF3B1",
    "POLR2B",
    # nonessential negative controls (AchillesNonessentialControls)
    "ABCG8",
    "ACTL7A",
    "ADAM2",
    "AICDA",
    # lineage-selective positive control
    "KRAS",
)

#: How the header spells a gene: ``EPHX2 (2053)`` — HGNC symbol plus Entrez id.
_HEADER_SUFFIX = " ("

#: Read size for the streamed files. One MiB keeps the syscall count sane on a
#: 428 MB body without holding a meaningful amount of it in memory.
_CHUNK = 1 << 20


class _DigestStream(io.RawIOBase):
    """A read-only stream that MD5s everything that passes through it.

    The point is to verify the *whole* file while keeping none of it: the CSV
    reader pulls rows off the front, this hashes each chunk on the way past, and
    at EOF :attr:`hexdigest` is the digest of the bytes actually consumed.
    """

    def __init__(self, raw: Any) -> None:
        super().__init__()
        self._raw = raw
        self._digest = hashlib.md5()
        self.n_bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        chunk = self._raw.read(len(b))
        if not chunk:
            return 0
        n = len(chunk)
        b[:n] = chunk
        self._digest.update(chunk)
        self.n_bytes += n
        return n

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _open(url: str, policy: EgressPolicy) -> Any:
    """Open *url* after the egress policy has approved it."""
    policy.check(url)
    request = urllib.request.Request(url, headers={"User-Agent": "drug-agent/0.1 (+setup script)"})
    return urllib.request.urlopen(request, timeout=120)


def _gene_symbol(header: str) -> str:
    """``'EPHX2 (2053)'`` -> ``'EPHX2'``; a header without an Entrez id is its own symbol."""
    cut = header.find(_HEADER_SUFFIX)
    return header[:cut] if cut > 0 and header.endswith(")") else header


def fetch_verbatim(spec: ReleaseFile, out_dir: Path, policy: EgressPolicy) -> dict[str, Any]:
    """Download one small file whole, verifying its published digest."""
    print(f"  {spec.name}: {spec.size_bytes:,} bytes", file=sys.stderr)
    with _open(spec.url, policy) as response:
        body = response.read()
    digest = hashlib.md5(body).hexdigest()
    if digest != spec.md5:
        raise SystemExit(f"{spec.name}: MD5 {digest} != published {spec.md5}; refusing to write the cache")
    (out_dir / spec.name).write_bytes(body)
    return {"file": spec.name, "bytes": len(body), "md5": digest, "verified": True}


def stream_gene_columns(
    spec: ReleaseFile,
    *,
    genes: tuple[str, ...],
    control_sets: dict[str, set[str]] | None,
    out_dir: Path,
    out_name: str,
    policy: EgressPolicy,
) -> dict[str, Any]:
    """Stream one gene-by-cell-line matrix and keep only *genes*.

    The file is cell-line-major — one row per model, one column per gene — so a
    single column cannot be range-requested: every byte has to pass through to
    reach it. It passes through *here*, and only the requested columns are kept.

    When *control_sets* is given, the per-cell-line mean over each named set of
    columns is accumulated on the same pass and returned. That is the only number
    this script computes rather than copies, and it is labelled as such.
    """
    control_sets = control_sets or {}
    print(f"  {spec.name}: streaming {spec.size_bytes:,} bytes for {len(genes)} genes", file=sys.stderr)
    with _open(spec.url, policy) as response:
        digest_stream = _DigestStream(response)
        buffered = io.BufferedReader(digest_stream, buffer_size=_CHUNK)  # type: ignore[arg-type]
        text = io.TextIOWrapper(buffered, encoding="utf-8", newline="")
        reader = csv.reader(text)
        header = next(reader)
        by_symbol = {_gene_symbol(h): i for i, h in enumerate(header[1:], start=1)}
        keep = [(g, by_symbol[g], header[by_symbol[g]]) for g in genes if g in by_symbol]
        missing = [g for g in genes if g not in by_symbol]
        control_cols = {
            name: [i for symbol, i in by_symbol.items() if symbol in members]
            for name, members in control_sets.items()
        }

        rows: list[list[str]] = []
        control_rows: list[list[str]] = []
        for record in reader:
            if not record:
                continue
            rows.append([record[0]] + [record[i] for _, i, _ in keep])
            if control_cols:
                summary = [record[0]]
                for name in sorted(control_cols):
                    values = [float(record[i]) for i in control_cols[name] if record[i] != ""]
                    summary += [f"{sum(values) / len(values):.6f}" if values else "", str(len(values))]
                control_rows.append(summary)

    if digest_stream.hexdigest != spec.md5:
        raise SystemExit(
            f"{spec.name}: MD5 {digest_stream.hexdigest} != published {spec.md5} "
            f"after {digest_stream.n_bytes:,} bytes; refusing to write the cache"
        )

    out_path = out_dir / out_name
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ModelID"] + [column for _, _, column in keep])
        writer.writerows(rows)

    result: dict[str, Any] = {
        "file": spec.name,
        "written": out_name,
        "bytes_streamed": digest_stream.n_bytes,
        "md5": digest_stream.hexdigest,
        "verified": True,
        "n_genes_in_release": len(header) - 1,
        "n_models_in_release": len(rows),
        "genes_cached": [column for _, _, column in keep],
        "genes_requested_not_found": missing,
    }
    if control_cols:
        names = sorted(control_cols)
        reference = out_dir / "control_reference.csv"
        with reference.open("w", newline="") as handle:
            writer = csv.writer(handle)
            columns = ["ModelID"]
            for name in names:
                columns += [f"{name}_mean_gene_effect", f"{name}_n_genes"]
            writer.writerow(columns)
            writer.writerows(control_rows)
        result["control_reference"] = {
            "written": "control_reference.csv",
            "computed_by": "scripts/fetch_depmap_subset.py",
            "formula": "arithmetic mean of the non-missing gene-effect values over the named columns",
            "sets": {name: len(cols) for name, cols in control_cols.items()},
        }
    return result


def read_gene_list(path: Path) -> set[str]:
    """The symbols of a one-column DepMap gene list (``Gene`` header, ``SYM (id)`` rows)."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return {_gene_symbol(row[0]) for row in rows[1:] if row and row[0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gene",
        action="append",
        default=None,
        metavar="SYMBOL",
        help="HGNC symbol to cache; repeatable. Added to the default panel unless --only is given.",
    )
    parser.add_argument(
        "--only", action="store_true", help="cache exactly --gene, dropping the default panel"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"cache directory (default: <repo>/{DEFAULT_CACHE_RELPATH})",
    )
    parser.add_argument(
        "--skip-dependency-probability",
        action="store_true",
        help="skip CRISPRGeneDependency.csv (saves a second 421 MB stream)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="permit egress to the figshare hosts. Without it the deny-by-default policy applies.",
    )
    args = parser.parse_args(argv)

    policy = (
        EgressPolicy(allow_network=True, allowlist=list(ALLOWED_HOSTS))
        if args.allow_network
        else EgressPolicy.from_env()
    )
    requested = tuple(args.gene or ())
    genes = requested if (args.only and requested) else tuple(dict.fromkeys(DEFAULT_GENES + requested))

    out_dir = args.out or Path(__file__).resolve().parents[1] / DEFAULT_CACHE_RELPATH
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{RELEASE} (figshare {FIGSHARE_ARTICLE}, doi {FIGSHARE_DOI}) -> {out_dir}", file=sys.stderr)

    verbatim = [f for f in RELEASE_FILES if f.mode == "verbatim"]
    files: dict[str, Any] = {}
    for spec in verbatim:
        files[spec.name] = fetch_verbatim(spec, out_dir, policy)

    control_sets = {
        "common_essential_control": read_gene_list(out_dir / "AchillesCommonEssentialControls.csv"),
        "nonessential_control": read_gene_list(out_dir / "AchillesNonessentialControls.csv"),
    }

    effect_spec = next(f for f in RELEASE_FILES if f.name == "CRISPRGeneEffect.csv")
    files[effect_spec.name] = stream_gene_columns(
        effect_spec,
        genes=genes,
        control_sets=control_sets,
        out_dir=out_dir,
        out_name="gene_effect.csv",
        policy=policy,
    )
    if not args.skip_dependency_probability:
        dep_spec = next(f for f in RELEASE_FILES if f.name == "CRISPRGeneDependency.csv")
        files[dep_spec.name] = stream_gene_columns(
            dep_spec,
            genes=genes,
            control_sets=None,
            out_dir=out_dir,
            out_name="gene_dependency.csv",
            policy=policy,
        )

    effect = files["CRISPRGeneEffect.csv"]
    snapshot = DataSnapshot(
        name=f"depmap_crispr_gene_effect_{RELEASE.replace(' ', '_').lower()}",
        source=FIGSHARE_ARTICLE_URL,
        kind="crispr_gene_effect",
        records=[{"gene": g} for g in effect["genes_cached"]],
        license=RELEASE_LICENSE,
        metadata={
            "release": RELEASE,
            "figshare_article": FIGSHARE_ARTICLE,
            "doi": FIGSHARE_DOI,
            "published": RELEASE_PUBLISHED,
            "files": files,
        },
    )
    card = snapshot.data_card()
    card["fetched_at"] = utcnow_iso()
    card["fetched_by"] = "scripts/fetch_depmap_subset.py"
    card["release"] = RELEASE
    card["doi"] = FIGSHARE_DOI
    card["source_url"] = FIGSHARE_ARTICLE_URL
    card["files"] = files
    card["genes_cached"] = effect["genes_cached"]
    card["genes_requested_not_found"] = effect["genes_requested_not_found"]
    card["n_genes_in_release"] = effect["n_genes_in_release"]
    card["n_models_in_release"] = effect["n_models_in_release"]
    card["coverage"] = (
        f"{len(effect['genes_cached'])} of {effect['n_genes_in_release']} genes of {RELEASE} are cached, "
        f"for all {effect['n_models_in_release']} cell lines. Every other gene of the release is absent "
        f"from this cache and target.essentiality.depmap refuses it by name."
    )
    (out_dir / "index.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")

    print(f"\nsnapshot_id: {card['snapshot_id']}", file=sys.stderr)
    print(card["coverage"], file=sys.stderr)
    if effect["genes_requested_not_found"]:
        print(f"NOT IN THE RELEASE: {', '.join(effect['genes_requested_not_found'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
