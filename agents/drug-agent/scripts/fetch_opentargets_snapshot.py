"""Fetch one target's Open Targets Platform evidence and persist it as a snapshot.

This is the **setup** half of ``target.evidence.opentargets``. It is the only part
of that capability permitted to reach the network, it runs once per target, and
what it writes is what the adapter reads for the rest of that target's life. The
adapter (:mod:`drug_agent.tools.target.opentargets_tool`) imports no HTTP client at
all and reports UNAVAILABLE naming this command when a target has no snapshot.

**A snapshot is real data, and it says so in a way that can be checked.** Every
file this writes carries the endpoint URL, the three GraphQL documents verbatim,
their variables, the API version and the data release the server reported at fetch
time, the UTC fetch timestamp, and a content hash plus snapshot id from
:class:`drug_agent.data.snapshot.DataSnapshot`. Anyone can re-run the recorded
query against the recorded release and diff. That is what separates a cached
snapshot from the fixture stand-in in ``fixtures/data/targets.json``, whose
"associations" were written by hand and whose ``source`` is the file itself.

**Paging is exhausted, not truncated.** ``associatedDiseases`` returns 620 rows for
EPHX2 and ``credibleSets`` 326; both are walked to the end at 200 per page and the
server's own ``count`` is stored beside the number of rows kept, so a partial fetch
is visible rather than implied. ``--max-associations`` / ``--max-credible-sets``
can cap either, and a cap that bites is recorded as ``truncated: true`` with both
numbers — never a quiet narrowing.

**Three documents, not one.** The core record, the association pages and the
credible-set pages are separate queries because they page independently. Each is
stored with the response it produced, so the snapshot is self-describing.

Usage::

    python scripts/fetch_opentargets_snapshot.py --allow-network ENSG00000120915
    python scripts/fetch_opentargets_snapshot.py --allow-network --symbol EPHX2

Egress goes through :class:`drug_agent.security.egress.EgressPolicy` with an
allowlist pinned to the Platform host. Without ``--allow-network`` the
deny-by-default policy applies and the script stops before the first request.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from drug_agent.data.snapshot import DataSnapshot
from drug_agent.security.egress import EgressPolicy
from drug_agent.util import content_hash, utcnow_iso

#: The Platform's public GraphQL endpoint. v4 is the current major; the exact
#: server build and data release are read from ``meta`` at fetch time rather than
#: assumed here, because they move independently of this script.
ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

#: The one host this script may reach.
ALLOWED_HOSTS = ("api.platform.opentargets.org",)

#: Where snapshots live, relative to the repository root.
DEFAULT_CACHE_RELPATH = ".external/opentargets"

#: Rows per page. 200 is accepted by both paged fields; 620 associations are four
#: requests rather than thirty-one.
PAGE_SIZE = 200

META_QUERY = """
query PlatformMeta {
  meta {
    name
    apiVersion { x y z }
    dataVersion { year month iteration }
  }
}
""".strip()

SEARCH_QUERY = """
query TargetSearch($queryString: String!) {
  search(queryString: $queryString, entityNames: ["target"], page: {index: 0, size: 10}) {
    total
    hits {
      id
      object { ... on Target { id approvedSymbol approvedName } }
    }
  }
}
""".strip()

#: Identity, what the protein is, whether it is drug-like, and what is known to go
#: wrong when it is hit. One request; none of it pages.
CORE_QUERY = """
query TargetCore($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    synonyms { label source }
    proteinIds { id source }
    targetClass { id label level }
    genomicLocation { chromosome start end strand }
    tractability { label modality value }
    safetyLiabilities {
      event
      eventId
      datasource
      literature
      url
      effects { direction dosing }
      biosamples { tissueLabel tissueId cellLabel cellFormat }
    }
  }
}
""".strip()

#: The association layer proper: one overall score per disease plus the per-datatype
#: breakdown that says *which kind* of evidence produced it.
ASSOCIATIONS_QUERY = """
query TargetAssociations($ensemblId: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    associatedDiseases(page: {index: $index, size: $size}) {
      count
      rows {
        score
        datatypeScores { id score }
        disease { id name therapeuticAreas { id name } }
      }
    }
  }
}
""".strip()

#: Gentropy's output, as the Platform serves it: fine-mapped credible sets and the
#: locus-to-gene scores assigning them to this target. See the adapter's module
#: docstring for why this is fetched rather than recomputed.
CREDIBLE_SETS_QUERY = """
query TargetCredibleSets($ensemblId: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    credibleSets(page: {index: $index, size: $size}) {
      count
      rows {
        studyLocusId
        studyType
        chromosome
        position
        beta
        standardError
        pValueMantissa
        pValueExponent
        finemappingMethod
        confidence
        variant { id rsIds }
        study {
          id
          studyType
          projectId
          traitFromSource
          pubmedId
          nSamples
          diseases { id name }
        }
        l2GPredictions(page: {index: 0, size: 5}) {
          count
          rows { score target { id approvedSymbol } }
        }
      }
    }
  }
}
""".strip()


class GraphQLError(RuntimeError):
    """The endpoint answered, and the answer was an error rather than data."""


def post(query: str, variables: dict[str, Any], policy: EgressPolicy) -> dict[str, Any]:
    """One GraphQL request, with the egress policy consulted first."""
    policy.check(ENDPOINT)
    body = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "drug-agent/0.1 (+setup script)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode())
    if payload.get("errors"):
        raise GraphQLError(json.dumps(payload["errors"])[:800])
    data = payload.get("data")
    if not isinstance(data, dict):
        raise GraphQLError(f"response carried no data object: {str(payload)[:400]}")
    return data


def resolve_symbol(symbol: str, policy: EgressPolicy) -> str:
    """The Ensembl gene id the Platform's own search returns for *symbol*.

    Exact-symbol only. The search endpoint happily returns near matches — EPHX2
    also matches ``ENSG00000289323``, "novel transcript, sense intronic to EPHX2" —
    and silently snapshotting the wrong gene is the kind of error that never
    announces itself downstream.
    """
    data = post(SEARCH_QUERY, {"queryString": symbol}, policy)
    hits = data.get("search", {}).get("hits") or []
    exact = [h for h in hits if (h.get("object") or {}).get("approvedSymbol") == symbol]
    if not exact:
        symbols = sorted({(h.get("object") or {}).get("approvedSymbol", "?") for h in hits})
        found = ", ".join(symbols) or "none"
        raise SystemExit(f"no target with approved symbol {symbol!r} (search returned: {found})")
    return str(exact[0]["object"]["id"])


def fetch_pages(
    query: str,
    *,
    ensembl_id: str,
    field: str,
    limit: int | None,
    policy: EgressPolicy,
) -> dict[str, Any]:
    """Walk one paged field to the end, or to *limit*, recording which happened."""
    rows: list[dict[str, Any]] = []
    count = 0
    index = 0
    while True:
        size = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(rows))
        if size <= 0:
            break
        data = post(query, {"ensemblId": ensembl_id, "index": index, "size": size}, policy)
        target = data.get("target") or {}
        block = target.get(field) or {}
        count = int(block.get("count") or 0)
        page = list(block.get("rows") or [])
        rows.extend(page)
        print(f"    {field}: {len(rows)}/{count}", file=sys.stderr)
        if not page or len(rows) >= count:
            break
        index += 1
    return {
        "count": count,
        "n_rows": len(rows),
        "truncated": len(rows) < count,
        "page_size": PAGE_SIZE,
        "rows": rows,
    }


def fetch_target(
    ensembl_id: str,
    *,
    policy: EgressPolicy,
    max_associations: int | None,
    max_credible_sets: int | None,
) -> dict[str, Any]:
    """Everything this platform reads about one target, plus how it was obtained."""
    meta = post(META_QUERY, {}, policy)["meta"]
    api = meta.get("apiVersion") or {}
    release = meta.get("dataVersion") or {}
    api_version = ".".join(str(api.get(k)) for k in ("x", "y", "z"))
    data_version = f"{release.get('year')}.{release.get('month')}"
    print(f"  {ensembl_id}: API {api_version}, data release {data_version}", file=sys.stderr)

    core = post(CORE_QUERY, {"ensemblId": ensembl_id}, policy).get("target")
    if core is None:
        raise SystemExit(f"the Platform has no target {ensembl_id!r} in release {data_version}")

    associations = fetch_pages(
        ASSOCIATIONS_QUERY,
        ensembl_id=ensembl_id,
        field="associatedDiseases",
        limit=max_associations,
        policy=policy,
    )
    credible_sets = fetch_pages(
        CREDIBLE_SETS_QUERY,
        ensembl_id=ensembl_id,
        field="credibleSets",
        limit=max_credible_sets,
        policy=policy,
    )
    return {
        "target": core,
        "associated_diseases": associations,
        "credible_sets": credible_sets,
        "queries": {
            "meta": {"document": META_QUERY, "variables": {}},
            "core": {"document": CORE_QUERY, "variables": {"ensemblId": ensembl_id}},
            "associated_diseases": {
                "document": ASSOCIATIONS_QUERY,
                "variables": {"ensemblId": ensembl_id, "index": "0..n", "size": PAGE_SIZE},
            },
            "credible_sets": {
                "document": CREDIBLE_SETS_QUERY,
                "variables": {"ensemblId": ensembl_id, "index": "0..n", "size": PAGE_SIZE},
            },
        },
        "api_version": api_version,
        "data_version": data_version,
        "server_name": meta.get("name"),
    }


def write_snapshot(payload: dict[str, Any], *, ensembl_id: str, out_dir: Path) -> dict[str, Any]:
    """Persist one target snapshot and return its data card."""
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # The hashed records are the scientific content — the association rows and the
    # credible sets. Identity and query text ride in metadata, so re-fetching the
    # same release yields the same snapshot id even if the server's banner changes.
    records = [{"kind": "association", **row} for row in payload["associated_diseases"]["rows"]] + [
        {"kind": "credible_set", **row} for row in payload["credible_sets"]["rows"]
    ]
    snapshot = DataSnapshot(
        name=f"opentargets_{ensembl_id}",
        source=ENDPOINT,
        kind="target_disease_evidence",
        records=records,
        license=(
            "Open Targets Platform data are released under CC0 1.0; individual datasources "
            "carry their own terms (see https://platform-docs.opentargets.org/licence)."
        ),
        metadata={
            "ensembl_id": ensembl_id,
            "approved_symbol": payload["target"].get("approvedSymbol"),
            "api_version": payload["api_version"],
            "data_version": payload["data_version"],
            "server_name": payload["server_name"],
            "query_hash": content_hash(payload["queries"]),
        },
    )
    card = snapshot.data_card()
    card["fetched_at"] = utcnow_iso()
    card["fetched_by"] = "scripts/fetch_opentargets_snapshot.py"
    card["endpoint"] = ENDPOINT
    card["ensembl_id"] = ensembl_id
    card["approved_symbol"] = payload["target"].get("approvedSymbol")
    card["n_associated_diseases"] = payload["associated_diseases"]["count"]
    card["n_associated_diseases_cached"] = payload["associated_diseases"]["n_rows"]
    card["associations_truncated"] = payload["associated_diseases"]["truncated"]
    card["n_credible_sets"] = payload["credible_sets"]["count"]
    card["n_credible_sets_cached"] = payload["credible_sets"]["n_rows"]
    card["credible_sets_truncated"] = payload["credible_sets"]["truncated"]

    document = {"data_card": card, "queries": payload["queries"], "payload": payload}
    path = targets_dir / f"{ensembl_id}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return card


def rebuild_index(out_dir: Path) -> dict[str, Any]:
    """Rewrite ``index.json`` from the snapshots actually on disk.

    Derived rather than appended, so a snapshot deleted by hand disappears from the
    index instead of leaving the adapter promising a target it cannot read.
    """
    targets_dir = out_dir / "targets"
    cards: dict[str, Any] = {}
    for path in sorted(targets_dir.glob("*.json")) if targets_dir.is_dir() else []:
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        card = document.get("data_card")
        if isinstance(card, dict) and card.get("ensembl_id"):
            cards[str(card["ensembl_id"])] = card
    index = {
        "source": ENDPOINT,
        "kind": "target_disease_evidence",
        "written_by": "scripts/fetch_opentargets_snapshot.py",
        "written_at": utcnow_iso(),
        "targets": cards,
        "symbols": {
            str(card.get("approved_symbol")): ensembl
            for ensembl, card in cards.items()
            if card.get("approved_symbol")
        },
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    (out_dir / "queries.graphql").write_text(
        "\n\n".join([META_QUERY, SEARCH_QUERY, CORE_QUERY, ASSOCIATIONS_QUERY, CREDIBLE_SETS_QUERY]) + "\n"
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ensembl_ids", nargs="*", metavar="ENSG", help="Ensembl gene ids to snapshot")
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        metavar="SYMBOL",
        help="HGNC symbol to resolve through the Platform's own search; repeatable",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help=f"cache directory (default: <repo>/{DEFAULT_CACHE_RELPATH})"
    )
    parser.add_argument("--max-associations", type=int, default=None, help="cap association rows")
    parser.add_argument("--max-credible-sets", type=int, default=None, help="cap credible-set rows")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="permit egress to the Open Targets Platform host; without it the deny-by-default policy applies",
    )
    args = parser.parse_args(argv)

    policy = (
        EgressPolicy(allow_network=True, allowlist=list(ALLOWED_HOSTS))
        if args.allow_network
        else EgressPolicy.from_env()
    )
    out_dir = args.out or Path(__file__).resolve().parents[1] / DEFAULT_CACHE_RELPATH
    out_dir.mkdir(parents=True, exist_ok=True)

    ensembl_ids = list(args.ensembl_ids)
    for symbol in args.symbol or []:
        ensembl_ids.append(resolve_symbol(symbol, policy))
    if not ensembl_ids:
        parser.error("name at least one Ensembl gene id, or a --symbol to resolve")

    print(f"{ENDPOINT} -> {out_dir}", file=sys.stderr)
    for ensembl_id in dict.fromkeys(ensembl_ids):
        payload = fetch_target(
            ensembl_id,
            policy=policy,
            max_associations=args.max_associations,
            max_credible_sets=args.max_credible_sets,
        )
        card = write_snapshot(payload, ensembl_id=ensembl_id, out_dir=out_dir)
        print(
            f"  {card['approved_symbol']} ({ensembl_id}): "
            f"{card['n_associated_diseases_cached']}/{card['n_associated_diseases']} associations, "
            f"{card['n_credible_sets_cached']}/{card['n_credible_sets']} credible sets, "
            f"snapshot {card['snapshot_id']}",
            file=sys.stderr,
        )
    index = rebuild_index(out_dir)
    print(f"index.json lists {len(index['targets'])} target(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
