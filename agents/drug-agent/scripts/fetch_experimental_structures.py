"""Populate the local experimental-structure cache from RCSB. The only network step.

``structure.experimental`` reads deposited coordinates out of a directory and
never opens a socket. This script is the other half of that arrangement: it
fetches the real files once, over the network, and writes a provenance record
beside them. Nothing here runs during a pipeline.

Two endpoints, both wwPDB/RCSB public services:

``https://files.rcsb.org/download/<ID>.pdb``
    the deposited coordinate file, in legacy PDB format (``--format mmcif`` asks
    for ``<ID>.cif`` instead, which is the only option for entries too large for
    the PDB format).
``https://search.rcsb.org/rcsbsearch/v2/query``
    ``--uniprot <accession>`` posts a two-clause exact-match query against
    ``rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers``
    and takes the entry ids back, best resolution first. The query is written into
    the index verbatim, so "which structures did we consider" is answerable later
    without re-running it.

Usage::

    export DRUGAGENT_ALLOW_NETWORK=1
    export DRUGAGENT_EGRESS_ALLOWLIST=files.rcsb.org,search.rcsb.org
    python scripts/fetch_experimental_structures.py 3I28 5AI0
    python scripts/fetch_experimental_structures.py --uniprot P34913 --limit 5
    python scripts/fetch_experimental_structures.py --reindex

Every request goes through :class:`~drug_agent.security.egress.EgressPolicy`, which
denies by default — so running this without deliberately opening egress fails with
the reason, rather than quietly reaching the internet from a machine that is
supposed to be offline.

**What the index is, and what it is not.** ``rcsb_fetch_index.json`` is a
:class:`~drug_agent.data.snapshot.DataSnapshot` data card: source URL, HTTP
status, byte count, SHA-256 and a UTC timestamp per entry, plus the query that
selected them and a content hash over the whole set. It carries **no scientific
values** — no resolution, no method, no ligand list. Those are parsed from the
coordinate files themselves on every call, so this file can misattribute a
download but can never become a source of numbers. A file already in the cache is
recorded as ``already_present`` and is *not* re-downloaded: its hash is computed
locally and its origin is stated as unknown rather than invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from drug_agent.data.snapshot import DataSnapshot  # noqa: E402
from drug_agent.security.egress import EgressPolicy, NetworkDeniedError  # noqa: E402
from drug_agent.util import utcnow_iso  # noqa: E402

#: Where the adapter looks by default. Overridable with ``--cache``, and with
#: ``DRUGAGENT_EXPERIMENTAL_ROOT`` on the adapter side.
DEFAULT_CACHE = REPO_ROOT / ".external" / "pdb"

DOWNLOAD_ENDPOINT = "https://files.rcsb.org/download/{entry}.{ext}"
SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"

#: What the adapter reads for provenance. Named here rather than imported so this
#: script keeps working if the adapter is not importable on the fetching machine.
INDEX_NAME = "rcsb_fetch_index.json"

FORMATS = {"pdb": "pdb", "mmcif": "cif"}

#: A polite, identifying user agent. RCSB asks for one and throttles anonymous
#: floods; this is a handful of requests, run by hand, at setup time.
USER_AGENT = "drug-agent/1.0 (structure.experimental cache setup)"

TIMEOUT_S = 120.0


def uniprot_query(accession: str, limit: int) -> dict[str, Any]:
    """The RCSB search body that finds every entry referencing *accession*.

    Two clauses rather than one: the accession alone would also match an
    identically-spelled id in another sequence database. Sorted by resolution so
    ``--limit`` keeps the best structures rather than an arbitrary page.
    """
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": accession.strip().upper(),
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_name"
                        ),
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": int(limit)},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def search_uniprot(accession: str, limit: int, policy: EgressPolicy) -> tuple[list[str], dict[str, Any]]:
    """Entry ids for *accession*, plus the query and total count for the record."""
    body = uniprot_query(accession, limit)
    policy.check(SEARCH_ENDPOINT)
    request = urllib.request.Request(
        SEARCH_ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        payload = json.loads(response.read().decode())
    ids = [str(row["identifier"]).upper() for row in payload.get("result_set", [])]
    return ids, {
        "endpoint": SEARCH_ENDPOINT,
        "query": body,
        "total_count": payload.get("total_count"),
        "returned": len(ids),
    }


def download(entry: str, ext: str, destination: Path, policy: EgressPolicy) -> dict[str, Any]:
    """Fetch one entry to *destination*, returning its provenance record."""
    url = DOWNLOAD_ENDPOINT.format(entry=entry.upper(), ext=ext)
    policy.check(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        status = int(response.status)
        raw = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return {
        "pdb_id": entry.upper(),
        "file": destination.name,
        "url": url,
        "http_status": status,
        "n_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at": utcnow_iso(),
        "origin": "downloaded",
    }


def record_existing(entry: str, path: Path) -> dict[str, Any]:
    """Record a file that was already in the cache, without claiming to have fetched it."""
    raw = path.read_bytes()
    return {
        "pdb_id": entry.upper(),
        "file": path.name,
        "url": None,
        "http_status": None,
        "n_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at": None,
        "origin": "already_present",
        "note": (
            "present in the cache before this script ran; the hash is computed here, "
            "the download is not evidenced"
        ),
    }


def load_existing_records(index_path: Path) -> dict[str, dict[str, Any]]:
    """Previous records, so one run for one entry does not erase the rest."""
    if not index_path.is_file():
        return {}
    try:
        card = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    records = card.get("records") if isinstance(card, dict) else None
    return {
        str(rec["pdb_id"]).upper(): rec
        for rec in records or []
        if isinstance(rec, dict) and rec.get("pdb_id")
    }


def cached_file(cache: Path, entry: str, ext: str) -> Path | None:
    """An existing cache file for *entry*, whatever suffix it was stored under."""
    for candidate in (f"{entry}.{ext}", f"{entry}.pdb", f"{entry}.cif", f"{entry}.ent"):
        path = cache / candidate
        if path.is_file():
            return path
    return None


def write_index(
    cache: Path,
    records: dict[str, dict[str, Any]],
    *,
    searches: list[dict[str, Any]],
    file_format: str,
) -> Path:
    """Write the data card. Provenance only — see the module docstring."""
    snapshot = DataSnapshot(
        name="rcsb_experimental_structures",
        source=DOWNLOAD_ENDPOINT.format(entry="<ID>", ext=FORMATS[file_format]),
        kind="experimental_structures",
        records=[records[key] for key in sorted(records)],
        license="RCSB PDB entries are released under CC0 1.0 Universal (public domain)",
        metadata={
            "download_endpoint": DOWNLOAD_ENDPOINT.format(entry="<ID>", ext=FORMATS[file_format]),
            "search_endpoint": SEARCH_ENDPOINT,
            "searches": searches,
            "file_format": file_format,
            "carries_scientific_values": False,
            "note": (
                "provenance only: resolution, method, chains and ligands are parsed from "
                "the coordinate files by structure.experimental on every call"
            ),
        },
    )
    # The data card describes the set; the records are the per-entry provenance the
    # adapter binds to each file. `data_card()` deliberately reports only
    # `n_records`, so they are carried alongside it rather than instead of it.
    document = {**snapshot.data_card(), "records": snapshot.records}
    index_path = cache / INDEX_NAME
    index_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    return index_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("entries", nargs="*", help="PDB entry ids, e.g. 3I28 5AI0")
    parser.add_argument("--uniprot", action="append", default=[], help="fetch every entry for an accession")
    parser.add_argument("--limit", type=int, default=5, help="entries per --uniprot search (default 5)")
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE, help=f"cache dir (default {DEFAULT_CACHE})"
    )
    parser.add_argument("--format", choices=sorted(FORMATS), default="pdb", help="coordinate format")
    parser.add_argument("--force", action="store_true", help="re-download entries already cached")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild the index from what is already cached, touching no network",
    )
    args = parser.parse_args(argv)

    cache: Path = args.cache
    ext = FORMATS[args.format]
    policy = EgressPolicy.from_env()
    records = load_existing_records(cache / INDEX_NAME)
    searches: list[dict[str, Any]] = []

    wanted = [entry.strip().upper() for entry in args.entries if entry.strip()]

    if args.reindex:
        if not cache.is_dir():
            print(f"nothing to reindex: {cache} does not exist", file=sys.stderr)
            return 2
        for path in sorted(cache.iterdir()):
            if path.name == INDEX_NAME or not path.is_file():
                continue
            entry = path.stem.upper()
            records.setdefault(entry, record_existing(entry, path))
        index_path = write_index(cache, records, searches=searches, file_format=args.format)
        print(f"reindexed {len(records)} entries -> {index_path}")
        return 0

    for accession in args.uniprot:
        try:
            found, detail = search_uniprot(accession, args.limit, policy)
        except NetworkDeniedError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"RCSB search for {accession} failed: {exc}", file=sys.stderr)
            return 1
        detail["accession"] = accession.upper()
        detail["ran_at"] = utcnow_iso()
        searches.append(detail)
        print(f"{accession}: {detail['total_count']} entries at RCSB, taking {len(found)}: {' '.join(found)}")
        wanted.extend(found)

    if not wanted:
        parser.error("give at least one PDB id, a --uniprot accession, or --reindex")

    cache.mkdir(parents=True, exist_ok=True)
    failures = 0
    for entry in dict.fromkeys(wanted):
        existing = cached_file(cache, entry, ext)
        if existing is not None and not args.force:
            records[entry] = records.get(entry) or record_existing(entry, existing)
            print(f"{entry}: already present at {existing} ({existing.stat().st_size} bytes)")
            continue
        destination = cache / f"{entry}.{ext}"
        try:
            records[entry] = download(entry, ext, destination, policy)
        except NetworkDeniedError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError) as exc:
            print(f"{entry}: download failed: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{entry}: {records[entry]['n_bytes']} bytes -> {destination}")

    index_path = write_index(cache, records, searches=searches, file_format=args.format)
    print(f"index: {index_path} ({len(records)} entries)")
    print("Verify: drugagent tools | grep structure.experimental")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
