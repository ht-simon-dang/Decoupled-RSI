"""Snapshot every public source the ZAK/MAP3K20 target package is built from.

This is the **only** part of the per-target program permitted to reach the
network, it runs once, and what it writes is what every later step reads. It
fetches nothing derived: four primary sources, verbatim, each stored beside the
URL that produced it, the HTTP status, the UTC fetch time, the row count, the
page count and a SHA-256 of the bytes.

Sources
-------
``uniprot``
    Q9NYL2 (the provisional mapping for the user's ``zak`` alias) as JSON and
    FASTA. This is where species, canonical sequence and construct come from.
``rcsb``
    PDB entry 6JUU — the structure the user named — as the deposited PDB file
    plus the entry, polymer-entity, nonpolymer-entity and chemical-component
    records. The chemical component ``C9R`` is the co-crystallized ligand and
    the only structural definition of what "compound 6r" means.
``chembl``
    Every activity row for the ChEMBL target that resolves from Q9NYL2, paged to
    exhaustion, plus the assay, document and molecule records those rows cite.
    ``page_meta.total_count`` is recorded next to the number of rows kept so a
    partial fetch is visible rather than implied.
``bindingdb``
    ``getLigandsByUniprots`` for Q9NYL2, and the HTML of the paper entry the
    brief links (entry 50007907). BindingDB re-publishes paper records credited
    to ChEMBL, so this exists to *measure the overlap*, not to add rows.

Nothing here interprets a measurement. Normalization, standardization,
de-duplication and stratification are :mod:`drug_agent.selfevolve.target.data`,
and they run offline against what this writes.

Usage::

    python scripts/research/fetch_zak_sources.py --allow-network
    python scripts/research/fetch_zak_sources.py --allow-network --out /tmp/snap

The default output root is ``.external/research/zak-c9r/``, which is gitignored:
raw source files stay out of source control and only their manifest and digests
are committed. Without ``--allow-network`` the deny-by-default egress policy
stops the script before the first request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from drug_agent.security.egress import EgressPolicy
from drug_agent.util import utcnow_iso

#: Where snapshots live, relative to the repository root. Gitignored.
DEFAULT_OUT_RELPATH = ".external/research/zak-c9r"

#: The provisional public mapping for the user's ``zak`` alias (Appendix A §2).
UNIPROT_ACCESSION = "Q9NYL2"
#: The structure the user named in the brief ("Design a drug on target zak with 6juu").
PDB_ID = "6JUU"
#: The co-crystallized ligand of 6JUU. A ligand identifier, not a compound name.
LIGAND_COMP_ID = "C9R"
#: BindingDB's entry id for the paper the brief links (PubMed 31244114).
BINDINGDB_ENTRY_ID = "50007907"

UNIPROT_BASE = "https://rest.uniprot.org"
RCSB_DATA_BASE = "https://data.rcsb.org/rest/v1/core"
RCSB_FILES_BASE = "https://files.rcsb.org/download"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
BINDINGDB_BASE = "https://www.bindingdb.org"

#: The only hosts this script may reach.
ALLOWED_HOSTS = (
    "rest.uniprot.org",
    "data.rcsb.org",
    "files.rcsb.org",
    "www.ebi.ac.uk",
    "www.bindingdb.org",
)

#: Rows per ChEMBL page. 1000 is the API's documented maximum.
PAGE_SIZE = 1000

#: Courtesy delay between requests to the same host.
THROTTLE_SECONDS = 0.2

USER_AGENT = "drug-agent-research/0.1 (per-target evolution program; step 5)"

#: Recorded so a reader knows what reuse terms the rows arrived under. These are
#: the publishers' stated terms at fetch time, transcribed, not legal advice.
LICENSES = {
    "uniprot": "CC BY 4.0 (UniProt)",
    "rcsb": "CC0 1.0 (RCSB PDB)",
    "chembl": "CC BY-SA 3.0 (ChEMBL, EMBL-EBI)",
    "bindingdb": "CC BY 3.0 US (BindingDB)",
}


class FetchError(RuntimeError):
    """A source could not be retrieved. Recorded, never silently skipped."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Fetcher:
    """Egress-policed HTTP with a per-request audit trail.

    Every call appends one entry to :attr:`log` whether it succeeded or not, so
    "partial coverage" is a number in the manifest rather than a missing file.
    """

    def __init__(self, policy: EgressPolicy) -> None:
        self.policy = policy
        self.log: list[dict[str, Any]] = []

    def get(self, url: str, *, timeout: float = 120.0) -> bytes:
        self.policy.check(url)
        started = time.perf_counter()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            self.log.append(
                {
                    "url": url,
                    "status": exc.code,
                    "ok": False,
                    "error": f"HTTPError {exc.code}",
                    "at": utcnow_iso(),
                }
            )
            raise FetchError(f"{url} -> HTTP {exc.code}") from exc
        except Exception as exc:
            self.log.append({"url": url, "status": None, "ok": False, "error": repr(exc), "at": utcnow_iso()})
            raise FetchError(f"{url} -> {exc!r}") from exc
        self.log.append(
            {
                "url": url,
                "status": status,
                "ok": True,
                "bytes": len(body),
                "seconds": round(time.perf_counter() - started, 3),
                "at": utcnow_iso(),
            }
        )
        time.sleep(THROTTLE_SECONDS)
        return body

    def get_json(self, url: str, **kw: Any) -> Any:
        return json.loads(self.get(url, **kw).decode("utf-8"))


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write(out: Path, relpath: str, data: bytes) -> dict[str, Any]:
    path = out / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": relpath, "sha256": _sha256(data), "bytes": len(data)}


def _write_json(out: Path, relpath: str, obj: Any) -> dict[str, Any]:
    return _write(out, relpath, json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def fetch_uniprot(f: Fetcher, out: Path) -> dict[str, Any]:
    entry_url = f"{UNIPROT_BASE}/uniprotkb/{UNIPROT_ACCESSION}.json"
    fasta_url = f"{UNIPROT_BASE}/uniprotkb/{UNIPROT_ACCESSION}.fasta"
    entry = f.get(entry_url)
    fasta = f.get(fasta_url)
    record = json.loads(entry.decode("utf-8"))
    return {
        "source": "uniprot",
        "license": LICENSES["uniprot"],
        "fetched_at": utcnow_iso(),
        "accession": UNIPROT_ACCESSION,
        "urls": [entry_url, fasta_url],
        "files": [
            _write(out, "uniprot/Q9NYL2.json", entry),
            _write(out, "uniprot/Q9NYL2.fasta", fasta),
        ],
        "entry_version": record.get("entryAudit", {}).get("entryVersion"),
        "sequence_version": record.get("entryAudit", {}).get("sequenceVersion"),
        "sequence_length": record.get("sequence", {}).get("length"),
        "organism": record.get("organism", {}).get("scientificName"),
        "n_records": 1,
    }


def fetch_rcsb(f: Fetcher, out: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    urls: list[str] = []

    pdb_url = f"{RCSB_FILES_BASE}/{PDB_ID}.pdb"
    urls.append(pdb_url)
    files.append(_write(out, f"rcsb/{PDB_ID}.pdb", f.get(pdb_url)))

    cif_url = f"{RCSB_FILES_BASE}/{PDB_ID}.cif"
    urls.append(cif_url)
    files.append(_write(out, f"rcsb/{PDB_ID}.cif", f.get(cif_url)))

    entry_url = f"{RCSB_DATA_BASE}/entry/{PDB_ID}"
    urls.append(entry_url)
    entry_bytes = f.get(entry_url)
    files.append(_write(out, f"rcsb/entry_{PDB_ID}.json", entry_bytes))
    entry = json.loads(entry_bytes.decode("utf-8"))

    ids = entry.get("rcsb_entry_container_identifiers", {})
    for eid in ids.get("polymer_entity_ids", []):
        url = f"{RCSB_DATA_BASE}/polymer_entity/{PDB_ID}/{eid}"
        urls.append(url)
        files.append(_write(out, f"rcsb/polymer_entity_{PDB_ID}_{eid}.json", f.get(url)))
    for eid in ids.get("non_polymer_entity_ids", []) or []:
        url = f"{RCSB_DATA_BASE}/nonpolymer_entity/{PDB_ID}/{eid}"
        urls.append(url)
        files.append(_write(out, f"rcsb/nonpolymer_entity_{PDB_ID}_{eid}.json", f.get(url)))

    comp_url = f"{RCSB_DATA_BASE}/chemcomp/{LIGAND_COMP_ID}"
    urls.append(comp_url)
    files.append(_write(out, f"rcsb/chemcomp_{LIGAND_COMP_ID}.json", f.get(comp_url)))

    return {
        "source": "rcsb",
        "license": LICENSES["rcsb"],
        "fetched_at": utcnow_iso(),
        "pdb_id": PDB_ID,
        "ligand_comp_id": LIGAND_COMP_ID,
        "title": entry.get("struct", {}).get("title"),
        "method": [m.get("method") for m in entry.get("exptl", [])],
        "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined"),
        "urls": urls,
        "files": files,
        "n_records": len(files),
    }


def _chembl_page_all(f: Fetcher, endpoint: str, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Page a ChEMBL collection to exhaustion, reporting coverage either way."""
    collection = endpoint.rstrip("s") + "s" if endpoint != "activity" else "activities"
    if endpoint == "molecule":
        collection = "molecules"
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    pages = 0
    while True:
        url = f"{CHEMBL_BASE}/{endpoint}.json?{query}&limit={PAGE_SIZE}&offset={offset}"
        page = f.get_json(url)
        meta = page.get("page_meta", {})
        total = meta.get("total_count") if total is None else total
        batch = page.get(collection, [])
        rows.extend(batch)
        pages += 1
        offset += PAGE_SIZE
        if not batch or (total is not None and offset >= total):
            break
    coverage = {
        "reported_total": total,
        "rows_retrieved": len(rows),
        "pages": pages,
        "complete": total is not None and len(rows) >= total,
    }
    return rows, coverage


def fetch_chembl(f: Fetcher, out: Path) -> dict[str, Any]:
    targets, target_cov = _chembl_page_all(f, "target", f"target_components__accession={UNIPROT_ACCESSION}")
    files = [_write_json(out, "chembl/targets.json", targets)]
    target_ids = [t["target_chembl_id"] for t in targets]
    if not target_ids:
        raise FetchError(f"no ChEMBL target resolves from {UNIPROT_ACCESSION}")

    activities: list[dict[str, Any]] = []
    activity_cov: dict[str, Any] = {}
    for tid in target_ids:
        rows, cov = _chembl_page_all(f, "activity", f"target_chembl_id={tid}")
        activities.extend(rows)
        activity_cov[tid] = cov
    files.append(_write_json(out, "chembl/activities.json", activities))

    assay_ids = sorted({a["assay_chembl_id"] for a in activities if a.get("assay_chembl_id")})
    doc_ids = sorted({a["document_chembl_id"] for a in activities if a.get("document_chembl_id")})
    mol_ids = sorted({a["molecule_chembl_id"] for a in activities if a.get("molecule_chembl_id")})

    def _by_ids(endpoint: str, field: str, ids: list[str], chunk: int = 50) -> list[dict[str, Any]]:
        acc: list[dict[str, Any]] = []
        for i in range(0, len(ids), chunk):
            batch = ids[i : i + chunk]
            rows, _ = _chembl_page_all(f, endpoint, f"{field}__in={','.join(batch)}")
            acc.extend(rows)
        return acc

    assays = _by_ids("assay", "assay_chembl_id", assay_ids)
    documents = _by_ids("document", "document_chembl_id", doc_ids)
    molecules = _by_ids("molecule", "molecule_chembl_id", mol_ids)
    files.append(_write_json(out, "chembl/assays.json", assays))
    files.append(_write_json(out, "chembl/documents.json", documents))
    files.append(_write_json(out, "chembl/molecules.json", molecules))

    status = f.get_json(f"{CHEMBL_BASE}/status.json")
    files.append(_write_json(out, "chembl/status.json", status))

    return {
        "source": "chembl",
        "license": LICENSES["chembl"],
        "fetched_at": utcnow_iso(),
        "release": status.get("chembl_db_version"),
        "api_version": status.get("api_version"),
        "target_ids": target_ids,
        "urls": [f"{CHEMBL_BASE}/activity.json?target_chembl_id={t}" for t in target_ids],
        "files": files,
        "coverage": {
            "targets": target_cov,
            "activities": activity_cov,
            "assays_requested": len(assay_ids),
            "assays_retrieved": len(assays),
            "documents_requested": len(doc_ids),
            "documents_retrieved": len(documents),
            "molecules_requested": len(mol_ids),
            "molecules_retrieved": len(molecules),
        },
        "n_records": len(activities),
    }


def fetch_bindingdb(f: Fetcher, out: Path) -> dict[str, Any]:
    rest_url = (
        f"{BINDINGDB_BASE}/rest/getLigandsByUniprots"
        f"?uniprot={UNIPROT_ACCESSION}&code=0&response=application/json"
    )
    payload = f.get_json(rest_url)
    # The service's response key is misspelled upstream; read whatever single
    # top-level key it returns rather than hardcoding the typo.
    body = payload[next(iter(payload))] if payload else {}
    affinities = body.get("affinities", []) if isinstance(body, dict) else []
    files = [
        _write_json(out, "bindingdb/getLigandsByUniprots_Q9NYL2.json", payload),
    ]

    entry_url = (
        f"{BINDINGDB_BASE}/rwd/jsp/dbsearch/PrimarySearch_ki.jsp?tag=entry&entryid={BINDINGDB_ENTRY_ID}"
    )
    errors: list[str] = []
    try:
        files.append(_write(out, f"bindingdb/entry_{BINDINGDB_ENTRY_ID}.html", f.get(entry_url)))
    except FetchError as exc:  # recorded as partial coverage, not a silent gap
        errors.append(str(exc))

    return {
        "source": "bindingdb",
        "license": LICENSES["bindingdb"],
        "fetched_at": utcnow_iso(),
        "accession": UNIPROT_ACCESSION,
        "entry_id": BINDINGDB_ENTRY_ID,
        "urls": [rest_url, entry_url],
        "files": files,
        "errors": errors,
        "coverage": {
            "rows_retrieved": len(affinities),
            "reported_total": None,  # the REST service reports no total
            "complete": None,
        },
        "n_records": len(affinities),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
SOURCES = {
    "uniprot": fetch_uniprot,
    "rcsb": fetch_rcsb,
    "chembl": fetch_chembl,
    "bindingdb": fetch_bindingdb,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--allow-network", action="store_true", help="required; without it egress is denied")
    parser.add_argument("--out", default=None, help=f"output root (default {DEFAULT_OUT_RELPATH})")
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(SOURCES),
        help="fetch only these sources (repeatable); default is all four",
    )
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else Path(DEFAULT_OUT_RELPATH)
    out.mkdir(parents=True, exist_ok=True)

    policy = EgressPolicy(allow_network=bool(args.allow_network), allowlist=list(ALLOWED_HOSTS))
    fetcher = Fetcher(policy)

    selected = args.only or sorted(SOURCES)
    started = time.perf_counter()
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name in selected:
        try:
            results[name] = SOURCES[name](fetcher, out)
            print(f"[ok] {name}: {results[name]['n_records']} records", file=sys.stderr)
        except Exception as exc:
            failures[name] = repr(exc)
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)

    manifest = {
        "snapshot_kind": "zak-c9r-sources",
        "created_at": utcnow_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "uniprot_accession": UNIPROT_ACCESSION,
        "pdb_id": PDB_ID,
        "ligand_comp_id": LIGAND_COMP_ID,
        "requested_sources": selected,
        "sources": results,
        "failed_sources": failures,
        "request_log": fetcher.log,
        "n_requests": len(fetcher.log),
        "n_request_failures": sum(1 for e in fetcher.log if not e["ok"]),
    }
    manifest_path = out / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"wrote {manifest_path}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
