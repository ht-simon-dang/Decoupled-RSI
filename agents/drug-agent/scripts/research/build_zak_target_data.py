"""Build the ZAK target manifest and measurement data card from the snapshots.

Offline. It reads only what ``scripts/research/fetch_zak_sources.py`` wrote, and
it writes ``target_manifest.json`` and ``data_card.json`` through the platform's
content-addressed artifact store, so both documents are addressable by digest and
neither is a loose file somebody can edit without the hash changing.

What the manifest grounds, and how
----------------------------------
The user's brief said ``Design a drug on target "zak" with 6juu`` and supplied no
receptor file. So the alias is resolved from the instruction outward, and every
link in the chain is recorded rather than assumed:

``zak`` -> **Q9NYL2**
    The provisional public mapping. The UniProt record supplies species and the
    canonical sequence.
Q9NYL2 -> **6JUU chain A**
    The deposited entity sequence is compared residue-by-residue with the UniProt
    sequence, using the deposition's own SIFTS alignment *and* an independent
    offset scan. Both numbers are stored. The PDB title is never the evidence.
6JUU -> **C9R**, the co-crystallized ligand
    Its structure comes from the RCSB chemical-component definition, not from the
    paper's compound table. ``C9R`` is compound **6r**. The paper's headline
    compound **6p** is a different molecule, and the manifest names it explicitly
    under ``not_the_reference_ligand`` with its own InChIKey, so a later reader
    cannot attach 6p's reported Kd = 8.0 nM to the ligand in the structure.
Q9NYL2 -> **CHEMBL3886**
    Resolved from the snapshot's target records by accession, and required to be
    a 1:1 resolution to a single *Homo sapiens* protein.

``residue_map`` is the author-numbered residue sequence of the chosen chain, read
out of the deposited PDB file. It is what :func:`pocket_matches_target` checks a
prepared pocket against, and it is the reason a pocket cut from a different
protein can no longer be handed to the generator unnoticed.

Usage::

    python scripts/research/build_zak_target_data.py
    python scripts/research/build_zak_target_data.py --snapshot .external/research/zak-c9r
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from drug_agent.selfevolve.target.data import (
    FAMILY_EDGE_RULES,
    STANDARDIZER_ID,
    build_data_card,
    deduplicate_records,
    ingest_source_rows,
    normalize_bindingdb_affinity,
    normalize_chembl_activity,
    parse_pdb_residues,
    resolve_reference_compound,
    sequence_identity,
    standardize_molecule,
)
from drug_agent.storage.artifacts import LocalArtifactStore
from drug_agent.util import canonical_json, utcnow_iso

DEFAULT_SNAPSHOT = Path(".external/research/zak-c9r")
DEFAULT_STORE = Path(".drugagent/research/artifacts")
DEFAULT_OUT = Path(".external/research/zak-c9r/derived")

ALIAS = "zak/c9r"
UNIPROT_ACCESSION = "Q9NYL2"
PDB_ID = "6JUU"
LIGAND_COMP_ID = "C9R"

#: Paper compound 6p. Recorded so the *negative* identity is part of the manifest
#: rather than a fact a reader has to already know. Its ChEMBL row is resolved
#: from the snapshot by this structure, never by the label "6p".
PAPER_COMPOUND_6P_SMILES = "COc1n[nH]c2ncc(-c3cn(-c4c(F)ccc(NS(=O)(=O)c5cccc(-c6ccccc6)c5)c4F)nn3)cc12"


def _load(snapshot: Path, relpath: str) -> Any:
    return json.loads((snapshot / relpath).read_text())


def _digest_of(snapshot: Path, relpath: str) -> str:
    return "sha256:" + hashlib.sha256((snapshot / relpath).read_bytes()).hexdigest()


def _source_entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    source = manifest["sources"].get(name, {})
    return {
        "source": name,
        "license": source.get("license"),
        "fetched_at": source.get("fetched_at"),
        "urls": source.get("urls", []),
        "files": source.get("files", []),
        "coverage": source.get("coverage"),
        "n_records": source.get("n_records"),
        "release": source.get("release"),
        "errors": source.get("errors", []),
    }


# ---------------------------------------------------------------------------
# Target manifest
# ---------------------------------------------------------------------------
def build_target_manifest(snapshot: Path) -> dict[str, Any]:
    snap = _load(snapshot, "snapshot_manifest.json")

    uniprot = _load(snapshot, "uniprot/Q9NYL2.json")
    reference_sequence = uniprot["sequence"]["value"]

    entry = _load(snapshot, f"rcsb/entry_{PDB_ID}.json")
    polymer = _load(snapshot, f"rcsb/polymer_entity_{PDB_ID}_1.json")
    ident = polymer["rcsb_polymer_entity_container_identifiers"]
    entity_sequence = polymer["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
    regions: list[dict[str, Any]] = []
    for align in polymer.get("rcsb_polymer_entity_align", []) or []:
        if align.get("reference_database_accession") == UNIPROT_ACCESSION:
            regions.extend(align.get("aligned_regions", []))
    check = sequence_identity(entity_sequence, reference_sequence, aligned_regions=regions or None)

    chain_id = (ident.get("auth_asym_ids") or ["A"])[0]
    pdb_text = (snapshot / f"rcsb/{PDB_ID}.pdb").read_text()
    residue_map = parse_pdb_residues(pdb_text)
    modelled = residue_map.get(chain_id, {})
    numbers = sorted(int("".join(c for c in k if c.isdigit() or c == "-")) for k in modelled)
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers)) if numbers else []

    chemcomp = _load(snapshot, f"rcsb/chemcomp_{LIGAND_COMP_ID}.json")
    descriptor = chemcomp.get("rcsb_chem_comp_descriptor", {})
    ligand = standardize_molecule(descriptor["SMILES_stereo"] or descriptor["SMILES"])
    deposited_key = descriptor.get("InChIKey")
    if deposited_key and deposited_key != ligand["inchikey"]:
        raise SystemExit(
            f"C9R InChIKey disagreement: RCSB says {deposited_key}, "
            f"local standardization says {ligand['inchikey']}"
        )
    six_p = standardize_molecule(PAPER_COMPOUND_6P_SMILES)
    if six_p["inchikey"] == ligand["inchikey"]:
        raise SystemExit("6p and C9R standardize to the same structure; the conflation guard is broken")

    targets = _load(snapshot, "chembl/targets.json")
    matching = [
        t
        for t in targets
        if any(c.get("accession") == UNIPROT_ACCESSION for c in t.get("target_components", []))
        and t.get("target_type") == "SINGLE PROTEIN"
        and t.get("organism") == "Homo sapiens"
    ]
    if len(matching) != 1:
        raise SystemExit(
            f"{UNIPROT_ACCESSION} resolves to {len(matching)} single human ChEMBL proteins; "
            "target identity is ambiguous and must be resolved by hand before ingestion"
        )
    chembl_target = matching[0]

    manifest: dict[str, Any] = {
        "manifest_kind": "target_manifest",
        # No wall-clock stamp: see the note in build_data_card. The snapshot this
        # was built from carries its own fetch time, which is the date that
        # actually describes the evidence.
        "determinism": (
            "byte-reproducible from the same source snapshot and standardizer; "
            "the build time lives in artifact_index.json"
        ),
        "built_from_snapshot_at": snap.get("created_at"),
        "alias": ALIAS,
        "user_instruction": 'Design a drug on target "zak" with 6juu',
        "alias_resolution": {
            "basis": "the user's brief named the alias and the PDB id; no receptor file was supplied",
            "local_receptor_supplied": False,
            "local_receptor_search": [
                "no 6JUU, ZAK, MAP3K20 or C9R file exists in the checkout "
                "(.external/pdb holds 3I28 and 5AI0 only)"
            ],
            "provenance": "public sources fetched by scripts/research/fetch_zak_sources.py",
        },
        "uniprot_accession": UNIPROT_ACCESSION,
        "uniprot_entry_version": uniprot.get("entryAudit", {}).get("entryVersion"),
        "uniprot_sequence_version": uniprot.get("entryAudit", {}).get("sequenceVersion"),
        "gene": (uniprot.get("genes") or [{}])[0].get("geneName", {}).get("value"),
        "gene_synonyms": [
            s.get("value") for s in (uniprot.get("genes") or [{}])[0].get("synonyms", []) or []
        ],
        "protein_name": uniprot.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value"),
        "organism": uniprot["organism"]["scientificName"],
        "taxon_id": uniprot["organism"]["taxonId"],
        "sequence": reference_sequence,
        "sequence_length": uniprot["sequence"]["length"],
        "sequence_md5": uniprot["sequence"].get("md5"),
        "structure": {
            "pdb_id": PDB_ID,
            "title": entry.get("struct", {}).get("title"),
            "method": [m.get("method") for m in entry.get("exptl", [])],
            "resolution_a": entry.get("rcsb_entry_info", {}).get("resolution_combined"),
            "deposited": entry.get("rcsb_accession_info", {}).get("deposit_date"),
            "chain_id": chain_id,
            "auth_asym_ids": ident.get("auth_asym_ids"),
            "entity_id": "1",
            "entity_sequence": entity_sequence,
            "entity_sequence_length": len(entity_sequence),
            "mutations": polymer.get("entity_poly", {}).get("rcsb_mutation_count", 0),
            "pdbx_mutation": polymer.get("rcsb_polymer_entity", {}).get("pdbx_mutation"),
            "construct_note": (
                "crystallized kinase-domain construct; the manifest's construct scope is "
                "the modelled span, not the full-length 800-residue protein"
            ),
            "modelled_residue_range": [numbers[0], numbers[-1]] if numbers else None,
            "n_modelled_residues": len(modelled),
            "n_unmodelled_seqres_residues": len(entity_sequence) - len(modelled),
            "missing_residue_numbers_in_range": missing,
            "n_missing_residues_in_range": len(missing),
            "residue_map": {chain_id: modelled},
            "preparation": {
                "state": "AS_DEPOSITED",
                "parent": f"RCSB {PDB_ID} deposited coordinates",
                "steps": [],
                "note": (
                    "no MD-derived or locally prepared ZAK state exists in this checkout; "
                    "if one appears later its parent and preparation lineage must be recorded here"
                ),
            },
            "files": {
                f"rcsb/{PDB_ID}.pdb": _digest_of(snapshot, f"rcsb/{PDB_ID}.pdb"),
                f"rcsb/{PDB_ID}.cif": _digest_of(snapshot, f"rcsb/{PDB_ID}.cif"),
            },
        },
        "sequence_check": check,
        "reference_ligand": {
            "comp_id": LIGAND_COMP_ID,
            "paper_compound_label": "6r",
            "label_evidence": (
                f"PDB entry title: {entry.get('struct', {}).get('title')!r}; "
                "the label is the deposition's, and identity is carried by the InChIKey below"
            ),
            "inchikey": ligand["inchikey"],
            "connectivity_key": ligand["connectivity_key"],
            "standard_smiles": ligand["standard_smiles"],
            "formula": ligand["formula"],
            "deposited_inchikey": deposited_key,
            "source_file": f"rcsb/chemcomp_{LIGAND_COMP_ID}.json",
            "source_sha256": _digest_of(snapshot, f"rcsb/chemcomp_{LIGAND_COMP_ID}.json"),
        },
        "not_the_reference_ligand": {
            "paper_compound_label": "6p",
            "why_recorded": (
                "6p is the publication's headline compound (reported Kd 8.0 nM, IC50 4.0 nM). "
                "It is NOT the ligand in 6JUU and its potency is not a label for C9R/6r."
            ),
            "inchikey": six_p["inchikey"],
            "connectivity_key": six_p["connectivity_key"],
            "standard_smiles": six_p["standard_smiles"],
            "formula": six_p["formula"],
        },
        "chembl_target": {
            "target_chembl_id": chembl_target["target_chembl_id"],
            "pref_name": chembl_target.get("pref_name"),
            "target_type": chembl_target.get("target_type"),
            "organism": chembl_target.get("organism"),
            "resolution": "1:1 from UniProt accession among SINGLE PROTEIN Homo sapiens targets",
            "n_candidates_considered": len(targets),
        },
        "source_snapshot": {
            "root": str(snapshot),
            "created_at": snap.get("created_at"),
            "n_requests": snap.get("n_requests"),
            "n_request_failures": snap.get("n_request_failures"),
            "files": {
                f["file"]: f["sha256"] for source in snap["sources"].values() for f in source.get("files", [])
            },
        },
        "contamination_notice": {
            "statement": (
                "Both pre-existing ZAK runs in running-history/ generated molecules against "
                ".external/pockets/seh_3i28_pocket10.pdb, a soluble epoxide hydrolase (PDB 3I28) "
                "pocket. They are contaminated evidence and are not baselines or Track A labels."
            ),
            "runs": [
                "running-history/requests/2026-09-11/102331__design-a-drug-on-target-zak-with-6juu__mrun_8056e0afe116",
                "running-history/requests/2026-09-11/114750__design-a-drug-on-target-zak-with-6juu__mrun_5fc27a50f518",
            ],
        },
    }
    manifest["manifest_hash"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_json({k: v for k, v in manifest.items() if k != "generated_at"}).encode("utf-8")
        ).hexdigest()
    )
    return manifest


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def build_measurements(snapshot: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    snap = _load(snapshot, "snapshot_manifest.json")
    chembl_source = snap["sources"]["chembl"]
    bdb_source = snap["sources"]["bindingdb"]

    chembl_raw = _load(snapshot, "chembl/activities.json")
    # A ChEMBL activity names a ChEMBL document; BindingDB names a PubMed id.
    # Without this index the same study wears two different names and every
    # cross-source mirror is counted as an independent experiment.
    document_index = {d["document_chembl_id"]: d for d in _load(snapshot, "chembl/documents.json")}
    chembl = ingest_source_rows(
        chembl_raw,
        normalize_chembl_activity,
        document_index=document_index,
        source="chembl",
        snapshot_sha256=_digest_of(snapshot, "chembl/activities.json"),
        snapshot_file="chembl/activities.json",
        fetched_at=chembl_source["fetched_at"],
        source_url=chembl_source["urls"][0],
        license=chembl_source["license"],
    )

    bdb_payload = _load(snapshot, "bindingdb/getLigandsByUniprots_Q9NYL2.json")
    bdb_raw = bdb_payload[next(iter(bdb_payload))]["affinities"]
    bdb = ingest_source_rows(
        bdb_raw,
        normalize_bindingdb_affinity,
        source="bindingdb",
        snapshot_sha256=_digest_of(snapshot, "bindingdb/getLigandsByUniprots_Q9NYL2.json"),
        snapshot_file="bindingdb/getLigandsByUniprots_Q9NYL2.json",
        fetched_at=bdb_source["fetched_at"],
        source_url=bdb_source["urls"][0],
        license=bdb_source["license"],
        target_id=manifest["chembl_target"]["target_chembl_id"],
    )

    merged = deduplicate_records(chembl["rows"] + bdb["rows"])
    card = build_data_card(
        merged,
        sources=[_source_entry(snap, "chembl"), _source_entry(snap, "bindingdb")],
        exclusions=chembl["exclusions"] + bdb["exclusions"],
        target_manifest=manifest,
    )
    card["family_edge_rules"] = dict(FAMILY_EDGE_RULES)
    card["reference_compounds"] = {
        "C9R_6r": resolve_reference_compound(manifest["reference_ligand"]["standard_smiles"], merged["rows"]),
        "paper_6p": resolve_reference_compound(
            manifest["not_the_reference_ligand"]["standard_smiles"], merged["rows"]
        ),
    }
    return {"rows": merged["rows"], "card": card}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="content-addressed artifact store root")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where the standardized rows are written")
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    if not (snapshot / "snapshot_manifest.json").is_file():
        print(
            f"no snapshot at {snapshot}; run scripts/research/fetch_zak_sources.py --allow-network first",
            file=sys.stderr,
        )
        return 2

    manifest = build_target_manifest(snapshot)
    built = build_measurements(snapshot, manifest)

    store = LocalArtifactStore(args.store)
    manifest_ref = store.put_json(
        manifest, kind="target_manifest", produced_by="scripts/research/build_zak_target_data.py"
    )
    card_ref = store.put_json(
        built["card"], kind="data_card", produced_by="scripts/research/build_zak_target_data.py"
    )
    rows_ref = store.put_json(
        built["rows"],
        kind="measurement_rows",
        produced_by="scripts/research/build_zak_target_data.py",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "target_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out / "data_card.json").write_text(json.dumps(built["card"], indent=1, sort_keys=True))
    (out / "measurements.json").write_text(json.dumps(built["rows"], indent=1, sort_keys=True))

    index = {
        "built_at": utcnow_iso(),
        "standardizer_id": STANDARDIZER_ID,
        "artifact_store": str(store.root),
        "target_manifest": manifest_ref.model_dump(mode="json"),
        "data_card": card_ref.model_dump(mode="json"),
        "measurement_rows": rows_ref.model_dump(mode="json"),
        "manifest_hash": manifest["manifest_hash"],
        "n_rows": built["card"]["n_rows"],
        "n_independent_molecules": built["card"]["n_independent_molecules"],
        "n_strata": built["card"]["n_strata"],
        "n_excluded": built["card"]["n_excluded"],
    }
    (out / "artifact_index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    print(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
