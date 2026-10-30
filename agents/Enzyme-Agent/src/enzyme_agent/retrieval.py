"""Bounded public database queries; raw records retain field-level evidence."""
import re
from datetime import datetime, timezone
import requests
from .config import atomic_json, data_root, digest


def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=(15, 60))
    r.raise_for_status()
    if len(r.content) > 8_000_000:
        raise ValueError("Database response exceeds size limit")
    record = {"url": r.url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "data": r.json()}
    record["sha256"] = digest(record)
    atomic_json(data_root() / "references" / (record["sha256"] + ".json"), record)
    return record


def uniprot(accession):
    if not re.fullmatch(r"[A-Z0-9]{6,10}", accession):
        raise ValueError("Invalid UniProt accession")
    return get_json(f"https://rest.uniprot.org/uniprotkb/{accession}.json")


def natural_search(ec, count=8):
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ec or ""):
        raise ValueError("Natural enzyme search requires a complete EC number")
    response = get_json("https://rest.uniprot.org/uniprotkb/search", {"query": f"(ec:{ec}) AND (reviewed:true)", "format": "json", "size": min(max(count, 1), 16)})
    return {"records": [{"accession": x["primaryAccession"], "sequence": x["sequence"]["value"],
                          "organism": x.get("organism"), "protein_description": x.get("proteinDescription"),
                          "features": x.get("features", []), "comments": x.get("comments", []),
                          "origin": "natural_database", "evidence_record_sha256": response["sha256"]} for x in response["data"].get("results", [])],
            "provenance": {k: response[k] for k in ("url", "retrieved_at", "sha256")},
            "limitations": "EC search is not homology proof; mature processing and substrate compatibility require reassessment"}


def natural_candidates(brief,count):
    from .evaluation import distance
    result=natural_search(brief.reaction.ec,count)
    sequences=[];provenances=[]
    for r in result['records']:
        sequence=r['sequence'];chains=[f for f in r['features'] if f['type']=='Chain']
        processing='Unresolved mature-chain boundaries'
        if len(chains)==1:
            start=chains[0]['location']['start'].get('value');end=chains[0]['location']['end'].get('value')
            if start and end:
                sequence=sequence[start-1:end];processing=f'UniProt annotated chain {start}..{end}; annotation evidence retained'
        if not 20<=len(sequence)<=1500 or set(sequence)-set('ACDEFGHIKLMNPQRSTVWY'): continue
        sequences.append(sequence)
        provenances.append({'accession':r['accession'],'organism':r['organism'],'source':'https://www.uniprot.org/uniprotkb/'+r['accession'],
            'record_sha256':r['evidence_record_sha256'],'origin':'natural_database','processing':processing,
            'reference_sequence_identity_proxy':1-distance(sequence,brief.reference.sequence) if brief.reference else None,
            'identity_method':'equal-length Hamming or SequenceMatcher; not phylogenetic homology proof'})
    if not sequences: raise ValueError('No compatible canonical natural sequences returned')
    return {'sequences':sequences,'per_sequence_provenance':provenances,'provenance':result['provenance'],
            'limitations':'Retrieved natural candidates are not new designs; substrate and product configuration need separate evidence'}


def pdb_record(code):
    if not re.fullmatch(r"[0-9][A-Za-z0-9]{3}", code):
        raise ValueError("Invalid PDB identifier")
    r = requests.get(f"https://files.rcsb.org/download/{code.upper()}.pdb", timeout=60)
    r.raise_for_status()
    if len(r.content) > 4_000_000:
        raise ValueError("PDB exceeds 4 MB")
    return {"pdb": r.text, "source": r.url, "sha256": digest(r.text)}
