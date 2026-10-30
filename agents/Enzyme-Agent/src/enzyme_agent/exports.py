"""Downstream sequence exports only include evaluated nominations."""


def nominated_ids(payload):
    selected={r['candidate_id'] for r in payload.get('portfolio',{}).get('selected',[])}
    return {c['id'] for c in payload.get('candidates',[]) if c['id'] in selected and c.get('evaluation',{}).get('status')=='passed'}


def nominated_fasta(payload):
    ids=nominated_ids(payload)
    return '\n'.join(f">{c['id']} origin={c['origin']} evaluation=passed\n{c['sequence']}" for c in payload.get('candidates',[]) if c['id'] in ids)


def validated_handoffs(payload):
    ids=nominated_ids(payload)
    return [h for h in payload.get('mrna_handoff',[]) if h['candidate_id'] in ids]
