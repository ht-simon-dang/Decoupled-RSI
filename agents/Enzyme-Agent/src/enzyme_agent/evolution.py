"""EMA4D outer-loop replay with a sealed evaluator and a fixed candidate pool."""
from .config import digest
from .evaluation import evaluator_hash, distance


def compare_policies(run):
    p=run['payload']
    if p.get('evaluator',{}).get('sha256')!=evaluator_hash():
        return {'status':'blocked','reason':'The evaluator changed since this run. Use the archived evaluator source or create a new fixed-evaluator run; no policy comparison was made.'}
    candidates=[c for c in p.get('candidates',[]) if not c['rejection_reasons'] and c.get('evaluation',{}).get('status')=='passed']
    budget=min(run['brief']['budget']['shortlist'],run['brief']['budget']['experiment_slots'])
    baseline=[x['candidate_id'] for x in p.get('portfolio',{}).get('selected',[])][:budget]
    selected=[]
    remaining=list(candidates)
    while remaining and len(selected)<budget:
        chosen=max(remaining,key=lambda c:(min((distance(c['sequence'],s['sequence']) for s in selected),default=1),
                    sum(e['status']=='not_assessed' for e in c['evidence'])))
        selected.append(chosen);remaining.remove(chosen)
    def score(ids):
        records=[c for c in candidates if c['id'] in ids]
        d=[distance(a['sequence'],b['sequence']) for i,a in enumerate(records) for b in records[:i]]
        return {'selected':ids,'count':len(ids),'mean_sequence_distance':sum(d)/len(d) if d else None,
                'measured_disease_success':None,'cost_additional_generation':0}
    return {'loop':'outer_workflow_policy','weights_changed':False,'generator_fixed':True,
            'candidate_pool_sha256':digest([c['id'] for c in candidates]),'brief_sha256':digest(run['brief']),
            'evaluator_sha256':evaluator_hash(),'budget':budget,
            'arms':{'fixed_pareto':score(baseline),'diversity_information':score([c['id'] for c in selected])},
            'conclusion':'Ablation replay only; no claim of prospective improvement. Family-held-out and real assay validation remain required.',
            'protected_rules':['No threshold changes','No predicted-to-measured promotion','No test-data adaptation','No negative-result reinterpretation']}
