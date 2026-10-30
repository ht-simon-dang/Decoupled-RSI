"""Append-only measurements and condition-specific failure learning."""
from .contracts import Brief, Observation, Evidence
from .evaluation import product_id, next_measurement, evaluator_hash

FAILURE_ACTIONS={
    "expression":("expression","Resolve host compatibility, expression and processing; do not train an activity-zero label"),
    "solubility":("stability","Investigate folding and aggregation with matched expression controls"),
    "processing":("processing","Investigate mature processing, glycosylation and assembly"),
    "catalysis":("catalysis","Check substrate compatibility, reaction conditions and catalytic residues"),
    "selectivity":("selectivity","Investigate off-target reactions and normal physiological substrate preservation"),
    "localization":("localization","Investigate delivery and substrate accessibility before pocket redesign"),
    "causal_hypothesis":("mechanism","Revisit causal chain, downstream product burden and disease endpoint"),
    "technical":("mechanism","Repeat/repair the measurement; exclude technical failure from biological training labels"),
}


def qualifies(observation, brief):
    if observation['status']!='measured' or observation['detection']!='quantified':
        return None
    for rule in brief.qualification_criteria:
        keys=('endpoint','layer','substrate_id','environment_id','unit')
        if all(observation[k]==getattr(rule,k) for k in keys) and observation['conditions']==rule.conditions:
            value=observation['value']
            return value>=rule.threshold if rule.direction=='ge' else value<=rule.threshold
    return None


def ingest(store,run_id,observations):
    run=store.get(run_id)
    if run['state'] not in {'complete','blocked','failed'}:
        raise ValueError("Feedback requires a stopped computational run")
    brief=Brief.model_validate(run['brief'])
    if run['payload'].get('evaluator',{}).get('sha256')!=evaluator_hash():
        raise ValueError('Evaluator changed; create a new project run or use its archived evaluator before importing measurements')
    candidates={c['id']:c for c in run['payload'].get('candidates',[])}
    prepared=[]
    for value in observations:
        observation=Observation.model_validate(value)
        c=candidates.get(observation.candidate_id)
        if not c or observation.product_id!=product_id(c['product']):
            raise ValueError("Observation candidate/product identity mismatch")
        if observation.environment_id not in {e.id for e in brief.environments}:
            raise ValueError("Unknown measurement environment")
        if observation.substrate_id not in {s.id for s in brief.panel}:
            raise ValueError("Unknown substrate; add a project panel revision before measurement import")
        if observation.qualification_threshold is not None:
            matches=[r for r in brief.qualification_criteria if r.endpoint==observation.endpoint and r.layer==observation.layer
                     and r.substrate_id==observation.substrate_id and r.environment_id==observation.environment_id
                     and r.conditions==observation.conditions and r.unit==observation.unit and r.threshold==observation.qualification_threshold
                     and r.direction==observation.threshold_direction and r.source==observation.threshold_source]
            if not matches:
                raise ValueError("Observation cannot add or relax the project's predeclared qualification threshold")
        prepared.append(observation.model_dump())
    inserted=store.add_observations(run_id,prepared)
    summary=summarize(store,run_id)
    payload={**run['payload'],'feedback':summary}
    # Append experimental evidence; never overwrite an existing prediction.
    for c in payload.get('candidates',[]):
        for o in summary['observations']:
            if o['candidate_id']!=c['id']:continue
            conditions={**o['conditions'],'measurement_id':o['id'],'layer':o['layer'],'batch':o['batch'],'replicate':o['replicate'],
                        'expression_host':o['expression_host'],'processing_state':o['processing_state'],'reference_control':o['reference_control'],'detection':o['detection']}
            if any(e.get('conditions',{}).get('measurement_id')==o['id'] for e in c['evidence']):continue
            value=o['value']
            if o['detection'] in {'below_limit','above_limit'}:value=('<' if o['detection']=='below_limit' else '>')+str(o['detection_limit'])
            c['evidence'].append(Evidence(endpoint=o['endpoint'],status=o['status'],value=value,unit=o['unit'],conditions=conditions,
                source=o['source'],version=o['version'],substrate_id=o['substrate_id'],environment_id=o['environment_id'],
                uncertainty='As reported by the experiment source; raw context retained',applicability='User-reported experiment, valid only for this product and these conditions').model_dump())
        if c['state']!='rejected':
            if c['id'] in summary['qualified_candidates']['disease_qualified']:c['state']='disease_system_validated'
            elif c['id'] in summary['qualified_candidates']['biochemical_qualified']:c['state']='biochemical_validated'
    from .workflow import report_text
    payload['report']=report_text(run_id,brief,payload)+'\n\n## Experimental feedback\n'+str(summary['denominators'])+'\n\n'+'\n'.join('- '+a['reason'] for a in summary['actions'])
    payload['feedback_report_sha256']=store.artifact({'feedback':summary,'candidates':payload.get('candidates',[]),'evaluator':payload['evaluator']})
    store.update(run_id,run['state'],payload)
    return {"inserted":inserted,"summary":summary}


def summarize(store,run_id):
    run=store.get(run_id);brief=Brief.model_validate(run['brief'])
    observations=store.observations(run_id)
    actions=[]
    buckets={k:set() for k in ['nominated','expression_success','usable_protein','measured','biochemical_qualified','disease_qualified']}
    buckets['nominated']={x['candidate_id'] for x in run['payload'].get('portfolio',{}).get('selected',[])}
    qualified_by_candidate={}
    for o in observations:
        cid=o['candidate_id'];q=qualifies(o,brief)
        if o['status']=='measured':
            buckets['measured'].add(cid)
        if q is True:
            qualified_by_candidate.setdefault(cid,set()).add(o['layer'])
            if o['layer']=='expression': buckets['expression_success'].add(cid)
            if o['layer']=='solubility': buckets['usable_protein'].add(cid)
            if o['layer']=='disease_substrate': buckets['biochemical_qualified'].add(cid)
        failure='technical' if o['status']=='technical_failure' else o['failure_category']
        if failure:
            bottleneck,reason=FAILURE_ACTIONS[failure]
            actions.append({'candidate_id':cid,'observation_id':o['id'],'bottleneck':bottleneck,'reason':reason,
                            'next_measurement':next_measurement(bottleneck)})
    # Matching disease-system and true-substrate measurements are necessary;
    # isolated artificial-reporter activity cannot establish disease validation.
    for cid,layers in qualified_by_candidate.items():
        if {'disease_substrate','disease_system'}.issubset(layers):
            buckets['disease_qualified'].add(cid)
    return {'observations':observations,'actions':actions,'denominators':{k:len(v) for k,v in buckets.items()},
        'qualified_candidates':{k:sorted(v) for k,v in buckets.items() if k in {'biochemical_qualified','disease_qualified'}},
        'qualification':'Only exact conditions/units matched to criteria declared before generation; no automatic clinical status',
        'unknown_qualification_count':sum(qualifies(o,brief) is None for o in observations)}


def next_brief(store,run_id):
    run=store.get(run_id);brief=Brief.model_validate(run['brief'])
    feedback=summarize(store,run_id)
    if not feedback['actions']:
        return {'action':'measure','reason':'No interpretable failure evidence supports another sequence-design round',
                'next_measurement':next_measurement(brief.bottleneck)}
    action=feedback['actions'][-1]
    values=brief.model_dump();values.update(parent_run_id=run_id,bottleneck=action['bottleneck'],seed=brief.seed+1)
    if action['bottleneck'] in {'processing','localization','mechanism'}:
        values.update(route='research',generators=['library'])
    return {'action':'proposed_next_round','reason':action['reason'],'brief':Brief.model_validate(values).model_dump()}
