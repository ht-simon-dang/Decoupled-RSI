"""Recoverable plan -> evidence -> generate -> independently assess -> nominate."""
from __future__ import annotations
import time
import fcntl
from .config import digest
from .contracts import Brief, Candidate, Evidence
from .evaluation import assess, make_candidate, portfolio, evaluator_hash, evaluator_sources, EVALUATOR_VERSION
from .gateway import Gateway
from .store import Store
from .structure import surface_positions, compare_structures
from .templates import mechanism_card
from .tool_evaluation import validation_record


class Engine:
    def __init__(self,store):
        self.store=store

    def run(self,brief):
        brief=Brief.model_validate(brief) if isinstance(brief,dict) else brief
        return self.resume(self.store.create(brief.model_dump()))

    def chat(self,instruction):
        return self.resume(self.store.create({}, {'instruction':instruction}))

    def resume(self,run_id):
        # One executor per run across CLI/MCP/Web processes. Failed attempts do
        # not reset the durable budget and a second process cannot duplicate work.
        self.store.get(run_id)
        lock_root=self.store.root/'locks';lock_root.mkdir(exist_ok=True)
        with (lock_root/f'{run_id}.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('This run already has an active executor')
            return self._resume(run_id)

    def _resume(self,run_id):
        run=self.store.get(run_id)
        if run['state']=='complete': return run
        if run['state']=='cancelled': raise ValueError('Queue the cancelled run before resuming')
        payload=run['payload'];payload.pop('error',None)
        start=time.monotonic();prior=payload.get('elapsed_seconds',0)
        def save(state):
            payload['elapsed_seconds']=prior+time.monotonic()-start
            self.store.update(run_id,state,payload)
        try:
            if not run['brief']:
                save('planning')
                gateway=Gateway(self.store,run_id,24,time.monotonic()+240)
                result=gateway.call('planning.qwen',{'instruction':payload['instruction']})
                payload['planning']=result.model_dump()
                if result.status!='ok':
                    payload['error']=result.error;save('blocked' if result.status=='unavailable' else 'failed');return self.store.get(run_id)
                if 'handoff' in result.data:
                    payload['handoff']=result.data;save('complete');return self.store.get(run_id)
                self.store.set_brief(run_id,result.data['brief'])
                run=self.store.get(run_id)
            brief=Brief.model_validate(run['brief'])
            previous_evaluator=payload.get('evaluator',{}).get('sha256')
            if previous_evaluator and previous_evaluator!=evaluator_hash():
                raise ValueError('Evaluator changed since this run began; create a new run to keep fixed evaluation rules')
            deadline=time.monotonic()+brief.budget.wall_seconds-payload.get('elapsed_seconds',0)
            gateway=Gateway(self.store,run_id,brief.budget.tool_calls,deadline)
            payload['mechanism']=mechanism_card(brief)
            payload['reaction_specification']={"reaction":brief.reaction.model_dump(),"panel":[s.model_dump() for s in brief.panel],
                "environments":[e.model_dump() for e in brief.environments],"criteria":[c.model_dump() for c in brief.qualification_criteria]}
            payload['evaluator']={'version':EVALUATOR_VERSION,'sha256':evaluator_hash(),'source_artifact_sha256':self.store.artifact(evaluator_sources())}
            save('evidence_review')
            contradicted=[k for k in ('direction_gate','product_gate','access_gate') if getattr(brief,k).status=='contradicted']
            if contradicted:
                payload.update(stop_reason='Contradicted prerequisite: '+', '.join(contradicted),computational_success=False)
                payload['report']=report_text(run_id,brief,payload);save('complete');return self.store.get(run_id)
            if brief.natural_search and 'natural_search' not in payload:
                save('retrieving')
                result=gateway.call('retrieval.natural',{'ec':brief.reaction.ec or '', 'count':min(brief.budget.candidates,16)})
                payload['natural_search']=result.model_dump()
                save('retrieved')
            if brief.route=='research':
                payload.update(stop_reason='Evidence exploration completed; resolve recorded uncertainties before sequence engineering',computational_success=False)
                payload['report']=report_text(run_id,brief,payload);save('complete');return self.store.get(run_id)
            if brief.route=='de_novo':
                mechanism_ready=brief.reaction.mechanism!='unknown' and brief.reaction.mechanism_sources and brief.reaction.states
                if not mechanism_ready:
                    payload['error']='De novo design requires evidenced catalytic roles, atom geometry and reaction-state constraints; return to mechanism exploration'
                    save('blocked');return self.store.get(run_id)
            if brief.reference and not brief.design_positions and brief.reference.pdb and any(g not in {'library','zymctrl','natural'} for g in brief.generators):
                values=brief.model_dump();values['design_positions']=surface_positions(brief.reference,min(8,brief.max_mutations))
                brief=Brief.model_validate(values);self.store.set_brief(run_id,brief.model_dump())
                payload['position_selection']={'method':'SASA ranking, excluding protected/Cys/Gly/Pro and within 10 A of protected CA','positions':brief.design_positions,
                    'status':'inferred','limitation':'Surface exposure is not proof of functional neutrality'}
            generated=payload.setdefault('generation',{})
            sources=[g for g in brief.generators if g!='library']
            if len(sources)>brief.budget.candidates:
                raise ValueError('Candidate budget smaller than number of generators')
            for index,g in enumerate(sources):
                if generated.get(g,{}).get('status')=='ok': continue
                save('generating')
                count=brief.budget.candidates//len(sources)+(index<brief.budget.candidates%len(sources))
                result=gateway.call('generation.'+g,{'brief':brief.model_dump(),'count':int(count)})
                generated[g]=result.model_dump();save('generated')
                if result.status!='ok':
                    payload['error']=f'{g}: {result.error}'
                    save('blocked' if result.status=='unavailable' else 'failed')
                    return self.store.get(run_id)
            candidates=[];seen={};duplicates=[]
            parent=None
            if brief.reference:
                parent=make_candidate(brief.reference.sequence,'reference',brief,{'sources':brief.reference.sources,'accession':brief.reference.accession})
                candidates.append(parent);seen[parent.id]=parent
            all_outputs=[]
            if 'library' in brief.generators:
                all_outputs += [(s,'user_library',{'source':'user supplied','novelty':'not claimed'}) for s in brief.library[:brief.budget.candidates]]
            for g,result in generated.items():
                if result['status']=='ok':
                    all_outputs += [(s,'natural_database' if g=='natural' else g,
                                     result['data'].get('per_sequence_provenance',[result['data']['provenance']]*len(result['data']['sequences']))[i])
                                    for i,s in enumerate(result['data']['sequences'])]
            for seq,g,provenance in all_outputs:
                c=make_candidate(seq,g,brief,provenance,parent.id if parent and g not in {'zymctrl','natural_database'} else None)
                if c.id in seen:
                    duplicates.append({'candidate_id':c.id,'source':g,'reason':'identical sequence and product configuration'})
                    continue
                candidates.append(c);seen[c.id]=c
            if not candidates: raise ValueError('No candidates; provide a reference, library or usable generator')
            stats={c.id:assess(c,brief) for c in candidates}
            payload.update(candidates=[c.model_dump() for c in candidates],descriptors=stats,duplicates=duplicates,
                           generation_counts={'requested':brief.budget.candidates,'returned':len(all_outputs),'unique_designs':sum(c.origin!='reference' for c in candidates),'reference':int(parent is not None)})
            save('assessing')
            folds=payload.setdefault('folds',{})
            evaluations=payload.setdefault('evaluations',{})
            payload['evaluation_policy']={'mandatory_all_candidates':True,'legacy_fold_limit_ignored':brief.budget.fold_candidates,
                'thresholds':brief.evaluation_policy.model_dump(),'threshold_scope':'Predeclared computational screens, not calibrated activity thresholds'}
            for c in candidates:
                cid=c.id
                records=evaluations.setdefault(cid,{})
                if c.state=='rejected':
                    c.evaluation=validation_record(c,brief,records)
                    c.evaluation['status']='failed'
                    continue
                if cid not in folds or folds[cid]['status']!='ok':
                    save('folding')
                    result=gateway.call('structure.esmfold2',{'sequence':c.sequence,'seed':brief.seed,'steps':brief.fold_steps,'loops':brief.fold_loops})
                    folds[cid]=result.model_dump();save('fold_assessed')
                result=folds[cid]
                records['structure.esmfold2']=result
                if result['status']=='ok':
                    data=result['data']
                    if data['sequence']!=c.sequence: raise ValueError('Fold/candidate sequence identity mismatch')
                    c.pdb=data['pdb']
                    c.evidence.append(Evidence(endpoint='mean_plddt',status='predicted',value=data['mean_plddt_fraction'],unit='fraction [0,1]',
                        source='ESMFold2-Fast',version=data['provenance']['revision'],conditions={'template_conditioned':False,'steps':float(brief.fold_steps)},
                        applicability='Single-chain unmodified structure confidence; not catalytic function'))
                    c.evidence.append(Evidence(endpoint='ptm',status='predicted',value=data['ptm'],unit='fraction [0,1]',
                        source='ESMFold2-Fast',version=data['provenance']['revision'],applicability='Fold confidence; not catalytic function'))
                    if records.get('evaluation.structure',{}).get('status')!='ok':
                        records['evaluation.structure']=gateway.call('evaluation.structure',{'brief':brief.model_dump(),'sequence':c.sequence,'pdb':c.pdb}).model_dump()
                        save('structure_assessed')
                    geometry=records['evaluation.structure'].get('data',{}).get('geometry',{})
                    if geometry.get('status')=='ok':
                        for endpoint,key in [('backbone_rmsd','backbone_rmsd_angstrom'),('catalytic_sidechain_rmsd','catalytic_sidechain_rmsd_angstrom')]:
                            if geometry.get(key) is not None:
                                c.evidence.append(Evidence(endpoint=endpoint,status='predicted',value=geometry[key],unit='angstrom',
                                    source='Independent fold versus reference geometry',version=EVALUATOR_VERSION,applicability=geometry['limitations']))
                else:
                    c.evidence.append(Evidence(endpoint='independent_fold',status='technical_failure' if result['status']=='error' else 'not_assessed',
                        source='ESMFold2-Fast',version='unavailable',applicability=result.get('error') or 'unavailable'))
                stability_applicable=bool(c.mutations and brief.reference and brief.reference.pdb and brief.route in {'conservative','product'})
                if stability_applicable:
                    if records.get('evaluation.thermompnn',{}).get('status')!='ok':
                        records['evaluation.thermompnn']=gateway.call('evaluation.thermompnn',{'brief':brief.model_dump(),'sequence':c.sequence}).model_dump()
                        save('stability_assessed')
                    stability=records['evaluation.thermompnn']
                    if stability['status']=='ok':
                        for row in stability['data']['mutations']:
                            c.evidence.append(Evidence(endpoint='point_mutation_ddg',status='predicted',value=row['ddg_kcal_mol'],unit='kcal/mol',
                                source='ThermoMPNN',version=stability['data']['provenance']['revision'],conditions={'mutation':row['mutation'],'background':'reference'},
                                applicability=stability['data']['scope']))
                    else:
                        c.evidence.append(Evidence(endpoint='point_mutation_ddg',status='technical_failure',source='ThermoMPNN',version='unavailable',applicability=stability.get('error') or 'unavailable'))
                c.evaluation=validation_record(c,brief,records)
                if c.evaluation['status']=='failed':
                    c.state='rejected';c.rejection_reasons.extend(c.evaluation['failures'])
                payload['candidates']=[x.model_dump() for x in candidates]
                save('candidate_evaluated')
            payload['portfolio']=portfolio(candidates,stats,brief)
            payload['candidates']=[c.model_dump() for c in candidates]
            payload['unavailable_assessments']=[{'candidate_id':cid,'tool':tool,'status':r['status'],'error':r.get('error')} for cid,records in evaluations.items() for tool,r in records.items() if r['status']!='ok']
            payload['evaluation_summary']={status:sum(c.evaluation.get('status')==status for c in candidates) for status in ('passed','failed','incomplete')}
            payload['evaluation_summary']['total']=len(candidates)
            payload['budget']={**self.store.cost(run_id),'limits':brief.budget.model_dump(),'candidate_slots_used':len(all_outputs),
                               'experiment_slots_proposed':len(payload['portfolio']['selected'])}
            gates_supported=all(getattr(brief,g).status=='supported' for g in ('direction_gate','product_gate','access_gate'))
            payload['evidence_readiness']='computational_priority' if gates_supported else 'exploratory'
            payload['computational_success']=bool(payload['portfolio']['selected']) and not payload['unavailable_assessments'] and gates_supported
            payload['scientific_success']={'enzyme_engineering':'not established without qualifying measurements','disease_function':'not established without disease-relevant measurements'}
            payload['stop_reason']='Computational budget completed; next advancement requires the specified measurements'
            if payload['evaluation_summary']['incomplete']:
                payload['stop_reason']='Required evaluation tools incomplete; affected candidates cannot be nominated'
                payload['error']='Required candidate evaluations are missing or failed to execute; see evaluations and unavailable_assessments'
            payload['mrna_handoff']=[mrna_handoff(c,brief) for c in candidates if c.id in {r['candidate_id'] for r in payload['portfolio']['selected']}]
            payload['report']=report_text(run_id,brief,payload)
            payload.pop('report_sha256',None)
            payload['report_sha256']=self.store.artifact(payload)
            save('blocked' if payload['evaluation_summary']['incomplete'] else 'complete')
        except InterruptedError:
            pass
        except Exception as exc:
            payload['error']=f'{type(exc).__name__}: {exc}'
            if self.store.get(run_id)['state']!='cancelled': save('failed')
        return self.store.get(run_id)


def mrna_handoff(candidate,brief):
    p=candidate.product
    return {'candidate_id':candidate.id,'coding_protein':p.precursor_sequence or candidate.sequence,'mature_sequence':candidate.sequence,
        'precursor_mature_relation':{'start':p.mature_start,'end':p.mature_end},'processing':p.processing,'modifications':p.modifications,
        'targeting':p.targeting,'compartments':[e.compartment for e in brief.environments],
        'protected_mature_positions':brief.reference.protected_positions if brief.reference else [],
        'required_functional_measurements':[brief.endpoint,'mature enzyme turnover and processing'],
        'unresolved':['Expression does not establish mature enzyme function','No nucleotide sequence or delivery efficacy inferred']}


def report_text(run_id,brief,payload):
    lines=[f'# {brief.name}',f'Run: {run_id}',f'Disease: {brief.disease}',f'Endpoint: {brief.endpoint}',
        '', '## Current conclusion',payload.get('stop_reason','In progress'),
        'Candidates must pass computational evaluation tools before nomination. Biochemical activity and disease-function remain unvalidated.',
        '', '## Reaction and route',brief.reaction.name,'Products: '+', '.join(brief.reaction.products),'Product fate: '+brief.reaction.product_fate,
        '', '## Evidence gaps']
    lines += ['- '+x for x in payload.get('mechanism',{}).get('gaps',[])]
    lines += ['', '## Portfolio']
    for row in payload.get('portfolio',{}).get('selected',[]):
        lines.append(f"- {row['candidate_id']} ({row['role']}): {row['reason']}. Next: {row['next_measurement']}")
    lines += ['', '## Required evaluation tools',str(payload.get('evaluation_summary',{}))]
    for c in payload.get('candidates',[]):
        e=c.get('evaluation',{})
        lines.append(f"- {c['id']}: {e.get('status','not_evaluated')}; required: {', '.join(e.get('required_tools',[]))}; missing: {e.get('missing_tools',[])}; failed: {e.get('failures',[])}")
    lines += ['', '## Failures and budget',f"Unavailable assessments: {len(payload.get('unavailable_assessments',[]))}",
        f"Duplicates: {len(payload.get('duplicates',[]))}",str(payload.get('budget',{})),
        '', 'All raw evidence, units, conditions, configuration identities and model revisions are retained in the JSON report.']
    return '\n'.join(lines)
