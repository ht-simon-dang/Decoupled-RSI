"""Recommendation boundary for WebUI, CLI, MCP and exports.

Explicit historical inspection uses history.preview_run, never this delivery list.
"""
from copy import deepcopy
from .contracts import Brief
from .filtering import POLICY_ID


def public_run(run):
    from .workflow import construct_from_card
    from .evaluation import portfolio
    raw = run.get('payload',{})
    view = {k:deepcopy(v) for k,v in run.items() if k not in {'payload','events'}}
    p = {k:deepcopy(raw[k]) for k in ('plan','report','stop_reason','error','elapsed_seconds','tool_calls','scientific_tool_seconds','replay_of') if k in raw}
    p.update(candidates=[],evaluations=[],portfolio={'selected':[],'frontier':[],'filter_results':[]})
    p['delivery'] = {'policy_id':POLICY_ID,'state':'withheld','generated_count':len(raw.get('candidates',[])),
                     'evaluated_count':len(raw.get('evaluations',[])),'delivered_count':0}
    if run.get('state') == 'complete' and raw.get('portfolio',{}).get('filter_policy_id') == POLICY_ID:
        try:
            current = portfolio([construct_from_card(c) for c in raw.get('candidates',[])],raw.get('evaluations',[]),Brief.model_validate(run['brief']))
            ids = {r['id'] for r in current['selected']}
            p.update(portfolio=current,candidates=[deepcopy(c) for c in raw.get('candidates',[]) if c['id'] in ids],
                     evaluations=[deepcopy(e) for e in raw.get('evaluations',[]) if e['construct_id'] in ids])
            p['delivery'].update(state='released' if ids else 'no_high_potential_candidates',delivered_count=len(ids))
        except (ValueError,KeyError,TypeError):
            p['delivery']['state']='invalid_evaluation_withheld'
    elif run.get('state') == 'complete':
        p['delivery']['state']='legacy_run_requires_reevaluation'
    # Raw checkpoints, tool arguments/output and feedback proposals contain unfiltered
    # sequences. They never cross this boundary, including while a run is in progress.
    if 'events' in run:
        view['events'] = [{k:e[k] for k in ('id','time','created','kind','run_id') if k in e} for e in run['events']]
    if isinstance(p.get('report'),dict):
        p['report']['recommendation']=str(len(p['candidates']))+' high-potential constructs released'
    view['payload']=p
    return view


def public_feedback(result,run):
    view=deepcopy(result)
    allowed={c['id'] for c in public_run(run)['payload']['candidates']}
    proposed=view.get('next_batch',[])
    view['next_batch']=[r for r in proposed if r['id'] in allowed]
    view['withheld_followup_count']=len(proposed)-len(view['next_batch'])
    view['delivery_policy']='Follow-up recommendations also require current independent evaluation and high-potential filter'
    return view
