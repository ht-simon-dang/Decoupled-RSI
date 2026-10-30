"""Calibrated endpoint acquisition with explicit roles and measurement cost."""
import math
from .evaluation import distance


def acquire(predictions,count=4,direction='max',costs=None,cost_budget=None):
    if direction not in {'max','min'} or not 1<=count<=16:raise ValueError('Invalid acquisition parameters')
    if not predictions:return {'selected':[],'reason':'No calibrated predictions; collect condition-matched measurements'}
    context_keys=('endpoint','unit','model_id','conditions_id','environment_id','substrate_id')
    if any(any(p[k]!=predictions[0][k] for k in context_keys) for p in predictions):raise ValueError('Acquisition cannot pool endpoints or assay conditions')
    costs=costs or [1.0]*len(predictions)
    if len(costs)!=len(predictions) or any(not math.isfinite(v) or v<=0 for v in costs):raise ValueError('One positive finite measurement cost is required per candidate')
    if cost_budget is None:cost_budget=float(count)
    if not math.isfinite(cost_budget) or cost_budget<=0:raise ValueError('Positive finite measurement budget required')
    selected=[];remaining=list(range(len(predictions)));spent=0.0
    while remaining and len(selected)<count:
        eligible=[i for i in remaining if spent+costs[i]<=cost_budget]
        if not eligible:break
        role='performance' if len(selected)%2==0 else 'information_diversity'
        if role=='performance':
            chosen=max(eligible,key=lambda i:(predictions[i]['upper'] if direction=='max' else -predictions[i]['lower'])/costs[i])
            rationale='Optimistic endpoint bound per measurement-cost unit; endpoint-specific research utility'
        else:
            chosen=max(eligible,key=lambda i:(min((distance(predictions[i]['sequence'],predictions[s['index']]['sequence']) for s in selected),default=1),
                         (predictions[i]['upper']-predictions[i]['lower'])/costs[i]))
            rationale='Sequence diversity among affordable candidates; interval width as a secondary information heuristic'
        selected.append({'index':chosen,'role':role,'cost':costs[chosen],'reason':rationale,**predictions[chosen]})
        spent+=costs[chosen];remaining.remove(chosen)
    return {'selected':selected,'measurement_cost_used':spent,'measurement_cost_budget':cost_budget,
            'scope':'Acquisition for one calibrated endpoint; not a therapeutic success probability. Conformal width is not pure epistemic uncertainty.',
            'rule':'Alternate optimistic endpoint-bound/cost with sequence diversity; no model weights are changed by acquisition'}
