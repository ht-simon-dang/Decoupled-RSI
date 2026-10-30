"""Group-separated predictor evaluation and measured pipeline denominators."""
from __future__ import annotations
import csv
import io
import math
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from .config import atomic_json, data_root, digest
from .contracts import AA
from .evaluation import distance, evaluator_hash

REQUIRED={'sequence','value','unit','endpoint','environment_id','conditions_id','substrate_id','reaction_id','family','cluster','source','split','status'}


def features(sequences):
    pairs=[a+b for a in AA for b in AA]
    return np.array([[s.count(a)/len(s) for a in AA]+[sum(s[i:i+2]==p for i in range(len(s)-1))/max(1,len(s)-1) for p in pairs]+[len(s)/1000] for s in sequences],dtype=float)


def dataset(csv_text, split_mode='cross_family'):
    rows=list(csv.DictReader(io.StringIO(csv_text)))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError('CSV requires: '+','.join(sorted(REQUIRED)))
    if len(rows)>2000:
        raise ValueError('Calibration dataset exceeds 2000-record application limit')
    if split_mode not in {'cross_family','within_family','substrate_transfer','reaction_transfer'}:
        raise ValueError('Unknown split mode')
    for row in rows:
        if any(not row[k].strip() for k in REQUIRED): raise ValueError('All required dataset fields must be nonempty')
        if row['status']!='measured': raise ValueError('Only measured, uncensored labels may train the function proxy')
        if set(row['sequence'])-set(AA) or not 20<=len(row['sequence'])<=1500: raise ValueError('Invalid sequence')
        row['value']=float(row['value'])
        if not math.isfinite(row['value']): raise ValueError('Nonfinite measurement')
        if row['split'] not in {'train','calibration','test'}: raise ValueError('Unknown dataset split')
    # Endpoint and assay conditions are fixed; units are never silently converted.
    for key in ('endpoint','unit','environment_id','conditions_id'):
        if len({r[key] for r in rows})!=1: raise ValueError(f'Calibration cannot pool different {key}')
    required_axes={'cluster'}
    if split_mode=='cross_family': required_axes|={'family','source'}
    if split_mode=='substrate_transfer': required_axes|={'substrate_id','source'}
    if split_mode=='reaction_transfer': required_axes|={'reaction_id','source'}
    leaks={}
    for axis in ('family','cluster','source','substrate_id','reaction_id'):
        seen={}
        for r in rows: seen.setdefault(r[axis],set()).add(r['split'])
        overlaps=[v for v,splits in seen.items() if len(splits)>1]
        leaks[axis]=overlaps
        if overlaps and axis in required_axes: raise ValueError(f'{axis} leakage across splits: {overlaps[:3]}')
    unique=set()
    for r in rows:
        identity=(r['sequence'],r['substrate_id'],r['conditions_id'])
        if identity in unique: raise ValueError('Duplicate sequence/substrate/condition measurement; aggregate replicates explicitly')
        unique.add(identity)
    # Verify obvious mislabelled clusters rather than relying only on user IDs.
    if split_mode=='cross_family':
        for i,a in enumerate(rows):
            for b in rows[:i]:
                if a['split']!=b['split'] and distance(a['sequence'],b['sequence'])<0.2:
                    raise ValueError('>=80% sequence similarity crosses the declared family split')
    counts={s:sum(r['split']==s for r in rows) for s in ('train','calibration','test')}
    if min(counts.values())<10: raise ValueError('Need at least 10 measured records in each train/calibration/test partition')
    return rows,{'counts':counts,'split_mode':split_mode,'leakage_audit':leaks,'enforced_axes':sorted(required_axes),
                 'scope':'Split validation cannot establish label authenticity or remove public model pretraining contamination'}


def metrics(y,pred):
    correlation=float(spearmanr(y,pred).statistic) if len(set(pred))>1 and len(set(y))>1 else None
    return {'rmse':float(np.sqrt(np.mean((y-pred)**2))),'mae':float(np.mean(abs(y-pred))),
            'spearman':correlation if correlation is None or math.isfinite(correlation) else None,'n':len(y)}


def calibrate(csv_text, model_id, split_mode='cross_family', alpha=0.1):
    if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',model_id): raise ValueError('Invalid model id')
    path=data_root()/'calibrations'/f'{model_id}.json'
    if path.exists(): raise ValueError('Calibration versions are immutable; choose a new model id')
    rows,audit=dataset(csv_text,split_mode)
    x=features([r['sequence'] for r in rows]);y=np.array([r['value'] for r in rows])
    masks={s:np.array([r['split']==s for r in rows]) for s in ('train','calibration','test')}
    train,cal,test=[masks[s] for s in ('train','calibration','test')]
    scaler=StandardScaler().fit(x[train]);z=scaler.transform(x)
    ridge=Ridge(alpha=10.0).fit(z[train],y[train])
    dummy=DummyRegressor(strategy='mean').fit(z[train],y[train])
    residual=abs(y[cal]-ridge.predict(z[cal]))
    quantile=math.ceil((len(residual)+1)*(1-alpha))/len(residual)
    if quantile>1: raise ValueError('Calibration set too small for requested conformal coverage')
    radius=float(np.quantile(residual,quantile,method='higher'))
    prediction=ridge.predict(z[test])
    info={"id":model_id,"version":"ridge-composition-conformal-1","dataset_sha256":digest(rows),"audit":audit,
          "endpoint":rows[0]['endpoint'],"unit":rows[0]['unit'],"environment_id":rows[0]['environment_id'],"conditions_id":rows[0]['conditions_id'],
          "train_families":sorted({r['family'] for r in rows if r['split']=='train'}),
          "substrates":sorted({r['substrate_id'] for r in rows if r['split']=='train'}),
          "mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"coef":ridge.coef_.tolist(),"intercept":float(ridge.intercept_),
          "radius":radius,"alpha":alpha,"test":metrics(y[test],prediction),"dummy_baseline":metrics(y[test],dummy.predict(z[test])),
          "test_interval_coverage":float(np.mean(abs(y[test]-prediction)<=radius)),"test_interval_width":2*radius,
          "training_sequences":[r['sequence'] for r in rows if r['split']=='train'],
          "evaluator_sha256":evaluator_hash(),"limitations":"Composition/dipeptide proxy, not an ESMC model; split-conformal marginal intervals depend on exchangeability; no clinical claim"}
    atomic_json(path,info)
    atomic_json(data_root()/'calibrations'/f'{model_id}.dataset.json',rows)
    return {k:v for k,v in info.items() if k not in {'mean','scale','coef','training_sequences'}}


def predict(model_id,sequences,endpoint,unit,environment_id,conditions_id,substrate_id):
    import json
    if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',model_id): raise ValueError('Invalid model id')
    model=json.loads((data_root()/'calibrations'/f'{model_id}.json').read_text())
    for key,value in [('endpoint',endpoint),('unit',unit),('environment_id',environment_id),('conditions_id',conditions_id)]:
        if model[key]!=value: raise ValueError(f'Prediction domain mismatch: {key}')
    if substrate_id not in model['substrates']: raise ValueError('Uncalibrated substrate')
    if any(set(s)-set(AA) or not 20<=len(s)<=1500 for s in sequences): raise ValueError('Invalid query sequence')
    x=features(sequences)
    values=((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coef'])+model['intercept']
    return [{'sequence':s,'endpoint':endpoint,'unit':unit,'value':float(v),'lower':float(v-model['radius']),'upper':float(v+model['radius']),
             'status':'predicted','model_id':model_id,'conditions_id':conditions_id,'environment_id':environment_id,'substrate_id':substrate_id,
             'source':'Local project calibration','version':model['version'],'dataset_sha256':model['dataset_sha256'],
             'evaluator_sha256':model['evaluator_sha256'],
             'nearest_training_identity':1-min(distance(s,t) for t in model['training_sequences']),
             'uncertainty':'Marginal conformal interval; out-of-domain coverage is not guaranteed'} for s,v in zip(sequences,values)]


def pipeline_metrics(run):
    p=run['payload'];c=p.get('candidates',[])
    designs=[x for x in c if x['origin'] not in {'reference','natural_database','user_library'}]
    distances=[distance(a['sequence'],b['sequence']) for i,a in enumerate(designs) for b in designs[:i]]
    measured=p.get('feedback',{}).get('denominators')
    cost=p.get('budget',{})
    return {'run_id':run['id'],'state':run['state'],'requested':p.get('generation_counts',{}).get('requested'),
        'returned':p.get('generation_counts',{}).get('returned',0),'unique_designs':len(designs),
        'retrieved_natural_candidates':sum(x['origin']=='natural_database' for x in c),
        'valid_designs':sum(not x['rejection_reasons'] for x in designs),
        'valid_fraction':sum(not x['rejection_reasons'] for x in designs)/len(designs) if designs else None,
        'mean_pairwise_distance':sum(distances)/len(distances) if distances else None,
        'computational_nominations':len(p.get('portfolio',{}).get('selected',[])),
        'experimental_denominators':measured,'experimental_hit_rate':None if not measured or not measured['measured'] else measured['biochemical_qualified']/measured['measured'],
        'tool_seconds':cost.get('tool_seconds'),'tool_calls':cost.get('tool_calls'),
        'cost_per_disease_qualified':cost.get('tool_seconds',0)/measured['disease_qualified'] if measured and measured['disease_qualified'] else None,
        'failure_distribution':{'rejected':sum(bool(x['rejection_reasons']) for x in c),'unavailable_assessment':len(p.get('unavailable_assessments',[]))},
        'evaluator':p.get('evaluator'),'scope':'Computational metrics only until experimental data are supplied'}
