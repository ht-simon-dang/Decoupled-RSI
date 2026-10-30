"""Input-scope checks and independent evidence for optional models."""
from .registry import MODELS,manifest


def predict(constructs,brief,gateway,output):
    from .evaluation import missing,measured
    cards={c.id:c for c in constructs}
    specs={
      'degscore':('degscore','sum of per-base DegScore 2.1',lambda c:len(c.sequence)<=2000 and c.molecule.modification in {'unmodified','unspecified'},'Unmodified RNA; Vienna structure input; chemical degradation control'),
      'saluki':('rna_stability','normalized stability consensus (not hours)',lambda c:brief.context.species in {'human','mouse'} and len(c.regions.transcript)<=12288 and c.molecule.modification in {'unmodified','unspecified'},'Human/mouse, <=12288 nt, unmodified RNA; synthetic splice channel zero'),
      'rnadegformer':('chemical_degradation','mean predicted deg_Mg_50C',lambda c:len(c.sequence)<=1024 and c.molecule.modification in {'unmodified','unspecified'},'Unmodified RNA, <=1024 nt application limit; short-RNA training/long-mRNA domain shift; chemical Mg/50 C assay, not cellular half-life'),
      'utr_lm':('ribosome_load','MRL',lambda c:len(c.regions.utr5)==50 and c.molecule.modification=='unmodified','Exactly 50 nt and explicit unmodified RNA; author unmod_1 reporter head'),
      'mrnabert':('embedding_norm','L2 norm (representation diagnostic)',lambda c:len(c.regions.utr5)+len(c.regions.cds)//3+len(c.regions.utr3)+2<=1024,'Mixed nucleotide/codon tokenization, <=1024 tokens; representation, not efficacy'),
    }
    for name,(metric,unit,applicable,scope) in specs.items():
      if name not in brief.evaluators:continue
      eligible=[c for c in constructs if applicable(c)]
      for c in constructs:
        if c not in eligible:output[c.id].append(missing(metric,name,scope,'out_of_scope',unit))
      if not eligible:continue
      args={'sequences':[c.regions.utr5 for c in eligible]} if name=='utr_lm' else {'constructs':[c.card() for c in eligible],'species':brief.context.species}
      result=gateway.call(name,args)
      if result['status']!='ok':
        for c in eligible:output[c.id].append(missing(metric,name,result.get('reason','Tool failed'),result['status'],unit))
        continue
      data=result['data']
      rows=[{'id':c.id,'score':s} for c,s in zip(eligible,data['scores'])] if name=='utr_lm' else data['predictions']
      if len(rows)!=len(eligible) or {r['id'] for r in rows}!={c.id for c in eligible}:raise ValueError(name+' returned wrong candidate identities')
      for row in rows:
        c=cards[row['id']];extra={k:v for k,v in data.items() if k not in {'predictions','scores'}}
        value=row.get('score',row.get('embedding_norm'))
        if name=='rnadegformer':
          import numpy as np
          per_base=np.asarray(row['per_base'])
          if per_base.shape!=(len(c.sequence),5):raise ValueError('RNAdegformer per-base output shape mismatch')
          value=float(per_base[:,3].mean());extra.update(per_base=row['per_base'],channel_means=dict(zip(data['channels'],map(float,per_base.mean(0)))))
        if name=='degscore':extra['per_base']=row['per_base']
        if name=='mrnabert':extra.update(embedding=row['embedding'],tokens=row['tokens'],functional_task_head='not available; no functional score inferred')
        output[c.id].append(measured(metric,value,unit,name,manifest().get({'saluki':'basenji','rnadegformer':'RNAdegformer-Webapp','utr_lm':'UTR-LM','degscore':'DegScore'}.get(name,name),{}).get('revision','installed'),scope,
          prediction=name!='mrnabert',independent=name not in {'mrnabert','degscore'} and not (name=='utr_lm' and 'ribosome_load' in c.optimized_metrics),
          provenance={**result.get('provenance',{}),**extra,'raw_artifact':result.get('raw_artifact')},
          uncertainty={'ensemble_sd':row['ensemble_sd'],'calibrated':False} if 'ensemble_sd' in row else None))
    return output
