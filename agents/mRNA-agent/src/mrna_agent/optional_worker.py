"""Pinned author model adapters. Invoked only by the fixed scientific worker."""
import os,sys,json,random,ast
from pathlib import Path
from scientific_worker import ROOT,verified


def codongpt(a):
    import torch
    from transformers import GPT2LMHeadModel
    from Bio.Data import CodonTable
    torch.set_num_threads(4);torch.manual_seed(a['seed'])
    p=ROOT/'models/codongpt';verified(p/'pytorch_model.bin');sys.path.insert(0,str(p))
    from tokenizer import CodonTokenizer
    tok=CodonTokenizer();model=GPT2LMHeadModel.from_pretrained(p,local_files_only=True).eval()
    if len(a['protein'])+2>model.config.n_positions:raise ValueError('CDS exceeds codonGPT context; no truncation')
    table=CodonTable.unambiguous_dna_by_id[1];syn={}
    for codon,aa in table.forward_table.items():syn.setdefault(aa,[]).append(tok.codon2id[codon])
    syn['*']=[tok.codon2id[c] for c in table.stop_codons]
    seqs=[]
    with torch.inference_mode():
      for j in range(a['count']):
        ids=torch.tensor([[tok.bos_token_id]]);cache=None;out=[]
        for aa in a['protein']+'*':
          response=model(ids,past_key_values=cache,use_cache=True);cache=response.past_key_values
          allowed=syn[aa];probs=response.logits[0,-1,allowed].softmax(-1)
          k=allowed[torch.multinomial(probs,1).item()];out.append(tok.id2codon[k]);ids=torch.tensor([[k]])
        seqs.append({'sequence':''.join(out).replace('T','U')})
    return {'sequences':seqs,'sampler':'author codon vocabulary; exact standard-code synonym mask, ancestral sampling','device':'cpu'}


def utrgan(a):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
    import numpy as np
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.random.set_seed(a['seed'])
    model=tf.keras.models.load_model(verified(ROOT/'external/UTRGAN/models/checkpoint_3000.h5'),compile=False)
    latent=tf.random.normal([a['count'],int(model.input_shape[-1])])
    preds=model(latent,training=False).numpy()
    # Author generator uses A,C,G,T and '*' padding. Remove padding only.
    alphabet='ACGT*'
    seqs=[''.join(alphabet[i] for i in row.argmax(-1)).replace('*','').replace('T','U') for row in preds]
    return {'sequences':[{'sequence':s} for s in seqs], 'sampler':'pretrained WGAN prior; no predictor reward optimization','shape':list(preds.shape)}


def saluki(a):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
    import numpy as np
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0,str(ROOT/'external/basenji'))
    from basenji.rnann import RnaNN
    base=ROOT/'models/saluki';params=json.loads(verified(base/'f0_c0/train/params.json').read_text())['model']
    model=RnaNN(params);head=0 if a['species']=='human' else 1
    x=np.zeros((len(a['constructs']),12288,6),dtype=np.float32)
    for i,c in enumerate(a['constructs']):
      r=c['regions'];seq=r['utr5']+r['cds']+r['utr3']
      if len(seq)>12288:raise ValueError('Saluki length exceeded')
      for j,nt in enumerate(seq):x[i,j,'ACGU'.index(nt)]=1
      x[i,len(r['utr5']):len(r['utr5'])+len(r['cds']):3,4]=1
      # Synthetic IVT transcript: explicit zero splice-site channel, not inferred.
    predictions=[];members=[]
    for path in sorted(base.glob(f'f*_c*/train/model{head}_best.h5')):
      model.models[head].load_weights(verified(path))
      predictions.append(model.models[head](x,training=False).numpy().ravel());members.append(str(path.relative_to(base)))
    if len(predictions)!=50:raise ValueError('Saluki requires all 50 ensemble members')
    values=np.asarray(predictions)
    return {'predictions':[{'id':c['id'],'score':float(values[:,i].mean()),'ensemble_sd':float(values[:,i].std())} for i,c in enumerate(a['constructs'])], 'members':members,'splice_sites':'zero for synthetic IVT; endogenous training domain shift','unit':'normalized stability consensus (not hours)'}


def utr_lm(a):
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(4)
    p=ROOT/'external/UTR-LM';sys.path.insert(0,str(p/'Scripts'))
    import esm
    from esm.model.esm2_secondarystructure import ESM2
    alphabet=esm.Alphabet(standard_toks='AGCT',prepend_toks=('<pad>','<eos>','<unk>'),append_toks=('<cls>','<mask>','<sep>'),prepend_bos=True,append_eos=True,use_msa=False)
    # Extract only author's model class: importing training script would initialize DDP.
    source=p/'Scripts/UTRLM_downstream/MJ3_Finetune_extract_append_predictor_Sample_10fold-lr-huber-DDP.py'
    node=next(n for n in ast.parse(verified(source).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CNN_linear')
    args=SimpleNamespace(nodes=40,cnn_layers=0,dropout3=.5,modelfile='SISS',avg_emb=False,bos_emb=True,magic=False)
    ns={'nn':torch.nn,'torch':torch,'embed_dim':128,'inp_len':50,'args':args,'layers':6,'heads':16,'alphabet':alphabet,'ESM2_SISS':ESM2}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    model=ns['CNN_linear']();weight=next((p/'Model/Downstream/MRL').glob('*unmod_1*fold0*.pt'))
    state=torch.load(verified(weight),map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);model.eval()
    tokens=torch.tensor([[alphabet.cls_idx]+alphabet.encode(s.replace('U','T'))+[alphabet.eos_idx] for s in a['sequences']])
    with torch.inference_mode():scores=model(tokens,need_head_weights=False,return_contacts=False).ravel().tolist()
    return {'scores':scores,'checkpoint':str(weight.relative_to(ROOT)),'unit':'MRL','training_library':'GSM3130435_egfp_unmod_1','ensemble':'official fold0 downstream head'}


def rnadegformer(a):
    import torch,numpy as np,tempfile
    torch.set_num_threads(4)
    p=ROOT/'external/RNAdegformer-Webapp';sys.path[:0]=[str(p),str(ROOT/'external')]
    # The pinned author Arnie RNAsoft wrapper accepts T but never passes it to
    # simfold_pf. Both nominal 37/50 C requests use the binary default 37 C.
    # Reuse that identical BPP once; do not claim two temperature conditions.
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
      conf=Path(temp)/'arnie.conf';conf.write_text('TMP: '+temp+'\nrnasoft: '+str(p/'MultiRNAFold')+'\n')
      os.environ['ARNIEFILE']=str(conf)
      from arnie.bpps import bpps
      from RNA_Inference.Network import NucleicTransformer
      os.environ['ARNIEFILE']=str(conf)
      device='cuda' if torch.cuda.is_available() else 'cpu';models=[]
      for i in range(10):
        model=NucleicTransformer(15,5,256,32,1024,5,True,kmers=[5],return_aw=True)
        state=torch.load(verified(p/f'RNA_Inference/best_weights/fold{i}top1.ckpt'),map_location='cpu',weights_only=True)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);models.append(model.eval().to(device))
      rows=[]
      with torch.inference_mode():
       for c in a['constructs']:
        seq=c['sequence'];n=len(seq);idx=np.arange(n);d=np.abs(idx[:,None]-idx[None,:]);distance=np.stack([np.divide(1.,d.astype(float)**k,out=np.zeros_like(d,dtype=float),where=d!=0) for k in (1,2,3)])
        tokens=torch.tensor([['ACGU'.index(s) for s in seq]],device=device);pred=[]
        for temperature in (37,):
         bp=bpps(seq,package='rnasoft',T=temperature)
         features=torch.as_tensor(np.concatenate([bp[None],distance]),dtype=torch.float32,device=device)[None]
         for model in models:pred.append(model(tokens,features)[0][0].float().cpu().numpy())
        values=np.stack(pred);mean=values.mean(0)
        rows.append({'id':c['id'],'per_base':mean.tolist(),'ensemble_sd':values.std(0).mean(0).tolist()})
    return {'predictions':rows,'channels':['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C'],'bpp_engine':'author RNAsoft/BLstar default 37 C','requested_bpp_temperatures':[37,50],'effective_bpp_temperature':37,'upstream_caveat':'Pinned author Arnie wrapper ignores T; identical temperature requests deduplicated','ensemble':'all 10 folds; identical author temperature-feature duplicates collapsed','device':device}


def mrnabert(a):
    import torch,numpy as np
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    torch.set_num_threads(4)
    p=ROOT/'models/mrnabert';verified(p/'pytorch_model.bin')
    tokenizer=AutoTokenizer.from_pretrained(p,trust_remote_code=True,local_files_only=True)
    model=AutoModelForMaskedLM.from_pretrained(p,trust_remote_code=True,local_files_only=True).eval()
    # Native attention path is deterministic and supports CPU without Triton kernels.
    import sys
    for module in list(sys.modules.values()):
      if module and getattr(module,'__name__','').endswith('bert_layers') and hasattr(module,'flash_attn_qkvpacked_func'):module.flash_attn_qkvpacked_func=None
    rows=[]
    for c in a['constructs']:
      r=c['regions'];tokens=list(r['utr5'])+[r['cds'][i:i+3] for i in range(0,len(r['cds']),3)]+list(r['utr3'])
      text=' '.join(tokens).replace('U','T');inputs=tokenizer(text,return_tensors='pt',truncation=False)
      if inputs['input_ids'].shape[1]>1024:raise ValueError('mRNABERT 1024-token context exceeded; no truncation')
      with torch.inference_mode():output=model.bert(**inputs,return_dict=True)
      hidden=output.last_hidden_state if hasattr(output,'last_hidden_state') else output[0]
      pooled=hidden[0,1:-1].float().mean(0).numpy()
      rows.append({'id':c['id'],'embedding':pooled.tolist(),'embedding_norm':float(np.linalg.norm(pooled)),'tokens':inputs['input_ids'].shape[1]})
    return {'predictions':rows,'role':'representation only; no validated mammalian task head','pooling':'mean non-special tokens'}


def degscore(a):
    import RNA
    sys.path.insert(0,str(ROOT/'external/DegScore'))
    verified(ROOT/'external/DegScore/DegScore.py')
    from DegScore import DegScore
    rows=[]
    for c in a['constructs']:
      structure,_=RNA.fold(c['sequence']);model=DegScore(c['sequence'],structure=structure)
      rows.append({'id':c['id'],'score':float(model.degscore),'per_base':model.degscore_by_position.tolist()})
    return {'predictions':rows,'structure_engine':'ViennaRNA '+RNA.__version__,'unit':'sum of DegScore 2.1 per-base predictions','role':'chemical degradation control; no intracellular half-life inference'}
