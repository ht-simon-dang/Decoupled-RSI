"""Official mRNAutilus weights with torch SDPA attention and bounded masked sampling.

Same projections, non-interleaved RoPE and masking as the author's FlashAttention
path. Uses PyTorch kernels for Blackwell compatibility; no reward/MCTS reranking.
"""
import sys,types,ast,math
from pathlib import Path
import torch
from torch import nn
from scientific_worker import ROOT,verified


def load_model():
    path=ROOT/'models/mrnautilus';sys.path.insert(0,str(path))
    name='src.rinalmo.model.attention'
    source=path/'src/rinalmo/model/attention.py'
    tree=ast.parse(verified(source).read_text())
    tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or not (n.module or '').startswith('flash_attn')]
    module=types.ModuleType(name);module.__file__=str(source);sys.modules[name]=module
    exec(compile(tree,str(source),'exec'),module.__dict__)
    class NativeAttention(nn.Module):
        def __init__(self,embed_dim,num_heads,attention_dropout=0.,causal=False,use_rot_emb=True,bias=False):
            super().__init__();self.num_heads=num_heads;self.head_dim=embed_dim//num_heads;self.use_rot_emb=use_rot_emb
            self.Wqkv=nn.Linear(embed_dim,3*embed_dim,bias=bias);self.out_proj=nn.Linear(embed_dim,embed_dim,bias=bias)
        def forward(self,x,key_padding_mask=None,return_attn_probs=False):
            b,n,d=x.shape;q,k,v=self.Wqkv(x).view(b,n,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
            if self.use_rot_emb:
                inv=1.0/(10000.0**(torch.arange(0,self.head_dim,2,device=x.device,dtype=torch.float32)/self.head_dim))
                freq=torch.outer(torch.arange(n,device=x.device,dtype=torch.float32),inv);angle=torch.cat([freq,freq],-1)[None,None]
                cos,sin=angle.cos().to(q.dtype),angle.sin().to(q.dtype)
                def rotate(z):a,c=z.chunk(2,-1);return torch.cat([-c,a],-1)
                q,k=q*cos+rotate(q)*sin,k*cos+rotate(k)*sin
            mask=key_padding_mask[:,None,None,:] if key_padding_mask is not None else None
            y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
            return self.out_proj(y.transpose(1,2).reshape(b,n,d)),None
    module.FlashMultiHeadSelfAttention=NativeAttention
    from src.rinalmo.config import model_config
    from src.rinalmo.model.model import RiNALMo
    from src.rinalmo.data.alphabet import Alphabet_uni
    config=model_config('mega');config.model.transformer.use_flash_attn=True
    model=RiNALMo(config)
    checkpoint=torch.load(verified(path/'best.ckpt'),map_location='cpu',weights_only=False)
    prefix='model.net.rm.';state={k[len(prefix):]:v for k,v in checkpoint['state_dict'].items() if k.startswith(prefix)}
    model.load_state_dict(state,strict=True)
    del checkpoint,state
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return model.eval().to(device),Alphabet_uni(),device


def generate(args):
    from Bio.Data import CodonTable
    torch.set_num_threads(4);torch.manual_seed(args['seed'])
    model,tok,device=load_model();r=args['regions'];n5=len(r['utr5']);n3=len(r['utr3']);cds=r['cds'].replace('U','T')
    length=n5+len(cds)//3+n3+2
    if length>1024:raise ValueError('mRNAutilus application limit 1024 mixed tokens; no truncation')
    template=torch.tensor(tok.batch_tokenize([r['utr5']],[cds],[r['utr3']],max_len=length),device=device)
    count=args['count'];x=template.repeat(count,1);mask=torch.ones_like(x,dtype=torch.bool);mask[:,0]=False;mask[:,-1]=False
    locks=set(args.get('fixed_regions',[]))
    if 'utr5' in locks:mask[:,1:1+n5]=False
    if 'cds' in locks:mask[:,1+n5:1+n5+len(cds)//3]=False
    if 'utr3' in locks:mask[:,1+n5+len(cds)//3:-1]=False
    x[mask]=tok.mask_idx
    allowed=torch.zeros((length,len(tok)),dtype=torch.bool,device=device)
    for pos in range(1,length-1):
        if pos<=n5 or pos>=1+n5+len(cds)//3:choices=[tok.get_idx(c) for c in 'ACGT']
        else:choices=tok.neighbors_idx[template[0,pos].item()]
        allowed[pos,choices]=True
    allowed[0,tok.cls_idx]=True;allowed[-1,tok.eos_idx]=True
    # Linear time grid for absorbing-mask diffusion, sample only still-masked sites.
    with torch.inference_mode():
      for step in range(8):
        logits=model(x)['logits'].float().masked_fill(~allowed[None],-torch.inf)
        probs=logits.softmax(-1);sample=torch.multinomial(probs.reshape(-1,len(tok)),1).reshape(count,length)
        reveal=(torch.rand(x.shape,device=device)<1/(8-step)) & x.eq(tok.mask_idx)
        x=torch.where(reveal,sample,x)
    rows=[]
    for tokens in x.tolist():
        parts={'utr5':''.join(tok.get_tkn(i) for i in tokens[1:1+n5]),'cds':''.join(tok.get_tkn(i) for i in tokens[1+n5:1+n5+len(cds)//3]),'utr3':''.join(tok.get_tkn(i) for i in tokens[1+n5+len(cds)//3:-1])}
        rows.append({k:v.replace('T','U') for k,v in parts.items()})
    return {'constructs':rows,'sampler':'8-step absorbing-mask diffusion with exact synonymous CDS constraints; no MCTS/reward reranking','attention':'native torch SDPA, non-interleaved RoPE','device':device}
