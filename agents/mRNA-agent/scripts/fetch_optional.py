"""Fetch official optional-model assets, pin revisions/hashes and enforce 50 GB quota."""
from pathlib import Path
import sys, json, hashlib, subprocess, requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mrna_agent.storage import data_root, quota, used_bytes
from fetch_assets import fetch
ROOT=data_root()
HF={'codongpt':('naniltx/codonGPT','ee7017c4bdd285206b87be2e65a28272ff4ac88e'), 'mrnabert':('YYLY66/mRNABERT','a1eb7df25804d23f08646e1cb996b234d7208a40'), 'mrnautilus':('atombio/mrnautilus','b04a676809be4b743842b207c743491cc607882b')}
def record():
 p=ROOT/'assets.json';d=json.loads(p.read_text())
 for r in (ROOT/'external').iterdir():
  if (r/'.git').exists():d[r.name]={'revision':subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip()}
 for name,(repo,rev) in HF.items():d[name]={'repo':repo,'revision':rev}
 for base in ['external/DegScore','external/UTRGAN','external/UTR-LM','external/RNAdegformer-Webapp','models/saluki','models/codongpt','models/mrnabert','models/mrnautilus']:
  for f in (ROOT/base).rglob('*'):
   if f.is_file() and '.git' not in f.parts:
    d[str(f.relative_to(ROOT))]={'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
 temporary=p.with_suffix('.new.json');temporary.write_text(json.dumps(d,indent=2));temporary.replace(p)
 print('Total bytes',used_bytes(),flush=True)
def hf():
 for name,(repo,rev) in HF.items():
  response=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true',timeout=30);response.raise_for_status()
  for item in response.json()['siblings']:
   f=item['rfilename'];fetch(f'https://huggingface.co/{repo}/resolve/{rev}/{f}',ROOT/'models'/name/f,item['size'],item.get('lfs',{}).get('sha256'))
 record()

def saluki():
 from remotezip import RemoteZip
 import re
 url='https://zenodo.org/records/6326409/files/datasets.zip?download=1'
 with RemoteZip(url) as z:
  rows=[i for i in z.infolist() if re.fullmatch(r'datasets/deeplearning/train_gru/f\d+_c\d+/train/(model[01]_best.h5|params.json)',i.filename)]
  assert len(rows)==150,len(rows)
  with quota(sum(i.file_size for i in rows)):
   for i in rows:
    p=ROOT/'models/saluki'/('/'.join(i.filename.split('/')[3:]));p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==i.file_size:continue
    p.write_bytes(z.read(i));print(str(p),flush=True)
 record()

if __name__=='__main__':
 hf()
 saluki()
 fetch('https://drive.usercontent.google.com/download?id=1mU37sSkEPNClR0OYBpLLF66g44TJEL3i&export=download&confirm=t',ROOT/'models/mrnautilus/best.ckpt',1778203267,'07e1b534008df579c58d65d5df6ec6c4dcf944d149133330049cbc054c1c4e29')
 record()
