"""Small official reference records, explicitly distinguished from engineering labels."""
from pathlib import Path
import hashlib
import io
import json
import os
import requests
from Bio import SeqIO


def main():
    root=Path(os.environ.get('MRNA_DATA_DIR',str(Path.home()/'.cache/decoupled-rsi/mrna'))).expanduser()/'datasets';root.mkdir(parents=True,exist_ok=True)
    refs=[]
    url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_000518.5&rettype=gb&retmode=text'
    p=root/'NM_000518.5.gb'
    if not p.exists():
        r=requests.get(url,timeout=40);r.raise_for_status();p.write_text(r.text)
    rec=SeqIO.read(p,'genbank');f=next(x for x in rec.features if x.type=='CDS');a,b=int(f.location.start),int(f.location.end);s=str(rec.seq).replace('T','U')
    refs.append({'id':rec.id,'kind':'natural_transcript','source':url,'protein':str(f.extract(rec.seq).translate(to_stop=True)),'regions':{'utr5':s[:a],'cds':s[a:b],'utr3':s[b:]},'context':{'species':'human','cell':'endogenous erythroid lineage; IVT expression unmeasured','modification':'natural transcript','delivery':'endogenous','measurement':'sequence reference only'},'evidence':'reference sequence; not an engineered high-expression label','record_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for accession in ['P42212','P08659']:
        url='https://rest.uniprot.org/uniprotkb/'+accession+'.fasta';p=root/(accession+'.fasta')
        if not p.exists():
            r=requests.get(url,timeout=40);r.raise_for_status();p.write_text(r.text)
        rec=SeqIO.read(p,'fasta');refs.append({'id':accession,'kind':'protein','source':url,'protein':str(rec.seq),'description':rec.description,'record_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (root/'references.json').write_text(json.dumps(refs,indent=2))
    print(json.dumps([{'id':r['id'],'protein_length':len(r['protein'])} for r in refs]))


if __name__=='__main__':main()
