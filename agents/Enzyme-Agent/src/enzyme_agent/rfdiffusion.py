"""Foundry RFdiffusion3 constrained all-atom redesign of an existing enzyme."""
import gzip
import io
import json
import os
import subprocess
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO, Select
from .config import data_root, digest, check_space
from .contracts import Unavailable
from .models import model_path
from .structure import validated_mapping, chain_records


def generate_rfd3(brief, count, work):
    if brief.route not in {'conservative','product'}:
        raise ValueError('This RFdiffusion3 adapter supports constrained reference redesign. De novo theozyme scaffolding needs a separate atom-mapped mechanism specification; it is not inferred from prose.')
    if not brief.reference or not brief.reference.pdb or not brief.design_positions:
        raise ValueError('RFdiffusion3 requires a complete reference structure and explicit editable positions')
    base = model_path('rfdiffusion3')
    runtime = data_root()/'foundry-env/bin/rfd3'
    checkpoint = base/'rfd3_latest.ckpt'
    if not runtime.is_file() or not checkpoint.is_file():
        raise Unavailable('Install the pinned Foundry Python 3.12 runtime and RFdiffusion3 checkpoint')
    manifest = json.loads((base/'manifest.json').read_text())
    structure,residues,labels = validated_mapping(brief.reference)
    editable=set(brief.design_positions)
    selected_ligands=set(brief.rfd3_ligands)
    found_ligands={r.resname for r in structure[0].get_residues() if r.id[0].startswith('H_')}
    if selected_ligands-found_ligands:
        raise ValueError('Specified RFdiffusion3 ligand is absent from reference structure')
    class Context(Select):
        def accept_residue(self,residue):
            return (residue.get_parent().id==brief.reference.chain and residue in residues) or residue.resname in selected_ligands
    writer=PDBIO();writer.set_structure(structure)
    writer.save(str(work/'reference.pdb'),Context())
    # Explicit lists also support non-contiguous original residue numbering.
    spec={'input':str(work/'reference.pdb'),'contig':','.join(labels),
        'select_unfixed_sequence':','.join(labels[p-1] for p in sorted(editable)),
        'select_fixed_atoms':{label:'ALL' for i,label in enumerate(labels,1) if i not in editable},
        'allow_ligand_on_existing_chain':True}
    if selected_ligands:
        spec['ligand']=','.join(sorted(selected_ligands))
        spec['select_fixed_atoms'].update({name:'ALL' for name in selected_ligands})
    (work/'design.json').write_text(json.dumps({'enzyme':spec}))
    out=work/'rfd3-output'
    check_space(300_000_000)
    env=dict(os.environ,FOUNDRY_CHECKPOINT_DIRS=str(base),WANDB_MODE='disabled',
        XDG_CACHE_HOME=str(data_root()/'cache'),TORCHINDUCTOR_CACHE_DIR=str(data_root()/'cache/inductor'),
        RFD3_COMPILE_CACHE_DIR=str(data_root()/'cache/rfd3'),PYTHONHASHSEED=str(brief.seed))
    command=[str(runtime),'design',f'inputs={work/"design.json"}',f'out_dir={out}',f'ckpt_path={checkpoint}',
        f'seed={brief.seed}',f'diffusion_batch_size=1',f'n_batches={count}',f'inference_sampler.num_timesteps={brief.rfd3_steps}',
        'inference_mode=legacy','prevalidate_inputs=True','skip_existing=False','dump_trajectories=False']
    log=work/'rfd3.log'
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=base/'source',timeout=1700)
    if result.returncode:
        raise RuntimeError('RFdiffusion3 failed: '+log.read_text(errors='replace')[-5000:])
    paths=sorted(out.glob('*.cif'))+sorted(out.glob('*.cif.gz'))
    if len(paths)!=count:
        raise RuntimeError(f'RFdiffusion3 output count mismatch: expected {count}, found {len(paths)}; '+log.read_text(errors='replace')[-2000:])
    sequences,details=[],[]
    for path in paths:
        text=gzip.open(path,'rt').read() if path.suffix=='.gz' else path.read_text()
        output_structure=MMCIFParser(QUIET=True).get_structure('generated',io.StringIO(text))
        writer=PDBIO();writer.set_structure(output_structure);buffer=io.StringIO();writer.save(buffer)
        pdb=buffer.getvalue()
        _,_,seq=chain_records(pdb,'A')
        if len(seq)!=len(brief.reference.sequence):raise ValueError('RFdiffusion3 changed the constrained chain length')
        if any(a!=b and i not in editable for i,(a,b) in enumerate(zip(brief.reference.sequence,seq),1)):
            raise ValueError('RFdiffusion3 changed a fixed sequence position')
        sequences.append(seq)
        details.append({'model':'RFdiffusion3','repo':manifest['repo'],'revision':manifest['revision'],
            'checkpoint_sha256':manifest['checkpoint']['sha256'],'seed':brief.seed,'steps':brief.rfd3_steps,
            'specification':{k:v for k,v in spec.items() if k!='input'},'input_pdb_sha256':digest(brief.reference.pdb),
            'design_pdb':pdb,'design_pdb_is_independent_evaluation':False,
            'scope':'All-atom constrained reference redesign; sequence from RFdiffusion3 sequence head; function unverified'})
    return {'sequences':sequences,'per_sequence_provenance':details,
        'provenance':{'model':'RFdiffusion3','revision':manifest['revision'],'seed':brief.seed,'steps':brief.rfd3_steps},
        'worker_log_tail':log.read_text(errors='replace')[-2000:]}
