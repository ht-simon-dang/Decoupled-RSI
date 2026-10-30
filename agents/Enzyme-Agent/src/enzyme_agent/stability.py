"""Real ThermoMPNN inference on point mutations in the reference background."""
import json
import sys
from types import SimpleNamespace
from .config import digest
from .contracts import Brief
from .models import model_path
from .structure import validated_mapping


def thermompnn(args, work):
    import torch
    from omegaconf import OmegaConf
    brief = Brief.model_validate(args['brief'])
    reference = brief.reference
    validated_mapping(reference)
    sequence = args['sequence']
    if len(sequence)!=len(reference.sequence):
        raise ValueError('ThermoMPNN requires a substitution-only reference mapping')
    changes = [(i,a,b) for i,(a,b) in enumerate(zip(reference.sequence,sequence)) if a!=b]
    if not changes:
        raise ValueError('No substitutions to evaluate; reference is the comparator')
    root = model_path('thermompnn')
    source = root/'source'
    manifest = json.loads((root/'manifest.json').read_text())
    sys.path.insert(0,str(source))
    from transfer_model import TransferModel
    from protein_mpnn_utils import alt_parse_PDB
    config = OmegaConf.create({'platform':{'thermompnn_dir':str(source)}, 'model':{
        'hidden_dims':[64,32], 'subtract_mut':True, 'num_final_layers':2,
        'freeze_weights':True, 'load_pretrained':True, 'lightattn':True}})
    torch.manual_seed(brief.seed)
    model = TransferModel(config)
    checkpoint = torch.load(source/'models/thermoMPNN_default.pt',map_location='cpu',weights_only=False)
    state = checkpoint.get('state_dict',checkpoint)
    # Lightning owns the training prefix; inference uses its exact TransferModel.
    state = {k.removeprefix('model.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    pdb = work/'reference.pdb';pdb.write_text(reference.pdb)
    parsed = alt_parse_PDB(str(pdb), reference.chain)
    if parsed[0]['seq']!=reference.sequence:
        raise ValueError('ThermoMPNN parser/reference residue map mismatch')
    mutations = [SimpleNamespace(position=i,wildtype=a,mutation=b) for i,a,b in changes]
    with torch.inference_mode():
        predictions,_ = model(parsed,mutations)
    rows = [{'mutation':f'{a}{i+1}{b}','mature_position':i+1,'ddg_kcal_mol':float(out['ddG'].item())}
            for (i,a,b),out in zip(changes,predictions)]
    return {'sequence_sha256':digest(sequence),'reference_sequence_sha256':digest(reference.sequence),
        'mutations':rows, 'net_multi_mutant_ddg':None,
        'sign_convention':'Positive ddG is destabilizing; negative is stabilizing',
        'scope':'Each substitution predicted separately in the wild-type background; no additive multi-mutant or thermostability claim',
        'provenance':{'model':'ThermoMPNN','revision':manifest['revision'],
            'weights_sha256':manifest['source_files']['models/thermoMPNN_default.pt']['sha256'], 'seed':brief.seed}}
