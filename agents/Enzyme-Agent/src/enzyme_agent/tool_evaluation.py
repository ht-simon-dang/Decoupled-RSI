"""Mandatory, independently executed candidate evaluation and fail-closed gates."""
import math
import numpy as np
from scipy.spatial import cKDTree
from .config import digest
from .contracts import Brief
from .structure import chain_records, compare_structures


def structure_evaluation(args):
    brief = Brief.model_validate(args['brief'])
    sequence, pdb = args['sequence'], args['pdb']
    _, residues, observed = chain_records(pdb)
    if observed != sequence:
        raise ValueError('Evaluation structure does not match candidate sequence')
    if any(any(a not in r for a in ('N','CA','C','O')) for r in residues):
        raise ValueError('Independent structure has incomplete backbone atoms')
    atoms = [(i,a) for i,r in enumerate(residues) for a in r if a.element not in {'H','D'}]
    coords = np.array([a.coord for _,a in atoms])
    if not np.isfinite(coords).all():
        raise ValueError('Structure has nonfinite coordinates')
    clashes = [f'{atoms[a][0]+1}:{atoms[a][1].name}/{atoms[b][0]+1}:{atoms[b][1].name}'
               for a,b in sorted(cKDTree(coords).query_pairs(1.2)) if abs(atoms[a][0]-atoms[b][0])>1]
    ca = np.array([r['CA'].coord for r in residues])
    distances = np.linalg.norm(np.diff(ca,axis=0),axis=1)
    breaks = [i+1 for i,d in enumerate(distances) if not 2.7<=d<=4.5]
    geometry = {'status':'not_applicable','reason':'No matched reference structure'}
    comparable = brief.reference and brief.reference.pdb and brief.route in {'conservative','product'}
    if comparable:
        geometry = compare_structures(pdb, brief.reference, brief.reference.catalytic_positions)
    policy = brief.evaluation_policy
    checks = {'complete_backbone':True, 'continuous_chain':not breaks,
              'nonlocal_steric_clashes':len(clashes)<=policy.max_nonlocal_clashes}
    if comparable:
        checks['reference_mapping'] = geometry['status']=='ok'
        if geometry['status']=='ok':
            checks['backbone_preservation'] = geometry['backbone_rmsd_angstrom']<=policy.max_backbone_rmsd
            if brief.reference.catalytic_positions:
                checks['catalytic_backbone_preservation'] = geometry['catalytic_ca_rmsd_angstrom'] is not None and geometry['catalytic_ca_rmsd_angstrom']<=policy.max_catalytic_ca_rmsd
                checks['catalytic_sidechain_preservation'] = geometry['catalytic_sidechain_rmsd_angstrom'] is not None and geometry['catalytic_sidechain_rmsd_angstrom']<=policy.max_catalytic_sidechain_rmsd
                checks['catalytic_identity'] = all(sequence[p-1]==brief.reference.sequence[p-1] for p in brief.reference.catalytic_positions) if len(sequence)==len(brief.reference.sequence) else False
    return {'sequence_sha256':digest(sequence),'pdb_sha256':digest(pdb),'checks':checks,'passed':all(checks.values()),
        'geometry':geometry, 'nonlocal_clashes':clashes, 'chain_break_positions':breaks,
        'provenance':{'tool':'Biopython / SciPy structure audit','version':'2'},
        'scope':'Independent fold quality and reference-relative geometry; no substrate conversion or transition-state validation'}


def validation_record(candidate, brief, results):
    required = ['structure.esmfold2','evaluation.structure']
    # ThermoMPNN has a point-mutation domain. It cannot evaluate a different
    # family or a whole multi-mutant free energy by summing individual results.
    applicable = bool(candidate.mutations and brief.reference and brief.reference.pdb and brief.route in {'conservative','product'})
    if applicable:
        required.append('evaluation.thermompnn')
    failures, missing = [], []
    for name in required:
        result = results.get(name)
        if not result or result['status']!='ok':
            missing.append(name)
            continue
        data = result['data']
        if name=='structure.esmfold2':
            if data.get('sequence') != candidate.sequence:
                failures.append('Fold sequence mismatch')
            for key,threshold in [('mean_plddt_fraction',brief.evaluation_policy.min_plddt),('ptm',brief.evaluation_policy.min_ptm)]:
                value = data.get(key)
                if value is None or not isinstance(value,(int,float)) or not math.isfinite(value) or not threshold<=value<=1:
                    failures.append(f'{key} below declared threshold or missing')
            provenance=data.get('provenance',{})
            if provenance.get('template_conditioned') is not False or provenance.get('forced_contacts') is not False:
                failures.append('Fold independence not established')
        elif name=='evaluation.structure':
            if data.get('sequence_sha256')!=digest(candidate.sequence) or data.get('pdb_sha256')!=digest(candidate.pdb):
                failures.append('Structure evaluation identity mismatch')
            if not data.get('passed') or not data.get('checks') or not all(data['checks'].values()):
                failures.extend(k for k,v in data.get('checks',{}).items() if not v)
                if not data.get('checks'):failures.append('Structure checks missing')
        elif name=='evaluation.thermompnn':
            if data.get('sequence_sha256')!=digest(candidate.sequence):
                failures.append('Stability evaluation identity mismatch')
            rows = data.get('mutations',[])
            if {r['mutation'] for r in rows}!=set(candidate.mutations):
                failures.append('Incomplete point-mutation stability evaluation')
            for row in rows:
                value=row.get('ddg_kcal_mol')
                if value is None or not math.isfinite(value) or value>brief.evaluation_policy.max_single_mutation_ddg:
                    failures.append(f"ThermoMPNN point-mutation risk: {row['mutation']}")
    if candidate.rejection_reasons:failures.extend(candidate.rejection_reasons)
    return {'status':'incomplete' if missing else 'failed' if failures else 'passed',
        'required_tools':required, 'missing_tools':missing, 'failures':failures,
        'policy':brief.evaluation_policy.model_dump(),
        'stability_scope':'Individual substitutions in reference background; multi-mutant epistasis and net stability remain unknown' if applicable else 'Point-mutation comparison not applicable',
        'scope':'Computational tool validation only; enzyme activity requires matched experimental measurements'}
