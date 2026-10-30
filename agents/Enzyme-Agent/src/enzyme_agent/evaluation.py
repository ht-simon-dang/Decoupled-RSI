"""Sealed evaluation rules: identities, evidence domains, Pareto and diversity.

Generator likelihood is never a functional endpoint; missing evidence stays missing.
"""
from __future__ import annotations
import math
from difflib import SequenceMatcher
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from .config import digest
from .contracts import Brief, Candidate, Evidence, Product

EVALUATOR_VERSION = "enzyme-evaluator-2-mandatory-tools"
OBJECTIVES = {"mutation_fraction": "min", "negative_plddt": "min"}


def product_id(product):
    return digest(product.model_dump() if hasattr(product,"model_dump") else product)[:24]


def make_candidate(sequence, origin, brief, provenance, parent_id=None):
    parent = brief.reference
    product = parent.product.model_copy(deep=True) if parent else Product()
    if origin in {"zymctrl","natural_database"}:
        product = Product(name="Exploratory enzyme; product configuration unresolved",processing=[provenance.get('processing','unknown')])
    if product.precursor_sequence and product.mature_start is not None:
        product.precursor_sequence = product.precursor_sequence[:product.mature_start-1]+sequence+product.precursor_sequence[product.mature_end:]
        product.mature_end = product.mature_start+len(sequence)-1
    identity = digest({"sequence":sequence,"product":product.model_dump()})[:24]
    mutations = []
    if parent and len(sequence)==len(parent.sequence):
        mutations=[f"{a}{i}{b}" for i,(a,b) in enumerate(zip(parent.sequence,sequence),1) if a!=b]
    return Candidate(id=identity, sequence=sequence, parent_id=parent_id, product=product,origin=origin,
                     provenance=provenance,design_intent=brief.design_intent,mutations=mutations)


def descriptors(sequence, ph=None):
    pa=ProteinAnalysis(sequence)
    return {"length":len(sequence),"molecular_weight_da":float(pa.molecular_weight()),"isoelectric_point":float(pa.isoelectric_point()),
            "gravy":float(pa.gravy()),"instability_index":float(pa.instability_index()),
            "charge_at_ph":float(pa.charge_at_pH(ph)) if ph is not None else None,
            "n_glycosylation_sequons":[i+1 for i in range(len(sequence)-2) if sequence[i]=='N' and sequence[i+1]!='P' and sequence[i+2] in 'ST']}


def assess(candidate: Candidate, brief: Brief):
    if brief.reference and candidate.origin not in {"reference","zymctrl","natural_database"}:
        parent=brief.reference.sequence
        if len(candidate.sequence)!=len(parent):
            candidate.rejection_reasons.append("Conservative engineering changed mature-chain length")
        else:
            for p in set(brief.reference.protected_positions+brief.reference.catalytic_positions):
                if candidate.sequence[p-1]!=parent[p-1]:
                    candidate.rejection_reasons.append(f"Protected residue {p} changed")
            if brief.design_positions and any(a!=b and i not in brief.design_positions for i,(a,b) in enumerate(zip(parent,candidate.sequence),1)):
                candidate.rejection_reasons.append("Generator changed a fixed position")
            if len(candidate.mutations)>brief.max_mutations:
                candidate.rejection_reasons.append("Mutation count exceeds project limit")
    ph=brief.environments[0].ph
    stats=descriptors(candidate.sequence,ph)
    for endpoint,key,unit in [("molecular_weight","molecular_weight_da","Da"),("isoelectric_point","isoelectric_point","dimensionless"),("gravy","gravy","Kyte-Doolittle hydropathy"),("instability_index","instability_index","index")]:
        candidate.evidence.append(Evidence(endpoint=endpoint,status="inferred",value=stats[key],unit=unit,
            source="Biopython ProtParam",version=EVALUATOR_VERSION,conditions={"sequence_state":"unmodified mature monomer"},
            uncertainty="Deterministic descriptor; no therapeutic calibration",applicability="Sequence descriptor; not activity, half-life or expression"))
    candidate.concerns += ["Glycosylation sequons do not establish modification occupancy", "Immunogenicity and effective lifetime require context-specific evidence"]
    if not brief.reference or not brief.reference.protection_sources:
        candidate.concerns.append("Protected-region evidence incomplete")
    for env in brief.environments:
        for substrate in brief.panel:
            candidate.evidence.append(Evidence(endpoint="substrate_conversion",status="not_assessed",source="project substrate panel",version=EVALUATOR_VERSION,
                substrate_id=substrate.id,environment_id=env.id,conditions={"role":substrate.role,"kind":substrate.kind,"compartment":env.compartment},
                applicability="Requires matched biochemical or material assay"))
            candidate.evidence.append(Evidence(endpoint="small_molecule_kinetics_prior",status="not_assessed" if substrate.kind=="small_molecule" else "not_applicable",
                source="domain guard",version=EVALUATOR_VERSION,substrate_id=substrate.id,environment_id=env.id,
                applicability="Small-molecule regressor requires established enzyme/substrate compatibility; no regression was called"))
    for endpoint in ("catalytic_turnover","off_target_protection","product_handling","localization","processing_occupancy","immune_response","disease_endpoint"):
        candidate.evidence.append(Evidence(endpoint=endpoint,status="not_assessed",source="project evidence gap",version=EVALUATOR_VERSION))
    if candidate.rejection_reasons:
        candidate.state="rejected"
    return stats


def distance(a,b):
    if len(a)==len(b):
        return sum(x!=y for x,y in zip(a,b))/len(a)
    return 1-SequenceMatcher(None,a,b,autojunk=False).ratio()


def pareto_layers(rows, objectives):
    remaining=list(rows)
    layers=[]
    while remaining:
        front=[]
        for a in remaining:
            dominated=False
            for b in remaining:
                if a is b: continue
                av=[a[k] for k in objectives]; bv=[b[k] for k in objectives]
                if all(y<=x for x,y in zip(av,bv)) and any(y<x for x,y in zip(av,bv)):
                    dominated=True;break
            if not dominated: front.append(a)
        if not front:
            raise ValueError("Non-finite Pareto objectives")
        layers.append(front)
        remaining=[r for r in remaining if r not in front]
    return layers


def portfolio(candidates, stats, brief):
    valid=[c for c in candidates if c.state!='rejected' and c.evaluation.get('status')=='passed']
    rows=[]
    for c in valid:
        s=stats[c.id]
        mutation=distance(c.sequence,brief.reference.sequence) if brief.reference else 0.0
        confidence=next(e.value for e in c.evidence if e.endpoint=='mean_plddt' and e.status=='predicted')
        rows.append({"id":c.id,"mutation_fraction":mutation,"negative_plddt":-confidence})
    # All candidates use identical descriptor scales. Unknown structure/kinetic
    # scores are not replaced by zero or compared with available scores.
    layers=pareto_layers(rows,OBJECTIVES)
    ranks={r['id']:i+1 for i,front in enumerate(layers) for r in front}
    byid={c.id:c for c in valid}
    selected=[]
    limit=min(brief.budget.shortlist,brief.budget.experiment_slots)
    ordered=sorted(valid,key=lambda c:(ranks[c.id],c.id))
    # Keep a reference comparator, then favor distinct sequences within each front.
    reference=next((c for c in ordered if c.origin=='reference'),None)
    if reference and limit:
        selected.append(reference.id)
    for front in layers:
        choices=[r['id'] for r in front if r['id'] not in selected]
        while choices and len(selected)<limit:
            chosen=max(choices,key=lambda i:min((distance(byid[i].sequence,byid[j].sequence) for j in selected),default=1))
            selected.append(chosen);choices.remove(chosen)
    result=[]
    for i,cid in enumerate(selected):
        c=byid[cid]
        c.state='computational_priority' if all(getattr(brief,g).status=='supported' for g in ('direction_gate','product_gate','access_gate')) else 'exploratory'
        role='reference' if c.origin=='reference' else ('discrimination' if i==1 else 'exploration' if i%2==0 else 'performance_proxy')
        result.append({"candidate_id":cid,"role":role,"pareto_front":ranks[cid],
            "reason":"Evaluated reference comparator" if role=='reference' else f"Passed required evaluation tools; Pareto front {ranks[cid]}; unresolved function requires measurement",
            "next_measurement":next_measurement(brief.bottleneck),"evidence_gaps":c.concerns})
    return {"selected":result,"pareto_fronts":[[r['id'] for r in f] for f in layers],"objectives":OBJECTIVES,
            "scope":"Only candidates passing required computational tools; independent fold confidence and sequence diversity; catalytic activity unverified",
            "excluded":[{'candidate_id':c.id,'evaluation':c.evaluation} for c in candidates if c.evaluation.get('status')!='passed'],
            "normalization":"None: Pareto comparison uses each raw objective separately; no weighted total or success probability",
            "evaluator_version":EVALUATOR_VERSION,"evaluator_hash":evaluator_hash()}


def next_measurement(bottleneck):
    return {"expression":"Measure usable expression in the declared host before assigning activity negatives",
        "processing":"Verify mature processing, modifications and oligomer assembly",
        "localization":"Measure access to the substrate compartment before redesigning the catalytic pocket",
        "selectivity":"Measure target and protected/off-target substrates under matched conditions",
        "mechanism":"Resolve reaction participants, catalytic roles and falsifiable mechanism",
        "catalysis":"Measure substrate turnover with product accounting and reference control",
        "stability":"Measure retained activity over time in the target environment, with processing controls"}[bottleneck]


def evaluator_hash():
    return digest(evaluator_sources())


def evaluator_sources():
    from pathlib import Path
    root=Path(__file__).parent
    return {name:(root/name).read_text() for name in ('contracts.py','evaluation.py','feedback.py','benchmark.py','structure.py','tool_evaluation.py','stability.py','folding.py','workflow.py')}


def measured_scenario(kcat_s, km_molar, enzyme_molar, substrate_molar):
    """Conditional Michaelis-Menten scenario, not an efficacy or exposure model."""
    values=[kcat_s,km_molar,enzyme_molar,substrate_molar]
    if any(not math.isfinite(v) or v<0 for v in values) or km_molar<=0:
        raise ValueError("Require finite nonnegative values and positive Km")
    return {"rate_molar_per_second":kcat_s*enzyme_molar*substrate_molar/(km_molar+substrate_molar),
            "assumptions":["Single-substrate steady-state Michaelis-Menten", "Constant active enzyme", "No transport, inhibition, substrate production or clearance model"],
            "scope":"User-parameterized scenario; not a patient efficacy prediction"}
