"""Fetch the human DNase I reference with explicit precursor/mature mapping."""
import json
from pathlib import Path
from enzyme_agent.config import atomic_json, data_root
from enzyme_agent.contracts import Reference, Product, Brief
from enzyme_agent.retrieval import uniprot, pdb_record
from enzyme_agent.structure import chain_records, surface_positions
from enzyme_agent.templates import template_brief

record = uniprot("P24855")
protein = record["data"]
pdb = pdb_record("4AWN")
precursor = protein["sequence"]["value"]
sequence = precursor[22:282]
_, residues, observed = chain_records(pdb["pdb"], "A")
assert sequence == observed, "Reference structure and canonical mature sequence differ"
protected, catalytic = set(), []
for feature in protein["features"]:
    if feature["type"] in {"Active site", "Binding site", "Glycosylation", "Disulfide bond"}:
        start, end = feature["location"]["start"]["value"]-22, feature["location"]["end"]["value"]-22
        protected.update([start,end])
        if feature["type"] == "Active site": catalytic.append(start)
        if feature["type"] == "Glycosylation": protected.update(range(start,start+3))
reference = Reference(name="Human DNase I (P24855, 4AWN chain A)", sequence=sequence, accession="P24855",
    sources=[record["url"],pdb["source"]], pdb=pdb["pdb"], protected_positions=sorted(protected), catalytic_positions=catalytic,
    protection_sources=["UniProt P24855 field-level evidence in raw record "+record["sha256"]+"; some catalytic annotations are by similarity"],
    product=Product(name="Human DNase I mature research protein", form="local_delivery", precursor_sequence=precursor,
        mature_start=23,mature_end=282,processing=["Signal peptide 1-22 removed"], modifications=["N-glycosylation at mature N18; N106 potential", "Disulfides C101-C104 and C173-C209"],
        oligomer="monomer reference",targeting="airway-local delivery; uptake/exposure not established for variants",expression_host="unspecified",
        auxiliary_components=["calcium", "magnesium"]))
atomic_json(data_root()/"references/cf_dnase_reference.json",reference.model_dump())
values=template_brief("cf_dnase").model_dump()
values.update(name="DNase I · constrained two-generator design",entry="enzyme",route="conservative",reference=reference.model_dump(),
    generators=["ligandmpnn","esm3"],design_positions=surface_positions(reference,8),
    design_intent="Explore solvent-accessible residues away from annotated protected sites; independently assess folding. Not an activity improvement claim.")
values["budget"].update(candidates=4,fold_candidates=3,shortlist=4)
brief=Brief.model_validate(values)
examples=Path(__file__).resolve().parents[1]/"examples"
examples.mkdir(parents=True,exist_ok=True)
(examples/"dnase_design.json").write_text(json.dumps(brief.model_dump(),ensure_ascii=False,indent=2))
values.update(name="DNase I · reference analysis",generators=["library"])
values["budget"].update(fold_candidates=0)
(examples/"dnase_reference.json").write_text(json.dumps(values,ensure_ascii=False,indent=2))
print(json.dumps({"length":len(sequence),"protected_positions":reference.protected_positions,"design_positions":brief.design_positions,"uniprot_sha256":record["sha256"]},indent=2))
