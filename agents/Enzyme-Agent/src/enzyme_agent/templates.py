"""Curated route cards. No invented concentration, kinetics, exposure or thresholds."""
from .contracts import Brief, Environment, Gate, Reaction, Substrate

TEMPLATES = {
    "fabry": {
        "name": "Fabry · α-galactosidase A", "disease": "Fabry disease", "accession": "P06280",
        "endpoint": "GL-3 reduction in a disease-relevant lysosomal system",
        "reaction": {"name": "Terminal alpha-D-galactose hydrolysis", "ec": "3.2.1.22", "products": ["D-galactose", "degalactosylated substrate"], "direction": "hydrolysis", "product_fate": "Requires project-specific downstream evidence"},
        "panel": [{"id": "gl3", "name": "Globotriaosylceramide (GL-3)", "kind": "glycan", "role": "target"}, {"id": "normal_glycans", "name": "Normal tissue alpha-galactosyl glycans", "kind": "glycan", "role": "protect"}],
        "environment": {"tissue": "disease-relevant cells", "compartment": "lysosome", "dependencies": {"cellular_uptake": "unknown", "lysosomal_trafficking": "unknown", "glycosylation": "unknown"}},
        "source": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/103979s5313lbl.pdf",
        "routes": [{"route": "enzyme_replacement", "reason": "Restore deficient alpha-galactosidase A function; uptake and lysosomal trafficking are necessary", "selected": True}, {"route": "substrate_reduction", "reason": "A small-molecule intervention should be handed to drug-agent", "selected": False}],
        "unknowns": ["Mature processing and glycosylation occupancy", "Cellular uptake and lysosomal localization", "GL-3 turnover under relevant conditions; artificial reporter alone is insufficient"],
    },
    "pku": {
        "name": "PKU · PAL bypass", "disease": "Phenylketonuria", "accession": None,
        "endpoint": "Phenylalanine reduction within a prespecified physiological target range",
        "reaction": {"name": "Phenylalanine ammonia-lyase bypass", "ec": "4.3.1.24", "products": ["trans-cinnamate", "ammonia"], "direction": "L-phenylalanine elimination", "product_fate": "Assess ammonia and trans-cinnamate handling explicitly"},
        "panel": [{"id": "phe", "name": "L-phenylalanine", "chemical_id": "CHEBI:17295", "kind": "small_molecule", "role": "target"}, {"id": "tyr", "name": "L-tyrosine", "kind": "small_molecule", "role": "off_target"}, {"id": "normal_phe", "name": "Physiologically required phenylalanine range", "kind": "small_molecule", "role": "protect"}, {"id": "ammonia", "name": "Ammonia burden", "kind": "small_molecule", "role": "harmful_product"}],
        "environment": {"tissue": "systemic", "compartment": "extracellular", "dependencies": {"active_lifetime": "unknown", "product_clearance": "unknown"}},
        "source": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/761079s034lbl.pdf",
        "routes": [{"route": "PAL_bypass", "reason": "Alternative reaction reduces Phe without restoring PAH", "selected": True}, {"route": "PAH_replacement", "reason": "Distinct intracellular enzyme/cofactor/delivery requirements; retain as alternative", "selected": False}],
        "unknowns": ["Product clearance and physiological lower bound", "Cofactor/autocatalytic maturation state", "Relevant activity duration and immune evidence"],
    },
    "cf_dnase": {
        "name": "Cystic fibrosis · DNase I", "disease": "Cystic fibrosis airway disease", "accession": "P24855",
        "endpoint": "Reduction of sputum DNA burden and viscoelasticity in the relevant airway material",
        "reaction": {"name": "DNA phosphodiester hydrolysis", "ec": "3.1.21.1", "products": ["DNA oligonucleotides"], "direction": "hydrolysis", "product_fate": "Requires local fragment clearance assessment"},
        "panel": [{"id": "extracellular_dna", "name": "Extracellular sputum DNA", "kind": "nucleic_acid", "role": "target"}, {"id": "protected_dna", "name": "Normal cellular DNA (compartment protected)", "kind": "nucleic_acid", "role": "protect"}, {"id": "actin", "name": "Airway actin", "kind": "protein", "role": "competitor"}],
        "environment": {"tissue": "airway", "compartment": "sputum", "cofactors": ["calcium", "magnesium"], "dependencies": {"local_delivery": "unknown", "formulation_retention": "unknown", "ions": "unknown"}},
        "source": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=d8c78a7e-ff99-48f3-8952-643ec2ea0f86",
        "routes": [{"route": "local_material_degradation", "reason": "Recombinant DNase I hydrolyzes sputum DNA; material endpoints must be measured", "selected": True}, {"route": "systemic_delivery", "reason": "Not supported by this indication-specific route card", "selected": False}],
        "unknowns": ["Delivery/formulation activity retention", "Activity in sputum and physiological inhibition", "Viscoelasticity endpoint; small-molecule affinity is not applicable"],
    },
}


def template_brief(key):
    t = TEMPLATES[key]
    return Brief(name=t["name"], entry="disease", template=key, disease=t["disease"], endpoint=t["endpoint"],
                 route="research", generators=["library"], reaction=Reaction(**t["reaction"]),
                 panel=[Substrate(**s) for s in t["panel"]], environments=[Environment(**t["environment"])],
                 direction_gate=Gate(status="supported", rationale=t["routes"][0]["reason"], sources=[t["source"]]))


def mechanism_card(brief):
    t = TEMPLATES.get(brief.template, {})
    gaps = list(t.get("unknowns", []))
    for name in ("direction_gate", "product_gate", "access_gate"):
        gate = getattr(brief, name)
        if gate.status != "supported":
            gaps.append(f"{name}: {gate.status}; {gate.rationale}")
    if not brief.reference:
        gaps.append("Reference enzyme, mature chain and protection evidence are required before engineering")
    if not brief.panel:
        gaps.append("Define target, off-target and protected substrates")
    if brief.endpoint == "unresolved":
        gaps.append("Define a falsifiable disease-related endpoint")
    return {"disease": brief.disease, "endpoint": brief.endpoint, "falsifier": brief.falsifier,
            "routes": t.get("routes", [{"route": brief.route, "reason": "User-specified hypothesis; evidence review required", "selected": True}]),
            "sources": [t["source"]] if t else [], "gaps": gaps,
            "reaction": brief.reaction.model_dump(), "environments": [e.model_dump() for e in brief.environments],
            "gates": {n: getattr(brief, n).model_dump() for n in ("direction_gate", "product_gate", "access_gate")},
            "stage": "exploratory" if gaps else "computational"}
