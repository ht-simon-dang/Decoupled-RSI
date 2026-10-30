"""Independent sequence-only folding; no design template or forced contacts."""
def fold(args, work):
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features, output_to_pdb
    from .models import model_path, MODELS
    from .structure import chain_records
    torch.manual_seed(args["seed"])
    model = EsmFold2Model.from_pretrained(model_path("esmfold2"), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    sequence = args["sequence"]
    with torch.inference_mode():
        output = model.infer_protein(sequence, num_loops=args["loops"], num_sampling_steps=args["steps"], num_diffusion_samples=1)
    features = prepare_protein_features(sequence, device=model.device)
    pdb = output_to_pdb(output, features)
    _, _, observed = chain_records(pdb)
    if observed != sequence:
        raise ValueError("Independent folding output sequence identity mismatch")
    values = output.plddt[0].detach().float().cpu().tolist()
    # Upstream ESMFold2 confidence is a fraction, not the 0..100 convention.
    return {"pdb": pdb, "sequence": sequence, "plddt_fraction": values,
            "mean_plddt_fraction": sum(values)/len(values), "ptm": float(output.ptm[0]),
            "provenance": {**MODELS["esmfold2"], "seed": args["seed"], "sampling_steps": args["steps"], "loops": args["loops"],
                "template_conditioned": False, "forced_contacts": False, "scope": "Single unmodified protein chain; oligomer/ligand/catalysis not assessed"}}
