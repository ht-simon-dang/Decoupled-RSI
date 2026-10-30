"""Isolated offline ESMFold2 preview; stdout is a single JSON result."""
import contextlib
import json
import sys


def main():
    args = json.loads(sys.stdin.read())
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model
        from transformers.models.esmfold2.protein_utils import prepare_protein_features, output_to_pdb
        torch.set_num_threads(4)
        torch.manual_seed(args["seed"])
        model = EsmFold2Model.from_pretrained(args["model_path"], local_files_only=True,
            trust_remote_code=False, dtype=torch.float32).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            output = model.infer_protein(args["sequence"], num_loops=args["loops"],
                num_sampling_steps=args["steps"], num_diffusion_samples=1)
        features = prepare_protein_features(args["sequence"], device=model.device)
        pdb = output_to_pdb(output, features)
        values = output.plddt[0].detach().float().cpu().tolist()
        result = {"pdb": pdb, "sequence": args["sequence"], "plddt_fraction": values,
                  "mean_plddt_fraction": sum(values) / len(values), "ptm": float(output.ptm[0])}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
