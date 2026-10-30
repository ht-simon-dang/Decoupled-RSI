from .models import MODELS, download, model_root


def download_backend(name):
    if name in MODELS: return download(name)
    if name == "toxinpred3":
        from .toxicity import export_model
        return export_model()
    if name == "boltzgen":
        from huggingface_hub import hf_hub_download
        from .structural import BOLTZ_REV, MOLS_REV
        cache = model_root()/"boltzgen"
        for file in ("boltzgen1_diverse.ckpt", "boltz2_conf_final.ckpt"):
            hf_hub_download("boltzgen/boltzgen-1", file, revision=BOLTZ_REV, cache_dir=cache)
        hf_hub_download("boltzgen/inference-data", "mols.zip", repo_type="dataset", revision=MOLS_REV, cache_dir=cache)
        return str(cache)
    raise ValueError("Unknown backend")
