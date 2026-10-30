"""Fixed JSON-in/JSON-out scientific subprocesses, also usable in Python 3.10.

No package imports: the official GEMORNA extension requires Python 3.10. Models
are loaded only from pinned administrator-installed assets, never user pickle files.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import hashlib

ROOT = Path(os.environ.get("MRNA_DATA_DIR", str(Path.home() / ".cache/decoupled-rsi/mrna"))).expanduser()


def verified(path):
    path = Path(path)
    records = json.loads((ROOT / "assets.json").read_text())
    expected = records.get(str(path.relative_to(ROOT)), {}).get("sha256")
    if not expected:
        raise ValueError("Asset is not registered in pinned manifest: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):h.update(block)
    if h.hexdigest() != expected:
        raise ValueError("Asset checksum mismatch: " + str(path))
    return path


def gemorna(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    path = ROOT / "external/GEMORNA"
    os.chdir(path)
    sys.path.insert(0, str(path / "src"))
    import generate
    from types import SimpleNamespace
    output = []
    for i in range(args["count"]):
        seed = args["seed"] + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), torch.inference_mode():
            generate.main(SimpleNamespace(mode=args["region"], ckpt_path=str(path / "checkpoints" / ("gemorna_" + args["region"] + ".pt")), protein_seq=args.get("protein"), utr_length=args.get("utr_length", "short")))
        raw = stream.getvalue()
        sequences = re.findall(r"(?<![A-Za-z])[ACGUT]{6,}(?![A-Za-z])", raw.upper())
        if not sequences:
            raise ValueError("GEMORNA returned no canonical sequence: " + raw[-1000:])
        output.append({"sequence": max(sequences, key=len).replace("T", "U"), "seed": seed, "raw": raw})
    return {"sequences": output, "device": "cpu", "upstream_sampler": True}


def ribonn(args):
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(4)
    path = ROOT / "external/RiboNN"
    sys.path.insert(0, str(path))
    from src.data import DataFrameDataset
    from src.model import RiboNN
    from src.utils.helpers import extract_config
    from src import predict
    import inspect
    # Target order is the author's order, read directly from the pinned source.
    source = inspect.getsource(predict.predict_using_models_trained_in_one_fold)
    choices = re.findall(r'training_data_columns = "([^"]+)"', source)
    species = args["species"]
    labels = choices[0 if species == "human" else 1].split(",")
    weights = ROOT / "models/ribonn" / species
    runs = pd.read_csv(weights / "runs.csv")
    selected = runs.sort_values("metrics.val_r2", ascending=False).groupby("params.test_fold", sort=True).head(1)
    config = extract_config(runs, selected.iloc[0].run_id)
    records = []
    for item in args["constructs"]:
        r = item["regions"]
        records.append({"tx_id": item["id"], "tx_sequence": (r["utr5"] + r["cds"] + r["utr3"]).replace("U", "T"),
                        "utr5_size": len(r["utr5"]), "cds_size": len(r["cds"]), "tx_size": sum(len(r[k]) for k in ("utr5", "cds", "utr3"))})
    kwargs = {k: config[k] for k in ("pad_5_prime", "split_utr5_cds_utr3_channels", "label_codons", "label_3rd_nt_of_codons", "label_utr5", "label_utr3", "label_splice_sites", "label_up_probs")}
    ds = DataFrameDataset(pd.DataFrame(records), target_column_pattern=None, max_utr5_len=1381, max_cds_utr3_len=11937, max_tx_len=13318, **kwargs)
    batch = torch.stack([ds[i] if isinstance(ds[i], torch.Tensor) else ds[i][0] for i in range(len(ds))])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for row in selected.itertuples():
        cfg = extract_config(runs, row.run_id)
        model = RiboNN(**cfg)
        model.load_state_dict(torch.load(verified(weights / row.run_id / "state_dict.pth"), map_location="cpu", weights_only=True))
        model.eval().to(device)
        with torch.inference_mode():
            results.append(model(batch.to(device)).float().cpu().numpy())
        del model
    values = np.stack(results)
    mean = values.mean(axis=0)
    return {"predictions": [{"id": r["tx_id"], "scores": dict(zip(labels, map(float, mean[i]))), "ensemble_sd": dict(zip(labels, map(float, values[:, i].std(axis=0))))} for i, r in enumerate(records)],
            "model_ids": selected.run_id.tolist(), "ensemble": "top validation model per fold; reduced author ensemble, not full 5-per-fold", "device": device}


def parade(args):
    import pandas as pd
    import numpy as np
    import torch
    torch.set_num_threads(4)
    path = ROOT / "external/parade/predictor/model"
    sys.path.insert(0, str(path))
    from predict import create_dataset_activity, MODEL_PATH_UTR5, MODEL_PATH_UTR3
    from pl_regressor import RNARegressor
    from utrdata_cl import UTRData
    region = args["region"]
    codes = args["cells"]
    frame = create_dataset_activity([s.replace("U", "T") for s in args["sequences"]], codes)
    ds = UTRData(df=frame, predict_cols=[], construct_type=region, features=("sequence", "positional", "conditions"), augment=False, augment_test_time=False, augment_kws=dict(extend_left=0, extend_right=0, shift_left=0, shift_right=0, revcomp=False))
    model = RNARegressor.load_from_checkpoint(verified(MODEL_PATH_UTR5 if region == "utr5" else MODEL_PATH_UTR3), weights_only=False, map_location="cpu").eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0)
    output = []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            pred, _ = model.predict_step(batch, i)
            output.extend(pred[:, 1].tolist())
    frame["score"] = output
    return {"predictions": [{"index": i, "scores": dict(zip(g.cell_type, map(float, g.score)))} for i, g in frame.groupby("id", sort=True)], "unit": "reporter mass-center score", "device": "cpu"}


def smart(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import numpy as np
    import tensorflow as tf
    import joblib
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    path = ROOT / "external/Smart5UTR"
    sys.path.insert(0, str(path))
    from Smart5UTR.dataloader import onehot_singleseq
    kind = args["kind"]
    model_path = path / ("models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5" if kind == "smart5utr" else "models/Optimus5-Prime/Optimus5_egfp_m1pseudo2_model_200k.h5")
    model = tf.keras.models.load_model(verified(model_path), compile=False)
    scaler = joblib.load(verified(path / ("models/egfp_m1pseudo2.scaler" if kind == "smart5utr" else "models/Optimus5-Prime/Optimus5_egfp_m1pseudo2_200k.scaler")))
    sequences = args["sequences"]
    if any(len(s) != 50 or set(s.upper()) - set("ACGTU") for s in sequences):
        raise ValueError("Smart5UTR/Optimus5 requires exactly 50 canonical nt; trimming/padding is disabled")
    if args.get("generate"):
        from Smart5UTR.model import split_autoencoder
        from Smart5UTR.dataloader import binary_mtx, decode_seq
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        tf.random.set_seed(args["seed"])
        encoder, decoder, width = split_autoencoder(model)
        latent = encoder(onehot_singleseq(sequences[0].replace("U", "T")), training=False).numpy()
        out = []
        # Bounded author-style latent perturbation, with all proposals retained for independent evaluation.
        for _ in range(args["count"]):
            noise = np.append(1.2 + np.random.normal(0, 1, (1, width - 20)) / 4, [1.2] * 20)
            decoded = decoder(latent * noise, training=False).numpy()
            out.append(decode_seq(binary_mtx(decoded)[0]).upper().replace("T", "U"))
        sequences = out
    x = np.concatenate([onehot_singleseq(s.replace("U", "T")) for s in sequences])
    values = model(x, training=False)
    if kind == "smart5utr":
        values = values[1]
    scores = scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel().tolist()
    return {"sequences": sequences, "scores": scores, "sampler": "bounded latent perturbations" if args.get("generate") else None, "unit": "MRL"}


if __name__ == "__main__":
    request = json.load(sys.stdin)
    from optional_worker import codongpt,utrgan,saluki,utr_lm,rnadegformer,mrnabert,degscore
    from mrnautilus_adapter import generate as mrnautilus
    handlers = {"degscore":degscore,"mrnautilus":mrnautilus,"codongpt":codongpt,"utrgan":utrgan,"saluki":saluki,"utr_lm":utr_lm,"rnadegformer":rnadegformer,"mrnabert":mrnabert,"gemorna": gemorna, "ribonn": ribonn, "parade": parade, "smart": smart}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = handlers[request["tool"]](request["arguments"])
        print(json.dumps({"status": "ok", "data": result}, allow_nan=False))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "reason": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
