"""ToxinPred3 published ExtraTrees inference, exported to pickle-free arrays.

The original sklearn 1.2 tree serialization is incompatible with sklearn 1.9.
Only the pinned, SHA-verified official artifact is read by the export command.
Runtime uses the original split thresholds and normalized leaf class counts.
"""
import hashlib
import io
import json
import urllib.request
import warnings
import zipfile

import numpy as np

from .contracts import AA, Analyze, ToolResult, Unavailable
from .models import model_root

REVISION = "dcdba833cd48cc8a2090ac90f3319fd9bcb87a1c"
MODEL_SHA = "bacaef2ff40136babe98edcde861ea79d62ea3420a6b1b5d36019fa0ddc2db3b"


class TreeSnapshot:
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def export_model():
    from joblib.numpy_pickle import NumpyUnpickler
    from sklearn.exceptions import InconsistentVersionWarning
    root = model_root() / "toxinpred3"
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/raghavagps/toxinpred3/{REVISION}/model/toxinpred3.0_model.pkl.zip"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        raw = handle.read("toxinpred3.0_model.pkl")
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA:
        raise ValueError("Official ToxinPred3 model checksum mismatch")
    path = root / "model.pkl"
    path.write_bytes(raw)
    class Loader(NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) == ("sklearn.tree._tree", "Tree"):
                return TreeSnapshot
            return super().find_class(module, name)
    with path.open("rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        forest = Loader(str(path), handle, False).load()
    if forest.n_features_in_ != 420 or list(forest.classes_) != [0, 1]:
        raise ValueError("Unexpected ToxinPred3 feature/class layout")
    arrays = {}
    for i, tree in enumerate(forest.estimators_):
        nodes, counts = tree.tree_.state["nodes"], tree.tree_.state["values"][:, 0, :]
        for key in ("left_child", "right_child", "feature", "threshold"):
            arrays[f"{i}_{key}"] = nodes[key]
        arrays[f"{i}_prob"] = counts[:, 1] / counts.sum(axis=1)
    arrays["n_trees"] = np.array(len(forest.estimators_))
    np.savez_compressed(root / "trees.npz", **arrays)
    (root / "manifest.json").write_text(json.dumps({"revision": REVISION, "source_sha256": MODEL_SHA,
        "array_sha256": hashlib.sha256((root / "trees.npz").read_bytes()).hexdigest(),
        "n_trees": len(forest.estimators_), "features": "AAC percent (20), DPC percent (400)"}))
    return str(root / "trees.npz")


def features(sequence):
    return np.asarray([100 * sequence.count(a) / len(sequence) for a in AA] +
                      [100 * sum(sequence[i:i+2] == a+b for i in range(len(sequence)-1)) / (len(sequence)-1)
                       for a in AA for b in AA], dtype=np.float32)


def evaluate_tree(arrays, i, x):
    node = 0
    while arrays[f"{i}_left_child"][node] != -1:
        side = "left_child" if x[arrays[f"{i}_feature"][node]] <= arrays[f"{i}_threshold"][node] else "right_child"
        node = int(arrays[f"{i}_{side}"][node])
    return float(arrays[f"{i}_prob"][node])


def toxicity(request: Analyze):
    if request.topology != "linear":
        raise Unavailable("ToxinPred3 is not validated for cyclic chemistry; linear precursor scores are not substituted")
    root = model_root() / "toxinpred3"
    if not (root / "trees.npz").exists():
        raise Unavailable("Run peptide-agent download toxinpred3")
    manifest = json.loads((root / "manifest.json").read_text())
    if hashlib.sha256((root / "trees.npz").read_bytes()).hexdigest() != manifest["array_sha256"]:
        raise ValueError("ToxinPred3 arrays checksum mismatch")
    with np.load(root / "trees.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    n = int(arrays["n_trees"])
    results = []
    for seq in request.sequences:
        x = features(seq)
        score = sum(evaluate_tree(arrays, i, x) for i in range(n)) / n
        results.append({"sequence": seq, "toxin_score": score,
                        "toxicity_flag": score >= 0.38, "threshold": 0.38})
    return ToolResult(tool="safety.toxinpred3", status="ok", trust="PREDICTED",
                      data={"candidates": results}, provenance={**manifest, "model": "ToxinPred3 ML-only ExtraTrees"},
                      warnings=["Published classifier score, not calibrated clinical toxicity probability.",
                                "MERCI hybrid is not used. Cyclic/noncanonical peptides are outside the validated domain."])
