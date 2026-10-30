"""Sample protein-ligand complexes with FlowDock — the drug-agent ⇄ FlowDock bridge.

This file is executed by whichever interpreter has FlowDock's dependencies. It
imports nothing from ``drug_agent``; the two sides talk over a JSON request /
response pair on disk, which is what lets the platform drive a stack it could
never import (FlowDock pins Python 3.10, torch 2.2.1+cu118, an ``openfold`` fork
and ``pytorch3d``).

**Why it shells out to ``flowdock/sample.py`` instead of calling ``sample(cfg)``.**
FlowDock's sampling entry point is ``@hydra.main(version_base="1.3",
config_path="../configs", config_name="sample.yaml")``, and the config tree it
composes depends on Hydra's own runtime: ``configs/paths/default.yaml`` resolves
``output_dir`` from ``${hydra:runtime.output_dir}`` and ``root_dir`` from the
``PROJECT_ROOT`` variable that ``rootutils.setup_root(...)`` sets at import time,
while ``flowdock/sample.py``'s ``__main__`` block registers six custom OmegaConf
resolvers before Hydra composes anything. Re-creating that by hand with
``hydra.compose`` would be a second, divergent implementation of the upstream
entry point. Running the documented command instead means this driver is wrong
only if the README is.

Request JSON (everything but ``out_dir`` is this driver's own contract):
    checkout_root       FlowDock checkout; defaults to the working directory
    ckpt_path           FlowDock checkpoint, e.g.
                        ``checkpoints/esmfold_prior_paper_weights-EMA.ckpt``
    input_receptor      receptor PDB path, or a sequence with chains split by ``|``
    input_ligand        ligand SMILES (fragments split by ``|``), or an SDF path
    input_template      optional template protein PDB for the ESMFold prior
    sample_id           identifier; also the output sub-directory's name
    n_samples           structures to sample
    chunk_size          structures sampled concurrently per batch segment
    num_steps           ODE integration steps
    sampler             ``ODE`` or ``VDODE``
    sampler_eta         variance-diminishing factor for ``VDODE``
    start_time          time at which to start sampling
    prior_type          ``gaussian``, ``harmonic`` or ``esmfold``
    esmfold_chunk_size  axial-attention chunking for ESMFold, or null
    auxiliary_estimation_only
                        estimate confidence and affinity for the *input*
                        coordinates instead of sampling new ones
    device              ``cuda...`` selects ``trainer=gpu``; anything else ``cpu``
    seed                RNG seed, or null
    out_dir             injected by the caller; receives ``result.json``

Response JSON (``out_dir/result.json``):
    poses[]             {rank, confidence_plddt_ligand, affinity_pk, sdf,
                         sdf_name, protein_pdb_name}
    n_poses             how many pose files were found
    sample_id           echoed back
    auxiliary_estimation
                        rows of FlowDock's own ``*_auxiliary_estimation.csv``,
                        which carries the per-ligand plDDT and affinity lists
    warnings[]          anything found but not understood

Both numbers come straight from FlowDock: ``multi_pose_sampling`` in
``flowdock/utils/sampling_utils.py`` writes them into the output filenames as
``lig_rank{rank}_plddt{confidence:.7f}_affinity{pK:.7f}.sdf``. They are read
back, never recomputed, and never combined.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: FlowDock's documented sampling entry point, relative to the checkout root.
SAMPLE_ENTRY = "flowdock/sample.py"

#: What ``multi_pose_sampling`` names a ranked ligand pose when
#: ``rank_outputs_by_confidence`` is on and the affinity head is enabled.
_RANKED_LIGAND = re.compile(
    r"^lig_rank(?P<rank>\d+)_plddt(?P<plddt>-?\d+\.\d+)"
    r"(?:_affinity(?P<affinity>-?\d+\.\d+))?\.sdf$"
)

#: The unranked fallback, written when ``rank_outputs_by_confidence`` is off.
_UNRANKED_LIGAND = re.compile(r"^lig_(?P<index>\d+)(?:_affinity(?P<affinity>-?\d+\.\d+))?\.sdf$")

#: The receptor written beside each ligand pose, matched only to report its name.
_RANKED_PROTEIN = re.compile(r"^prot_rank(?P<rank>\d+)_plddt-?\d+\.\d+.*\.pdb$")

#: Aggregate files ``multi_pose_sampling`` always writes; not per-pose outputs.
_NOT_A_POSE = frozenset({"lig_all.sdf", "lig_ref.sdf"})

#: How many lines of a failed FlowDock run are quoted back in the error.
_TAIL_LINES = 25


def _quote(value):
    """Wrap *value* for Hydra's override grammar.

    FlowDock's README passes ligands as ``input_ligand='"c1cc2c(cc1O)CCCC2"'``:
    the inner double quotes are part of the value handed to Hydra, because a bare
    SMILES contains parentheses, brackets and equals signs that Hydra's override
    parser would otherwise read as grammar. Backslashes matter too — ``C/C=C\\C``
    is an ordinary SMILES — so both are escaped rather than stripped.
    """
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return '"' + text + '"'


def _flag(value):
    """Hydra spells booleans lowercase."""
    return "true" if value else "false"


def _overrides(request, out_path, sample_id, ckpt_path):
    """The command line from FlowDock's README, filled in from *request*."""
    device = str(request.get("device") or "cpu")
    prior_type = str(request.get("prior_type") or "esmfold")
    template = request.get("input_template")
    seed = request.get("seed")
    esmfold_chunk_size = request.get("esmfold_chunk_size")
    args = [
        "ckpt_path=" + _quote(ckpt_path),
        "sampling_task=batched_structure_sampling",
        "prior_type=" + prior_type,
        "model.cfg.prior_type=" + prior_type,
        "input_receptor=" + _quote(request["input_receptor"]),
        "input_ligand=" + _quote(request["input_ligand"]),
        "input_template=" + (_quote(template) if template else "null"),
        "sample_id=" + _quote(sample_id),
        "out_path=" + _quote(out_path),
        "n_samples=" + str(int(request.get("n_samples") or 5)),
        "chunk_size=" + str(int(request.get("chunk_size") or request.get("n_samples") or 5)),
        "num_steps=" + str(int(request.get("num_steps") or 40)),
        "sampler=" + str(request.get("sampler") or "VDODE"),
        "sampler_eta=" + str(float(request.get("sampler_eta") or 1.0)),
        "start_time=" + _quote(request.get("start_time") or "1.0"),
        "use_template=" + _flag(template),
        "separate_pdb=true",
        "rank_outputs_by_confidence=true",
        "plddt_ranking_type=ligand",
        "visualize_sample_trajectories=false",
        "auxiliary_estimation_only=" + _flag(request.get("auxiliary_estimation_only")),
        "esmfold_chunk_size=" + (str(int(esmfold_chunk_size)) if esmfold_chunk_size else "null"),
        "trainer=" + ("gpu" if device.startswith("cuda") else "cpu"),
    ]
    if seed is not None:
        args.append("seed=" + str(int(seed)))
    return args


def _tail(path):
    """The last lines of the FlowDock log, for an error that has to explain itself."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "(FlowDock produced no output)"
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-_TAIL_LINES:]) or "(FlowDock produced no output)"


def _read_auxiliary_csv(sample_dir, sample_id):
    """FlowDock's own per-sample estimation table, returned verbatim as rows."""
    path = sample_dir / (sample_id + "_auxiliary_estimation.csv")
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _collect(sample_dir, warnings):
    """Read every per-pose SDF FlowDock wrote, with the numbers in its filenames."""
    proteins = {}
    for pdb in sorted(sample_dir.glob("prot_rank*.pdb")):
        matched = _RANKED_PROTEIN.match(pdb.name)
        if matched:
            proteins[int(matched.group("rank"))] = pdb.name

    poses = []
    for sdf in sorted(sample_dir.glob("lig_*.sdf")):
        if sdf.name in _NOT_A_POSE:
            continue
        ranked = _RANKED_LIGAND.match(sdf.name)
        unranked = None if ranked else _UNRANKED_LIGAND.match(sdf.name)
        if ranked is None and unranked is None:
            warnings.append("unrecognised FlowDock output file " + sdf.name)
            continue
        matched = ranked if ranked is not None else unranked
        rank = int(matched.group("rank")) if ranked is not None else int(matched.group("index")) + 1
        affinity = matched.group("affinity")
        poses.append(
            {
                "rank": rank,
                "confidence_plddt_ligand": float(ranked.group("plddt")) if ranked else None,
                "affinity_pk": float(affinity) if affinity is not None else None,
                "sdf": sdf.read_text(),
                "sdf_name": sdf.name,
                "protein_pdb_name": proteins.get(rank),
            }
        )
    poses.sort(key=lambda pose: pose["rank"])
    return poses


def run(request):
    """Run FlowDock's sampling entry point once and return what it wrote."""
    root = Path(request.get("checkout_root") or Path.cwd()).resolve()
    entry = root / SAMPLE_ENTRY
    if not entry.is_file():
        raise RuntimeError("FlowDock entry point not found at " + str(entry))

    ckpt_path = Path(request["ckpt_path"]).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.is_file():
        raise RuntimeError(
            "FlowDock checkpoint not found at "
            + str(ckpt_path)
            + "; fetch flowdock_checkpoints.tar.gz from https://zenodo.org/records/15066450"
        )

    out_dir = Path(request["out_dir"])
    out_path = out_dir / "samples"
    out_path.mkdir(parents=True, exist_ok=True)
    sample_id = str(request.get("sample_id") or "sample")

    argv = [sys.executable, str(entry), *_overrides(request, out_path, sample_id, ckpt_path)]
    env = dict(os.environ)
    env.setdefault("HYDRA_FULL_ERROR", "1")
    log_path = out_dir / "flowdock_sample.log"
    with log_path.open("wb") as log_file:
        returncode = subprocess.call(argv, cwd=str(root), env=env, stdout=log_file, stderr=log_file)
    if returncode != 0:
        raise RuntimeError("FlowDock sampling exited " + str(returncode) + ":\n" + _tail(log_path))

    warnings = []
    sample_dir = out_path / sample_id
    if not sample_dir.is_dir():
        warnings.append("FlowDock wrote no " + sample_id + " directory; reading " + str(out_path))
        sample_dir = out_path
    poses = _collect(sample_dir, warnings)
    if not poses:
        raise RuntimeError(
            "FlowDock exited 0 but wrote no ligand poses under " + str(sample_dir) + ":\n" + _tail(log_path)
        )
    return {
        "poses": poses,
        "n_poses": len(poses),
        "sample_id": sample_id,
        "auxiliary_estimation": _read_auxiliary_csv(sample_dir, sample_id),
        "warnings": warnings,
    }


if __name__ == "__main__":
    _driver.main(run)
