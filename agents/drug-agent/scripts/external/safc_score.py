"""Score molecules with the frozen SAFC target bundle, inside its own environment.

SAFC's score is not a property of a molecule alone. It is read off an ensemble:
every candidate is docked into each conformation of an MD-derived receptor
ensemble, each docked complex is encoded by the frozen official EquiScore graph
encoder into 128 dimensions, those per-pose embeddings are averaged per ligand,
and the bundle's own trained target head turns that one vector into a score. The
four stages are the bundle's own scripts, run here in order:

    01  ligand prep    RDKit ETKDG + MMFF -> SDF -> Open Babel -> PDBQT
    02  ensemble dock  AutoDock Vina, one run per conformation, that row's box
    03  complexes      05c_build_official_equiscore_inputs.py
    04  embed + score  05d_extract_official_equiscore_embeddings.py, then
                       07_predict_official_equiscore_tl.py

Why this driver takes a persistent ``workdir``
----------------------------------------------
``ExternalBackend.run_json`` hands every driver a fresh ``out_dir`` and deletes
it in a ``finally``. That is right for a tool whose output *is* its result, and
wrong for this one: the expensive artifact here is the pose cache — one Vina run
per ligand per conformation, 41 of them per molecule on the ZAK bundle — and a
150-molecule submission is thousands of docks over hours. Destroying that on
return would make every interruption start from zero. So ``workdir`` is a path
the caller owns and keeps, ``out_dir`` still receives ``result.json``, and a dock
whose pose file is already on disk is not repeated. Re-invoking after a kill
resumes.

What it refuses to do
---------------------
Nothing here invents a pose, an embedding or a score. A ligand that will not
embed, a dock that writes no scored model, a complex the builder rejects and a
pose the encoder cannot featurise are each recorded with their reason and
carried into the result, because a molecule nobody scored must never come back
looking like a clean one. ``pose_count`` travels with every score for the same
reason: a score averaged over 12 surviving poses is not the same measurement as
one averaged over 41, and the reader has to be able to see which they have.

The score is a ranking signal, not a calibrated probability — the bundle's own
guidance says so, and this driver copies its wording into the provenance block
rather than restating it in its own words.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _driver

VINA_RESULT = re.compile(r"^REMARK VINA RESULT:\s+([-+0-9.eE]+)")
#: Columns 05c requires of the docking manifest it consumes.
MANIFEST_COLUMNS = (
    "inchikey",
    "receptor",
    "state_arm",
    "pose_pdbqt",
    "receptor_pdb",
    "vina_score",
)


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(part) for part in cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _resolve_executable(value: str, label: str) -> str:
    found = shutil.which(value)
    if found:
        return found
    if Path(value).is_file():
        return str(Path(value).resolve())
    raise RuntimeError(
        f"{label} is not usable here ({value!r}); SAFC docks nothing without it"
    )


# ------------------------------------------------------------------ stage 01


def _prepare_ligands(
    molecules: list[dict],
    *,
    workdir: Path,
    obabel: str,
    seed: int,
) -> dict[str, dict]:
    """SMILES -> one PDBQT per unique InChIKey, keyed by ``sample_id``.

    The 3D conformer is embedded once per *molecule*, not per conformation: the
    ensemble varies the receptor, and giving each receptor its own ligand
    conformer would confound the two.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    sdf_dir = workdir / "ligands_sdf"
    pdbqt_dir = workdir / "ligands_pdbqt"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, dict] = {}
    for entry in molecules:
        sample_id = str(entry["sample_id"])
        smiles = str(entry.get("smiles") or entry.get("canonical_smiles") or "")
        record: dict = {"sample_id": sample_id, "smiles": smiles}
        prepared[sample_id] = record

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            record["status"] = "unparseable_smiles"
            continue
        inchikey = Chem.MolToInchiKey(mol)
        record["inchikey"] = inchikey
        record["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        pdbqt = pdbqt_dir / f"{inchikey}.pdbqt"
        if pdbqt.is_file() and pdbqt.stat().st_size > 0:
            record["status"] = "ok"
            record["pdbqt"] = str(pdbqt)
            continue

        sdf = sdf_dir / f"{inchikey}.sdf"
        if not sdf.is_file():
            confmol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed)
            if AllChem.EmbedMolecule(confmol, params) != 0:
                record["status"] = "embed_failed"
                continue
            try:
                AllChem.MMFFOptimizeMolecule(confmol)
            except Exception:  # noqa: BLE001 - optimisation is best-effort
                pass
            Chem.MolToMolFile(confmol, str(sdf))

        result = _run([obabel, str(sdf), "-O", str(pdbqt)], timeout=300)
        if not pdbqt.is_file() or pdbqt.stat().st_size == 0:
            record["status"] = "obabel_failed"
            record["detail"] = (result.stderr or result.stdout or "").strip()[-300:]
            continue
        record["status"] = "ok"
        record["pdbqt"] = str(pdbqt)
    return prepared


# ------------------------------------------------------------------ stage 02


def _conformations(target_root: Path) -> list[dict]:
    manifest = target_root / "structures" / "conformation_manifest.csv"
    if not manifest.is_file():
        raise RuntimeError(f"conformation manifest missing: {manifest}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"conformation manifest is empty: {manifest}")
    return rows


def _first_affinity(pose: Path) -> float | None:
    with pose.open() as handle:
        for line in handle:
            match = VINA_RESULT.match(line)
            if match:
                return float(match.group(1))
    return None


def _dock_ensemble(
    prepared: dict[str, dict],
    conformations: list[dict],
    *,
    target_root: Path,
    workdir: Path,
    vina: str,
    jobs: int,
    vina_cpu: int,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    dock_timeout: float,
) -> tuple[Path, list[dict]]:
    """Dock each prepared ligand into every conformation. Already-written poses stand."""
    pose_root = workdir / "poses"
    ligands = {
        record["inchikey"]: record["pdbqt"]
        for record in prepared.values()
        if record.get("status") == "ok" and record.get("pdbqt")
    }

    tasks: list[tuple[str, str, dict, Path]] = []
    for inchikey, ligand_pdbqt in sorted(ligands.items()):
        for conformation in conformations:
            pose = pose_root / inchikey / f"{conformation['conformation_id']}.pdbqt"
            tasks.append((inchikey, ligand_pdbqt, conformation, pose))
    pending = [task for task in tasks if not (task[3].is_file() and task[3].stat().st_size > 0)]
    print(
        f"[dock] {len(ligands)} ligands x {len(conformations)} conformations = {len(tasks)} docks; "
        f"{len(tasks) - len(pending)} cached, {len(pending)} to run",
        flush=True,
    )

    def one(task: tuple[str, str, dict, Path]) -> dict | None:
        inchikey, ligand_pdbqt, conformation, pose = task
        pose.parent.mkdir(parents=True, exist_ok=True)
        size = float(conformation.get("box_size_angstrom") or 22.0)
        cmd = [
            vina,
            "--receptor", str(target_root / conformation["receptor_pdbqt"]),
            "--ligand", ligand_pdbqt,
            "--center_x", conformation["box_cx"],
            "--center_y", conformation["box_cy"],
            "--center_z", conformation["box_cz"],
            "--size_x", str(size), "--size_y", str(size), "--size_z", str(size),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--cpu", str(vina_cpu),
            "--seed", str(seed),
            "--out", str(pose),
        ]
        try:
            result = _run(cmd, timeout=dock_timeout)
        except subprocess.TimeoutExpired:
            return {
                "inchikey": inchikey,
                "conformation_id": conformation["conformation_id"],
                "status": f"dock_timeout>{dock_timeout:g}s",
            }
        if not pose.is_file() or pose.stat().st_size == 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            return {
                "inchikey": inchikey,
                "conformation_id": conformation["conformation_id"],
                "status": "no_pose",
                "detail": tail[-1] if tail else "",
            }
        return None

    failures: list[dict] = []
    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = [pool.submit(one, task) for task in pending]
            for future in as_completed(futures):
                outcome = future.result()
                done += 1
                if outcome is not None:
                    failures.append(outcome)
                if done % 100 == 0 or done == len(pending):
                    print(f"[dock]   {done}/{len(pending)}, {len(failures)} failed", flush=True)

    rows: list[dict] = []
    for inchikey, _ligand, conformation, pose in tasks:
        if not (pose.is_file() and pose.stat().st_size > 0):
            continue
        affinity = _first_affinity(pose)
        if affinity is None:
            failures.append({
                "inchikey": inchikey,
                "conformation_id": conformation["conformation_id"],
                "status": "pose_without_score",
            })
            continue
        rows.append({
            "inchikey": inchikey,
            "receptor": conformation["receptor"],
            "state_arm": conformation["state_arm"],
            "pose_pdbqt": str(pose),
            "receptor_pdb": str(target_root / conformation["receptor_pdb"]),
            "vina_score": affinity,
        })

    manifest = workdir / "complex_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[dock] {len(rows)} scored poses -> {manifest}", flush=True)
    return manifest, failures


# ------------------------------------------------------------- stages 03, 04


def _stage(label: str, cmd: list[str]) -> str:
    print(f"[{label}] running", flush=True)
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout or '').strip()[-1200:]}"
        )
    return result.stdout


def _embed_and_score(
    prepared: dict[str, dict],
    *,
    bundle: Path,
    target_root: Path,
    workdir: Path,
    manifest: Path,
    batch_size: int,
    threads: int,
    device: str,
    pocket_cutoff: float,
) -> tuple[Path, dict]:
    scripts = bundle / "safc" / "scripts"
    scored = [record for record in prepared.values() if record.get("status") == "ok"]

    inference_csv = workdir / "inference_molecules.csv"
    seen: set[str] = set()
    with inference_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["inchikey", "canonical_smiles", "molecule_name"]
        )
        writer.writeheader()
        for record in scored:
            if record["inchikey"] in seen:
                continue
            seen.add(record["inchikey"])
            writer.writerow({
                "inchikey": record["inchikey"],
                "canonical_smiles": record["canonical_smiles"],
                "molecule_name": record["sample_id"],
            })

    # 05c normally builds training and inference complexes together and refuses to
    # finish while a *training* ligand has no pose. This path scores candidates
    # against an already-trained frozen head and docks no training ligand, so an
    # empty training view is the truthful declaration and satisfies that guard
    # without weakening it. Passing the bundle's real 67-row training_view.csv
    # would make the guard fail correctly, for a run that never promised those poses.
    empty_training = workdir / "training_view_empty.csv"
    empty_training.write_text("inchikey,canonical_smiles,label\n")

    inputs_dir = workdir / "equiscore_inputs"
    _stage("05c", [
        sys.executable, "-I", str(scripts / "05c_build_official_equiscore_inputs.py"),
        "--manifest", str(manifest),
        "--training-csv", str(empty_training),
        "--inference-csv", str(inference_csv),
        "--output-dir", str(inputs_dir),
        "--pocket-cutoff", str(pocket_cutoff),
    ])

    embed_dir = workdir / "equiscore_embeddings"
    checkpoint = (
        bundle / "safc" / "third_party" / "EquiScore" / "workdir"
        / "official_weight" / "save_model_screen.pt"
    )
    _stage("05d", [
        sys.executable, "-I", str(scripts / "05d_extract_official_equiscore_embeddings.py"),
        "--input-manifest", str(inputs_dir / "input_manifest.csv"),
        "--upstream-dir", str(bundle / "safc" / "third_party" / "EquiScore"),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(embed_dir),
        "--batch-size", str(batch_size),
        "--threads", str(threads),
        "--device", device,
    ])
    # Named, not matched. 05d writes `pose_embeddings.csv` beside
    # `embedding_failures.csv`, and a glob over "*embedding*" sorts the empty
    # failures file first — which reaches stage 07 as "no columns to parse",
    # a message that describes the wrong file and hides a successful encode.
    embeddings = next(
        (
            path
            for path in (embed_dir / "pose_embeddings.csv", embed_dir / "embeddings.csv")
            if path.is_file() and path.stat().st_size > 0
        ),
        None,
    )
    if embeddings is None:
        present = [
            f"{path.name} ({path.stat().st_size}B)" for path in sorted(embed_dir.glob("*"))
        ]
        raise RuntimeError(
            f"the encoder wrote no usable pose_embeddings.csv into {embed_dir}; found {present}"
        )

    candidates = workdir / "candidates.csv"
    with candidates.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "canonical_smiles"])
        writer.writeheader()
        for record in scored:
            writer.writerow({
                "sample_id": record["sample_id"],
                "canonical_smiles": record["canonical_smiles"],
            })

    output = workdir / "safc_scores.csv"
    summary_path = workdir / "safc_summary.json"
    _stage("07", [
        sys.executable, "-I", str(scripts / "07_predict_official_equiscore_tl.py"),
        "--candidates", str(candidates),
        "--embeddings", str(embeddings),
        "--model", str(target_root / "models" / "official_equiscore_tl" / "primary_model.joblib"),
        "--output", str(output),
        "--summary", str(summary_path),
    ])

    embed_summary: dict = {}
    for name in ("embedding_summary.json", "summary.json"):
        candidate = embed_dir / name
        if candidate.is_file():
            try:
                embed_summary = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                embed_summary = {}
            break
    return output, embed_summary


# --------------------------------------------------------------------- entry


def _read_scores(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(request: dict) -> dict:
    task = str(request.get("task") or "score")
    if task != "score":
        raise ValueError(f"unknown task {task!r}; this driver only scores")

    bundle = Path(str(request.get("root") or request["bundle"])).resolve()
    target = str(request.get("target") or "zak")
    target_root = bundle / "targets" / target
    if not target_root.is_dir():
        raise RuntimeError(f"bundle has no target {target!r} at {target_root}")

    molecules = list(request.get("molecules") or [])
    if not molecules:
        raise ValueError("no molecules were given to score")

    workdir = Path(str(request.get("workdir") or request["out_dir"])).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    vina = _resolve_executable(str(request.get("vina") or "vina"), "AutoDock Vina")
    obabel = _resolve_executable(str(request.get("obabel") or "obabel"), "Open Babel")
    seed = int(request.get("seed") or 42)

    prepared = _prepare_ligands(
        molecules, workdir=workdir, obabel=obabel, seed=seed
    )
    ready = [record for record in prepared.values() if record.get("status") == "ok"]
    print(f"[prep] {len(ready)}/{len(molecules)} ligands prepared", flush=True)
    if not ready:
        raise RuntimeError("no ligand could be prepared; nothing was docked")

    manifest, dock_failures = _dock_ensemble(
        prepared,
        _conformations(target_root),
        target_root=target_root,
        workdir=workdir,
        vina=vina,
        jobs=int(request.get("jobs") or 10),
        vina_cpu=int(request.get("vina_cpu") or 2),
        exhaustiveness=int(request.get("exhaustiveness") or 8),
        num_modes=int(request.get("num_modes") or 3),
        seed=seed,
        dock_timeout=float(request.get("dock_timeout_seconds") or 900.0),
    )

    scores_csv, embed_summary = _embed_and_score(
        prepared,
        bundle=bundle,
        target_root=target_root,
        workdir=workdir,
        manifest=manifest,
        batch_size=int(request.get("batch_size") or 8),
        threads=int(request.get("threads") or 8),
        device=str(request.get("device") or "cpu"),
        pocket_cutoff=float(request.get("pocket_cutoff") or 8.0),
    )

    by_sample = {row["sample_id"]: row for row in _read_scores(scores_csv)}
    scores: list[dict] = []
    unscored: list[dict] = []
    for sample_id, record in prepared.items():
        row = by_sample.get(sample_id)
        if record.get("status") != "ok" or row is None:
            unscored.append({
                "sample_id": sample_id,
                "reason": record.get("status") or "no_score_row",
                "detail": record.get("detail", ""),
            })
            continue
        entry = {
            "sample_id": sample_id,
            "inchikey": record["inchikey"],
            "canonical_smiles": record["canonical_smiles"],
            "safc_score": float(row["safc_tl_score"]),
            "pose_count": int(float(row.get("equiscore_pose_count") or 0)),
        }
        for source, destination in (
            ("equiscore_zero_shot", "equiscore_zero_shot"),
            ("vina_redock_kcal_mol", "vina_ensemble_mean_kcal_mol"),
        ):
            if row.get(source) not in (None, ""):
                entry[destination] = float(row[source])
        scores.append(entry)

    target_config: dict = {}
    config_path = target_root / "config" / "target.json"
    if config_path.is_file():
        try:
            target_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            target_config = {}

    manifest_path = bundle / "bundle_manifest.json"
    bundle_manifest: dict = {}
    if manifest_path.is_file():
        try:
            bundle_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            bundle_manifest = {}

    return {
        "scores": scores,
        "unscored": unscored,
        "n_requested": len(molecules),
        "n_scored": len(scores),
        "n_unscored": len(unscored),
        "dock_failures": dock_failures[:200],
        "n_dock_failures": len(dock_failures),
        "workdir": str(workdir),
        "scores_csv": str(scores_csv),
        "provenance": {
            "bundle_version": (bundle / "VERSION").read_text().strip()
            if (bundle / "VERSION").is_file() else "",
            "target": target,
            "n_conformations": (target_config.get("structures") or {}).get("n_conformations"),
            "representation": (target_config.get("model") or {}).get("representation"),
            "primary_model": (target_config.get("model") or {}).get("primary_model"),
            "encoder_checkpoint_sha256": (target_config.get("model") or {}).get(
                "encoder_checkpoint_sha256"
            ),
            "selected_head": (bundle_manifest.get("target") or {}).get("final_model", {}).get("head"),
            "encoder_frozen": embed_summary.get("encoder_frozen"),
            "embedding_dim": embed_summary.get("embedding_dim"),
            "device": embed_summary.get("device"),
            "dgl_version": embed_summary.get("dgl_version"),
            "torch_version": embed_summary.get("torch_version"),
            # Copied from the bundle's own interpretation guidance rather than
            # paraphrased, because the distinction it draws is the whole point.
            "interpretation": (
                "Treat scores as ranking or prioritization evidence, not calibrated "
                "binding probabilities."
            ),
        },
    }


if __name__ == "__main__":
    _driver.main(run)
