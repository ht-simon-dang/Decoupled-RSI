"""Reproducibility bundle writer: emits ``runs/<run-id>/`` + RunManifest."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..contracts.run import RunManifest, Task
from ..security.redaction import redact_env_names
from ..util import canonical_json, hash_bytes, utcnow_iso

if TYPE_CHECKING:
    from ..orchestration.context import RunContext

_KEY_PACKAGES = [
    "rdkit",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "pydantic",
    "fastapi",
    "typer",
    "sqlalchemy",
    "pyarrow",
]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in _KEY_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "absent"
    return out


def _write(path: Path, data: bytes | str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    b = data.encode("utf-8") if isinstance(data, str) else data
    path.write_bytes(b)
    return hash_bytes(b)


def _dump_group(root: Path, name: str, objects: list[dict[str, Any]], hashes: dict[str, str]) -> None:
    for obj in objects:
        oid = obj.get("candidate_id") or obj.get("id") or "item"
        p = root / name / f"{oid}.json"
        hashes[str(p.relative_to(root.parent))] = _write(p, canonical_json(obj))


def create_bundle(ctx: RunContext) -> dict[str, Any]:
    S = ctx.store
    rid = ctx.run.id
    root = Path(ctx.settings.runs_dir or ".drugagent/runs") / rid
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    brief = S.get_object("brief", ctx.run.brief_id) or {}
    hashes["research_brief.yaml"] = _write(
        root / "research_brief.yaml", yaml.safe_dump(brief, sort_keys=False)
    )

    tasks = [Task.model_validate(t) for t in S.list_tasks(rid)]
    plan = {
        "run_id": rid,
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "state": str(t.state),
                "depends_on": t.depends_on,
                "capability_id": t.capability_id,
                "approval_gate": t.approval_gate,
                "cache_hit": t.cache_hit,
                "error": t.error,
            }
            for t in tasks
        ],
    }
    hashes["execution_plan.json"] = _write(root / "execution_plan.json", canonical_json(plan))

    events = S.list_events(rid)
    hashes["task_events.jsonl"] = _write(root / "task_events.jsonl", "\n".join(json.dumps(e) for e in events))

    evidence = S.list_objects("evidence", run_id=rid)
    hashes["evidence.jsonl"] = _write(root / "evidence.jsonl", "\n".join(json.dumps(e) for e in evidence))

    scorecards = S.list_objects("scorecard", run_id=rid)
    hashes["scorecards.json"] = _write(root / "scorecards.json", canonical_json(scorecards))

    portfolios = S.list_objects("portfolio", run_id=rid)
    hashes["pareto_portfolio.json"] = _write(
        root / "pareto_portfolio.json", canonical_json(portfolios[0] if portfolios else {})
    )

    rejected = [s for s in scorecards if s.get("decision") == "REJECT"]
    hashes["rejected_candidates.json"] = _write(root / "rejected_candidates.json", canonical_json(rejected))

    # candidates.parquet + candidates.sdf
    candidates = S.list_objects("candidate", run_id=rid)
    _write_candidates(root, candidates, scorecards, hashes)

    # grouped specialist outputs
    _dump_group(root, "synthesis", S.list_objects("synthesis", run_id=rid), hashes)
    _dump_group(
        root,
        "structures",
        S.list_objects("pose", run_id=rid) + S.list_objects("simulation", run_id=rid),
        hashes,
    )
    _dump_group(root, "perturbation", S.list_objects("perturbation", run_id=rid), hashes)
    _dump_group(root, "safety", S.list_objects("safety", run_id=rid), hashes)
    _dump_group(root, "data_cards", S.list_objects("data_card", run_id=rid), hashes)

    # model cards
    baseline = S.list_objects("baseline_metrics", run_id=rid)
    model_card = {
        "id": "activity_model",
        "type": "RandomForest on Morgan fingerprints",
        "endpoint": "pIC50",
        "metrics": baseline[0] if baseline else {},
        "intended_use": "research prioritization only; not clinical",
        "limitations": "synthetic demo labels; scaffold-split evaluation",
    }
    hashes["model_cards/activity_model.json"] = _write(
        root / "model_cards" / "activity_model.json", canonical_json(model_card)
    )

    # Generator model card — one per de-novo batch, so a reproducer can see
    # exactly what produced the structures.
    for gen in S.list_objects("generation_batch", run_id=rid):
        # Retained for journals written before the surrogate backend was removed;
        # nothing produces a True here any more.
        surrogate = bool(gen.get("is_surrogate"))
        # Derived from the batch, never hard-coded: with twenty-one generators in
        # the catalogue, labelling every model card "PFM (Perturbed Flow
        # Matching)" and citing PFM's repository put a false provenance claim in
        # the reproducibility bundle — the one artifact whose whole job is to say
        # truthfully what produced these molecules.
        engine = str(gen.get("origin_tag") or gen.get("backend") or "unknown").split(".")[0]
        card = {
            "id": gen.get("id"),
            "type": (
                "BRICS recombination surrogate"
                if surrogate
                else ("PFM (Perturbed Flow Matching)" if engine == "pfm" else engine)
            ),
            "upstream": "https://github.com/kurisu92725/PFM" if engine == "pfm" else None,
            "backend": gen.get("backend"),
            "model_id": gen.get("model_id"),
            "model_version": gen.get("model_version"),
            "is_surrogate": surrogate,
            "origin_tag": gen.get("origin_tag"),
            "trust_label": gen.get("trust_label"),
            "seed": gen.get("seed"),
            "pocket": gen.get("pocket"),
            "config": gen.get("config"),
            "metrics": gen.get("metrics"),
            "warnings": gen.get("warnings"),
            "errors": gen.get("errors"),
            "intended_use": "research prioritization only; not clinical",
            "limitations": (
                "Generated structures are HYPOTHESES: not validated for synthesizability, "
                "activity, or safety. The surrogate is not structure-conditioned and carries "
                "no binding-mode information."
                if surrogate
                else "Generated structures are HYPOTHESES: pocket-conditioned samples with no "
                "experimental validation. Reconstruction from the generated point cloud fails "
                "for a fraction of samples; validity/uniqueness are reported above."
            ),
        }
        stem = gen.get("id", "generator")
        hashes[f"model_cards/{stem}.json"] = _write(
            root / "model_cards" / f"{stem}.json", canonical_json(card)
        )

    # figures (dependency-free SVG funnel)
    _write_funnel_svg(root, S, rid, hashes)

    # report (export from artifacts if present, else regenerate)
    rep = S.get_object("report", rid)
    if rep:
        md = ctx.artifacts.get_bytes(rep["md_ref"])
        htmlb = ctx.artifacts.get_bytes(rep["html_ref"])
        hashes["report.md"] = _write(root / "report.md", md)
        hashes["report.html"] = _write(root / "report.html", htmlb)

    # manifest
    manifest = _build_manifest(ctx, tasks, hashes)
    hashes["manifest.json"] = _write(root / "manifest.json", canonical_json(manifest.model_dump(mode="json")))
    S.put_object("manifest", manifest, run_id=rid)

    return {
        "bundle_dir": str(root),
        "n_files": len(hashes),
        "manifest": str(root / "manifest.json"),
        "report_html": str(root / "report.html"),
    }


def _write_candidates(
    root: Path, candidates: list[dict], scorecards: list[dict], hashes: dict[str, str]
) -> None:
    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    sc_by = {s["candidate_id"]: s for s in scorecards}
    rows = []
    for c in candidates:
        sc = sc_by.get(c["id"], {})
        rows.append(
            {
                "candidate_id": c["id"],
                "inchikey": c.get("inchikey"),
                "canonical_smiles": c.get("canonical_smiles"),
                "origin": c.get("origin"),
                "passed_hard_filters": c.get("passed_hard_filters"),
                "utility": sc.get("utility"),
                "pareto_rank": sc.get("pareto_rank"),
                "diversity_cluster": sc.get("diversity_cluster"),
                "decision": sc.get("decision"),
            }
        )
    df = pd.DataFrame(rows)
    pq = root / "candidates.parquet"
    df.to_parquet(pq, index=False)
    hashes["candidates.parquet"] = hash_bytes(pq.read_bytes())

    sdf_path = root / "candidates.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for c in candidates:
        mol = Chem.MolFromSmiles(c.get("canonical_smiles") or "")
        if mol is None:
            continue
        mol.SetProp("_Name", c["id"])
        if c.get("inchikey"):
            mol.SetProp("inchikey", c["inchikey"])
        origin = str(c.get("origin") or "unknown")
        mol.SetProp("origin", origin)
        # An SDF is what leaves the system and gets opened elsewhere, so a
        # de-novo structure must announce itself there too, not only in the report.
        if origin in {"pfm", "pfm_mock"}:  # pfm_mock: pre-removal runs only
            mol.SetProp("de_novo", "true")
            mol.SetProp(
                "trust_label",
                "PROPOSED — generated structure, not validated for synthesizability, "
                "activity, or safety" + ("; SURROGATE output, not PFM" if origin == "pfm_mock" else ""),
            )
        sc = sc_by.get(c["id"], {})
        if sc.get("decision"):
            mol.SetProp("decision", str(sc["decision"]))
        writer.write(mol)
    writer.close()
    hashes["candidates.sdf"] = hash_bytes(sdf_path.read_bytes())


def _write_funnel_svg(root: Path, S: Any, rid: str, hashes: dict[str, str]) -> None:
    stages = [
        ("sourced", len(S.list_objects("raw_candidate", run_id=rid))),
        ("standardized", len(S.list_objects("candidate", run_id=rid))),
    ]
    cands = S.list_objects("candidate", run_id=rid)
    stages.append(("passed_filters", sum(1 for c in cands if c.get("passed_hard_filters"))))
    ports = S.list_objects("portfolio", run_id=rid)
    stages.append(("shortlist", len(ports[0]["shortlist"]) if ports else 0))
    maxv = max((v for _, v in stages), default=1) or 1
    bars = ""
    for i, (k, v) in enumerate(stages):
        w = int(360 * v / maxv)
        y = 20 + i * 40
        bars += (
            f'<rect x="120" y="{y}" width="{w}" height="24" fill="#3b6fb0" rx="4"/>'
            f'<text x="10" y="{y + 17}" font-size="13" fill="#333">{k}</text>'
            f'<text x="{130 + w}" y="{y + 17}" font-size="13" fill="#333">{v}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="560" height="{40 * len(stages) + 40}">'
        f'<text x="10" y="14" font-size="14" font-weight="bold">Candidate funnel</text>{bars}</svg>'
    )
    hashes["figures/funnel.svg"] = _write(root / "figures" / "funnel.svg", svg)


def _build_manifest(ctx: RunContext, tasks: list[Task], hashes: dict[str, str]) -> RunManifest:
    import os

    # The dependency set this run executed under, hashed into the manifest.
    # `environment.lock.yml` (written by `make env-export`) is a full solve and
    # is preferred when present; `environment.yml` is the declared spec that
    # every checkout has. Both are conda specs — there is no other package
    # manager in this project.
    lock_hash = None
    for spec in (Path("environment.lock.yml"), Path("environment.yml")):
        if spec.exists():
            lock_hash = hash_bytes(spec.read_bytes())
            break
    tool_versions = {p.capability_id: p.version for p in ctx.registry.profiles()}
    return RunManifest(
        run_id=ctx.run.id,
        brief_id=ctx.run.brief_id,
        profile=ctx.run.profile,
        deterministic=ctx.run.deterministic,
        repo_revision=_git("rev-parse", "HEAD"),
        repo_dirty=(bool(_git("status", "--porcelain"))),
        lock_hash=lock_hash,
        os=platform.system(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        gpu=None,
        seeds={"run": ctx.run.seed},
        config_hash=ctx.run.config_hash,
        data_snapshots={
            c["name"]: c.get("snapshot_id", "")
            for c in ctx.store.list_objects("data_card", run_id=ctx.run.id)
        },
        model_versions={"activity_model": "baseline-rf-1.0.0"},
        tool_versions=tool_versions,
        package_versions=_package_versions(),
        env_vars_present=redact_env_names({k: "" for k in os.environ if k.startswith("DRUGAGENT_")}),
        started_at=ctx.run.started_at,
        finished_at=utcnow_iso(),
        task_states={t.name: str(t.state) for t in tasks},
        artifact_hashes=hashes,
        capability_matrix=ctx.registry.capability_matrix(),
    )
