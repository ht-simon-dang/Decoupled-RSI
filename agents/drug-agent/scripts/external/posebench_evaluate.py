"""Score one method's saved predictions with PoseBench, inside PoseBench's own environment.

This runs the evaluation upstream documents — ``python3
posebench/analysis/inference_analysis.py method=<method> dataset=<dataset>
repeat_index=<n>`` from the checkout root — and then reads back the scoring table
that entry point writes. It does not reimplement the scoring: PoseBench builds the
prediction/crystal/receptor table itself and runs ``PoseBusters(config="redock",
top_n=None)`` over it, and this driver's only jobs are to launch that correctly and
to turn ``bust_results.csv`` into per-target rows the adapter can summarise.

Three details the checkout makes mandatory:

``PROJECT_ROOT``
    ``configs/analysis/inference_analysis.yaml`` and ``configs/paths/default.yaml``
    both interpolate ``${oc.env:PROJECT_ROOT}``. OmegaConf raises if it is unset, so
    the driver exports it before launching the child.

*no chdir*
    The entry point is ``@hydra.main(version_base="1.3", ...)``, and hydra 1.3 does
    not change the working directory for a 1.3-versioned job. Every path the config
    resolves — the predictions directory, the input table — is relative to the
    checkout root, so the child is launched with ``cwd`` set there, exactly as the
    README's commands assume.

*the output directory is computed, not passed*
    ``posebench.resolve_method_output_dir`` is the same function the config's
    ``${resolve_method_output_dir:...}`` resolver calls, so calling it here finds the
    directory the child will write ``bust_results.csv`` into rather than guessing at
    the naming scheme, which differs per method family.

Checks with no value in the table are reported as ``None`` and left for the adapter
to count as failures. That is stricter than upstream's ``.all(axis=1)`` over a table
containing NaN, which treats an absent value as passing, and the divergence is
deliberate: it is surfaced as a flagged target rather than folded into a rate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver

#: The entry point, relative to the checkout root.
ENTRY_POINT = ("posebench", "analysis", "inference_analysis.py")

#: Upstream's accuracy verdict column. Spelled exactly as PoseBench writes it,
#: including the glyphs — this is a dataframe key, not prose.
ACCURACY_COLUMN = "rmsd_≤_2å"

#: The symmetry-corrected distance itself, present in a full report.
RMSD_COLUMN = "rmsd"

#: The per-target identifier `main()` appends to the scoring table.
ID_COLUMN = "mol_id"

#: How much of the child's stderr goes into a structured failure.
ERROR_TAIL_CHARS = 4000


def _flag(value: bool) -> str:
    """Hydra override spelling for a boolean."""
    return "true" if value else "false"


def run(request):
    root = Path(request.get("root") or Path.cwd()).resolve()
    method = str(request["method"])
    dataset = str(request["dataset"])
    repeat_index = int(request.get("repeat_index", 1))
    pocket_only = bool(request.get("pocket_only_baseline", False))
    v1_baseline = bool(request.get("v1_baseline", False))
    relax_protein = bool(request.get("relax_protein", False))
    force_rescore = bool(request.get("force_rescore", False))
    full_report = bool(request.get("full_report", True))
    vina_binding_site_method = str(request.get("vina_binding_site_method", "p2rank"))
    ensemble_ranking_method = str(request.get("ensemble_ranking_method", "consensus"))

    entry = root.joinpath(*ENTRY_POINT)
    if not entry.is_file():
        raise FileNotFoundError(f"PoseBench entry point missing: {entry}")

    import pandas as pd
    import posebench
    from posebench import resolve_method_input_csv_path, resolve_method_output_dir

    output_dir = root / resolve_method_output_dir(
        method,
        dataset,
        vina_binding_site_method,
        ensemble_ranking_method,
        repeat_index,
        pocket_only,
        v1_baseline,
    )
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"no predictions to score: {output_dir} does not exist. PoseBench evaluates a "
            f"method's saved output; run that method first, or download its published "
            f"predictions from https://doi.org/10.5281/zenodo.19138652."
        )

    env = dict(os.environ)
    env["PROJECT_ROOT"] = str(root)
    env["PYTHONPATH"] = (
        os.pathsep.join([str(root), env["PYTHONPATH"]]) if env.get("PYTHONPATH") else str(root)
    )

    overrides = [
        f"method={method}",
        f"dataset={dataset}",
        f"repeat_index={repeat_index}",
        f"pocket_only_baseline={_flag(pocket_only)}",
        f"v1_baseline={_flag(v1_baseline)}",
        f"relax_protein={_flag(relax_protein)}",
        f"force_rescore={_flag(force_rescore)}",
        f"full_report={_flag(full_report)}",
        f"vina_binding_site_method={vina_binding_site_method}",
        f"ensemble_ranking_method={ensemble_ranking_method}",
    ]
    # stdout is inherited so the child's progress lands in the capture the adapter
    # already spills to a file; stderr is captured so a traceback can go into the
    # structured failure instead of only into that tail.
    proc = subprocess.run(
        [sys.executable, str(entry), *overrides],
        cwd=str(root),
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    # `main()` scores the unrelaxed predictions first and writes them here; the
    # `_relaxed` pass, when the method has one, writes a sibling directory.
    bust_csv = output_dir / "bust_results.csv"
    if not bust_csv.is_file():
        tail = (proc.stderr or "")[-ERROR_TAIL_CHARS:].strip()
        raise RuntimeError(
            f"inference_analysis.py exited {proc.returncode} and wrote no {bust_csv}."
            + (f"\n{tail}" if tail else "")
        )

    from posebench.analysis.inference_analysis import (
        BUST_TEST_COLUMNS,
        DOCKGEN_BUST_TEST_COLUMNS,
    )

    # DockGen's crystal ligands are PDB, so upstream drops the graph-based
    # assertions for it. Use its list rather than a list of our own.
    wanted = DOCKGEN_BUST_TEST_COLUMNS if dataset == "dockgen" else BUST_TEST_COLUMNS
    frame = pd.read_csv(bust_csv)
    check_columns = [c for c in wanted if c != ACCURACY_COLUMN and c in frame.columns]
    absent_columns = [c for c in wanted if c not in frame.columns]

    warnings = []
    if absent_columns:
        warnings.append(
            f"{bust_csv.name} has no column for {', '.join(absent_columns)}; "
            f"validity was judged on the {len(check_columns)} columns it does have"
        )
    if proc.returncode != 0:
        warnings.append(f"inference_analysis.py exited {proc.returncode} but left a scoring table")

    rows = []
    for index, record in enumerate(frame.to_dict(orient="records")):
        qc_flags = []
        raw_id = record.get(ID_COLUMN)
        target_id = f"row_{index}" if raw_id is None or pd.isna(raw_id) else str(raw_id)

        raw_rmsd = record.get(RMSD_COLUMN)
        rmsd = None if raw_rmsd is None or pd.isna(raw_rmsd) else float(raw_rmsd)

        raw_hit = record.get(ACCURACY_COLUMN)
        if raw_hit is None or pd.isna(raw_hit):
            within_2a = False
            qc_flags.append(f"{ACCURACY_COLUMN}_missing")
        else:
            within_2a = bool(raw_hit)

        checks = {}
        for column in check_columns:
            value = record.get(column)
            checks[column] = None if value is None or pd.isna(value) else bool(value)

        rows.append(
            {
                "target_id": target_id,
                "rmsd": rmsd,
                "rmsd_within_2a": within_2a,
                "checks": checks,
                "qc_flags": qc_flags,
            }
        )

    # The denominator for coverage: how many targets the method was asked for. For
    # the methods whose target list is a directory rather than a CSV, upstream
    # parses the input data directory instead, so there is no row count to read and
    # coverage is reported as unavailable rather than as a guess.
    n_targets_expected = 0
    input_csv = root / resolve_method_input_csv_path(method, dataset, pocket_only)
    if input_csv.is_file():
        n_targets_expected = len(pd.read_csv(input_csv))
    else:
        warnings.append(
            f"target count unavailable: {input_csv} is not a CSV file, so coverage cannot be computed"
        )

    return {
        "rows": rows,
        "n_targets_expected": n_targets_expected,
        "bust_results_csv": str(bust_csv),
        "check_columns": check_columns,
        "warnings": warnings,
        "posebench_version": getattr(posebench, "__version__", "") or str(request.get("upstream_ref") or ""),
    }


if __name__ == "__main__":
    _driver.main(run)
