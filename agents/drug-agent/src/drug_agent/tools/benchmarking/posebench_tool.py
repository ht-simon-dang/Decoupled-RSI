"""PoseBench — how often a docking or co-folding method is right, over a whole dataset.

PoseBench is benchmark infrastructure, not a scientific tool. It does not produce a
pose and it does not score a molecule. It takes a *method* that has already written
its predictions to disk, walks a fixed list of targets, pairs each prediction with
the crystal ligand, runs PoseBusters in ``redock`` mode over every pair, and reports
what fraction of the dataset that method got right. The unit of its answer is a
method, not a candidate, and this adapter models it that way: ``evaluate`` returns a
:class:`~drug_agent.tools.benchmarking.MethodBenchmark`, never a
:class:`~drug_agent.contracts.structure.Pose`.

**Why it cannot run on this machine.** Five blockers, each verified against the
checkout at ``.external/PoseBench`` (upstream ``c5d728d2a31d``, MIT, 654 MB on disk).

1. *No environment here can import the evaluation path.*
   ``posebench/analysis/inference_analysis.py`` imports ``hydra``, ``omegaconf``,
   ``rootutils`` and ``beartype`` at module scope, and the
   ``posebench.utils.data_utils`` it pulls in adds ``prody``, ``biopandas`` and
   ``Bio``. Re-probed on 2026-09-09 across all twenty-five interpreters under this
   host's conda root: ``rootutils``, ``beartype``, ``prody`` and ``biopandas`` are in
   none of them. The file that would build the environment is a better fit here than
   it was on aarch64 and is still not usable:
   ``environments/posebench_environment.yaml`` is a fully pinned linux-64 export —
   it names ``binutils_impl_linux-64=2.39=he00db2b_1``,
   ``gcc_linux-64=10.4.0=h9215b83_13``, ``kernel-headers_linux-64=2.6.32=he073ed8_17``
   and ``cudatoolkit=11.8.0=h4ba93d1_13``. This machine **is** linux-64, so the export
   now names the right platform, but ``cudatoolkit 11.8`` predates this GPU's sm_120
   by three major CUDA versions. The analysis path itself is pure Python over pandas,
   RDKit and PoseBusters and never touches CUDA, so a hand-built environment is a real
   option; the pinned export is not, for a reason that moved from architecture to
   accelerator rather than going away.

2. *The benchmark datasets are not downloaded.* The four the harness accepts —
   PoseBusters Benchmark, Astex Diverse, DockGen and CASP15 — live in
   ``data/<dataset>_set/`` and none of those directories exists. The whole ``data/``
   tree in the checkout is 408 KB: an id list, and per-dataset input tables under
   ``data/test_cases/`` that name files which are not there. Upstream's README gives
   the download as four archives from Zenodo record 19138652, "~3 GB" total.

3. *There is nothing to score.* An evaluation reads a method's predictions from
   ``forks/<Method>/inference/<method>_<dataset>_output*_<repeat>/``; no such
   directory exists under any fork. Upstream's README publishes the reference
   predictions for ten methods as a second Zenodo download, "~19 GB" total.
   Producing them locally instead is a larger commitment than downloading them:
   the README's own figures are "~80 GB total" for the method conda environments
   and "~8.25 GB total" for their checkpoints, all pinned to CUDA 11.8 on linux-64.

4. *CASP15 needs a second interpreter that is not here.*
   ``configs/analysis/inference_analysis_casp.yaml`` shells its lDDT-PLI scoring out
   to ``${oc.env:HOME}/mambaforge/envs/casp15_ligand_scoring/bin/python3``. That
   environment does not exist on this host, so ``dataset="casp15"`` is blocked
   separately from and in addition to blockers 1-3.

5. *The shipped per-method input tables are not portable.* Every path in, for
   example, ``forks/DiffDock/inference/diffdock_posebusters_benchmark_inputs.csv``
   is absolute and rooted at the authors' cluster
   (``/cluster/pixstor/chengji-lab/acmwhb/...``). The evaluation path survives this,
   because ``create_mol_table`` reads only the complex-name column and rebuilds
   every file path from ``input_data_dir``; running the *prediction* half would
   require regenerating those tables.

**What the platform does instead: it does not benchmark.** No method-level success
rate is published anywhere in this package, and none is estimated. The related
capability that is real here is ``structure.validate.posebusters``, which runs the
same PoseBusters battery over a single pose — the same checks, one molecule at a
time, and no rate over a dataset. A benchmark number invented from anything other
than scored targets would be the easiest number in this repository to mistake for
evidence about a method, so there is no surrogate.

**What it would compute, once installed.** One row per target, then rates over those
rows. Per target: ``rmsd`` (Å, symmetry-corrected against the crystal ligand),
``rmsd_within_2a`` (upstream's accuracy verdict, its ``rmsd_≤_2å`` column), and
``checks`` — the validity columns upstream fixes per dataset in
``BUST_TEST_COLUMNS``: sanitization, molecular formula and bonds, tetrahedral
chirality, double-bond stereochemistry, bond lengths and angles, internal steric
clash, aromatic-ring and double-bond flatness, internal energy, minimum distance to
protein and to organic and inorganic cofactors, and volume overlap with each. DockGen
uses a shorter list, because its crystal ligands are PDB rather than SDF and the
graph-based assertions cannot be made. Over those rows: ``rmsd_within_2a_rate``,
``pb_valid_rate``, ``success_rate`` (the conjunction — upstream's headline pairing),
``median_rmsd`` (Å) and ``coverage`` (scored targets over requested targets, unitless).

**Coverage is not decoration.** Upstream drops every target whose prediction file is
missing before scoring, after logging a warning, so a rate computed from what
remains is a rate over the targets the method did not crash on. This adapter
therefore reads the method's own input table for the denominator and reports
``coverage`` beside every rate. In this checkout that denominator is 308 rows for
PoseBusters Benchmark, 85 for Astex Diverse and 122 for DockGen in DiffDock's tables;
the ensemble tables under ``data/test_cases/`` list 428, 85, 189 and 19.

**Data provenance: a benchmark result inherits terms this repository did not set.**
The preprocessed datasets do not ship crystal receptors alone. Upstream's README
states that the Zenodo archives "provide pre-holo-aligned protein structures
predicted by AlphaFold 3 (and alternatively MIT-licensed ESMFold)", and that "users
must ensure their usage of such predicted protein structures from AlphaFold 3 aligns
with AlphaFold 3's Terms of Use". PoseBench's own licence is MIT and does not cover
that. So before any number produced here leaves this machine, three things have to be
checked separately: which structure set the ``data/<dataset>_set/`` archive actually
contains (the AlphaFold 3 one or the ESMFold one — the directory name
``<dataset>_holo_aligned_predicted_structures`` is the same either way), the terms
attached to that generator's outputs, and the licence of the downloaded method
predictions, which are ten other projects' output and not PoseBench's. Those terms
are carried onto every result as ``data_restrictions`` rather than left on a card,
because a rate copied into a report loses its card.

**What it deliberately does not do.** It does not run inference: producing the
predictions is each method's own job, in each method's own environment, and this
adapter scores what is already on disk. It does not rank methods, because a single
dataset does not establish an ordering. It does not turn a benchmark result into a
capability — that some method scores well on PoseBusters Benchmark says nothing
about whether it runs here.
"""

from __future__ import annotations

from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import measure, note
from .._ext.spec import ExternalSpec
from . import BenchmarkError, BenchmarkToolAdapter, MethodBenchmark

#: This adapter's contract with PoseBench, not PoseBench's own version.
POSEBENCH_ADAPTER_VERSION = "1.0.0"

#: The datasets the harness accepts, from the NOTE on the ``dataset`` key in
#: ``configs/analysis/inference_analysis.yaml``.
DATASETS: tuple[str, ...] = ("posebusters_benchmark", "astex_diverse", "dockgen", "casp15")

#: The methods the harness can score, from the NOTE on the ``method`` key in the
#: same config. These are names of *other people's* predictions on disk; none of
#: them is a backend this package can run.
METHODS: tuple[str, ...] = (
    "diffdock",
    "fabind",
    "dynamicbind",
    "neuralplexer",
    "flowdock",
    "rfaa",
    "chai-lab",
    "boltz",
    "alphafold3",
    "vina",
    "ensemble",
)

#: Terms a result inherits from the data it was computed over. Quoted from
#: upstream's README rather than paraphrased, because a paraphrase of a licence
#: obligation is not one.
DATA_RESTRICTIONS = (
    "PoseBench's code is MIT, but its preprocessed datasets are not only crystal "
    "structures: upstream states the Zenodo archives 'provide pre-holo-aligned protein "
    "structures predicted by AlphaFold 3 (and alternatively MIT-licensed ESMFold)' and "
    "that 'users must ensure their usage of such predicted protein structures from "
    "AlphaFold 3 aligns with AlphaFold 3's Terms of Use'. Any benchmark number computed "
    "here inherits those terms, plus the licence of whichever method's downloaded "
    "predictions were scored. Check which structure set the local data/<dataset>_set "
    "archive contains before publishing a result."
)

SPEC = ExternalSpec(
    name="PoseBench",
    env_prefix="POSEBENCH",
    checkout_relpath=".external/PoseBench",
    # Every one of these is on the evaluation path: `.project-root` is the indicator
    # `rootutils.setup_root` looks for, the config is what hydra composes, and the
    # three modules are what the entry point imports before it does anything.
    sentinel_files=(
        ".project-root",
        "posebench/__init__.py",
        "posebench/analysis/inference_analysis.py",
        "posebench/utils/data_utils.py",
        "configs/analysis/inference_analysis.yaml",
        "data/posebusters_pdb_ccd_ids.txt",
    ),
    driver_name="posebench_evaluate.py",
    # `rootutils` is the narrow marker of a PoseBench environment: it is the import
    # the entry point uses to find the project root and almost nothing else needs it.
    selection_modules=("rootutils",),
    # What a broken install actually loses, read off the import statements of
    # `analysis/inference_analysis.py` and the `utils/data_utils` it imports —
    # not the headline dependency.
    runtime_modules=(
        "hydra",
        "omegaconf",
        "rootutils",
        "beartype",
        "prody",
        "biopandas",
        "Bio",
        "pandas",
        "numpy",
        "requests",
        "tqdm",
        "rdkit",
        "posebusters",
    ),
    # A benchmark's "weights" are its dataset: without one there is nothing to score
    # and no rate to report, so the readiness probe checks for the default dataset of
    # `configs/analysis/inference_analysis.yaml`. Each operation additionally checks
    # the dataset it was actually asked for, so a host holding only DockGen is not
    # blocked from benchmarking DockGen.
    weight_relpaths=("data/posebusters_benchmark_set",),
    conda_envs=("posebench", "PoseBench", "drug-agent"),
    upstream_url="https://github.com/BioinfoMachineLearning/PoseBench",
    upstream_ref="c5d728d2a31d",
    license="PoseBench MIT (LICENSE, Copyright (c) 2024 BioinfoMachineLearning)",
    install_hint=(
        "pip install -e .external/PoseBench into a dedicated environment, then unpack the "
        "Zenodo 19138652 dataset and prediction archives under .external/PoseBench/"
    ),
    # The scoring itself is PoseBusters over CPU. The methods being scored need GPUs;
    # this capability does not run them.
    needs_gpu=False,
)


class PoseBenchAdapter(BenchmarkToolAdapter):
    """Dataset-level evaluation of structure-prediction and docking methods."""

    capability_id = "benchmark.pose.posebench"
    spec = SPEC
    engine = "posebench"
    version = POSEBENCH_ADAPTER_VERSION
    produces_nothing = "benchmarks nothing"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # A budget, not a measurement: one evaluation is the PoseBusters redock
        # battery over every target in the dataset, and the largest of the four
        # lists 428 of them.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 7200.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Structure-prediction and docking benchmarks (PoseBench)",
            description=(
                "Runs PoseBench's evaluation over one dataset (PoseBusters Benchmark, Astex "
                "Diverse, DockGen or CASP15) for a method whose predictions are already on "
                "disk, and reports per-target RMSD and PoseBusters validity plus the rates "
                "over them. Scores methods, not molecules: it produces no pose and runs no "
                "inference."
            ),
            output_schema_ref="MethodBenchmark",
            # The scoring is deterministic over fixed prediction files. The
            # predictions themselves are usually not, which is why the repeat index
            # is part of the config recorded on every result.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.EXPENSIVE,
            data_restrictions=DATA_RESTRICTIONS,
            known_failure_modes=[
                "dataset not downloaded, so there are no crystal ligands to score against",
                "predictions directory absent or empty for the requested method and repeat index",
                "method produced files for only part of the dataset, which upstream drops before "
                "scoring; reported as coverage below 1.0 rather than as an error",
                "a validity column with no value, counted as not valid and flagged rather than "
                "passed through",
                "CASP15 scoring shells out to a separate casp15_ligand_scoring interpreter",
                "one evaluation is PoseBusters over the whole dataset and can exceed its budget",
            ],
            validation_checks=[
                "every rate is computed from counted targets, never from a backend's own claim",
                "coverage is reported beside every rate, with the method's input table as denominator",
                "rates are absent rather than zero when nothing was scored",
                "success_rate is the conjunction of accuracy and validity, not either alone",
            ],
        )

    # --- operations --------------------------------------------------------
    def evaluate(
        self,
        *,
        method: str,
        dataset: str = "posebusters_benchmark",
        repeat_index: int = 1,
        pocket_only_baseline: bool = False,
        v1_baseline: bool = False,
        relax_protein: bool = False,
        force_rescore: bool = False,
        full_report: bool = True,
        vina_binding_site_method: str = "p2rank",
        ensemble_ranking_method: str = "consensus",
    ) -> MethodBenchmark:
        """Score one method's saved predictions over one dataset.

        Runs nothing itself: the predictions must already exist under the method's
        fork directory for this ``repeat_index``. Returns the rates and the counts
        they were computed from.
        """
        self.require_available()
        request = self._request(
            method=method,
            dataset=dataset,
            repeat_index=repeat_index,
            pocket_only_baseline=pocket_only_baseline,
            v1_baseline=v1_baseline,
            relax_protein=relax_protein,
            force_rescore=force_rescore,
            full_report=full_report,
            vina_binding_site_method=vina_binding_site_method,
            ensemble_ranking_method=ensemble_ranking_method,
        )
        with measure(self.capability_id, "evaluate", backend=self.backend) as metered:
            report = self._evaluate(request, metered)
        return report

    def compare(
        self,
        methods: list[str],
        *,
        dataset: str = "posebusters_benchmark",
        repeat_index: int = 1,
        pocket_only_baseline: bool = False,
        v1_baseline: bool = False,
        relax_protein: bool = False,
        force_rescore: bool = False,
        full_report: bool = True,
        vina_binding_site_method: str = "p2rank",
        ensemble_ranking_method: str = "consensus",
    ) -> list[MethodBenchmark]:
        """Evaluate several methods on the same dataset and return each one's metrics.

        One method failing does not stop the rest: its summary comes back with no
        targets, the failure in ``warnings``, and every rate absent rather than
        zero — a method that could not be scored has not scored zero.
        """
        self.require_available()
        with measure(self.capability_id, "compare", backend=self.backend) as metered:
            reports: list[MethodBenchmark] = []
            for method in methods:
                request = self._request(
                    method=method,
                    dataset=dataset,
                    repeat_index=repeat_index,
                    pocket_only_baseline=pocket_only_baseline,
                    v1_baseline=v1_baseline,
                    relax_protein=relax_protein,
                    force_rescore=force_rescore,
                    full_report=full_report,
                    vina_binding_site_method=vina_binding_site_method,
                    ensemble_ranking_method=ensemble_ranking_method,
                )
                try:
                    reports.append(self._evaluate(request, {}))
                except BenchmarkError as exc:
                    note(self.capability_id, "compare", f"{method}: {exc}"[:200])
                    reports.append(
                        self.summarise(
                            [],
                            method=method,
                            dataset=dataset,
                            n_targets_expected=0,
                            engine_version="",
                            config=dict(request),
                            warnings=[str(exc)],
                            data_restrictions=DATA_RESTRICTIONS,
                        )
                    )
            metered["benchmark_n_methods"] = float(len(reports))
            scored = [r for r in reports if r.n_scored]
            metered["benchmark_n_methods_scored"] = float(len(scored))
        return reports

    # --- real backend ------------------------------------------------------
    def _request(self, **fields: Any) -> dict[str, Any]:
        """Validate the arguments against the harness' own accepted values."""
        method = str(fields["method"])
        dataset = str(fields["dataset"])
        if method not in METHODS:
            raise ValueError(f"unknown PoseBench method {method!r}; expected one of {list(METHODS)}")
        if dataset not in DATASETS:
            raise ValueError(f"unknown PoseBench dataset {dataset!r}; expected one of {list(DATASETS)}")
        request = dict(fields)
        request["method"] = method
        request["dataset"] = dataset
        request["upstream_ref"] = self.spec.upstream_ref
        return request

    def _evaluate(self, request: dict[str, Any], metered: dict[str, float]) -> MethodBenchmark:
        dataset = str(request["dataset"])
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() has already refused
            raise BenchmarkError("no PoseBench checkout")
        dataset_dir = root / "data" / f"{dataset}_set"
        if not dataset_dir.is_dir():
            raise BenchmarkError(
                f"the {dataset} dataset is not downloaded: {dataset_dir} does not exist. "
                f"Fetch {dataset}_set.tar.gz from https://doi.org/10.5281/zenodo.19138652 "
                f"(~3 GB for all four datasets) and extract it under {root}."
            )
        run = self.run_driver(request, error_cls=BenchmarkError)
        payload = run.payload
        rows = payload.get("rows") or []
        report = self.summarise(
            rows,
            method=str(request["method"]),
            dataset=dataset,
            n_targets_expected=int(payload.get("n_targets_expected") or 0),
            engine_version=str(payload.get("posebench_version") or ""),
            results_path=str(payload.get("bust_results_csv") or ""),
            config=dict(request),
            warnings=list(payload.get("warnings") or []),
            data_restrictions=DATA_RESTRICTIONS,
        )
        for key, value in report.metrics_for_ledger().items():
            metered[key] = value
        metered["benchmark_driver_seconds"] = float(run.seconds)
        return report


__all__ = [
    "DATASETS",
    "DATA_RESTRICTIONS",
    "METHODS",
    "POSEBENCH_ADAPTER_VERSION",
    "SPEC",
    "PoseBenchAdapter",
]
