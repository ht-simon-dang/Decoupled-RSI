"""OpenFE — relative binding free energies by hybrid-topology replica exchange in OpenMM.

OpenFE (https://github.com/OpenFreeEnergy/openfe, MIT) plans and executes
alchemical free-energy calculations. The protocol this adapter binds is
``RelativeHybridTopologyProtocol`` (``src/openfe/protocols/openmm_rfe/``): for one
edge A→B it builds a Perses-style hybrid topology, samples it across 11 lambda
windows with Hamiltonian replica exchange in OpenMM, and estimates ΔG with MBAR —
once with the ligand in the protein and once in water — so that
ΔΔG_bind = ΔG_complex − ΔG_solvent. Upstream's defaults, read from
``RelativeHybridTopologyProtocol._default_settings`` and ``IntegratorSettings`` in
the pinned checkout, are 3 independent repeats, 1 ns equilibration and 5 ns
production per window at a 4 fs timestep, 298.15 K and 1 bar, AM1-BCC ligand
charges via AmberTools. That is 1,250,000 production integrator steps per replica,
and 3 × 2 × 11 × 1,500,000 = 99,000,000 integrator steps per edge before any
wall-clock is spent on setup, minimisation or analysis. Wall time on this machine
has not been measured, because nothing here can run it yet.

**Status on this machine: UNAVAILABLE.** The blockers, each verified:

1. No interpreter on this machine has the backend. ``drug-agent``,
   ``drug-agent-boltz``, ``drug-agent-aizynth``, ``drug-agent-unimol`` and ``pfm``
   were each asked to import ``openfe``, ``gufe``, ``openmm``, ``openmmtools``,
   ``openff.toolkit``, ``openmmforcefields``, ``lomap``, ``kartograf``,
   ``konnektor``, ``pymbar`` and ``mdtraj``; every one raised
   ``ModuleNotFoundError`` for every module. The environment this spec looks for,
   ``drug-agent-openfe``, does not exist.
2. ``pip install openfe`` cannot fix that. The PyPI distribution named ``openfe``
   is version 0.0.12 of *"OpenFE: automated feature generation with expert-level
   performance"* (IIIS-Li-Group/OpenFE), requiring ``lightgbm``; it has no relation
   to Open Free Energy. The real package is on conda-forge only, and its
   ``environment.yml`` and ``docs/installation.rst`` say so.
3. The architecture blocker that stops several other upstreams does **not** apply,
   and did not on aarch64 either. Re-run on this x86-64 host on 2026-09-09 —
   ``conda create -n _openfe_dryrun --dry-run --override-channels -c conda-forge openfe``
   — resolved 420 packages for ``linux-64``: ``openfe 1.12.0`` (noarch),
   ``openmm 8.4.0`` (linux-64, py313), ``ambertools 24.8`` (linux-64, with
   ``cuda-nvrtc 13.3.33``), ``gufe 1.12.0``, ``kartograf 2.0.0``, ``konnektor
   0.4.0``, ``lomap2 3.3.0``, ``pymbar 4.2.0``, ``mdtraj 1.11.1``, ``openmmtools
   0.26.0``, ``openmmforcefields 0.15.1``, on Python 3.13.15 — the same versions the
   aarch64 solve returned, at a different package count because the two platforms
   pull different low-level libraries. The install is one command; it has still not
   been run.
4. Whether OpenMM's CUDA platform initialises on this GPU (NVIDIA RTX PRO 6000
   Blackwell Workstation Edition, sm_120, driver 595.71.05, CUDA 13.2) is
   unverified. OpenFE's ``OpenMMEngineSettings.compute_platform`` defaults to
   ``"cuda"``, and ``omm_compute.get_openmm_platform`` warns that any other platform
   "may significantly impact simulation performance". ``probe()`` answers this the
   moment the environment exists.
5. The platform holds no prepared receptor. ``.external/pdb/3I28.pdb`` and
   ``5AI0.pdb`` are raw PDB entries — no hydrogens, crystallographic ligand and
   waters still present (3I28: ligand ``34N``, 153 ``HOH`` records);
   ``.external/pockets/seh_3i28_pocket10.pdb`` is a 752-atom pocket cut whose
   chain breaks are uncapped. An RBFE complex leg needs a whole,
   protonated, ligand-free protein a force field can parameterise, and an SDF of
   ligands posed in its frame. Neither exists in this repository.
6. The driver, ``scripts/external/openfe_rbfe.py``, was written against the pinned
   checkout (``678bbd4a23b9``) and cross-checked against ``openfe v1.12.0`` and
   ``gufe v1.12.0`` sources for the calls that live in gufe, but it has never been
   executed: there is no interpreter to execute it in.

What the platform does instead: nothing. This capability reports UNAVAILABLE,
``backend="none"``, and every operation raises
:class:`~drug_agent.tools.base.ToolUnavailable`. No docking score, no MMFF strain
and no descriptor is returned in place of a ΔΔG.

**What it does once installed.** Two operations mirror the upstream CLI so that
their outputs are interchangeable with it.

``plan_network``
    ``openfe plan-rbfe-network``. Loads the ligand SDF exactly as
    ``openfecli/parameters/molecules.py`` does, the receptor via
    ``ProteinComponent.from_pdb_file``, assigns partial charges with
    ``bulk_assign_partial_charges``, maps ligands with Kartograf (upstream's
    default since v1.7.0, with the CLI's exact settings) or LOMAP, plans a
    minimal-spanning, minimal-redundant, maximal or radial network scored by
    ``default_lomap_score``, and builds the ``AlchemicalNetwork`` with
    ``RBFEAlchemicalNetworkPlanner``. It writes the same files the CLI writes —
    ``<name>.json``, ``ligand_network.graphml``, ``transformations/*.json`` — so
    ``openfe quickrun`` can execute any edge this adapter planned. The returned
    :class:`RBFENetworkPlan` lists each edge with its mapping score and the number
    of mapped atoms, which is where an operator decides whether an edge is worth
    99 million integrator steps.

``run_edges``
    ``openfe quickrun`` per leg: ``Transformation.from_json``, ``create()``,
    ``execute_DAG`` with ``raise_error=False`` and retries, ``protocol.gather``.
    One :class:`~drug_agent.contracts.structure.SimulationResult` per edge, the
    ``candidate_id`` being ligand B and ``metadata`` naming ligand A:

        ``ddG_bind_kcal_mol``                 ΔG_complex − ΔG_solvent
        ``ddG_bind_uncertainty_kcal_mol``     leg uncertainties in quadrature
        ``dG_complex_kcal_mol``               mean over repeats, complex leg
        ``dG_complex_uncertainty_kcal_mol``   unbiased std over repeats (upstream
                                              ``compute_uncertainty``, ddof=1)
        ``dG_solvent_kcal_mol``               mean over repeats, solvent leg
        ``dG_solvent_uncertainty_kcal_mol``   as above
        ``mbar_error_complex_kcal_mol``       mean per-repeat MBAR error
        ``mbar_error_solvent_kcal_mol``       as above
        ``forward_reverse_max_gap_kcal_mol``  largest |forward − reverse| estimate
                                              over the fractions MBAR evaluated
        ``n_repeats``, ``lambda_windows``, ``n_replicas``
        ``production_iterations_mean``        uncorrelated production samples
        ``equilibration_iterations_mean``     iterations discarded as equilibration

    ``n_steps`` is the production integrator steps of one replica
    (``production_length / timestep``). ``converged`` is true only when both legs'
    DAGs finished with no failed unit, every repeat's forward/reverse analysis was
    obtained from the full set of uncorrelated samples, and the estimate is
    finite. Anything less is reported with ``converged=False`` and a ``qc_flag``
    naming the leg — a failed leg is a fact about that edge, not a malfunction of
    the tool. Each leg also writes a ``<key>_results.json`` in ``quickrun``'s own
    format, so ``openfe gather`` reads this adapter's output unchanged.

``probe``
    Versions of ``openfe``, ``gufe``, ``openmm`` and the OpenMM platforms the
    interpreter can load. Cheap, real, and the answer to blocker 4.

**What it deliberately does not do.** It does not pose ligands: a ligand that is
not already in the receptor frame is rejected, not docked. It does not compute
absolute binding free energies, though upstream ships that protocol. It does not
run on arbitrary generated chemistry: the network planner needs a common core and
the protocol validates the mapping before it will build a hybrid system. It does
not resume an interrupted leg; ``openfe quickrun --resume`` on the same
transformation JSON does, which is why the plan is written in the CLI's layout. And
it does not rank: an edge's ΔΔG is a difference between two named compounds, and
turning a set of edges into per-ligand affinities is a separate estimation
(``cinnabar``) that this adapter leaves to the specialist agent.

**Why the checkout is required even though the package comes from conda-forge.**
The driver calls upstream entry points by name with named arguments, and upstream
renames both between releases (``environment.yml`` in the checkout already tracks
``gufe@main`` ahead of a release). The checkout at the pinned sha is what the
driver was written against; requiring it keeps that reference on disk to diff
against, and ``upstream_ref`` records the commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...contracts.structure import SimulationResult
from ...observability.toolmeter import measure, note
from .._ext.spec import ExternalSpec
from . import FreeEnergyError, FreeEnergyToolAdapter

#: This adapter's contract with OpenFE, not OpenFE's own version.
OPENFE_ADAPTER_VERSION = "1.0.0"

#: Budget for one ``run_edges`` driver call. An edge at upstream defaults is 99
#: million integrator steps across 66 replicas; a day is a ceiling, not an estimate.
RUN_TIMEOUT_S = 86_400.0
#: Budget for planning: AM1-BCC charging runs one semi-empirical calculation per
#: ligand and the network planner scores every candidate mapping.
PLAN_TIMEOUT_S = 3_600.0

#: Network planners the driver exposes, by the name the request carries.
NETWORKS: tuple[str, ...] = ("minimal_spanning", "minimal_redundant", "maximal", "radial")
#: Atom mappers the driver exposes.
MAPPERS: tuple[str, ...] = ("kartograf", "lomap")
#: The two legs of a binding edge. ``vacuum`` exists upstream for hydration only.
LEGS: tuple[str, ...] = ("complex", "solvent")

SPEC = ExternalSpec(
    name="OpenFE",
    env_prefix="OPENFE",
    checkout_relpath=".external/openfe",
    # The modules the plan and run paths import from the upstream tree, verified
    # present at the pinned commit.
    sentinel_files=(
        "src/openfe/__init__.py",
        "src/openfe/protocols/openmm_rfe/hybridtop_protocols.py",
        "src/openfe/protocols/openmm_rfe/hybridtop_units.py",
        "src/openfe/setup/alchemical_network_planner/relative_alchemical_network_planner.py",
        "src/openfe/setup/ligand_network_planning.py",
        "src/openfecli/commands/quickrun.py",
    ),
    driver_name="openfe_rbfe.py",
    selection_modules=("openfe",),
    # Every top-level package ``hybridtop_units.py``, ``openmm_utils/*`` and
    # ``setup/*`` import at module scope. Top-level names only: ``openff`` is a
    # namespace package holding ``toolkit``/``units``/``interchange``, and the
    # foreign-interpreter probe can see the namespace directory but not a dotted
    # submodule, so ``openff.toolkit`` would read as absent when it is installed.
    runtime_modules=(
        "openfe",
        "gufe",
        "openmm",
        "openmmtools",
        "openmmforcefields",
        "openff",
        "lomap",
        "kartograf",
        "konnektor",
        "pymbar",
        "mdtraj",
    ),
    conda_envs=("drug-agent-openfe",),
    upstream_url="https://github.com/OpenFreeEnergy/openfe",
    upstream_ref="678bbd4a23b9",
    license="OpenFE MIT (checkout LICENSE); conda-forge build; not installed here",
    install_hint="make openfe-install",
    needs_gpu=True,
)


@dataclass
class RBFEEdge:
    """One planned perturbation A→B and the transformation files that realise it."""

    ligand_a: str
    ligand_b: str
    #: ``default_lomap_score`` of the chosen mapping; higher is a better map.
    mapping_score: float | None = None
    n_mapped_atoms: int = 0
    #: Leg name (``complex``/``solvent``) to the transformation JSON the driver wrote.
    legs: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.ligand_a}__{self.ligand_b}"

    @property
    def complete(self) -> bool:
        """Both legs of the binding cycle were planned."""
        return all(leg in self.legs for leg in LEGS)


@dataclass
class RBFENetworkPlan:
    """The output of ``plan_network``: what would be simulated, and where it was written."""

    work_dir: str = ""
    protein_pdb: str = ""
    ligands: list[str] = field(default_factory=list)
    edges: list[RBFEEdge] = field(default_factory=list)
    network_json: str = ""
    ligand_network_graphml: str = ""
    mapper: str = ""
    network: str = ""
    engine_version: str = ""
    #: The protocol settings baked into every transformation, as plain values.
    settings: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers that say whether this network is worth running."""
        out: dict[str, float] = {
            "openfe_n_ligands": float(len(self.ligands)),
            "openfe_n_edges": float(self.n_edges),
            "openfe_n_complete_edges": float(sum(1 for e in self.edges if e.complete)),
        }
        scores = [e.mapping_score for e in self.edges if e.mapping_score is not None]
        if scores:
            out["openfe_mapping_score_mean"] = float(sum(scores) / len(scores))
            out["openfe_mapping_score_min"] = float(min(scores))
        return out


class OpenFEAdapter(FreeEnergyToolAdapter):
    """Relative binding free energies with OpenFE's hybrid-topology protocol."""

    capability_id = "binding.freeenergy.openfe"
    spec = SPEC
    engine = "openfe"
    version = OPENFE_ADAPTER_VERSION
    protocol_id = "rbfe.hybrid_topology.repex"

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        plan_timeout_s: float | None = None,
    ) -> None:
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else RUN_TIMEOUT_S)
        self.plan_timeout_s = float(plan_timeout_s if plan_timeout_s is not None else PLAN_TIMEOUT_S)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Relative binding free energies (OpenFE RBFE)",
            description=(
                "Alchemical ΔΔG_bind between pairs of closely related, pre-posed ligands: "
                "hybrid-topology Hamiltonian replica exchange in OpenMM over 11 lambda "
                "windows, complex and solvent legs, MBAR estimates with repeat-level "
                "uncertainties. Plans the network and runs edges; poses nothing, ranks "
                "nothing, and is not a screen for generated chemistry."
            ),
            input_schema_ref="RBFENetworkPlan",
            output_schema_ref="SimulationResult",
            # Replica exchange over Langevin dynamics with random initial velocities
            # and Monte Carlo swaps: two runs of the same edge give two estimates
            # inside the same error bar, not the same number.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "ligand pair with no common core, or a net-charge change the mapping cannot carry: "
                "the protocol's validate() refuses the edge before any simulation",
                "receptor with uncapped chain breaks or non-standard residues that "
                "openmmforcefields cannot parameterise",
                "AM1-BCC charging runs an AmberTools sqm calculation per ligand and fails on "
                "some tautomers and unusual valences",
                "OpenMM CUDA platform absent: OpenFE warns that a non-CUDA platform may "
                "significantly impact performance and the run proceeds on CPU",
                "insufficient sampling: forward and reverse estimates disagree, MBAR fails at "
                "low sample fractions, converged is False",
                "driver exceeds its wall-clock budget and is killed; this driver does not "
                "resume a partial leg (openfe quickrun --resume does)",
            ],
            validation_checks=[
                "every ligand has a unique name and 3D coordinates before planning starts",
                "an edge missing either leg is reported with converged=False and no ddG",
                "ddG is complex minus solvent with uncertainties in quadrature, as openfe gather computes it",
                "engine is 'openfe' on every result; n_steps is one replica's production integrator steps",
                "NaN and infinite estimates are dropped from observables rather than written",
            ],
        )

    # --- operations --------------------------------------------------------
    def probe(self) -> dict[str, Any]:
        """Backend versions and the OpenMM platforms the interpreter can load."""
        self.require_available()
        with measure(self.capability_id, "probe", backend=self.backend) as metered:
            run = self.ext.run_json({"operation": "probe"}, timeout_s=300.0, error_cls=FreeEnergyError)
            payload = dict(run.payload)
            metered["openfe_driver_seconds"] = float(run.seconds)
            metered["openfe_cuda_platform"] = (
                1.0 if "CUDA" in (payload.get("openmm_platforms") or []) else 0.0
            )
        return payload

    def plan_network(
        self,
        *,
        protein_pdb: str | Path,
        ligands_sdf: str | Path,
        work_dir: str | Path,
        cofactors_sdf: str | Path | None = None,
        mapper: str = "kartograf",
        network: str = "minimal_spanning",
        central_ligand: str | None = None,
        protocol_repeats: int = 3,
        production_ns: float | None = None,
        equilibration_ns: float | None = None,
        compute_platform: str | None = None,
        partial_charge_method: str = "am1bcc",
        n_processes: int = 1,
        name: str = "rbfe",
    ) -> RBFENetworkPlan:
        """Plan the ligand network and write every transformation under *work_dir*.

        *ligands_sdf* must hold ligands already posed in the receptor frame with
        explicit hydrogens and unique names; *protein_pdb* must be a prepared,
        ligand-free receptor. Nothing here poses, protonates or repairs.
        """
        self.require_available()
        with measure(self.capability_id, "plan_network", backend=self.backend) as metered:
            plan = self._plan_network(
                protein_pdb=Path(protein_pdb),
                ligands_sdf=Path(ligands_sdf),
                work_dir=Path(work_dir),
                cofactors_sdf=Path(cofactors_sdf) if cofactors_sdf else None,
                mapper=mapper,
                network=network,
                central_ligand=central_ligand,
                protocol_repeats=protocol_repeats,
                production_ns=production_ns,
                equilibration_ns=equilibration_ns,
                compute_platform=compute_platform,
                partial_charge_method=partial_charge_method,
                n_processes=n_processes,
                name=name,
            )
            for key, value in plan.metrics_for_ledger().items():
                metered[key] = value
            if plan.warnings:
                note(self.capability_id, "plan_network", plan.warnings[0][:200])
        return plan

    def run_edges(
        self,
        plan: RBFENetworkPlan,
        *,
        edges: list[RBFEEdge] | None = None,
        work_dir: str | Path | None = None,
        receptor_id: str | None = None,
        candidate_ids: dict[str, str] | None = None,
        n_retries: int = 2,
    ) -> list[SimulationResult]:
        """Execute both legs of each edge and close the cycle.

        *edges* defaults to every edge in *plan*; pass a subset to run the edges
        whose mapping scores justified the cost. *candidate_ids* maps ligand names
        to the platform's candidate ids; a ligand not in it keeps its SDF name.
        """
        self.require_available()
        with measure(self.capability_id, "run_edges", backend=self.backend) as metered:
            results = self._run_edges(
                plan,
                edges=list(edges) if edges is not None else list(plan.edges),
                work_dir=Path(work_dir) if work_dir else Path(plan.work_dir) / "results",
                receptor_id=receptor_id,
                candidate_ids=dict(candidate_ids or {}),
                n_retries=n_retries,
                metered=metered,
            )
        return results

    # --- real backend ------------------------------------------------------
    def _plan_network(
        self,
        *,
        protein_pdb: Path,
        ligands_sdf: Path,
        work_dir: Path,
        cofactors_sdf: Path | None,
        mapper: str,
        network: str,
        central_ligand: str | None,
        protocol_repeats: int,
        production_ns: float | None,
        equilibration_ns: float | None,
        compute_platform: str | None,
        partial_charge_method: str,
        n_processes: int,
        name: str,
    ) -> RBFENetworkPlan:
        if mapper not in MAPPERS:
            raise FreeEnergyError(f"unknown mapper {mapper!r}; choose one of {', '.join(MAPPERS)}")
        if network not in NETWORKS:
            raise FreeEnergyError(f"unknown network {network!r}; choose one of {', '.join(NETWORKS)}")
        if network == "radial" and not central_ligand:
            raise FreeEnergyError("a radial network needs central_ligand, the name of the hub ligand")
        for label, path in (("protein_pdb", protein_pdb), ("ligands_sdf", ligands_sdf)):
            if not path.is_file():
                raise FreeEnergyError(f"{label} not found at {path}")
        if cofactors_sdf is not None and not cofactors_sdf.is_file():
            raise FreeEnergyError(f"cofactors_sdf not found at {cofactors_sdf}")
        work_dir.mkdir(parents=True, exist_ok=True)

        request: dict[str, Any] = {
            "operation": "plan",
            "name": name,
            "protein_pdb": str(protein_pdb.resolve()),
            "ligands_sdf": str(ligands_sdf.resolve()),
            "cofactors_sdf": str(cofactors_sdf.resolve()) if cofactors_sdf else None,
            "work_dir": str(work_dir.resolve()),
            "mapper": mapper,
            "network": network,
            "central_ligand": central_ligand,
            "protocol_repeats": int(protocol_repeats),
            "production_ns": production_ns,
            "equilibration_ns": equilibration_ns,
            "compute_platform": compute_platform,
            "partial_charge_method": partial_charge_method,
            "n_processes": int(n_processes),
        }
        run = self.ext.run_json(request, timeout_s=self.plan_timeout_s, error_cls=FreeEnergyError)
        payload = run.payload
        edges = [
            RBFEEdge(
                ligand_a=str(item.get("ligand_a") or ""),
                ligand_b=str(item.get("ligand_b") or ""),
                mapping_score=self.finite(item.get("mapping_score")),
                n_mapped_atoms=int(item.get("n_mapped_atoms") or 0),
                legs={str(k): str(v) for k, v in dict(item.get("legs") or {}).items()},
            )
            for item in list(payload.get("edges") or [])
        ]
        plan = RBFENetworkPlan(
            work_dir=str(work_dir.resolve()),
            protein_pdb=str(protein_pdb.resolve()),
            ligands=[str(n) for n in list(payload.get("ligands") or [])],
            edges=edges,
            network_json=str(payload.get("network_json") or ""),
            ligand_network_graphml=str(payload.get("ligand_network_graphml") or ""),
            mapper=mapper,
            network=network,
            engine_version=str(payload.get("engine_version") or "unknown"),
            settings=dict(payload.get("settings") or {}),
            warnings=[str(w) for w in list(payload.get("warnings") or [])],
        )
        if not plan.edges:
            plan.warnings.append("the planner produced no edges: no pair of ligands shares a mappable core")
        return plan

    def _run_edges(
        self,
        plan: RBFENetworkPlan,
        *,
        edges: list[RBFEEdge],
        work_dir: Path,
        receptor_id: str | None,
        candidate_ids: dict[str, str],
        n_retries: int,
        metered: dict[str, float],
    ) -> list[SimulationResult]:
        if not edges:
            return []
        work_dir.mkdir(parents=True, exist_ok=True)
        request: dict[str, Any] = {
            "operation": "run",
            "work_dir": str(work_dir.resolve()),
            "n_retries": int(n_retries),
            "edges": [{"ligand_a": e.ligand_a, "ligand_b": e.ligand_b, "legs": dict(e.legs)} for e in edges],
        }
        run = self.run_driver(request, error_cls=FreeEnergyError)
        payload = run.payload
        engine_version = str(payload.get("engine_version") or plan.engine_version or "unknown")
        receptor = receptor_id or (Path(plan.protein_pdb).stem if plan.protein_pdb else None)

        results: list[SimulationResult] = []
        for item in list(payload.get("edges") or []):
            results.append(
                self._edge_result(
                    dict(item),
                    engine_version=engine_version,
                    receptor_id=receptor,
                    candidate_ids=candidate_ids,
                    plan_settings=plan.settings,
                )
            )

        metered["openfe_driver_seconds"] = float(run.seconds)
        metered["openfe_n_edges"] = float(len(results))
        converged = [r for r in results if r.converged]
        metered["openfe_n_converged"] = float(len(converged))
        if results:
            metered["openfe_fraction_converged"] = float(len(converged) / len(results))
        uncertainties = [
            r.observables["ddG_bind_uncertainty_kcal_mol"]
            for r in converged
            if "ddG_bind_uncertainty_kcal_mol" in r.observables
        ]
        if uncertainties:
            metered["openfe_ddg_uncertainty_mean_kcal_mol"] = float(sum(uncertainties) / len(uncertainties))
        for warning in list(payload.get("warnings") or [])[:3]:
            note(self.capability_id, "run_edges", str(warning)[:200])
        return results

    def _edge_result(
        self,
        item: dict[str, Any],
        *,
        engine_version: str,
        receptor_id: str | None,
        candidate_ids: dict[str, str],
        plan_settings: dict[str, Any],
    ) -> SimulationResult:
        """Close one edge's cycle from the driver's per-leg records."""
        ligand_a = str(item.get("ligand_a") or "")
        ligand_b = str(item.get("ligand_b") or "")
        legs: dict[str, dict[str, Any]] = {
            str(k): dict(v) for k, v in dict(item.get("legs") or {}).items() if isinstance(v, dict)
        }
        flags: list[str] = []
        observables: dict[str, Any] = {}
        converged = True
        n_steps = 0

        for leg in LEGS:
            record = legs.get(leg)
            if record is None:
                flags.append(f"leg_missing:{leg}")
                converged = False
                continue
            n_steps = max(n_steps, int(record.get("n_steps") or 0))
            if not record.get("ok"):
                flags.append(f"leg_failed:{leg}")
                converged = False
                continue
            dg = self.finite(record.get("dg_kcal_mol"))
            unc = self.finite(record.get("uncertainty_kcal_mol"))
            if dg is None or unc is None:
                flags.append(f"estimate_missing:{leg}")
                converged = False
                continue
            observables[f"dG_{leg}_kcal_mol"] = dg
            observables[f"dG_{leg}_uncertainty_kcal_mol"] = unc
            mbar = [self.finite(pair[1]) for pair in list(record.get("individual") or []) if len(pair) == 2]
            mbar_clean = [m for m in mbar if m is not None]
            if mbar_clean:
                observables[f"mbar_error_{leg}_kcal_mol"] = sum(mbar_clean) / len(mbar_clean)
            if record.get("forward_reverse_incomplete"):
                flags.append(f"mbar_incomplete:{leg}")
                converged = False
            gap = self.finite(record.get("forward_reverse_max_gap_kcal_mol"))
            if gap is not None:
                prev = observables.get("forward_reverse_max_gap_kcal_mol")
                observables["forward_reverse_max_gap_kcal_mol"] = gap if prev is None else max(prev, gap)
            for key in ("production_iterations", "equilibration_iterations"):
                values = [self.finite(v) for v in list(record.get(key) or [])]
                clean = [v for v in values if v is not None]
                if clean:
                    prev_mean = observables.get(f"{key}_mean")
                    mean = sum(clean) / len(clean)
                    observables[f"{key}_mean"] = mean if prev_mean is None else (prev_mean + mean) / 2.0

        if all(f"dG_{leg}_kcal_mol" in observables for leg in LEGS):
            ddg, ddg_unc = self.ddg_from_legs(
                observables["dG_complex_kcal_mol"],
                observables["dG_solvent_kcal_mol"],
                observables["dG_complex_uncertainty_kcal_mol"],
                observables["dG_solvent_uncertainty_kcal_mol"],
            )
            observables["ddG_bind_kcal_mol"] = ddg
            observables["ddG_bind_uncertainty_kcal_mol"] = ddg_unc
        else:
            flags.append("ddg_unavailable")
            converged = False

        first = next(iter(legs.values()), {})
        for key in ("n_repeats", "lambda_windows", "n_replicas"):
            value = self.finite(first.get(key, plan_settings.get(key)))
            if value is not None:
                observables[key] = value

        metadata: dict[str, Any] = {
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "candidate_id_a": candidate_ids.get(ligand_a, ligand_a),
            "protocol_class": "openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol",
            "upstream_ref": self.spec.upstream_ref,
            "legs": {
                leg: {
                    "ok": bool(rec.get("ok")),
                    "results_json": rec.get("results_json"),
                    "transformation": rec.get("transformation"),
                    "wall_seconds": self.finite(rec.get("wall_seconds")),
                    "error": rec.get("error"),
                }
                for leg, rec in legs.items()
            },
            "settings": dict(plan_settings),
        }
        return self.simulation_result(
            candidate_id=candidate_ids.get(ligand_b, ligand_b),
            receptor_id=receptor_id,
            engine_version=engine_version,
            n_steps=n_steps,
            observables=observables,
            converged=converged,
            qc_flags=flags,
            metadata=metadata,
            artifact_ref=str(item.get("edge_dir") or "") or None,
        )


__all__ = [
    "LEGS",
    "MAPPERS",
    "NETWORKS",
    "OPENFE_ADAPTER_VERSION",
    "PLAN_TIMEOUT_S",
    "RUN_TIMEOUT_S",
    "SPEC",
    "OpenFEAdapter",
    "RBFEEdge",
    "RBFENetworkPlan",
]
