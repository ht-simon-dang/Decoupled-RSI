"""Plan and run relative binding free-energy edges with OpenFE — the drug-agent ⇄ OpenFE bridge.

This file is executed by the interpreter that has OpenFE installed — on this
machine that would be the conda environment ``drug-agent-openfe`` created from
conda-forge, which pairs ``openfe 1.12.0`` with ``openmm 8.4.0`` and ``gufe 1.12.0``
on Python 3.13. The platform's own environment (Python 3.12, rdkit 2026.03) has
none of it, so the JSON request/response pair on disk is the whole interface.

It imports nothing from ``drug_agent``: only the standard library, ``openfe`` and
the packages OpenFE itself depends on, exactly as ``scripts/external/_driver.py``
requires.

**What it mirrors, call for call.** ``plan`` is ``openfe plan-rbfe-network``
(``src/openfecli/commands/plan_rbfe_network.py`` → ``plan_rbfe_network_main``) and
writes the same files ``plan_alchemical_network_output`` writes, so ``openfe
quickrun`` can execute any transformation planned here. ``run`` is ``openfe
quickrun`` per leg (``src/openfecli/commands/quickrun.py``): ``Transformation.from_json``,
``create()``, ``execute_DAG(..., raise_error=False, n_retries=...)``,
``protocol.gather([dagresult])``, then ``get_estimate()``/``get_uncertainty()`` and
a ``<key>_results.json`` in quickrun's own layout so ``openfe gather`` reads it.
The entry points and keyword names were read from the pinned checkout
(``678bbd4a23b9``) and, for the calls that live in gufe, from ``gufe v1.12.0``.

**What it has not done.** This driver has never been executed: no interpreter on
the machine it was written on can import ``openfe``. Everything above is a claim
about source code, not about a run.

Request JSON — common:
    operation           "probe" | "plan" | "run"
    out_dir             directory to write result.json (injected by the caller)

Request JSON — plan:
    name                network name; prefix of every transformation name
    protein_pdb         prepared, ligand-free receptor PDB (or PDBx/mmCIF)
    ligands_sdf         ligands posed in the receptor frame, explicit H, unique names
    cofactors_sdf       optional SDF of cofactors kept in the complex leg
    work_dir            where <name>.json, ligand_network.graphml, transformations/ go
    mapper              "kartograf" (CLI default) | "lomap"
    network             "minimal_spanning" | "minimal_redundant" | "maximal" | "radial"
    central_ligand      hub ligand name; required for "radial"
    protocol_repeats    independent repeats per leg (upstream default 3)
    production_ns       override of production_length (upstream default 5.0)
    equilibration_ns    override of equilibration_length (upstream default 1.0)
    compute_platform    "cuda" | "opencl" | "cpu" (upstream default "cuda")
    partial_charge_method  "am1bcc" | "am1bccelf10" | "nagl" | "espaloma"
    n_processes         processors for partial-charge assignment

Request JSON — run:
    work_dir            where each <ligand_a>__<ligand_b>/<leg>/ directory is made
    n_retries           execute_DAG retries per failed unit
    edges[]             {ligand_a, ligand_b, legs: {complex: path, solvent: path}}

Response JSON — plan (``out_dir/result.json``):
    engine_version, ligands[], network_json, ligand_network_graphml, settings{}
    edges[]             {ligand_a, ligand_b, mapping_score, n_mapped_atoms, legs{}}
    warnings[]

Response JSON — run:
    engine_version, warnings[]
    edges[]             {ligand_a, ligand_b, edge_dir, legs: {leg: {ok, dg_kcal_mol,
                        uncertainty_kcal_mol, individual[[dg, mbar_err]],
                        forward_reverse_max_gap_kcal_mol, forward_reverse_incomplete,
                        equilibration_iterations[], production_iterations[], n_steps,
                        n_repeats, lambda_windows, n_replicas, wall_seconds,
                        results_json, transformation, error}}}
"""

from __future__ import annotations

import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path insert above: the shared driver contract lives beside this file.
import _driver

#: The two legs an RBFE edge is made of, by the suffix EasyChemicalSystemGenerator
#: puts on the ChemicalSystem name (``<ligand>_complex`` / ``<ligand>_solvent``).
_LEG_SUFFIXES = {"_complex": "complex", "_solvent": "solvent", "_vacuum": "vacuum"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _versions() -> dict[str, str]:
    """Installed versions of the packages whose behaviour the results depend on."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for dist in (
        "openfe",
        "gufe",
        "openmm",
        "openmmtools",
        "openff-toolkit",
        "lomap2",
        "kartograf",
        "pymbar",
    ):
        try:
            out[dist] = str(version(dist))
        except PackageNotFoundError:
            out[dist] = "unknown"
    return out


def _kcal(quantity: Any) -> float | None:
    """A Quantity's magnitude in kcal/mol, or ``None`` when it is not finite."""
    from openff.units import unit

    try:
        value = float(quantity.to(unit.kilocalorie_per_mole).m)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_ligands(sdf: str) -> list[Any]:
    """SmallMoleculeComponents from an SDF, exactly as ``openfecli`` loads them.

    ``removeHs=False`` because the protocol needs explicit hydrogens; a record that
    RDKit cannot parse, that has no conformer, or that has no name is a preparation
    error and stops the plan — an RBFE network silently planned without one of its
    ligands is a different network from the one that was asked for.
    """
    from openfe import SmallMoleculeComponent
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    ligands: list[Any] = []
    for index, mol in enumerate(supplier):
        if mol is None:
            raise ValueError(f"record {index} of {sdf} could not be parsed by RDKit")
        if mol.GetNumConformers() == 0:
            raise ValueError(f"record {index} of {sdf} has no 3D coordinates; ligands must be posed")
        ligands.append(SmallMoleculeComponent(mol))
    names = [str(lig.name) for lig in ligands]
    if any(not name for name in names):
        raise ValueError(f"every ligand in {sdf} needs a name (the SDF title line or an ofe-name property)")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"ligand names in {sdf} must be unique; duplicated: {', '.join(duplicates)}")
    return ligands


def _load_protein(path: str) -> Any:
    """``ProteinComponent`` from PDB or PDBx, as ``openfecli/parameters/protein.py`` does."""
    from openfe import ProteinComponent

    lower = path.lower()
    if lower.endswith((".cif", ".pdbx")):
        return ProteinComponent.from_pdbx_file(path)
    return ProteinComponent.from_pdb_file(path)


def _build_mapper(name: str) -> Any:
    """The atom mapper with the settings upstream itself uses for it.

    Kartograf's kwargs are the CLI's defaults
    (``openfecli/parameters/plan_network_options.py``); LOMAP's are
    ``RelativeAlchemicalNetworkPlanner``'s defaults.
    """
    from openfe.setup import KartografAtomMapper, LomapAtomMapper

    if name == "lomap":
        return LomapAtomMapper(time=20, threed=True, max3d=1.0, element_change=True, shift=False)
    if name == "kartograf":
        return KartografAtomMapper(
            atom_max_distance=0.95,
            atom_map_hydrogens=True,
            map_hydrogens_on_hydrogens_only=True,
            map_exact_ring_matches_only=True,
            allow_partial_fused_rings=True,
            allow_bond_breaks=False,
        )
    raise ValueError(f"unknown mapper {name!r}; expected 'kartograf' or 'lomap'")


def _build_network_planner(name: str, central_ligand: str | None) -> Any:
    """The ligand-network planner, bound the way ``plan_network_options`` binds it."""
    from openfe.setup.ligand_network_planning import (
        generate_maximal_network,
        generate_minimal_redundant_network,
        generate_minimal_spanning_network,
        generate_radial_network,
    )

    planners = {
        "minimal_spanning": generate_minimal_spanning_network,
        "minimal_redundant": generate_minimal_redundant_network,
        "maximal": generate_maximal_network,
    }
    if name == "radial":
        if not central_ligand:
            raise ValueError("a radial network needs central_ligand")
        return partial(generate_radial_network, central_ligand=central_ligand)
    try:
        return planners[name]
    except KeyError:
        raise ValueError(
            f"unknown network {name!r}; expected one of {', '.join([*planners, 'radial'])}"
        ) from None


def _build_protocol(request: dict[str, Any]) -> Any:
    """``RelativeHybridTopologyProtocol`` on upstream's default settings plus the overrides asked for."""
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    from openff.units import unit

    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.protocol_repeats = int(request.get("protocol_repeats") or settings.protocol_repeats)
    if request.get("production_ns") is not None:
        settings.simulation_settings.production_length = float(request["production_ns"]) * unit.nanosecond
    if request.get("equilibration_ns") is not None:
        settings.simulation_settings.equilibration_length = (
            float(request["equilibration_ns"]) * unit.nanosecond
        )
    if request.get("compute_platform"):
        settings.engine_settings.compute_platform = str(request["compute_platform"])
    return RelativeHybridTopologyProtocol(settings)


def _settings_summary(settings: Any, partial_charge_method: str) -> dict[str, Any]:
    """The protocol settings that decide cost and meaning, as plain JSON values."""
    from openff.units import unit

    sim = settings.simulation_settings
    integrator = settings.integrator_settings
    production_fs = float(sim.production_length.to(unit.femtosecond).m)
    timestep_fs = float(integrator.timestep.to(unit.femtosecond).m)
    return {
        "protocol_repeats": int(settings.protocol_repeats),
        "lambda_windows": int(settings.lambda_settings.lambda_windows),
        "n_replicas": int(sim.n_replicas),
        "sampler_method": str(sim.sampler_method),
        "equilibration_ns": float(sim.equilibration_length.to(unit.nanosecond).m),
        "production_ns": float(sim.production_length.to(unit.nanosecond).m),
        "timestep_fs": timestep_fs,
        "n_steps_production_per_replica": round(production_fs / timestep_fs),
        "temperature_K": float(settings.thermo_settings.temperature.to(unit.kelvin).m),
        "compute_platform": settings.engine_settings.compute_platform,
        "forcefields": [str(ff) for ff in settings.forcefield_settings.forcefields],
        "small_molecule_forcefield": str(settings.forcefield_settings.small_molecule_forcefield),
        "partial_charge_method": partial_charge_method,
    }


def _leg_of(chemical_system: Any) -> str:
    """Which leg a ChemicalSystem belongs to, from the suffix the generator gave it."""
    name = str(getattr(chemical_system, "name", "") or "")
    for suffix, leg in _LEG_SUFFIXES.items():
        if name.endswith(suffix):
            return leg
    return "unknown"


def _format_exception(exception: Any) -> str:
    """A ``ProtocolUnitFailure.exception`` as one line, as quickrun formats it."""
    try:
        return f"{exception[0]}: {exception[1][0]}"
    except (IndexError, KeyError, TypeError):
        return str(exception)


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------
def _probe(request: dict[str, Any]) -> dict[str, Any]:
    """Versions and the OpenMM platforms this interpreter can load."""
    import openmm

    platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": _versions(),
        "openmm_platforms": platforms,
    }


def _plan(request: dict[str, Any]) -> dict[str, Any]:
    """``openfe plan-rbfe-network`` without the click layer."""
    from openfe import SolventComponent
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
    from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
    from openfe.setup.alchemical_network_planner import RBFEAlchemicalNetworkPlanner
    from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score

    warnings: list[str] = []
    name = str(request.get("name") or "rbfe")
    work_dir = Path(str(request.get("work_dir") or "")).resolve()
    if not str(request.get("work_dir") or ""):
        raise ValueError("request carries no work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    ligands = _load_ligands(str(request["ligands_sdf"]))
    if len(ligands) < 2:
        raise ValueError("a relative network needs at least two ligands")
    protein = _load_protein(str(request["protein_pdb"]))
    cofactors = _load_ligands(str(request["cofactors_sdf"])) if request.get("cofactors_sdf") else []
    # Water with Na+/Cl- at 0.15 M, neutralised: SolventComponent's own defaults
    # and what plan-rbfe-network uses.
    solvent = SolventComponent()

    charge_settings = OpenFFPartialChargeSettings()
    method = str(request.get("partial_charge_method") or charge_settings.partial_charge_method)
    charge_settings.partial_charge_method = method
    n_processes = max(1, int(request.get("n_processes") or 1))
    charged = bulk_assign_partial_charges(
        molecules=ligands,
        overwrite=False,
        method=charge_settings.partial_charge_method,
        toolkit_backend=charge_settings.off_toolkit_backend,
        generate_n_conformers=charge_settings.number_of_conformers,
        nagl_model=charge_settings.nagl_model,
        processors=n_processes,
    )
    if cofactors:
        cofactors = bulk_assign_partial_charges(
            molecules=cofactors,
            overwrite=False,
            method=charge_settings.partial_charge_method,
            toolkit_backend=charge_settings.off_toolkit_backend,
            generate_n_conformers=charge_settings.number_of_conformers,
            nagl_model=charge_settings.nagl_model,
            processors=n_processes,
        )

    protocol = _build_protocol(request)
    planner = RBFEAlchemicalNetworkPlanner(
        name=name,
        mappers=[_build_mapper(str(request.get("mapper") or "kartograf"))],
        mapping_scorer=default_lomap_score,
        ligand_network_planner=_build_network_planner(
            str(request.get("network") or "minimal_spanning"), request.get("central_ligand")
        ),
        protocol=protocol,
    )
    alchemical_network = planner(ligands=charged, solvent=solvent, protein=protein, cofactors=cofactors)
    # The CLI reads the same private attribute; the planner exposes no public one.
    ligand_network = planner._ligand_network

    # Files in plan_alchemical_network_output's layout, so quickrun runs them as-is.
    network_json = work_dir / f"{name}.json"
    alchemical_network.to_json(network_json)
    graphml = work_dir / "ligand_network.graphml"
    graphml.write_text(ligand_network.to_graphml())
    transformations_dir = work_dir / "transformations"
    transformations_dir.mkdir(parents=True, exist_ok=True)

    legs_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for transformation in sorted(alchemical_network.edges, key=lambda t: str(t.name or t.key)):
        filename = f"{transformation.name or transformation.key}.json"
        path = transformations_dir / filename
        transformation.to_json(path)
        ligand_a = str(transformation.stateA.components["ligand"].name)
        ligand_b = str(transformation.stateB.components["ligand"].name)
        leg = _leg_of(transformation.stateA)
        legs_by_pair.setdefault((ligand_a, ligand_b), {})[leg] = str(path)

    edges: list[dict[str, Any]] = []
    for mapping in sorted(
        ligand_network.edges, key=lambda m: (str(m.componentA.name), str(m.componentB.name))
    ):
        ligand_a = str(mapping.componentA.name)
        ligand_b = str(mapping.componentB.name)
        try:
            score: float | None = float(default_lomap_score(mapping))
        except Exception as exc:  # a scorer failure is a fact about the edge
            score = None
            warnings.append(f"could not score mapping {ligand_a}->{ligand_b}: {exc}")
        legs = legs_by_pair.get((ligand_a, ligand_b), {})
        if set(legs) != {"complex", "solvent"}:
            warnings.append(
                f"edge {ligand_a}->{ligand_b} has legs {sorted(legs) or 'none'}, not complex+solvent"
            )
        edges.append(
            {
                "ligand_a": ligand_a,
                "ligand_b": ligand_b,
                "mapping_score": score,
                "n_mapped_atoms": len(mapping.componentA_to_componentB),
                "legs": legs,
            }
        )

    return {
        "engine_version": _versions().get("openfe", "unknown"),
        "versions": _versions(),
        "ligands": [str(lig.name) for lig in charged],
        "network_json": str(network_json),
        "ligand_network_graphml": str(graphml),
        "settings": _settings_summary(protocol.settings, method),
        "edges": edges,
        "warnings": warnings,
    }


def _run_leg(transformation_path: str, leg_dir: Path, n_retries: int) -> dict[str, Any]:
    """One leg, as ``openfe quickrun`` runs it, reduced to numbers the adapter can read."""
    from gufe.protocols.protocoldag import execute_DAG
    from gufe.tokenization import JSON_HANDLER
    from gufe.transformations.transformation import Transformation
    from openff.units import unit

    leg_dir.mkdir(parents=True, exist_ok=True)
    transformation = Transformation.from_json(transformation_path)
    settings = transformation.protocol.settings
    production_fs = float(settings.simulation_settings.production_length.to(unit.femtosecond).m)
    timestep_fs = float(settings.integrator_settings.timestep.to(unit.femtosecond).m)
    record: dict[str, Any] = {
        "ok": False,
        "transformation": transformation_path,
        "n_steps": round(production_fs / timestep_fs),
        "n_repeats": int(settings.protocol_repeats),
        "lambda_windows": int(settings.lambda_settings.lambda_windows),
        "n_replicas": int(settings.simulation_settings.n_replicas),
    }

    started = time.perf_counter()
    dag = transformation.create()
    # No cache_basedir: this driver does not resume, and a stale cache is a
    # failure mode quickrun handles with its own DAG cache that is not reproduced here.
    dagresult = execute_DAG(
        dag,
        shared_basedir=leg_dir,
        scratch_basedir=leg_dir,
        keep_shared=True,
        raise_error=False,
        n_retries=int(n_retries),
    )
    protocol_result = transformation.protocol.gather([dagresult])
    record["wall_seconds"] = time.perf_counter() - started
    ok = bool(dagresult.ok())
    record["ok"] = ok

    estimate = uncertainty = None
    if ok:
        estimate = protocol_result.get_estimate()
        uncertainty = protocol_result.get_uncertainty()
        record["dg_kcal_mol"] = _kcal(estimate)
        record["uncertainty_kcal_mol"] = _kcal(uncertainty)
        record["individual"] = [
            [_kcal(dg), _kcal(err)] for dg, err in protocol_result.get_individual_estimates()
        ]
        record["equilibration_iterations"] = [float(x) for x in protocol_result.equilibration_iterations()]
        record["production_iterations"] = [float(x) for x in protocol_result.production_iterations()]
        gaps: list[float] = []
        incomplete = False
        for analysis in protocol_result.get_forward_and_reverse_energy_analysis():
            if analysis is None:
                incomplete = True
                continue
            forward = analysis["forward_DGs"].to(unit.kilocalorie_per_mole).m
            reverse = analysis["reverse_DGs"].to(unit.kilocalorie_per_mole).m
            gaps.extend(abs(float(f) - float(r)) for f, r in zip(forward, reverse, strict=True))
        finite_gaps = [g for g in gaps if math.isfinite(g)]
        record["forward_reverse_max_gap_kcal_mol"] = max(finite_gaps) if finite_gaps else None
        record["forward_reverse_incomplete"] = incomplete
    else:
        failures = list(dagresult.protocol_unit_failures)
        record["error"] = "; ".join(f"{f.name}: {_format_exception(f.exception)}" for f in failures[-3:]) or (
            "the DAG did not complete and reported no unit failure"
        )

    # quickrun's own output file, so `openfe gather` treats this leg as one of its own.
    results_json = leg_dir / f"{transformation.key}_results.json"
    out_dict = {
        "estimate": estimate,
        "uncertainty": uncertainty,
        "protocol_result": protocol_result.to_dict(),
        "unit_results": {unit_.key: unit_.to_keyed_dict() for unit_ in dagresult.protocol_unit_results},
    }
    with results_json.open("w") as fh:
        json.dump(out_dict, fh, cls=JSON_HANDLER.encoder)
    record["results_json"] = str(results_json)
    return record


def _run(request: dict[str, Any]) -> dict[str, Any]:
    """Execute every leg of every requested edge, in order, and report per leg."""
    work_dir = Path(str(request.get("work_dir") or "")).resolve()
    if not str(request.get("work_dir") or ""):
        raise ValueError("request carries no work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    n_retries = int(request.get("n_retries") or 0)
    warnings: list[str] = []

    edges_out: list[dict[str, Any]] = []
    for edge in list(request.get("edges") or []):
        ligand_a = str(edge.get("ligand_a") or "")
        ligand_b = str(edge.get("ligand_b") or "")
        legs = {str(k): str(v) for k, v in dict(edge.get("legs") or {}).items()}
        if not ligand_a or not ligand_b or not legs:
            raise ValueError(f"edge {edge!r} needs ligand_a, ligand_b and at least one leg")
        edge_dir = work_dir / f"{ligand_a}__{ligand_b}"
        leg_records: dict[str, Any] = {}
        for leg, path in sorted(legs.items()):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"transformation for {ligand_a}->{ligand_b} {leg} not found at {path}"
                )
            try:
                leg_records[leg] = _run_leg(path, edge_dir / leg, n_retries)
            except Exception as exc:  # one leg crashing is that edge's failure, not the run's
                leg_records[leg] = {
                    "ok": False,
                    "transformation": path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                warnings.append(f"{ligand_a}->{ligand_b} {leg}: {type(exc).__name__}: {exc}")
        edges_out.append(
            {"ligand_a": ligand_a, "ligand_b": ligand_b, "edge_dir": str(edge_dir), "legs": leg_records}
        )

    return {
        "engine_version": _versions().get("openfe", "unknown"),
        "versions": _versions(),
        "edges": edges_out,
        "warnings": warnings,
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch on ``operation``; anything else is a caller error, reported structurally."""
    operation = str(request.get("operation") or "")
    if operation == "probe":
        return _probe(request)
    if operation == "plan":
        return _plan(request)
    if operation == "run":
        return _run(request)
    raise ValueError(f"unknown operation {operation!r}; expected 'probe', 'plan' or 'run'")


if __name__ == "__main__":
    _driver.main(run)
