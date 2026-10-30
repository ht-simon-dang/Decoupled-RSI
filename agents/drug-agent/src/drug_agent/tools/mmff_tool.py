"""MMFF94s conformer minimisation — a real force-field calculation.

This file was ``openmm_tool.py`` and it computed nothing. ``_simulate`` built a
"stability score" from rotatable-bond count and molecular weight plus a hash
jitter, reported an ``interaction_energy_kj_mol`` for a system that contained no
receptor, and labelled the lot ``engine="mock"``. Under the make-it-or-fail rule
(:mod:`drug_agent.tools.base`) that had to become either a real computation or
an ``UNAVAILABLE``. It became real.

**What it does now.** For each candidate: embed an ETKDGv3 conformer ensemble
under a fixed seed, minimise every conformer with MMFF94s, and report the
resulting energetics. Every observable is a quantity the force field actually
produced:

  ``e_min_kcal_mol``        lowest minimised conformer energy
  ``strain_span_kcal_mol``  E_max − E_min across the minimised ensemble — the
                            conformational strain window
  ``rmsd_relax_nm``         heavy-atom RMSD from the embedded geometry to its
                            own minimum. This is the honest version of the
                            number the old proxy invented.
  ``rgyr_nm``               radius of gyration of the lowest-energy conformer
  ``grad_rms_kcal_mol_A``   residual RMS gradient at the minimum

``converged`` is every conformer's MMFF return flag being 0, not a threshold on
a fabricated RMSD. ``n_steps`` is the iteration cap actually handed to the
minimiser rather than a caller default that was previously ignored.

**What it deliberately does not do.** This is a ligand-only calculation, so
there is no ``interaction_energy_kj_mol`` and no ``receptor_id`` in the result —
emitting either would replace a labelled mock with an unlabelled one. It is also
not molecular dynamics, and does not claim to be: MD of a protein-ligand complex
needs docked poses (``structure.dock`` produces none), ligand force-field
parameters (AM1-BCC charge assignment measured 4.8 s at 21 atoms rising to 632 s
at 93), and a solvated system. MMFF94s minimisation is a different, cheaper,
real calculation, and it is named for what it is.

**Cost, by what is in the batch.** 81 ms/ligand was measured on the library-only
sEH demo (``fixtures/briefs/target_seh.yaml``: 84 seed/analogue/library
compounds) and holds only there. The de-novo brief
(``fixtures/briefs/target_seh_pfm.yaml``) adds 40 PFM-generated structures, and
they are ~30x costlier: over that 124-candidate batch the mean is 994 ms/ligand
(the 84 catalogue compounds 121 ms/ligand, the 40 generated ones 2.83 s/ligand,
worst single molecule 41.4 s). Cumulenes, bridgeheads, spiro systems and 7-8
membered rings make ETKDGv3 embedding retry and give MMFF94s a rougher surface
to walk. Both runs: converged for every candidate, bit-identical across reruns
and across thread counts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdMolTransforms  # noqa: F401

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ..contracts.structure import SimulationResult
from ..observability.toolmeter import measure
from .base import ToolAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Conformers embedded per candidate. Small enough to stay inside the batch
#: budget, large enough that the strain span means something.
_N_CONFORMERS = 8
#: Minimiser iteration cap, reported as ``n_steps``.
_MAX_ITERS = 500


class SimulationAdapter(ToolAdapter):
    capability_id = "structure.simulate"

    def __init__(self, backend: str = "auto") -> None:
        self.backend = "rdkit-mmff94s"

    @property
    def profile(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.capability_id,
            version="2.0.0",
            title="Conformer minimisation (MMFF94s)",
            description=(
                "Ligand-only ETKDGv3 conformer ensemble minimised with MMFF94s; reports "
                "minimum energy, conformational strain span and relaxation RMSD. Not "
                "molecular dynamics and not a complex calculation."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="SimulationResult",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            status=CapabilityStatus.AVAILABLE,
            provides_mock=False,
            backend=self.backend,
            runtime_requirements={"gpu": False},
            license="RDKit BSD-3-Clause; MMFF94s parameters",
            known_failure_modes=[
                "embedding failure on strained or exotic ring systems",
                "no MMFF94s parameters for the element set (e.g. boronic acids, selenium)",
            ],
            validation_checks=["energies finite", "convergence flag from the minimiser"],
            unavailable_reason=None,
        )

    def available(self) -> bool:
        return True

    def simulate(
        self,
        candidate_id: str,
        smiles: str,
        receptor_id: str | None = None,
        protocol: str = "mmff94s_min",
        n_steps: int = _MAX_ITERS,
    ) -> SimulationResult:
        self.require_available()
        with measure(self.capability_id, "simulate", backend=self.backend) as metered:
            result = self._simulate(candidate_id, smiles, protocol, n_steps)
            metered["converged"] = 1.0 if result.converged else 0.0
            if result.stability_score is not None:
                metered["stability_score"] = float(result.stability_score)
        return result

    def _simulate(self, candidate_id: str, smiles: str, protocol: str, n_steps: int) -> SimulationResult:
        engine_version = Chem.rdBase.rdkitVersion
        base = Chem.MolFromSmiles(smiles)
        if base is None:
            return SimulationResult(
                candidate_id=candidate_id,
                protocol=protocol,
                engine=self.backend,
                engine_version=engine_version,
                converged=False,
                qc_flags=["unparseable_smiles"],
            )

        mol = Chem.AddHs(base)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D  # fixed: the whole pipeline is reproducible
        params.useSmallRingTorsions = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=_N_CONFORMERS, params=params))
        if not cids:
            return SimulationResult(
                candidate_id=candidate_id,
                protocol=protocol,
                engine=self.backend,
                engine_version=engine_version,
                converged=False,
                qc_flags=["embedding_failed"],
            )

        pre = Chem.Mol(mol)  # geometry before minimisation, for the relaxation RMSD
        try:
            converged_energies = AllChem.MMFFOptimizeMoleculeConfs(
                mol, maxIters=n_steps, mmffVariant="MMFF94s"
            )
        except Exception:
            return SimulationResult(
                candidate_id=candidate_id,
                protocol=protocol,
                engine=self.backend,
                engine_version=engine_version,
                converged=False,
                qc_flags=["no_mmff94s_parameters"],
            )

        flags = [f for f, _e in converged_energies]
        energies = [e for _f, e in converged_energies]
        if not energies or any(not math.isfinite(e) for e in energies):
            return SimulationResult(
                candidate_id=candidate_id,
                protocol=protocol,
                engine=self.backend,
                engine_version=engine_version,
                converged=False,
                qc_flags=["non_finite_energy"],
            )

        best = min(range(len(energies)), key=lambda i: energies[i])
        e_min, e_max = min(energies), max(energies)
        strain_span = e_max - e_min

        observables = {
            "e_min_kcal_mol": round(float(e_min), 4),
            "strain_span_kcal_mol": round(float(strain_span), 4),
            "rmsd_relax_nm": round(self._relaxation_rmsd(pre, mol, cids[best]), 4),
            "rgyr_nm": round(float(rdMolDescriptors.CalcRadiusOfGyration(mol, confId=cids[best])) / 10.0, 4),
            "grad_rms_kcal_mol_A": round(self._residual_gradient(mol, cids[best]), 6),
        }
        # All MMFF return flags 0 == every conformer reached the convergence
        # criterion within maxIters. A non-zero flag means "hit the cap".
        converged = all(f == 0 for f in flags)
        return SimulationResult(
            candidate_id=candidate_id,
            receptor_id=None,  # ligand-only: there is no receptor in this calculation
            protocol=protocol,
            engine=self.backend,
            engine_version=engine_version,
            n_steps=n_steps,
            observables=observables,
            # Strain span is the physically meaningful discriminator; expose it
            # as a bounded score rather than inventing a separate "stability".
            stability_score=round(1.0 / (1.0 + max(0.0, float(strain_span))), 4),
            converged=converged,
            trust_label=TrustLabel.CALCULATED,
            qc_flags=[] if converged else ["minimiser_hit_iteration_cap"],
            metadata={
                "n_conformers": len(cids),
                "mmff_variant": "MMFF94s",
                "embedding": "ETKDGv3",
            },
        )

    @staticmethod
    def _relaxation_rmsd(pre: Chem.Mol, post: Chem.Mol, conf_id: int) -> float:
        """Heavy-atom RMSD (nm) from the embedded geometry to its own minimum."""
        heavy = [a.GetIdx() for a in post.GetAtoms() if a.GetAtomicNum() > 1]
        a = pre.GetConformer(conf_id)
        b = post.GetConformer(conf_id)
        total = 0.0
        for i in heavy:
            pa, pb = a.GetAtomPosition(i), b.GetAtomPosition(i)
            total += (pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2
        return math.sqrt(total / max(len(heavy), 1)) / 10.0  # A -> nm

    @staticmethod
    def _residual_gradient(mol: Chem.Mol, conf_id: int) -> float:
        """RMS force at the minimum, from the MMFF94s field itself."""
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is None:
            return float("nan")
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return float("nan")
        grad = ff.CalcGrad()
        if not grad:
            return 0.0
        return math.sqrt(sum(g * g for g in grad) / len(grad))
