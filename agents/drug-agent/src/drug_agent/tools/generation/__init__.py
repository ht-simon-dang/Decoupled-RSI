"""De-novo generators: nineteen ways to propose a molecule nobody has made.

The catalogue this package implements divides generation four ways, and the
divisions matter because they decide what a proposal even *is*.

**Pocket-conditioned 3D** — TargetDiff, Pocket2Mol, PocketFlow, MolCRAFT, MolJO,
MolPilot, PocketXMol, DrugFlow, FLOWR, FLOWR.root, PILOT, OMTRA and PFM sample atom
positions and types inside a binding site and rebuild a molecular graph from the
point cloud. Their failure mode is not chemistry, it is *reconstruction*: bond
perception over a generated point cloud invents allenes and four-membered rings at
a rate that has nothing to do with how well the model learned the pocket. The
measured PFM baseline on this repository is the warning — of 40 samples, 21
contained an allene and 21 a four-membered ring, leaving 2 chemically ordinary
molecules. Every adapter here therefore reports validity and uniqueness as
first-class metrics, because a collapse in either is how a broken checkpoint, a
wrong pocket or a bad reconstruction announces itself.

**Ligand-conditioned 3D** — ShEPhERD generates against a reference ligand's shape,
electrostatics and pharmacophores rather than a pocket. It is not a pocket
generator and must never be labelled as one.

**Unconditional 3D** — FlowMol3 samples all-atom molecules with no target at all.
It is a chemical prior and a control, not a substitute for a conditioned model.

**Synthesis-aware** — REINVENT4, SynFormer, PrexSyn, ReaSyn and RxnFlow generate in
the space of reactions and building blocks instead of atoms. What they buy is that
the output comes with a route; what they do not buy is proof that the route works.
A generated synthesis is a hypothesis about chemistry, exactly like a generated
pose is a hypothesis about binding.

:class:`GenerativeToolAdapter` owns everything that happens *after* a model emits
samples, because that half is identical for all nineteen and is where the honesty
lives: RDKit parsing, canonical de-duplication, the heavy-atom window, QC flags,
and the assembly of a :class:`~drug_agent.tools.pfm_tool.GenerationBatch` whose
``validity`` and ``uniqueness`` are computed from real counts. A subclass supplies
only :meth:`_sample`, which returns raw records from its own backend.

**Trust.** Every molecule leaves here as ``TrustLabel.PROPOSED``. It has been
generated, not screened; it becomes evidence only after the same RDKit, activity,
safety and synthesis pipeline that judges every other candidate has judged it.

**Refusal.** A generator with no backend reports UNAVAILABLE and raises
:class:`~drug_agent.tools.base.ToolUnavailable` from ``require_available()``, per
:mod:`drug_agent.tools.base`. That is a deliberate divergence from
:mod:`drug_agent.tools.pfm_tool`, which raises its own ``GenerationError`` because
its one caller catches exactly that and degrades the run rather than failing the
task. PFM's contract is pinned by tests and is not changed here; these nineteen are
new surface with no such caller, so they follow the base rule instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from ...contracts.brief import PocketSpec
from ...observability.toolmeter import measure, note
from .._ext.adapter import ExternalToolAdapter
from ..pfm_tool import GeneratedMolecule, GenerationBatch

RDLogger.DisableLog("rdApp.*")


class GenerationBackendError(RuntimeError):
    """The selected generative backend ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the model was
    installed, sampling started, and it did not produce usable output — a CUDA
    fault, a checkpoint mismatch, a driver crash.
    """


class GenerativeToolAdapter(ExternalToolAdapter):
    """Sampling, screening and batch assembly shared by every de-novo generator."""

    #: What the model conditions on. One of ``"pocket"``, ``"ligand"`` or
    #: ``"none"``. Recorded on the batch config so a ligand-conditioned model can
    #: never be read as a pocket-conditioned one.
    conditioning: str = "pocket"
    #: ``True`` when the backend generates in reaction/building-block space and can
    #: therefore report a route alongside the molecule.
    synthesis_aware: bool = False
    produces_nothing = "generates nothing"
    #: What :meth:`_sample` last loaded, folded onto the batch config as
    #: ``sample_provenance``. A backend that can be pointed at more than one set
    #: of weights fills it with at least the path and the sha256 of the file it
    #: actually read, because a checkpoint override that silently does not take
    #: effect produces a batch that is indistinguishable from an ordinary one.
    _sample_provenance: dict[str, Any] = {}

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The PocketSpec/batch fields this backend REFUSES to sample without.

        Derived from the adapter's own state, never hand-listed, so a model added
        to the catalogue describes itself and cannot drift out of date. Read by
        the master agent's prompt (so it can pick a generator it is able to feed)
        and by the pre-flight check (so a doomed combination is refused before the
        pipeline spends a cycle on it).

        The default is what ``conditioning`` implies. Subclasses whose contract is
        narrower than their conditioning — PocketXMol's de-novo tasks also need a
        numeric centre, MolPilot needs a reference-ligand SDF — override it.
        """
        return ("pocket_pdb",) if self.conditioning == "pocket" else ()

    # --- what a subclass supplies -----------------------------------------
    def _sample(
        self,
        *,
        n_molecules: int,
        pocket: dict[str, Any],
        context_smiles: list[str],
        seed: int,
        batch_size: int,
        timeout_s: float,
        checkpoint: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Raw samples from this backend, as ``(records, warnings, errors)``.

        Each record carries at least ``smiles``; optionally ``sdf``, ``qc_flags``
        and ``route``. Everything after this — parsing, de-duplication, filtering,
        metric computation — belongs to the base class, so that nineteen backends
        cannot disagree about what "validity" counts.

        ``checkpoint`` selects weights other than the backend's default. Only the
        backends that can be pointed at a fine-tuned checkpoint accept it, and
        :meth:`_generate` forwards it **only when it is not None**, so a backend
        that has not implemented it is called byte-identically to before and a
        caller who asks one for an override gets a refusal rather than a batch
        quietly sampled from the default weights.
        """
        raise NotImplementedError

    # --- the operation every generator exposes ----------------------------
    def generate(
        self,
        *,
        n_molecules: int,
        pocket: PocketSpec | dict[str, Any] | None = None,
        context_smiles: list[str] | None = None,
        seed: int = 0,
        batch_size: int = 10,
        min_heavy_atoms: int = 10,
        max_heavy_atoms: int = 70,
        keep_invalid: bool = False,
        max_runtime_seconds: float | None = None,
        checkpoint: str | Path | None = None,
    ) -> GenerationBatch:
        """Sample ``n_molecules`` ligands, de-duplicated and validity-filtered.

        ``pocket`` accepts either the typed :class:`PocketSpec` or its
        ``model_dump``; the pipeline passes the dict and direct callers reach for
        the contract object, and accepting both here beats an ``AttributeError``
        three frames deeper.

        ``checkpoint`` points the backend at weights other than its default — the
        output of a target-adaptation run rather than the released prior. It is
        the half of adaptation that is easy to leave out and impossible to notice
        missing: a fine-tuning run that writes a real checkpoint no sampling call
        ever loads still succeeds, and every batch afterwards reports the
        incumbent's numbers with no error anywhere. ``None`` is the default and
        keeps today's behaviour exactly, down to which weights are read.
        """
        self.require_available()
        with measure(self.capability_id, "generate", backend=self.backend) as metered:
            return self._generate(
                metered,
                n_molecules=n_molecules,
                pocket=pocket,
                context_smiles=context_smiles,
                seed=seed,
                batch_size=batch_size,
                min_heavy_atoms=min_heavy_atoms,
                max_heavy_atoms=max_heavy_atoms,
                keep_invalid=keep_invalid,
                max_runtime_seconds=max_runtime_seconds,
                checkpoint=checkpoint,
            )

    def _generate(
        self,
        metered: dict[str, float],
        *,
        n_molecules: int,
        pocket: PocketSpec | dict[str, Any] | None,
        context_smiles: list[str] | None,
        seed: int,
        batch_size: int,
        min_heavy_atoms: int,
        max_heavy_atoms: int,
        keep_invalid: bool,
        max_runtime_seconds: float | None,
        checkpoint: str | Path | None = None,
    ) -> GenerationBatch:
        started = time.perf_counter()
        if isinstance(pocket, PocketSpec):
            pocket = pocket.model_dump(mode="json")
        pocket = pocket or {}
        cfg = {
            "backend": self.backend,
            "conditioning": self.conditioning,
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "min_heavy_atoms": int(min_heavy_atoms),
            "max_heavy_atoms": int(max_heavy_atoms),
            "pocket": pocket,
            "upstream_ref": self.spec.upstream_ref,
        }
        # Forwarded only when asked for, so every backend that has never heard of
        # a checkpoint override is still called with exactly the arguments it was
        # called with before this parameter existed.
        override = {} if checkpoint is None else {"checkpoint": checkpoint}
        if checkpoint is not None:
            cfg["checkpoint"] = str(checkpoint)
        try:
            raw, warnings, errors = self._sample(
                n_molecules=n_molecules,
                pocket=pocket,
                context_smiles=list(context_smiles or []),
                seed=seed,
                batch_size=batch_size,
                timeout_s=max_runtime_seconds or self.timeout_s,
                **override,
            )
        except TypeError as exc:
            # Narrow on purpose: only the interpreter's own "got an unexpected
            # keyword argument 'checkpoint'" is translated. A TypeError raised
            # *inside* a working _sample must keep its own traceback.
            unsupported = "unexpected keyword argument" in str(exc) and "checkpoint" in str(exc)
            if checkpoint is not None and unsupported:
                raise GenerationBackendError(
                    f"{self.backend} cannot be pointed at a checkpoint: its _sample() takes no "
                    "'checkpoint' argument. Refusing rather than sampling from the default "
                    "weights under a request that asked for adapted ones."
                ) from exc
            raise
        # Which weights were actually read, recorded whether or not an override
        # was asked for — a silent fall-back to the default prior is only
        # detectable after the fact if the default path is stamped too.
        provenance = dict(getattr(self, "_sample_provenance", None) or {})
        if provenance:
            cfg["sample_provenance"] = provenance
        batch = GenerationBatch(
            backend=self.backend,
            model_id=self.capability_id,
            model_version=self.version,
            n_requested=int(n_molecules),
            n_raw=len(raw),
            warnings=warnings,
            errors=errors,
            config=cfg,
        )
        self._screen(
            batch,
            raw,
            pocket=pocket,
            min_heavy_atoms=min_heavy_atoms,
            max_heavy_atoms=max_heavy_atoms,
            keep_invalid=keep_invalid,
            n_molecules=n_molecules,
        )
        batch.runtime_seconds = time.perf_counter() - started
        self._meter(metered, batch, n_molecules=n_molecules)
        if errors:
            note(self.capability_id, "generate", errors[0][:200])
        return batch

    def _screen(
        self,
        batch: GenerationBatch,
        raw: list[dict[str, Any]],
        *,
        pocket: dict[str, Any],
        min_heavy_atoms: int,
        max_heavy_atoms: int,
        keep_invalid: bool,
        n_molecules: int,
    ) -> None:
        """Parse, de-duplicate and window the raw samples onto *batch*.

        The counts are what make ``validity`` and ``uniqueness`` meaningful:
        ``n_valid`` is how many RDKit could parse at all, ``n_unique`` how many
        distinct canonical structures those were. Both are computed here, from the
        same raw list, for every backend — a generator is not allowed to report its
        own validity.
        """
        pocket_id = pocket.get("receptor_id") or pocket.get("pdb_id")
        seen: set[str] = set()
        n_valid = 0
        for idx, item in enumerate(raw):
            smiles = item.get("smiles") or ""
            rdmol = Chem.MolFromSmiles(smiles) if smiles else None
            if rdmol is None:
                if keep_invalid:
                    batch.molecules.append(
                        GeneratedMolecule(
                            smiles=smiles,
                            origin=self.engine,
                            sample_index=idx,
                            model_id=self.capability_id,
                            model_version=self.version,
                            backend=self.backend,
                            pocket_id=pocket_id,
                            sdf_block=item.get("sdf"),
                            qc_flags=["unparseable_smiles"],
                        )
                    )
                continue
            n_valid += 1
            canon = Chem.MolToSmiles(rdmol)
            if canon in seen:
                continue
            seen.add(canon)
            heavy = rdmol.GetNumHeavyAtoms()
            flags: list[str] = list(item.get("qc_flags") or [])
            if heavy < min_heavy_atoms:
                flags.append("below_min_heavy_atoms")
            if heavy > max_heavy_atoms:
                flags.append("above_max_heavy_atoms")
            if flags and not keep_invalid:
                continue
            batch.molecules.append(
                GeneratedMolecule(
                    smiles=canon,
                    origin=self.engine,
                    sample_index=idx,
                    model_id=self.capability_id,
                    model_version=self.version,
                    backend=self.backend,
                    pocket_id=pocket_id,
                    n_heavy_atoms=heavy,
                    sdf_block=item.get("sdf"),
                    qc_flags=flags,
                )
            )
        batch.n_valid = n_valid
        batch.n_unique = len(seen)
        batch.molecules = batch.molecules[: int(n_molecules)]

    def _meter(self, metered: dict[str, float], batch: GenerationBatch, *, n_molecules: int) -> None:
        """Record the numbers that say whether this generation was any good.

        ``n_kept`` is the one that proves a run produced something: a backend that
        failed entirely still returns a well-formed empty batch carrying its own
        name, so validity and uniqueness alone can read as healthy over nothing.
        """
        metrics = batch.metrics()
        metered["n_kept"] = float(len(batch.molecules))
        metered["n_requested"] = float(n_molecules)
        for key in ("generation_validity", "generation_uniqueness"):
            value = metrics.get(key)
            if value is not None:
                metered[key] = float(value)


__all__ = ["GenerationBackendError", "GenerativeToolAdapter"]
