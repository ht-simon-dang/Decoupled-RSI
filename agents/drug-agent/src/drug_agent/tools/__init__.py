"""The tool pool — one class per file, each file named for what it binds.

Each adapter wraps a tool behind a typed contract and a ``CapabilityProfile``.
The rule is **make it or fail** (:mod:`drug_agent.tools.base`, ADR-0011): a tool
either computes the real thing or reports ``UNAVAILABLE`` and raises
:class:`~drug_agent.tools.base.ToolUnavailable` from every operation. It never
substitutes a fabricated number for one it cannot compute. The previous
"always provides a deterministic mock" guarantee is gone, along with the mocks.

A file is named for the **external library it binds** where one exists, so the
listing answers "what third-party science is in here?" without opening anything.
The seven built-in tools — the ones whose computation is in this repository — sit
at the top level:

  ``rdkit_tool.py``       ``chem.standardize``       RDKit
  ``sklearn_tool.py``     ``model.activity``         scikit-learn
  ``pfm_tool.py``         ``molecule.generate.pfm``  PFM, via a subprocess
  ``mmff_tool.py``        ``structure.simulate``     RDKit ETKDGv3 + MMFF94s
  ``synthesis_tool.py``   ``synth.assess``           own SA score (RDKit)
  ``safety_tool.py``      ``safety.assess``          own models (sklearn + RDKit)
  ``envelope_tool.py``    ``platform.envelope``      probes this platform
  ``vina_tool.py``        ``structure.dock``         AutoDock Vina, via a subprocess
  ``singlecell_tool.py``  ``perturb.assess``         *nothing — UNAVAILABLE*
  ``llm_provider_tool.py``                           the LLM-calling seam

The thirty adapters over upstream projects (ADR-0012) are grouped by what they do,
because thirty more files in one directory would stop being a listing and start
being a wall:

  ``generation/``      nineteen de-novo generators, pocket-, ligand- and
                       unconditioned, plus the synthesis-aware ones
  ``structure/``       GNINA, DiffDock-L, FlowDock, Boltz-2 — and PoseBusters,
                       which judges what the other four produce. ``vina_tool`` is
                       the fifth producer and stays at this level because it is
                       the ``structure.dock`` capability itself, not an adapter
                       over a checkout
  ``properties/``      ADMET-AI, Chemprop, Uni-Mol
  ``retrosynthesis/``  AiZynthFinder
  ``freeenergy/``      OpenFE
  ``benchmarking/``    PoseBench
  ``_ext/``            the shared machinery: discovery, readiness probing, and
                       launching a driver in somebody else's environment

Each package's ``__init__.py`` holds the abstraction its members actually share —
the RDKit validity and de-duplication screen every generator runs after sampling,
pose parsing and ranking for the structure tools, route-tree flattening for
retrosynthesis — so a subclass supplies only the part that is genuinely its own.
They are **not** re-exported here: reach for
``drug_agent.tools.structure.gnina_tool.GninaAdapter``, which says where it lives,
and keeps importing this package from pulling in thirty modules to render a table.

``vina_tool`` binds AutoDock Vina as of repair R1: the search runs as a subprocess
out of the ``drug-agent-vina`` environment, because vina pins libboost 1.86 while
the pinned rdkit needs 1.90 and nothing is gained by making them argue. Ligand and
receptor PDBQT preparation and pose read-back stay in this interpreter, with RDKit,
Meeko and Open Babel. A host with no vina binary gets the same UNAVAILABLE it
always got, for a now much shorter reason.

``singlecell_tool`` is still named for what it would bind rather than what it has:
real QC, a real donor split and a real disease signature, but no model relating a
molecule to a transcriptional response, so ``score_candidate`` raises and the
capability as a whole is ``UNAVAILABLE``. It explains itself in
``unavailable_reason`` and in its module docstring, which also says what making it
real would take.

Two consequences worth knowing before reading a run:

  * ``structure.dock`` now produces real ``Pose`` objects where the vina binary is
    resolvable, and produces nothing at all where it is not. It is still a
    separate id from ``structure.dock.gnina`` and from the three predicted-complex
    capabilities under ``structure/``, on a separate scale, and no run averages or
    substitutes one for another. A ``dock_score`` prediction is the specialist
    agent's to build; this layer returns coordinates and the engine's own number.
  * ``restoration_score`` is the primary objective of the phenotype brief, and
    nothing can supply it — so a phenotype program does not rank. That is the
    intended, honest outcome; it must not be patched with a fallback.

``synth.assess`` and ``synth.route.aizynthfinder`` are likewise two capabilities,
not one: the first is the transparent RDKit complexity score and still reports no
route, and the second is a real retrosynthetic search. A reader of a plan can
always tell which produced what they are looking at.

:mod:`drug_agent.tools.registry` describes the pool as
:class:`~drug_agent.tools.registry.ToolCard` records — identity, operations, and
live status — which is the same vocabulary the run journal meters in.
"""

from __future__ import annotations

from .base import ToolAdapter
from .llm_provider_tool import FakeProvider, LLMProvider, StubOpenAIProvider, make_provider
from .mmff_tool import SimulationAdapter
from .pfm_tool import (
    GeneratedMolecule,
    GenerationBatch,
    GenerationError,
    PFMGeneratorAdapter,
    resolve_pfm_paths,
)
from .rdkit_tool import Cheminformatics
from .registry import TOOL_OPERATIONS, ToolCard, tool_index, tool_pool
from .safety_tool import SafetyAdapter
from .singlecell_tool import SCDataset, SingleCellAdapter
from .sklearn_tool import BaselineActivityModel
from .synthesis_tool import SynthesisAdapter
from .vina_tool import DockingAdapter

__all__ = [
    "BaselineActivityModel",
    "Cheminformatics",
    "DockingAdapter",
    "FakeProvider",
    "GeneratedMolecule",
    "GenerationBatch",
    "GenerationError",
    "LLMProvider",
    "PFMGeneratorAdapter",
    "SCDataset",
    "SafetyAdapter",
    "SimulationAdapter",
    "SingleCellAdapter",
    "StubOpenAIProvider",
    "SynthesisAdapter",
    "TOOL_OPERATIONS",
    "ToolAdapter",
    "ToolCard",
    "make_provider",
    "resolve_pfm_paths",
    "tool_index",
    "tool_pool",
]
