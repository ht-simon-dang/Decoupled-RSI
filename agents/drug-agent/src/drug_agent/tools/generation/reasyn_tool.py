"""ReaSyn — the synthetic pathway to a molecule, or to the nearest one that has a pathway.

ReaSyn (NVIDIA-Digital-Bio, *Exploring Synthesizable Chemical Space with Iterative
Pathway Refinements*, arXiv:2509.16084) is a synthesis-aware generator. It does not
sample atoms. It takes one molecule and searches the space of reaction pathways —
Enamine building blocks joined by 115 SMARTS reaction templates — for a pathway
whose product is that molecule. When no such pathway exists, it returns pathways to
the closest product it *can* reach. Upstream calls this *projection onto
synthesizable space*.

**A projected molecule is a different molecule. Re-score it from scratch.**
This is the one thing about this tool that cannot be got wrong. The output of a
projection is not the input with a synthesis attached; it is a new compound,
reachable by known chemistry, that happens to be similar to the input. Similar is
not the same. A methyl becomes an ethyl, a ring loses a nitrogen, a fluorine moves —
and affinity, solubility, permeability, hERG liability and every other property are
now unmeasured for the thing actually in hand. The similarity ReaSyn reports
(``score``, a Morgan Tanimoto over the input) is a statement about fingerprints, not
about biology, and reaches 1.0 only on exact reconstruction. Therefore:

* no prediction, docking score, assay value or ranking position may be carried from
  the input molecule onto its projected analogue, at any distance downstream;
* every molecule this adapter returns leaves as ``TrustLabel.PROPOSED`` and must
  traverse the same standardization, activity, safety and synthesis pipeline as any
  other candidate, exactly as if it had been drawn from a catalogue;
* every molecule is flagged, in ``qc_flags``, as either
  ``reasyn_exact_reconstruction`` (canonical SMILES equal to the input, so the input
  itself is makeable) or ``reasyn_projected_analogue`` (a different compound). The
  flag exists so that "is this the molecule I asked about?" is answerable three
  artifacts later, when the SMILES is just a string in a table.

**What it would compute here.** For each input SMILES, the pathways found and their
products, keyed as the driver returns them:

    ``smiles``                  the product actually reachable by the pathway
    ``target``                  the input molecule it was projected from
    ``route``                   ``Stack.get_action_string`` — a ``;``-joined,
                                bottom-up sequence of building-block SMILES and
                                ``R<n>`` reaction-template indices
    ``num_steps``               reactions in the pathway (0 = a bare building block)
    ``similarity_to_input``     Morgan Tanimoto of product against input, in [0, 1]
    ``exact_reconstruction``    whether the product *is* the input
    ``qc_flags``                the projection flags above

The base class in :mod:`drug_agent.tools.generation` keeps ``smiles``, ``sdf`` and
``qc_flags`` from each record and computes validity and uniqueness itself. The route
string is therefore available to the driver boundary and to this module, and is not
today carried onto :class:`~drug_agent.tools.pfm_tool.GeneratedMolecule`, which has
no field for it. That is a real limitation of the current contract, named rather
than hidden: the flags travel, the pathway does not.

**What it deliberately does not do.** It does not verify chemistry. A pathway is a
sequence of template matches, not an executed synthesis — no conditions, no
stoichiometry, no yield, no orthogonality check, no evidence that the building
blocks are still purchasable (the index upstream trained on is Enamine US Stock as
retrieved in October 2023, per the model card). It does not score binding, and it
does not rank. A route from ReaSyn is a hypothesis about chemistry in precisely the
sense that a pose from a docking engine is a hypothesis about binding.

**Why this capability is UNAVAILABLE on this machine.** The checkout at
``.external/reasyn`` is the real code at ``8a0b02178e90``, and it is complete; what
is absent is everything the code loads at run time.

1. **The two model checkpoints are not present.**
   ``.external/reasyn/data/trained_model/`` contains one file, its ``.gitignore``
   (which is ``*``, so the weights were never in the repository). The run path
   ``reasyn/sampler/parallel.py::run_sampling_one`` asserts it was given a list of
   exactly two checkpoints, and ``reasyn/sampler/sampler.py::Sampler.__init__``
   unpacks them and asserts the first has ``model_type == 'autoregressive'`` and the
   second ``'editflow'``. So both of
   ``data/trained_model/nv-reasyn-ar-166m-v2.ckpt`` (166M parameters) and
   ``data/trained_model/nv-reasyn-eb-174m-v2.ckpt`` (174M parameters) must exist;
   one is not enough. Upstream's README directs to
   https://huggingface.co/nvidia/NV-ReaSyn-AR-166M-v2 and
   https://huggingface.co/nvidia/NV-ReaSyn-EB-174M-v2, or the equivalent NGC
   resources, under the NVIDIA Open Model License.

2. **The building-block index and reaction matrix are not present, and are not
   downloadable.** ``.external/reasyn/data/processed/comp_2048/`` likewise contains
   only its ``.gitignore``. There is no inference config: ``run_sampling_one`` reads
   the config out of the checkpoint (``ckpt["hyper_parameters"]["config"]``), and
   that config points ``chem.fpindex`` at ``data/processed/comp_2048/fpindex.pkl``
   and ``chem.rxn_matrix`` at ``data/processed/comp_2048/matrix.pkl`` — the same two
   relative paths ``configs/train_ar.yml`` carries — then unpickles both before any
   sampling starts. Upstream builds them with ``scripts/preprocess.py`` from the
   Enamine US Stock catalog, which its README states is "available upon request"
   rather than by download, or from SynFormer's preprocessed release converted by
   ``scripts/convert_processed_1.py`` and ``scripts/convert_processed_2.py``. The
   115 reaction templates (``data/rxn_templates/comprehensive.txt``, also absent)
   are an input to that preprocessing, not to sampling — ``matrix.pkl`` carries the
   reactions once built.

3. **Five of the run path's imports are missing from this interpreter.** Probed in
   the ``drug-agent`` environment (Python 3.12.14): ``omegaconf``
   (``reasyn/sampler/parallel.py``), ``einops`` and ``torchtyping``
   (``reasyn/utils/editflow_utils.py``), ``rotary_embedding_torch``
   (``reasyn/models/transformer_rope.py``) and ``editdistance``
   (``reasyn/chem/fpindex.py``) are all absent, and no conda environment named
   ``reasyn`` exists here. This blocker is the cheap one: all five are pure Python
   and install with pip. **The platform is not an obstacle to ReaSyn on either
   architecture** — there is no compiled extension in its dependency set that lacked
   an aarch64 build, and on this x86-64 host the question does not arise; torch with
   working CUDA is already present. The blocker is data, not silicon, which is why
   moving machine did not move this row.

4. **The fingerprint index pickle is version-coupled to scikit-learn.**
   ``reasyn/chem/fpindex.py`` builds an ``sklearn.neighbors.BallTree``, so
   ``fpindex.pkl`` restores one. Upstream's README pins ``scikit-learn==1.6.0`` to
   load that pickle and ``1.2.2`` for its hit-expansion evaluation; this environment
   has 1.9.0. Supplying the file in blocker 2 is therefore necessary but may not be
   sufficient, and this is stated in advance rather than discovered as an
   unpickling error mid-run.

**What the platform does instead: nothing.** Under :mod:`drug_agent.tools.base`
there is no surrogate. ``generate()`` raises
:class:`~drug_agent.tools.base.ToolUnavailable`, the run proceeds with no
synthesizable-projection arm, and that absence is visible on the capability card.
The alternative — projecting with some cheaper analogue enumeration and calling the
result synthesizable — would put molecules with fabricated routes in front of
someone who asked for real ones, which is the failure this rule exists to prevent.
Route *feasibility* for a molecule the platform already holds is a different
question, answered separately by the retrosynthesis engine; it does not substitute
for projection, because it cannot propose the reachable analogue that projection
exists to find.
"""

from __future__ import annotations

import math
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import note
from .._ext.spec import ExternalSpec
from . import GenerationBackendError, GenerativeToolAdapter

#: This adapter's contract with ReaSyn, not ReaSyn's own version.
REASYN_ADAPTER_VERSION = "1.0.0"

SPEC = ExternalSpec(
    name="ReaSyn",
    env_prefix="REASYN",
    checkout_relpath=".external/reasyn",
    # Files the run path really imports: run_sampling_one lives in parallel.py and
    # pulls in the Sampler, the ReaSyn network and the Molecule wrapper.
    sentinel_files=(
        "reasyn/sampler/parallel.py",
        "reasyn/sampler/sampler.py",
        "reasyn/models/reasyn.py",
        "reasyn/chem/mol.py",
    ),
    driver_name="reasyn_sample.py",
    # Selection identifies the environment ReaSyn was installed into (upstream's
    # env.yml builds a torch + omegaconf env named `reasyn`); readiness below then
    # rejects a half-built one module by module.
    selection_modules=("torch", "omegaconf"),
    # What a broken install loses, in the order it is worth reading: the five
    # pip-only imports first, then the heavy ones a torch environment already has.
    runtime_modules=(
        "omegaconf",
        "einops",
        "torchtyping",
        "rotary_embedding_torch",
        "editdistance",
        "torch",
        "rdkit",
        "sklearn",
        "joblib",
        "numpy",
        "pandas",
        "tqdm",
    ),
    # Both checkpoints, because Sampler asserts one of each model_type; and both
    # pickles, because run_sampling_one unpickles them before sampling begins.
    weight_relpaths=(
        "data/trained_model/nv-reasyn-ar-166m-v2.ckpt",
        "data/trained_model/nv-reasyn-eb-174m-v2.ckpt",
        "data/processed/comp_2048/fpindex.pkl",
        "data/processed/comp_2048/matrix.pkl",
    ),
    conda_envs=("reasyn", "drug-agent"),
    upstream_url="https://github.com/NVIDIA-Digital-Bio/reasyn",
    upstream_ref="8a0b02178e90",
    license="Apache-2.0 (source); NVIDIA Open Model License (NV-ReaSyn v2 weights, absent here)",
    # Every readiness failure repeats this string, so it stays one line: the exact
    # paths come from weight_relpaths above, and the full provenance from the module
    # docstring. What it must carry is where upstream says the missing files live.
    install_hint=(
        "make reasyn-install; weights from huggingface.co/nvidia/NV-ReaSyn-AR-166M-v2 and "
        "NV-ReaSyn-EB-174M-v2, building-block index from Enamine US Stock (on request) — "
        "see .external/reasyn/README.md"
    ),
    needs_gpu=True,
)

#: Fraction of the adapter's timeout handed to the sampler as its own per-molecule
#: budget. The remainder covers checkpoint load, index unpickling and CUDA startup,
#: which upstream's ``time_limit`` does not account for.
_SAMPLER_BUDGET_FRACTION = 0.9

#: Floor on that per-molecule budget. A search cut off early returns only
#: bare building blocks, which is a worse answer than a timeout.
_MIN_SAMPLER_SECONDS = 60.0


class ReaSynAdapter(GenerativeToolAdapter):
    """Synthesizable-analogue projection and pathway prediction with ReaSyn."""

    capability_id = "molecule.generate.reasyn"
    spec = SPEC
    engine = "reasyn"
    version = REASYN_ADAPTER_VERSION
    #: Conditioned on a reference *molecule*, never on a pocket. Recorded on the
    #: batch config so this can never be read as structure-based generation.
    conditioning = "ligand"
    #: Generates in reaction/building-block space, so every product arrives with a
    #: pathway — a hypothesis about chemistry, not a validated synthesis.
    synthesis_aware = True

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Synthesizable-analogue projection (ReaSyn)",
            description=(
                "Projects an input molecule onto synthesizable chemical space. An autoregressive "
                "pathway model and an Edit-Bridge refiner search Enamine building blocks joined by "
                "115 reaction templates, returning the reachable product closest to the input "
                "together with its pathway. The product is usually not the input: it is a new "
                "compound that must be re-scored from scratch and may never inherit the input's "
                "predictions."
            ),
            input_schema_ref="GenerationSpec",
            output_schema_ref="Candidate",
            # Upstream exposes no seed: the search draws from numpy.random.choice for
            # parent states and torch.multinomial for tokens. The driver seeds all
            # three generators, which narrows the spread without making it a function.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "the product is an analogue, not the input — `score` is a Morgan Tanimoto and "
                "equals 1.0 only on exact reconstruction, so a projected molecule that inherits "
                "the input's activity or safety numbers is silently wrong",
                "no pathway is found for an input molecule: Sampler.get_dataframe returns an empty "
                "frame and that input contributes nothing to the batch",
                "a route with num_steps == 0 is a bare catalogue building block, not a synthesis; "
                "it is flagged reasyn_no_reaction_step rather than dropped",
                "Stack.get_action_string names reactions as R<n> indices into the 115-template "
                "SynFormer set — the string is not an executable procedure and carries no "
                "conditions, stoichiometry, yield or orthogonality check",
                "the building-block index upstream trained on is Enamine US Stock as retrieved in "
                "October 2023 (model card), so a route can name a block no longer purchasable",
                "unpickling data/processed/comp_2048/fpindex.pkl restores an sklearn BallTree; the "
                "README pins scikit-learn 1.6.0 to load it, and another version can fail outright",
                "the per-molecule time limit truncates the search rather than raising, so a short "
                "budget degrades pathway quality quietly",
                "run_sampling_one cannot express upstream's --no_exact_break, so the search halts "
                "as soon as it reconstructs the input exactly; an already-synthesizable input "
                "therefore yields fewer analogues here than upstream's CLI would return",
            ],
            validation_checks=[
                "context_smiles is required and non-empty: a ligand-conditioned projector with no "
                "input molecule refuses instead of sampling something",
                "every product SMILES is re-parsed by RDKit in the domain base, which counts "
                "validity and uniqueness itself — the model does not report its own",
                "each product is flagged reasyn_exact_reconstruction or reasyn_projected_analogue, "
                "decided by canonical SMILES equality with the input molecule",
                "both checkpoints are checked to exist before launch, and the driver requires the "
                "pair in [autoregressive, edit-bridge] order that Sampler asserts",
                "a pocket passed to a ligand-conditioned generator is recorded as a warning, never "
                "silently accepted as conditioning",
            ],
        )

    # --- the real backend --------------------------------------------------
    def _sample(
        self,
        *,
        n_molecules: int,
        pocket: dict[str, Any],
        context_smiles: list[str],
        seed: int,
        batch_size: int,
        timeout_s: float,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Project every molecule in *context_smiles* and return the products found.

        ``batch_size`` has no counterpart upstream — ReaSyn's unit of work is one
        input molecule, and its search width is ``search_width`` × ``exhaustiveness``
        inside that molecule — so it is used only to bound how many pathways per
        input are kept, alongside the ``n_molecules`` the caller asked for.
        """
        if not context_smiles:
            raise ValueError(
                f"{self.capability_id} is ligand-conditioned: it projects an input molecule onto "
                "synthesizable space and has nothing to sample without context_smiles."
            )
        root = self.ext.root
        if root is None:  # require_available() has already covered this
            raise GenerationBackendError("ReaSyn checkout disappeared between probe and call")

        warnings: list[str] = []
        if pocket:
            warnings.append(
                "ReaSyn conditions on a reference molecule, not a binding site; the pocket in this "
                "request was ignored and the output is not pocket-conditioned."
            )

        n_inputs = len(context_smiles)
        per_input = max(1, math.ceil(int(n_molecules) / n_inputs))
        request = {
            "root": str(root),
            "smiles": list(context_smiles),
            "model_paths": [str(root / rel) for rel in self.spec.weight_relpaths[:2]],
            "device": self.ext.device,
            "seed": int(seed),
            "max_results": max(per_input, int(batch_size)),
            "time_limit_seconds": int(
                max(_MIN_SAMPLER_SECONDS, timeout_s * _SAMPLER_BUDGET_FRACTION / n_inputs)
            ),
        }
        run = self.run_driver(request, error_cls=GenerationBackendError, seed=seed, timeout_s=timeout_s)
        payload = run.payload

        records = [r for r in (payload.get("molecules") or []) if isinstance(r, dict)]
        warnings.extend(str(w) for w in (payload.get("warnings") or []))
        errors = [str(e) for e in (payload.get("errors") or [])]

        n_projected = int(payload.get("n_projected") or 0)
        if n_projected < n_inputs:
            warnings.append(
                f"ReaSyn found a pathway for {n_projected} of {n_inputs} input molecules; "
                "the rest contributed nothing."
            )
        note(
            self.capability_id,
            "generate",
            f"ReaSyn projected {n_inputs} molecule(s) in {run.seconds:.1f}s of driver wall time",
        )
        return records, warnings, errors


__all__ = ["REASYN_ADAPTER_VERSION", "SPEC", "ReaSynAdapter"]
