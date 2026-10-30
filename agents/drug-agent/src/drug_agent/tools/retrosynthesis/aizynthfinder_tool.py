"""AiZynthFinder — a searched retrosynthetic route to purchasable precursors.

This is the capability :mod:`drug_agent.tools.synthesis_tool` gave up. That module's
``_mock_route`` returned a :class:`~drug_agent.contracts.synthesis.SynthesisRoute`
whose every field was hash-derived — the step count from ``hash(smiles) % 2``, the
reaction names drawn from a fixed list of six, the reactants left empty so a "step"
was a label with no chemistry in it — and for the shipped candidate
``O=C(O)c1ccc(-c2ccc(F)cc2F)cc1O`` it emitted an amide coupling, a Buchwald-Hartwig
and two reductive aminations on a molecule containing no amide, no amine and no
ester. It was removed under ADR-0011 and ``routes`` has been empty since. This
adapter fills it with a real Monte-Carlo tree search over a template library.

**What it does now.** Runs AiZynthFinder's MCTS backwards from the target, expanding
each molecule with the USPTO template policy, rejecting proposed reactions with the
USPTO feasibility filter, and stopping a branch when its molecule is found in the
ZINC stock file. Every route the search kept is returned as a real ``SynthesisRoute``
whose steps carry the reaction SMILES the template produced and whose leaves carry
the catalogue that matched them.

    ``routes[].steps[].description``   the mapped reaction SMILES, e.g.
                                       ``Br[c:1]...>>OB(O)[c:2]...``
    ``routes[].steps[].reaction_type`` the template library's own classification
                                       string, often ``"0.0 Unrecognized"``
    ``routes[].steps[].reactants``     the precursor SMILES of that disconnection
    ``routes[].steps[].confidence``    ``policy_probability`` in [0, 1] — the
                                       expansion network's probability that this
                                       template applies, unitless
    ``routes[].building_blocks[]``     the leaves, with ``available`` from the
                                       engine's stock lookup and ``catalog`` from
                                       ``AiZynthFinder.stock_info()``
    ``routes[].overall_confidence``    the search's own "state score" in [0, 1]
    ``routes[].feasible``              whether *this* route's leaves are all in stock
    ``building_block_availability``    fraction of the best route's leaves in stock
    ``ext["aizynthfinder"]``           the engine's ``extract_statistics()`` verbatim

Re-measured on this machine on 2026-09-09 (x86-64, conda env ``drug-agent-aizynth``:
Python 3.11.16, aizynthfinder 4.4.1, rdkit 2023.09.6, onnxruntime 1.29.0, numpy
1.26.4), one ``plan()`` call on ``O=C(O)c1ccc(-c2ccc(F)cc2F)cc1O`` at the default 100
iterations against the public USPTO models and the ZINC stock file: solved on
iteration 1 after 0.61 s, the search finishing in 8.20 s over 281 tree nodes, 113
routes found of which 50 solved, five extracted and returned; 20.2 s of adapter wall
time, the difference being interpreter start and model load in the child. Each route
is a single disconnection to two precursors, every precursor matched in ZINC, so
``building_block_availability`` is 1.0 and ``risk_factors`` is empty. The biaryl
disconnection ``_mock_route`` missed is in the returned set — this run led with the
carbonic-acid route and carried ``Fc1ccc(Br)c(F)c1`` plus ``O=C(O)c1ccc(Br)cc1O`` at
the same top score — found by search rather than asserted.

The aarch64 host reported 280 nodes, 114 routes and 49 solved for the same call. The
architecture did not change those; the search is stochastic and the paragraph below
is why.

**Reproducibility, measured rather than assumed.** The search is stochastic and does
not become reproducible by seeding. Ten runs of this target under identical settings
here gave tree sizes from 269 to 289 nodes and 47 to 52 solved routes; three of those
runs seeded ``random`` and ``numpy.random`` identically and still differed from one
another. MCTS breaks ties with ``np.random.choice`` and ``random.choice``
(``aizynthfinder/search/mcts/node.py``), and upstream exposes no seed. The top state
score was 0.9976287063411217 in every one of the ten, so the score was stable while
the tree was not. All five extracted routes tie at exactly that score, so which of them comes
back first varies too: one run led with the carbonic-acid disconnection below, another
with the Suzuki coupling. ``determinism`` on the card therefore says STOCHASTIC, and a
caller that needs one route should read the steps rather than trust the ordering.

**What it deliberately does not do.** It does not price anything, so
``BuildingBlock.price_class`` stays ``None``. It does not claim a reaction works:
``confidence`` is the expansion network's probability that a template *applies*, and
a route whose steps are all classified ``"0.0 Unrecognized"`` is one the template
library could not name — every step of every route returned for the test target was
classified that way. It does not compute synthetic accessibility: that is
``synth.assess``, and ``sa_score`` is left ``None`` here rather than filled from a
second, unrelated calculation. It does not validate chemistry — one of the five tied
routes for the test target disconnects the benzoic acid to carbonic acid plus the
arene, a legal template match and not a synthesis anyone would run. The state score
ranks routes within one search; it is not a probability of success.

**What "solved" means.** A leaf is in stock when its SMILES is in the ZINC stock file
the engine loaded — a catalogue membership test against a 2020 snapshot. It is not a
quote, a lead time, or evidence the material can be bought today. ``feasible`` carries
only that assertion, and :attr:`SynthesisPlan.limitation` says so on every plan.

AiZynthFinder pins rdkit 2023.09 and numpy < 2 against this environment's rdkit
2026.03 and numpy 2.5, so it cannot share this interpreter. It runs in the conda
environment ``drug-agent-aizynth`` through ``scripts/external/aizynth_plan.py``, over
the JSON request/response pair described in :mod:`drug_agent.tools._ext.backend`. When
that environment, the driver, the config or the 663 MB stock file is absent, this
capability is UNAVAILABLE and plans no routes. Under
:mod:`drug_agent.tools.base` there is no stand-in: an invented route is what this
module exists to have replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.synthesis import SynthesisPlan, SynthesisRoute
from ...observability.toolmeter import measure
from .._ext.spec import ExternalSpec
from . import RetrosynthesisError, RetrosynthesisToolAdapter

#: This adapter's contract with AiZynthFinder, not AiZynthFinder's own version.
AIZYNTHFINDER_ADAPTER_VERSION = "1.0.0"

SPEC = ExternalSpec(
    name="AiZynthFinder",
    env_prefix="AIZYNTH",
    # The models, not the source checkout: the driver imports the pip-installed
    # `aizynthfinder` from its own conda environment and needs only the data
    # directory `download_public_data` writes, which is where config.yml lives.
    checkout_relpath=".external/aizynthfinder-data",
    sentinel_files=(
        "uspto_model.onnx",
        "uspto_templates.csv.gz",
        "uspto_filter_model.onnx",
    ),
    driver_name="aizynth_plan.py",
    selection_modules=("aizynthfinder",),
    # What a half-built install actually loses. `onnxruntime` runs both policies;
    # `tables` is what `pandas.read_hdf` needs to open zinc_stock.hdf5 and is a
    # separate wheel, so an install without it selects fine and then fails at the
    # first stock lookup; `rdchiral` applies the retro templates.
    runtime_modules=(
        "aizynthfinder",
        "rdkit",
        "rdchiral",
        "onnxruntime",
        "pandas",
        "tables",
        "networkx",
    ),
    weight_relpaths=("config.yml", "zinc_stock.hdf5"),
    conda_envs=("drug-agent-aizynth",),
    upstream_url="https://github.com/MolecularAI/aizynthfinder",
    upstream_ref="21ff546d5f22",
    license=(
        "AiZynthFinder MIT (Genheden & Bjerrum 2020). The USPTO expansion, ring-breaker and "
        "filter models and the ZINC stock file are the public release `download_public_data` "
        "fetches from Zenodo records 7797465 and 7341155 and from figshare."
    ),
    install_hint="make aizynth-install",
    needs_gpu=False,
)

#: AiZynthFinder's own defaults, restated here because the adapter passes them
#: explicitly rather than letting the config file decide silently.
_DEFAULT_ITERATIONS = 100
_DEFAULT_TIME_LIMIT_S = 120
_DEFAULT_MAX_TRANSFORMS = 6


class AiZynthFinderAdapter(RetrosynthesisToolAdapter):
    """Monte-Carlo tree search for a retrosynthetic route to purchasable starting material."""

    capability_id = "synth.route.aizynthfinder"
    spec = SPEC
    engine = "aizynthfinder"
    version = AIZYNTHFINDER_ADAPTER_VERSION
    produces_nothing = "plans no routes"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # The subprocess budget must comfortably exceed the search's own time limit
        # plus the one-off cost of loading a 663 MB stock file and two ONNX models.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 900.0)

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Retrosynthetic route search (AiZynthFinder)",
            description=(
                "Monte-Carlo tree search backwards from a target using the public USPTO "
                "template expansion policy and feasibility filter, stopping at molecules "
                "present in the ZINC stock file. Returns the routes the search kept, each "
                "with its reaction SMILES, per-step policy probability and stock-matched "
                "building blocks. Prices nothing and validates no chemistry."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="SynthesisPlan",
            # MCTS breaks ties with unseeded RNGs and upstream exposes no seed; five
            # runs of one target here gave 269-288 nodes for an identical top score.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            data_restrictions=(
                "Templates are extracted from USPTO patent reactions; the stock is a ZINC "
                "snapshot published in 2020, not live supplier inventory."
            ),
            known_failure_modes=[
                "unparseable or empty SMILES (reported as a risk factor, not raised)",
                "target unsanitizable by RDKit, which AiZynthFinder rejects before searching",
                "no solved route within the iteration or time limit, so every route has "
                "leaves that are not in stock",
                "templates classified '0.0 Unrecognized', so reaction_type names nothing",
                "chemically implausible but template-legal disconnections rank highly",
                "the 663 MB stock file makes the first search in a process slow to start",
            ],
            validation_checks=[
                "every returned route carries at least one step with real reactant SMILES",
                "building_block_availability is the measured fraction of leaves in stock",
                "step confidences are the policy's own probabilities, clamped to [0, 1]",
                "sa_score is None: this capability does not compute synthetic accessibility",
            ],
        )

    # --- operations --------------------------------------------------------
    def plan(
        self,
        *,
        candidate_id: str,
        smiles: str,
        iteration_limit: int = _DEFAULT_ITERATIONS,
        time_limit_s: int = _DEFAULT_TIME_LIMIT_S,
        max_transforms: int = _DEFAULT_MAX_TRANSFORMS,
        max_routes: int = 5,
        stock: tuple[str, ...] = ("zinc",),
        expansion: tuple[str, ...] = ("uspto",),
        filter_policies: tuple[str, ...] = ("uspto",),
        return_first: bool = False,
    ) -> SynthesisPlan:
        """Search for routes to *smiles* and return them as a :class:`SynthesisPlan`."""
        self.require_available()
        with measure(self.capability_id, "plan", backend=self.backend) as metered:
            plan = self._plan(
                candidate_id=candidate_id,
                smiles=smiles,
                iteration_limit=iteration_limit,
                time_limit_s=time_limit_s,
                max_transforms=max_transforms,
                max_routes=max_routes,
                stock=stock,
                expansion=expansion,
                filter_policies=filter_policies,
                return_first=return_first,
            )
            for key, value in self._ledger_metrics(plan).items():
                metered[key] = value
        return plan

    def plan_batch(self, candidates: list[tuple[str, str]], **kwargs: Any) -> list[SynthesisPlan]:
        """Plan several candidates, each given as ``(candidate_id, smiles)``.

        One search per candidate, because AiZynthFinder's tree is rooted at a single
        target and there is no batched entry point upstream. A molecule the engine
        cannot parse becomes a plan with a risk factor rather than an exception; a
        backend that fails still raises, because that is not a fact about the
        molecule.
        """
        self.require_available()
        with measure(self.capability_id, "plan_batch", backend=self.backend) as metered:
            plans = [self._plan(candidate_id=cid, smiles=smi, **kwargs) for cid, smi in candidates]
            if plans:
                metered["aizynth_n_candidates"] = float(len(plans))
                metered["aizynth_solved_fraction"] = float(
                    sum(1 for p in plans if any(r.feasible for r in p.routes)) / len(plans)
                )
        return plans

    # --- real backend ------------------------------------------------------
    def _plan(
        self,
        *,
        candidate_id: str,
        smiles: str,
        iteration_limit: int = _DEFAULT_ITERATIONS,
        time_limit_s: int = _DEFAULT_TIME_LIMIT_S,
        max_transforms: int = _DEFAULT_MAX_TRANSFORMS,
        max_routes: int = 5,
        stock: tuple[str, ...] = ("zinc",),
        expansion: tuple[str, ...] = ("uspto",),
        filter_policies: tuple[str, ...] = ("uspto",),
        return_first: bool = False,
    ) -> SynthesisPlan:
        # Imported here, not at module scope: `tool_pool()` builds every adapter on
        # every call, and a library import in a constructor is what turns a listing
        # into a multi-second operation once there are thirty of these.
        from rdkit import Chem

        target = (smiles or "").strip()
        if not target or Chem.MolFromSmiles(target) is None:
            # A molecule this platform cannot parse is a fact about the molecule.
            # Six seconds of tree search would report the same thing more slowly.
            return self._empty_plan(candidate_id, ["unparseable_smiles"])

        request = {
            "target_smiles": target,
            "config_file": self._config_file(),
            "stock": list(stock),
            "expansion": list(expansion),
            "filter": list(filter_policies),
            "iteration_limit": int(iteration_limit),
            "time_limit": int(time_limit_s),
            "max_transforms": int(max_transforms),
            "max_routes": int(max_routes),
            "return_first": bool(return_first),
        }
        run = self.run_driver(request, error_cls=RetrosynthesisError)
        return self._plan_from_payload(candidate_id, target, run.payload)

    def _config_file(self) -> str:
        """The absolute path of the config the driver loads.

        ``ExternalBackend`` has already established that the root exists and holds
        every sentinel and weight, so this is a lookup, not a probe.
        """
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() has already run
            raise RetrosynthesisError("AiZynthFinder model directory did not resolve")
        return str(Path(root) / "config.yml")

    def _plan_from_payload(self, candidate_id: str, target: str, payload: dict[str, Any]) -> SynthesisPlan:
        """Turn one driver payload into a plan, keeping the engine's own statistics."""
        engine_version = str(payload.get("engine_version") or "unknown")
        raw_routes = payload.get("routes") or []
        routes: list[SynthesisRoute] = []
        for item in raw_routes:
            tree = item.get("tree")
            if not isinstance(tree, dict):
                continue
            routes.append(
                self.route_from_tree(
                    tree,
                    target_smiles=target,
                    score=self._clamped(item.get("score")),
                    solved=bool(item.get("is_solved", False)),
                    engine_version=engine_version,
                )
            )

        risk: list[str] = []
        if not routes:
            risk.append("no_route_found")
        if not payload.get("is_solved", False):
            risk.append("search_unsolved")
        for warning in payload.get("warnings") or []:
            risk.append(f"config: {warning}")

        best = routes[0] if routes else None
        availability = self.availability(best.building_blocks) if best else None
        if availability is not None and availability < 1.0:
            risk.append("precursors_not_in_stock")

        stats = payload.get("stats")
        return SynthesisPlan(
            candidate_id=candidate_id,
            # Synthetic accessibility is `synth.assess`. Filling it from an
            # unrelated calculation here would put two different numbers behind
            # one field name.
            sa_score=None,
            sa_label=None,
            routes=routes,
            best_route_id=best.id if best else None,
            risk_factors=risk,
            building_block_availability=availability,
            trust_label=TrustLabel.PREDICTED,
            limitation=(
                "Routes are searched, not validated. A leaf marked available was found in "
                "the ZINC stock file the engine loaded — a catalogue membership test, not a "
                "quote, a lead time or evidence the material can be bought. Step confidences "
                "are the expansion policy's probability that a USPTO template applies, not a "
                "probability that the reaction works. No laboratory validation is implied; "
                "confirm precursors and conditions with the chemistry provider."
            ),
            ext={"aizynthfinder": stats} if isinstance(stats, dict) else {},
        )

    def _empty_plan(self, candidate_id: str, risk: list[str]) -> SynthesisPlan:
        """A plan with no routes and a stated reason.

        ``building_block_availability`` stays ``None`` rather than 0.0: nothing was
        checked, and 0.0 would read as "we looked and found none".
        """
        return SynthesisPlan(
            candidate_id=candidate_id,
            sa_score=None,
            routes=[],
            best_route_id=None,
            risk_factors=risk,
            building_block_availability=None,
            trust_label=TrustLabel.PREDICTED,
        )

    @staticmethod
    def _clamped(value: Any) -> float | None:
        """The engine's state score as a float in the contract's range, or ``None``."""
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, score))

    @staticmethod
    def _ledger_metrics(plan: SynthesisPlan) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        A search that stops finding routes, or starts finding only routes with
        unpurchasable leaves, is how a broken stock file or a missing policy model
        announces itself while every call still returns a well-formed plan.
        """
        out: dict[str, float] = {"aizynth_n_routes": float(len(plan.routes))}
        if plan.building_block_availability is not None:
            out["aizynth_building_block_availability"] = float(plan.building_block_availability)
        if plan.routes:
            best = plan.routes[0]
            out["aizynth_n_solved_routes"] = float(sum(1 for r in plan.routes if r.feasible))
            out["aizynth_best_n_steps"] = float(best.n_steps)
            if best.overall_confidence is not None:
                out["aizynth_top_score"] = float(best.overall_confidence)
        stats = plan.ext.get("aizynthfinder")
        if isinstance(stats, dict):
            for key in ("search_time", "number_of_nodes", "first_solution_time"):
                value = stats.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[f"aizynth_{key}"] = float(value)
        return out


__all__ = ["AIZYNTHFINDER_ADAPTER_VERSION", "SPEC", "AiZynthFinderAdapter"]
