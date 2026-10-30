"""Retrosynthesis: the disconnection this package used to invent, now searched for real.

This is the capability :mod:`drug_agent.tools.synthesis_tool` deliberately gave up.
Its ``_mock_route`` returned a :class:`~drug_agent.contracts.synthesis.SynthesisRoute`
whose every field was hash-derived — the step count from ``hash(smiles) % 2``, the
reaction names drawn from a fixed list of six, the reactants left empty so a "step"
was a label with no chemistry in it. For the shipped candidate
``O=C(O)c1ccc(-c2ccc(F)cc2F)cc1O`` it emitted an amide coupling, a Buchwald-Hartwig
and two reductive aminations on a molecule containing no amide, no amine and no
ester, while missing the one obvious real disconnection: the Suzuki coupling across
the biaryl. It was removed under ADR-0011, and ``routes`` has been empty since.

A route is an object, not a number, so it could never be approximated — the only
honest options were a real search engine or nothing. This package is the engine.

:class:`RetrosynthesisToolAdapter` owns the mapping from a search engine's route
tree to this project's contracts, which is where the care is needed. Upstream
planners return a nested alternation of molecule and reaction nodes; the contracts
want an ordered list of :class:`RouteStep` plus the
:class:`BuildingBlock` leaves. :meth:`route_from_tree` walks the tree once,
flattens it depth-first so step ``0`` is the first thing a chemist would do, and
records which leaves the engine's stock file says are actually purchasable.

**What a solved route is and is not.** A planner reports a route as *solved* when
every leaf is in its stock file. That is a statement about a catalogue, not about a
laboratory: it does not mean the reactions work, that the conditions are known, or
that the material is in stock today at a price anyone would pay. So
``feasible`` reflects only what the engine asserted, ``building_block_availability``
is the measured fraction of leaves in stock, and
:attr:`SynthesisPlan.limitation` keeps saying that no laboratory validation is
implied. Confirm precursors with the eventual chemistry provider.
"""

from __future__ import annotations

from typing import Any

from ...contracts.enums import Severity, TrustLabel
from ...contracts.synthesis import BuildingBlock, RouteStep, SynthesisRoute
from .._ext.adapter import ExternalToolAdapter


class RetrosynthesisError(RuntimeError):
    """The selected planner ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the engine was
    installed with its policy models and stock, the search started, and it did not
    return a usable tree.
    """


class RetrosynthesisToolAdapter(ExternalToolAdapter):
    """Shared route-tree flattening for the retrosynthesis backends."""

    produces_nothing = "plans no routes"

    def route_from_tree(
        self,
        tree: dict[str, Any],
        *,
        target_smiles: str,
        score: float | None,
        solved: bool | None,
        engine_version: str,
    ) -> SynthesisRoute:
        """Flatten one planner route tree into a :class:`SynthesisRoute`.

        The tree alternates molecule nodes and reaction nodes. Reactions become
        ordered :class:`RouteStep` records; molecule leaves with no children become
        :class:`BuildingBlock` records carrying whatever the engine said about
        stock. Depth-first, deepest first, so the returned order is the order a
        chemist would run it rather than the order the search discovered it.
        """
        steps: list[RouteStep] = []
        blocks: list[BuildingBlock] = []
        self._walk(tree, steps, blocks, depth=0)
        # The search builds the tree backwards from the target, so the deepest
        # reaction is the first bench step. Sorting by depth descending turns the
        # retrosynthetic tree into a forward synthesis.
        steps.sort(key=lambda pair: -pair.order)
        for order, step in enumerate(steps):
            step.order = order
        return SynthesisRoute(
            target_smiles=target_smiles,
            n_steps=len(steps),
            steps=steps,
            building_blocks=blocks,
            feasible=solved,
            overall_confidence=score,
            engine=self.engine,
            engine_version=engine_version,
            trust_label=TrustLabel.PREDICTED,
        )

    def _walk(
        self,
        node: dict[str, Any],
        steps: list[RouteStep],
        blocks: list[BuildingBlock],
        *,
        depth: int,
    ) -> None:
        """Collect reaction steps and stock leaves from one subtree."""
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        children = node.get("children") or []
        if kind == "reaction":
            meta = node.get("metadata") or {}
            reactants = [c.get("smiles", "") for c in children if isinstance(c, dict)]
            steps.append(
                RouteStep(
                    # Depth is carried in `order` and rewritten by the caller once
                    # the whole tree is known; a step cannot know its own position.
                    order=depth,
                    reaction_type=str(meta.get("classification") or "") or None,
                    description=str(node.get("smiles") or "reaction"),
                    reactants=[r for r in reactants if r],
                    product=None,
                    confidence=self._confidence(meta),
                    risk=Severity.LOW,
                )
            )
        elif kind == "mol" and not children:
            blocks.append(
                BuildingBlock(
                    smiles=str(node.get("smiles") or ""),
                    available=bool(node.get("in_stock", False)),
                    catalog=str(node.get("stock") or "") or None,
                    # The engine reports membership in a catalogue, not a price.
                    # Inventing a price class would be exactly the fabrication this
                    # package exists to have removed.
                    price_class=None,
                )
            )
        for child in children:
            self._walk(child, steps, blocks, depth=depth + 1)

    @staticmethod
    def _confidence(meta: dict[str, Any]) -> float | None:
        """The policy's own probability for this disconnection, if it reported one.

        This is the model's confidence that the template applies, not a probability
        that the reaction will work at the bench. It is passed through unchanged and
        clamped to the contract's range; ``None`` when the engine said nothing.
        """
        raw = meta.get("policy_probability")
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, value))

    @staticmethod
    def availability(blocks: list[BuildingBlock]) -> float | None:
        """Fraction of leaves the engine's stock file says are purchasable.

        ``None`` when there were no leaves at all — 0.0 would read as "we checked
        and found none", which is a different claim.
        """
        if not blocks:
            return None
        return round(sum(1 for b in blocks if b.available) / len(blocks), 4)


__all__ = ["RetrosynthesisError", "RetrosynthesisToolAdapter"]
