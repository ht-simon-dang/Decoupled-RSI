"""Plan retrosynthetic routes for one target with AiZynthFinder — the drug-agent ⇄ AiZynth bridge.

This file is executed by the interpreter that has AiZynthFinder installed, which on
this machine is the conda environment ``drug-agent-aizynth`` (Python 3.11,
aizynthfinder 4.4.1, rdkit 2023.09.6, numpy < 2). The platform's own environment
pins rdkit 2026.03.5 and numpy 2.5, so the two cannot share an interpreter and the
JSON request/response pair on disk is the whole interface.

It imports nothing from ``drug_agent``: only the standard library and
``aizynthfinder`` itself, exactly as ``scripts/external/_driver.py`` requires.

**Why the raw route tree is returned rather than a route object.** The caller owns
the mapping from a planner's bipartite molecule/reaction tree to this project's
``SynthesisRoute`` contract — that mapping lives in
``drug_agent.tools.retrosynthesis.RetrosynthesisToolAdapter.route_from_tree`` and is
shared with every other planner. This driver's job is to run the search faithfully
and hand back what AiZynthFinder actually produced, unsummarised.

**The one annotation added to the tree.** ``ReactionTree.to_dict()`` records whether
a leaf is ``in_stock`` but not *which* stock it was found in. The names come from
``AiZynthFinder.stock_info()``, which asks the loaded stock the same question, and
are written onto each leaf as ``stock`` so the caller can fill ``BuildingBlock.catalog``
with a catalogue the engine really matched rather than a placeholder.

Request JSON:
    target_smiles       SMILES of the molecule to plan a route to (required)
    config_file         path to AiZynthFinder's config.yml (required)
    stock               stock keys to select, e.g. ["zinc"]
    expansion           expansion policy keys, e.g. ["uspto"]
    filter              filter policy keys, e.g. ["uspto"]
    iteration_limit     MCTS iterations before the search stops
    time_limit          seconds before the search stops
    max_transforms      maximum retrosynthetic depth
    max_routes          most routes to extract after the search
    min_routes          fewest routes to extract after the search
    return_first        stop at the first solved route
    out_dir             directory to write result.json (injected by the caller)

Response JSON (``out_dir/result.json``):
    target_smiles, engine_version, is_solved, search_time_s
    routes[]            {tree, score, scores, is_solved, created_at_iteration, n_steps}
    stats               AiZynthFinder's own ``extract_statistics()``, JSON-safe
    stock_info          {leaf smiles: [stock names]}
    selected            {stock, expansion, filter}
    warnings[]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported after the sys.path insert above: the shared driver contract lives beside this file.
import _driver

#: AiZynthFinder's multi-objective search returns a score *dict*; with the default
#: single reward that dict holds one key, and this is it.
_DEFAULT_REWARD = "state score"


def _select(collection: Any, keys: list[str], what: str, warnings: list[str]) -> list[str]:
    """Select *keys* on one of AiZynthFinder's context collections, skipping absent ones.

    A key named in the request that the config never loaded is a configuration
    error worth reporting, not a crash: the search can still run on whatever else
    was asked for, and the caller sees which selection did not happen.
    """
    loaded = list(collection.items)
    chosen = [key for key in keys if key in loaded]
    for key in keys:
        if key not in loaded:
            warnings.append(f"{what} {key!r} is not in the config (loaded: {', '.join(loaded) or 'none'})")
    if chosen:
        collection.select(chosen)
    return chosen


def _scalar_score(score: Any) -> float | None:
    """One float out of AiZynthFinder's score, whatever shape this version uses.

    4.4.1 reports ``{"state score": 0.99}`` because the search is multi-objective
    internally even with one objective configured. Older versions returned a bare
    float. Anything else is reported as no score rather than guessed at.
    """
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    if isinstance(score, dict):
        if _DEFAULT_REWARD in score:
            return float(score[_DEFAULT_REWARD])
        for value in score.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _scores_dict(score: Any) -> dict[str, float]:
    """Every objective the search scored this route on, as floats."""
    if not isinstance(score, dict):
        value = _scalar_score(score)
        return {_DEFAULT_REWARD: value} if value is not None else {}
    out: dict[str, float] = {}
    for key, value in score.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = float(value)
    return out


def _annotate_stock(node: Any, stock_info: dict[str, list[str]]) -> int:
    """Write catalogue names onto in-stock leaves; return the reaction count below *node*.

    ``to_dict`` drops the ``children`` key entirely on a leaf, so "no children" is
    how a building block identifies itself.
    """
    if not isinstance(node, dict):
        return 0
    children = node.get("children") or []
    if node.get("type") == "mol" and not children:
        names = stock_info.get(node.get("smiles", "")) or []
        if names:
            node["stock"] = ",".join(names)
    n_steps = 1 if node.get("type") == "reaction" else 0
    for child in children:
        n_steps += _annotate_stock(child, stock_info)
    return n_steps


def _jsonable(value: Any) -> Any:
    """Coerce AiZynthFinder's statistics into something ``json.dumps`` accepts."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run one AiZynthFinder tree search and return its routes."""
    from aizynthfinder.aizynthfinder import AiZynthFinder

    target = str(request.get("target_smiles") or "").strip()
    if not target:
        raise ValueError("request carries no target_smiles")
    config_file = str(request.get("config_file") or "")
    if not config_file or not Path(config_file).is_file():
        raise FileNotFoundError(f"AiZynthFinder config not found at {config_file!r}")

    warnings: list[str] = []
    finder = AiZynthFinder(configfile=config_file)

    selected = {
        "stock": _select(finder.stock, list(request.get("stock") or ["zinc"]), "stock", warnings),
        "expansion": _select(
            finder.expansion_policy, list(request.get("expansion") or ["uspto"]), "expansion policy", warnings
        ),
        "filter": _select(
            finder.filter_policy, list(request.get("filter") or ["uspto"]), "filter policy", warnings
        ),
    }
    if not selected["stock"]:
        raise ValueError("no stock could be selected; every leaf would read as unpurchasable")
    if not selected["expansion"]:
        raise ValueError("no expansion policy could be selected; the search cannot propose a disconnection")

    search = finder.config.search
    search.iteration_limit = int(request.get("iteration_limit") or search.iteration_limit)
    search.time_limit = int(request.get("time_limit") or search.time_limit)
    search.max_transforms = int(request.get("max_transforms") or search.max_transforms)
    search.return_first = bool(request.get("return_first", False))
    post = finder.config.post_processing
    post.max_routes = int(request.get("max_routes") or post.max_routes)
    post.min_routes = int(request.get("min_routes") or post.min_routes)

    finder.target_smiles = target
    search_time = float(finder.tree_search())
    finder.build_routes()
    stats = _jsonable(finder.extract_statistics())
    stock_info = {str(k): [str(n) for n in v] for k, v in finder.stock_info().items()}

    routes: list[dict[str, Any]] = []
    for index in range(len(finder.routes)):
        entry = finder.routes[index]
        tree = entry["reaction_tree"].to_dict()
        n_steps = _annotate_stock(tree, stock_info)
        metadata = entry.get("route_metadata") or {}
        routes.append(
            {
                "tree": tree,
                "score": _scalar_score(entry.get("score")),
                "scores": _scores_dict(entry.get("all_score") or entry.get("score")),
                "is_solved": bool(metadata.get("is_solved", False)),
                "created_at_iteration": metadata.get("created_at_iteration"),
                "n_steps": n_steps,
            }
        )

    return {
        "target_smiles": finder.target_smiles,
        "engine_version": _engine_version(),
        "is_solved": bool(stats.get("is_solved", False)),
        "search_time_s": search_time,
        "routes": routes,
        "stats": stats,
        "stock_info": stock_info,
        "selected": selected,
        "warnings": warnings,
    }


def _engine_version() -> str:
    """The installed aizynthfinder distribution version, or ``unknown``."""
    try:
        from importlib.metadata import version

        return str(version("aizynthfinder"))
    except Exception:
        # A missing version must not fail a search that otherwise worked.
        return "unknown"


if __name__ == "__main__":
    _driver.main(run)
