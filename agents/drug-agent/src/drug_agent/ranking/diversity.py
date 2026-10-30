"""Diversity: scaffold/fingerprint clustering + diverse shortlist selection.

Ensures the shortlist is chemically diverse (acceptance criterion #12) rather
than a cluster of near-duplicates. Clustering is a simple, deterministic
Tanimoto/Butina-style sphere-exclusion over Morgan fingerprints.
"""

from __future__ import annotations

import numpy as np

from ..tools.rdkit_tool import Cheminformatics


def _tanimoto_matrix(fps: np.ndarray) -> np.ndarray:
    inter = fps @ fps.T
    counts = fps.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim


def cluster_by_scaffold(
    smiles_list: list[str], *, cutoff: float = 0.6, chem: Cheminformatics | None = None
) -> list[int]:
    """Butained-style sphere-exclusion clustering. Returns per-item cluster id.

    Deterministic: seeds are chosen by descending neighbour count, ties broken
    by index.
    """
    chem = chem or Cheminformatics()
    n = len(smiles_list)
    if n == 0:
        return []
    fps = np.vstack([chem.fingerprint(s) for s in smiles_list]).astype(np.float64)
    sim = _tanimoto_matrix(fps)
    neighbours = (sim >= cutoff).sum(axis=1)
    order = sorted(range(n), key=lambda i: (-neighbours[i], i))
    assigned = [-1] * n
    cluster_id = 0
    for seed in order:
        if assigned[seed] != -1:
            continue
        members = [j for j in range(n) if assigned[j] == -1 and sim[seed, j] >= cutoff]
        for m in members:
            assigned[m] = cluster_id
        cluster_id += 1
    return assigned


def select_diverse(
    candidate_ids: list[str],
    smiles_list: list[str],
    scores: list[float],
    k: int,
    *,
    clusters: list[int] | None = None,
    chem: Cheminformatics | None = None,
) -> list[str]:
    """Pick ``k`` diverse candidates: best-per-cluster first, then max-min fill.

    ``scores`` is higher-is-better (e.g. utility or -pareto_rank). Returns the
    selected candidate IDs in selection order.
    """
    chem = chem or Cheminformatics()
    n = len(candidate_ids)
    if n == 0 or k <= 0:
        return []
    if clusters is None:
        clusters = cluster_by_scaffold(smiles_list, chem=chem)
    fps = np.vstack([chem.fingerprint(s) for s in smiles_list]).astype(np.float64)

    # 1) best-scoring member of each cluster (round-robin by cluster best score)
    by_cluster: dict[int, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    cluster_best = {c: max(idxs, key=lambda i: (scores[i], -i)) for c, idxs in by_cluster.items()}
    ordered_clusters = sorted(cluster_best, key=lambda c: (-scores[cluster_best[c]], c))

    selected: list[int] = []
    for c in ordered_clusters:
        if len(selected) >= k:
            break
        selected.append(cluster_best[c])

    # 2) fill remaining slots via max-min diversity to the selected set
    while len(selected) < k and len(selected) < n:
        remaining = [i for i in range(n) if i not in selected]
        if not remaining:
            break
        sel_fp = fps[selected]

        def min_sim(i: int, sel_fp: np.ndarray = sel_fp) -> float:
            inter = sel_fp @ fps[i]
            union = sel_fp.sum(axis=1) + fps[i].sum() - inter
            sims = np.where(union > 0, inter / union, 0.0)
            return float(sims.max()) if len(sims) else 0.0

        # choose the remaining item least similar to current selection,
        # tie-broken by higher score then lower index
        pick = min(remaining, key=lambda i: (min_sim(i), -scores[i], i))
        selected.append(pick)

    return [candidate_ids[i] for i in selected[:k]]


def mean_pairwise_distance(smiles_list: list[str], chem: Cheminformatics | None = None) -> float:
    """1 - mean pairwise Tanimoto over the set (diversity metric)."""
    chem = chem or Cheminformatics()
    if len(smiles_list) < 2:
        return 0.0
    fps = np.vstack([chem.fingerprint(s) for s in smiles_list]).astype(np.float64)
    sim = _tanimoto_matrix(fps)
    iu = np.triu_indices(len(smiles_list), k=1)
    return float(1.0 - sim[iu].mean())
