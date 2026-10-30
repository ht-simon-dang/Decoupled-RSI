"""Pareto-front extraction and hypervolume (§10 step 5, §12.3 portfolio metrics).

All objectives are internally converted to *maximization* of normalized [0,1]
scores by the caller, so this module works purely on "higher is better" vectors.
"""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (>= on all, > on at least one)."""
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated rows.

    ``points`` is (n, d); higher is better on every dimension. NaNs are treated
    as the worst possible value (a missing objective cannot help a candidate).
    """
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    pts = np.where(np.isnan(points), -np.inf, points)
    n = pts.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j or not on_front[j]:
                continue
            if dominates(pts[j], pts[i]) and not np.array_equal(pts[i], pts[j]):
                on_front[i] = False
                break
    return on_front


def pareto_rank(points: np.ndarray) -> np.ndarray:
    """Assign a non-domination rank (1 = first front) to every point."""
    pts = np.where(np.isnan(points), -np.inf, points)
    n = pts.shape[0]
    ranks = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    current = 1
    while remaining.any():
        idx = np.where(remaining)[0]
        sub = pts[idx]
        front = pareto_front(sub)
        for k, is_front in zip(idx, front, strict=True):
            if is_front:
                ranks[k] = current
                remaining[k] = False
        current += 1
    return ranks


def hypervolume(front: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Hypervolume dominated by the front w.r.t. a reference (worst) point.

    Uses an inclusion–exclusion-free grid-free approach for the common 2–3D
    case via sorting; falls back to a Monte-Carlo estimate for higher d. Inputs
    are maximization, normalized to [0,1]; default reference is the origin.
    """
    if front.size == 0:
        return 0.0
    pts = np.atleast_2d(front)
    d = pts.shape[1]
    ref = reference if reference is not None else np.zeros(d)
    pts = pts[np.all(pts >= ref, axis=1)]
    if pts.size == 0:
        return 0.0
    if d == 1:
        return float(pts.max() - ref[0])
    if d == 2:
        # Sweep points by descending x; accumulate rectangles of increasing y.
        order = np.argsort(-pts[:, 0])
        pts = pts[order]
        hv = 0.0
        prev_y = ref[1]
        for x, y in pts:
            if y <= prev_y:
                continue
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
        return float(hv)
    # d >= 3: Monte-Carlo estimate in the unit box scaled to bounds.
    rng = np.random.default_rng(12345)
    hi = pts.max(axis=0)
    samples = rng.uniform(ref, hi, size=(20000, d))
    dominated = np.array([np.any(np.all(pts >= s, axis=1)) for s in samples])
    vol_box = float(np.prod(hi - ref))
    return float(dominated.mean() * vol_box)
