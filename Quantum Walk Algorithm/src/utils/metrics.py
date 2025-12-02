"""Utility functions for analyzing walk dynamics."""

from __future__ import annotations

import numpy as np


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute total variation distance between distributions."""

    if p.shape != q.shape:
        raise ValueError("Distributions must share shape.")
    return 0.5 * float(np.sum(np.abs(p - q)))


def mixing_time(distributions: np.ndarray, stationary: np.ndarray, epsilon: float = 0.05) -> int:
    """Return the first index where TV-distance <= epsilon."""

    for idx, dist in enumerate(distributions):
        if total_variation_distance(dist, stationary) <= epsilon:
            return idx
    return len(distributions) - 1


def peak_hitting_probability(distributions: np.ndarray, target: int) -> tuple[int, float]:
    """Return the step index and value of maximum probability on the target vertex."""

    probs = distributions[:, target]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])
