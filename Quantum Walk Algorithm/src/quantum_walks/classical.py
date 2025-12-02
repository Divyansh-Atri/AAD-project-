"""Classical random walk implementations used for comparison baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .graph import Graph


@dataclass
class ClassicalRandomWalk:
    """Simple unbiased random walk on an undirected graph."""

    graph: Graph

    def __post_init__(self) -> None:
        degrees = self.graph.degree_matrix.diagonal()
        if np.any(degrees == 0):
            raise ValueError("Graph contains isolated vertices.")
        self.transition = self.graph.adjacency / degrees[:, None]

    def initial_distribution(self, vertex: int) -> np.ndarray:
        """Delta distribution concentrated at ``vertex``."""

        if vertex < 0 or vertex >= self.graph.size:
            raise ValueError("Vertex index out of range.")
        probs = np.zeros(self.graph.size)
        probs[vertex] = 1.0
        return probs

    def step(self, distribution: np.ndarray) -> np.ndarray:
        """Apply one random-walk step."""

        return distribution @ self.transition

    def simulate(self, steps: int, vertex: int = 0) -> np.ndarray:
        """Return probability distributions over ``steps`` iterations."""

        history = []
        current = self.initial_distribution(vertex)
        history.append(current)
        for _ in range(steps):
            current = self.step(current)
            history.append(current)
        return np.stack(history)

    def expected_hitting_time(self, start: int, target: int, horizon: int = 10_000) -> float:
        """Monte-Carlo estimate of hitting time using repeated simulations."""

        rng = np.random.default_rng(42)
        total_steps = 0
        trials = 1000
        for _ in range(trials):
            position = start
            for step in range(1, horizon + 1):
                if position == target:
                    total_steps += step - 1
                    break
                probs = self.transition[position]
                position = rng.choice(self.graph.size, p=probs)
            else:
                total_steps += horizon
        return total_steps / trials
