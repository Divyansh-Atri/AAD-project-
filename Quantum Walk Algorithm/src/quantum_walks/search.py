"""Quantum walk spatial search routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtqw import DiscreteTimeQuantumWalk1D


@dataclass
class SpatialSearch1D:
    """Spatial search using DTQW with a phase-marked vertex."""

    walk: DiscreteTimeQuantumWalk1D
    marked_vertex: int

    def __post_init__(self) -> None:
        if self.marked_vertex < 0 or self.marked_vertex >= self.walk.length:
            raise ValueError("Marked vertex outside lattice.")

    def initial_state(self) -> np.ndarray:
        uniform = np.ones(self.walk.length * 2, dtype=complex)
        return uniform / np.linalg.norm(uniform)

    def oracle(self, state: np.ndarray) -> np.ndarray:
        reshaped = state.reshape(self.walk.length, 2)
        reshaped[self.marked_vertex] *= -1
        return reshaped.reshape(-1)

    def step(self, state: np.ndarray) -> np.ndarray:
        marked = self.oracle(state)
        return self.walk.step(marked)

    def run(self, steps: int) -> np.ndarray:
        state = self.initial_state()
        history = []
        for _ in range(steps):
            state = self.step(state)
            history.append(self.walk.probabilities(state))
        return np.array(history)
