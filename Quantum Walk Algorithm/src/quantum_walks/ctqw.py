"""Continuous-time quantum walk (CTQW) implementation for small graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .graph import Graph

HamiltonianChoice = Literal["adjacency", "laplacian"]


@dataclass
class ContinuousTimeQuantumWalk:
    """Continuous-time quantum walk driven by a graph Hamiltonian."""

    graph: Graph
    gamma: float = 1.0
    hamiltonian_type: HamiltonianChoice = "adjacency"

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("Gamma must be positive.")
        if self.hamiltonian_type == "adjacency":
            self.hamiltonian = self.graph.adjacency
        elif self.hamiltonian_type == "laplacian":
            self.hamiltonian = self.graph.laplacian
        else:
            raise ValueError("Unknown Hamiltonian choice.")
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.hamiltonian)

    @property
    def dimension(self) -> int:
        """Return the Hilbert space dimension (number of vertices)."""

        return self.graph.size

    def initial_state(self, vertex: int) -> np.ndarray:
        """Return a delta localized state at ``vertex``."""

        if vertex < 0 or vertex >= self.dimension:
            raise ValueError("Vertex index out of range.")
        state = np.zeros(self.dimension, dtype=complex)
        state[vertex] = 1.0
        return state

    def evolve(self, state: np.ndarray, time: float) -> np.ndarray:
        """Evolve ``state`` for duration ``time`` using spectral decomposition."""

        phases = np.exp(-1j * self.gamma * self._eigenvalues * time)
        coeffs = self._eigenvectors.conj().T @ state
        evolved = self._eigenvectors @ (phases * coeffs)
        return evolved

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return vertex occupation probabilities."""

        return np.abs(state) ** 2

    def simulate(self, times: Iterable[float], vertex: int = 0) -> np.ndarray:
        """Return probability distributions for each time in ``times``."""

        state = self.initial_state(vertex)
        history = []
        for t in times:
            evolved = self.evolve(state, t)
            history.append(self.probabilities(evolved))
        return np.stack(history)
