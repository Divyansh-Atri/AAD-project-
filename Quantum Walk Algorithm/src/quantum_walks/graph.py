"""Graph utilities tailored for quantum and classical walk simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

ArrayLike = Sequence[Sequence[float]]


@dataclass(frozen=True)
class Graph:
    """Simple undirected graph represented by its adjacency matrix.

    Attributes
    ----------
    adjacency : np.ndarray
        Square (n x n) adjacency matrix with entries in {0, 1} for unweighted graphs
        or non-negative reals for weighted graphs.
    """

    adjacency: np.ndarray

    def __post_init__(self) -> None:
        adjacency = np.asarray(self.adjacency, dtype=float)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        if not np.allclose(adjacency, adjacency.T, atol=1e-10):
            raise ValueError("Graph must be undirected (adjacency symmetric).")
        object.__setattr__(self, "adjacency", adjacency)

    @property
    def size(self) -> int:
        """Return the number of vertices."""

        return self.adjacency.shape[0]

    @property
    def degree_matrix(self) -> np.ndarray:
        """Return the diagonal degree matrix."""

        degrees = np.sum(self.adjacency, axis=1)
        return np.diag(degrees)

    @property
    def laplacian(self) -> np.ndarray:
        """Return the combinatorial Laplacian L = D - A."""

        return self.degree_matrix - self.adjacency

    @classmethod
    def from_adjacency(cls, adjacency: ArrayLike) -> "Graph":
        """Construct a graph from an adjacency-like structure."""

        return cls(np.array(adjacency, dtype=float))

    @classmethod
    def line(cls, length: int) -> "Graph":
        """Create a path/line graph with ``length`` vertices."""

        if length < 2:
            raise ValueError("Line graph needs at least two vertices.")
        adjacency = np.zeros((length, length), dtype=float)
        for i in range(length - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0
        return cls(adjacency)

    @classmethod
    def cycle(cls, length: int) -> "Graph":
        """Create a cycle graph with ``length`` vertices."""

        if length < 3:
            raise ValueError("Cycle graph needs at least three vertices.")
        adjacency = cls.line(length).adjacency.copy()
        adjacency[0, -1] = 1.0
        adjacency[-1, 0] = 1.0
        return cls(adjacency)

    @classmethod
    def complete(cls, size: int) -> "Graph":
        """Create a complete graph K_n."""

        if size < 1:
            raise ValueError("Complete graph needs at least one vertex.")
        adjacency = np.ones((size, size), dtype=float) - np.eye(size)
        return cls(adjacency)

    def neighbors(self, vertex: int) -> List[int]:
        """Return the indices of neighbors of ``vertex``."""

        if vertex < 0 or vertex >= self.size:
            raise IndexError("Vertex index out of range.")
        return list(np.nonzero(self.adjacency[vertex])[0])
