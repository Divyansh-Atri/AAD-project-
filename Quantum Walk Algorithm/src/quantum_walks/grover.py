"""Grover search reference implementation for small instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GroverSearch:
    """Grover's amplitude amplification for a single marked item."""

    num_items: int
    target_index: int

    def __post_init__(self) -> None:
        if self.num_items <= 1:
            raise ValueError("Need at least two items for Grover search.")
        if self.target_index < 0 or self.target_index >= self.num_items:
            raise ValueError("Target index out of range.")
        self.state = np.ones(self.num_items, dtype=complex) / np.sqrt(self.num_items)

    @staticmethod
    def oracle(num_items: int, target_index: int) -> np.ndarray:
        oracle_matrix = np.eye(num_items, dtype=complex)
        oracle_matrix[target_index, target_index] = -1
        return oracle_matrix

    @staticmethod
    def diffusion(num_items: int) -> np.ndarray:
        return 2 * np.ones((num_items, num_items), dtype=complex) / num_items - np.eye(
            num_items, dtype=complex
        )

    def step(self) -> None:
        """Apply one Grover iteration (oracle + diffusion)."""

        oracle = self.oracle(self.num_items, self.target_index)
        diffusion = self.diffusion(self.num_items)
        self.state = diffusion @ (oracle @ self.state)

    def run(self, iterations: int | None = None) -> List[float]:
        """Run Grover iterations and record success probability each step."""

        if iterations is None:
            iterations = int(np.floor(np.pi * np.sqrt(self.num_items) / 4))
        probs = []
        for _ in range(iterations):
            self.step()
            probs.append(self.success_probability)
        return probs

    @property
    def success_probability(self) -> float:
        """Return probability of measuring the marked item."""

        return float(np.abs(self.state[self.target_index]) ** 2)
