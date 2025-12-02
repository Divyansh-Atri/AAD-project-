"""Discrete-time quantum walk (DTQW) implementations on line and cycle graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

CoinType = Literal["hadamard", "balanced", "custom"]


def hadamard_coin() -> np.ndarray:
    """Return the canonical Hadamard coin."""

    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def balanced_coin(theta: float) -> np.ndarray:
    """Return a one-parameter balanced coin.

    Parameters
    ----------
    theta : float
        Rotation angle controlling bias between left/right directions.
    """

    return np.array(
        [
            [np.sin(theta), np.cos(theta)],
            [np.cos(theta), -np.sin(theta)],
        ],
        dtype=complex,
    )


@dataclass
class DiscreteTimeQuantumWalk1D:
    """DTQW on a one-dimensional lattice with two coin states.

    The walk supports both line (reflective boundaries) and cycle (periodic) graphs.
    """

    length: int
    topology: Literal["line", "cycle"] = "line"
    coin_type: CoinType = "hadamard"
    theta: float = np.pi / 4

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError("Walk length must be at least 2.")
        if self.topology not in {"line", "cycle"}:
            raise ValueError("Topology must be either 'line' or 'cycle'.")
        self.coin = self._build_coin()

    def _build_coin(self) -> np.ndarray:
        if self.coin_type == "hadamard":
            return hadamard_coin()
        if self.coin_type == "balanced":
            return balanced_coin(self.theta)
        raise ValueError("For custom coins, override the 'coin' attribute explicitly.")

    @property
    def hilbert_dimension(self) -> int:
        """Total Hilbert space dimension (position x coin)."""

        return self.length * 2

    def initial_state(
        self, position: int = 0, coin_state: Tuple[complex, complex] | None = None
    ) -> np.ndarray:
        """Prepare a localized initial state.

        Parameters
        ----------
        position : int
            Lattice site to localize the walker.
        coin_state : tuple of complex, optional
            Amplitudes for (left, right) coin basis. Defaults to |R>.
        """

        if position < 0 or position >= self.length:
            raise ValueError("Initial position outside lattice.")
        if coin_state is None:
            coin_state = (0.0, 1.0)
        state = np.zeros((self.length, 2), dtype=complex)
        state[position, 0] = coin_state[0]
        state[position, 1] = coin_state[1]
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("Initial state has zero norm.")
        return (state / norm).reshape(-1)

    def step(self, state: np.ndarray) -> np.ndarray:
        """Apply one full DTQW step (coin + shift)."""

        reshaped = state.reshape(self.length, 2)
        after_coin = (self.coin @ reshaped.T).T
        after_shift = np.zeros_like(after_coin)

        for position in range(self.length):
            left_amp, right_amp = after_coin[position]
            # Left move
            target_left = position - 1
            if self.topology == "cycle":
                target_left %= self.length
                after_shift[target_left, 0] += left_amp
            else:  # line with reflection
                if target_left < 0:
                    after_shift[0, 1] += left_amp
                else:
                    after_shift[target_left, 0] += left_amp
            # Right move
            target_right = position + 1
            if self.topology == "cycle":
                target_right %= self.length
                after_shift[target_right, 1] += right_amp
            else:
                if target_right >= self.length:
                    after_shift[self.length - 1, 0] += right_amp
                else:
                    after_shift[target_right, 1] += right_amp

        return after_shift.reshape(-1)

    def evolve(self, steps: int, state: np.ndarray) -> np.ndarray:
        """Iterate the walk for ``steps`` steps from ``state``."""

        current = state.copy()
        for _ in range(steps):
            current = self.step(current)
        return current

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return position probability distribution for ``state``."""

        reshaped = state.reshape(self.length, 2)
        return np.sum(np.abs(reshaped) ** 2, axis=1)

    def simulate(self, steps: int, position: int = 0) -> np.ndarray:
        """Convenience helper returning probability distributions over time."""

        histories = []
        current = self.initial_state(position)
        histories.append(self.probabilities(current))
        for _ in range(steps):
            current = self.step(current)
            histories.append(self.probabilities(current))
        return np.stack(histories)
