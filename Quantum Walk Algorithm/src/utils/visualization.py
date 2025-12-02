"""Visualization helpers for probability distributions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def plot_distribution_evolution(distributions: np.ndarray, title: str = "Evolution") -> None:
    """Line plot showing probability mass over positions."""

    steps, positions = distributions.shape
    plt.figure(figsize=(10, 4))
    for pos in range(positions):
        plt.plot(range(steps), distributions[:, pos], label=f"v{pos}")
    plt.xlabel("Step")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()


def interactive_surface(distributions: np.ndarray, title: str = "Quantum Walk") -> go.Figure:
    """Return a Plotly surface showing probability vs position/time."""

    steps = np.arange(distributions.shape[0])
    positions = np.arange(distributions.shape[1])
    fig = go.Figure(
        data=[
            go.Surface(
                z=distributions,
                x=positions,
                y=steps,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Position", yaxis_title="Step", zaxis_title="Probability"),
    )
    return fig
