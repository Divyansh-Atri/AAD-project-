# Quantum Walk Algorithms

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## Overview

**Quantum Walk Algorithms** is a comprehensive research and implementation project exploring the dynamics of quantum walks on graphs. This repository provides "from-scratch" Python implementations of core quantum algorithms, avoiding black-box libraries for the algorithmic logic to ensure transparency and educational value.

Key implementations include:
*   **Discrete-Time Quantum Walks (DTQW):** On line and cycle graphs with configurable coins (Hadamard, Balanced).
*   **Continuous-Time Quantum Walks (CTQW):** Driven by adjacency or Laplacian Hamiltonians.
*   **Classical Random Walks:** For baseline comparison of mixing and hitting times.
*   **Grover's Search:** A reference implementation of the amplitude amplification algorithm.
*   **Spatial Search:** DTQW-based search on graphs.


## Features

*   **No Black Boxes:** Algorithms are implemented from first principles using NumPy/SciPy.
*   **Rich Visualization:** Interactive Plotly surfaces and Matplotlib static plots for analyzing probability distributions.
*   **IBM Quantum Integration:** Run Grover's algorithm on real quantum hardware via Qiskit Runtime.

## Project Structure

```
├── src/
│   ├── quantum_walks/    # Core algorithm implementations
│   └── utils/            # Visualization and metrics
├── notebooks/            # Interactive theory and demos
```

## Installation


1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Interactive Notebook

The best way to explore the algorithms is through the interactive notebook:

```bash
jupyter notebook notebooks/quantum_walks_showcase.ipynb
```

### Library Usage

You can import the algorithms directly in your Python scripts:

```python
from src.quantum_walks.dtqw import DiscreteTimeQuantumWalk1D

# Initialize a walk on a line of length 21
walk = DiscreteTimeQuantumWalk1D(length=21, topology="line")
state = walk.initial_state(position=10)

# Evolve for 10 steps
final_state = walk.evolve(steps=10, state=state)
probs = walk.probabilities(final_state)
print(probs)
```


## License

This project is licensed under the MIT License.
