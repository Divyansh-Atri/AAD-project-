# Grover's Quantum Search Algorithm

A complete implementation of Grover's algorithm with comprehensive time and space complexity analysis, capable of running on both simulator and IBM Quantum hardware.

## Overview

Grover's algorithm is a quantum algorithm that finds a marked item in an unsorted database of N items in O(√N) time, providing a quadratic speedup over the O(N) time required by classical algorithms.

### Key Components

1.  **Oracle**: Marks the target state by flipping its phase.
2.  **Diffusion Operator**: Amplifies the amplitude of the marked state.
3.  **Superposition**: Initializes all states equally.
4.  **Measurement**: Collapses to the marked state with high probability.

## Features

*   **Single & Multiple State Search**: Find one or multiple target states simultaneously.
*   **IBM Quantum Integration**: Support for execution on real quantum computers.
*   **Automatic Configuration**: Loads API token securely from environment variables.
*   **Optimal Iterations**: Automatically calculates the optimal number of iterations: π/4 × √(N/M).
*   **Complexity Verification**: Statistical validation of O(√N) time complexity.
*   **Proportionality Analysis**: Verifies k = π/(4√M) for each marked state count.
*   **High-Precision Timing**: Uses nanosecond-level timing for accurate benchmarking.
*   **Visual Results**: Generates comprehensive plots and analysis.

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup IBM Quantum (Optional)

Create a `.env` file in the project directory:

```env
IBM_QUANTUM_TOKEN=your_token_here
```

You can obtain your token from the IBM Quantum dashboard.

### 4. Execution

**Simple implementation:**
```bash
python grovers_algorithm.py
```

**Comprehensive complexity benchmark (simulator):**
```bash
python batch_benchmark.py
```

**Run on IBM Quantum hardware:**
```bash
python batch_benchmark.py --ibm
```

## Configuration

You can configure the search parameters in `grovers_algorithm.py` or `batch_benchmark.py` within the `main()` function:

```python
n_qubits = 3  # Number of qubits (search space: 2^n states)

# Single state search
marked_states = '101'

# OR Multiple states search
marked_states = ['101', '110', '011']
```

## Scripts Description

### grovers_algorithm.py

This script provides a basic implementation with visualization. It is best for learning and quick demonstrations.
*   Demonstrates Grover's algorithm on a small example.
*   Compares simulator versus IBM hardware results.
*   Shows quantum speedup over classical search.

### batch_benchmark.py

This advanced benchmarking script performs rigorous statistical analysis to verify the theoretical complexity of Grover's algorithm.

#### Key Features

*   **High-Precision Timing**: Uses `time.perf_counter()` with nanosecond resolution.
*   **Statistical Validation**: Linear regression with R-squared goodness-of-fit analysis.
*   **Dual Analysis Mode**:
    *   Unfiltered: Shows raw performance of all circuits.
    *   Filtered: Uses high-accuracy circuits only (>50% IBM, >90% simulator).
*   **Complete Proportionality Verification**:
    *   Tests k = π/(4√M) for each marked state count M.
    *   Validates the theoretical formula across various M values.
*   **Comprehensive Metrics**:
    *   Time Complexity: O(√N) verification.
    *   Space Complexity: O(n) memory scaling.
    *   Circuit depth and gate count analysis.
    *   Success rate distributions.
    *   Quantum speedup calculations.

#### Test Configuration

*   **2-10 qubits**: Covers search spaces from 4 to 1024 states.
*   **Systematic tests**: Approximately 100-120 tests covering critical fractions and random variations.
*   **Individual Circuit Timing**: Measures each circuit individually for precise build, transpile, and execution times.

## Output Analysis

The benchmark confirms the theoretical proportionality with high accuracy.

*   **Single marked state (M=1)**: Verifies the canonical Grover constant π/4.
*   **Complete formula verification**: Confirms k = π/(4√M) holds across all M values.
*   **Linear relationship**: Proves Iterations ∝ √N with R-squared > 0.99.

### Generated Files

**Basic Implementation:**
*   `grover_simulator_results.png`: Simulator results visualization.
*   `grover_ibm_results.png`: IBM Quantum hardware results.

**Complexity Benchmark:**
Each run generates data files and plots for both unfiltered and filtered analysis, including:
*   Detailed data text files with timing information.
*   Main complexity plots (3x3 grid).
*   Space/time scaling details.
*   Proportionality verification plots.

## Requirements

*   Python 3.8+
*   Qiskit 1.0+
*   qiskit-ibm-runtime
*   qiskit-aer
*   matplotlib
*   numpy
*   python-dotenv
*   IBM Quantum account (optional, for hardware access)

## Algorithm Details

### Time Complexity
*   **Classical**: O(N)
*   **Quantum**: O(√N)
*   **Verification**: Linear regression confirms Iterations ∝ √N.

### Space Complexity
*   **Qubits**: O(n) where n is the number of qubits.
*   **States**: O(2^n).
*   **Memory**: State vector requires 2^n × 16 bytes.

### Success Probability
For optimal iterations, the probability of measuring the marked state approaches 100% in ideal conditions. Real quantum hardware will have lower success rates due to noise and decoherence.

## Resources

*   [Qiskit Documentation](https://qiskit.org/documentation/)
*   [IBM Quantum Experience](https://quantum.ibm.com/)
*   [Grover's Algorithm Paper](https://arxiv.org/abs/quant-ph/9605043)
*   [Proportionality Summary](PROPORTIONALITY_SUMMARY.md)
