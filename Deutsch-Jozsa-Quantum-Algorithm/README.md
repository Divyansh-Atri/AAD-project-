# Deutsch-Jozsa Quantum Algorithm

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.2.2-6133BD)](https://qiskit.org/)


A comprehensive implementation of the Deutsch-Jozsa quantum algorithm with visualization tools, classical comparison, error mitigation, and real IBM quantum hardware deployment.

## Project Overview

This project provides a production-grade implementation of the Deutsch-Jozsa algorithm, one of the foundational quantum algorithms demonstrating exponential quantum advantage over classical approaches.

### The Problem

Determine whether a black-box boolean function f: {0,1}^n -> {0,1} is:
- **Constant:** Returns the same value for all inputs
- **Balanced:** Returns 0 for exactly half the inputs and 1 for the other half

### Quantum Advantage

| Approach | Queries Required | Complexity |
|----------|-----------------|------------|
| **Classical (Deterministic)** | 2^(n-1) + 1 | Exponential |
| **Classical (Probabilistic)** | O(log n) | Logarithmic (probabilistic) |
| **Quantum (Deutsch-Jozsa)** | **1** | **Constant** (guaranteed) |

**Example:** For n=10 qubits:
- Classical needs: **513 queries**
- Quantum needs: **1 query**
- **Speedup: 513x**

## Features

### Core Implementation
- Complete Deutsch-Jozsa algorithm for n-qubit functions
- Multiple oracle types (constant, balanced, custom)
- Statevector simulation and measurement
- Circuit optimization and transpilation

### Advanced Features
- **Error Mitigation:** Readout error mitigation & zero-noise extrapolation
- **Hardware Deployment:** Real IBM Quantum computer execution
- **Noise Analysis:** NISQ device characterization

### Visualization & Analysis
- Circuit diagrams and quantum state visualization
- Classical vs quantum complexity comparison
- Interference pattern analysis
- Performance benchmarking

## Project Structure

```
Deutsch-Jozsa-Quantum-Algorithm/
├── src/                              # Source code
│   ├── deutsch_jozsa.py             # Main algorithm
│   ├── oracles.py                   # Oracle construction
│   ├── visualization.py             # Plotting tools
│   ├── analysis.py                  # Classical comparison
│   ├── error_mitigation.py          # Error mitigation
│   └── hardware_deployment.py        # IBM Quantum
│
├── tests/                            # Test suite
│   ├── test_deutsch_jozsa.py
│   ├── test_oracles.py
│   └── test_error_mitigation.py
│
├── docs/                             # Documentation
│   ├── theory.md
│   ├── results.md
│   └── hardware_results.md
│
├── notebooks/                        # Jupyter notebooks
│   └── deutsch_jozsa_tutorial.ipynb
│
├── examples/                         # Usage examples
│   └── simple_example.py
│
├── run_on_hardware.py               # Hardware execution script
├── test_backends.py                 # Backend availability checker
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Deutsch-Jozsa-Quantum-Algorithm"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.deutsch_jozsa import DeutschJozsa

# Create instance for 3-qubit function
dj = DeutschJozsa(n_qubits=3)

# Test constant function
result = dj.run('constant', {'output_value': 0})
print(f"Result: {result}")  # Output: 'constant'
counts = dj.get_counts()
print(f"Measurements: {counts}")  # {'000': 1024}

# Test balanced function
dj2 = DeutschJozsa(n_qubits=3)
result = dj2.run('balanced', {'balance_type': 'first_bit'})
print(f"Result: {result}")  # Output: 'balanced'
```

### Run Tests

```bash
# Run all tests
pytest -v
```

## Hardware Deployment

### Run on Real IBM Quantum Computer

1. **Get IBM Quantum Credentials:**
   - Sign up at [quantum.ibm.com](https://quantum.ibm.com)
   - Get API token from Account page
   - Copy your instance CRN

2. **Configure Credentials:**
   ```bash
   cp setup_ibm_template.py setup_ibm.py
   # Edit setup_ibm.py with your credentials
   python setup_ibm.py
   ```

3. **Execute on Hardware:**
   ```bash
   python run_on_hardware.py
   ```

## Results

### Simulator Performance
- **Accuracy:** 100% (perfect identification)
- **Circuit Depth:** 4-8 gates (depending on n)
- **Execution Time:** <100ms

### Real Quantum Hardware (IBM ibm_torino)
- **Backend:** 133-qubit quantum computer
- **Accuracy:** **99.7%** (1021/1024 correct)
- **Status:** Successfully executed

### Query Complexity Comparison

| n (qubits) | Classical | Quantum | Speedup |
|------------|-----------|---------|---------|
| 3 | 5 | 1 | 5x |
| 4 | 9 | 1 | 9x |
| 5 | 17 | 1 | 17x |
| 10 | 513 | 1 | **513x** |

