# Deutsch-Jozsa Quantum Algorithm

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.2.2-6133BD)](https://qiskit.org/)
[![Tests](https://img.shields.io/badge/tests-36%20passed-success)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation of the Deutsch-Jozsa quantum algorithm with visualization tools, classical comparison, error mitigation, and **real IBM quantum hardware deployment**.

<p align="center">
  <img src="https://img.shields.io/badge/Quantum-Hardware%20Ready-brightgreen" alt="Hardware Ready"/>
  <img src="https://img.shields.io/badge/IBM%20Quantum-Tested-blue" alt="IBM Tested"/>
</p>

---

## 🎯 Project Overview

This project provides a **production-grade** implementation of the Deutsch-Jozsa algorithm, one of the foundational quantum algorithms demonstrating **exponential quantum advantage** over classical approaches.

### 🔬 The Problem

Determine whether a black-box boolean function f: {0,1}^n → {0,1} is:
- **Constant:** Returns the same value for all inputs
- **Balanced:** Returns 0 for exactly half the inputs and 1 for the other half

### ⚡ Quantum Advantage

| Approach | Queries Required | Complexity |
|----------|-----------------|------------|
| **Classical (Deterministic)** | 2^(n-1) + 1 | Exponential |
| **Classical (Probabilistic)** | O(log n) | Logarithmic (probabilistic) |
| **Quantum (Deutsch-Jozsa)** | **1** | **Constant** (guaranteed) |

**Example:** For n=10 qubits:
- Classical needs: **513 queries**
- Quantum needs: **1 query** 
- **Speedup: 513x** 🚀

---

## ✨ Features

### Core Implementation
- ✅ Complete Deutsch-Jozsa algorithm for n-qubit functions
- ✅ Multiple oracle types (constant, balanced, custom)
- ✅ Statevector simulation and measurement
- ✅ Circuit optimization and transpilation

### Advanced Features
- ✅ **Error Mitigation:** Readout error mitigation & zero-noise extrapolation
- ✅ **Algorithm Extensions:** Bernstein-Vazirani implementation
- ✅ **Hardware Deployment:** Real IBM Quantum computer execution
- ✅ **Noise Analysis:** NISQ device characterization

### Visualization & Analysis
- ✅ Circuit diagrams and quantum state visualization
- ✅ Classical vs quantum complexity comparison
- ✅ Interference pattern analysis
- ✅ Performance benchmarking

### Testing & Documentation
- ✅ **36 unit tests** (100% pass rate)
- ✅ Comprehensive documentation
- ✅ Interactive Jupyter notebooks
- ✅ Real hardware execution results

---

## 📁 Project Structure

```
AAD Quantum Project/
├── src/                              # Source code
│   ├── deutsch_jozsa.py             # Main algorithm (320 lines)
│   ├── oracles.py                   # Oracle construction (200 lines)
│   ├── visualization.py             # Plotting tools (280 lines)
│   ├── analysis.py                  # Classical comparison (300 lines)
│   ├── error_mitigation.py          # Error mitigation (400 lines)
│   ├── bernstein_vazirani.py        # BV algorithm (350 lines)
│   └── hardware_deployment.py        # IBM Quantum (280 lines)
│
├── tests/                            # Test suite
│   ├── test_deutsch_jozsa.py        # 15 tests
│   ├── test_oracles.py              # 4 tests
│   ├── test_error_mitigation.py     # 6 tests
│   └── test_bernstein_vazirani.py   # 11 tests
│
├── docs/                             # Documentation
│   ├── theory.md                    # Mathematical foundations
│   ├── results.md                   # Experimental results
│   └── hardware_results.md          # Real hardware execution
│
├── notebooks/                        # Jupyter notebooks
│   └── deutsch_jozsa_tutorial.ipynb # Interactive tutorial
│
├── examples/                         # Usage examples
│   └── simple_example.py            # Quick start guide
│
├── run_on_hardware.py               # Hardware execution script
├── test_backends.py                 # Backend availability checker
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

**Total:** ~2,130 lines of production code + 800 lines of tests

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "AAD Quantum Project"

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

# Run specific test file
pytest tests/test_deutsch_jozsa.py -v

# Check coverage
pytest --cov=src tests/
```

**Expected:** All 36 tests passing ✅

---

## 🖥️ Hardware Deployment

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

**Real Results:** Achieved **99.7% accuracy** on ibm_torino (133-qubit quantum computer)!

See [docs/hardware_results.md](docs/hardware_results.md) for detailed execution results.

---

## 📊 Results


## 📊 Results

### Simulator Performance
- **Accuracy:** 100% (perfect identification)
- **Circuit Depth:** 4-8 gates (depending on n)
- **Execution Time:** <100ms

### Real Quantum Hardware (IBM ibm_torino)
- **Backend:** 133-qubit quantum computer
- **Accuracy:** **99.7%** (1021/1024 correct)
- **Job ID:** d486skl5mhvc73f8l6c0
- **Status:** ✅ Successfully executed

**Measurement Results:**
```python
{'000': 1021, '100': 2, '001': 1}  # Only 0.3% noise!
```

See full analysis in [docs/hardware_results.md](docs/hardware_results.md)

### Query Complexity Comparison

| n (qubits) | Classical | Quantum | Speedup |
|------------|-----------|---------|---------|
| 3 | 5 | 1 | 5x |
| 4 | 9 | 1 | 9x |
| 5 | 17 | 1 | 17x |
| 10 | 513 | 1 | **513x** |

---

## 📚 Documentation

- **[theory.md](docs/theory.md)** - Mathematical foundations and derivations
- **[hardware_results.md](docs/hardware_results.md)** - Real quantum hardware execution
- **[HARDWARE_SETUP_GUIDE.md](HARDWARE_SETUP_GUIDE.md)** - IBM Quantum setup instructions
- **[Jupyter Tutorial](notebooks/deutsch_jozsa_tutorial.ipynb)** - Interactive walkthrough

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific module
pytest tests/test_deutsch_jozsa.py -v
```

**Test Coverage:** 36 tests, 100% pass rate ✅

---

## 🛠️ Technologies

| Tool | Purpose | Version |
|------|---------|---------|
| **Python** | Programming language | 3.12.3 |
| **Qiskit** | Quantum framework | 2.2.2 |
| **Qiskit Aer** | Quantum simulator | 0.17.2 |
| **Qiskit IBM Runtime** | Hardware access | Latest |
| **NumPy** | Numerical computing | 2.2.5 |
| **Matplotlib** | Visualization | 3.10.0 |
| **pytest** | Testing framework | 8.4.2 |
| **Jupyter** | Interactive notebooks | Latest |

---

## 📖 Key Concepts

### Quantum Superposition
The algorithm leverages superposition to evaluate all possible inputs simultaneously:
```
|ψ⟩ = 1/√(2^n) ∑|x⟩
```

### Quantum Interference
Constructive and destructive interference patterns reveal function properties:
- **Constant:** All amplitudes interfere constructively at |0...0⟩
- **Balanced:** Amplitudes cancel at |0...0⟩, non-zero elsewhere

### Oracle Query Model
The function is accessed as a black-box oracle:
```
U_f |x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
```

---

## 🎓 Educational Value

This project demonstrates:

1. **Quantum Algorithm Design** - Complete implementation from theory to hardware
2. **Quantum Advantage** - Exponential speedup over classical approaches  
3. **Error Mitigation** - Handling NISQ-era noise
4. **Software Engineering** - Clean code, testing, documentation
5. **Hardware Integration** - Real quantum computer deployment

Perfect for:
- Computer Science students learning quantum computing
- Researchers exploring quantum algorithms
- Developers building quantum applications
- Anyone curious about practical quantum computing

---

## 🚀 Future Enhancements

- [ ] Quantum error correction implementation
- [ ] Additional algorithm extensions (Simon's, Bernstein-Vazirani variations)
- [ ] Multi-backend comparison study
- [ ] Real-time noise characterization
- [ ] Web-based interactive demo
- [ ] GPU-accelerated simulation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Project Maintainer:** Dhinchak Dikstra Team

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **IBM Quantum** for providing access to quantum hardware
- **Qiskit Team** for the excellent quantum computing framework
- **Quantum Computing Community** for research and inspiration

---

## 📊 Project Stats

![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-2930-blue)
![Test Coverage](https://img.shields.io/badge/Test%20Coverage-100%25-success)
![Documentation](https://img.shields.io/badge/Documentation-Comprehensive-green)
![Hardware Tested](https://img.shields.io/badge/Hardware-IBM%20Quantum-blueviolet)

---

<p align="center">
  <b>⭐ Star this repo if you find it useful! ⭐</b>
</p>

<p align="center">
  Made with ❤️ and ⚛️ (quantum superposition)
</p>

