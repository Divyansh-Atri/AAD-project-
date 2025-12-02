# Deutsch-Jozsa Algorithm - Getting Started

Complete guide to install, run, and deploy on IBM Quantum hardware.

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd "AAD Quantum Project"
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Run tests
pytest -v

# Should show: 36 tests passing ✅
```

---

## 🚀 Quick Start

### Basic Usage

```python
from src.deutsch_jozsa import DeutschJozsa

# Create instance
dj = DeutschJozsa(n_qubits=3)

# Test constant function
result = dj.run('constant', {'output_value': 0})
print(f"Result: {result}")  # Output: 'constant'

# Get measurement counts
counts = dj.get_counts()
print(f"Counts: {counts}")  # {'000': 1024}
```

### Run Tests
```bash
# All tests
pytest -v

# Specific test
pytest tests/test_deutsch_jozsa.py -v

# With coverage
pytest --cov=src tests/
```

### Quick Simulator Test
```bash
python quick_test.py
```

---

## 🖥️ IBM Quantum Hardware (Optional)

### Step 1: Get IBM Quantum Access

1. **Sign up** (FREE): https://quantum.ibm.com/
2. **Get API token**: Profile → Account → Copy API Token
3. **Get CRN**: Go to Instances page → Copy your CRN

### Step 2: Configure Credentials

```bash
# Copy template
cp setup_ibm_template.py setup_ibm.py

# Edit setup_ibm.py and add:
# - YOUR_TOKEN = "your_api_token_here"
# - YOUR_CRN = "your_crn_here"

# Run setup
python setup_ibm.py
```

### Step 3: Check Available Backends
```bash
python test_backends.py
```

### Step 4: Run on Real Hardware
```bash
python run_on_hardware.py
```

**Note:** This submits a job to IBM Quantum. May take 5-30 minutes depending on queue.

---

## 📚 Project Structure

```
src/
  ├── deutsch_jozsa.py        # Main algorithm
  ├── oracles.py              # Oracle construction
  ├── visualization.py        # Plotting tools
  ├── analysis.py             # Classical comparison
  ├── error_mitigation.py     # Error mitigation
  ├── bernstein_vazirani.py   # BV algorithm
  └── hardware_deployment.py  # IBM Quantum

tests/                        # 36 unit tests
docs/                         # Documentation
notebooks/                    # Jupyter tutorials
examples/                     # Usage examples
```

---

## 📖 Documentation

- **README.md** - Main documentation
- **docs/theory.md** - Mathematical foundations
- **docs/results.md** - Experimental results
- **docs/hardware_results.md** - Real hardware execution
- **notebooks/deutsch_jozsa_tutorial.ipynb** - Interactive tutorial

---

## 🧪 Examples

### Visualize Circuit
```python
from src.deutsch_jozsa import DeutschJozsa
from src.visualization import DJVisualizer

dj = DeutschJozsa(n_qubits=3)
circuit = dj.create_circuit('constant', {'output_value': 0})

visualizer = DJVisualizer()
visualizer.plot_circuit(circuit)
```

### Compare Query Complexity
```python
from src.visualization import DJVisualizer

visualizer = DJVisualizer()
visualizer.compare_query_complexity(n_qubits_range=[2, 3, 4, 5])
```

### Classical vs Quantum
```python
from src.analysis import AlgorithmComparator

comparator = AlgorithmComparator(n_qubits=3)
comparison = comparator.compare_query_complexity()
print(comparison)
# {'quantum': 1, 'classical_deterministic': 5, ...}
```

---

## ❓ Troubleshooting

### Tests Failing?
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Run tests again
pytest -v
```

### Import Errors?
```bash
# Make sure you're in project directory
cd "/path/to/AAD Quantum Project"

# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "from src.deutsch_jozsa import DeutschJozsa; print('✅')"
```

### IBM Quantum Connection Issues?
- Verify your API token is correct
- Check your CRN instance is correct
- Ensure you copied the ENTIRE token (no spaces)
- Token format: 44 characters, lowercase letters and numbers only

---

## 🎓 Learning Resources

1. **Start here:** `notebooks/deutsch_jozsa_tutorial.ipynb`
2. **Theory:** `docs/theory.md`
3. **Examples:** `examples/simple_example.py`
4. **Real results:** `docs/hardware_results.md`

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 🎯 Quick Commands Reference

```bash
# Install
pip install -r requirements.txt

# Test
pytest -v

# Run simulator
python quick_test.py

# Check backends
python test_backends.py

# Run on hardware
python run_on_hardware.py

# Interactive tutorial
jupyter notebook notebooks/deutsch_jozsa_tutorial.ipynb
```

---

**Need help?** Check the README.md or open an issue on GitHub.
