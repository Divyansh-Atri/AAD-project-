# Quantum Random Number Generator (QRNG)

A C++ implementation of a Quantum Random Number Generator (QRNG) with comprehensive statistical analysis of multiple RNG algorithms.

## Quick Start

### Prerequisites
- C++17 compatible compiler (GCC/Clang)
- CMake 3.14+
- Git
- curl (required for TRUE_IBM_QUANTUM mode in the comparison tool)

### Installation

```bash
# From the repository root
mkdir -p build && cd build
cmake .. && cmake --build .
```

## Running the QRNG

### Basic Usage
Run the QRNG with default settings (1 qubit, 1000 shots):
```bash
./qrng_app
```

### Custom Parameters
```bash
./qrng_app --qubits 2 --shots 10000 --algorithm MERSENNE_TWISTER --seed 42
```

### Available Algorithms
- `MERSENNE_TWISTER` - Industry standard PRNG (default)
- `XOSHIRO` - Fast, high-quality PRNG
- `PCG` - Modern alternative with good statistical properties
- `QUANTUM_SIMULATED` - Simulated quantum measurements
- `TRUE_IBM_QUANTUM` - Real quantum source via network (falls back to local source if offline)

Note: The main `qrng_app` supports the first four algorithms. `TRUE_IBM_QUANTUM` is exercised by the `compare_algorithms` tool.

### Optional: IBM Quantum client (Python)
Generate true quantum bits using IBM/Qiskit (saved to `true_quantum.bin`):
```bash
python3 fetch_ibm_bits.py
```
Requirements: Python 3, qiskit, qiskit-ibm-runtime, and an IBM Quantum API token configured as per your environment.

Security note: The script currently contains an `API_TOKEN` constant. Replace it with your token before running, or modify the script to read from an environment variable for better security.

## Comparing Algorithms

### Run All Algorithms
Compare all implemented algorithms with the same parameters:
```bash
./compare_algorithms [qubits] [shots] [seed]
```
Example:
```bash
./compare_algorithms 1 100000 42
```

### Understanding the Output
The comparison tool provides:
- Bits generated and time taken
- 0s/1s ratio and distribution
- Shannon and min entropy values
- Statistical test results (chi-square, runs test)
- Pass/fail status for all tests

### Example Output
```
=== Testing Mersenne Twister ===
Bits generated: 10000
Time taken: 0.040529 ms
0s/1s ratio: 4939/5061 (49.39% / 50.61%)
Shannon entropy: 0.999893 bits/bit
Min entropy: 0.982506 bits/bit
Chi-square p-value: 0.222465
Runs test p-value: 0.480651
All tests passed: YES
```

## Features

### Implemented Algorithms
- **Mersenne Twister (MT19937)**: Industry standard PRNG
- **Xoshiro256**: High-quality, fast PRNG
- **PCG**: Modern PRNG with excellent properties
- **Quantum Simulated**: Simulates quantum measurements
- **True IBM Quantum**: Attempts to fetch bits from an online QRNG source with graceful fallback

### Statistical Analysis
- **Frequency tests**: Monobit, block frequency
- **Runs tests**: Tests for independence of bits
- **Entropy analysis**: Shannon and min entropy calculations
- **Performance metrics**: Generation speed and memory usage

## Testing Methodology

### Test Setup
- **Hardware**: Standard x86_64 system
- **OS**: Linux
- **Compiler**: g++ 13.3.0
- **Test Parameters**:
  - Qubits: 1
  - Shots: 100,000
  - Seed: 42 (for reproducibility)

### Statistical Tests Performed
1. **Frequency Test**
   - Measures the proportion of 0s and 1s
   - Ideal: ~50% each
   
2. **Runs Test**
   - Tests for independence between consecutive bits
   - p-value > 0.05 indicates random behavior

3. **Entropy Analysis**
   - **Shannon Entropy**: Measures information density (max 1.0)
   - **Min Entropy**: Measures predictability (higher is better)

### Test Results Summary

| Algorithm | Time (ms) | 0s/1s Ratio | Shannon Entropy | Min Entropy | Chi-square p-value | Runs p-value |
|-----------|----------:|------------:|----------------:|------------:|-------------------:|-------------:|
| Mersenne Twister | 3.03 | 50.10/49.90 | 0.999997 | 0.997002 | 0.510696 | 0.774897 |
| Xoshiro256** | 1.06 | 49.90/50.10 | 0.999997 | 0.997117 | 0.527089 | 0.161812 |
| PCG | 0.40 | 50.23/49.77 | 0.999985 | 0.993494 | 0.152904 | 0.785546 |
| Quantum Simulated | 36.33 | 49.89/50.11 | 0.999997 | 0.996916 | 0.498579 | 0.733795 |

### Interpreting Results
- **All tests passed** for all algorithms (p-values > 0.05)
- **PCG** is the fastest algorithm
- **Quantum Simulated** provides true randomness but is slower
- **Mersenne Twister** and **Xoshiro** offer a good balance

## Documentation

- [Detailed Results](docs/DETAILED_RESULTS.md) - Example runs and explanations

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Building the Project

### Linux/macOS

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Optional: Run tests
ctest --output-on-failure
```

### Windows

```cmd
# Create build directory
mkdir build
cd build

# Configure with CMake (adjust generator if needed)
cmake .. -G "Visual Studio 16 2019" -A x64

# Build the project
cmake --build . --config Release

# Optional: Run tests
ctest -C Release --output-on-failure
```

## Usage

### Command Line Interface

```
Usage: ./qrng_app [--qubits N] [--shots N] [--seed N] [--algorithm ALGO]
Options:
  --qubits N       Number of qubits (default: 1)
  --shots N        Number of measurement shots (default: 1000)
  --seed N         Random seed (default: 42)
  --algorithm      One of: MERSENNE_TWISTER, XOSHIRO, PCG, QUANTUM_SIMULATED
  --help           Show this help message

Examples:
  ./qrng_app --qubits 4 --shots 10000
  ./qrng_app --algorithm PCG --shots 50000 --seed 42
```

### Example Output

```
QRNG Generation Results:
------------------------
Qubits: 1
Shots: 1000
Total bits: 1000
Ones: 486 (48.6%)
Zeros: 514 (51.4%)
Shannon Entropy: 0.999434 bits/bit
Min Entropy: 0.96016 bits/bit
Chi-square test p-value: 0.375921
Runs test p-value: 0.732758
Generation time: 0.04156 ms

First 20 bits: 11101010000111111000
```

## Benchmarking

To run the benchmark:

```bash
./compare_algorithms 1 100000 42
```

This compares all algorithms with a fixed configuration and prints per-algorithm statistics.

## Project Structure

```
project-QRNG/
	├── CMakeLists.txt          # Main build configuration
	├── README.md               # Project overview and documentation
	├── fetch_ibm_bits.py      # Python script to fetch true quantum bits from IBM
	│
	├── docs/                  # Additional documentation
	│   └── DETAILED_RESULTS.md # Example runs and explanations
	│
	├── include/               # Public header files
	│   └── qrng.h             # Main QRNG class interface
	│
	├── src/                   # C++ implementation files
	│   ├── qrng.cpp           # Core QRNG implementation
	│   ├── main.cpp           # Command-line interface (builds qrng_app)
	│   └── compare_algorithms.cpp  # Algorithm comparison tool
	│
	└── true_quantum.bin      # (Optional) binary file generated by fetch_ibm_bits.py
```

### Key Components

- **QRNG Core (`include/qrng.h`, `src/qrng.cpp`)**
  - Implements multiple RNG algorithms
  - Handles random bit generation and statistical analysis
  - Configurable parameters (qubits, shots, seed)

- **Command-line Tools**
  - `qrng_app`: Main application for random number generation
  - `compare_algorithms`: Tool to compare different RNG algorithms


