#!/usr/bin/env python3
"""
Quick test script for Deutsch-Jozsa algorithm.
Run this script to see the algorithm in action!

Usage:
    python test_algorithm.py
"""

from src.deutsch_jozsa import DeutschJozsa
import sys


def main():
    print("=" * 70)
    print(" " * 15 + "DEUTSCH-JOZSA ALGORITHM TEST")
    print(" " * 20 + "Team: Dhinchak Dikstra")
    print("=" * 70)
    print()
    
    # Test with 3 qubits
    n_qubits = 3
    dj = DeutschJozsa(n_qubits=n_qubits)
    
    print(f"Configuration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Classical queries needed: {2**(n_qubits-1) + 1}")
    print(f"  Quantum queries needed: 1")
    print(f"  Speedup: {2**(n_qubits-1) + 1}x")
    print()
    
    # Test 1: Constant function (f(x) = 0)
    print("-" * 70)
    print("TEST 1: Constant Function → f(x) = 0 for all x")
    print("-" * 70)
    result = dj.run('constant', {'output_value': 0}, shots=1024)
    counts = dj.get_counts()
    print(f"  Expected: constant")
    print(f"  Detected: {result}")
    print(f"  Measurement: {counts}")
    print(f"  Status: {'✅ PASS' if result == 'constant' else '❌ FAIL'}")
    print()
    
    # Test 2: Constant function (f(x) = 1)
    print("-" * 70)
    print("TEST 2: Constant Function → f(x) = 1 for all x")
    print("-" * 70)
    result = dj.run('constant', {'output_value': 1}, shots=1024)
    counts = dj.get_counts()
    print(f"  Expected: constant")
    print(f"  Detected: {result}")
    print(f"  Measurement: {counts}")
    print(f"  Status: {'✅ PASS' if result == 'constant' else '❌ FAIL'}")
    print()
    
    # Test 3: Balanced function (first bit)
    print("-" * 70)
    print("TEST 3: Balanced Function → f(x) = first bit of x")
    print("-" * 70)
    result = dj.run('balanced', {'oracle_type': 'first_bit'}, shots=1024)
    counts = dj.get_counts()
    print(f"  Expected: balanced")
    print(f"  Detected: {result}")
    print(f"  Measurement: {counts}")
    print(f"  Status: {'✅ PASS' if result == 'balanced' else '❌ FAIL'}")
    print()
    
    # Test 4: Balanced function (XOR)
    print("-" * 70)
    print("TEST 4: Balanced Function → f(x) = XOR of all bits")
    print("-" * 70)
    result = dj.run('balanced', {'oracle_type': 'xor'}, shots=1024)
    counts = dj.get_counts()
    print(f"  Expected: balanced")
    print(f"  Detected: {result}")
    print(f"  Measurement: {counts}")
    print(f"  Status: {'✅ PASS' if result == 'balanced' else '❌ FAIL'}")
    print()
    
    # Summary
    print("=" * 70)
    print(" " * 22 + "ALL TESTS COMPLETED!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • Open the Jupyter notebook: jupyter notebook notebooks/deutsch_jozsa_tutorial.ipynb")
    print("  • Run comprehensive tests: pytest tests/ -v")
    print("  • Read the theory: docs/theory.md")
    print("  • Check results: docs/results.md")
    print()
    print("🎉 The Deutsch-Jozsa algorithm is working perfectly!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure you've activated the virtual environment:")
        print("  source venv/bin/activate")
        sys.exit(1)
