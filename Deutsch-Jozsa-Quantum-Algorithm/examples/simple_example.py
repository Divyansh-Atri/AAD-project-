"""
Simple example demonstrating Deutsch-Jozsa algorithm usage.

This script shows the basic usage of the Deutsch-Jozsa algorithm
implementation for quick testing and demonstration.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deutsch_jozsa import DeutschJozsa
from visualization import DJVisualizer
import matplotlib.pyplot as plt


def simple_example():
    """Run a simple example of the Deutsch-Jozsa algorithm."""
    
    print("=" * 70)
    print("DEUTSCH-JOZSA ALGORITHM - SIMPLE EXAMPLE")
    print("Team: Dhinchak Dikstra")
    print("=" * 70)
    print()
    
    # Create algorithm instance
    n_qubits = 3
    dj = DeutschJozsa(n_qubits=n_qubits)
    
    print(f"Number of qubits: {n_qubits}")
    print(f"Classical queries needed (deterministic): {2**(n_qubits-1) + 1}")
    print(f"Quantum queries needed: 1")
    print(f"Speedup: {2**(n_qubits-1) + 1}x")
    print()
    
    # Test 1: Constant function
    print("-" * 70)
    print("TEST 1: Constant Function (f(x) = 0)")
    print("-" * 70)
    
    result = dj.run(
        function_type='constant',
        oracle_params={'output_value': 0},
        shots=1024
    )
    
    counts = dj.get_counts()
    
    print(f"Result: {result}")
    print(f"Measurement counts: {counts}")
    print(f"Success: {'✅' if result == 'constant' else '❌'}")
    print()
    
    # Test 2: Balanced function
    print("-" * 70)
    print("TEST 2: Balanced Function (XOR)")
    print("-" * 70)
    
    result = dj.run(
        function_type='balanced',
        oracle_params={'oracle_type': 'xor'},
        shots=1024
    )
    
    counts = dj.get_counts()
    
    print(f"Result: {result}")
    print(f"Measurement counts: {counts}")
    print(f"Success: {'✅' if result == 'balanced' else '❌'}")
    print()
    
    # Visualize
    print("-" * 70)
    print("VISUALIZATION")
    print("-" * 70)
    print("Generating plots...")
    
    viz = DJVisualizer()
    
    # Create comparison plot
    fig = viz.compare_query_complexity(range(1, 11))
    plt.savefig('query_complexity_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: query_complexity_comparison.png")
    
    plt.close()
    
    # Get and visualize circuit
    circuit = dj.get_circuit()
    circuit_fig = circuit.draw(output='mpl', style='iqp')
    plt.savefig('deutsch_jozsa_circuit.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: deutsch_jozsa_circuit.png")
    
    plt.close()
    
    print()
    print("=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Check the generated PNG files")
    print("2. Open notebooks/deutsch_jozsa_tutorial.ipynb for full tutorial")
    print("3. Run tests with: pytest tests/ -v")
    print("4. Read docs/theory.md for mathematical details")
    print()


if __name__ == "__main__":
    simple_example()
