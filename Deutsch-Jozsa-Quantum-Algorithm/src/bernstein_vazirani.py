"""
Bernstein-Vazirani Algorithm Implementation

Extension of the Deutsch-Jozsa algorithm that finds a hidden bit string.
This demonstrates a practical application where quantum algorithms provide
exponential speedup over classical approaches.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
from typing import List, Dict, Optional


class BernsteinVazirani:
    """
    Implementation of the Bernstein-Vazirani algorithm.
    
    Given a function f(x) = s·x (mod 2) where s is a hidden bit string,
    the algorithm finds s with a single query, compared to n classical queries.
    
    This is a direct extension of Deutsch-Jozsa that solves a more practical problem.
    """
    
    def __init__(self, n_qubits: int = 3):
        """
        Initialize Bernstein-Vazirani algorithm.
        
        Args:
            n_qubits: Number of qubits (length of hidden string)
        """
        if n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.n_qubits = n_qubits
        self.simulator = AerSimulator()
        self.circuit = None
        self.result = None
        self.hidden_string = None
    
    def create_oracle(self, hidden_string: str) -> QuantumCircuit:
        """
        Create oracle for hidden bit string s.
        
        The oracle implements f(x) = s·x (mod 2) = s₀x₀ ⊕ s₁x₁ ⊕ ... ⊕ sₙ₋₁xₙ₋₁
        
        Args:
            hidden_string: Binary string (e.g., "101")
            
        Returns:
            QuantumCircuit: Oracle implementing the function
        """
        if len(hidden_string) != self.n_qubits:
            raise ValueError(f"Hidden string must have length {self.n_qubits}")
        
        if not all(c in '01' for c in hidden_string):
            raise ValueError("Hidden string must be binary (only 0s and 1s)")
        
        oracle = QuantumCircuit(self.n_qubits + 1, name=f'Oracle(s={hidden_string})')
        
        # Apply CNOT for each bit that is 1 in the hidden string
        for i, bit in enumerate(hidden_string):
            if bit == '1':
                oracle.cx(i, self.n_qubits)
        
        self.hidden_string = hidden_string
        return oracle
    
    def create_circuit(self, hidden_string: str) -> QuantumCircuit:
        """
        Create complete Bernstein-Vazirani circuit.
        
        Circuit structure (same as Deutsch-Jozsa):
        1. Initialize output qubit to |1⟩
        2. Apply Hadamard gates to all qubits
        3. Apply oracle
        4. Apply Hadamard gates to input qubits
        5. Measure input qubits
        
        Args:
            hidden_string: The hidden bit string to find
            
        Returns:
            QuantumCircuit: Complete circuit
        """
        # Create quantum and classical registers
        qr = QuantumRegister(self.n_qubits + 1, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Step 1: Initialize output qubit to |1⟩
        circuit.x(self.n_qubits)
        circuit.barrier()
        
        # Step 2: Apply Hadamard gates to all qubits
        for i in range(self.n_qubits + 1):
            circuit.h(i)
        circuit.barrier()
        
        # Step 3: Apply oracle
        oracle = self.create_oracle(hidden_string)
        circuit.compose(oracle, inplace=True)
        circuit.barrier()
        
        # Step 4: Apply Hadamard gates to input qubits
        for i in range(self.n_qubits):
            circuit.h(i)
        circuit.barrier()
        
        # Step 5: Measure input qubits
        circuit.measure(range(self.n_qubits), range(self.n_qubits))
        
        self.circuit = circuit
        return circuit
    
    def run(self, hidden_string: str, shots: int = 1024) -> str:
        """
        Run Bernstein-Vazirani algorithm to find the hidden string.
        
        Args:
            hidden_string: The hidden bit string (for oracle creation)
            shots: Number of measurement shots
            
        Returns:
            str: The discovered hidden string
        """
        # Create and run circuit
        circuit = self.create_circuit(hidden_string)
        
        # Transpile and execute
        transpiled = transpile(circuit, self.simulator)
        job = self.simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        self.result = counts
        
        # The measurement should give us the hidden string directly
        # Find the most common measurement outcome
        discovered_string = max(counts, key=counts.get)
        
        return discovered_string
    
    def get_counts(self) -> Dict[str, int]:
        """Get measurement counts from last run."""
        if self.result is None:
            raise RuntimeError("Algorithm has not been run yet")
        return self.result
    
    def get_circuit(self) -> QuantumCircuit:
        """Get the current circuit."""
        if self.circuit is None:
            raise RuntimeError("Circuit has not been created yet")
        return self.circuit
    
    def verify_result(self, discovered: str, expected: str) -> bool:
        """
        Verify if discovered string matches expected.
        
        Args:
            discovered: String found by algorithm
            expected: Expected hidden string
            
        Returns:
            bool: True if match
        """
        return discovered == expected


def compare_with_classical():
    """
    Compare Bernstein-Vazirani with classical approach.
    
    Classical approach needs n queries to determine each bit of the hidden string.
    Quantum approach needs only 1 query.
    """
    print("=" * 70)
    print("BERNSTEIN-VAZIRANI vs CLASSICAL COMPARISON")
    print("=" * 70)
    print()
    
    for n in [3, 5, 8, 10]:
        # Generate random hidden string
        hidden_string = ''.join(np.random.choice(['0', '1']) for _ in range(n))
        
        print(f"n = {n} bits")
        print(f"Hidden string: {hidden_string}")
        print()
        
        # Quantum approach
        bv = BernsteinVazirani(n_qubits=n)
        discovered = bv.run(hidden_string)
        
        print(f"  Quantum queries: 1")
        print(f"  Discovered: {discovered}")
        print(f"  Correct: {'✅' if discovered == hidden_string else '❌'}")
        print()
        
        # Classical approach
        print(f"  Classical queries: {n}")
        print(f"  Speedup: {n}x")
        print()
        print("-" * 70)
        print()


def demonstrate_bernstein_vazirani():
    """Demonstrate the Bernstein-Vazirani algorithm."""
    print("=" * 70)
    print("BERNSTEIN-VAZIRANI ALGORITHM DEMONSTRATION")
    print("=" * 70)
    print()
    
    print("The Bernstein-Vazirani algorithm finds a hidden bit string s")
    print("given a function f(x) = s·x (mod 2) with a single query.")
    print()
    
    # Test cases
    test_cases = [
        ("101", 3),
        ("1111", 4),
        ("10101", 5),
    ]
    
    for hidden_string, n_qubits in test_cases:
        print("-" * 70)
        print(f"Test: Finding hidden string of length {n_qubits}")
        print(f"Hidden string: {hidden_string}")
        print()
        
        bv = BernsteinVazirani(n_qubits=n_qubits)
        
        # Run algorithm
        discovered = bv.run(hidden_string, shots=1024)
        counts = bv.get_counts()
        
        print(f"Measurement counts: {counts}")
        print(f"Discovered string: {discovered}")
        print(f"Match: {'✅ YES' if discovered == hidden_string else '❌ NO'}")
        print()
    
    print("=" * 70)
    print()
    
    # Comparison with classical
    compare_with_classical()


def analyze_circuit_similarity():
    """
    Analyze similarity between Deutsch-Jozsa and Bernstein-Vazirani.
    
    Shows how BV is a generalization of DJ.
    """
    from src.deutsch_jozsa import DeutschJozsa
    
    print("=" * 70)
    print("DEUTSCH-JOZSA vs BERNSTEIN-VAZIRANI COMPARISON")
    print("=" * 70)
    print()
    
    n = 3
    
    # Deutsch-Jozsa (balanced XOR is special case of BV with s=111)
    print("Deutsch-Jozsa (balanced, XOR):")
    dj = DeutschJozsa(n_qubits=n)
    dj_circuit = dj.create_circuit('balanced', {'oracle_type': 'xor'})
    print(f"  Circuit depth: {dj_circuit.depth()}")
    print(f"  Gate count: {dj_circuit.size()}")
    print(f"  Problem: Determine if function is constant or balanced")
    print(f"  Output: One of two categories (constant/balanced)")
    print()
    
    # Bernstein-Vazirani with s=111 (same as DJ balanced XOR)
    print("Bernstein-Vazirani (s='111'):")
    bv = BernsteinVazirani(n_qubits=n)
    bv_circuit = bv.create_circuit('111')
    print(f"  Circuit depth: {bv_circuit.depth()}")
    print(f"  Gate count: {bv_circuit.size()}")
    print(f"  Problem: Find hidden bit string s")
    print(f"  Output: Exact bit string (e.g., '111')")
    print()
    
    print("Key Difference:")
    print("  • DJ: Determines property of function (constant vs balanced)")
    print("  • BV: Determines exact hidden string")
    print("  • DJ is a special case of BV (checks if s = 0...0 or not)")
    print()
    
    print("Similarity:")
    print("  • Both use same circuit structure")
    print("  • Both achieve exponential speedup")
    print("  • Both use quantum interference and phase kickback")
    print()


if __name__ == "__main__":
    demonstrate_bernstein_vazirani()
    print("\n")
    analyze_circuit_similarity()
