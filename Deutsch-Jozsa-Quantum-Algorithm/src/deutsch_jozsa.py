"""
Deutsch-Jozsa Algorithm Implementation

This module provides a comprehensive implementation of the Deutsch-Jozsa algorithm,
demonstrating quantum advantage for determining whether a function is constant or balanced.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
from typing import Literal, Dict, Optional, List


class DeutschJozsa:
    """
    Implementation of the Deutsch-Jozsa algorithm.
    
    The algorithm determines whether a boolean function f: {0,1}^n -> {0,1}
    is constant (same output for all inputs) or balanced (different outputs 
    for exactly half of the inputs) using a single query.
    
    Attributes:
        n_qubits (int): Number of input qubits
        simulator: Qiskit Aer simulator backend
    """
    
    def __init__(self, n_qubits: int = 3):
        """
        Initialize the Deutsch-Jozsa algorithm.
        
        Args:
            n_qubits: Number of input qubits (default: 3)
        """
        if n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.n_qubits = n_qubits
        self.simulator = AerSimulator()
        self.circuit = None
        self.result = None
        
    def create_constant_oracle(self, output_value: int = 0) -> QuantumCircuit:
        """
        Create an oracle for a constant function.
        
        A constant function returns the same value (0 or 1) for all inputs.
        
        Args:
            output_value: The constant output value (0 or 1)
            
        Returns:
            QuantumCircuit: Oracle circuit
        """
        oracle = QuantumCircuit(self.n_qubits + 1, name=f'Constant({output_value})')
        
        if output_value == 1:
            # Flip the output qubit to make f(x) = 1 for all x
            oracle.x(self.n_qubits)
        # If output_value == 0, do nothing (identity operation)
        
        return oracle
    
    def create_balanced_oracle(self, oracle_type: str = 'random') -> QuantumCircuit:
        """
        Create an oracle for a balanced function.
        
        A balanced function returns 0 for half the inputs and 1 for the other half.
        
        Args:
            oracle_type: Type of balanced oracle to create:
                        'random' - randomly generated balanced function
                        'first_bit' - function depends on first qubit
                        'xor' - XOR of all input bits
                        
        Returns:
            QuantumCircuit: Oracle circuit
        """
        oracle = QuantumCircuit(self.n_qubits + 1, name=f'Balanced({oracle_type})')
        
        if oracle_type == 'first_bit':
            # Function depends only on the first qubit
            oracle.cx(0, self.n_qubits)
            
        elif oracle_type == 'xor':
            # Function is XOR of all input qubits
            for i in range(self.n_qubits):
                oracle.cx(i, self.n_qubits)
                
        elif oracle_type == 'random':
            # Create a random balanced function
            # Apply random pattern of CNOTs
            np.random.seed()  # Use current time for randomness
            
            # Randomly choose which qubits to use as controls
            num_controls = np.random.randint(1, self.n_qubits + 1)
            control_qubits = np.random.choice(self.n_qubits, num_controls, replace=False)
            
            for control in control_qubits:
                oracle.cx(control, self.n_qubits)
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
        
        return oracle
    
    def create_circuit(self, 
                      function_type: Literal['constant', 'balanced'] = 'constant',
                      oracle_params: Optional[Dict] = None) -> QuantumCircuit:
        """
        Create the complete Deutsch-Jozsa circuit.
        
        Circuit structure:
        1. Initialize qubits (input in |0⟩, output in |1⟩)
        2. Apply Hadamard gates to all qubits
        3. Apply oracle
        4. Apply Hadamard gates to input qubits
        5. Measure input qubits
        
        Args:
            function_type: Type of function ('constant' or 'balanced')
            oracle_params: Parameters for oracle creation
            
        Returns:
            QuantumCircuit: Complete Deutsch-Jozsa circuit
        """
        if oracle_params is None:
            oracle_params = {}
        
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
        if function_type == 'constant':
            output_value = oracle_params.get('output_value', 0)
            oracle = self.create_constant_oracle(output_value)
        elif function_type == 'balanced':
            oracle_type = oracle_params.get('oracle_type', 'random')
            oracle = self.create_balanced_oracle(oracle_type)
        else:
            raise ValueError(f"Unknown function type: {function_type}")
        
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
    
    def run(self, 
            function_type: Literal['constant', 'balanced'] = 'constant',
            oracle_params: Optional[Dict] = None,
            shots: int = 1024) -> str:
        """
        Run the Deutsch-Jozsa algorithm.
        
        Args:
            function_type: Type of function to test
            oracle_params: Parameters for oracle creation
            shots: Number of measurement shots
            
        Returns:
            str: Result ('constant' or 'balanced')
        """
        # Create circuit
        circuit = self.create_circuit(function_type, oracle_params)
        
        # Transpile for simulator
        transpiled_circuit = transpile(circuit, self.simulator)
        
        # Run simulation
        job = self.simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        self.result = counts
        
        # Analyze result
        # If all measurements are '000...0', function is constant
        # Otherwise, function is balanced
        zero_state = '0' * self.n_qubits
        
        if zero_state in counts and len(counts) == 1:
            return 'constant'
        else:
            return 'balanced'
    
    def get_counts(self) -> Dict[str, int]:
        """
        Get measurement counts from the last run.
        
        Returns:
            Dict mapping measurement outcomes to counts
        """
        if self.result is None:
            raise RuntimeError("Algorithm has not been run yet")
        return self.result
    
    def get_circuit(self) -> QuantumCircuit:
        """
        Get the current circuit.
        
        Returns:
            QuantumCircuit: The current circuit
        """
        if self.circuit is None:
            raise RuntimeError("Circuit has not been created yet")
        return self.circuit
    
    def get_statevector(self, 
                       function_type: Literal['constant', 'balanced'] = 'constant',
                       oracle_params: Optional[Dict] = None) -> np.ndarray:
        """
        Get the statevector at different stages of the algorithm.
        
        Args:
            function_type: Type of function
            oracle_params: Oracle parameters
            
        Returns:
            np.ndarray: Final statevector
        """
        from qiskit_aer import AerSimulator
        
        # Create circuit without measurement
        qr = QuantumRegister(self.n_qubits + 1, 'q')
        circuit = QuantumCircuit(qr)
        
        # Build circuit stages
        circuit.x(self.n_qubits)
        for i in range(self.n_qubits + 1):
            circuit.h(i)
        
        if function_type == 'constant':
            output_value = oracle_params.get('output_value', 0) if oracle_params else 0
            oracle = self.create_constant_oracle(output_value)
        else:
            oracle_type = oracle_params.get('oracle_type', 'random') if oracle_params else 'random'
            oracle = self.create_balanced_oracle(oracle_type)
        
        circuit.compose(oracle, inplace=True)
        
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Save statevector
        circuit.save_statevector()
        
        # Run with statevector simulator
        sv_sim = AerSimulator(method='statevector')
        job = sv_sim.run(circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        return np.array(statevector)


def demonstrate_algorithm():
    """
    Demonstrate the Deutsch-Jozsa algorithm with examples.
    """
    print("=" * 70)
    print("DEUTSCH-JOZSA ALGORITHM DEMONSTRATION")
    print("=" * 70)
    
    n_qubits = 3
    dj = DeutschJozsa(n_qubits=n_qubits)
    
    print(f"\nTesting with {n_qubits} qubits")
    print(f"Classical deterministic queries needed: {2**(n_qubits-1) + 1}")
    print("Quantum queries needed: 1\n")
    
    # Test constant function (output = 0)
    print("-" * 70)
    print("Test 1: Constant function (always returns 0)")
    result = dj.run(function_type='constant', oracle_params={'output_value': 0})
    print(f"Result: {result}")
    print(f"Measurement counts: {dj.get_counts()}")
    
    # Test constant function (output = 1)
    print("\n" + "-" * 70)
    print("Test 2: Constant function (always returns 1)")
    result = dj.run(function_type='constant', oracle_params={'output_value': 1})
    print(f"Result: {result}")
    print(f"Measurement counts: {dj.get_counts()}")
    
    # Test balanced function
    print("\n" + "-" * 70)
    print("Test 3: Balanced function (XOR of all bits)")
    result = dj.run(function_type='balanced', oracle_params={'oracle_type': 'xor'})
    print(f"Result: {result}")
    print(f"Measurement counts: {dj.get_counts()}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_algorithm()
