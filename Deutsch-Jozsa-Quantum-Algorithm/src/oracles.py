"""
Oracle Construction Utilities for Deutsch-Jozsa Algorithm

This module provides utilities for creating various types of oracles
for the Deutsch-Jozsa algorithm.
"""

from qiskit import QuantumCircuit
import numpy as np
from typing import List, Optional


class OracleFactory:
    """Factory class for creating various types of quantum oracles."""
    
    @staticmethod
    def create_custom_balanced_oracle(n_qubits: int, 
                                      control_pattern: List[int]) -> QuantumCircuit:
        """
        Create a balanced oracle with a custom control pattern.
        
        Args:
            n_qubits: Number of input qubits
            control_pattern: List of qubit indices to use as controls
            
        Returns:
            QuantumCircuit: Custom balanced oracle
        """
        oracle = QuantumCircuit(n_qubits + 1, name='CustomBalanced')
        
        for control in control_pattern:
            if 0 <= control < n_qubits:
                oracle.cx(control, n_qubits)
        
        return oracle
    
    @staticmethod
    def create_multi_controlled_oracle(n_qubits: int,
                                       control_qubits: List[int],
                                       flip_output: bool = True) -> QuantumCircuit:
        """
        Create an oracle with multi-controlled operations.
        
        Args:
            n_qubits: Number of input qubits
            control_qubits: Indices of control qubits
            flip_output: Whether to flip output when all controls are |1⟩
            
        Returns:
            QuantumCircuit: Multi-controlled oracle
        """
        oracle = QuantumCircuit(n_qubits + 1, name='MultiControlled')
        
        if len(control_qubits) == 1:
            oracle.cx(control_qubits[0], n_qubits)
        elif len(control_qubits) == 2:
            oracle.ccx(control_qubits[0], control_qubits[1], n_qubits)
        else:
            # For more than 2 controls, use MCX gate
            oracle.mcx(control_qubits, n_qubits)
        
        return oracle
    
    @staticmethod
    def verify_oracle_balance(n_qubits: int, oracle: QuantumCircuit) -> dict:
        """
        Verify whether an oracle implements a balanced or constant function.
        
        This method tests the oracle on all possible inputs to determine
        if it's truly balanced or constant.
        
        Args:
            n_qubits: Number of input qubits
            oracle: Oracle circuit to verify
            
        Returns:
            dict: Verification results including function type and output distribution
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        outputs = []
        
        # Test all possible inputs
        for i in range(2**n_qubits):
            # Create circuit for testing
            test_circuit = QuantumCircuit(n_qubits + 1, 1)
            
            # Prepare input state
            binary = format(i, f'0{n_qubits}b')
            for j, bit in enumerate(binary):
                if bit == '1':
                    test_circuit.x(j)
            
            # Initialize output qubit to |1⟩ for phase kickback
            test_circuit.x(n_qubits)
            test_circuit.h(n_qubits)
            
            # Apply oracle
            test_circuit.compose(oracle, inplace=True)
            
            # Measure output qubit
            test_circuit.h(n_qubits)
            test_circuit.measure(n_qubits, 0)
            
            # Run simulation
            simulator = AerSimulator()
            job = simulator.run(transpile(test_circuit, simulator), shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get output
            output = int(list(counts.keys())[0])
            outputs.append(output)
        
        # Analyze outputs
        ones_count = sum(outputs)
        total_count = len(outputs)
        
        if ones_count == 0 or ones_count == total_count:
            function_type = 'constant'
        elif ones_count == total_count // 2:
            function_type = 'balanced'
        else:
            function_type = 'neither'
        
        return {
            'function_type': function_type,
            'ones_count': ones_count,
            'total_count': total_count,
            'outputs': outputs
        }
    
    @staticmethod
    def create_nested_oracle(n_qubits: int, depth: int = 2) -> QuantumCircuit:
        """
        Create a more complex balanced oracle with nested operations.
        
        Args:
            n_qubits: Number of input qubits
            depth: Depth of nesting (complexity)
            
        Returns:
            QuantumCircuit: Nested oracle
        """
        oracle = QuantumCircuit(n_qubits + 1, name=f'Nested(d={depth})')
        
        for d in range(depth):
            # Apply different patterns at each depth level
            for i in range(n_qubits):
                if (i + d) % 2 == 0:
                    oracle.cx(i, n_qubits)
        
        return oracle


def generate_test_oracles(n_qubits: int) -> dict:
    """
    Generate a suite of test oracles for algorithm validation.
    
    Args:
        n_qubits: Number of input qubits
        
    Returns:
        dict: Dictionary of oracle name -> oracle circuit
    """
    factory = OracleFactory()
    
    oracles = {
        'constant_0': QuantumCircuit(n_qubits + 1, name='Constant(0)'),
        'constant_1': create_constant_one_oracle(n_qubits),
        'balanced_first': create_first_bit_oracle(n_qubits),
        'balanced_xor': create_xor_oracle(n_qubits),
        'balanced_custom': factory.create_custom_balanced_oracle(
            n_qubits, [0, 2] if n_qubits > 2 else [0]
        ),
    }
    
    return oracles


def create_constant_one_oracle(n_qubits: int) -> QuantumCircuit:
    """Create oracle for f(x) = 1 for all x."""
    oracle = QuantumCircuit(n_qubits + 1, name='Constant(1)')
    oracle.x(n_qubits)
    return oracle


def create_first_bit_oracle(n_qubits: int) -> QuantumCircuit:
    """Create oracle where f(x) depends on first bit."""
    oracle = QuantumCircuit(n_qubits + 1, name='Balanced(first)')
    oracle.cx(0, n_qubits)
    return oracle


def create_xor_oracle(n_qubits: int) -> QuantumCircuit:
    """Create oracle where f(x) is XOR of all bits."""
    oracle = QuantumCircuit(n_qubits + 1, name='Balanced(xor)')
    for i in range(n_qubits):
        oracle.cx(i, n_qubits)
    return oracle
