"""
Unit tests for Deutsch-Jozsa algorithm implementation.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deutsch_jozsa import DeutschJozsa
from src.oracles import OracleFactory


class TestDeutschJozsa:
    """Test cases for DeutschJozsa class."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        dj = DeutschJozsa(n_qubits=3)
        assert dj.n_qubits == 3
        assert dj.simulator is not None
    
    def test_invalid_qubits(self):
        """Test that invalid qubit count raises error."""
        with pytest.raises(ValueError):
            DeutschJozsa(n_qubits=0)
        
        with pytest.raises(ValueError):
            DeutschJozsa(n_qubits=-1)
    
    def test_constant_oracle_creation(self):
        """Test constant oracle creation."""
        dj = DeutschJozsa(n_qubits=3)
        
        # Test oracle for f(x) = 0
        oracle_0 = dj.create_constant_oracle(output_value=0)
        assert oracle_0 is not None
        assert oracle_0.num_qubits == 4
        
        # Test oracle for f(x) = 1
        oracle_1 = dj.create_constant_oracle(output_value=1)
        assert oracle_1 is not None
        assert oracle_1.num_qubits == 4
    
    def test_balanced_oracle_creation(self):
        """Test balanced oracle creation."""
        dj = DeutschJozsa(n_qubits=3)
        
        # Test different oracle types
        oracle_types = ['first_bit', 'xor', 'random']
        
        for oracle_type in oracle_types:
            oracle = dj.create_balanced_oracle(oracle_type=oracle_type)
            assert oracle is not None
            assert oracle.num_qubits == 4
    
    def test_circuit_creation(self):
        """Test circuit creation."""
        dj = DeutschJozsa(n_qubits=3)
        
        # Test constant function circuit
        circuit = dj.create_circuit(function_type='constant')
        assert circuit is not None
        assert circuit.num_qubits == 4
        assert circuit.num_clbits == 3
        
        # Test balanced function circuit
        circuit = dj.create_circuit(function_type='balanced')
        assert circuit is not None
    
    def test_constant_function_detection(self):
        """Test that constant functions are correctly identified."""
        dj = DeutschJozsa(n_qubits=3)
        
        # Test f(x) = 0
        result = dj.run(function_type='constant', 
                       oracle_params={'output_value': 0})
        print(f"\n  ✓ Constant function f(x)=0 detected as: {result}")
        assert result == 'constant'
        
        # Test f(x) = 1
        result = dj.run(function_type='constant', 
                       oracle_params={'output_value': 1})
        print(f"  ✓ Constant function f(x)=1 detected as: {result}")
        assert result == 'constant'
    
    def test_balanced_function_detection(self):
        """Test that balanced functions are correctly identified."""
        dj = DeutschJozsa(n_qubits=3)
        
        # Test different balanced oracles
        oracle_types = ['first_bit', 'xor']
        
        for oracle_type in oracle_types:
            result = dj.run(function_type='balanced',
                          oracle_params={'oracle_type': oracle_type})
            print(f"\n  ✓ Balanced function ({oracle_type}) detected as: {result}")
            assert result == 'balanced'
    
    def test_multiple_runs_consistency(self):
        """Test that multiple runs give consistent results."""
        print(f"\n  ⟳ Testing consistency across multiple runs...")
        dj = DeutschJozsa(n_qubits=3)
        
        # Run constant function multiple times
        results = []
        for _ in range(5):
            result = dj.run(function_type='constant', 
                          oracle_params={'output_value': 0})
            results.append(result)
        
        assert all(r == 'constant' for r in results)
        
        # Run balanced function multiple times
        results = []
        for _ in range(5):
            result = dj.run(function_type='balanced',
                          oracle_params={'oracle_type': 'xor'})
            results.append(result)
        
        assert all(r == 'balanced' for r in results)
    
    def test_get_counts(self):
        """Test getting measurement counts."""
        dj = DeutschJozsa(n_qubits=3)
        dj.run(function_type='constant')
        
        counts = dj.get_counts()
        assert counts is not None
        assert isinstance(counts, dict)
        assert '000' in counts
    
    def test_get_circuit(self):
        """Test getting circuit."""
        dj = DeutschJozsa(n_qubits=3)
        dj.create_circuit(function_type='constant')
        
        circuit = dj.get_circuit()
        assert circuit is not None
    
    def test_get_circuit_before_creation(self):
        """Test that getting circuit before creation raises error."""
        dj = DeutschJozsa(n_qubits=3)
        
        with pytest.raises(RuntimeError):
            dj.get_circuit()
    
    def test_different_qubit_counts(self):
        """Test algorithm with different numbers of qubits."""
        for n in range(1, 6):
            dj = DeutschJozsa(n_qubits=n)
            
            result = dj.run(function_type='constant')
            assert result == 'constant'
            
            result = dj.run(function_type='balanced', 
                          oracle_params={'oracle_type': 'xor'})
            assert result == 'balanced'


class TestOracleFactory:
    """Test cases for OracleFactory."""
    
    def test_custom_balanced_oracle(self):
        """Test custom balanced oracle creation."""
        factory = OracleFactory()
        
        oracle = factory.create_custom_balanced_oracle(
            n_qubits=3,
            control_pattern=[0, 2]
        )
        
        assert oracle is not None
        assert oracle.num_qubits == 4
    
    def test_multi_controlled_oracle(self):
        """Test multi-controlled oracle creation."""
        factory = OracleFactory()
        
        # Single control
        oracle = factory.create_multi_controlled_oracle(
            n_qubits=3,
            control_qubits=[0]
        )
        assert oracle is not None
        
        # Two controls
        oracle = factory.create_multi_controlled_oracle(
            n_qubits=3,
            control_qubits=[0, 1]
        )
        assert oracle is not None
    
    def test_nested_oracle(self):
        """Test nested oracle creation."""
        factory = OracleFactory()
        
        oracle = factory.create_nested_oracle(n_qubits=3, depth=2)
        assert oracle is not None
        assert oracle.num_qubits == 4


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
