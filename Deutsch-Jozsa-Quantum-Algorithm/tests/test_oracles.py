"""
Unit tests for oracle implementations.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.oracles import (
    OracleFactory,
    generate_test_oracles,
    create_constant_one_oracle,
    create_first_bit_oracle,
    create_xor_oracle
)


class TestOracleCreation:
    """Test oracle creation functions."""
    
    def test_constant_one_oracle(self):
        """Test constant-1 oracle."""
        oracle = create_constant_one_oracle(n_qubits=3)
        assert oracle is not None
        assert oracle.num_qubits == 4
    
    def test_first_bit_oracle(self):
        """Test first-bit oracle."""
        oracle = create_first_bit_oracle(n_qubits=3)
        assert oracle is not None
        assert oracle.num_qubits == 4
    
    def test_xor_oracle(self):
        """Test XOR oracle."""
        oracle = create_xor_oracle(n_qubits=3)
        assert oracle is not None
        assert oracle.num_qubits == 4
    
    def test_generate_test_oracles(self):
        """Test test oracle generation."""
        oracles = generate_test_oracles(n_qubits=3)
        
        assert isinstance(oracles, dict)
        assert 'constant_0' in oracles
        assert 'constant_1' in oracles
        assert 'balanced_first' in oracles
        assert 'balanced_xor' in oracles
        
        for name, oracle in oracles.items():
            assert oracle is not None
            assert oracle.num_qubits == 4


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
