"""
Tests for Bernstein-Vazirani algorithm.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bernstein_vazirani import BernsteinVazirani


class TestBernsteinVazirani:
    """Test Bernstein-Vazirani algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        bv = BernsteinVazirani(n_qubits=3)
        assert bv.n_qubits == 3
        assert bv.simulator is not None
    
    def test_invalid_qubits(self):
        """Test invalid qubit count."""
        with pytest.raises(ValueError):
            BernsteinVazirani(n_qubits=0)
    
    def test_oracle_creation(self):
        """Test oracle creation."""
        bv = BernsteinVazirani(n_qubits=3)
        oracle = bv.create_oracle('101')
        assert oracle is not None
        assert oracle.num_qubits == 4
    
    def test_invalid_hidden_string_length(self):
        """Test invalid hidden string length."""
        bv = BernsteinVazirani(n_qubits=3)
        with pytest.raises(ValueError):
            bv.create_oracle('10')  # Too short
    
    def test_invalid_hidden_string_chars(self):
        """Test invalid characters in hidden string."""
        bv = BernsteinVazirani(n_qubits=3)
        with pytest.raises(ValueError):
            bv.create_oracle('1a1')  # Invalid character
    
    def test_circuit_creation(self):
        """Test circuit creation."""
        bv = BernsteinVazirani(n_qubits=3)
        circuit = bv.create_circuit('101')
        assert circuit is not None
        assert circuit.num_qubits == 4
        assert circuit.num_clbits == 3
    
    def test_find_hidden_string(self):
        """Test finding hidden strings."""
        test_strings = ['101', '111', '000', '010']
        
        for hidden in test_strings:
            bv = BernsteinVazirani(n_qubits=len(hidden))
            discovered = bv.run(hidden, shots=1024)
            assert discovered == hidden
    
    def test_different_lengths(self):
        """Test with different string lengths."""
        for n in range(2, 6):
            hidden = '1' * n
            bv = BernsteinVazirani(n_qubits=n)
            discovered = bv.run(hidden)
            assert discovered == hidden
    
    def test_get_counts(self):
        """Test getting counts."""
        bv = BernsteinVazirani(n_qubits=3)
        bv.run('101')
        counts = bv.get_counts()
        assert counts is not None
        assert '101' in counts
    
    def test_get_circuit(self):
        """Test getting circuit."""
        bv = BernsteinVazirani(n_qubits=3)
        bv.create_circuit('101')
        circuit = bv.get_circuit()
        assert circuit is not None
    
    def test_verify_result(self):
        """Test result verification."""
        bv = BernsteinVazirani(n_qubits=3)
        assert bv.verify_result('101', '101') == True
        assert bv.verify_result('101', '100') == False


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
