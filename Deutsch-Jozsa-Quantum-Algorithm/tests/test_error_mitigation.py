"""
Tests for error mitigation module.
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.error_mitigation import ErrorMitigator, NoiseAnalyzer
from src.deutsch_jozsa import DeutschJozsa


class TestErrorMitigator:
    """Test error mitigation functionality."""
    
    def test_initialization(self):
        """Test error mitigator initialization."""
        mitigator = ErrorMitigator(n_qubits=3)
        assert mitigator.n_qubits == 3
        assert mitigator.calibration_data is None
    
    def test_noise_model_creation(self):
        """Test noise model creation."""
        mitigator = ErrorMitigator(n_qubits=3)
        noise_model = mitigator.create_noise_model(
            gate_error_rate=0.001,
            measurement_error_rate=0.01
        )
        assert noise_model is not None
    
    def test_readout_error_mitigation(self):
        """Test readout error mitigation."""
        mitigator = ErrorMitigator(n_qubits=3)
        
        # Simulate noisy counts
        raw_counts = {'000': 900, '001': 50, '010': 50, '100': 24}
        
        mitigated_counts = mitigator.apply_readout_error_mitigation(raw_counts)
        
        print(f"\n  ✓ Readout error mitigation: {len(raw_counts)} states → {len(mitigated_counts)} mitigated")
        
        assert mitigated_counts is not None
        assert isinstance(mitigated_counts, dict)
        assert '000' in mitigated_counts
    
    def test_calibration_matrix_creation(self):
        """Test calibration matrix creation."""
        mitigator = ErrorMitigator(n_qubits=2)
        
        cal_matrix = mitigator._create_ideal_calibration_matrix(error_rate=0.01)
        
        assert cal_matrix.shape == (4, 4)
        assert np.allclose(np.sum(cal_matrix, axis=0), 1.0)  # Column sums to 1


class TestNoiseAnalyzer:
    """Test noise analysis functionality."""
    
    def test_initialization(self):
        """Test noise analyzer initialization."""
        analyzer = NoiseAnalyzer(n_qubits=3)
        assert analyzer.n_qubits == 3
        assert analyzer.error_mitigator is not None
    
    def test_noise_impact_analysis(self):
        """Test noise impact analysis with small dataset."""
        dj = DeutschJozsa(n_qubits=2)
        circuit = dj.create_circuit('constant', {'output_value': 0})
        
        analyzer = NoiseAnalyzer(n_qubits=2)
        
        # Test with just 2 error rates and 2 trials for speed
        results = analyzer.analyze_noise_impact(
            circuit,
            error_rates=[0, 0.01],
            shots=100,
            num_trials=2
        )
        
        assert 'results' in results
        assert len(results['results']) == 2
        assert results['n_qubits'] == 2


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
