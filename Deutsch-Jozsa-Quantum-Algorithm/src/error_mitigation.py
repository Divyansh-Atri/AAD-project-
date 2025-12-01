"""
Error Mitigation and Noise Analysis Module

This module provides error mitigation techniques and noise analysis tools
for the Deutsch-Jozsa algorithm on real quantum hardware.
"""

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ErrorMitigationResult:
    """Store error mitigation results."""
    raw_counts: Dict[str, int]
    mitigated_counts: Dict[str, int]
    raw_result: str
    mitigated_result: str
    improvement: float
    noise_level: float


class ErrorMitigator:
    """
    Implements error mitigation techniques for quantum circuits.
    
    Techniques:
    1. Readout error mitigation
    2. Zero-noise extrapolation
    3. Measurement error mitigation
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize error mitigator.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
        self.calibration_data = None
    
    def create_noise_model(self, 
                          gate_error_rate: float = 0.001,
                          measurement_error_rate: float = 0.01,
                          t1: float = 50e-6,
                          t2: float = 70e-6) -> NoiseModel:
        """
        Create a realistic noise model for simulation.
        
        Args:
            gate_error_rate: Single-qubit gate error rate
            measurement_error_rate: Measurement error rate
            t1: T1 relaxation time (seconds)
            t2: T2 dephasing time (seconds)
            
        Returns:
            NoiseModel: Configured noise model
        """
        noise_model = NoiseModel()
        
        # Single-qubit gate errors
        single_qubit_error = depolarizing_error(gate_error_rate, 1)
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x'])
        
        # Two-qubit gate errors (higher error rate)
        two_qubit_error = depolarizing_error(gate_error_rate * 10, 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
        
        # Measurement errors
        measurement_error = depolarizing_error(measurement_error_rate, 1)
        noise_model.add_all_qubit_quantum_error(measurement_error, ['measure'])
        
        return noise_model
    
    def apply_readout_error_mitigation(self,
                                      counts: Dict[str, int],
                                      calibration_matrix: Optional[np.ndarray] = None) -> Dict[str, int]:
        """
        Apply readout error mitigation to measurement counts.
        
        This uses a calibration matrix to correct for measurement errors.
        
        Args:
            counts: Raw measurement counts
            calibration_matrix: Calibration matrix (if None, uses ideal approximation)
            
        Returns:
            Mitigated counts
        """
        if calibration_matrix is None:
            # Simple approximation: assume symmetric readout errors
            # In practice, get this from calibration experiments
            measurement_error = 0.01
            calibration_matrix = self._create_ideal_calibration_matrix(measurement_error)
        
        # Convert counts to probability vector
        total_shots = sum(counts.values())
        n_states = 2**self.n_qubits
        
        prob_vector = np.zeros(n_states)
        for state_str, count in counts.items():
            state_int = int(state_str, 2)
            prob_vector[state_int] = count / total_shots
        
        # Apply inverse calibration matrix
        try:
            inv_cal_matrix = np.linalg.inv(calibration_matrix)
            mitigated_prob = inv_cal_matrix @ prob_vector
            
            # Ensure probabilities are valid (non-negative)
            mitigated_prob = np.maximum(mitigated_prob, 0)
            mitigated_prob = mitigated_prob / np.sum(mitigated_prob)
            
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            inv_cal_matrix = np.linalg.pinv(calibration_matrix)
            mitigated_prob = inv_cal_matrix @ prob_vector
            mitigated_prob = np.maximum(mitigated_prob, 0)
            mitigated_prob = mitigated_prob / np.sum(mitigated_prob)
        
        # Convert back to counts
        mitigated_counts = {}
        for i, prob in enumerate(mitigated_prob):
            if prob > 0.001:  # Threshold to avoid noise
                state_str = format(i, f'0{self.n_qubits}b')
                mitigated_counts[state_str] = int(prob * total_shots)
        
        return mitigated_counts
    
    def _create_ideal_calibration_matrix(self, error_rate: float) -> np.ndarray:
        """
        Create an ideal calibration matrix with symmetric errors.
        
        Args:
            error_rate: Measurement error rate
            
        Returns:
            Calibration matrix
        """
        n_states = 2**self.n_qubits
        cal_matrix = np.eye(n_states) * (1 - error_rate)
        
        # Add symmetric off-diagonal errors
        cal_matrix += (error_rate / (n_states - 1)) * (1 - np.eye(n_states))
        
        return cal_matrix
    
    def zero_noise_extrapolation(self,
                                circuit,
                                noise_levels: List[float],
                                shots: int = 1024) -> Tuple[str, Dict]:
        """
        Apply zero-noise extrapolation by running at different noise levels.
        
        Args:
            circuit: Quantum circuit to run
            noise_levels: List of noise scaling factors
            shots: Number of shots per noise level
            
        Returns:
            Tuple of (extrapolated result, analysis data)
        """
        results = []
        
        for noise_scale in noise_levels:
            # Create noise model with scaled errors
            noise_model = self.create_noise_model(
                gate_error_rate=0.001 * noise_scale,
                measurement_error_rate=0.01 * noise_scale
            )
            
            # Run with noise
            simulator = AerSimulator(noise_model=noise_model)
            transpiled = transpile(circuit, simulator)
            job = simulator.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Analyze result
            zero_state = '0' * self.n_qubits
            zero_prob = counts.get(zero_state, 0) / shots
            
            results.append({
                'noise_scale': noise_scale,
                'zero_probability': zero_prob,
                'counts': counts
            })
        
        # Extrapolate to zero noise
        noise_scales = [r['noise_scale'] for r in results]
        zero_probs = [r['zero_probability'] for r in results]
        
        # Linear extrapolation
        coeffs = np.polyfit(noise_scales, zero_probs, deg=1)
        extrapolated_prob = coeffs[1]  # Intercept at noise=0
        
        # Determine result based on extrapolated probability
        if extrapolated_prob > 0.9:
            extrapolated_result = 'constant'
        else:
            extrapolated_result = 'balanced'
        
        analysis = {
            'measurements': results,
            'extrapolated_probability': extrapolated_prob,
            'fit_coefficients': coeffs
        }
        
        return extrapolated_result, analysis


class NoiseAnalyzer:
    """Analyze noise effects on Deutsch-Jozsa algorithm."""
    
    def __init__(self, n_qubits: int):
        """Initialize noise analyzer."""
        self.n_qubits = n_qubits
        self.error_mitigator = ErrorMitigator(n_qubits)
    
    def analyze_noise_impact(self,
                            circuit,
                            error_rates: List[float],
                            shots: int = 1024,
                            num_trials: int = 10) -> Dict:
        """
        Analyze impact of different error rates on algorithm success.
        
        Args:
            circuit: Quantum circuit to test
            error_rates: List of error rates to test
            shots: Shots per trial
            num_trials: Number of trials per error rate
            
        Returns:
            Analysis results
        """
        results = []
        
        for error_rate in error_rates:
            print(f"  Testing error rate: {error_rate*100:.2f}%")
            
            successes = 0
            trial_data = []
            
            for trial in range(num_trials):
                # Create noise model
                noise_model = self.error_mitigator.create_noise_model(
                    gate_error_rate=error_rate,
                    measurement_error_rate=error_rate * 10
                )
                
                # Run with noise
                simulator = AerSimulator(noise_model=noise_model)
                transpiled = transpile(circuit, simulator)
                job = simulator.run(transpiled, shots=shots)
                result = job.result()
                counts = result.get_counts()
                
                # Check if correct
                zero_state = '0' * self.n_qubits
                is_constant = (zero_state in counts and 
                             counts[zero_state] > shots * 0.9)
                
                trial_data.append({
                    'counts': counts,
                    'is_constant': is_constant
                })
                
                if is_constant:
                    successes += 1
            
            success_rate = successes / num_trials
            
            results.append({
                'error_rate': error_rate,
                'success_rate': success_rate,
                'trials': trial_data
            })
            
            print(f"    Success rate: {success_rate*100:.1f}%")
        
        return {
            'n_qubits': self.n_qubits,
            'results': results,
            'shots_per_trial': shots,
            'num_trials': num_trials
        }
    
    def plot_noise_analysis(self, analysis_results: Dict) -> plt.Figure:
        """
        Plot noise analysis results.
        
        Args:
            analysis_results: Results from analyze_noise_impact
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        error_rates = [r['error_rate'] for r in analysis_results['results']]
        success_rates = [r['success_rate'] for r in analysis_results['results']]
        
        # Plot 1: Success rate vs error rate
        axes[0].plot(error_rates, success_rates, 'o-', linewidth=2, markersize=8)
        axes[0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
        axes[0].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
        axes[0].set_xlabel('Gate Error Rate', fontsize=12)
        axes[0].set_ylabel('Algorithm Success Rate', fontsize=12)
        axes[0].set_title(f'Noise Impact on Deutsch-Jozsa (n={analysis_results["n_qubits"]})', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        
        # Plot 2: Error rate vs required fidelity
        fidelities = success_rates
        axes[1].semilogy([1-e for e in error_rates], fidelities, 'o-', 
                        linewidth=2, markersize=8, color='purple')
        axes[1].set_xlabel('Gate Fidelity (1 - error_rate)', fontsize=12)
        axes[1].set_ylabel('Algorithm Success Rate', fontsize=12)
        axes[1].set_title('Required Gate Fidelity', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].axhline(y=0.95, color='red', linestyle='--', 
                       alpha=0.5, label='95% target')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def compare_with_without_mitigation(self,
                                       circuit,
                                       noise_model: NoiseModel,
                                       shots: int = 1024) -> ErrorMitigationResult:
        """
        Compare results with and without error mitigation.
        
        Args:
            circuit: Quantum circuit
            noise_model: Noise model to use
            shots: Number of shots
            
        Returns:
            Comparison results
        """
        # Run with noise (no mitigation)
        simulator = AerSimulator(noise_model=noise_model)
        transpiled = transpile(circuit, simulator)
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        raw_counts = result.get_counts()
        
        # Analyze raw result
        zero_state = '0' * self.n_qubits
        raw_result = 'constant' if (zero_state in raw_counts and 
                                    raw_counts[zero_state] > shots * 0.9) else 'balanced'
        
        # Apply error mitigation
        mitigated_counts = self.error_mitigator.apply_readout_error_mitigation(raw_counts)
        
        # Analyze mitigated result
        mitigated_result = 'constant' if (zero_state in mitigated_counts and 
                                         mitigated_counts[zero_state] > shots * 0.9) else 'balanced'
        
        # Calculate improvement
        raw_zero_prob = raw_counts.get(zero_state, 0) / shots
        mit_zero_prob = mitigated_counts.get(zero_state, 0) / shots
        improvement = (mit_zero_prob - raw_zero_prob) / raw_zero_prob if raw_zero_prob > 0 else 0
        
        return ErrorMitigationResult(
            raw_counts=raw_counts,
            mitigated_counts=mitigated_counts,
            raw_result=raw_result,
            mitigated_result=mitigated_result,
            improvement=improvement,
            noise_level=0.01  # Default
        )


def demonstrate_error_mitigation():
    """Demonstrate error mitigation techniques."""
    from src.deutsch_jozsa import DeutschJozsa
    
    print("=" * 70)
    print("ERROR MITIGATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    n_qubits = 3
    dj = DeutschJozsa(n_qubits=n_qubits)
    circuit = dj.create_circuit('constant', {'output_value': 0})
    
    # Remove measurements for noise analysis
    circuit_no_measure = circuit.copy()
    circuit_no_measure.remove_final_measurements()
    
    analyzer = NoiseAnalyzer(n_qubits)
    
    # Test different error rates
    print("Testing impact of different error rates...")
    print()
    
    error_rates = [0, 0.001, 0.005, 0.01, 0.02]
    analysis = analyzer.analyze_noise_impact(circuit, error_rates, shots=1024, num_trials=5)
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for result in analysis['results']:
        print(f"Error rate {result['error_rate']*100:.2f}%: "
              f"Success rate {result['success_rate']*100:.1f}%")
    
    print()
    print("Error mitigation analysis complete!")


if __name__ == "__main__":
    demonstrate_error_mitigation()
