"""
Comparative Analysis Tools for Deutsch-Jozsa Algorithm

This module provides utilities for comparing quantum and classical approaches,
analyzing performance, and conducting experiments.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Store performance metrics for algorithm runs."""
    algorithm_type: str
    n_qubits: int
    queries_made: int
    execution_time: float
    success_rate: float
    theoretical_queries: int


class ClassicalDeutschJozsa:
    """
    Classical implementation of the Deutsch-Jozsa problem for comparison.
    """
    
    def __init__(self, n_bits: int):
        """
        Initialize classical algorithm.
        
        Args:
            n_bits: Number of input bits
        """
        self.n_bits = n_bits
        self.query_count = 0
    
    def deterministic_solve(self, oracle: Callable[[int], int]) -> str:
        """
        Solve using deterministic classical algorithm.
        
        Worst case: needs 2^(n-1) + 1 queries to guarantee correct answer.
        
        Args:
            oracle: Function that takes integer input and returns 0 or 1
            
        Returns:
            str: 'constant' or 'balanced'
        """
        self.query_count = 0
        
        # Query first value
        first_output = oracle(0)
        self.query_count += 1
        
        # Need to query half + 1 values to be certain
        max_queries = 2**(self.n_bits - 1) + 1
        same_count = 1
        different_count = 0
        
        for i in range(1, min(2**self.n_bits, max_queries)):
            output = oracle(i)
            self.query_count += 1
            
            if output == first_output:
                same_count += 1
            else:
                different_count += 1
                # Found different output - must be balanced
                return 'balanced'
        
        # All sampled outputs were the same
        return 'constant'
    
    def probabilistic_solve(self, oracle: Callable[[int], int], 
                          confidence: float = 0.99) -> str:
        """
        Solve using probabilistic classical algorithm.
        
        Randomly sample inputs until confident about the answer.
        
        Args:
            oracle: Function that takes integer input and returns 0 or 1
            confidence: Desired confidence level (0-1)
            
        Returns:
            str: 'constant' or 'balanced'
        """
        self.query_count = 0
        
        # Calculate number of samples needed for desired confidence
        # For balanced function, probability of all samples being same = (1/2)^k
        # Want (1/2)^k < (1 - confidence)
        k = int(np.ceil(-np.log2(1 - confidence)))
        
        samples = []
        tested_inputs = set()
        
        while len(samples) < k and len(tested_inputs) < 2**self.n_bits:
            # Random input
            input_val = np.random.randint(0, 2**self.n_bits)
            
            if input_val not in tested_inputs:
                output = oracle(input_val)
                samples.append(output)
                tested_inputs.add(input_val)
                self.query_count += 1
        
        # Check if all samples are the same
        if len(set(samples)) == 1:
            return 'constant'
        else:
            return 'balanced'
    
    def get_query_count(self) -> int:
        """Get number of queries made."""
        return self.query_count


class AlgorithmComparator:
    """Compare quantum and classical algorithms."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize comparator.
        
        Args:
            n_qubits: Number of qubits/bits to use
        """
        self.n_qubits = n_qubits
    
    def compare_query_complexity(self) -> Dict[str, int]:
        """
        Compare theoretical query complexity.
        
        Returns:
            Dictionary with complexity for each approach
        """
        return {
            'quantum': 1,
            'classical_deterministic': 2**(self.n_qubits - 1) + 1,
            'classical_probabilistic_99': int(np.ceil(-np.log2(0.01))),
            'classical_probabilistic_99_9': int(np.ceil(-np.log2(0.001)))
        }
    
    def calculate_speedup(self) -> Dict[str, float]:
        """
        Calculate quantum speedup over classical approaches.
        
        Returns:
            Dictionary with speedup factors
        """
        complexities = self.compare_query_complexity()
        quantum_queries = complexities['quantum']
        
        return {
            'vs_deterministic': complexities['classical_deterministic'] / quantum_queries,
            'vs_probabilistic_99': complexities['classical_probabilistic_99'] / quantum_queries,
            'vs_probabilistic_99_9': complexities['classical_probabilistic_99_9'] / quantum_queries
        }
    
    def generate_complexity_table(self, max_qubits: int = 10) -> str:
        """
        Generate a comparison table of query complexities.
        
        Args:
            max_qubits: Maximum number of qubits to include
            
        Returns:
            Formatted string table
        """
        header = f"{'n':<5} {'Quantum':<12} {'Classical Det.':<18} {'Speedup':<12}"
        separator = "-" * 50
        
        lines = [separator, header, separator]
        
        for n in range(1, max_qubits + 1):
            quantum = 1
            classical = 2**(n-1) + 1
            speedup = classical / quantum
            
            line = f"{n:<5} {quantum:<12} {classical:<18} {speedup:.1f}x"
            lines.append(line)
        
        lines.append(separator)
        return "\n".join(lines)


class ExperimentRunner:
    """Run experiments and collect performance data."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.results = []
    
    def run_quantum_experiment(self, dj_algorithm, function_type: str, 
                              oracle_params: Dict = None, 
                              num_trials: int = 10) -> PerformanceMetrics:
        """
        Run quantum algorithm experiment.
        
        Args:
            dj_algorithm: DeutschJozsa instance
            function_type: Type of function to test
            oracle_params: Oracle parameters
            num_trials: Number of trials to run
            
        Returns:
            PerformanceMetrics object
        """
        n_qubits = dj_algorithm.n_qubits
        successes = 0
        total_time = 0
        
        for _ in range(num_trials):
            start_time = time.time()
            result = dj_algorithm.run(function_type, oracle_params)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            if result == function_type:
                successes += 1
        
        avg_time = total_time / num_trials
        success_rate = successes / num_trials
        
        metrics = PerformanceMetrics(
            algorithm_type='quantum',
            n_qubits=n_qubits,
            queries_made=1,
            execution_time=avg_time,
            success_rate=success_rate,
            theoretical_queries=1
        )
        
        self.results.append(metrics)
        return metrics
    
    def run_classical_experiment(self, classical_alg: ClassicalDeutschJozsa,
                                oracle: Callable, 
                                true_type: str,
                                method: str = 'deterministic',
                                num_trials: int = 10) -> PerformanceMetrics:
        """
        Run classical algorithm experiment.
        
        Args:
            classical_alg: ClassicalDeutschJozsa instance
            oracle: Oracle function
            true_type: True function type
            method: 'deterministic' or 'probabilistic'
            num_trials: Number of trials
            
        Returns:
            PerformanceMetrics object
        """
        n_bits = classical_alg.n_bits
        successes = 0
        total_queries = 0
        total_time = 0
        
        for _ in range(num_trials):
            start_time = time.time()
            
            if method == 'deterministic':
                result = classical_alg.deterministic_solve(oracle)
            else:
                result = classical_alg.probabilistic_solve(oracle)
            
            elapsed = time.time() - start_time
            
            total_time += elapsed
            total_queries += classical_alg.get_query_count()
            
            if result == true_type:
                successes += 1
        
        avg_time = total_time / num_trials
        avg_queries = total_queries / num_trials
        success_rate = successes / num_trials
        
        theoretical = (2**(n_bits-1) + 1) if method == 'deterministic' else int(np.ceil(-np.log2(0.01)))
        
        metrics = PerformanceMetrics(
            algorithm_type=f'classical_{method}',
            n_qubits=n_bits,
            queries_made=int(avg_queries),
            execution_time=avg_time,
            success_rate=success_rate,
            theoretical_queries=theoretical
        )
        
        self.results.append(metrics)
        return metrics
    
    def get_results_summary(self) -> str:
        """
        Get formatted summary of all results.
        
        Returns:
            Formatted string with results
        """
        if not self.results:
            return "No results available."
        
        lines = [
            "=" * 80,
            "EXPERIMENT RESULTS SUMMARY",
            "=" * 80,
            ""
        ]
        
        for metrics in self.results:
            lines.append(f"Algorithm: {metrics.algorithm_type}")
            lines.append(f"  n_qubits: {metrics.n_qubits}")
            lines.append(f"  Queries made: {metrics.queries_made}")
            lines.append(f"  Theoretical queries: {metrics.theoretical_queries}")
            lines.append(f"  Execution time: {metrics.execution_time*1000:.2f} ms")
            lines.append(f"  Success rate: {metrics.success_rate*100:.1f}%")
            lines.append("")
        
        return "\n".join(lines)


def create_test_oracles(n_bits: int) -> Dict[str, Callable]:
    """
    Create test oracle functions for classical algorithm testing.
    
    Args:
        n_bits: Number of input bits
        
    Returns:
        Dictionary of oracle name -> oracle function
    """
    oracles = {}
    
    # Constant oracle returning 0
    oracles['constant_0'] = lambda x: 0
    
    # Constant oracle returning 1
    oracles['constant_1'] = lambda x: 1
    
    # Balanced oracle: return first bit
    oracles['balanced_first'] = lambda x: (x >> (n_bits - 1)) & 1
    
    # Balanced oracle: return XOR of all bits
    def xor_oracle(x):
        result = 0
        for i in range(n_bits):
            result ^= (x >> i) & 1
        return result
    oracles['balanced_xor'] = xor_oracle
    
    # Balanced oracle: return parity of even positions
    def even_parity(x):
        result = 0
        for i in range(0, n_bits, 2):
            result ^= (x >> i) & 1
        return result
    oracles['balanced_even'] = even_parity
    
    return oracles
