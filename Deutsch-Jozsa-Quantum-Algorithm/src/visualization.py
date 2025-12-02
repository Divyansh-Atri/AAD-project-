"""
Visualization Tools for Deutsch-Jozsa Algorithm

This module provides visualization utilities for analyzing and presenting
the Deutsch-Jozsa algorithm results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from qiskit.visualization import plot_histogram, circuit_drawer, plot_bloch_multivector


class DJVisualizer:
    """Visualization tools for Deutsch-Jozsa algorithm."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    def plot_measurement_results(self, counts: Dict[str, int], 
                                title: str = "Measurement Results") -> plt.Figure:
        """
        Plot measurement results as a bar chart.
        
        Args:
            counts: Dictionary of measurement outcomes
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig = plot_histogram(counts, title=title, color='#3498db')
        return fig
    
    def plot_circuit(self, circuit, output_format: str = 'mpl') -> plt.Figure:
        """
        Plot quantum circuit.
        
        Args:
            circuit: Qiskit QuantumCircuit
            output_format: Output format ('mpl', 'text', or 'latex')
            
        Returns:
            Circuit drawing
        """
        return circuit.draw(output=output_format, style='iqp')
    
    def compare_query_complexity(self, n_qubits_range: List[int]) -> plt.Figure:
        """
        Compare classical vs quantum query complexity.
        
        Args:
            n_qubits_range: Range of qubit numbers to compare
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classical_queries = [2**(n-1) + 1 for n in n_qubits_range]
        quantum_queries = [1] * len(n_qubits_range)
        
        ax.plot(n_qubits_range, classical_queries, 'o-', 
                label='Classical (Deterministic)', linewidth=2, markersize=8)
        ax.plot(n_qubits_range, quantum_queries, 's-', 
                label='Quantum', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Qubits (n)', fontsize=12)
        ax.set_ylabel('Number of Queries', fontsize=12)
        ax.set_title('Query Complexity: Classical vs Quantum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add annotations
        for i, n in enumerate(n_qubits_range[::2]):  # Annotate every other point
            idx = i * 2
            if idx < len(n_qubits_range):
                speedup = classical_queries[idx]
                ax.annotate(f'{speedup}x', 
                           xy=(n_qubits_range[idx], classical_queries[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def visualize_statevector(self, statevector: np.ndarray, 
                            title: str = "Quantum State") -> plt.Figure:
        """
        Visualize quantum statevector.
        
        Args:
            statevector: State vector to visualize
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot amplitudes
        amplitudes = np.abs(statevector)
        phases = np.angle(statevector)
        
        n_states = len(statevector)
        indices = range(n_states)
        labels = [format(i, f'0{int(np.log2(n_states))}b') for i in indices]
        
        ax1.bar(indices, amplitudes, color='#3498db', alpha=0.7)
        ax1.set_xlabel('Basis State', fontsize=11)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('State Amplitudes', fontsize=12, fontweight='bold')
        ax1.set_xticks(indices)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot phases
        colors = plt.cm.hsv((phases + np.pi) / (2 * np.pi))
        ax2.bar(indices, amplitudes, color=colors, alpha=0.7)
        ax2.set_xlabel('Basis State', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title('State Phases (colored)', fontsize=12, fontweight='bold')
        ax2.set_xticks(indices)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_interference_pattern(self, n_qubits: int) -> plt.Figure:
        """
        Visualize interference patterns in Deutsch-Jozsa algorithm.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        stages = [
            "Initial State |0⟩⊗n|1⟩",
            "After First Hadamards",
            "After Oracle (Balanced)",
            "After Final Hadamards"
        ]
        
        # Simplified visualization of probability amplitudes at each stage
        for idx, (ax, stage) in enumerate(zip(axes.flat, stages)):
            n_states = 2**n_qubits
            x = np.arange(n_states)
            
            if idx == 0:
                # Initial state: only |0...01⟩ has amplitude
                amplitudes = np.zeros(n_states)
                amplitudes[0] = 1
            elif idx == 1:
                # After Hadamards: uniform superposition
                amplitudes = np.ones(n_states) / np.sqrt(n_states)
            elif idx == 2:
                # After oracle: phase flip on half the states
                amplitudes = np.ones(n_states) / np.sqrt(n_states)
                amplitudes[::2] *= -1  # Flip phase on even states
            else:
                # After final Hadamards: interference causes concentration
                amplitudes = np.zeros(n_states)
                amplitudes[1:] = 0.5 / np.sqrt(n_states)  # Non-zero states
            
            colors = ['red' if a < 0 else 'blue' for a in amplitudes]
            ax.bar(x, np.abs(amplitudes), color=colors, alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_title(stage, fontsize=11, fontweight='bold')
            ax.set_xlabel('Basis State')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Quantum Interference in Deutsch-Jozsa Algorithm', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_algorithm_summary(self, results: Dict, n_qubits: int) -> plt.Figure:
        """
        Create a comprehensive summary visualization.
        
        Args:
            results: Dictionary containing algorithm results
            n_qubits: Number of qubits used
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Deutsch-Jozsa Algorithm Analysis (n={n_qubits} qubits)', 
                    fontsize=16, fontweight='bold')
        
        # Circuit visualization
        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(0.5, 0.5, 'Circuit Diagram\n(Use .plot_circuit() for actual circuit)', 
                ha='center', va='center', fontsize=12)
        ax1.axis('off')
        
        # Measurement results
        ax2 = fig.add_subplot(gs[1, 0])
        if 'counts' in results:
            counts = results['counts']
            ax2.bar(counts.keys(), counts.values(), color='#3498db', alpha=0.7)
            ax2.set_title('Measurement Results', fontweight='bold')
            ax2.set_xlabel('State')
            ax2.set_ylabel('Counts')
            ax2.tick_params(axis='x', rotation=45)
        
        # Query complexity comparison
        ax3 = fig.add_subplot(gs[1, 1:])
        n_range = range(1, n_qubits + 3)
        classical = [2**(n-1) + 1 for n in n_range]
        quantum = [1] * len(n_range)
        ax3.plot(n_range, classical, 'o-', label='Classical', linewidth=2)
        ax3.plot(n_range, quantum, 's-', label='Quantum', linewidth=2)
        ax3.set_yscale('log')
        ax3.set_title('Query Complexity Comparison', fontweight='bold')
        ax3.set_xlabel('Number of Qubits')
        ax3.set_ylabel('Queries (log scale)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Algorithm info
        ax4 = fig.add_subplot(gs[2, :])
        info_text = f"""
        Algorithm: Deutsch-Jozsa
        Problem: Determine if function is constant or balanced
        
        Classical Complexity: O(2^(n-1)) queries (deterministic)
        Quantum Complexity: O(1) query
        
        Speedup: {2**(n_qubits-1)}x for {n_qubits} qubits
        
        Result: {results.get('result', 'N/A')}
        Success Rate: {results.get('success_rate', 'N/A')}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        
        return fig


def plot_multiple_runs(results_list: List[Dict[str, int]], 
                      labels: List[str]) -> plt.Figure:
    """
    Plot results from multiple algorithm runs for comparison.
    
    Args:
        results_list: List of measurement count dictionaries
        labels: Labels for each run
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, len(results_list), 
                            figsize=(6*len(results_list), 4))
    
    if len(results_list) == 1:
        axes = [axes]
    
    for ax, counts, label in zip(axes, results_list, labels):
        ax.bar(counts.keys(), counts.values(), color='#3498db', alpha=0.7)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Measurement Outcome')
        ax.set_ylabel('Counts')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
