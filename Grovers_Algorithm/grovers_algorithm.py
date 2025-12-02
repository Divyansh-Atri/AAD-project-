"""
Grover's Algorithm Implementation
----------------------------------
This script implements Grover's quantum search algorithm to find a marked item
in an unsorted database. The algorithm provides a quadratic speedup over classical search.

Example: Search for |11⟩ in a 2-qubit system (4 possible states)
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv


def setup_ibm_account():
    """
    Setup IBM Quantum account using token from .env file.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Try to load existing account
        service = QiskitRuntimeService()
        print("[+] IBM Quantum account already configured")
        return True
    except Exception:
        # No account configured, load from .env
        print("\nSetting up IBM Quantum account from .env file...")
        
        # Load environment variables
        load_dotenv()
        token = os.getenv('IBM_QUANTUM_TOKEN')
        
        if not token:
            print("[-] IBM_QUANTUM_TOKEN not found in .env file")
            print("\nTo use IBM Quantum:")
            print("1. Create a .env file in the project directory")
            print("2. Add: IBM_QUANTUM_TOKEN=your_token_here")
            print("3. Get your token from https://quantum.ibm.com/")
            return False
        
        try:
            # Save account
            QiskitRuntimeService.save_account(
                channel='ibm_quantum_platform',
                token=token,
                overwrite=True
            )
            
            print("[+] Account saved successfully!")
            
            # Test connection
            service = QiskitRuntimeService()
            backends = service.backends()
            print(f"[+] Found {len(backends)} backend(s) available")
            
            return True
            
        except Exception as e:
            print(f"[-] Error setting up account: {e}")
            return False


def create_oracle(n_qubits, marked_states):
    """
    Create the oracle that marks the target state(s) by flipping their phase.
    
    Args:
        n_qubits: Number of qubits
        marked_states: The state(s) to mark (string or list of strings, e.g., '11' or ['11', '01'])
    
    Returns:
        QuantumCircuit: Oracle circuit
    """
    oracle = QuantumCircuit(n_qubits, name='Oracle')
    
    # Convert single state to list for uniform processing
    if isinstance(marked_states, str):
        marked_states = [marked_states]
    
    # Mark each target state
    for marked_state in marked_states:
        # Flip qubits that should be 0 in the marked state
        for i, bit in enumerate(reversed(marked_state)):
            if bit == '0':
                oracle.x(i)
        
        # Multi-controlled Z gate (flips phase of marked state)
        if n_qubits > 1:
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)
        else:
            oracle.z(0)
        
        # Flip back the qubits
        for i, bit in enumerate(reversed(marked_state)):
            if bit == '0':
                oracle.x(i)
    
    return oracle


def create_diffusion_operator(n_qubits):
    """
    Create the Grover diffusion operator (inversion about average).
    
    Args:
        n_qubits: Number of qubits
    
    Returns:
        QuantumCircuit: Diffusion operator circuit
    """
    diffusion = QuantumCircuit(n_qubits, name='Diffusion')
    
    # Apply H gates
    diffusion.h(range(n_qubits))
    
    # Apply X gates
    diffusion.x(range(n_qubits))
    
    # Multi-controlled Z gate
    if n_qubits > 1:
        diffusion.h(n_qubits - 1)
        diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        diffusion.h(n_qubits - 1)
    else:
        diffusion.z(0)
    
    # Apply X gates
    diffusion.x(range(n_qubits))
    
    # Apply H gates
    diffusion.h(range(n_qubits))
    
    return diffusion


def grovers_algorithm(n_qubits, marked_states, iterations=None):
    """
    Implement Grover's algorithm for searching.
    
    Args:
        n_qubits: Number of qubits
        marked_states: The state(s) to search for (string or list of strings)
        iterations: Number of Grover iterations (default: optimal number)
    
    Returns:
        QuantumCircuit: Complete Grover's algorithm circuit
    """
    # Convert to list if single state
    if isinstance(marked_states, str):
        marked_states_list = [marked_states]
    else:
        marked_states_list = marked_states
    
    # Calculate optimal number of iterations if not provided
    if iterations is None:
        N = 2 ** n_qubits
        M = len(marked_states_list)  # Number of marked states
        # Optimal iterations for multiple marked states
        iterations = int(np.pi / 4 * np.sqrt(N / M))
    
    # Create quantum circuit
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Initialize in superposition
    circuit.h(range(n_qubits))
    circuit.barrier()
    
    # Apply Grover iterations
    oracle = create_oracle(n_qubits, marked_states)
    diffusion = create_diffusion_operator(n_qubits)
    
    for i in range(iterations):
        circuit.append(oracle, range(n_qubits))
        circuit.barrier()
        circuit.append(diffusion, range(n_qubits))
        circuit.barrier()
    
    # Measure
    circuit.measure(qr, cr)
    
    return circuit


def run_on_simulator(circuit):
    """
    Run the circuit on a local simulator.
    
    Args:
        circuit: Quantum circuit to run
    
    Returns:
        dict: Measurement results
    """
    from qiskit_aer import AerSimulator
    
    # Create simulator
    simulator = AerSimulator()
    
    # Transpile circuit for simulator
    transpiled = transpile(circuit, simulator)
    
    # Run simulation
    job = simulator.run(transpiled, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    return counts


def run_on_ibm_quantum(circuit, backend_name=None):
    """
    Run the circuit on an IBM quantum computer.
    
    Args:
        circuit: Quantum circuit to run
        backend_name: Name of the backend (default: least busy)
    
    Returns:
        dict: Measurement results
    """
    try:
        # Load IBM Quantum account
        service = QiskitRuntimeService()
    except Exception:
        print("\n[-] No IBM Quantum account configured.")
        print("Please run setup first or the script will prompt you.")
        raise
    
    # Select backend
    if backend_name is None:
        # Get least busy backend with enough qubits
        backend = service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=circuit.num_qubits
        )
    else:
        backend = service.backend(backend_name)
    
    print(f"Running on backend: {backend.name}")
    print(f"Backend status: {backend.status()}")
    
    # Transpile circuit for the backend
    transpiled = transpile(circuit, backend, optimization_level=3)
    
    print(f"\nCircuit stats:")
    print(f"  Original depth: {circuit.depth()}")
    print(f"  Transpiled depth: {transpiled.depth()}")
    print(f"  Number of gates: {transpiled.size()}")
    
    # Create sampler and run
    sampler = Sampler(backend)
    job = sampler.run([transpiled], shots=1024)
    
    print(f"\nJob submitted! Job ID: {job.job_id()}")
    print("Waiting for results...")
    
    result = job.result()
    
    # Extract counts from PrimitiveResult
    counts = result[0].data.c.get_counts()
    
    return counts


def plot_results(counts, marked_states, title="Grover's Algorithm Results"):
    """
    Plot the measurement results.
    
    Args:
        counts: Measurement counts dictionary
        marked_states: The target state(s) (string or list)
        title: Plot title
    """
    # Convert to list if single state
    if isinstance(marked_states, str):
        marked_states_list = [marked_states]
    else:
        marked_states_list = marked_states
    
    # Sort counts by state
    sorted_counts = dict(sorted(counts.items()))
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    states = list(sorted_counts.keys())
    values = list(sorted_counts.values())
    colors = ['red' if state in marked_states_list else 'blue' for state in states]
    
    ax.bar(states, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Quantum State', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    if len(marked_states_list) == 1:
        target_label = f'Target: |{marked_states_list[0]}⟩'
    else:
        target_label = f'Targets: {", ".join([f"|{s}⟩" for s in marked_states_list])}'
    
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label=target_label),
        Patch(facecolor='blue', alpha=0.7, label='Other states')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    return fig


def analyze_results(counts, marked_states):
    """
    Analyze and print statistics about the results.
    
    Args:
        counts: Measurement counts
        marked_states: The target state(s) (string or list)
    """
    # Convert to list if single state
    if isinstance(marked_states, str):
        marked_states_list = [marked_states]
    else:
        marked_states_list = marked_states
    
    total_shots = sum(counts.values())
    
    # Calculate total success for all target states
    total_target_counts = sum(counts.get(state, 0) for state in marked_states_list)
    success_rate = (total_target_counts / total_shots) * 100
    
    print("\n" + "="*60)
    print("GROVER'S ALGORITHM RESULTS")
    print("="*60)
    
    if len(marked_states_list) == 1:
        print(f"Target state: |{marked_states_list[0]}⟩")
    else:
        print(f"Target states: {', '.join([f'|{s}⟩' for s in marked_states_list])}")
    
    print(f"Total measurements: {total_shots}")
    print(f"Target state(s) measured: {total_target_counts} times")
    print(f"Success rate: {success_rate:.2f}%")
    
    N = 2 ** len(marked_states_list[0])  # Total states in search space
    M = len(marked_states_list)  # Number of marked states
    classical_rate = (M / N) * 100
    
    print(f"\nTheoretical success rate (ideal): ~100%")
    print(f"Classical random search: {classical_rate:.2f}%")
    if classical_rate > 0:
        print(f"Quantum speedup demonstrated: {success_rate / classical_rate:.2f}x")
    print("="*60)
    
    print("\nAll measurement results:")
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_shots) * 100
        bar = "=" * int(percentage / 2)
        marker = " <- TARGET" if state in marked_states_list else ""
        print(f"|{state}⟩: {count:4d} ({percentage:5.2f}%) {bar}{marker}")
        print(f"|{state}⟩: {count:4d} ({percentage:5.2f}%) {bar}{marker}")


def main():
    """
    Main function to run Grover's algorithm.
    """
    print("="*60)
    print("GROVER'S QUANTUM SEARCH ALGORITHM")
    print("="*60)
    
    # Configuration - You can set single state or multiple states
    n_qubits = 3  # Number of qubits (search space size: 2^n)
    
    # Example 1: Single state search
    marked_states = '101'
    
    # Example 2: Multiple states search
    # marked_states = ['101', '110']  # Search for multiple states
    
    print(f"\nConfiguration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Search space size: {2**n_qubits} states")
    
    if isinstance(marked_states, str):
        print(f"  Target state: |{marked_states}⟩ (decimal: {int(marked_states, 2)})")
        M = 1
    else:
        print(f"  Target states: {', '.join([f'|{s}⟩ (decimal: {int(s, 2)})' for s in marked_states])}")
        M = len(marked_states)
    
    # Calculate optimal iterations
    N = 2 ** n_qubits
    optimal_iterations = int(np.pi / 4 * np.sqrt(N / M))
    print(f"  Number of marked states: {M}")
    print(f"  Optimal iterations: {optimal_iterations}")
    
    # Create circuit
    print("\nBuilding quantum circuit...")
    circuit = grovers_algorithm(n_qubits, marked_states)
    
    print(f"\nCircuit created!")
    print(f"  Total qubits: {circuit.num_qubits}")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Number of gates: {circuit.size()}")
    
    # Draw circuit
    print("\nCircuit diagram:")
    print(circuit.draw(output='text', fold=-1))
    
    # Run on simulator first
    print("\n" + "="*60)
    print("STEP 1: Running on local simulator...")
    print("="*60)
    sim_counts = run_on_simulator(circuit)
    analyze_results(sim_counts, marked_states)
    
    # Plot simulator results
    title = "Grover's Algorithm - Simulator Results"
    if M > 1:
        title += f" ({M} target states)"
    fig_sim = plot_results(sim_counts, marked_states, title)
    plt.savefig('/home/abhi/qiskit-clean/grover_simulator_results.png', dpi=300, bbox_inches='tight')
    print("\nSimulator results saved to: grover_simulator_results.png")
    
    # Ask user if they want to run on real quantum hardware
    print("\n" + "="*60)
    print("STEP 2: Run on IBM Quantum Hardware")
    print("="*60)
    
    # Check if IBM Quantum account is configured, if not, set it up
    try:
        service = QiskitRuntimeService()
        ibm_configured = True
    except Exception:
        ibm_configured = False
    
    if not ibm_configured:
        print("\nIBM Quantum account not found. Setting up now...")
        ibm_configured = setup_ibm_account()
    
    if ibm_configured:
        try:
            print("\nConnecting to IBM Quantum...")
            ibm_counts = run_on_ibm_quantum(circuit)
            
            print("\n" + "="*60)
            print("IBM QUANTUM HARDWARE RESULTS")
            print("="*60)
            analyze_results(ibm_counts, marked_states)
            
            # Plot IBM results
            ibm_title = "Grover's Algorithm - IBM Quantum Hardware Results"
            if M > 1:
                ibm_title += f" ({M} target states)"
            fig_ibm = plot_results(ibm_counts, marked_states, ibm_title)
            plt.savefig('/home/abhi/qiskit-clean/grover_ibm_results.png', dpi=300, bbox_inches='tight')
            print("\nIBM Quantum results saved to: grover_ibm_results.png")
            
            # Compare results
            print("\n" + "="*60)
            print("COMPARISON: Simulator vs IBM Quantum")
            print("="*60)
            
            # Calculate success rates for all marked states
            marked_states_list = [marked_states] if isinstance(marked_states, str) else marked_states
            sim_success = sum(sim_counts.get(s, 0) for s in marked_states_list) / sum(sim_counts.values()) * 100
            ibm_success = sum(ibm_counts.get(s, 0) for s in marked_states_list) / sum(ibm_counts.values()) * 100
            
            print(f"Simulator success rate: {sim_success:.2f}%")
            print(f"IBM Quantum success rate: {ibm_success:.2f}%")
            print(f"Difference: {abs(sim_success - ibm_success):.2f}%")
            print("\nNote: Real quantum hardware has noise and decoherence,")
            print("which can reduce the success rate compared to simulation.")
            
        except Exception as e:
            print(f"\nCould not run on IBM Quantum: {e}")
            print("\nThe simulator results are available above.")
    else:
        print("\nSkipping IBM Quantum execution. Running on simulator only.")
        print("Add IBM_QUANTUM_TOKEN to .env file to enable IBM Quantum hardware.")
    
    print("\n" + "="*60)
    print("Grover's algorithm execution complete!")
    print("="*60)


if __name__ == "__main__":
    main()
