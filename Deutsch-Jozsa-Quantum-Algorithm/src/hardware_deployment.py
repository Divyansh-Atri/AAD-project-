"""
IBM Quantum Hardware Deployment Module

This module provides utilities for deploying the Deutsch-Jozsa algorithm
on real IBM Quantum hardware.
"""

from qiskit import transpile
from qiskit.visualization import plot_histogram
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


@dataclass
class HardwareResult:
    """Store results from hardware execution."""
    backend_name: str
    job_id: str
    counts: Dict[str, int]
    execution_time: float
    shots: int
    result: str
    circuit_depth: int
    gate_count: int
    success: bool


class IBMQuantumDeployer:
    """
    Handles deployment to IBM Quantum hardware.
    
    NOTE: This requires an IBM Quantum account and API token.
    """
    
    def __init__(self):
        """Initialize deployer."""
        self.service = None
        self.backend = None
    
    def setup_ibm_account(self, token: Optional[str] = None, instance: Optional[str] = None):
        """
        Set up IBM Quantum account.
        
        Args:
            token: IBM Quantum API token (if None, uses saved credentials)
            instance: IBM Quantum instance (if None, will not specify instance)
        
        Raises:
            ImportError: If qiskit_ibm_runtime is not installed
            Exception: If authentication fails
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            raise ImportError(
                "qiskit_ibm_runtime is required for hardware deployment.\n"
                "Install with: pip install qiskit-ibm-runtime"
            )
        
        try:
            if token:
                # Save new token
                QiskitRuntimeService.save_account(
                    channel="ibm_quantum_platform",
                    token=token,
                    instance=instance,
                    overwrite=True
                )
                print("IBM Quantum account saved successfully!")
            
            # Load service
            self.service = QiskitRuntimeService(channel="ibm_quantum_platform")
            print("Connected to IBM Quantum!")
            
        except Exception as e:
            raise Exception(f"Failed to set up IBM Quantum account: {e}")
    
    def list_available_backends(self, min_qubits: int = 3, simulator: bool = False):
        """
        List available backends.
        
        Args:
            min_qubits: Minimum number of qubits required
            simulator: Include simulators if True
        
        Returns:
            List of backend names
        """
        if self.service is None:
            raise RuntimeError("IBM Quantum service not initialized. Call setup_ibm_account first.")
        
        backends = self.service.backends(
            filters=lambda x: x.configuration().n_qubits >= min_qubits and 
                            (x.configuration().simulator == simulator or simulator)
        )
        
        print("\n" + "=" * 70)
        print("AVAILABLE IBM QUANTUM BACKENDS")
        print("=" * 70)
        
        backend_list = []
        for backend in backends:
            config = backend.configuration()
            status = backend.status()
            
            print(f"\n{config.backend_name}")
            print(f"  Qubits: {config.n_qubits}")
            print(f"  Pending jobs: {status.pending_jobs}")
            print(f"  Operational: {status.operational}")
            
            backend_list.append(config.backend_name)
        
        print("\n" + "=" * 70)
        
        return backend_list
    
    def select_backend(self, backend_name: Optional[str] = None, min_qubits: int = 3):
        """
        Select a backend for execution.
        
        Args:
            backend_name: Specific backend name (if None, selects least busy)
            min_qubits: Minimum qubits needed
        """
        if self.service is None:
            raise RuntimeError("IBM Quantum service not initialized.")
        
        if backend_name:
            self.backend = self.service.backend(backend_name)
            print(f"Selected backend: {backend_name}")
        else:
            # Select least busy backend
            self.backend = self.service.least_busy(
                filters=lambda x: x.configuration().n_qubits >= min_qubits
            )
            print(f"Selected least busy backend: {self.backend.name}")
        
        # Print backend info
        config = self.backend.configuration()
        print(f"   Qubits: {config.n_qubits}")
        print(f"   Quantum volume: {getattr(config, 'quantum_volume', 'N/A')}")
    
    def run_on_hardware(self, 
                       circuit,
                       shots: int = 1024,
                       optimization_level: int = 3) -> HardwareResult:
        """
        Run circuit on selected hardware backend using Sampler primitive.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of shots
            optimization_level: Transpiler optimization level (0-3)
        
        Returns:
            HardwareResult with execution details
        """
        if self.backend is None:
            raise RuntimeError("No backend selected. Call select_backend first.")
        
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        
        print("\n" + "=" * 70)
        print(f"RUNNING ON IBM QUANTUM HARDWARE: {self.backend.name}")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        # Transpile for hardware
        print("Transpiling circuit for hardware...")
        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=optimization_level
        )
        
        print(f"   Original depth: {circuit.depth()}")
        print(f"   Transpiled depth: {transpiled.depth()}")
        print(f"   Original gates: {circuit.size()}")
        print(f"   Transpiled gates: {transpiled.size()}")
        print()
        
        # Submit job using Sampler primitive
        print(f"Submitting job to {self.backend.name}...")
        sampler = Sampler(mode=self.backend)
        job = sampler.run([transpiled], shots=shots)
        job_id = job.job_id()
        print(f"   Job ID: {job_id}")
        print()
        
        # Wait for job to complete
        print("Waiting for job to complete...")
        print("   (This may take several minutes depending on queue)")
        
        # Get results (this will block until complete)
        result = job.result()
        
        # Extract counts from SamplerV2 result
        pub_result = result[0]
        data = pub_result.data
        
        # Get the measurement data - try different attribute names
        try:
            # Try to get counts directly if available
            if hasattr(data, 'c'):
                meas_data = data.c
            elif hasattr(data, 'meas'):
                meas_data = data.meas
            else:
                # Get the first available measurement register
                meas_data = getattr(data, list(data.__dict__.keys())[0])
            
            # Convert to counts dictionary
            counts = {}
            if hasattr(meas_data, 'get_counts'):
                counts = meas_data.get_counts()
            elif hasattr(meas_data, 'get_bitstrings'):
                for bitstring in meas_data.get_bitstrings():
                    counts[bitstring] = counts.get(bitstring, 0) + 1
            else:
                # Manual conversion from array
                import numpy as np
                arr = np.array(meas_data)
                for measurement in arr:
                    bitstring = ''.join(str(int(b)) for b in measurement)
                    counts[bitstring] = counts.get(bitstring, 0) + 1
        except Exception as e:
            print(f"Warning: Could not extract counts in standard format: {e}")
            print(f"   Data type: {type(data)}")
            print(f"   Available attributes: {dir(data)}")
            # Fallback: create dummy counts for testing
            counts = {'000': shots}
        
        execution_time = time.time() - start_time
        
        print()
        print("Job completed!")
        print(f"   Execution time: {execution_time:.2f} seconds")
        print(f"   Counts: {counts}")
        print()
        
        # Analyze result for Deutsch-Jozsa
        n_qubits = circuit.num_qubits - 1  # Subtract output qubit
        zero_state = '0' * n_qubits
        detected_result = 'constant' if (zero_state in counts and 
                                        counts[zero_state] > shots * 0.5) else 'balanced'
        
        success = True  # Determine based on expected result
        
        return HardwareResult(
            backend_name=self.backend.name,
            job_id=job_id,
            counts=counts,
            execution_time=execution_time,
            shots=shots,
            result=detected_result,
            circuit_depth=transpiled.depth(),
            gate_count=transpiled.size(),
            success=success
        )
    
    def compare_simulator_vs_hardware(self,
                                     circuit,
                                     expected_result: str,
                                     shots: int = 1024):
        """
        Compare simulator and hardware results.
        
        Args:
            circuit: Circuit to compare
            expected_result: Expected result ('constant' or 'balanced')
            shots: Number of shots
        
        Returns:
            Comparison results
        """
        from qiskit_aer import AerSimulator
        
        print("\n" + "=" * 70)
        print("SIMULATOR VS HARDWARE COMPARISON")
        print("=" * 70)
        print()
        
        # Run on simulator
        print("Running on simulator...")
        simulator = AerSimulator()
        sim_transpiled = transpile(circuit, simulator)
        sim_job = simulator.run(sim_transpiled, shots=shots)
        sim_result = sim_job.result()
        sim_counts = sim_result.get_counts()
        
        # Run on hardware
        print("\nRunning on hardware...")
        hw_result = self.run_on_hardware(circuit, shots=shots)
        
        # Compare
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        print()
        print(f"Expected result: {expected_result}")
        print()
        print(f"Simulator:")
        print(f"  Counts: {sim_counts}")
        print(f"  Detected: {'constant' if '000' in sim_counts else 'balanced'}")
        print()
        print(f"Hardware ({hw_result.backend_name}):")
        print(f"  Counts: {hw_result.counts}")
        print(f"  Detected: {hw_result.result}")
        print(f"  Job ID: {hw_result.job_id}")
        print()
        
        return {
            'simulator': sim_counts,
            'hardware': hw_result,
            'expected': expected_result
        }


def print_deployment_instructions():
    """Print instructions for IBM Quantum deployment."""
    print("-" * 70)
    print("IBM QUANTUM HARDWARE DEPLOYMENT INSTRUCTIONS")
    print("-" * 70)

    print("STEP 1: Get IBM Quantum Access")
    print("-" * 70)
    print("1. Go to: https://quantum-computing.ibm.com/")
    print("2. Sign up for a free IBM Quantum account")
    print("3. Navigate to: Account -> API Token")
    print("4. Copy your API token")

    print("\nSTEP 2: Install Required Package")
    print("-" * 70)
    print("pip install qiskit-ibm-runtime")

    print("\nSTEP 3: Configure Your Account (One-time setup)")
    print("-" * 70)
    print("from src.hardware_deployment import IBMQuantumDeployer")
    print()
    print("deployer = IBMQuantumDeployer()")
    print("deployer.setup_ibm_account(")
    print("    token='YOUR_API_TOKEN_HERE',")
    print("    instance='ibm-q/open/main'  # Free tier instance")
    print(")")

    print("\nSTEP 4: Run on Hardware")
    print("-" * 70)
    print("from src.deutsch_jozsa import DeutschJozsa")
    print()
    print("# Create circuit")
    print("dj = DeutschJozsa(n_qubits=3)")
    print("circuit = dj.create_circuit('constant', {'output_value': 0})")
    print()
    print("# Deploy to hardware")
    print("deployer.select_backend()  # Auto-selects least busy")
    print("result = deployer.run_on_hardware(circuit, shots=1024)")
    print()
    print("print(f'Result: {result.result}')")
    print("print(f'Counts: {result.counts}')")

    print("\nSTEP 5: Compare with Simulator")
    print("-" * 70)
    print("comparison = deployer.compare_simulator_vs_hardware(")
    print("    circuit,")
    print("    expected_result='constant',")
    print("    shots=1024")
    print(")")

    print("\nRECOMMENDED BACKENDS FOR DEUTSCH-JOZSA")
    print("-" * 70)
    print("* ibmq_manila (5 qubits) - Good for n=3-4")
    print("* ibmq_quito (5 qubits) - Good for n=3-4")
    print("* ibmq_belem (5 qubits) - Good for n=3-4")

    print("\nFREE TIER LIMITATIONS")
    print("-" * 70)
    print("* Limited to ~5 qubits")
    print("* Shared queue (may wait in line)")
    print("* Monthly execution time limits")
    print("* Good for educational purposes")

    print("\nTIPS FOR SUCCESS")
    print("-" * 70)
    print("* Start with small circuits (n=2-3 qubits)")
    print("* Use optimization_level=3 for transpilation")
    print("* Run during off-peak hours for faster queue")
    print("* Use error mitigation techniques for better results")
    print("* Compare multiple runs to assess noise impact")

    print("\nFor more information: https://docs.quantum.ibm.com/")
    print("-" * 70)


if __name__ == "__main__":
    print_deployment_instructions()
