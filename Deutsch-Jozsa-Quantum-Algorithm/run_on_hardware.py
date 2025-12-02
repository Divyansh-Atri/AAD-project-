"""
Run Deutsch-Jozsa Algorithm on Real IBM Quantum Hardware
"""

from src.deutsch_jozsa import DeutschJozsa
from src.hardware_deployment import IBMQuantumDeployer
from src.visualization import DJVisualizer
import matplotlib.pyplot as plt

print("=" * 70)
print("DEUTSCH-JOZSA ON REAL QUANTUM HARDWARE")
print("=" * 70)
print()

# Step 1: Setup connection
print("Step 1: Connecting to IBM Quantum...")
deployer = IBMQuantumDeployer()
deployer.setup_ibm_account()
print()

# Step 2: Select backend (automatically chooses least busy)
print("Step 2: Selecting quantum computer...")
deployer.select_backend(min_qubits=5)
print()

# Step 3: Create Deutsch-Jozsa circuit
print("Step 3: Creating Deutsch-Jozsa circuit...")
dj = DeutschJozsa(n_qubits=3)
circuit = dj.create_circuit('constant', {'output_value': 0})
print(f"✅ Circuit created: {circuit.num_qubits} qubits, depth {circuit.depth()}")
print()

# Step 4: Run on real quantum hardware!
print("Step 4: Running on REAL quantum hardware...")
print("⚠️  NOTE: This may take 5-30 minutes depending on queue!")
print("         The job will be submitted and we'll wait for results.")
print()

try:
    result = deployer.run_on_hardware(circuit, shots=1024, optimization_level=3)
    
    print()
    print("=" * 70)
    print("🎉 HARDWARE EXECUTION COMPLETE!")
    print("=" * 70)
    print(f"Backend: {result.backend_name}")
    print(f"Job ID: {result.job_id}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    print(f"Circuit depth: {result.circuit_depth}")
    print(f"Gate count: {result.gate_count}")
    print(f"Detected result: {result.result}")
    print(f"Expected: constant")
    print(f"Success: {'✅ YES' if result.result == 'constant' else '❌ NO'}")
    print()
    print(f"Measurement counts: {result.counts}")
    print()
    
    # Step 5: Visualize results
    print("Step 5: Creating visualization...")
    visualizer = DJVisualizer()
    fig = visualizer.plot_measurement_results(
        result.counts,
        title=f"Deutsch-Jozsa on {result.backend_name} (Real Quantum Hardware)"
    )
    plt.savefig('hardware_result.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved to: hardware_result.png")
    print()
    
    print("=" * 70)
    print("🏆 CONGRATULATIONS!")
    print("=" * 70)
    print("You just ran your Deutsch-Jozsa algorithm on a REAL quantum computer!")
    print()
    print(f"Your quantum computation ran on: {result.backend_name}")
    print(f"Located in: IBM Quantum Data Center")
    print(f"Job ID: {result.job_id}")
    print()
    print("Your project is now 100% complete! 🎉")
    print("=" * 70)
    
except KeyboardInterrupt:
    print()
    print("⚠️  Execution cancelled by user")
    print()
except Exception as e:
    print()
    print("❌ Error during execution:")
    print(f"   {e}")
    print()
    print("This might be due to:")
    print("  - Queue is too long (try again later)")
    print("  - Backend is down for maintenance")
    print("  - Network connection issue")
    print()
