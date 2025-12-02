"""
Test IBM Quantum Connection - List Available Backends
"""

from src.hardware_deployment import IBMQuantumDeployer

print("=" * 70)
print("TESTING IBM QUANTUM CONNECTION")
print("=" * 70)
print()

# Initialize deployer
deployer = IBMQuantumDeployer()

# Connect (uses saved credentials)
print("Connecting to IBM Quantum...")
deployer.setup_ibm_account()

print()
print("=" * 70)
print("LISTING AVAILABLE QUANTUM COMPUTERS")
print("=" * 70)

# List available backends
available = deployer.list_available_backends(min_qubits=3)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Connection successful!")
print(f"✅ Found {len(available)} available quantum backend(s)")
print()
print("You can now run your Deutsch-Jozsa algorithm on real quantum hardware!")
print()
