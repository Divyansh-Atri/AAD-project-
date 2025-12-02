import sys
import os
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- CONFIGURATION ---
API_TOKEN = 'PI_TOKEN'  
NUM_BITS_NEEDED = 10000  # Number of random bits to generate
OUTPUT_FILE = "true_quantum.bin"
# ---------------------

def get_quantum_random_bits(num_bits):
    """
    Generate random bits using IBM's quantum computers.
    
    Args:
        num_bits (int): Number of random bits to generate
        
    Returns:
        str: String of '0's and '1's with length equal to num_bits
    """
    try:
        print("Connecting to IBM Quantum...")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_TOKEN)
        
        # Select the least busy real quantum computer
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Connected to backend: {backend.name}")

        # Create a circuit with multiple qubits in superposition
        num_qubits = min(100, backend.num_qubits) 
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))  # Apply Hadamard gate to create superposition
        qc.measure_all()

        # Optimize the circuit for the target hardware
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)

        # Calculate number of shots needed
        shots = (num_bits // num_qubits) + 10
        print(f"Submitting job for {shots} shots to quantum computer...")
        
        # Run the job
        sampler = Sampler(backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        
        # Process the results
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        
        # Convert counts to a bit string
        bit_string = ""
        for bits, count in counts.items():
            bit_string += bits * count
            if len(bit_string) >= num_bits:
                break
                
        return bit_string[:num_bits]
        
    except Exception as e:
        print(f"Error generating quantum random bits: {str(e)}")
        sys.exit(1)

def save_bits_to_file(bits, filename):
    """Save bits to a binary file."""
    try:
        # Convert '0'/'1' string to bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) < 8:
                byte = byte.ljust(8, '0')  # Pad with zeros if needed
            byte_array.append(int(byte, 2))
            
        with open(filename, "wb") as f:
            f.write(byte_array)
            
        print(f"\nSUCCESS: Saved {len(bits)} quantum random bits to '{filename}' ({len(byte_array)} bytes)")
        return True
        
    except Exception as e:
        print(f"Error saving to file: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Generating {NUM_BITS_NEEDED} quantum random bits...")
    bits = get_quantum_random_bits(NUM_BITS_NEEDED)
    save_bits_to_file(bits, OUTPUT_FILE)
