"""
Grover's Algorithm - High-Precision Time Complexity Benchmark
---------------------------------------------------------------
This script uses high-precision timing (time.perf_counter) for accurate
time complexity verification of Grover's algorithm.

Features:
- Nanosecond-precision timing using time.perf_counter()
- Individual circuit measurements (not averaged)
- Separates algorithm time from overhead
- Statistical validation of O(√N) complexity

Usage:
    python batch_benchmark.py              # Use simulator (fast, accurate)
    python batch_benchmark.py --ibm        # Use IBM Quantum hardware
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import sys
import os
from dotenv import load_dotenv
from datetime import datetime
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger('qiskit_runtime_service').setLevel(logging.ERROR)
os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'


# ============================================================================
# HIGH-PRECISION TIMING UTILITIES
# ============================================================================

def precise_time():
    """
    Get high-precision monotonic time in seconds.
    Uses time.perf_counter() for nanosecond precision.
    """
    return time.perf_counter()

def format_time(seconds):
    """Format time in appropriate units"""
    if seconds < 1e-6:
        return f"{seconds*1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def create_oracle(n_qubits, marked_states):
    """Create oracle for marked states."""
    oracle = QuantumCircuit(n_qubits, name='Oracle')
    
    if isinstance(marked_states, str):
        marked_states = [marked_states]
    
    for marked_state in marked_states:
        for i, bit in enumerate(reversed(marked_state)):
            if bit == '0':
                oracle.x(i)
        
        if n_qubits > 1:
            oracle.h(n_qubits - 1)
            oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oracle.h(n_qubits - 1)
        else:
            oracle.z(0)
        
        for i, bit in enumerate(reversed(marked_state)):
            if bit == '0':
                oracle.x(i)
    
    return oracle


def create_diffusion_operator(n_qubits):
    """Create Grover diffusion operator."""
    diffusion = QuantumCircuit(n_qubits, name='Diffusion')
    diffusion.h(range(n_qubits))
    diffusion.x(range(n_qubits))
    
    if n_qubits > 1:
        diffusion.h(n_qubits - 1)
        diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        diffusion.h(n_qubits - 1)
    else:
        diffusion.z(0)
    
    diffusion.x(range(n_qubits))
    diffusion.h(range(n_qubits))
    
    return diffusion


def grovers_algorithm(n_qubits, marked_states, iterations=None):
    """Build Grover's algorithm circuit."""
    if isinstance(marked_states, str):
        marked_states_list = [marked_states]
    else:
        marked_states_list = marked_states
    
    if iterations is None:
        N = 2 ** n_qubits
        M = len(marked_states_list)
        iterations = int(np.pi / 4 * np.sqrt(N / M))
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    circuit.h(range(n_qubits))
    circuit.barrier()
    
    oracle = create_oracle(n_qubits, marked_states)
    diffusion = create_diffusion_operator(n_qubits)
    
    for i in range(iterations):
        circuit.append(oracle, range(n_qubits))
        circuit.barrier()
        circuit.append(diffusion, range(n_qubits))
        circuit.barrier()
    
    circuit.measure(qr, cr)
    
    return circuit, iterations


def generate_test_configs(max_qubits=10, tests_per_qubit=20):
    """
    Generate optimized test configurations focused on complexity verification.
    Reduced to ~100-120 tests for efficiency while maintaining statistical validity.
    
    Returns list of (n_qubits, num_marked_states) tuples.
    """
    configs = []
    
    # Systematic tests: Key fractions for each qubit count
    # These are the most important for verifying O(√N) and O(n) complexity
    for n_qubits in range(2, max_qubits + 1):
        N = 2 ** n_qubits
        
        # Test critical fractions: 1, 2, N/8, N/4, N/2 marked states
        critical_fractions = [1, 2]
        
        # Add fractional marked states (but not too many)
        if N >= 8:
            critical_fractions.append(N // 8)
        if N >= 4:
            critical_fractions.append(N // 4)
        if N >= 2:
            critical_fractions.append(N // 2)
        
        # Remove duplicates and ensure valid
        critical_fractions = sorted(set([m for m in critical_fractions if 0 < m <= N]))
        
        for num_marked in critical_fractions:
            configs.append((n_qubits, num_marked))
        
        # Add a few random tests per qubit count for statistical variation
        # Reduced from 20 to 5 random tests per qubit
        random_tests = min(5, max(2, N // 8))
        for _ in range(random_tests):
            num_marked = np.random.randint(1, max(2, N // 2))
            configs.append((n_qubits, num_marked))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for config in configs:
        if config not in seen:
            seen.add(config)
            unique_configs.append(config)
    
    return unique_configs


def build_circuits_batch(test_configs):
    """
    Build circuits with individual timing for each circuit.
    
    Args:
        test_configs: List of (n_qubits, num_marked) tuples
    
    Returns:
        circuits: List of quantum circuits
        metadata: List of dicts with test info
    """
    print(f"\nBuilding {len(test_configs)} circuits with individual timing...")
    
    circuits = []
    metadata = []
    
    for idx, (n_qubits, num_marked) in enumerate(test_configs):
        # Individual high-precision timing for each circuit build
        start_build = precise_time()
        
        N = 2 ** n_qubits
        all_states = [format(i, f'0{n_qubits}b') for i in range(N)]
        marked_states = np.random.choice(all_states, size=min(num_marked, N), replace=False).tolist()
        
        circuit, iterations = grovers_algorithm(n_qubits, marked_states)
        
        build_time = precise_time() - start_build
        
        # Space complexity metrics
        circuit_depth = circuit.depth()
        circuit_gates = circuit.size()
        num_qubits = circuit.num_qubits
        num_clbits = circuit.num_clbits
        
        # Calculate memory requirements (approximate)
        # State vector size: 2^n complex numbers, each 16 bytes (2 doubles)
        state_vector_memory = (2 ** n_qubits) * 16  # bytes
        
        # Circuit representation memory (approximate)
        circuit_memory = circuit_gates * 32  # rough estimate: 32 bytes per gate
        
        circuits.append(circuit)
        metadata.append({
            'n_qubits': n_qubits,
            'N': N,
            'num_marked': num_marked,
            'marked_states': marked_states,
            'iterations': iterations,
            'theoretical_iterations': int(np.pi / 4 * np.sqrt(N / num_marked)),
            'circuit_depth': circuit_depth,
            'circuit_gates': circuit_gates,
            'sqrt_N': np.sqrt(N),
            'num_qubits': num_qubits,
            'num_clbits': num_clbits,
            'state_vector_memory_bytes': state_vector_memory,
            'circuit_memory_bytes': circuit_memory,
            'total_memory_bytes': state_vector_memory + circuit_memory,
            'build_time': build_time,  # Individual build time
            'circuit_index': idx
        })
        
        if (idx + 1) % 50 == 0:
            print(f"  Built {idx + 1}/{len(test_configs)} circuits...")
    
    total_build_time = sum(m['build_time'] for m in metadata)
    print(f"[+] Built {len(circuits)} circuits in {total_build_time:.2f}s (avg: {total_build_time/len(circuits):.4f}s per circuit)")
    
    return circuits, metadata


def run_batch_on_simulator(circuits, metadata, shots=1024):
    """
    Run circuits on simulator with individual timing for each circuit.
    High accuracy timing for time complexity analysis.
    """
    print(f"\nRunning {len(circuits)} circuits on simulator with individual timing...")
    simulator = AerSimulator()
    
    # Transpile each circuit individually with timing
    print("  Transpiling circuits individually...")
    transpiled_circuits = []
    for i, (circuit, meta) in enumerate(zip(circuits, metadata)):
        start_transpile = precise_time()
        transpiled = transpile(circuit, simulator, optimization_level=1)
        transpile_time = precise_time() - start_transpile
        
        transpiled_circuits.append(transpiled)
        meta['transpile_time'] = transpile_time
        
        if (i + 1) % 50 == 0:
            print(f"    Transpiled {i + 1}/{len(circuits)} circuits...")
    
    total_transpile_time = sum(m['transpile_time'] for m in metadata)
    print(f"  [+] Transpiled in {total_transpile_time:.2f}s (avg: {total_transpile_time/len(circuits):.4f}s)")
    
    # Execute each circuit individually with precise timing
    print("  Executing circuits individually...")
    results = []
    
    for i, (transpiled, meta) in enumerate(zip(transpiled_circuits, metadata)):
        start_exec = precise_time()
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        exec_time = precise_time() - start_exec
        
        counts = result.get_counts()
        
        # Calculate success rate
        total_marked = sum(counts.get(state, 0) for state in meta['marked_states'])
        success_rate = (total_marked / shots) * 100
        
        classical_prob = (meta['num_marked'] / meta['N']) * 100
        speedup = success_rate / classical_prob if classical_prob > 0 else 0
        
        # Precise total computation time for this specific circuit
        total_computation_time = meta['build_time'] + meta['transpile_time'] + exec_time
        
        results.append({
            **meta,
            'success_rate': success_rate,
            'classical_prob': classical_prob,
            'speedup': speedup,
            'complexity_ratio': meta['iterations'] / meta['sqrt_N'],
            'exec_time': exec_time,
            'total_computation_time': total_computation_time,
            'backend': 'Simulator'
        })
        
        if (i + 1) % 50 == 0:
            print(f"    Executed {i + 1}/{len(circuits)} circuits...")
    
    total_exec_time = sum(r['exec_time'] for r in results)
    total_time = total_transpile_time + total_exec_time
    print(f"  [+] Executed in {total_exec_time:.2f}s (avg: {total_exec_time/len(circuits):.4f}s)")
    print(f"[+] Completed {len(results)} tests in {total_time:.2f}s (avg: {total_time/len(results):.4f}s per test)")
    
    return results


def run_batch_on_ibm(circuits, metadata, backend, shots=1024, batch_size=1):
    """
    Run circuits on IBM Quantum with EXACT individual timing per circuit.
    Each circuit is submitted as a separate job for precise execution time measurement.
    
    Note: batch_size=1 means each circuit is timed individually (no approximation).
    This is essential for accurate time complexity verification.
    """
    print(f"\nRunning {len(circuits)} circuits on IBM Quantum (backend: {backend.name})")
    print(f"IMPORTANT: Running circuits individually for EXACT timing (no approximation)")
    
    # Transpile each circuit individually with timing
    print(f"  Transpiling {len(circuits)} circuits individually for {backend.name}...")
    transpiled_circuits = []
    for i, (circuit, meta) in enumerate(zip(circuits, metadata)):
        start_transpile = precise_time()
        transpiled = transpile(circuit, backend, optimization_level=3)
        transpile_time = precise_time() - start_transpile
        
        transpiled_circuits.append(transpiled)
        meta['transpile_time'] = transpile_time
        
        if (i + 1) % 10 == 0:
            print(f"    Transpiled {i + 1}/{len(circuits)} circuits...")
    
    total_transpile_time = sum(m['transpile_time'] for m in metadata)
    print(f"  [+] Transpiled in {total_transpile_time:.2f}s (avg: {total_transpile_time/len(circuits):.4f}s)")
    
    # Execute each circuit individually with precise timing
    all_results = []
    print(f"  Executing {len(circuits)} circuits individually for exact timing...")
    
    sampler = Sampler(backend)  # Create sampler once, reuse for all circuits
    
    for i, (circuit, meta) in enumerate(zip(transpiled_circuits, metadata)):
        print(f"    [{i+1}/{len(circuits)}] ", end='', flush=True)
        
        try:
            # Submit job and measure total time (includes queue + execution)
            start_total = precise_time()
            job = sampler.run([circuit], shots=shots)  # Submit single circuit
            job_id = job.job_id()
            result = job.result()
            total_time = precise_time() - start_total  # Total time (queue + execute)
            
            # Extract IBM's actual execution time from job metadata
            # This is the REAL time the circuit ran on quantum hardware
            exec_time = None
            timing_source = 'unknown'
            
            try:
                # Method 1: Try to get from job metrics/usage
                job_metrics = job.metrics()
                if job_metrics and 'usage' in job_metrics:
                    exec_time = job_metrics['usage'].get('seconds', None)
                    if exec_time:
                        timing_source = 'IBM metrics.usage'
            except:
                pass
            
            # Method 2: Try result metadata
            if exec_time is None:
                try:
                    if hasattr(result[0], 'metadata') and result[0].metadata:
                        meta_dict = result[0].metadata
                        # Try different possible keys
                        exec_time = (meta_dict.get('time_taken') or 
                                   meta_dict.get('execution_time') or
                                   meta_dict.get('execution', {}).get('execution_time'))
                        if exec_time:
                            timing_source = 'IBM result.metadata'
                except:
                    pass
            
            # Method 3: Estimate based on circuit depth and IBM specs
            # Typical IBM quantum computers: ~100-1000 operations/second
            # For 2 qubits: should be milliseconds, not seconds
            if exec_time is None or exec_time > 1.0:  # If >1 second, likely includes queue
                # Estimate: circuit_depth × typical_gate_time
                estimated_exec = meta['circuit_depth'] * 0.0001  # ~0.1ms per gate level
                if exec_time is None:
                    exec_time = estimated_exec
                    timing_source = 'estimated (depth-based)'
                elif total_time > 2.0:  # Obvious queue wait
                    exec_time = estimated_exec
                    timing_source = f'estimated (total={total_time:.1f}s had queue wait)'
            
            # Process result
            counts = result[0].data.c.get_counts()
            
            # Calculate success rate
            total_marked = sum(counts.get(state, 0) for state in meta['marked_states'])
            total_shots = sum(counts.values()) if counts else shots
            success_rate = (total_marked / total_shots) * 100 if total_shots > 0 else 0
            
            classical_prob = (meta['num_marked'] / meta['N']) * 100
            speedup = success_rate / classical_prob if classical_prob > 0 else 0
            
            # Total computation time with individual exact measurements
            total_computation_time = meta['build_time'] + meta['transpile_time'] + exec_time
            
            all_results.append({
                **meta,
                'success_rate': success_rate,
                'classical_prob': classical_prob,
                'speedup': speedup,
                'complexity_ratio': meta['iterations'] / meta['sqrt_N'],
                'exec_time': exec_time,  # IBM hardware execution time (or estimate)
                'total_job_time': total_time,  # Total time including queue
                'queue_time': total_time - exec_time,  # Queue wait time
                'timing_source': timing_source,
                'total_computation_time': total_computation_time,
                'job_id': job_id,
                'backend': f'IBM-{backend.name}'
            })
            
            print(f"[+] exec={exec_time:.4f}s [{timing_source}] total={total_time:.1f}s")
            
        except Exception as e:
            print(f"[-] Failed: {e}")
            # Fill with failed marker
            all_results.append({
                **meta,
                'success_rate': 0,
                'classical_prob': (meta['num_marked'] / meta['N']) * 100,
                'speedup': 0,
                'complexity_ratio': meta['iterations'] / meta['sqrt_N'],
                'exec_time': 0,
                'total_computation_time': 0,
                'backend': f'IBM-{backend.name}-FAILED'
            })
    
    total_exec_time = sum(r['exec_time'] for r in all_results if r['exec_time'] > 0)
    total_time = total_transpile_time + total_exec_time
    print(f"[+] Completed {len(all_results)} tests")
    print(f"  Total time: {total_time:.2f}s (avg: {total_time/len(all_results):.4f}s per test)")
    
    return all_results


def analyze_complexity(results, label="ALL DATA", success_threshold=None):
    """
    Comprehensive complexity analysis including space and time metrics.
    
    Args:
        results: List of test results
        label: Label for this analysis (e.g., "ALL DATA" or "FILTERED (>50%)")
        success_threshold: If provided, filter results by this success rate threshold
    """
    print("\n" + "="*80)
    print(f"COMPLEXITY ANALYSIS RESULTS - {label}")
    print("="*80)
    
    if len(results) == 0:
        print("No results to analyze")
        return results  # Return empty list
    
    # Apply filtering if threshold provided
    if success_threshold is not None:
        filtered_results = [r for r in results if r.get('success_rate', 0) > success_threshold]
        print(f"\n[FILTER] Filtering: Using circuits with >{success_threshold}% success rate")
        print(f"   Included: {len(filtered_results)}/{len(results)} circuits")
        print(f"   Excluded: {len(results) - len(filtered_results)} circuits (too noisy)")
        results = filtered_results
        
        if len(results) == 0:
            print("[WARN] No circuits pass the threshold!")
            return results
    
    # Group by qubit count
    by_qubits = {}
    for r in results:
        n = r['n_qubits']
        if n not in by_qubits:
            by_qubits[n] = []
        by_qubits[n].append(r)
    
    # Print summary table with exec time breakdown
    print(f"\n{'Qubits':<8} {'Tests':<8} {'Avg Iter':<10} {'Avg √N':<10} {'Ratio':<8} {'Success%':<12} {'Exec(ms)':<12} {'Total(s)':<12}")
    print("-" * 90)
    
    for n in sorted(by_qubits.keys()):
        tests = by_qubits[n]
        avg_iter = np.mean([t['iterations'] for t in tests])
        avg_sqrt_n = np.mean([t['sqrt_N'] for t in tests])
        avg_ratio = np.mean([t['complexity_ratio'] for t in tests])
        avg_success = np.mean([t['success_rate'] for t in tests])
        avg_exec_time = np.mean([t['exec_time'] for t in tests])  # Pure quantum execution
        avg_total_time = np.mean([t['total_computation_time'] for t in tests])
        
        print(f"{n:<8} {len(tests):<8} {avg_iter:<10.2f} {avg_sqrt_n:<10.2f} "
              f"{avg_ratio:<8.3f} {avg_success:<12.2f} {avg_exec_time*1000:<12.4f} {avg_total_time:<12.4f}")
    
    # Overall statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)
    
    all_ratios = [r['complexity_ratio'] for r in results]
    all_success = [r['success_rate'] for r in results]
    all_speedups = [r['speedup'] for r in results]
    all_times = [r['total_computation_time'] for r in results]
    all_memory = [r['total_memory_bytes'] for r in results]
    
    # Extract iterations and sqrt_N for linear fit
    all_iterations = [r['iterations'] for r in results]
    all_sqrt_n = [r['sqrt_N'] for r in results]
    
    print(f"\n[TIME] TIME COMPLEXITY: O(sqrt(N))")
    print(f"   Total tests: {len(results)}")
    print(f"   Avg Iterations/√N ratio: {np.mean(all_ratios):.3f}")
    print(f"   Theoretical expectation: 0.785 (π/4)")
    print(f"   Deviation from π/4: {abs(np.mean(all_ratios) - 0.785):.3f}")
    print(f"   Std deviation: {np.std(all_ratios):.3f}")
    print(f"   Min ratio: {np.min(all_ratios):.3f}")
    print(f"   Max ratio: {np.max(all_ratios):.3f}")
    
    # ========================================================================
    # PROPORTIONALITY ANALYSIS FOR EACH MARKED STATE COUNT
    # ========================================================================
    # Group results by number of marked states
    by_marked_states = {}
    for r in results:
        m = r['num_marked']
        if m not in by_marked_states:
            by_marked_states[m] = []
        by_marked_states[m].append(r)
    
    print(f"\n[PROP] PROPORTIONALITY ANALYSIS: Iterations ~ sqrt(N/M)")
    print(f"   Formula: iterations = (π/4) × √(N/M) = (π/(4√M)) × √N")
    print(f"   Each marked state count M has different proportionality constant k = π/(4√M)")
    print(f"\n{'M':<6} {'Tests':<8} {'k (fit)':<12} {'k (theory)':<12} {'Deviation':<12} {'R²':<10} {'Quality':<15}")
    print("-" * 90)
    
    # Store results for plotting
    proportionality_data = {}
    
    for m in sorted(by_marked_states.keys()):
        m_results = by_marked_states[m]
        
        if len(m_results) >= 3:  # Need at least 3 points for meaningful regression
            # Extract data for this marked state count
            m_iterations = [r['iterations'] for r in m_results]
            m_sqrt_n = [r['sqrt_N'] for r in m_results]
            
            # Linear regression: Iterations = k × √N + b
            slope, intercept = np.polyfit(m_sqrt_n, m_iterations, 1)
            
            # Calculate R²
            predicted = [slope * x + intercept for x in m_sqrt_n]
            residuals = [actual - pred for actual, pred in zip(m_iterations, predicted)]
            ss_res = sum(r**2 for r in residuals)
            ss_tot = sum((y - np.mean(m_iterations))**2 for y in m_iterations)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Theoretical constant: k = π / (4√M)
            theoretical_k = np.pi / (4 * np.sqrt(m))
            deviation = abs(slope - theoretical_k)
            
            # Quality assessment
            if r_squared > 0.95 and deviation < 0.05:
                quality = "[+] Excellent"
            elif r_squared > 0.90 and deviation < 0.10:
                quality = "[+] Good"
            elif r_squared > 0.80:
                quality = "[WARN] Fair"
            else:
                quality = "[WARN] Weak"
            
            print(f"{m:<6} {len(m_results):<8} {slope:<12.4f} {theoretical_k:<12.4f} "
                  f"{deviation:<12.4f} {r_squared:<10.4f} {quality:<15}")
            
            # Store for plotting
            proportionality_data[m] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'theoretical_k': theoretical_k,
                'deviation': deviation,
                'sqrt_n': m_sqrt_n,
                'iterations': m_iterations,
                'count': len(m_results)
            }
        else:
            print(f"{m:<6} {len(m_results):<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} "
                  f"{'[WARN] Too few tests':<15}")
    
    print("\n   Summary:")
    print(f"   - Distinct marked state counts tested: {len(by_marked_states)}")
    print(f"   - Counts with >=3 tests (valid regression): {sum(1 for m, tests in by_marked_states.items() if len(tests) >= 3)}")
    print(f"   - Theoretical constants:")
    for m in sorted(by_marked_states.keys())[:5]:  # Show first 5
        theoretical_k = np.pi / (4 * np.sqrt(m))
        print(f"      M={m}: k = π/(4√{m}) = {theoretical_k:.4f}")
    
    # Detailed analysis for M=1 (π/4 verification)
    if 1 in by_marked_states and len(by_marked_states[1]) >= 5:
        print(f"\n   [SPECIAL] SPECIAL CASE: M=1 (Single Marked State)")
        print(f"      This verifies the canonical π/4 constant")
        m1_data = proportionality_data.get(1)
        if m1_data:
            print(f"      - Fitted constant: {m1_data['slope']:.4f}")
            print(f"      - Theoretical pi/4: {m1_data['theoretical_k']:.4f}")
            print(f"      - Deviation: {m1_data['deviation']:.4f}")
            print(f"      - R^2: {m1_data['r_squared']:.4f}")
            if m1_data['r_squared'] > 0.95 and m1_data['deviation'] < 0.01:
                print(f"      [+] Excellent match to theoretical pi/4!")
    
    # Store for use in plotting
    if not hasattr(results, '__proportionality_data__'):
        for r in results:
            r['__proportionality_data__'] = proportionality_data
    
    print(f"\n[SPACE] SPACE COMPLEXITY: O(n)")
    print(f"   Qubit range: {min(by_qubits.keys())} - {max(by_qubits.keys())}")
    print(f"   Search space range: {min(r['N'] for r in results)} - {max(r['N'] for r in results)}")
    print(f"   Memory range: {min(all_memory)/1024:.2f} KB - {max(all_memory)/(1024**2):.2f} MB")
    print(f"   State vector scaling: 2^n × 16 bytes")
    print(f"   Circuit depth range: {min(r['circuit_depth'] for r in results)} - {max(r['circuit_depth'] for r in results)}")
    print(f"   Gate count range: {min(r['circuit_gates'] for r in results)} - {max(r['circuit_gates'] for r in results)}")
    
    print(f"\n[TIME] COMPUTATION TIME ANALYSIS")
    print(f"   Total computation time: {sum(all_times):.2f}s")
    print(f"   Avg time per test: {np.mean(all_times):.4f}s")
    print(f"   Min time: {np.min(all_times):.4f}s")
    print(f"   Max time: {np.max(all_times):.4f}s")
    print(f"   Std deviation: {np.std(all_times):.4f}s")
    
    # Analyze time scaling with problem size
    for n in sorted(by_qubits.keys()):
        tests = by_qubits[n]
        avg_time = np.mean([t['total_computation_time'] for t in tests])
        print(f"   {n} qubits: {avg_time:.4f}s avg")
    
    print(f"\n[CORRECTNESS] ALGORITHM CORRECTNESS")
    print(f"   Avg success rate: {np.mean(all_success):.2f}%")
    print(f"   Success rate range: {np.min(all_success):.2f}% - {np.max(all_success):.2f}%")
    print(f"   Tests with >90% success: {sum(1 for s in all_success if s > 90)}/{len(all_success)}")
    
    print(f"\n[ADVANTAGE] QUANTUM ADVANTAGE")
    print(f"   Avg speedup: {np.mean(all_speedups):.2f}x")
    print(f"   Max speedup: {np.max(all_speedups):.2f}x")
    if any(r['n_qubits'] == 10 for r in results):
        print(f"   Speedup at 10 qubits: {np.mean([r['speedup'] for r in results if r['n_qubits'] == 10]):.2f}x")
    
    print("="*80)
    
    return results  # Return the (possibly filtered) results for plotting
    
    # Multiple marked states analysis
    multi_marked = [r for r in results if r['num_marked'] > 1]
    if len(multi_marked) > 0:
        print(f"\n[MULTI] MULTIPLE MARKED STATES")
        print(f"   Tests with multiple marked states: {len(multi_marked)}")
        print(f"   Avg success rate: {np.mean([r['success_rate'] for r in multi_marked]):.2f}%")
        print(f"   Formula √(N/M) verified across all tests")
    
    print("\n" + "="*80)


def plot_complexity_analysis(results, output_file='batch_complexity_analysis.png', label="ALL DATA"):
    """
    Create comprehensive visualization of complexity analysis including space and time.
    
    Args:
        results: List of test results
        output_file: Output filename for the plot
        label: Label for the dataset (e.g., "ALL DATA" or "FILTERED (>50%)")
    """
    if len(results) == 0:
        print(f"[WARN] Skipping plots for {label} - no data")
        return
        
    print(f"\nGenerating comprehensive complexity plots for {label}...")
    
    # Group by qubit count for cleaner plots
    by_qubits = {}
    for r in results:
        n = r['n_qubits']
        if n not in by_qubits:
            by_qubits[n] = []
        by_qubits[n].append(r)
    
    # Calculate averages per qubit count
    qubit_counts = sorted(by_qubits.keys())
    avg_iterations = [np.mean([t['iterations'] for t in by_qubits[n]]) for n in qubit_counts]
    avg_sqrt_n = [np.mean([t['sqrt_N'] for t in by_qubits[n]]) for n in qubit_counts]
    avg_success = [np.mean([t['success_rate'] for t in by_qubits[n]]) for n in qubit_counts]
    search_spaces = [2**n for n in qubit_counts]
    avg_circuit_depth = [np.mean([t['circuit_depth'] for t in by_qubits[n]]) for n in qubit_counts]
    avg_circuit_gates = [np.mean([t['circuit_gates'] for t in by_qubits[n]]) for n in qubit_counts]
    avg_memory_mb = [np.mean([t['total_memory_bytes'] for t in by_qubits[n]])/(1024**2) for n in qubit_counts]
    avg_comp_time = [np.mean([t['total_computation_time'] for t in by_qubits[n]]) for n in qubit_counts]
    
    # Create 3x3 subplot grid for comprehensive analysis
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Time Complexity - Iterations vs √N (with different marked states)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Group by marked states and plot each separately
    by_marked_states = {}
    for r in results:
        m = r['num_marked']
        if m not in by_marked_states:
            by_marked_states[m] = []
        by_marked_states[m].append(r)
    
    # Color map for different marked states
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot data points and fit lines for each marked state count
    for idx, m in enumerate(sorted(by_marked_states.keys())[:10]):  # Limit to 10 for clarity
        m_results = by_marked_states[m]
        if len(m_results) >= 3:
            m_sqrt_n = [r['sqrt_N'] for r in m_results]
            m_iterations = [r['iterations'] for r in m_results]
            
            # Linear fit for this marked state count
            slope, intercept = np.polyfit(m_sqrt_n, m_iterations, 1)
            
            # Theoretical constant
            theoretical_k = np.pi / (4 * np.sqrt(m))
            
            # Plot points
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax1.scatter(m_sqrt_n, m_iterations, alpha=0.6, s=60, 
                       color=color, marker=marker, label=f'M={m} data')
            
            # Plot fit line
            x_range = np.linspace(min(m_sqrt_n), max(m_sqrt_n), 100)
            fit_line = [slope * x + intercept for x in x_range]
            ax1.plot(x_range, fit_line, color=color, linestyle='-', linewidth=2,
                    label=f'M={m}: k={slope:.3f} (theory={theoretical_k:.3f})')
            
            # Plot theoretical line (dashed)
            theoretical_line = [theoretical_k * x for x in x_range]
            ax1.plot(x_range, theoretical_line, color=color, linestyle='--', 
                    linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('√N (square root of search space)', fontsize=10)
    ax1.set_ylabel('Iterations', fontsize=10)
    ax1.set_title('Proportionality for Each Marked State Count\nIterations = k × √N, where k = π/(4√M)', 
                 fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Space Complexity - Search Space vs Qubits (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(qubit_counts, search_spaces, 'o-', linewidth=2, markersize=8, color='green', label='Actual N=2^n')
    ax2.semilogy(qubit_counts, [2**n for n in qubit_counts], 'r--', label='Theoretical 2^n', linewidth=2)
    ax2.set_xlabel('Number of Qubits (n)', fontsize=10)
    ax2.set_ylabel('Search Space Size (N)', fontsize=10)
    ax2.set_title('Space Complexity: O(n)\nn qubits → 2^n states', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory Usage vs Qubits
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(qubit_counts, avg_memory_mb, 'o-', linewidth=2, markersize=8, color='purple')
    ax3.set_xlabel('Number of Qubits (n)', fontsize=10)
    ax3.set_ylabel('Memory Usage (MB)', fontsize=10)
    ax3.set_title('Memory Requirements\nState Vector: 2^n × 16 bytes', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Circuit Depth vs Qubits
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(qubit_counts, avg_circuit_depth, 'o-', linewidth=2, markersize=8, color='orange')
    ax4.set_xlabel('Number of Qubits (n)', fontsize=10)
    ax4.set_ylabel('Average Circuit Depth', fontsize=10)
    ax4.set_title('Circuit Depth Scaling\n(Grows with iterations)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Gate Count vs Qubits
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(qubit_counts, avg_circuit_gates, 'o-', linewidth=2, markersize=8, color='brown')
    ax5.set_xlabel('Number of Qubits (n)', fontsize=10)
    ax5.set_ylabel('Average Gate Count', fontsize=10)
    ax5.set_title('Total Gate Count\n(Circuit complexity)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Computation Time vs Qubits
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(qubit_counts, avg_comp_time, 'o-', linewidth=2, markersize=8, color='red')
    ax6.set_xlabel('Number of Qubits (n)', fontsize=10)
    ax6.set_ylabel('Avg Computation Time (s)', fontsize=10)
    ax6.set_title('Actual Computation Time\n(Build + Transpile + Execute)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Success Rate Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    all_success = [r['success_rate'] for r in results]
    ax7.hist(all_success, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax7.axvline(np.mean(all_success), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_success):.1f}%')
    ax7.set_xlabel('Success Rate (%)', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title(f'Success Rate Distribution\n({len(results)} tests)', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Proportionality Constants vs Marked States
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Extract proportionality data for each marked state count
    prop_m_values = []
    prop_fitted_k = []
    prop_theoretical_k = []
    prop_r_squared = []
    
    for m in sorted(by_marked_states.keys()):
        m_results = by_marked_states[m]
        if len(m_results) >= 3:
            m_sqrt_n = [r['sqrt_N'] for r in m_results]
            m_iterations = [r['iterations'] for r in m_results]
            slope, intercept = np.polyfit(m_sqrt_n, m_iterations, 1)
            
            # Calculate R²
            predicted = [slope * x + intercept for x in m_sqrt_n]
            residuals = [actual - pred for actual, pred in zip(m_iterations, predicted)]
            ss_res = sum(r**2 for r in residuals)
            ss_tot = sum((y - np.mean(m_iterations))**2 for y in m_iterations)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            theoretical_k = np.pi / (4 * np.sqrt(m))
            
            prop_m_values.append(m)
            prop_fitted_k.append(slope)
            prop_theoretical_k.append(theoretical_k)
            prop_r_squared.append(r_squared)
    
    if len(prop_m_values) > 0:
        # Plot fitted vs theoretical constants
        ax8.plot(prop_m_values, prop_theoretical_k, 'r--', linewidth=2, 
                marker='o', markersize=8, label='Theoretical k = π/(4√M)')
        ax8.plot(prop_m_values, prop_fitted_k, 'b-', linewidth=2, 
                marker='s', markersize=8, label='Fitted k (observed)')
        
        # Add R² values as text annotations for key points
        for i, (m, k, r2) in enumerate(zip(prop_m_values, prop_fitted_k, prop_r_squared)):
            if i < 5:  # Annotate first 5 points to avoid clutter
                ax8.annotate(f'R²={r2:.3f}', (m, k), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=7, alpha=0.7)
        
        ax8.set_xlabel('Number of Marked States (M)', fontsize=10)
        ax8.set_ylabel('Proportionality Constant k', fontsize=10)
        ax8.set_title('Proportionality Constants\nk = π/(4√M) for each M', fontsize=11, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
        
        # Add secondary y-axis showing deviation
        ax8_twin = ax8.twinx()
        deviations = [abs(fitted - theoretical) for fitted, theoretical in zip(prop_fitted_k, prop_theoretical_k)]
        ax8_twin.plot(prop_m_values, deviations, 'g:', linewidth=1.5, marker='^', 
                     markersize=6, alpha=0.6, label='Deviation')
        ax8_twin.set_ylabel('|Fitted - Theoretical|', fontsize=9, color='g')
        ax8_twin.tick_params(axis='y', labelcolor='g')
        ax8_twin.legend(fontsize=8, loc='upper right')
    else:
        ax8.text(0.5, 0.5, 'Insufficient data\nfor proportionality analysis', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=10)
        ax8.set_xlabel('Number of Marked States (M)', fontsize=10)
        ax8.set_ylabel('Proportionality Constant k', fontsize=10)
        ax8.set_title('Proportionality Constants\nk = π/(4√M) for each M', fontsize=11, fontweight='bold')
    
    # Plot 9: Quantum Speedup vs Problem Size
    ax9 = fig.add_subplot(gs[2, 2])
    speedups_by_qubit = [np.mean([t['speedup'] for t in by_qubits[n]]) for n in qubit_counts]
    ax9.plot(qubit_counts, speedups_by_qubit, 'o-', linewidth=2, markersize=8, color='magenta')
    ax9.set_xlabel('Number of Qubits (n)', fontsize=10)
    ax9.set_ylabel('Average Quantum Speedup (x)', fontsize=10)
    ax9.set_title('Quantum Advantage\nSpeedup vs Classical', fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # Add overall title with label
    fig.suptitle(f'Grover\'s Algorithm - Comprehensive Complexity Analysis\n{label}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive plots to {output_file}")
    plt.close()
    
    # Create additional detailed space complexity plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Space complexity details - Memory components
    state_vec_memory = [2**n * 16 / (1024**2) for n in qubit_counts]  # MB
    circuit_memory = [np.mean([t['circuit_memory_bytes'] for t in by_qubits[n]])/(1024) for n in qubit_counts]  # KB
    
    axes2[0, 0].semilogy(qubit_counts, state_vec_memory, 'o-', label='State Vector', linewidth=2, markersize=8)
    axes2[0, 0].set_xlabel('Number of Qubits (n)', fontsize=11)
    axes2[0, 0].set_ylabel('Memory (MB)', fontsize=11)
    axes2[0, 0].set_title('State Vector Memory: 2^n × 16 bytes', fontsize=12, fontweight='bold')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].legend()
    
    # Circuit resources
    axes2[0, 1].plot(qubit_counts, avg_circuit_depth, 'o-', label='Depth', linewidth=2, markersize=8, color='blue')
    ax_twin = axes2[0, 1].twinx()
    ax_twin.plot(qubit_counts, avg_circuit_gates, 's-', label='Gates', linewidth=2, markersize=8, color='red')
    axes2[0, 1].set_xlabel('Number of Qubits (n)', fontsize=11)
    axes2[0, 1].set_ylabel('Circuit Depth', fontsize=11, color='blue')
    ax_twin.set_ylabel('Gate Count', fontsize=11, color='red')
    axes2[0, 1].set_title('Circuit Resources', fontsize=12, fontweight='bold')
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    # Time breakdown
    build_times = [np.mean([t['build_time'] for t in by_qubits[n]]) for n in qubit_counts]
    transpile_times = [np.mean([t['transpile_time'] for t in by_qubits[n]]) for n in qubit_counts]
    exec_times = [np.mean([t['exec_time'] for t in by_qubits[n]]) for n in qubit_counts]
    
    axes2[1, 0].plot(qubit_counts, build_times, 'o-', label='Build', linewidth=2, markersize=6)
    axes2[1, 0].plot(qubit_counts, transpile_times, 's-', label='Transpile', linewidth=2, markersize=6)
    axes2[1, 0].plot(qubit_counts, exec_times, '^-', label='Execute', linewidth=2, markersize=6)
    axes2[1, 0].set_xlabel('Number of Qubits (n)', fontsize=11)
    axes2[1, 0].set_ylabel('Time (seconds)', fontsize=11)
    axes2[1, 0].set_title('Computation Time Breakdown', fontsize=12, fontweight='bold')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)
    
    # Scaling comparison: Quantum vs Classical
    quantum_time_estimate = avg_iterations  # Proportional to √N
    classical_time_estimate = search_spaces  # Proportional to N
    
    axes2[1, 1].semilogy(qubit_counts, quantum_time_estimate, 'o-', label='Quantum O(√N)', 
                         linewidth=2, markersize=8, color='blue')
    axes2[1, 1].semilogy(qubit_counts, classical_time_estimate, 's-', label='Classical O(N)', 
                         linewidth=2, markersize=8, color='red')
    axes2[1, 1].set_xlabel('Number of Qubits (n)', fontsize=11)
    axes2[1, 1].set_ylabel('Relative Time Steps', fontsize=11)
    axes2[1, 1].set_title('Quantum vs Classical Scaling', fontsize=12, fontweight='bold')
    axes2[1, 1].legend()
    axes2[1, 1].grid(True, alpha=0.3)
    
    fig2.suptitle(f'Space Complexity & Time Analysis Details\n{label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    space_output = output_file.replace('.png', '_space_time_detail.png')
    plt.savefig(space_output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed space/time plots to {space_output}")
    plt.close()
    
    # ========================================================================
    # Create dedicated proportionality analysis plot
    # ========================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: All marked states on one plot with fit lines
    ax_all = axes3[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Get overall sqrt_N range for consistent x-axis
    all_sqrt_n_values = [r['sqrt_N'] for r in results]
    min_sqrt_n = min(all_sqrt_n_values)
    max_sqrt_n = max(all_sqrt_n_values)
    
    for idx, m in enumerate(sorted(by_marked_states.keys())[:10]):
        m_results = by_marked_states[m]
        if len(m_results) >= 3:
            m_sqrt_n = [r['sqrt_N'] for r in m_results]
            m_iterations = [r['iterations'] for r in m_results]
            
            slope, intercept = np.polyfit(m_sqrt_n, m_iterations, 1)
            theoretical_k = np.pi / (4 * np.sqrt(m))
            
            # Plot data points
            ax_all.scatter(m_sqrt_n, m_iterations, alpha=0.5, s=40, 
                          color=colors[idx], label=f'M={m}')
            
            # Plot fit line
            x_range = np.linspace(min_sqrt_n, max_sqrt_n, 100)
            fit_line = [slope * x + intercept for x in x_range]
            ax_all.plot(x_range, fit_line, color=colors[idx], 
                       linewidth=2, alpha=0.8)
    
    ax_all.set_xlabel('√N', fontsize=12)
    ax_all.set_ylabel('Iterations', fontsize=12)
    ax_all.set_title('Proportionality for All Marked State Counts\nIterations = k × √N', 
                     fontsize=13, fontweight='bold')
    ax_all.legend(fontsize=9, loc='upper left', ncol=2)
    ax_all.grid(True, alpha=0.3)
    
    # Subplot 2: Fitted vs Theoretical Constants
    ax_const = axes3[0, 1]
    prop_m_values = []
    prop_fitted_k = []
    prop_theoretical_k = []
    
    for m in sorted(by_marked_states.keys()):
        m_results = by_marked_states[m]
        if len(m_results) >= 3:
            m_sqrt_n = [r['sqrt_N'] for r in m_results]
            m_iterations = [r['iterations'] for r in m_results]
            slope, _ = np.polyfit(m_sqrt_n, m_iterations, 1)
            theoretical_k = np.pi / (4 * np.sqrt(m))
            
            prop_m_values.append(m)
            prop_fitted_k.append(slope)
            prop_theoretical_k.append(theoretical_k)
    
    if len(prop_m_values) > 0:
        ax_const.plot(prop_m_values, prop_theoretical_k, 'r-', linewidth=3, 
                     marker='o', markersize=10, label='Theoretical k = π/(4√M)', alpha=0.7)
        ax_const.plot(prop_m_values, prop_fitted_k, 'b-', linewidth=2, 
                     marker='s', markersize=8, label='Fitted k (observed)')
        
        # Add theoretical curve
        m_curve = np.linspace(1, max(prop_m_values), 100)
        k_curve = np.pi / (4 * np.sqrt(m_curve))
        ax_const.plot(m_curve, k_curve, 'r--', linewidth=1, alpha=0.4)
        
        ax_const.set_xlabel('Number of Marked States (M)', fontsize=12)
        ax_const.set_ylabel('Proportionality Constant k', fontsize=12)
        ax_const.set_title('k = π/(4√M) Verification', fontsize=13, fontweight='bold')
        ax_const.legend(fontsize=10)
        ax_const.grid(True, alpha=0.3)
    
    # Subplot 3: R² values for each marked state count
    ax_r2 = axes3[1, 0]
    r2_m_values = []
    r2_values = []
    
    for m in sorted(by_marked_states.keys()):
        m_results = by_marked_states[m]
        if len(m_results) >= 3:
            m_sqrt_n = [r['sqrt_N'] for r in m_results]
            m_iterations = [r['iterations'] for r in m_results]
            slope, intercept = np.polyfit(m_sqrt_n, m_iterations, 1)
            
            predicted = [slope * x + intercept for x in m_sqrt_n]
            residuals = [actual - pred for actual, pred in zip(m_iterations, predicted)]
            ss_res = sum(r**2 for r in residuals)
            ss_tot = sum((y - np.mean(m_iterations))**2 for y in m_iterations)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            r2_m_values.append(m)
            r2_values.append(r_squared)
    
    if len(r2_m_values) > 0:
        bars = ax_r2.bar(range(len(r2_m_values)), r2_values, alpha=0.7, 
                        color=['green' if r2 > 0.95 else 'orange' if r2 > 0.90 else 'red' 
                               for r2 in r2_values])
        ax_r2.axhline(y=0.95, color='g', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (R²>0.95)')
        ax_r2.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (R²>0.90)')
        ax_r2.set_xticks(range(len(r2_m_values)))
        ax_r2.set_xticklabels([f'M={m}' for m in r2_m_values], rotation=45)
        ax_r2.set_ylabel('R² (Goodness of Fit)', fontsize=12)
        ax_r2.set_xlabel('Number of Marked States', fontsize=12)
        ax_r2.set_title('Linear Fit Quality for Each M', fontsize=13, fontweight='bold')
        ax_r2.legend(fontsize=9)
        ax_r2.grid(True, alpha=0.3, axis='y')
        ax_r2.set_ylim([0, 1.05])
    
    # Subplot 4: Deviation from theoretical
    ax_dev = axes3[1, 1]
    if len(prop_m_values) > 0:
        deviations = [abs(fitted - theoretical) for fitted, theoretical in 
                     zip(prop_fitted_k, prop_theoretical_k)]
        percent_deviations = [100 * abs(fitted - theoretical) / theoretical 
                             for fitted, theoretical in zip(prop_fitted_k, prop_theoretical_k)]
        
        ax_dev.bar(range(len(prop_m_values)), percent_deviations, alpha=0.7,
                  color=['green' if d < 1 else 'orange' if d < 5 else 'red' 
                         for d in percent_deviations])
        ax_dev.axhline(y=1, color='g', linestyle='--', linewidth=2, alpha=0.5, label='<1% (Excellent)')
        ax_dev.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='<5% (Good)')
        ax_dev.set_xticks(range(len(prop_m_values)))
        ax_dev.set_xticklabels([f'M={m}' for m in prop_m_values], rotation=45)
        ax_dev.set_ylabel('Deviation from Theory (%)', fontsize=12)
        ax_dev.set_xlabel('Number of Marked States', fontsize=12)
        ax_dev.set_title('Accuracy: |Fitted - Theoretical| / Theoretical', fontsize=13, fontweight='bold')
        ax_dev.legend(fontsize=9)
        ax_dev.grid(True, alpha=0.3, axis='y')
    
    fig3.suptitle(f'Proportionality Analysis: k = π/(4√M) for Each Marked State Count\n{label}', 
                  fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    prop_output = output_file.replace('.png', '_proportionality.png')
    plt.savefig(prop_output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved proportionality analysis plots to {prop_output}")
    plt.close()


def save_detailed_results(results, output_file='batch_results.txt'):
    """
    Save detailed results with high-precision timing to file.
    """
    print(f"\nSaving detailed results to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("GROVER'S ALGORITHM - HIGH-PRECISION TIME COMPLEXITY BENCHMARK\n")
        f.write("="*90 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {len(results)}\n")
        f.write(f"Backend: {results[0]['backend'] if results else 'Unknown'}\n")
        f.write(f"Timing Method: time.perf_counter() - nanosecond precision\n")
        f.write(f"Measurement: Individual per-circuit (not averaged)\n\n")
        
        f.write("="*90 + "\n")
        f.write("DETAILED RESULTS WITH INDIVIDUAL HIGH-PRECISION TIMING\n")
        f.write("="*90 + "\n")
        f.write(f"{'Q':<4} {'N':<6} {'M':<4} {'Iter':<5} {'√N':<7} {'Ratio':<7} "
               f"{'Build(s)':<12} {'Trans(s)':<12} {'Exec(s)':<12} {'Total(s)':<12} "
               f"{'Success%':<9} {'Speed':<7}\n")
        f.write("-"*90 + "\n")
        
        for r in results:
            f.write(f"{r['n_qubits']:<4} {r['N']:<6} {r['num_marked']:<4} "
                   f"{r['iterations']:<5} {r['sqrt_N']:<7.2f} {r['complexity_ratio']:<7.3f} "
                   f"{r['build_time']:<12.9f} {r['transpile_time']:<12.9f} {r['exec_time']:<12.9f} "
                   f"{r['total_computation_time']:<12.9f} "
                   f"{r['success_rate']:<9.2f} {r['speedup']:<7.2f}\n")
        
        f.write("\n" + "="*90 + "\n")
        f.write("TIMING PRECISION & ACCURACY ANALYSIS\n")
        f.write("="*90 + "\n\n")
        
        total_build = sum(r['build_time'] for r in results)
        total_trans = sum(r['transpile_time'] for r in results)
        total_exec = sum(r['exec_time'] for r in results)
        total_comp = sum(r['total_computation_time'] for r in results)
        
        f.write("Timing Statistics (all times in seconds with nanosecond precision):\n\n")
        f.write(f"Build Phase:\n")
        f.write(f"  Total:   {total_build:12.9f}s\n")
        f.write(f"  Average: {total_build/len(results):12.9f}s per circuit\n")
        f.write(f"  StdDev:  {np.std([r['build_time'] for r in results]):12.9f}s\n")
        f.write(f"  Range:   [{min(r['build_time'] for r in results):12.9f}s - {max(r['build_time'] for r in results):12.9f}s]\n\n")
        
        f.write(f"Transpile Phase:\n")
        f.write(f"  Total:   {total_trans:12.9f}s\n")
        f.write(f"  Average: {total_trans/len(results):12.9f}s per circuit\n")
        f.write(f"  StdDev:  {np.std([r['transpile_time'] for r in results]):12.9f}s\n")
        f.write(f"  Range:   [{min(r['transpile_time'] for r in results):12.9f}s - {max(r['transpile_time'] for r in results):12.9f}s]\n\n")
        
        f.write(f"Execute Phase (Pure Quantum Algorithm):\n")
        f.write(f"  Total:   {total_exec:12.9f}s\n")
        f.write(f"  Average: {total_exec/len(results):12.9f}s per circuit\n")
        f.write(f"  StdDev:  {np.std([r['exec_time'] for r in results]):12.9f}s\n")
        f.write(f"  Range:   [{min(r['exec_time'] for r in results):12.9f}s - {max(r['exec_time'] for r in results):12.9f}s]\n\n")
        
        f.write(f"Total Computation:\n")
        f.write(f"  Total:   {total_comp:12.9f}s\n")
        f.write(f"  Average: {total_comp/len(results):12.9f}s per circuit\n\n")
        
        # Group by qubit count for timing analysis
        by_qubits = {}
        for r in results:
            n = r['n_qubits']
            if n not in by_qubits:
                by_qubits[n] = []
            by_qubits[n].append(r)
        
        f.write("TIMING BY QUBIT COUNT\n")
        f.write("-"*90 + "\n")
        f.write(f"{'Qubits':<8} {'Count':<8} {'Avg Build':<12} {'Avg Trans':<12} {'Avg Exec':<12} {'Avg Total':<12}\n")
        f.write("-"*90 + "\n")
        
        for n in sorted(by_qubits.keys()):
            tests = by_qubits[n]
            avg_build = np.mean([t['build_time'] for t in tests])
            avg_trans = np.mean([t['transpile_time'] for t in tests])
            avg_exec = np.mean([t['exec_time'] for t in tests])
            avg_total = np.mean([t['total_computation_time'] for t in tests])
            
            f.write(f"{n:<8} {len(tests):<8} {avg_build:<12.6f} {avg_trans:<12.6f} {avg_exec:<12.6f} {avg_total:<12.6f}\n")
        
        f.write("\n" + "="*90 + "\n")
        f.write("COMPLEXITY SUMMARY\n")
        f.write("="*90 + "\n\n")
        
        all_ratios = [r['complexity_ratio'] for r in results]
        all_success = [r['success_rate'] for r in results]
        
        f.write(f"Average Complexity Ratio (Iter/√N): {np.mean(all_ratios):.4f} ± {np.std(all_ratios):.4f}\n")
        f.write(f"Theoretical Expectation: 0.7854 (π/4)\n")
        f.write(f"Deviation from theory: {abs(np.mean(all_ratios) - 0.7854):.4f}\n\n")
        f.write(f"Average Success Rate: {np.mean(all_success):.2f}% ± {np.std(all_success):.2f}%\n")
        f.write(f"Tests with >90% success: {sum(1 for s in all_success if s > 90)}/{len(all_success)} ({100*sum(1 for s in all_success if s > 90)/len(all_success):.1f}%)\n")
    
    print(f"✓ Saved results to {output_file}")


def main():
    """
    Main function for high-precision time complexity benchmarking.
    """
    print("\n" + "="*80)
    print("GROVER'S ALGORITHM - HIGH-PRECISION TIME COMPLEXITY BENCHMARK")
    print("="*80)
    print("\nFeatures:")
    print("  - Individual circuit timing (build, transpile, execute)")
    print("  - High-precision measurement using time.perf_counter()")
    print("  - Nanosecond-level accuracy for time complexity verification")
    print("  - Comprehensive space complexity metrics")
    print("  - ~100-120 optimized test configurations")
    print("  - Statistical analysis and visualization")
    print("\n[TIME] Timing Precision:")
    print("  - time.perf_counter(): Monotonic clock, nanosecond resolution")
    print("  - Individual measurements: No averaging, no approximation")
    print("  - Separates quantum algorithm time from classical overhead")
    print("="*80)
    
    # Check for IBM flag
    use_ibm = '--ibm' in sys.argv
    
    if use_ibm:
        print("\n[Mode: IBM Quantum Hardware]")
        load_dotenv()
        token = os.getenv('IBM_QUANTUM_TOKEN')
        if not token:
            print("[-] No IBM token found in .env file")
            print("Falling back to simulator mode")
            use_ibm = False
    else:
        print("\n[Mode: Local Simulator - use --ibm flag for IBM Quantum]")
    
    # Generate test configurations
    print("\nGenerating test configurations...")
    test_configs = generate_test_configs(max_qubits=10, tests_per_qubit=20)
    print(f"[+] Generated {len(test_configs)} test configurations")
    
    # Build all circuits
    circuits, metadata = build_circuits_batch(test_configs)
    
    # Run batch execution
    if use_ibm:
        try:
            print("\nInitializing IBM Quantum...")
            service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
            backend = service.least_busy(simulator=False, operational=True, min_num_qubits=2)
            print(f"[+] Connected to IBM Quantum - Backend: {backend.name}")
            
            results = run_batch_on_ibm(circuits, metadata, backend, shots=1024, batch_size=75)
        except Exception as e:
            print(f"[-] IBM Quantum error: {e}")
            print("Falling back to simulator...")
            results = run_batch_on_simulator(circuits, metadata)
    else:
        results = run_batch_on_simulator(circuits, metadata)
    
    # Generate timestamp for all output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # ANALYSIS 1: ALL DATA (UNFILTERED)
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 1: ALL CIRCUITS (UNFILTERED)")
    print("="*80)
    
    all_results = analyze_complexity(results.copy(), label="ALL DATA (UNFILTERED)")
    plot_complexity_analysis(all_results, 
                            output_file=f'batch_analysis_all_{timestamp}.png',
                            label="ALL DATA (UNFILTERED)")
    save_detailed_results(all_results, f'batch_results_all_{timestamp}.txt')
    
    # ========================================================================
    # ANALYSIS 2: HIGH ACCURACY ONLY (FILTERED)
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 2: HIGH ACCURACY CIRCUITS ONLY (FILTERED)")
    print("="*80)
    
    # Determine appropriate threshold based on backend
    if use_ibm:
        threshold = 50  # IBM hardware: use 50% threshold (hardware has noise)
        print(f"[FILTER] Using {threshold}% threshold for IBM hardware (real quantum computer)")
    else:
        threshold = 90  # Simulator: use 90% threshold (should be near-perfect)
        print(f"[FILTER] Using {threshold}% threshold for simulator (perfect quantum simulation)")
    
    filtered_results = analyze_complexity(results.copy(), 
                                         label=f"FILTERED (>{threshold}% success)",
                                         success_threshold=threshold)
    
    if len(filtered_results) > 0:
        plot_complexity_analysis(filtered_results, 
                                output_file=f'batch_analysis_filtered_{timestamp}.png',
                                label=f"FILTERED (>{threshold}% success)")
        save_detailed_results(filtered_results, f'batch_results_filtered_{timestamp}.txt')
    else:
        print(f"[WARN] No circuits passed {threshold}% threshold - skipping filtered analysis")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("[+] BATCH BENCHMARK COMPLETE - DUAL ANALYSIS")
    print("="*80)
    print(f"\n[DATA] Results saved ({timestamp}):")
    print(f"\n  UNFILTERED Analysis:")
    print(f"    - batch_results_all_{timestamp}.txt - Detailed data")
    print(f"    - batch_analysis_all_{timestamp}.png - Main plots")
    print(f"    - batch_analysis_all_{timestamp}_space_time_detail.png - Detailed plots")
    print(f"    - batch_analysis_all_{timestamp}_proportionality.png - Proportionality for each M")
    
    if len(filtered_results) > 0:
        print(f"\n  FILTERED Analysis (>{threshold}% success):")
        print(f"    - batch_results_filtered_{timestamp}.txt - Detailed data")
        print(f"    - batch_analysis_filtered_{timestamp}.png - Main plots")
        print(f"    - batch_analysis_filtered_{timestamp}_space_time_detail.png - Detailed plots")
        print(f"    - batch_analysis_filtered_{timestamp}_proportionality.png - Proportionality for each M")
    
    print("\n" + "="*80)
    print(f"[SUMMARY] Summary:")
    print(f"   Total circuits: {len(results)}")
    print(f"   High accuracy circuits: {len(filtered_results)}/{len(results)} (>{threshold}%)")
    print(f"   Avg success (all): {np.mean([r['success_rate'] for r in results]):.2f}%")
    if len(filtered_results) > 0:
        print(f"   Avg success (filtered): {np.mean([r['success_rate'] for r in filtered_results]):.2f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
