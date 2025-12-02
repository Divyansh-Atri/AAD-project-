#include "qrng.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

void run_test(const std::string& name, AlgorithmType algo, int qubits, int shots, uint64_t seed = 0) {
    std::cout << "\n=== Testing " << name << " ===\n";
    
    QRNGConfig config;
    config.num_qubits = qubits;
    config.num_shots = shots;
    config.seed = seed;
    config.algorithm = algo;
    
    QRNG qrng(config);
    auto result = qrng.generate();
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Bits generated: " << result.random_bits.size() << "\n";
    std::cout << "Time taken: " << result.generation_time_ms << " ms\n";
    std::cout << "0s/1s ratio: " << result.zeros << "/" << result.ones 
              << " (" << (result.zeros * 100.0 / result.random_bits.size()) << "% / " 
              << (result.ones * 100.0 / result.random_bits.size()) << "%)\n";
    std::cout << "Shannon entropy: " << result.shannon_entropy << " bits/bit\n";
    std::cout << "Min entropy: " << result.min_entropy << " bits/bit\n";
    std::cout << "Chi-square p-value: " << result.chi_square << "\n";
    std::cout << "Runs test p-value: " << result.runs_pvalue << "\n";
    std::cout << "All tests passed: " << (result.stats.all_tests_passed ? "YES" : "NO") << "\n";
}

int main(int argc, char* argv[]) {
    const int qubits = (argc > 1) ? std::stoi(argv[1]) : 1;
    const int shots = (argc > 2) ? std::stoi(argv[2]) : 10000;
    const uint64_t seed = (argc > 3) ? std::stoull(argv[3]) : 42;
    
    std::cout << "Comparing QRNG algorithms\n";
    std::cout << "Qubits: " << qubits << "\n";
    std::cout << "Shots: " << shots << "\n";
    std::cout << "Seed: " << seed << "\n\n";
    
    // Test each algorithm
    run_test("Mersenne Twister", AlgorithmType::MERSENNE_TWISTER, qubits, shots, seed);
    run_test("Xoshiro256**", AlgorithmType::XOSHIRO, qubits, shots, seed);
    run_test("PCG", AlgorithmType::PCG, qubits, shots, seed);
    run_test("Simulated Quantum", AlgorithmType::QUANTUM_SIMULATED, qubits, shots, seed);
    run_test("REAL IBM QUANTUM", AlgorithmType::TRUE_IBM_QUANTUM, qubits, shots, seed);
    return 0;
}
