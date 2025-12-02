#include "qrng.h"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    try {
        // Default configuration
        QRNGConfig config;
        config.num_qubits = 1;
        config.num_shots = 1000;
        config.seed = 42;  // Fixed seed for reproducibility

        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--qubits" && i + 1 < argc) {
                config.num_qubits = std::stoi(argv[++i]);
            } else if (arg == "--shots" && i + 1 < argc) {
                config.num_shots = std::stoi(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                config.seed = std::stoull(argv[++i]);
            } else if (arg == "--algorithm" && i + 1 < argc) {
                std::string algo = argv[++i];
                if (algo == "MERSENNE_TWISTER") config.algorithm = AlgorithmType::MERSENNE_TWISTER;
                else if (algo == "XOSHIRO") config.algorithm = AlgorithmType::XOSHIRO;
                else if (algo == "PCG") config.algorithm = AlgorithmType::PCG;
                else if (algo == "QUANTUM_SIMULATED") config.algorithm = AlgorithmType::QUANTUM_SIMULATED;
                else std::cerr << "Warning: Unknown algorithm " << algo << ", using default.\n";
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [--qubits N] [--shots N] [--seed N] [--algorithm ALGO]\n"
                          << "  --qubits N    Number of qubits (default: 1)\n"
                          << "  --shots N     Number of measurement shots (default: 1000)\n"
                          << "  --seed N      Random seed (default: 42)\n"
                          << "  --algorithm   Algorithm: MERSENNE_TWISTER, XOSHIRO, PCG, QUANTUM_SIMULATED\n"
                          << "  --help        Show this help message\n";
                return 0;
            }
        }

        // Create and run QRNG
        QRNG qrng(config);
        auto result = qrng.generate();

        // Print results
        std::cout << "QRNG Generation Results:" << std::endl;
        std::cout << "------------------------" << std::endl;
        std::cout << "Qubits: " << config.num_qubits << std::endl;
        std::cout << "Shots: " << config.num_shots << std::endl;
        std::cout << "Total bits: " << result.random_bits.size() << std::endl;
        std::cout << "Ones: " << result.ones << " (" 
                  << (result.ones * 100.0 / result.random_bits.size()) << "%)" << std::endl;
        std::cout << "Zeros: " << result.zeros << " (" 
                  << (result.zeros * 100.0 / result.random_bits.size()) << "%)" << std::endl;
        std::cout << "Shannon Entropy: " << result.shannon_entropy << " bits/bit" << std::endl;
        std::cout << "Min Entropy: " << result.min_entropy << " bits/bit" << std::endl;
        std::cout << "Chi-square test p-value: " << result.chi_square << std::endl;
        std::cout << "Runs test p-value: " << result.runs_pvalue << std::endl;
        std::cout << "Generation time: " << result.generation_time_ms << " ms" << std::endl;
        
        // Print first 20 bits as a sample
        std::cout << "\nFirst 20 bits: ";
        for (int i = 0; i < 20 && i < result.random_bits.size(); ++i) {
            std::cout << (int)result.random_bits[i];
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
