#ifndef QRNG_H
#define QRNG_H

#include <vector>
#include <cstdint>
#include <string>


struct QRNGResult {
    std::vector<uint8_t> random_bits;
    double generation_time_ms = 0.0;
    std::string error_message;
    uint64_t ones = 0;
    uint64_t zeros = 0;
    double chi_square = 0.0;
    double runs_pvalue = 0.0;
    double shannon_entropy = 0.0;
    double min_entropy = 0.0;
    struct {
        bool all_tests_passed = false;
    } stats;
};

enum class AlgorithmType {
    MERSENNE_TWISTER,  // Default high-quality PRNG
    XOSHIRO,           // Fast PRNG
    PCG,               // Another good PRNG
    QUANTUM_SIMULATED,  // Simulated quantum measurements
    TRUE_IBM_QUANTUM
};

struct QRNGConfig {
    int num_qubits = 1;
    int num_shots = 1000;
    uint64_t seed = 0; // 0 means use time-based seed
    AlgorithmType algorithm = AlgorithmType::MERSENNE_TWISTER;
};

class QRNG {
public:
    QRNG() = default;
    explicit QRNG(const QRNGConfig& config);
    ~QRNG() = default;

    // Generate random bits
    QRNGResult generate() const;
    
    // Generate random bits with specified parameters
    QRNGResult generate(int qubits, int shots) const;

    // Statistical tests
    double frequency_test(const std::vector<uint8_t>& bits) const;
    double runs_test(const std::vector<uint8_t>& bits) const;
    double chi_square_test(const std::vector<uint8_t>& bits) const;
    double calculate_shannon_entropy(const std::vector<uint8_t>& bits) const;
    double calculate_min_entropy(const std::vector<uint8_t>& bits) const;

private:
    // Helper methods
    std::vector<uint8_t> generate_random_bits(size_t count) const;
    std::vector<uint8_t> generate_pseudo_random_bits(size_t count) const;
    QRNGConfig config_;
};

#endif // QRNG_H
