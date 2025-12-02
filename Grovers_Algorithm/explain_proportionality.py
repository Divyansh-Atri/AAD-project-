"""
Explain why the proportionality constant isn't matching theoretical value
"""
import numpy as np

print("="*80)
print("WHY YOUR PROPORTIONALITY CONSTANT ISN'T CLOSE TO π/4 = 0.785")
print("="*80)

# Your actual filtered data (from batch_results_filtered_20251202_063954.txt)
# Only tests with 1-2 marked states
data_1_2_marked = [
    (2, 1, 1, 2.00),   # n_qubits, num_marked, iterations, sqrt_N
    (3, 1, 2, 2.83),
    (3, 2, 1, 2.83),
    (4, 1, 3, 4.00),
    (4, 2, 2, 4.00),
    (5, 1, 4, 5.66),
    (5, 2, 3, 5.66),
    (6, 1, 6, 8.00),
    (6, 2, 4, 8.00),
    (7, 1, 8, 11.31),
    (7, 2, 6, 11.31),
    (8, 1, 12, 16.00),
    (8, 2, 8, 16.00),
    (9, 1, 17, 22.63),
    (9, 2, 12, 22.63),
    (10, 1, 25, 32.00),
    (10, 2, 17, 32.00),
]

print("\n[DATA] YOUR DATA (1-2 marked states only):")
print("-"*80)
print(f"{'Qubits':<8} {'Marked':<8} {'Iter':<8} {'√N':<8} {'Ratio':<8} {'Theoretical':<12}")
print("-"*80)

for n_qubits, num_marked, iterations, sqrt_n in data_1_2_marked:
    ratio = iterations / sqrt_n
    theoretical = (np.pi / 4) * np.sqrt(2**n_qubits / num_marked) / sqrt_n
    print(f"{n_qubits:<8} {num_marked:<8} {iterations:<8} {sqrt_n:<8.2f} {ratio:<8.3f} {theoretical:<12.3f}")

# Calculate proportionality
sqrt_n_vals = [d[3] for d in data_1_2_marked]
iter_vals = [d[2] for d in data_1_2_marked]

slope, intercept = np.polyfit(sqrt_n_vals, iter_vals, 1)
predicted = [slope * x + intercept for x in sqrt_n_vals]
residuals = [actual - pred for actual, pred in zip(iter_vals, predicted)]
r_squared = 1 - (sum(r**2 for r in residuals) / sum((y - np.mean(iter_vals))**2 for y in iter_vals))

print(f"\n[FIT] LINEAR FIT RESULTS:")
print(f"   Best fit: Iterations = {slope:.3f} × √N + {intercept:.3f}")
print(f"   R² = {r_squared:.4f}")
print(f"   Proportionality constant: {slope:.3f}")
print(f"   Theoretical constant: 0.785 (π/4)")
print(f"   Deviation: {abs(slope - 0.785):.3f}")

print(f"\n[ANALYSIS] ROOT CAUSE ANALYSIS:")
print("="*80)

# Calculate what each marked state scenario contributes
marked_1 = [d for d in data_1_2_marked if d[1] == 1]
marked_2 = [d for d in data_1_2_marked if d[1] == 2]

if marked_1:
    avg_ratio_1 = np.mean([d[2]/d[3] for d in marked_1])
    print(f"\n[1] Tests with 1 marked state:")
    print(f"   Count: {len(marked_1)}")
    print(f"   Average ratio: {avg_ratio_1:.3f}")
    print(f"   Theoretical: 0.785 (π/4 × √(N/1)/√N = π/4)")
    print(f"   Match: {'[+] GOOD' if abs(avg_ratio_1 - 0.785) < 0.1 else '[-] MISMATCH'}")

if marked_2:
    avg_ratio_2 = np.mean([d[2]/d[3] for d in marked_2])
    theoretical_2 = (np.pi/4) * np.sqrt(1/2)  # π/4 × √(N/2)/√N
    print(f"\n[2] Tests with 2 marked states:")
    print(f"   Count: {len(marked_2)}")
    print(f"   Average ratio: {avg_ratio_2:.3f}")
    print(f"   Theoretical: {theoretical_2:.3f} (π/4 × √(N/2)/√N)")
    print(f"   Match: {'[+] GOOD' if abs(avg_ratio_2 - theoretical_2) < 0.1 else '[-] MISMATCH'}")

print(f"\n[WARN] THE PROBLEM:")
print("="*80)
print("When you mix tests with 1 marked AND 2 marked states:")
print(f"  - 1 marked: ratio ~ 0.785")
print(f"  - 2 marked: ratio ~ 0.555")
print(f"  - Combined: ratio ~ {slope:.3f} (weighted average)")
print()
print("The linear fit finds the AVERAGE proportionality across both scenarios,")
print("which is NOT π/4 = 0.785!")

print(f"\n[SOL] SOLUTION:")
print("="*80)
print("For accurate π/4 verification, use ONLY 1 marked state tests:")

marked_1_sqrt = [d[3] for d in marked_1]
marked_1_iter = [d[2] for d in marked_1]
slope_1, intercept_1 = np.polyfit(marked_1_sqrt, marked_1_iter, 1)
predicted_1 = [slope_1 * x + intercept_1 for x in marked_1_sqrt]
residuals_1 = [actual - pred for actual, pred in zip(marked_1_iter, predicted_1)]
r_squared_1 = 1 - (sum(r**2 for r in residuals_1) / sum((y - np.mean(marked_1_iter))**2 for y in marked_1_iter))

print(f"\n   Iterations = {slope_1:.3f} × √N + {intercept_1:.3f}")
print(f"   R² = {r_squared_1:.4f}")
print(f"   Proportionality constant: {slope_1:.3f}")
print(f"   Deviation from π/4: {abs(slope_1 - 0.785):.3f}")
print(f"   {'[+] EXCELLENT MATCH!' if abs(slope_1 - 0.785) < 0.05 else '[WARN] Still some deviation'}")

print("\n" + "="*80)
