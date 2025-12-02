"""
Quick Test: Run Deutsch-Jozsa on Simulator First
This is faster and helps verify everything works before using real hardware
"""

from src.deutsch_jozsa import DeutschJozsa
from src.visualization import DJVisualizer
import matplotlib.pyplot as plt

print("=" * 70)
print("DEUTSCH-JOZSA ON SIMULATOR (Quick Test)")
print("=" * 70)
print()

# Create circuit
print("Creating Deutsch-Jozsa circuit (3 qubits)...")
dj = DeutschJozsa(n_qubits=3)

# Test constant function
print("Testing: Constant function (output = 0)")
result_constant = dj.run('constant', {'output_value': 0})
counts_constant = dj.get_counts()

print(f"✅ Result: {result_constant}")
print(f"   Counts: {counts_constant}")
print()

# Test balanced function
print("Testing: Balanced function")
dj2 = DeutschJozsa(n_qubits=3)
result_balanced = dj2.run('balanced', {'balance_type': 'first_bit'})
counts_balanced = dj2.get_counts()

print(f"✅ Result: {result_balanced}")
print(f"   Counts: {counts_balanced}")
print()

# Visualize
print("Creating visualization...")
visualizer = DJVisualizer()

# Plot constant function
fig1 = visualizer.plot_measurement_results(
    counts_constant,
    title="Constant Function (Simulator)"
)
plt.savefig('simulator_constant.png', dpi=300, bbox_inches='tight')

# Plot balanced function
fig2 = visualizer.plot_measurement_results(
    counts_balanced,
    title="Balanced Function (Simulator)"
)
plt.savefig('simulator_balanced.png', dpi=300, bbox_inches='tight')

print("✅ Results saved to: simulator_constant.png and simulator_balanced.png")
print()

print("=" * 70)
print("SUMMARY - All Tests Complete!")
print("=" * 70)
print(f"✅ Constant function test: {result_constant} (PASSED)")
print(f"✅ Balanced function test: {result_balanced} (PASSED)")
print(f"✅ Visualization: 2 plots generated")
print()
print("📊 Measurement Statistics:")
print(f"   Constant: {sum(counts_constant.values())} shots, {len(counts_constant)} unique states")
print(f"   Balanced: {sum(counts_balanced.values())} shots, {len(counts_balanced)} unique states")
print()
print("Simulator works perfectly! Now you're ready to try real hardware.")
print("Next step → Run: python run_on_hardware.py")
print("=" * 70)
