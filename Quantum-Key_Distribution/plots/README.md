# QKD BB84 Simulation - Extracted Plots

This folder contains all plots extracted from the `bb84_simulation.ipynb` notebook.

## Plot Descriptions

### 1. BB84 Demo - 20 Qubits Without Eve (4-Panel)
**File:** `01_bb84_demo_20qubits_no_eve_4panel.png`

A comprehensive 4-panel visualization showing normal BB84 operation:
- **Top Left:** Alice vs Bob basis choices (Z = Blue, X = Red)
- **Top Right:** Bit agreement distribution (violin plot)
- **Bottom Left:** Sifting results (Green = Kept, Red = Discarded)
- **Bottom Right:** Running match percentage across all 20 qubits

Demonstrates the protocol's baseline behavior without eavesdropping.

---

### 2. Eve Attack Comparison (6-Panel)
**File:** `02_eve_attack_comparison_6panel.png`

Comprehensive analysis of Eve's eavesdropping impact (80% interception rate):
- **Row 1, Left:** Eve interception pattern (Red = Intercepted, Green = Clean)
- **Row 1, Middle:** Basis flow visualization (Alice → Eve → Bob)
- **Row 1, Right:** Sifting results showing error introduction
- **Row 2, Left:** Text comparison (No Eve vs With Eve scenarios)
- **Row 2, Middle:** Eve's attack breakdown statistics
- **Row 2, Right:** QBER comparison bar chart with 11% threshold

Shows how eavesdropping exceeds the security threshold and is detected.

---

### 3. BB84 QBER Under Depolarizing Noise
**File:** `03_bb84_qber_under_depolarizing_noise.png`

Line plot showing measured QBER vs depolarizing probability (0 to 0.25).
Demonstrates how quantum channel noise affects error rates in BB84 protocol.

---

### 4. BB84 Intercept-Resend Attack
**File:** `04_bb84_intercept_resend_attack.png`

Line plot comparing:
- **Blue line:** Simulated QBER under varying attack probabilities
- **Red dashed line:** Theoretical prediction (QBER = 0.25 × attack probability)

Validates the theoretical security model against intercept-resend attacks.

---

### 5. QBER Time-Series: Eve Starts Attacking
**File:** `05_qber_timeseries_eve_starts_attacking.png`

Bar chart showing QBER over 20 consecutive BB84 sessions:
- **Green bars:** Secure sessions (before Eve attacks)
- **Red bars:** Compromised sessions (Eve attacking at 100%)
- **Orange dashed line:** 11% abort threshold
- **Purple dotted line:** Marks when Eve starts attacking (session 10)

Demonstrates real-time detection of a sudden eavesdropping attack.

---

### 6. BB84 Sifting: Key Length & Efficiency
**File:** `06_bb84_sifting_key_length_efficiency.png`

Two-panel analysis of basis reconciliation:
- **Left:** Average sifted key length vs initial qubits sent, with 50% theoretical line
- **Right:** Sifting efficiency percentage across different qubit counts

Shows that basis matching naturally yields ~50% efficiency (as expected).

---

### 7. Noise-Induced Bit-Flip Rates by Basis
**File:** `07_noise_induced_bitflip_rates_by_basis.png`

Line plot comparing bit error rates under depolarizing noise:
- **Z Basis:** Error rates for |0⟩ and |1⟩ states
- **X Basis:** Error rates for |+⟩ and |−⟩ states

Demonstrates that both bases are equally affected by channel noise.

---

### 8. BB84 QBER Heatmap: Combined Noise & Attack Impact
**File:** `08_bb84_qber_heatmap_noise_and_attack.png`

2D heatmap showing QBER under combined threats:
- **X-axis:** Channel noise probability (0 to 0.15)
- **Y-axis:** Eve attack probability (0 to 0.6)
- **Color scale:** Red-Yellow-Green showing QBER percentage
- **Blue dashed contour:** 11% abort threshold line
- **Annotations:** QBER percentage in each cell

Identifies safe and danger zones for protocol operation.

---

### 9. Parity-Based Error Correction Performance
**File:** `09_parity_error_correction_performance.png`

Two-panel evaluation of error correction:
- **Left:** Success rate vs number of errors (color-coded: green > 90%, orange > 50%, red otherwise)
- **Right:** Theoretical vs actual performance comparison

Shows the parity scheme reliably corrects 0-1 errors but fails with multiple errors.

---

### 10. Additional Plot
**File:** `plot_10.png`

Additional visualization generated during notebook execution (appears to be a duplicate or supplementary plot).

---

## Technical Details

- **Extraction Date:** December 2, 2025
- **Resolution:** 300 DPI
- **Format:** PNG with white background
- **Source Notebook:** `bb84_simulation.ipynb`
- **Total Plots:** 10 (9 main + 1 additional)
- **Total Size:** ~2.2 MB

## Plot Summary

| # | Filename | Type | Panels | Description |
|---|----------|------|--------|-------------|
| 01 | bb84_demo_20qubits_no_eve_4panel | Multi-panel | 4 | Normal BB84 operation |
| 02 | eve_attack_comparison_6panel | Multi-panel | 6 | Eavesdropping detection |
| 03 | bb84_qber_under_depolarizing_noise | Single | 1 | Noise impact analysis |
| 04 | bb84_intercept_resend_attack | Single | 1 | Attack validation |
| 05 | qber_timeseries_eve_starts_attacking | Single | 1 | Time-series attack detection |
| 06 | bb84_sifting_key_length_efficiency | Multi-panel | 2 | Sifting efficiency |
| 07 | noise_induced_bitflip_rates_by_basis | Single | 1 | Basis fidelity |
| 08 | bb84_qber_heatmap_noise_and_attack | Single | 1 | Combined threat analysis |
| 09 | parity_error_correction_performance | Multi-panel | 2 | Error correction |
| 10 | plot_10 | Single | 1 | Additional |

## Usage

These high-resolution plots are suitable for:
- Academic papers and theses
- Conference presentations
- Educational materials on quantum cryptography
- Technical documentation
- Research reports on QKD security

---

**Note:** All plots demonstrate various aspects of the BB84 Quantum Key Distribution protocol, including normal operation, eavesdropping detection, noise analysis, and error correction mechanisms.
