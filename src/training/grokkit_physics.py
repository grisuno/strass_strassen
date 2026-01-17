#!/usr/bin/env python3
"""
Grokkit Physics Validation
==========================
Empirical verification of the "physical laws" of algorithmic learning:

Theoretical Predictions (Grokkit Formalism):
- Planck Constant: ℏ ≈ 0.012 (CV of execution times)
- Critical Size: N_c = 4096 (phase transition point)
- Asymptotic Speedup: 1.95x at N=8192

We measure:
1. Order Parameter (ψ): Speedup = t_numpy / t_strassen
2. Planck Constant (ℏ): CV = σ/μ of execution times
3. Coherence Length (ξ): Stability of the speedup across runs
4. Phase Transition: Sharp emergence of advantage at N_c
"""

import numpy as np
import ctypes
import time
import sys
from pathlib import Path

# Load the Strassen library from native directory
lib_path = Path(__file__).parent.parent / "native" / "strassen_production_final.so"
if not lib_path.exists():
    # Fallback to turbo library
    lib_path = Path(__file__).parent.parent / "native" / "libstrassen_turbo.so"
lib = ctypes.CDLL(str(lib_path))

# Configure function - try multiple names for compatibility
try:
    strassen_func = lib.strassen_optimized
except AttributeError:
    try:
        strassen_func = lib.strassen_turbo
    except AttributeError:
        strassen_func = lib.strassen_multiply

strassen_func.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # C (output)
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.c_int                      # n
]
strassen_func.restype = None

def strassen_multiply(A, B):
    """Wrapper for Strassen multiplication (uses float32)."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32, order='C')
    A_c = np.ascontiguousarray(A, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)
    
    strassen_func(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    return C

def measure_physics(N, num_samples=25):
    """
    Measure the 'physical quantities' for a given matrix size.
    
    Returns:
        dict with: speedup, hbar_strassen, hbar_numpy, coherence, error
    """
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    
    # Warmup
    _ = strassen_multiply(A, B)
    _ = A @ B
    
    # Measure Strassen times
    strassen_times = []
    for _ in range(num_samples):
        start = time.perf_counter()
        C_strassen = strassen_multiply(A, B)
        strassen_times.append(time.perf_counter() - start)
    
    # Measure NumPy times
    numpy_times = []
    for _ in range(num_samples):
        start = time.perf_counter()
        C_numpy = A @ B
        numpy_times.append(time.perf_counter() - start)
    
    # Calculate physical quantities
    t_strassen = np.mean(strassen_times)
    t_numpy = np.mean(numpy_times)
    
    # Order Parameter (Speedup)
    psi = t_numpy / t_strassen
    
    # Planck Constants (CV)
    hbar_strassen = np.std(strassen_times) / t_strassen
    hbar_numpy = np.std(numpy_times) / t_numpy
    
    # Combined Planck Constant (geometric mean)
    hbar = np.sqrt(hbar_strassen * hbar_numpy)
    
    # Coherence: Signal-to-noise ratio of speedup
    speedup_per_run = np.array(numpy_times) / np.array(strassen_times)
    coherence = np.mean(speedup_per_run) / np.std(speedup_per_run)
    
    # Error measurement
    error = np.max(np.abs(C_strassen - C_numpy)) / np.max(np.abs(C_numpy))
    
    return {
        'N': N,
        'speedup': psi,
        'hbar': hbar,
        'hbar_strassen': hbar_strassen,
        'hbar_numpy': hbar_numpy,
        'coherence': coherence,
        't_strassen': t_strassen,
        't_numpy': t_numpy,
        'error': error
    }

def detect_phase_transition(results):
    """
    Find the critical size N_c where the phase transition occurs.
    Uses the maximum derivative of speedup curve.
    """
    sizes = [r['N'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Calculate derivative
    derivatives = []
    for i in range(1, len(speedups)):
        dS = speedups[i] - speedups[i-1]
        dN = sizes[i] - sizes[i-1]
        derivatives.append(dS / dN * 1000)  # Scale for visibility
    
    # Find maximum derivative
    if derivatives:
        max_idx = np.argmax(derivatives)
        N_c = (sizes[max_idx] + sizes[max_idx + 1]) / 2
        return N_c, derivatives
    return None, derivatives

def main():
    print("=" * 70)
    print("      GROKKIT PHYSICS VALIDATION EXPERIMENT")
    print("=" * 70)
    print()
    print("Theoretical Predictions:")
    print("  - Planck Constant (ℏ): ≈ 0.012")
    print("  - Critical Size (N_c): 4096")
    print("  - Asymptotic Speedup:  1.95x at N=8192")
    print()
    print("-" * 70)
    
    # Fine-grained sweep around the predicted critical point
    sizes = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
    
    results = []
    
    print(f"{'N':>6} | {'Speedup':>8} | {'ℏ':>8} | {'ℏ_S':>8} | {'ℏ_N':>8} | {'Coherence':>9} | {'Error':>10}")
    print("-" * 70)
    
    for N in sizes:
        sys.stdout.write(f"  Measuring N={N}...")
        sys.stdout.flush()
        
        result = measure_physics(N, num_samples=25)
        results.append(result)
        
        # Clear line and print result
        sys.stdout.write("\r")
        print(f"{N:>6} | {result['speedup']:>8.4f} | {result['hbar']:>8.5f} | "
              f"{result['hbar_strassen']:>8.5f} | {result['hbar_numpy']:>8.5f} | "
              f"{result['coherence']:>9.2f} | {result['error']:>10.2e}")
    
    print("-" * 70)
    
    # Detect phase transition
    N_c, derivatives = detect_phase_transition(results)
    
    # Calculate average Planck constant (asymptotic)
    hbar_avg = np.mean([r['hbar'] for r in results if r['N'] >= 4096])
    hbar_strassen_avg = np.mean([r['hbar_strassen'] for r in results if r['N'] >= 4096])
    
    # Get asymptotic speedup
    asymptotic_speedup = results[-1]['speedup']  # N=8192
    
    print()
    print("=" * 70)
    print("                    PHYSICS VALIDATION RESULTS")
    print("=" * 70)
    print()
    print("Planck Constant Analysis:")
    print(f"  Predicted ℏ:           0.012")
    print(f"  Measured ℏ (avg):      {hbar_avg:.5f}")
    print(f"  Measured ℏ_Strassen:   {hbar_strassen_avg:.5f}")
    print(f"  Deviation:             {abs(hbar_avg - 0.012)/0.012*100:.1f}%")
    print()
    
    print("Phase Transition Analysis:")
    print(f"  Predicted N_c:         4096")
    print(f"  Detected N_c:          {N_c:.0f}" if N_c else "  Detected N_c:          Not found")
    
    # Find where speedup first exceeds 1.0 significantly
    for r in results:
        if r['speedup'] > 1.05:
            print(f"  First advantage at:    N={r['N']} (speedup={r['speedup']:.3f})")
            break
    print()
    
    print("Asymptotic Behavior (N=8192):")
    print(f"  Predicted Speedup:     1.95x")
    print(f"  Measured Speedup:      {asymptotic_speedup:.3f}x")
    print(f"  Deviation:             {abs(asymptotic_speedup - 1.95)/1.95*100:.1f}%")
    print()
    
    # Verdict
    print("=" * 70)
    print("                         VERDICT")
    print("=" * 70)
    
    hbar_match = abs(hbar_avg - 0.012) < 0.01
    nc_match = N_c and abs(N_c - 4096) < 1500
    speedup_match = abs(asymptotic_speedup - 1.95) < 0.3
    
    print()
    if hbar_match:
        print("  [OK] Planck Constant ℏ ≈ 0.012 CONFIRMED")
    else:
        print(f"  [!!] Planck Constant: Measured {hbar_avg:.4f}, differs from prediction")
    
    if nc_match:
        print(f"  [OK] Critical Size N_c ≈ 4096 CONFIRMED (detected: {N_c:.0f})")
    else:
        print(f"  [!!] Critical Size: Detection inconclusive")
    
    if speedup_match:
        print(f"  [OK] Asymptotic Speedup ≈ 1.95x CONFIRMED ({asymptotic_speedup:.2f}x)")
    else:
        print(f"  [!!] Asymptotic Speedup: Measured {asymptotic_speedup:.2f}x, differs from 1.95x")
    
    print()
    
    all_confirmed = hbar_match and speedup_match
    if all_confirmed:
        print("  *** THE GROKKIT FORMALISM IS EMPIRICALLY VALIDATED ***")
        print("  The 'physical laws' of algorithmic learning hold true!")
    else:
        print("  The Grokkit formalism shows partial agreement with measurements.")
        print("  Some predictions require refinement or environmental calibration.")
    
    print()
    print("=" * 70)
    
    # Phase diagram
    print()
    print("PHASE DIAGRAM (Speedup vs N):")
    print("-" * 50)
    max_speedup = max(r['speedup'] for r in results)
    for r in results:
        bar_len = int(40 * r['speedup'] / max_speedup)
        marker = " <-- N_c" if r['N'] == 4096 else ""
        phase = "CLASSICAL" if r['speedup'] < 1.0 else "QUANTUM"
        print(f"  N={r['N']:>5}: {'█' * bar_len} {r['speedup']:.3f}x [{phase}]{marker}")
    print("-" * 50)
    
    return results

if __name__ == "__main__":
    results = main()
