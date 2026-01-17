#!/usr/bin/env python3
"""
BENCHMARK FINAL: Strassen Hybrid vs NumPy BLAS
==============================================
La prueba definitiva: ¿Puede nuestro modelo Grokkeado + optimizaciones BLAS
superar al método tradicional?

Author: Matrix Agent
"""

import numpy as np
import ctypes
import time
import os

# Load the hybrid library from native directory
lib_path = os.path.join(os.path.dirname(__file__), '..', 'native', 'strassen_hybrid.so')
if not os.path.exists(lib_path):
    # Fallback to turbo library
    lib_path = os.path.join(os.path.dirname(__file__), '..', 'native', 'libstrassen_turbo.so')
lib = ctypes.CDLL(lib_path)

# Configure function signatures
# The library exports strassen_turbo (or strassen_hybrid via symlink)
try:
    strassen_func = lib.strassen_hybrid
except AttributeError:
    strassen_func = lib.strassen_turbo

strassen_func.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
strassen_func.restype = None

try:
    lib.get_threads.restype = ctypes.c_int
    get_threads = lib.get_threads
except AttributeError:
    lib.get_num_threads.restype = ctypes.c_int
    get_threads = lib.get_num_threads

def strassen_hybrid_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply using our Strassen Hybrid implementation"""
    n = A.shape[0]
    A_f = np.ascontiguousarray(A, dtype=np.float32)
    B_f = np.ascontiguousarray(B, dtype=np.float32)
    C_f = np.zeros((n, n), dtype=np.float32)
    
    strassen_func(
        C_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    return C_f

def numpy_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard NumPy BLAS multiplication"""
    return A @ B

def benchmark(func, A, B, warmup=2, runs=5):
    """Run benchmark with warmup"""
    # Warmup
    for _ in range(warmup):
        _ = func(A, B)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = func(A, B)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.min(times), np.mean(times), np.max(times), C

def main():
    print("=" * 70)
    print("  BENCHMARK FINAL: Strassen Grokked Hybrid vs NumPy BLAS")
    print("=" * 70)
    print(f"\n  OpenMP Threads: {get_threads()}")
    print(f"  NumPy Config: {np.show_config.__module__}")
    print()
    
    # Test sizes (powers of 2)
    sizes = [128, 256, 512, 1024, 2048, 4096]
    
    print(f"{'Size':>6} | {'NumPy (ms)':>12} | {'Hybrid (ms)':>12} | {'Speedup':>8} | {'Winner':>10} | {'Error':>10}")
    print("-" * 70)
    
    results = []
    
    for n in sizes:
        try:
            # Generate random matrices
            np.random.seed(42)
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            
            # Benchmark NumPy
            np_min, np_mean, np_max, C_np = benchmark(numpy_multiply, A, B)
            
            # Benchmark Hybrid
            hy_min, hy_mean, hy_max, C_hy = benchmark(strassen_hybrid_multiply, A, B)
            
            # Calculate error
            error = np.max(np.abs(C_np - C_hy)) / np.max(np.abs(C_np))
            
            # Speedup (positive = hybrid faster)
            speedup = np_mean / hy_mean
            winner = "HYBRID" if speedup > 1.0 else "NumPy"
            
            np_ms = np_mean * 1000
            hy_ms = hy_mean * 1000
            
            print(f"{n:>6} | {np_ms:>12.3f} | {hy_ms:>12.3f} | {speedup:>7.2f}x | {winner:>10} | {error:>10.2e}")
            
            results.append({
                'size': n,
                'numpy_ms': np_ms,
                'hybrid_ms': hy_ms,
                'speedup': speedup,
                'winner': winner,
                'error': error
            })
            
        except MemoryError:
            print(f"{n:>6} | {'OOM':>12} | {'OOM':>12} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10}")
            break
    
    print("-" * 70)
    
    # Summary
    hybrid_wins = sum(1 for r in results if r['winner'] == 'HYBRID')
    numpy_wins = len(results) - hybrid_wins
    
    print(f"\n  RESUMEN:")
    print(f"  - Victorias Strassen Hybrid: {hybrid_wins}")
    print(f"  - Victorias NumPy BLAS:      {numpy_wins}")
    
    if results:
        best_speedup = max(r['speedup'] for r in results)
        best_size = [r['size'] for r in results if r['speedup'] == best_speedup][0]
        print(f"  - Mejor speedup:             {best_speedup:.2f}x @ {best_size}x{best_size}")
    
    print("\n" + "=" * 70)
    
    # GFLOPS analysis
    print("\n  ANÁLISIS DE RENDIMIENTO (GFLOPS):")
    print(f"{'Size':>6} | {'NumPy GFLOPS':>14} | {'Hybrid GFLOPS':>14} | {'Ops Saved':>10}")
    print("-" * 56)
    
    for r in results:
        n = r['size']
        flops_standard = 2 * n**3  # Standard: 2n³ ops
        flops_strassen = 7 * (n/2)**2.807 * 2  # Approximate for Strassen
        
        gflops_np = (flops_standard / (r['numpy_ms'] / 1000)) / 1e9
        gflops_hy = (flops_standard / (r['hybrid_ms'] / 1000)) / 1e9  # Use same ops for fair comparison
        ops_saved = (1 - (7/8) ** np.log2(n/64)) * 100 if n > 64 else 0  # Approximate ops reduction
        
        print(f"{n:>6} | {gflops_np:>14.2f} | {gflops_hy:>14.2f} | {ops_saved:>9.1f}%")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
