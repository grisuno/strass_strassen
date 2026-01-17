#!/usr/bin/env python3
"""
SCIENTIFIC BENCHMARK: Strassen AVX-512 vs OpenBLAS
===================================================
Generates raw data for scientific analysis.
All measurements in milliseconds with statistical analysis.
"""

import numpy as np
import ctypes
import time
import json
import os
from datetime import datetime

os.chdir('/workspace/strass')

# Load libraries
lib_strassen = ctypes.CDLL('./libstrassen_avx512.so')
lib_strassen.strassen_avx512.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
lib_strassen.strassen_avx512.restype = None

lib_strassen.standard_avx512.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
lib_strassen.standard_avx512.restype = None

lib_strassen.get_threads.restype = ctypes.c_int

def strassen_multiply(A, B):
    n = A.shape[0]
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    lib_strassen.strassen_avx512(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    return C

def standard_avx512_multiply(A, B):
    n = A.shape[0]
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    lib_strassen.standard_avx512(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    return C

def numpy_multiply(A, B):
    return A @ B

def benchmark_function(func, A, B, runs=5, warmup=2):
    """Benchmark with statistical analysis"""
    # Warmup
    for _ in range(warmup):
        _ = func(A, B)
    
    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        C = func(A, B)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    return {
        'min': min(times),
        'max': max(times),
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'samples': times,
        'result': C
    }

def main():
    timestamp = datetime.now().isoformat()
    threads = lib_strassen.get_threads()
    
    print("=" * 80)
    print("SCIENTIFIC BENCHMARK: Strassen AVX-512 vs OpenBLAS vs Standard AVX-512")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"OpenMP Threads: {threads}")
    print(f"Strassen Threshold: 2048")
    print(f"Block Size: 64")
    print()
    
    # Test sizes
    sizes = [512, 1024, 2048, 4096, 8192]
    
    results = {
        'metadata': {
            'timestamp': timestamp,
            'threads': threads,
            'strassen_threshold': 2048,
            'block_size': 64,
            'dtype': 'float32'
        },
        'benchmarks': []
    }
    
    header = f"{'Size':<8} | {'NumPy/BLAS':<14} | {'Std AVX-512':<14} | {'Strassen':<14} | {'vs BLAS':<10} | {'vs Std':<10} | {'Error':<12}"
    print(header)
    print("-" * 100)
    
    for n in sizes:
        try:
            # Generate random matrices
            np.random.seed(42)
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            
            runs = 5 if n <= 4096 else 3
            
            # Benchmark all three
            r_numpy = benchmark_function(numpy_multiply, A, B, runs=runs)
            r_std = benchmark_function(standard_avx512_multiply, A, B, runs=runs)
            r_strassen = benchmark_function(strassen_multiply, A, B, runs=runs)
            
            # Calculate error
            ref = r_numpy['result']
            err_std = np.abs(r_std['result'] - ref).max() / np.abs(ref).max()
            err_str = np.abs(r_strassen['result'] - ref).max() / np.abs(ref).max()
            
            # Speedup calculations
            speedup_vs_blas = r_numpy['mean'] / r_strassen['mean']
            speedup_vs_std = r_std['mean'] / r_strassen['mean']
            
            # FLOPS calculation
            flops = 2 * n**3
            gflops_numpy = flops / (r_numpy['mean'] / 1000) / 1e9
            gflops_std = flops / (r_std['mean'] / 1000) / 1e9
            gflops_strassen = flops / (r_strassen['mean'] / 1000) / 1e9
            
            # Store results
            result = {
                'size': n,
                'numpy_blas': {
                    'mean_ms': round(r_numpy['mean'], 3),
                    'std_ms': round(r_numpy['std'], 3),
                    'min_ms': round(r_numpy['min'], 3),
                    'max_ms': round(r_numpy['max'], 3),
                    'gflops': round(gflops_numpy, 2)
                },
                'standard_avx512': {
                    'mean_ms': round(r_std['mean'], 3),
                    'std_ms': round(r_std['std'], 3),
                    'min_ms': round(r_std['min'], 3),
                    'max_ms': round(r_std['max'], 3),
                    'gflops': round(gflops_std, 2),
                    'error': float(err_std)
                },
                'strassen_avx512': {
                    'mean_ms': round(r_strassen['mean'], 3),
                    'std_ms': round(r_strassen['std'], 3),
                    'min_ms': round(r_strassen['min'], 3),
                    'max_ms': round(r_strassen['max'], 3),
                    'gflops': round(gflops_strassen, 2),
                    'error': float(err_str)
                },
                'speedup_vs_blas': round(speedup_vs_blas, 4),
                'speedup_vs_std_avx512': round(speedup_vs_std, 4)
            }
            results['benchmarks'].append(result)
            
            # Print row
            winner = ""
            if speedup_vs_blas > 1.0:
                winner = " << STRASSEN WINS"
            print(f"{n:<8} | {r_numpy['mean']:>10.2f} ms | {r_std['mean']:>10.2f} ms | {r_strassen['mean']:>10.2f} ms | {speedup_vs_blas:>8.3f}x | {speedup_vs_std:>8.3f}x | {err_str:.2e}{winner}")
            
        except MemoryError:
            print(f"{n:<8} | OUT OF MEMORY")
            break
        except Exception as e:
            print(f"{n:<8} | ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("-" * 100)
    print()
    
    # Summary statistics
    print("=" * 80)
    print("RAW DATA SUMMARY")
    print("=" * 80)
    
    print("\nGFLOPS Performance:")
    print(f"{'Size':<8} | {'NumPy/BLAS':<14} | {'Std AVX-512':<14} | {'Strassen':<14}")
    print("-" * 60)
    for r in results['benchmarks']:
        print(f"{r['size']:<8} | {r['numpy_blas']['gflops']:>10.2f}     | {r['standard_avx512']['gflops']:>10.2f}     | {r['strassen_avx512']['gflops']:>10.2f}")
    
    print("\nSpeedup Analysis:")
    for r in results['benchmarks']:
        vs_blas = r['speedup_vs_blas']
        vs_std = r['speedup_vs_std_avx512']
        status_blas = "FASTER" if vs_blas > 1 else "SLOWER"
        status_std = "FASTER" if vs_std > 1 else "SLOWER"
        print(f"  {r['size']}x{r['size']}: Strassen is {vs_blas:.3f}x vs BLAS ({status_blas}), {vs_std:.3f}x vs Std AVX-512 ({status_std})")
    
    # Save to JSON
    output_file = 'benchmark_scientific_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw data saved to: {output_file}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
