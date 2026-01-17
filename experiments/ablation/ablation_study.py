#!/usr/bin/env python3
"""
ABLACIÓN RIGUROSA - Strassen vs OpenBLAS
=========================================
Protocolo científico:
1. Múltiples runs por configuración (N_RUNS = 5)
2. Warm-up runs descartados
3. Estadísticas: media, std, min, max
4. Control de variables: mismas matrices, misma seed
5. Medición de error numérico
"""

import ctypes
import numpy as np
import time
import json
import gc
from dataclasses import dataclass
from typing import Dict, List

# Configuración del estudio
N_RUNS = 5           # Runs por configuración
WARMUP_RUNS = 2      # Runs de calentamiento (descartados)
SIZES = [2048, 4096, 8192]  # Tamaños a probar

@dataclass
class BenchmarkResult:
    name: str
    size: int
    times: List[float]
    gflops: List[float]
    error: float
    
    @property
    def mean_time(self) -> float:
        return np.mean(self.times)
    
    @property
    def std_time(self) -> float:
        return np.std(self.times)
    
    @property
    def min_time(self) -> float:
        return np.min(self.times)
    
    @property
    def max_time(self) -> float:
        return np.max(self.times)
    
    @property
    def mean_gflops(self) -> float:
        return np.mean(self.gflops)

def load_libraries():
    """Cargar bibliotecas con manejo de errores"""
    libs = {}
    
    # OpenBLAS
    try:
        libs['openblas'] = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libopenblas.so.0')
        libs['openblas'].cblas_sgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
    except Exception as e:
        print(f"Error loading OpenBLAS: {e}")
        return None
    
    # Strassen Optimized
    try:
        libs['optimized'] = ctypes.CDLL('./strassen_optimized.so')
        libs['optimized'].strassen_optimized.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        libs['optimized'].get_threshold.restype = ctypes.c_int
    except Exception as e:
        print(f"Error loading Strassen Optimized: {e}")
    
    # Strassen Hybrid AVX-512 (original)
    try:
        libs['hybrid'] = ctypes.CDLL('./libstrassen_hybrid_avx512.so')
        libs['hybrid'].strassen_hybrid_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
    except Exception as e:
        print(f"Error loading Hybrid: {e}")
    
    # Strassen Fusion
    try:
        libs['fusion'] = ctypes.CDLL('./strassen_fusion.so')
        libs['fusion'].strassen_fusion.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
    except Exception as e:
        print(f"Error loading Fusion: {e}")
    
    return libs

def run_openblas(libs, A, B, C, n):
    """Ejecutar multiplicación con OpenBLAS"""
    libs['openblas'].cblas_sgemm(
        101, 111, 111, n, n, n,
        ctypes.c_float(1.0),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.c_float(0.0),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )

def run_strassen(libs, name, func_name, A, B, C, n):
    """Ejecutar multiplicación con Strassen"""
    func = getattr(libs[name], func_name)
    func(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n)
    )

def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup):
    """Benchmark una implementación"""
    times = []
    
    # Warmup runs
    for _ in range(warmup):
        C.fill(0)
        gc.collect()
        if algo_name == 'openblas':
            run_openblas(libs, A, B, C, n)
        else:
            run_strassen(libs, algo_name, func_name, A, B, C, n)
    
    # Measured runs
    for _ in range(n_runs):
        C.fill(0)
        gc.collect()
        
        start = time.perf_counter()
        if algo_name == 'openblas':
            run_openblas(libs, A, B, C, n)
        else:
            run_strassen(libs, algo_name, func_name, A, B, C, n)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Calcular error relativo
    if C_ref is not None:
        max_ref = np.max(np.abs(C_ref))
        if max_ref > 0:
            error = float(np.max(np.abs(C - C_ref)) / max_ref)
        else:
            error = 0.0
    else:
        error = 0.0
    
    # Calcular GFLOPS
    flops = 2 * n**3
    gflops = [flops / t / 1e9 for t in times]
    
    return BenchmarkResult(
        name=algo_name,
        size=n,
        times=times,
        gflops=gflops,
        error=error
    )

def run_ablation(libs, sizes, n_runs, warmup):
    """Ejecutar ablación completa"""
    results = {}
    
    algos = [
        ('openblas', None),
        ('optimized', 'strassen_optimized'),
        ('hybrid', 'strassen_hybrid_avx512'),
        ('fusion', 'strassen_fusion'),
    ]
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"SIZE: {n}x{n} ({n*n*4/1e6:.1f} MB per matrix)")
        print(f"{'='*60}")
        
        # Crear matrices
        np.random.seed(42)
        A = np.ascontiguousarray(np.random.randn(n, n).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(n, n).astype(np.float32))
        C = np.zeros((n, n), dtype=np.float32, order='C')
        
        results[n] = {}
        C_ref = None
        
        for algo_name, func_name in algos:
            if algo_name not in libs:
                continue
                
            print(f"\n  Testing {algo_name}...", end=' ', flush=True)
            
            try:
                result = benchmark_single(
                    libs, algo_name, func_name, 
                    A, B, C, C_ref, n, n_runs, warmup
                )
                results[n][algo_name] = result
                
                if algo_name == 'openblas':
                    C_ref = C.copy()
                
                print(f"Done: {result.mean_time:.3f}s ± {result.std_time:.3f}s "
                      f"({result.mean_gflops:.1f} GFLOPS)")
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Limpiar memoria
        del A, B, C, C_ref
        gc.collect()
    
    return results

def analyze_results(results):
    """Analizar y presentar resultados"""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    analysis = {}
    
    for n in sorted(results.keys()):
        print(f"\n--- N = {n} ---")
        print(f"{'Algorithm':<15} {'Mean (s)':<12} {'Std (s)':<10} {'Min (s)':<10} "
              f"{'GFLOPS':<10} {'Speedup':<10} {'Error':<12}")
        print("-" * 80)
        
        if 'openblas' not in results[n]:
            continue
            
        blas_time = results[n]['openblas'].mean_time
        analysis[n] = {}
        
        for algo_name, result in results[n].items():
            speedup = blas_time / result.mean_time
            print(f"{algo_name:<15} {result.mean_time:<12.4f} {result.std_time:<10.4f} "
                  f"{result.min_time:<10.4f} {result.mean_gflops:<10.1f} "
                  f"{speedup:<10.2f}x {result.error:<12.2e}")
            
            analysis[n][algo_name] = {
                'mean_time': result.mean_time,
                'std_time': result.std_time,
                'min_time': result.min_time,
                'max_time': result.max_time,
                'mean_gflops': result.mean_gflops,
                'speedup_vs_blas': speedup,
                'relative_error': result.error,
                'times': result.times,
                'gflops': result.gflops
            }
    
    return analysis

def main():
    print("="*60)
    print("ABLATION STUDY: Strassen Implementations vs OpenBLAS")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Measured runs per config: {N_RUNS}")
    print(f"  - Warmup runs (discarded): {WARMUP_RUNS}")
    print(f"  - Matrix sizes: {SIZES}")
    print()
    
    # Cargar bibliotecas
    libs = load_libraries()
    if libs is None:
        print("Failed to load libraries!")
        return
    
    print(f"Loaded libraries: {list(libs.keys())}")
    
    if 'optimized' in libs:
        threshold = libs['optimized'].get_threshold()
        print(f"Optimized Strassen threshold: {threshold}")
    
    # Ejecutar ablación
    results = run_ablation(libs, SIZES, N_RUNS, WARMUP_RUNS)
    
    # Analizar resultados
    analysis = analyze_results(results)
    
    # Guardar resultados
    output = {
        'config': {
            'n_runs': N_RUNS,
            'warmup_runs': WARMUP_RUNS,
            'sizes': SIZES
        },
        'results': analysis
    }
    
    with open('ablation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to ablation_results.json")
    
    # Conclusiones
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    for n in sorted(analysis.keys()):
        blas_time = analysis[n]['openblas']['mean_time']
        best_strassen = None
        best_speedup = 0
        
        for algo, data in analysis[n].items():
            if algo != 'openblas' and data['speedup_vs_blas'] > best_speedup:
                best_speedup = data['speedup_vs_blas']
                best_strassen = algo
        
        if best_speedup > 1.0:
            print(f"N={n}: {best_strassen} BEATS OpenBLAS by {best_speedup:.2f}x")
        else:
            print(f"N={n}: OpenBLAS wins (best Strassen: {best_strassen} at {best_speedup:.2f}x)")

if __name__ == '__main__':
    main()
