#!/usr/bin/env python3
"""
LÍMITE DE RUIDO DE GRADIENTE COMPUTACIONAL - BENCHMARK
===========================================
Análisis del punto donde Strassen deja de ser realizable.
"""

import numpy as np
import time
import json
import threadpoolctl

THRESHOLD = 2048

def strassen_numpy(A, B, threshold=THRESHOLD):
    """Strassen recursivo con NumPy para productos base."""
    n = A.shape[0]
    
    if n <= threshold:
        return A @ B
    
    h = n // 2
    
    # Extraer cuadrantes
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    
    # 7 productos de Strassen
    M1 = strassen_numpy(A11 + A22, B11 + B22, threshold)
    M2 = strassen_numpy(A21 + A22, B11, threshold)
    M3 = strassen_numpy(A11, B12 - B22, threshold)
    M4 = strassen_numpy(A22, B21 - B11, threshold)
    M5 = strassen_numpy(A11 + A12, B22, threshold)
    M6 = strassen_numpy(A21 - A11, B11 + B12, threshold)
    M7 = strassen_numpy(A12 - A22, B21 + B22, threshold)
    
    # Reconstruir C
    C = np.empty((n, n), dtype=A.dtype)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    
    return C


def measure_single_sgemm(n, threads=1):
    """Mide tiempo de un solo sgemm de tamaño n."""
    with threadpoolctl.threadpool_limits(limits=threads, user_api='blas'):
        A = np.random.randn(n, n).astype(np.float32)
        B = np.random.randn(n, n).astype(np.float32)
        t0 = time.perf_counter()
        C = A @ B
        return time.perf_counter() - t0


def run_planck_analysis():
    """Ejecuta el análisis del Límite de Planck."""
    
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║ LÍMITE DE RUIDO DE GRADIENTE COMPUTACIONAL - BENCHMARK - STRASSEN ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║ Objetivo: Encontrar N_max donde overhead > t_op                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    results = []
    Ns = [2048, 4096, 8192]
    
    # Forzar 1 thread
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
        print(f"{'N':>6} {'t_strassen':>12} {'7*t_op':>12} {'overhead':>12} {'t_blas':>12} {'speedup':>10} {'estado':>15}")
        print("─" * 85)
        
        for N in Ns:
            h = N // 2
            
            # Medir t_op (un sgemm de tamaño N/2)
            t_op = measure_single_sgemm(h, threads=1)
            
            # Medir Strassen
            A = np.random.randn(N, N).astype(np.float32)
            B = np.random.randn(N, N).astype(np.float32)
            
            t0 = time.perf_counter()
            C1 = strassen_numpy(A, B)
            t_strassen = time.perf_counter() - t0
            
            # Medir BLAS directo
            t0 = time.perf_counter()
            C2 = A @ B
            t_blas = time.perf_counter() - t0
            
            # Calcular overhead
            expected_7ops = 7.0 * t_op
            overhead = t_strassen - expected_7ops
            speedup = t_blas / t_strassen
            
            # Estado
            if overhead > t_op:
                estado = "PLANCK_LIMIT"
            elif overhead > 0.5 * t_op:
                estado = "NEAR_LIMIT"
            elif speedup > 1.1:
                estado = "OPTIMAL"
            else:
                estado = "MARGINAL"
            
            print(f"{N:>6} {t_strassen:>12.3f} {expected_7ops:>12.3f} {overhead:>12.3f} {t_blas:>12.3f} {speedup:>9.2f}x {estado:>15}")
            
            results.append({
                'N': N,
                't_strassen': t_strassen,
                't_op': t_op,
                'expected_7ops': expected_7ops,
                'overhead': overhead,
                't_blas': t_blas,
                'speedup': speedup,
                'estado': estado,
                'error': float(np.abs(C1 - C2).max())
            })
    
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                         ANÁLISIS                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Análisis de overhead
    print("  Desglose del overhead de Strassen:")
    print("  ───────────────────────────────────")
    for r in results:
        N = r['N']
        ratio = r['overhead'] / r['t_op'] if r['t_op'] > 0 else float('inf')
        print(f"  N={N}: overhead = {r['overhead']:.3f}s = {ratio:.2f} × t_op")
        print(f"         -> O(n^2) additions = {18 * (N//2)**2 / 1e9:.2f} GFlops additional")
    
    print()
    print("  Interpretación física:")
    print("  ───────────────────────")
    print("  - overhead < 0   -> The 7 multiplications are more efficient than 1")
    print("  - overhead ~ t_op -> At the coherence limit")
    print("  - overhead > t_op -> Strassen loses algorithmic advantage")
    print()
    
    # Guardar resultados
    with open('planck_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Resultados guardados en: planck_results.json")
    
    return results


if __name__ == '__main__':
    results = run_planck_analysis()
