#!/usr/bin/env python3
"""
ANÁLISIS DE COHERENCIA TEMPORAL
================================
Medición de la estabilidad del algoritmo Strassen.
"""

import numpy as np
import time
import json
import threadpoolctl

THRESHOLD = 2048

def strassen_numpy(A, B, threshold=THRESHOLD):
    n = A.shape[0]
    if n <= threshold:
        return A @ B
    
    h = n // 2
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    
    M1 = strassen_numpy(A11 + A22, B11 + B22, threshold)
    M2 = strassen_numpy(A21 + A22, B11, threshold)
    M3 = strassen_numpy(A11, B12 - B22, threshold)
    M4 = strassen_numpy(A22, B21 - B11, threshold)
    M5 = strassen_numpy(A11 + A12, B22, threshold)
    M6 = strassen_numpy(A21 - A11, B11 + B12, threshold)
    M7 = strassen_numpy(A12 - A22, B21 + B22, threshold)
    
    C = np.empty((n, n), dtype=A.dtype)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    return C


def run_coherence_analysis():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           ANÁLISIS DE COHERENCIA TEMPORAL - STRASSEN             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    results = []
    Ns = [2048, 4096, 8192]
    runs = 5
    
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
        for N in Ns:
            print(f"{'='*60}")
            print(f"N = {N}")
            print('='*60)
            
            A = np.random.randn(N, N).astype(np.float32)
            B = np.random.randn(N, N).astype(np.float32)
            
            # Benchmark Strassen
            times_strassen = []
            for _ in range(runs):
                t0 = time.perf_counter()
                C1 = strassen_numpy(A, B)
                times_strassen.append(time.perf_counter() - t0)
            
            # Benchmark NumPy
            times_numpy = []
            for _ in range(runs):
                t0 = time.perf_counter()
                C2 = A @ B
                times_numpy.append(time.perf_counter() - t0)
            
            t_strassen = np.mean(times_strassen)
            t_numpy = np.mean(times_numpy)
            speedup = t_numpy / t_strassen
            
            # Varianza (proxy de coherencia)
            cv_strassen = np.std(times_strassen) / np.mean(times_strassen) * 100
            cv_numpy = np.std(times_numpy) / np.mean(times_numpy) * 100
            coherence_ratio = cv_numpy / cv_strassen if cv_strassen > 0 else 0
            
            # GFlops
            flops = 2 * N**3
            gflops_strassen = flops / t_strassen / 1e9
            gflops_numpy = flops / t_numpy / 1e9
            
            # Energía conceptual
            energy_ratio = t_strassen / t_numpy
            
            print(f"\n  Tiempos:")
            print(f"    Strassen: {t_strassen:.3f}s ± {np.std(times_strassen)*1000:.1f}ms (CV={cv_strassen:.1f}%)")
            print(f"    NumPy:    {t_numpy:.3f}s ± {np.std(times_numpy)*1000:.1f}ms (CV={cv_numpy:.1f}%)")
            print(f"    Speedup:  {speedup:.2f}x")
            
            print(f"\n  Rendimiento:")
            print(f"    Strassen: {gflops_strassen:.2f} GFlops")
            print(f"    NumPy:    {gflops_numpy:.2f} GFlops")
            
            print(f"\n  Coherencia:")
            print(f"    Ratio de coherencia: {coherence_ratio:.2f}")
            interp = 'Strassen más coherente' if coherence_ratio > 1 else 'NumPy más coherente'
            print(f"    Interpretación: {interp}")
            
            print(f"\n  Análisis Termodinámico:")
            print(f"    Ratio de energía: {energy_ratio:.3f}")
            if energy_ratio < 1:
                print(f"    -> Strassen occupies the MINIMUM of computational energy")
            else:
                print(f"    -> NumPy is more energy-efficient")
            
            results.append({
                'N': N,
                't_strassen': t_strassen,
                't_numpy': t_numpy,
                'speedup': speedup,
                'cv_strassen': cv_strassen,
                'cv_numpy': cv_numpy,
                'coherence_ratio': coherence_ratio,
                'gflops_strassen': gflops_strassen,
                'gflops_numpy': gflops_numpy,
                'energy_ratio': energy_ratio
            })
            print()
    
    # Conclusiones
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    CONCLUSIONES EXPERIMENTALES                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Encontrar punto de transición
    for r in results:
        N = r['N']
        speedup = r['speedup']
        cv = r['cv_strassen']
        
        if speedup > 1.05:
            status = "[OK] OPTIMAL"
            desc = "Strassen domina, algoritmo grokkeado es físicamente realizable"
        elif speedup > 0.95:
            status = "~ TRANSICIÓN"
            desc = "Cerca del límite de Planck computacional"
        else:
            status = "[X] SUB-OPTIMAL"
            desc = "Overhead de Strassen supera el beneficio algorítmico"
        
        print(f"  N={N}: speedup={speedup:.2f}x, CV={cv:.1f}%")
        print(f"    {status}: {desc}")
        print()
    
    print("  Interpretación física del experimento:")
    print("  ─────────────────────────────────────────")
    print("  El algoritmo Strassen 'grokkeado' exhibe las siguientes propiedades:")
    print()
    print("  1. COHERENCIA ESPECTRAL:")
    avg_cv = np.mean([r['cv_strassen'] for r in results])
    print(f"     Average CV = {avg_cv:.1f}% -> {'High coherence' if avg_cv < 5 else 'Moderate coherence'}")
    print()
    print("  2. LÍMITE DE PLANCK:")
    n_optimal = [r['N'] for r in results if r['speedup'] > 1.0]
    if n_optimal:
        print(f"     N_crítico ≈ {min(n_optimal)}: Strassen es realizable para N >= {min(n_optimal)}")
    else:
        print("     Strassen no supera a BLAS en los tamaños probados")
    print()
    print("  3. ENERGÍA COMPUTACIONAL:")
    min_energy = min(results, key=lambda x: x['energy_ratio'])
    print(f"     Mínimo de energía en N={min_energy['N']} (ratio={min_energy['energy_ratio']:.3f})")
    print()
    
    with open('coherence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Resultados guardados en: coherence_results.json")
    
    return results


if __name__ == '__main__':
    run_coherence_analysis()
