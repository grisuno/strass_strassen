#!/usr/bin/env python3
import ctypes
import numpy as np
import time
import json
import gc
import sys

N_RUNS = 3
WARMUP = 1

print('Loading libraries...', flush=True)
openblas = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libopenblas.so.0')
openblas.cblas_sgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

optimized = ctypes.CDLL('./strassen_optimized.so')
optimized.strassen_optimized.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]

n = 8192
print(f'N={n}x{n}', flush=True)

np.random.seed(42)
print('Allocating matrices...', flush=True)
A = np.ascontiguousarray(np.random.randn(n,n).astype(np.float32))
B = np.ascontiguousarray(np.random.randn(n,n).astype(np.float32))
C = np.zeros((n,n), dtype=np.float32, order='C')

# OpenBLAS
print('OpenBLAS warmup...', flush=True)
for _ in range(WARMUP):
    C.fill(0)
    openblas.cblas_sgemm(101, 111, 111, n, n, n,
        ctypes.c_float(1.0),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.c_float(0.0),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)

print('OpenBLAS measuring...', flush=True)
times_blas = []
for run in range(N_RUNS):
    C.fill(0)
    gc.collect()
    t0 = time.perf_counter()
    openblas.cblas_sgemm(101, 111, 111, n, n, n,
        ctypes.c_float(1.0),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.c_float(0.0),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
    t = time.perf_counter() - t0
    times_blas.append(t)
    print(f'  BLAS run {run+1}: {t:.2f}s', flush=True)

C_ref = C.copy()

# Strassen
print('Strassen warmup...', flush=True)
for _ in range(WARMUP):
    C.fill(0)
    optimized.strassen_optimized(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n))

print('Strassen measuring...', flush=True)
times_opt = []
for run in range(N_RUNS):
    C.fill(0)
    gc.collect()
    t0 = time.perf_counter()
    optimized.strassen_optimized(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n))
    t = time.perf_counter() - t0
    times_opt.append(t)
    print(f'  Strassen run {run+1}: {t:.2f}s', flush=True)

err = np.max(np.abs(C - C_ref)) / np.max(np.abs(C_ref))

mean_blas = np.mean(times_blas)
std_blas = np.std(times_blas)
mean_opt = np.mean(times_opt)
std_opt = np.std(times_opt)
speedup = mean_blas / mean_opt
gflops_blas = (2 * n**3) / mean_blas / 1e9
gflops_opt = (2 * n**3) / mean_opt / 1e9

print('='*50, flush=True)
print(f'OpenBLAS:  {mean_blas:.2f}s +/- {std_blas:.2f}s | {gflops_blas:.1f} GFLOPS', flush=True)
print(f'Strassen:  {mean_opt:.2f}s +/- {std_opt:.2f}s | {gflops_opt:.1f} GFLOPS', flush=True)
print(f'Speedup:   {speedup:.3f}x', flush=True)
print(f'Error:     {err:.2e}', flush=True)

if speedup > 1.0:
    print(f'*** STRASSEN WINS by {speedup:.2f}x ***', flush=True)
else:
    print(f'*** OpenBLAS WINS by {1/speedup:.2f}x ***', flush=True)

results = {
    'n': n,
    'openblas': {'mean': mean_blas, 'std': std_blas, 'times': times_blas, 'gflops': gflops_blas},
    'strassen': {'mean': mean_opt, 'std': std_opt, 'times': times_opt, 'gflops': gflops_opt, 'speedup': speedup, 'error': float(err)}
}

with open('ablation_8192_result.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Done!', flush=True)
