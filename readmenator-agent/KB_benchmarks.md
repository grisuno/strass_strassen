# Subsystem: benchmarks

## src/benchmarks/benchmark_final.py
- Layer: utility
- Language: py
- Symbols:
  - `strassen_hybrid_multiply` (function, line 45) `def strassen_hybrid_multiply(A, B)`
  - `numpy_multiply` (function, line 60) `def numpy_multiply(A, B)`
  - `benchmark` (function, line 64) `def benchmark(func, A, B, warmup, runs)`
  - `main` (function, line 80) `def main()`

## src/benchmarks/benchmark_scientific.py
- Layer: utility
- Language: py
- Symbols:
  - `strassen_multiply` (function, line 38) `def strassen_multiply(A, B)`
  - `standard_avx512_multiply` (function, line 51) `def standard_avx512_multiply(A, B)`
  - `numpy_multiply` (function, line 64) `def numpy_multiply(A, B)`
  - `benchmark_function` (function, line 67) `def benchmark_function(func, A, B, runs, warmup)`
  - `main` (function, line 91) `def main()`

## src/benchmarks/benchmark_strassen.py
- Layer: utility
- Language: py
- Symbols:
  - `BenchmarkResult` (class, line 29) `class BenchmarkResult`
  - `BenchmarkConfig` (class, line 45) `class BenchmarkConfig`
  - `load_config` (method, line 61) `def load_config(config_path)`
  - `get_dtype` (method, line 93) `def get_dtype(dtype_str)`
  - `estimate_memory_mb` (method, line 104) `def estimate_memory_mb(n, dtype, batch_size)`
  - `benchmark_resolution` (method, line 114) `def benchmark_resolution(n, cfg, dtype)`
  - `run_benchmark` (method, line 218) `def run_benchmark(cfg)`
  - `save_results` (method, line 310) `def save_results(results, filepath)`
  - `main` (method, line 322) `def main()`

## src/benchmarks/strassen_numpy.py
- Layer: utility
- Language: py
- Symbols:
  - `_load_weights` (function, line 19) `def _load_weights()`
  - `strassen_2x2_numpy` (function, line 29) `def strassen_2x2_numpy(A, B)`
  - `strassen_numpy` (function, line 44) `def strassen_numpy(A, B)`
  - `strassen_hybrid` (function, line 78) `def strassen_hybrid(A, B, threshold)`
  - `multiplication_count` (function, line 112) `def multiplication_count(n)`
