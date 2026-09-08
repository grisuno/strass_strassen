# Subsystem: ablation

## experiments/ablation/ablation_8192.py
- Layer: utility
- Language: py

## experiments/ablation/ablation_study.py
- Layer: utility
- Language: py
- Symbols:
  - `BenchmarkResult` (class, line 27) `class BenchmarkResult`
  - `load_libraries` (method, line 54) `def load_libraries()`
  - `run_openblas` (method, line 111) `def run_openblas(libs, A, B, C, n)`
  - `run_strassen` (method, line 122) `def run_strassen(libs, name, func_name, A, B, C, n)`
  - `benchmark_single` (method, line 132) `def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)`
  - `run_ablation` (method, line 180) `def run_ablation(libs, sizes, n_runs, warmup)`
  - `analyze_results` (method, line 233) `def analyze_results(results)`
  - `main` (method, line 273) `def main()`
  - `mean_time` (method, line 35) `def mean_time(self)`
  - `std_time` (method, line 39) `def std_time(self)`
  - `min_time` (method, line 43) `def min_time(self)`
  - `max_time` (method, line 47) `def max_time(self)`
  - `mean_gflops` (method, line 51) `def mean_gflops(self)`
