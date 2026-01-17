#!/usr/bin/env python3
"""
Strassen Zero-Shot Expansion Benchmark
======================================
Tests Strassen algorithm across multiple matrix resolutions
using the same 2x2 grokked operator recursively.

Author: grisun0
"""

import torch
import time
import json
import gc
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from strassen import strassen, strassen_2x2, multiplication_count


@dataclass
class BenchmarkResult:
    resolution: int
    strassen_time_ms: float
    standard_time_ms: float
    speedup: float
    strassen_mults: int
    standard_mults: int
    mult_reduction: float
    max_error: float
    mean_error: float
    memory_mb: float
    passed: bool
    error_msg: str = ""


@dataclass
class BenchmarkConfig:
    resolutions: List[int] = field(default_factory=list)
    max_memory_gb: float = 8.0
    dtype: str = "float32"
    warmup_iterations: int = 3
    test_iterations: int = 10
    samples_per_resolution: int = 100
    tolerance_float: float = 1e-4
    tolerance_int: float = 0
    verbose: bool = True
    save_results: bool = True
    results_file: str = "benchmark_results.json"
    verify_correctness: bool = True
    compare_with_standard: bool = True


def load_config(config_path: str = "config.toml") -> BenchmarkConfig:
    """Load configuration from TOML file."""
    path = Path(__file__).parent / config_path
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    cfg = BenchmarkConfig()
    
    if "expansion" in data:
        cfg.resolutions = data["expansion"].get("resolutions", cfg.resolutions)
        cfg.max_memory_gb = data["expansion"].get("max_memory_gb", cfg.max_memory_gb)
        cfg.dtype = data["expansion"].get("dtype", cfg.dtype)
    
    if "benchmark" in data:
        cfg.warmup_iterations = data["benchmark"].get("warmup_iterations", cfg.warmup_iterations)
        cfg.test_iterations = data["benchmark"].get("test_iterations", cfg.test_iterations)
        cfg.verify_correctness = data["benchmark"].get("verify_correctness", cfg.verify_correctness)
        cfg.compare_with_standard = data["benchmark"].get("compare_with_standard", cfg.compare_with_standard)
    
    if "test" in data:
        cfg.samples_per_resolution = data["test"].get("samples_per_resolution", cfg.samples_per_resolution)
        cfg.tolerance_float = data["test"].get("tolerance_float", cfg.tolerance_float)
        cfg.tolerance_int = data["test"].get("tolerance_int", cfg.tolerance_int)
    
    if "logging" in data:
        cfg.verbose = data["logging"].get("verbose", cfg.verbose)
        cfg.save_results = data["logging"].get("save_results", cfg.save_results)
        cfg.results_file = data["logging"].get("results_file", cfg.results_file)
    
    return cfg


def get_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    return dtype_map.get(dtype_str, torch.float32)


def estimate_memory_mb(n: int, dtype: torch.dtype, batch_size: int = 1) -> float:
    """Estimate memory usage for matrix multiplication."""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    # A, B, C matrices + intermediate
    matrices = 4 * batch_size * n * n * bytes_per_elem
    # Recursion overhead (approximate)
    recursion_overhead = 2 * matrices
    return (matrices + recursion_overhead) / (1024 * 1024)


def benchmark_resolution(n: int, cfg: BenchmarkConfig, dtype: torch.dtype) -> BenchmarkResult:
    """Benchmark Strassen vs standard matmul for given resolution."""
    
    is_int = dtype in [torch.int32, torch.int64]
    tolerance = cfg.tolerance_int if is_int else cfg.tolerance_float
    
    # Generate test matrices
    if is_int:
        A = torch.randint(-10, 11, (n, n), dtype=dtype)
        B = torch.randint(-10, 11, (n, n), dtype=dtype)
    else:
        A = torch.randn(n, n, dtype=dtype)
        B = torch.randn(n, n, dtype=dtype)
    
    # Memory estimation
    mem_mb = estimate_memory_mb(n, dtype)
    
    # Warmup
    for _ in range(cfg.warmup_iterations):
        try:
            _ = strassen(A.float(), B.float()) if is_int else strassen(A, B)
            if cfg.compare_with_standard:
                _ = A @ B if is_int else A @ B
        except Exception as e:
            return BenchmarkResult(
                resolution=n,
                strassen_time_ms=0,
                standard_time_ms=0,
                speedup=0,
                strassen_mults=0,
                standard_mults=0,
                mult_reduction=0,
                max_error=float('inf'),
                mean_error=float('inf'),
                memory_mb=mem_mb,
                passed=False,
                error_msg=str(e)
            )
    
    # Benchmark Strassen
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    strassen_times = []
    for _ in range(cfg.test_iterations):
        start = time.perf_counter()
        if is_int:
            C_strassen = strassen(A.float(), B.float())
        else:
            C_strassen = strassen(A, B)
        strassen_times.append((time.perf_counter() - start) * 1000)
    
    strassen_time = sum(strassen_times) / len(strassen_times)
    
    # Benchmark standard
    standard_times = []
    if cfg.compare_with_standard:
        for _ in range(cfg.test_iterations):
            start = time.perf_counter()
            if is_int:
                C_standard = (A.float() @ B.float())
            else:
                C_standard = A @ B
            standard_times.append((time.perf_counter() - start) * 1000)
        standard_time = sum(standard_times) / len(standard_times)
    else:
        if is_int:
            C_standard = (A.float() @ B.float())
        else:
            C_standard = A @ B
        standard_time = 0
    
    # Verify correctness
    if is_int:
        C_strassen = C_strassen.round()
    
    error = (C_strassen - C_standard).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    # Calculate multiplication counts
    strassen_mults = multiplication_count(n)
    standard_mults = n ** 3
    
    # Results
    speedup = standard_time / strassen_time if strassen_time > 0 else 0
    mult_reduction = 1 - (strassen_mults / standard_mults)
    passed = max_error <= tolerance if is_int else max_error < tolerance * n
    
    return BenchmarkResult(
        resolution=n,
        strassen_time_ms=strassen_time,
        standard_time_ms=standard_time,
        speedup=speedup,
        strassen_mults=strassen_mults,
        standard_mults=standard_mults,
        mult_reduction=mult_reduction,
        max_error=max_error,
        mean_error=mean_error,
        memory_mb=mem_mb,
        passed=passed
    )


def run_benchmark(cfg: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    
    dtype = get_dtype(cfg.dtype)
    results = []
    max_mem_bytes = cfg.max_memory_gb * 1024 * 1024 * 1024
    
    print("=" * 70)
    print("STRASSEN ZERO-SHOT EXPANSION BENCHMARK")
    print("=" * 70)
    print(f"Resolutions: {cfg.resolutions}")
    print(f"Data type: {cfg.dtype}")
    print(f"Max memory: {cfg.max_memory_gb} GB")
    print(f"Iterations: {cfg.test_iterations}")
    print("=" * 70)
    print()
    
    header = f"{'N':>6} | {'Strassen':>10} | {'Standard':>10} | {'Speedup':>8} | {'Mults Red':>9} | {'MaxErr':>10} | {'Mem MB':>8} | {'Status':>8}"
    print(header)
    print("-" * len(header))
    
    for n in cfg.resolutions:
        # Check memory limit
        est_mem = estimate_memory_mb(n, dtype)
        if est_mem > cfg.max_memory_gb * 1024:
            print(f"{n:>6} | {'SKIPPED':>10} | {'':>10} | {'':>8} | {'':>9} | {'':>10} | {est_mem:>8.1f} | {'OOM':>8}")
            results.append(BenchmarkResult(
                resolution=n,
                strassen_time_ms=0,
                standard_time_ms=0,
                speedup=0,
                strassen_mults=0,
                standard_mults=0,
                mult_reduction=0,
                max_error=0,
                mean_error=0,
                memory_mb=est_mem,
                passed=False,
                error_msg="Out of memory limit"
            ))
            continue
        
        try:
            result = benchmark_resolution(n, cfg, dtype)
            results.append(result)
            
            status = "PASS" if result.passed else "FAIL"
            print(f"{n:>6} | {result.strassen_time_ms:>10.3f} | {result.standard_time_ms:>10.3f} | "
                  f"{result.speedup:>8.2f}x | {result.mult_reduction*100:>8.1f}% | "
                  f"{result.max_error:>10.2e} | {result.memory_mb:>8.1f} | {status:>8}")
            
        except Exception as e:
            print(f"{n:>6} | {'ERROR':>10} | {'':>10} | {'':>8} | {'':>9} | {'':>10} | {est_mem:>8.1f} | {'ERROR':>8}")
            results.append(BenchmarkResult(
                resolution=n,
                strassen_time_ms=0,
                standard_time_ms=0,
                speedup=0,
                strassen_mults=0,
                standard_mults=0,
                mult_reduction=0,
                max_error=0,
                mean_error=0,
                memory_mb=est_mem,
                passed=False,
                error_msg=str(e)
            ))
        
        gc.collect()
    
    print()
    print("=" * 70)
    
    # Summary
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed and r.error_msg == ""]
    skipped = [r for r in results if r.error_msg != ""]
    
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    
    if passed:
        max_res = max(r.resolution for r in passed)
        print(f"Maximum successful resolution: {max_res}x{max_res}")
        
        avg_mult_reduction = sum(r.mult_reduction for r in passed) / len(passed)
        print(f"Average multiplication reduction: {avg_mult_reduction*100:.1f}%")
    
    print("=" * 70)
    
    return results


def save_results(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results to JSON."""
    path = Path(__file__).parent / filepath
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {path}")


def main():
    cfg = load_config()
    results = run_benchmark(cfg)
    
    if cfg.save_results:
        save_results(results, cfg.results_file)
    
    # Return max successful resolution
    passed = [r for r in results if r.passed]
    if passed:
        return max(r.resolution for r in passed)
    return 0


if __name__ == "__main__":
    max_res = main()
    sys.exit(0 if max_res > 0 else 1)
