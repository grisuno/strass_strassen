# `src.benchmarks.benchmark_final`

> Source: [src/benchmarks/benchmark_final.py:1](../../src/benchmarks/benchmark_final.py#L1) | language `py` | coverage 80.0% | [API index](../index.md)

BENCHMARK FINAL: Strassen Hybrid vs NumPy BLAS ============================================== La prueba definitiva: ¿Puede nuestro modelo Grokkeado + optimizaciones BLAS superar al método tradicional?

Author: Matrix Agent

## Contents

- **Functions:** [`strassen_hybrid_multiply`](#strassen_hybrid_multiply), [`numpy_multiply`](#numpy_multiply), [`benchmark`](#benchmark), [`main`](#main)

## Call Graph

Solid arrows stay in this module, dashed arrows leave it.

```mermaid
flowchart LR
    n_src_benchmarks_benchmark_final_py__main(["main"])
    n_src_benchmarks_benchmark_final_py__benchmark(["benchmark"])
    n_src_benchmarks_benchmark_final_py__main --> n_src_benchmarks_benchmark_final_py__benchmark
```

## Functions

<a id="strassen_hybrid_multiply"></a>

### `strassen_hybrid_multiply`

`function` | [src/benchmarks/benchmark_final.py:45](../../src/benchmarks/benchmark_final.py#L45)

```python
def strassen_hybrid_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray
```

Multiply using our Strassen Hybrid implementation

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` |  |  |
| `B` | `np.ndarray` |  |  |

**Returns:** `np.ndarray`

---

<a id="numpy_multiply"></a>

### `numpy_multiply`

`function` | [src/benchmarks/benchmark_final.py:60](../../src/benchmarks/benchmark_final.py#L60)

```python
def numpy_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray
```

Standard NumPy BLAS multiplication

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` |  |  |
| `B` | `np.ndarray` |  |  |

**Returns:** `np.ndarray`

---

<a id="benchmark"></a>

### `benchmark`

`function` | [src/benchmarks/benchmark_final.py:64](../../src/benchmarks/benchmark_final.py#L64)

```python
def benchmark(func, A, B, warmup=2, runs=5)
```

Run benchmark with warmup

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `func` |  |  |  |
| `A` |  |  |  |
| `B` |  |  |  |
| `warmup` |  | `2` |  |
| `runs` |  | `5` |  |

*Called by:* [`main`](#main)

---

<a id="main"></a>

### `main`

`function` | [src/benchmarks/benchmark_final.py:80](../../src/benchmarks/benchmark_final.py#L80)

```python
def main()
```

*Calls:* [`benchmark`](#benchmark)

---
