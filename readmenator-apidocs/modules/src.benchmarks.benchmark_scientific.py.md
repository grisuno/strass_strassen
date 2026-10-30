# `src.benchmarks.benchmark_scientific`

> Source: [src/benchmarks/benchmark_scientific.py:1](../../src/benchmarks/benchmark_scientific.py#L1) | language `py` | coverage 33.3% | [API index](../index.md)

SCIENTIFIC BENCHMARK: Strassen AVX-512 vs OpenBLAS =================================================== Generates raw data for scientific analysis.

All measurements in milliseconds with statistical analysis.

## Contents

- **Functions:** [`strassen_multiply`](#strassen_multiply), [`standard_avx512_multiply`](#standard_avx512_multiply), [`numpy_multiply`](#numpy_multiply), [`benchmark_function`](#benchmark_function), [`main`](#main)

## Call Graph

Solid arrows stay in this module, dashed arrows leave it.

```mermaid
flowchart LR
    n_src_benchmarks_benchmark_scientific_py__main(["main"])
    n_src_benchmarks_benchmark_scientific_py__benchmark_function(["benchmark_function"])
    n_src_benchmarks_benchmark_scientific_py__main --> n_src_benchmarks_benchmark_scientific_py__benchmark_function
```

## Functions

<a id="strassen_multiply"></a>

### `strassen_multiply`

`function` | [src/benchmarks/benchmark_scientific.py:38](../../src/benchmarks/benchmark_scientific.py#L38)

```python
def strassen_multiply(A, B)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` |  |  |  |
| `B` |  |  |  |

---

<a id="standard_avx512_multiply"></a>

### `standard_avx512_multiply`

`function` | [src/benchmarks/benchmark_scientific.py:51](../../src/benchmarks/benchmark_scientific.py#L51)

```python
def standard_avx512_multiply(A, B)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` |  |  |  |
| `B` |  |  |  |

---

<a id="numpy_multiply"></a>

### `numpy_multiply`

`function` | [src/benchmarks/benchmark_scientific.py:64](../../src/benchmarks/benchmark_scientific.py#L64)

```python
def numpy_multiply(A, B)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` |  |  |  |
| `B` |  |  |  |

---

<a id="benchmark_function"></a>

### `benchmark_function`

`function` | [src/benchmarks/benchmark_scientific.py:67](../../src/benchmarks/benchmark_scientific.py#L67)

```python
def benchmark_function(func, A, B, runs=5, warmup=2)
```

Benchmark with statistical analysis

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `func` |  |  |  |
| `A` |  |  |  |
| `B` |  |  |  |
| `runs` |  | `5` |  |
| `warmup` |  | `2` |  |

*Called by:* [`main`](#main)

---

<a id="main"></a>

### `main`

`function` | [src/benchmarks/benchmark_scientific.py:91](../../src/benchmarks/benchmark_scientific.py#L91)

```python
def main()
```

*Calls:* [`benchmark_function`](#benchmark_function)

---
