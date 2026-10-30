# `src.benchmarks.strassen_numpy`

> Source: [src/benchmarks/strassen_numpy.py:1](../../src/benchmarks/strassen_numpy.py#L1) | language `py` | coverage 100.0% | [API index](../index.md)

Strassen Matrix Multiplication - NumPy Optimized ================================================= Zero-shot expansion using the grokked 2x2 operator.

Author: grisun0

## Contents

- **Functions:** [`strassen_2x2_numpy`](#strassen_2x2_numpy), [`strassen_numpy`](#strassen_numpy), [`strassen_hybrid`](#strassen_hybrid), [`multiplication_count`](#multiplication_count)

## Call Graph

Solid arrows stay in this module, dashed arrows leave it.

```mermaid
flowchart LR
    n_src_benchmarks_strassen_numpy_py__strassen_numpy(["strassen_numpy"])
    n_src_benchmarks_strassen_numpy_py__strassen_2x2_numpy(["strassen_2x2_numpy"])
    n_src_benchmarks_strassen_numpy_py__strassen_numpy --> n_src_benchmarks_strassen_numpy_py__strassen_2x2_numpy
```

## Functions

<a id="strassen_2x2_numpy"></a>

### `strassen_2x2_numpy`

`function` | [src/benchmarks/strassen_numpy.py:29](../../src/benchmarks/strassen_numpy.py#L29)

```python
def strassen_2x2_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray
```

Strassen 2x2 using grokked coefficients.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` |  |  |
| `B` | `np.ndarray` |  |  |

**Returns:** `np.ndarray`

*Called by:* [`strassen_numpy`](#strassen_numpy)

---

<a id="strassen_numpy"></a>

### `strassen_numpy`

`function` | [src/benchmarks/strassen_numpy.py:44](../../src/benchmarks/strassen_numpy.py#L44)

```python
def strassen_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray
```

Recursive Strassen using NumPy.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` |  |  |
| `B` | `np.ndarray` |  |  |

**Returns:** `np.ndarray`

*Calls:* [`strassen_2x2_numpy`](#strassen_2x2_numpy)

---

<a id="strassen_hybrid"></a>

### `strassen_hybrid`

`function` | [src/benchmarks/strassen_numpy.py:78](../../src/benchmarks/strassen_numpy.py#L78)

```python
def strassen_hybrid(A: np.ndarray, B: np.ndarray, threshold: int = 64) -> np.ndarray
```

Hybrid Strassen: use Strassen for large matrices, NumPy for small.

This is faster because NumPy matmul is highly optimized for small matrices.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` |  |  |
| `B` | `np.ndarray` |  |  |
| `threshold` | `int` | `64` |  |

**Returns:** `np.ndarray`

---

<a id="multiplication_count"></a>

### `multiplication_count`

`function` | [src/benchmarks/strassen_numpy.py:112](../../src/benchmarks/strassen_numpy.py#L112)

```python
def multiplication_count(n: int) -> int
```

Count multiplications used by Strassen.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `n` | `int` |  |  |

**Returns:** `int`

---
