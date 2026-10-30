# `experiments.ablation.ablation_8192`

> Source: [experiments/ablation/ablation_8192.py:1](../../experiments/ablation/ablation_8192.py#L1) | language `py` | coverage 0.0% | [API index](../index.md)

## Contents

- **Constants:** [`N_RUNS`](#n_runs), [`WARMUP`](#warmup), [`A`](#a), [`B`](#b), [`C`](#c)

## Constants

<a id="n_runs"></a>

### `N_RUNS`

`constant` | [experiments/ablation/ablation_8192.py:9](../../experiments/ablation/ablation_8192.py#L9)

```python
N_RUNS = 3
```

---

<a id="warmup"></a>

### `WARMUP`

`constant` | [experiments/ablation/ablation_8192.py:10](../../experiments/ablation/ablation_8192.py#L10)

```python
WARMUP = 1
```

---

<a id="a"></a>

### `A`

`constant` | [experiments/ablation/ablation_8192.py:35](../../experiments/ablation/ablation_8192.py#L35)

```python
A = np.ascontiguousarray(np.random.randn(n, n).astype(np.float32))
```

---

<a id="b"></a>

### `B`

`constant` | [experiments/ablation/ablation_8192.py:36](../../experiments/ablation/ablation_8192.py#L36)

```python
B = np.ascontiguousarray(np.random.randn(n, n).astype(np.float32))
```

---

<a id="c"></a>

### `C`

`constant` | [experiments/ablation/ablation_8192.py:37](../../experiments/ablation/ablation_8192.py#L37)

```python
C = np.zeros((n, n), dtype=np.float32, order='C')
```

---
