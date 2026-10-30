# `src/native/strassen_optimal.c`

> Source: [src/native/strassen_optimal.c:1](../../src/native/strassen_optimal.c#L1) | language `c` | coverage 33.3% | [API index](../index.md)

## Contents

- **Functions:** [`strassen_optimal`](#strassen_optimal)
- **Macros:** [`STRASSEN_THRESHOLD`](#strassen_threshold)

## Functions

<a id="strassen_optimal"></a>

### `strassen_optimal`

`function` | [src/native/strassen_optimal.c:131](../../src/native/strassen_optimal.c#L131)

```c
void strassen_optimal(float* C, float* A, float* B, int n)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `C` | `float*` |  |  |
| `A` | `float*` |  |  |
| `B` | `float*` |  |  |
| `n` | `int` |  |  |

---

## Macros

<a id="strassen_threshold"></a>

### `STRASSEN_THRESHOLD`

`macro` | [src/native/strassen_optimal.c:15](../../src/native/strassen_optimal.c#L15)

```c
#define STRASSEN_THRESHOLD
```

Only use Strassen for huge matrices where O(n^2.807) wins

---
