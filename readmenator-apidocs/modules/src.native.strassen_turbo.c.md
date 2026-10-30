# `src/native/strassen_turbo.c`

> Source: [src/native/strassen_turbo.c:1](../../src/native/strassen_turbo.c#L1) | language `c` | coverage 42.9% | [API index](../index.md)

## Contents

- **Functions:** [`strassen_turbo_recursive`](#strassen_turbo_recursive), [`strassen_turbo`](#strassen_turbo), [`get_num_threads`](#get_num_threads)
- **Macros:** [`THRESHOLD`](#threshold), [`BLOCK_SIZE`](#block_size), [`ALIGN`](#align)

## Call Graph

Solid arrows stay in this module, dashed arrows leave it.

```mermaid
flowchart LR
    n_src_native_strassen_turbo_c__strassen_turbo(["strassen_turbo"])
    n_src_native_strassen_turbo_c__strassen_turbo_recursive(["strassen_turbo_recursive"])
    n_src_native_strassen_turbo_c__strassen_turbo --> n_src_native_strassen_turbo_c__strassen_turbo_recursive
```

## Functions

<a id="strassen_turbo_recursive"></a>

### `strassen_turbo_recursive`

`function` | [src/native/strassen_turbo.c:124](../../src/native/strassen_turbo.c#L124)

```c
void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)
```

Strassen recursive with parallelism

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `C` | `float*` |  |  |
| `A` | `float*` |  |  |
| `B` | `float*` |  |  |
| `n` | `int` |  |  |
| `depth` | `int` |  |  |

*Called by:* [`strassen_turbo`](#strassen_turbo)

---

<a id="strassen_turbo"></a>

### `strassen_turbo`

`function` | [src/native/strassen_turbo.c:261](../../src/native/strassen_turbo.c#L261)

```c
void strassen_turbo(float* C, float* A, float* B, int n)
```

Public API

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `C` | `float*` |  |  |
| `A` | `float*` |  |  |
| `B` | `float*` |  |  |
| `n` | `int` |  |  |

*Calls:* [`strassen_turbo_recursive`](#strassen_turbo_recursive)

---

<a id="get_num_threads"></a>

### `get_num_threads`

`function` | [src/native/strassen_turbo.c:267](../../src/native/strassen_turbo.c#L267)

```c
int get_num_threads(void)
```

Get number of threads

**Returns:** `int`

---

## Macros

<a id="threshold"></a>

### `THRESHOLD`

`macro` | [src/native/strassen_turbo.c:20](../../src/native/strassen_turbo.c#L20)

```c
#define THRESHOLD
```

---

<a id="block_size"></a>

### `BLOCK_SIZE`

`macro` | [src/native/strassen_turbo.c:21](../../src/native/strassen_turbo.c#L21)

```c
#define BLOCK_SIZE
```

---

<a id="align"></a>

### `ALIGN`

`macro` | [src/native/strassen_turbo.c:22](../../src/native/strassen_turbo.c#L22)

```c
#define ALIGN
```

---
