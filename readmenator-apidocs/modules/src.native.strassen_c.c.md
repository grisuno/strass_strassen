# `src/native/strassen_c.c`

> Source: [src/native/strassen_c.c:1](../../src/native/strassen_c.c#L1) | language `c` | coverage 60.0% | [API index](../index.md)

## Contents

- **Functions:** [`strassen_recursive`](#strassen_recursive), [`strassen_multiply`](#strassen_multiply), [`standard_multiply`](#standard_multiply)
- **Macros:** [`THRESHOLD`](#threshold)

## Call Graph

Solid arrows stay in this module, dashed arrows leave it.

```mermaid
flowchart LR
    n_src_native_strassen_c_c__strassen_multiply(["strassen_multiply"])
    n_src_native_strassen_c_c__strassen_recursive(["strassen_recursive"])
    n_src_native_strassen_c_c__strassen_multiply --> n_src_native_strassen_c_c__strassen_recursive
```

## Functions

<a id="strassen_recursive"></a>

### `strassen_recursive`

`function` | [src/native/strassen_c.c:65](../../src/native/strassen_c.c#L65)

```c
void strassen_recursive(float* C, float* A, float* B, int n)
```

Strassen recursive

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `C` | `float*` |  |  |
| `A` | `float*` |  |  |
| `B` | `float*` |  |  |
| `n` | `int` |  |  |

*Called by:* [`strassen_multiply`](#strassen_multiply)

---

<a id="strassen_multiply"></a>

### `strassen_multiply`

`function` | [src/native/strassen_c.c:164](../../src/native/strassen_c.c#L164)

```c
void strassen_multiply(float* C, float* A, float* B, int n)
```

Public API

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `C` | `float*` |  |  |
| `A` | `float*` |  |  |
| `B` | `float*` |  |  |
| `n` | `int` |  |  |

*Calls:* [`strassen_recursive`](#strassen_recursive)

---

<a id="standard_multiply"></a>

### `standard_multiply`

`function` | [src/native/strassen_c.c:169](../../src/native/strassen_c.c#L169)

```c
void standard_multiply(float* C, float* A, float* B, int n)
```

Standard multiply for comparison

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `C` | `float*` |  |  |
| `A` | `float*` |  |  |
| `B` | `float*` |  |  |
| `n` | `int` |  |  |

---

## Macros

<a id="threshold"></a>

### `THRESHOLD`

`macro` | [src/native/strassen_c.c:12](../../src/native/strassen_c.c#L12)

```c
#define THRESHOLD
```

---
