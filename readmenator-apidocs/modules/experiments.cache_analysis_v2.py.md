# `experiments.cache_analysis_v2`

> Source: [experiments/cache_analysis_v2.py:1](../../experiments/cache_analysis_v2.py#L1) | language `py` | coverage 100.0% | [API index](../index.md)

Corrected cache analysis including model parameters and optimizer state.

## Contents

- **Functions:** [`cache_analysis`](#cache_analysis)

## Functions

<a id="cache_analysis"></a>

### `cache_analysis`

`function` | [experiments/cache_analysis_v2.py:7](../../experiments/cache_analysis_v2.py#L7)

```python
def cache_analysis()
```

Full memory analysis for training.

Model: Bilinear with rank-8 (target 7 active)
- U: 8 x 4 = 32 floats
- V: 8 x 4 = 32 floats
- W: 4 x 8 = 32 floats
Total params: 96 floats = 384 bytes

AdamW optimizer state (per param):
- m (momentum): 96 floats
- v (variance): 96 floats
Total optimizer: 192 floats = 768 bytes

Per-batch memory:
- Input batch: B x 8 floats (A, B flattened)
- Hidden: B x 8 floats (intermediate M)
- Output: B x 4 floats (C)
- Gradients: same as forward
Total per-sample: ~40 floats x 2 (forward+backward) = 80 floats = 320 bytes

Total training memory = model + optimizer + batch

---
