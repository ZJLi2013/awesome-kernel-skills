---
name: softmax-kernel-optimization
description: >-
  Optimize fused softmax kernels in Triton for NVIDIA and AMD GPUs.
  Covers online max/sum reduction, multi-row processing, numerical stability, and warp-level patterns.
  Use when writing or optimizing softmax, log-softmax, or similar row-wise normalization kernels.
---

# Softmax Kernel Optimization

## Overview

Softmax: `out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`

Softmax is **memory-bound** for typical LLM shapes (large vocab, moderate batch).
The key metric is **GBps** (fraction of HBM peak bandwidth).

The fused kernel avoids 3 separate passes (max, exp-sum, normalize) by doing everything
in one program launch per row, keeping data in registers/shared memory.

## Core Technique

### Safe Softmax (Numerically Stable)

1. Load entire row into registers (FP32)
2. Find row maximum: `m = max(x)`
3. Compute shifted exponentials: `e = exp(x - m)`
4. Sum exponentials: `s = sum(e)`
5. Normalize: `out = e / s`

### Multi-Row Processing

For small row widths (< 2048), processing one row per program wastes GPU resources.
Process `NUM_ROWS_PER_PROGRAM` rows per program to improve occupancy.

Heuristic: `ROWS_PER_PROG = max(1, 4096 / n_cols)`

### BLOCK_SIZE Selection

`BLOCK_SIZE = next_power_of_2(n_cols)` -- must cover the entire row in one program.
For very large rows (>32K), consider multi-pass or split approaches.

## Autotune Notes

Softmax has fewer knobs than GEMM since it is a row-wise 1D kernel:
- `BLOCK_SIZE`: determined by `n_cols` (not a free parameter)
- `num_warps`: 4 for n_cols <= 2048, 8 for n_cols <= 8192, 16 for larger
- `NUM_ROWS_PER_PROGRAM`: 1-8 depending on row width

### AMD-Specific

- `num_warps=4` gives 256 threads (wavefront-64), sufficient for most row sizes
- `num_stages=2` (never use 0 on HIP)

## Verification

```bash
python skills/kernels/softmax/test_softmax.py
```

- Correctness: `torch.allclose(triton_result, torch.softmax(x, dim=-1), atol=1e-3, rtol=1e-3)`
- Performance: report GBps = `(read_bytes + write_bytes) / latency_s / 1e9`

## Common Pitfalls

1. **Skipping max subtraction**: `exp(x)` overflows for large logits in FP16/FP32
2. **FP16 reduction**: sum of exponentials can overflow; always accumulate in FP32
3. **BLOCK_SIZE < n_cols**: partial row load produces wrong results (must cover full row)
4. **Single row per program with tiny n_cols**: GPU occupancy collapse

## Agent Instructions

When optimizing a softmax kernel:

1. Confirm the kernel is memory-bound (GBps metric)
2. Match BLOCK_SIZE to row width (next_power_of_2)
3. Use multi-row processing for small rows
4. Profile memory throughput; target >60% of HBM peak
5. For very large vocabularies (>128K), consider chunked/online approaches

## References

- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Online Normalizer Calculation (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
