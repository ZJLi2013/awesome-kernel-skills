---
name: rotary-embedding-kernel
description: >-
  Optimize Rotary Position Embedding (RoPE) kernels in Triton for NVIDIA and AMD GPUs.
  Covers interleaved/sequential layouts, sincos precomputation, and multi-row processing.
  Use when writing or optimizing RoPE, position embeddings, or rotary transformations.
---

# Rotary Embedding (RoPE) Kernel Optimization

## Overview

RoPE applies a rotation to pairs of elements: `(x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)`.
Applied per-position in the sequence dimension.

RoPE is **memory-bound** (4 reads + 2 writes per pair, minimal compute).
The key metric is **GBps**.

## Core Technique

### Paired Element Layout

Two common layouts:
1. **Interleaved**: `[x0, x1, x2, x3, ...]` where pairs are `(x0,x1), (x2,x3), ...`
2. **Split-half**: `[x_first_half | x_second_half]` where pairs are `(x[i], x[i+D/2])`

This template uses **interleaved** layout (common in LLaMA/GPT):
- Load with stride-2 indexing: `x1 = load(base + i*2)`, `x2 = load(base + i*2 + 1)`

### Multi-Row Processing

Process `ROWS_PER_PROG` rows per program for better occupancy on small head dims.

## Verification

```bash
python skills/kernels/rotary-embedding/test_rotary.py
```

## Common Pitfalls

1. **Wrong pair layout**: interleaved vs split-half must match the model's convention
2. **Broadcasting cos/sin**: cos/sin are `[seq_len, D/2]`, must broadcast over batch/heads
3. **On AMD**: `tl.math.tanh` not available -- not needed for RoPE but watch related kernels

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [AutoKernel RoPE Starter](https://github.com/RightNow-AI/autokernel)
