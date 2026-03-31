---
name: flash-attention-kernel
description: >-
  Optimize FlashAttention-style fused attention kernels in Triton for NVIDIA and AMD GPUs.
  Covers online softmax, tiled QK/AV GEMM, causal masking, and memory-efficient attention.
  Use when writing or optimizing self-attention, cross-attention, or any QKV attention kernel.
---

# FlashAttention Kernel Optimization

## Overview

FlashAttention fuses the entire attention computation (Q@K^T -> softmax -> @V) into a single
kernel, avoiding materialization of the full N x N attention matrix in HBM.

For typical LLM shapes, attention is **IO-bound** (dominated by HBM reads of Q/K/V).
The key metrics are **TFLOPS** and **memory savings** (O(N) instead of O(N^2)).

## Core Technique

### Online Softmax (Milakov & Gimelshein)

The key insight: compute softmax incrementally as K/V tiles stream through:
1. Maintain running `m_i` (row max) and `l_i` (exp sum) per query row
2. For each K tile: compute `qk = Q @ K^T`, update max, rescale accumulator
3. After all K tiles: divide accumulator by final `l_i`

This avoids two passes over the attention matrix.

### Tiled Computation

- Outer loop: Q tiles of `[BLOCK_M, D]`
- Inner loop: K/V tiles of `[BLOCK_N, D]`
- Two GEMMs per inner iteration: `QK = Q @ K^T` and `acc += softmax(QK) @ V`
- Both GEMMs use tensor cores (WMMA/WGMMA/MFMA)

### Causal Masking

For causal (autoregressive) attention:
- Early termination: skip K/V tiles where all positions are masked
- `kv_end = min(N, (pid_m + 1) * BLOCK_M)` for causal
- Apply `where(causal_mask, qk, -inf)` for partial tiles

## Autotune Notes

| Parameter | Typical Values |
|-----------|---------------|
| BLOCK_M | 64, 128 |
| BLOCK_N | 32, 64 |
| D (head dim) | 64, 128 (constexpr) |
| num_warps | 4 (D<=64), 8 (D>64) |
| num_stages | 2 |

AMD: same structure but `num_warps` maps to wavefronts of 64.

## Verification

```bash
python skills/kernels/flash-attention/test_flash_attention.py
```

- Correctness: compare vs `torch.nn.functional.scaled_dot_product_attention`
- Performance: TFLOPS = `4 * B * H * N * N * D / latency / 1e12` (2 GEMMs, fwd only)

## Common Pitfalls

1. **Missing online softmax rescaling**: forgetting `acc *= alpha[:, None]` when max updates
2. **Wrong causal mask direction**: `offs_m >= offs_n` not `offs_m > offs_n`
3. **D not constexpr**: head dimension must be known at compile time for tensor core tiling
4. **`tl.trans` on AMD**: works but may need explicit `.contiguous()` on input K

## References

- [FlashAttention: Fast and Memory-Efficient Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
