---
name: gemm-kernel-optimization
description: >-
  Optimize dense matrix multiplication (GEMM) kernels in Triton for NVIDIA and AMD GPUs.
  Covers tiled blocking, L2 cache grouping, tensor core utilization, and dual-platform autotune.
  Use when writing or optimizing matmul, linear layers, or any GEMM-based kernel.
---

# GEMM Kernel Optimization

## Overview

General Matrix Multiply (C = A @ B) is the most compute-intensive operation in deep learning.
GEMM is **compute-bound**: the key metric is **TFLOPS** (target: >60% of hardware peak).

| Bound | Metric | Target (H100) | Target (4090) | Target (MI300X) |
|-------|--------|---------------|---------------|-----------------|
| Compute | TFLOPS | >500 FP16 | >150 FP16 | >600 FP16 |

## Core Technique

### Algorithm: Tiled Blocked GEMM

1. Partition output C into `[BLOCK_M, BLOCK_N]` tiles, one per program instance
2. Accumulate in FP32: iterate K dimension in `BLOCK_K` chunks
3. Each iteration: load A tile `[BLOCK_M, BLOCK_K]`, B tile `[BLOCK_K, BLOCK_N]`, call `tl.dot`
4. Store result tile back to C (cast to output dtype)

### L2 Cache Grouping

Naive row-major block ordering causes excessive HBM reads. Group `GROUP_SIZE_M` rows of blocks
before moving to the next column to maximize L2 reuse of B tiles.

Empirically: GROUP_SIZE_M=8 gives 10-15% improvement on A100/H100.

### Tensor Core Utilization

- `tl.dot(a, b, acc)` maps to WMMA (Ampere) or WGMMA (Hopper) or MFMA (CDNA)
- Accumulator must be FP32 for accuracy
- BLOCK_K must be >= 16 for tensor core eligibility
- On MI300X, MFMA prefers BLOCK_K=32 with wavefront-64

## Autotune Configs

### NVIDIA (Ampere/Ada/Hopper)

```python
NVIDIA_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
]
```

### AMD MI300X (CDNA3, gfx942)

```python
AMD_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
]
```

Key AMD differences:
- Wavefront = 64 threads (not 32); `num_warps=4` means 256 threads
- `num_stages` must be >= 1 (0 crashes on HIP)
- `waves_per_eu` hint for occupancy tuning (2-4 compute-bound, 4-8 memory-bound)

## Verification

```bash
python skills/kernels/gemm/test_gemm.py
```

- Correctness: `torch.allclose(triton_result, torch.matmul(A, B), atol=1e-2, rtol=1e-2)` for FP16
- Performance: report TFLOPS = `2*M*N*K / latency_s / 1e12`

## Common Pitfalls

1. **Missing mask on K boundary**: when K is not divisible by BLOCK_K, unmasked loads read garbage
2. **Wrong accumulator dtype**: using FP16 accumulator causes catastrophic precision loss
3. **GROUP_SIZE_M=1**: disables L2 grouping, 10-20% perf regression on large matrices
4. **BLOCK_K too small on CDNA**: MI300X MFMA prefers BLOCK_K >= 32
5. **Forgetting `.to(output_dtype)` before store**: silent precision issues

## Agent Instructions

When optimizing a GEMM kernel:

1. Start with `triton_template.py` as baseline
2. Profile with NCU/rocprof to determine if memory or compute bound
3. If compute-bound and < 60% peak: check tensor core utilization, increase BLOCK_K
4. If memory-bound: check L2 hit rate, adjust GROUP_SIZE_M, consider split-K
5. Run `test_gemm.py` after every change to verify correctness + measure TFLOPS

## References

- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [AutoKernel program.md Tier 1](https://github.com/RightNow-AI/autokernel)
- [CUTLASS: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
