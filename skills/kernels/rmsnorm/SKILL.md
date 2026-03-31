---
name: rmsnorm-kernel-optimization
description: >-
  Optimize RMS Normalization kernels in Triton for NVIDIA and AMD GPUs.
  Covers row-parallel reduction, vectorized loads, warp shuffle patterns, and rsqrt usage.
  Use when writing or optimizing RMSNorm, LayerNorm, or similar normalization kernels.
---

# RMSNorm Kernel Optimization

## Overview

RMS Normalization: `out = x / sqrt(mean(x^2) + eps) * weight`

RMSNorm is **memory-bound**: each element is read once and written once, with minimal compute.
The key metric is **GBps** (fraction of HBM peak bandwidth).

| Bound | Metric | Target (4090) | Target (MI300X) |
|-------|--------|---------------|-----------------|
| Memory | % peak BW | >60% | >60% |

AutoKernel reported ~34.7% of H100 peak bandwidth for a baseline RMSNorm.

## Core Technique

### Row-Parallel Design

One Triton program per row (token). Each program:
1. Loads the entire row into registers (FP32 for numerical stability)
2. Computes sum of squares via `tl.sum(x * x)`
3. Applies `rsqrt(mean_sq + eps)` -- single fast instruction
4. Multiplies by weight vector
5. Stores result

### Key Optimizations

- **FP32 accumulation**: always compute reduction in float32, even for FP16/BF16 inputs
- **Vectorized loads**: BLOCK_SIZE should be >= hidden_dim, next_power_of_2 for alignment
- **`rsqrt` intrinsic**: `1/sqrt(x)` is a single HW instruction on both NVIDIA and AMD
- **Multi-row per program**: for small hidden dims, process multiple rows per program to improve occupancy

### NVIDIA vs AMD Differences

- AMD wavefront=64: fewer programs needed for same thread count
- MI300X LDS = 64 KB/CU (vs 228 KB/SM on H100): less shared memory, but 256 MB L2
- Both: BLOCK_SIZE should be power of 2, >= hidden_dim

## Verification

```bash
python skills/kernels/rmsnorm/test_rmsnorm.py
```

- Correctness: compare vs `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight`
- Performance: report GBps = `(read_bytes + write_bytes) / latency_s / 1e9`

## Common Pitfalls

1. **FP16 reduction**: computing `sum(x^2)` in FP16 overflows for large hidden dims
2. **Missing mask**: when hidden_dim is not power of 2, unmasked loads cause errors
3. **Separate sqrt + divide**: use `rsqrt` instead of `sqrt` + `divide` (2x fewer instructions)
4. **Large BLOCK_SIZE on AMD**: LDS is only 64 KB/CU; keep register pressure manageable

## Agent Instructions

When optimizing a RMSNorm kernel:

1. Profile to confirm memory-bound (GBps should be the focus metric)
2. Ensure FP32 accumulation for the reduction
3. Try multi-row per program for small hidden dims (< 1024)
4. Check vectorized load efficiency with NCU/rocprof
5. Run test script after every change

## References

- [AutoKernel RMSNorm experiments](https://github.com/RightNow-AI/autokernel)
- [Triton Fused Softmax Tutorial (reduction pattern)](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
