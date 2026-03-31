---
name: kernel-benchmark
description: >-
  Unified kernel benchmarking protocol producing JSON results with latency, TFLOPS, GBps,
  and comparison against PyTorch baselines. Covers warmup, timing, and cross-platform reporting.
  Use when benchmarking kernels, measuring performance, or comparing implementations.
---

# Kernel Benchmark Skill

## Overview

Consistent benchmarking methodology across all kernels and platforms.
Every benchmark produces a JSON result for automated tracking.

## Protocol

### 1. Warmup

Run the kernel `n_warmup` times (default: 25) to:
- Trigger JIT compilation (Triton autotune)
- Warm up GPU caches and frequency scaling
- Call `torch.cuda.synchronize()` after warmup

### 2. Timed Runs

```python
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_iter):  # default: 100
    kernel(inputs)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / n_iter
```

### 3. Metrics Computation

| Kernel Type | Primary Metric | Formula |
|-------------|---------------|---------|
| Compute-bound (GEMM, attention) | TFLOPS | `flops / elapsed / 1e12` |
| Memory-bound (norm, softmax, RoPE) | GBps | `total_bytes / elapsed / 1e9` |

FLOPS formulas:
- GEMM: `2 * M * N * K`
- Attention (fwd): `4 * B * H * N * N * D` (2 GEMMs)
- Softmax/Norm: N/A (use GBps)

Bytes formulas:
- Softmax: `rows * cols * elem_size * 2` (read + write)
- RMSNorm: `rows * cols * elem_size * 3` (read x, read w, write out)

### 4. Baseline Comparison

Always compare against PyTorch:
```python
vs_pytorch = triton_metric / pytorch_metric
```

### 5. Output Format

```json
{
  "kernel": "gemm",
  "gpu": "NVIDIA GeForce RTX 4090",
  "vendor": "nvidia",
  "correctness": true,
  "latency_us": 101.9,
  "tflops": 168.68,
  "gbps": null,
  "vs_pytorch": 1.014,
  "dtype": "float16",
  "shape": "2048x2048x2048",
  "status": "pass"
}
```

## Agent Instructions

When benchmarking:

1. Run correctness check first (never report perf for incorrect kernels)
2. Use enough warmup iterations for autotune to settle (25-50)
3. Use enough timed iterations for stable results (100-200)
4. Always `torch.cuda.synchronize()` before and after timing
5. Report both absolute metrics and vs-PyTorch ratio
6. Save JSON to `experiments/` for tracking
