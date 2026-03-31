---
name: gpu-kernel-bottleneck-diagnosis
description: >-
  Classifies GPU kernel bottlenecks (memory, compute, latency) from profiling metrics and applies
  GEAK-style workload guidance: what to prefer, consider, or deprioritize. Use after NCU/rocprof
  or when choosing between fusion, tiling, and scheduling changes without a clear winner.
---

# Bottleneck Diagnosis (GEAK / Workload Guidance)

Integrates **GEAK-style `workload_guidance`** with the **[AutoKernel](https://github.com/RightNow-AI/autokernel) six-tier playbook**: diagnosis picks **which tier** to emphasize next.

## Three bottleneck types

### 1. Memory-bound

- **Signals**: **Roofline** memory regime: **HBM BW % of peak** tends **high** while **compute % of peak** stays **low**; low **arithmetic intensity**; load/store stalls dominate in timelines.
- **Guidance**: **Prefer** algorithmic **fusion** (fewer round-trips), **tiling** that improves **reuse** (L2 grouping, larger K-tiles), **vectorized loads**, **Split-K** only if it does not inflate traffic.
- **Deprioritize**: Blind **instruction shaving** when bytes dominate.

### 2. Compute-bound

- **Signals**: High **% peak TFLOPS** (or MFMA/TC active cycles), **VALU/MFMA** hot, memory counters sub-saturated.
- **Guidance**: **Prefer** **MFMA/WGMMA-friendly** shapes, **inner-loop** cleanup, **mixed precision** with safe accumulators, **fast math** where numerics allow.
- **Deprioritize**: Extra **fusion** that only saves launches but increases **register** pressure without BW win.

### 3. Latency-bound

- **Signals**: Many **short kernels**, low **occupancy**, pipeline bubbles, **launch overhead** visible in timeline; often **low duration** per kernel.
- **Guidance**: **Prefer** **kernel fusion**, **persistent** loops, **fewer synchronizations**, overlapping **comm + compute** in multi-GPU cases.
- **Deprioritize**: Heavy **Split-K** atomics or complex schedules before addressing **launch granularity**.

## Metrics-driven flowchart (roofline-style)

Use **measured** `% peak memory bandwidth` and `% peak FP16/FP8 TFLOPS` (or vendor counters):

```text
IF dram_bw_pct is high AND tflops_pct is low
   → memory-bound → Tiers 1–2 first, then fusion (Tier 3)
ELIF tflops_pct is high AND dram_bw_pct is moderate/low
   → compute-bound → Tier 3 math/MMA, then Tier 1 occupancy/reg tuning
ELIF both percentages are low AND many short kernels / low occupancy
   → latency-bound → Tier 4 persistence/fusion, reduce launches
ELSE
   → mixed / unclear → gather second metrics (L2 hit, TC active, wave occupancy)
```

## Priority framework (GEAK-style)

| Priority | Meaning | Example actions |
|----------|---------|-----------------|
| **Prefer First** | Highest ROI for this diagnosis | Memory-bound → L2 tiling + vector loads |
| **Consider** | Secondary, validate with profile | Compute-bound → fast math after MMA layout fixed |
| **Deprioritize** | Low ROI or harmful until root cause shifts | Latency-bound → micro-optimizing div/sqrt in cold code |

## Mapping to AutoKernel tiers

| Bottleneck | Primary tiers |
|------------|----------------|
| Memory | **1–2** (tile + hierarchy), **3** fusion |
| Compute | **1** occupancy, **3** math, **5** vendor MMA |
| Latency | **4** scheduling, **3** fusion, **multi-GPU** overlap |

## Agent Instructions

1. Ingest **metrics.json**-style fields (`dram_throughput_pct`, `sm_throughput_pct` / MFMA, `occupancy_pct`, kernel count, duration) from the profiling skill.
2. Apply the **flowchart**; label the workload **memory / compute / latency** with one paragraph of evidence.
3. Emit recommendations tagged **Prefer First**, **Consider**, **Deprioritize** aligned with GEAK workload guidance habits.
4. Point to the **next AutoKernel tier file** (tier1…tier5) matching the diagnosis; avoid jumping to Hopper-only features on AMD (and vice versa).
5. If metrics conflict, request **one extra counter pass** (e.g. L2 hit, TC active) before suggesting invasive Tier 4 rewrites.
