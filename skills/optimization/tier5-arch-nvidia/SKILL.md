---
name: gpu-opt-tier5-nvidia-arch
description: >-
  NVIDIA-specific caps and tuning: Hopper SM90 (WGMMA, TMA, cluster SMEM), Ampere SM80
  (cp.async, TF32, 2:4 sparsity), Ada SM89 consumer (4090-class). Use when targeting a known
  compute capability or interpreting NCU metrics for a specific chip.
---

# Tier 5 (NVIDIA): Architecture-Specific Optimization

**AutoKernel Tier 5** maps generic techniques to **SKU capabilities**. This file is the **NVIDIA** branch.

## Compute capability quick reference

| Arch | CC | Examples | Notes |
|------|-----|----------|--------|
| Ampere | **8.0** | A100 | TF32 TC, cp.async, 2:4 structured sparsity (sparse TC) |
| Ada Lovelace | **8.9** | RTX 4090 | **16384** "CUDA cores" (marketing aggregate), **~1008 GB/s** ref bandwidth class |
| Hopper | **9.0** | H100 | **WGMMA**, **TMA**, **cluster** SMEM, up to **~228 KB SMEM**/SM (config-dependent) |

*Exact SMEM/L1 splits and clocks vary by SKU; always verify in NCU "Launch Statistics" and occupancy calculator.*

## H100 / SM90 (Hopper)

- **WGMMA**: prefer **tensor-core-native** tile multiples; pair with **TMA** for async bulk loads.
- **TMA**: reduces load instruction overhead; needs **correct tensor descriptors** and alignment.
- **Cluster**: **distributed shared memory** across blocks in a cluster for multi-block cooperation.
- **Autotune tips**: sweep **cluster size**, **TMA vs cp.async** paths, **WGMMA K-dimension** tiling; watch **register** explosion on wide tiles.

## A100 / SM80 (Ampere)

- **cp.async**: pipeline global→shared; tune **`num_stages`** with occupancy.
- **TF32**: training default for many frameworks; know accuracy implications vs FP16/BF16 TC.
- **2:4 sparsity**: only wins when weights are **structured-sparse** and paths are enabled end-to-end.

## 4090 / Ada (SM89)

- High **FP16/BF16** throughput for consumer bandwidth; often **memory-bound** on large models despite high FLOPS.
- Prefer **vectorized loads**, **fusion**, and **L2-friendly** traversal; autotune similarly to Ampere but respect **smaller L2** vs datacenter parts in some metrics.

## Autotune workflow (NVIDIA)

1. Fix **Tier 1–2** (tiling + memory) before Hopper-only features.
2. Use **NCU** to separate **DRAM vs L1TEX vs TC** bottlenecks.
3. For Hopper, add **WGMMA/TMA** knobs only when baseline kernel already uses compatible abstractions (e.g. CUTLASS 3.x, cuBLASLt, or hand-written PTX).

## NCU metrics cheat sheet (orientation)

| Symptom in NCU | Likely lever |
|----------------|--------------|
| Low `sm__throughput` + low `dram__throughput` | Occupancy / launch / tail (Tier 1/4) |
| High DRAM, low TC active | Memory path / vectorization (Tier 2) |
| High TC, reg spills | Tile size / WGMMA shape / fewer live tensors |

## Blackwell / future SKUs

- Treat unseen **CC** values as **unknown**: fall back to **prior-gen patterns**, then re-measure; do not assume Hopper knobs map 1:1 without vendor release notes.

## Agent Instructions

1. Detect **CC** (`nvidia-smi`, NCU header) and load **SKU-specific peak** FLOPS/BW for roofline.
2. Choose instruction family: **WGMMA+TMA (9.0)** vs **WMMA/cp.async (8.x)** vs legacy paths.
3. Record **SMEM usage** per kernel; on Hopper, include **cluster** constraints if used.
4. Cross-link **GEMM/flash** skills for op-specific templates; keep this file for **capability gating** only.
5. Treat numbers like **16384 cores** and **1008 GB/s** as **marketing/spec sheet** anchors—always **measure** on the user's card.
