---
name: gpu-opt-tier1-block-tiling
description: >-
  Chooses block/tile dimensions, warp counts, pipeline stages, and L2-aware block grouping
  for GPU kernels (Triton/CUDA/HIP). Use when autotuning matmul-like kernels, tuning occupancy,
  or aligning tile geometry with M/N/K and NVIDIA vs AMD thread models.
---

# Tier 1: Block, Tile, and Warp Configuration

This skill is **Tier 1** in [AutoKernel](https://github.com/RightNow-AI/autokernel)'s tiered optimization playbook: fix launch geometry before memory micro-optimizations.

## BLOCK_SIZE_M / N / K

- **Eligibility**: For tensor/MMA paths, `BLOCK_K` is often ≥16 (NVIDIA WMMA/WGMMA) or **≥32** (AMD MFMA on CDNA).
- **Square vs rectangular**: Wider `BLOCK_N` helps B reuse in row layouts; taller `BLOCK_M` helps A reuse. Balance against register/SMEM pressure.
- **Dimension-driven rules**:
  - **Small M or N** (e.g. `<128`): prefer smaller tiles (64×64, 64×128) so enough blocks exist to fill the GPU.
  - **Very large K**: larger `BLOCK_K` can amortize loads but raises SMEM/regs; validate occupancy.
  - **Skinny matrices**: prioritize the large dimension in tile width/height so warps stay busy.

## num_warps and num_stages

- **num_warps**: More warps hide latency but increase register footprint and can **lower occupancy**. Typical search: **2, 4, 8** on NVIDIA; on AMD, remember **wavefront-64** — `num_warps=4` ⇒ 256 threads per block.
- **num_stages**: Software-pipelined loads (cp.async, multi-buffer SMEM) need `num_stages≥2` on NVIDIA. On **HIP/ROCm**, **`num_stages` must be ≥1** (0 is invalid).
- **Coordination**: High `num_stages` + large tiles can spill registers; co-tune with `num_warps`.

## Occupancy tuning

- If **DRAM/L2 metrics are low** and warps are often stalled on memory, try **more warps** or smaller tiles first.
- If **compute units are underfed** and occupancy is already high, favor **larger tiles** or better instruction mix (later tiers).

## GROUP_SIZE_M (L2-aware tiling)

- Process **`GROUP_SIZE_M` consecutive block-rows** before advancing along N to reuse B slices in L2.
- Common search space: **4, 8, 16**; **8** is a strong default on Ampere/Hopper-class workloads.

## Typical autotune search spaces

| Vendor | BLOCK_M×N×K (examples) | num_warps | num_stages | Notes |
|--------|-------------------------|-----------|------------|--------|
| NVIDIA | 64–256 each axis, K often 32–64 | 2, 4, 8 | 2–5 | WGMMA kernels may need arch-specific caps |
| AMD | 64–256, **K often 32** | 2, 4, 8 (×64 threads/warp) | 1–3 | Add `waves_per_eu` (e.g. 2–8) when supported |

## AMD: wavefront-64

- One warp = **64** threads. Tile shapes that are multiples of **64** along the thread-dependent axis often map more cleanly.
- Pair with **`waves_per_eu`** hints: **2–4** when compute-bound, **4–8** when memory-bound (empirical starting points).

## Triton-oriented defaults (starting points)

- NVIDIA GEMM-style grids often center on **128×256×64**, **128×128×32**, **64×128×32** with **`GROUP_SIZE_M=8`**; shrink M/N when grid is too coarse.
- AMD CDNA grids often center on **128×128×32** or **256×128×32** with **`num_warps` 4 or 8**; always include at least one **64×64×32** candidate when occupancy is poor.
- Keep **total threads per block** compatible with hardware limits (e.g. 1024 max); on AMD, **multiples of 64** per warp axis reduce tail divergence in loads.

## Dividers, padding, and grid coverage

- If **M, N, or K** are not divisible by tile sizes, ensure **masks** or **epilogue tiles**; autotune should still evaluate **rounded-down** main path configs.
- Prefer **BLOCK_* divisors** that match **MMA instruction shapes** (e.g. 16/32 multiples) over arbitrary primes unless sparsity or fusion forces otherwise.

## Relationship to other skills

- Pairs with **gemm-kernel-optimization** for concrete config tuples; this file is the **decision logic** behind those tuples.
- After Tier 1 sweeps plateau, hand off to **tier2-memory-access** if loads are sector-inefficient or L2 metrics are poor.

## Agent Instructions

1. Read problem sizes **M, N, K** and pick a **tile scale** (small M/N → smaller blocks).
2. Build a **compact config grid**: 3–6 `(BLOCK_M, BLOCK_N, BLOCK_K)` candidates × `num_warps` × `num_stages` (respect HIP `num_stages≥1`).
3. Always sweep **`GROUP_SIZE_M`** (e.g. 4/8/16) for GEMM-like kernels with large inner dimension.
4. After a winner emerges, confirm **occupancy and register/SMEM limits** before locking configs.
5. Cross-check deeper tuning against **AutoKernel Tier 2+** (memory hierarchy) if bandwidth or bank conflicts dominate.
