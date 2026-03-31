---
name: gpu-opt-tier2-memory-hierarchy
description: >-
  Optimizes global, shared/LDS, and cache behavior for GPU kernels: coalescing, pipelining,
  bank conflicts, vector loads, and register vs shared tradeoffs. Use when NCU/rocprof shows
  low memory throughput, poor coalescing, or L2/LDS pressure.
---

# Tier 2: Memory Hierarchy Optimization

**AutoKernel Tier 2** focus: after launch geometry (Tier 1), maximize bytes moved usefully per cycle across HBM → L2 → L1/LDS → registers.

## Global memory: coalescing

- Prefer **contiguous threads** accessing **consecutive addresses** (stride-1 along the fastest varying index in global load/store).
- **Vectorized loads** (`float4`, `uint4`, Triton `block_ptr` with wide `block_shape`): fewer instructions, better line utilization when aligned.
- **NVIDIA**: aim for **128-byte** sector-friendly access patterns per warp where possible.
- **AMD**: **256-byte** granularity is often optimal for full-line utilization on global loads; verify with rocprof transaction counters.

## Software pipelining

- **NVIDIA Ampere+**: `cp.async` from global to shared; **Hopper**: **TMA** bulk tensor loads if using supported APIs/kernels.
- Use **multi-buffer SMEM** (ties to `num_stages` in Triton) to overlap load with math.
- **AMD**: explicit prefetch via buffer lines or unrolled loads + LDS; pipeline depth co-tuned with register pressure.

## Shared memory / LDS: banks and layout

- **NVIDIA**: **32 banks**, **4-byte** width; **stride-32** FP32 columns → bank conflicts. Mitigations:
  - **Pad inner dimension** (e.g. `+1` column) to skew addresses across banks.
  - **Swizzled / XOR layouts** for MMA-friendly tiles (CUTLASS/Triton patterns).
- **AMD LDS**: on the order of **~64 KB per CU** (architecture-dependent); treat as **scarce** — spill to registers or shrink tiles if occupancy collapses.

## L2-aware behavior

- **Tile ordering** (e.g. `GROUP_SIZE_M`) and **problem splitting** (Split-K, stream-K in Tier 4) change L2 reuse; pair Tier 2 vectorization with Tier 1 grouping.

## Register vs shared (scratchpad) tradeoff

- **More SMEM/LDS**: enables reuse and wider loads but can **drop max blocks/SM**.
- **More registers**: faster per-thread state but can **reduce occupancy**; profile spill.
- Prefer **SMEM for tile staging** and **regs for per-lane accumulators** in classic GEMM.

## NVIDIA bank-conflict mental model

- Address within warp: `bank = (offset_in_bytes / 4) % 32` for classic layouts; **32 lanes** hitting the same bank on the same cycle → serial replays.
- **Padding** `N_stride + 1` (FP32) is the minimal skew; larger pads sometimes help **double-buffer** pointer math stay conflict-free across stages.
- **Swizzle** maps logical `(row,col)` to physical addresses so MMA loads spread across banks (see CUTLASS/Triton shared layouts).

## Store coalescing and atomics

- **Global stores** follow the same contiguous-lane rule as loads; misaligned **partial tiles** at matrix edges often need **predicated narrow stores** rather than scalar scatter.
- **Atomics** break coalescing and hurt bandwidth; avoid in inner loops unless algorithmically required (e.g. some reductions).

## AMD LDS pressure signals

- If **VGPR spill** or **low wave count** appears when LDS grows, **shrink tile** or move stable data to **regs** / **global** with prefetch.
- **rocprof**: rising **LDS bank conflicts** or **stall_cycles** alongside LDS instructions flags layout issues analogous to NVIDIA bank replay.

## Agent Instructions

1. From profiling, classify **global vs L2 vs LDS** bottlenecks (sector efficiency, hit rates, LDS bank conflicts if exposed).
2. Fix **access patterns first** (stride, thread→address mapping), then add **vector widths** aligned to 16/128/256 B as vendor-appropriate.
3. If using pipelining, align **`num_stages`** with buffer counts and verify **bank conflict-free** SMEM/LDS layouts (pad or swizzle).
4. On **AMD**, assume **64-byte L1 line** behavior when reasoning about spatial locality; validate with **TCC/L2** counters.
5. Escalate to **AutoKernel Tier 3** if math dominates after memory patterns are clean.
