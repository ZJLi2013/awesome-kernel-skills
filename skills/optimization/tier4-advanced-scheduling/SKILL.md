---
name: gpu-opt-tier4-advanced-scheduling
description: >-
  Applies Split-K, persistent kernels, Stream-K/Stream-K++, warp specialization, and multi-GPU
  fused schedules. Use when simple tiling leaves tails, load imbalance, or excessive kernel
  launch overhead on large GPUs.
---

# Tier 4: Advanced Scheduling

**AutoKernel Tier 4** addresses **parallelism structure** beyond per-block tiling: how work maps to SMs over time and across chips.

## Split-K with reduction

- **Idea**: Multiple blocks partial-sum along K; **atomic** or **second-phase reduction** to finalize C.
- **Use when**: Increasing `BLOCK_K` alone overspills SMEM/regs or hurts occupancy, but more **K-parallelism** improves utilization.
- **Cost**: Extra **HBM traffic** for partials unless reduction is fused; watch **atomics** on small tiles.

## Persistent kernels

- **Idea**: Launch **~#SMs** blocks (or similar) that **loop internally** over many output tiles, reusing registers/SMEM across tiles.
- **Benefits**: Amortizes **launch latency**, improves **L2 locality** with controlled traversal order.
- **Risks**: Complex **grid-stride** indexing; harder debugging. Pair with **ordered tile scheduling** for cache affinity.

## Stream-K / Stream-K++

- **Problem**: Classic 2D tiling leaves **tail imbalance** (last waves idle).
- **Idea**: **Stream** fractional K contributions across blocks so SMs stay busy; **Stream-K++** refines reduction and scheduling for fewer barriers.
- **When**: Large rectangular GEMMs on **high SM-count** GPUs where occupancy is uneven.

## Warp specialization (Hopper-class)

- **Pattern**: **Producer warps** (TMA / async copy / transport) vs **consumer warps** (WGMMA / math).
- **Goal**: Overlap **bulk memory movement** with **tensor-core compute** without starving either side.
- **When**: Kernels already use **TMA/WGMMA** pipelines; not a first step for simple Triton dot kernels.

## Multi-GPU: compute + communication fusion

- **Overlap**: **NCCL/RCCL** collectives scheduled alongside **independent compute** on separate streams (or fused schedules in framework runtimes).
- **Fusion patterns**: **All-reduce + gradient GEMM** pipelining, **tensor-parallel** shards with **next-layer prefetch**.
- **Watch**: **Bubble** minimization dominates; profile both **network** and **GPU** timelines.

## Picking a strategy (quick matrix)

| Symptom | Try first | Watch for |
|---------|-----------|-----------|
| K too large for one tile’s SMEM | **Split-K** | Partial traffic + atomics |
| Huge grid, tiny per-kernel time | **Persistent** | Correctness of tile iterator |
| Classic 2D tail idle on big GPU | **Stream-K** | Reduction complexity |
| Hopper + TMA in play | **Warp specialization** | Producer/consumer deadlock |
| TP/PP pipeline bubbles | **Comm/compute overlap** | Ordering / race bugs |

## Interaction with AutoKernel Tier 6

- Some playbooks treat **tooling/automation** (search, DSL lowering) as a final tier; scheduling choices here should still be **expressed as reproducible config knobs** for autotuners.

## Agent Instructions

1. Only adopt Tier 4 after **Tier 1–3** are sane; confirm imbalance or launch overhead with a **timeline profiler** (NSYS/rocprof traces).
2. For **tail-heavy** matmuls, prototype **Stream-K** or **persistent** variants against a **Split-K** baseline; compare **effective TFLOPS** and **code complexity**.
3. On **Hopper**, if using advanced APIs, evaluate **warp-specialized** pipelines vs monolithic blocks using profiling counters for **TMA vs math overlap**.
4. For **multi-GPU**, map **critical path** (comm vs compute) and prefer **fused/overlapped** schedules where safe.
5. Align documentation and tuning notes with **AutoKernel Tier 5** (vendor-specific caps) before shipping production configs.
