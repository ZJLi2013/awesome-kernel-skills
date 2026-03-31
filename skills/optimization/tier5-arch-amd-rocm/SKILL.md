---
name: gpu-opt-tier5-amd-rocm
description: >-
  AMD ROCm/CDNA3 tuning: MI300X topology, LDS/L2, wavefront-64 and Triton num_warps, MFMA,
  HIP/Triton limitations, rocprof counters, and coalescing. Use when optimizing for gfx942/MI300
  or comparing against NVIDIA H100 baselines.
---

# Tier 5 (AMD ROCm): CDNA3 / MI300-Class Optimization

**AutoKernel Tier 5** vendor fork for **ROCm** and **CDNA**.

## MI300X (representative specs, verify for SKU)

| Quantity | Typical class |
|----------|----------------|
| CUs | **304** (full die class) |
| HBM bandwidth | **~5.3 TB/s** (spec tier) |
| HBM capacity | **192 GB** HBM3 class |
| LDS per CU | **~64 KB** (architecture guidance) |
| L2 | **~256 MB** shared |

## Wavefront-64 and Triton

- **One warp = 64 threads**. `num_warps=4` ⇒ **256** threads/block (common baseline).
- Map **lane layouts** so contiguous lanes hit contiguous memory for **global coalescing**.

## Occupancy hints: `waves_per_eu`

- **Compute-heavy**: often start **2–4** waves/EU.
- **Memory-heavy**: try **4–8** waves/EU to hide latency (empirical; co-tune with LDS footprint).

## MFMA matrix cores

- Prefer **MFMA-friendly** `BLOCK_K` (commonly **32** for FP16-class) and layouts that compilers can lower to **matrix fragments**.
- Profile **SQ_ACTIVE_INST_MFMA** vs **VALU** to confirm tensor path use.

## HIP / Triton practical limitations

- **`num_stages >= 1`** on HIP (stage-0 patterns can fail).
- Some math: **`tl.math.tanh`** (and similar) may be missing or require **polyfill**—check Triton version and ROCm support matrix.

## Memory: coalescing and lines

- **256-byte** optimal global transactions are a useful mental model for **full-line** utilization; align leading dimensions when possible.
- **64-byte L1** line behavior informs **spatial locality** next to vector width.

## rocprof (counters mindset)

- Track **TCC** (L2) hits/misses, **SQ** activity, **MFMA** ops, **VALU** busy.
- Derive **MFMA efficiency** and **memory BW utilization** similar to the profiling skill; compare against **5.3 TB/s** peak class.

## vs H100 (orientation table)

| Aspect | H100 (Hopper) | MI300X (CDNA3) |
|--------|----------------|----------------|
| Thread warp | 32 | **64** |
| Matrix API | WGMMA + TMA | **MFMA** |
| Peak BW / FLOPS | SKU-specific | **HBM3 + MFMA** class |
| Tools | NCU, NSYS | **rocprof**, ROCm SMI |
| Approximate FLOPS/BW class | H100 HBM3 + Hopper TC | **304 CU** + **5.3 TB/s** + **MFMA** class |

## Kernel agent integration

- When generating HIP/Triton, cross-check **gfx target**, **ROCm version**, and **Triton support** before emitting MFMA-dependent tile sizes.
- Prefer **measured** `rocprof`/`rocprofv2` stats over static caps when advising “peak” percentages in diagnosis.

## Agent Instructions

1. Confirm **GPU name / gfx** (`rocminfo`) and **ROCm + Triton** versions before promising intrinsics.
2. Build autotune grids with **`BLOCK_K=32`** prominent; tune **`num_warps`** on a **64-thread warp** basis.
3. Set **`waves_per_eu`** in the memory-vs-compute direction indicated by rocprof.
4. Replace unsupported **Triton math** with **handwritten** or **torch** fallbacks when required.
5. For cross-vendor claims, cite **measured** NCU vs rocprof rooflines, not spec sheets alone.
