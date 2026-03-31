---
name: gpu-opt-tier3-compute-fusion
description: >-
  Tightens inner-loop math, mixed precision, fused epilogues, and fast math intrinsics for GPU
  kernels. Use when kernels are compute-bound, epilogue-bound, or dominated by elementwise ops
  around matmul-like cores.
---

# Tier 3: Compute Optimization and Fusion

**AutoKernel Tier 3**: reduce instructions per useful flop and fuse memory round-trips into the same kernel pass.

## Mixed precision

- Use **FP16/BF16/FP8** for inputs where acceptable; keep **FP32 (or wider) accumulators** for reductions and matmul (`tl.dot` acc) to preserve accuracy.
- Match **layout and instruction** (e.g. tensor/MMA width) to the narrow type actually used.

## Fused epilogue patterns

Fuse into the same kernel to avoid extra global traffic:

| Pattern | When it matters |
|---------|------------------|
| **Bias** (row/col) | LLM linear layers, conv fc |
| **Scale / per-channel scale** | quantization, norm fusion |
| **Activation** (GELU, ReLU, clamp) | MLP blocks |
| **Quantize to int8/fp8** | inference export |

**Rule**: if the epilogue is **memory-bandwidth limited**, fusion is high ROI; if already **compute-saturated**, fusion still saves launches but may need register tuning.

## Inner loop hygiene

- **Hoist invariants** out of the K (inner) loop; keep hot paths branch-free.
- **Reduce branches** on lanes: predication and masks beat divergent `if` in inner loops.
- **Unroll** only where it hides latency without blowing register budget.

## Fast math intrinsics (CUDA/HIP style)

| Intrinsic / pattern | Typical use | Caveat |
|---------------------|-------------|--------|
| `__expf`, `__exp10f` | softmax, activations | lower precision vs libm |
| `rsqrtf`, `__frsqrt_rn` | norm, attention | acceptable for ML norms with care |
| `__sincosf` | rotary, trig | pair eval saves bandwidth |
| `__fdividef` | non-critical divisions | faster, less IEEE |

**When each matters**: **transcendentals** dominate in softmax/attention/layer-norm; **fdivide** helps cheap color-space or scaling loops. Prefer **intrinsics in hot paths**, library precision on **numerically sensitive** reductions.

## Compiler and framework flags (orientation)

- CUDA **`--use-fast_math`** (or equivalent) globally enables aggressive approximations; prefer **scoped intrinsics** when you need **deterministic** training behavior.
- In Triton, prefer **explicit casts** and **`tl.math`** / libdevice-style ops that match the desired **IEEE vs fast** semantics for the pass.

## When *not* to fuse

- If the epilogue is **already bandwidth-light** and fusion **spills registers** causing occupancy collapse, measure **end-to-end** against a two-kernel sequence.
- If **different dtypes** or **layout transforms** force expensive **repack** in-register, a dedicated **transpose/reorder** kernel may win.

## Agent Instructions

1. Confirm **bottleneck is compute** (high % peak FLOPS, low memory util) before aggressive math shortcuts.
2. Identify **epilogue inputs/outputs**; if separate kernels exist between matmul and bias/act/quantize, plan a **single-store** fused path.
3. Audit the **innermost loop** for invariant loads and branches; refactor to **mask/predication**.
4. Select **fast math** only where error budgets allow; document accuracy tradeoffs for training vs inference.
5. If still bound by **launch or tail effects**, continue to **AutoKernel Tier 4** (scheduling).
