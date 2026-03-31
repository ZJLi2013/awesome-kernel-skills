# Experiment Log

All experiments follow the [experiment-driven-doc](https://github.com/RightNow-AI/autokernel) workflow:
hypothesis -> design -> execute -> results -> analysis -> next step.

## Experiment Overview

| Exp | Kernel(s) | Platform | Status | Key Result | Conclusion |
|-----|-----------|----------|--------|------------|------------|
| W1-G | GEMM | 4090 + MI300X | done | 168.7 / 94.4 TFLOPS | Correct; AMD needs wider autotune |
| W1-R | RMSNorm | 4090 + MI300X | done | 2482 / 876 GBps | Fused kernel >> PyTorch reference |
| W1-S | Softmax | 4090 + MI300X | done | 1747 / 700 GBps | Correct; AMD PyTorch has optimized softmax |
| W2-FA | FlashAttention | 4090 + MI300X | done | PASS / PASS | Online softmax correct on both |
| W2-CE | CrossEntropy | 4090 + MI300X | done | PASS / PASS | Fused log-softmax+NLL correct |
| W2-RO | RotaryEmbedding | 4090 + MI300X | done | PASS / PASS | Interleaved RoPE correct |
| W3-MOE | FusedMoE | 4090 + MI300X | done | PASS / PASS | Grouped GEMM with pure-Python routing correct |
| **OPT-GEMM** | GEMM optimize-loop | 4090 (MI300X deferred) | **done** | [see detail](gemm_optimization.md) | 5 iters: beats cuBLAS at 2048+ (1.03-1.14x), 1024 gap is Triton limit |

## Final Cross-Platform Summary (Polish Pass)

**NVIDIA RTX 4090** (PyTorch NGC 25.02, Triton bundled): **7/7 passed**, 0 skipped
**AMD MI300X** (ROCm 6.4, PyTorch 2.6.0): **7/7 passed**, 0 skipped

All 7 Triton kernel templates work correctly on both NVIDIA (CUDA) and AMD (ROCm/HIP).

### Fixes Applied During Polish
1. Softmax: `return` inside `for` loop unsupported on ROCm Triton -- replaced with conditional mask
2. RMSNorm: output now cast to input dtype (was storing float32)
3. Softmax: row bounds check added for multi-row processing edge cases
4. detect_gpu.py: Ada Lovelace (SM89) properly classified
5. Benchmark metrics: `speedup` ratio standardized (higher = Triton faster)

---

## Exp-W1-G: GEMM (Wave 1)

### Hypothesis
Tiled blocked GEMM with L2 cache grouping and dual NVIDIA/AMD autotune achieves >60% peak TFLOPS on both platforms and passes correctness.

### Results

| GPU | Triton TFLOPS | PyTorch TFLOPS | Ratio | Correctness | Shapes Tested |
|-----|--------------|----------------|-------|-------------|---------------|
| RTX 4090 | 168.68 | 166.33 | 1.014x | PASS (4/4) | 128-2048 |
| MI300X | 94.35 | 150.61 | 0.626x | PASS (4/4) | 128-2048 |

### Analysis
- **4090**: On par with cuBLAS. L2 grouping + autotune working well.
- **MI300X**: 62.6% of rocBLAS. The autotune search space for AMD is smaller (5 configs vs 7 for NVIDIA). Expanding BLOCK_K and adding `waves_per_eu` variants would likely improve this.

### Next Step
- Expand AMD autotune configs
- Consider persistent kernel / stream-K for large matrices

---

## Exp-W1-R: RMSNorm (Wave 1)

### Hypothesis
Row-parallel fused RMSNorm achieves better bandwidth utilization than naive PyTorch.

### Results

| GPU | Triton GBps | PyTorch GBps | Speedup | Correctness |
|-----|-------------|--------------|---------|-------------|
| RTX 4090 | 2482.4 | 260.3 | 9.5x | PASS (4/4) |
| MI300X | 875.8 | 322.3 | 2.7x | PASS (4/4) |

### Analysis
- **4090**: 9.5x speedup shows the power of kernel fusion (single pass vs 3 ops).
- **MI300X**: 2.7x speedup. GBps of 876 is ~16.5% of MI300X peak -- room for improvement.

---

## Exp-W1-S: Softmax (Wave 1)

### Hypothesis
Fused single-pass softmax with multi-row processing outperforms PyTorch's built-in softmax.

### Results

| GPU | Triton GBps | PyTorch GBps | Speedup | Correctness |
|-----|-------------|--------------|---------|-------------|
| RTX 4090 | 1746.9 | 1108.3 | 1.58x | PASS (5/5) |
| MI300X | 700.1 | 848.4 | 0.82x | PASS (5/5) |

### Analysis
- **4090**: 1.58x speedup. Multi-row processing effective.
- **MI300X**: 0.82x -- PyTorch ROCm's optimized softmax is faster. Needs AMD-specific tuning.

---

## Exp-W2-FA: FlashAttention (Wave 2)

### Results

| GPU | Correctness | Shapes Tested |
|-----|-------------|---------------|
| RTX 4090 | PASS (4/4) | B1-2, H4-8, N128-1024, D64-128 |
| MI300X | PASS (4/4) | Same configs |

### Analysis
Online softmax + tiled QK/AV pattern works correctly on both platforms. Causal masking verified.

---

## Exp-W2-CE: CrossEntropy (Wave 2)

### Results

| GPU | Correctness | Shapes Tested |
|-----|-------------|---------------|
| RTX 4090 | PASS (4/4) | B32-128, V1000-128256 |
| MI300X | PASS (4/4) | Same configs |

### Analysis
Fused log-softmax + NLL correct for vocabulary sizes up to 128K.

---

## Exp-W2-RO: RotaryEmbedding (Wave 2)

### Results

| GPU | Correctness | Shapes Tested |
|-----|-------------|---------------|
| RTX 4090 | PASS (4/4) | seq128-1024, dim64-128 |
| MI300X | PASS (4/4) | Same configs |

---

## Exp-W3-MOE: FusedMoE (Wave 3)

### Results

| GPU | Correctness | Shapes Tested |
|-----|-------------|---------------|
| RTX 4090 | PASS (3/3) | M16-64, E4-8, N64-256, K32-128 |
| MI300X | PASS (3/3) | Same configs |

### Analysis
Pure-Python routing + Triton grouped GEMM kernel works correctly. The routing overhead (CPU-side Python) is the bottleneck for production use; would need C++/CUDA custom ops for high-throughput inference.
