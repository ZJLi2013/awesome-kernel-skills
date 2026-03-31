---
name: cross-entropy-kernel
description: >-
  Optimize fused cross-entropy loss kernels in Triton for NVIDIA and AMD GPUs.
  Covers fused log-softmax + NLL, online log-sum-exp, and large vocabulary handling.
  Use when writing or optimizing cross-entropy, label-smoothed CE, or similar loss kernels.
---

# Cross-Entropy Kernel Optimization

## Overview

Fused cross-entropy: `loss = -log_softmax(logits)[target]` computed in a single kernel pass
per row, without materializing the full softmax output.

Cross-entropy is **memory-bound** for large vocabularies (30K-250K).
The key metric is **GBps** (only need to read logits once, write one scalar per row).

## Core Technique

### Fused Log-Softmax + NLL

Per-row computation:
1. Load row of logits into FP32
2. Find max: `m = max(logits)`
3. Compute `log_sum_exp = log(sum(exp(logits - m)))`
4. Loss = `-(logits[target] - m - log_sum_exp)`

Only the target index value is needed -- no need to compute full softmax.

### Large Vocabulary Handling

For vocab > 64K (e.g., LLaMA's 128K):
- BLOCK_SIZE = next_power_of_2(vocab) can be very large
- Consider chunked approach: process in BLOCK_SIZE chunks, accumulate online

## Verification

```bash
python skills/kernels/cross-entropy/test_cross_entropy.py
```

- Correctness: compare vs `torch.nn.functional.cross_entropy(logits, targets)`
- Performance: GBps = `(batch * vocab * elem_size) / latency / 1e9`

## Common Pitfalls

1. **Integer overflow in target indexing**: target must be int64 for large vocabularies
2. **Missing FP32 accumulation**: log-sum-exp in FP16 is numerically unstable
3. **Reduction to scalar**: remember to average over batch dimension

## References

- [Liger-Kernel Fused Cross Entropy](https://github.com/linkedin/Liger-Kernel)
- [AutoKernel Cross-Entropy Starter](https://github.com/RightNow-AI/autokernel)
