---
name: fused-moe-kernel
description: >-
  Optimize Fused Mixture-of-Experts (MoE) kernels in Triton for NVIDIA and AMD GPUs.
  Covers token-expert routing, grouped GEMM, cache-aware scheduling, and top-k gating fusion.
  Use when writing or optimizing MoE layers, expert routing, or grouped matrix multiplication.
---

# Fused MoE Kernel Optimization

## Overview

Fused MoE combines token routing, expert GEMM, and output accumulation into a single kernel,
avoiding expensive scatter/gather operations in global memory.

MoE kernels are **compute-bound** for large expert dimensions and **latency-bound** for
small token counts per expert. The key metric is **TFLOPS**.

## Core Technique

### Token-Expert Routing

1. Top-k gating: select k experts per token (typically k=2)
2. Sort tokens by assigned expert
3. Pad each expert's token count to be divisible by BLOCK_SIZE_M

### Grouped GEMM Structure

The kernel is a standard tiled GEMM with an extra indirection:
- `sorted_token_ids` maps program blocks to actual token rows
- `expert_ids` maps each M-block to an expert (selects which weight matrix)
- Block-level early exit: skip if `pid_m * BLOCK_M >= num_tokens_post_padded`

### Optimization Patterns

- **L2 cache grouping**: same as standard GEMM (GROUP_SIZE_M)
- **Persistent cache-aware**: for many experts, persistent kernel with tile scheduling
- **Compute + comm fusion**: overlap expert compute with all-to-all communication

## Key Parameters

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| BLOCK_SIZE_M | 16, 32, 64, 128 | 16 for small token counts |
| BLOCK_SIZE_N | 32, 64, 128 | Expert hidden dim |
| BLOCK_SIZE_K | 32, 64, 128 | Input feature dim |
| GROUP_SIZE_M | 1, 4, 8 | L2 grouping |
| top_k | 1, 2 | Experts per token |

Small-M heuristic: when `M <= num_experts`, use `BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, GROUP_M=1`.

## Verification

```bash
python skills/kernels/fused-moe/test_fused_moe.py
```

- Correctness: compare vs sequential per-expert `torch.matmul` with top-k selection
- Performance: TFLOPS = `2 * M * top_k * N * K / latency / 1e12`

## Common Pitfalls

1. **Incorrect token ID mapping**: `sorted_token_ids // top_k` maps back to original token
2. **Padding tokens**: padded IDs must be >= num_valid_tokens for masking
3. **Expert weight layout**: typically `[E, N, K]` (expert-major), not `[E, K, N]`
4. **Routed weight multiplication**: must apply `topk_weights` to correct output rows

## References

- [Accelerating MoEs with Persistent Cache-Aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [vLLM Fused MoE Kernel](https://github.com/vllm-project/vllm)
- [SGLang Fused MoE](https://www.emergentmind.com/topics/fused-moe-triton-kernel)
