"""
Fused MoE Triton Template -- Simplified grouped GEMM with top-k routing.
Works on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

This is a simplified standalone version (no vllm/sglang custom ops dependency).
It implements the core fused MoE GEMM pattern with pure PyTorch routing.

Usage:
    from skills.kernels.fused_moe.triton_template import fused_moe_simple
    out = fused_moe_simple(hidden, w, gating_output, topk=2)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_kernel(
    A_ptr, B_ptr, C_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    b_ptrs = B_ptr + off_expert * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MUL_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * moe_weight[:, None]

    acc = acc.to(C_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _moe_align_block_size_py(topk_ids, block_size, num_experts):
    """Pure-Python MoE block alignment (no custom C++ ops needed)."""
    M, top_k = topk_ids.shape
    max_num_tokens_padded = M * top_k + num_experts * (block_size - 1)

    sorted_ids = torch.full((max_num_tokens_padded,), M * top_k, dtype=torch.int32, device=topk_ids.device)
    expert_ids_out = torch.zeros(max_num_tokens_padded // block_size + 1, dtype=torch.int32, device=topk_ids.device)

    flat_ids = topk_ids.view(-1)
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for i in range(flat_ids.numel()):
        tokens_per_expert[flat_ids[i]] += 1

    cumsum = 0
    expert_start = {}
    for e in range(num_experts):
        expert_start[e] = cumsum
        padded = ((tokens_per_expert[e].item() + block_size - 1) // block_size) * block_size
        expert_ids_out[cumsum // block_size: (cumsum + padded) // block_size] = e
        cumsum += padded

    num_tokens_post_padded = torch.tensor([cumsum], dtype=torch.int32, device=topk_ids.device)

    write_pos = {e: expert_start[e] for e in range(num_experts)}
    for i in range(flat_ids.numel()):
        e = flat_ids[i].item()
        sorted_ids[write_pos[e]] = i
        write_pos[e] += 1

    return sorted_ids, expert_ids_out[:cumsum // block_size], num_tokens_post_padded


def fused_moe_simple(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int = 2,
    mul_routed_weight: bool = True,
) -> torch.Tensor:
    """
    Simplified fused MoE: gating + grouped GEMM in one pass.
    hidden_states: [M, K], w: [E, N, K], gating_output: [M, E] (pre-softmax)
    Returns: [M, N]
    """
    M, K = hidden_states.shape
    E, N, K2 = w.shape
    assert K == K2

    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    BLOCK_M = 64 if M > 32 else 16
    sorted_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size_py(topk_ids, BLOCK_M, E)

    out = torch.zeros(M * topk, N, device=hidden_states.device, dtype=hidden_states.dtype)

    config = {"BLOCK_M": BLOCK_M, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    grid = lambda META: (
        triton.cdiv(sorted_ids.shape[0], META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_moe_kernel[grid](
        hidden_states, w, out,
        topk_weights.view(-1), sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, topk_ids.numel(),
        hidden_states.stride(0), hidden_states.stride(1),
        w.stride(0), w.stride(2), w.stride(1),
        out.stride(0), out.stride(1),
        top_k=topk,
        MUL_WEIGHT=mul_routed_weight,
        **config,
    )

    out = out.view(M, topk, N).sum(dim=1)
    return out
