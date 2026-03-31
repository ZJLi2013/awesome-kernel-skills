"""
FlashAttention Triton Template -- Fused multi-head attention with online softmax.
Works on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

Usage:
    from skills.kernels.flash_attention.triton_template import flash_attention
    O = flash_attention(Q, K, V, causal=True)
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M_size, N_size,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(0)

    qo_offset = pid_z * stride_qz + pid_h * stride_qh
    kv_offset = pid_z * stride_kz + pid_h * stride_kh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M_size, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else N_size

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + kv_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_size, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        qk = tl.where(offs_n[None, :] < N_size, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_size, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < M_size)


def flash_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    causal: bool = True, sm_scale: float = None,
) -> torch.Tensor:
    """Triton FlashAttention forward pass. Q/K/V: [B, H, N, D]."""
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    Z, H, M_size, D = Q.shape
    _, _, N_size, _ = K.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)
    assert D in (16, 32, 64, 128, 256), f"Head dim {D} not supported (must be power of 2, 16-256)"

    BLOCK_M = 128 if D <= 64 else 64
    BLOCK_N = 64 if D <= 64 else 32
    num_warps = 4 if D <= 64 else 8

    grid = (triton.cdiv(M_size, BLOCK_M), H, Z)

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, M_size, N_size,
        sm_scale,
        IS_CAUSAL=causal,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=2,
    )
    return O
