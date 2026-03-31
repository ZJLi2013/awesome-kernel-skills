"""
RMSNorm Triton Template -- Row-parallel RMS normalization.
Works on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

Usage:
    from skills.kernels.rmsnorm.triton_template import rmsnorm
    out = rmsnorm(x, weight, eps=1e-6)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    N,
    stride_x_row, stride_x_col,
    stride_o_row, stride_o_col,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0.0).to(tl.float32)

    sq_mean = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(sq_mean + eps)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * w

    tl.store(OUT_ptr + row * stride_o_row + offs * stride_o_col, out.to(OUT_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton RMSNorm: out = x / rms(x) * weight."""
    assert x.is_cuda
    orig_shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8

    _rmsnorm_kernel[(M,)](
        x, weight, out,
        N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out.view(orig_shape)
