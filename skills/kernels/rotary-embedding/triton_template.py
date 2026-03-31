"""
Rotary Embedding (RoPE) Triton Template -- Interleaved layout.
Works on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

Usage:
    from skills.kernels.rotary_embedding.triton_template import rotary_embedding
    out = rotary_embedding(x, cos, sin)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    seq_len, head_dim, half_dim,
    stride_x_row, stride_cos_row, stride_sin_row, stride_out_row,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask_half = col_offsets < half_dim

    for row_off in range(ROWS_PER_PROG):
        row_idx = row_start + row_off
        row_valid = row_idx < seq_len
        load_mask = mask_half & row_valid

        x_base = X_ptr + row_idx * stride_x_row
        x1 = tl.load(x_base + col_offsets * 2, mask=load_mask, other=0.0)
        x2 = tl.load(x_base + col_offsets * 2 + 1, mask=load_mask, other=0.0)

        cos = tl.load(COS_ptr + row_idx * stride_cos_row + col_offsets, mask=load_mask, other=1.0)
        sin = tl.load(SIN_ptr + row_idx * stride_sin_row + col_offsets, mask=load_mask, other=0.0)

        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos

        out_base = OUT_ptr + row_idx * stride_out_row
        tl.store(out_base + col_offsets * 2, rx1, mask=load_mask)
        tl.store(out_base + col_offsets * 2 + 1, rx2, mask=load_mask)


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Triton RoPE: interleaved layout. x: [*, head_dim], cos/sin: [seq, D/2]."""
    assert x.is_cuda
    orig_shape = x.shape
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    assert head_dim % 2 == 0

    x_flat = x.contiguous().view(-1, head_dim)
    n_rows = x_flat.shape[0]

    cos_flat = cos.contiguous().view(-1, half_dim)
    sin_flat = sin.contiguous().view(-1, half_dim)

    if cos_flat.shape[0] < n_rows:
        reps = (n_rows + cos_flat.shape[0] - 1) // cos_flat.shape[0]
        cos_flat = cos_flat.repeat(reps, 1)[:n_rows]
        sin_flat = sin_flat.repeat(reps, 1)[:n_rows]

    out = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    ROWS_PER_PROG = 4
    num_warps = 2 if half_dim <= 64 else 4 if half_dim <= 256 else 8

    grid = (triton.cdiv(n_rows, ROWS_PER_PROG),)
    _rotary_kernel[grid](
        x_flat, cos_flat, sin_flat, out,
        n_rows, head_dim, half_dim,
        x_flat.stride(0), cos_flat.stride(0), sin_flat.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=num_warps,
        num_stages=2,
    )
    return out.view(orig_shape)
