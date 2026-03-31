"""
Softmax Triton Template -- Fused row-wise softmax with multi-row processing.
Works on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

Usage:
    from skills.kernels.softmax.triton_template import softmax
    out = softmax(x)  # x: [*, N] -> out: [*, N]
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    BLOCK_SIZE: tl.constexpr,
    NUM_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * NUM_ROWS
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    for row_off in range(NUM_ROWS):
        row_idx = row_start + row_off
        row_valid = row_idx < n_rows

        in_ptr = input_ptr + row_idx * stride_in_row
        out_ptr = output_ptr + row_idx * stride_out_row

        load_mask = col_mask & row_valid

        row = tl.load(in_ptr + col_offsets, mask=load_mask, other=float("-inf")).to(tl.float32)

        row_max = tl.max(row, axis=0)
        row = row - row_max
        numerator = tl.exp(row)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator

        tl.store(out_ptr + col_offsets, result.to(output_ptr.dtype.element_ty), mask=load_mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton fused softmax along the last dimension."""
    assert x.is_cuda
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if n_cols <= 2048:
        num_warps = 4
    elif n_cols <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    ROWS_PER_PROG = min(4, max(1, 4096 // max(n_cols, 1)))
    grid = (triton.cdiv(n_rows, ROWS_PER_PROG),)

    _softmax_kernel[grid](
        x, output,
        n_rows, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_ROWS=ROWS_PER_PROG,
        num_warps=num_warps,
        num_stages=2,
    )
    return output.view(orig_shape)
