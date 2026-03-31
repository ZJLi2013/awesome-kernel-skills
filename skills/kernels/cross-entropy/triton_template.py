"""
Cross-Entropy Triton Template -- Fused log-softmax + NLL loss.
Works on both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.

Usage:
    from skills.kernels.cross_entropy.triton_template import cross_entropy
    loss = cross_entropy(logits, targets)  # scalar mean loss
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_kernel(
    logits_ptr, targets_ptr, losses_ptr,
    n_cols, stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx * stride_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    logits = tl.load(row_start + col_offsets, mask=mask, other=float("-inf")).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    shifted = logits - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    log_sum_exp = tl.log(sum_exp)

    target = tl.load(targets_ptr + row_idx)
    target_logit = tl.load(row_start + target).to(tl.float32)
    loss = -(target_logit - row_max - log_sum_exp)

    tl.store(losses_ptr + row_idx, loss)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Triton fused cross-entropy loss (mean reduction)."""
    assert logits.is_cuda and targets.is_cuda

    if logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1)

    n_rows, n_cols = logits.shape
    losses = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8 if BLOCK_SIZE <= 8192 else 16

    _cross_entropy_kernel[(n_rows,)](
        logits, targets, losses,
        n_cols, logits.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )
    return losses.mean()
