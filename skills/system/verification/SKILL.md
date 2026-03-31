---
name: kernel-verification
description: >-
  5-stage kernel correctness verification protocol for Triton and CUDA kernels.
  Covers numerical correctness, dtype sensitivity, edge cases, determinism, and stress testing.
  Use when verifying kernel correctness, running validation, or setting up test harnesses.
---

# Kernel Verification Skill (5-Stage)

## Overview

Every kernel optimization must pass correctness verification before performance is measured.
This skill defines a 5-stage protocol adapted from AutoKernel.

## Stage 1: Basic Correctness

Compare kernel output against PyTorch reference on standard shapes.

```python
ref = pytorch_reference(inputs)
out = triton_kernel(inputs)
assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
```

### Tolerance Guidelines

| Dtype | atol | rtol | Notes |
|-------|------|------|-------|
| float32 | 1e-5 | 1e-5 | Strict |
| float16 | 1e-2 | 1e-2 | FP16 has limited precision |
| bfloat16 | 1e-1 | 1e-1 | BF16 has even less mantissa |
| fp8 | 5e-1 | 5e-1 | Very coarse |

## Stage 2: Dtype Sensitivity

Test all target dtypes (fp16, bf16, fp32) and mixed combinations.
Many bugs only appear in specific precision modes.

## Stage 3: Edge Cases

Test boundary conditions:
- Dimensions not divisible by block size (e.g., M=127, K=33)
- Very small inputs (1x1, 1xN)
- Very large inputs (stress memory limits)
- Zero-filled inputs, inf/nan inputs

## Stage 4: Determinism

Run the same input 10 times and verify bitwise identical output.
Non-determinism indicates race conditions or uninitialized memory.

```python
results = [triton_kernel(inputs) for _ in range(10)]
for r in results[1:]:
    assert torch.equal(r, results[0]), "Non-deterministic!"
```

## Stage 5: Stress Test

Run with large inputs and many iterations to catch:
- Memory leaks (monitor GPU memory)
- Numerical drift over repeated applications
- Rare race conditions

## Agent Instructions

When verifying a kernel:

1. Always run Stage 1 before measuring performance
2. Run Stage 2 if the kernel will be used with multiple dtypes
3. Run Stage 3 for kernels that will handle variable-length inputs (attention, MoE)
4. Run Stage 4 for training kernels (backward pass must be deterministic)
5. Run Stage 5 before declaring a kernel "production ready"

## Quick Test Template

```python
def verify_kernel(kernel_fn, ref_fn, input_gen, n_shapes=5, n_dtypes=2, n_repeats=3):
    for shape in generate_shapes(n_shapes):
        for dtype in [torch.float16, torch.bfloat16][:n_dtypes]:
            inputs = input_gen(shape, dtype)
            ref = ref_fn(*inputs)
            for _ in range(n_repeats):
                out = kernel_fn(*inputs)
                assert torch.allclose(out.float(), ref.float(), atol=get_atol(dtype), rtol=get_rtol(dtype))
    return True
```
