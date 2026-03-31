"""FlashAttention correctness + performance test."""

import json
import math
import sys
import time
from pathlib import Path

import torch

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "triton_template", Path(__file__).resolve().parent / "triton_template.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
flash_attention = _mod.flash_attention


def sdpa_ref(Q, K, V, causal=True):
    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), is_causal=causal
    ).to(Q.dtype)


def test_correctness(dtype=torch.float16):
    configs = [
        (1, 4, 128, 64), (2, 8, 256, 64), (1, 4, 512, 128), (2, 4, 1024, 64),
    ]
    results = []
    for B, H, N, D in configs:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
        K = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
        V = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

        ref = sdpa_ref(Q, K, V, causal=True)
        out = flash_attention(Q, K, V, causal=True)

        ok = torch.allclose(out.float(), ref.float(), atol=1e-1, rtol=1e-1)
        max_diff = (out.float() - ref.float()).abs().max().item()
        results.append({"shape": f"B{B}H{H}N{N}D{D}", "correct": ok, "max_diff": max_diff})

        status = "PASS" if ok else "FAIL"
        print(f"  {status} B={B} H={H} N={N} D={D}: max_diff={max_diff:.6f}")

    return results


def benchmark(B=2, H=32, N=1024, D=64, dtype=torch.float16, n_warmup=10, n_iter=50):
    Q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    K = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    V = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    for _ in range(n_warmup):
        flash_attention(Q, K, V, causal=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        flash_attention(Q, K, V, causal=True)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    for _ in range(n_warmup):
        sdpa_ref(Q, K, V, causal=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        sdpa_ref(Q, K, V, causal=True)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iter

    flops = 4.0 * B * H * N * N * D
    triton_tflops = flops / triton_time / 1e12
    pytorch_tflops = flops / pytorch_time / 1e12

    return {
        "triton_us": round(triton_time * 1e6, 1),
        "pytorch_us": round(pytorch_time * 1e6, 1),
        "triton_tflops": round(triton_tflops, 2),
        "pytorch_tflops": round(pytorch_tflops, 2),
        "vs_pytorch": round(triton_tflops / pytorch_tflops, 3) if pytorch_tflops > 0 else 0,
    }


def main():
    json_mode = "--json" in sys.argv
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

    print(f"=== FlashAttention Test on {gpu_name} ===\n")

    print("Correctness:")
    corr = test_correctness()
    all_correct = all(r["correct"] for r in corr)

    print(f"\nBenchmark (B=2, H=32, N=1024, D=64, FP16, causal):")
    perf = benchmark()
    print(f"  Triton: {perf['triton_tflops']} TFLOPS ({perf['triton_us']} us)")
    print(f"  PyTorch SDPA: {perf['pytorch_tflops']} TFLOPS ({perf['pytorch_us']} us)")
    print(f"  Ratio: {perf['vs_pytorch']}x")

    result = {
        "kernel": "flash_attention",
        "gpu": gpu_name,
        "correctness": all_correct,
        "latency_us": perf["triton_us"],
        "tflops": perf["triton_tflops"],
        "vs_pytorch": perf["vs_pytorch"],
        "status": "pass" if all_correct else "fail",
    }

    if json_mode:
        print(json.dumps(result))

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
