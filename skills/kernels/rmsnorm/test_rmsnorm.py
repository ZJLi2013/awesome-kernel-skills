"""RMSNorm correctness + performance test."""

import json
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
rmsnorm = _mod.rmsnorm


def rmsnorm_ref(x, weight, eps=1e-6):
    rms = torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(torch.float32) * rms * weight.to(torch.float32)).to(x.dtype)


def test_correctness(dtype=torch.float16):
    configs = [
        (32, 128), (64, 1024), (128, 4096), (256, 8192),
    ]
    results = []
    for M, N in configs:
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype)

        ref = rmsnorm_ref(x, w)
        out = rmsnorm(x, w)

        ok = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
        max_diff = (out.float() - ref.float()).abs().max().item()
        results.append({"shape": f"{M}x{N}", "correct": ok, "max_diff": max_diff})

        status = "PASS" if ok else "FAIL"
        print(f"  {status} {M}x{N}: max_diff={max_diff:.6f}")

    return results


def benchmark(M=2048, N=4096, dtype=torch.float16, n_warmup=50, n_iter=200):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)

    for _ in range(n_warmup):
        rmsnorm(x, w)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        rmsnorm(x, w)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    for _ in range(n_warmup):
        rmsnorm_ref(x, w)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        rmsnorm_ref(x, w)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iter

    elem_size = x.element_size()
    total_bytes = M * N * elem_size * 3  # read x, read w (broadcast), write out
    triton_gbps = total_bytes / triton_time / 1e9
    pytorch_gbps = total_bytes / pytorch_time / 1e9

    return {
        "triton_us": round(triton_time * 1e6, 1),
        "pytorch_us": round(pytorch_time * 1e6, 1),
        "triton_gbps": round(triton_gbps, 1),
        "pytorch_gbps": round(pytorch_gbps, 1),
        "speedup": round(pytorch_time / triton_time, 3) if triton_time > 0 else 0,
    }


def main():
    json_mode = "--json" in sys.argv
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

    print(f"=== RMSNorm Test on {gpu_name} ===\n")

    print("Correctness:")
    corr = test_correctness()
    all_correct = all(r["correct"] for r in corr)

    print(f"\nBenchmark (2048x4096, FP16):")
    perf = benchmark()
    print(f"  Triton: {perf['triton_gbps']} GBps ({perf['triton_us']} us)")
    print(f"  PyTorch: {perf['pytorch_gbps']} GBps ({perf['pytorch_us']} us)")
    print(f"  Speedup: {perf['speedup']}x")

    result = {
        "kernel": "rmsnorm",
        "gpu": gpu_name,
        "correctness": all_correct,
        "latency_us": perf["triton_us"],
        "gbps": perf["triton_gbps"],
        "speedup": perf["speedup"],
        "status": "pass" if all_correct else "fail",
    }

    if json_mode:
        print(json.dumps(result))

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
