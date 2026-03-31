"""Cross-Entropy correctness + performance test."""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "triton_template", Path(__file__).resolve().parent / "triton_template.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
cross_entropy = _mod.cross_entropy


def test_correctness(dtype=torch.float16):
    configs = [
        (32, 1000), (64, 32000), (128, 50257), (32, 128256),
    ]
    results = []
    for B, V in configs:
        logits = torch.randn(B, V, device="cuda", dtype=dtype)
        targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

        ref = F.cross_entropy(logits.float(), targets)
        out = cross_entropy(logits, targets)

        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        diff = (out - ref).abs().item()
        results.append({"shape": f"B{B}V{V}", "correct": ok, "diff": diff})

        status = "PASS" if ok else "FAIL"
        print(f"  {status} B={B} V={V}: diff={diff:.6f}")

    return results


def benchmark(B=128, V=32000, dtype=torch.float16, n_warmup=50, n_iter=200):
    logits = torch.randn(B, V, device="cuda", dtype=dtype)
    targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

    for _ in range(n_warmup):
        cross_entropy(logits, targets)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        cross_entropy(logits, targets)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    for _ in range(n_warmup):
        F.cross_entropy(logits.float(), targets)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        F.cross_entropy(logits.float(), targets)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iter

    total_bytes = B * V * logits.element_size()
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

    print(f"=== Cross-Entropy Test on {gpu_name} ===\n")

    print("Correctness:")
    corr = test_correctness()
    all_correct = all(r["correct"] for r in corr)

    print(f"\nBenchmark (B=128, V=32000, FP16):")
    perf = benchmark()
    print(f"  Triton: {perf['triton_gbps']} GBps ({perf['triton_us']} us)")
    print(f"  PyTorch: {perf['pytorch_gbps']} GBps ({perf['pytorch_us']} us)")
    print(f"  Speedup: {perf['speedup']}x")

    result = {
        "kernel": "cross_entropy",
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
