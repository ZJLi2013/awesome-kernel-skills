"""Rotary Embedding correctness + performance test."""

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
rotary_embedding = _mod.rotary_embedding


def rope_ref(x, cos, sin):
    """Reference: interleaved layout RoPE."""
    d = x.shape[-1]
    half = d // 2
    x_flat = x.view(-1, d).float()
    cos_flat = cos.view(-1, half).float()
    sin_flat = sin.view(-1, half).float()
    if cos_flat.shape[0] < x_flat.shape[0]:
        reps = (x_flat.shape[0] + cos_flat.shape[0] - 1) // cos_flat.shape[0]
        cos_flat = cos_flat.repeat(reps, 1)[:x_flat.shape[0]]
        sin_flat = sin_flat.repeat(reps, 1)[:x_flat.shape[0]]
    x1 = x_flat[:, 0::2]
    x2 = x_flat[:, 1::2]
    out = torch.empty_like(x_flat)
    out[:, 0::2] = x1 * cos_flat - x2 * sin_flat
    out[:, 1::2] = x1 * sin_flat + x2 * cos_flat
    return out.view(x.shape).to(x.dtype)


def test_correctness(dtype=torch.float16):
    configs = [(128, 64), (256, 128), (512, 64), (1024, 128)]
    results = []
    for seq, dim in configs:
        x = torch.randn(seq, dim, device="cuda", dtype=dtype)
        cos = torch.randn(seq, dim // 2, device="cuda", dtype=dtype)
        sin = torch.randn(seq, dim // 2, device="cuda", dtype=dtype)

        ref = rope_ref(x, cos, sin)
        out = rotary_embedding(x, cos, sin)

        ok = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
        max_diff = (out.float() - ref.float()).abs().max().item()
        results.append({"shape": f"S{seq}D{dim}", "correct": ok, "max_diff": max_diff})

        status = "PASS" if ok else "FAIL"
        print(f"  {status} seq={seq} dim={dim}: max_diff={max_diff:.6f}")

    return results


def benchmark(seq=4096, dim=128, dtype=torch.float16, n_warmup=50, n_iter=200):
    x = torch.randn(seq, dim, device="cuda", dtype=dtype)
    cos = torch.randn(seq, dim // 2, device="cuda", dtype=dtype)
    sin = torch.randn(seq, dim // 2, device="cuda", dtype=dtype)

    for _ in range(n_warmup):
        rotary_embedding(x, cos, sin)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        rotary_embedding(x, cos, sin)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    total_bytes = seq * dim * dtype.itemsize * 2 + seq * (dim // 2) * dtype.itemsize * 2
    gbps = total_bytes / triton_time / 1e9

    return {"triton_us": round(triton_time * 1e6, 1), "gbps": round(gbps, 1)}


def main():
    json_mode = "--json" in sys.argv
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

    print(f"=== Rotary Embedding Test on {gpu_name} ===\n")

    print("Correctness:")
    corr = test_correctness()
    all_correct = all(r["correct"] for r in corr)

    print(f"\nBenchmark (seq=4096, dim=128, FP16):")
    perf = benchmark()
    print(f"  Triton: {perf['gbps']} GBps ({perf['triton_us']} us)")

    result = {
        "kernel": "rotary_embedding",
        "gpu": gpu_name,
        "correctness": all_correct,
        "latency_us": perf["triton_us"],
        "gbps": perf["gbps"],
        "status": "pass" if all_correct else "fail",
    }

    if json_mode:
        print(json.dumps(result))

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
