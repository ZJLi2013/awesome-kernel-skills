"""Fused MoE correctness + performance test."""

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
fused_moe_simple = _mod.fused_moe_simple


def moe_ref(hidden, w, gating_output, topk=2):
    """Reference: sequential per-expert matmul with top-k routing."""
    M, K = hidden.shape
    E, N, _ = w.shape
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    out = torch.zeros(M, N, device=hidden.device, dtype=hidden.dtype)
    for k in range(topk):
        for e in range(E):
            mask = topk_ids[:, k] == e
            if mask.any():
                h = hidden[mask].float()
                w_e = w[e].float()
                result = h @ w_e.T
                out[mask] += (result * topk_weights[mask, k:k+1]).to(out.dtype)
    return out


def test_correctness(dtype=torch.float16):
    configs = [(16, 8, 64, 32, 2), (32, 4, 128, 64, 2), (64, 8, 256, 128, 2)]
    results = []
    for M, E, N, K, topk in configs:
        hidden = torch.randn(M, K, device="cuda", dtype=dtype)
        w = torch.randn(E, N, K, device="cuda", dtype=dtype)
        gating = torch.randn(M, E, device="cuda", dtype=dtype)

        ref = moe_ref(hidden, w, gating, topk)
        out = fused_moe_simple(hidden, w, gating, topk)

        ok = torch.allclose(out.float(), ref.float(), atol=5e-1, rtol=5e-1)
        max_diff = (out.float() - ref.float()).abs().max().item()
        results.append({"shape": f"M{M}E{E}N{N}K{K}", "correct": ok, "max_diff": max_diff})

        status = "PASS" if ok else "FAIL"
        print(f"  {status} M={M} E={E} N={N} K={K}: max_diff={max_diff:.4f}")

    return results


def benchmark(M=256, E=8, N=1024, K=512, topk=2, dtype=torch.float16, n_warmup=10, n_iter=50):
    hidden = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(E, N, K, device="cuda", dtype=dtype)
    gating = torch.randn(M, E, device="cuda", dtype=dtype)

    for _ in range(n_warmup):
        fused_moe_simple(hidden, w, gating, topk)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        fused_moe_simple(hidden, w, gating, topk)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    flops = 2.0 * M * topk * N * K
    tflops = flops / triton_time / 1e12

    return {"triton_us": round(triton_time * 1e6, 1), "tflops": round(tflops, 2)}


def main():
    json_mode = "--json" in sys.argv
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

    print(f"=== Fused MoE Test on {gpu_name} ===\n")

    print("Correctness:")
    corr = test_correctness()
    all_correct = all(r["correct"] for r in corr)

    print(f"\nBenchmark (M=256, E=8, N=1024, K=512, top2, FP16):")
    perf = benchmark()
    print(f"  Triton: {perf['tflops']} TFLOPS ({perf['triton_us']} us)")

    result = {
        "kernel": "fused_moe",
        "gpu": gpu_name,
        "correctness": all_correct,
        "latency_us": perf["triton_us"],
        "tflops": perf["tflops"],
        "status": "pass" if all_correct else "fail",
    }

    if json_mode:
        print(json.dumps(result))

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
