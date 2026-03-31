"""GEMM correctness + multi-size performance benchmark.

Supports the optimize-loop workflow:
  python test_gemm.py                   # human-readable output
  python test_gemm.py --json            # structured JSON for logging
  python test_gemm.py --sizes 1024,4096 # custom sizes (square MxMxM)
"""

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
matmul = _mod.matmul

BENCH_SIZES = [1024, 2048, 4096, 8192]


def test_correctness(dtype=torch.float16, sizes=None):
    if sizes is None:
        sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    results = []
    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)

        ref = torch.matmul(A, B)
        out = matmul(A, B)

        atol = 1e-1 if dtype == torch.float16 else 5e-2
        rtol = 1e-1 if dtype == torch.float16 else 5e-2
        ok = torch.allclose(out, ref, atol=atol, rtol=rtol)

        max_diff = (out - ref).abs().max().item()
        results.append({"shape": f"{M}x{N}x{K}", "correct": ok, "max_diff": max_diff})

        status = "PASS" if ok else "FAIL"
        print(f"  {status} {M}x{N}x{K}: max_diff={max_diff:.6f}")

    return results


def benchmark_one(M, N, K, dtype=torch.float16, n_warmup=25, n_iter=100):
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    for _ in range(n_warmup):
        matmul(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        matmul(A, B)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    for _ in range(n_warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iter

    flops = 2.0 * M * N * K
    triton_tflops = flops / triton_time / 1e12
    pytorch_tflops = flops / pytorch_time / 1e12

    return {
        "size": f"{M}x{N}x{K}",
        "triton_us": round(triton_time * 1e6, 1),
        "pytorch_us": round(pytorch_time * 1e6, 1),
        "triton_tflops": round(triton_tflops, 2),
        "pytorch_tflops": round(pytorch_tflops, 2),
        "vs_pytorch": round(triton_tflops / pytorch_tflops, 3) if pytorch_tflops > 0 else 0,
    }


def benchmark_multi(sizes=None, dtype=torch.float16):
    if sizes is None:
        sizes = BENCH_SIZES
    results = []
    for s in sizes:
        r = benchmark_one(s, s, s, dtype=dtype)
        results.append(r)
        print(f"  {r['size']:>15s}  Triton {r['triton_tflops']:7.2f}  PyTorch {r['pytorch_tflops']:7.2f}  ratio {r['vs_pytorch']:.3f}x")
    return results


def main():
    json_mode = "--json" in sys.argv
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

    sizes = BENCH_SIZES
    for arg in sys.argv[1:]:
        if arg.startswith("--sizes"):
            continue
        if not arg.startswith("--"):
            try:
                sizes = [int(x) for x in arg.split(",")]
            except ValueError:
                pass
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--sizes" and i < len(sys.argv) - 1:
            try:
                sizes = [int(x) for x in sys.argv[i + 1].split(",")]
            except ValueError:
                pass

    print(f"=== GEMM Optimize-Loop Benchmark on {gpu_name} ===\n")

    print("Correctness:")
    corr = test_correctness()
    all_correct = all(r["correct"] for r in corr)
    print()

    print(f"Benchmark (FP16, sizes: {sizes}):")
    perfs = benchmark_multi(sizes)
    print()

    avg_ratio = sum(p["vs_pytorch"] for p in perfs) / len(perfs) if perfs else 0

    result = {
        "kernel": "gemm",
        "gpu": gpu_name,
        "correctness": all_correct,
        "benchmarks": perfs,
        "avg_vs_pytorch": round(avg_ratio, 3),
        "status": "pass" if all_correct else "fail",
    }

    print(f"Average vs PyTorch (cuBLAS/rocBLAS): {avg_ratio:.3f}x")
    print(f"Correctness: {'ALL PASS' if all_correct else 'FAIL'}")

    if json_mode:
        print("\n--- JSON ---")
        print(json.dumps(result, indent=2))

    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
