"""Run all kernel tests and produce a JSON summary report."""

import json
import subprocess
import sys
import time
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills" / "kernels"

KERNEL_TESTS = {
    "gemm": "skills/kernels/gemm/test_gemm.py",
    "softmax": "skills/kernels/softmax/test_softmax.py",
    "rmsnorm": "skills/kernels/rmsnorm/test_rmsnorm.py",
    "flash_attention": "skills/kernels/flash-attention/test_flash_attention.py",
    "cross_entropy": "skills/kernels/cross-entropy/test_cross_entropy.py",
    "rotary_embedding": "skills/kernels/rotary-embedding/test_rotary.py",
    "fused_moe": "skills/kernels/fused-moe/test_fused_moe.py",
}


def run_test(kernel_name, test_path, project_root):
    full_path = project_root / test_path
    if not full_path.exists():
        return {"kernel": kernel_name, "status": "skipped", "reason": "test file not found"}

    try:
        result = subprocess.run(
            [sys.executable, str(full_path), "--json"],
            capture_output=True, text=True, timeout=300,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass
            return {"kernel": kernel_name, "status": "pass", "output": result.stdout[-500:]}
        else:
            return {
                "kernel": kernel_name, "status": "fail",
                "returncode": result.returncode,
                "stderr": result.stderr[-500:],
            }
    except subprocess.TimeoutExpired:
        return {"kernel": kernel_name, "status": "timeout"}
    except Exception as e:
        return {"kernel": kernel_name, "status": "error", "error": str(e)}


def main():
    project_root = Path(__file__).resolve().parent.parent

    # detect GPU first
    detect_script = project_root / "scripts" / "detect_gpu.py"
    gpu_result = subprocess.run(
        [sys.executable, str(detect_script)], capture_output=True, text=True
    )
    gpu_info = {}
    if gpu_result.returncode == 0:
        try:
            gpu_info = json.loads(gpu_result.stdout)
        except json.JSONDecodeError:
            pass

    print(f"GPU: {gpu_info.get('name', 'unknown')} ({gpu_info.get('vendor', 'unknown')})")
    print(f"Running {len(KERNEL_TESTS)} kernel tests...\n")

    results = []
    for name, path in KERNEL_TESTS.items():
        print(f"  [{name}] ", end="", flush=True)
        t0 = time.time()
        r = run_test(name, path, project_root)
        r["gpu"] = gpu_info.get("name", "unknown")
        r["vendor"] = gpu_info.get("vendor", "unknown")
        r["elapsed_s"] = round(time.time() - t0, 2)
        results.append(r)
        status = r.get("status", "unknown")
        print(f"{status} ({r['elapsed_s']}s)")

    report = {"gpu_info": gpu_info, "results": results}
    report_path = project_root / "experiments" / "test_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults written to {report_path}")

    passed = sum(1 for r in results if r.get("status") == "pass" or r.get("correctness") is True)
    total = len(results)
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"Summary: {passed}/{total - skipped} passed, {skipped} skipped")

    return 0 if passed == total - skipped else 1


if __name__ == "__main__":
    sys.exit(main())
