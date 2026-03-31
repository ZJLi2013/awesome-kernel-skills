"""Auto-detect GPU vendor (NVIDIA / AMD) and return hardware specs for kernel tuning."""

import torch
import json
import sys


def detect_gpu():
    if not torch.cuda.is_available():
        return {"vendor": "none", "error": "No CUDA/ROCm device found"}

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)

    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None

    info = {
        "vendor": "amd" if is_hip else "nvidia",
        "name": name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024**3), 1),
        "multiprocessor_count": props.multi_processor_count,
        "is_hip": is_hip,
    }

    if is_hip:
        info["wavefront_size"] = 64
        info["notes"] = "CDNA: num_warps controls wavefronts of 64 threads"
    else:
        info["warp_size"] = 32
        sm = props.major * 10 + props.minor
        if sm >= 100:
            info["arch_family"] = "blackwell"
        elif sm >= 90:
            info["arch_family"] = "hopper"
        elif sm >= 89:
            info["arch_family"] = "ada"
        elif sm >= 80:
            info["arch_family"] = "ampere"
        elif sm >= 75:
            info["arch_family"] = "turing"
        else:
            info["arch_family"] = "older"

    return info


def is_nvidia():
    info = detect_gpu()
    return info.get("vendor") == "nvidia"


def is_amd():
    info = detect_gpu()
    return info.get("vendor") == "amd"


if __name__ == "__main__":
    info = detect_gpu()
    print(json.dumps(info, indent=2))
    if info.get("vendor") == "none":
        sys.exit(1)
