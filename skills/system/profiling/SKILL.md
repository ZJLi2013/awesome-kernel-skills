---
name: kernel-profiling
description: >-
  Profile GPU kernels using NCU (NVIDIA) or rocprof (AMD) to collect performance metrics.
  Produces structured metrics.json with throughput, bandwidth, occupancy, and bottleneck classification.
  Use when profiling kernels, collecting NCU/rocprof data, or diagnosing performance issues.
---

# Kernel Profiling Skill

## Overview

Profiling is the first step in any kernel optimization loop. This skill covers collecting
hardware performance counters and deriving actionable metrics on both NVIDIA and AMD GPUs.

## NVIDIA: Nsight Compute (NCU)

### Quick Command

```bash
ncu --set full -o profile_output python your_kernel.py
```

### Key Metrics to Collect

| Metric | What It Tells You |
|--------|-------------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth utilization |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/Tex cache pressure |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Global load efficiency |

### Roofline Derivation

```python
arithmetic_intensity = flops / bytes_transferred
peak_flops = gpu_spec["fp16_tflops"] * 1e12
peak_bw = gpu_spec["hbm_bw_bytes"]
roofline_bound = min(peak_flops, arithmetic_intensity * peak_bw)
achieved_pct = actual_flops / roofline_bound
```

## AMD: rocprof

### Quick Command

```bash
rocprof --stats python your_kernel.py
# Or with specific counters:
rocprof -i counters.txt python your_kernel.py
```

### Key Counters

```text
pmc : SQ_INSTS_VALU SQ_INSTS_MFMA_MOPS_FP16 SQ_INSTS_MFMA_MOPS_BF16
pmc : SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_MFMA SQ_BUSY_CYCLES
pmc : TCC_HIT TCC_MISS TCC_REQ
pmc : TCC_EA_RDREQ TCC_EA_WRREQ
```

### Derived Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| GPU Utilization | `GRBM_GUI_ACTIVE / GRBM_COUNT` | >90% |
| MFMA Efficiency | `SQ_ACTIVE_INST_MFMA / SQ_BUSY_CYCLES` | >50% for matmul |
| L2 Hit Rate | `TCC_HIT / TCC_REQ` | >80% for tiled kernels |
| Memory BW Util | `(reads + writes) * line_size / time / peak_bw` | >70% for mem-bound |

## Output Format

Produce a `metrics.json`:

```json
{
  "kernel": "matmul",
  "gpu": "RTX 4090",
  "latency_us": 101.9,
  "tflops": 168.7,
  "gbps": null,
  "occupancy_pct": 85.2,
  "sm_throughput_pct": 72.1,
  "dram_throughput_pct": 45.3,
  "bottleneck": "compute",
  "roofline_pct": 68.5
}
```

## Agent Instructions

1. Detect GPU vendor (`scripts/detect_gpu.py`)
2. Run appropriate profiler (NCU or rocprof)
3. Parse output into structured metrics
4. Classify bottleneck (memory / compute / latency)
5. Feed into bottleneck-diagnosis skill for optimization recommendations
