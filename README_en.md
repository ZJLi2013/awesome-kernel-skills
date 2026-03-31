# Awesome Kernel Skills

**[English](README_en.md)** | **[中文](README_cn.md)**

A structured collection of GPU kernel optimization skills for AI coding agents (Cursor, Claude Code, Codex). Each skill is a self-contained markdown document with runnable Triton templates, dual NVIDIA/AMD autotune configs, and verification scripts.

**Verified on**: NVIDIA RTX 4090 + AMD MI300X -- 7/7 kernels pass on both platforms.

## Quick Start

```bash
# On NVIDIA GPU (PyTorch NGC container)
docker run --gpus all --rm -it --ipc=host -v $(pwd):/workspace nvcr.io/nvidia/pytorch:25.12-py3
cd /workspace && python scripts/run_all_tests.py

# On AMD GPU (ROCm PyTorch container)
docker run --device=/dev/kfd --device=/dev/dri --group-add video --rm -it --ipc=host \
  -v $(pwd):/workspace rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
cd /workspace && python scripts/run_all_tests.py
```

## Skill Index

### System Skills

| Skill | Path | Description |
|-------|------|-------------|
| **Optimize Loop** | `skills/system/optimize-loop/` | **Iterative optimization orchestrator** -- chains all skills into a profile → diagnose → optimize → verify → benchmark loop |
| Profiling | `skills/system/profiling/` | NCU / rocprof metric collection |
| Bottleneck Diagnosis | `skills/system/bottleneck-diagnosis/` | Memory / compute / latency bound classification (GEAK workload guidance) |
| Verification | `skills/system/verification/` | 5-stage correctness verification protocol |
| Benchmark | `skills/system/benchmark/` | Unified latency / TFLOPS / GBps benchmark |

### Optimization Skills (6-Tier Playbook)

| Tier | Skill | Path |
|------|-------|------|
| 1 | Block / Tile / Warp Config | `skills/optimization/tier1-block-tiling/` |
| 2 | Memory Access & Hierarchy | `skills/optimization/tier2-memory-access/` |
| 3 | Compute & Fusion | `skills/optimization/tier3-compute-fusion/` |
| 4 | Advanced Scheduling | `skills/optimization/tier4-advanced-scheduling/` |
| 5 | NVIDIA Architecture (H100/A100/4090) | `skills/optimization/tier5-arch-nvidia/` |
| 5 | AMD ROCm Architecture (MI300X/CDNA3) | `skills/optimization/tier5-arch-amd-rocm/` |

### Kernel Skills

| Kernel | Bound Type | Path | Key Metric | 4090 | MI300X |
|--------|-----------|------|------------|------|--------|
| GEMM | Compute | `skills/kernels/gemm/` | TFLOPS | 168.7 | 94.4 |
| Softmax | Memory | `skills/kernels/softmax/` | GBps | 1747 | 700 |
| RMSNorm | Memory | `skills/kernels/rmsnorm/` | GBps | 2482 | 876 |
| FlashAttention | IO / Compute | `skills/kernels/flash-attention/` | TFLOPS | PASS | PASS |
| Cross-Entropy | Memory | `skills/kernels/cross-entropy/` | GBps | PASS | PASS |
| Rotary Embedding | Memory | `skills/kernels/rotary-embedding/` | GBps | PASS | PASS |
| Fused MoE | Compute | `skills/kernels/fused-moe/` | TFLOPS | PASS | PASS |

## How to Use

### Use Case 1: As Domain Knowledge for AI Coding Agents

The primary design intent -- inject GPU kernel optimization expertise into agents like Cursor, Claude Code, or Codex so they can write and optimize kernels autonomously.

**Setup**: Add the `skills/` directory to your agent's skill path (e.g. `.cursor/rules/` or skill references). The agent will index all `SKILL.md` files automatically.

**Example 1: Write a new kernel**

```
You:   Write a causal FlashAttention Triton kernel for head_dim=128, targeting MI300X.

Agent reads:
  1. flash-attention/SKILL.md    → online softmax algorithm, tiled QK/AV structure
  2. tier5-arch-amd-rocm/SKILL.md → wavefront-64, MFMA, num_stages>=1
  3. tier1-block-tiling/SKILL.md  → BLOCK_M/N selection for D=128
  4. flash-attention/triton_template.py → runnable starting point

Agent outputs: a working, AMD-tuned FlashAttention kernel with test script.
```

**Example 2: Iteratively optimize an existing kernel**

```
You:   Optimize my softmax kernel to reach 70% of HBM peak bandwidth on 4090.

Agent reads: optimize-loop/SKILL.md → understands the iterative loop protocol
  Iter 0: baseline → 1747 GBps (61% peak)
  Iter 1: profile → diagnose memory-bound → tier2: vectorized loads → 1880 GBps (+7.6%)
  Iter 2: profile → still memory-bound → tier1: BLOCK_SIZE 8192 → 1955 GBps (+4.0%)
  Iter 3: profile → tier3: fuse with residual add → 2010 GBps → TARGET MET (70.2%)

Agent outputs: optimized kernel + log in experiments/ with every iteration.
```

### Use Case 2: As an Engineer's Reference Manual + Runnable Templates

Even without AI agents, this project serves as a structured Triton kernel optimization handbook with code you can run immediately.

**Typical LLM inference optimization workflow**:

```
 Identify bottleneck operator (e.g. decode-phase softmax is slow)
   │
   ▼
 Step 1: profiling/SKILL.md
   Run NCU or rocprof → dram_bw=85%, sm_throughput=15%
   │
   ▼
 Step 2: bottleneck-diagnosis/SKILL.md
   High DRAM + low compute → memory-bound → recommend Tier 1-2 first
   │
   ▼
 Step 3: softmax/SKILL.md + triton_template.py
   Grab the fused softmax implementation (dual-platform autotune included)
   │
   ▼
 Step 4: tier1-block-tiling/SKILL.md + tier2-memory-access/SKILL.md
   Follow diagnosis: tune BLOCK_SIZE, multi-row processing, vectorized loads
   │
   ▼
 Step 5: test_softmax.py
   Verify correctness + measure GBps → record in experiments/
```

**Common LLM inference tasks mapped to skills**:

| Inference Task | Skills to Use | What You Get |
|---------------|---------------|--------------|
| Prefill GEMM too slow | `gemm/` + `tier1` + `tier5-nvidia` | Template as baseline, autotune configs from SKILL.md |
| Decode attention memory-bound | `flash-attention/` + `tier2` | Online softmax principles, KV cache access patterns |
| MoE routing + expert GEMM fusion | `fused-moe/` + `tier4` | Grouped GEMM + token routing fusion pattern |
| RMSNorm / RoPE small-op overhead | `rmsnorm/` + `rotary-embedding/` + `tier3` | Drop-in fused kernels replacing PyTorch eager |
| Porting to MI300X | `tier5-arch-amd-rocm/` | Wavefront-64, MFMA, `waves_per_eu` tuning guide |
| Not sure what to optimize first | `profiling/` → `bottleneck-diagnosis/` | Data-driven profile → diagnose → act loop |
| Want continuous auto-optimization | `optimize-loop/` (orchestrates all above) | Iterative loop until target met or max iterations |

## Architecture

```
skills/
  system/         -- workflow skills (profiling, verification, benchmark, diagnosis)
  optimization/   -- technique skills (6-tier playbook from AutoKernel)
  kernels/        -- per-kernel skills (SKILL.md + triton_template.py + test)
experiments/      -- experiment logs (experiment-driven-doc format)
scripts/          -- shared utilities (GPU detection, test runner)
```

Each kernel skill directory contains:
- `SKILL.md` -- Agent-readable skill document (Cursor SKILL.md convention)
- `triton_template.py` -- Working Triton kernel with dual NVIDIA/AMD autotune
- `test_*.py` -- Correctness + performance benchmark script

## Related Projects

### Agent Skill Collections

| Project | Focus | Stars |
|---------|-------|-------|
| [HF Custom CUDA Kernels Skills](https://github.com/huggingface/kernels) | CUDA kernel builder + agent skill for Claude/Codex/Cursor | 500+ |
| [agent-gpu-skills](https://github.com/slowlyC/agent-gpu-skills) | Multi-level GPU skills (PTX/CUDA, CUTLASS, Triton, SGLang) | 48 |
| [agent-skills.md](https://agent-skills.md/) | Community skills marketplace (includes cuda, kernel-refine skills) | -- |

### Kernel Agent Systems

| Project | Approach | Benchmark |
|---------|----------|-----------|
| [AutoKernel](https://github.com/RightNow-AI/autokernel) | Iterative agent-driven search, 6-tier playbook | 5.29x on RMSNorm (H100) |
| [KernelAgent (PyTorch/Meta)](https://github.com/meta-pytorch/KernelAgent) | Hardware-guided multi-agent orchestration | 100% on KernelBench 250 tasks |
| [KernelSkill](https://github.com/0satan0/KernelMem/) | Multi-agent + dual-level memory (skills + short-term) | 100% L1-L3, 5.44x L1 speedup |
| [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) | Large-scale agentic RL for CUDA generation | 98.8% pass, 2.11x vs torch.compile |
| [GEAK](https://github.com/AMD-AGI/GEAK) | HIP/Triton optimization agent (AMD-focused) | AMD CDNA3 knowledge base |

### Triton Kernel Collections

| Project | Description | Stars |
|---------|-------------|-------|
| [Awesome-Triton-Kernels](https://github.com/zinccat/Awesome-Triton-Kernels) | Curated Triton kernel collection by category | 184 |
| [triton-index](https://github.com/gpu-mode/triton-index) | Searchable catalog of released Triton kernels | 298 |
| [triton-resources](https://github.com/rkinas/triton-resources) | Learning path with daily challenges + benchmarks | 467 |
| [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) | CUDA/Triton learning materials, papers, lectures | 2060 |

### Key Differences

This project (`awesome_kernel_skills`) differs from the above in that it:
1. **Skills-first**: Every kernel is packaged as a Cursor/Claude-compatible `SKILL.md` with structured agent instructions
2. **Dual-platform**: All templates include both NVIDIA and AMD ROCm autotune configs, verified on real hardware
3. **6-tier optimization playbook**: System/optimization/kernel skills organized following AutoKernel's tiered approach
4. **Runnable + verified**: Every kernel ships with a test that produces JSON benchmarks on both platforms

## Roadmap

### Layer 1: New Kernel Skills (Hardware-Driven)

As new instructions and data formats arrive, new kernel skills will be added:

| Hardware Generation | New Capability | Planned Skill |
|---------------------|---------------|---------------|
| H100 / MI300X | FP8 (E4M3/E5M2) GEMM | `kernels/fp8-gemm/` |
| Blackwell | FP4 GEMM, 2nd-gen TMA | `kernels/fp4-gemm/` |
| Ampere+ | 2:4 Structured Sparsity | `kernels/sparse-gemm/` |
| Cross-gen | MX Formats (MXFP4/MXFP6, micro-scaling) | `kernels/mx-gemm/` |
| Cross-gen | Quantized KV Cache Attention | `kernels/quantized-attention/` |

### Layer 2: Updated Optimization Knowledge

New hardware doesn't just add instructions -- it changes kernel design paradigms:

| Shift | Impact on Existing Skills |
|-------|--------------------------|
| **Micro-scaling formats** | Tier 2 (memory): per-block scaling factors change load/broadcast patterns; Tier 3 (fusion): quantize-GEMM-dequant must be fused |
| **Warp specialization** | Tier 4 (scheduling): producer/consumer warp model replaces uniform warp design |
| **Cluster-level SMEM** | Tier 5 (Hopper+): multi-block cooperation via distributed shared memory |
| **HBM4 / CXL** | Tier 2: new memory hierarchy levels change tiling and prefetch strategies |

### Layer 3: Smarter Optimization Loop

The most impactful evolution -- moving from "human writes rules, agent selects" toward self-improving agents:

```
Current:    human curates SKILL.md  →  agent follows instructions
                                         ↓
Near-term:  agent logs successful optimizations  →  stores as new reusable skills
            (optimize-loop feedback → experiments/ → new SKILL.md entries)
                                         ↓
Future:     agent reads ISA specs + runs experiments  →  discovers optimization
            rules autonomously (RL-driven, à la CUDA-Agent)
```

| Direction | Approach | Reference |
|-----------|----------|-----------|
| Skill accumulation (current) | Human-curated, agent-consumed | This project, agent-gpu-skills |
| Skill self-generation | Agent stores successful optimization trajectories as new skills | [KernelSkill](https://arxiv.org/abs/2603.10085) dual-level memory |
| RL-driven exploration | Agent discovers rules via reward (correctness + speedup), no human skills needed | [CUDA-Agent](https://arxiv.org/abs/2602.24286) |

A concrete near-term step for this project: enhance `optimize-loop/SKILL.md` so that after each successful optimization, the agent appends a structured record to a `learned-patterns/` directory -- building a project-local skill memory that grows with use.

## Sources

Distilled from:
- [AutoKernel](https://github.com/RightNow-AI/autokernel) -- 6-tier playbook + 9 kernel starters
- [HF Custom CUDA Kernels Skills](https://huggingface.co/blog/custom-cuda-kernels-agent-skills) -- SKILL.md format
- [agent-gpu-skills](https://github.com/slowlyC/agent-gpu-skills) -- Multi-level GPU agent skills (cuda-skill / cutlass-skill / triton-skill / sglang-skill), CUDA ecosystem focused
- [GEAK](https://github.com/AMD-AGI/GEAK) -- AMD CDNA3 knowledge base
- [KernelAgent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) -- PyTorch multi-agent orchestration
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) -- Reference implementations
- [KernelSkill](https://arxiv.org/abs/2603.10085) -- Multi-agent framework with expert optimization skills
- [CUDA-Agent](https://arxiv.org/abs/2602.24286) -- Agentic RL for CUDA kernel generation
