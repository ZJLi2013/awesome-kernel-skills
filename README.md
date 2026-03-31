# Awesome Kernel Skills

**[English](README_en.md)** | **[中文](README_cn.md)**

A structured collection of GPU kernel optimization skills for AI coding agents.

GPU 算子优化技能的结构化集合，面向 AI 编程智能体。

---

**Two ways to use / 两种使用方式**:
1. As domain knowledge for AI coding agents (Cursor / Claude Code / Codex) -- agents read `SKILL.md` to write optimized kernels
2. As an engineer's reference manual with runnable Triton templates and a profile → diagnose → optimize workflow

**Verified on / 已验证平台**: NVIDIA RTX 4090 + AMD MI300X -- 7/7 kernels pass on both platforms.

```
skills/
  system/         -- optimize-loop, profiling, verification, benchmark, diagnosis
  optimization/   -- 6-tier playbook (block-tiling → arch-specific)
  kernels/        -- GEMM, Softmax, RMSNorm, FlashAttention, CrossEntropy, RoPE, FusedMoE
experiments/      -- experiment logs
scripts/          -- GPU detection, test runner
```

```bash
# Quick start
docker run --gpus all --rm -it --ipc=host -v $(pwd):/workspace nvcr.io/nvidia/pytorch:25.12-py3
cd /workspace && python scripts/run_all_tests.py
```
