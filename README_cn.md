# Awesome Kernel Skills

**[English](README_en.md)** | **[中文](README_cn.md)**

将 [AutoKernel](https://github.com/RightNow-AI/autokernel) 的 6 层调优手册拆解为 **18 个独立 SKILL.md**（约 1,500 行），覆盖 7 个 LLM 核心算子 + NVIDIA/AMD 双平台。

**两种用法**：
1. **工程师学习路径**：两周读完 18 个 SKILL.md + 自测，系统掌握 kernel 优化的 profile → 诊断 → 调优闭环 → 详见 [LEARNING_PATH.md](LEARNING_PATH.md)
2. **AI 智能体技能注入**：将 `skills/` 目录挂载到 Cursor / Claude Code / Codex，智能体自动获得算子优化的领域知识

**已验证平台**：NVIDIA RTX 4090 + AMD MI300X —— 7/7 算子在两个平台均通过。

## 与 AutoKernel 的关系

本项目的知识体系蒸馏自 [AutoKernel](https://github.com/RightNow-AI/autokernel)——一个自主 GPU 算子优化智能体。

AutoKernel 将所有知识打包在一个 900 行的 `program.md` 中，由 `orchestrate.py` 驱动 agent 自动跑实验。**本项目做的事情是把这些知识拆解、模块化、补充双平台细节，让人类工程师可以逐个消化。**

```
AutoKernel                              awesome_kernel_skills
┌──────────────────────┐                ┌──────────────────────────────────┐
│ program.md (900行)   │   拆解为 →     │ 18 个 SKILL.md（各 49~251 行）    │
│  Tier 1~6 playbook   │                │  6 个 optimization tier skills   │
│  9 kernel starters   │                │  7 个 kernel skills              │
│  orchestration loop  │                │  5 个 system skills              │
├──────────────────────┤                ├──────────────────────────────────┤
│ orchestrate.py       │                │ optimize-loop/SKILL.md           │
│ bench.py / verify.py │   对应 →       │ benchmark/ + verification/       │
│ reference.py         │                │ test_*.py (PyTorch 对比)          │
│ kernels/*.py         │                │ triton_template.py (双平台)       │
└──────────────────────┘                └──────────────────────────────────┘
```

## 学习路径（两周，每天 2~3 小时）

18 个 SKILL.md 按依赖关系分三层阅读，详见 **[LEARNING_PATH.md](LEARNING_PATH.md)**：

| 周 | 内容 | 文件数 | 行数 | 目标 |
|----|------|--------|------|------|
| 第 1 周 | **优化技能 Tier 1~5** → 建立思维框架 | 6 | ~404 | 看到任何 kernel 能判断瓶颈类型 + 选择 tier |
| 第 1 周 | **算子技能** → 用框架理解 7 个算子 | 7 | ~516 | 对每个算子说出核心算法 + 3 个 pitfall |
| 第 2 周 | **系统技能** → 理解编排逻辑 | 5 | ~595 | 理解 profile → 诊断 → 调优 → 验证闭环 |
| 第 2 周 | **对照 AutoKernel** → 交叉验证 | - | - | LEARNING_PATH.md 终极自测通过 |

**毕业标志**：给定一个未见过的算子，30 分钟内完成瓶颈判断 → tier 优先级 → kernel 伪代码 → pitfalls → NVIDIA/AMD 差异。

## 快速开始（运行模板 + 测试）

```bash
# NVIDIA GPU（PyTorch NGC 容器）
docker run --gpus all --rm -it --ipc=host -v $(pwd):/workspace nvcr.io/nvidia/pytorch:25.12-py3
cd /workspace && python scripts/run_all_tests.py

# AMD GPU（ROCm PyTorch 容器）
docker run --device=/dev/kfd --device=/dev/dri --group-add video --rm -it --ipc=host \
  -v $(pwd):/workspace rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
cd /workspace && python scripts/run_all_tests.py
```

## 技能索引

### 系统技能

| 技能 | 路径 | 说明 |
|------|------|------|
| **优化循环** | `skills/system/optimize-loop/` | **迭代优化编排器** —— 将所有技能串联为 profile → 诊断 → 优化 → 验证 → 基准测试的闭环 |
| 性能分析 | `skills/system/profiling/` | NCU / rocprof 指标采集 |
| 瓶颈诊断 | `skills/system/bottleneck-diagnosis/` | 内存 / 计算 / 延迟瓶颈分类（GEAK 工作负载指南） |
| 正确性验证 | `skills/system/verification/` | 五阶段正确性验证协议 |
| 基准测试 | `skills/system/benchmark/` | 统一延迟 / TFLOPS / GBps 基准 |

### 优化技能（6 层调优手册）

| 层级 | 技能 | 路径 |
|------|------|------|
| 1 | 分块 / Tile / Warp 配置 | `skills/optimization/tier1-block-tiling/` |
| 2 | 内存访问与层级 | `skills/optimization/tier2-memory-access/` |
| 3 | 计算优化与算子融合 | `skills/optimization/tier3-compute-fusion/` |
| 4 | 高级调度 | `skills/optimization/tier4-advanced-scheduling/` |
| 5 | NVIDIA 架构特定优化（H100/A100/4090） | `skills/optimization/tier5-arch-nvidia/` |
| 5 | AMD ROCm 架构特定优化（MI300X/CDNA3） | `skills/optimization/tier5-arch-amd-rocm/` |

### 算子技能

| 算子 | 瓶颈类型 | 路径 | 关键指标 | 4090 | MI300X |
|------|----------|------|----------|------|--------|
| GEMM（矩阵乘） | 计算瓶颈 | `skills/kernels/gemm/` | TFLOPS | 168.7 | 94.4 |
| Softmax | 访存瓶颈 | `skills/kernels/softmax/` | GBps | 1747 | 700 |
| RMSNorm | 访存瓶颈 | `skills/kernels/rmsnorm/` | GBps | 2482 | 876 |
| FlashAttention | IO / 计算 | `skills/kernels/flash-attention/` | TFLOPS | PASS | PASS |
| 交叉熵 | 访存瓶颈 | `skills/kernels/cross-entropy/` | GBps | PASS | PASS |
| 旋转位置编码 | 访存瓶颈 | `skills/kernels/rotary-embedding/` | GBps | PASS | PASS |
| 融合 MoE | 计算瓶颈 | `skills/kernels/fused-moe/` | TFLOPS | PASS | PASS |

## 使用方式

### 场景一：工程师学习（主要场景）

**18 个 SKILL.md 已全部写完**，覆盖 AutoKernel 6 层调优手册的全部知识。每个 SKILL.md 包含：
- **Core Technique**：算法原理 + back-of-envelope 分析
- **Common Pitfalls**：高频踩坑点（算子技能）
- **Autotune Configs / Notes**：NVIDIA + AMD 双平台参数
- **Agent Instructions**：给 AI 的执行指令（学习时可跳过）

阅读顺序和自测问题详见 **[LEARNING_PATH.md](LEARNING_PATH.md)**。

**LLM 推理常见任务与技能映射**：

| 推理优化任务 | 对应技能 | 你能得到什么 |
|-------------|---------|-------------|
| Prefill GEMM 太慢 | `gemm/` + `tier1` + `tier5-nvidia` | Tiled GEMM 原理 + L2 grouping + tensor core 约束 |
| Decode attention 访存瓶颈 | `flash-attention/` + `tier2` | Online softmax 原理，tiled QK/AV 的 SRAM 压力分析 |
| MoE 路由 + expert GEMM 融合 | `fused-moe/` + `tier4` | Grouped GEMM + token routing 融合 + persistent 调度 |
| RMSNorm / RoPE 小算子开销大 | `rmsnorm/` + `rotary-embedding/` + `tier3` | 为什么 fused 比 eager 快 3~10x 的物理解释 |
| 需要迁移到 MI300X | `tier5-arch-amd-rocm/` | Wavefront-64、MFMA、`waves_per_eu` 的直觉 |
| 不确定先优化什么 | `profiling/` → `bottleneck-diagnosis/` | NCU/rocprof 关键 metric 的含义 + 瓶颈判断 flowchart |

### 场景二：AI 智能体技能注入

将 `skills/` 目录添加到智能体的技能路径（如 `.cursor/rules/` 或技能引用），智能体自动获得 GPU 算子优化知识。

```
你：帮我写一个 causal FlashAttention Triton kernel，head_dim=128，需要跑在 MI300X 上。

智能体读取：
  1. flash-attention/SKILL.md      → online softmax 算法、tiled QK/AV 结构
  2. tier5-arch-amd-rocm/SKILL.md  → wavefront-64、MFMA、num_stages>=1
  3. tier1-block-tiling/SKILL.md   → D=128 时的 BLOCK_M/N 选择
  4. flash-attention/triton_template.py → 可运行的起点代码

智能体输出：一个可运行的、AMD 调优过的 FlashAttention kernel + 测试脚本。
```

## 项目结构

```
skills/
  system/         -- 工作流技能（性能分析、验证、基准测试、诊断）
  optimization/   -- 优化技术技能（来自 AutoKernel 的 6 层调优手册）
  kernels/        -- 算子技能（SKILL.md + triton_template.py + 测试脚本）
experiments/      -- 实验日志（实验驱动文档格式）
scripts/          -- 共享工具（GPU 检测、测试运行器）
```

每个算子技能目录包含：
- `SKILL.md` —— 智能体可读的技能文档（Cursor SKILL.md 规范）
- `triton_template.py` —— 可运行的 Triton 算子，含 NVIDIA/AMD 双平台自动调优
- `test_*.py` —— 正确性 + 性能基准测试脚本

## 相关项目

### 智能体技能集合

| 项目 | 方向 | Stars |
|------|------|-------|
| [HF Custom CUDA Kernels Skills](https://github.com/huggingface/kernels) | CUDA 算子构建器 + Claude/Codex/Cursor 智能体技能 | 500+ |
| [agent-gpu-skills](https://github.com/slowlyC/agent-gpu-skills) | 多层 GPU 技能（PTX/CUDA、CUTLASS、Triton、SGLang） | 48 |
| [agent-skills.md](https://agent-skills.md/) | 社区技能市场（含 cuda、kernel-refine 技能） | -- |

### 算子智能体系统

| 项目 | 方法 | 评测结果 |
|------|------|----------|
| [AutoKernel](https://github.com/RightNow-AI/autokernel) | 迭代式智能体驱动搜索，6 层调优手册 | RMSNorm 5.29x（H100） |
| [KernelAgent (PyTorch/Meta)](https://github.com/meta-pytorch/KernelAgent) | 硬件引导的多智能体编排 | KernelBench 250 题 100% |
| [KernelSkill](https://github.com/0satan0/KernelMem/) | 多智能体 + 双层记忆（技能 + 短期记忆） | L1-L3 100%，L1 5.44x 加速 |
| [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) | 大规模智能体 RL 生成 CUDA 算子 | 98.8% 通过率，2.11x vs torch.compile |
| [GEAK](https://github.com/AMD-AGI/GEAK) | HIP/Triton 优化智能体（AMD 侧重） | AMD CDNA3 知识库 |

### Triton 算子合集

| 项目 | 说明 | Stars |
|------|------|-------|
| [Awesome-Triton-Kernels](https://github.com/zinccat/Awesome-Triton-Kernels) | 按类别分类的 Triton 算子精选集 | 184 |
| [triton-index](https://github.com/gpu-mode/triton-index) | 可搜索的已发布 Triton 算子目录 | 298 |
| [triton-resources](https://github.com/rkinas/triton-resources) | 每日挑战学习路径 + 基准测试 | 467 |
| [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) | CUDA/Triton 学习资料、论文、讲座 | 2060 |

### 本项目的定位

`awesome_kernel_skills` = **AutoKernel 知识的模块化拆解 + 工程师学习路径**

| 维度 | AutoKernel | 本项目 |
|------|-----------|--------|
| 形态 | 可运行的 agent 系统（Python 脚本 + 编排器） | 可阅读的知识模块（18 个 SKILL.md + 学习路径） |
| 受众 | AI agent（自主跑实验） | **工程师**（读懂原理）+ AI agent（注入知识） |
| 平台 | 主要 NVIDIA，后加 AMD 分支 | 从设计起就是 NVIDIA + AMD 双平台 |
| 知识组织 | 1 个大文件（program.md, 900 行） | 18 个独立模块（各 49~251 行），有阅读顺序 |
| 可运行 | 完整 benchmark + orchestration | 7 个可运行 Triton 模板 + 测试脚本 |

## 演进路线

### 第零层：补充学习基础设施

| 方向 | 说明 | 优先级 |
|------|------|--------|
| 模型级编排策略 | Profile 整个模型 → Amdahl 排序 → 决定先优化哪个 kernel。当前只覆盖 single-kernel 循环，待补充 multi-kernel 优先级决策 | 中 |
| Anti-Patterns 汇总 | 将 7 个 kernel 的 Common Pitfalls + AutoKernel 通用反模式汇总为一份 checklist | 低 |

### 第一层：新增算子技能（硬件驱动）

随着新指令和数据格式的引入，将持续新增算子技能：

| 硬件代际 | 新增能力 | 计划技能 |
|----------|---------|---------|
| H100 / MI300X | FP8 (E4M3/E5M2) GEMM | `kernels/fp8-gemm/` |
| Blackwell | FP4 GEMM、第二代 TMA | `kernels/fp4-gemm/` |
| Ampere+ | 2:4 结构化稀疏 | `kernels/sparse-gemm/` |
| 跨代际 | MX 格式（MXFP4/MXFP6，micro-scaling） | `kernels/mx-gemm/` |
| 跨代际 | 量化 KV Cache Attention | `kernels/quantized-attention/` |

### 第二层：更新优化知识（范式变迁）

新硬件不只是新增指令 —— 它改变了 kernel 的设计范式：

| 变迁 | 对现有技能的影响 |
|------|-----------------|
| **Micro-scaling 格式** | Tier 2（访存）：per-block scaling factor 改变加载/广播模式；Tier 3（融合）：quantize-GEMM-dequant 必须融合 |
| **Warp 特化** | Tier 4（调度）：producer/consumer warp 模型取代统一 warp 设计 |
| **Cluster 级共享内存** | Tier 5（Hopper+）：多 block 通过分布式共享内存协作 |
| **HBM4 / CXL** | Tier 2：新内存层级改变 tiling 和预取策略 |

### 第三层：更智能的优化循环（最根本的演进）

从"人写规则、agent 选择"演进到自我提升的 agent：

```
当前：     人工策展 SKILL.md  →  智能体遵循指令
                                  ↓
近期：     智能体记录成功的优化  →  存储为新的可复用技能
           （optimize-loop 反馈 → experiments/ → 新 SKILL.md 条目）
                                  ↓
未来：     智能体读 ISA spec + 做实验  →  自主发现优化规则
           （RL 驱动，类似 CUDA-Agent）
```

| 方向 | 方法 | 参考 |
|------|------|------|
| 技能积累（当前） | 人工策展，agent 消费 | 本项目、agent-gpu-skills |
| 技能自生成 | Agent 将成功的优化轨迹存为新技能 | [KernelSkill](https://arxiv.org/abs/2603.10085) 双层记忆 |
| RL 自主探索 | Agent 通过奖励（正确性 + 加速比）自己发现规则，不依赖人写技能 | [CUDA-Agent](https://arxiv.org/abs/2602.24286) |

近期可落地的一步：增强 `optimize-loop/SKILL.md`，让每次成功优化后 agent 自动追加一条结构化记录到 `learned-patterns/` 目录 —— 构建一个随使用增长的项目级技能记忆。

## 知识来源

蒸馏自以下项目：
- [AutoKernel](https://github.com/RightNow-AI/autokernel) —— 6 层调优手册 + 9 个算子基线
- [HF Custom CUDA Kernels Skills](https://huggingface.co/blog/custom-cuda-kernels-agent-skills) —— SKILL.md 格式参考
- [agent-gpu-skills](https://github.com/slowlyC/agent-gpu-skills) —— 多层 GPU 智能体技能（cuda-skill / cutlass-skill / triton-skill / sglang-skill），侧重 CUDA 生态
- [GEAK](https://github.com/AMD-AGI/GEAK) —— AMD CDNA3 知识库
- [KernelAgent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) —— PyTorch 多智能体编排
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) —— 参考实现
- [KernelSkill](https://arxiv.org/abs/2603.10085) —— 多智能体框架 + 专家优化技能
- [CUDA-Agent](https://arxiv.org/abs/2602.24286) —— 智能体 RL 驱动的 CUDA 算子生成
