# Kernel Skills 知识内化路径

18 个 SKILL.md，约 1,500 行。目标不是"跑实验"（那是 agent 的事），而是**读懂每一条建议背后的物理直觉**。

---

## 阅读总览

```
18 个 SKILL.md 分三类：

系统技能（5 个）—— HOW：优化流程怎么跑
  optimize-loop (251L)  profiling (97L)  bottleneck-diagnosis (71L)
  verification (89L)    benchmark (87L)

优化技能（6 个）—— WHY：每层调优的物理原理
  tier1-block-tiling (73L)       tier2-memory-access (65L)
  tier3-compute-fusion (64L)     tier4-advanced-scheduling (63L)
  tier5-arch-nvidia (65L)        tier5-arch-amd-rocm (74L)

算子技能（7 个）—— WHAT：具体算子的算法 + 陷阱
  gemm (103L)  softmax (84L)  rmsnorm (77L)  flash-attention (76L)
  cross-entropy (55L)  rotary-embedding (49L)  fused-moe (72L)
```

**建议阅读顺序**：先 tier1~5（建立优化思维框架） → 再 kernel skills（用框架理解具体算子） → 最后 system skills（理解流程编排）。

每个 SKILL.md 里，**"Agent Instructions" 段落可以跳过**——那是给 AI 的执行指令，不是知识。**重点读 Core Technique、Common Pitfalls、以及各种表格**。

---

## 第一周：优化技能层（6 个 SKILL.md）—— 建立思维框架

这 6 个文件是整个知识体系的骨架。读完它们，你面对任何 kernel 都能说出"先查什么、再改什么"。

### Day 1-2：Tier 1 + Tier 2（发射几何 + 内存层级）

#### `tier1-block-tiling/SKILL.md`（73 行）

**核心要理解的**：
- BLOCK_SIZE 不是"越大越好"，而是在 **arithmetic intensity** 和 **occupancy** 之间做 trade-off
- `num_warps` 增多 → 隐藏延迟 → 但寄存器压力增大 → 可能反降 occupancy
- GROUP_SIZE_M 的本质：让相邻 thread block 访问相邻 B 列，提高 L2 命中率
- AMD wavefront=64 意味着 `num_warps=4` 就是 256 线程，和 NVIDIA `num_warps=8` 的线程总数一样

**自测问题**：
1. 给定 M=1024, N=1024, K=1024 的 GEMM，BLOCK_M=128, BLOCK_N=128，总共多少个 thread block？4090 有 128 SM，每个 SM 能跑几波？
2. 如果把 BLOCK_M 从 128 改成 64，thread block 数量翻倍，但每个 block 的 arithmetic intensity 降低——什么情况下这是值得的？
3. 你的 GEMM 实验里，1024 size 只有 0.57x cuBLAS。实验日志说"65 blocks for 128 SMs, poor occupancy"——用 Tier 1 的知识解释为什么。

#### `tier2-memory-access/SKILL.md`（65 行）

**核心要理解的**：
- Coalescing 的物理含义：同一个 warp 的 32（或 64）个线程，在同一个 cycle 访问**连续地址**，GPU 把它们合并成一次 128-byte（NVIDIA）或 256-byte（AMD）事务。如果地址不连续，就变成多次事务
- Bank conflict：shared memory 有 32 个 bank，每个 4 bytes。如果同一 warp 的多个线程碰巧访问同一个 bank 的不同地址 → 串行化。解法：padding（+1 列）或 swizzle
- Software pipelining（`num_stages`）的本质：把"加载下一轮数据"和"计算当前数据"在时间上重叠。`num_stages=3` 意味着 3 个缓冲区轮转
- Register vs Shared 的 trade-off：SMEM 增多 → block 能同时 stage 更多数据 → 但 max blocks/SM 下降

**自测问题**：
1. 为什么你的 GEMM 实验里 `tl.multiple_of` pointer hints 导致 -16.5% regression？（提示：Triton 编译器在 Ada 上已经做了类似的优化，手动 hint 干扰了编译器的优化 pass）
2. 一个 softmax kernel 每行 load 一次、store 一次，为什么是 memory-bound？请用 arithmetic intensity 估算。
3. NVIDIA shared memory 32 banks × 4 bytes，画出访问 `shared[threadIdx.x][0]` 时 32 个线程的 bank 分布。如果改成 `shared[threadIdx.x][threadIdx.x]` 呢？

---

### Day 3：Tier 3（计算优化与融合）

#### `tier3-compute-fusion/SKILL.md`（64 行）

**核心要理解的**：
- **Fused epilogue** 的 ROI 判断：如果 epilogue（bias/activation/quantize）是 memory-bandwidth limited（即数据要重新从 HBM 读一遍），融合的收益巨大。如果已经 compute-saturated，融合只是省了 launch，但可能增加 register pressure
- **Inner loop hygiene**：K-loop 里的每一条多余指令都被放大 `ceil(K/BLOCK_K)` 倍。分支、invariant load、类型转换——都要移出去
- **Fast math 的精度 trade-off**：`__expf` 比 `exp` 快但精度低。在 softmax 里（训练）要小心；在 RoPE 的 sincos 里通常可以用

**自测问题**：
1. RMSNorm 的 forward：`out = x / sqrt(mean(x^2) + eps) * weight`。如果你把它拆成 3 个 kernel（求均方、rsqrt、乘以 weight），vs 1 个 fused kernel，数据要从 HBM 读写几次？
2. 为什么 GEMM kernel 的 accumulator 必须用 FP32，即使输入是 FP16？（提示：FP16 的 range 是 ±65504，两个 FP16 数 dot product 后结果可能已经溢出）
3. 什么情况下"不应该 fuse"？文档里说了两种情况，用你自己的话解释。

---

### Day 4：Tier 4（高级调度）

#### `tier4-advanced-scheduling/SKILL.md`（63 行）

**核心要理解的**：
- **Split-K**：把 K 维度拆给多个 block 并行做 partial sum，然后再 reduce。适用场景：M、N 很小（grid 不够大），但 K 很大
- **Persistent kernel**：只 launch ≈ #SM 个 block，每个 block 内循环处理多个 tile。好处：省 launch overhead、可以控制 tile 遍历顺序（L2 友好）。坏处：代码复杂、你的 GEMM 实验里在 4090 上 regression 了 2.5%
- **Warp specialization**：Hopper 的 producer/consumer warp 模型。还不到你现阶段需要的深度
- **何时用 Tier 4**：前提是 Tier 1~3 已经 plateau，且 profiler 显示 tail imbalance 或 launch overhead

**自测问题**：
1. 你的 GEMM 实验 Iter 3：persistent kernel regression 2.5%。日志分析说"hardware scheduler handles tile dispatch faster than software loop"。为什么在 4090 上是这样，但在 H100（更多 SM）上可能不同？
2. Split-K 的代价是什么？（额外的 HBM 写 partial sum + atomic/reduction）什么时候这个代价大于收益？
3. Stream-K 解决的是什么问题？用"最后一波 thread block 空闲"来解释。

---

### Day 5：Tier 5a + 5b（NVIDIA + AMD 架构特化）

#### `tier5-arch-nvidia/SKILL.md`（65 行）

**核心要理解的**：
- 三代架构的关键差异：Ampere（cp.async）→ Hopper（TMA + WGMMA + cluster）→ Ada（高 FP16 throughput 但消费级带宽）
- 4090 的特殊性：FLOPS 很高但 memory bandwidth 相对低 → 很多 kernel 实际上是 memory-bound → fusion 和 L2 优化比算力优化更重要
- NCU metrics 速查：低 sm_throughput + 低 dram_throughput = occupancy/launch 问题（Tier 1/4）；高 DRAM + 低 TC = memory path（Tier 2）

#### `tier5-arch-amd-rocm/SKILL.md`（74 行）

**核心要理解的**：
- Wavefront-64 的全局影响：所有关于"warp 内行为"的直觉都要乘以 2
- MI300X 的内存优势：5.3 TB/s HBM3 + 256 MB L2。Memory-bound kernel 理论上应该比 H100 更快——但你的 Softmax 实验里反而输给 PyTorch（0.82x）
- MFMA vs WMMA/WGMMA：不同的 matrix fragment shape，BLOCK_K=32 是 CDNA 的 sweet spot
- `waves_per_eu`：compute-bound 用 2~4，memory-bound 用 4~8（控制每个 EU 的 wave 数量来平衡 occupancy）
- HIP/Triton 的实际限制：`num_stages>=1`、`tl.math.tanh` 缺失等

**自测问题**：
1. MI300X 有 304 CU × 5.3 TB/s HBM3。为什么你的 GEMM 只达到 0.63x rocBLAS？用 Tier 5 AMD 知识列出可能的原因（autotune 空间小？MFMA 利用率低？调度？）
2. 4090 上 Softmax 1.58x，MI300X 上 0.82x。两者都是 memory-bound kernel。为什么 AMD 反而更差？（提示：PyTorch ROCm 有定制的 softmax kernel，且 MI300X 的高带宽需要更高的并行度才能喂饱）
3. 如果要把一个 NVIDIA Triton kernel 移植到 AMD，你需要改哪些东西？列一个 checklist。

---

## 第二周：算子技能层（7 个 SKILL.md）—— 用框架理解具体算子

有了 Tier 1~5 的框架，现在读每个算子时，要练习的能力是：**一看到算子就能判断瓶颈类型 → 对应到 tier 优先级 → 知道哪些优化有效**。

### Day 6：Compute-Bound 算子

#### `gemm/SKILL.md`（103 行）—— 最重要，花 2 小时

**对照你的实验读**：这是唯一一个你已经做过 5 轮优化的算子。结合 `experiments/gemm_optimization.md` 一起读。

**核心要理解的**：
- Back-of-envelope：GEMM 的 arithmetic intensity = `2MNK / (2(MK+KN+MN) × elem_size)`。当 M=N=K=2048, FP16：intensity ≈ 2048/6 ≈ 341 ops/byte。4090 的 compute/BW 比 ≈ 330 TFLOPS / 1 TB/s ≈ 330 ops/byte → compute-bound（intensity > compute/BW 比）
- L2 grouping 为什么对 GEMM 特别重要：不分组时，相邻 block 读的 B 列完全不同，L2 命中率低
- Tensor core 的硬约束：BLOCK_K ≥ 16（NVIDIA WMMA）或 ≥ 32（AMD MFMA），否则 tl.dot 走不了 tensor core
- Common Pitfall #1（K 边界 mask）：你的 EVEN_K 实验（Iter 2b）就是处理这个

**自测**：打开 `triton_template.py`，逐行解释 kernel 代码。能否不看 SKILL.md 说出每段代码对应的优化技巧？

#### `flash-attention/SKILL.md`（76 行）

**核心要理解的**：
- Online softmax 的关键 insight：不需要先完整算出 max，再算 exp，再算 sum。可以一边扫 K tiles 一边维护 running max `m_i` 和 running sum `l_i`，遇到更大的 max 时用 `alpha = exp(m_old - m_new)` 修正之前的累加
- 两个 GEMM：`QK = Q @ K^T`（给出 attention scores）和 `acc += softmax(QK) @ V`（加权求和）。两个都走 tensor core
- Causal mask 的 early termination：如果当前 K tile 的所有 position 都在当前 Q tile 之后（全被 mask），直接 skip。`kv_end = min(N, (pid_m+1)*BLOCK_M)`
- 瓶颈类型：IO-bound（读 Q/K/V 从 HBM）+ compute-bound（两个 GEMM），混合型
- Pitfall #1（rescaling）：更新 m_i 后忘了 `acc *= alpha[:, None]` → 之前的累加值没有被修正 → 输出错误

**自测**：
1. 不看代码，能否画出 FlashAttention 的 tile loop 结构？（外循环 Q tile，内循环 K/V tile，每个内循环里做什么？）
2. 为什么 FlashAttention 是 O(N) 内存而不是 O(N^2)？（因为不需要 materialize 完整的 N×N attention matrix）
3. D（head_dim）为什么必须是 constexpr？（因为 tensor core tiling 需要编译期知道 D 来确定 fragment shape）

#### `fused-moe/SKILL.md`（72 行）

**核心要理解的**：
- MoE 的特殊性：不是一个大 GEMM，而是多个小 GEMM（每个 expert 处理不同子集的 token）。核心挑战是 routing + GEMM 的融合
- Token-Expert 映射的间接性：`sorted_token_ids` 让 block 不直接读 token index，而是通过排序后的映射表 → 让同一个 expert 的 token 连续排列 → GEMM 更规整
- 小 M 问题：如果每个 expert 只分到几个 token（M 很小），grid 不够大 → 需要 BLOCK_M=16 这种极小 tile
- Pitfall #1（token ID 映射）：`sorted_token_ids // top_k` 才是原始 token index，除以 top_k 是因为同一个 token 可能被路由到 k 个 expert

**自测**：
1. 一个有 8 个 expert、top_k=2 的 MoE 层，64 个 token。平均每个 expert 分到多少 token？这个 M 值对 GEMM kernel 意味着什么？
2. vLLM 的 fused_moe 为什么用 persistent kernel？（因为 expert 数量多、每个 expert 的 M 小 → 大量小 GEMM → 用 persistent 减少 launch overhead + 控制 L2 遍历）

---

### Day 7：Memory-Bound 算子

#### `softmax/SKILL.md`（84 行）

**核心要理解的**：
- Back-of-envelope：Softmax 每个元素读 1 次、写 1 次，计算 max + exp + sum + div ≈ 4 ops/element。Arithmetic intensity ≈ 4 ops / (2 × 2 bytes) ≈ 1 op/byte → 极度 memory-bound
- BLOCK_SIZE = next_power_of_2(n_cols)：必须覆盖整行，因为 softmax 是 per-row 归约
- Multi-row processing 的直觉：如果 n_cols=512，一个 program 只处理 1 行 → 占用资源太少 → GPU 利用率低。处理 4~8 行 → 每个 program 有更多工作 → 更好的 latency hiding
- Pitfall #2（FP16 reduction）：exp 的结果可以很大，FP16 只有 ±65504 range → sum 会溢出

**自测**：
1. 为什么 softmax 不能像 GEMM 那样做 tiling？（因为 softmax 需要整行的 max 和 sum，不能只看局部。除非用 online softmax 分 chunk——但那是 FlashAttention 的做法）
2. MI300X 上 Softmax 0.82x vs PyTorch。如果你要改进，根据 Tier 1~2 知识，你会先尝试什么？

#### `rmsnorm/SKILL.md`（77 行）

**核心要理解的**：
- 和 Softmax 同为 row-parallel、memory-bound，但更简单：只需要 sum(x^2)，不需要 max
- `rsqrt` 是关键：`1/sqrt(x)` 在 GPU 上是单条指令。用 `sqrt` + `divide` 就是 2 条指令、2 倍开销
- 为什么 4090 上 9.5x speedup 这么大：PyTorch eager 的 RMSNorm 是 3 个独立 kernel（square → mean → multiply），每个都要读写 HBM。Fused 只读写 1 次
- Pitfall #4（AMD LDS 64 KB）：BLOCK_SIZE 不能太大，否则 register + LDS 超预算 → occupancy 塌陷

**自测**：
1. RMSNorm 的 arithmetic intensity 是多少？（每个元素：读 x + 读 weight → 2 reads；写 out → 1 write；计算 x^2 + sum + rsqrt + multiply ≈ 4 ops。Intensity ≈ 4/(3×2) ≈ 0.67 ops/byte → 极度 memory-bound）
2. 为什么在 MI300X 上只有 2.7x speedup（vs 4090 的 9.5x）？可能的原因？（提示：MI300X 的 PyTorch eager RMSNorm 可能已经有一定程度的优化，或者 Triton 在 AMD 上的 codegen 没有充分利用 5.3 TB/s 带宽）

#### `cross-entropy/SKILL.md`（55 行）+ `rotary-embedding/SKILL.md`（49 行）

这两个是最薄的 SKILL.md，可以快速过。

**Cross-Entropy 核心**：
- Fused log-softmax + NLL：不 materialize 完整 softmax，只算 `logits[target] - max - log(sum(exp))`
- 大 vocab（128K）的挑战：BLOCK_SIZE = next_power_of_2(128K) = 131072 → 寄存器装不下 → 需要 chunked approach
- Pitfall #1：target 是 int64，大 vocab 时 int32 会溢出

**RoPE 核心**：
- 纯 memory-bound（4 reads + 2 writes, 几乎没有计算）
- 两种 layout（interleaved vs split-half）必须和模型一致
- 是最适合"drop-in 替换 PyTorch eager"的算子（简单、收益确定）

---

### Day 8-9：系统技能层（5 个 SKILL.md）—— 理解编排逻辑

#### `optimize-loop/SKILL.md`（251 行）—— 最长，但你已经通过 GEMM 实验体验过

**重点不是读流程图（你已经做过了），而是理解设计决策**：
- 为什么"一次只改一个优化点"？（归因：如果同时改两个东西，性能变了你不知道是哪个导致的）
- 为什么"revert on regression"？（防止累积负面改动。你的 persistent kernel 实验就是正确 revert 的案例）
- "Max 5 iterations"的含义：diminishing returns。如果 5 轮没达标，问题可能不是 tuning 而是 algorithmic

#### `profiling/SKILL.md`（97 行）

**核心要理解的（不是命令本身，而是 metrics 的含义）**：
- NCU 的 5 个关键 metric：sm_throughput、dram_throughput、l1tex_throughput、occupancy、global_load_efficiency
- rocprof 的关键 counter：SQ_INSTS_MFMA（tensor core 使用量）、TCC_HIT/TCC_REQ（L2 命中率）、SQ_WAVE_CYCLES（occupancy）
- Roofline 推导：`arithmetic_intensity = flops / bytes`。把这个点画到 roofline 图上，看它落在 memory-bound 区还是 compute-bound 区

#### `bottleneck-diagnosis/SKILL.md`（71 行）

**核心就是那个 flowchart**——熟练到看到 profiling 数据能 1 秒判断：

```
高 DRAM + 低 compute → memory-bound → Tier 1-2
高 compute + 低 DRAM → compute-bound → Tier 3, 然后 Tier 1/5
两个都低 → latency-bound → Tier 4（fusion/persistent）
```

以及 GEAK 的 Prefer/Consider/Deprioritize 框架——避免在 memory-bound kernel 上浪费时间做 instruction shaving。

#### `verification/SKILL.md`（89 行）+ `benchmark/SKILL.md`（87 行）

快速过即可。关键 takeaway：
- 5-stage verification：基本正确性 → dtype 敏感度 → 边界条件 → 确定性 → 压力测试
- Benchmark 要用 GPU warmup + 多次取 median，不是跑一次

---

## 第三周：交叉理解 + 对照 AutoKernel program.md

### Day 10-11：对照阅读

打开 `kernels/autokernel/program.md`（你本地已有），和上面 18 个 SKILL.md 做交叉对照：

| 对照点 | AutoKernel program.md | awesome_kernel_skills |
|--------|----------------------|----------------------|
| Tier 1 Block Tuning | 第 435~446 行 | `tier1-block-tiling/SKILL.md` |
| Tier 2 Memory | 第 448~469 行 | `tier2-memory-access/SKILL.md` |
| Tier 3 Compute | 第 471~486 行 | `tier3-compute-fusion/SKILL.md` |
| Tier 4 Advanced | 第 488~510 行 | `tier4-advanced-scheduling/SKILL.md` |
| Tier 5 Arch | 第 512~545 行 | `tier5-arch-nvidia` + `tier5-arch-amd-rocm` |
| Tier 6 Kernel-specific | 第 547~576 行 | 7 个 `kernels/*/SKILL.md` |
| Anti-patterns | 第 586~595 行 | 分散在各 SKILL.md 的 Common Pitfalls |

**关注差异**：AutoKernel 的 program.md 在某些地方更具体（例如 CUDA C++ 的 wmma API 示例），你的 SKILL.md 在 AMD 方面更详细。两者互补。

### Day 12-14：综合自测

**终极自测**：给自己一个没见过的算子（比如 LayerNorm 或 SiLU），不看任何文档，能否在 30 分钟内：

1. 判断瓶颈类型（memory/compute/latency）+ back-of-envelope 推算
2. 列出 Tier 优先级（该先调什么）
3. 写出 Triton kernel 的关键结构（不需要能跑，伪代码即可）
4. 列出 3 个可能的 Common Pitfalls
5. 说出 NVIDIA vs AMD 的关键差异点

如果 30 分钟内能完成以上 5 点，Phase 1 知识内化基本完成。

---

## 进度追踪

### SKILL.md 阅读 + 自测 Checklist

| # | SKILL.md | 行数 | 阅读 | 自测完成 | 笔记 |
|---|----------|------|------|---------|------|
| 1 | tier1-block-tiling | 73 | [ ] | [ ] | |
| 2 | tier2-memory-access | 65 | [ ] | [ ] | |
| 3 | tier3-compute-fusion | 64 | [ ] | [ ] | |
| 4 | tier4-advanced-scheduling | 63 | [ ] | [ ] | |
| 5 | tier5-arch-nvidia | 65 | [ ] | [ ] | |
| 6 | tier5-arch-amd-rocm | 74 | [ ] | [ ] | |
| 7 | gemm | 103 | [ ] | [ ] | |
| 8 | flash-attention | 76 | [ ] | [ ] | |
| 9 | fused-moe | 72 | [ ] | [ ] | |
| 10 | softmax | 84 | [ ] | [ ] | |
| 11 | rmsnorm | 77 | [ ] | [ ] | |
| 12 | cross-entropy | 55 | [ ] | [ ] | |
| 13 | rotary-embedding | 49 | [ ] | [ ] | |
| 14 | optimize-loop | 251 | [ ] | [ ] | |
| 15 | profiling | 97 | [ ] | [ ] | |
| 16 | bottleneck-diagnosis | 71 | [ ] | [ ] | |
| 17 | verification | 89 | [ ] | [ ] | |
| 18 | benchmark | 87 | [ ] | [ ] | |

### 毕业标志

| 能力 | 验证方式 | 达标 |
|------|---------|------|
| 看到算子，30s 内判断 memory/compute bound | back-of-envelope arithmetic intensity | [ ] |
| 给定 profiling 数据，选出正确的 tier 优先级 | bottleneck flowchart | [ ] |
| 对 7 个算子各说出 ≥2 个 Common Pitfall | 不看文档 | [ ] |
| 解释 NVIDIA vs AMD 的 5 个核心差异 | wavefront、MFMA、LDS、num_stages、waves_per_eu | [ ] |
| 对照 AutoKernel program.md 能说出 skill 的对应关系 | 交叉阅读 | [ ] |
| 新算子 30 分钟综合自测通过 | Day 12-14 终极自测 | [ ] |
