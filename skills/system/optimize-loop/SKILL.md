---
name: iterative-kernel-optimization-loop
description: >-
  Orchestrates continuous kernel optimization by chaining profiling, bottleneck diagnosis,
  tier-based optimization, verification, and benchmarking into an iterative loop.
  This is the "main program" that drives the other skills. Use when you want to
  iteratively optimize a kernel until it meets a performance target, similar to AutoKernel.
---

# Iterative Kernel Optimization Loop

## Overview

This skill is the **orchestration layer** that chains the other skills into an automated,
iterative optimization cycle. It turns `awesome_kernel_skills` from a static knowledge
collection into an **agent-driven optimization engine**, similar to
[AutoKernel](https://github.com/RightNow-AI/autokernel)'s `program.md`.

The agent (Cursor / Claude Code / Codex) acts as the execution engine -- this skill
tells it **what to do, in what order, and when to stop**.

## The Loop

```
                    ┌──────────────────────────────┐
                    │  INPUT                        │
                    │  - kernel source (.py)        │
                    │  - target metric & threshold  │
                    │  - target GPU (nvidia/amd)    │
                    └──────────────┬───────────────┘
                                   ▼
              ┌─── Phase 0: BASELINE ───────────────────┐
              │  Run test_*.py → record baseline perf   │
              │  If no test exists, create one first     │
              └──────────────┬──────────────────────────┘
                             ▼
        ┌────────────────────────────────────────────┐
        │  Phase 1: PROFILE                          │
   ┌───▶│  Skill: profiling/SKILL.md                 │
   │    │  Action: run NCU or rocprof                │
   │    │  Output: metrics.json                      │
   │    └──────────────┬─────────────────────────────┘
   │                   ▼
   │    ┌────────────────────────────────────────────┐
   │    │  Phase 2: DIAGNOSE                         │
   │    │  Skill: bottleneck-diagnosis/SKILL.md      │
   │    │  Action: classify memory/compute/latency   │
   │    │  Output: bottleneck type + tier priority    │
   │    └──────────────┬─────────────────────────────┘
   │                   ▼
   │    ┌────────────────────────────────────────────┐
   │    │  Phase 3: OPTIMIZE                         │
   │    │  Skills: tier1~5 SKILL.md (by priority)    │
   │    │  Action: apply ONE optimization at a time  │
   │    │  Rule: modify kernel, keep changes small   │
   │    └──────────────┬─────────────────────────────┘
   │                   ▼
   │    ┌────────────────────────────────────────────┐
   │    │  Phase 4: VERIFY                           │
   │    │  Skill: verification/SKILL.md              │
   │    │  Action: run test → correctness check      │
   │    │  If FAIL → revert change, try next tier    │
   │    └──────────────┬─────────────────────────────┘
   │                   ▼
   │    ┌────────────────────────────────────────────┐
   │    │  Phase 5: BENCHMARK                        │
   │    │  Skill: benchmark/SKILL.md                 │
   │    │  Action: measure TFLOPS or GBps            │
   │    │  Compare vs previous iteration             │
   │    └──────────────┬─────────────────────────────┘
   │                   ▼
   │    ┌────────────────────────────────────────────┐
   │    │  Phase 6: DECIDE                           │
   │    │  - Target met?          → DONE, log result │
   │    │  - Perf improved?       → log, continue    │
   │    │  - Perf regressed?      → revert, try next │
   │    │  - All tiers exhausted? → DONE, log best   │
   │    │  - Max iterations hit?  → DONE, log best   │
   │    └──────────────┬─────────────────────────────┘
   │                   │ not done
   └───────────────────┘
```

## Configuration

Before starting the loop, establish these parameters:

```yaml
kernel_path: skills/kernels/softmax/triton_template.py
test_path: skills/kernels/softmax/test_softmax.py
target_metric: gbps            # "tflops" or "gbps"
target_value: 800              # target GBps (or TFLOPS)
target_pct_of_peak: 0.70       # alternative: 70% of HBM peak
max_iterations: 5              # stop after N optimization rounds
gpu: auto                      # "nvidia", "amd", or "auto" (detect)
```

## Phase Details

### Phase 0: Baseline

1. Detect GPU via `scripts/detect_gpu.py`
2. Run `test_*.py --json` to get baseline correctness + performance
3. Record in optimization log:

```markdown
## Iteration 0 (Baseline)
- GPU: NVIDIA RTX 4090
- Correctness: PASS
- GBps: 1747.0
- Target: 2000 GBps (70% of peak)
- Gap: 253 GBps (12.6%)
```

### Phase 1: Profile

Read `profiling/SKILL.md` and execute:
- NVIDIA: `ncu --set full -o iter{N}_profile python test_*.py`
- AMD: `rocprof --stats python test_*.py`

Extract key metrics into structured form.

### Phase 2: Diagnose

Read `bottleneck-diagnosis/SKILL.md` and apply the flowchart:

```
IF dram_bw_pct high AND compute_pct low → memory-bound → Tier 1-2, then 3
ELIF compute_pct high                   → compute-bound → Tier 3, then 1, 5
ELIF both low                           → latency-bound → Tier 4, then 3
```

Output a **tier priority list**, e.g. `[tier2, tier1, tier3]`.

### Phase 3: Optimize

For the highest-priority **untried** tier:
1. Read the corresponding `tier{N}/SKILL.md`
2. Read the kernel's own `SKILL.md` for kernel-specific guidance
3. Apply **ONE** focused change (do not change multiple things at once)
4. Common changes by tier:

| Tier | Typical Change |
|------|---------------|
| 1 | Adjust BLOCK_SIZE, num_warps, GROUP_SIZE_M |
| 2 | Improve coalescing, add vectorized loads, fix bank conflicts |
| 3 | Fuse epilogue, switch to fast math intrinsics, fix accumulator dtype |
| 4 | Switch to persistent kernel, add Split-K |
| 5 | Use arch-specific features (TMA, WGMMA, MFMA shapes) |

**Rule**: Save a copy of the kernel before modifying so you can revert.

### Phase 4: Verify

Run `test_*.py` (correctness portion only):
- **PASS** → proceed to benchmark
- **FAIL** → revert the change, mark this tier+change as "tried/failed", pick next

### Phase 5: Benchmark

Run `test_*.py --json` (full benchmark):
- Record TFLOPS or GBps
- Compare vs previous iteration

### Phase 6: Decide

```python
if current_metric >= target_value:
    status = "TARGET_MET"
elif current_metric > best_metric:
    best_metric = current_metric
    best_kernel = save_snapshot()
    status = "IMPROVED"
elif current_metric <= prev_metric:
    revert_to_previous()
    mark_tier_change_as_tried()
    status = "REGRESSED"

if iteration >= max_iterations or all_tiers_exhausted():
    restore_best_kernel()
    status = "MAX_ITERATIONS"
```

## Optimization Log Format

Append to `experiments/README.md` after each iteration:

```markdown
## Optimization: {kernel_name} on {gpu}

| Iter | Action | Tier | Correctness | Metric | Delta | Status |
|------|--------|------|-------------|--------|-------|--------|
| 0 | baseline | - | PASS | 1747 GBps | - | - |
| 1 | increase BLOCK_SIZE 4096→8192 | T1 | PASS | 1820 GBps | +4.2% | improved |
| 2 | vectorized loads (float4) | T2 | PASS | 1955 GBps | +7.4% | improved |
| 3 | multi-row=8 | T1 | FAIL | - | - | reverted |
| 4 | fuse with layernorm | T3 | PASS | 2010 GBps | +2.8% | TARGET_MET |

Final: 2010 GBps (70.2% of peak), 4 iterations, target met.
```

## Guardrails

1. **One change per iteration**: Never apply two optimizations simultaneously. You cannot attribute improvement (or regression) otherwise.
2. **Always verify before benchmark**: Never measure performance of an incorrect kernel.
3. **Revert on regression**: Do not keep a change that makes things slower unless it enables a larger subsequent optimization (document why).
4. **Respect platform**: Check `tier5-arch-nvidia` or `tier5-arch-amd-rocm` before using arch-specific features. Never use Hopper TMA on AMD.
5. **Max 5 iterations default**: Diminishing returns kick in fast. If stuck after 5 rounds, the kernel may need an algorithmic redesign rather than tuning.
6. **Log everything**: Every iteration must be recorded. Future optimization attempts should read the log to avoid repeating failed changes.

## When Profiling Is Not Available

If NCU/rocprof cannot be run (e.g. container restrictions, no root), fall back to:

1. Use `test_*.py --json` benchmark numbers as the only signal
2. Classify based on kernel type from the kernel's `SKILL.md` (it states bound type)
3. Follow the tier priority for that bound type
4. Iterate on autotune configs (Tier 1) first -- this is always safe

## Agent Instructions

When asked to optimize a kernel iteratively:

1. **Read this skill first** to understand the loop structure
2. Establish target: ask the user or derive from GPU peak specs
3. Run Phase 0 to get baseline
4. Execute the loop: Profile → Diagnose → Optimize → Verify → Benchmark → Decide
5. After each iteration, update the optimization log in `experiments/`
6. Stop when target is met, max iterations reached, or all viable tiers are exhausted
7. Report final result with summary table

When profiling tools are available, always start with Phase 1 (profile).
When they are not, skip to Phase 3 using the kernel's documented bound type.

## Relationship to Other Skills

```
optimize-loop (this skill) ─── orchestrates ───▶ all other skills
  │
  ├─ Phase 1 → profiling/SKILL.md
  ├─ Phase 2 → bottleneck-diagnosis/SKILL.md
  ├─ Phase 3 → tier1~5/SKILL.md + kernels/*/SKILL.md
  ├─ Phase 4 → verification/SKILL.md
  └─ Phase 5 → benchmark/SKILL.md
```

## References

- [AutoKernel program.md](https://github.com/RightNow-AI/autokernel) -- The original iterative agent-driven kernel optimization loop
- [KernelSkill dual-level memory](https://arxiv.org/abs/2603.10085) -- Reusable optimization skills + short-term backtracking memory
