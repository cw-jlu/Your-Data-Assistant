# Data Agent 前置 Handoff 生成流程修改意见书

> **注意：此提案涉及的手写引导回路（GuidedDataUnderstandingLoop）已于 2026-05-09 被移除。**
> 当前架构直接将 data catalog JSON 注入给主 Agent，不再生成 Handoff。此文档仅做历史参考。

## 1. 现状与运行瓶颈分析（已过时）

结合当前的代码架构（`src/data_agent_baseline/inspectors/data_understanding_agent.py`）以及实际的运行日志（`artifacts/runs/20260507T140404Z/task_163`），系统在生成 Data Understanding Handoff 时的表现如下：

### 代码理论侧
当前架构通过 `GuidedDataUnderstandingLoop` 来生成 Handoff。虽然系统引入了 `profile_guided_fast_path` 配置试图加速，但实际上这个“快速通道”仅仅跳过了 `overview` 阶段（以及在非常严苛的单表情况下的 `fabric` 阶段）。流程中最为耗时的 **`grounding`**（字段映射）和 **`contract`**（答题契约）阶段，即使在 Global Data Profile 已经生成的情况下，依然会被强制放入大语言模型（LLM）的串行调用链路中。

### 实际运行侧（以 Task 163 为例）
从 `task_163` 的 `data_understanding_trace.json` 日志中可以清晰看到以下执行序列：
1. `deterministic_context` (纯规则提取上下文)
2. `overview_skipped` (成功命中了 Global Profile 快速通道)
3. `grounding` (第 1 次调用 LLM，并决定调用工具)
4. `grounding_tools` (执行工具探查)
5. `grounding` (第 2 次调用 LLM 综合工具结果)
6. `fabric` (第 3 次调用 LLM，因为确定性连接路径未能覆盖多表场景)
7. `contract` (第 4 次调用 LLM)

**结论**：即使在理想的前提下，生成一份 Handoff 仍需 **4 次以上的 LLM 串行调用**。这是导致前置流程“太慢了”的核心瓶颈。高质量的 Global Data Profile 虽然被传入，但仅充当了上下文背景，并未实现跳过繁琐分步推理的初衷。

---

## 2. 核心优化目标

在**前置 Global Data Profile 生成质量不错**的前提下，充分信任该画像提供的实体关系与结构认知。打破 `GuidedDataUnderstandingLoop` 僵化的状态机机制，将 Handoff 生成的 LLM 交互次数从 4~5 次**压缩至 1 次（一镜到底），甚至 0 次（纯规则装配）**。

---

## 3. 具体修改方案建议

为了大幅削减时间开销，提供以下三种方案。推荐采用**方案 A** 或**方案 A+B 混合**。

### 方案 A：引入“一镜到底”合并生成（One-Shot Generation）[🌟 最优先推荐]
**修改逻辑**：
当 `profile_guided_fast_path` 为 `True` 且 `global_data_profile` 有效时，**彻底跳过** `GuidedDataUnderstandingLoop` 分步状态机。
直接构建一个名为 `OneShotHandoffDraft` 的大 Pydantic 模型（合并 grounding、fabric 和 contract），将 `question`、`context_bundle` 和 `global_data_profile` 作为上下文，请求 LLM **一次性输出完整的 Handoff JSON**。

**预期收益**：
- LLM 串行等待次数直接从 4-5 次变为 1 次。
- 考虑到现在的大模型在长上下文下的遵循能力已非常强大，单次 Prompt 只要规则清晰，足以完成多步骤逻辑，整体延迟将下降 **70% 以上**。

### 方案 B：强化“纯确定性”旁路（Enhanced Deterministic Fallback）
**修改逻辑**：
目前的 `_deterministic_fabric_from_context` 退化条件过于苛刻（例如只有选定表 <= 1 才能跳过 LLM）。
由于 Global Data Profile 质量已经很好，我们可以增强规则提取（`_build_handoff`）：
1. 用向量检索（Semantic Index）强制绑定高分 Grounded Concepts。
2. 用图算法（结合 schema 和 profile 中的外键声明）硬连 `JoinPaths`。
如果规则构建出的 Handoff 能够完全覆盖问题中的名词并且连通图无断点，则实现 **0 次 LLM 交互**，直接返回。如果置信度低，再走 LLM（如方案 A）。

**预期收益**：
对结构清晰的标准问题做到毫秒级生成。

### 方案 C：合并过度细分的 Prompt 阶段（Fallback for Guided Loop）
如果必须保留一定程度的重试和反思机制（例如出于精度考虑）：
至少应该将 `overview`、`grounding`、`fabric` 三步合并为一步：“`grounding_and_fabric`”。因为字段的对齐（Grounding）与表连接（Fabric）在数据库分析中高度耦合，强行拆分不仅增加了耗时，还可能导致模型上下文割裂。

---

## 4. 代码级实施路径（基于方案A的简述）

1. **修改入口路由** (`data_understanding_agent.py`):
   在 `DataUnderstandingAgent.run` 中，当触发快速通道时，不再实例化 `GuidedDataUnderstandingLoop`。
   ```python
   if self.config.profile_guided_fast_path and _is_valid_global_data_profile(global_data_profile):
       # 新增一镜到底的调用函数
       handoff = self._generate_one_shot_handoff(task.question, context_bundle, global_data_profile)
   elif use_guided_llm:
       # 保留原有慢速分步通道作为兜底或高精度模式
       guided_result = GuidedDataUnderstandingLoop(...).run(...)
   ```

2. **新增 Prompt 与 Pydantic Model**:
   在 `prompts.py` 中新增 `ONE_SHOT_HANDOFF_SYSTEM_PROMPT`。定义一个包含 `GroundedConcept`、`JoinPath` 和 `AnswerContract` 嵌套的单一返回结构。

3. **对比验证**:
   完成代码修改后，运行 `tests/test_runner.py` 或使用 `artifacts/` 下的测评工具。对比新旧两条路径的 `primary_proxy_score` 和 `duration`，确保加速后精度无显著退化。