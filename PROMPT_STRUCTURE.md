# 🤖 Data Agent 整体 Prompt 结构白皮书 (v2.1)

本项目采用 **“全知静态导航 + 动态历史反思”** 的双轨 Prompt 架构，旨在通过最大化背景上下文质量来降低 Agent 的推理压力。

---

## 1. 静态基础层 (Static Foundation)
*在任务开始（Step 1）时即完整注入，为 Agent 提供全局视野。*

| 组件 | 内容描述 | 目的 |
| :--- | :--- | :--- |
| **System Prompt** | ReAct 协议定义、JSON 输出规范、工具说明书。 | 确立 Agent 的思维框架与行为边界。 |
| **Data Roadmap** | 包含 [DB] 样本、[FK] 外键、[KNOWLEDGE] 业务定义。 | 让 Agent 拥有“上帝视角”，无需盲目探索文件。 |
| **Semantic Index** | 每个文档的 3-5 点语义概览（DOC SEMANTICS）。 | 指引 Agent 决定何时去“翻阅”哪个文档。 |
| **Pre-RAG Context** | 针对当前问题，由 PageRAG 预先检索到的相关原文。 | **冷启动加速**：很多简单问题在 Step 1 就能直接根据此内容给出答案。 |

---

## 2. 动态增长层 (Dynamic Growth)
*随着 ReAct 循环（思考 -> 行动 -> 观察）不断追加，记录 Agent 的决策演进。*

### A. 履历链 (History Chain) —— `role: assistant` & `role: user`
每完成一个步骤，都会在消息队列尾部顺序追加：
1.  **Assistant 回复**：上一步的 `thought`（思维过程）、`reflection`（自我反思）和 `action`。
2.  **User 观察 (Observation)**：工具执行的真实反馈，格式为 JSON。
    *   *示例：* `{"ok": true, "tool": "python_executor", "content": "Query result: 42"}`

### B. 反思提示 (Reflection Hint) —— 注入于 `Observation`
这是**容错机制**的动态体现。当 Agent 连续报错达到阈值（默认 3 次）时：
*   **注入方式**：在最新的 Observation JSON 中添加 `error_hint` 字段。
*   **内容**：提示 Agent “你已经连续错了 X 次，请仔细检查 JSON 格式和工具规范，重新审视你的方案”。

### C. 熔断与反馈 (Circuit Breaker)
*   **死循环检测**：如果动态检测到 Agent 连续执行了 3 次完全相同的 Action，系统会触发异常观察，强迫模型停止重复行为。

---

## 3. 完整 Prompt 拼接逻辑 (Pseudo-code)

```python
# 1. 初始构建 (Step 1)
Prompt = System_Instructions + Data_Roadmap + Task_Question

# 2. 循环演进 (Step N)
For each completed_step:
    Prompt += Assistant_Response(N-1)
    Prompt += Observation(N-1)
    
    If consecutive_errors >= threshold:
        Prompt.last_observation += Reflection_Warning
```

---

## 4. 优势总结
1.  **极高信噪比**：通过 PageRAG 预检索和 Title-only 策略，只投喂高度相关的片段，避免 Context Window 膨胀。
2.  **自我纠正**：动态的 `reflection_hint` 机制显著提升了 Agent 在遇到编码、路径或语法错误时的“生还率”。
3.  **零探索成本**：静态 Roadmap 确保 Agent 在起跑线上就掌握了数据库结构和业务词典，将有限的步数全部用于解决核心逻辑。
