# 🤖 Data Agent 整体 Prompt 结构白皮书 (v3.0)

本项目采用 **“混合导航 + 动态历史反思”** 的双轨 Prompt 架构。通过在系统起始阶段注入高质量的数据路线图（Roadmap），将 Agent 从盲目的文件探索中解放出来，专注于逻辑推理与代码执行。在 V3.0 版本中，我们彻底移除了不稳定的向量检索（PageRAG），全面拥抱基于物理坐标的树状检索（PageIndex Lite）。

---

## 1. 系统提示词架构 (System Message Structure)

系统提示词是 Agent 的“大脑预装程序”，由以下部分按顺序拼接而成：

### A. 核心指令 (ReAct System Prompt)
- **定义**：确立 ReAct 思维链协议（Thought -> Action -> Observation）。
- **规范**：强制要求输出严格格式化的 JSON 块。
- **JSON Compliance**：新增 Rule 19/20，要求极其简练的 Thought 以防止截断，并强调转义与闭合。
- **行为准则**：包括如何处理缺失数据、如何调用 `answer` 工具等。

### B. 混合数据路线图 (Hybrid Data Roadmap)
*这是系统的核心竞争力，包含 Schema 图谱与大文件树状索引。系统会根据任务难度（Easy / Medium / Hard / Extreme）动态决定挂载哪些模块。*

| 模块名称 | 内容组成 | 注入逻辑 |
| :--- | :--- | :--- |
| **[DB] 数据库结构** | 表名、全量列名、总行数。 | 自动扫描所有 `.db` 文件，移除样本以保持轻量。 |
| **[CSV/JSON] 文件结构** | 文件名、全量列名、总记录数。 | 自动识别结构，展示全量字段，不含样本。 |
| **[FK] 关联关系** | 物理外键关系、基于同名字段的逻辑关联建议。 | 自动发现跨表 `ID` 关联。 |
| **[PageIndex] 文档树状索引** | 将超长 Markdown 文件切分为带起始/终止行号的目录树。 | 针对 Extreme/Hard 任务。取代旧版 PageRAG，解决“大海捞针”问题。 |
| **[Knowledge] 业务摘要** | 将 `knowledge.md` 作为一个整体进行 LLM 语义摘要。 | 与 PageIndex 深度融合，强制确保业务规则对 Agent 可见。为保持业务规则的完整性，对该文件关闭自动切块。 |
| **[LLM Summary] 节点摘要** | 异步并发调用大模型，为每个长文本区块生成 15 词极简摘要。 | 与 PageIndex 深度融合，赋予物理区块以语义标签。 |

### C. 工具说明书 (Tool Descriptions)
- 列出所有可用工具及其 JSON Schema 格式（`read_json`, `execute_python`, `get_doc_structure`, `read_doc_lines`, `answer` 等）。

### D. 响应示例 (Response Examples)
- 提供 1-2 个标准的 JSON 响应模板，防止格式崩溃。

---

## 2. 任务提示词 (Task Message Structure) —— `role: user`

在系统消息之后，紧接着发送具体的任务描述：
```text
Question: [用户提出的原始问题]
All tool file paths are relative to the task context directory. 
When you have the final table, call the `answer` tool.
```

---

## 3. 动态对话历史 (Conversation History)

Agent 运行过程中的每一轮迭代都会作为对话历史追加：

- **Assistant (role: assistant)**: 
  ```json
  {
    "thought": "我需要查找严重血栓患者，让我看看 knowledge.md 里怎么定义的...",
    "reflection": "我应该用 get_doc_structure 看看结构，或者直接用 read_doc_lines 读取特定章节...",
    "data_sufficient": false,
    "action": "read_doc_lines",
    "action_input": { "path": "doc/knowledge.md", "start_line": 1, "end_line": 50 }
  }
  ```
- **User (role: user)**: 
  ```text
  Observation:
  {
    "ok": true,
    "tool": "read_doc_lines",
    "content": { "content": "..." }
  }
  ```

### 容错增强：反思注入 (Reflection Hint)
当 Agent 连续报错时，系统会在最新的 `Observation` 中注入警告：
> "[SYSTEM WARNING] You have encountered multiple consecutive errors. Please carefully analyze the error messages above... Rethink your current approach."

---

## 4. 为什么 V3.0 这样设计？

1.  **极简 Roadmap 导航**：通过在 Roadmap 中加入全量列名（无样本数据），Agent 在第一步就能精准锁定目标字段并构建正确的逻辑，同时避免了因样本数据过大导致的上下文溢出。
2.  **确定性替代模糊性**：彻底摒弃基于 Embedding 的 Vector RAG，改为“物理行坐标 (Line Ranges) + 精确切块工具”，彻底消灭大模型提取数据时的幻觉和遗漏问题。
3.  **并发初始化**：通过 `asyncio` 并发计算章节摘要，将庞大文档结构的“读图”时间压缩至最低，保证系统不超时。
4.  **减少 Token 往返**：高质量的 Roadmap 减少了 Agent 为了解结构而进行的无效工具调用，将有限的 `max_steps` 全部用于核心解题。
