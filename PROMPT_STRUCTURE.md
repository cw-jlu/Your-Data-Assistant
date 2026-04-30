# 🤖 Data Agent 整体 Prompt 结构白皮书 (v2.2)

本项目采用 **“全知混合导航 + 动态历史反思”** 的双轨 Prompt 架构。通过在系统起始阶段注入高质量的数据路线图（Roadmap），将 Agent 从盲目的文件探索中解放出来，专注于逻辑推理与代码执行。

---

## 1. 系统提示词架构 (System Message Structure)

系统提示词是 Agent 的“大脑预装程序”，由以下部分按顺序拼接而成：

### A. 核心指令 (ReAct System Prompt)
- **定义**：确立 ReAct 思维链协议（Thought -> Action -> Observation）。
- **规范**：强制要求输出严格格式化的 JSON 块。
- **行为准则**：包括如何处理缺失数据、如何调用 `answer` 工具等。

### B. 混合数据路线图 (Hybrid Data Roadmap)
*这是系统的核心竞争力，包含全量 Schema 知识图谱。*

| 模块名称 | 内容组成 | 注入逻辑 |
| :--- | :--- | :--- |
| **[DB] 数据库结构** | 表名、列名(类型)、**总行数**、前 3 行样本。 | 自动扫描所有 `.db` 文件。 |
| **[CSV/JSON] 文件结构** | 文件名、**推断列名**、**总记录数**、前 3 条样本。 | 智能识别 JSON 包装结构（如 `records` 列表）。 |
| **[FK] 关联关系** | 物理外键关系、基于同名字段的逻辑关联建议。 | 自动发现跨表 `ID` 关联。 |
| **[KNOWLEDGE] 业务定义** | `knowledge.md` 中的所有业务规则、指标公式、SQL 示例。 | **全量注入**（无行数截断），确保示例 SQL 可见。 |
| **[DOC SEMANTICS] 文档语义** | 其他 MD/TXT 文件的 LLM 摘要总结。 | 无点数限制，由 LLM 提取核心业务实体。 |
| **[PageRAG] 文档检索** | 针对当前问题的文档目录 (Catalog) 与相关片段。 | 动态检索，提供针对性的背景知识。 |

### C. 工具说明书 (Tool Descriptions)
- 列出所有可用工具及其 JSON Schema 格式（`read_json`, `execute_python`, `answer` 等）。

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
    "thought": "我需要查找严重血栓患者...",
    "reflection": "上一步我发现 A 字段在 JSON 中...",
    "data_sufficient": false,
    "action": "execute_python",
    "action_input": { "code": "..." }
  }
  ```
- **User (role: user)**: 
  ```text
  Observation:
  {
    "ok": true,
    "tool": "execute_python",
    "content": { "success": true, "output": "Found 3 patients..." }
  }
  ```

### 容错增强：反思注入 (Reflection Hint)
当 Agent 连续报错时，系统会在最新的 `Observation` 中注入警告：
> "[SYSTEM WARNING] You have encountered multiple consecutive errors. Please carefully analyze the error messages above... Rethink your current approach."

---

## 4. 为什么这样设计？

1.  **消除首轮冷启动**：通过在 Roadmap 中加入样本数据（Sample），Agent 在第一步就能写出正确的 Python 过滤代码（例如知道性别字段值是 "M" 而不是 "Male"），无需先去“试读”文件。
2.  **业务规则前置**：通过全量注入 `knowledge.md`，Agent 能在第一时间掌握复杂的医学指标定义和 SQL 查询范式（如 `Thrombosis = 2` 代表严重）。
3.  **减少 Token 往返**：高质量的 Roadmap 减少了 Agent 为了解结构而进行的无效工具调用，将有限的 `max_steps` 全部用于核心解题。
