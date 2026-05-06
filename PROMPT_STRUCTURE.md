# 🤖 Data Agent 完整 Prompt 结构白皮书 (v3.1)

> 本文档描述当前代码中 **实际生效** 的 Prompt 拼接结构，与 `prompt.py` / `react.py` 保持同步。

---

## 一、系统消息 (`role: system`) 拼接顺序

由 `prompt.py` 的 `build_system_prompt()` 按以下顺序拼接：

```
[A] react_system_prompt.txt      ← 核心行为准则
[B] Data Roadmap                 ← 动态数据地图（由 react.py 按难度生成）
[C] Tool Descriptions            ← 工具说明（由 registry.py 自动生成）
[D] response_examples.txt        ← 输出格式示例
[E] 输出格式强制声明              ← 硬编码在 build_system_prompt() 末尾
```

---

## 二、各块详细内容

### [A] `react_system_prompt.txt` — 核心行为准则

**文件**：`agents/prompts/react_system_prompt.txt`

```
You are an elite Data Agent. Your goal is to solve data extraction tasks with 100% precision.

### CORE OPERATING PRINCIPLES
1. Logic Audit (Mandatory): state formula in `thought` before any code
2. Exhaustive Search: return ALL relevant rows, not just samples
3. Type-Safe Alignment: use .astype(str) when joining/filtering
4. Knowledge Primacy: Always check `knowledge.md` for business definitions
5. Large File Strategy: use execute_python for files >1MB
6. No Laziness: verify from full dataset, not previews

### TOOL RULES
- SQL: ONLY for .db files
- Python: use code_lines for multi-line scripts
- answer: ONLY include requested columns (extra = 50% penalty)

### OUTPUT SPECIFICATION
- One JSON block inside ```json fence
- Fields: thought, reflection, data_sufficient, action, action_input
- No text outside the JSON block

### DATA ROADMAP
Below is the schema and relationship map. Use it to identify join paths.
```

> ⚠️ 末尾 `### DATA ROADMAP` 是一个"接头标记"，紧接着会被 [B] 的 Roadmap 内容接上。

---

### [B] Data Roadmap — 动态数据地图

**生成逻辑**：`react.py` 的 `run()` 方法，**按任务难度分流**：

| 难度 | 策略 | 内容 |
|------|------|------|
| `easy` | 精简模式 | 仅文件列表（`list_context_tree` depth=2），避免过载 |
| `medium` / `hard` / `extreme` | 全量模式 | Schema KG + PageIndex 文档树 |

**全量模式下的 Roadmap 结构**（由 `db_navigator.py` + `pageindex_lite.py` 生成）：

```json
=== DATA ROADMAP (SCHEMA & RELATIONS ONLY) ===
{
  "IMPORTANT": "READ knowledge_docs FIRST before writing any query or code. ...",
  "data_assets": {
    "knowledge_docs": ["context/doc/knowledge.md"],   ← 主动扫描，优先阅读
    "databases": [
      {
        "path": "db/data.db",
        "tables": {
          "results": {
            "columns": ["raceId", "driverId", "points"],
            "foreign_keys": [{"from": "raceId", "to_table": "races", "to_column": "raceId"}],
            "row_count": 25840        ← 让 Agent 感知数据规模
          }
        }
      }
    ],
    "csv_files": [{"path": "csv/races.csv", "columns": ["raceId", "year", "name"]}],
    "json_files": [{"path": "json/config.json"}]
  },
  "relationships": {
    "shared_fields": {
      "raceid": ["csv/races.csv", "db/data.db"],   ← 所有跨文件同名字段，无过滤
      "year":   ["csv/races.csv", "db/data.db"]
    },
    "note": "Fields appearing in 2+ files — use as JOIN hints or to align data across sources."
  }
}

=== PAGEINDEX TREE STRUCTURES ===
Document: doc/knowledge.md          ← 文档树状结构，含行号，供 read_doc_lines 精准读取
[0001] Business Definitions (lines 1-80)
  [0002] Scoring Rules (lines 10-40) - Describes how points are allocated per race position
  [0003] DNF Definition (lines 41-55) - DNF means Did Not Finish; counts as 0 points
...
```

**关系提取逻辑（无硬编码）**：
- `all_columns`：以**文件**为粒度统计每个字段名出现在哪些文件（同 DB 内多表算同一个文件，避免重复）
- `shared_fields`：出现在 2+ 个不同文件中的字段，**全量列出**，按文件数降序，不过滤不分类

---

### [C] Tool Descriptions — 工具说明

**生成逻辑**：`registry.py` 的 `describe_for_prompt()` 自动生成

当前注册的工具：

| 工具名 | 用途 |
|--------|------|
| `list_context` | 列出 context 目录结构 |
| `read_csv` | 读取 CSV 预览（max_rows 行） |
| `read_json` | 读取 JSON 预览（max_chars 字符） |
| `read_doc` | 按页读取文本文档 |
| `read_doc_lines` | **精准行读取**，配合 PageIndex 行号使用 |
| `get_doc_structure` | 获取 Markdown 文档的树状结构（同 PageIndex） |
| `inspect_sqlite_schema` | 查看 .db 文件表结构 |
| `execute_context_sql` | 对 .db 执行只读 SQL |
| `execute_python` | 执行 Python 代码，以 context 目录为工作目录 |
| `answer` | **提交最终答案**（唯一终止动作） |

---

### [D] `response_examples.txt` — 输出格式示例

**文件**：`agents/prompts/response_examples.txt`

提供 3 个示例：
1. SQL 查询标准格式（`.db` 文件场景）
2. Python 计算 + Logic Audit 格式（含 `.astype(str)` 类型对齐）
3. 精准提交格式（仅包含请求列，无多余列）

> ⚠️ **当前缺口**：没有示范"先读 knowledge.md"的例子。

---

### [E] 输出格式强制声明

硬编码在 `build_system_prompt()` 末尾：
```
You must always return a single ```json fenced block containing one JSON object
with keys `thought`, `reflection`, `data_sufficient`, `action`, and `action_input`, and no extra text.
```

---

## 三、用户消息 (`role: user`) — 任务提示词

由 `build_task_prompt()` 生成，结构固定：

```
Question: {task.question}
All tool file paths are relative to the task context directory.
When you have the final table, call the `answer` tool.
```

---

## 四、对话历史 (`role: assistant` / `role: user` 交替)

每一步由 `_build_messages()` 拼接历史记录，含**历史压缩**逻辑：

- **正常步骤**：直接追加 `assistant` + `user(Observation)` 轮次
- **历史压缩**：连续失败序列（后面跟了成功步骤的）被折叠为最后一条，思考前注入 `[System: N previous failed attempts omitted]`，节省 Token 同时保留调试上下文
- **连续错误时**：在 Observation 中追加 `[SYSTEM WARNING]` 反思提醒（连续报错 ≥ 3 次触发）

---

## 五、`answer` 工具拦截器 — Final Review

**文件**：`agents/prompts/final_review_prompt.txt`  
**触发**：Agent 第一次调用 `answer` 时，`registry.py` 拦截，返回该 Prompt 要求 Agent 二次核查。

拦截内容（checklist）：
1. **Formatting**：列名和行数是否与问题完全匹配，无多余列
2. **Aggregation**：SUM/AVG/Percentage 数学逻辑是否正确
3. **Time Reference**：年龄/时长计算使用当前年份 **2026**
4. **Knowledge Rules**：是否查阅了 knowledge.md 中的阈值/定义

通过后再次调用 `answer`，第二次正常提交。

---

## 六、`answer` 工具的 normalize 流程

提交时 `answer_normalizer.py` 自动执行：
1. **cell 规范化**：`str.strip()` 去除空格
2. **行去重**：`_dedupe_rows()` 
3. **形状推断**：`_infer_expected_shape()` 根据问题语义推断期望列数
4. **自动裁剪**：若列数超出预期，`_trim_columns()` 智能选列（文本列/数值列）
5. **shape 校验**：列数不符时 raise `ValueError`，强迫 Agent 修正后重试
