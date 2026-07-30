# Tool Call Error Analysis — Run 20260506T061609Z

**分析日期**: 2026-05-06
**Run ID**: `20260506T061609Z`
**任务数**: 4 (task_38, task_80, task_86, task_89)
**模型**: qwen3.5-35b-a3b
**最终结果**: 4/4 任务成功

---

## 错误总览

| 错误类别 | task_38 | task_80 | task_86 | task_89 | 合计 |
|----------|:-------:|:-------:|:-------:|:-------:|:----:|
| 工具调用路径不正确（CSV 当数据库） | 1 | 1 | 0 | 1 | **3** |
| SQL 语法错误（GROUP BY 缺失） | 0 | 1 | 0 | 0 | **1** |
| 合约验证失败（使用禁用字段） | 0 | 0 | 0 | 3 | **3** |
| **总计** | **1** | **2** | **0** | **4** | **7** |

---

## 类别 A: 工具调用时文件路径不正确

**根因**: 模型在 CSV 文件上调用了仅支持 SQLite 数据库的工具，导致 `"file is not a database"` 错误。

### task_38 — Step 6

| 字段 | 值 |
|---|---|
| 节点 | `tool` |
| 调用工具 | `inspect_sqlite_schema` |
| 传入参数 | `{"path": "csv/disp.csv"}` |
| 错误信息 | `"file is not a database"` |
| 是否恢复 | 是 — 模型在 Step 7 中意识到 CSV 不是数据库，切换为 `read_csv` |
| 浪费步数 | **1 step** |

### task_80 — Step 6

| 字段 | 值 |
|---|---|
| 节点 | `tool` |
| 调用工具 | `execute_context_sql` |
| 传入参数 | `{"path": "csv/qualifying.csv", "sql": "SELECT number, q3 FROM qualifying WHERE raceId = 903 AND q3 != '' ORDER BY q3", "limit": 20}` |
| 错误信息 | `"file is not a database"` |
| 是否恢复 | 是 — 模型在 Step 7 识别错误，Step 9 改用 `execute_python` 配合 `csv.DictReader` 完成计算 |
| 浪费步数 | **1 step** |

### task_89 — Step 6

| 字段 | 值 |
|---|---|
| 节点 | `tool` |
| 调用工具 | `inspect_sqlite_schema` |
| 传入参数 | `{"path": "csv/results.csv"}` |
| 错误信息 | `"file is not a database"` |
| 是否恢复 | 是 — 模型在 Step 7 识别错误，Step 8 改用 `read_csv` 成功读取 schema |
| 浪费步数 | **1 step** |

### 模式分析

这三个错误展现了一致的错误模式：
1. 模型在 Data Understanding handoff 中收到 CSV 文件路径作为数据源
2. 模型在 `model` 步的推理中确定需要检查文件 schema
3. 模型错误地选择了仅支持 SQLite 的工具而非 `read_csv`
4. 每次都能在下一步恢复，但每个错误消耗了 1 个额外步数

**建议**: 在 `inspect_sqlite_schema` 和 `execute_context_sql` 的工具描述中增加更明确的约束说明，或当路径后缀为 `.csv` 时增加前端校验。

---

## 类别 B: SQL 语法错误

**根因**: 生成的 SQL 语句中聚合函数与普通列混用，缺少 `GROUP BY`。

### task_80 — Step 2 (DataUnderstandingAgent inspector)

| 字段 | 值 |
|---|---|
| 节点 | `understand_and_explore_data` |
| 调用工具 | `execute_probe_query` |
| SQL 语句 | `SELECT COUNT(*), q3 FROM qualifying WHERE raceId = 903 AND q3 IS NOT NULL AND q3 LIKE '1:54.%'` |
| 错误信息 | `Binder Error: column "q3" must appear in the GROUP BY clause or must be part of an aggregate function` |
| 是否恢复 | 是 — 同批复用的另一个查询 `SELECT number, q3 FROM qualifying WHERE raceId = 903 ORDER BY q3 LIMIT 20` 成功执行，模型使用了该查询结果 |
| 浪费步数 | **0 step**（与成功查询同批复用） |

### 根因分析

用户问题中的时间 `0:01:54` 在数据中标准化为 `1:54.000`，但实际数据中不存在精确匹配。DataUnderstandingAgent 试图统计匹配模式的行数，但在 `COUNT(*)`（聚合）旁边放置了普通列 `q3`。这反映出模型在需要"先探查数据再做聚合"的场景下倾向于合并为单条 SQL 而非分成两条。

---

## 类别 C: 合约验证失败

**根因**: DataUnderstandingAgent 在构建 `answer_contract` 时引用了被领域知识标记为禁用的字段。

### task_89 — Step 2 (DataUnderstandingAgent 合约阶段)

| 尝试 | 阶段 | 错误 |
|------|------|------|
| 1 | `contract` (phase) | `Contract uses rejected fields: ['csv/results.csv.position']` |
| 2 | `contract` (retry) | `Contract uses rejected fields: ['csv/results.csv.position']` |
| 3 | `contract` (fallback) | 合并前两次错误，输出部分 handoff |

**背景**: 问题要求找到 2008 年中国大奖赛第二名车手的完成时间。`csv/results.csv.position` 被 knowledge.md 标记为不可靠，推荐使用 `csv/results.csv.positionOrder` 作为最终排名字段。模型在 reasoning 中讨论过这一点，但生成的合约 JSON 中仍包含了 `position` 字段。

**是否恢复**: 部分恢复 — fallback handoff 提供了足够的字段映射（`finish time -> csv/results.csv.time`, `ranked second -> csv/results.csv.positionOrder`），主 solver 基于此成功完成任务。

**浪费步数**: 合约阶段内部消耗了 3 个 sub-step（phase → retry → fallback），但未增加外层主 loop 步数。

**值得注意的是**: 重试并没有修正错误——模型在 retry 中重复了相同的错误。这说明重试 prompt 可能没有足够明确地指出被拒字段的问题。

---

## 汇总统计

| 指标 | 数值 |
|---|---|
| 总工具调用错误数 | **7** |
| 直接浪费的 model 步数 | **3** (task_38: 1, task_80: 1, task_89: 1) |
| 合约内部浪费的 sub-step | **3** (task_89 contract phase/retry/fallback) |
| 模型自恢复率 | **100%** (7/7 错误均被恢复) |
| 零错误任务 | **task_86** (唯一干净的执行) |
| 浪费严重程度 | 低 — 所有任务最终成功，浪费集中在少量额外步数和 sub-step |

---

## 改善建议

1. **工具路由优化**: 对 CSV 文件增加前端路径校验，当用户传入 `.csv` 路径到 `inspect_sqlite_schema` 或 `execute_context_sql` 时，在错误发生前拦截并引导至 `read_csv`。

2. **SQL 生成提示增强**: 在 DataUnderstandingAgent 的 probe query 提示中增加 "当在 SELECT 中同时使用聚合函数和普通列时，必须包含 GROUP BY" 的约束。

3. **合约重试机制改进**: 当合约验证因禁用字段失败时，在 retry prompt 中直接列出具体被拒字段（当前 task_89 的重试未能修正，说明信息传递可能不够明确）。

4. **参考 task_86 的成功模式**: task_86 是唯一零错误的执行，其路径为 `list_context → read_json → execute_python → answer`，全程避开了 SQLite 工具，直接使用 Python 处理数据。
