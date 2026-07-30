## Data Agent 截断设计评估报告

### 一、概述

本报告对 `kddcup2026-data-agents-starter-kit` 项目中 Data Agent 的所有截断（truncation）机制进行全面审计。截断设计遍布整个 agent 的数据流——从底层 token 级裁剪、SQL 行数限制、工具输出格式化、答案校验上下文裁剪，到消息历史压缩和日志预览。报告共识别出 **20+ 处截断机制**，分布在 12 个源文件中，并对其潜在 bug、不合理之处和优化方向给出建议。

---

### 二、截断架构总览

数据从源头到最终模型调用的截断链路如下：

```
[数据源] ──SQL LIMIT/行数限制──▶ [工具执行层] ──format_result 统一截断──▶
[模型上下文] ──reasoning/image 压缩──▶ [LLM] ──答案提交──▶
[答案校验器] ──truncate_answer_content 二次截断──▶ [Validator LLM]
```

截断可分为以下六大类：

| 类别 | 涉及文件 | 数量 |
|------|----------|------|
| Token 级文本裁剪 | `token_utils.py`, `truncation.py` | 3 处 |
| SQL/数据查询行数限制 | `probe_engine.py`, `sqlite.py`, `registry.py` | 6 处 |
| 工具输出格式化截断 | `registry.py` (`format_result`) | 1 处 |
| 答案校验上下文截断 | `langgraph_runtime.py`, `answer_validator.py` | 3 处 |
| 消息历史/上下文窗口压缩 | `langgraph_runtime.py`, `multimodal.py` | 4 处 |
| 日志/预览截断 | `langgraph_runtime.py` | 3 处 |

---

### 三、各截断机制详细分析

#### 3.1 Token 级基础裁剪

**文件**: `token_utils.py` (全文 59 行)

这是整个截断体系的底层基座，使用 Qwen3.5-35B-A3B tokenizer 进行 token 计数和裁剪。

- **`count_tokens(text)`** — 编码文本并返回 token 数。用于所有需要判断是否需要截断的场景。
- **`truncate_by_tokens(text, max_tokens)`** — 编码 → 切片 `encoded[:max_tokens]` → 解码回文本。

**文件**: `truncation.py` (全文 73 行)

在 `token_utils` 之上构建的结构化截断工具集：

- **`TRUNCATION_SUFFIX`** — 截断标记字符串：`"\n\n...（内容已被截断，返回过长。如未获取到所需信息，请尝试其他方式而非直接读取全部文档。）"`
- **`truncate_str(text, max_tokens=10000)`** — 对单个字符串按 token 数截断，超过则调用 `truncate_by_tokens` 并拼接 `TRUNCATION_SUFFIX`。
- **`truncate_content(obj, max_str_tokens=2000, max_list_items=200)`** — 递归遍历任意嵌套的 JSON 结构（str/list/dict），对字符串按 `max_str_tokens` 截断，对列表按 `max_list_items` 切片，超出则在列表末尾追加 `TRUNCATION_SUFFIX`。
- **`truncate_answer_content(answer, max_str_tokens=2000, max_list_items=200)`** — 专为答案表格设计的截断：保留 `columns` 键不做裁剪，仅对 `rows` 和其他键调用 `truncate_content`。

**配置来源**: `config.py` 中的 `ToolConfig`

```python
class ToolConfig:
    max_output_tokens: int = 10000   # 工具输出每字符串最大 token 数
    max_list_items: int = 200        # 工具输出列表最大元素数
```

**实际调用时**，`format_result` 使用 `ToolConfig.max_output_tokens`（默认 10000）覆盖 `truncate_content` 的默认值（2000），因此实际运行时字符串截断阈值为 **10000 tokens**。

---

#### 3.2 SQL/数据查询行数限制

| 位置 | 机制 | 默认值 | 硬上限 |
|------|------|--------|--------|
| `probe_engine.py:174` `execute_probe_query()` | `fetchmany(limit+1)` + `rows[:limit]` | 200 行 | 无（由调用方决定） |
| `registry.py:630` `_execute_probe_query()` | `min(limit, 200)` 强制钳位 | 5 行（agent 传入） | **200 行** |
| `sqlite.py:13` `execute_read_only_sql()` | `fetchmany(limit+1)` + `rows[:limit]` | 200 行 | 无 |
| `registry.py:597` `_execute_context_sql()` | 直接透传 limit 参数 | 200 行 | **无硬上限** |
| `registry.py:653` `_get_column_distinct_values()` | `min(top_n, 200)` 强制钳位 | 20 值 | **200 值** |
| `registry.py:303` `_search_semantic_catalog()` | `max(1, min(limit, 100))` 强制钳位 | 20 条 | **100 条** |

**提交路径特殊处理**：当通过 `submit_tool_result` 提交最终答案时，`_execute_submit_source_tool()` 会对 `execute_probe_query` 和 `execute_context_sql` 传入 `limit=None`，即**不限制行数**，确保答案的完整性。这是一个合理的设计——探查时使用 limit 预览，提交时放开限制获取全量数据。

**`truncated` 标志位**：`probe_engine.py` 和 `sqlite.py` 均使用 `fetchmany(limit+1)` 的技巧——多取一行用于判断是否存在更多数据，但只返回前 `limit` 行，并在结果中设置 `truncated: true/false` 标志。

---

#### 3.3 工具输出格式化截断（统一截断层）

**文件**: `registry.py:931-948` — `ToolRegistry.format_result()`

```python
def format_result(self, action, result):
    payload = {"ok": result.ok, "content": result.content}
    if result.answer is not None:
        payload["answer"] = truncate_answer_content(
            result.answer.to_dict(),
            max_str_tokens=self.tool_config.max_output_tokens,
            max_list_items=self.tool_config.max_list_items,
        )
    if action != "submit_tool_result":
        payload["content"] = truncate_content(
            payload["content"],
            max_str_tokens=self.tool_config.max_output_tokens,
            max_list_items=self.tool_config.max_list_items,
        )
    return payload
```

这是**所有工具结果到达模型之前的最后一道截断关卡**。关键设计：

- `submit_tool_result` 的 `content`（提交状态摘要）不做截断，因为其中不含大量数据。
- `answer` 字段总是被截断——这是模型在提交后看到的"已提交答案预览"。
- 非提交操作的 `content` 总是被截断。

---

#### 3.4 答案校验上下文截断

**文件**: `langgraph_runtime.py:194-205` + `1854-1859`

当 agent 提交答案后，答案校验器（answer_validator）接收的不是完整答案，而是一个截断副本：

```python
answer_dict_for_validator = _submitted_answer_for_validator_context(
    answer_dict_full,
    max_str_tokens=self.tools.tool_config.max_output_tokens,  # 10000
    max_list_items=self.tools.tool_config.max_list_items,      # 200
)
answer_truncated_for_validator = answer_dict_for_validator != answer_dict_full
```

**答案校验器的截断感知处理**（`answer_validator.py:293-300`）：

1. 深拷贝答案，避免修改图状态。
2. 如果 `rows` 列表的最后一个元素是 `TRUNCATION_SUFFIX` 字符串，将其弹出。
3. 如果 `answer_truncated=True`，在验证请求中加入免责声明：*"The JSON below is a truncated validator-context preview. The stored submitted answer remains complete..."*

---

#### 3.5 消息历史/上下文窗口压缩

**文件**: `langgraph_runtime.py`

| 机制 | 位置 | 说明 |
|------|------|------|
| Reasoning 历史剥离 | 586-626 行 | 从旧 AI 消息中移除 reasoning 内容，`reasoning_history_limit` 控制保留最近 N 条 |
| 图像消息压缩 | 547-583 行 | 将已处理的图像数据替换为文本摘要，笔记截断至 `compressed_image_note_chars`（默认 600 字符） |
| 视频帧数限制 | `multimodal.py:63-70` | `stable_frames[:max_attached_frames]`，默认 16 帧 |
| 进程校验步骤限制 | 1615-1616 行 | `recent_steps[-recent_step_limit:]`，仅取最近 8 步发给进程校验器 |

---

#### 3.6 日志/预览截断

| 机制 | 位置 | 限制 |
|------|------|------|
| `_preview_text()` | `langgraph_runtime.py:262-265` | 默认 180 字符 |
| Profile 预览 | `langgraph_runtime.py:969` | 500 字符 |
| 歧义分析预览 | `langgraph_runtime.py:1039` | `json.dumps(...)[:500]`，直接字符切片 |

这些截断仅影响 trace/log 输出，不影响模型可见数据，风险较低。

---

#### 3.7 数据探查器（Inspector）截断

**文件**: `semantic_catalog.py`

| 机制 | 默认值 | 说明 |
|------|--------|------|
| `MAX_RELATION_DISTINCT_VALUES = 200_000` | 硬编码 | 单个字段 distinct value 分析的上限 |
| `_flatten_json_fields()` 数组采样 | `value[:20]` | JSON schema 推断时仅取前 20 个数组元素 |
| 文档预览 token 限制 | `budget.max_doc_tokens = 500` | 文档内容截断至 500 tokens |
| 搜索结果硬编码上限 | `[:10]` assets, `[:20]` fields | 搜索结果数量限制，不可配置 |

---

#### 3.8 歧义分析器截断

**文件**: `ambiguity_analyzer.py`

| 机制 | 默认值 | 说明 |
|------|--------|------|
| `_build_schemas_listing()` 字段数上限 | `max_fields = 200` | 硬编码，超出后追加 `"... (N more fields omitted)"` |
| `_build_knowledge_listing()` token 预算 | `max_tokens = 2048` | 所有知识文档共享 2048 token 预算 |

---

#### 3.9 其他截断

| 机制 | 位置 | 默认值 |
|------|------|--------|
| Python 执行超时 | `registry.py:53` | 30 秒 |
| Agent 最大步数 | `config.py:38` | 16 步 |
| 文件系统目录递归深度 | `filesystem.py:21-55` | `max_depth=4` |
| 文档搜索分页 | `langgraph_tools.py:175-178` | `page_size=20` |

---

### 四、潜在 Bug 与不合理之处

#### BUG-1（中风险）：`_execute_context_sql` 缺少硬上限

**位置**: `registry.py:597`

```python
limit = int(action_input.get("limit", 200))  # 直接透传，无上限钳位
```

与 `_execute_probe_query` 的 `min(limit, 200)` 和 `_get_column_distinct_values` 的 `min(top_n, 200)` 不同，`_execute_context_sql` 完全信任模型传入的 limit 值。模型可以请求 `limit=1000000`，导致一次查询返回百万行数据。虽然 `format_result` 的 `truncate_content` 会随后截断列表至 200 项，但这意味着：

- **大量内存被无效占用**：先加载百万行到内存，再丢弃绝大部分。
- **执行时间浪费**：数据库执行全量查询远慢于带 LIMIT 的查询。
- **与 probe query 的行为不一致**：两种 SQL 工具的安全边界不同。

---

#### BUG-2（中风险）：答案校验器仅清理顶层 rows 的 `TRUNCATION_SUFFIX`

**位置**: `answer_validator.py:298-300`

```python
if isinstance(answer, dict) and "rows" in answer and isinstance(answer["rows"], list):
    if answer["rows"] and answer["rows"][-1] == TRUNCATION_SUFFIX:
        answer["rows"].pop()
```

`truncate_content` 是递归的：如果 `rows` 中的某个单元格值（如长文本字段）也被截断，该字符串末尾会拼接 `TRUNCATION_SUFFIX`。但校验器只清理了 `rows` 列表末尾的截断标记，**单元格级别的截断标记仍然保留**。

这可能导致：
- 校验器看到被截断的单元格值（如 `"2024-03-01T12:00:00Z\n\n...（内容已被截断..."`），误判日期格式不合规。
- 校验器看到被截断的名称字段，误判内容不完整。

---

#### BUG-3（低风险）：`truncate_by_tokens` 可能在多字节字符边界处截断产生乱码

**位置**: `token_utils.py:56-59`

```python
encoded = tokenizer.encode(text, add_special_tokens=False)
if len(encoded) <= max_tokens:
    return text
return tokenizer.decode(encoded[:max_tokens], skip_special_tokens=True)
```

Tokenizer 的 encode/decode 不一定保证在字符边界处截断。对于 CJK 字符或 emoji，一个字符可能对应多个 token。切片 `encoded[:max_tokens]` 可能在某个字符的 token 序列中间截断，decode 后可能产生不完整的字符或末尾乱码。紧接着拼接的 `TRUNCATION_SUFFIX` 虽然提供了截断提示，但前面的乱码可能干扰模型理解。

---

#### BUG-4（低风险）：`ToolConfig` 缺少参数校验

**位置**: `config.py:122-124`

```python
class ToolConfig:
    max_output_tokens: int = 10000
    max_list_items: int = 200
```

与 `AgentConfig` 和 `ProcessValidatorConfig` 不同，`ToolConfig` 没有 `__post_init__` 校验方法。如果通过 YAML 配置将 `max_output_tokens` 设为 0 或负数：
- `truncate_str` 会返回空字符串（`max_tokens <= 0` 的分支）。
- `truncate_content` 的 `max_list_items=0` 会将所有列表截断为空。

这不会导致崩溃，但会导致所有工具输出被完全清空，agent 将无法获得任何有效信息。

---

#### BUG-5（中风险）：`truncate_content` 默认值与 `format_result` 实际使用值不一致

**位置**: `truncation.py:21-25` vs `registry.py:943-946`

`truncate_content` 的函数签名默认 `max_str_tokens=2000`，但 `format_result` 总是传入 `ToolConfig.max_output_tokens=10000`。这意味着：
- 直接调用 `truncate_content(text)` 时截断阈值为 2000 tokens。
- 通过 `format_result` 调用时截断阈值为 10000 tokens。

如果未来有新的调用路径忘记传入 `ToolConfig` 值，会意外使用 2000 的较低阈值，导致过度截断。

---

#### BUG-6（中风险）：搜索结果硬编码上限不可配置

**位置**: `semantic_catalog.py:969-970`

```python
"relevant_assets": sorted(...)[:10],
"relevant_fields": sorted(...)[:20],
```

以及 `ambiguity_analyzer.py:521` 的 `max_fields = 200`。

这些硬编码常量不可通过 YAML 配置调整，且截断后**没有任何提示标记**（不像 `truncate_content` 会追加 `TRUNCATION_SUFFIX`）。对于大型数据集，agent 可能因搜索结果不足而遗漏关键数据资产，且无法感知结果被截断。

---

#### BUG-7（低风险）：`_preview_text` 与 `analysis_preview` 截断方式不一致

**位置**: `langgraph_runtime.py:262-265` vs `1039`

```python
# 方式1：使用 _preview_text 函数
_preview_text(text, limit=180)

# 方式2：直接字符串切片
json.dumps(result, ensure_ascii=False)[:500]
```

方式2直接在 JSON 字符串上切片，可能在 JSON 结构的中间截断（如 `{"key": "val`），导致 trace 中出现不完整的 JSON。虽然这只影响日志可读性，不影响功能，但会给调试带来困扰。

---

#### BUG-8（低风险）：`_submitted_answer_for_validator_context` 的截断检测可能不可靠

**位置**: `langgraph_runtime.py:1859`

```python
answer_truncated_for_validator = answer_dict_for_validator != answer_dict_full
```

这里使用 Python 的深度 `!=` 比较来检测截断是否发生。由于 `truncate_answer_content` 总是创建新的 dict 对象，在某些边界情况下（如 `columns` 不是 list 类型时，`truncated["columns"] = columns` 直接引用原始对象），比较结果可能不完全准确。虽然实际场景中答案格式通常是标准的，但这种依赖对象结构比较的方式较为脆弱。

---

#### 设计问题-1：双重截断导致 token 预算计算模糊

工具输出在 `format_result` 中被截断至 10000 tokens。模型看到这份截断后的数据，构造提交请求时可能复述其中的内容。提交后，`_submitted_answer_for_validator_context` 再次对答案截断至 10000 tokens。

这两次截断使用相同的 token 预算（`ToolConfig.max_output_tokens`），但由于第一次截断可能改变数据结构（添加 `TRUNCATION_SUFFIX`、截断列表），第二次截断的输入已经不同于原始数据。这使得"校验器实际看到了多少数据"变得难以精确推理。

---

#### 设计问题-2：`TRUNCATION_SUFFIX` 是中文文本，国际化受限

`TRUNCATION_SUFFIX` 使用中文提示：*"内容已被截断，返回过长。如未获取到所需信息，请尝试其他方式而非直接读取全部文档。"*

如果未来需要支持非中文的 LLM 或多语言场景，这个硬编码的中文后缀可能不被模型正确理解。

---

### 五、优化建议

#### 5.1 高优先级

1. **为 `_execute_context_sql` 添加硬上限**：参照 `_execute_probe_query`，加入 `limit = min(int(action_input.get("limit", 200)), 10000)` 或类似的安全上限，防止模型请求超大结果集。

2. **扩展答案校验器的 `TRUNCATION_SUFFIX` 清理逻辑**：不仅清理 `rows` 列表末尾的标记，还应递归遍历 rows 中每个单元格的字符串值，移除或替换其中的 `TRUNCATION_SUFFIX`。或者在截断前对答案做一份"校验器专用"的清洗副本。

3. **将硬编码常量提取为可配置参数**：将 `semantic_catalog.py` 中的 `[:10]` / `[:20]` 和 `ambiguity_analyzer.py` 中的 `max_fields=200` / `max_tokens=2048` 提取到 `config.py` 中的配置类，允许通过 YAML 调整。

#### 5.2 中优先级

4. **为 `ToolConfig` 添加 `__post_init__` 校验**：确保 `max_output_tokens > 0` 和 `max_list_items > 0`，与 `AgentConfig` 等其他配置类保持一致的校验风格。

5. **统一 `truncate_content` 的默认值**：将函数签名的默认值改为与 `ToolConfig` 一致的 10000，或者去掉默认值、强制调用方显式传入，消除"默认 2000 vs 实际 10000"的陷阱。

6. **为搜索结果截断添加提示标记**：在 `semantic_catalog.py` 的搜索结果被 `[:10]` / `[:20]` 截断时，在返回结果中添加 `"truncated": true` 或 `"total_count": N` 字段，让 agent 感知到结果不完整。

#### 5.3 低优先级

7. **改用字符级安全截断**：在 `truncate_by_tokens` 后，对解码结果做一次 `rstrip` 或末尾完整性检查，避免残留不完整的 CJK 字符。

8. **统一日志预览截断方式**：将 `analysis_preview` 的 `json.dumps(...)[:500]` 改为使用 `_preview_text(json.dumps(...), limit=500)`，保持 trace 输出格式一致。

9. **将 `TRUNCATION_SUFFIX` 改为可配置或双语**：在 `config.py` 中定义截断后缀文本，允许根据部署环境切换语言。

10. **考虑引入全局 token 预算管理**：当前各处截断独立运作，缺少对整体上下文窗口 token 使用量的统筹。可以引入一个 `ContextBudget` 管理器，统一分配和追踪系统提示、历史消息、工具输出等各部分的 token 预算。

---

### 六、截断参数速查表

| 参数名 | 默认值 | 所在配置类 | 作用范围 | 可配置 |
|--------|--------|-----------|----------|--------|
| `max_output_tokens` | 10000 | `ToolConfig` | 工具输出字符串 token 上限 | YAML `tool.max_output_tokens` |
| `max_list_items` | 200 | `ToolConfig` | 工具输出列表最大元素数 | YAML `tool.max_list_items` |
| `max_steps` | 16 | `AgentConfig` | Agent 最大推理步数 | YAML `agent.max_steps` |
| `max_tokens` | 8192 | `AgentConfig` | 模型输出 token 上限 | YAML `agent.max_tokens` |
| `validation_retry_limit` | 2 | `AgentConfig` | 答案校验最大重试次数 | YAML `agent.validation_retry_limit` |
| `reasoning_history_limit` | None | `AgentConfig` | 保留 reasoning 的最近 AI 消息数 | YAML `agent.reasoning_history_limit` |
| `compressed_image_note_chars` | 600 | `AgentConfig` | 图像压缩后笔记字符上限 | YAML `agent.compressed_image_note_chars` |
| `max_attached_frames` | 16 | `VideoPreprocessingConfig` | 视频最大附加帧数 | YAML `video_preprocessing.max_attached_frames` |
| `catalog_top_distinct_values` | 50 | `DataInspectorSampleBudget` | catalog 中 distinct value 取样数 | YAML `data_inspector.sample_budget.catalog_top_distinct_values` |
| `max_doc_tokens` | 500 | `DataInspectorSampleBudget` | 文档预览 token 上限 | YAML `data_inspector.sample_budget.max_doc_tokens` |
| `recent_step_limit` | 8 | `ProcessValidatorConfig` | 进程校验器最近步骤数 | YAML `process_validator.recent_step_limit` |
| `EXECUTE_PYTHON_TIMEOUT_SECONDS` | 30 | 模块常量 | Python 执行超时 | 不可配置 |
| `MAX_RELATION_DISTINCT_VALUES` | 200000 | 模块常量 | 单字段 distinct value 分析上限 | 不可配置 |
| 搜索结果 `[:10]` / `[:20]` | 10 / 20 | 硬编码 | 搜索结果最大资产数/字段数 | 不可配置 |
| `max_fields` | 200 | 硬编码 | 歧义分析 schema 字段上限 | 不可配置 |
| `max_tokens` (知识文档) | 2048 | 硬编码 | 知识文档共享 token 预算 | 不可配置 |
| `_preview_text` limit | 180 / 500 | 函数默认值 | 日志预览字符上限 | 不可配置 |
| Probe query 硬上限 | 200 | 代码 `min()` | 探查查询最大行数 | 不可配置 |
| Distinct values 硬上限 | 200 | 代码 `min()` | 去重值最大数量 | 不可配置 |
| Catalog search 硬上限 | 100 | 代码 `min()` | 语义搜索结果上限 | 不可配置 |
