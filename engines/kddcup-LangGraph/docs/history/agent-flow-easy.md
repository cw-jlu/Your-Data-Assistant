# Agent 流程图 — easy.yaml 配置

## 配置概览

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `prompt_version` | 1 | V1 提示词 |
| `enable_data_inspector` | true | 开启全局数据探查 |
| `enable_ambiguity_analysis` | true | 开启歧义分析 |
| `enable_answer_validator` | true | 开启答案校验 |
| `max_steps` | 200 | Agent 主循环最大步数 |
| `empty_stop_retry_limit` | 2 | 空 stop 修复重试上限 |
| `validation_retry_limit` | 2 | 答案校验重试上限 |
| `temperature` | 0.0 | 模型温度 |

---

## 一、整体流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                        LangGraph Agent                           │
│                                                                  │
│  ┌─────────────┐                                                │
│  │  init_state │  初始化系统提示词、状态字段                         │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────┐                                   │
│  │ global_data_exploration  │  全局数据探查                        │
│  │  (enable_data_inspector) │  build_semantic_catalog()           │
│  └──────────┬───────────────┘  扫描所有 CSV/JSON/SQLite/Doc       │
│             │                  生成字段级统计 (基数, min/max,       │
│             ▼                  distinct values, 关联推断)         │
│  ┌──────────────────────┐                                       │
│  │  analyze_ambiguity   │  歧义分析                               │
│  │  (enable_ambiguity_  │  LLM 识别语义风险 → ambiguities[]       │
│  │   analysis)          │  + non_ambiguous_candidates[]          │
│  └──────────┬───────────┘  ← invoke_model_with_retries           │
│             │              失败 → 空后备结果 (不阻塞)               │
│             ▼                                                    │
│  ┌──────────────────┐                                           │
│  │  receive_problem │  组装任务提示词                              │
│  │                  │  用户问题 + 完整 Catalog + 歧义分析结果       │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ╔══════════════════════════════════════════════════════════╗    │
│  ║              主 Agent 循环 (max_steps=200)                ║    │
│  ║                                                          ║    │
│  ║          ┌──────────────┐                                ║    │
│  ║  ┌──────►│  model_step  │◄───────────────────────┐       ║    │
│  ║  │       │  LLM 推理    │                        │       ║    │
│  ║  │       │  ← retry×3   │                        │       ║    │
│  ║  │       └──────┬───────┘                        │       ║    │
│  ║  │              │                                │       ║    │
│  ║  │    ┌─────────┼─────────┬──────────┬───────────┘       ║    │
│  ║  │    │         │         │          │                    ║    │
│  ║  │    ▼         ▼         ▼          ▼                    ║    │
│  ║  │ ┌──────┐ ┌───────┐ ┌────────┐ ┌──────────┐           ║    │
│  ║  │ │ tool │ │repair │ │validate│ │ finalize │           ║    │
│  ║  │ │_step │ │_step  │ │_answer │ │          │           ║    │
│  ║  │ │      │ │(≤2次) │ │ (≤2次) │ │          │           ║    │
│  ║  │ └──┬───┘ └───┬───┘ └───┬────┘ └──────────┘           ║    │
│  ║  │    │         │         │                               ║    │
│  ║  │    │         └──►model─┘                               ║    │
│  ║  │    │              (无条件)                               ║    │
│  ║  │    │                                                   ║    │
│  ║  │    ├── model_step (无answer 且 未达max_steps)           ║    │
│  ║  │    ├── validate_answer (有answer)                       ║    │
│  ║  │    └── finalize (failure_reason 或 达max_steps)         ║    │
│  ║  │                                                        ║    │
│  ║  │    validate_answer ─┬─ 通过 → finalize                  ║    │
│  ║  │                     └─ 拒绝 → model_step (反馈消息)      ║    │
│  ║  └──────────────────────────────────────────────────────┘   ║
│  ╚══════════════════════════════════════════════════════════════╝
│                                                                  │
│           ▼                                                      │
│  ┌──────────────┐                                               │
│  │   finalize   │  设置 failure_reason (如适用)                    │
│  └──────────────┘  产出 trace.json / prediction.csv              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、节点详解

### 2.1 init_state
- 构建系统提示词 (V1: `build_system_prompt`)
- 初始化状态字段: `messages=[SystemMessage]`, `step_count=0`, `empty_stop_retry_count=0`, `validation_retry_count=0`

### 2.2 global_data_exploration
- 调用 `DataUnderstandingAgent.explore_data_globally()`
- 扫描 task 目录下所有数据资产:
  - CSV: 字段名/类型/基数/TOP-N distinct values/min-max
  - SQLite: 表结构/行数/字段统计
  - JSON: 字段路径/类型/统计
  - Doc: 文档 token 数/预览
- 产出 `global_data_profile` (完整 catalog, 含 cardinality, distinct values, min/max, join relationships)
- 失败不阻塞，记录错误日志

### 2.3 analyze_ambiguity
- 将问题和 schema 送 LLM 进行语义歧义分析
- 产出: `question_intent{entities, filters, metrics, requested_output, grain}`, `ambiguities[{id, phrase, type, clarifying_question, required_verification}]` (最多2个), `resolved_by_knowledge[]`, `non_ambiguous_candidates[{phrase, candidate_fields, note}]`
- 8 种歧义类型: `field_binding`, `metric_definition`, `entity_resolution`, `filter_semantics`, `time_range`, `grain`, `join_path`, `output_format`
- 内部使用 `invoke_model_with_retries`
- 失败 → 返回空后备结果 (不阻塞)
- 产出 `ambiguity_analysis` 填充到 state

### 2.4 receive_problem
- 组装 `<user_query>` 消息
- 追加完整 Catalog + 歧义分析结果 (`<ambiguity_analysis>`) 作为 preamble
- 注入 HumanMessage 到消息列表

---

## 三、主循环路由规则

### route_after_model (model_step 之后)

```
有 failure_reason? ──yes──► finalize
有 answer? ──yes──► validate_answer
AIMessage + tool_calls? ──yes──► tool_step
AIMessage + finish_reason="stop" + 无tool_calls
  + empty_stop_retry_count < 2? ──yes──► repair_step
其他 ──► finalize
```

### route_after_tool (tool_step 之后)

```
有 failure_reason? ──yes──► finalize
有 answer? ──yes──► validate_answer
step_count >= 200? ──yes──► finalize
其他 ──► model_step
```

### route_after_validation (validate_answer 之后)

```
无answer 且 无failure_reason? ──yes──► model_step (重试)
其他 ──► finalize
```

---

## 四、重试机制 (easy.yaml 配置下)

```
┌─────────────────────────────────────────────────────────┐
│ 层1: LLM API 重试 (model_retry.py)                       │
│   触发: 429/500/502/503/504                              │
│   延迟: 5s → 15s → 30s (递进)                            │
│   应用: model_step / validator / ambiguity_analyzer       │
├─────────────────────────────────────────────────────────┤
│ 层2: Empty-Stop 修复 (repair_step)                       │
│   触发: LLM 返回 stop 但无 tool_call                     │
│   上限: 2次                                              │
│   动作: 注入 "请立即调用工具" 提示                         │
│   重置: 任何工具执行成功后归零                            │
├─────────────────────────────────────────────────────────┤
│ 层3: 答案校验重试 (validate_answer_step)                  │
│   上限: 2次                                              │
│   动作: 校验失败 → 注入反馈 → 回到 model_step             │
│   失败: 校验器崩溃 → 直接接受答案 (fail-open)             │
│   耗尽: 达到上限 → 接受当前答案                           │
└─────────────────────────────────────────────────────────┘
```

---

## 五、可用工具

| 工具 | 说明 |
|------|------|
| `list_context` | 列出 task 目录下的文件树 |
| `read_doc` | 读取文档文件 (支持多种格式) |
| `lookup_doc_outline` | 获取文档目录结构 |
| `search_doc` | 在文档中搜索关键词/正则 |
| `execute_context_sql` | 在 SQLite/CSV/JSON 上执行只读 SQL 查询 |
| `execute_probe_query` | 通过 SQL 对 CSV/JSON/SQLite 进行快速批量探查 |
| `get_column_distinct_values` | 快速获取列的去重值及频次 |
| `execute_python` | 在隔离沙箱中执行 Python 代码 (30s 超时) |
| `answer` | 提交最终答案表 |

---

## 六、数据流

```
task.context_dir (CSV/JSON/SQLite/Doc)
         │
         ▼
global_data_exploration ──► semantic_catalog (结构化)
         │
         ▼
analyze_ambiguity ──► ambiguity_analysis (歧义清单 + 候选字段)
         │
         ▼
receive_problem ──► HumanMessage (问题 + 完整 catalog + 歧义分析)
         │
         ▼
   ┌─ model_step ◄─── tool_step ──┐
   │     │                          │
   │     ├── list_context              │
   │     ├── execute_probe_query       │
   │     ├── read_doc                  │
   │     ├── execute_context_sql       │
   │     ├── execute_python            │
   │     └── answer ────────────────┘
   │                    │
   └── repair_step ─────┘ (空 stop)
         │
         ▼
   validate_answer ──► 通过 → finalize
         │              拒绝 → model_step
         ▼
   finalize ──► prediction.csv + trace.json
```
