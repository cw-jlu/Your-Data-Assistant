# 配置文件说明

配置文件位于 `configs/` 目录，使用 YAML 格式。运行命令时通过 `-c` 指定：

```bash
uv run python -m data_agent_baseline run-task -c configs/easy.yaml
```

---

## 一、`dataset` — 数据集路径

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `root_path` | 路径 | `data/input` | 公开数据集的根目录，任务数据从此目录下按 `task_id` 查找 |

---

## 二、`agent` — 主 Agent 配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | 字符串 | `gpt-4.1-mini` | 模型名称。如果同时设置了 `model_env`，则从环境变量读取，忽略此值 |
| `model_env` | 字符串 | 无 | 从环境变量读取模型名的变量名（仅 docker.yaml 使用） |
| `api_base` | URL | `https://api.openai.com/v1` | API 地址 |
| `api_base_env` | 字符串 | 无 | 从环境变量读取 API 地址的变量名（仅 docker.yaml 使用） |
| `api_key` | 字符串 | 空 | API Key。留空则从 `api_key_env` 指定的环境变量读取 |
| `api_key_env` | 字符串 | 无 | 环境变量名。程序会先读环境变量，读不到再读 `.env` 文件 |
| `max_steps` | 整数 | 16 | 主 Agent 最大 ReAct 轮数（model_step -> tool_step 循环次数） |
| `temperature` | 浮点 | 0.0 | 模型温度，0 表示确定性输出 |
| `enable_data_inspector` | 布尔 | `false` | **总开关**。必须设为 `true`，下面的 `data_inspector` 配置才会生效 |
| `enable_answer_validator` | 布尔 | `true` | 是否在提交答案前进行自我验证 |
| `validation_retry_limit` | 整数 | 2 | 答案校验最多打回主 Agent 的次数；达到后接受当前答案 |
| `prompt_version` | 整数 | 1 | 系统提示词版本（1=v1 英文, 2=v2） |
| `model_request_timeout_seconds` | 浮点 | 120 | 单次模型请求超时秒数 |

---

## 三、`data_inspector` — 数据探查配置

### 3.1 `enable_data_inspector: false` 时的流程

`global_data_exploration` 节点直接返回 `{}`（空操作），Agent 的 `messages` 里只有 system prompt + 任务问题，靠自己用工具探索数据。

流程：`START -> init_state -> global_data_exploration(no-op) -> receive_problem -> model_step -> [ReAct循环] -> finalize`

- `global_data_profile` 保持 `None`（不构建 catalog，不调 LLM）
- `inspector` 保持 `None`
- `receive_problem` 不注入 catalog 消息

### 3.2 数据采样参数 `sample_budget`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `catalog_top_distinct_values` | 整数 | 50 | 每个字段展示的去重值数量，取频率最高的前 N 个。`cardinality` 字段会报告真实的去重总数 |
| `max_doc_chars` | 整数 | 2000 | 文档类文件（knowledge.md）在 catalog 中预览的最大字符数 |

---

## 四、`tool` — 工具输出截断配置

控制 Agent 调用工具后，返回内容的截断阈值。除最终提交工具 `submit_tool_result` 外，其他工具输出会统一经过此截断处理。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_output_chars` | 整数 | 8000 | 工具输出中字符串字段的最大字符数，超过部分会被截断 |
| `max_list_items` | 整数 | 200 | 工具输出中列表字段的最大条目数，超过部分会被截断 |

> **注意**：`execute_python` 的输出（stdout/stderr）不再单独使用 4000 字符的硬编码截断，而是与其他工具统一使用此处的配置。

---

## 五、`run` — 运行配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `output_dir` | 路径 | `artifacts/runs` | 运行产物输出目录 |
| `log_dir` | 路径 | 无 | 日志目录（仅 docker 使用） |
| `output_layout` | 字符串 | `run_dir` | 产物目录结构：`run_dir`（每 run 一个子目录）或 `flat`（平铺） |
| `run_id` | 字符串 | 空 | 运行 ID，留空自动生成 |
| `max_workers` | 整数 | 4 | 并行执行任务的最大线程数 |
| `extract_structured_doc_max_workers` | 整数 | 2 | 全局同时执行 `extract_structured_doc` 的最大任务数 |
| `task_timeout_seconds` | 整数 | 600 | 单个任务超时秒数 |
| `task_ids` | 列表 | 无 | 要执行的任务 ID 列表。不填则执行全部 |

---

## 六、典型使用场景

### 场景 1：开启 Data Inspector（推荐）
```yaml
agent:
  enable_data_inspector: true
```

### 场景 2：完全不用 data inspector（裸 Agent）
```yaml
agent:
  enable_data_inspector: false
```
