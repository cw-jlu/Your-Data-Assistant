# 配置与运行手册

本文依据当前 `config.py`、CLI、Runner、评分器和 Dockerfile 编写。除非特别说明，“当前配置”指提交镜像实际复制并使用的 `configs/docker.yaml`，不是 Python 默认值，也不是其他本地 preset。

## 1. 能力支持与当前启用状态

| 能力 | 代码支持 | `configs/docker.yaml` 当前状态 |
| --- | --- | --- |
| 全局数据理解 / Semantic Catalog | 支持 | **启用**：`enable_data_inspector: true` |
| Semantic View | 支持 | **启用**：`semantic_views.enabled: true` |
| one-to-one 维表补充 | 支持，可配置 | **允许** |
| 答案校验 | 支持 | **启用**，最多拒绝 2 次 |
| 过程校验 | 支持 | **启用**，每 12 个模型步检查一次 |
| 问题歧义预分析 | 支持 | **未启用** |
| 视频预处理和视频理解摘要 | 支持 | **启用** |
| 已使用图片消息压缩 | 支持 | **启用** |
| 推理历史剥离 | 支持 | **未启用** |
| 结构化文档抽取 | 支持 | **可用**，全局最多 2 个并发抽取 |
| 单任务硬超时 | 支持 | **启用**，基础 1200 秒；使用文档抽取后加 1800 秒 |
| 批量并发 | 支持 | **启用**，8 个活跃任务 |
| 指定任务列表 | 支持 | **未指定**，因此遍历 `/input` 下全部任务 |
| flat 输出 / 独立日志目录 | 支持 | **启用**：预测根 `/output`，运行汇总根 `/logs` |
| 跳过已有预测 | CLI 支持 | **Docker 入口启用**：`--skip-completed` |
| 重复运行 | CLI 支持 | Docker 入口不使用；本地按需执行 |
| 本地公开集评分和运行对比 | CLI 支持 | Docker 入口不依赖；hidden 数据无 `gold.csv` 时不可评分 |

配置中出现一个字段不等于功能已启用。功能开关、阈值的有效前提及 CLI 是否实际传参都必须同时检查。

## 2. 安装与基本检查

项目要求 Python 3.10 及以上，命令入口由 `pyproject.toml` 注册为 `dabench`。

```powershell
uv sync --frozen
# 需要运行测试时再安装 dev 可选依赖：
uv sync --frozen --extra dev
uv run dabench status --config configs/full.yaml
```

数据集目录应满足：

```text
data/input/
└── task_<id>/
    ├── task.json            # 至少包含 task_id、question
    └── context/             # 任务数据资产

data/output/
└── task_<id>/
    └── gold.csv             # 仅公开 demo 本地评分需要
```

所有 YAML 相对路径均相对于项目根目录解析，而不是相对于 YAML 文件所在目录或当前终端目录。

## 3. 环境变量与模型接口

### 3.1 模型接口要求

运行时使用 `langchain_openai.ChatOpenAI` 的派生类，因此模型服务必须提供 OpenAI-compatible Chat Completions 接口，并支持项目使用的 tool calling。涉及图片或视频稳定帧时，模型还需要兼容多模态 `image_url` 输入。

固定请求行为：

- `base_url` 使用 `agent.api_base` 并移除末尾 `/`。
- `model`、`temperature`、`max_tokens` 来自 YAML 或环境变量。
- `top_p` 固定为 `0.8`。
- `extra_body.repetition_penalty` 固定为 `1.1`。
- ChatOpenAI 客户端 `max_retries=3`。
- 项目外层还会对 429、500、502、503、504 和网络错误重试；前三次等待 5/15/30 秒，之后每次等待 10 秒。当前外层重试没有总次数上限。
- 单次调用达到 `model_request_timeout_seconds` 时抛出超时；该超时不进入网络错误重试。
- Provider 返回的 `reasoning_content` / `reasoning` 会保留到 trace。

若服务不接受 `repetition_penalty`、tool schema、`parallel_tool_calls` 或多模态字段，应在服务兼容层处理；当前 YAML 没有关闭这些请求字段的开关。

### 3.2 YAML 与环境变量解析规则

模型名称和 API 地址支持两种写法：

```yaml
agent:
  model: qwen3.5-35b-a3b
  api_base: http://127.0.0.1:8000/v1
```

或：

```yaml
agent:
  model_env: MODEL_NAME
  api_base_env: MODEL_API_URL
```

解析规则：

- 配置了 `model_env` 或 `api_base_env` 后，对应进程环境变量必须存在；不会从 `.env` 回退，缺失时配置加载立即失败。
- `api_key` 非空时优先使用 YAML 明文值。
- `api_key` 为空且配置了 `api_key_env` 时，先读取进程环境变量，再读取项目根目录 `.env`。
- 两处都没有密钥时，配置仍可加载，但创建模型时会报 `Missing model API key`。
- `.env` 只为 `api_key_env` 提供回退，不负责 `model_env` 和 `api_base_env`。

本地 PowerShell 示例：

```powershell
$env:QwenAPI = "<api-key>"
uv run dabench run-task task_1 --config configs/easy.yaml
```

Docker 当前要求：

| 环境变量 | 用途 | 是否必须 |
| --- | --- | --- |
| `MODEL_NAME` | `agent.model` | 必须 |
| `MODEL_API_URL` | OpenAI-compatible base URL | 必须 |
| `MODEL_API_KEY` | 模型密钥 | 实际运行必须 |
| `DABENCH_SUBMISSION_CONFIG` | Docker 入口使用的配置路径 | 可覆盖，默认 `configs/docker.yaml` |
| `DABENCH_LOG_ROOT` | `runtime.log` 所在根目录 | 可覆盖，默认 `/logs` |
| `DABENCH_RETRY_SLEEP_SECONDS` | Docker 两轮运行之间的等待时间 | 可覆盖，默认 60 秒 |
| `HF_HUB_OFFLINE` | Hugging Face 运行时离线模式 | 镜像固定为 `1` |
| `TRANSFORMERS_OFFLINE` | Transformers 运行时离线模式 | 镜像固定为 `1` |

注意：覆盖 `DABENCH_LOG_ROOT` 只改变 Docker entrypoint 的 `runtime.log` 路径；`configs/docker.yaml` 中的 `run.log_dir` 仍然是 `/logs`。

Docker 构建参数：

| build arg | 默认值 | 用途 |
| --- | --- | --- |
| `HF_ENDPOINT` | `https://huggingface.co` | 构建期模型下载端点。 |
| `PRELOAD_FASTER_WHISPER_MODEL` | `base` | 构建期预载 faster-whisper；设为空字符串可跳过。 |
| `VERIFY_QWEN_TOKENIZER_CACHE` | `1` | 校验 `assets/huggingface/Qwen3.5-35B-A3B/` tokenizer 缓存；设 `0` 跳过。 |

## 4. 全部 YAML 配置项

### 4.1 `dataset`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `dataset.root_path` | `data/input` | `/input` | 任务数据集根目录。 |

### 4.2 `agent`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `agent.model` | `gpt-4.1-mini` | 由 `MODEL_NAME` 提供 | 模型名称；设置 `model_env` 后此字面值不生效。 |
| `agent.model_env` | `null` | `MODEL_NAME` | 模型名环境变量名；存在时必须可读取。 |
| `agent.api_base` | `https://api.openai.com/v1` | 由 `MODEL_API_URL` 提供 | OpenAI-compatible API 根地址。 |
| `agent.api_base_env` | `null` | `MODEL_API_URL` | API 地址环境变量名。 |
| `agent.api_key` | 空字符串 | 空字符串 | 可直接配置密钥；不建议提交到仓库。 |
| `agent.api_key_env` | `null` | `MODEL_API_KEY` | 密钥环境变量名；可回退项目 `.env`。 |
| `agent.max_steps` | 16 | 64 | 单题主模型轮次上限，不包含工具、校验和强制提交节点。 |
| `agent.temperature` | 0.0 | 0.2 | 主模型采样温度。 |
| `agent.model_request_timeout_seconds` | 1800 | 2400 | 单次模型调用超时；YAML 中用 `0` 或 `null` 关闭。加载器拒绝负数；只有直接构造 Python 配置时负数才会被模型层视为关闭。 |
| `agent.max_tokens` | 8192 | 16000 | 单次模型最大输出；`0` 或 `null` 不向客户端传该限制。 |
| `agent.enable_data_inspector` | `false` | **`true`** | 是否在主图前生成全局数据画像。 |
| `agent.enable_answer_validator` | `true` | **`true`** | 是否校验已提交答案。 |
| `agent.validation_retry_limit` | 2 | 2 | 答案校验允许拒绝并回环的次数。 |
| `agent.enable_process_validator` | `false` | **`true`** | 是否校验证据链和求解过程。 |
| `agent.enable_ambiguity_analysis` | `false` | **`false`** | 是否在主求解前运行歧义分析。代码支持，当前关闭。 |
| `agent.strip_reasoning_history` | `false` | `false` | 向后续请求发送历史时是否剥离推理文本。 |
| `agent.reasoning_history_limit` | `null` | 10 | 保留的推理历史数量上限；null 表示不设该上限。 |
| `agent.prompt_version` | 1 | 3 | 系统提示版本。当前可选构建器为 1、2、3；未知值回退版本 1。 |
| `agent.compress_used_image_messages` | `true` | `true` | 图片使用后是否压缩其历史消息。 |
| `agent.compressed_image_note_chars` | 600 | 600 | 压缩图片观察说明的字符上限。 |

### 4.3 `data_inspector.sample_budget`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `catalog_top_distinct_values` | 50 | 5 | 每个字段写入 Catalog 的 Top 高频值数量。 |
| `max_doc_tokens` | 500 | 1000 | 普通文档写入 Catalog 预览的 token 上限；`knowledge.md` 保留全文。 |

### 4.4 `data_inspector.semantic_views`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | `true` | **`true`** | 是否构建派生 Semantic View。 |
| `min_confidence` | 0.95 | 0.95 | 关系进入 View 的最低置信度。 |
| `min_distinct_match_ratio` | 0.90 | 0.90 | 来源 distinct 值匹配率下限。 |
| `strict_distinct_match_ratio` | 0.95 | 0.95 | 低于该值时保留 View 但增加警告。 |
| `min_target_uniqueness_ratio` | 1.0 | 1.0 | 目标键唯一性下限。 |
| `max_views` | 30 | 30 | 最多构建的 View 数。 |
| `max_dimension_fields_per_view` | 12 | 12 | 每个维表最多附加字段数。 |
| `max_dimensions_per_view` | 2 | 2 | 每个 View 最多连接维表数。 |
| `min_payload_fields` | 1 | 1 | 维表至少需要多少个可附加字段。 |
| `allow_one_to_one_enrichment` | `false` | **`true`** | 是否允许 one-to-one 关系参与 enrichment。 |

### 4.5 `process_validator`

这些参数只有 `agent.enable_process_validator=true` 时影响主流程。Docker 当前已启用。

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `checkpoint_model_interval` | 10 | 12 | 每隔多少个主模型步执行过程检查。必须大于 0。 |
| `retry_limit` | 5 | 1 | 被拒绝后的修正预算；0 表示不调用过程校验器。 |
| `recent_step_limit` | 8 | 200 | 提供给过程校验器的近期 trace 步数。 |
| `evidence_max_items` | 12 | 100 | 支持证据最大条目数。 |
| `evidence_max_str_tokens` | 300 | 10000 | 单个证据字符串 token 上限。 |
| `evidence_max_list_items` | 5 | 200 | 证据列表元素上限。 |

### 4.6 `video_preprocessing`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | `true` | **`true`** | 是否处理视频并生成时间线/稳定帧/摘要。 |
| `sample_fps` | 2.0 | 2.0 | 变化检测采样帧率，必须大于 0。 |
| `diff_threshold` | 0.025 | 0.010 | 判断画面变化的像素比例阈值。 |
| `pixel_delta` | 25 | 25 | 单像素变化阈值。 |
| `min_stable_duration` | 1.0 | 1.0 | 稳定片段最短秒数。 |
| `resize_width` | 320 | 320 | 差异计算用缩放宽度。 |
| `jpg_quality` | 95 | 95 | 稳定帧 JPEG 质量，范围 1–100。 |
| `max_attached_frames` | 16 | 32 | 注入主模型上下文的帧数上限。 |
| `asr_model` | `base` | `base` | faster-whisper 模型。 |
| `asr_device` | `cpu` | `cpu` | ASR 设备。 |
| `asr_compute_type` | `int8` | `int8` | ASR 计算精度。 |

`configs/video.yaml` 和 `configs/resume.yaml` 还写有 `video_preprocessing.dedup`、`hash_threshold`。当前 `VideoPreprocessingConfig` 和加载器没有这两个字段，它们会被静默忽略；因此是“YAML 中存在”，不是“代码支持或已启用”。

### 4.7 `tool`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `tool.max_output_tokens` | 10000 | 10000 | 普通工具返回给模型的字符串 token 上限。 |
| `tool.max_list_items` | 200 | 100 | 普通工具返回给模型的列表元素上限。 |

### 4.8 `tool.structured_doc`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `min_chunk_lines` | 25 | 30 | 文档抽取分块的最小行数。 |
| `max_chunk_lines` | 40 | 40 | 文档抽取分块的最大行数；不得小于最小值。 |
| `max_selected_lines_for_llm_extraction` | 400 | 500 | 自动选块后允许进入 LLM 抽取的最大行数。 |
| `default_max_model_calls` | 20 | 20 | 抽取工具默认模型调用预算。 |
| `hard_max_model_calls` | 20 | 20 | 抽取模型调用硬上限。 |
| `inspect_doc_structure_max_model_calls` | 3 | 3 | 文档结构检查调用预算，且不得超过硬上限。 |

### 4.9 `tool.structured_doc.llm`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `temperature` | 0.0 | 0.0 | 正常抽取温度。 |
| `top_p` | 1.0 | 1.0 | 抽取 top-p。 |
| `repetition_penalty` | 1.0 | 1.0 | 抽取重复惩罚。 |
| `max_tokens` | 16384 | 16384（YAML 未写，继承默认） | 抽取调用最大输出。 |
| `repair_temperature` | 0.3 | 0.3（YAML 未写，继承默认） | 修复无效抽取响应时的温度。 |

### 4.10 `run`

| 配置项 | 代码默认值 | Docker 当前有效值 | 说明 |
| --- | --- | --- | --- |
| `run.output_dir` | `artifacts/runs` | `/output` | prediction 根；run_dir 模式下也是运行根。 |
| `run.log_dir` | `null` | `/logs` | flat 模式的 summary/日志 run 根；flat 时必填。 |
| `run.output_layout` | `run_dir` | **`flat`** | 仅支持 `run_dir`、`flat`。 |
| `run.run_id` | `null` | `null` | 空值自动生成 UTC `YYYYMMDDTHHMMSSZ`；显式值必须是单个目录名。 |
| `run.max_workers` | 4 | 8 | 批量活跃任务上限。必须至少 1。 |
| `run.extract_structured_doc_max_workers` | 2 | 2 | 全局结构化文档抽取并发上限，必须大于 0。 |
| `run.task_timeout_seconds` | 600 | 1200 | 单任务硬超时；`<=0` 关闭。 |
| `run.extract_structured_doc_timeout_bonus_seconds` | 0 | 1800 | 任务调用文档抽取后一次性增加的超时秒数。 |
| `run.task_ids` | `null` | `null` | 可选任务列表；空值表示全部任务，空字符串忽略，重复项保序去重。 |

加载器当前不会拒绝未知顶层或嵌套字段。拼错配置名可能被静默忽略；修改 YAML 后应结合 `summary.json` 中已记录的部分有效字段、trace 和实际行为确认，因为 summary 并不回写全部配置项。

## 5. 现有配置文件定位

| 文件 | 当前用途和实际状态 |
| --- | --- |
| `configs/docker.yaml` | Docker 提交配置；环境变量模型，flat 输出，8 并发，过程/答案校验均开。 |
| `configs/easy.yaml` | DashScope 本地/远端配置；未设置 task_ids，当前遍历全部任务；6 并发。 |
| `configs/full.yaml` | 本地 `127.0.0.1:8000/v1`；**当前 task_ids 只含 `task_21`，并非全量**。 |
| `configs/selected.yaml` | 本地模型；当前只选 `task_173`、`task_169`、`task_199`；过程校验关闭。 |
| `configs/video.yaml` | DashScope；选择一组视频任务；过程校验关闭；两个旧视频去重字段被忽略。 |
| `configs/resume.yaml` | flat 输出到 `artifacts/resume_predictions`，日志 run 写到 `artifacts/runs`；过程校验关闭；两个旧视频去重字段被忽略。 |

文件名只是约定，最终任务范围以 `run.task_ids` 的实际内容为准。

## 6. CLI 完整命令

统一形式：

```powershell
uv run dabench <command> [arguments] [options]
```

| 命令 | 完整形式 | 说明 |
| --- | --- | --- |
| `status` | `dabench status --config PATH` | 检查项目、配置和数据集路径，统计任务数。 |
| `inspect-task` | `dabench inspect-task TASK_ID --config PATH` | 查看题目元信息和原始 context 文件。 |
| `run-task` | `dabench run-task TASK_ID --config PATH` | 运行单题并写产物。 |
| `repeat-task` | `dabench repeat-task TASK_ID [--runs N|-n N] [--workers N|-w N] --config PATH` | 同一题重复运行并直接计算公开 gold 指标；默认 10 次、8 worker。 |
| `run-benchmark` | `dabench run-benchmark --config PATH [--limit N] [--skip-completed]` | 批量运行配置选择的任务。 |
| `score-run` | `dabench score-run [RUN_ID] [--lambda FLOAT ...]` | 对已有 run 评分；省略 RUN_ID 时取 `artifacts/runs` 名字最新的目录。 |
| `compare-runs` | `dabench compare-runs RUN_REF RUN_REF [...]` | 比较多个已有 `score.json`；RUN_REF 可为 `artifacts/runs` 下 ID 或包含 score.json 的路径。 |

## 7. 典型运行流程

### 7.1 单题检查与运行

```powershell
uv run dabench inspect-task task_21 --config configs/full.yaml
uv run dabench run-task task_21 --config configs/full.yaml
```

`run-task` 会创建独立 run 目录，但不写 batch `summary.json`，因此不能直接交给 `score-run`。

### 7.2 重复运行

```powershell
uv run dabench repeat-task task_21 --runs 10 --workers 4 --config configs/full.yaml
```

实际并发为 `min(workers, runs)`。每次运行写入 `run_01`、`run_02` 等子目录；命令直接读取 `data/output/<task>/gold.csv` 并汇总 full cover、recall、redundancy 和 proxy score。没有公开 gold 时会失败。

### 7.3 批量运行

运行 YAML 中的 `run.task_ids`；若为空则运行数据集全部任务：

```powershell
uv run dabench run-benchmark --config configs/selected.yaml
```

只运行选中范围的前 5 题：

```powershell
uv run dabench run-benchmark --config configs/easy.yaml --limit 5
```

跳过 prediction 根中已经存在 `prediction.csv` 的题：

```powershell
uv run dabench run-benchmark --config configs/resume.yaml --skip-completed
```

批量结束后会自动尝试本地评分；缺少公开 gold 或输出布局不满足评分器要求时，只打印 `Scoring skipped`，不把已经完成的 Agent 运行改成失败。

### 7.4 本地评分

```powershell
uv run dabench score-run
uv run dabench score-run 20260721T120000Z
uv run dabench score-run 20260721T120000Z --lambda 0.1 --lambda 0.3
```

评分范围严格来自该 run 的 `summary.json.tasks`，并读取：

```text
artifacts/runs/<run_id>/<task_id>/prediction.csv
data/output/<task_id>/gold.csv
```

输出 `score.json` 和 `score_report.md`。本地分数是公开 demo 的 proxy 诊断，不代表 hidden test 官方最终分数。

CLI `score-run` 的运行根目前硬编码为 `artifacts/runs`。使用自定义 `run.output_dir` 或 Docker flat 布局时，不能通过参数直接指定任意 run 路径；需要将产物组织到该布局，或调用 Python 评分 API。

### 7.5 运行对比

先确保每个目标目录都有 `score.json`：

```powershell
uv run dabench compare-runs 20260720T120000Z 20260721T120000Z
uv run dabench compare-runs artifacts/standard/baseline artifacts/standard/candidate
```

对比项包括任务数、预测数、主 proxy score、平均 recall/冗余率、模型步数、P95 耗时和主要失败原因。`compare-runs` 只读已有评分文件，不自动重新评分。

## 8. 输出目录结构

### 8.1 `run_dir` 布局

```text
<run.output_dir>/
└── <run_id>/
    ├── summary.json                 # batch；run-task 不生成
    ├── task_status.jsonl            # batch 任务状态
    ├── tool_gate_events.jsonl       # 启用抽取门控时可能存在
    ├── score.json                   # 成功执行本地评分后
    ├── score_report.md
    └── task_<id>/
        ├── trace.json               # 实时原子更新，最终去掉 partial
        ├── prediction.csv           # 只有存在 AnswerTable 时生成
        ├── semantic_catalog.json    # 启用/触发 Catalog 时可能存在
        ├── global_data_profile.json
        ├── ambiguity_analysis.json  # 启用歧义分析时可能存在
        ├── generated_context/       # PDF/视频生成资产
        ├── video_preprocessing/
        ├── video_understanding/
        └── structured_doc/          # 文档抽取调试记录可能存在
```

### 8.2 重复运行布局

`repeat-task` 在 run_dir 模式下使用单独的层级：

```text
<run.output_dir>/<run_id>/
├── summary.json
├── run_01/task_<id>/trace.json + prediction.csv
├── run_02/task_<id>/trace.json + prediction.csv
└── ...
```

该 summary 是重复实验汇总，不是 batch `summary.json.tasks` 格式；重复实验已经在命令内部完成评分。

### 8.3 Docker `flat` 布局

```text
/output/
└── task_<id>/
    ├── prediction.csv
    ├── trace.json
    └── 任务级画像与预处理资产...

/logs/
├── runtime.log
└── <run_id>/
    ├── summary.json
    ├── task_status.jsonl
    └── tool_gate_events.jsonl       # 门控启用时
```

flat 模式要求 `run.log_dir`。预测和 trace 位于 `prediction_output_root`，batch 汇总位于 `run_output_dir`；两者不是同一目录。

## 9. 并发与超时

| 控制项 | 边界 |
| --- | --- |
| `run.max_workers` | 普通批量路径的活跃任务数。外部传入共享 model/tools 的代码调用会强制单 worker。 |
| `extract_structured_doc_max_workers` | 当小于任务 worker 数时启用专用门控。等待抽取的任务不占活跃槽，因此物理子进程数可能大于 `max_workers`。 |
| `task_timeout_seconds` | Agent 任务级墙钟硬超时；超时先 terminate，1 秒后仍存活则 kill。`<=0` 关闭。 |
| `extract_structured_doc_timeout_bonus_seconds` | 只要任务请求过文档抽取即增加一次；门控等待时间从有效任务耗时中扣除。 |
| `model_request_timeout_seconds` | 每次主模型、校验器或视频理解相关模型请求的请求级上限，不等于整题上限。 |
| `execute_python` | 工具自身固定 30 秒进程级超时，不受 YAML 调整。 |
| `max_steps` | 主模型轮次上限，不是秒数；达到后还可尝试一次强制提交。 |

特别边界：`execute_task` 在显式传入 model/tools 时绕过任务子进程硬超时；`run_benchmark` 单 worker 分支会复用共享实例，因此该分支没有 `task_timeout_seconds` 的进程级强制终止，只剩请求级和工具级超时。

## 10. 中断恢复与跳过机制

### 10.1 实时 trace 与异常恢复

- 每个节点执行中和完成后都会原子更新 `trace.json`。
- 任务超时或子进程异常时，父进程尝试从 trace 恢复已经提交的完整答案。
- 最终失败结果如果没有步骤，会保留 partial trace 中已有的步骤、Inspector、全局画像和答案，并标记 `finalized_from_partial_trace=true`。
- 恢复到答案不等于任务成功；`failure_reason` 仍会保留，便于审计。

### 10.2 Ctrl+C

批量收到 `KeyboardInterrupt` 时会把未完成任务写为 `Interrupted by user.`，在 batch `summary.json` 中记录 `interrupted=true`。专用抽取门控路径会终止仍存活的任务子进程。

### 10.3 `--skip-completed`

跳过判据只有一个：

```text
<prediction_output_root>/<task_id>/prediction.csv 已存在
```

- 只有 trace、失败记录或预处理资产不会触发跳过，任务会重跑。
- 不校验已有 CSV 的内容、时间或配置版本；换模型/配置重跑前应使用新的输出根。
- flat 模式可在新 run_id 下复用固定 prediction 根，适合恢复。
- run_dir 模式每次创建全新且不得已存在的 run 目录，通常无法用同一个 run_id 原地续跑；`--skip-completed` 在这种布局下主要用于程序内部已指向同一 prediction 根的场景。

### 10.4 Docker 两轮机制

当前 Dockerfile 固定执行两轮：

1. `.venv/bin/dabench run-benchmark --skip-completed --config ...`
2. 记录退出码，等待 `DABENCH_RETRY_SLEEP_SECONDS`。
3. 再执行同一命令；已有 `prediction.csv` 被跳过，失败题会重试。
4. 容器最终返回第二轮退出码。

它不是单个失败请求的模型重试机制，而是整个 benchmark 的第二次恢复扫描。

## 11. Docker 构建、挂载与提交

### 11.1 构建

```powershell
docker build -t <team>:<version> .
```

如不需要视频或构建环境不能下载 Whisper：

```powershell
docker build `
  --build-arg PRELOAD_FASTER_WHISPER_MODEL= `
  --build-arg VERIFY_QWEN_TOKENIZER_CACHE=0 `
  -t <team>:<version> .
```

默认构建会：

- 从 `python:3.11-slim` 开始。
- 使用 `uv sync --frozen --no-dev` 创建 `/app/.venv`。
- 只复制源码、`configs/docker.yaml` 和 Hugging Face 缓存；其他本地 preset 不进入镜像。
- 预载 faster-whisper `base`。
- 验证 Qwen tokenizer 缓存。
- 运行时强制 Hugging Face/Transformers 离线。

### 11.2 本地挂载运行

```powershell
$projectDir = (Get-Location).Path
$inputDir = Join-Path $projectDir "data\input"
$outputDir = Join-Path $projectDir "docker_sim\output"
$logsDir = Join-Path $projectDir "docker_sim\logs"
New-Item -ItemType Directory -Force $outputDir, $logsDir | Out-Null

docker run --rm `
  --cpus=16 `
  --memory=64g `
  --memory-swap=64g `
  -v "${inputDir}:/input:ro" `
  -v "${outputDir}:/output:rw" `
  -v "${logsDir}:/logs:rw" `
  -e MODEL_API_URL="https://your-model-endpoint/v1" `
  -e MODEL_API_KEY="<api-key>" `
  -e MODEL_NAME="your-model-name" `
  <team>:<version>
```

挂载契约：

- `/input`：只读任务输入。
- `/output`：持久化每题预测、trace 和任务级资产。
- `/logs`：`runtime.log` 及每轮 run 的 summary。

### 11.3 导出和提交镜像

```powershell
docker save -o <team>_<version>.tar <team>:<version>
```

若平台要求 gzip：

```powershell
gzip -k <team>_<version>.tar
```

提交前建议在干净的 `docker_sim/output`、`docker_sim/logs` 上完整启动一次，并确认：

```powershell
Get-ChildItem $outputDir -Recurse
Get-Content (Join-Path $logsDir "runtime.log") -Tail 100
```

仓库代码只定义容器入口、目录契约和镜像导出方式；最终文件命名、压缩格式、大小限制和上传位置以比赛平台当期要求为准。

## 12. 常见错误与定位

| 现象 | 常见原因 | 定位与处理 |
| --- | --- | --- |
| `Missing environment variable MODEL_NAME/MODEL_API_URL` | Docker 环境变量未传；这两个字段不从 `.env` 回退。 | 检查 `docker run -e` 或平台注入变量。 |
| `Missing model API key` | `api_key` 为空，环境和 `.env` 都没有 `api_key_env` 对应值。 | 本地检查项目 `.env`；Docker 检查 `MODEL_API_KEY`。不要把真实密钥写进提交仓库。 |
| 401/403 | 密钥无效或 endpoint 权限不足。 | 查看 `trace.json` 模型步骤的 `model_response.error` 和 `/logs/runtime.log`。 |
| 404、模型不存在 | `api_base` 不含正确 `/v1`，或 `model` 名与服务端不一致。 | 先用 `status` 验证配置可加载，再核对服务端模型列表和 Chat Completions 路径。 |
| 服务端报未知参数 | OpenAI 兼容层不接受 `repetition_penalty`、tool calling 或多模态字段。 | 查看原始 HTTP 错误；当前配置不能关闭这些固定字段，需要调整服务兼容层或代码。 |
| 长时间处于 retrying | 429/5xx/网络错误的外层重试无总次数上限。 | 查 trace 的 `request_retry`、runtime.log 和服务健康状态；必要时中断任务。 |
| `Model invocation timed out` | 单次请求超过模型请求超时。 | 调整 `agent.model_request_timeout_seconds`，同时确保小于合理的任务总超时。超时本身不会自动重试。 |
| `Task timed out after ...` | 整题超过任务级墙钟上限。 | 查看最后 trace 节点；若文档抽取确实必要，检查 bonus 是否配置及 gate 日志。 |
| `Python execution timed out after 30 seconds` | 模型生成的 Python 过慢。 | 将工作改写为 DuckDB SQL、减少数据扫描或拆分逻辑；30 秒当前不可由 YAML 调整。 |
| 数据集为空或任务找不到 | `dataset.root_path` 错、缺 task.json/context，或 task_id 不匹配目录名。 | 运行 `status` 和 `inspect-task`；检查数据布局。 |
| `run_id` 目录已存在 | run_dir 创建使用 `exist_ok=false`。 | 清空 `run.run_id` 自动生成新 ID，或改为新的单目录名。不要填路径。 |
| flat 模式启动失败 | 未配置 `run.log_dir`。 | 同时设置 `output_layout: flat`、`output_dir`、`log_dir`。 |
| 只生成 trace，没有 prediction.csv | Agent 未成功调用 `submit_tool_result`，或提交/校验失败。 | 查看 `trace.json.failure_reason`，搜索最后的 `force_answer`、`tool`、`validate_*` 步骤。 |
| 文档抽取返回 `missing_doc_structure` | 未先建立结构缓存。 | 先调用 `inspect_doc_structure`，再用精确 fields 调 `extract_structured_doc`。 |
| 文档抽取 input-too-large | fields 太多或选中行超过限制。 | 缩小 fields；必要时用 `search_doc`/`read_doc` 配合 Python 解析；查 `structured_doc` 日志。 |
| Semantic View 未出现 | 关系置信度、匹配率、唯一性不足，或实际行数改变。 | 查 `semantic_catalog.json.relationship_warnings` 和关系 evidence。 |
| `score-run requires summary.json` | 使用了 `run-task` 目录或不完整 run。 | 使用 batch run；单题请用 `repeat-task` 的内置评分。 |
| `Gold output directory/file not found` | hidden 数据无 gold，或公开数据未放在 `data/output`。 | 本地评分只适用于公开 demo；Agent 运行结果本身仍可使用。 |
| `score-run` 找不到自定义输出 | CLI 固定从 `artifacts/runs` 解析 run ID。 | 使用标准布局，或调用评分 API；`compare-runs` 可直接接收带 score.json 的路径。 |
| `--skip-completed` 没跳过 | prediction 根不一致，或题目只有 trace 没有 CSV。 | 按 summary 的 `prediction_output_root` 检查 `<task>/prediction.csv`。 |
| YAML 修改无效 | 字段拼错或属于未支持旧字段；加载器忽略未知键。 | 对照本文完整字段表，并检查 summary 中已记录的字段、trace 与实际行为。 |
| Docker 构建缺 tokenizer | `assets/huggingface/Qwen3.5-35B-A3B/` 未进入构建上下文。 | 补齐缓存，或仅在确认替代 tokenizer 行为可接受时设 `VERIFY_QWEN_TOKENIZER_CACHE=0`。 |
| Docker 构建无法下载 Whisper | 构建网络/HF endpoint 不可用。 | 指定 `HF_ENDPOINT`，或将 `PRELOAD_FASTER_WHISPER_MODEL` 设为空并确认运行时已有模型。 |
| 本地 `.venv` 指向已删除 Python | Python 安装移动或旧虚拟环境失效。 | 重新安装/选择 Python 3.10+，让 uv 重新同步环境；不要继续使用损坏的 `.venv`。 |

建议定位顺序：

1. `status` 确认配置和数据根。
2. `inspect-task` 确认目标题及资产。
3. 查看任务 `trace.json` 的 `failure_reason` 和最后 3 个步骤。
4. 查看 batch `summary.json`、`task_status.jsonl`。
5. 涉及抽取并发时查看 `tool_gate_events.jsonl`。
6. Docker 中最后查看 `/logs/runtime.log`，区分 CLI、进程退出和第二轮恢复行为。

## 13. 代码依据

- `src/data_agent_baseline/config.py`
- `src/data_agent_baseline/cli.py`
- `src/data_agent_baseline/agents/model.py`
- `src/data_agent_baseline/model_retry.py`
- `src/data_agent_baseline/run/runner.py`
- `src/data_agent_baseline/scoring.py`
- `src/data_agent_baseline/benchmark/dataset.py`
- `configs/*.yaml`
- `Dockerfile`、`.dockerignore`、`pyproject.toml`
