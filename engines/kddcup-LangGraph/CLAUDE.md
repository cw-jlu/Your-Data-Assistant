# CLAUDE.md

KDD Cup 2026 DataAgent-Bench 竞赛 starter kit — 基于 LangGraph 的 ReAct LLM agent，通过工具调用完成数据分析任务并输出 `prediction.csv`。

## 常用命令

```bash
# 安装依赖
uv sync

# 查看项目状态
uv run dabench status --config configs/full.yaml

# 查看单个任务
uv run dabench inspect-task task_1 --config configs/selected.yaml

# 运行单个任务
uv run dabench run-task task_11 --config configs/selected.yaml

# 运行全量 benchmark
uv run dabench run-benchmark --config configs/full.yaml

# 限制任务数运行
uv run dabench run-benchmark --config configs/full.yaml --limit 5

# 评分（默认最新 run）
uv run dabench score-run

# 运行全部测试
uv run pytest tests/ -x -v

# 运行指定测试文件
uv run pytest tests/test_runner.py -x -v

# 跳过慢测
uv run pytest tests/ -x -v --ignore=tests/test_eval_increment0.py

# 代码检查
uv run ruff check src/ tests/
```

## 技术栈

- Python >=3.10, `uv` 包管理, Hatchling 构建
- LangGraph + LangChain + OpenAI-compatible API
- Typer CLI + Rich 终端, PyYAML 配置
- pandas / polars / duckdb / pyarrow / openpyxl 数据分析
- pymupdf / faster-whisper / opencv 多模态处理

## 数据目录

### `data/` — 题目与标准答案（60个任务，不在 git 内，需单独下载）

```
data/
├── input/                          # 题目定义 + 上下文数据
│   ├── task_1/                     # 每个任务一个目录（task_1 ~ task_60）
│   │   ├── task.json               #   任务问题定义 {"task_id", "question"}
│   │   └── context/                #   上下文数据（按类型分子目录）
│   │       ├── knowledge.md        #   领域知识指南（列/表含义、业务规则）
│   │       ├── csv/                #   CSV 表格数据（.csv）
│   │       ├── json/               #   JSON 表格数据（.json, 含 table + records）
│   │       ├── doc/                #   文档（.md 说明 + .pdf 报告/公告）
│   │       ├── db/                 #   SQLite 数据库（sub_db.sqlite, 57/60 个任务有）
│   │       └── video/              #   简报视频（briefing.mp4, 30/60 个任务有）
│   └── task_N/...
│
└── output/                         # 标准答案（ground truth）
    ├── task_1/
    │   └── gold.csv                #   标准答案（gold.csv, 每个任务一个）
    └── task_N/...
```

**任务领域分布：**
| 前缀 | 领域 | 涉及任务数 |
|---|---|---|
| `lc_` | A 股上市公司（IPO、股本、分红、高管） | ~14 |
| `mf_` | 公募基金（净值、经理、费率、分红） | ~24 |
| `ed_` | 宏观经济（GDP、货币、CPI、PMI） | ~14 |
| 临床 | 医疗数据（患者、诊断、处方、实验室） | ~6 |

**文件命名：** CSV/JSON/Doc 文件名使用领域前缀（`lc_` / `mf_` / `ed_` / `qt_`）+ 内容描述。约 30 个任务的 `question` 引用视频，视频中包含问题所需的具体筛选条件（准入线、时间口径等），需结合 ASR 文字识别 + 关键帧截图来获取。

### `artifacts/` — Agent 运行输出（日志、预测、评分、调试）

```
artifacts/
├── runs/                           # 正式运行输出（按时间戳分目录）
│   └── 20260620T072429Z/           #   每次 benchmark 运行一个目录
│       ├── summary.json            #   运行配置 + 结果元数据
│       ├── score.json              #   评分结果（recall, redundancy, lambda 分析）
│       ├── score_report.md         #   可读评分报告（中文）
│       ├── task_status.jsonl       #   每任务状态（成功/失败、输出路径）
│       ├── tool_gate_events.jsonl  #   工具门控事件流（extract_structured_doc 等）
│       └── task_N/                 #   单任务输出
│           ├── prediction.csv      #   Agent 预测答案
│           ├── trace.json          #   完整执行轨迹（所有 tool call + LLM 交互）
│           ├── global_data_profile.json  # 数据资产统计画像
│           ├── semantic_catalog.json     # 语义目录（表/列/关联关系）
│           └── context_preprocessing_manifest.json  # 上下文预处理清单
│
├── debug/                          # 调试用 LLM 全量调用日志
│   └── task_N_full_context/        #   某任务的完整上下文 dump
│       ├── call_01_model.json      #   第 N 次 LLM 调用消息（JSON）
│       ├── call_01_model.txt       #   同上（纯文本，方便 grep）
│       ├── call_NN_main_agent.json/txt  #   后续 agent 调用
│       └── run_summary.json        #   该运行的摘要
│
├── test/                           # 测试数据画像（60 个任务）
│   ├── _summary.json               #   聚合摘要
│   └── task_N/                     #   单任务画像
│       ├── global_data_profile.json
│       └── semantic_catalog.json
│
├── sample/                         # 实验性 agent 配置（sample recipes）
│   ├── 0baseline/                  #   基线版本
│   ├── 06video_agent/              #   视频处理 agent
│   ├── 07过程校验agent/            #   过程校验 agent
│   ├── 08long_doc/                 #   长文档处理（当前分支相关）
│   └── ...                         #   其他实验变体
│
└── phase1/                         # Phase 1 历史数据
    ├── history/                    #   历史运行（48 个时间戳目录）
    │   └── YYYYMMDDTHHMMSSZ/
    │       └── task_N/
    │           ├── prediction.csv  #   历史预测
    │           └── trace.json      #   历史轨迹
    ├── sample/                     #   Phase 1 实验配置（10+ 变体）
    └── debug/                      #   Phase 1 调试 dump
```

**关键文件说明：**
| 文件 | 说明 |
|---|---|
| `prediction.csv` | Agent 最终预测，通常 2 列 + 少量数据行 |
| `trace.json` | 完整执行轨迹，所有 step / tool call / LLM 交互，体积较大（数百 KB） |
| `summary.json` | 运行级元数据（task_count, elapsed, token 消耗等） |
| `score.json` | 评分结果 + 多 lambda 分析（recall / redundancy 权衡） |
| `score_report.md` | 中文评分报告，含概览表 + 逐任务明细 |
| `gold.csv` | 标准答案（在 `data/output/` 下），用来和 `prediction.csv` 对比评分 |

## 架构速查

```
src/data_agent_baseline/
├── cli.py              # CLI 入口 (status/inspect-task/run-task/run-benchmark/score-run)
├── config.py           # 配置 dataclass 和 YAML 加载
├── scoring.py          # 评分逻辑 (recall, redundancy)
├── agents/
│   ├── langgraph_runtime.py  # LangGraph 主循环 + tool calling
│   ├── prompt.py / prompt2.py # System/task prompt (prompt_version 控制)
│   ├── model.py         # OpenAI-compatible 模型封装
│   ├── state.py         # LangGraph 状态定义
│   ├── answer_validator.py    # 答案校验 agent
│   ├── process_validator.py   # 过程校验 agent
│   ├── question_analyzer.py   # 问题分析
│   └── ambiguity_analyzer.py  # 歧义分析
├── tools/
│   ├── registry.py      # 工具注册、分发、submit_tool_result
│   ├── filesystem.py    # list_context, read_doc, search_doc
│   ├── sqlite.py        # SQLite schema + 只读查询
│   ├── python_exec.py   # execute_python (子进程隔离, 30s 超时)
│   ├── probe_engine.py  # execute_probe_query, get_column_distinct_values
│   ├── structured_doc_extractor.py  # extract_structured_doc (长文档结构化提取)
│   ├── doc_structure.py # inspect_doc_structure
│   └── langgraph_tools.py  # LangGraph 工具定义 + 参数 schema
├── inspectors/
│   ├── data_understanding_agent.py  # 数据探查 agent
│   ├── semantic_catalog.py          # 语义目录构建
│   └── semantic_views.py            # 语义视图匹配
├── run/
│   ├── runner.py        # 单任务/批量执行编排
│   ├── context_preprocessor.py  # 上下文预处理 (含 video)
│   └── video_preprocessor.py   # 视频稳定帧提取 + ASR
└── benchmark/
    ├── dataset.py       # 任务发现与加载
    └── schema.py        # PublicTask, AnswerTable 等数据结构
```

## 核心设计约束

- **文件隔离**：工具操作的文件路径必须相对 `context/`，`resolve_context_path()` 阻止越界
- **SQL 只读**：仅允许 SELECT/WITH/PRAGMA 开头，SQLite 以只读 URI 连接
- **Python 沙箱**：`execute_python` 在独立子进程运行，固定 30s 超时
- **任务级容错**：批量运行中单任务失败不影响全局，超时/异常都会记录
- **Config 优先**：所有行为由 YAML 驱动，`configs/` 下仅保留 docker/full/selected 三个配置

## 评分规则

评分将 Agent 产出的 `prediction.csv` 与标准答案 `gold.csv` 进行**列级匹配**，计算 Recall、Redundancy Rate 和 Proxy Score。核心实现在 `scoring.py`。

### 评分流程

```
prediction.csv ──→ 加载列向量 ──→ 单元格归一化 ──→ 列签名匹配 ──→ 贪心搜索最优匹配 ──→ Recall / Redundancy / Proxy Score
gold.csv ────────→ 加载列向量 ──→ 单元格归一化 ──→
```

### 单元格归一化 (`normalize_cell`)

匹配前所有单元格经过统一归一化：

| 类型 | 规则 | 示例 |
|---|---|---|
| NULL 同义词 | `""`, `"null"`, `"none"`, `"nan"`, `"nat"`, `"<na>"` → `""` | |
| 数值 | 四舍五入到 0.01 精度（`Decimal.quantize("0.01")`） | `"3.14159"` → `"3.14"` |
| 日期 | 严格 ISO 8601 零填充（`YYYY-MM-DD`） | `"2024-3-1"` → 不归一化（格式不匹配） |
| 日期时间 | 归一化到 UTC，后缀 `Z` | `"2024-03-01T12:00:00+08:00"` → `"2024-03-01T04:00:00Z"` |
| 普通文本 | 保持原样（大小写敏感） | |

### 列签名匹配

每列的**签名** = 该列所有单元格值去重排序后的 tuple。gold 列和 prediction 列的签名相同即视为**内容匹配**。

支持三种匹配模式：

| 模式 | 说明 | 触发条件 |
|---|---|---|
| **1-to-1** | 一个 gold 列签名 = 一个 prediction 列签名 | 默认 |
| **2-to-1 (name)** | 两个 gold 列（均含 "name"）合并后签名 = 一个 prediction 列签名 | First Name + Last Name 合并为一列 Full Name |
| **1-to-2 (name)** | 一个 gold 列签名 = 两个 prediction 列（均含 "name"）合并后签名 | 一列 Full Name 拆分为 First/Last Name 两列 |

匹配通过**贪心搜索**选取全局最优组合：优先覆盖 gold 列数最多，其次匹配 prediction 列数最多。

### 核心指标

| 指标 | 公式 | 含义 |
|---|---|---|
| **Recall**（覆盖率） | `覆盖的 gold 列数 / gold 总列数` | Agent 找到了多少标准答案列 |
| **Redundancy Rate**（冗余率） | `多余 prediction 列数 / prediction 总列数` | 预测结果中有多少是无关列 |
| **Full Cover** | `Recall == 1.0` 且 `gold_column_count > 0` | 是否完全覆盖所有 gold 列 |

### Proxy Score（代理分数）

```
Proxy Score(λ) = max(Recall - λ × Redundancy Rate, 0)
```

- λ 控制对冗余列的惩罚力度：λ=0 只看召回不惩罚冗余；λ 越大冗余惩罚越重
- **默认主分 λ = 0.1**，用于稳定比较
- **λ 网格**：(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)，用于多 λ 敏感度分析
- 每个任务得分 = 该任务 recall 和 redundancy 代入公式；运行级得分为所有任务得分均值

### 边界情况

- `prediction.csv` 缺失 → Recall=0, Redundancy=0, reason="prediction.csv is missing."
- `prediction.csv` 格式错误 → Recall=0, Redundancy=0, reason=具体错误信息
- gold 列数为 0 且 prediction 列数为 0 → Recall=0, Redundancy=0（双方均无列）
- `prediction.csv` 仅 header 无数据行 → 列向量为空，Recall=0

### 评分报告结构

`score-run` 输出 `score.json` + `score_report.md`，报告包含：

1. **执行摘要** — Primary Score、Mean Recall、Mean Redundancy、总 Token 消耗
2. **多 λ 代理分数** — 各 λ 下的 Proxy Score 表格
3. **按难度拆分** — easy/medium/hard/extreme 各组的得分和覆盖
4. **失败原因分析** — 各失败原因的任务计数
5. **运行时摘要** — 总耗时、P95 耗时、模型轮数统计
6. **节点耗时分析** — 每个 LangGraph 节点的调用次数/耗时分布
7. **工具调用统计** — 每种工具的调用总次数、每任务均值
8. **Token 消耗统计** — Input/Output/Reasoning 总量和每任务均值
9. **复盘任务列表** — Primary Score < 0.5 的任务明细
10. **全量任务附录** — 所有任务逐行得分详情

### 评分 CLI

```bash
# 对最新 run 评分
uv run dabench score-run

# 对指定 run 评分
uv run dabench score-run --run-id 20260620T072429Z

# 自定义 λ 网格
uv run dabench score-run --lambda 0.0,0.1,0.3,0.5

# 比较多个 run 的得分
uv run dabench compare-runs <run-id-1> <run-id-2>
```

## 编码约定

- 行宽 100 (ruff); Python 3.10 语法 (`from __future__ import annotations`)
- dataclass 用 `frozen=True, slots=True`；配置对象不可变
- 工具函数命名：公开工具用 `_tool` 后缀，内部用 `_` 前缀
- 路径操作统一用 `pathlib.Path`，不用 `os.path`
- 日志输出用 `rich.Console`，不用 `print`
- 类型注解完整覆盖公开接口

## 关键配置字段 (configs/*.yaml)

| 区域 | 字段 | 说明 |
|---|---|---|
| `agent` | `max_steps` | 单任务最大模型轮数 |
| `agent` | `enable_data_inspector` | 启用数据探查 agent |
| `agent` | `enable_answer_validator` | 启用答案校验 |
| `agent` | `enable_process_validator` | 启用过程校验 |
| `agent` | `prompt_version` | 1=prompt.py, 2=prompt2.py |
| `agent` | `compress_used_image_messages` | 压缩已使用的图片消息 |
| `tool` | `max_output_tokens` | 工具输出最大 token |
| `tool.structured_doc` | `min_chunk_lines / max_chunk_lines` | 文档分块大小 |
| `tool.structured_doc` | `hard_max_model_calls` | 文档提取最大 LLM 调用次数 |
| `run` | `max_workers` | 批量并发数 |
| `run` | `extract_structured_doc_max_workers` | 文档提取专用并发 |
| `run` | `task_timeout_seconds` | 单任务超时 |

## 当前分支: `long-doc-processing`

- 聚焦长文档的结构化提取和分块处理优化
- 核心文件：`tools/structured_doc_extractor.py`, `tools/doc_structure.py`
- 活跃子系统：answer_validator, process_validator, context_preprocessor
- 视频预处理 (`video_preprocessor.py`) 也在近期加入

## 注意事项

- 测试数据不在 git 内 (`data/` 在 .gitignore 中)；公开 demo 需单独下载
- 模型 API 密钥通过 `.env` 文件或环境变量注入，不要硬编码在 YAML
- `run-task` 不生成 `summary.json`，无法直接用 `score-run` 评分
- Docker 评估用 `configs/docker.yaml`，入口和环境变量不同
- `extract_structured_doc` 调用后任务超时自动延长 (`extract_structured_doc_timeout_bonus_seconds`)
- 图片消息压缩 (`compress_used_image_messages`) 截断为 600 字符占位文本
