<div align="center">

# DataAgent-Bench Starter Kit

[English](README.md) | 中文

[![官方网站](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo 数据集](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1lICQVM_LfyQ5DMEIZjssq6aaOPTWCtNd/view?usp=share_link)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> 面向 KDD Cup 2026 DataAgent-Bench 挑战的官方 starter kit。仓库默认读取 `data/input/`，并为后续评测生成预测结果。

## Overview

| 项目               | 内容                                       |
| ------------------ | ------------------------------------------ |
| 数据输入           | `data/input/`                     |
| 公开 demo 标准答案 | `data/output/task_<id>/gold.csv`  |
| hidden test 数据   | 仅提供 `input/`，不提供 `output/`      |
| 入口命令           | `uv run dabench <command> --config PATH` |
| 默认输出目录       | `artifacts/runs/`                        |

## 快速开始

1. 请先按照 `uv` 官方安装指南安装 `uv`：

   - https://docs.astral.sh/uv/getting-started/installation/
2. 在 macOS 和 Linux 上，官方独立安装命令为：

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. 安装项目依赖：

   ```bash
   uv sync
   ```
4. 检查数据集根目录是否可见：

   ```bash
   uv run dabench status --config configs/full.yaml
   ```
5. 运行 baseline：

   ```bash
   uv run dabench run-benchmark --config configs/full.yaml
   ```
6. 基于该次运行 `summary.json` 记录的任务列表，对最新一次运行结果做公开 demo 本地评分：

   ```bash
   uv run dabench score-run
   ```

## 数据集

公开 demo 数据集默认位于 `data/input/`。每个任务目录结构如下：

```text
data/input/task_<id>/
├── task.json
└── context/
```

公开 demo 的标准答案文件单独放在 `data/output/task_<id>/gold.csv`。
hidden test set 只提供 `input/`，不会包含 `output/`。

`task.json` 包含：

- `task_id`
- `difficulty`
- `question`

`context/` 中可能包含一种或多种数据：

- CSV 文件
- JSON 文件
- SQLite / DB 文件
- 文本文档

## 配置

`configs/` 目录刻意只保留三份配置：

| 配置 | 作用 |
| --- | --- |
| `configs/docker.yaml` | Docker 评测入口。读取 `/input`，预测写到 `/output`，日志和调试产物写到 `/logs`。 |
| `configs/full.yaml` | 本地全量运行。遍历 `data/input` 下所有 `task_<id>`。 |
| `configs/selected.yaml` | 本地选择任务运行。只运行 `run.task_ids` 中列出的任务。 |

`configs/selected.yaml` 和 `configs/full.yaml` 结构一致，只是额外配置了 `run.task_ids`：

```yaml
dataset:
  root_path: data/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: ""
  api_key_env: YOUR_API_KEY_NAME
  max_steps: 16
  temperature: 0.0
  model_request_timeout_seconds: 120

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
  task_ids:
    - task_11
    - task_25
```

配置字段说明：

| 字段                         | 含义                                                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset.root_path`        | 公开 demo `input/` 数据集根目录。相对路径按项目根目录解析。                                                                                                                                  |
| `agent.model`              | 模型名称。                                                                                                                                                                                     |
| `agent.model_env`          | 可选的模型名称环境变量；设置后会覆盖 `agent.model`。                                                                                                                                           |
| `agent.api_base`           | OpenAI-compatible 接口根地址。                                                                                                                                                                 |
| `agent.api_base_env`       | 可选的接口地址环境变量；设置后会覆盖 `agent.api_base`。                                                                                                                                         |
| `agent.api_key`            | API key，直接从配置文件读取。使用 `.env` 时留空。                                                                                                                                            |
| `agent.api_key_env`        | API key 环境变量名。加载器会先读取系统环境变量，再回退到项目根目录 `.env`。                                                                                                                   |
| `agent.max_steps`          | 单个任务允许的最大模型轮数。                                                                                                                                                                   |
| `agent.temperature`        | 模型采样温度。                                                                                                                                                                                 |
| `agent.model_request_timeout_seconds` | 单次模型请求超时秒数。设为 `0`、负数或 `null` 可关闭请求级超时。                                                                                                                                        |
| `run.output_dir`           | 运行产物输出目录。                                                                                                                                                                             |
| `run.log_dir`              | 可选日志/调试产物目录。`run.output_layout: flat` 时必填，预测写入 `run.output_dir`，trace 和 summary 写入 `run.log_dir`。                                                                       |
| `run.output_layout`        | `run_dir` 表示本地 `output_dir/<run_id>/` 布局；`flat` 表示 Docker 评测的 `output_dir/<task_id>/prediction.csv` 布局。                                                                          |
| `run.run_id`               | 可选，指定运行目录名。不传时默认使用 UTC 时间戳；必须是单个目录名，已存在会报错。                                                                                                              |
| `run.max_workers`          | `run-benchmark` 并行 worker 数。                                                                                                                                                             |
| `run.extract_structured_doc_max_workers` | `extract_structured_doc` 的全局并发执行上限；等待队列中的任务不占用活跃 benchmark worker 槽位。                                                                                              |
| `run.task_timeout_seconds` | 单个任务允许的最长墙钟时间。设为 `0` 或负数可关闭任务级超时。                                                                                                                                |
| `run.task_ids`             | 可选任务 ID 数组，供 `run-benchmark` 选择任务使用。空白项会被忽略，重复 ID 会按原顺序去重。                                                                                                  |

## CLI

```bash
uv run dabench <command> [options]
```

| 命令              | 作用                                                                                        | 示例                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `status`        | 查看项目路径、配置路径、数据集根目录和公开任务数量。                                        | `uv run dabench status --config configs/full.yaml`            |
| `inspect-task`  | 查看任务元信息，并列出 `context/` 下可访问文件。                                          | `uv run dabench inspect-task task_1 --config configs/selected.yaml` |
| `run-task`      | 对单个任务运行 baseline，并写出结果。                                                       | `uv run dabench run-task task_1 --config configs/selected.yaml`     |
| `run-benchmark` | 批量运行全部任务；如果配置了 `run.task_ids`，则只运行这些任务。                            | `uv run dabench run-benchmark --config configs/full.yaml`       |
| `score-run`     | 对某次运行目录按公开 demo `gold.csv` 做本地评测，输出 Recall / 冗余率诊断，并给出默认 `λ=0.1` 主分与多组 `λ` 代理分数；仅评分该次运行 `summary.json` 中记录的任务，不传 `run_id` 时默认评分最新一次运行。 | `uv run dabench score-run 20260407T022447Z --lambda 0.1 --lambda 0.3`          |

`run-benchmark` 还支持 `--limit N`，用于限制任务数量。
当配置里存在 `run.task_ids` 时，`run-benchmark` 只运行这些任务；否则遍历数据集根目录下的所有 `task_<id>`。
全量运行用 `configs/full.yaml`，选择任务运行用 `configs/selected.yaml`。
涉及任务执行的命令需要传 `--config PATH`；`score-run` 直接读取已有产物，不需要配置文件，但目标运行目录必须包含 `summary.json`。

如果你想把密钥放在 `.env` 中，可以在项目根目录创建 `.env`，并在配置里写入对应变量名。例如：

```yaml
agent:
  model: deepseek-chat
  api_base: https://api.deepseek.com
  api_key: ""
  api_key_env: DEEPSEEK_API_KEY
  max_steps: 16
  temperature: 0.0
```

然后在项目根目录 `.env` 中写入 `DEEPSEEK_API_KEY=...`。
Docker 评测使用 `configs/docker.yaml`，会直接读取平台注入的 `MODEL_API_URL`、`MODEL_API_KEY` 和 `MODEL_NAME`。

## Docker 提交环境模拟

仓库根目录现在包含一个 `Dockerfile`，用于本地模拟提交环境。
镜像遵循评测平台约定：`/input` 是只读任务输入，`/output` 只写预测结果，`/logs` 写运行日志和调试产物。
镜像会保留 `/app` 下的源码目录，并固定使用：

```dockerfile
ENTRYPOINT ["/bin/sh", "-c", "mkdir -p /output /logs && uv run dabench run-benchmark --config configs/docker.yaml >/logs/runtime.log 2>&1"]
```

在项目根目录构建镜像：

```powershell
docker build -t team0042:v1 .
```

用 PowerShell 准备本地挂载目录：

```powershell
$proj = "C:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit"
$inputDir = Join-Path $proj "data\input"
$outputDir = Join-Path $proj "docker_sim\output"
$logsDir = Join-Path $proj "docker_sim\logs"
New-Item -ItemType Directory -Force $outputDir, $logsDir | Out-Null
```

按评测平台同样的 `/input`、`/output`、`/logs` 约定启动容器：

```powershell
docker run --rm `
  --cpus=16 `
  --memory=64g `
  --memory-swap=64g `
  -v "${inputDir}:/input:ro" `
  -v "${outputDir}:/output:rw" `
  -v "${logsDir}:/logs:rw" `
  -e MODEL_API_URL="https://your-model-endpoint/v1" `
  -e MODEL_API_KEY="your-api-key" `
  -e MODEL_NAME="your-model-name" `
  team0042:v1
```

可选的镜像导出与回载：

```powershell
docker save -o team0042_v1.tar team0042:v1
docker load -i team0042_v1.tar
```

查看输出与日志：

```powershell
Get-ChildItem $outputDir -Recurse
Get-Content (Join-Path $logsDir "runtime.log") -Tail 100
```

## Tools

当前暴露给模型的工具有：

| 工具                      | 作用                                              | 输入                         |
| ------------------------- | ------------------------------------------------- | ---------------------------- |
| `list_context`          | 列出 `context/` 下的文件和目录。                | `max_depth`                |
| `inspect_all_schema`    | 一次性查看 CSV、JSON、SQLite schema 以及经过数据验证的 join 关系。 | `include_relationships`、`max_depth` |
| `read_doc`              | 读取文本文档预览。                                | `path`、`max_chars`      |
| `execute_context_sql`   | 对 `context/` 内 SQLite / DB 文件执行只读 SQL。 | `path`、`sql`、`limit` |
| `execute_python`        | 在任务 `context/` 的临时副本目录内执行任意 Python 代码。  | `code`                     |
| `submit_tool_result`    | 重新执行 `execute_probe_query` 或 `execute_python`，并使用其完整输出提交最终答案表格。 | `tool_name`、`tool_args`、`columns` |

所有文件路径都必须是相对于任务 `context/` 目录的相对路径。

## 评分

仓库现在提供了一个面向公开 demo 的本地评分命令：

```bash
uv run dabench score-run [run_id] [--lambda FLOAT ...]
```

如果不传 `run_id`，命令会默认评分 `artifacts/runs/` 下名字最新的一次运行目录。
评分器会先读取 `artifacts/runs/<run_id>/summary.json` 里的任务 ID，再把对应的 `prediction.csv` 与 `data/output/task_<id>/gold.csv` 按新版官方规则进行比较；本地不会假装知道官方未公开的唯一 `λ`，而是输出代理评测结果：

- 每道题都会输出 `recall` 和 `redundancy_rate` 两个核心指标。
- 默认主分固定采用 `λ=0.1`。
- CLI 还会输出多组 `recall - λ * redundancy_rate` 代理分数。
- 默认本地 `λ` 网格为 `0.0, 0.05, 0.1, 0.2, 0.3, 0.5`。
- 评分时忽略列名。
- 每一列按“无序值向量”比较，因此列内行顺序不影响得分。
- 比较前会先做值规范化：空值别名归一为空字符串、数值四舍五入到两位小数、日期转 ISO 日期、带时区的 datetime 转 UTC。
- name field 同时支持 `first_name + last_name` 两列形式和单列 full name 形式。
- `full_cover` 仍会保留在单题诊断信息中，但不再作为聚合评分主视图。

这个本地评分器只适用于公开 demo 任务，因为 hidden test 不提供 `gold.csv`。
`run-task` 生成的运行目录不会包含 `summary.json`，因此不能直接用 `score-run` 评分。

## 输出

每个任务运行后可能生成：

- `trace.json`
- `prediction.csv`

单任务产物路径：

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

批量运行还会额外生成：

```text
artifacts/runs/<run_id>/summary.json
```

对某次运行评分后还会生成：

```text
artifacts/runs/<run_id>/score.json
artifacts/runs/<run_id>/score_report.md
```

其中 `score.json` 面向程序消费，`score_report.md` 是便于人工复盘的中文评测报告。

## Contact

- 问题反馈： https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- 官方网站： https://dataagent.top
- Discord： https://discord.com/invite/7eFwJQN3Fx
- 微信公众号：`数据智能与分析实验室 DIAL`

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://dataagent.top">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://dataagent.top&bgcolor=ffffff&color=111827&margin=8"
            alt="Official website QR code"
            width="144"
          />
        </a>
        <br />
        官方网站
      </td>
      <td align="center">
        <a href="https://discord.com/invite/7eFwJQN3Fx">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://discord.com/invite/7eFwJQN3Fx&bgcolor=ffffff&color=111827&margin=8"
            alt="Discord QR code"
            width="144"
          />
        </a>
        <br />
        Discord
      </td>
      <td align="center">
        <img
          src="https://dataagent.top/HKUSTGZ_DIAL.jpg"
          alt="WeChat official account QR code"
          width="144"
        />
        <br />
        微信公众号
      </td>
    </tr>
  </table>
</div>

## 主要模块

| 模块                                             | 责任                                                        |
| ------------------------------------------------ | ----------------------------------------------------------- |
| `src/data_agent_baseline/benchmark/dataset.py` | 公开数据集加载器                                            |
| `src/data_agent_baseline/tools/filesystem.py`  | `list_context`、`read_doc`、旧版 CSV/JSON 预览 helper |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python`                                          |
| `src/data_agent_baseline/tools/sqlite.py`      | SQLite schema helper、`execute_context_sql`          |
| `src/data_agent_baseline/tools/registry.py`    | 工具注册与终止型 `submit_tool_result`                     |
| `src/data_agent_baseline/agents/prompt.py`     | tool-calling system prompt 与 task prompt                   |
| `src/data_agent_baseline/agents/langgraph_runtime.py` | 基于原生 tool calling 的 LangGraph runtime         |
| `src/data_agent_baseline/agents/state.py`      | LangGraph 运行状态定义                                      |
| `src/data_agent_baseline/run/runner.py`        | 单任务和批量运行逻辑                                        |
