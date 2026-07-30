<div align="center">

# DataAgent-Bench Starter Kit

English | [中文](README.zh.md)

[![Official Website](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1lICQVM_LfyQ5DMEIZjssq6aaOPTWCtNd/view?usp=share_link)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> Official starter kit for the KDD Cup 2026 DataAgent-Bench challenge. The repository reads tasks from `data/input/` and writes predictions for downstream evaluation.

## Overview

| Item                     | Value                                      |
| ------------------------ | ------------------------------------------ |
| Dataset input            | `data/input/`                     |
| Public demo ground truth | `data/output/task_<id>/gold.csv`  |
| Hidden test data         | `input/` only, no `output/`            |
| Entry command            | `uv run dabench <command> --config PATH` |
| Default run output       | `artifacts/runs/`                        |

## Quick Start

1. Install `uv` by following the official guide:

   - https://docs.astral.sh/uv/getting-started/installation/
2. On macOS and Linux, the standalone installer is:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Install project dependencies:

   ```bash
   uv sync
   ```
4. Confirm the dataset root is visible:

   ```bash
   uv run dabench status --config configs/full.yaml
   ```
5. Run the baseline:

   ```bash
   uv run dabench run-benchmark --config configs/full.yaml
   ```
6. Score the latest run against the public demo gold files using task IDs from that run's `summary.json`:

   ```bash
   uv run dabench score-run
   ```

## Dataset

The public demo dataset lives under `data/input/`. Each task directory follows this structure:

```text
data/input/task_<id>/
├── task.json
└── context/
```

The corresponding public demo answers live separately under `data/output/task_<id>/gold.csv`.
Hidden test sets only include `input/`, so there is no `output/` directory there.

`task.json` contains:

- `task_id`
- `difficulty`
- `question`

The `context/` directory may contain one or more of:

- CSV files
- JSON files
- SQLite / DB files
- Text documents

## Configuration

The `configs/` directory intentionally keeps only three configs:

| Config | Purpose |
| --- | --- |
| `configs/docker.yaml` | Docker evaluation entry. Reads `/input`, writes predictions to `/output`, writes logs/debug artifacts to `/logs`. |
| `configs/full.yaml` | Local full public run. Runs every `task_<id>` under `data/input`. |
| `configs/selected.yaml` | Local selected-task run. Runs only IDs listed in `run.task_ids`. |

`configs/selected.yaml` uses the same shape as `configs/full.yaml`, with `run.task_ids` added:

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

Config fields:

| Field                        | Meaning                                                                                                                                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset.root_path`        | Root directory of the public demo `input/` dataset. Relative paths are resolved from the project root.                                                                                                                                         |
| `agent.model`              | Model name.                                                                                                                                                                                                                                      |
| `agent.model_env`          | Optional environment variable name for the model name. When set, it overrides `agent.model`.                                                                                                                                                   |
| `agent.api_base`           | OpenAI-compatible API base URL.                                                                                                                                                                                                                  |
| `agent.api_base_env`       | Optional environment variable name for the API base URL. When set, it overrides `agent.api_base`.                                                                                                                                              |
| `agent.api_key`            | API key, read directly from the config file. Leave empty when using `.env`.                                                                                                                                                                     |
| `agent.api_key_env`        | API key environment variable name. The loader checks the process environment first, then falls back to the project root `.env` file.                                                                                                            |
| `agent.max_steps`          | Maximum model turns per task.                                                                                                                                                                                                                    |
| `agent.temperature`        | Sampling temperature.                                                                                                                                                                                                                            |
| `agent.model_request_timeout_seconds` | Per-request model timeout in seconds. Set to `0`, a negative value, or `null` to disable the request timeout.                                                                                                                                             |
| `run.output_dir`           | Output directory for run artifacts.                                                                                                                                                                                                              |
| `run.log_dir`              | Optional log/debug artifact directory. Required for `run.output_layout: flat`, where predictions go to `run.output_dir` and traces/summaries go to `run.log_dir`.                                                                              |
| `run.output_layout`        | `run_dir` for local runs under `output_dir/<run_id>/`; `flat` for Docker evaluation outputs under `output_dir/<task_id>/prediction.csv`.                                                                                                      |
| `run.run_id`               | Optional run directory name. Defaults to a UTC timestamp if omitted. Must be a single directory name; existing run directories are rejected.                                                                                                     |
| `run.max_workers`          | Parallel worker count for `run-benchmark`.                                                                                                                                                                                                     |
| `run.extract_structured_doc_max_workers` | Global concurrent execution limit for `extract_structured_doc`; queued tasks do not consume active benchmark worker slots.                                                                                                                   |
| `run.task_timeout_seconds` | Maximum wall-clock time per task. Set to `0` or a negative value to disable the task-level timeout.                                                                                                                                            |
| `run.extract_structured_doc_timeout_bonus_seconds` | Extra seconds added once per task when it calls `extract_structured_doc`. Defaults to `0`; ignored when task timeouts are disabled. |
| `run.task_ids`             | Optional task ID list used by `run-benchmark`. Empty values are ignored and duplicates are de-duplicated in order.                                                                                                                            |

## CLI

```bash
uv run dabench <command> [options]
```

| Command           | Purpose                                                                                                                    | Example                                                                           |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `status`        | Show project paths, config path, dataset root, and public task counts.                                                     | `uv run dabench status --config configs/full.yaml`            |
| `inspect-task`  | Show task metadata and list accessible files under `context/`.                                                           | `uv run dabench inspect-task task_1 --config configs/selected.yaml` |
| `run-task`      | Run the baseline on one task and write outputs.                                                                            | `uv run dabench run-task task_1 --config configs/selected.yaml`     |
| `run-benchmark` | Run all tasks, or only `run.task_ids` when the config lists task IDs.                                                       | `uv run dabench run-benchmark --config configs/full.yaml`       |
| `score-run`     | Evaluate one run against the public demo `gold.csv` files, expose recall / redundancy diagnostics, and report a default primary score at `λ=0.1` plus the multi-`λ` proxy grid. Only task IDs recorded in that run's `summary.json` are scored. Defaults to the latest run when `run_id` is omitted. | `uv run dabench score-run 20260407T022447Z --lambda 0.1 --lambda 0.3`          |

`run-benchmark` also supports `--limit N` to cap the number of tasks.
When `run.task_ids` is present, `run-benchmark` runs only those tasks; otherwise it traverses all `task_<id>` directories under the configured dataset root.
Use `configs/full.yaml` for all tasks and `configs/selected.yaml` for selected tasks.
Commands that execute tasks require `--config PATH`; `score-run` reads existing artifacts and does not need a config file, but it now requires the target run directory to include `summary.json`.

To avoid storing secrets in YAML, you can leave `agent.api_key` empty and put the key name in `agent.api_key_env`. Example:

```yaml
agent:
  model: deepseek-chat
  api_base: https://api.deepseek.com
  api_key: ""
  api_key_env: DEEPSEEK_API_KEY
  max_steps: 16
  temperature: 0.0
```

Then create a project root `.env` file with the matching variable, for example `DEEPSEEK_API_KEY=...`.
For Docker evaluation, `configs/docker.yaml` reads `MODEL_API_URL`, `MODEL_API_KEY`, and `MODEL_NAME` directly from the environment injected by the platform.

## Docker Submission Simulation

The repository now includes a root `Dockerfile` for local evaluation-style simulation.
The image follows the evaluation platform contract: `/input` is read-only task input, `/output` receives only predictions, and `/logs` receives runtime logs plus debug artifacts.
It keeps the source tree under `/app` and uses:

```dockerfile
ENTRYPOINT ["/bin/sh", "-c", "mkdir -p /output /logs && uv run dabench run-benchmark --config configs/docker.yaml >/logs/runtime.log 2>&1"]
```

Build the image from the project root:

```powershell
docker build -t team0042:v1 .
```

Prepare local bind-mount directories in PowerShell:

```powershell
$proj = "C:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit"
$inputDir = Join-Path $proj "data\input"
$outputDir = Join-Path $proj "docker_sim\output"
$logsDir = Join-Path $proj "docker_sim\logs"
New-Item -ItemType Directory -Force $outputDir, $logsDir | Out-Null
```

Run the container with the same `/input`, `/output`, `/logs` contract used by the evaluation platform:

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

Optional image export and reload:

```powershell
docker save -o team0042_v1.tar team0042:v1
docker load -i team0042_v1.tar
```

Inspect the generated outputs and logs:

```powershell
Get-ChildItem $outputDir -Recurse
Get-Content (Join-Path $logsDir "runtime.log") -Tail 100
```

## Tools

The baseline exposes these tools to the model:

| Tool                      | Purpose                                                               | Inputs                       |
| ------------------------- | --------------------------------------------------------------------- | ---------------------------- |
| `list_context`          | List files and directories under `context/`.                        | `max_depth`                |
| `inspect_all_schema`    | Inspect all CSV, JSON, and SQLite schemas plus data-validated join relationships. | `include_relationships`, `max_depth` |
| `read_doc`              | Read a text document preview.                                         | `path`, `max_chars`      |
| `execute_context_sql`   | Execute read-only SQL against a SQLite / DB file in `context/`.     | `path`, `sql`, `limit` |
| `execute_python`        | Execute arbitrary Python code inside a temporary copy of the task `context/` directory. | `code`                     |
| `submit_tool_result`    | Submit the final answer by re-executing `execute_probe_query` or `execute_python` and using its complete output. | `tool_name`, `tool_args`, `columns` |

All file paths passed to tools must be relative to the task `context/` directory.

## Scoring

The repository now includes a local scoring command for the public demo set:

```bash
uv run dabench score-run [run_id] [--lambda FLOAT ...]
```

If `run_id` is omitted, the command scores the latest run directory under `artifacts/runs/`.
The scorer reads task IDs from `artifacts/runs/<run_id>/summary.json`, compares each matching `prediction.csv` with `data/output/task_<id>/gold.csv`, and intentionally reports a local proxy evaluation instead of pretending to know the official hidden `λ`:

- Each task exposes both `recall` and `redundancy_rate`.
- The default primary score is fixed at `λ=0.1`.
- The CLI also reports multiple proxy scores using `recall - λ * redundancy_rate`.
- The default local `λ` grid is `0.0, 0.05, 0.1, 0.2, 0.3, 0.5`.
- Column names are ignored.
- Each column is compared as an unordered value vector, so row order does not matter within a column.
- Cell values are normalized before comparison: null aliases collapse to empty strings, numerics round to 2 decimals, dates normalize to ISO dates, and timezone-aware datetimes convert to UTC.
- Name fields support both `first_name + last_name` and combined full-name columns.
- `full_cover` remains available as a task-level diagnostic, but it is no longer used as the aggregate scoring view.

This local scorer only works for the public demo tasks because hidden test sets do not ship with `gold.csv`.
Runs produced by `run-task` do not include `summary.json`, so they cannot be scored directly with `score-run`.

## Outputs

Each successful task run may produce:

- `trace.json`
- `prediction.csv`

Per-task outputs are written to:

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

Benchmark runs also write:

```text
artifacts/runs/<run_id>/summary.json
```

Scoring a run also writes:

```text
artifacts/runs/<run_id>/score.json
artifacts/runs/<run_id>/score_report.md
```

`score.json` is the machine-readable summary, while `score_report.md` is a human-friendly Chinese report for review.

## Contact

- Open issues: https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- Official website: https://dataagent.top
- Discord: https://discord.com/invite/7eFwJQN3Fx
- WeChat official account: `数据智能与分析实验室 DIAL`

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
        Official Website
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
        WeChat Official Account
      </td>
    </tr>
  </table>
</div>

## Main Modules

| Module                                           | Responsibility                                              |
| ------------------------------------------------ | ----------------------------------------------------------- |
| `src/data_agent_baseline/benchmark/dataset.py` | Public dataset loader                                       |
| `src/data_agent_baseline/tools/filesystem.py`  | `list_context`, `read_doc`, legacy CSV/JSON previews |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python`                                          |
| `src/data_agent_baseline/tools/sqlite.py`      | SQLite schema helpers, `execute_context_sql`          |
| `src/data_agent_baseline/tools/registry.py`    | Tool registration and terminal `submit_tool_result`       |
| `src/data_agent_baseline/agents/prompt.py`     | Tool-calling system prompt and task prompt                  |
| `src/data_agent_baseline/agents/langgraph_runtime.py` | LangGraph runtime with native tool calling           |
| `src/data_agent_baseline/agents/state.py`      | LangGraph state schema                                      |
| `src/data_agent_baseline/run/runner.py`        | Single-task and benchmark execution                         |
