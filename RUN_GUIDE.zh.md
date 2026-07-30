# DataAgent-Bench 运行指南（含注意事项）

本文档面向当前仓库结构（public 与 starter-kit 同级），汇总常用运行方式、配置建议和常见问题。

## 1. 先决条件

1. 已安装 uv（建议先执行 uv --version 验证）。
2. 命令在 starter-kit 目录执行。
3. 首次运行前先安装依赖：

```powershell
Set-Location d:\code\python\kdd\starter-kit
uv sync
```

## 2. 目录与配置约定

- 任务输入目录：../public/input
- 官方公开答案：../public/output/task_<id>/gold.csv
- 运行产物目录（默认）：artifacts/runs

说明：

- gold.csv 是官方参考答案，不是你本次运行产生的预测文件。
- status 命令只检查状态，不会生成 prediction.csv。

## 3. 推荐准备方式（本地配置文件）

建议不要直接修改示例配置，复制一份本地配置：

```powershell
Set-Location d:\code\python\kdd\starter-kit
Copy-Item configs/react_baseline.example.yaml configs/react_baseline.local.yaml
```

然后编辑 configs/react_baseline.local.yaml，至少确认以下字段：

```yaml
dataset:
  root_path: ../public/input

agent:
  model: 你的模型名
  api_base: 你的 OpenAI-compatible 接口地址
  api_key: 你的密钥
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 1
  task_timeout_seconds: 600
```

注意事项：

- root_path 推荐写相对路径 ../public/input，跨机器更稳。
- run_id 留空时自动按时间戳生成；手动指定时必须是新目录名，重复会报错。
- max_workers 太大时可能触发接口限流，建议从 1 到 2 逐步增加。
- api_key 建议通过本地配置或环境变量管理，不要提交到仓库。

## 4. 运行方式一：CLI（最常用）

### 4.1 环境与数据检查

```powershell
Set-Location d:\code\python\kdd\starter-kit
uv run dabench status --config configs/react_baseline.local.yaml
```

用途：确认配置文件、数据目录是否可见，以及公开任务数量。

### 4.2 查看单任务元信息

```powershell
uv run dabench inspect-task task_11 --config configs/react_baseline.local.yaml
```

用途：查看题目描述及 context 内可访问文件列表。

### 4.3 运行单任务

```powershell
uv run dabench run-task task_11 --config configs/react_baseline.local.yaml
```

用途：调试 prompt、工具调用和单题输出。

### 4.4 运行批量任务（公开集）

先小规模压测：

```powershell
uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 1
```

再全量运行：

```powershell
uv run dabench run-benchmark --config configs/react_baseline.local.yaml
```

用途：正式批量跑公开数据集，输出 summary 和各任务结果。

## 5. 运行方式二：main.py（评测入口兼容）

```powershell
Set-Location d:\code\python\kdd\starter-kit
uv run python main.py
```

行为说明：

1. 本地环境：读取 configs/react_baseline.example.yaml。
2. 若配置中的数据目录不存在，会尝试回退到 ../public/input。
3. 若检测到 /input 存在（容器评测环境），会自动改用 /input 与 /output。
4. 可通过环境变量 BENCHMARK_LIMIT 限制任务数量（便于冒烟测试）。

示例：

```powershell
$env:BENCHMARK_LIMIT = "2"
uv run python main.py
```

## 6. 结果输出与定位

默认输出根目录：artifacts/runs

目录结构示例：

```text
artifacts/runs/<run_id>/
  summary.json
  task_11/
    prediction.csv
    trace.json
  task_19/
    prediction.csv
    trace.json
```

含义：

- prediction.csv：模型最终预测结果。
- trace.json：任务过程轨迹（动作、观测、错误信息等）。
- summary.json：批量运行汇总。

## 7. 常见问题与排查

### 7.1 status 显示成功但没有预测文件

原因：status 不执行任务，只做状态检查。

处理：执行 run-task 或 run-benchmark。

### 7.2 报找不到数据目录

排查：

1. 确认当前目录是 starter-kit。
2. 配置中 dataset.root_path 是否正确。
3. 先运行 status 验证 dataset_root 显示 present。

### 7.3 run_id 已存在导致报错

原因：同名输出目录已存在。

处理：

1. 删除旧目录后重试，或
2. 更换 run.run_id，或
3. 留空让系统自动生成。

### 7.4 接口超时或速率限制

建议：

1. 降低 run.max_workers。
2. 增大 run.task_timeout_seconds。
3. 先用 --limit 小规模运行验证稳定性。

## 8. 推荐执行顺序（稳妥版）

1. uv sync
2. status
3. inspect-task（可选）
4. run-benchmark --limit 1
5. run-benchmark（全量）
6. 检查 artifacts/runs/<run_id>/summary.json 与各 task 的 prediction.csv
