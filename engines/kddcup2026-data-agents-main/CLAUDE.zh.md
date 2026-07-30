# CLAUDE.md

> 🌐 **Language**: [English](CLAUDE.md) · [한국어](CLAUDE.ko.md) · **中文**

本文件为 Claude Code (claude.ai/code) 在本仓库工作时提供指南。

## 项目

KDD Cup 2026 DABench / DataAgent-Bench 挑战的 ReAct 风格 baseline (`data-agent-baseline`, package `data_agent_baseline`)。从 `data/public/input/` 读取任务,运行 LLM 驱动的 ReAct 智能体配合 per-task 工具,为每个任务写出 `prediction.csv` 和 `trace.json`。

Python ≥3.10, 由 `uv` 管理。CLI 入口 `dabench` 在 `[project.scripts]` 中定义,resolve 到 `data_agent_baseline.cli:main`。

## 常用命令

> **macOS dev + DGX vLLM 分离工作流 (team1438):** vLLM 仅由 DGX (`<VLLM_HOST>:8000`) 服务。构建/测试/提交打包都在 macOS local 上进行。下面命令都在 macOS 上跑,LLM endpoint 通过 env 注入 — `export MODEL_API_URL=http://<VLLM_HOST>:8000/v1; export MODEL_API_KEY=local; export MODEL_NAME=qwen3.5-35b-a3b`。`Dockerfile` 钉死了 `--platform=linux/amd64`,所以即使在 Apple Silicon 上也能产出兼容主办方 amd64 环境的镜像。详细步骤见 [`README.zh.md`](README.zh.md) §3。

```bash
uv sync                                                            # 从 uv.lock 安装 deps
uv sync --extra dev                                                # 包括 pytest + ruff
uv run dabench status        --config configs/local.yaml           # 显示路径 + 数据集存在性
uv run dabench inspect-task task_<id> --config configs/local.yaml  # 任务元数据 + context 树
uv run dabench run-task     task_<id> --config configs/local.yaml  # 单一任务 end-to-end
uv run dabench run-benchmark          --config configs/local.yaml  # 所有任务
uv run dabench run-benchmark          --config configs/local.yaml --task-set data/public/holdout_ids.txt  # 仅 holdout
uv run dabench run-benchmark          --config configs/local.yaml --limit 5  # smoke
uv run ruff check src                                              # lint (line-length 100, py310)
uv run pytest                                                      # tests/ (gitignored, 本地跑)
uv run pytest tests/path/to/test_x.py::test_name                   # 单测

# Holdout split (在 data/public/input/ 放数据后跑一次):
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public

# 对 gold 做本地 mock 打分:
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> --gold data/public/output \
    --input data/public/input --lambda-values 0.05 0.10 0.20

# 构建并打分 Docker 提交:
bash scripts/build_submission.sh v3               # → submissions/dabench_v3.tar.gz
bash scripts/local_eval.sh v3                     # 镜像对完整 holdout
bash scripts/local_eval.sh v3 data/public/holdout_ids.txt  # 仅 holdout 子集
```

被追踪的 configs: `configs/eval.yaml` (Docker submission, env-overridable) 和 `configs/local.yaml` (本地 vLLM 假定的开发默认值 — 复制作为新 dev config 的起点)。其他 `configs/*` 均 gitignored。被追踪的 docs: `docs/ARCHITECTURE.md`, `docs/SYSTEM_FLOW.md`, `docs/DATA_ANALYSIS.md`, `docs/SUBMISSION_LOG.md`, `docs/HARNESS_STRUCTURE.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/qwen_endpoint_capabilities.md`,以及它们的 `.ko.md` / `.zh.md` 变体。其它 docs 均 gitignored。

## 架构

运行时是一个严格的 layered pipeline。修改任何一层几乎都需要重新阅读其上下两层。

`**cli.py` (Typer)** — 声明 4 个 subcommand 并 orchestrate progress UI。`PROJECT_ROOT` 定义为 `Path(__file__).resolve().parents[2]`;YAML 中的相对路径由 `config.py` 基于此 root 解析。

`**config.py`** — 把 YAML 加载为 frozen dataclass (`AppConfig` → `DatasetConfig | AgentConfig | RunConfig`)。优先级 **env > YAML > default**。评测容器依赖此机制:主办方在 runtime 注入 `MODEL_API_URL`、`MODEL_API_KEY`、`MODEL_NAME`,我们的 YAML 不能抢先。其它 env knob: `DABENCH_INPUT_DIR`, `DABENCH_OUTPUT_DIR`, `DABENCH_RUN_ID`, `DABENCH_MAX_WORKERS`, `DABENCH_TASK_TIMEOUT`, `DABENCH_FLAT_OUTPUT`, `DABENCH_LOG_FILE`, `DABENCH_LAMBDA` (mock scorer), `DABENCH_REPEAT_MAX`, `DABENCH_PASS_SAFETY_MARGIN` (multi-pass), `DABENCH_DISABLE_SELF_CONSISTENCY`。所有 path 返回 absolute `Path`。`run_id` 被规范化:空字符串 → `None` → 自动生成的 UTC timestamp;`runner.py` 中的 resolver 拒绝 `.`、`..`、含 `/` 或 `\`。

`**benchmark/dataset.py` + `schema.py`** — `DABenchPublicDataset` 发现 `task_<N>` 目录,按数字后缀排序,验证每个 `task.json` *精确* 拥有 `{task_id, difficulty, question}` 这套 key。键不匹配时 raise — 不要静默扩展 schema。`PublicTask` 同时暴露 `task_dir` 和 `context_dir`;工具只能从 `context_dir` 读取。

`**agents/*`* — ReAct 循环:

- `model.py` — `ModelAdapter` Protocol;`OpenAIModelAdapter` 对任意 OpenAI 兼容的 `api_base` 调用 `chat.completions.create`。`ScriptedModelAdapter` 用于测试。
- `prompt.py` — system / task / observation prompt builder。Contract 严格:模型必须返回 **正好一个 JSON object** (键 `thought`、`action`、`action_input`),包在单一 ```json fenced block 中,前后无任何文本。修改 contract 需要 lockstep 更新 `react.parse_model_step`。**G-3:** knowledge.md cap 5000 字符 + question-keyword H2/H3 reorder。
- `react.py` — `ReActAgent.run` 迭代到 `max_steps`,解析模型响应,通过 `ToolRegistry.execute` dispatch,把 observation 作为下一条 user message feed 回去。某工具返回 `is_terminal=True` 时终止 (只有 `answer` 会)。Parser 异常被捕获并以 `__error__` step record surface — 让 agent 在同一 task 内自我恢复。
- `self_consistency.py` (G-2) — `SelfConsistencyAgent` 在 hard/extreme tier 上以新的 OpenAIModelAdapter (temperature=0.5) 实例化 k=3 个 ReActAgent,按 `column_signature` multiset majority vote。
- `runtime.py` — `StepRecord`、`AgentRuntimeState`、`AgentRunResult` dataclass (序列化到 trace.json)。

`**tools/registry.py**` — agent 工具表面的唯一真理来源。`create_default_tool_registry()` 把 names → `ToolSpec` (描述 + JSON schema 示例) 和 names → `ToolHandler` callable 连线。`describe_for_prompt()` 是模型实际看到的内容,所以 spec 文本是 prompt 的一部分;改动会影响 agent 行为。新增工具 = handler 函数 + `ToolSpec` entry + (若 terminal) 返回 `ToolExecutionResult(is_terminal=True, answer=..., normalized_answer=...)`。`_answer` handler 同时 emit raw `AnswerTable` (LLM literal 输出) 和 `normalized_answer` (经过 `scoring.normalize.normalize_answer_table`);`runner._write_task_outputs` 默认把 normalized 变体写入 `prediction.csv`。

`**tools/filesystem.py`、`tools/sqlite.py`、`tools/python_exec.py`** — 所有工具输入都是 *`context/` 下的相对路径*。`resolve_context_path` 强制此 invariant;绝不接受 absolute path 或逃逸 context root 的 path。SQL 执行只读。

`**run/runner.py**` — 5 个关注点 layered:

1. **Run dir 创建** (`create_run_output_dir(..., flat=...)`) — `flat=False` 用 `mkdir(exist_ok=False)` 并以 `<run_id>/` wrap (拒绝冲突)。`flat=True` (eval 模式) 直接写入 `output_root/task_<id>/...`,允许预先存在的 mount;内部仍然合成 run_id 用于 trace metadata。通过 `config.run.flat_output_dir` / `DABENCH_FLAT_OUTPUT` 连线。
2. **Per-task timeout 隔离** (`_run_single_task_with_timeout`) — `task_timeout_seconds > 0` 时 task 在 `multiprocessing.Process` + queue 中运行;父进程在 timeout 时 kill 子进程。`task_timeout_seconds <= 0` 则 in-process 跑。**副作用:** timeout 模式下每个 task 在子进程内构造自己的 `OpenAIModelAdapter` 和 `ToolRegistry`;不能传 `model=` / `tools=` override。当提供 override 时 runner 会 fallback 到 single-process 模式 (强制 `max_workers=1`) — 用于 `ScriptedModelAdapter` 测试。
3. **并行** — `run_benchmark` 在 `>1` workers 时使用 `ThreadPoolExecutor(max_workers=...)` (每个 worker 为 task spawn 自己的 subprocess),提供 override 或 `max_workers == 1` 时则共享 model/tools 串行循环。`--task-set <file>` (或 `task_filter` kwarg) 把 run 限制到 task id 列表 — 用于 holdout-only 迭代。
4. **Multi-pass orchestrator (H-1)** — `run_benchmark_with_passes` 在 `repeat_max > 1` 时把同一 task set 跑 N 次然后 cross-run vote。Per-pass 输出到 `output_dir/_runs/run_<i>/`,最后由 `cross_run_vote` 写出 voted final 到 `output_dir/task_<id>/`。`pass_safety_margin × last_pass_duration` budget guard 保证回归安全;voter 失败时 `_fallback_copy_pass` 复制 pass 0。
5. **Runtime log** (`RuntimeLogger`) — 当 `config.run.log_file` 设置时,runner 把 JSON Lines 事件 (`benchmark_start`、`task_done`、`benchmark_end` + 5 个 multi-pass 事件) append 到该路径。eval rules 要求 `/logs/runtime.log`;`configs/eval.yaml` 自动连线。

`execute_python` (`tools/python_exec.py`) 也是 subprocess 隔离 + 硬 30s timeout (`tools/registry.py` 的 `EXECUTE_PYTHON_TIMEOUT_SECONDS`),`chdir` 到 `task.context_dir`,在 fd 级别捕获 stdout/stderr。Nested multiprocessing: `task_timeout_seconds > 0` 且 agent 调 `execute_python` 时,你会得到 process-in-a-process。在不容忍此的 debugger 下不要跑 benchmark。

`**scoring/`** — 官方 DataAgent-Bench scorer 的本地镜像。竞赛规则 `Score = Recall − λ·(ExtraCols/PredictedCols)`,column-signature matching 忽略列名和行序 (规范化后:numeric → 2 decimals, dates → ISO 8601, nulls → `""`, strings → trimmed case-sensitive)。

- `scoring/normalize.py` — `normalize_value`、`normalize_column`、`normalize_answer_table`、`column_signature`。无 policy 时 per-cell 自动判断类型。连线到 `_answer` 以与 raw 一起产出 normalized 变体。
- `scoring/mock_scorer.py` — 给定 `predictions/` 和 `gold/` root,计算 per-task `score`、`recall`、`matched/gold/predicted` columns。CLI 支持 `--lambda-values` 做 λ-sensitivity sweep。leaderboard 的 λ 未公开;我们默认 0.10,column-ablation 在 `{0.05, 0.10, 0.20}` 上 gate。
- `scoring/cross_run_vote.py` (H-1) — N 个 benchmark output root 之间的 column-multiset majority voter。CLI + library。`vote_across_runs(prediction_roots, output_dir)` 让我们能做 host-side ensemble 测量 (与容器内 in-runner voting 算法相同)。
- `scoring/holdout.py` — 确定性 80/20 hash 拆分 (`blake2b(salt + task_id)`),产生 `train_ids.txt` / `holdout_ids.txt`。holdout 是 prompt 迭代的唯一诚实信号;不要让它泄漏到 prompt 设计中。

**提交流程。** `Dockerfile` + `configs/eval.yaml` + `scripts/build_submission.sh` 产生 `submissions/dabench_<version>.tar.gz`。tarball 超过 10 GB 上限时 build script abort。`scripts/local_eval.sh` 用 `host.docker.internal:8000` 的 self-hosted vLLM 对 `data/public/input/` 跑构建好的镜像,把预测送过 mock scorer。

## 数据集布局

```
data/public/input/task_<id>/
  task.json    # {task_id, difficulty, question} — 精确 key 集合,无附加
  context/     # CSV / JSON / SQLite / 文本文件;所有工具 path 相对于此
data/public/output/task_<id>/gold.csv  # 公开 demo gold (独立 tree)
```

Hidden test set 只 ship `input/` — 永远不要写依赖 `output/` 存在的代码。

## Run 输出

```
artifacts/runs/<run_id>/
  <task_id>/
    trace.json        # 完整 step history + answer + failure_reason
    prediction.csv    # 仅当 agent 调用 answer 工具时存在
  summary.json        # 仅 benchmark run
```

`run_id` 默认 UTC timestamp `YYYYMMDDTHHMMSSZ`。整个 `artifacts/` 树 gitignored,只有 `artifacts/.gitkeep` 例外。

## gitignored 项目 (别期待在 repo 中找到)

`data/`、`tests/`、`evaluation/`、`artifacts/*` (除 `artifacts/.gitkeep`)、`submissions/`、`vllm_logs/`、大部分 `docs/*` (除 7 个被白名单的 docs × 3 种语言变体: ARCHITECTURE / SYSTEM_FLOW / SYSTEM_ARCHITECTURE / DATA_ANALYSIS / SUBMISSION_LOG / HARNESS_STRUCTURE / qwen_endpoint_capabilities)、大部分 `configs/*` (除 `eval.yaml`、`local.yaml`)。`.codex` 也 ignored。在以上树中新增追踪文件时,记得在 `.gitignore` 添加 `!path/to/file`。

## Plan & 竞赛上下文

要记住的关键规则:

- 评测时 LLM 通过 `MODEL_API_URL`/`MODEL_API_KEY`/`MODEL_NAME` 锁定为 `qwen3.5-35b-a3b`;不要硬编码。
- **单一模型政策 (team1438):** 规则技术上允许辅助模型 (embedding、retrieval) 在硬件预算内,但我们队伍只用 qwen 主求解器 — harness 的任何层都没有辅助 LLM、sentence-transformers、FAISS、ONNX vision/embedding、外部 web/vision API。`AgentConfig.model`/`api_base`/`api_key` 默认为空字符串,缺失 env 时显式 fail 而不是静默调用外部 provider。
- 计算资源: 16 vCPU / 64 GB RAM / 无 GPU / 所有任务 **共 12 小时**。除 LLM endpoint 外网络封锁。
- 提交上限: 1/天, 30/Phase-1。每次提交记录在 [`docs/SUBMISSION_LOG.zh.md`](docs/SUBMISSION_LOG.zh.md)。
- 挂载: `/input` (RO)、`/output` (RW)、`/logs` (RW)。`configs/eval.yaml` 已 wired 这些路径和 `flat_output_dir: true`。
