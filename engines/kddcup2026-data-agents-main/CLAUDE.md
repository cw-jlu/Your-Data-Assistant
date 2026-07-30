# CLAUDE.md

> 🌐 **Language**: **English** · [한국어](CLAUDE.ko.md) · [中文](CLAUDE.zh.md)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ReAct-style baseline (`data-agent-baseline`, package `data_agent_baseline`) for the KDD Cup 2026 DABench / DataAgent-Bench challenge. Reads tasks from `data/public/input/`, runs an LLM-driven ReAct agent against per-task tools, and writes a `prediction.csv` plus `trace.json` for each task.

Python ≥3.10, managed by `uv`. CLI entry point `dabench` is defined in `[project.scripts]` and resolves to `data_agent_baseline.cli:main`.

## Common commands

> **macOS dev + DGX vLLM split workflow (team1438):** vLLM is dedicated to DGX (`<VLLM_HOST>:8000`) for serving only. Build / test / submission packaging happens on macOS local. The commands below all run on macOS with the LLM endpoint injected via env — `export MODEL_API_URL=http://<VLLM_HOST>:8000/v1; export MODEL_API_KEY=local; export MODEL_NAME=qwen3.5-35b-a3b`. The `Dockerfile` pins `--platform=linux/amd64`, so even on Apple Silicon you produce an image compatible with the eval amd64 environment. See `[README.md](README.md)` §3 for the detailed steps.

```bash
uv sync                                                            # install deps from uv.lock
uv sync --extra dev                                                # include pytest + ruff
uv run dabench status        --config configs/local.yaml           # show paths + dataset presence
uv run dabench inspect-task task_<id> --config configs/local.yaml  # task metadata + context tree
uv run dabench run-task     task_<id> --config configs/local.yaml  # one task end-to-end
uv run dabench run-benchmark          --config configs/local.yaml  # all tasks
uv run dabench run-benchmark          --config configs/local.yaml --task-set data/public/holdout_ids.txt  # holdout-only
uv run dabench run-benchmark          --config configs/local.yaml --limit 5  # smoke
uv run ruff check src                                              # lint (line-length 100, py310)
uv run pytest                                                      # tests/ (gitignored, but run locally)
uv run pytest tests/path/to/test_x.py::test_name                   # single test

# Holdout split (run once after dropping data into data/public/input/):
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public

# Local mock scoring against gold:
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> --gold data/public/output \
    --input data/public/input --lambda-values 0.05 0.10 0.20

# Build + score a Docker submission:
bash scripts/build_submission.sh v3               # → submissions/dabench_v3.tar.gz
bash scripts/local_eval.sh v3                     # full holdout against the image
bash scripts/local_eval.sh v3 data/public/holdout_ids.txt  # holdout subset only
```

Tracked configs: `configs/eval.yaml` (Docker submission, env-overridable) and `configs/local.yaml` (developer default for local vLLM — copy as the starting point for new dev configs). All other `configs/*` are gitignored. Tracked docs: `docs/ARCHITECTURE.md`, `docs/SYSTEM_FLOW.md`, `docs/DATA_ANALYSIS.md`, `docs/SUBMISSION_LOG.md`, `docs/HARNESS_STRUCTURE.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/qwen_endpoint_capabilities.md`, plus their `.ko.md` / `.zh.md` variants. Other docs are gitignored.

## Architecture

The runtime is a strict layered pipeline. Touching a layer almost always requires re-reading the layer above and below.

`**cli.py` (Typer)** — declares four subcommands and orchestrates progress UI. Defines `PROJECT_ROOT` as `Path(__file__).resolve().parents[2]`; relative paths in YAML are resolved against this root by `config.py`.

`**config.py`** — loads YAML into frozen dataclasses (`AppConfig` → `DatasetConfig | AgentConfig | RunConfig`). Precedence is **env > YAML > default**. The eval container relies on this: judges inject `MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME` at runtime and our YAML must not preempt them. Other env knobs: `DABENCH_INPUT_DIR`, `DABENCH_OUTPUT_DIR`, `DABENCH_RUN_ID`, `DABENCH_MAX_WORKERS`, `DABENCH_TASK_TIMEOUT`, `DABENCH_FLAT_OUTPUT`, `DABENCH_LOG_FILE`, `DABENCH_LAMBDA` (mock scorer), `DABENCH_REPEAT_MAX`, `DABENCH_PASS_SAFETY_MARGIN` (multi-pass), `DABENCH_DISABLE_SELF_CONSISTENCY`. All paths come back as absolute `Path`s. `run_id` is normalized: empty string → `None` → auto-generated UTC timestamp; the resolver in `runner.py` rejects `.`, `..`, or anything containing `/` or `\`.

`**benchmark/dataset.py` + `schema.py`** — `DABenchPublicDataset` discovers `task_<N>` directories, sorts by numeric suffix, and validates that each `task.json` has *exactly* the keys `{task_id, difficulty, question}`. Raises on a key mismatch — do not silently extend the schema. `PublicTask` exposes both `task_dir` and `context_dir`; tools must only read from `context_dir`.

`**agents/`** — the ReAct loop:

- `model.py` — `ModelAdapter` Protocol; `OpenAIModelAdapter` calls `chat.completions.create` against any OpenAI-compatible `api_base`. `ScriptedModelAdapter` is for tests.
- `prompt.py` — system / task / observation prompt builders. The contract is hard: the model must return **exactly one JSON object** with keys `thought`, `action`, `action_input`, wrapped in a single ```json fenced block, no surrounding text. Changing this contract requires updating `react.parse_model_step` in lockstep. **G-3:** knowledge.md cap 5000 chars + question-keyword H2/H3 reorder.
- `react.py` — `ReActAgent.run` iterates up to `max_steps`, parses the model response, dispatches via `ToolRegistry.execute`, and feeds each observation back as the next user message. Terminates when a tool returns `is_terminal=True` (only `answer` does). Parser exceptions are caught and surfaced as `__error__` step records so the agent can recover within the same task.
- `self_consistency.py` (G-2) — `SelfConsistencyAgent` instantiates k=3 ReActAgents on hard/extreme tier with fresh OpenAIModelAdapter instances (temperature=0.5) and majority-votes by `column_signature` multiset.
- `runtime.py` — `StepRecord`, `AgentRuntimeState`, `AgentRunResult` dataclasses serialized into `trace.json`.

`**tools/registry.py`** — single source of truth for the agent's tool surface. `create_default_tool_registry()` wires names → `ToolSpec` (description + JSON schema example) and names → `ToolHandler` callables. `describe_for_prompt()` is what the model actually sees, so spec text is part of the prompt and changes affect agent behavior. Adding a tool means: handler function, `ToolSpec` entry, and (if terminal) returning `ToolExecutionResult(is_terminal=True, answer=..., normalized_answer=...)`. The `_answer` handler emits **both** the raw `AnswerTable` (the LLM's literal output) and the `normalized_answer` (run through `scoring.normalize.normalize_answer_table`); `runner._write_task_outputs` writes the normalized variant to `prediction.csv` by default so our submission CSV already matches the official scorer's normalization.

`**tools/filesystem.py`, `tools/sqlite.py`, `tools/python_exec.py`** — all tool inputs are *relative paths under `context/*`. `resolve_context_path` enforces this invariant; never accept absolute paths or paths that escape the context root. SQL execution is read-only.

`**run/runner.py`** — five concerns layered together:

1. **Run directory creation** (`create_run_output_dir(..., flat=...)`) — `flat=False` uses `mkdir(exist_ok=False)` and wraps with `<run_id>/` (rejecting collisions). `flat=True` (eval mode) writes directly under `output_root/task_<id>/...`, allowing a pre-existing mount; an internal run_id is still synthesized for trace metadata. Wired through `config.run.flat_output_dir` / `DABENCH_FLAT_OUTPUT`.
2. **Per-task timeout isolation** (`_run_single_task_with_timeout`) — when `task_timeout_seconds > 0` the task runs in a `multiprocessing.Process` with a queue; the parent kills the child on timeout. Set `task_timeout_seconds <= 0` to run in-process. **Side effect:** under timeout mode each task constructs its own `OpenAIModelAdapter` and `ToolRegistry` inside the child; you cannot pass `model=` / `tools=` overrides. The runner detects this and falls back to single-process mode (and forces `max_workers=1`) when overrides are supplied — useful for tests with `ScriptedModelAdapter`.
3. **Parallelism** — `run_benchmark` uses `ThreadPoolExecutor(max_workers=...)` for `>1` workers (each worker spawns its own subprocess for the task), and a serial loop with shared model/tools when overrides are provided or `max_workers == 1`. `--task-set <file>` (or `task_filter` kwarg) restricts the run to a list of task ids — used for holdout-only iteration.
4. **Multi-pass orchestrator (H-1)** — `run_benchmark_with_passes` runs the same task set N times when `repeat_max > 1` and cross-run votes. Per-pass output goes to `output_dir/_runs/run_<i>/`, and at the end `cross_run_vote` writes the voted final into `output_dir/task_<id>/`. The `pass_safety_margin × last_pass_duration` budget guard ensures regression safety; on voter failure `_fallback_copy_pass` copies pass 0.
5. **Runtime log** (`RuntimeLogger`) — when `config.run.log_file` is set, the runner appends JSON-Lines events (`benchmark_start`, `task_done`, `benchmark_end`, plus 5 multi-pass events) to that path. The eval rules require logs at `/logs/runtime.log`; `configs/eval.yaml` wires that automatically.

`execute_python` (`tools/python_exec.py`) is *also* subprocess-isolated with a hard 30s timeout (`EXECUTE_PYTHON_TIMEOUT_SECONDS` in `tools/registry.py`), `chdir`'d into `task.context_dir`, with stdout/stderr captured at the file-descriptor level. Nested multiprocessing: when `task_timeout_seconds > 0` *and* the agent calls `execute_python`, you get a process-in-a-process. Avoid running benchmarks under a debugger that doesn't tolerate that.

`**scoring/`** — local mirror of the official DataAgent-Bench scorer. The competition rule is `Score = Recall − λ·(ExtraCols/PredictedCols)` with column-signature matching ignoring column names and row order, after normalization (numerics → 2 decimals, dates → ISO 8601, nulls → `""`, strings → trimmed case-sensitive).

- `scoring/normalize.py` — `normalize_value`, `normalize_column`, `normalize_answer_table`, `column_signature`. Auto-detects per-cell type when no policy is given. Wired into `_answer` to produce a normalized variant alongside the raw one.
- `scoring/mock_scorer.py` — given `predictions/` and `gold/` roots, computes per-task `score`, `recall`, `matched/gold/predicted` columns. CLI supports `--lambda-values` for λ-sensitivity sweeps. The leaderboard λ is undisclosed; we default to 0.10 and gate column-ablation on agreement across `{0.05, 0.10, 0.20}`.
- `scoring/cross_run_vote.py` (H-1) — column-multiset majority voter across N benchmark output roots. CLI + library. `vote_across_runs(prediction_roots, output_dir)` enables host-side ensemble measurement (same algorithm the in-runner orchestrator uses inside the container).
- `scoring/holdout.py` — deterministic 80/20 hash-based split (`blake2b(salt + task_id)`), produces `train_ids.txt` / `holdout_ids.txt`. The holdout is the only honest signal we have for prompt iteration; do not let it leak into prompt design.

**Submission flow.** `Dockerfile` + `configs/eval.yaml` + `scripts/build_submission.sh` produce `submissions/dabench_<version>.tar.gz`. The build script aborts if the tarball exceeds the 10 GB cap. `scripts/local_eval.sh` runs the built image against `data/public/input/` with a self-hosted vLLM on `host.docker.internal:8000` and pipes the predictions through the mock scorer.

## Dataset layout

```
data/public/input/task_<id>/
  task.json    # {task_id, difficulty, question} — exact key set, no extras
  context/     # CSV / JSON / SQLite / text files; all tool paths relative to here
data/public/output/task_<id>/gold.csv  # public demo ground truth (separate tree)
```

Hidden test sets ship `input/` only — never write code that depends on `output/` existing.

## Run outputs

```
artifacts/runs/<run_id>/
  <task_id>/
    trace.json        # full step history + answer + failure_reason
    prediction.csv    # only present when the agent called the answer tool
  summary.json        # benchmark runs only
```

`run_id` defaults to a UTC timestamp `YYYYMMDDTHHMMSSZ`. The whole `artifacts/` tree is gitignored except for `artifacts/.gitkeep`.

## Things that are gitignored (so don't expect them in the repo)

`data/`, `tests/`, `evaluation/`, `artifacts/*` (except `artifacts/.gitkeep`), `submissions/`, `vllm_logs/`, most of `docs/*` (except the seven whitelisted docs × 3 language variants: ARCHITECTURE / SYSTEM_FLOW / SYSTEM_ARCHITECTURE / DATA_ANALYSIS / SUBMISSION_LOG / HARNESS_STRUCTURE / qwen_endpoint_capabilities), and most of `configs/*` (except `eval.yaml`, `local.yaml`). `.codex` is also ignored. When adding a new tracked file in any of those trees, also add a `!path/to/file` line in `.gitignore`.

## Plan & competition context

Key rules to remember:

- Eval-time LLM is locked to `qwen3.5-35b-a3b` via `MODEL_API_URL`/`MODEL_API_KEY`/`MODEL_NAME`; do not hardcode them.
- **Single-model policy (team1438):** the rules technically allow auxiliary models (embeddings, retrieval) within the hardware budget, but our team uses the qwen main solver only — no auxiliary LLM, sentence-transformers, FAISS, ONNX vision/embedding, or external web/vision API at any layer of the harness. `AgentConfig.model`/`api_base`/`api_key` defaults are empty strings so a missing env var fails loudly rather than silently calling an external provider.
- Compute envelope: 16 vCPU / 64 GB RAM / no GPU / **12 hours total** for all tasks. Network is blocked except the LLM endpoint.
- Submission cap: 1/day, 30/Phase-1. Track every submission in `[docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md)`.
- Mounts: `/input` (RO), `/output` (RW), `/logs` (RW). `configs/eval.yaml` is wired for those paths and `flat_output_dir: true`.

