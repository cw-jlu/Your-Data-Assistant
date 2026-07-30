# CLI

The `dabench` command is the only entrypoint. It is registered via
`pyproject.toml`'s `[project.scripts]` and must be invoked through `uv run`:

```bash
uv run dabench <subcommand> --config <path/to/config.yaml> [options]
```

> Do **not** invoke as `python -m agents.cli` — Typer + the module runner
> produces no output in this project. The supported executable entrypoint is the
> `dabench` script declared in `pyproject.toml`.

## Subcommands

`dabench` exposes five subcommands.

| Command | Purpose |
| --- | --- |
| `status` | Print project paths, dataset root, and public task counts. |
| `inspect-task <task_id>` | Show task metadata and the file tree under `context/`. |
| `run-task <task_id>` | Run the agent on a single task and write outputs. |
| `run-benchmark` | Run the agent across the dataset (optionally `--limit N` and/or `--range S-E`). |
| `dashboard` | Launch the SQLite traces dashboard for runs created with tracing enabled. |

The runtime commands (`status`, `inspect-task`, `run-task`, `run-benchmark`) require
`--config <path>`. The YAML is validated by Pydantic v2 on entry; a schema violation
aborts with non-zero exit before any task runs. `dashboard` reads an existing SQLite
trace database and accepts `--db` instead of `--config`.

## Examples

```bash
# Smoke-check the dataset and config wiring
uv run dabench status --config configs/react.example.yaml

# Look at one task without running the agent
uv run dabench inspect-task task_1 --config configs/react.local.yaml

# Run one task; outputs land under <output_dir>/<run_id>/task_1/
uv run dabench run-task task_1 --config configs/react.local.yaml

# Run the first 5 tasks
uv run dabench run-benchmark --limit 5 --config configs/react.local.yaml

# Run a contiguous task subset (inclusive on both ends)
uv run dabench run-benchmark --range 5-10 --config configs/react.local.yaml
uv run dabench run-benchmark --range task_5-task_10 --config configs/react.local.yaml

# Combine: range filters first, then --limit truncates → runs task_2..task_4
uv run dabench run-benchmark --range 2-8 --limit 3 --config configs/react.local.yaml

# View traces from runs whose config enabled SQLite tracing
uv run dabench dashboard --db artifacts/traces.db
```

## Config schema

The supported YAML fields are validated by the frozen Pydantic dataclasses in
`src/agents/config.py`; the example below mirrors
`configs/react.example.yaml`.

```yaml
dataset:
  root_path: data/public/input            # relative paths anchored at PROJECT_ROOT

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY                   # ignored when api_keys is set
  max_steps: 16
  temperature: 0.0
  allow_parallel_tool_calls: true
  max_tokens: 16384                       # 0 = use endpoint default
  enable_thinking: true                   # Qwen3 extra_body passthrough; null = omit

  # Optional dual-key pool + cross-process rate limiter.
  # Both blocks must be present together or both absent.
  # api_keys:
  #   - sk-key-A
  #   - sk-key-B
  # rate_limit:
  #   rpm_per_key: 600
  #   tpm_per_key: 1000000
  #   safety_factor: 0.85
  #   reserve_output_tokens: 16384
  #   max_retries: 6
  #   backoff_initial_seconds: 1.0
  #   backoff_max_seconds: 30.0
  #   # state_dir: artifacts/ratelimit/<run_id>/  # default

run:
  output_dir: artifacts/runs
  run_id:                                 # null = auto `<UTC YYYYMMDD>-<NNN>`, e.g. 20260430-001
  max_workers: 8
  task_timeout_seconds: 600               # 0 or negative = disable

# Optional debug tracing. Omit the block, or leave enabled false, for normal benchmark runs.
# tracing:
#   enabled: true
#   db_path: artifacts/traces.db
```

### Validation behavior

- Strict bool/int coercion. Quoted YAML scalars like `"false"` or `"16384"` are rejected.
  Use unquoted scalars (`false`, `16384`).
- `api_keys` and `rate_limit` are paired: setting one without the other is rejected.
- `agent.rate_limit.reserve_output_tokens` must be ≤ `agent.max_tokens` (when both are set).
- `tracing` is optional and defaults to disabled. The dashboard only shows runs that wrote
  spans to `tracing.db_path`.
- All path fields resolve relative to the project root.

## Outputs

```
<output_dir>/<run_id>/
├── summary.json
└── <task_id>/
    └── prediction.csv          # absent if the agent never called `answer`
```

`summary.json` is the canonical machine-readable run output. CLI progress text is
informational only.

Debug traces are not written into the run directory. Enable `tracing.enabled: true` to
write spans to SQLite (`artifacts/traces.db` by default), then open them with
`dabench dashboard`.

## Exit codes

- `0` on success
- non-zero on missing config, missing task, validation error, or unrecoverable runtime error
