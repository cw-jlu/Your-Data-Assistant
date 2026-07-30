# Agent protocol

The agent uses OpenAI native function calling to get tool calls out of an LLM.
vLLM with `--enable-auto-tool-choice --tool-call-parser qwen3_coder` works.

Multi-call per turn is supported. JSON arguments are validated by the SDK.

The model adapters strip any `<think>...</think>` block from
`message.content` before parsing.

## Default tool inventory

Registration order is fixed; `to_openai_tools()` sorts by name on output for cache
stability.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `inspect_files` | _(none)_ | One-shot summary of every file under `context/`: columns, dtypes, row counts, table schemas, and PDF `page_count` / `paragraph_count` / `entity_groups_sample`. |
| `preview_file` | `path` | Single-file preview by extension: CSV head/tail, JSON structure, SQLite CREATE TABLE statements with row counts, PDF entity group samples, or text head. |
| `execute_context_sql` | `path`, `sql`, `limit` | SQL against a SQLite/DB file in virtual `context/`: `SELECT`/`WITH`/`PRAGMA`, `EXPLAIN QUERY PLAN`, and local `CREATE INDEX IF NOT EXISTS` for discovery. |
| `execute_python` | `code` | Run arbitrary Python with cwd pinned to the task `context/`. |
| `answer` | `columns`, `rows`, `from_csv` | Submit the final answer table inline or from a CSV artifact. **Terminal**: halts the loop on success. |

All `path` fields are resolved relative to the task's `context_dir`. Path traversal,
absolute paths, and symlinks escaping the sandbox are rejected.

## Loop control

The driver bounds these counters:

- `max_steps` — total recorded steps before the loop halts with a
  `failure_reason` mentioning `max_steps`.
- `max_empty_tool_call_retries` — total empty-call turns tolerated. Halt
  fires when `empty_tool_call_retries >= max_empty_tool_call_retries`
  (i.e. with `N=1`, the first empty turn is the failure turn).

Model request exceptions are recorded as `model_error` steps and consume the normal
`max_steps` budget.

Internal event kinds (`model_error`, `parse_error`, `empty_tool_calls`,
`tool_error`, `observation`) are an internal `Literal`; each recorded error
step serializes as `action="__error__"` plus `error_payload` so downstream
tracing and tests see one stable shape.

## Step Records

`AgentRunResult.to_dict()` returns the in-memory task result shape consumed by the
runner:

```json
{
  "task_id": "task_1",
  "succeeded": true,
  "failure_reason": null,
  "steps": [...],
  "answer": { "columns": [...], "rows": [...] },
  "elapsed_seconds": 12.3,
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801,
    "cached_tokens": 0,
    "reasoning_tokens": 12
  }
}
```

Each `steps` entry:

```json
{
  "step_index": 0,
  "turn_index": 0,
  "thought": "...",
  "action": "preview_file",
  "action_input": { "path": "data.csv" },
  "observation": { "ok": true, "content": {...} },
  "error_payload": null,
  "usage": { ... },
  "latency_ms": 412
}
```

Error steps have `action == "__error__"` and a populated `error_payload`:

```json
{
  "step_index": 3,
  "turn_index": 3,
  "action": "__error__",
  "error_payload": { "kind": "tool_error", "tool": "preview_file", "error": "..." }
}
```

## `summary.json`

```json
{
  "run_id": "20260430-001",
  "task_count": 5,
  "succeeded_task_count": 4,
  "max_workers": 4,
  "tasks": [
    {
      "task_id": "task_1",
      "task_output_dir": "artifacts/runs/.../task_1",
      "prediction_csv_path": "artifacts/runs/.../task_1/prediction.csv",
      "succeeded": true,
      "failure_reason": null
    }
  ]
}
```

The `succeeded` flag is purely "did the agent call `answer`?" — it does not assert that
the answer is **correct** against any ground truth.

Debug tracing is stored separately from run artifacts. Set `tracing.enabled: true` in
the config to write SQLite spans to `tracing.db_path` for `dabench dashboard`.
