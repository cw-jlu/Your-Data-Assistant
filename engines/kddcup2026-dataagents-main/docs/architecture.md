# Architecture

`agents` is a single-process ReAct loop that drives an OpenAI-compatible LLM
through a fixed set of tools, scoped to a per-task `context/` directory. The runner wraps
each task in a subprocess so wall-clock timeouts and crashes can be cleanly bounded.

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLI (Typer)                                                           │
│   dabench {status, inspect-task, run-task, run-benchmark, dashboard}   │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │   AppConfig     │   Pydantic-validated YAML
                └────────┬────────┘
                         │
              ┌──────────▼──────────┐
              │  build_application  │   src/agents/application.py
              └──────────┬──────────┘
                         │  AgentApp{ dataset, registry, model_adapter,
                         │            agent_config, protocol }
              ┌──────────▼──────────┐
              │     ReActAgent      │   src/agents/agent.py
              │  ┌───────┐ ┌──────┐ │
              │  │protoco│→│dispat│ │
              │  │l adapt│ │ cher │ │
              │  └───┬───┘ └──┬───┘ │
              │      └────┬───┘     │
              │      ┌────▼────┐    │
              │      │recorder │    │
              │      └────┬────┘    │
              └───────────┼─────────┘
                          ▼
                 prediction.csv + summary.json
                 optional traces.db
```

## Live capabilities

These are the current implementation anchors for the main runtime surfaces.

| Capability | Implementation | What lives there |
| --- | --- | --- |
| ReAct loop | `src/agents/agent.py` | Bounded loop, model-call state machine, tool dispatch, and step callbacks. |
| Runtime support | `src/agents/runtime/` | **Agent runtime** (not infrastructure runtime): run state, step records, message replay, media attachment, budget guardrails, step events, and callback notification. Deliberately depends upward on `prompts/` and `tools/` because it assembles agent-facing messages from both. |
| Protocol adapter | `src/agents/llm/protocol.py` | Native function-calling responses normalized into a common turn shape. |
| Model types | `src/agents/llm/types.py` | Provider-neutral message, response, tool-call, usage, and adapter protocol types shared by runtime, tracing, verification, and tests. |
| Model I/O | `src/agents/llm/openai.py` | OpenAI-compatible `chat.completions` adapters, Qwen3 `<think>` stripping, usage capture, latency capture, and native tool-call parsing. |
| Tool inventory | `src/agents/tools/registry.py` plus `src/agents/tools/dispatcher.py` | `ToolDefinition`, immutable `ToolRegistry`, default tool inventory, Pydantic input validation, OpenAI tool schema export, and tool-call dispatch. |
| Answer verification | `src/agents/verification/` | Optional no-tools answer-table verifier plus terminal-answer rejection/fallback policy. |
| Application factory | `src/agents/application.py` | Wires dataset, registry, model adapter, verifier, and loop config into `AgentApp`. |
| Subprocess runner | `src/agents/runs/runner.py` plus `runs/subprocess.py` and `runs/artifacts.py` | Per-task subprocess timeout isolation, deterministic key pinning, task artifacts, and `summary.json`. |
| SQLite tracing | `src/agents/tracing/`, `src/agents/dashboard/`, and `dashboard/frontend/` | Opt-in span capture to `tracing.db_path`, FastAPI dashboard routes, and the Vite frontend that reads them. |
| Rate limiting | `src/agents/llm/rate_limit.py` plus `src/agents/llm/token_bucket.py` | Cross-process RPM/TPM token buckets with `filelock`, atomic state writes, and Retry-After handling. |

## Native function calling

The agent uses OpenAI native function calling exclusively. Response normalization lives
in `agents/llm/protocol.py:adapt_model_response`: the model emits zero-or-more
`tool_calls` in `choices[0].message`, and the dispatcher iterates them in order; the
first terminal-and-successful call halts the loop. Turns replay as assistant/tool
messages with tool-call IDs.

## Subprocess isolation

`run_single_task` always runs the agent in a child process when no test injection is
active. The parent posts the `AppConfig`; the child calls `build_application` itself
(the `AgentApp` is intentionally not pickled). On timeout or crash, the parent returns a
synthetic failure payload. When `tracing.enabled: true`, the child also writes spans to
the configured SQLite database for dashboard inspection.

## Key pool + cross-process rate limiter

When `agent.api_keys` is set, task index `i` pins to `api_keys[i % len(api_keys)]` for the
entire run. The `RateLimitedAdapter` then guards each request with a `FileTokenBucket`
backed by a `filelock`-protected JSON file under `<output_dir>/../ratelimit/<run_id>/`.
Atomic state writes (write-temp + `os.replace`) prevent torn reads across worker
processes.

## Wire-format contract

The following outputs are treated as stable runtime artifacts and are covered by focused
unit and integration tests:

- `summary.json` — per-run aggregate (run id, task count, succeeded count, per-task entries)
- `prediction.csv` — answer table
- `traces.db` — optional SQLite tracing store when `tracing.enabled: true`
- `ratelimit/bucket_<sha1>.json` — cross-process bucket state
