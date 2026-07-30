# Development

## Toolchain

The project is managed by [uv](https://docs.astral.sh/uv/). All Python
invocations must go through `uv run` so the locked environment is used.

```bash
# Install runtime + dev dependencies
uv sync --extra dev

# Run the CLI / scripts
uv run dabench status --config configs/react.example.yaml
uv run pytest
uv run ruff format src/
uv run pyright
```

Avoid invoking `python`, `python3`, or `.venv/bin/python` directly — they may bypass
`uv`'s sync state.

## Quality gates

The repo ships four exit-code-clean gates. Each is independent and idempotent.

| Gate | Command | Contract |
| --- | --- | --- |
| Format | `uv run ruff format --check .` | Exit `0`; reports zero files would be reformatted. |
| Lint | `uv run ruff check .` | Exit `0`; `All checks passed`. |
| Types | `uv run pyright` | Exit `0`. Strict mode on `src/agents`, basic on `tests/`. |
| Tests | `uv run pytest` | Exit `0`. Coverage ≥ 65% (gate fails below). |

### Pre-commit (opt-in)

A pre-commit chain is shipped at `.pre-commit-config.yaml` running
`ruff format` → `ruff check --fix` → `pyright src` on changed files. It is opt-in:

```bash
uv run pre-commit install      # one-time, per clone
uv run pre-commit run --all-files
```

GitHub Actions runs these gates on pull requests and pushes to `main`; local runs use the
same commands.

## Project layout

```
src/agents/
├── __init__.py                  # Public ReAct/model exports
├── cli.py                       # Typer entrypoint (dabench)
├── config.py                    # Frozen config dataclasses + Pydantic validation
├── agent.py                     # ReAct loop driver
├── runtime/                     # Run state, message/media helpers, budget, recording
│   ├── state.py                 # StepRecord, AgentRuntimeState, AgentRunResult
│   ├── messages.py              # System/user/replay message construction + trace serialization
│   ├── media.py                 # Task-local video attachment
│   ├── budget.py                # Budget warning prompts + final-step guardrail
│   └── recorder.py              # Step events + step_callback
├── llm/                         # Model adapters, token accounting, rate limits
│   ├── types.py                 # Provider-neutral model protocol and response types
│   ├── openai.py                # OpenAI-compatible adapters (native function calling)
│   ├── protocol.py              # ModelResponse → ToolCall normalization
│   ├── rate_limit.py            # RateLimitedAdapter
│   ├── token_bucket.py          # Cross-process token bucket
│   └── tokenizer.py             # Qwen tokenizer loader
├── prompts/                     # System, task, observation, and recovery prompts
├── tools/                       # ToolDefinition, dispatcher, handlers, filesystem previews
├── verification/                # Answer self-check + terminal-answer fallback policy
├── benchmark/                   # DABenchPublicDataset and task schemas
├── application.py               # AgentApp factory
├── runs/                        # Runner, subprocess isolation, artifact writing
├── tracing/                     # Trace/span capture and SQLite store
├── dashboard/                   # FastAPI traces dashboard backend
└── etl/                         # Optional prose-to-CSV ETL helpers
```

The dashboard frontend source lives outside the Python package under
`dashboard/frontend/`. Its Vite build writes ignored static assets into
`src/agents/dashboard/static/` for `dabench dashboard` to serve.

Tests live under `tests/`, with shared fixtures in `tests/helpers/` (notably
`scripted_adapters.py`).

## Contributing rules

Commit-message format and `CHANGELOG.md` maintenance live in their own document:
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Repository-specific developer rules below.

- **Don't bypass `uv`.** `pip install`, raw `python`, and `.venv/bin/python` are out.
- **Keep wire formats stable.** `summary.json`, `prediction.csv`, optional
  `traces.db`, and `bucket_<sha1>.json` are runtime contracts. Any intentional format
  change needs focused tests and a `CHANGELOG.md` entry when users can observe it.
- **Add tests for new code paths.** Coverage gate is 65%; treat it as a floor.
