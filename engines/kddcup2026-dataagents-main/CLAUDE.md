# CLAUDE.md

每次回复，称呼我为K神。

Project conventions for AI coding assistants (Claude Code, Codex, Cursor, etc.) and human contributors.

## Python execution: always use `uv run`

This project is managed by **uv** (`uv.lock`, `pyproject.toml`). All Python invocations must go through `uv run`.

```bash
# Do
uv run python -c "..."
uv run pytest
uv run ruff format src/

# Don't
.venv/bin/python ...   # bypasses uv sync state
python3 ...            # may hit a different interpreter
python ...             # same issue as python3
pip install ...        # use `uv add` or `uv sync --extra dev` instead
```

## Benchmark CLI: drive via `dabench`

```bash
uv run dabench status        --config configs/react.example.yaml
uv run dabench inspect-task <task_id> --config <cfg>
uv run dabench run-task     <task_id> --config <cfg>
uv run dabench run-benchmark         --config <cfg> [--limit N] [--range S-E]
```

Do not invoke via `uv run python -m agents.cli`.

## Code contribution rules

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — canonical source for commit messages, CHANGELOG policy, and contribution rules. It takes precedence over this file.

## KDD Cup 2026 constraints

- Heterogeneous data analysis: `context/` = any subset of `csv/`, `db/` (SQLite), `json/`, `doc/` (`.md`/`.pdf`), `video/` (MP4), `knowledge.md`; detect what exists. Not pure text-to-SQL.
- Phase-2 `task.json`: only `task_id` + `question` (no `difficulty`); questions may be Chinese.
- Inference-time only — no model training.
- Target model: `qwen3.5-35b-a3b` (judge-required model string; ~3B active params, supports video). See `scripts/eval_local.sh`.
- Online evaluation deploys via vLLM (`--tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --media-io-kwargs '{"video": {"num_frames": -1}}'`). Video sampling defaults `fps=2`, `do_sample_frames=True`; override `fps` via `extra_body`. Video input uses base64 data URI inline (no URL reference — local deployment has no external fetch path).
- Prefer workflow engines, tool use, multi-agent decomposition, and deterministic post-processing.
- Constrain tool I/O and use structured intermediate files rather than dumping large state into conversation history.

## Fix and optimization policy

- Fix root causes. No workarounds, threshold tweaks, skip flags, or symptom masking.
- Every optimization must generalize — no rules, guards, or heuristics that only fit one known task.
- Larger refactors are acceptable when they produce a more robust architecture.
- Do not validate changes exclusively on the task that motivated the fix.

## Evaluation

Run results are stored under `artifacts/runs/<run_id>/`, each task producing `<task_id>/prediction.csv`. Traces are written to `artifacts/traces.db` (SQLite). Score with:

```bash
uv run python scripts/score_predictions.py \
    --pred artifacts/runs/<run_id> \
    --gold <gold_dir> \
    --lambda 0.1 \
    --output artifacts/runs/<run_id>/scores.json \
    --verbose
```

- The grader ignores column names, column order, and row order.
- Each column is normalized (numbers quantized to 2 decimal places, dates standardized, nulls unified) then hashed.
- Columns are matched greedily across prediction and gold; extra predicted columns incur a penalty.

See `scripts/score_predictions.py` for details.

## Claims and evidence

Cite file:line or trace fields for mechanism-level claims. State correlation as correlation, not as causation. Re-check evidence when challenged.
