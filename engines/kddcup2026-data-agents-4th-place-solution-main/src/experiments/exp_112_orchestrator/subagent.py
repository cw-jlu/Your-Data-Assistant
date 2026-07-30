"""1-shot SQL / Python sub-agents.

Sub-agents are called as TOOLS by the orchestrator. They are intentionally
1-shot (= single LLM call → single execution) to keep latency contained.
The orchestrator does multi-step planning; specialists do focused execution.

SQL sub-agent:
- Input: sub_question, view catalog, knowledge.md slice
- Output: AnswerTable from one DuckDB SELECT

Python sub-agent:
- Input: sub_question, optional input_table (= rows the orchestrator pulled),
        knowledge.md slice
- Output: AnswerTable from one Python script
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_112_orchestrator.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
)


SQL_SPECIALIST_SYSTEM = """\
You are a SQL specialist. Generate exactly ONE DuckDB SQL statement that answers the
sub-question. Output format: a single ```sql fenced block with the query, no extra
text, no explanation.

Rules:
- Start with SELECT or WITH.
- Name every column explicitly. Never SELECT *.
- For superlative questions (lowest/highest/min/max), use WHERE col = (SELECT MIN/MAX(col) FROM ...) — never LIMIT 1.
- Apply ROUND() only at the outermost SELECT.
- Cross-file JOINs are allowed (= every CSV / JSON / sqlite-DB is exposed as a view).
- Read all caveats in the knowledge.md slice carefully — formula scope and column-name
  ambiguity are common pitfalls.
"""


PYTHON_SPECIALIST_SYSTEM = """\
You are a Python specialist. Generate exactly ONE Python snippet that answers the
sub-question. Output format: a single ```python fenced block, no extra text.

Rules:
- Set `answer_df` (a pandas DataFrame) at the end. Include only the columns the
  question requires — no extra metadata columns.
- The snippet runs with the task's context dir as cwd. Files are at
  `csv/...`, `json/...`, `db/...` paths.
- Pre-loaded helpers: `pd` (pandas), `pl` (polars optional), `sqlite3`. You can
  also use `duckdb.connect(':memory:')` if you prefer.
- For superlative questions, use filter-back not nlargest(1) — preserves ties.
- Read all caveats in the knowledge.md slice carefully.
"""


@dataclass(frozen=True, slots=True)
class SubAgentResult:
    ok: bool
    columns: list[str]
    rows: list[list[Any]]
    code: str
    error: str | None = None
    raw_response: str | None = None


def _extract_fenced(text: str, lang: str) -> str:
    m = re.search(rf"```{lang}\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: any fenced block
    m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]"


def _knowledge_slice(task: PublicTask, max_chars: int = 6000) -> str:
    p = task.context_dir / "knowledge.md"
    if not p.exists():
        return "(no knowledge.md)"
    return _truncate(p.read_text(encoding="utf-8", errors="replace"), max_chars)


def call_sql_specialist(
    *,
    task: PublicTask,
    sub_question: str,
    model: ModelAdapter,
) -> SubAgentResult:
    catalog = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)
    user_prompt = (
        f"Sub-question: {sub_question}\n\n"
        f"## Available DuckDB views\n{catalog}\n\n"
        f"## knowledge.md (= caveats, definitions)\n{knowledge}\n\n"
        "Output ONE ```sql block."
    )
    messages = [
        ModelMessage(role="system", content=SQL_SPECIALIST_SYSTEM),
        ModelMessage(role="user", content=user_prompt),
    ]
    raw = model.complete(messages)
    sql = _extract_fenced(raw, "sql")
    try:
        result = execute_sql(task.context_dir, sql, limit=10000)
    except Exception as exc:
        return SubAgentResult(
            ok=False, columns=[], rows=[], code=sql, error=f"{type(exc).__name__}: {exc}", raw_response=raw,
        )
    columns = [str(c) for c in (result.get("columns") or [])]
    rows = result.get("rows") or []
    if not columns:
        return SubAgentResult(ok=False, columns=[], rows=[], code=sql, error="SQL returned no columns.", raw_response=raw)
    return SubAgentResult(ok=True, columns=columns, rows=rows, code=sql, raw_response=raw)


def call_python_specialist(
    *,
    task: PublicTask,
    sub_question: str,
    model: ModelAdapter,
    input_table: dict[str, Any] | None = None,
) -> SubAgentResult:
    catalog = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)
    input_section = ""
    if input_table:
        cols = input_table.get("columns", [])
        rows = input_table.get("rows", [])
        input_section = (
            f"\n## Input table (= passed by orchestrator, available as `input_df` in your snippet)\n"
            f"columns: {cols}\nrows ({len(rows)}): {_truncate(repr(rows[:50]), 2000)}\n"
        )
    user_prompt = (
        f"Sub-question: {sub_question}\n\n"
        f"## Data files (also exposed as DuckDB views; you can use either)\n{catalog}\n"
        f"{input_section}\n"
        f"## knowledge.md (= caveats)\n{knowledge}\n\n"
        "Output ONE ```python block. Set `answer_df` at the end."
    )
    messages = [
        ModelMessage(role="system", content=PYTHON_SPECIALIST_SYSTEM),
        ModelMessage(role="user", content=user_prompt),
    ]
    raw = model.complete(messages)
    code = _extract_fenced(raw, "python")
    return _execute_python_snippet(task, code, raw, input_table=input_table)


def _execute_python_snippet(
    task: PublicTask,
    code: str,
    raw: str,
    *,
    input_table: dict[str, Any] | None = None,
) -> SubAgentResult:
    """Run code in a subprocess with cwd=task.context_dir, return answer_df rows."""
    import subprocess
    import sys
    import tempfile
    import json as _json

    pre = """
import pandas as pd
try:
    import polars as pl
except Exception:
    pl = None
import sqlite3
import json as _json
"""
    if input_table is not None:
        pre += (
            f"\ninput_df = pd.DataFrame({_json.dumps(input_table.get('rows', []))}, "
            f"columns={_json.dumps(input_table.get('columns', []))})\n"
        )
    post = """
import json as _json
import sys as _sys
import pandas as _pd
if 'answer_df' in dir() and isinstance(answer_df, _pd.DataFrame):
    out = {'columns': [str(c) for c in answer_df.columns], 'rows': [[(None if _pd.isna(v) else v) for v in r] for r in answer_df.to_records(index=False).tolist()]}
elif 'answer_table' in dir() and isinstance(answer_table, dict):
    out = answer_table
else:
    out = {'__error__': 'No answer_df or answer_table set'}
print('___ANSWER_BEGIN___')
print(_json.dumps(out, default=str, ensure_ascii=False))
print('___ANSWER_END___')
"""
    full_code = pre + "\n" + code + "\n" + post
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        script_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=str(task.context_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return SubAgentResult(ok=False, columns=[], rows=[], code=code, error="timeout (60s)", raw_response=raw)
    finally:
        try:
            Path(script_path).unlink()
        except Exception:
            pass

    out = proc.stdout
    err = proc.stderr
    m = re.search(r"___ANSWER_BEGIN___\n(.+?)\n___ANSWER_END___", out, flags=re.DOTALL)
    if not m:
        return SubAgentResult(
            ok=False, columns=[], rows=[], code=code,
            error=f"No answer extracted. stderr:\n{err[:1000]}\nstdout tail:\n{out[-500:]}",
            raw_response=raw,
        )
    try:
        parsed = _json.loads(m.group(1))
    except Exception as exc:
        return SubAgentResult(
            ok=False, columns=[], rows=[], code=code,
            error=f"JSON parse error: {exc}", raw_response=raw,
        )
    if "__error__" in parsed:
        return SubAgentResult(
            ok=False, columns=[], rows=[], code=code,
            error=parsed["__error__"], raw_response=raw,
        )
    columns = [str(c) for c in parsed.get("columns") or []]
    rows = [list(r) for r in parsed.get("rows") or []]
    if not columns:
        return SubAgentResult(ok=False, columns=[], rows=[], code=code, error="empty columns", raw_response=raw)
    return SubAgentResult(ok=True, columns=columns, rows=rows, code=code, raw_response=raw)
