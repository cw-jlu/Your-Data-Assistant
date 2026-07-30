"""Phase 2: self_refine loop (= ReFoRCE's main answer-generation algorithm).

Closely mirrors `self_refine()` in references/ReFoRCE/methods/ReFoRCE/agent.py:169-265:

  1. Build a chat session with the self_refine prompt.
  2. Generate ONE SQL.
  3. Execute it.
  4. If error → next prompt = "error: ... please correct".
  5. If empty → next prompt = "simplify some conditions".
  6. If has rows:
     - Compute result signature (= sorted normalized first-column values, ReFoRCE
       compares value-only).
     - If signature in past_results → SELF-CONSISTENT → COMMIT.
     - Else → push signature, next prompt = self_consistency_prompt.
  7. Loop until max_iter or commit.
  8. Early stop: if last 4 results are all empty → abandon.

Returns the committed result, or None if no consistent answer emerged.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import pandas as pd

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_116_reforce_loop.tools.duckdb_unified import execute_sql, describe_catalog
from experiments.exp_116_reforce_loop.prompts import (
    SELF_REFINE_SYSTEM,
    build_self_refine_prompt,
    build_self_consistency_prompt,
    build_error_correct_prompt,
    build_empty_simplify_prompt,
)
from experiments.exp_116_reforce_loop.exploration import _knowledge_slice
import re


@dataclass(frozen=True, slots=True)
class SelfRefineResult:
    ok: bool
    columns: list[str]
    rows: list[list[Any]]
    winning_sql: str
    n_iterations: int
    commit_reason: str  # "self_consistent" | "max_iter_exhausted" | "early_stop_empty" | "all_failed"
    history: list[dict[str, Any]]  # per-iter SQL + result summary


def _extract_sql(raw: str) -> str:
    m = re.search(r"```sql\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _normalize_value(v: Any) -> tuple:
    """Normalize to a sort-safe (kind_tag, payload) tuple.

    Mixing str / float / None during a sort raises TypeError in Python 3.
    Use a tag-prefixed tuple so all comparisons stay within homogeneous
    types: (0, "") for None, (1, float) for numerics, (2, str) for strings.
    """
    if v is None:
        return (0, "")
    if isinstance(v, (int, float)):
        try:
            return (1, round(float(v), 3))
        except Exception:
            return (2, str(v).strip().lower())
    s = str(v).strip()
    try:
        return (1, round(float(s), 3))
    except Exception:
        return (2, s.lower())


def _result_signature(columns: list[str], rows: list[list[Any]]) -> tuple:
    """Order-insensitive signature for vote / consistency check."""
    if not columns:
        return ("__empty__",)
    n_cols = len(columns)
    norm = []
    for r in rows:
        if len(r) != n_cols:
            continue
        norm.append(tuple(_normalize_value(v) for v in r))
    norm.sort()
    return (n_cols, len(norm), tuple(norm))


def _csv_preview(columns: list[str], rows: list[list[Any]], max_rows: int = 10, max_chars: int = 1500) -> str:
    if not columns:
        return "(empty)"
    df = pd.DataFrame(rows, columns=columns)
    text = df.head(max_rows).to_csv(index=False)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return text


def _detect_anomalies(columns: list[str], rows: list[list[Any]]) -> tuple[list, list]:
    """Detect nested values + entirely-empty/zero columns (= ReFoRCE agent.py:225-241)."""
    if not columns or not rows:
        return [], []
    df = pd.DataFrame(rows, columns=columns).fillna("")
    nested = [
        v for r in df.values.tolist() for v in r
        if isinstance(v, str) and "\n" in v
    ]
    df_str = df.astype(str)
    empty_cols = df_str.columns[((df_str == "0") | (df_str == "")).all()].tolist()
    return nested[:3], empty_cols


def run_self_refine(
    *,
    task: PublicTask,
    model: ModelAdapter,
    exploration_findings: str,
    max_iter: int = 5,
    early_stop_empty: int = 4,
) -> SelfRefineResult:
    catalog = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)

    initial_user_prompt = build_self_refine_prompt(
        task=task, catalog=catalog, knowledge=knowledge,
        exploration_findings=exploration_findings,
    )
    messages: list[ModelMessage] = [
        ModelMessage(role="system", content=SELF_REFINE_SYSTEM),
        ModelMessage(role="user", content=initial_user_prompt),
    ]

    past_signatures: list[tuple] = []
    past_sql_for_sig: dict[tuple, str] = {}
    past_columns_for_sig: dict[tuple, list[str]] = {}
    past_rows_for_sig: dict[tuple, list[list[Any]]] = {}

    error_record: list[str] = []  # for early_stop tracking
    history: list[dict[str, Any]] = []

    for itercount in range(max_iter):
        try:
            raw = model.complete(messages)
        except Exception as exc:
            history.append({"iter": itercount, "error": f"LLM call failed: {exc}"})
            return SelfRefineResult(
                ok=False, columns=[], rows=[], winning_sql="",
                n_iterations=itercount, commit_reason="all_failed",
                history=history,
            )
        sql = _extract_sql(raw)
        if not sql or not sql.lstrip().lower().startswith(("select", "with")):
            messages.append(ModelMessage(role="assistant", content=raw))
            messages.append(ModelMessage(
                role="user",
                content="Please output exactly one complete SQL query in ```sql``` fenced format. Start with SELECT or WITH.",
            ))
            history.append({"iter": itercount, "sql": sql, "error": "no SELECT/WITH"})
            error_record.append("no_sql")
            continue

        try:
            result = execute_sql(task.context_dir, sql, limit=10000)
            err: str | None = None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"

        messages.append(ModelMessage(role="assistant", content=raw))
        if err is not None:
            history.append({"iter": itercount, "sql": sql, "error": err})
            error_record.append("exec_error")
            messages.append(ModelMessage(
                role="user",
                content=build_error_correct_prompt(prior_sql=sql, error=err),
            ))
            continue

        columns = [str(c) for c in (result.get("columns") or [])]
        rows = [list(r) for r in (result.get("rows") or [])]
        n_rows = len(rows)
        history.append({"iter": itercount, "sql": sql, "columns": columns, "n_rows": n_rows})

        if n_rows == 0:
            error_record.append("empty")
            # ReFoRCE early stop: 4 consecutive empties → abandon
            if early_stop_empty > 0 and len(error_record) >= early_stop_empty:
                if all(e == "empty" for e in error_record[-early_stop_empty:]):
                    return SelfRefineResult(
                        ok=False, columns=columns, rows=[], winning_sql=sql,
                        n_iterations=itercount + 1, commit_reason="early_stop_empty",
                        history=history,
                    )
            messages.append(ModelMessage(
                role="user",
                content=build_empty_simplify_prompt(prior_sql=sql),
            ))
            continue

        # Non-empty result — consistency check
        error_record.append("ok")
        sig = _result_signature(columns, rows)
        if sig in past_signatures:
            # SELF-CONSISTENT → commit
            return SelfRefineResult(
                ok=True,
                columns=past_columns_for_sig[sig],
                rows=past_rows_for_sig[sig],
                winning_sql=past_sql_for_sig[sig],
                n_iterations=itercount + 1,
                commit_reason="self_consistent",
                history=history,
            )
        past_signatures.append(sig)
        past_sql_for_sig[sig] = sql
        past_columns_for_sig[sig] = columns
        past_rows_for_sig[sig] = rows

        # Build self_consistency verify prompt for next iteration
        nested, empty_cols = _detect_anomalies(columns, rows)
        verify_prompt = build_self_consistency_prompt(
            task=task,
            prior_sql=sql,
            prior_csv_preview=_csv_preview(columns, rows),
            prior_columns=columns,
            prior_n_rows=n_rows,
            nested_values=nested,
            empty_columns=empty_cols,
        )
        messages.append(ModelMessage(role="user", content=verify_prompt))

    # max_iter exhausted — fall back to first non-empty result if any
    if past_signatures:
        first_sig = past_signatures[0]
        return SelfRefineResult(
            ok=True,
            columns=past_columns_for_sig[first_sig],
            rows=past_rows_for_sig[first_sig],
            winning_sql=past_sql_for_sig[first_sig],
            n_iterations=max_iter,
            commit_reason="max_iter_exhausted",
            history=history,
        )

    return SelfRefineResult(
        ok=False, columns=[], rows=[], winning_sql="",
        n_iterations=max_iter, commit_reason="all_failed",
        history=history,
    )
