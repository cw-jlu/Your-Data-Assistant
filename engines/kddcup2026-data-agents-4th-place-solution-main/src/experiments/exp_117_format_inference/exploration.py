"""Phase 1: column exploration (= ReFoRCE pillar (d)).

Generate 3-8 short DuckDB SELECT queries to understand column values BEFORE
the answer SQL. Results are formatted as 'few-shot examples' to inject into
the self_refine prompt.

Mirrors ReFoRCE's `exploration()` in agent.py:129-167.
"""
from __future__ import annotations

import json as _json
import re
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_117_format_inference.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
)
from experiments.exp_117_format_inference.prompts import (
    EXPLORATION_SYSTEM,
    build_exploration_user_prompt,
)


def _knowledge_slice(task: PublicTask, max_chars: int = 4000) -> str:
    p = task.context_dir / "knowledge.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n[...truncated]" if len(text) > max_chars else "")


def _extract_queries(raw: str) -> list[str]:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    payload = m.group(1).strip() if m else raw.strip()
    try:
        arr = _json.loads(payload)
        if isinstance(arr, list):
            return [str(q).strip() for q in arr if isinstance(q, str) and q.strip()]
    except Exception:
        pass
    candidates = []
    for line in payload.split("\n"):
        s = line.strip().rstrip(",").strip().strip('"\'')
        if s.lower().startswith(("select", "with")):
            candidates.append(s)
    return candidates


def _format_findings(query_results: list[dict[str, Any]]) -> str:
    """Format as 'Query / Answer' pairs, mirroring ReFoRCE's pre_info layout."""
    parts = []
    for qr in query_results:
        sql = qr["sql"]
        if qr.get("error"):
            parts.append(f"Query:\n{sql}\nAnswer:\nERROR: {qr['error'][:200]}\n")
        else:
            cols = qr.get("columns", [])
            rows = qr.get("rows", [])
            sample = rows[:8]
            tbl = "| " + " | ".join(cols) + " |\n"
            for r in sample:
                tbl += "| " + " | ".join("" if v is None else str(v)[:60] for v in r) + " |\n"
            parts.append(f"Query:\n{sql}\nAnswer:\n{tbl}")
    return "\n".join(parts)


def run_exploration_phase(
    *,
    task: PublicTask,
    model: ModelAdapter,
    max_queries: int = 8,
    max_findings_chars: int = 8000,
) -> dict[str, Any]:
    """Generate exploratory SQLs, execute them, return formatted findings + meta."""
    catalog = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)
    user = build_exploration_user_prompt(task, catalog, knowledge)
    raw = model.complete([
        ModelMessage(role="system", content=EXPLORATION_SYSTEM),
        ModelMessage(role="user", content=user),
    ])
    queries = _extract_queries(raw)[:max_queries]

    query_results: list[dict[str, Any]] = []
    successful = 0
    failed = 0
    chars_so_far = 0
    for sql in queries:
        try:
            result = execute_sql(task.context_dir, sql, limit=50)
            qr = {"sql": sql, "columns": result.get("columns", []), "rows": result.get("rows", []), "error": None}
            successful += 1
        except Exception as exc:
            qr = {"sql": sql, "columns": [], "rows": [], "error": f"{type(exc).__name__}: {exc}"}
            failed += 1
        # Budget control on cumulative findings size
        formatted_one = _format_findings([qr])
        if chars_so_far + len(formatted_one) > max_findings_chars:
            break
        chars_so_far += len(formatted_one)
        query_results.append(qr)

    findings_block = _format_findings(query_results) if query_results else ""

    return {
        "findings": findings_block,
        "queries_attempted": len(queries),
        "successful": successful,
        "failed": failed,
        "raw_response": raw,
    }
