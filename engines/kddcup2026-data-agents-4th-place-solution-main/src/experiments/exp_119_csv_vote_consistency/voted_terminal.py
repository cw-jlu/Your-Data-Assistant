"""Voted answer terminal (= ReFoRCE's vote + consistency at commit time).

When the agent calls answer_from_sql, instead of immediately committing the
agent's SQL, this module:

1. Treats the agent's SQL as candidate 0.
2. Generates K-1 alternative SQLs at temp=1.0 from a focused specialist
   prompt (= using the same intent as the agent's terminal call).
3. Executes all K candidates against DuckDB.
4. Compares result row-sets (= column-name-agnostic, value-set match).
5. CONSISTENCY CHECK: requires a UNIQUE max-vote winner. If multiple
   clusters tie at the max, this is "no consensus" and the call fails
   non-terminally so the agent retries.

Mirrors the ReFoRCE vote_result design (= agent.py:334-396, utils.py:100-145).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_119_csv_vote_consistency.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
)


VOTER_SYSTEM = """\
You are a SQL specialist. Generate exactly ONE DuckDB SQL statement that
answers the task. Output a single ```sql fenced block, no extra text.

Rules:
- Start with SELECT or WITH.
- Name every column explicitly. NEVER `SELECT *`.
- For superlatives (lowest/highest/min/max), use filter-back, NOT LIMIT 1:
      SELECT col FROM t WHERE val = (SELECT MIN(val) FROM t)
- Apply ROUND() only at the outermost SELECT level.
- Pure SELECT/WITH only.
"""


@dataclass(frozen=True, slots=True)
class Candidate:
    sql: str
    columns: list[str] | None
    rows: list[list[Any]] | None
    error: str | None


def _extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _knowledge_slice(task: PublicTask, max_chars: int = 4000) -> str:
    p = task.context_dir / "knowledge.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n[...truncated]" if len(text) > max_chars else "")


def _norm_value(v: Any) -> tuple:
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
    """Order-insensitive, value-only signature.

    Sort rows lexicographically (= ignore order). Two candidates match if
    same shape AND same multiset of value tuples.
    """
    if not columns:
        return ("__empty__",)
    n_cols = len(columns)
    norm = []
    for r in rows:
        if len(r) != n_cols:
            continue
        norm.append(tuple(_norm_value(v) for v in r))
    norm.sort()
    return (n_cols, len(norm), tuple(norm))


def _execute_candidate(task: PublicTask, sql: str) -> Candidate:
    if not sql or not sql.lstrip().lower().startswith(("select", "with")):
        return Candidate(sql=sql, columns=None, rows=None, error="not a SELECT/WITH")
    try:
        result = execute_sql(task.context_dir, sql, limit=10000)
    except Exception as exc:
        return Candidate(sql=sql, columns=None, rows=None, error=f"{type(exc).__name__}: {exc}")
    columns = [str(c) for c in (result.get("columns") or [])]
    rows = [list(r) for r in (result.get("rows") or [])]
    if not columns:
        return Candidate(sql=sql, columns=None, rows=None, error="no columns returned")
    return Candidate(sql=sql, columns=columns, rows=rows, error=None)


def _generate_alt_sql(
    task: PublicTask, sub_question: str, format_csv: str,
    knowledge: str, catalog: str, voter_model: ModelAdapter, seed: int,
) -> Candidate:
    user = (
        f"## Task\n{sub_question}\n\n"
        f"## DuckDB views\n{catalog}\n\n"
        f"## knowledge.md\n{knowledge}\n\n"
        + (f"## Expected answer format\n```csv\n{format_csv}\n```\n\n" if format_csv else "")
        + f"<!--variant {seed}-->\n"
        "Output ONE ```sql block now."
    )
    try:
        raw = voter_model.complete([
            ModelMessage(role="system", content=VOTER_SYSTEM),
            ModelMessage(role="user", content=user),
        ])
    except Exception as exc:
        return Candidate(sql="", columns=None, rows=None, error=f"LLM error: {exc}")
    sql = _extract_sql(raw)
    return _execute_candidate(task, sql)


@dataclass(frozen=True, slots=True)
class VotedResult:
    ok: bool
    columns: list[str]
    rows: list[list[Any]]
    winning_sql: str
    cluster_sizes: list[int]
    has_unique_max: bool
    decision: str  # "unique_max" | "tied" | "all_failed"
    error: str | None = None


def voted_terminal(
    *,
    task: PublicTask,
    agent_sql: str,
    sub_question: str,
    format_csv: str,
    voter_model: ModelAdapter,
    k: int = 5,
) -> VotedResult:
    """Build K candidates (incl. agent's SQL), vote, require unique max."""
    catalog = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)

    candidates: list[Candidate] = []
    # Candidate 0 = the agent's own SQL
    candidates.append(_execute_candidate(task, agent_sql))

    # K-1 alternative candidates from the voter model
    n_alt = max(0, k - 1)
    if n_alt > 0:
        with ThreadPoolExecutor(max_workers=n_alt) as ex:
            futures = [
                ex.submit(
                    _generate_alt_sql, task, sub_question, format_csv,
                    knowledge, catalog, voter_model, seed=i + 1,
                )
                for i in range(n_alt)
            ]
            for fut in as_completed(futures):
                candidates.append(fut.result())

    successful = [c for c in candidates if c.error is None and c.columns]
    if not successful:
        return VotedResult(
            ok=False, columns=[], rows=[], winning_sql="",
            cluster_sizes=[], has_unique_max=False, decision="all_failed",
            error=f"all {k} candidates failed",
        )

    sig_to_indices: dict[tuple, list[int]] = {}
    for i, c in enumerate(successful):
        sig = _result_signature(c.columns, c.rows or [])
        sig_to_indices.setdefault(sig, []).append(i)

    clusters = sorted(sig_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True)
    sizes = [len(idxs) for _, idxs in clusters]
    max_size = sizes[0]
    n_at_max = sum(1 for s in sizes if s == max_size)
    has_unique = n_at_max == 1

    if has_unique:
        winner = successful[clusters[0][1][0]]
        return VotedResult(
            ok=True, columns=list(winner.columns or []),
            rows=list(winner.rows or []),
            winning_sql=winner.sql, cluster_sizes=sizes,
            has_unique_max=True, decision="unique_max",
        )

    return VotedResult(
        ok=False, columns=[], rows=[], winning_sql="",
        cluster_sizes=sizes, has_unique_max=False, decision="tied",
        error=f"vote tied: {n_at_max} clusters at size {max_size}",
    )
